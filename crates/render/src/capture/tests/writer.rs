//! The writer: its size limit at the edge, the room it keeps for closing, refusals that change
//! nothing, the end, and shader sources stored once by content.

use super::super::{
    CaptureBuffer, CaptureError, CaptureSink, CaptureWriter, Create, Created, Destroy, Limit,
    Record, RecordKind, ShaderCode, SourceRecord,
};
use super::{naga, read, start, writer};
use crate::command::CommandList;
use crate::id::{BufferId, FenceValue, ResourceId};
use crate::resource::{BufferDesc, BufferUsage};
use crate::shader::{ShaderDesc, ShaderInterface, ShaderOrigin, ShaderSource, ShaderStage};
use std::sync::Arc;

/// The room kept for closing, by hand: a `Limit` with the largest fence (tag, length, and a
/// ten-byte varint) and `End` (tag and a zero length).
const CLOSE: u64 = (1 + 1 + 10) + (1 + 1);

fn made_buffer(index: u32) -> Record {
    Record::CreateBuffer(Create {
        desc: BufferDesc {
            size: 64,
            usage: BufferUsage::STORAGE,
        },
        result: Created::Made(BufferId::new(index, 0)),
    })
}

/// A writer limited to `limit` bytes, and its buffer.
fn limited(limit: u64) -> Result<(CaptureWriter, Arc<CaptureBuffer>), CaptureError> {
    let buffer = Arc::new(CaptureBuffer::new());
    let sink: Arc<dyn CaptureSink> = Arc::<CaptureBuffer>::clone(&buffer);
    CaptureWriter::new(sink, start(), limit).map(|w| (w, buffer))
}

/// Bytes of the header and `Start`.
fn opening() -> u64 {
    writer().0.written()
}

/// Bytes `record` takes, with its tag and length.
fn size(record: &Record) -> u64 {
    let mut body = Vec::new();
    super::super::record::encode_body(record, &mut body).expect("encode");
    let len = u32::try_from(body.len()).expect("small");
    let prefix = super::super::write::prefix(record.kind(), len).expect("prefix");
    u64::try_from(prefix.len() + body.len()).expect("small")
}

#[test]
fn the_limit_must_hold_the_start_and_the_closing_records() {
    let needed = opening() + CLOSE;
    assert!(limited(needed).is_ok());
    let error = limited(needed - 1).map(|_| ()).expect_err("one byte short");
    assert_eq!(
        error,
        CaptureError::LimitTooSmall {
            limit: needed - 1,
            needed
        }
    );
    assert_eq!(
        error.to_string(),
        format!(
            "a capture limit of {} bytes cannot hold its start, which needs {needed}",
            needed - 1
        )
    );
}

#[test]
fn a_record_past_the_limit_is_refused_and_the_capture_closes_within_it() {
    let record = made_buffer(1);
    let exact = opening() + size(&record) + CLOSE;
    let (mut w, _) = limited(exact).expect("fits");
    assert_eq!(w.write(&record), Ok(()));
    assert_eq!(w.written(), exact - CLOSE);

    let (mut w, buffer) = limited(exact - 1).expect("fits");
    let error = w.write(&record).expect_err("one byte short");
    assert_eq!(
        error,
        CaptureError::OverLimit {
            record: 1,
            needed: exact,
            limit: exact - 1
        }
    );
    assert_eq!(
        error.to_string(),
        format!(
            "record 1 would take the capture to {exact} bytes, past its limit of {}",
            exact - 1
        )
    );
    // The closing records use the room kept for them.
    assert_eq!(w.close_at_limit(), Ok(()));
    let bytes = buffer.bytes();
    assert!(u64::try_from(bytes.len()).expect("small") <= exact - 1);
    let records = read(&bytes).expect("a closed capture");
    assert_eq!(
        records.iter().map(Record::kind).collect::<Vec<_>>(),
        [RecordKind::Start, RecordKind::Limit, RecordKind::End]
    );
}

#[test]
fn the_limit_record_names_the_last_accepted_fence() {
    let (mut w, buffer) = writer();
    for (fence, made) in [(4, true), (0, false), (6, true)] {
        let list = w.encode_list(&CommandList::new()).expect("empty");
        let result = if made {
            Created::Made(FenceValue::new(fence))
        } else {
            Created::Refused("no".to_owned())
        };
        w.write_submit(list, &result).expect("in order");
    }
    w.close_at_limit().expect("closes");
    let records = read(&buffer.bytes()).expect("reads");
    assert_eq!(
        records.get(4),
        Some(&Record::Limit(Limit {
            fence: FenceValue::new(6)
        }))
    );
    // With no list, zero.
    let (mut w, buffer) = writer();
    w.close_at_limit().expect("closes");
    assert_eq!(
        read(&buffer.bytes()).expect("reads").get(1),
        Some(&Record::Limit(Limit {
            fence: FenceValue::ZERO
        }))
    );
}

#[test]
fn a_refused_record_changes_nothing() {
    let (mut w, buffer) = writer();
    w.write(&made_buffer(1)).expect("first");
    let written = w.written();
    // An id still live: refused, and the next record keeps its position.
    assert_eq!(
        w.write(&made_buffer(1)),
        Err(CaptureError::IdLive {
            record: 2,
            id: ResourceId::Buffer(BufferId::new(1, 0))
        })
    );
    assert_eq!(w.written(), written);
    // A source out of order, and a shader naming one not given.
    assert_eq!(
        w.write(&Record::ShaderSource(SourceRecord {
            index: 3,
            bytes: vec![1]
        })),
        Err(CaptureError::SourceIndex {
            record: 2,
            index: 3,
            expected: 0
        })
    );
    // A fence not above the last.
    let list = w.encode_list(&CommandList::new()).expect("empty");
    assert_eq!(
        w.write_submit(list, &Created::Made(FenceValue::ZERO)),
        Err(CaptureError::FenceOrder {
            record: 2,
            fence: 0
        })
    );
    assert_eq!(w.written(), written);
    w.write(&Record::Destroy(Destroy {
        id: ResourceId::Buffer(BufferId::new(1, 0)),
        after: FenceValue::ZERO,
    }))
    .expect("record 2");
    w.write(&made_buffer(1)).expect("live no more");
    w.finish().expect("end");
    assert_eq!(read(&buffer.bytes()).map(|r| r.len()), Ok(5));
}

#[test]
fn nothing_is_written_after_the_end() {
    let (mut w, buffer) = writer();
    assert!(!w.is_finished());
    w.finish().expect("end");
    assert!(w.is_finished());
    let bytes = buffer.bytes();
    assert_eq!(w.finish(), Err(CaptureError::Finished));
    assert_eq!(w.write(&made_buffer(1)), Err(CaptureError::Finished));
    assert_eq!(w.close_at_limit(), Err(CaptureError::Finished));
    let list = w.encode_list(&CommandList::new()).expect("encodes");
    assert_eq!(
        w.write_submit(list, &Created::Made(FenceValue::new(1))),
        Err(CaptureError::Finished)
    );
    assert_eq!(buffer.bytes(), bytes);
    assert_eq!(CaptureError::Finished.to_string(), "the capture has ended");
}

/// A shader with `origin`.
fn shader(origin: Option<ShaderOrigin>, source: ShaderSource) -> ShaderDesc {
    ShaderDesc {
        stage: ShaderStage::Vertex,
        entry: "main".to_owned(),
        source,
        interface: ShaderInterface::default(),
        origin,
    }
}


#[test]
fn shader_code_is_its_origin_once_by_content_or_its_own_code() {
    let (mut w, buffer) = writer();
    let a: Arc<[u8]> = Arc::from(vec![1, 2, 3]);
    // Equal bytes in another allocation: the same source.
    let a_again: Arc<[u8]> = Arc::from(vec![1, 2, 3]);
    let b: Arc<[u8]> = Arc::from(vec![4]);
    let origin = |record: &Arc<[u8]>, stage_index| {
        Some(ShaderOrigin {
            record: Arc::clone(record),
            stage_index,
        })
    };
    let codes = [
        w.shader_code(&shader(origin(&a, 7), naga())),
        w.shader_code(&shader(origin(&b, 8), ShaderSource::SpirV(vec![1]))),
        w.shader_code(&shader(origin(&a_again, 9), naga())),
        w.shader_code(&shader(None, ShaderSource::SpirV(vec![5, 6]))),
        w.shader_code(&shader(None, ShaderSource::Msl("m".to_owned()))),
        w.shader_code(&shader(None, naga())),
    ];
    assert_eq!(
        codes,
        [
            Ok(ShaderCode::Origin {
                source: 0,
                stage_index: 7
            }),
            Ok(ShaderCode::Origin {
                source: 1,
                stage_index: 8
            }),
            Ok(ShaderCode::Origin {
                source: 0,
                stage_index: 9
            }),
            Ok(ShaderCode::SpirV(vec![5, 6])),
            Ok(ShaderCode::Msl("m".to_owned())),
            Ok(ShaderCode::Missing),
        ]
    );
    w.finish().expect("end");
    let sources: Vec<Record> = read(&buffer.bytes())
        .expect("reads")
        .into_iter()
        .filter(|r| r.kind() == RecordKind::ShaderSource)
        .collect();
    assert_eq!(
        sources,
        [
            Record::ShaderSource(SourceRecord {
                index: 0,
                bytes: vec![1, 2, 3]
            }),
            Record::ShaderSource(SourceRecord {
                index: 1,
                bytes: vec![4]
            }),
        ]
    );
}

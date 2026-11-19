//! Every reading check, each with its exact error, and their documented order: two faults in
//! one file for each pair of neighbouring checks, the first reported.

use super::super::{
    BoundedField, CaptureError, CaptureLimits, CaptureReader, Create, CreateShader, Created,
    Destroy, Limit, Record, RecordKind, ShaderCode, SourceRecord, Start, Submit, VERSION,
};
use super::{END, file, framed, framed_any, prefix, read, read_with, start};
use crate::command::CommandList;
use crate::id::{BufferId, FenceValue, IdKind, ResourceId, ShaderId, TextureId};
use crate::resource::{BufferDesc, BufferUsage};
use crate::shader::{ShaderInterface, ShaderStage};
use std::io;

/// A record's body as postcard writes it (the fault cases need bodies; the golden test pins the
/// encoding itself).
fn body<T: serde::Serialize>(value: &T) -> Vec<u8> {
    postcard::to_allocvec(value).expect("encode")
}

/// A destroy of buffer `index` after fence zero, which no list is needed to reach.
fn destroy(index: u32) -> Vec<u8> {
    destroy_after(ResourceId::Buffer(BufferId::new(index, 0)), 0)
}

fn destroy_after(id: ResourceId, after: u64) -> Vec<u8> {
    framed(
        RecordKind::Destroy.tag(),
        &body(&Destroy {
            id,
            after: FenceValue::new(after),
        }),
    )
}

fn sampler_made(index: u32) -> Vec<u8> {
    framed(
        RecordKind::CreateSampler.tag(),
        &body(&Create {
            desc: super::sampler_desc(),
            result: Created::Made(crate::id::SamplerId::new(index, 0)),
        }),
    )
}

fn sampler(index: u32) -> ResourceId {
    ResourceId::Sampler(crate::id::SamplerId::new(index, 0))
}

fn buffer_made(index: u32, generation: u32) -> Vec<u8> {
    framed(
        RecordKind::CreateBuffer.tag(),
        &body(&Create {
            desc: BufferDesc {
                size: 16,
                usage: BufferUsage::VERTEX,
            },
            result: Created::Made(BufferId::new(index, generation)),
        }),
    )
}

fn texture_made(index: u32) -> Vec<u8> {
    framed(
        RecordKind::CreateTexture.tag(),
        &body(&Create {
            desc: crate::resource::TextureDesc {
                dimension: crate::resource::TextureDimension::D2,
                format: crate::format::PixelFormat::R8Unorm,
                size: crate::resource::Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_layers: 1,
                },
                mip_levels: 1,
                sample_count: 1,
                usage: crate::resource::TextureUsage::SAMPLED,
            },
            result: Created::Made(TextureId::new(index, 0)),
        }),
    )
}

/// A `Submit` of an empty list with `result`, by hand.
fn submit(result: &Created<FenceValue>) -> Vec<u8> {
    let mut b = vec![0x00]; // no commands
    b.extend(body(result));
    framed(RecordKind::Submit.tag(), &b)
}

fn made(fence: u64) -> Vec<u8> {
    submit(&Created::Made(FenceValue::new(fence)))
}

fn source(index: u32) -> Vec<u8> {
    framed(
        RecordKind::ShaderSource.tag(),
        &body(&SourceRecord {
            index,
            bytes: vec![0x5A],
        }),
    )
}

fn shader(source: u32, id: u32) -> Vec<u8> {
    framed(
        RecordKind::CreateShader.tag(),
        &body(&CreateShader {
            stage: ShaderStage::Vertex,
            entry: "main".to_owned(),
            interface: ShaderInterface::default(),
            code: ShaderCode::Origin {
                source,
                stage_index: 0,
            },
            result: Created::Made(ShaderId::new(id, 0)),
        }),
    )
}

fn limit() -> Vec<u8> {
    framed(
        RecordKind::Limit.tag(),
        &body(&Limit {
            fence: FenceValue::new(1),
        }),
    )
}

fn start_record() -> Vec<u8> {
    let b = body(&start());
    let mut out = vec![RecordKind::Start.tag()];
    out.extend(body(&u32::try_from(b.len()).expect("small")));
    out.extend(b);
    out
}

/// The header alone.
fn header() -> Vec<u8> {
    let mut h = b"SQRC".to_vec();
    h.push(0x02);
    h
}

#[test]
fn version_1_is_refused_before_its_body() {
    // A version-1 file as the old writer laid it out: a list mark, a fence, a list, the end.
    let v1 = [
        b'S', b'Q', b'R', b'C', 0x01, 0x01, 0x01, 0x01, 0x09, 0x07, 0x0B, 0x00,
    ];
    let error = read(&v1).expect_err("version 1");
    assert_eq!(error, CaptureError::UnsupportedVersion { found: 1 });
    assert_eq!(
        error.to_string(),
        "capture layout version 1 is not supported (this build reads 2)"
    );
    // Any other version, followed by nothing at all: the version is checked first.
    for found in [0, 3, VERSION + 1, u32::MAX] {
        let mut file = b"SQRC".to_vec();
        file.extend(body(&found));
        assert_eq!(
            read(&file),
            Err(CaptureError::UnsupportedVersion { found }),
            "{found}"
        );
    }
}

#[test]
fn a_foreign_file_is_bad_magic_before_its_version() {
    for bytes in [
        &b"PK\x03\x04\x01"[..],
        b"SQRD\x02",
        b"sqrc\x02",
        b"XQRC\x01",
    ] {
        assert_eq!(read(bytes), Err(CaptureError::BadMagic), "{bytes:?}");
    }
    assert_eq!(
        CaptureError::BadMagic.to_string(),
        "not a render capture (bad magic)"
    );
    // Shorter than the magic: cut short when it starts as the magic does.
    for bytes in [&b""[..], b"S", b"SQ", b"SQR"] {
        assert_eq!(read(bytes), Err(CaptureError::Truncated), "{bytes:?}");
    }
    for bytes in [&b"X"[..], b"SX", b"SQX"] {
        assert_eq!(read(bytes), Err(CaptureError::BadMagic), "{bytes:?}");
    }
    assert_eq!(
        CaptureError::Truncated.to_string(),
        "the capture ends early"
    );
}

#[test]
fn a_malformed_version_is_a_header_error() {
    let mut file = b"SQRC".to_vec();
    file.extend([0x80; 5]);
    assert_eq!(
        read(&file),
        Err(CaptureError::HeaderDecode(
            postcard::Error::DeserializeBadVarint
        ))
    );
    // A version that runs out of bytes is cut short.
    assert_eq!(read(b"SQRC\x80"), Err(CaptureError::Truncated));
    assert_eq!(read(b"SQRC"), Err(CaptureError::Truncated));
}

#[test]
fn truncation_at_every_byte_is_sticky() {
    let whole = file(&[&buffer_made(1, 0), &made(1)]);
    assert!(read(&whole).is_ok());
    for len in 0..whole.len() {
        let cut = &whole[..len];
        let n = u64::try_from(len).expect("small");
        match CaptureReader::new(cut, n) {
            Err(error) => {
                assert_eq!(error, CaptureError::Truncated, "{len} bytes, in the header");
                assert!(len <= header().len(), "{len}");
            }
            Ok(mut reader) => {
                let error = loop {
                    match reader.next_record() {
                        Ok(Some(_)) => {}
                        Ok(None) => unreachable!("{len} bytes read to the end"),
                        Err(error) => break error,
                    }
                };
                assert_eq!(error, CaptureError::Truncated, "{len} bytes");
                assert_eq!(reader.next_record(), Err(CaptureError::Truncated), "{len}");
                assert_eq!(reader.next_record(), Err(CaptureError::Truncated), "{len}");
            }
        }
    }
}

#[test]
fn an_input_shorter_than_its_length_is_truncated() {
    let whole = file(&[&buffer_made(1, 0)]);
    let short = &whole[..whole.len() - 1];
    let len = u64::try_from(whole.len()).expect("small");
    let mut reader = CaptureReader::new(short, len).expect("header");
    assert!(matches!(reader.next_record(), Ok(Some(Record::Start(_)))));
    assert!(matches!(
        reader.next_record(),
        Ok(Some(Record::CreateBuffer(_)))
    ));
    assert_eq!(reader.next_record(), Err(CaptureError::Truncated));
    assert_eq!(reader.next_record(), Err(CaptureError::Truncated));
}

/// An input that always fails.
struct Failing;

impl io::Read for Failing {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::other("the disk is unwell"))
    }
}

#[test]
fn a_failing_input_is_an_io_error_and_sticky() {
    // The header reads, then the input fails.
    let mut reader =
        CaptureReader::new(io::Read::chain(&b"SQRC\x02"[..], Failing), 100).expect("header");
    let expected = CaptureError::Io {
        kind: io::ErrorKind::Other,
        message: "the disk is unwell".to_owned(),
    };
    assert_eq!(reader.next_record(), Err(expected.clone()));
    assert_eq!(reader.next_record(), Err(expected.clone()));
    assert_eq!(
        expected.to_string(),
        "could not read the capture: the disk is unwell"
    );
}

#[test]
fn after_end_nothing_more_is_read() {
    let bytes = file(&[]);
    let len = u64::try_from(bytes.len()).expect("small");
    let mut reader = CaptureReader::new(&bytes[..], len).expect("header");
    assert!(matches!(reader.next_record(), Ok(Some(Record::Start(_)))));
    assert_eq!(reader.position(), 1);
    assert_eq!(reader.next_record(), Ok(Some(Record::End)));
    assert_eq!(reader.next_record(), Ok(None));
    assert_eq!(reader.next_record(), Ok(None));
    assert_eq!(reader.position(), 2);
}

#[test]
fn bytes_after_end_are_refused_and_sticky() {
    let mut bytes = file(&[]);
    bytes.extend([0x00, 0x07]);
    let len = u64::try_from(bytes.len()).expect("small");
    let mut reader = CaptureReader::new(&bytes[..], len).expect("header");
    assert!(matches!(reader.next_record(), Ok(Some(Record::Start(_)))));
    let error = CaptureError::TrailingBytes { count: 2 };
    assert_eq!(reader.next_record(), Err(error.clone()));
    assert_eq!(reader.next_record(), Err(error.clone()));
    assert_eq!(error.to_string(), "2 bytes follow the capture's End record");
    // One byte is as many too many.
    let mut one = file(&[]);
    one.push(0x00);
    assert_eq!(read(&one), Err(CaptureError::TrailingBytes { count: 1 }));
}

#[test]
fn an_unknown_tag_is_refused() {
    let first_unknown = u8::try_from(RecordKind::ALL.len()).expect("small");
    let error = read(&file(&[&[first_unknown, 0]])).expect_err("unknown");
    assert_eq!(
        error,
        CaptureError::UnknownRecord {
            tag: first_unknown,
            record: 1
        }
    );
    assert_eq!(error.to_string(), "record 1 has unknown tag 17");
    assert_eq!(
        read(&file(&[&[0xFF, 0]])),
        Err(CaptureError::UnknownRecord {
            tag: 0xFF,
            record: 1
        })
    );
}

#[test]
fn a_body_that_does_not_decode_is_refused() {
    // A buffer size whose varint runs past the body.
    let cut = framed(RecordKind::CreateBuffer.tag(), &[0xFF]);
    assert_eq!(
        read(&file(&[&cut])),
        Err(CaptureError::Decode {
            record: 1,
            source: postcard::Error::DeserializeUnexpectedEnd
        })
    );
    // A resource kind that does not exist.
    let bad_kind = framed(RecordKind::Destroy.tag(), &[0x06, 0x01, 0x01, 0x01]);
    let error = read(&file(&[&bad_kind])).expect_err("bad enum");
    assert_eq!(
        error,
        CaptureError::Decode {
            record: 1,
            source: postcard::Error::SerdeDeCustom
        }
    );
    // A record length that is not a varint.
    let mut bad_len = prefix();
    bad_len.extend([RecordKind::Destroy.tag(), 0x80, 0x80, 0x80, 0x80, 0x80]);
    bad_len.extend(END);
    assert_eq!(
        read(&bad_len),
        Err(CaptureError::Decode {
            record: 1,
            source: postcard::Error::DeserializeBadVarint
        })
    );
}

#[test]
fn a_body_shorter_than_its_length_is_refused() {
    let mut long = body(&Destroy {
        id: ResourceId::Buffer(BufferId::new(1, 1)),
        after: FenceValue::new(1),
    });
    long.push(0x00);
    let error = read(&file(&[&framed(RecordKind::Destroy.tag(), &long)])).expect_err("left over");
    assert_eq!(
        error,
        CaptureError::RecordTrailingBytes {
            record: 1,
            count: 1
        }
    );
    assert_eq!(
        error.to_string(),
        "record 1 has 1 bytes left over after its body"
    );
}

#[test]
fn start_comes_first_and_once() {
    let mut no_start = header();
    no_start.extend(destroy(1));
    no_start.extend(END);
    let error = read(&no_start).expect_err("no start");
    assert_eq!(
        error,
        CaptureError::MissingStart {
            record: 0,
            kind: RecordKind::Destroy
        }
    );
    assert_eq!(
        error.to_string(),
        "record 0 (Destroy) comes before the Start record"
    );
    let mut only_end = header();
    only_end.extend(END);
    assert_eq!(
        read(&only_end),
        Err(CaptureError::MissingStart {
            record: 0,
            kind: RecordKind::End
        })
    );
    let twice = file(&[&start_record()]);
    let error = read(&twice).expect_err("twice");
    assert_eq!(error, CaptureError::DuplicateStart { record: 1 });
    assert_eq!(error.to_string(), "record 1 is a second Start record");
    // One Start is accepted.
    assert_eq!(read(&file(&[])).map(|r| r.len()), Ok(2));
}

#[test]
fn only_end_follows_limit() {
    assert_eq!(read(&file(&[&limit()])).map(|r| r.len()), Ok(3));
    let error = read(&file(&[&limit(), &destroy(1)])).expect_err("after limit");
    assert_eq!(
        error,
        CaptureError::AfterLimit {
            record: 2,
            kind: RecordKind::Destroy
        }
    );
    assert_eq!(
        error.to_string(),
        "record 2 (Destroy) follows the Limit record"
    );
    assert_eq!(
        read(&file(&[&limit(), &limit()])),
        Err(CaptureError::AfterLimit {
            record: 2,
            kind: RecordKind::Limit
        })
    );
}

#[test]
fn accepted_fences_rise_from_above_zero() {
    let fence = |record, fence| Err(CaptureError::FenceOrder { record, fence });
    assert_eq!(read(&file(&[&made(0)])), fence(1, 0));
    assert_eq!(read(&file(&[&made(1)])).map(|r| r.len()), Ok(3));
    // Equal, below, and one above the one before.
    assert_eq!(read(&file(&[&made(5), &made(5)])), fence(2, 5));
    assert_eq!(read(&file(&[&made(5), &made(4)])), fence(2, 4));
    assert!(read(&file(&[&made(5), &made(6)])).is_ok());
    // A refused list has no fence and does not move the last one.
    let refused = submit(&Created::Refused("no".to_owned()));
    assert!(read(&file(&[&made(5), &refused, &made(6)])).is_ok());
    assert_eq!(read(&file(&[&made(5), &refused, &made(5)])), fence(3, 5));
    assert_eq!(
        CaptureError::FenceOrder {
            record: 2,
            fence: 5
        }
        .to_string(),
        "record 2: fence value 5 is not above the one before it"
    );
}

#[test]
fn shader_sources_come_in_order_and_are_named_after_they_come() {
    assert_eq!(
        read(&file(&[&source(1)])),
        Err(CaptureError::SourceIndex {
            record: 1,
            index: 1,
            expected: 0
        })
    );
    assert!(read(&file(&[&source(0), &source(1)])).is_ok());
    let error = read(&file(&[&source(0), &source(0)])).expect_err("again");
    assert_eq!(
        error,
        CaptureError::SourceIndex {
            record: 2,
            index: 0,
            expected: 1
        }
    );
    assert_eq!(
        error.to_string(),
        "record 2: shader source 0 should be source 1"
    );
    // Named before it is given; then the last given; then one past it.
    let error = read(&file(&[&shader(0, 1)])).expect_err("not given");
    assert_eq!(
        error,
        CaptureError::UnknownSource {
            record: 1,
            index: 0
        }
    );
    assert_eq!(
        error.to_string(),
        "record 1: shader source 0 has not been given"
    );
    assert!(read(&file(&[&source(0), &shader(0, 1)])).is_ok());
    assert_eq!(
        read(&file(&[&source(0), &shader(1, 1)])),
        Err(CaptureError::UnknownSource {
            record: 2,
            index: 1
        })
    );
}

#[test]
fn a_create_may_not_return_a_live_id() {
    let id = ResourceId::Buffer(BufferId::new(3, 4));
    let error = read(&file(&[&buffer_made(3, 4), &buffer_made(3, 4)])).expect_err("live");
    assert_eq!(error, CaptureError::IdLive { record: 2, id });
    assert_eq!(
        error.to_string(),
        "record 2 creates buffer 3 (generation 4), which is still live"
    );
    // Destroyed in between, or another generation, or another slot: accepted.
    let destroyed = framed(
        RecordKind::Destroy.tag(),
        &body(&Destroy {
            id,
            after: FenceValue::new(9),
        }),
    );
    assert!(read(&file(&[&buffer_made(3, 4), &destroyed, &buffer_made(3, 4)])).is_ok());
    assert!(read(&file(&[&buffer_made(3, 4), &buffer_made(3, 5)])).is_ok());
    assert!(read(&file(&[&buffer_made(3, 4), &buffer_made(4, 4)])).is_ok());
    // A refused create returns no id.
    let refused = framed(
        RecordKind::CreateBuffer.tag(),
        &body(&Create {
            desc: BufferDesc {
                size: 16,
                usage: BufferUsage::VERTEX,
            },
            result: Created::<BufferId>::Refused("no".to_owned()),
        }),
    );
    assert!(read(&file(&[&refused, &refused])).is_ok());
}

#[test]
fn live_ids_are_counted_per_kind() {
    let limits = CaptureLimits {
        live_resources: 2,
        ..CaptureLimits::SHIPPED
    };
    let at_cap = file(&[&buffer_made(1, 0), &buffer_made(2, 0)]);
    assert!(read_with(&at_cap, limits).is_ok());
    let error = read_with(
        &file(&[&buffer_made(1, 0), &buffer_made(2, 0), &buffer_made(3, 0)]),
        limits,
    )
    .expect_err("past the cap");
    assert_eq!(
        error,
        CaptureError::TooManyLive {
            record: 3,
            kind: IdKind::Buffer,
            max: 2
        }
    );
    assert_eq!(
        error.to_string(),
        "record 3: more than 2 buffer ids would be live"
    );
    // Another kind has its own count, and a destroy frees one.
    let other_kind = file(&[
        &buffer_made(1, 0),
        &buffer_made(2, 0),
        &texture_made(3),
        &texture_made(4),
        &destroy(1),
        &buffer_made(5, 0),
    ]);
    assert!(read_with(&other_kind, limits).is_ok());
    // A destroy of an id that is not live frees nothing.
    let not_live = file(&[
        &buffer_made(1, 0),
        &buffer_made(2, 0),
        &destroy(9),
        &buffer_made(5, 0),
    ]);
    assert_eq!(
        read_with(&not_live, limits),
        Err(CaptureError::TooManyLive {
            record: 4,
            kind: IdKind::Buffer,
            max: 2
        })
    );
}

#[test]
fn held_ids_are_counted_until_a_list_reaches_their_fence() {
    let limits = CaptureLimits {
        live_resources: 2,
        ..CaptureLimits::SHIPPED
    };
    let too_many = |record| {
        Err(CaptureError::TooManyLive {
            record,
            kind: IdKind::Sampler,
            max: 2,
        })
    };
    // Each sampler is destroyed after fence 5, past every list so far: replay holds it, so
    // the reader counts it. Two held are accepted; the third create is refused at its record.
    let held = |n: u32| -> Vec<Vec<u8>> {
        (1..=n)
            .flat_map(|i| [sampler_made(i), destroy_after(sampler(i), 5)])
            .collect()
    };
    let refs = |records: &[Vec<u8>]| -> Vec<u8> {
        file(&records.iter().map(Vec::as_slice).collect::<Vec<_>>())
    };
    assert!(read_with(&refs(&held(2)), limits).is_ok());
    let mut third = held(2);
    third.push(sampler_made(3));
    // Start, then two pairs: the third create is record 5.
    assert_eq!(read_with(&refs(&third), limits).map(|_| ()), too_many(5));
    // A list at the fence releases them: the third create is accepted.
    let mut reached = held(2);
    reached.push(made(5));
    reached.push(sampler_made(3));
    assert!(read_with(&refs(&reached), limits).is_ok());
    // A list one below the fence does not.
    let mut below = held(2);
    below.push(made(4));
    below.push(sampler_made(3));
    assert_eq!(read_with(&refs(&below), limits).map(|_| ()), too_many(6));
    // A destroy after a fence a list has already reached frees its id at once: with list 5
    // first, the destroys after 5 hold nothing.
    let mut at_fence = vec![made(5)];
    at_fence.extend(held(2));
    at_fence.push(sampler_made(3));
    assert!(read_with(&refs(&at_fence), limits).is_ok());
    // And one past it holds: list 4 first, destroys after 5.
    let mut past = vec![made(4)];
    past.extend(held(2));
    past.push(sampler_made(3));
    assert_eq!(read_with(&refs(&past), limits).map(|_| ()), too_many(6));
    // Held ids are counted per kind: a buffer is still accepted.
    let mut other_kind = held(2);
    other_kind.push(buffer_made(1, 0));
    assert!(read_with(&refs(&other_kind), limits).is_ok());
    // A list that releases one fence keeps the others: destroys after 5 and 6, list 5, then two
    // creates; the second passes the bound.
    let mut partly = vec![
        sampler_made(1),
        destroy_after(sampler(1), 5),
        sampler_made(2),
        destroy_after(sampler(2), 6),
        made(5),
        sampler_made(3),
    ];
    assert!(read_with(&refs(&partly), limits).is_ok());
    partly.push(sampler_made(4));
    assert_eq!(read_with(&refs(&partly), limits).map(|_| ()), too_many(7));
    // The hostile case: many samplers destroyed after the largest fence, with no list.
    let hostile: Vec<Vec<u8>> = (1..=100)
        .flat_map(|i| [sampler_made(i), destroy_after(sampler(i), u64::MAX)])
        .collect();
    assert_eq!(read_with(&refs(&hostile), limits).map(|_| ()), too_many(5));
    // The writer applies the same count.
    let (mut w, _) = super::writer_with(limits);
    for i in 1..=2 {
        w.write(&Record::CreateSampler(Create {
            desc: super::sampler_desc(),
            result: Created::Made(crate::id::SamplerId::new(i, 0)),
        }))
        .expect("create");
        w.write(&Record::Destroy(Destroy {
            id: sampler(i),
            after: FenceValue::new(5),
        }))
        .expect("destroy");
    }
    assert_eq!(
        w.write(&Record::CreateSampler(Create {
            desc: super::sampler_desc(),
            result: Created::Made(crate::id::SamplerId::new(3, 0)),
        })),
        too_many(5)
    );
}

/// The documented order, with two faults present at once for each pair of neighbouring checks.
#[test]
fn the_first_check_in_the_documented_order_is_reported() {
    // Magic, then version: a foreign file with version 1.
    assert_eq!(read(b"XQRC\x01"), Err(CaptureError::BadMagic));
    // Version, then records: version 1 followed by an unknown tag.
    assert_eq!(
        read(b"SQRC\x01\xFF\x00"),
        Err(CaptureError::UnsupportedVersion { found: 1 })
    );
    // Tag, then length: an unknown tag with a length past every cap.
    let mut f = prefix();
    f.extend([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);
    assert_eq!(
        read(&f),
        Err(CaptureError::UnknownRecord {
            tag: 0xFF,
            record: 1
        })
    );
    // Length cap, then truncation: a Destroy past its cap, with nothing behind it.
    let mut f = prefix();
    f.extend([RecordKind::Destroy.tag(), 0x81, 0x80, 0x04]); // 65,537
    assert!(matches!(read(&f), Err(CaptureError::RecordTooLarge { .. })));
    // Truncation, then decoding: a body that would not decode either, cut short.
    let mut f = prefix();
    f.extend([RecordKind::Destroy.tag(), 0x05, 0x09]);
    assert_eq!(read(&f), Err(CaptureError::Truncated));
    // Decoding, then left-over bytes: a bad enum with a byte to spare.
    let f = file(&[&framed(
        RecordKind::Destroy.tag(),
        &[0x06, 0x01, 0x01, 0x01, 0x00],
    )]);
    assert_eq!(
        read(&f),
        Err(CaptureError::Decode {
            record: 1,
            source: postcard::Error::SerdeDeCustom
        })
    );
    // Left-over bytes, then field bounds: a Start with a long build id and a byte to spare.
    let mut long = start();
    long.build = "b".repeat(5000);
    let mut b = body(&long);
    b.push(0x00);
    let mut f = header();
    f.push(RecordKind::Start.tag());
    f.extend(body(&u32::try_from(b.len()).expect("small")));
    f.extend(b);
    f.extend(END);
    assert_eq!(
        read(&f),
        Err(CaptureError::RecordTrailingBytes {
            record: 0,
            count: 1
        })
    );
    // Decoding, then the order checks: a malformed first record that is not Start.
    let mut f = header();
    f.extend(framed(RecordKind::Destroy.tag(), &[0x06, 0x01, 0x01, 0x01]));
    assert!(matches!(
        read(&f),
        Err(CaptureError::Decode { record: 0, .. })
    ));
    // Field bounds, then the order checks: a long entry name on a shader naming no source.
    let long_entry = framed_any(
        RecordKind::CreateShader.tag(),
        &body(&CreateShader {
            stage: ShaderStage::Vertex,
            entry: "e".repeat(5000),
            interface: ShaderInterface::default(),
            code: ShaderCode::Origin {
                source: 7,
                stage_index: 0,
            },
            result: Created::Made(ShaderId::new(1, 0)),
        }),
    );
    assert!(matches!(
        read(&file(&[&long_entry])),
        Err(CaptureError::TooLarge {
            what: BoundedField::EntryName,
            ..
        })
    ));
}

/// The order checks, in their documented order, each pair with both faults present.
#[test]
fn the_order_checks_come_in_their_documented_order() {
    // Start first, then Start once, then Limit, then the kind's own check.
    let mut f = header();
    f.extend(made(0));
    assert_eq!(
        read(&f),
        Err(CaptureError::MissingStart {
            record: 0,
            kind: RecordKind::Submit
        })
    );
    assert_eq!(
        read(&file(&[&limit(), &start_record()])),
        Err(CaptureError::DuplicateStart { record: 2 })
    );
    assert_eq!(
        read(&file(&[&limit(), &made(0)])),
        Err(CaptureError::AfterLimit {
            record: 2,
            kind: RecordKind::Submit
        })
    );
    // Within a CreateShader: the source it names, then the id it returns.
    assert_eq!(
        read(&file(&[&source(0), &shader(0, 1), &shader(4, 1)])),
        Err(CaptureError::UnknownSource {
            record: 3,
            index: 4
        })
    );
    // The order checks, then bytes after End: End first, with a byte behind it.
    let mut f = header();
    f.extend(END);
    f.push(0x00);
    assert_eq!(
        read(&f),
        Err(CaptureError::MissingStart {
            record: 0,
            kind: RecordKind::End
        })
    );
}

#[test]
fn a_submit_reads_its_list_then_its_result() {
    // Built by hand: two markers, then the fence. Swapping the two reads would fail.
    let mut b = vec![0x02, 0x09, 0x01, 0x02, 0x09, 0x03, 0x04];
    b.extend([0x00, 0x07]);
    let records = read(&file(&[&framed(RecordKind::Submit.tag(), &b)])).expect("a list");
    let Some(Record::Submit(Submit { list, result })) = records.get(1) else {
        unreachable!("a submit: {records:?}")
    };
    assert_eq!(*result, Created::Made(FenceValue::new(7)));
    assert_eq!(list.commands.len(), 2);
    assert_eq!(
        list,
        &CommandList {
            commands: vec![
                crate::command::Command::Marker(crate::command::Marker { submit: 1, draw: 2 }),
                crate::command::Command::Marker(crate::command::Marker { submit: 3, draw: 4 }),
            ],
        }
    );
}

#[test]
fn the_reader_reports_its_position() {
    let bytes = file(&[&destroy(1)]);
    let len = u64::try_from(bytes.len()).expect("small");
    let mut reader = CaptureReader::new(&bytes[..], len).expect("header");
    assert_eq!(reader.position(), 0);
    assert!(matches!(
        reader.next_record(),
        Ok(Some(Record::Start(Start { .. })))
    ));
    assert_eq!(reader.position(), 1);
    assert!(matches!(reader.next_record(), Ok(Some(Record::Destroy(_)))));
    assert_eq!(reader.position(), 2);
}

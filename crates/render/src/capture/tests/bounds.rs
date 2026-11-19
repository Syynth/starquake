//! Every bound at its edge, on both sides, through the reader and the writer: the accepted
//! side of each edge reaches the next check, so the two sides give different exact outcomes.

use super::super::replay::MAX_REPLAY_FRAMES;
use super::super::{
    BoundedField, CaptureError, CaptureLimits, CreateShader, Created, MAX_DESCRIPTION_BYTES,
    MAX_LIST_COMMANDS, MAX_LIST_DECODED_BYTES, MAX_LIST_DUMPS, MAX_LIVE_RESOURCES,
    MAX_RECORD_BYTES, MAX_SHADER_CODE_BYTES, MAX_SHADER_SOURCE_BYTES, MAX_SHADER_SOURCES_BYTES,
    MAX_TEXT_BYTES, Record, RecordKind, SOURCE_ENTRY_BYTES, ShaderCode, SourceRecord, Start,
    Submit,
};
use super::{END, file, framed_any, length, prefix, read, read_with, start, writer, writer_with};
use crate::caps::{GUEST_VERTEX_BUFFERS, GUEST_VIEWPORTS};
use crate::command::{
    Binding, BindingResource, BufferSlice, Command, CommandList, DispatchDesc, DispatchParams,
    DrawDesc, DrawParams, DynamicValues, MAX_COMMAND_BINDINGS, Marker, Rect, StagingSlice,
    Viewport,
};
use crate::id::{BufferId, FenceValue, PipelineId, ShaderId};
use crate::pipeline::DepthBias;
use crate::shader::{ShaderInterface, ShaderStage};

fn usize_of(n: u64) -> usize {
    usize::try_from(n).expect("fits")
}

#[test]
fn the_bounds_are_the_documented_values() {
    // Written out, so a wrong constant fails here and not only a swapped one.
    assert_eq!(MAX_RECORD_BYTES, 1_073_741_824);
    assert_eq!(MAX_DESCRIPTION_BYTES, 65_536);
    assert_eq!(MAX_LIST_COMMANDS, 1_048_576);
    assert_eq!(MAX_LIST_DECODED_BYTES, 2_147_483_648);
    assert_eq!(MAX_SHADER_SOURCE_BYTES, 2_097_152);
    assert_eq!(MAX_SHADER_CODE_BYTES, 16_777_216);
    assert_eq!(MAX_TEXT_BYTES, 4096);
    assert_eq!(MAX_LIVE_RESOURCES, 1_048_576);
    assert_eq!(MAX_COMMAND_BINDINGS, 4096);
    assert_eq!(MAX_REPLAY_FRAMES, 1024);
    assert_eq!(MAX_SHADER_SOURCES_BYTES, 1_073_741_824);
    assert_eq!(MAX_LIST_DUMPS, 1024);
    // A source's table entry: an `Arc<[u8]>` (pointer and length) and the `Arc`'s two counts.
    assert_eq!(SOURCE_ENTRY_BYTES, 32);
    assert_eq!(
        u64::try_from(size_of::<std::sync::Arc<[u8]>>() + 2 * size_of::<usize>()).expect("small"),
        SOURCE_ENTRY_BYTES
    );
    assert_eq!(
        CaptureLimits::SHIPPED,
        CaptureLimits {
            record_bytes: MAX_RECORD_BYTES,
            description_bytes: MAX_DESCRIPTION_BYTES,
            list_commands: MAX_LIST_COMMANDS,
            list_decoded_bytes: MAX_LIST_DECODED_BYTES,
            live_resources: MAX_LIVE_RESOURCES,
            source_bytes: MAX_SHADER_SOURCES_BYTES,
            list_dumps: MAX_LIST_DUMPS,
        }
    );
}

#[test]
fn kind_caps() {
    let s = CaptureLimits::SHIPPED;
    for kind in RecordKind::ALL {
        let expected = match kind {
            RecordKind::End => 0,
            RecordKind::Submit | RecordKind::ReadbackResult => 1 << 30,
            RecordKind::ShaderSource => (2 << 20) + (64 << 10),
            RecordKind::CreateShader => (16 << 20) + (64 << 10),
            _ => 64 << 10,
        };
        assert_eq!(s.max_body(kind), expected, "{kind}");
    }
    // The whole-record bound caps every kind's own.
    let low = CaptureLimits {
        record_bytes: 100,
        ..s
    };
    for kind in RecordKind::ALL {
        let expected = if kind == RecordKind::End { 0 } else { 100 };
        assert_eq!(low.max_body(kind), expected, "{kind}");
    }
}

/// A file whose next record is `kind` with length `len` and no body bytes behind it.
fn bare_length(kind: RecordKind, len: u64) -> Vec<u8> {
    let mut f = prefix();
    f.push(kind.tag());
    f.extend(length(len));
    f
}

#[test]
fn each_kind_cap_at_its_edge() {
    let s = CaptureLimits::SHIPPED;
    for kind in RecordKind::ALL {
        let max = s.max_body(kind);
        // One past the cap: refused before the bytes are looked for.
        let over = read(&bare_length(kind, max + 1));
        assert_eq!(
            over,
            Err(CaptureError::RecordTooLarge {
                record: 1,
                kind,
                len: max + 1,
                max
            }),
            "{kind}"
        );
        // At the cap: the next check, the bytes left, refuses it (no body follows; for End,
        // nothing follows a valid End, so the order checks pass and the file reads).
        let at = read(&bare_length(kind, max));
        if kind == RecordKind::End {
            assert!(at.is_ok(), "{at:?}");
        } else {
            assert_eq!(at, Err(CaptureError::Truncated), "{kind}");
        }
    }
    let error = CaptureError::RecordTooLarge {
        record: 1,
        kind: RecordKind::Destroy,
        len: 65_537,
        max: 65_536,
    };
    assert_eq!(
        error.to_string(),
        "record 1 (Destroy) is 65537 bytes long, more than the 65536 allowed"
    );
}

#[test]
fn the_writer_refuses_a_body_past_its_kind_cap() {
    // A pipeline whose vertex layout takes its body past the description cap: each attribute
    // here encodes in five bytes.
    let pipeline = |attributes: usize| {
        let mut desc = super::golden::golden_pipeline();
        let attribute = desc.vertex_input.attributes[0];
        desc.vertex_input.attributes = vec![attribute; attributes];
        Record::CreateGraphicsPipeline(super::super::Create {
            desc,
            result: Created::Made(PipelineId::new(1, 0)),
        })
    };
    let per = 5;
    // From 128 to 16,383 attributes the count is a two-byte varint, so the body grows by exactly
    // `per` bytes an attribute; the edge lies in that range.
    let base_count = 128;
    let base = {
        let mut body = Vec::new();
        super::super::record::encode_body(&pipeline(base_count), &mut body).expect("encode");
        u64::try_from(body.len()).expect("small")
    };
    let fit = base_count + usize_of((MAX_DESCRIPTION_BYTES - base) / per);
    assert!(fit < 1 << 14);
    let len_of = |n: usize| base + per * u64::try_from(n - base_count).expect("small");
    assert!(len_of(fit) <= MAX_DESCRIPTION_BYTES && len_of(fit + 1) > MAX_DESCRIPTION_BYTES);
    let (mut w, _) = writer();
    assert_eq!(
        w.write(&pipeline(fit + 1)),
        Err(CaptureError::RecordTooLarge {
            record: 1,
            kind: RecordKind::CreateGraphicsPipeline,
            len: len_of(fit + 1),
            max: MAX_DESCRIPTION_BYTES
        })
    );
    // Refused, so nothing changed: the next record is still record 1.
    assert_eq!(w.write(&pipeline(fit)), Ok(()));
}

/// A `Start` body with `build` and `backend`, framed.
fn start_with(build: &str, backend: &str) -> Vec<u8> {
    let s = Start {
        build: build.to_owned(),
        backend: backend.to_owned(),
        ..start()
    };
    let mut f = b"SQRC\x02".to_vec();
    f.extend(framed_any(
        RecordKind::Start.tag(),
        &postcard::to_allocvec(&s).expect("encode"),
    ));
    f.extend(END);
    f
}

#[test]
fn texts_at_their_edge() {
    let max = usize_of(MAX_TEXT_BYTES);
    let at = "t".repeat(max);
    let over = "t".repeat(max + 1);
    let too_long = |what| {
        Err(CaptureError::TooLarge {
            record: 0,
            what,
            len: MAX_TEXT_BYTES + 1,
            max: MAX_TEXT_BYTES,
        })
    };
    assert!(read(&start_with(&at, &at)).is_ok());
    assert_eq!(
        read(&start_with(&over, "")).map(|_| ()),
        too_long(BoundedField::BuildId)
    );
    assert_eq!(
        read(&start_with("", &over)).map(|_| ()),
        too_long(BoundedField::BackendName)
    );
    // The build id is checked before the backend name.
    assert_eq!(
        read(&start_with(&over, &over)).map(|_| ()),
        too_long(BoundedField::BuildId)
    );
    assert_eq!(
        CaptureError::TooLarge {
            record: 0,
            what: BoundedField::BuildId,
            len: 4097,
            max: 4096
        }
        .to_string(),
        "record 0: the build id is 4097 bytes long, more than the 4096 allowed"
    );

    // The writer applies the same bound.
    let sink: std::sync::Arc<dyn super::super::CaptureSink> =
        std::sync::Arc::new(super::super::CaptureBuffer::new());
    let long_start = Start {
        backend: over.clone(),
        ..start()
    };
    assert_eq!(
        super::super::CaptureWriter::new(sink, long_start, u64::MAX).map(|_| ()),
        too_long(BoundedField::BackendName)
    );

    // Entry names and refusal messages.
    let shader = |entry: &str, result| {
        Record::CreateShader(CreateShader {
            stage: ShaderStage::Vertex,
            entry: entry.to_owned(),
            interface: ShaderInterface::default(),
            code: ShaderCode::Missing,
            result,
        })
    };
    let (mut w, _) = writer();
    let made = || Created::Made(ShaderId::new(1, 0));
    assert!(matches!(
        w.write(&shader(&over, made())),
        Err(CaptureError::TooLarge {
            what: BoundedField::EntryName,
            ..
        })
    ));
    assert!(matches!(
        w.write(&shader("main", Created::Refused(over.clone()))),
        Err(CaptureError::TooLarge {
            what: BoundedField::RefusalMessage,
            ..
        })
    ));
    assert_eq!(w.write(&shader(&at, Created::Refused(at.clone()))), Ok(()));
    let (mut w, buffer) = writer();
    assert_eq!(w.write(&shader(&at, made())), Ok(()));
    w.finish().expect("end");
    assert!(read(&buffer.bytes()).is_ok());
    // A submit's refusal goes through the same check.
    let (mut w, _) = writer();
    let list = w.encode_list(&CommandList::new()).expect("empty");
    assert!(matches!(
        w.write_submit(list, &Created::Refused(over)),
        Err(CaptureError::TooLarge {
            what: BoundedField::RefusalMessage,
            ..
        })
    ));
}

#[test]
fn a_refusal_message_is_cut_to_the_bound_at_a_character_boundary() {
    let max = usize_of(MAX_TEXT_BYTES);
    // A two-byte character straddles the bound.
    let text = format!("{}é", "a".repeat(max - 1));
    let error: Result<(), String> = Err(text);
    let Created::Refused(cut) = Created::<()>::of(&error) else {
        unreachable!("refused")
    };
    assert_eq!(cut.len(), max - 1);
    let exact: Result<(), String> = Err("a".repeat(max));
    let Created::Refused(kept) = Created::<()>::of(&exact) else {
        unreachable!("refused")
    };
    assert_eq!(kept.len(), max);
}

#[test]
fn shader_source_and_code_at_their_edge() {
    let source = |len: u64| {
        Record::ShaderSource(SourceRecord {
            index: 0,
            bytes: vec![0x11; usize_of(len)],
        })
    };
    let (mut w, _) = writer();
    assert_eq!(
        w.write(&source(MAX_SHADER_SOURCE_BYTES + 1)),
        Err(CaptureError::TooLarge {
            record: 1,
            what: BoundedField::ShaderSource,
            len: MAX_SHADER_SOURCE_BYTES + 1,
            max: MAX_SHADER_SOURCE_BYTES
        })
    );
    assert_eq!(w.write(&source(MAX_SHADER_SOURCE_BYTES)), Ok(()));

    let code = |code| {
        Record::CreateShader(CreateShader {
            stage: ShaderStage::Fragment,
            entry: "main".to_owned(),
            interface: ShaderInterface::default(),
            code,
            result: Created::Made(ShaderId::new(1, 1)),
        })
    };
    let words = usize_of(MAX_SHADER_CODE_BYTES) / size_of::<u32>();
    let (mut w, _) = writer();
    assert_eq!(
        w.write(&code(ShaderCode::SpirV(vec![0; words + 1]))),
        Err(CaptureError::TooLarge {
            record: 1,
            what: BoundedField::SpirVCode,
            len: MAX_SHADER_CODE_BYTES + 4,
            max: MAX_SHADER_CODE_BYTES
        })
    );
    assert_eq!(
        w.write(&code(ShaderCode::Msl(
            "m".repeat(usize_of(MAX_SHADER_CODE_BYTES) + 1)
        ))),
        Err(CaptureError::TooLarge {
            record: 1,
            what: BoundedField::MslCode,
            len: MAX_SHADER_CODE_BYTES + 1,
            max: MAX_SHADER_CODE_BYTES
        })
    );
    // At the edge: written, and read back. SPIR-V of zero words is one byte a word, so the
    // body stays under the kind's cap; MSL at the edge with a short entry fits too.
    let (mut w, buffer) = writer();
    assert_eq!(w.write(&source(MAX_SHADER_SOURCE_BYTES)), Ok(()));
    assert_eq!(w.write(&code(ShaderCode::SpirV(vec![0; words]))), Ok(()));
    let msl = code(ShaderCode::Msl("m".repeat(usize_of(MAX_SHADER_CODE_BYTES))));
    let Record::CreateShader(mut second) = msl else {
        unreachable!("a shader")
    };
    second.result = Created::Made(ShaderId::new(2, 1));
    assert_eq!(w.write(&Record::CreateShader(second)), Ok(()));
    w.finish().expect("end");
    assert!(read(&buffer.bytes()).is_ok());
}

/// A list of `n` markers.
fn markers(n: usize) -> CommandList {
    CommandList {
        commands: (0..n)
            .map(|i| {
                Command::Marker(Marker {
                    submit: u64::try_from(i).expect("small"),
                    draw: 1,
                })
            })
            .collect(),
    }
}

/// `list` as a `Submit` record accepted with fence 1, framed.
fn submit_record(list: &CommandList) -> Vec<u8> {
    let mut b = postcard::to_allocvec(list).expect("encode");
    b.extend(postcard::to_allocvec(&Created::Made(FenceValue::new(1))).expect("encode"));
    framed_any(RecordKind::Submit.tag(), &b)
}

#[test]
fn a_lists_command_count_at_its_edge() {
    let limits = CaptureLimits {
        list_commands: 3,
        ..CaptureLimits::SHIPPED
    };
    assert!(read_with(&file(&[&submit_record(&markers(3))]), limits).is_ok());
    let error = read_with(&file(&[&submit_record(&markers(4))]), limits).expect_err("long");
    assert_eq!(
        error,
        CaptureError::ListTooLong {
            record: 1,
            count: 4,
            max: 3
        }
    );
    assert_eq!(
        error.to_string(),
        "record 1: the list holds 4 commands, more than the 3 allowed"
    );
    let (w, _) = writer_with(limits);
    assert!(w.encode_list(&markers(3)).is_ok());
    assert_eq!(
        w.encode_list(&markers(4)).map(|_| ()),
        Err(CaptureError::ListTooLong {
            record: 1,
            count: 4,
            max: 3
        })
    );

    // The shipped count is checked before any command is decoded: one past it with no command
    // bytes is refused for its count; at it, decoding finds the commands missing.
    let bare = |count: u64| {
        let mut b = postcard::to_allocvec(&count).expect("encode");
        b.push(0x00);
        file(&[&framed_any(RecordKind::Submit.tag(), &b)])
    };
    assert_eq!(
        read(&bare(MAX_LIST_COMMANDS + 1)),
        Err(CaptureError::ListTooLong {
            record: 1,
            count: MAX_LIST_COMMANDS + 1,
            max: MAX_LIST_COMMANDS
        })
    );
    assert_eq!(
        read(&bare(MAX_LIST_COMMANDS)),
        Err(CaptureError::Decode {
            record: 1,
            source: postcard::Error::DeserializeUnexpectedEnd
        })
    );
}

/// An upload of `len` bytes.
fn upload(len: usize) -> Command {
    Command::Upload {
        from: StagingSlice::from_bytes(vec![0x33; len]),
        to: crate::command::CopyRegion::Buffer(BufferSlice {
            buffer: BufferId::new(1, 0),
            offset: 0,
            size: 1,
        }),
    }
}

fn draw(bindings: usize, vertex_buffers: usize, viewports: usize, scissors: usize) -> Command {
    let slice = BufferSlice {
        buffer: BufferId::new(2, 0),
        offset: 0,
        size: 4,
    };
    Command::Draw(DrawDesc {
        pipeline: PipelineId::new(3, 0),
        bindings: (0..bindings)
            .map(|i| Binding {
                group: 0,
                binding: u32::try_from(i).expect("small"),
                resource: BindingResource::Buffer(slice),
            })
            .collect(),
        vertex_buffers: vec![slice; vertex_buffers],
        index_buffer: None,
        dynamic: DynamicValues {
            viewports: vec![
                Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                viewports
            ],
            scissors: vec![Rect::default(); scissors],
            blend_constant: [0.0; 4],
            stencil_reference: 0,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            depth_bias: DepthBias::default(),
            line_width: 1.0,
        },
        params: DrawParams::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
    })
}

fn dispatch(bindings: usize) -> Command {
    let Command::Draw(d) = draw(bindings, 0, 0, 0) else {
        unreachable!("a draw")
    };
    Command::Dispatch(DispatchDesc {
        pipeline: PipelineId::new(4, 0),
        bindings: d.bindings,
        params: DispatchParams::Direct { x: 1, y: 1, z: 1 },
    })
}

#[test]
fn a_lists_decoded_charge_at_its_edge() {
    let command = u64::try_from(size_of::<Command>()).expect("small");
    let binding = u64::try_from(size_of::<Binding>()).expect("small");
    let slice = u64::try_from(size_of::<BufferSlice>()).expect("small");
    let viewport = u64::try_from(size_of::<Viewport>()).expect("small");
    let rect = u64::try_from(size_of::<Rect>()).expect("small");
    // Commands of different sizes: a marker, a 5-byte upload, a draw with two bindings, one
    // vertex buffer, one viewport and one scissor, and a dispatch with three bindings.
    let list = CommandList {
        commands: vec![
            Command::Marker(Marker { submit: 1, draw: 2 }),
            upload(5),
            draw(2, 1, 1, 1),
            dispatch(3),
        ],
    };
    let charge = 4 * command + 5 + 2 * binding + slice + viewport + rect + 3 * binding;
    let at = CaptureLimits {
        list_decoded_bytes: charge,
        ..CaptureLimits::SHIPPED
    };
    let under = CaptureLimits {
        list_decoded_bytes: charge - 1,
        ..at
    };
    let f = file(&[&submit_record(&list)]);
    assert!(read_with(&f, at).is_ok());
    let error = read_with(&f, under).expect_err("one byte short");
    assert_eq!(
        error,
        CaptureError::ListTooBig {
            record: 1,
            command: 3,
            max: charge - 1
        }
    );
    assert_eq!(
        error.to_string(),
        format!(
            "record 1: command 3 takes the list past {} bytes decoded",
            charge - 1
        )
    );
    let (w, _) = writer_with(at);
    assert!(w.encode_list(&list).is_ok());
    let (w2, _) = writer_with(under);
    assert_eq!(w2.encode_list(&list).map(|_| ()), Err(error));
    // Charged across commands: with the budget at the first three commands' charge, the fourth
    // passes it.
    let first_three = charge - 3 * binding - command;
    let three = CaptureLimits {
        list_decoded_bytes: first_three,
        ..at
    };
    assert_eq!(
        read_with(&f, three),
        Err(CaptureError::ListTooBig {
            record: 1,
            command: 3,
            max: first_three
        })
    );
}

#[test]
fn the_lists_nested_in_a_command_at_their_edge() {
    let bindings = usize_of(MAX_COMMAND_BINDINGS.into());
    let buffers = usize_of(GUEST_VERTEX_BUFFERS.into());
    let viewports = usize_of(GUEST_VIEWPORTS.into());
    let cases = [
        (
            draw(bindings, 0, 0, 0),
            draw(bindings + 1, 0, 0, 0),
            BoundedField::DrawBindings,
        ),
        (
            draw(0, buffers, 0, 0),
            draw(0, buffers + 1, 0, 0),
            BoundedField::DrawVertexBuffers,
        ),
        (
            draw(0, 0, viewports, 0),
            draw(0, 0, viewports + 1, 0),
            BoundedField::DrawViewports,
        ),
        (
            draw(0, 0, 0, viewports),
            draw(0, 0, 0, viewports + 1),
            BoundedField::DrawScissors,
        ),
        (
            dispatch(bindings),
            dispatch(bindings + 1),
            BoundedField::DispatchBindings,
        ),
    ];
    for (at, over, what) in cases {
        let at = CommandList { commands: vec![at] };
        let over = CommandList {
            commands: vec![over],
        };
        assert!(read(&file(&[&submit_record(&at)])).is_ok(), "{what}");
        // Postcard keeps no message from a decode error raised inside a command.
        assert_eq!(
            read(&file(&[&submit_record(&over)])),
            Err(CaptureError::Decode {
                record: 1,
                source: postcard::Error::SerdeDeCustom
            }),
            "{what}"
        );
        let (w, _) = writer();
        assert!(w.encode_list(&at).is_ok(), "{what}");
        let len = match &over.commands[0] {
            Command::Draw(d) => [
                d.bindings.len(),
                d.vertex_buffers.len(),
                d.dynamic.viewports.len(),
                d.dynamic.scissors.len(),
            ]
            .into_iter()
            .max()
            .expect("four"),
            Command::Dispatch(d) => d.bindings.len(),
            _ => unreachable!("a draw or dispatch"),
        };
        let max = match what {
            BoundedField::DrawBindings | BoundedField::DispatchBindings => MAX_COMMAND_BINDINGS,
            BoundedField::DrawVertexBuffers => GUEST_VERTEX_BUFFERS,
            _ => GUEST_VIEWPORTS,
        };
        assert_eq!(
            w.encode_list(&over).map(|_| ()),
            Err(CaptureError::TooLarge {
                record: 1,
                what,
                len: u64::try_from(len).expect("small"),
                max: max.into()
            }),
            "{what}"
        );
    }
}

#[test]
fn a_submit_record_writes_and_reads_back() {
    let list = CommandList {
        commands: vec![upload(3), draw(1, 1, 1, 1), dispatch(1)],
    };
    let (mut w, buffer) = writer();
    let encoded = w.encode_list(&list).expect("fits");
    w.write_submit(encoded, &Created::Made(FenceValue::new(9)))
        .expect("written");
    w.finish().expect("end");
    let records = read(&buffer.bytes()).expect("reads");
    assert_eq!(
        records.get(1),
        Some(&Record::Submit(Submit {
            list,
            result: Created::Made(FenceValue::new(9))
        }))
    );
}

#[test]
fn the_end_record_has_no_body() {
    let mut f = prefix();
    f.extend([0x00, 0x01, 0x00]);
    assert_eq!(
        read(&f),
        Err(CaptureError::RecordTooLarge {
            record: 1,
            kind: RecordKind::End,
            len: 1,
            max: 0
        })
    );
}

/// A `ShaderSource` record of `len` bytes, framed.
fn source_record(index: u32, len: usize) -> Vec<u8> {
    let body = postcard::to_allocvec(&SourceRecord {
        index,
        bytes: vec![0xA5; len],
    })
    .expect("encode");
    framed_any(RecordKind::ShaderSource.tag(), &body)
}

#[test]
fn the_sources_charge_at_its_edge() {
    // Two sources of different lengths, each charged its length and the entry cost.
    let both = 3 + SOURCE_ENTRY_BYTES + 2 + SOURCE_ENTRY_BYTES;
    let limits = |source_bytes| CaptureLimits {
        source_bytes,
        ..CaptureLimits::SHIPPED
    };
    let two = file(&[&source_record(0, 3), &source_record(1, 2)]);
    assert!(read_with(&two, limits(both)).is_ok());
    let error = read_with(&two, limits(both - 1)).expect_err("one byte short");
    assert_eq!(
        error,
        CaptureError::SourcesTooLarge {
            record: 2,
            needed: both,
            max: both - 1
        }
    );
    assert_eq!(
        error.to_string(),
        format!(
            "record 2: the shader sources would be charged {both} bytes, more than the {} allowed",
            both - 1
        )
    );
    // An empty source is charged the entry cost: one fits a bound of exactly that, two do not.
    let empties = file(&[&source_record(0, 0), &source_record(1, 0)]);
    assert_eq!(
        read_with(&empties, limits(SOURCE_ENTRY_BYTES)),
        Err(CaptureError::SourcesTooLarge {
            record: 2,
            needed: 2 * SOURCE_ENTRY_BYTES,
            max: SOURCE_ENTRY_BYTES
        })
    );
    assert!(read_with(&empties, limits(2 * SOURCE_ENTRY_BYTES)).is_ok());
    // The index is checked before the charge: a wrong index past the charge is reported as such.
    let both_faults = file(&[&source_record(0, 3), &source_record(5, 2)]);
    assert_eq!(
        read_with(&both_faults, limits(both - 1)),
        Err(CaptureError::SourceIndex {
            record: 2,
            index: 5,
            expected: 1
        })
    );
    // The writer applies the same charge, through the shader-source path the wrapper uses.
    let (mut w, _) = writer_with(limits(both - 1));
    assert_eq!(w.shader_source(&[1, 2, 3]), Ok(0));
    assert_eq!(
        w.shader_source(&[4, 5]),
        Err(CaptureError::SourcesTooLarge {
            record: 2,
            needed: both,
            max: both - 1
        })
    );
    // A refused source changes nothing: a smaller one still fits, at the next index.
    assert_eq!(w.shader_source(&[6]), Ok(1));
}

#[test]
fn the_sources_charge_saturates_rather_than_wrapping() {
    use super::super::order::Order;
    use super::super::record::{Facts, SourceFact};
    let source = |index, charge| Facts {
        source: Some(SourceFact { index, charge }),
        ..Facts::default()
    };
    let limits = CaptureLimits {
        source_bytes: u64::MAX,
        ..CaptureLimits::SHIPPED
    };
    let mut order = Order::default();
    order.apply(RecordKind::Start, Facts::default());
    order.apply(RecordKind::ShaderSource, source(0, u64::MAX - 1));
    // Exactly at the bound with the second, then one past it, which would wrap.
    assert_eq!(
        order.check(RecordKind::ShaderSource, source(1, 1), limits),
        Ok(())
    );
    assert_eq!(
        order.check(RecordKind::ShaderSource, source(1, 2), limits),
        Err(CaptureError::SourcesTooLarge {
            record: 2,
            needed: u64::MAX,
            max: u64::MAX
        })
    );
}

/// A dumped (or not) present of frame `frame`.
fn present(frame: u64, dump: bool) -> Command {
    Command::Present(crate::command::PresentDesc {
        view: crate::id::ViewId::new(1, 0),
        src_rect: Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
        transform: crate::command::PresentTransform::Identity,
        frame,
        dump,
    })
}

/// A list that starts with a present that is not dumped, then `dumped` dumped presents.
fn dumps(dumped: u64) -> CommandList {
    let mut commands = vec![present(0, false)];
    commands.extend((1..=dumped).map(|frame| present(frame, true)));
    CommandList { commands }
}

#[test]
fn a_lists_dumped_presents_at_their_edge() {
    let max = MAX_LIST_DUMPS;
    // Presents that are not dumped are not counted: the one in front and one more at the end.
    let mut at = dumps(max);
    at.commands.push(present(max + 1, false));
    assert!(read(&file(&[&submit_record(&at)])).is_ok());
    let over = dumps(max + 1);
    // The first command is not dumped, so the extra dumped present is command `max + 1`.
    let expected = CaptureError::TooManyDumps {
        record: 1,
        command: max + 1,
        count: max + 1,
        max,
    };
    assert_eq!(read(&file(&[&submit_record(&over)])), Err(expected.clone()));
    assert_eq!(
        expected.to_string(),
        "record 1: command 1025 is dumped present 1025, more than the 1024 allowed"
    );
    let (w, _) = writer();
    assert!(w.encode_list(&at).is_ok());
    assert_eq!(w.encode_list(&over).map(|_| ()), Err(expected));
    // A test may lower it.
    let limits = CaptureLimits {
        list_dumps: 2,
        ..CaptureLimits::SHIPPED
    };
    assert!(read_with(&file(&[&submit_record(&dumps(2))]), limits).is_ok());
    assert_eq!(
        read_with(&file(&[&submit_record(&dumps(3))]), limits),
        Err(CaptureError::TooManyDumps {
            record: 1,
            command: 3,
            count: 3,
            max: 2
        })
    );
}

#[test]
fn bounded_fields_in_words() {
    // Every field, its words and whether its length counts elements, written out.
    let cases = [
        (BoundedField::BuildId, "build id", false),
        (BoundedField::BackendName, "backend name", false),
        (BoundedField::EntryName, "entry name", false),
        (BoundedField::RefusalMessage, "refusal message", false),
        (BoundedField::ShaderSource, "shader source", false),
        (BoundedField::SpirVCode, "SPIR-V code", false),
        (BoundedField::MslCode, "MSL code", false),
        (BoundedField::DrawBindings, "draw bindings", true),
        (BoundedField::DrawVertexBuffers, "draw vertex buffers", true),
        (BoundedField::DrawViewports, "draw viewports", true),
        (BoundedField::DrawScissors, "draw scissors", true),
        (BoundedField::DispatchBindings, "dispatch bindings", true),
    ];
    for (field, words, list) in cases {
        assert_eq!(field.to_string(), words);
        assert_eq!(field.is_list(), list, "{words}");
    }
    let error = |what| CaptureError::TooLarge {
        record: 1,
        what,
        len: 4097,
        max: 4096,
    };
    assert_eq!(
        error(BoundedField::DrawBindings).to_string(),
        "record 1: the draw bindings list holds 4097 elements, more than the 4096 allowed"
    );
    assert_eq!(
        error(BoundedField::EntryName).to_string(),
        "record 1: the entry name is 4097 bytes long, more than the 4096 allowed"
    );
}

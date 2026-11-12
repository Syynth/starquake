//! Tests for ring entries, their word encoding and their text form.
//!
//! The golden words below are laid out by hand from the layout table in `encode.rs`, not built
//! from its field constants, so a wrong constant cannot agree with itself.

use std::borrow::Cow;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use starquake_render::command::Rect;

use super::encode::{BIND_SUBCHANNEL, KINDS};
use super::*;
use crate::interface::{
    ChannelId, DisplayId, ImageDesc, LayerId, PresentRequest, SYNCPOINT_COUNT, SyncpointId,
};
use crate::sema::{PayloadSize, Reduction, ReductionFormat, ReductionOp, ReleaseForm};

fn config() -> Config {
    Config {
        cases: 2048,
        rng_seed: RngSeed::Fixed(0x7269_6e67),
        failure_persistence: None,
        ..Config::default()
    }
}

fn subchannel(value: u8) -> Subchannel {
    Subchannel::new(value).unwrap()
}

fn syncpoint(value: u8) -> SyncpointId {
    SyncpointId::new(value).unwrap()
}

/// One entry of every kind and variant, every field a distinct non-zero value, with its words
/// laid out by hand.
fn golden() -> Vec<(RingEntry<'static>, Vec<u64>)> {
    let mut entries = golden_engine();
    entries.extend(golden_host());
    entries
}

/// Binds, writes and runs.
fn golden_engine() -> Vec<(RingEntry<'static>, Vec<u64>)> {
    vec![
        (
            RingEntry::Bind {
                channel: ChannelId(0x11),
                subchannel: subchannel(5),
                class: ClassId(0xc797),
            },
            vec![0x0000_0011_0000_0501, 0x0000_0000_0000_c797],
        ),
        (
            RingEntry::Write {
                channel: ChannelId(0x12),
                class: ClassId(0xc7c0),
                method: 0x0e23,
                value: 0xdead_beef,
            },
            vec![0xdead_beef_000e_2302, 0x0000_c7c0_0000_0012],
        ),
        (
            RingEntry::Run {
                channel: ChannelId(0x13),
                class: ClassId(0xa140),
                method: 0x6d,
                words: Cow::Owned(vec![0x1111_1111, 0x2222_2222, 0x3333_3333]),
            },
            vec![
                0x0000_0003_0000_6d03,
                0x0000_a140_0000_0013,
                0x2222_2222_1111_1111,
                0x0000_0000_3333_3333,
            ],
        ),
    ]
}

/// Everything else.
fn golden_host() -> Vec<(RingEntry<'static>, Vec<u64>)> {
    vec![
        (
            RingEntry::Release(Release {
                address: 0x2000,
                payload: 0x7,
                form: ReleaseForm {
                    size: PayloadSize::Four,
                    timestamp: false,
                },
                op: ReleaseOp::Write,
            }),
            vec![0x0000_0000_0000_0004, 0x2000, 0x7],
        ),
        (
            // Reduce (bit 8), eight bytes (9), timestamp (10), RED_INC = 3 at 14:12, signed = 1
            // at 17:16.
            RingEntry::Release(Release {
                address: 0x1_2345_6780,
                payload: 0x9abc,
                form: ReleaseForm {
                    size: PayloadSize::Eight,
                    timestamp: true,
                },
                op: ReleaseOp::Reduce(Reduction {
                    op: ReductionOp::Inc,
                    format: ReductionFormat::Signed,
                }),
            }),
            vec![0x0000_0000_0001_3704, 0x1_2345_6780, 0x9abc],
        ),
        (
            RingEntry::Syncpoint { id: syncpoint(191) },
            vec![0x0000_0000_0000_bf05],
        ),
        (
            RingEntry::WaitForIdle {
                channel: ChannelId(0x14),
            },
            vec![0x0000_0014_0000_0006],
        ),
        (
            RingEntry::Invalidate {
                channel: ChannelId(0x15),
                what: Invalidation::L2FlushDirty,
            },
            vec![0x0000_0015_0000_1007],
        ),
        (
            RingEntry::Marker {
                token: MarkerToken(0x0102_0304_0506_0708),
            },
            vec![0x0000_0000_0000_0008, 0x0102_0304_0506_0708],
        ),
        (
            RingEntry::Present(PresentRequest {
                image: ImageDesc {
                    address: 0x8_0000_0000,
                    width: 1280,
                    height: 720,
                    pitch: 20,
                    block_height_log2: 4,
                    format: 1,
                },
                shown: Rect {
                    x: 3,
                    y: 5,
                    width: 1270,
                    height: 710,
                },
                layer: LayerId(0x21),
                display: DisplayId(0x22),
            }),
            vec![
                0x0000_0001_0000_0009,
                0x8_0000_0000,
                0x0000_02d0_0000_0500,
                0x0000_0004_0000_0014,
                0x0000_0005_0000_0003,
                0x0000_02c6_0000_04f6,
                0x21,
                0x22,
            ],
        ),
        (
            RingEntry::CaptureControl(CaptureCommand::Start),
            vec![0x0000_0000_0000_010a],
        ),
        (
            RingEntry::CaptureControl(CaptureCommand::Stop),
            vec![0x0000_0000_0000_020a],
        ),
    ]
}

#[test]
fn every_kind_has_golden_words() {
    let entries = golden();
    let mut kinds: Vec<Kind> = entries.iter().map(|(entry, _)| Kind::of(entry)).collect();
    kinds.dedup();
    assert_eq!(kinds.len(), KINDS.len());
    for (entry, words) in entries {
        let mut out = Vec::new();
        encode(&entry, &mut out).unwrap();
        assert_eq!(out, words, "{entry}");
        assert_eq!(encoded_words(&entry), words.len(), "{entry}");
        assert_eq!(decode(&words), Ok((entry.clone(), words.len())), "{entry}");
        assert_eq!(entry_words(words[0]), Ok((Kind::of(&entry), words.len())));
    }
}

#[test]
fn decode_reads_only_its_own_entry() {
    let (first, mut words) = golden().swap_remove(1);
    words.extend([0xffff_ffff_ffff_ffff, 0]);
    assert_eq!(decode(&words), Ok((first, 2)));
}

#[test]
fn encode_appends_and_leaves_the_output_alone_on_an_error() {
    let mut out = vec![0x55];
    encode(&golden()[0].0, &mut out).unwrap();
    assert_eq!(
        out,
        vec![0x55, 0x0000_0011_0000_0501, 0x0000_0000_0000_c797]
    );
    let too_long = RingEntry::Run {
        channel: ChannelId(1),
        class: ClassId(2),
        method: 3,
        words: Cow::Owned(vec![0; MAX_RUN_WORDS + 1]),
    };
    let before = out.clone();
    assert_eq!(
        encode(&too_long, &mut out),
        Err(EncodeError::RunTooLong {
            words: MAX_RUN_WORDS + 1
        })
    );
    assert_eq!(out, before);
}

#[test]
fn a_run_of_the_most_words_encodes_and_one_more_does_not() {
    let run = |count: usize| RingEntry::Run {
        channel: ChannelId(1),
        class: ClassId(2),
        method: 3,
        words: Cow::Owned((0..).take(count).collect()),
    };
    let mut out = Vec::new();
    encode(&run(MAX_RUN_WORDS), &mut out).unwrap();
    assert_eq!(out.len(), 2 + MAX_RUN_WORDS / 2);
    assert_eq!(decode(&out), Ok((run(MAX_RUN_WORDS), out.len())));
    assert_eq!(
        encode(&run(MAX_RUN_WORDS + 1), &mut Vec::new())
            .unwrap_err()
            .to_string(),
        "a run of 8193 words is longer than the most one entry carries, 8192"
    );
    // A header claiming one word more is refused before its length is trusted.
    let count = u32::try_from(MAX_RUN_WORDS + 1).unwrap();
    assert_eq!(
        entry_words((u64::from(count) << 32) | 3),
        Err(DecodeError::RunTooLong { words: count })
    );
}

#[test]
fn a_run_line_of_the_most_words_parses_and_one_more_does_not() {
    // The text form accepts exactly what `encode` accepts.
    let line = |count: usize| {
        let words: Vec<String> = (0..count).map(|word| format!("{word:#x}")).collect();
        format!(
            "run channel=1 class=0x2 method=0x3 words={}",
            words.join(",")
        )
    };
    let parsed = parse_entry(&line(MAX_RUN_WORDS)).unwrap();
    assert_eq!(
        parsed,
        RingEntry::Run {
            channel: ChannelId(1),
            class: ClassId(2),
            method: 3,
            words: Cow::Owned((0..).take(MAX_RUN_WORDS).collect()),
        }
    );
    encode(&parsed, &mut Vec::new()).unwrap();
    let error = parse_entry(&line(MAX_RUN_WORDS + 1)).unwrap_err();
    assert_eq!(
        error,
        ParseError::RunTooLong {
            words: MAX_RUN_WORDS + 1
        }
    );
    assert_eq!(
        error.to_string(),
        format!(
            "a run of {} words is longer than the most one entry carries, {MAX_RUN_WORDS}",
            MAX_RUN_WORDS + 1
        )
    );
    // The count is checked before the words, so a bad word past the limit is not what is
    // reported.
    let bad_word = format!("{},zz", line(MAX_RUN_WORDS));
    assert_eq!(
        parse_entry(&bad_word),
        Err(ParseError::RunTooLong {
            words: MAX_RUN_WORDS + 1
        })
    );
}

#[test]
fn an_empty_run_is_two_words() {
    let run = RingEntry::Run {
        channel: ChannelId(7),
        class: ClassId(8),
        method: 9,
        words: Cow::Borrowed(&[]),
    };
    let mut out = Vec::new();
    encode(&run, &mut out).unwrap();
    assert_eq!(out, vec![0x0000_0000_0000_0903, 0x0000_0008_0000_0007]);
    assert_eq!(decode(&out), Ok((run.into_owned(), 2)));
}

#[test]
fn malformed_words_are_exact_errors() {
    let cases: Vec<(Vec<u64>, DecodeError)> = vec![
        (vec![], DecodeError::Empty),
        (vec![0], DecodeError::UnknownKind { kind: 0 }),
        (vec![11], DecodeError::UnknownKind { kind: 11 }),
        (
            vec![0x0000_0011_0000_0501],
            DecodeError::Truncated {
                kind: Kind::Bind,
                needed: 2,
                available: 1,
            },
        ),
        // Write: bit 24 is above the method.
        (
            vec![0x0100_0002, 0],
            DecodeError::Reserved {
                kind: Kind::Write,
                word: 0,
                bits: 0x0100_0000,
            },
        ),
        // Bind: the class word's high half.
        (
            vec![0x0501, 0x1_0000_c797],
            DecodeError::Reserved {
                kind: Kind::Bind,
                word: 1,
                bits: 0x1_0000_0000,
            },
        ),
        (vec![0x0801, 0], DecodeError::Subchannel { value: 8 }),
        // Run of one word: the padding half must be zero.
        (
            vec![0x0000_0001_0000_0003, 0, 0x5_0000_0001],
            DecodeError::Reserved {
                kind: Kind::Run,
                word: 2,
                bits: 0x5_0000_0000,
            },
        ),
    ];
    for (words, error) in cases {
        assert_eq!(decode(&words), Err(error), "{words:x?}");
    }
}

#[test]
fn malformed_host_entries_are_exact_errors() {
    let cases: Vec<(Vec<u64>, DecodeError)> = vec![
        // Release: a plain write must not carry reduction fields.
        (
            vec![0x3004, 0, 0],
            DecodeError::Reserved {
                kind: Kind::Release,
                word: 0,
                bits: 0x3000,
            },
        ),
        // Release: the high half of the header.
        (
            vec![0x1_0000_0004, 0, 0],
            DecodeError::Reserved {
                kind: Kind::Release,
                word: 0,
                bits: 0x1_0000_0000,
            },
        ),
        (
            vec![0x2_0104, 0, 0],
            DecodeError::Reduction(crate::sema::DecodeError::UndefinedFormat {
                word: "ring release",
                value: 2,
            }),
        ),
        (vec![0xc005], DecodeError::Syncpoint { value: 192 }),
        (
            vec![0x1_0000_bf05],
            DecodeError::Reserved {
                kind: Kind::Syncpoint,
                word: 0,
                bits: 0x1_0000_0000,
            },
        ),
        (
            vec![0x0106],
            DecodeError::Reserved {
                kind: Kind::WaitForIdle,
                word: 0,
                bits: 0x100,
            },
        ),
        (vec![0x1107], DecodeError::Invalidation { value: 0x11 }),
        (
            vec![0x1_0000_0008, 0],
            DecodeError::Reserved {
                kind: Kind::Marker,
                word: 0,
                bits: 0x1_0000_0000,
            },
        ),
        (
            vec![0x109, 0, 0, 0, 0, 0, 0, 0],
            DecodeError::Reserved {
                kind: Kind::Present,
                word: 0,
                bits: 0x100,
            },
        ),
        (vec![0x000a], DecodeError::Capture { value: 0 }),
        (vec![0x030a], DecodeError::Capture { value: 3 }),
        (
            vec![0x1_0000_010a],
            DecodeError::Reserved {
                kind: Kind::CaptureControl,
                word: 0,
                bits: 0x1_0000_0000,
            },
        ),
    ];
    for (words, error) in cases {
        assert_eq!(decode(&words), Err(error), "{words:x?}");
    }
}

#[test]
fn decode_errors_read_as_plain_words() {
    assert_eq!(
        DecodeError::Empty.to_string(),
        "there is no ring entry here"
    );
    assert_eq!(
        decode(&[0]).unwrap_err().to_string(),
        "ring entry kind 0 is not defined"
    );
    assert_eq!(
        decode(&[0x0000_0003_0000_6d03, 0]).unwrap_err().to_string(),
        "a run entry needs 4 words but 2 are left"
    );
    assert_eq!(
        decode(&[0x0106]).unwrap_err().to_string(),
        "a wait-for-idle entry sets unused bits 0x100 in its word 0"
    );
    assert_eq!(
        decode(&[0x0801, 0]).unwrap_err().to_string(),
        "subchannel 8 is out of range"
    );
    assert_eq!(
        decode(&[0xc005]).unwrap_err().to_string(),
        "syncpoint 192 is out of range"
    );
    assert_eq!(
        decode(&[0x1107]).unwrap_err().to_string(),
        "memory operation 0x11 is not defined"
    );
    assert_eq!(
        decode(&[0x030a]).unwrap_err().to_string(),
        "capture command 3 is not defined"
    );
    assert_eq!(
        decode(&[0x2_0104, 0, 0]).unwrap_err().to_string(),
        "ring release reduction format 2 is not defined"
    );
}

#[test]
fn the_subchannel_field_is_wider_than_the_subchannels() {
    // The field can hold out-of-range values, which is what makes the range check reachable.
    assert!(BIND_SUBCHANNEL.max() >= SUBCHANNEL_COUNT);
}

#[test]
fn subchannels_are_checked_at_their_bound() {
    // Eight subchannels: the 3-bit field of `NVC76F_DMA_METHOD_SUBCHANNEL 15:13`.
    assert_eq!(SUBCHANNEL_COUNT, 8);
    assert_eq!(Subchannel::new(7).map(Subchannel::get), Some(7));
    assert_eq!(Subchannel::new(8), None);
}

#[test]
fn invalidations_follow_the_mem_op_d_operations() {
    // Hand-copied from `clc76f.h`'s `NVC76F_MEM_OP_D_OPERATION_*`.
    let table = [
        (Invalidation::Membar, 0x5),
        (Invalidation::MmuTlbInvalidate, 0x9),
        (Invalidation::MmuTlbInvalidateTargeted, 0xa),
        (Invalidation::MmuOperation, 0xb),
        (Invalidation::L2PeermemInvalidate, 0xd),
        (Invalidation::L2SysmemInvalidate, 0xe),
        (Invalidation::L2CleanComptags, 0xf),
        (Invalidation::L2FlushDirty, 0x10),
        (Invalidation::L2WaitForSysPendingReads, 0x15),
        (Invalidation::AccessCounterClear, 0x16),
    ];
    for (what, value) in table {
        assert_eq!(what.operation(), value);
        assert_eq!(Invalidation::from_operation(value), Some(what));
        // OPERATION is bits 31:27 of the word; the low bits carry other fields.
        assert_eq!(
            Invalidation::from_mem_op_d((value << 27) | 0x7ff_ffff),
            Some(what)
        );
    }
    for value in (0..32).filter(|value| !table.iter().any(|&(_, op)| op == *value)) {
        assert_eq!(Invalidation::from_operation(value), None, "{value:#x}");
    }
    assert_eq!(MEM_OP_D_OPERATION.get(0xf800_0000), 0x1f);
}

#[test]
fn entries_name_their_channel_and_own_their_words() {
    let words = [1, 2, 3];
    let run = RingEntry::Run {
        channel: ChannelId(4),
        class: ClassId(5),
        method: 6,
        words: Cow::Borrowed(&words),
    };
    assert_eq!(run.channel(), Some(ChannelId(4)));
    let owned: RingEntry<'static> = run.clone().into_owned();
    assert_eq!(owned, run);
    for (entry, _) in golden() {
        let expected = match &entry {
            RingEntry::Bind { channel, .. }
            | RingEntry::Write { channel, .. }
            | RingEntry::Run { channel, .. }
            | RingEntry::WaitForIdle { channel }
            | RingEntry::Invalidate { channel, .. } => Some(*channel),
            _ => None,
        };
        assert_eq!(entry.channel(), expected, "{entry}");
        assert_eq!(entry.clone().into_owned(), entry);
    }
}

// ---------------------------------------------------------------------------------------------
// The text form
// ---------------------------------------------------------------------------------------------

#[test]
fn every_golden_entry_has_its_text() {
    let expected = [
        "bind channel=17 subchannel=5 class=0xc797",
        "write channel=18 class=0xc7c0 method=0xe23 value=0xdeadbeef",
        "run channel=19 class=0xa140 method=0x6d words=0x11111111,0x22222222,0x33333333",
        "release address=0x2000 payload=0x7 size=4 timestamp=no op=write",
        "release address=0x123456780 payload=0x9abc size=8 timestamp=yes op=inc.signed",
        "syncpoint id=191",
        "wait-for-idle channel=20",
        "invalidate channel=21 what=l2-flush-dirty",
        "marker token=0x102030405060708",
        "present address=0x800000000 width=1280 height=720 pitch=20 block-height=4 format=1 \
         shown=3,5,1270,710 layer=33 display=34",
        "capture command=start",
        "capture command=stop",
    ];
    let entries: Vec<RingEntry<'static>> = golden().into_iter().map(|(entry, _)| entry).collect();
    for (entry, text) in entries.iter().zip(expected) {
        assert_eq!(entry.to_string(), text);
        assert_eq!(parse_entry(text).as_ref(), Ok(entry));
    }
    let text = write_text(&entries);
    let mut lines = expected.join("\n");
    lines.push('\n');
    assert_eq!(text, lines);
    assert_eq!(parse_text(&text), Ok(entries));
}

#[test]
fn hand_written_lines_parse() {
    let cases = [
        // Numbers in either base, wherever they appear.
        (
            "write channel=0x12 class=51136 method=3619 value=0x1",
            RingEntry::Write {
                channel: ChannelId(0x12),
                class: ClassId(0xc7c0),
                method: 0xe23,
                value: 1,
            },
        ),
        // Extra spaces between fields.
        (
            "  wait-for-idle   channel=3  ",
            RingEntry::WaitForIdle {
                channel: ChannelId(3),
            },
        ),
        // A run without words, and one with decimal words.
        (
            "run channel=1 class=2 method=3 words=",
            RingEntry::Run {
                channel: ChannelId(1),
                class: ClassId(2),
                method: 3,
                words: Cow::Owned(vec![]),
            },
        ),
        (
            "run channel=1 class=2 method=3 words=10,0x10,4294967295",
            RingEntry::Run {
                channel: ChannelId(1),
                class: ClassId(2),
                method: 3,
                words: Cow::Owned(vec![10, 0x10, u32::MAX]),
            },
        ),
        // Every reduction name and both formats.
        (
            "release address=0 payload=0 size=4 timestamp=no op=add.unsigned",
            reduce(ReductionOp::Add, ReductionFormat::Unsigned),
        ),
        (
            "release address=0 payload=0 size=4 timestamp=no op=min.signed",
            reduce(ReductionOp::Min, ReductionFormat::Signed),
        ),
        (
            "release address=0 payload=0 size=4 timestamp=no op=max.unsigned",
            reduce(ReductionOp::Max, ReductionFormat::Unsigned),
        ),
        (
            "release address=0 payload=0 size=4 timestamp=no op=dec.unsigned",
            reduce(ReductionOp::Dec, ReductionFormat::Unsigned),
        ),
        (
            "release address=0 payload=0 size=4 timestamp=no op=and.unsigned",
            reduce(ReductionOp::And, ReductionFormat::Unsigned),
        ),
        (
            "release address=0 payload=0 size=4 timestamp=no op=or.unsigned",
            reduce(ReductionOp::Or, ReductionFormat::Unsigned),
        ),
        (
            "release address=0 payload=0 size=4 timestamp=no op=xor.unsigned",
            reduce(ReductionOp::Xor, ReductionFormat::Unsigned),
        ),
        (
            "invalidate channel=0 what=membar",
            RingEntry::Invalidate {
                channel: ChannelId(0),
                what: Invalidation::Membar,
            },
        ),
        ("syncpoint id=0", RingEntry::Syncpoint { id: syncpoint(0) }),
    ];
    for (line, entry) in cases {
        assert_eq!(parse_entry(line), Ok(entry), "{line}");
    }
}

fn reduce(op: ReductionOp, format: ReductionFormat) -> RingEntry<'static> {
    RingEntry::Release(Release {
        address: 0,
        payload: 0,
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: false,
        },
        op: ReleaseOp::Reduce(Reduction { op, format }),
    })
}

#[test]
fn every_invalidation_has_a_name_that_reads_back() {
    let names = [
        "membar",
        "mmu-tlb-invalidate",
        "mmu-tlb-invalidate-targeted",
        "mmu-operation",
        "l2-peermem-invalidate",
        "l2-sysmem-invalidate",
        "l2-clean-comptags",
        "l2-flush-dirty",
        "l2-wait-for-sys-pending-reads",
        "access-counter-clear",
    ];
    let mut seen = Vec::new();
    for name in names {
        let entry = parse_entry(&format!("invalidate channel=1 what={name}")).unwrap();
        assert_eq!(
            entry.to_string(),
            format!("invalidate channel=1 what={name}")
        );
        let what = match entry {
            RingEntry::Invalidate { what, .. } => Some(what),
            _ => None,
        }
        .unwrap();
        assert!(!seen.contains(&what), "{name}");
        seen.push(what);
    }
}

#[test]
fn malformed_lines_are_exact_errors() {
    let cases = [
        ("", "the line is empty"),
        ("   ", "the line is empty"),
        ("draw channel=1", "\"draw\" is not an entry kind"),
        (
            "wait-for-idle",
            "a wait-for-idle entry needs channel next, but the line ends",
        ),
        (
            "write class=1 channel=2 method=3 value=4",
            "a write entry needs channel next, not \"class=1\"",
        ),
        (
            "write channel",
            "a write entry needs channel next, not \"channel\"",
        ),
        (
            "wait-for-idle channel=1 channel=2",
            "a wait-for-idle entry has nothing after its last field, not \"channel=2\"",
        ),
        ("wait-for-idle channel=x", "\"x\" is not a valid channel"),
        ("wait-for-idle channel=", "\"\" is not a valid channel"),
        ("wait-for-idle channel=0x", "\"0x\" is not a valid channel"),
        ("wait-for-idle channel=+1", "\"+1\" is not a valid channel"),
        (
            "wait-for-idle channel=0x+1",
            "\"0x+1\" is not a valid channel",
        ),
        ("wait-for-idle channel=-1", "\"-1\" is not a valid channel"),
        (
            "wait-for-idle channel=18446744073709551616",
            "\"18446744073709551616\" is not a valid channel",
        ),
    ];
    assert_parse_errors(&cases);
}

/// Checks that each line fails to parse with exactly its message.
fn assert_parse_errors(cases: &[(&str, &str)]) {
    for &(line, message) in cases {
        assert_eq!(
            parse_entry(line).map_err(|error| error.to_string()),
            Err(message.to_owned()),
            "{line}"
        );
    }
}

#[test]
fn out_of_range_and_unnamed_values_are_exact_errors() {
    let cases = [
        (
            "wait-for-idle channel=4294967296",
            "channel 4294967296 is out of range",
        ),
        (
            "write channel=1 class=2 method=0x10000 value=4",
            "method 65536 is out of range",
        ),
        (
            "bind channel=1 subchannel=8 class=2",
            "subchannel 8 is out of range",
        ),
        (
            "bind channel=1 subchannel=256 class=2",
            "subchannel 256 is out of range",
        ),
        ("syncpoint id=192", "id 192 is out of range"),
        (
            "release address=0 payload=0 size=2 timestamp=no op=write",
            "size 2 is out of range",
        ),
        (
            "release address=0 payload=0 size=4 timestamp=maybe op=write",
            "\"maybe\" is not a valid timestamp",
        ),
        (
            "release address=0 payload=0 size=4 timestamp=no op=add",
            "\"add\" is not a valid op",
        ),
        (
            "release address=0 payload=0 size=4 timestamp=no op=add.float",
            "\"add.float\" is not a valid op",
        ),
        (
            "release address=0 payload=0 size=4 timestamp=no op=mul.signed",
            "\"mul.signed\" is not a valid op",
        ),
        (
            "invalidate channel=1 what=flush",
            "\"flush\" is not a valid what",
        ),
        ("capture command=pause", "\"pause\" is not a valid command"),
        (
            "run channel=1 class=2 method=3 words=1,,2",
            "\"\" is not a valid words",
        ),
        (
            "run channel=1 class=2 method=3 words=0x100000000",
            "words 4294967296 is out of range",
        ),
        (
            "present address=0 width=1 height=1 pitch=1 block-height=0 format=1 shown=0,0,1 \
             layer=0 display=0",
            "\"0,0,1\" is not a valid shown",
        ),
        (
            "present address=0 width=1 height=1 pitch=1 block-height=0 format=1 shown=0,0,1,1,1 \
             layer=0 display=0",
            "\"0,0,1,1,1\" is not a valid shown",
        ),
    ];
    assert_parse_errors(&cases);
}

#[test]
fn a_bad_line_in_a_text_is_reported_by_its_number() {
    let text = "wait-for-idle channel=1\n\n   \nsyncpoint id=500\nmarker token=1\n";
    let error = parse_text(text).unwrap_err();
    assert_eq!(
        error,
        TextError {
            line: 4,
            error: ParseError::OutOfRange {
                key: "id",
                value: 500
            },
        }
    );
    assert_eq!(error.to_string(), "line 4: id 500 is out of range");
    assert_eq!(parse_text(""), Ok(vec![]));
}

// ---------------------------------------------------------------------------------------------
// Generated round trips
// ---------------------------------------------------------------------------------------------

fn reduction() -> impl Strategy<Value = Reduction> {
    (
        prop::sample::select(vec![
            ReductionOp::Add,
            ReductionOp::Min,
            ReductionOp::Max,
            ReductionOp::Inc,
            ReductionOp::Dec,
            ReductionOp::And,
            ReductionOp::Or,
            ReductionOp::Xor,
        ]),
        prop::sample::select(vec![ReductionFormat::Unsigned, ReductionFormat::Signed]),
    )
        .prop_map(|(op, format)| Reduction { op, format })
}

fn release() -> impl Strategy<Value = Release> {
    (
        any::<u64>(),
        any::<u64>(),
        any::<bool>(),
        any::<bool>(),
        prop::option::of(reduction()),
    )
        .prop_map(|(address, payload, eight, timestamp, reduce)| Release {
            address,
            payload,
            form: ReleaseForm {
                size: if eight {
                    PayloadSize::Eight
                } else {
                    PayloadSize::Four
                },
                timestamp,
            },
            op: reduce.map_or(ReleaseOp::Write, ReleaseOp::Reduce),
        })
}

fn present() -> impl Strategy<Value = PresentRequest> {
    (
        any::<u64>(),
        prop::array::uniform5(any::<u32>()),
        prop::array::uniform4(any::<u32>()),
        any::<u64>(),
        any::<u64>(),
    )
        .prop_map(|(address, image, shown, layer, display)| PresentRequest {
            image: ImageDesc {
                address,
                width: image[0],
                height: image[1],
                pitch: image[2],
                block_height_log2: image[3],
                format: image[4],
            },
            shown: Rect {
                x: shown[0],
                y: shown[1],
                width: shown[2],
                height: shown[3],
            },
            layer: LayerId(layer),
            display: DisplayId(display),
        })
}

/// Any entry the encoder accepts.
fn entry() -> impl Strategy<Value = RingEntry<'static>> {
    let channel = any::<u32>().prop_map(ChannelId);
    let class = any::<u32>().prop_map(ClassId);
    prop_oneof![
        (
            channel.clone(),
            0..u8::try_from(SUBCHANNEL_COUNT).unwrap(),
            class.clone()
        )
            .prop_map(|(channel, value, class)| RingEntry::Bind {
                channel,
                subchannel: subchannel(value),
                class,
            }),
        (channel.clone(), class.clone(), any::<u16>(), any::<u32>()).prop_map(
            |(channel, class, method, value)| RingEntry::Write {
                channel,
                class,
                method,
                value,
            }
        ),
        (
            channel.clone(),
            class,
            any::<u16>(),
            prop::collection::vec(any::<u32>(), 0..9)
        )
            .prop_map(|(channel, class, method, words)| RingEntry::Run {
                channel,
                class,
                method,
                words: Cow::Owned(words),
            }),
        release().prop_map(RingEntry::Release),
        (0..u8::try_from(SYNCPOINT_COUNT).unwrap())
            .prop_map(|id| RingEntry::Syncpoint { id: syncpoint(id) }),
        channel
            .clone()
            .prop_map(|channel| RingEntry::WaitForIdle { channel }),
        (
            channel,
            prop::sample::select(vec![
                Invalidation::Membar,
                Invalidation::MmuTlbInvalidate,
                Invalidation::MmuTlbInvalidateTargeted,
                Invalidation::MmuOperation,
                Invalidation::L2PeermemInvalidate,
                Invalidation::L2SysmemInvalidate,
                Invalidation::L2CleanComptags,
                Invalidation::L2FlushDirty,
                Invalidation::L2WaitForSysPendingReads,
                Invalidation::AccessCounterClear,
            ])
        )
            .prop_map(|(channel, what)| RingEntry::Invalidate { channel, what }),
        any::<u64>().prop_map(|token| RingEntry::Marker {
            token: MarkerToken(token)
        }),
        present().prop_map(RingEntry::Present),
        prop::sample::select(vec![CaptureCommand::Start, CaptureCommand::Stop])
            .prop_map(RingEntry::CaptureControl),
    ]
}

proptest! {
    #![proptest_config(config())]

    /// Encoding then decoding gives the entry back, taking exactly its words.
    #[test]
    fn words_round_trip(entry in entry()) {
        let mut words = Vec::new();
        encode(&entry, &mut words).unwrap();
        prop_assert_eq!(words.len(), encoded_words(&entry));
        prop_assert_eq!(entry_words(words[0]), Ok((Kind::of(&entry), words.len())));
        prop_assert_eq!(decode(&words), Ok((entry, words.len())));
    }

    /// Writing then reading the text form gives the entry back.
    #[test]
    fn text_round_trips(entry in entry()) {
        let text = entry.to_string();
        prop_assert_eq!(parse_entry(&text), Ok(entry));
    }

    /// A sequence of entries round-trips through the words and through the text, in order.
    #[test]
    fn sequences_round_trip(entries in prop::collection::vec(entry(), 0..16)) {
        let mut words = Vec::new();
        for entry in &entries {
            encode(entry, &mut words).unwrap();
        }
        let mut decoded = Vec::new();
        let mut rest = words.as_slice();
        while !rest.is_empty() {
            let (entry, used) = decode(rest).unwrap();
            decoded.push(entry);
            rest = &rest[used..];
        }
        prop_assert_eq!(&decoded, &entries);
        prop_assert_eq!(parse_text(&write_text(&entries)), Ok(entries));
    }

    /// An encoding with one bit flipped decodes to an error, or to an entry that encodes back to
    /// exactly the words it took: nothing decodes that the encoder would not write.
    #[test]
    fn a_flipped_bit_is_an_error_or_another_exact_entry(
        entry in entry(),
        word in any::<prop::sample::Index>(),
        bit in 0..64_u32,
    ) {
        let mut words = Vec::new();
        encode(&entry, &mut words).unwrap();
        let index = word.index(words.len());
        words[index] ^= 1 << bit;
        if let Ok((decoded, used)) = decode(&words) {
            let mut again = Vec::new();
            encode(&decoded, &mut again).unwrap();
            prop_assert_eq!(&again[..], &words[..used]);
        }
    }

    /// Arbitrary words never panic the decoder.
    #[test]
    fn arbitrary_words_decode_or_error(words in prop::collection::vec(any::<u64>(), 0..12)) {
        if let Ok((decoded, used)) = decode(&words) {
            let mut again = Vec::new();
            encode(&decoded, &mut again).unwrap();
            prop_assert_eq!(&again[..], &words[..used]);
        }
    }

    /// Arbitrary text never panics the parser.
    #[test]
    fn arbitrary_text_parses_or_errors(line in "[a-z0-9=,x\\- ]{0,40}") {
        if let Ok(entry) = parse_entry(&line) {
            prop_assert_eq!(parse_entry(&entry.to_string()), Ok(entry));
        }
    }
}

//! Golden bytes of the header and every record kind, each field a distinct non-zero value
//! where it can be: each written as the bytes below and read back from them. The tags, kind
//! bytes and layouts are written out as literals here, so a changed constant fails.

use std::time::Duration;

use starquake_memory::{Fault, PageRange};
use starquake_shader::lookup::{GuestStage, ProgramRequest};

use super::*;
use crate::back::fence::Waiter;
use crate::capture::format::one;
use crate::capture::{
    Access, BackRecord, ContentHash, EngineStateKind, FrontRecord, Header, Output, ShaderAnswer,
    StateRecord,
};
use crate::interface::{ChannelConfig, Fence, Kind, RegionId, Ticket, VmUpdate};
use crate::memory::GpuFault;
use crate::ring::MarkerToken;

/// `value`'s low `width` bytes, little-endian.
fn le(value: u128, width: usize) -> Vec<u8> {
    value.to_le_bytes()[..width].to_vec()
}

fn u32le(value: u32) -> Vec<u8> {
    le(value.into(), 4)
}

fn u64le(value: u64) -> Vec<u8> {
    le(value.into(), 8)
}

/// Three hashes whose sixteen bytes all differ.
const HASH_A: u128 = 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100;
const HASH_B: u128 = 0x1f1e_1d1c_1b1a_1918_1716_1514_1312_1110;
const HASH_C: u128 = 0x2f2e_2d2c_2b2a_2928_2726_2524_2322_2120;

fn hash_bytes(value: u128) -> Vec<u8> {
    le(value, 16)
}

fn range(start: u64, size: u64) -> PageRange {
    PageRange::new(start, size).unwrap()
}

fn front(record: FrontRecord, expected: &[Vec<u8>]) {
    let expected = expected.concat();
    assert_eq!(one::front(&record).unwrap(), expected, "{record:?}");
    assert_eq!(one::read_front(&expected).unwrap(), record);
}

fn back(record: BackRecord, expected: &[Vec<u8>]) {
    let expected = expected.concat();
    assert_eq!(one::back(&record).unwrap(), expected, "{record:?}");
    assert_eq!(one::read_back(&expected).unwrap(), record);
}

fn state(record: StateRecord, expected: &[Vec<u8>]) {
    let expected = expected.concat();
    assert_eq!(one::state(&record), expected, "{record:?}");
    assert_eq!(one::read_state(&expected).unwrap(), record);
}

#[test]
fn the_header() {
    let header = Header {
        build_id: "ab".to_owned(),
        caps: ContentHash::new(HASH_A),
        memory_allowed: 0x1122_3344_5566_7788,
        setting: WaitWhileCompiling::Skip,
        mode: Mode::AtRecord,
        spaces: vec![AsId(0x21), AsId(0x22)],
        channels: vec![super::super::CapturedChannel {
            id: ChannelId(3),
            config: ChannelConfig {
                space: AsId(0x22),
                syncpoint: SyncpointId::new(7).unwrap(),
            },
        }],
    };
    let expected = [
        b"SQGC".to_vec(),
        u32le(1),
        u32le(2),
        b"ab".to_vec(),
        hash_bytes(HASH_A),
        u64le(0x1122_3344_5566_7788),
        vec![1, 1],
        u32le(2),
        u32le(0x21),
        u32le(0x22),
        u32le(1),
        u32le(3),
        u32le(0x22),
        vec![7],
    ]
    .concat();
    assert_eq!(one::header(&header).unwrap(), expected);
    assert_eq!(one::read_header(&expected).unwrap(), header);
    // The other setting and mode are byte 0.
    let mut other = header.clone();
    other.setting = WaitWhileCompiling::Wait;
    other.mode = Mode::Fenced;
    let bytes = one::header(&other).unwrap();
    let at = 4 + 4 + 4 + 2 + 16 + 8;
    assert_eq!(bytes[at..at + 2], [0, 0]);
    assert_eq!(one::read_header(&bytes).unwrap(), other);
}

#[test]
fn starting_state_records() {
    state(
        StateRecord::AddressTable {
            space: AsId(0x31),
            blob: ContentHash::new(HASH_A),
        },
        &[vec![0x01], u32le(0x31), hash_bytes(HASH_A)],
    );
    state(
        StateRecord::Syncpoints {
            blob: ContentHash::new(HASH_B),
        },
        &[vec![0x02], hash_bytes(HASH_B)],
    );
    let kinds = [
        (EngineStateKind::RegisterFile, 1),
        (EngineStateKind::ShadowFile, 2),
        (EngineStateKind::Pusher, 3),
        (EngineStateKind::Bindings, 4),
        (EngineStateKind::MacroCode, 5),
        (EngineStateKind::DataRam, 6),
        (EngineStateKind::QmdTable, 7),
    ];
    assert_eq!(kinds.len(), EngineStateKind::ALL.len());
    for (kind, byte) in kinds {
        let owner = 0x4100 + u32::from(byte);
        state(
            StateRecord::Engine {
                kind,
                owner,
                blob: ContentHash::new(HASH_C),
            },
            &[vec![0x03, byte], u32le(owner), hash_bytes(HASH_C)],
        );
    }
}

#[test]
fn front_records_of_the_driver() {
    front(FrontRecord::Turn, &[vec![0x10]]);
    front(
        FrontRecord::Submit {
            channel: ChannelId(0x51),
            submission: Submission {
                entries: vec![
                    GpfifoEntry(0x8182_8384_8586_8788),
                    GpfifoEntry(0x9192_9394_9596_9798),
                ],
                wait: Some(Fence {
                    id: SyncpointId::new(9).unwrap(),
                    value: 0x6162_6364,
                }),
                increments: 0x71,
                wfi: true,
            },
        },
        &[
            vec![0x11],
            u32le(0x51),
            vec![1, 9],
            u32le(0x6162_6364),
            u32le(0x71),
            vec![1],
            u32le(2),
            u64le(0x8182_8384_8586_8788),
            u64le(0x9192_9394_9596_9798),
        ],
    );
    front(
        FrontRecord::Submit {
            channel: ChannelId(0x52),
            submission: Submission {
                entries: vec![],
                wait: None,
                increments: 0x72,
                wfi: false,
            },
        },
        &[vec![0x11], u32le(0x52), vec![0], u32le(0x72), vec![0], u32le(0)],
    );
    let r = range(0x1000, 0x3000);
    let rb = [u64le(0x1000), u64le(0x3000)].concat();
    front(
        FrontRecord::Update {
            space: AsId(0x23),
            update: VmUpdate::Unmap { range: r },
            ticket: Some(Ticket(0xa1)),
        },
        &[vec![0x12], u32le(0x23), vec![0], rb.clone(), vec![1], u64le(0xa1)],
    );
    front(
        FrontRecord::Update {
            space: AsId(0x24),
            update: VmUpdate::ModifyKind {
                range: r,
                kind: Kind(0x5b),
            },
            ticket: None,
        },
        &[vec![0x12], u32le(0x24), vec![1], rb.clone(), vec![0x5b, 0]],
    );
    front(
        FrontRecord::Update {
            space: AsId(0x25),
            update: VmUpdate::DestroyMemory { range: r },
            ticket: Some(Ticket(0xa2)),
        },
        &[vec![0x12], u32le(0x25), vec![2], rb.clone(), vec![1], u64le(0xa2)],
    );
    front(
        FrontRecord::Update {
            space: AsId(0x26),
            update: VmUpdate::DestroySparseRegion {
                range: r,
                region: RegionId(0xc1),
            },
            ticket: Some(Ticket(0xa3)),
        },
        &[vec![0x12], u32le(0x26), vec![3], rb, u32le(0xc1), vec![1], u64le(0xa3)],
    );
    front(
        FrontRecord::Barrier {
            ticket: Ticket(0xd1d2),
        },
        &[vec![0x13], u64le(0xd1d2)],
    );
    front(
        FrontRecord::Present {
            request: PresentRequest {
                image: ImageDesc {
                    address: 0x1_0203_0405,
                    width: 0x11,
                    height: 0x12,
                    pitch: 0x13,
                    block_height_log2: 0x14,
                    format: 0x15,
                },
                shown: Rect {
                    x: 0x21,
                    y: 0x22,
                    width: 0x23,
                    height: 0x24,
                },
                layer: LayerId(0x31),
                display: DisplayId(0x32),
            },
        },
        &[
            vec![0x14],
            u64le(0x1_0203_0405),
            u32le(0x11),
            u32le(0x12),
            u32le(0x13),
            u32le(0x14),
            u32le(0x15),
            u32le(0x21),
            u32le(0x22),
            u32le(0x23),
            u32le(0x24),
            u64le(0x31),
            u64le(0x32),
        ],
    );
    front(
        FrontRecord::Destroy {
            channel: ChannelId(0xe1),
        },
        &[vec![0x15], u32le(0xe1)],
    );
}

#[test]
fn front_records_from_the_back_and_the_clock() {
    front(
        FrontRecord::Report {
            markers: vec![MarkerToken(0x11), MarkerToken(0x22)],
            released: true,
            incremented: false,
        },
        &[vec![0x16, 1, 0], u32le(2), u64le(0x11), u64le(0x22)],
    );
    front(
        FrontRecord::Report {
            markers: vec![],
            released: false,
            incremented: true,
        },
        &[vec![0x16, 0, 1], u32le(0)],
    );
    front(
        FrontRecord::Syncpoint {
            id: SyncpointId::new(0x0b).unwrap(),
            value: 0x1234_5678,
        },
        &[vec![0x17, 0x0b], u32le(0x1234_5678)],
    );
    front(
        FrontRecord::Latch {
            channel: ChannelId(0x13),
        },
        &[vec![0x18], u32le(0x13)],
    );
    front(
        FrontRecord::Clock {
            now: Duration::from_nanos(0x01_0203_0405),
        },
        &[vec![0x19], u64le(0x01_0203_0405)],
    );
}

/// Every access, with the bytes after its tag, in both logs.
fn accesses() -> Vec<(Access, Vec<u8>)> {
    let head = |tag: u8, space: u32, va: u64| [vec![tag], u32le(space), u64le(va)].concat();
    vec![
        (
            Access::Read {
                space: AsId(0x61),
                va: 0x7172,
                len: 0x10,
                result: Ok(ContentHash::new(HASH_A)),
            },
            [head(0x30, 0x61, 0x7172), u64le(0x10), vec![0], hash_bytes(HASH_A)].concat(),
        ),
        (
            Access::Read {
                space: AsId(0x62),
                va: 0x7273,
                len: 0x20,
                result: Err(GpuFault::Unmapped { va: 0x8182 }),
            },
            [head(0x30, 0x62, 0x7273), u64le(0x20), vec![1, 0], u64le(0x8182)].concat(),
        ),
        (
            Access::Read {
                space: AsId(0x63),
                va: 0x7374,
                len: 0x30,
                result: Err(GpuFault::OutOfRange {
                    va: 0x8283,
                    len: 0x9192,
                }),
            },
            [
                head(0x30, 0x63, 0x7374),
                u64le(0x30),
                vec![1, 1],
                u64le(0x8283),
                u64le(0x9192),
            ]
            .concat(),
        ),
        (
            Access::Read {
                space: AsId(0x64),
                va: 0x7475,
                len: 0x40,
                result: Err(GpuFault::Guest {
                    va: 0x8384,
                    fault: Fault { address: 0x9394 },
                }),
            },
            [
                head(0x30, 0x64, 0x7475),
                u64le(0x40),
                vec![1, 2],
                u64le(0x8384),
                u64le(0x9394),
            ]
            .concat(),
        ),
        (
            Access::Write {
                space: AsId(0x65),
                va: 0x7576,
                data: ContentHash::new(HASH_B),
                result: Ok(()),
            },
            [head(0x31, 0x65, 0x7576), hash_bytes(HASH_B), vec![0]].concat(),
        ),
        (
            Access::Write {
                space: AsId(0x66),
                va: 0x7677,
                data: ContentHash::new(HASH_C),
                result: Err(GpuFault::ShimWrite { va: 0x8586 }),
            },
            [head(0x31, 0x66, 0x7677), hash_bytes(HASH_C), vec![1, 3], u64le(0x8586)].concat(),
        ),
        (
            Access::GpuOnly {
                space: AsId(0x67),
                va: 0x7778,
                len: 0x50,
                answer: true,
            },
            [head(0x32, 0x67, 0x7778), u64le(0x50), vec![1]].concat(),
        ),
        (
            Access::Generation {
                space: AsId(0x68),
                value: 0x8788,
            },
            [vec![0x33], u32le(0x68), u64le(0x8788)].concat(),
        ),
    ]
}

#[test]
fn accesses_in_either_log() {
    for (access, bytes) in accesses() {
        front(FrontRecord::Access(access), &[bytes.clone()]);
        back(BackRecord::Access(access), &[bytes]);
    }
}

#[test]
fn back_records() {
    back(BackRecord::Apply, &[vec![0x20]]);
    back(BackRecord::Close, &[vec![0x21]]);
    back(
        BackRecord::Complete { completed: 0x77 },
        &[vec![0x22], u64le(0x77)],
    );
    back(
        BackRecord::Elapsed {
            elapsed: Duration::from_nanos(0x0203_0405),
        },
        &[vec![0x23], u64le(0x0203_0405)],
    );
    back(
        BackRecord::Timestamp {
            value: 0x0304_0506,
        },
        &[vec![0x24], u64le(0x0304_0506)],
    );
    let ops = [
        (AcquireOp::Equal, 0),
        (AcquireOp::GreaterOrEqual, 1),
        (AcquireOp::CircularGreaterOrEqual, 2),
        (AcquireOp::And, 3),
        (AcquireOp::Nor, 4),
    ];
    let mut waiters = Vec::new();
    let mut expected = vec![vec![0x25], u32le(5)];
    for (index, (op, byte)) in (0u32..).zip(ops) {
        let latched = index % 2 == 0;
        waiters.push((
            Waiter {
                channel: ChannelId(0x10 + index),
                space: AsId(0x20 + index),
                address: 0x3000 + u64::from(index),
                op,
                payload: 0x40 + index,
            },
            latched,
        ));
        expected.extend([
            u32le(0x10 + index),
            u32le(0x20 + index),
            u64le(0x3000 + u64::from(index)),
            vec![byte],
            u32le(0x40 + index),
            vec![u8::from(latched)],
        ]);
    }
    back(BackRecord::Waiters { waiters }, &expected);
    back(
        BackRecord::Drain {
            space: AsId(0x27),
            ranges: vec![range(0x1000, 0x2000), range(0x5000, 0x1000)],
        },
        &[
            vec![0x26],
            u32le(0x27),
            u32le(2),
            u64le(0x1000),
            u64le(0x2000),
            u64le(0x5000),
            u64le(0x1000),
        ],
    );
}

#[test]
fn shader_records() {
    let stages = [
        (GuestStage::VertexCullBeforeFetch, 0),
        (GuestStage::Vertex, 1),
        (GuestStage::TessellationInit, 2),
        (GuestStage::Tessellation, 3),
        (GuestStage::Geometry, 4),
        (GuestStage::Pixel, 5),
        (GuestStage::Compute, 6),
    ];
    let requests: Vec<ProgramRequest> = (0u64..)
        .zip(stages)
        .map(|(index, (stage, _))| ProgramRequest {
            stage,
            address: 0x9000 + index,
        })
        .collect();
    let mut body = vec![vec![0x27], u32le(7)];
    for (index, (_, byte)) in (0u64..).zip(stages) {
        body.extend([vec![byte], u64le(0x9000 + index)]);
    }
    let answers = [
        (ShaderAnswer::Ready, vec![0]),
        (
            ShaderAnswer::Pending { ticket: 0xabcd },
            [vec![1], u64le(0xabcd)].concat(),
        ),
        (ShaderAnswer::Rejected, vec![2]),
    ];
    for (answer, answer_bytes) in answers {
        let mut expected = body.clone();
        expected.push(answer_bytes.clone());
        back(
            BackRecord::ShaderLookup {
                requests: requests.clone(),
                answer,
            },
            &expected,
        );
        back(
            BackRecord::ShaderWait {
                ticket: 0xbeef,
                answer,
            },
            &[vec![0x28], u64le(0xbeef), answer_bytes],
        );
    }
}

#[test]
fn outputs() {
    let ring = Output::Ring {
        hash: ContentHash::new(HASH_A),
    };
    let expected = [vec![0x40], hash_bytes(HASH_A)].concat();
    assert_eq!(one::output(&ring), expected);
    assert_eq!(one::read_output(&expected).unwrap(), ring);
    let render = Output::Render {
        text: ContentHash::new(HASH_B),
    };
    let expected = [vec![0x41], hash_bytes(HASH_B)].concat();
    assert_eq!(one::output(&render), expected);
    assert_eq!(one::read_output(&expected).unwrap(), render);
}

#[test]
fn the_tags_are_distinct_and_the_table_names_each_kind_once() {
    let tags = super::super::TAGS;
    for (i, (name, tag)) in tags.iter().enumerate() {
        for (other, other_tag) in &tags[i + 1..] {
            assert_ne!(tag, other_tag, "{name} and {other}");
        }
    }
    // The literal tags the golden tests above wrote, in table order.
    let literal: Vec<u8> = tags.iter().map(|(_, tag)| *tag).collect();
    assert_eq!(
        literal,
        [
            0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x20,
            0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x30, 0x31, 0x32, 0x33, 0x40, 0x41
        ]
    );
}

#[test]
fn the_content_hash_is_xxh3_128_with_seed_zero() {
    // The xxHash specification's value for the empty input, XXH3_128bits("") with seed 0.
    assert_eq!(
        ContentHash::of(b"").value(),
        0x99aa_06d3_0147_98d8_6001_c324_468d_497f
    );
    assert_eq!(
        ContentHash::of(b"").to_string(),
        "99aa06d3014798d86001c324468d497f"
    );
}

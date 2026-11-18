//! 216's entry layout and decoding, against PR's `GpuSparseMapRequest`, `DecodeGpuSparseMap` and
//! `DecodeNativeSparseRemap` (`eden-pr4588/src/core/hle/service/am/service/gpu_memory_registration.h`
//! and `eden-pr4588/src/core/hle/service/nvdrv/devices/native_sparse_remap.h` @ e5fb3b97).

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::*;

/// The constants, against PR's values written out: the struct's size and offsets, the batch
/// bound, the cache bound, and the page size behind its `& 0xfff` mask and `>> 12` shift.
#[test]
fn the_constants_are_prs() {
    assert_eq!(ENTRY_SIZE, 40);
    assert_eq!(MAX_ENTRIES, 256);
    assert_eq!(MAX_CACHE, 1);
    assert_eq!(PAGE_SIZE, 0x1000);
    assert_eq!(
        [
            offset::ADDRESS,
            offset::SIZE,
            offset::READ_ONLY,
            offset::CACHE,
            offset::RESERVED0,
            offset::KIND,
            offset::FALLBACK,
            offset::COMPTAG_MODE,
            offset::RESERVED1,
            offset::MEMORY,
            offset::RESERVED2,
            offset::SOURCE_OFFSET,
        ],
        [0, 8, 16, 17, 18, 19, 20, 21, 22, 24, 28, 32]
    );
}

/// Every field read from PR's offsets at its width: golden bytes laid out by hand, every byte
/// distinct and non-zero, so a field read from the wrong place or at the wrong width reads
/// different bytes.
#[test]
fn every_field_is_read_from_prs_offsets() {
    let bytes: [u8; ENTRY_SIZE] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, // address
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, // size
        0x21, // read_only
        0x22, // cache
        0x23, // reserved0
        0x24, // kind
        0x25, // fallback_kind
        0x26, // comptag_mode
        0x27, 0x28, // reserved1
        0x31, 0x32, 0x33, 0x34, // memory
        0x41, 0x42, 0x43, 0x44, // reserved2
        0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, // source_offset
    ];
    assert_eq!(
        SparseMapRequest::parse(&bytes),
        SparseMapRequest {
            address: 0x0807_0605_0403_0201,
            size: 0x1817_1615_1413_1211,
            read_only: 0x21,
            cache: 0x22,
            reserved0: 0x23,
            kind: 0x24,
            fallback: 0x25,
            comptag_mode: 0x26,
            reserved1: 0x2827,
            memory: 0x3433_3231,
            reserved2: 0x4443_4241,
            source_offset: 0x5857_5655_5453_5251,
        }
    );
}

/// The tests' encoder: each field at its [`offset`], little endian.
fn encode(request: &SparseMapRequest) -> [u8; ENTRY_SIZE] {
    let mut bytes = [0; ENTRY_SIZE];
    let mut put = |at: usize, field: &[u8]| bytes[at..at + field.len()].copy_from_slice(field);
    put(offset::ADDRESS, &request.address.to_le_bytes());
    put(offset::SIZE, &request.size.to_le_bytes());
    put(offset::READ_ONLY, &[request.read_only]);
    put(offset::CACHE, &[request.cache]);
    put(offset::RESERVED0, &[request.reserved0]);
    put(offset::KIND, &[request.kind]);
    put(offset::FALLBACK, &[request.fallback]);
    put(offset::COMPTAG_MODE, &[request.comptag_mode]);
    put(offset::RESERVED1, &request.reserved1.to_le_bytes());
    put(offset::MEMORY, &request.memory.to_le_bytes());
    put(offset::RESERVED2, &request.reserved2.to_le_bytes());
    put(offset::SOURCE_OFFSET, &request.source_offset.to_le_bytes());
    bytes
}

fn request() -> impl Strategy<Value = SparseMapRequest> {
    (
        (any::<u64>(), any::<u64>(), any::<u8>(), any::<u8>()),
        (any::<u8>(), any::<u8>(), any::<u8>(), any::<u8>()),
        (any::<u16>(), any::<u32>(), any::<u32>(), any::<u64>()),
    )
        .prop_map(
            |(
                (address, size, read_only, cache),
                (reserved0, kind, fallback, comptag_mode),
                (reserved1, memory, reserved2, source_offset),
            )| SparseMapRequest {
                address,
                size,
                read_only,
                cache,
                reserved0,
                kind,
                fallback,
                comptag_mode,
                reserved1,
                memory,
                reserved2,
                source_offset,
            },
        )
}

proptest! {
    #![proptest_config(Config {
        cases: 512,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x5350_4152_5345_4D50),
        failure_persistence: None,
        ..Config::default()
    })]

    /// Build, then parse: every field comes back, so two reads swapped (fields of the same
    /// width hold different random values) fail.
    #[test]
    fn every_entry_reads_back_as_built(request in request()) {
        prop_assert_eq!(SparseMapRequest::parse(&encode(&request)), request);
    }
}

/// An entry PR accepts: a mapped range of memory 7 from a page into it, cached, kinds 6 and 6.
fn accepted() -> SparseMapRequest {
    SparseMapRequest {
        address: 0x81_0000_0000,
        size: 2 * PAGE_SIZE,
        read_only: 0,
        cache: 1,
        reserved0: 0,
        kind: 6,
        fallback: 6,
        comptag_mode: 0,
        reserved1: 0,
        memory: 7,
        reserved2: 0,
        source_offset: PAGE_SIZE,
    }
}

/// One accepting case per feature: a handle with a source offset, cached and not; no handle
/// (back to the region's pages) with no offset; any kind pair (kinds are not checked here).
#[test]
fn accepted_entries_decode_to_their_range() {
    let entry = accepted();
    assert_eq!(
        decode(&entry),
        Some(SparseMap {
            range: PageRange::new(0x81_0000_0000, 2 * PAGE_SIZE).unwrap(),
            handle: 7,
            source_offset: PAGE_SIZE,
            kind: 6,
            fallback: 6,
        })
    );
    let uncached = SparseMapRequest { cache: 0, ..entry };
    assert!(decode(&uncached).is_some());
    let sparse = SparseMapRequest {
        memory: 0,
        source_offset: 0,
        kind: 0xAB,
        fallback: 0xCD,
        ..entry
    };
    assert_eq!(
        decode(&sparse),
        Some(SparseMap {
            range: PageRange::new(0x81_0000_0000, 2 * PAGE_SIZE).unwrap(),
            handle: 0,
            source_offset: 0,
            kind: 0xAB,
            fallback: 0xCD,
        })
    );
}

/// Each check, refused at its edge and accepted one step inside it; the multi-byte fields with
/// only a byte above their low byte set.
#[test]
fn each_check_refuses_at_its_edge() {
    let entry = accepted();
    let refused = |changed: SparseMapRequest| decode(&changed).is_none();
    // 1. the size.
    assert!(refused(SparseMapRequest { size: 0, ..entry }));
    assert!(!refused(SparseMapRequest {
        size: PAGE_SIZE,
        ..entry
    }));
    // 2. alignment, at the mask's lowest and highest bits.
    for misaligned in [1, PAGE_SIZE / 2, PAGE_SIZE - 1] {
        assert!(refused(SparseMapRequest {
            address: entry.address + misaligned,
            ..entry
        }));
        assert!(refused(SparseMapRequest {
            size: entry.size + misaligned,
            ..entry
        }));
        assert!(refused(SparseMapRequest {
            source_offset: entry.source_offset + misaligned,
            ..entry
        }));
    }
    // 3. the cache byte.
    assert!(refused(SparseMapRequest {
        cache: MAX_CACHE + 1,
        ..entry
    }));
    // 4. the read-only byte, the tag mode and the reserved fields.
    assert!(refused(SparseMapRequest {
        read_only: 1,
        ..entry
    }));
    assert!(refused(SparseMapRequest {
        comptag_mode: 1,
        ..entry
    }));
    assert!(refused(SparseMapRequest {
        reserved0: 1,
        ..entry
    }));
    for reserved1 in [1, 0x100] {
        assert!(refused(SparseMapRequest { reserved1, ..entry }));
    }
    for reserved2 in [1, 0x100, 0x1_0000, 0x100_0000] {
        assert!(refused(SparseMapRequest { reserved2, ..entry }));
    }
    // 5. the range's end: one that ends at 2^64 does not fit, one a page lower does.
    let top = 0u64.wrapping_sub(entry.size);
    assert!(refused(SparseMapRequest {
        address: top,
        ..entry
    }));
    assert!(!refused(SparseMapRequest {
        address: top - PAGE_SIZE,
        ..entry
    }));
    // 6. the source's end, likewise.
    assert!(refused(SparseMapRequest {
        source_offset: top,
        ..entry
    }));
    assert!(!refused(SparseMapRequest {
        source_offset: top - PAGE_SIZE,
        ..entry
    }));
    // 7. a source with no handle; a handle with none is fine.
    assert!(refused(SparseMapRequest { memory: 0, ..entry }));
    assert!(!refused(SparseMapRequest {
        source_offset: 0,
        ..entry
    }));
}

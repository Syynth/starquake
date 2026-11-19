//! Tests of the GPU-only store on its own; `vm_memory/tests.rs` drives it through the production
//! memory and the address table.

use super::*;

const PAGE: usize = PAGE_BYTES;

/// Regions with distinct, non-adjacent-looking ids, so a key mix-up shows.
const R: RegionId = RegionId(7);
const BEFORE: RegionId = RegionId(6);
const AFTER: RegionId = RegionId(8);

fn bytes(store: &GpuOnlyStore, region: RegionId, offset: u64, len: usize) -> Vec<u8> {
    let mut out = vec![0xa5; len];
    store.read(region, offset, &mut out).unwrap();
    out
}

#[test]
fn a_fresh_store_holds_nothing_and_reads_zero() {
    let store = GpuOnlyStore::new();
    assert_eq!(store.resident_pages(), 0);
    assert_eq!(store.resident_pages_of(R), 0);
    assert_eq!(bytes(&store, R, 0, PAGE + 3), vec![0; PAGE + 3]);
    // A read allocates nothing.
    assert_eq!(store.resident_pages(), 0);
    let default = GpuOnlyStore::default();
    assert_eq!(default.resident_pages(), 0);
}

#[test]
fn a_write_reads_back_and_its_neighbours_read_zero() {
    let store = GpuOnlyStore::new();
    let at = 5 * PAGE_SIZE + 0x10;
    store.write(R, at, &[1, 2, 3, 4]).unwrap();
    assert_eq!(bytes(&store, R, at - 2, 8), [0, 0, 1, 2, 3, 4, 0, 0]);
    assert_eq!(store.resident_pages(), 1);
    // The page's other bytes, and the pages either side, read zero.
    assert_eq!(bytes(&store, R, 5 * PAGE_SIZE, 0x10), vec![0; 0x10]);
    assert_eq!(bytes(&store, R, 4 * PAGE_SIZE, PAGE), vec![0; PAGE]);
    assert_eq!(bytes(&store, R, 6 * PAGE_SIZE, PAGE), vec![0; PAGE]);
    // Another region at the same offset is another page.
    assert_eq!(bytes(&store, AFTER, at, 4), [0; 4]);
}

#[test]
fn the_first_write_to_a_page_allocates_it_and_later_ones_do_not() {
    let store = GpuOnlyStore::new();
    store.write(R, 0, &[9]).unwrap();
    assert_eq!(store.resident_pages(), 1);
    store.write(R, PAGE_SIZE - 1, &[8]).unwrap();
    assert_eq!(store.resident_pages(), 1);
    // The page's last byte and the next page's first: one more page.
    store.write(R, PAGE_SIZE - 1, &[7, 6]).unwrap();
    assert_eq!(store.resident_pages(), 2);
    assert_eq!(bytes(&store, R, 0, 1), [9]);
    assert_eq!(bytes(&store, R, PAGE_SIZE - 1, 2), [7, 6]);
    // An empty write allocates nothing.
    store.write(BEFORE, 0, &[]).unwrap();
    assert_eq!(store.resident_pages_of(BEFORE), 0);
}

#[test]
fn a_write_across_pages_lands_in_order() {
    let store = GpuOnlyStore::new();
    let data: Vec<u8> = (0..2 * PAGE + 6)
        .map(|i| u8::try_from(i % 251).unwrap() + 1)
        .collect();
    let at = 3 * PAGE_SIZE - 3;
    store.write(R, at, &data).unwrap();
    assert_eq!(store.resident_pages(), 4);
    assert_eq!(bytes(&store, R, at, data.len()), data);
}

#[test]
fn freeing_a_region_frees_its_pages_and_no_one_elses() {
    let store = GpuOnlyStore::new();
    for region in [BEFORE, R, AFTER] {
        store
            .write(region, 0, &[u8::try_from(region.0).unwrap()])
            .unwrap();
        store
            .write(region, 9 * PAGE_SIZE, &[u8::try_from(region.0).unwrap()])
            .unwrap();
    }
    // A page at the very last offset of the region before, and at the very first of the one
    // after: the edges of the region's keys.
    store.write(BEFORE, u64::MAX - PAGE_SIZE, &[0x66]).unwrap();
    store.write(AFTER, 0, &[0x88]).unwrap();
    assert_eq!(store.resident_pages(), 7);

    assert_eq!(store.free_region(R), 2);
    assert_eq!(store.resident_pages_of(R), 0);
    assert_eq!(bytes(&store, R, 0, 1), [0]);
    assert_eq!(bytes(&store, R, 9 * PAGE_SIZE, 1), [0]);
    assert_eq!(store.resident_pages_of(BEFORE), 3);
    assert_eq!(store.resident_pages_of(AFTER), 2);
    assert_eq!(bytes(&store, BEFORE, u64::MAX - PAGE_SIZE, 1), [0x66]);
    assert_eq!(bytes(&store, BEFORE, 9 * PAGE_SIZE, 1), [6]);
    assert_eq!(bytes(&store, AFTER, 0, 1), [0x88]);

    // Freeing a region with nothing in it frees nothing and leaves the others.
    assert_eq!(store.free_region(R), 0);
    assert_eq!(store.resident_pages(), 5);
    // A freed region's next write allocates afresh.
    store.write(R, 0, &[1]).unwrap();
    assert_eq!(store.resident_pages_of(R), 1);
}

#[test]
fn the_first_and_last_region_ids_free_alone() {
    let store = GpuOnlyStore::new();
    let first = RegionId(0);
    let last = RegionId(u32::MAX);
    let next_to_last = RegionId(u32::MAX - 1);
    for region in [first, RegionId(1), next_to_last, last] {
        store.write(region, PAGE_SIZE, &[1]).unwrap();
    }
    assert_eq!(store.free_region(last), 1);
    assert_eq!(store.resident_pages_of(next_to_last), 1);
    assert_eq!(store.free_region(first), 1);
    assert_eq!(store.resident_pages_of(RegionId(1)), 1);
    assert_eq!(store.resident_pages(), 2);
}

#[test]
fn an_access_whose_end_is_past_the_last_offset_is_refused() {
    let store = GpuOnlyStore::new();
    // Ends exactly at `u64::MAX`: representable, accepted.
    store.write(R, u64::MAX - 2, &[1, 2]).unwrap();
    assert_eq!(bytes(&store, R, u64::MAX - 2, 2), [1, 2]);
    // One byte further: refused, with nothing written.
    let error = GpuOnlyRangeError {
        region: R,
        offset: u64::MAX - 2,
        len: 3,
    };
    assert_eq!(store.write(R, u64::MAX - 2, &[3, 4, 5]), Err(error));
    assert_eq!(bytes(&store, R, u64::MAX - 2, 2), [1, 2]);
    let mut buf = [0xa5; 3];
    assert_eq!(store.read(R, u64::MAX - 2, &mut buf), Err(error));
    assert_eq!(buf, [0xa5; 3]);
    assert_eq!(store.resident_pages(), 1);
}

#[test]
fn the_range_error_reads_as_plain_words() {
    let error = GpuOnlyRangeError {
        region: RegionId(3),
        offset: 0xffff_ffff_ffff_fffe,
        len: 0x10,
    };
    assert_eq!(
        error.to_string(),
        "a GPU-only access of 0x10 bytes at offset 0xfffffffffffffffe of sparse region 3 runs \
         past the last offset"
    );
}

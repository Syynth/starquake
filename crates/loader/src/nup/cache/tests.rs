//! The decompressed-file cache: eviction at exactly its cap, least recently used first, room
//! made before a fill, and sizes past the cap refused before a fill.

use std::cell::Cell;

use super::*;

/// A file id for the tests.
fn id(index: u32) -> FileId {
    FileId(index)
}

/// `size` bytes, each `fill`.
fn bytes(size: u64, fill: u8) -> Vec<u8> {
    vec![fill; usize::try_from(size).unwrap()]
}

/// Asks the cache for `file` of `size` bytes, counting fills in `fills`.
fn get(cache: &mut DecompressedCache, file: u32, size: u64, fills: &Cell<u32>) -> Vec<u8> {
    let byte = u8::try_from(file).unwrap();
    cache
        .get_or_fill(id(file), size, || {
            fills.set(fills.get() + 1);
            Ok::<_, ()>(bytes(size, byte))
        })
        .unwrap()
        .to_vec()
}

#[test]
fn files_that_fill_the_cap_exactly_stay_and_one_byte_more_evicts_the_oldest() {
    const CAP: u64 = 10;
    let fills = Cell::new(0);
    let mut cache = DecompressedCache::new(CAP);
    assert_eq!((cache.cap(), cache.held()), (CAP, 0));
    get(&mut cache, 1, 4, &fills);
    get(&mut cache, 2, CAP - 4, &fills);
    // Exactly at the cap: both held.
    assert_eq!(cache.held(), CAP);
    assert!(cache.contains(id(1)) && cache.contains(id(2)));
    // One byte past it: the least recently used goes, and only it.
    get(&mut cache, 3, 1, &fills);
    assert!(!cache.contains(id(1)));
    assert!(cache.contains(id(2)) && cache.contains(id(3)));
    assert_eq!(cache.held(), CAP - 4 + 1);
    assert_eq!(fills.get(), 3);
}

#[test]
fn a_hit_counts_as_a_use_and_a_re_read_after_eviction_returns_the_same_bytes() {
    const CAP: u64 = 10;
    let fills = Cell::new(0);
    let mut cache = DecompressedCache::new(CAP);
    let first = get(&mut cache, 1, 4, &fills);
    get(&mut cache, 2, 3, &fills);
    get(&mut cache, 3, 3, &fills);
    // Using 1 again makes 2 the oldest.
    assert_eq!(get(&mut cache, 1, 4, &fills), first);
    assert_eq!(fills.get(), 3);
    get(&mut cache, 4, 3, &fills);
    assert!(!cache.contains(id(2)));
    assert!(cache.contains(id(1)) && cache.contains(id(3)) && cache.contains(id(4)));
    // A file that needs the whole cap evicts every other one.
    get(&mut cache, 5, CAP, &fills);
    assert_eq!(cache.held(), CAP);
    assert!(!cache.contains(id(1)));
    // File 1 was dropped: it is filled again, with the same bytes.
    assert_eq!(get(&mut cache, 1, 4, &fills), first);
    assert_eq!(fills.get(), 6);
    assert!(!cache.contains(id(5)));
}

#[test]
fn a_file_larger_than_the_cap_is_refused_before_it_is_filled() {
    const CAP: u64 = 10;
    let mut cache = DecompressedCache::new(CAP);
    let fills = Cell::new(0);
    get(&mut cache, 1, 4, &fills);
    for size in [CAP + 1, u64::MAX] {
        let error = cache
            .get_or_fill(id(2), size, || -> Result<Vec<u8>, ()> {
                unreachable!("a file past the cap is never filled")
            })
            .unwrap_err();
        assert_eq!(error, CacheError::TooLarge { size, cap: CAP });
    }
    // Nothing was evicted for it.
    assert!(cache.contains(id(1)));
    assert_eq!(cache.held(), 4);
    // At the cap exactly, it fits.
    assert_eq!(get(&mut cache, 2, CAP, &fills).len(), 10);
}

#[test]
fn room_is_made_before_the_fill_runs() {
    const CAP: u64 = 10;
    let fills = Cell::new(0);
    let mut cache = DecompressedCache::new(CAP);
    get(&mut cache, 1, 4, &fills);
    get(&mut cache, 2, 4, &fills);
    get(&mut cache, 3, 2, &fills);
    // A 5-byte file needs two of them gone first; its fill fails, so what is left is what was
    // left when it ran.
    let error = cache
        .get_or_fill(id(4), 5, || Err::<Vec<u8>, _>("no"))
        .unwrap_err();
    assert_eq!(error, CacheError::Fill("no"));
    assert_eq!(cache.held(), 2);
    assert!(cache.held() <= CAP - 5);
    assert!(!cache.contains(id(1)) && !cache.contains(id(2)));
    assert!(cache.contains(id(3)) && !cache.contains(id(4)));
}

#[test]
fn a_fill_of_the_wrong_size_is_refused_and_not_kept() {
    let mut cache = DecompressedCache::new(10);
    for got in [3u64, 5] {
        let error = cache
            .get_or_fill(id(1), 4, || Ok::<_, ()>(bytes(got, 1)))
            .unwrap_err();
        assert_eq!(error, CacheError::WrongSize { size: 4, got });
        assert_eq!(cache.held(), 0);
        assert!(!cache.contains(id(1)));
    }
}

//! A package's `RomFS` served as an image, on synthetic containers built in the test: reads at
//! any offset, two files read interleaved, the cache at its cap, sticky failures and hostile
//! sizes.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use super::*;
use crate::nup::PackageKind;
use crate::nup::build::{self, Built, Content, PackageSpec, ROMFS_DIR_NAME, Tree};
use crate::read_at::Positional;
use crate::romfs::RomFs;

/// A synthetic title id.
const TITLE_ID: u64 = 0x0100_0000_0000_7000;

/// Bytes whose every value differs from its neighbours', so a read from the wrong offset shows.
fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|at| u8::try_from(at % 251).unwrap().wrapping_mul(7) ^ seed)
        .collect()
}

/// The `RomFS` the tests serve: raw and compressed files, empty files and an empty directory,
/// at several depths. Paths are from the `RomFS` root.
fn files() -> Vec<(&'static str, Content)> {
    vec![
        ("a.bin", Content::raw(pattern(300, 1))),
        ("data/big.zst", Content::zstd(pattern(5000, 2))),
        ("data/small.zst", Content::zstd(pattern(40, 3))),
        ("data/deep/raw.txt", Content::raw(pattern(17, 4))),
        ("data/deep/also.zst", Content::zstd(pattern(1000, 5))),
        ("empty", Content::raw(Vec::new())),
        ("z.bin", Content::raw(pattern(1, 6))),
    ]
}

/// A base package whose tree is `files` under `romfs/`, and an `empty/` directory there.
fn container(files: &[(&str, Content)]) -> Built {
    let tree = files.iter().fold(
        Tree::new()
            .with_dir(&format!("{ROMFS_DIR_NAME}/hollow"))
            .unwrap(),
        |tree, (path, content)| {
            tree.with_file(&format!("{ROMFS_DIR_NAME}/{path}"), content.clone())
                .unwrap()
        },
    );
    build::build(&[PackageSpec::new(PackageKind::Base, TITLE_ID, 0, tree)]).unwrap()
}

/// A source over `bytes` that counts reads and can be made to fail reads from `fail_from` on.
struct Source {
    bytes: Vec<u8>,
    reads: AtomicUsize,
    failing: AtomicBool,
    fail_from: u64,
}

impl Source {
    fn new(bytes: Vec<u8>) -> Arc<Self> {
        Arc::new(Self {
            bytes,
            reads: AtomicUsize::new(0),
            failing: AtomicBool::new(false),
            fail_from: u64::MAX,
        })
    }

    fn failing_from(bytes: Vec<u8>, fail_from: u64) -> Arc<Self> {
        Arc::new(Self {
            bytes,
            reads: AtomicUsize::new(0),
            failing: AtomicBool::new(true),
            fail_from,
        })
    }

    fn reads(&self) -> usize {
        self.reads.load(Ordering::SeqCst)
    }
}

impl ReadAt for Source {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if self.failing.load(Ordering::SeqCst) && offset >= self.fail_from {
            return Err(io::Error::other("the disk is on fire"));
        }
        self.bytes.read_at(offset, buf)
    }

    fn size(&self) -> io::Result<u64> {
        self.bytes.size()
    }
}

/// The storage of `built`'s base `RomFS`, over `source`, with a cache of `cap` bytes.
fn storage(built: &Built, source: Arc<Source>, cap: u64) -> RomFsStorage<Positional<Arc<Source>>> {
    let nup = Nup::open(Positional::new(source)).unwrap();
    let dir = DirId(u32::try_from(built.dir_index(0, ROMFS_DIR_NAME).unwrap()).unwrap());
    RomFsStorage::new(nup, dir, cap).unwrap()
}

/// The storage of the default files with the shipping cap.
fn default_storage() -> (Built, RomFsStorage<Positional<Arc<Source>>>) {
    let built = container(&files());
    let source = Source::new(built.bytes.clone());
    let storage = storage(&built, source, CACHE_CAP);
    (built, storage)
}

/// The whole image, read in one go.
fn whole<S: ReadAt>(storage: &RomFsStorage<S>) -> Vec<u8> {
    let mut out = vec![0; usize::try_from(storage.size()).unwrap()];
    assert_eq!(storage.read_at(0, &mut out).unwrap(), out.len());
    out
}

/// The image parsed by the loader's `RomFS` reader, reading positionally through the storage.
fn parsed<S: ReadAt>(storage: &RomFsStorage<S>) -> RomFs<Positional<&RomFsStorage<S>>> {
    RomFs::new(Positional::new(storage)).unwrap()
}

#[test]
fn the_image_holds_every_file_at_its_path_with_its_bytes() {
    let (_, storage) = default_storage();
    let romfs = parsed(&storage);
    for (path, content) in files() {
        let entry = romfs.find_file(&format!("/{path}")).unwrap().unwrap();
        let mut buf = vec![0xEE; content.data.len()];
        let read = romfs.read_file_shared(&entry, 0, &mut buf).unwrap();
        assert_eq!((read, buf), (content.data.len(), content.data), "{path}");
    }
    assert!(romfs.find_dir("/hollow").unwrap().is_some());
    assert_eq!(romfs.walk().unwrap().files.len(), files().len());
    // The same bytes through the trait a positional reader sees.
    let mut head = [0u8; 8];
    assert_eq!(ReadAt::read_at(&storage, 0, &mut head).unwrap(), 8);
    assert_eq!(u64::from_le_bytes(head), 0x50);
    assert_eq!(ReadAt::size(&storage).unwrap(), storage.size());
}

#[test]
fn reads_at_any_offset_including_a_files_exact_end_and_one_past_it() {
    let (_, storage) = default_storage();
    let image = whole(&storage);
    let size = storage.size();
    // Every offset of the image, a few lengths each, against the one whole read.
    for offset in (0..size + 2).step_by(7).chain([size - 1, size, size + 1]) {
        for len in [1usize, 16, 333] {
            let mut buf = vec![0xEE; len];
            let read = storage.read_at(offset, &mut buf).unwrap();
            let start = usize::try_from(offset).unwrap().min(image.len());
            let end = (start + len).min(image.len());
            assert_eq!(&buf[..read], &image[start..end], "{len} at {offset}");
        }
    }
    // Inside each file: its last byte, its exact end and one past it.
    let romfs = parsed(&storage);
    for (path, content) in files() {
        let entry = romfs.find_file(&format!("/{path}")).unwrap().unwrap();
        let len = u64::try_from(content.data.len()).unwrap();
        let mut one = [0xEE];
        if let Some(last) = content.data.last() {
            assert_eq!(
                romfs.read_file_shared(&entry, len - 1, &mut one).unwrap(),
                1
            );
            assert_eq!(one[0], *last, "{path}");
        }
        assert_eq!(romfs.read_file_shared(&entry, len, &mut one).unwrap(), 0);
        assert_eq!(
            romfs.read_file_shared(&entry, len + 1, &mut one).unwrap(),
            0
        );
    }
}

#[test]
fn two_files_open_at_once_read_interleaved() {
    let (_, storage) = default_storage();
    let romfs = parsed(&storage);
    let big = romfs.find_file("/data/big.zst").unwrap().unwrap();
    let raw = romfs.find_file("/a.bin").unwrap().unwrap();
    let (mut got_big, mut got_raw) = (Vec::new(), Vec::new());
    let mut chunk = [0u8; 37];
    // Alternate between the two, each at its own offset.
    loop {
        let at_big = u64::try_from(got_big.len()).unwrap();
        let at_raw = u64::try_from(got_raw.len()).unwrap();
        let read_big = romfs.read_file_shared(&big, at_big, &mut chunk).unwrap();
        got_big.extend_from_slice(&chunk[..read_big]);
        let read_raw = romfs.read_file_shared(&raw, at_raw, &mut chunk).unwrap();
        got_raw.extend_from_slice(&chunk[..read_raw]);
        if read_big == 0 && read_raw == 0 {
            break;
        }
    }
    assert_eq!(got_big, pattern(5000, 2));
    assert_eq!(got_raw, pattern(300, 1));
}

#[test]
fn two_threads_read_two_files_at_once() {
    let (_, storage) = default_storage();
    let storage = Arc::new(storage);
    let (done, results) = mpsc::channel();
    for (path, expected) in [
        ("/data/deep/also.zst", pattern(1000, 5)),
        ("/a.bin", pattern(300, 1)),
    ] {
        let storage = Arc::clone(&storage);
        let done = done.clone();
        std::thread::spawn(move || {
            let romfs = RomFs::new(Positional::new(&*storage)).unwrap();
            let entry = romfs.find_file(path).unwrap().unwrap();
            let mut got = Vec::new();
            let mut chunk = [0u8; 13];
            loop {
                let at = u64::try_from(got.len()).unwrap();
                let read = romfs.read_file_shared(&entry, at, &mut chunk).unwrap();
                if read == 0 {
                    break;
                }
                got.extend_from_slice(&chunk[..read]);
            }
            done.send(got == expected).unwrap();
        });
    }
    for _ in 0..2 {
        let matched = results.recv_timeout(Duration::from_secs(30)).unwrap();
        assert!(matched);
    }
}

/// The index of `path` (from the `RomFS` root) in `built`'s base package.
fn index(built: &Built, path: &str) -> usize {
    built
        .file_index(0, &format!("{ROMFS_DIR_NAME}/{path}"))
        .unwrap()
}

/// Reads `path`'s whole data through the storage's image, by its placement.
fn read_file<S: ReadAt>(storage: &RomFsStorage<S>, path: &str) -> Result<Vec<u8>, StorageError> {
    let romfs = parsed(storage);
    let entry = romfs.find_file(&format!("/{path}")).unwrap().unwrap();
    let extent = romfs.file_extent(&entry).unwrap();
    let mut out = vec![0; usize::try_from(extent.end - extent.start).unwrap()];
    storage.read_at(extent.start, &mut out)?;
    Ok(out)
}

#[test]
fn the_cache_evicts_at_exactly_its_cap_and_a_re_read_returns_the_same_bytes() {
    let built = container(&files());
    let source = Source::new(built.bytes.clone());
    // Room for `big` and `small` exactly.
    let cap = 5000 + 40;
    let storage = storage(&built, Arc::clone(&source), cap);
    let big = read_file(&storage, "data/big.zst").unwrap();
    read_file(&storage, "data/small.zst").unwrap();
    assert_eq!(storage.cached_bytes(), cap);
    // Both are cached: reading them again reads nothing from the source.
    let reads = source.reads();
    assert_eq!(read_file(&storage, "data/big.zst").unwrap(), big);
    read_file(&storage, "data/small.zst").unwrap();
    assert_eq!(source.reads(), reads);
    // `also` does not fit beside them: `big`, used least recently, goes.
    read_file(&storage, "data/deep/also.zst").unwrap();
    assert_eq!(storage.cached_bytes(), 40 + 1000);
    let reads = source.reads();
    read_file(&storage, "data/small.zst").unwrap();
    assert_eq!(source.reads(), reads, "small stayed");
    // `big` is decompressed again, with the same bytes.
    assert_eq!(read_file(&storage, "data/big.zst").unwrap(), big);
    assert!(source.reads() > reads);
    assert_eq!(big, pattern(5000, 2));
}

#[test]
fn a_compressed_file_larger_than_the_cache_is_an_error_in_words() {
    let built = container(&files());
    let cap = 4999;
    let storage = storage(&built, Source::new(built.bytes.clone()), cap);
    let error = read_file(&storage, "data/big.zst").unwrap_err();
    let file = index(&built, "data/big.zst");
    assert_eq!(
        error,
        StorageError::TooLargeForCache {
            file,
            size: 5000,
            cap
        }
    );
    assert_eq!(
        error.to_string(),
        format!(
            "file {file} of the RomFS is 5000 bytes decompressed, more than the 4999-byte cache \
             of decompressed files holds"
        )
    );
    // Raw files do not go through the cache, however large.
    let storage = storage_with_cap(&built, 0);
    assert_eq!(read_file(&storage, "a.bin").unwrap(), pattern(300, 1));
}

fn storage_with_cap(built: &Built, cap: u64) -> RomFsStorage<Positional<Arc<Source>>> {
    storage(built, Source::new(built.bytes.clone()), cap)
}

#[test]
fn a_hostile_size_cannot_force_an_allocation_past_the_cap() {
    // The record of `small` claims a terabyte; the frame still holds 40 bytes. The size is
    // compared with the cap before anything is allocated or decompressed.
    let mut built = container(&files());
    let small = index(&built, "data/small.zst");
    built.edit_file(small, |record| record.size = 1 << 40);
    let source = Source::new(built.bytes.clone());
    let storage = storage(&built, Arc::clone(&source), 1 << 20);
    let romfs = parsed(&storage);
    let entry = romfs.find_file("/data/small.zst").unwrap().unwrap();
    assert_eq!(entry.size, 1 << 40);
    let reads = source.reads();
    let mut buf = [0u8; 16];
    let start = romfs.file_extent(&entry).unwrap().start;
    let error = storage.read_at(start, &mut buf).unwrap_err();
    assert_eq!(
        error,
        StorageError::TooLargeForCache {
            file: small,
            size: 1 << 40,
            cap: 1 << 20
        }
    );
    assert_eq!(source.reads(), reads, "nothing was read for it");
    assert_eq!(storage.cached_bytes(), 0);

    // Under the cap but not what the frame holds: the frame's own size is checked first.
    let mut built = container(&files());
    built.edit_file(small, |record| record.size = 41);
    let storage = storage_with_cap(&built, 1 << 20);
    let error = read_file(&storage, "data/small.zst").unwrap_err();
    assert_eq!(
        error,
        StorageError::Damaged {
            file: small,
            message: format!(
                "file {small}'s zstd frame declares 40 bytes of content, not its size 41"
            ),
        }
    );
    assert_eq!(storage.cached_bytes(), 0);
}

#[test]
fn a_failed_read_is_sticky_for_its_file_and_no_other() {
    // `small`'s stored frame decodes to other bytes than its hash says.
    let mut entries = files();
    let damaged = build::zstd_frame(&pattern(40, 9)).unwrap();
    entries[2].1 = Content::zstd(pattern(40, 3)).with_stored(damaged);
    let built = container(&entries);
    let storage = storage_with_cap(&built, CACHE_CAP);
    let small = index(&built, "data/small.zst");
    let expected = StorageError::Damaged {
        file: small,
        message: format!("file {small}'s content does not match its SHA-256"),
    };
    // Read twice: the same error both times.
    assert_eq!(read_file(&storage, "data/small.zst"), Err(expected.clone()));
    assert_eq!(read_file(&storage, "data/small.zst"), Err(expected.clone()));
    assert_eq!(storage.cached_bytes(), 0);
    // Other files read as ever.
    assert_eq!(
        read_file(&storage, "data/big.zst").unwrap(),
        pattern(5000, 2)
    );
    assert_eq!(read_file(&storage, "a.bin").unwrap(), pattern(300, 1));
    // A read that spans the damaged file and another fails, whole.
    let romfs = parsed(&storage);
    let entry = romfs.find_file("/data/small.zst").unwrap().unwrap();
    let start = romfs.file_extent(&entry).unwrap().start;
    let mut span = vec![0u8; 4096];
    assert_eq!(storage.read_at(start - 8, &mut span), Err(expected));
}

#[test]
fn a_source_failure_is_sticky_even_once_the_source_recovers() {
    let built = container(&files());
    let a_bin = index(&built, "a.bin");
    let (extent_at, _) = built.layout.extents[a_bin];
    // The source fails at `a.bin`'s extent and past it.
    let source = Source::failing_from(built.bytes.clone(), u64::try_from(extent_at).unwrap());
    let storage = storage(&built, Arc::clone(&source), CACHE_CAP);
    let error = read_file(&storage, "a.bin").unwrap_err();
    let expected = StorageError::Io {
        file: a_bin,
        message: "reading the container failed: the disk is on fire".to_owned(),
    };
    assert_eq!(error, expected);
    source.failing.store(false, Ordering::SeqCst);
    // The source works again; the file's reads still fail the same way.
    assert_eq!(read_file(&storage, "a.bin"), Err(expected.clone()));
    // Through the positional trait, the error is carried in an `io::Error`.
    let romfs = parsed(&storage);
    let entry = romfs.find_file("/a.bin").unwrap().unwrap();
    let start = romfs.file_extent(&entry).unwrap().start;
    let error = ReadAt::read_at(&storage, start, &mut [0u8; 4]).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert_eq!(error.to_string(), expected.to_string());
    // A file it never failed on reads.
    assert_eq!(read_file(&storage, "z.bin").unwrap(), pattern(1, 6));
}

#[test]
fn a_directory_from_another_container_is_refused() {
    let built = container(&files());
    let nup = Nup::open(Positional::new(built.bytes.clone())).unwrap();
    let past = nup.dir_count();
    let error = RomFsStorage::new(nup, DirId(u32::try_from(past).unwrap()), CACHE_CAP).unwrap_err();
    assert_eq!(error, StorageError::NoSuchDir(past));
    assert_eq!(
        error.to_string(),
        format!("the RomFS directory {past} is not in this container")
    );
}

#[test]
fn a_tree_that_cannot_be_an_image_is_refused_in_words() {
    // A file whose path passes the RomFS limit.
    let name = "n".repeat(crate::romfs::MAX_PATH_LEN);
    let built = container(&[(name.as_str(), Content::raw(b"x".to_vec()))]);
    let nup = Nup::open(Positional::new(built.bytes.clone())).unwrap();
    let dir = DirId(u32::try_from(built.dir_index(0, ROMFS_DIR_NAME).unwrap()).unwrap());
    let error = RomFsStorage::new(nup, dir, CACHE_CAP).unwrap_err();
    assert_eq!(
        error.to_string(),
        "the RomFS could not be laid out as an image: a path of the RomFS is 769 bytes, longer \
         than the 768 a path may be"
    );
}

#[test]
fn building_the_image_reads_nothing_from_the_source() {
    let built = container(&files());
    let source = Source::new(built.bytes.clone());
    let nup = Nup::open(Positional::new(Arc::clone(&source))).unwrap();
    let reads = source.reads();
    let dir = DirId(u32::try_from(built.dir_index(0, ROMFS_DIR_NAME).unwrap()).unwrap());
    let storage = RomFsStorage::new(nup, dir, CACHE_CAP).unwrap();
    assert_eq!(source.reads(), reads);
    // The header and tables read without the source too.
    let (metadata_at, _) = storage.image.metadata();
    let mut tables = vec![0u8; usize::try_from(storage.size() - metadata_at).unwrap()];
    storage.read_at(metadata_at, &mut tables).unwrap();
    storage.read_at(0, &mut [0u8; 0x50]).unwrap();
    assert_eq!(source.reads(), reads);
    assert_eq!(storage.nup().file_count(), built.layout.files.len());
}

//! The `RomFS` builder: golden bytes for a small tree, a build-then-parse round trip through the
//! loader's own `RomFS` reader, and each limit at its edge.

use std::collections::BTreeSet;
use std::io::Cursor;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use super::*;
use crate::romfs::{HASH_SEED, RomFs};

/// The tree of the golden image: directories `/c`, `/d` and `/d/e`, files `/y`, `/d/x` and
/// `/d/w`, listed out of path order so that the sort shows.
const GOLDEN_DIRS: [DirIn<'static>; 4] = [
    DirIn {
        parent: 0,
        name: b"",
    },
    DirIn {
        parent: 0,
        name: b"d",
    },
    DirIn {
        parent: 1,
        name: b"e",
    },
    DirIn {
        parent: 0,
        name: b"c",
    },
];

/// The golden tree's files; the key is the file's content.
fn golden_files() -> Vec<FileIn<'static, &'static [u8]>> {
    vec![
        FileIn {
            parent: 0,
            name: b"y",
            size: 3,
            key: b"YYY",
        },
        FileIn {
            parent: 1,
            name: b"x",
            size: 0x11,
            key: b"@ABCDEFGHIJKLMNOP",
        },
        FileIn {
            parent: 1,
            name: b"w",
            size: 5,
            key: b"wwwww",
        },
    ]
}

/// The image's bytes, file data filled from each file's key.
fn bytes_of(image: &Image<&[u8]>) -> Vec<u8> {
    let mut out = vec![0xEE; usize::try_from(image.size()).unwrap()];
    let read = image
        .read_with(0, &mut out, |placed, offset, part| {
            let start = usize::try_from(offset).unwrap();
            part.copy_from_slice(&placed.key[start..start + part.len()]);
            Ok::<(), ()>(())
        })
        .unwrap();
    assert_eq!(read, out.len());
    out
}

/// The golden image's header fields, in order: `header_size`, then offset and size of the
/// directory buckets, the directory entries, the file buckets and the file entries, then the
/// data offset. Computed by an independent model of Eden's `RomFSBuildContext::Build`.
const GOLDEN_HEADER: [u64; 10] = [0x50, 0x234, 20, 0x248, 108, 0x2B4, 12, 0x2C0, 108, 0x200];

/// The golden image's metadata, from the same model: five directory buckets, four directory
/// entries (`""`, `c`, `d`, `d/e`), three file buckets, three file entries (`d/w`, `d/x`, `y`).
const GOLDEN_METADATA: [u8; 0xF8] = [
    0x50, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x18, 0x00, 0x00, 0x00,
    0x48, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x34, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF,
    0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00,
    0x64, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0x34, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x65, 0x00, 0x00, 0x00,
    0x48, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x24, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00,
    0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00, 0x77, 0x00, 0x00, 0x00,
    0x34, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00,
    0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x30, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x79, 0x00, 0x00, 0x00,
];

#[test]
fn a_small_tree_builds_to_these_bytes() {
    let image = build(&GOLDEN_DIRS, &golden_files()).unwrap();
    let mut expected = vec![0u8; 0x32C];
    for (field, value) in expected[..HEADER_SIZE]
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(GOLDEN_HEADER)
    {
        field.copy_from_slice(&value.to_le_bytes());
    }
    // `d/w` at the partition's start, `d/x` at the next 16-byte boundary, `y` after it.
    expected[0x200..0x205].copy_from_slice(b"wwwww");
    expected[0x210..0x221].copy_from_slice(b"@ABCDEFGHIJKLMNOP");
    expected[0x230..0x233].copy_from_slice(b"YYY");
    expected[0x234..].copy_from_slice(&GOLDEN_METADATA);
    assert_eq!(bytes_of(&image), expected);
    assert_eq!(image.size(), 0x32C);
    assert_eq!(image.metadata(), (0x234, &GOLDEN_METADATA[..]));
    let placed: Vec<(u64, u64)> = image.files().iter().map(|f| (f.offset, f.size)).collect();
    assert_eq!(placed, [(0x200, 5), (0x210, 0x11), (0x230, 3)]);
    // The shipping constants are the ones the bytes show.
    assert_eq!(GOLDEN_HEADER[9], FILE_PARTITION_OFFSET);
    assert_eq!(GOLDEN_HEADER[0], u64::try_from(HEADER_SIZE).unwrap());
}

#[test]
fn hash_bucket_counts_follow_edens_rule() {
    // Below 3 entries, 3 buckets; below 19, the next odd count; then the first count no prime
    // up to 17 divides.
    let expected = [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 5),
        (17, 17),
        (18, 19),
        (19, 19),
        (20, 23),
        (24, 29),
        (289, 293),
        (361, 361),
    ];
    for (entries, count) in expected {
        assert_eq!(hash_table_count(entries), count, "{entries} entries");
    }
}

/// A tree of `depth` nested directories and a file at the bottom, every name one byte except
/// the file's, which is `name_len` bytes.
fn nested(depth: usize, name_len: usize) -> (Vec<DirIn<'static>>, Vec<u8>) {
    let mut dirs = vec![DirIn {
        parent: 0,
        name: b"",
    }];
    for index in 1..=depth {
        dirs.push(DirIn {
            parent: index - 1,
            name: b"n",
        });
    }
    (dirs, vec![b'f'; name_len])
}

#[test]
fn a_path_at_the_limit_builds_and_one_past_it_does_not() {
    // `/f...`: one separator and the name.
    let at_limit = vec![b'f'; MAX_PATH_LEN - 1];
    let file = |name: &'static [u8]| FileIn {
        parent: 0,
        name,
        size: 1,
        key: (),
    };
    let at_limit: &'static [u8] = Box::leak(at_limit.into_boxed_slice());
    let root = [GOLDEN_DIRS[0]];
    assert!(build(&root, &[file(at_limit)]).is_ok());
    let past: &'static [u8] = Box::leak(vec![b'f'; MAX_PATH_LEN].into_boxed_slice());
    assert_eq!(
        build(&root, &[file(past)]),
        Err(BuildError::PathTooLong {
            len: MAX_PATH_LEN + 1
        })
    );
    assert_eq!(
        BuildError::PathTooLong {
            len: MAX_PATH_LEN + 1
        }
        .to_string(),
        "a path of the RomFS is 769 bytes, longer than the 768 a path may be"
    );

    // The same through nesting: ten directories of `/n` and the file's `/` and name.
    let (dirs, name) = nested(10, MAX_PATH_LEN - 21);
    let name: &'static [u8] = Box::leak(name.into_boxed_slice());
    let deep = FileIn {
        parent: 10,
        name,
        size: 0,
        key: (),
    };
    assert!(build(&dirs, &[deep]).is_ok());
    let (dirs, name) = nested(10, MAX_PATH_LEN - 20);
    let name: &'static [u8] = Box::leak(name.into_boxed_slice());
    let deep = FileIn {
        parent: 10,
        name,
        size: 0,
        key: (),
    };
    assert_eq!(
        build(&dirs, &[deep]),
        Err(BuildError::PathTooLong {
            len: MAX_PATH_LEN + 1
        })
    );
    // A directory's own path counts too: one directory named to the limit, and one past it.
    let long: &'static [u8] = Box::leak(vec![b'd'; MAX_PATH_LEN - 1].into_boxed_slice());
    let longer: &'static [u8] = Box::leak(vec![b'd'; MAX_PATH_LEN].into_boxed_slice());
    let with = |name| [GOLDEN_DIRS[0], DirIn { parent: 0, name }];
    assert!(build::<()>(&with(long), &[]).is_ok());
    assert_eq!(
        build::<()>(&with(longer), &[]),
        Err(BuildError::PathTooLong {
            len: MAX_PATH_LEN + 1
        })
    );
}

#[test]
fn parents_must_come_first_and_exist() {
    let root = GOLDEN_DIRS[0];
    assert_eq!(build::<()>(&[], &[]), Err(BuildError::NoRoot));
    // A directory naming itself, or one after it, as its parent.
    let own = [
        root,
        DirIn {
            parent: 1,
            name: b"a",
        },
    ];
    assert_eq!(
        build::<()>(&own, &[]),
        Err(BuildError::BadParent {
            kind: "directory",
            index: 1,
            parent: 1
        })
    );
    let later = [
        root,
        DirIn {
            parent: 2,
            name: b"a",
        },
        DirIn {
            parent: 0,
            name: b"b",
        },
    ];
    assert!(matches!(
        build::<()>(&later, &[]),
        Err(BuildError::BadParent { index: 1, .. })
    ));
    // A file in a directory that does not exist; one in the last that does is fine.
    let file = |parent| FileIn {
        parent,
        name: b"f",
        size: 0,
        key: (),
    };
    let error = build(&[root], &[file(1)]).unwrap_err();
    assert_eq!(
        error,
        BuildError::BadParent {
            kind: "file",
            index: 0,
            parent: 1
        }
    );
    assert_eq!(
        error.to_string(),
        "file 0 of the RomFS tree names parent 1, which is not before it"
    );
    assert!(build(&[root], &[file(0)]).is_ok());
}

#[test]
fn the_metadata_limit_holds_at_its_edge() {
    let files = golden_files();
    let size = u64::try_from(GOLDEN_METADATA.len()).unwrap();
    assert!(build_within(&GOLDEN_DIRS, &files, size).is_ok());
    let limit = size - 1;
    assert_eq!(
        build_within(&GOLDEN_DIRS, &files, limit),
        Err(BuildError::MetadataTooLarge { size, limit })
    );
    assert_eq!(
        BuildError::MetadataTooLarge { size, limit }.to_string(),
        "the RomFS's tables would be 248 bytes, more than the 247 this emulator builds"
    );
    // The shipping limit keeps every entry offset inside a `u32`.
    assert!(MAX_METADATA_SIZE < u64::from(u32::MAX));
}

/// One root file of `size` bytes, named `a`.
fn one_file(size: u64) -> Result<Image<()>, BuildError> {
    let file = FileIn {
        parent: 0,
        name: b"a",
        size,
        key: (),
    };
    build(&[GOLDEN_DIRS[0]], &[file])
}

#[test]
fn file_sizes_that_overflow_the_image_are_refused_at_the_exact_edge() {
    let metadata = one_file(1).unwrap().metadata().1.len();
    let metadata = u64::try_from(metadata).unwrap();
    // The largest file that fits: the metadata, 4-aligned, ends exactly at `u64::MAX`'s edge.
    let largest = ((u64::MAX - metadata) & !(TABLE_ALIGNMENT - 1)) - FILE_PARTITION_OFFSET;
    let image = one_file(largest).unwrap();
    assert!(image.size() > u64::MAX - TABLE_ALIGNMENT - metadata);
    assert_eq!(one_file(largest + 1), Err(BuildError::TooLarge));
    assert_eq!(one_file(u64::MAX), Err(BuildError::TooLarge));
    // Two files whose sizes add up past `u64::MAX`.
    let two = [
        FileIn {
            parent: 0,
            name: b"a",
            size: u64::MAX - 8,
            key: (),
        },
        FileIn {
            parent: 0,
            name: b"b",
            size: 16,
            key: (),
        },
    ];
    assert_eq!(build(&[GOLDEN_DIRS[0]], &two), Err(BuildError::TooLarge));
    assert_eq!(
        BuildError::TooLarge.to_string(),
        "the RomFS's files add up to more bytes than an image can hold"
    );
}

#[test]
fn a_name_with_high_bytes_hashes_as_the_console_reads_it() {
    // `é` in Latin-1: a byte that differs signed and unsigned.
    let name: &[u8] = &[0xE9];
    let file = FileIn {
        parent: 0,
        name,
        size: 1,
        key: (),
    };
    let image = build(&[GOLDEN_DIRS[0]], &[file]).unwrap();
    let (_, metadata) = image.metadata();
    let buckets = hash_table_count(1);
    let file_hash_at = usize::try_from(ONE_FILE_FILE_HASH_AT).unwrap();
    let bucket_of = |hash: u32| usize::try_from(u64::from(hash) % buckets).unwrap();
    // The console's hash: over unsigned bytes.
    let unsigned = path_hash(0, name);
    // Eden's builder on a host whose `char` is signed: each byte sign-extended.
    let signed = name.iter().fold(HASH_SEED, |hash, &byte| {
        hash.rotate_right(5) ^ i32::from(i8::from_le_bytes([byte])).cast_unsigned()
    });
    assert_ne!(bucket_of(unsigned), bucket_of(signed));
    let word = |bucket: usize| {
        let at = file_hash_at + bucket * 4;
        u32::from_le_bytes(metadata[at..at + 4].try_into().unwrap())
    };
    assert_eq!(word(bucket_of(unsigned)), 0);
    assert_eq!(word(bucket_of(signed)), ENTRY_EMPTY);
}

/// Where the file buckets start in the metadata of a tree of the root and one file: after the
/// root's three buckets and its 0x18-byte entry.
const ONE_FILE_FILE_HASH_AT: u64 = 3 * 4 + 0x18;

#[test]
fn reads_at_any_offset_match_the_whole_image() {
    let image = build(&GOLDEN_DIRS, &golden_files()).unwrap();
    let whole = bytes_of(&image);
    let size = image.size();
    for offset in 0..=size + 1 {
        for len in [0usize, 1, 7, 0x40] {
            let mut buf = vec![0xEE; len];
            let read = image
                .read_with(offset, &mut buf, |placed, into, part| {
                    let start = usize::try_from(into).unwrap();
                    part.copy_from_slice(&placed.key[start..start + part.len()]);
                    Ok::<(), ()>(())
                })
                .unwrap();
            let start = usize::try_from(offset).unwrap().min(whole.len());
            let end = (start + len).min(whole.len());
            assert_eq!(&buf[..read], &whole[start..end], "{len} at {offset:#x}");
        }
    }
    assert!(image.segment_at(size).is_none());
    assert!(image.segment_at(size - 1).is_some());
}

#[test]
fn a_file_callback_error_is_the_reads_error() {
    let image = build(&GOLDEN_DIRS, &golden_files()).unwrap();
    let mut buf = [0u8; 0x20];
    // The header alone reads without touching a file.
    let read = image.read_with(0, &mut buf, |_, _, _| Err("touched"));
    assert_eq!(read, Ok(0x20));
    let read = image.read_with(0x1F0, &mut buf, |_, _, _| Err("touched"));
    assert_eq!(read, Err("touched"));
}

/// A generated tree: up to six directories, each under an earlier one, and up to eight files,
/// with short names from a small alphabet so that names share prefixes and collide in buckets.
/// Generated directories: each one's parent and name.
type Dirs = Vec<(usize, String)>;
/// Generated files: each one's parent, name and content.
type Files = Vec<(usize, String, Vec<u8>)>;

fn trees() -> impl Strategy<Value = (Dirs, Files)> {
    let name = "[ab.][ab/.]{0,2}".prop_map(|name: String| name.replace('/', "-"));
    let dirs = prop::collection::vec((any::<prop::sample::Index>(), name.clone()), 0..6);
    let files = prop::collection::vec(
        (
            any::<prop::sample::Index>(),
            name,
            prop::collection::vec(any::<u8>(), 0..40),
        ),
        0..8,
    );
    (dirs, files).prop_map(|(dirs, files)| {
        let dirs: Vec<(usize, String)> = dirs
            .into_iter()
            .enumerate()
            .map(|(at, (parent, name))| (parent.index(at + 1), name))
            .collect();
        let count = dirs.len() + 1;
        let files = files
            .into_iter()
            .map(|(parent, name, data)| (parent.index(count), name, data))
            .collect();
        (dirs, files)
    })
}

fn config() -> Config {
    Config {
        cases: 256,
        rng_seed: RngSeed::Fixed(0x524F_4D46),
        failure_persistence: None,
        ..Config::default()
    }
}

/// Every path of a generated tree, with names made unique per directory.
fn unique(dirs: &[(usize, String)], files: &[(usize, String, Vec<u8>)]) -> (Dirs, Files) {
    let mut taken: BTreeSet<(usize, String)> = BTreeSet::new();
    let mut claim = |parent: usize, name: &str, salt: usize| {
        // `.` and `..` are not names a lookup path may hold.
        let mut name = if name.is_empty() || name == "." || name == ".." {
            "e".to_owned()
        } else {
            name.to_owned()
        };
        while !taken.insert((parent, name.clone())) {
            name.push_str(&salt.to_string());
        }
        name
    };
    let dirs = dirs
        .iter()
        .enumerate()
        .map(|(at, (parent, name))| (*parent, claim(*parent, name, at)))
        .collect();
    let files = files
        .iter()
        .enumerate()
        .map(|(at, (parent, name, data))| (*parent, claim(*parent, name, at + 100), data.clone()))
        .collect();
    (dirs, files)
}

proptest! {
    #![proptest_config(config())]

    /// Build, then parse with the loader's `RomFS` reader: every directory and file is found by
    /// its path through the hash tables, listed under its parent, and reads back its bytes, so
    /// a swapped field, link or offset fails.
    #[test]
    fn a_built_image_parses_back_to_its_tree((dirs, files) in trees()) {
        let (dirs, files) = unique(&dirs, &files);
        let mut dir_in = vec![GOLDEN_DIRS[0]];
        dir_in.extend(dirs.iter().map(|(parent, name)| DirIn { parent: *parent, name: name.as_bytes() }));
        let file_in: Vec<FileIn<'_, &[u8]>> = files
            .iter()
            .map(|(parent, name, data)| FileIn {
                parent: *parent,
                name: name.as_bytes(),
                size: u64::try_from(data.len()).unwrap(),
                key: data.as_slice(),
            })
            .collect();
        let image = build(&dir_in, &file_in).unwrap();
        let bytes = bytes_of(&image);
        let mut romfs = RomFs::new(Cursor::new(bytes)).unwrap();

        let mut paths = vec![String::new()];
        for (parent, name) in &dirs {
            let path = format!("{}/{name}", paths[*parent]);
            paths.push(path);
        }
        let tree = romfs.walk().unwrap();
        prop_assert_eq!(tree.dirs.len(), paths.len());
        prop_assert_eq!(tree.files.len(), files.len());
        for path in &paths {
            let shown = if path.is_empty() { "/" } else { path.as_str() };
            prop_assert!(romfs.find_dir(shown).unwrap().is_some(), "{}", shown);
        }
        for (parent, name, data) in &files {
            let path = format!("{}/{name}", paths[*parent]);
            let entry = romfs.find_file(&path).unwrap().unwrap();
            prop_assert_eq!(&romfs.read_file(&entry).unwrap(), data);
            prop_assert_eq!(entry.offset % FILE_ALIGNMENT, 0);
        }
        // Each directory lists its files and subdirectories in path order.
        for (path, dir) in &tree.dirs {
            let contents = romfs.read_dir(dir).unwrap();
            let names: Vec<&str> = contents.files.iter().map(|f| f.name.as_str()).collect();
            let mut sorted = names.clone();
            sorted.sort_unstable();
            prop_assert_eq!(names, sorted, "files of {}", path);
            let names: Vec<&str> = contents.dirs.iter().map(|d| d.name.as_str()).collect();
            let mut sorted = names.clone();
            sorted.sort_unstable();
            prop_assert_eq!(names, sorted, "directories of {}", path);
        }
    }
}

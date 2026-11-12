//! Unit tests for the NUP reader. Every container here is synthetic, built by [`build`] from
//! the specification; nothing is derived from a real title.

use std::io::{Cursor, Read};

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use super::build::{self, Built, Content, Entry, PackageSpec, Tree};
use super::layout::{
    CHILD_SLOT_SIZE, Codec, DIR_RECORD_SIZE, EXTENT_ALIGN, FILE_RECORD_SIZE, MAGIC,
    PACKAGE_FLAG_STANDALONE_UPDATE, PACKAGE_RECORD_SIZE, PackageKind, RawDir, RawFile, RawPackage,
    RawSuperblock, RawTocHeader, SUPERBLOCK_SIZE, TOC_HEADER_SIZE, TOC_MAGIC,
};
use super::*;

/// Fixed seed so property runs are reproducible.
const PROPTEST_SEED: u64 = 0x5EED_0000_0095;

fn config(cases: u32) -> Config {
    Config {
        cases,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        ..Config::default()
    }
}

const BASE_ID: u64 = 0x0100_0000_0000_1000;
const UPDATE_ID: u64 = 0x0100_0000_0000_1800;
const ADDON_ID: u64 = 0x0100_0000_0000_2001;

/// Bytes that compress well, so a zstd extent is smaller than its content.
fn compressible(len: usize) -> Vec<u8> {
    (0..len).map(|i| b"starquake "[i % 10]).collect()
}

fn base_tree() -> Tree {
    Tree::new()
        .with_file("exefs/main", Content::raw(b"base main".to_vec()))
        .unwrap()
        .with_file("exefs/main.npdm", Content::raw(b"base npdm".to_vec()))
        .unwrap()
        .with_file("exefs/subsdk9", Content::raw(b"base only".to_vec()))
        .unwrap()
        .with_file("romfs/data/level.bin", Content::zstd(compressible(4000)))
        .unwrap()
        .with_file("romfs/data/empty", Content::raw(Vec::new()))
        .unwrap()
        .with_file("romfs/readme.txt", Content::raw(b"hello".to_vec()))
        .unwrap()
        .with_file("romfs/copy.txt", Content::raw(b"hello".to_vec()))
        .unwrap()
}

fn update_tree() -> Tree {
    Tree::new()
        .with_file("exefs/main", Content::raw(b"update main".to_vec()))
        .unwrap()
        .with_file("exefs/main.npdm", Content::zstd(compressible(600)))
        .unwrap()
        .with_file("romfs/readme.txt", Content::raw(b"hello again".to_vec()))
        .unwrap()
}

/// A base (version 0) and an update (version 0x20000, 2.0.0).
fn sample_specs() -> [PackageSpec; 2] {
    [
        PackageSpec::new(PackageKind::Base, BASE_ID, 0, base_tree()),
        PackageSpec::new(PackageKind::Update, UPDATE_ID, 0x2_0000, update_tree()),
    ]
}

fn sample() -> Built {
    build::build(&sample_specs()).unwrap()
}

fn open(bytes: &[u8]) -> Result<Nup<Cursor<&[u8]>>, NupError> {
    Nup::open(Cursor::new(bytes))
}

fn open_built(built: &Built) -> Result<Nup<Cursor<&[u8]>>, NupError> {
    open(&built.bytes)
}

fn file_id(built: &Built, package: usize, path: &str) -> FileId {
    FileId(u32::try_from(built.file_index(package, path).unwrap()).unwrap())
}

// ---------------------------------------------------------------------------------------------
// Record layouts
// ---------------------------------------------------------------------------------------------

/// Little-endian bytes of `value`, `width` bytes wide, for writing golden layouts by hand.
fn le(value: u64, width: usize) -> Vec<u8> {
    value.to_le_bytes()[..width].to_vec()
}

#[test]
fn the_superblock_layout_is_pinned_by_its_golden_bytes() {
    let record = RawSuperblock {
        magic: MAGIC,
        header_size: 0x0102,
        header_version: 0x0304,
        toc_offset: 0x0506_0708_090A_0B0C,
        toc_size: 0x1112_1314_1516_1718,
        toc_sha256: [0x21; 32],
        blob_offset: 0x3132_3334_3536_3738,
        file_size: 0x4142_4344_4546_4748,
        package_count: 0x5152_5354,
        flags: 0x6162_6364,
        uuid: [0x71; 16],
        reserved: [0x81; 32],
    };
    let mut golden = b"NUP\x01".to_vec();
    golden.extend(le(0x0102, 2));
    golden.extend(le(0x0304, 2));
    golden.extend(le(0x0506_0708_090A_0B0C, 8));
    golden.extend(le(0x1112_1314_1516_1718, 8));
    golden.extend([0x21; 32]);
    golden.extend(le(0x3132_3334_3536_3738, 8));
    golden.extend(le(0x4142_4344_4546_4748, 8));
    golden.extend(le(0x5152_5354, 4));
    golden.extend(le(0x6162_6364, 4));
    golden.extend([0x71; 16]);
    golden.extend([0x81; 32]);
    assert_eq!(golden.len(), SUPERBLOCK_SIZE);
    assert_eq!(record.encode().to_vec(), golden);
    assert_eq!(RawSuperblock::decode(&golden).unwrap(), record);
}

#[test]
fn the_toc_header_layout_is_pinned_by_its_golden_bytes() {
    let record = RawTocHeader {
        magic: TOC_MAGIC,
        package_count: 0x0102_0304,
        dir_count: 0x1112_1314_1516_1718,
        file_count: 0x2122_2324_2526_2728,
        string_pool_offset: 0x3132_3334_3536_3738,
        string_pool_size: 0x4142_4344_4546_4748,
        reserved: 0x5152_5354_5556_5758,
    };
    let mut golden = b"TOC\x01".to_vec();
    golden.extend(le(0x0102_0304, 4));
    for word in [
        0x1112_1314_1516_1718,
        0x2122_2324_2526_2728,
        0x3132_3334_3536_3738,
        0x4142_4344_4546_4748,
        0x5152_5354_5556_5758,
    ] {
        golden.extend(le(word, 8));
    }
    assert_eq!(golden.len(), TOC_HEADER_SIZE);
    assert_eq!(record.encode().to_vec(), golden);
    assert_eq!(RawTocHeader::decode(&golden).unwrap(), record);
}

#[test]
fn the_package_record_layout_is_pinned_by_its_golden_bytes() {
    let record = RawPackage {
        title_id: 0x0102_0304_0506_0708,
        root_dir: 0x1112_1314,
        exefs_dir: 0x2122_2324,
        romfs_dir: 0x3132_3334,
        version: 0x4142_4344,
        kind: 0x51,
        flags: 0x52,
        reserved: 0x5354,
        reserved2: 0x6162_6364,
        meta_offset: 0x7172_7374_7576_7778,
        meta_size: 0x8182_8384,
        reserved3: [0x91; 20],
    };
    let mut golden = le(0x0102_0304_0506_0708, 8);
    golden.extend(le(0x1112_1314, 4));
    golden.extend(le(0x2122_2324, 4));
    golden.extend(le(0x3132_3334, 4));
    golden.extend(le(0x4142_4344, 4));
    golden.extend([0x51, 0x52]);
    golden.extend(le(0x5354, 2));
    golden.extend(le(0x6162_6364, 4));
    golden.extend(le(0x7172_7374_7576_7778, 8));
    golden.extend(le(0x8182_8384, 4));
    golden.extend([0x91; 20]);
    assert_eq!(golden.len(), PACKAGE_RECORD_SIZE);
    assert_eq!(record.encode().to_vec(), golden);
    assert_eq!(RawPackage::decode(&golden).unwrap(), record);
}

#[test]
fn the_dir_record_layout_is_pinned_by_its_golden_bytes() {
    let record = RawDir {
        name_offset: 0x0102_0304,
        name_len: 0x1112_1314,
        first_child: 0x2122_2324,
        child_count: 0x3132_3334,
        reserved: 0x4142_4344_4546_4748,
    };
    let mut golden = le(0x0102_0304, 4);
    golden.extend(le(0x1112_1314, 4));
    golden.extend(le(0x2122_2324, 4));
    golden.extend(le(0x3132_3334, 4));
    golden.extend(le(0x4142_4344_4546_4748, 8));
    assert_eq!(golden.len(), DIR_RECORD_SIZE);
    assert_eq!(record.encode().to_vec(), golden);
    assert_eq!(RawDir::decode(&golden).unwrap(), record);
}

#[test]
fn the_file_record_layout_is_pinned_by_its_golden_bytes() {
    let record = RawFile {
        name_offset: 0x0102_0304,
        name_len: 0x1112_1314,
        size: 0x2122_2324_2526_2728,
        extent_offset: 0x3132_3334_3536_3738,
        extent_size: 0x4142_4344_4546_4748,
        sha256: [0x51; 32],
        codec: 0x61,
        codec_flags: 0x62,
        reserved2: 0x6364,
        reserved3: [0x71; 12],
    };
    let mut golden = le(0x0102_0304, 4);
    golden.extend(le(0x1112_1314, 4));
    golden.extend(le(0x2122_2324_2526_2728, 8));
    golden.extend(le(0x3132_3334_3536_3738, 8));
    golden.extend(le(0x4142_4344_4546_4748, 8));
    golden.extend([0x51; 32]);
    golden.extend([0x61, 0x62]);
    golden.extend(le(0x6364, 2));
    golden.extend([0x71; 12]);
    assert_eq!(golden.len(), FILE_RECORD_SIZE);
    assert_eq!(record.encode().to_vec(), golden);
    assert_eq!(RawFile::decode(&golden).unwrap(), record);
}

#[test]
fn kind_and_codec_bytes_are_the_specs_and_the_next_value_is_reserved() {
    // SPEC §3.1.2 and §4.1.
    let kinds: Vec<(u8, &str)> = PackageKind::ALL
        .iter()
        .map(|k| (k.to_byte(), k.name()))
        .collect();
    assert_eq!(
        kinds,
        [
            (0, "base"),
            (1, "update"),
            (2, "addon"),
            (3, "addon_update"),
            (4, "system")
        ]
    );
    for kind in PackageKind::ALL {
        assert_eq!(PackageKind::from_byte(kind.to_byte()), Some(kind));
    }
    let first_reserved = u8::try_from(PackageKind::ALL.len()).unwrap();
    assert_eq!(PackageKind::from_byte(first_reserved), None);
    assert_eq!(PackageKind::from_byte(u8::MAX), None);

    let codecs: Vec<(u8, &str)> = Codec::ALL.iter().map(|c| (c.to_byte(), c.name())).collect();
    assert_eq!(codecs, [(0, "raw"), (1, "zstd")]);
    for codec in Codec::ALL {
        assert_eq!(Codec::from_byte(codec.to_byte()), Some(codec));
    }
    assert_eq!(
        Codec::from_byte(u8::try_from(Codec::ALL.len()).unwrap()),
        None
    );
}

#[test]
fn each_decoder_refuses_a_record_one_byte_short() {
    type Decode = fn(&[u8]) -> Result<(), NupError>;
    let cases: [(usize, Decode, &str); 5] = [
        (
            SUPERBLOCK_SIZE,
            |b| RawSuperblock::decode(b).map(drop),
            "superblock",
        ),
        (
            TOC_HEADER_SIZE,
            |b| RawTocHeader::decode(b).map(drop),
            "table of contents header",
        ),
        (
            PACKAGE_RECORD_SIZE,
            |b| RawPackage::decode(b).map(drop),
            "package",
        ),
        (
            DIR_RECORD_SIZE,
            |b| RawDir::decode(b).map(drop),
            "directory",
        ),
        (FILE_RECORD_SIZE, |b| RawFile::decode(b).map(drop), "file"),
    ];
    for (size, decode, name) in cases {
        let bytes = vec![0u8; size];
        assert!(decode(&bytes).is_ok(), "{name} at its size");
        let error = decode(&bytes[..size - 1]).unwrap_err();
        assert!(
            matches!(error, NupError::RecordTruncated { record } if record == name),
            "{name}: {error}"
        );
    }
}

proptest! {
    #![proptest_config(config(256))]

    /// Build then parse: a decoder that read two fields in the wrong order would not give back
    /// what was encoded.
    #[test]
    fn every_record_round_trips(
        words in proptest::array::uniform8(any::<u64>()),
        bytes in proptest::collection::vec(any::<u8>(), 128),
    ) {
        let [a, b, c, d, e, f, g, h] = words;
        let lo = |w: u64| u32::try_from(w & 0xFFFF_FFFF).unwrap();
        let hi = |w: u64| u32::try_from(w >> 32).unwrap();
        let half = |w: u64| u16::try_from(w & 0xFFFF).unwrap();
        let byte = |w: u64| u8::try_from(w & 0xFF).unwrap();
        let array = |at: usize| -> [u8; 32] { bytes[at..at + 32].try_into().unwrap() };

        let superblock = RawSuperblock {
            magic: [byte(a), byte(b), byte(c), byte(d)],
            header_size: half(a >> 8),
            header_version: half(b >> 8),
            toc_offset: c,
            toc_size: d,
            toc_sha256: array(0),
            blob_offset: e,
            file_size: f,
            package_count: lo(g),
            flags: hi(g),
            uuid: array(32)[..16].try_into().unwrap(),
            reserved: array(64),
        };
        prop_assert_eq!(RawSuperblock::decode(&superblock.encode()).unwrap(), superblock);

        let header = RawTocHeader {
            magic: [byte(h), byte(g), byte(f), byte(e)],
            package_count: lo(a),
            dir_count: b,
            file_count: c,
            string_pool_offset: d,
            string_pool_size: e,
            reserved: f,
        };
        prop_assert_eq!(RawTocHeader::decode(&header.encode()).unwrap(), header);

        let package = RawPackage {
            title_id: a,
            root_dir: lo(b),
            exefs_dir: hi(b),
            romfs_dir: lo(c),
            version: hi(c),
            kind: byte(d),
            flags: byte(d >> 8),
            reserved: half(d >> 16),
            reserved2: hi(d),
            meta_offset: e,
            meta_size: lo(f),
            reserved3: array(96)[..20].try_into().unwrap(),
        };
        prop_assert_eq!(RawPackage::decode(&package.encode()).unwrap(), package);

        let dir = RawDir {
            name_offset: lo(a),
            name_len: hi(a),
            first_child: lo(b),
            child_count: hi(b),
            reserved: c,
        };
        prop_assert_eq!(RawDir::decode(&dir.encode()).unwrap(), dir);

        let file = RawFile {
            name_offset: lo(d),
            name_len: hi(d),
            size: e,
            extent_offset: f,
            extent_size: g,
            sha256: array(16),
            codec: byte(h),
            codec_flags: byte(h >> 8),
            reserved2: half(h >> 16),
            reserved3: array(48)[..12].try_into().unwrap(),
        };
        prop_assert_eq!(RawFile::decode(&file.encode()).unwrap(), file);
    }
}

// ---------------------------------------------------------------------------------------------
// Opening and reading a good container
// ---------------------------------------------------------------------------------------------

#[test]
fn a_base_and_an_update_open_with_their_packages_and_trees() {
    let built = sample();
    let nup = open_built(&built).unwrap();

    let packages = nup.packages();
    assert_eq!(packages.len(), 2);
    let base = &packages[0];
    assert_eq!(
        (
            base.index,
            base.kind,
            base.title_id,
            base.version,
            base.standalone_update
        ),
        (0, PackageKind::Base, BASE_ID, 0, false)
    );
    let update = &packages[1];
    assert_eq!(
        (update.kind, update.title_id, update.version),
        (PackageKind::Update, UPDATE_ID, 0x2_0000)
    );
    assert_eq!(update.version_parts(), (2, 0, 0));
    assert_eq!(base.meta, None);

    let dir_of = |package: usize, path: &str| {
        DirId(u32::try_from(built.dir_index(package, path).unwrap()).unwrap())
    };
    assert_eq!(base.exefs, Some(dir_of(0, "exefs")));
    assert_eq!(base.romfs, Some(dir_of(0, "romfs")));
    assert_eq!(update.root, dir_of(1, ""));

    let names = |dir| -> Vec<String> {
        nup.children(dir)
            .iter()
            .map(|&n| String::from_utf8(nup.name(n).to_vec()).unwrap())
            .collect()
    };
    assert_eq!(names(base.root), ["exefs", "romfs"]);
    assert_eq!(
        names(dir_of(0, "romfs")),
        ["copy.txt", "data", "readme.txt"]
    );

    let walked: Vec<(String, FileId)> = nup
        .walk_files(base.root)
        .into_iter()
        .map(|(path, file)| (String::from_utf8(path).unwrap(), file))
        .collect();
    let expected: Vec<(String, FileId)> = [
        "exefs/main",
        "exefs/main.npdm",
        "exefs/subsdk9",
        "romfs/copy.txt",
        "romfs/data/empty",
        "romfs/data/level.bin",
        "romfs/readme.txt",
    ]
    .iter()
    .map(|path| ((*path).to_owned(), file_id(&built, 0, path)))
    .collect();
    assert_eq!(walked, expected);

    assert_eq!(
        nup.resolve(base.root, "romfs/data/level.bin"),
        Some(Node::File(file_id(&built, 0, "romfs/data/level.bin")))
    );
    assert_eq!(
        nup.resolve(base.root, "/romfs//readme.txt/"),
        Some(Node::File(file_id(&built, 0, "romfs/readme.txt")))
    );
    assert_eq!(nup.resolve(base.root, "romfs/missing"), None);
    assert_eq!(nup.resolve(base.root, "romfs/readme.txt/below"), None);
    assert_eq!(
        nup.lookup(base.root, b"exefs"),
        Some(Node::Dir(dir_of(0, "exefs")))
    );
    assert_eq!(nup.dir_count(), built.layout.dir_records.len());
    assert_eq!(nup.file_count(), built.layout.file_records.len());
}

#[test]
fn stored_and_compressed_files_read_back_through_the_streaming_reader() {
    let built = sample();
    let mut nup = open_built(&built).unwrap();

    let raw = file_id(&built, 0, "romfs/readme.txt");
    assert_eq!(nup.file(raw).unwrap().codec, Codec::Raw);
    assert_eq!(nup.read_to_vec(raw, u64::MAX).unwrap(), b"hello");

    let zstd = file_id(&built, 0, "romfs/data/level.bin");
    let entry = nup.file(zstd).unwrap().clone();
    assert_eq!(entry.codec, Codec::Zstd);
    assert!(entry.extent_size < entry.size, "{entry:?}");
    // In small reads, as a streaming consumer would.
    let mut reader = nup.reader(zstd).unwrap();
    let mut out = Vec::new();
    let mut chunk = [0u8; 7];
    loop {
        let count = reader.read(&mut chunk).unwrap();
        if count == 0 {
            break;
        }
        out.extend_from_slice(&chunk[..count]);
    }
    assert_eq!(out, compressible(4000));

    let empty = file_id(&built, 0, "romfs/data/empty");
    assert_eq!(nup.read_to_vec(empty, 0).unwrap(), b"");
    nup.verify_file(empty).unwrap();
}

#[test]
fn read_to_vec_refuses_a_file_over_its_limit_and_accepts_one_at_it() {
    let built = sample();
    let mut nup = open_built(&built).unwrap();
    let file = file_id(&built, 0, "romfs/readme.txt");
    let size = nup.file(file).unwrap().size;
    assert_eq!(nup.read_to_vec(file, size).unwrap().len(), 5);
    let error = nup.read_to_vec(file, size - 1).unwrap_err();
    assert!(
        matches!(
            error,
            NupError::FileTooLarge {
                size: 5,
                limit: 4,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn a_file_id_from_another_container_is_an_error_not_a_panic() {
    let built = sample();
    let mut nup = open_built(&built).unwrap();
    let foreign = FileId(u32::try_from(nup.file_count()).unwrap());
    assert!(nup.file(foreign).is_none());
    assert_eq!(nup.name(Node::File(foreign)), b"");
    assert_eq!(nup.children(DirId(u32::MAX)).len(), 0);
    let error = nup.reader(foreign).unwrap_err();
    assert!(matches!(error, NupError::NoSuchFile(index) if index == nup.file_count()));
    assert!(matches!(
        nup.read_to_vec(foreign, 1),
        Err(NupError::NoSuchFile(_))
    ));
}

#[test]
fn verify_checks_every_file_and_hashes_a_shared_extent_once() {
    let built = sample();
    // `romfs/copy.txt` and `romfs/readme.txt` hold the same bytes, so they share an extent.
    assert_eq!(
        built.layout.extents[built.file_index(0, "romfs/copy.txt").unwrap()],
        built.layout.extents[built.file_index(0, "romfs/readme.txt").unwrap()]
    );
    let mut nup = open_built(&built).unwrap();
    let summary = nup.verify().unwrap();
    let files = built.layout.file_records.len();
    let distinct: std::collections::BTreeSet<_> = built
        .layout
        .extents
        .iter()
        .filter(|&&(_, size)| size != 0)
        .collect();
    let bytes: u64 = [
        "base main".len(),
        "base npdm".len(),
        "base only".len(),
        4000,
        "hello".len(), // once for the shared extent
        "update main".len(),
        600,
        "hello again".len(),
    ]
    .iter()
    .map(|&n| to_u64(n))
    .sum();
    assert_eq!(
        summary,
        VerifySummary {
            files: to_u64(files),
            extents: to_u64(distinct.len()),
            bytes,
        }
    );
    assert_eq!(
        distinct.len(),
        files - 2,
        "one shared extent, one empty file"
    );
}

#[test]
fn a_meta_block_is_located_but_not_interpreted() {
    let mut base = PackageSpec::new(PackageKind::Base, BASE_ID, 0, base_tree());
    base.meta = vec![0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xAB, 0xCD];
    let built = build::build(&[base]).unwrap();
    let nup = open_built(&built).unwrap();
    let (offset, size) = built.layout.meta[0].unwrap();
    assert_eq!(
        nup.packages()[0].meta,
        Some(to_u64(offset)..to_u64(offset + size))
    );
}

// ---------------------------------------------------------------------------------------------
// The title: base, newest update, ExeFS
// ---------------------------------------------------------------------------------------------

fn exefs_names<S>(nup: &Nup<S>, title: &Title) -> Vec<String> {
    title
        .exefs_files(nup)
        .into_iter()
        .map(|f| String::from_utf8(nup.name(Node::File(f)).to_vec()).unwrap())
        .collect()
}

#[test]
fn an_update_replaces_the_whole_exefs() {
    let built = sample();
    let mut nup = open_built(&built).unwrap();
    let title = nup.title().unwrap();
    assert_eq!(
        (title.base, title.update, title.program),
        (Some(0), Some(1), 0)
    );
    let exefs = title.exefs.unwrap();
    assert_eq!(exefs.package, 1);
    // `subsdk9` is only in the base's ExeFS: Eden replaces the ExeFS, it does not merge it
    // (the divergence from the spec's path shadowing, in the `title` module docs).
    assert_eq!(exefs_names(&nup, &title), ["main", "main.npdm"]);
    assert_eq!(title.exefs_file(&nup, "subsdk9"), None);
    let main = title.exefs_file(&nup, "main").unwrap();
    assert_eq!(main, file_id(&built, 1, "exefs/main"));
    assert_eq!(nup.read_to_vec(main, u64::MAX).unwrap(), b"update main");
    let npdm = title.exefs_file(&nup, "main.npdm").unwrap();
    assert_eq!(nup.read_to_vec(npdm, u64::MAX).unwrap(), compressible(600));
}

#[test]
fn without_an_update_the_exefs_is_the_bases() {
    let built =
        build::build(&[PackageSpec::new(PackageKind::Base, BASE_ID, 0, base_tree())]).unwrap();
    let nup = open_built(&built).unwrap();
    let title = nup.title().unwrap();
    assert_eq!(
        (title.base, title.update, title.program),
        (Some(0), None, 0)
    );
    assert_eq!(exefs_names(&nup, &title), ["main", "main.npdm", "subsdk9"]);
    assert_eq!(
        title.exefs_file(&nup, "subsdk9"),
        Some(file_id(&built, 0, "exefs/subsdk9"))
    );
}

#[test]
fn an_update_without_an_exefs_keeps_the_bases() {
    let romfs_only = Tree::new()
        .with_file("romfs/patch.bin", Content::raw(b"patch".to_vec()))
        .unwrap();
    let built = build::build(&[
        PackageSpec::new(PackageKind::Base, BASE_ID, 0, base_tree()),
        PackageSpec::new(PackageKind::Update, UPDATE_ID, 0x1_0000, romfs_only),
    ])
    .unwrap();
    let nup = open_built(&built).unwrap();
    assert_eq!(nup.packages()[1].exefs, None);
    let title = nup.title().unwrap();
    assert_eq!(title.update, Some(1));
    assert_eq!(title.exefs.unwrap().package, 0);
}

#[test]
fn the_newest_of_several_updates_wins() {
    let update = |version, main: &str| {
        let tree = Tree::new()
            .with_file("exefs/main", Content::raw(main.as_bytes().to_vec()))
            .unwrap();
        PackageSpec::new(PackageKind::Update, UPDATE_ID, version, tree)
    };
    let built = build::build(&[
        update(0x1_0000, "v1"),
        PackageSpec::new(PackageKind::Base, BASE_ID, 0, base_tree()),
        update(0x3_0000, "v3"),
        update(0x2_0000, "v2"),
    ])
    .unwrap();
    let mut nup = open_built(&built).unwrap();
    let title = nup.title().unwrap();
    assert_eq!(
        (title.base, title.update, title.program),
        (Some(1), Some(2), 1)
    );
    let main = title.exefs_file(&nup, "main").unwrap();
    assert_eq!(nup.read_to_vec(main, u64::MAX).unwrap(), b"v3");
}

#[test]
fn two_updates_sharing_the_newest_version_are_ambiguous_but_older_twins_are_not() {
    let update = |version| {
        let tree = Tree::new()
            .with_file("exefs/main", Content::raw(vec![1]))
            .unwrap();
        PackageSpec::new(PackageKind::Update, UPDATE_ID, version, tree)
    };
    let base = || PackageSpec::new(PackageKind::Base, BASE_ID, 0, base_tree());
    let built = build::build(&[base(), update(7), update(7)]).unwrap();
    let error = open_built(&built).unwrap().title().unwrap_err();
    assert!(
        matches!(
            error,
            NupError::AmbiguousUpdate {
                first: 1,
                second: 2,
                version: 7
            }
        ),
        "{error}"
    );
    let built = build::build(&[base(), update(7), update(7), update(8)]).unwrap();
    assert_eq!(open_built(&built).unwrap().title().unwrap().update, Some(3));
}

#[test]
fn a_standalone_update_is_the_program_of_a_container_without_a_base() {
    let mut update = PackageSpec::new(PackageKind::Update, UPDATE_ID, 0x2_0000, update_tree());
    update.flags = PACKAGE_FLAG_STANDALONE_UPDATE;
    let built = build::build(&[update]).unwrap();
    let nup = open_built(&built).unwrap();
    assert!(nup.packages()[0].standalone_update);
    let title = nup.title().unwrap();
    assert_eq!(
        (title.base, title.update, title.program),
        (None, Some(0), 0)
    );
    assert_eq!(title.exefs.unwrap().package, 0);
}

#[test]
fn a_container_of_add_ons_holds_no_program() {
    let addon = Tree::new()
        .with_file("romfs/dlc.bin", Content::raw(vec![1, 2, 3]))
        .unwrap();
    let built = build::build(&[PackageSpec::new(PackageKind::AddOn, ADDON_ID, 0, addon)]).unwrap();
    let error = open_built(&built).unwrap().title().unwrap_err();
    assert!(matches!(error, NupError::NoProgram), "{error}");
    assert_eq!(
        error.to_string(),
        "the container holds no program: no base and no update package"
    );
}

// ---------------------------------------------------------------------------------------------
// The superblock
// ---------------------------------------------------------------------------------------------

#[test]
fn a_source_shorter_than_the_superblock_is_refused() {
    let built = sample();
    let short = &built.bytes[..SUPERBLOCK_SIZE - 1];
    let error = open(short).unwrap_err();
    assert!(
        matches!(error, NupError::SourceTooShort { len } if len == to_u64(SUPERBLOCK_SIZE - 1)),
        "{error}"
    );
    // At exactly the superblock's size the next check speaks: its file size is wrong.
    let error = open(&built.bytes[..SUPERBLOCK_SIZE]).unwrap_err();
    assert!(
        matches!(error, NupError::FileSizeMismatch { .. }),
        "{error}"
    );
}

#[test]
fn each_bad_superblock_field_has_its_own_error() {
    let check = |edit: fn(&mut RawSuperblock), want: fn(&NupError) -> bool| {
        let mut built = sample();
        built.edit_superblock(edit);
        let error = open_built(&built).unwrap_err();
        assert!(want(&error), "{error}");
    };
    check(
        |s| s.magic = *b"NUQ\x01",
        |e| matches!(e, NupError::BadMagic(m) if m == b"NUQ\x01"),
    );
    check(
        |s| s.magic = *b"NUP\x02",
        |e| matches!(e, NupError::UnsupportedVersion(2)),
    );
    check(
        |s| s.header_size = u16::try_from(SUPERBLOCK_SIZE - 1).unwrap(),
        |e| matches!(e, NupError::HeaderTooSmall(size) if usize::from(*size) == SUPERBLOCK_SIZE - 1),
    );
    check(
        |s| s.reserved[31] = 1,
        |e| matches!(e, NupError::SuperblockReservedNotZero),
    );
    check(
        |s| s.flags = 1,
        |e| matches!(e, NupError::UnknownSuperblockFlags(1)),
    );
    check(
        |s| s.file_size += 1,
        |e| matches!(e, NupError::FileSizeMismatch { declared, actual } if *declared == *actual + 1),
    );
    check(
        |s| s.package_count = 0,
        |e| matches!(e, NupError::NoPackages),
    );
    check(
        |s| s.toc_sha256[0] ^= 1,
        |e| matches!(e, NupError::TocHashMismatch),
    );
}

#[test]
fn a_header_bigger_than_version_1s_moves_the_table_of_contents_with_it() {
    // SPEC §2 rules 2 and 3: `header_size` may exceed 0x80, and `toc_offset` must equal it.
    let mut built = sample();
    built.edit_superblock(|s| s.header_size += 16);
    let error = open_built(&built).unwrap_err();
    assert!(
        matches!(error, NupError::TocNotAfterHeader { toc_offset, header_size }
            if toc_offset == to_u64(SUPERBLOCK_SIZE) && usize::from(header_size) == SUPERBLOCK_SIZE + 16),
        "{error}"
    );
}

#[test]
fn a_header_bigger_than_version_1s_opens_when_the_table_of_contents_follows_it() {
    // SPEC §2 rules 2 and 3, the accepting side: `header_size` above 0x80 with the table of
    // contents really at `header_size` opens, and its content reads back.
    let header_size = SUPERBLOCK_SIZE + 16;
    let built = build::build_with_header(&sample_specs(), header_size).unwrap();
    assert_eq!(built.layout.toc_offset, header_size);
    assert!(
        built.bytes[SUPERBLOCK_SIZE..header_size]
            .iter()
            .all(|&b| b == 0)
    );
    let mut nup = open_built(&built).unwrap();
    assert_eq!(usize::from(nup.superblock().header_size), header_size);
    assert_eq!(nup.superblock().toc_offset, to_u64(header_size));
    let plain = open_built(&sample()).unwrap().verify().unwrap();
    assert_eq!(nup.verify().unwrap(), plain);
    let title = nup.title().unwrap();
    assert_eq!(exefs_names(&nup, &title), ["main", "main.npdm"]);
    // A header below the superblock's size cannot be built.
    assert_eq!(
        build::build_with_header(&sample_specs(), SUPERBLOCK_SIZE - 1).unwrap_err(),
        build::BuildError::NotAContainer
    );
}

#[test]
fn the_table_of_contents_must_end_inside_the_file() {
    let mut built = sample();
    let file_size = to_u64(built.bytes.len());
    // Ending exactly at the end of the file is allowed by this check (the blob check then
    // speaks); one byte further is not.
    built.edit_superblock(|s| s.toc_size = file_size - s.toc_offset);
    assert!(matches!(
        open_built(&built).unwrap_err(),
        NupError::BlobOverlapsToc { .. }
    ));
    built.edit_superblock(|s| s.toc_size += 1);
    assert!(matches!(
        open_built(&built).unwrap_err(),
        NupError::TocOutOfBounds
    ));
    built.edit_superblock(|s| s.toc_size = u64::MAX);
    assert!(matches!(
        open_built(&built).unwrap_err(),
        NupError::TocOutOfBounds
    ));
}

#[test]
fn the_blob_region_is_aligned_after_the_table_of_contents_and_inside_the_file() {
    let built = sample();
    let toc_end = to_u64(built.layout.toc_offset + built.layout.toc_size);
    let blob = to_u64(built.layout.blob_offset);

    let mut misaligned = built.clone();
    misaligned.edit_superblock(|s| s.blob_offset += 1);
    let error = open_built(&misaligned).unwrap_err();
    assert!(
        matches!(error, NupError::BlobMisaligned(at) if at == blob + 1),
        "{error}"
    );

    // The aligned offset just below the end of the table of contents overlaps it.
    let below = (toc_end - 1) / EXTENT_ALIGN * EXTENT_ALIGN;
    let mut overlapping = built.clone();
    overlapping.edit_superblock(|s| s.blob_offset = below);
    let error = open_built(&overlapping).unwrap_err();
    assert!(
        matches!(error, NupError::BlobOverlapsToc { blob_offset, toc_end: end } if blob_offset == below && end == toc_end),
        "{error}"
    );

    // The first aligned offset past the end of the file is refused, whether or not the file's
    // size is itself a multiple of the alignment.
    let mut past = built.clone();
    let file_size = to_u64(built.bytes.len());
    let beyond = (file_size / EXTENT_ALIGN + 1) * EXTENT_ALIGN;
    assert!(beyond > file_size && beyond - file_size <= EXTENT_ALIGN);
    past.edit_superblock(|s| s.blob_offset = beyond);
    assert!(matches!(
        open_built(&past).unwrap_err(),
        NupError::BlobOutOfBounds
    ));
}

#[test]
fn a_blob_region_may_start_exactly_at_the_end_of_the_file() {
    // Only empty files: no extents and no meta block, so the blob region is empty and starts
    // where the file ends. That is the accepted edge of `blob_offset <= file_size`.
    let tree = Tree::new()
        .with_file("romfs/empty", Content::raw(Vec::new()))
        .unwrap();
    let built = build::build(&[PackageSpec::new(PackageKind::Base, BASE_ID, 0, tree)]).unwrap();
    assert_eq!(built.layout.blob_offset, built.bytes.len());
    let mut nup = open_built(&built).unwrap();
    assert_eq!(
        nup.verify().unwrap(),
        VerifySummary {
            files: 1,
            extents: 0,
            bytes: 0
        }
    );
    // One alignment step further is past the end.
    let mut past = built.clone();
    past.edit_superblock(|s| s.blob_offset += EXTENT_ALIGN);
    assert!(matches!(
        open_built(&past).unwrap_err(),
        NupError::BlobOutOfBounds
    ));
}

#[test]
fn a_blob_offset_must_be_aligned_to_the_full_alignment() {
    // Every offset that is a multiple of a smaller power of two but not of `EXTENT_ALIGN`.
    let built = sample();
    let blob = to_u64(built.layout.blob_offset);
    for bit in 0..EXTENT_ALIGN.ilog2() {
        let delta = 1u64 << bit;
        let mut misaligned = built.clone();
        misaligned.edit_superblock(|s| s.blob_offset += delta);
        let error = open_built(&misaligned).unwrap_err();
        assert!(
            matches!(error, NupError::BlobMisaligned(at) if at == blob + delta),
            "+{delta}: {error}"
        );
    }
}

#[test]
fn the_table_of_contents_limit_holds_at_its_edge() {
    let built = sample();
    let size = to_u64(built.layout.toc_size);
    let at = Limits { max_toc_size: size };
    assert!(Nup::open_with_limits(Cursor::new(&built.bytes[..]), at).is_ok());
    let below = Limits {
        max_toc_size: size - 1,
    };
    let error = Nup::open_with_limits(Cursor::new(&built.bytes[..]), below).unwrap_err();
    assert!(
        matches!(error, NupError::TocTooLarge { size: s, limit } if s == size && limit == size - 1),
        "{error}"
    );
    assert_eq!(Limits::default().max_toc_size, DEFAULT_MAX_TOC_SIZE);
}

// ---------------------------------------------------------------------------------------------
// The table of contents header
// ---------------------------------------------------------------------------------------------

#[test]
fn each_bad_toc_header_field_has_its_own_error() {
    let check = |edit: &dyn Fn(&mut RawTocHeader, &Built), want: &dyn Fn(&NupError) -> bool| {
        let mut built = sample();
        let snapshot = built.clone();
        built.edit_toc_header(|h| edit(h, &snapshot));
        let error = open_built(&built).unwrap_err();
        assert!(want(&error), "{error}");
    };
    check(
        &|h, _| h.magic = *b"TOC\x02",
        &|e| matches!(e, NupError::BadTocMagic(m) if m == b"TOC\x02"),
    );
    check(&|h, _| h.package_count += 1, &|e| {
        matches!(
            e,
            NupError::PackageCountMismatch {
                superblock: 2,
                toc: 3
            }
        )
    });
    check(&|h, _| h.file_count = u64::MAX, &|e| {
        matches!(
            e,
            NupError::TocTruncated {
                needed: u64::MAX,
                ..
            }
        )
    });
    // One record more than fits.
    check(
        &|h, b| {
            let room = b.layout.toc_size - (b.layout.children_offset - b.layout.toc_offset);
            h.dir_count += to_u64(room / DIR_RECORD_SIZE + 1);
        },
        &|e| matches!(e, NupError::TocTruncated { .. }),
    );
    // The pool may not start inside the records, nor end past the table.
    check(
        &|h, b| h.string_pool_offset = to_u64(b.layout.children_offset - b.layout.toc_offset - 1),
        &|e| matches!(e, NupError::StringPoolOutOfBounds),
    );
    check(&|h, _| h.string_pool_size += 1, &|e| {
        matches!(e, NupError::StringPoolOutOfBounds)
    });
    check(&|h, _| h.string_pool_size = u64::MAX, &|e| {
        matches!(e, NupError::StringPoolOutOfBounds)
    });
    // A pool that starts one byte later leaves a children array of a partial slot.
    check(
        &|h, _| {
            h.string_pool_offset += 1;
            h.string_pool_size -= 1;
        },
        &|e| matches!(e, NupError::ChildrenMisaligned(bytes) if bytes % to_u64(CHILD_SLOT_SIZE) == 1),
    );
}

#[test]
fn a_table_of_contents_shorter_than_its_header_is_refused() {
    let mut built = sample();
    built.edit_superblock(|s| s.toc_size = to_u64(TOC_HEADER_SIZE - 1));
    // The table of contents hash must match the shortened table for the check to be reached.
    build::rehash_toc(&mut built.bytes).unwrap();
    let error = open_built(&built).unwrap_err();
    assert!(
        matches!(
            error,
            NupError::RecordTruncated {
                record: "table of contents header"
            }
        ),
        "{error}"
    );
}

// ---------------------------------------------------------------------------------------------
// Packages
// ---------------------------------------------------------------------------------------------

fn package_error(index: usize, edit: impl FnOnce(&mut RawPackage)) -> NupError {
    let mut built = sample();
    built.edit_package(index, edit);
    open_built(&built).unwrap_err()
}

#[test]
fn each_bad_package_field_has_its_own_error() {
    let reserved_kind = u8::try_from(PackageKind::ALL.len()).unwrap();
    let error = package_error(1, |p| p.kind = reserved_kind);
    assert!(
        matches!(error, NupError::UnknownPackageKind { package: 1, kind } if kind == reserved_kind),
        "{error}"
    );
    let error = package_error(1, |p| p.flags = 0x02);
    assert!(
        matches!(
            error,
            NupError::UnknownPackageFlags {
                package: 1,
                flags: 0x02
            }
        ),
        "{error}"
    );
    let error = package_error(0, |p| p.reserved = 1);
    assert!(
        matches!(error, NupError::PackageReservedNotZero { package: 0 }),
        "{error}"
    );
    let error = package_error(0, |p| p.reserved2 = 1);
    assert!(
        matches!(error, NupError::PackageReservedNotZero { package: 0 }),
        "{error}"
    );
}

#[test]
fn the_highest_defined_kind_is_accepted() {
    let mut built = sample();
    let highest = PackageKind::ALL[PackageKind::ALL.len() - 1];
    built.edit_package(1, |p| p.kind = highest.to_byte());
    assert_eq!(open_built(&built).unwrap().packages()[1].kind, highest);
}

/// An edit of one directory field of a package record.
type Edit = fn(&mut RawPackage, u32);

#[test]
fn a_package_whose_root_is_an_empty_directory_opens() {
    // No files and no child slots: the children array is empty, so the string pool starts
    // exactly where the records end, and the blob region starts where the file ends.
    let built =
        build::build(&[PackageSpec::new(PackageKind::Base, BASE_ID, 0, Tree::new())]).unwrap();
    assert_eq!(built.layout.children_offset, built.layout.pool_offset);
    let mut nup = open_built(&built).unwrap();
    assert_eq!((nup.dir_count(), nup.file_count()), (1, 0));
    let root = nup.packages()[0].root;
    assert_eq!(nup.children(root), &[]);
    assert_eq!(nup.walk_files(root), Vec::new());
    assert_eq!(nup.verify().unwrap(), VerifySummary::default());
}

#[test]
fn a_package_directory_must_exist_and_no_dir_means_none() {
    let built = sample();
    let dirs = u32::try_from(built.layout.dir_records.len()).unwrap();
    let edits: [(&str, Edit); 3] = [
        ("root_dir", |p, d| p.root_dir = d),
        ("exefs_dir", |p, d| p.exefs_dir = d),
        ("romfs_dir", |p, d| p.romfs_dir = d),
    ];
    for (field, edit) in edits {
        let error = package_error(0, |p| edit(p, dirs));
        assert!(
            matches!(&error, NupError::DirOutOfRange { package: 0, field: f, dir } if *f == field && *dir == dirs),
            "{error}"
        );
    }
    let mut built = sample();
    built.edit_package(0, |p| p.romfs_dir = super::layout::NO_DIR);
    assert_eq!(open_built(&built).unwrap().packages()[0].romfs, None);
}

#[test]
fn a_meta_block_lies_between_the_table_of_contents_and_the_end_of_the_file() {
    // The builder puts the meta block after the extents, ending exactly at the end of the file.
    let mut base = PackageSpec::new(PackageKind::Base, BASE_ID, 0, base_tree());
    base.meta = vec![0x5A; 24];
    let built = build::build(&[base]).unwrap();
    let (offset, size) = built.layout.meta[0].unwrap();
    assert_eq!(offset + size, built.bytes.len());
    assert!(open_built(&built).is_ok());
    let mut past = built.clone();
    past.edit_package(0, |p| p.meta_size += 1);
    let error = open_built(&past).unwrap_err();
    assert!(
        matches!(error, NupError::MetaOutOfBounds { package: 0 }),
        "{error}"
    );

    // A meta block may also sit in the padding between the table of contents and the blob
    // region, starting exactly at the end of the table of contents (the spec only gives its
    // offset; see "Meta blocks" in the module docs).
    let sample = sample();
    let toc_end = to_u64(sample.layout.toc_offset + sample.layout.toc_size);
    let gap = to_u64(sample.layout.blob_offset) - toc_end;
    assert!(
        gap > 0,
        "the sample's blob region is padded after its table of contents"
    );
    let mut in_gap = sample.clone();
    in_gap.edit_package(0, |p| {
        p.meta_offset = toc_end;
        p.meta_size = u32::try_from(gap).unwrap();
    });
    assert!(open_built(&in_gap).is_ok());

    let error = package_error(0, |p| {
        p.meta_offset = toc_end - 1;
        p.meta_size = 1;
    });
    assert!(
        matches!(error, NupError::MetaOutOfBounds { package: 0 }),
        "{error}"
    );
    let error = package_error(0, |p| {
        p.meta_offset = u64::MAX;
        p.meta_size = 1;
    });
    assert!(
        matches!(error, NupError::MetaOutOfBounds { package: 0 }),
        "{error}"
    );
    // A size of 0 means no meta block, whatever the offset.
    let mut none = sample.clone();
    none.edit_package(0, |p| p.meta_offset = u64::MAX);
    assert_eq!(open_built(&none).unwrap().packages()[0].meta, None);
}

#[test]
fn two_base_packages_are_refused() {
    let error = package_error(1, |p| p.kind = PackageKind::Base.to_byte());
    assert!(
        matches!(
            error,
            NupError::MultipleBases {
                first: 0,
                second: 1
            }
        ),
        "{error}"
    );
}

#[test]
fn an_update_without_a_base_needs_the_standalone_flag() {
    let update = PackageSpec::new(PackageKind::Update, UPDATE_ID, 1, update_tree());
    let built = build::build(std::slice::from_ref(&update)).unwrap();
    let error = open_built(&built).unwrap_err();
    assert!(
        matches!(error, NupError::UpdateWithoutBase { package: 0 }),
        "{error}"
    );
    let mut flagged = update;
    flagged.flags = PACKAGE_FLAG_STANDALONE_UPDATE;
    assert!(open_built(&build::build(&[flagged]).unwrap()).is_ok());
}

#[test]
fn a_content_directory_outside_its_package_is_refused() {
    let built = sample();
    let base_exefs = u32::try_from(built.dir_index(0, "exefs").unwrap()).unwrap();
    let error = package_error(1, |p| p.exefs_dir = base_exefs);
    assert!(
        matches!(error, NupError::ContentDirOutsidePackage { package: 1, field: "exefs_dir", dir }
            if dir == usize::try_from(base_exefs).unwrap()),
        "{error}"
    );
    // The same for the `RomFS` directory, which is checked on its own.
    let base_romfs = u32::try_from(built.dir_index(0, "romfs").unwrap()).unwrap();
    let error = package_error(1, |p| p.romfs_dir = base_romfs);
    assert!(
        matches!(error, NupError::ContentDirOutsidePackage { package: 1, field: "romfs_dir", dir }
            if dir == usize::try_from(base_romfs).unwrap()),
        "{error}"
    );
    assert_eq!(
        error.to_string(),
        format!("package 1's romfs_dir is directory {base_romfs}, which is not in its tree")
    );
}

// ---------------------------------------------------------------------------------------------
// What the reader does not check (the "Looser" list in the module docs)
// ---------------------------------------------------------------------------------------------

/// An edit of a built container.
type BuiltEdit = fn(&mut Built);

#[test]
fn reserved_fields_and_the_headers_version_and_uuid_are_not_checked() {
    // Each edit is made on its own, so a check added for any one of them fails here.
    let edits: [(&str, BuiltEdit); 7] = [
        ("header_version", |b| {
            b.edit_superblock(|s| s.header_version = u16::MAX);
        }),
        ("uuid", |b| {
            b.edit_superblock(|s| s.uuid = [0xA5; 16]);
        }),
        ("toc header reserved", |b| {
            b.edit_toc_header(|h| h.reserved = u64::MAX);
        }),
        ("dir reserved", |b| {
            b.edit_dir(1, |d| d.reserved = u64::MAX);
        }),
        ("file reserved2", |b| {
            b.edit_file(0, |f| f.reserved2 = u16::MAX);
        }),
        ("file reserved3", |b| {
            b.edit_file(0, |f| f.reserved3 = [0xFF; 12]);
        }),
        ("package reserved3", |b| {
            b.edit_package(0, |p| p.reserved3 = [0xFF; 20]);
        }),
    ];
    for (what, edit) in edits {
        let mut built = sample();
        edit(&mut built);
        let opened = open_built(&built);
        assert!(opened.is_ok(), "{what}: {:?}", opened.as_ref().err());
        let verified = opened.unwrap().verify();
        assert!(verified.is_ok(), "{what}: {:?}", verified.as_ref().err());
    }
}

#[test]
fn a_name_need_not_be_utf8() {
    let name: &[u8] = &[0xFF, 0xFE, b'x'];
    let mut tree = base_tree();
    tree.entries
        .insert(name.to_vec(), Entry::File(Content::raw(b"odd".to_vec())));
    let built = build::build(&[PackageSpec::new(PackageKind::Base, BASE_ID, 0, tree)]).unwrap();
    let mut nup = open_built(&built).unwrap();
    let root = nup.packages()[0].root;
    let found = nup.lookup(root, name);
    assert!(matches!(found, Some(Node::File(_))), "{found:?}");
    let Some(Node::File(file)) = found else {
        unreachable!("checked above");
    };
    assert_eq!(nup.name(Node::File(file)), name);
    assert_eq!(nup.read_to_vec(file, u64::MAX).unwrap(), b"odd");
    assert!(nup.walk_files(root).iter().any(|(path, _)| path == name));
    nup.verify().unwrap();
}

#[test]
fn a_standalone_update_next_to_a_base_is_accepted() {
    let mut built = sample();
    built.edit_package(1, |p| p.flags = PACKAGE_FLAG_STANDALONE_UPDATE);
    let nup = open_built(&built).unwrap();
    assert!(nup.packages()[1].standalone_update);
    assert_eq!(nup.packages()[0].kind, PackageKind::Base);
}

#[test]
fn bytes_after_the_string_pool_inside_the_table_of_contents_are_accepted() {
    let built = sample();
    let toc_end = built.layout.toc_offset + built.layout.toc_size;
    let gap = built.layout.blob_offset - toc_end;
    assert!(
        gap > 0,
        "the sample pads its table of contents to the blob region"
    );
    // Grow the table of contents over the padding, up to the blob region and no further.
    let mut grown = built.clone();
    grown.edit_superblock(|s| s.toc_size += to_u64(gap));
    build::rehash_toc(&mut grown.bytes).unwrap();
    let mut nup = open_built(&grown).unwrap();
    nup.verify().unwrap();
    let mut over = built.clone();
    over.edit_superblock(|s| s.toc_size += to_u64(gap) + 1);
    build::rehash_toc(&mut over.bytes).unwrap();
    assert!(matches!(
        open_built(&over).unwrap_err(),
        NupError::BlobOverlapsToc { .. }
    ));
}

// ---------------------------------------------------------------------------------------------
// Directories, files and the tree
// ---------------------------------------------------------------------------------------------

#[test]
fn a_name_must_end_inside_the_string_pool() {
    let built = sample();
    let pool_len = built.layout.toc_offset + built.layout.toc_size - built.layout.pool_offset;
    let pool_len = u32::try_from(pool_len).unwrap();
    // A name ending exactly at the end of the pool is fine (it reads the pool's last bytes);
    // here only the check is of interest, so the next check that fails is the sort order.
    let mut at_end = sample();
    at_end.edit_file(0, |f| {
        f.name_offset = pool_len - f.name_len;
    });
    assert!(!matches!(
        open_built(&at_end),
        Err(NupError::NameOutOfPool { .. })
    ));

    let mut past = sample();
    past.edit_file(0, |f| f.name_offset = pool_len - f.name_len + 1);
    let error = open_built(&past).unwrap_err();
    assert!(
        matches!(
            error,
            NupError::NameOutOfPool {
                node: "file",
                index: 0
            }
        ),
        "{error}"
    );
    let mut dir = sample();
    dir.edit_dir(1, |d| d.name_offset = u32::MAX);
    let error = open_built(&dir).unwrap_err();
    assert!(
        matches!(
            error,
            NupError::NameOutOfPool {
                node: "directory",
                index: 1
            }
        ),
        "{error}"
    );
}

#[test]
fn a_child_range_must_end_inside_the_children_array() {
    let built = sample();
    let slots = (built.layout.pool_offset - built.layout.children_offset) / CHILD_SLOT_SIZE;
    let slots = u32::try_from(slots).unwrap();
    let mut past = sample();
    past.edit_dir(0, |d| d.first_child = slots - d.child_count + 1);
    let error = open_built(&past).unwrap_err();
    assert!(
        matches!(error, NupError::ChildRangeOutOfBounds { dir: 0 }),
        "{error}"
    );
    let mut overflow = sample();
    overflow.edit_dir(0, |d| d.first_child = u32::MAX);
    assert!(matches!(
        open_built(&overflow).unwrap_err(),
        NupError::ChildRangeOutOfBounds { dir: 0 }
    ));
}

#[test]
fn overlapping_child_ranges_are_refused() {
    let mut built = sample();
    // Directory 0's range starts one slot early, over the end of nothing (it is the first), so
    // move directory 1's range onto directory 0's first slot instead.
    built.edit_dir(1, |d| d.first_child = 0);
    let error = open_built(&built).unwrap_err();
    assert!(matches!(error, NupError::ChildRangesOverlap), "{error}");
    // A directory that gives up its slots leaves them to nobody.
    let mut unused = sample();
    unused.edit_dir(0, |d| d.child_count -= 1);
    assert!(matches!(
        open_built(&unused).unwrap_err(),
        NupError::ChildRangesOverlap
    ));
}

#[test]
fn a_child_slot_past_the_last_node_is_refused() {
    let built = sample();
    let nodes =
        u32::try_from(built.layout.dir_records.len() + built.layout.file_records.len()).unwrap();
    let mut past = sample();
    past.set_child(0, nodes);
    let error = open_built(&past).unwrap_err();
    assert!(
        matches!(error, NupError::NodeOutOfRange { slot: 0, node } if node == nodes),
        "{error}"
    );
}

#[test]
fn a_node_with_two_parents_is_refused() {
    let built = sample();
    // Slot 0 is the base root's first child (its `exefs`); point it at the update's root.
    let update_root = u32::try_from(built.dir_index(1, "").unwrap()).unwrap();
    let mut twice = sample();
    twice.set_child(0, update_root);
    let error = open_built(&twice).unwrap_err();
    assert!(
        matches!(error, NupError::NodeReferencedTwice(Node::Dir(DirId(d))) if d == update_root),
        "{error}"
    );
    assert_eq!(
        error.to_string(),
        format!("directory {update_root} appears more than once in the tree")
    );
    // Two packages sharing a root.
    let error = package_error(1, |p| p.root_dir = 0);
    assert!(
        matches!(error, NupError::NodeReferencedTwice(Node::Dir(DirId(0)))),
        "{error}"
    );
    assert_eq!(
        error.to_string(),
        "directory 0 appears more than once in the tree"
    );
    // A file with two places is named as a file.
    let file_node = u32::try_from(built.layout.dir_records.len()).unwrap();
    let mut file_twice = sample();
    file_twice.set_child(0, file_node);
    let error = open_built(&file_twice).unwrap_err();
    assert_eq!(
        error.to_string(),
        "file 0 appears more than once in the tree"
    );
}

#[test]
fn an_unreachable_directory_is_refused() {
    // A cycle that no root reaches: the update's `romfs` lists itself as its only child, and
    // the slot that named its file now names nothing else. Built by hand from the sample.
    let built = sample();
    let romfs = built.dir_index(1, "romfs").unwrap();
    let file = built.file_index(1, "romfs/readme.txt").unwrap();
    let dirs = built.layout.dir_records.len();
    let mut cyclic = sample();
    // Find the slot that holds the file and make it name the directory itself.
    let slot = (0..(built.layout.pool_offset - built.layout.children_offset) / CHILD_SLOT_SIZE)
        .find(|&slot| {
            let at = built.layout.children_offset + slot * CHILD_SLOT_SIZE;
            u32::from_le_bytes(built.bytes[at..at + 4].try_into().unwrap())
                == u32::try_from(dirs + file).unwrap()
        })
        .unwrap();
    cyclic.set_child(slot, u32::try_from(romfs).unwrap());
    // The update root's `romfs` slot now has to stop naming it, or `romfs` has two parents:
    // move that slot onto the orphaned file instead.
    let root_slot = (0..slot)
        .find(|&s| {
            let at = built.layout.children_offset + s * CHILD_SLOT_SIZE;
            u32::from_le_bytes(built.bytes[at..at + 4].try_into().unwrap())
                == u32::try_from(romfs).unwrap()
        })
        .unwrap();
    cyclic.set_child(root_slot, u32::try_from(dirs + file).unwrap());
    // The update's root now lists a file called `romfs`'s position, which breaks the sort
    // order only if names compare wrongly; reachability is checked first.
    let error = open_built(&cyclic).unwrap_err();
    assert!(
        matches!(error, NupError::UnreachableNode(Node::Dir(DirId(d))) if usize::try_from(d).unwrap() == romfs),
        "{error}"
    );
    assert_eq!(
        error.to_string(),
        format!("directory {romfs} is not reachable from any package root")
    );
}

/// A package whose only file is `depth` directories below its root.
fn nested(depth: usize) -> Built {
    let path = format!("{}f", "d/".repeat(depth));
    let tree = Tree::new()
        .with_file(&path, Content::raw(b"x".to_vec()))
        .unwrap();
    build::build(&[PackageSpec::new(PackageKind::Base, BASE_ID, 0, tree)]).unwrap()
}

#[test]
fn a_tree_nested_past_the_depth_limit_is_refused_and_one_at_it_is_not() {
    // The deepest directory of a tree at the limit is `MAX_TREE_DEPTH` levels below the root.
    let built = nested(MAX_TREE_DEPTH);
    let mut nup = open_built(&built).unwrap();
    let root = nup.packages()[0].root;
    let walked = nup.walk_files(root);
    assert_eq!(walked.len(), 1);
    assert_eq!(
        walked[0].0.len(),
        2 * MAX_TREE_DEPTH + 1,
        "{MAX_TREE_DEPTH} names and `f`"
    );
    nup.verify().unwrap();

    let built = nested(MAX_TREE_DEPTH + 1);
    let error = open_built(&built).unwrap_err();
    assert!(
        matches!(error, NupError::TreeTooDeep { dir, limit }
            if dir == MAX_TREE_DEPTH + 1 && limit == MAX_TREE_DEPTH),
        "{error}"
    );
    assert_eq!(
        error.to_string(),
        format!(
            "directory {} is nested more than {MAX_TREE_DEPTH} levels below its package root",
            MAX_TREE_DEPTH + 1
        )
    );
}

#[test]
fn the_depth_limit_is_per_package_not_cumulative() {
    // Two packages, each at the limit: depth is measured from a package's own root.
    let tree = || {
        Tree::new()
            .with_file(
                &format!("{}f", "d/".repeat(MAX_TREE_DEPTH)),
                Content::raw(b"x".to_vec()),
            )
            .unwrap()
    };
    let built = build::build(&[
        PackageSpec::new(PackageKind::Base, BASE_ID, 0, tree()),
        PackageSpec::new(PackageKind::Update, UPDATE_ID, 1, tree()),
    ])
    .unwrap();
    assert!(open_built(&built).is_ok());
}

#[test]
fn walking_files_names_each_by_its_own_path_across_siblings_and_levels() {
    // After a deep directory is left, the next sibling's path must not carry its name.
    let leaf = || Content::raw(b"x".to_vec());
    let tree = Tree::new()
        .with_file("a/b/c/z", leaf())
        .unwrap()
        .with_file("a/b/y", leaf())
        .unwrap()
        .with_file("a/w", leaf())
        .unwrap()
        .with_file("a/x", leaf())
        .unwrap()
        .with_file("v", leaf())
        .unwrap();
    let built = build::build(&[PackageSpec::new(PackageKind::Base, BASE_ID, 0, tree)]).unwrap();
    let nup = open_built(&built).unwrap();
    let root = nup.packages()[0].root;
    let paths: Vec<String> = nup
        .walk_files(root)
        .into_iter()
        .map(|(path, file)| {
            assert_eq!(
                file,
                file_id(&built, 0, std::str::from_utf8(&path).unwrap())
            );
            String::from_utf8(path).unwrap()
        })
        .collect();
    assert_eq!(paths, ["a/b/c/z", "a/b/y", "a/w", "a/x", "v"]);
    // Below a directory, paths are relative to it.
    let b = nup.resolve(root, "a/b").and_then(|n| match n {
        Node::Dir(d) => Some(d),
        Node::File(_) => None,
    });
    let below: Vec<String> = nup
        .walk_files(b.unwrap())
        .into_iter()
        .map(|(path, _)| String::from_utf8(path).unwrap())
        .collect();
    assert_eq!(below, ["c/z", "y"]);
}

#[test]
fn unsorted_or_duplicate_children_are_refused() {
    let built = sample();
    let romfs = built.dir_index(0, "romfs").unwrap();
    // `copy.txt` < `data` < `readme.txt`; renaming `copy.txt` to `readme.txt`'s name makes a
    // duplicate, and to a name after it breaks the order.
    let copy = built.file_index(0, "romfs/copy.txt").unwrap();
    let readme = built.file_index(0, "romfs/readme.txt").unwrap();
    let readme_name = {
        let mut name = (0, 0);
        let mut probe = sample();
        probe.edit_file(readme, |f| name = (f.name_offset, f.name_len));
        name
    };
    let mut duplicate = sample();
    duplicate.edit_file(copy, |f| (f.name_offset, f.name_len) = readme_name);
    let error = open_built(&duplicate).unwrap_err();
    assert!(
        matches!(error, NupError::ChildrenNotSorted { dir } if dir == romfs),
        "{error}"
    );

    let mut unsorted = sample();
    unsorted.edit_file(copy, |f| {
        // `readme.txt` without its first byte: `eadme.txt`, which sorts after `data`.
        f.name_offset = readme_name.0 + 1;
        f.name_len = readme_name.1 - 1;
    });
    let error = open_built(&unsorted).unwrap_err();
    assert!(
        matches!(error, NupError::ChildrenNotSorted { dir } if dir == romfs),
        "{error}"
    );
}

fn file_error(path: &str, edit: impl FnOnce(&mut RawFile, &Built)) -> NupError {
    let mut built = sample();
    let index = built.file_index(0, path).unwrap();
    let snapshot = built.clone();
    built.edit_file(index, |f| edit(f, &snapshot));
    open_built(&built).unwrap_err()
}

#[test]
fn each_bad_file_field_has_its_own_error() {
    let readme = sample().file_index(0, "romfs/readme.txt").unwrap();
    let empty = sample().file_index(0, "romfs/data/empty").unwrap();

    let unknown = u8::try_from(Codec::ALL.len()).unwrap();
    let error = file_error("romfs/readme.txt", |f, _| f.codec = unknown);
    assert!(
        matches!(error, NupError::UnknownCodec { file, codec } if file == readme && codec == unknown),
        "{error}"
    );
    let error = file_error("romfs/readme.txt", |f, _| f.codec_flags = 1);
    assert!(
        matches!(error, NupError::CodecFlagsNotZero { file, flags: 1 } if file == readme),
        "{error}"
    );
    let edits: [fn(&mut RawFile, &Built); 3] = [
        |f, _| f.extent_size = 16,
        |f, b| f.extent_offset = to_u64(b.layout.blob_offset),
        |f, _| f.codec = Codec::Zstd.to_byte(),
    ];
    for edit in edits {
        let error = file_error("romfs/data/empty", edit);
        assert!(
            matches!(error, NupError::EmptyFileWithExtent { file } if file == empty),
            "{error}"
        );
    }
    let error = file_error("romfs/readme.txt", |f, _| f.extent_size = 0);
    assert!(
        matches!(error, NupError::MissingExtent { file } if file == readme),
        "{error}"
    );
    let error = file_error("romfs/readme.txt", |f, _| f.extent_size += 1);
    assert!(
        matches!(error, NupError::RawSizeMismatch { file, size: 5, extent_size: 6 } if file == readme),
        "{error}"
    );
}

#[test]
fn a_raw_extent_of_the_wrong_size_is_refused() {
    // Smaller than the file, too (the larger case is in `each_bad_file_field_has_its_own_error`).
    let error = file_error("romfs/readme.txt", |f, _| f.extent_size -= 1);
    assert!(
        matches!(
            error,
            NupError::RawSizeMismatch {
                size: 5,
                extent_size: 4,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn an_extent_lies_inside_the_blob_region_and_is_aligned() {
    let built = sample();
    let file_size = to_u64(built.bytes.len());
    let level = built.file_index(0, "romfs/data/level.bin").unwrap();

    // Before the blob region.
    let error = file_error("romfs/data/level.bin", |f, b| {
        f.extent_offset = to_u64(b.layout.blob_offset) - EXTENT_ALIGN;
    });
    assert!(
        matches!(error, NupError::ExtentOutOfBounds { file } if file == level),
        "{error}"
    );
    // Running one byte past the end of the file (a zstd extent may be any size).
    let error = file_error("romfs/data/level.bin", |f, _| {
        f.extent_size = file_size - f.extent_offset + 1;
    });
    assert!(
        matches!(error, NupError::ExtentOutOfBounds { file } if file == level),
        "{error}"
    );
    let error = file_error("romfs/data/level.bin", |f, _| f.extent_size = u64::MAX);
    assert!(
        matches!(error, NupError::ExtentOutOfBounds { file } if file == level),
        "{error}"
    );
    // Every offset that is a multiple of a smaller power of two but not of `EXTENT_ALIGN`.
    for bit in 0..EXTENT_ALIGN.ilog2() {
        let delta = 1u64 << bit;
        let error = file_error("romfs/data/level.bin", |f, _| f.extent_offset += delta);
        assert!(
            matches!(error, NupError::ExtentMisaligned { file, offset }
                if file == level && offset % EXTENT_ALIGN == delta),
            "+{delta}: {error}"
        );
    }
    // Ending exactly at the end of the file passes the bounds check (and then overlaps the
    // extents after it, which is the next check).
    let error = file_error("romfs/data/level.bin", |f, _| {
        f.extent_size = file_size - f.extent_offset;
    });
    assert!(matches!(error, NupError::RegionsOverlap { .. }), "{error}");
}

#[test]
fn overlapping_extents_are_refused() {
    let built = sample();
    let first = built.file_index(0, "exefs/main").unwrap();
    let (offset, size) = built.layout.extents[first];
    // Make the 5-byte `romfs/readme.txt` start where the 9-byte `exefs/main` does: the same
    // offset with another size is an overlap, not a shared extent.
    assert_ne!(size, 5);
    let error = file_error("romfs/readme.txt", |f, _| f.extent_offset = to_u64(offset));
    // Regions are ordered by start, then end: the shorter one is `first`.
    let readme = built.file_index(0, "romfs/readme.txt").unwrap();
    assert!(
        matches!(
            error,
            NupError::RegionsOverlap {
                first: Region::Extent(a),
                second: Region::Extent(b)
            } if (a, b) == (readme, first)
        ),
        "{error}"
    );
    assert_eq!(
        error.to_string(),
        format!("file {readme}'s extent and file {first}'s extent overlap")
    );
}

#[test]
fn files_sharing_an_extent_must_agree() {
    let built = sample();
    let copy = built.file_index(0, "romfs/copy.txt").unwrap();
    let readme = built.file_index(0, "romfs/readme.txt").unwrap();
    let error = file_error("romfs/readme.txt", |f, _| f.sha256[0] ^= 1);
    assert!(
        matches!(error, NupError::SharedExtentMismatch { first, second }
            if (first, second) == (copy.min(readme), copy.max(readme))),
        "{error}"
    );
}

#[test]
fn files_sharing_an_extent_must_agree_on_each_of_size_codec_and_hash() {
    let built = sample();
    let copy = built.file_index(0, "romfs/copy.txt").unwrap();
    let readme = built.file_index(0, "romfs/readme.txt").unwrap();
    let both = (copy.min(readme), copy.max(readme));
    // Only the codec differs (a zstd frame is not checked against a raw extent at open).
    let error = file_error("romfs/readme.txt", |f, _| f.codec = Codec::Zstd.to_byte());
    assert!(
        matches!(error, NupError::SharedExtentMismatch { first, second } if (first, second) == both),
        "{error}"
    );
    // Only the size differs. Raw files cannot show it (a raw extent is its file's size, so a
    // different size is a different extent); two zstd files can.
    let tree = Tree::new()
        .with_file("romfs/a", Content::zstd(compressible(300)))
        .unwrap()
        .with_file("romfs/b", Content::zstd(compressible(300)))
        .unwrap();
    let mut pair = build::build(&[PackageSpec::new(PackageKind::Base, BASE_ID, 0, tree)]).unwrap();
    assert_eq!(pair.layout.extents[0], pair.layout.extents[1]);
    assert!(open_built(&pair).is_ok());
    pair.edit_file(1, |f| f.size += 1);
    let error = open_built(&pair).unwrap_err();
    assert!(
        matches!(
            error,
            NupError::SharedExtentMismatch {
                first: 0,
                second: 1
            }
        ),
        "{error}"
    );
}

#[test]
fn a_meta_block_over_an_extent_is_refused() {
    let built = sample();
    let (offset, size) = built.layout.extents[built.file_index(0, "exefs/main").unwrap()];
    let error = package_error(0, |p| {
        p.meta_offset = to_u64(offset + size - 1);
        p.meta_size = 1;
    });
    let main = built.file_index(0, "exefs/main").unwrap();
    assert!(
        matches!(
            error,
            NupError::RegionsOverlap {
                first: Region::Extent(a),
                second: Region::Meta(0)
            } if a == main
        ),
        "{error}"
    );
    assert_eq!(
        error.to_string(),
        format!("file {main}'s extent and package 0's meta block overlap")
    );
    // Starting exactly where the extent ends is no overlap.
    let mut after = sample();
    after.edit_package(0, |p| {
        p.meta_offset = to_u64(offset + size);
        p.meta_size = 1;
    });
    assert!(open_built(&after).is_ok());
}

// ---------------------------------------------------------------------------------------------
// Reading damaged content
// ---------------------------------------------------------------------------------------------

#[test]
fn a_damaged_stored_file_fails_its_hash_and_verify_names_it() {
    let mut built = sample();
    let readme = built.file_index(1, "romfs/readme.txt").unwrap();
    let (offset, _) = built.layout.extents[readme];
    built.bytes[offset] ^= 0x20;
    let mut nup = open_built(&built).unwrap();

    let error = nup
        .verify_file(FileId(u32::try_from(readme).unwrap()))
        .unwrap_err();
    assert!(
        matches!(error, NupError::HashMismatch { file } if file == readme),
        "{error}"
    );

    // Through `Read`, the error is `InvalidData` and carries the `NupError`.
    let mut out = Vec::new();
    let io = nup
        .reader(FileId(u32::try_from(readme).unwrap()))
        .unwrap()
        .read_to_end(&mut out)
        .unwrap_err();
    assert_eq!(io.kind(), std::io::ErrorKind::InvalidData);
    assert!(matches!(
        NupError::from_io(io),
        NupError::HashMismatch { .. }
    ));

    let error = nup.verify().unwrap_err();
    assert_eq!(
        (error.package, error.path.as_str()),
        (1, "romfs/readme.txt")
    );
    assert!(matches!(error.source, NupError::HashMismatch { file } if file == readme));
    assert_eq!(
        error.to_string(),
        format!("package 1, romfs/readme.txt: file {readme}'s content does not match its SHA-256")
    );
}

#[test]
fn a_read_that_failed_fails_again_instead_of_ending_the_file() {
    // After `ContentTooLong`, `TrailingExtentBytes` or `HashMismatch` the next read repeats the
    // error; it is never a clean end of file.
    let data = compressible(3000);
    let mut too_long = build::zstd_frame(&compressible(3001)).unwrap();
    too_long[6..14].copy_from_slice(&3000u64.to_le_bytes());
    let mut trailing = build::zstd_frame(&data).unwrap();
    trailing.push(0);
    let cases: [(&str, Vec<u8>, Want); 2] = [
        ("too long", too_long, |e| {
            matches!(e, NupError::ContentTooLong { .. })
        }),
        ("trailing", trailing, |e| {
            matches!(e, NupError::TrailingExtentBytes { .. })
        }),
    ];
    for (what, stored, want) in cases {
        let (built, file) = with_stored(&data, stored);
        assert_sticky(&built, file, what, want);
    }
    // A raw file whose stored bytes were damaged.
    let mut built = sample();
    let readme = file_id(&built, 1, "romfs/readme.txt");
    let (offset, _) = built.layout.extents[readme.index()];
    built.bytes[offset] ^= 0x20;
    assert_sticky(&built, readme, "hash", |e| {
        matches!(e, NupError::HashMismatch { .. })
    });
}

/// What an error should be.
type Want = fn(&NupError) -> bool;

/// Reads `file` until it fails, then again: the second read must carry the same error.
fn assert_sticky(built: &Built, file: FileId, what: &str, want: Want) {
    let mut nup = open_built(built).unwrap();
    let mut reader = nup.reader(file).unwrap();
    let mut buf = vec![0u8; 8192];
    let first = loop {
        match reader.read(&mut buf) {
            Ok(0) => unreachable!("{what}: the file ended cleanly"),
            Ok(_) => {}
            Err(error) => break NupError::from_io(error),
        }
    };
    assert!(want(&first), "{what}: {first}");
    for again in 0..2 {
        let error = reader.read(&mut buf).unwrap_err();
        assert_eq!(
            error.kind(),
            std::io::ErrorKind::InvalidData,
            "{what} {again}"
        );
        let error = NupError::from_io(error);
        assert!(want(&error), "{what} {again}: {error}");
        assert_eq!(error.to_string(), first.to_string(), "{what} {again}");
    }
    // An empty buffer is no way around it either.
    assert!(reader.read(&mut []).is_err(), "{what}");
}

#[test]
fn the_end_checks_run_in_the_documented_order() {
    // Too long, then trailing bytes, then the hash: with all three faults at once the first
    // one speaks, and each fault alone is reported as itself.
    let data = compressible(3000);
    let wrong_hash = |built: &mut Built| {
        let file = built.file_index(0, "romfs/f").unwrap();
        built.edit_file(file, |f| f.sha256[0] ^= 1);
    };
    let verify = |built: &Built, file| open_built(built).unwrap().verify_file(file).unwrap_err();

    let mut trailing = build::zstd_frame(&data).unwrap();
    trailing.push(0);
    let (mut built, file) = with_stored(&data, trailing.clone());
    wrong_hash(&mut built);
    let error = verify(&built, file);
    assert!(
        matches!(error, NupError::TrailingExtentBytes { .. }),
        "{error}"
    );

    let mut too_long = build::zstd_frame(&compressible(3001)).unwrap();
    too_long[6..14].copy_from_slice(&3000u64.to_le_bytes());
    too_long.push(0);
    let (mut built, file) = with_stored(&data, too_long);
    wrong_hash(&mut built);
    let error = verify(&built, file);
    assert!(matches!(error, NupError::ContentTooLong { .. }), "{error}");

    let (mut built, file) = with_stored(&data, build::zstd_frame(&data).unwrap());
    wrong_hash(&mut built);
    let error = verify(&built, file);
    assert!(matches!(error, NupError::HashMismatch { .. }), "{error}");
}

#[test]
fn verify_checks_every_empty_file_against_its_hash() {
    // Empty files all have the extent (0, 0), but each has its own record and hash; none is
    // skipped as a shared extent already read.
    let tree = Tree::new()
        .with_file("romfs/a", Content::raw(Vec::new()))
        .unwrap()
        .with_file("romfs/b", Content::raw(Vec::new()))
        .unwrap();
    let mut built = build::build(&[PackageSpec::new(PackageKind::Base, BASE_ID, 0, tree)]).unwrap();
    assert_eq!(open_built(&built).unwrap().verify().unwrap().files, 2);
    let b = built.file_index(0, "romfs/b").unwrap();
    built.edit_file(b, |f| f.sha256[0] ^= 1);
    let mut nup = open_built(&built).unwrap();
    let error = nup.verify().unwrap_err();
    assert_eq!((error.package, error.path.as_str()), (0, "romfs/b"));
    assert!(matches!(error.source, NupError::HashMismatch { file } if file == b));
}

#[test]
fn an_io_error_without_a_nup_error_stays_an_io_error() {
    let error = NupError::from_io(std::io::Error::other("disk on fire"));
    assert!(error.is_io());
    assert!(!NupError::NoProgram.is_io());
}

/// A container whose only file is `data` with its extent replaced by `stored`.
fn with_stored(data: &[u8], stored: Vec<u8>) -> (Built, FileId) {
    let tree = Tree::new()
        .with_file("romfs/f", Content::zstd(data.to_vec()).with_stored(stored))
        .unwrap();
    let built = build::build(&[PackageSpec::new(PackageKind::Base, BASE_ID, 0, tree)]).unwrap();
    let file = file_id(&built, 0, "romfs/f");
    (built, file)
}

#[test]
fn a_zstd_frame_must_declare_the_files_size() {
    let data = compressible(3000);
    // `ruzstd`'s own frame, which declares no content size.
    let bare =
        ruzstd::encoding::compress_to_vec(&data[..], ruzstd::encoding::CompressionLevel::Fastest);
    let (built, file) = with_stored(&data, bare);
    let mut nup = open_built(&built).unwrap();
    let error = nup.reader(file).unwrap_err();
    assert!(
        matches!(
            error,
            NupError::ZstdContentSize {
                size: 3000,
                frame: 0,
                ..
            }
        ),
        "{error}"
    );

    // A frame that declares another size.
    let (built, file) = with_stored(&data, build::zstd_frame(&data[..2999]).unwrap());
    let error = open_built(&built).unwrap().reader(file).unwrap_err();
    assert!(
        matches!(
            error,
            NupError::ZstdContentSize {
                size: 3000,
                frame: 2999,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn bytes_after_the_zstd_frame_are_refused() {
    let data = compressible(3000);
    let mut stored = build::zstd_frame(&data).unwrap();
    stored.push(0);
    let (built, file) = with_stored(&data, stored);
    let error = open_built(&built).unwrap().verify_file(file).unwrap_err();
    assert!(
        matches!(error, NupError::TrailingExtentBytes { .. }),
        "{error}"
    );
}

#[test]
fn a_cut_zstd_frame_or_garbage_is_a_decode_error() {
    let data = compressible(3000);
    let mut cut = build::zstd_frame(&data).unwrap();
    cut.truncate(cut.len() - 4);
    let (built, file) = with_stored(&data, cut);
    let error = open_built(&built).unwrap().verify_file(file).unwrap_err();
    assert!(matches!(error, NupError::Zstd { .. }), "{error}");

    let (built, file) = with_stored(&data, vec![0xA5; 64]);
    let error = open_built(&built).unwrap().reader(file).unwrap_err();
    assert!(matches!(error, NupError::Zstd { .. }), "{error}");
}

#[test]
fn a_zstd_frame_that_holds_more_or_less_than_it_declares_is_refused() {
    let data = compressible(3000);
    let declare = |frame: &mut Vec<u8>, size: u64| {
        // `zstd_frame` writes the 8-byte content size right after the magic, the descriptor
        // and the window descriptor.
        frame[6..14].copy_from_slice(&size.to_le_bytes());
    };
    // Holds 3001 bytes, declares 3000.
    let mut longer = build::zstd_frame(&compressible(3001)).unwrap();
    declare(&mut longer, 3000);
    let (built, file) = with_stored(&data, longer);
    let error = open_built(&built).unwrap().verify_file(file).unwrap_err();
    assert!(matches!(error, NupError::ContentTooLong { .. }), "{error}");

    // Holds 2999 bytes, declares 3000.
    let mut shorter = build::zstd_frame(&compressible(2999)).unwrap();
    declare(&mut shorter, 3000);
    let (built, file) = with_stored(&data, shorter);
    let error = open_built(&built).unwrap().verify_file(file).unwrap_err();
    assert!(
        matches!(error, NupError::ContentTruncated { .. }),
        "{error}"
    );
}

// ---------------------------------------------------------------------------------------------
// Hand-written zstd frames (RFC 8878), not made by the builder's encoder
// ---------------------------------------------------------------------------------------------

/// `Content_Checksum_flag`, bit 2 of the frame header descriptor (RFC 8878, 3.1.1.1.1).
const CONTENT_CHECKSUM_FLAG: u8 = 0b0000_0100;
/// `Window_Descriptor`'s window is `1 << (WINDOW_LOG_BASE + exponent)` bytes with a zero
/// mantissa (RFC 8878, 3.1.1.1.2); the exponent is the byte's top five bits.
const WINDOW_LOG_BASE: u32 = 10;

/// A raw block (RFC 8878, 3.1.1.2): a three-byte little-endian header holding `Last_Block` in
/// bit 0, `Block_Type` raw (0) in bits 1-2 and `Block_Size` from bit 3, then the bytes.
fn raw_block(last: bool, data: &[u8]) -> Vec<u8> {
    let header = (u32::try_from(data.len()).unwrap() << 3) | u32::from(last);
    let mut out = header.to_le_bytes()[..3].to_vec();
    out.extend_from_slice(data);
    out
}

/// The frame magic, taken from a frame the encoder wrote.
fn frame_magic() -> Vec<u8> {
    build::zstd_frame(&[0]).unwrap()[..4].to_vec()
}

/// A frame: magic, `descriptor`, an optional window descriptor, the content size field, then
/// `body` (the blocks and any checksum).
fn hand_frame(descriptor: u8, window: Option<u8>, content_size: &[u8], body: &[u8]) -> Vec<u8> {
    let mut frame = frame_magic();
    frame.push(descriptor);
    frame.extend(window);
    frame.extend_from_slice(content_size);
    frame.extend_from_slice(body);
    frame
}

/// A single-segment frame whose content size is the one byte `data.len()`: `Frame_Content_Size
/// _flag` 0 with `Single_Segment_flag` set means a one-byte field (RFC 8878, 3.1.1.1.4).
fn single_segment_frame(data: &[u8], flags: u8, body: &[u8]) -> Vec<u8> {
    hand_frame(
        build::SINGLE_SEGMENT_FLAG | flags,
        None,
        &[u8::try_from(data.len()).unwrap()],
        body,
    )
}

const HELLO: &[u8] = b"hello";

/// The container holding `HELLO` stored as `frame`.
fn hello_in(frame: Vec<u8>) -> (Built, FileId) {
    with_stored(HELLO, frame)
}

fn read_back(built: &Built, file: FileId) -> Result<Vec<u8>, NupError> {
    open_built(built).unwrap().read_to_vec(file, u64::MAX)
}

#[test]
fn a_single_segment_frame_with_a_one_byte_content_size_reads_back() {
    let frame = single_segment_frame(HELLO, 0, &raw_block(true, HELLO));
    // Magic, descriptor, size, block header, content.
    assert_eq!(frame.len(), 4 + 1 + 1 + 3 + HELLO.len());
    let (built, file) = hello_in(frame);
    assert_eq!(read_back(&built, file).unwrap(), HELLO);
    open_built(&built).unwrap().verify().unwrap();
}

#[test]
fn a_frame_with_a_content_checksum_ends_with_its_four_bytes() {
    // `ruzstd` reads the four checksum bytes but does not compare them with the content (the
    // file's SHA-256 is the integrity check), so any four bytes stand for the checksum here.
    let checksum = [0xDE, 0xAD, 0xBE, 0xEF];
    let mut body = raw_block(true, HELLO);
    body.extend_from_slice(&checksum);
    let frame = single_segment_frame(HELLO, CONTENT_CHECKSUM_FLAG, &body);
    let (built, file) = hello_in(frame.clone());
    assert_eq!(read_back(&built, file).unwrap(), HELLO);
    open_built(&built).unwrap().verify().unwrap();

    // One byte after the checksum is a byte after the frame.
    let mut longer = frame.clone();
    longer.push(0);
    let (built, file) = hello_in(longer);
    let error = open_built(&built).unwrap().verify_file(file).unwrap_err();
    assert!(
        matches!(error, NupError::TrailingExtentBytes { file: f } if f == file.index()),
        "{error}"
    );

    // A frame that announces a checksum but stops before it is cut short inside the frame.
    let (built, file) = hello_in(frame[..frame.len() - checksum.len()].to_vec());
    let error = open_built(&built).unwrap().verify_file(file).unwrap_err();
    assert!(matches!(error, NupError::Zstd { .. }), "{error}");
}

#[test]
fn content_followed_by_an_empty_last_block_reads_back() {
    let mut body = raw_block(false, HELLO);
    body.extend(raw_block(true, &[]));
    let frame = single_segment_frame(HELLO, 0, &body);
    let (built, file) = hello_in(frame.clone());
    assert_eq!(read_back(&built, file).unwrap(), HELLO);
    open_built(&built).unwrap().verify().unwrap();

    // The empty block is part of the frame: dropping its bytes cuts the frame short.
    let (built, file) = hello_in(frame[..frame.len() - 1].to_vec());
    let error = open_built(&built).unwrap().verify_file(file).unwrap_err();
    assert!(matches!(error, NupError::Zstd { .. }), "{error}");
}

#[test]
fn a_window_over_the_decoders_limit_is_a_decode_error_and_one_at_it_is_not() {
    use ruzstd::decoding::DEFAULT_MAX_WINDOW_SIZE;

    // A frame with a window descriptor (no single-segment flag) and an eight-byte content size.
    let framed = |exponent: u32| {
        let window = u8::try_from(exponent << 3).unwrap();
        hand_frame(
            build::FCS_FLAG_8_BYTES,
            Some(window),
            &to_u64(HELLO.len()).to_le_bytes(),
            &raw_block(true, HELLO),
        )
    };
    let at_limit = DEFAULT_MAX_WINDOW_SIZE.ilog2() - WINDOW_LOG_BASE;
    assert_eq!(
        1u64 << (WINDOW_LOG_BASE + at_limit),
        DEFAULT_MAX_WINDOW_SIZE,
        "the limit is a power of two, so a zero-mantissa window can sit exactly on it"
    );
    let (built, file) = hello_in(framed(at_limit));
    assert_eq!(read_back(&built, file).unwrap(), HELLO);

    let (built, file) = hello_in(framed(at_limit + 1));
    let error = open_built(&built).unwrap().reader(file).unwrap_err();
    assert!(
        matches!(&error, NupError::Zstd { file: f, .. } if *f == file.index()),
        "{error}"
    );
}

// ---------------------------------------------------------------------------------------------
// Untrusted input never panics
// ---------------------------------------------------------------------------------------------

/// Opens, takes the title, walks and verifies whatever opens. Errors are fine; panics are not.
fn exercise(bytes: &[u8]) {
    let Ok(mut nup) = open(bytes) else {
        return;
    };
    let _ = nup.title();
    for package in nup.packages().to_vec() {
        let _ = nup.walk_files(package.root);
    }
    let _ = nup.verify();
}

proptest! {
    #![proptest_config(config(512))]

    #[test]
    fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..1024)) {
        exercise(&bytes);
    }

    /// Damage a good container (and fix its table of contents hash, so the damage is read).
    #[test]
    fn a_damaged_container_never_panics(
        flips in proptest::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..8),
        rehash in any::<bool>(),
    ) {
        let mut built = sample();
        for (at, value) in flips {
            let index = at.index(built.bytes.len());
            built.bytes[index] = value;
        }
        if rehash {
            let _ = build::rehash_toc(&mut built.bytes);
        }
        exercise(&built.bytes);
    }

    /// Damage only the table of contents, the part with the most structure.
    #[test]
    fn a_damaged_table_of_contents_never_panics(
        flips in proptest::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..4),
    ) {
        let mut built = sample();
        let start = built.layout.toc_offset;
        for (at, value) in flips {
            let index = start + at.index(built.layout.toc_size);
            built.bytes[index] = value;
        }
        build::rehash_toc(&mut built.bytes).unwrap();
        exercise(&built.bytes);
    }
}

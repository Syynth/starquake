//! [`open`]: a title's program and its RomFS, served as the image the guest reads, on synthetic
//! containers built in the test.

use starquake_loader::nup::storage::StorageError;
use starquake_loader::read_at::Positional;
use starquake_loader::romfs::RomFs as RomFsImage;

use super::*;

/// A tree with the hello ExeFS and `romfs` files at `romfs/<path>`.
fn with_romfs(files: &[(&str, Content)]) -> Tree {
    files
        .iter()
        .fold(exefs_tree(&HELLO), |tree, (path, content)| {
            tree.with_file(&format!("{ROMFS_DIR_NAME}/{path}"), content.clone())
                .unwrap()
        })
}

/// The bytes of a container of `packages`.
fn bytes(packages: &[PackageSpec]) -> Vec<u8> {
    build::build(packages).unwrap().bytes
}

/// The content of `path` in the RomFS image `romfs` serves, read through the loader's reader.
fn file_in(romfs: &RomFs, path: &str) -> Option<Vec<u8>> {
    let image = RomFsImage::new(Positional::new(Image(romfs.clone()))).unwrap();
    let entry = image.find_file(path).unwrap()?;
    let mut out = vec![0; usize::try_from(entry.size).unwrap()];
    let read = image.read_file_shared(&entry, 0, &mut out).unwrap();
    assert_eq!(read, out.len());
    Some(out)
}

/// A [`RomFs`] as a positional source, so the loader's reader can parse it.
struct Image(RomFs);

impl ReadAt for Image {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read_at(offset, buf)
    }

    fn size(&self) -> io::Result<u64> {
        Ok(self.0.size())
    }
}

#[test]
fn a_title_opens_as_its_program_and_its_romfs() {
    let files = [
        ("hello.txt", Content::raw(b"hello".to_vec())),
        ("data/big.bin", Content::zstd(vec![7; 3000])),
    ];
    let title = open(bytes(&[base(with_romfs(&files))])).unwrap();
    assert_eq!(title.system, expected(&HELLO));
    let romfs = title.romfs.unwrap();
    assert_eq!(file_in(&romfs, "/hello.txt").unwrap(), b"hello");
    assert_eq!(file_in(&romfs, "/data/big.bin").unwrap(), vec![7; 3000]);
    // The ExeFS is not in it.
    assert_eq!(file_in(&romfs, "/main"), None);
    // Reads past the end read nothing.
    assert_eq!(romfs.read_at(romfs.size(), &mut [0; 4]).unwrap(), 0);
}

#[test]
fn the_romfs_is_the_newest_packages_never_the_bases() {
    let base_files = [
        ("hello.txt", Content::raw(b"base".to_vec())),
        ("only-in-base", Content::raw(b"x".to_vec())),
    ];
    let update_files = [("hello.txt", Content::raw(b"update".to_vec()))];
    let packages = [
        base(with_romfs(&base_files)),
        update(1, with_romfs(&update_files)),
    ];
    let title = open(bytes(&packages)).unwrap();
    let romfs = title.romfs.unwrap();
    assert_eq!(file_in(&romfs, "/hello.txt").unwrap(), b"update");
    assert_eq!(file_in(&romfs, "/only-in-base"), None);
}

#[test]
fn a_newest_update_without_the_bases_romfs_is_refused_in_words() {
    let base_files = [("hello.txt", Content::raw(b"base".to_vec()))];
    let packages = [base(with_romfs(&base_files)), update(1, exefs_tree(&HELLO))];
    let error = open(bytes(&packages)).unwrap_err();
    assert!(
        matches!(
            error,
            TitleError::Container(NupError::IncompleteUpdate {
                package: 1,
                part: Part::RomFs
            })
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "the title could not be opened: the newest update (package 1) has no RomFS of its own \
         and would need the base's; an update that patches its base is not supported yet"
    );
    // The program alone refuses it the same way: the run and its data agree on the package.
    let error = System::from_nup(Cursor::new(bytes(&packages))).unwrap_err();
    assert!(
        matches!(
            error,
            TitleError::Container(NupError::IncompleteUpdate {
                part: Part::RomFs,
                ..
            })
        ),
        "{error:?}"
    );
}

#[test]
fn a_title_without_a_romfs_runs_with_no_data_storage() {
    let title = open(bytes(&[base(exefs_tree(&HELLO))])).unwrap();
    assert_eq!(title.system, expected(&HELLO));
    assert_eq!(title.romfs, None);
}

#[test]
fn a_romfs_that_cannot_be_an_image_is_refused_after_the_program_loads() {
    let long = "n".repeat(starquake_loader::romfs::MAX_PATH_LEN);
    let files = [(long.as_str(), Content::raw(b"x".to_vec()))];
    let error = open(bytes(&[base(with_romfs(&files))])).unwrap_err();
    assert!(
        matches!(error, TitleError::RomFs(StorageError::Build(_))),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "the title's RomFS could not be served: the RomFS could not be laid out as an image: a \
         path of the RomFS is 769 bytes, longer than the 768 a path may be"
    );
    // With a program that does not load as well, the program's error comes first.
    let broken = exefs_tree(&[(MAIN_MODULE, NSO)]);
    let tree = files.iter().fold(broken, |tree, (path, content)| {
        tree.with_file(&format!("{ROMFS_DIR_NAME}/{path}"), content.clone())
            .unwrap()
    });
    let error = open(bytes(&[base(tree)])).unwrap_err();
    assert!(
        matches!(error, TitleError::ExeFs(ExeFsError::MissingNpdm)),
        "{error:?}"
    );
}

#[test]
fn a_compressed_romfs_file_past_the_cache_is_an_error_when_read_not_when_opened() {
    let files = [("big.bin", Content::zstd(vec![1; 100]))];
    let title = open_with(bytes(&[base(with_romfs(&files))]), READ_LIMITS, 99).unwrap();
    let romfs = title.romfs.unwrap();
    let mut all = vec![0; usize::try_from(romfs.size()).unwrap()];
    let error = romfs.read_at(0, &mut all).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(
        error.to_string().ends_with(
            "is 100 bytes decompressed, more than the 99-byte cache of decompressed files holds"
        ),
        "{error}"
    );
    // At the cap exactly, it is served.
    let title = open_with(bytes(&[base(with_romfs(&files))]), READ_LIMITS, 100).unwrap();
    let romfs = title.romfs.unwrap();
    assert_eq!(file_in(&romfs, "/big.bin").unwrap(), vec![1; 100]);
}

#[test]
fn a_romfs_is_equal_to_its_clones_and_to_no_other() {
    let files = [("f", Content::raw(b"f".to_vec()))];
    let one = open(bytes(&[base(with_romfs(&files))]))
        .unwrap()
        .romfs
        .unwrap();
    let other = open(bytes(&[base(with_romfs(&files))]))
        .unwrap()
        .romfs
        .unwrap();
    assert_eq!(one.clone(), one);
    assert_ne!(one, other);
    assert_eq!(one.size(), other.size());
    assert_eq!(
        format!("{one:?}"),
        format!("RomFs {{ size: {}, .. }}", one.size())
    );
}

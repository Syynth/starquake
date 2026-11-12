//! The normaliser against the reference's cases: each branch of `PathNormalizer`
//! (`eden/src/core/file_sys/fs_path_utility.h`), with the answer worked out by hand from the
//! reference's code.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::*;

/// The service's buffer: `EntryNameLengthMax + 1`.
const MAX: usize = ENTRY_NAME_LENGTH_MAX + 1;

/// `normalize` as `fsp-srv` calls it: not a Windows path, not drive-relative, checking characters.
fn plain(path: &[u8]) -> Result<Vec<u8>, PathError> {
    normalize(path, MAX, false, false, false)
}

#[test]
fn normalised_paths_come_back_as_they_are() {
    for path in [&b"/"[..], b"/a", b"/a/b", b"/.a", b"/..a", b"/a.", b"/a..b"] {
        assert_eq!(
            plain(path).unwrap(),
            path,
            "{:?}",
            String::from_utf8_lossy(path)
        );
        assert_eq!(is_normalized(path, false), Ok(true));
    }
}

#[test]
fn separators_collapse_and_a_trailing_one_goes() {
    let cases: [(&[u8], &[u8]); 5] = [
        (b"//", b"/"),
        (b"/a/", b"/a"),
        (b"//a//b", b"/a/b"),
        (b"/a///b///", b"/a/b"),
        (b"///a", b"/a"),
    ];
    for (path, expected) in cases {
        assert_eq!(plain(path).unwrap(), expected);
    }
}

#[test]
fn current_directory_components_go() {
    let cases: [(&[u8], &[u8]); 5] = [
        (b"/.", b"/"),
        (b"/./a", b"/a"),
        (b"/a/./b", b"/a/b"),
        (b"/a/.", b"/a"),
        (b"/././.", b"/"),
    ];
    for (path, expected) in cases {
        assert_eq!(plain(path).unwrap(), expected);
    }
}

#[test]
fn parent_directory_components_remove_the_one_before() {
    let cases: [(&[u8], &[u8]); 6] = [
        (b"/a/..", b"/"),
        (b"/a/../b", b"/b"),
        (b"/a/b/..", b"/a"),
        (b"/a/b/../../c", b"/c"),
        (b"/a/./../b", b"/b"),
        (b"/ab/cd/../ef", b"/ab/ef"),
    ];
    for (path, expected) in cases {
        assert_eq!(plain(path).unwrap(), expected);
    }
}

#[test]
fn climbing_above_the_root_is_unobtainable_unless_windows() {
    for path in [&b"/.."[..], b"/a/../..", b"/./..", b"/../a"] {
        assert_eq!(plain(path), Err(PathError::DirectoryUnobtainable));
    }
    // A Windows path may: the root's separator goes too, and an empty output is `/`.
    assert_eq!(normalize(b"/..", MAX, true, false, false).unwrap(), b"/");
}

#[test]
fn a_path_that_does_not_start_with_a_separator_is_drive_relative_or_refused() {
    for path in [&b"a"[..], b"", b"./a", b"\\a"] {
        assert_eq!(plain(path), Err(PathError::InvalidFormat));
    }
    let relative = |path| normalize(path, MAX, false, true, false);
    assert_eq!(relative(b"a").unwrap(), b"/a");
    assert_eq!(relative(b"a/b/").unwrap(), b"/a/b");
    assert_eq!(relative(b"").unwrap(), b"/");
    assert_eq!(relative(b"./a").unwrap(), b"/a");
    assert_eq!(relative(b".."), Err(PathError::DirectoryUnobtainable));
}

#[test]
fn each_invalid_character_is_refused_unless_all_are_allowed() {
    for c in INVALID_CHARACTERS {
        let path = [b'/', b'a', c, b'b'];
        assert_eq!(
            plain(&path),
            Err(PathError::InvalidCharacter),
            "{}",
            c as char
        );
        assert_eq!(
            normalize(&path, MAX, false, false, true).unwrap(),
            path,
            "{}",
            c as char
        );
        assert_eq!(
            is_normalized(&path, false),
            Err(PathError::InvalidCharacter)
        );
        assert_eq!(is_normalized(&path, true), Ok(true));
    }
    // The characters around them are allowed.
    for c in [b'9', b';', b'=', b'@', b'{', b'}', b'\\', b'"', 0x80, 0xFF] {
        assert!(!is_invalid_character(c));
        assert!(plain(&[b'/', c]).is_ok());
    }
}

#[test]
fn the_path_ends_at_its_first_zero_byte() {
    assert_eq!(plain(b"/a\0/b").unwrap(), b"/a");
    assert_eq!(plain(b"/a\0:").unwrap(), b"/a");
    assert_eq!(is_normalized(b"/a\0/", false), Ok(true));
}

#[test]
fn the_longest_path_fits_and_one_byte_more_does_not() {
    // `/` and a component that ends exactly at the last byte before the terminator.
    let mut path = vec![b'/'];
    path.resize(ENTRY_NAME_LENGTH_MAX, b'a');
    assert_eq!(plain(&path).unwrap(), path);
    path.push(b'a');
    assert_eq!(plain(&path), Err(PathError::TooLong));
    // What is removed does not count: the same length with a `.` component in it fits.
    let mut shorter = b"/./".to_vec();
    shorter.resize(ENTRY_NAME_LENGTH_MAX + 2, b'a');
    assert_eq!(plain(&shorter).unwrap().len(), ENTRY_NAME_LENGTH_MAX);
}

#[test]
fn a_separator_that_would_fill_the_buffer_is_too_long() {
    // Room for `/a` and its terminator: the second separator would leave none.
    assert_eq!(
        normalize(b"/a/b", 3, false, false, false),
        Err(PathError::TooLong)
    );
    // Room for `/a/` and a terminator, but not for `b` as well.
    assert_eq!(
        normalize(b"/a/b", 4, false, false, false),
        Err(PathError::TooLong)
    );
    assert_eq!(normalize(b"/a/b", 5, false, false, false).unwrap(), b"/a/b");
}

#[test]
fn the_last_byte_of_the_buffer_holds_the_path_and_not_its_terminator() {
    // The reference's final check is `max_out_size >= total_len - 1` ("Probable nintendo bug"):
    // a one-byte buffer holds `/`, with its terminator one past the end.
    assert_eq!(normalize(b"/", 1, false, false, false).unwrap(), b"/");
    assert_eq!(normalize(b"", 1, false, true, false).unwrap(), b"/");
}

#[test]
fn a_buffer_of_no_bytes_holds_no_path() {
    for path in [&b"/"[..], b"/a", b"/.", b"/a/.."] {
        assert_eq!(
            normalize(path, 0, false, false, false),
            Err(PathError::TooLong)
        );
    }
}

#[test]
fn backslashes_next_to_a_parent_component_become_separators() {
    let needed: [&[u8]; 5] = [b"/a\\..\\b", b"/a\\..", b"/..\\a", b"\\..\\", b"/a\\../b"];
    for path in needed {
        assert!(
            is_parent_directory_path_replacement_needed(path),
            "{:?}",
            String::from_utf8_lossy(path)
        );
    }
    let not_needed: [&[u8]; 6] = [b"/a/../b", b"/", b"/a", b"a\\..\\b", b"/a\\b", b"/a\\.b"];
    for path in not_needed {
        assert!(
            !is_parent_directory_path_replacement_needed(path),
            "{:?}",
            String::from_utf8_lossy(path)
        );
    }
    assert_eq!(replace_parent_directory_path(b"/a\\..\\b"), b"/a/../b");
    assert_eq!(replace_parent_directory_path(b"/a\\.."), b"/a/..");
    assert_eq!(replace_parent_directory_path(b"\\..\\"), b"/../");
    // Only the separators next to the `..` change.
    assert_eq!(
        replace_parent_directory_path(b"/x\\y\\..\\z"),
        b"/x\\y/../z"
    );

    assert_eq!(plain(b"/a\\..\\b").unwrap(), b"/b");
    assert_eq!(plain(b"/a\\..").unwrap(), b"/");
    // `x\\y` is one component, which the `..` removes.
    assert_eq!(plain(b"/x\\y\\..\\z").unwrap(), b"/z");
    // A backslash elsewhere is an ordinary character.
    assert_eq!(plain(b"/a\\b").unwrap(), b"/a\\b");
    // So `is_normalized` passes a path whose replacement climbs above the root, as in the
    // reference: the two functions see backslashes differently.
    assert_eq!(is_normalized(b"/..\\", false), Ok(true));
    assert_eq!(plain(b"/..\\"), Err(PathError::DirectoryUnobtainable));
}

#[test]
fn is_normalized_names_each_way_a_path_is_not() {
    let not: [&[u8]; 7] = [
        b"//", b"/a/", b"/./a", b"/a/.", b"/a/..", b"/../a", b"/a//b",
    ];
    for path in not {
        assert_eq!(
            is_normalized(path, false),
            Ok(false),
            "{:?}",
            String::from_utf8_lossy(path)
        );
    }
    assert_eq!(is_normalized(b"", false), Err(PathError::InvalidFormat));
    assert_eq!(is_normalized(b"a", false), Err(PathError::InvalidFormat));
    // The first byte is not checked for characters: it must be the separator.
    assert_eq!(is_normalized(b":", false), Err(PathError::InvalidFormat));
}

#[test]
fn errors_come_in_the_order_of_the_path() {
    // A path that does not start with a separator is refused before its characters are read.
    assert_eq!(plain(b"a:b"), Err(PathError::InvalidFormat));
    // A climb above the root before an invalid character, and the other way round.
    assert_eq!(plain(b"/../a:b"), Err(PathError::DirectoryUnobtainable));
    assert_eq!(plain(b"/a:b/../.."), Err(PathError::InvalidCharacter));
    // An invalid character in a component that does not fit: the character is found first.
    assert_eq!(
        normalize(b"/abc:", 3, false, false, false),
        Err(PathError::InvalidCharacter)
    );
    // Too long before a later climb above the root.
    assert_eq!(
        normalize(b"/abc/../..", 3, false, false, false),
        Err(PathError::TooLong)
    );
    // `is_normalized`: a not-normalised path stops the scan before a later invalid character.
    assert_eq!(is_normalized(b"//:", false), Ok(false));
    assert_eq!(
        is_normalized(b"/:/", false),
        Err(PathError::InvalidCharacter)
    );
}

#[test]
fn each_error_has_the_reference_s_result() {
    // `eden/src/core/file_sys/errors.h`, module FS (2): description in bits 9 and up.
    let cases = [
        (PathError::TooLong, 6003),
        (PathError::InvalidCharacter, 6004),
        (PathError::InvalidFormat, 6005),
        (PathError::DirectoryUnobtainable, 6006),
    ];
    for (error, description) in cases {
        let code = error.result();
        assert_eq!(code.module(), FS_MODULE);
        assert_eq!(code.description(), description);
    }
    assert_eq!(PathError::TooLong.result().raw(), 2 | (6003 << 9));
}

fn config() -> Config {
    Config {
        cases: 2048,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(0x5041_5448_4E4F_524D),
        rng_algorithm: RngAlgorithm::ChaCha,
        ..Config::default()
    }
}

/// Paths built from the bytes that matter: separators, dots, a letter, a backslash and an
/// invalid character.
fn paths() -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(
        prop_oneof![
            Just(b'/'),
            Just(b'.'),
            Just(b'a'),
            Just(b'b'),
            Just(b'\\'),
            Just(b':')
        ],
        0..24,
    )
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn what_normalize_gives_is_normalized_and_normalizes_to_itself(path in paths()) {
        if let Ok(normalized) = plain(&path) {
            prop_assert_eq!(is_normalized(&normalized, false), Ok(true));
            prop_assert_eq!(plain(&normalized), Ok(normalized.clone()));
            prop_assert!(normalized.len() < MAX);
        }
    }

    /// Without backslashes: `is_normalized` takes one as an ordinary character, but `normalize`
    /// first turns one next to `..` into a separator (`backslashes_next_to_a_parent_component_\
    /// become_separators`).
    #[test]
    fn a_normalized_path_is_its_own_normal_form(path in paths()) {
        if is_normalized(&path, false) == Ok(true) && !path.contains(&ALTERNATE_DIRECTORY_SEPARATOR) {
            prop_assert_eq!(plain(&path), Ok(path.clone()));
        }
    }
}

//! The file-system path normaliser: Horizon's rules for turning the path a guest sends into one
//! canonical path, or refusing it.
//!
//! Ported from `PathNormalizer` and the helpers it calls (`IsCurrentDirectory`,
//! `IsParentDirectory`, `StringTraits::IsInvalidCharacter`) in
//! `eden/src/core/file_sys/fs_path_utility.h` @ cf654052 (GPL-3.0-or-later, from yuzu,
//! GPL-2.0-or-later). The result codes are `eden/src/core/file_sys/errors.h`'s.
//!
//! A path is a C string: it ends at its first zero byte, or at the end of the slice, whichever
//! comes first. Every read past the end of the slice sees a zero byte, which is what the
//! reference's terminator gives it.
//!
//! - [`normalize`]: the canonical form of a path. It starts with `/`, has no empty, `.` or `..`
//!   components and no trailing separator, and is at most `max_out_size - 1` bytes.
//! - [`is_normalized`]: whether a path is already in that form.
//!
//! # Where this differs from the reference
//!
//! Each difference is a case where the reference's C++ reads or writes outside its buffers, or
//! asserts something it then ignores. None changes the answer for a path that does not reach
//! it. Each is pinned by the test named, in `fsp/path/tests.rs`:
//!
//! - `Normalize` writes into a caller's buffer of `max_out_size` bytes without checking the
//!   leading separator it adds to a drive-relative path, and the terminator it adds at the end
//!   can land one byte past the buffer ("Probable nintendo bug" there). Here the output is a
//!   growing vector, so nothing is written out of bounds; the answers are the reference's
//!   (`the_last_byte_of_the_buffer_holds_the_path_and_not_its_terminator`).
//! - When a component would not fit, the reference copies the part that fits before it fails with
//!   `ResultTooLongPath`; here nothing is returned but the error. With `max_out_size` zero, the
//!   reference's length arithmetic wraps; here every path is `ResultTooLongPath`
//!   (`a_buffer_of_no_bytes_holds_no_path`).
//! - Removing a component for `..` walks back over the output with a counter that, in the
//!   reference, wraps below zero if the output does not start with a separator (its `ASSERT`s say
//!   that cannot happen, and only log when it does). Here the counter stops at zero.
//! - `IsNormalized` leaves its length output unset when it finds a path is not normalised; this
//!   port returns whether the path is normalised and nothing else, since no caller here uses the
//!   length.

use crate::framework::result::code;
use starquake_kernel::result::ResultCode;

#[cfg(test)]
mod tests;

/// `ErrorModule::FS` (`eden/src/core/hle/result.h`).
pub const FS_MODULE: u32 = 2;

/// `EntryNameLengthMax` (`eden/src/core/file_sys/fs_directory.h`): the longest path, without its
/// terminator.
pub const ENTRY_NAME_LENGTH_MAX: usize = 0x300;

/// The separator (`StringTraits::DirectorySeparator`).
pub const DIRECTORY_SEPARATOR: u8 = b'/';
/// The separator Windows paths also use (`StringTraits::AlternateDirectorySeparator`).
pub const ALTERNATE_DIRECTORY_SEPARATOR: u8 = b'\\';
/// `StringTraits::Dot`.
const DOT: u8 = b'.';
/// `StringTraits::NullTerminator`.
const NULL_TERMINATOR: u8 = 0;

/// The characters a path component may not hold unless all are allowed
/// (`StringTraits::InvalidCharacters`).
pub const INVALID_CHARACTERS: [u8; 6] = *b":*?<>|";

/// `ResultTooLongPath` (FS 6003).
pub const TOO_LONG_PATH: ResultCode = code(FS_MODULE, 6003);
/// `ResultInvalidCharacter` (FS 6004).
pub const INVALID_CHARACTER: ResultCode = code(FS_MODULE, 6004);
/// `ResultInvalidPathFormat` (FS 6005).
pub const INVALID_PATH_FORMAT: ResultCode = code(FS_MODULE, 6005);
/// `ResultDirectoryUnobtainable` (FS 6006).
pub const DIRECTORY_UNOBTAINABLE: ResultCode = code(FS_MODULE, 6006);

/// Why a path was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PathError {
    /// The path is longer than the output may be (`ResultTooLongPath`).
    #[error("the path is too long")]
    TooLong,
    /// A component holds a character paths may not hold (`ResultInvalidCharacter`).
    #[error("the path holds a character paths may not hold")]
    InvalidCharacter,
    /// The path does not start the way it must (`ResultInvalidPathFormat`).
    #[error("the path is not in a form paths may take")]
    InvalidFormat,
    /// A `..` climbs above the root (`ResultDirectoryUnobtainable`).
    #[error("the path climbs above its root")]
    DirectoryUnobtainable,
}

impl PathError {
    /// The result code the reference fails with.
    #[must_use]
    pub fn result(self) -> ResultCode {
        match self {
            Self::TooLong => TOO_LONG_PATH,
            Self::InvalidCharacter => INVALID_CHARACTER,
            Self::InvalidFormat => INVALID_PATH_FORMAT,
            Self::DirectoryUnobtainable => DIRECTORY_UNOBTAINABLE,
        }
    }
}

/// The byte at `index`, or the terminator past the end.
fn at(path: &[u8], index: usize) -> u8 {
    path.get(index).copied().unwrap_or(NULL_TERMINATOR)
}

/// `StringTraits::IsInvalidCharacter`.
#[must_use]
pub fn is_invalid_character(c: u8) -> bool {
    INVALID_CHARACTERS.contains(&c)
}

/// `IsCurrentDirectory(path + start)`: a `.` component starts there.
fn is_current_directory(path: &[u8], start: usize) -> bool {
    let next = at(path, start.saturating_add(1));
    at(path, start) == DOT && (next == NULL_TERMINATOR || next == DIRECTORY_SEPARATOR)
}

/// `IsParentDirectory(path + start)`: a `..` component starts there.
fn is_parent_directory(path: &[u8], start: usize) -> bool {
    let third = at(path, start.saturating_add(2));
    at(path, start) == DOT
        && at(path, start.saturating_add(1)) == DOT
        && (third == NULL_TERMINATOR || third == DIRECTORY_SEPARATOR)
}

/// Whether `c` is either separator.
fn is_separator(c: u8) -> bool {
    c == DIRECTORY_SEPARATOR || c == ALTERNATE_DIRECTORY_SEPARATOR
}

/// `PathNormalizer::IsParentDirectoryPathReplacementNeeded`: whether a path that starts with a
/// separator has a `..` component next to a backslash, which [`normalize`] rewrites with forward
/// separators first.
///
/// The reference's loop advances the path pointer and keeps its index at zero, so it looks at
/// every run of four bytes in turn, and then at the last three; this port walks the same runs.
#[must_use]
pub fn is_parent_directory_path_replacement_needed(path: &[u8]) -> bool {
    if !is_separator(at(path, 0)) {
        return false;
    }
    if at(path, 1) == NULL_TERMINATOR || at(path, 2) == NULL_TERMINATOR {
        return false;
    }
    // The pointer the reference advances; its index stays 0. Each turn reads one byte further
    // than the last, so the loop ends at the terminator.
    let mut base = 0usize;
    while at(path, base.saturating_add(3)) != NULL_TERMINATOR {
        let dots =
            at(path, base.saturating_add(1)) == DOT && at(path, base.saturating_add(2)) == DOT;
        if dots {
            let c0 = at(path, base);
            let c3 = at(path, base.saturating_add(3));
            // `c3` is never the terminator inside the loop; the reference tests for it anyway.
            if c0 == ALTERNATE_DIRECTORY_SEPARATOR && (is_separator(c3) || c3 == NULL_TERMINATOR) {
                return true;
            }
            if c3 == ALTERNATE_DIRECTORY_SEPARATOR && is_separator(c0) {
                return true;
            }
        }
        base = base.saturating_add(1);
    }
    at(path, base) == ALTERNATE_DIRECTORY_SEPARATOR
        && at(path, base.saturating_add(1)) == DOT
        && at(path, base.saturating_add(2)) == DOT
}

/// `PathNormalizer::ReplaceParentDirectoryPath`: the path with every separator next to a `..`
/// component made a forward one, and its first byte a forward separator.
fn replace_parent_directory_path(src: &[u8]) -> Vec<u8> {
    let len = src
        .iter()
        .position(|byte| *byte == NULL_TERMINATOR)
        .unwrap_or(src.len());
    // Room for every byte of the path; the reference's buffer is `EntryNameLengthMax + 1` and its
    // callers' paths fit it.
    let mut dst = vec![NULL_TERMINATOR; len.max(1)];
    let mut put = |index: usize, byte: u8| {
        if let Some(slot) = dst.get_mut(index) {
            *slot = byte;
        }
    };
    put(0, DIRECTORY_SEPARATOR);
    let mut i = 1usize;
    while at(src, i) != NULL_TERMINATOR {
        let before = at(src, i.saturating_sub(1));
        let dots = at(src, i) == DOT && at(src, i.saturating_add(1)) == DOT;
        let after = at(src, i.saturating_add(2));
        if is_separator(before) && dots && is_separator(after) {
            put(i.saturating_sub(1), DIRECTORY_SEPARATOR);
            put(i, DOT);
            put(i.saturating_add(1), DOT);
            put(i.saturating_add(2), DIRECTORY_SEPARATOR);
            i = i.saturating_add(3);
        } else {
            if before == ALTERNATE_DIRECTORY_SEPARATOR && dots && after == NULL_TERMINATOR {
                put(i.saturating_sub(1), DIRECTORY_SEPARATOR);
                put(i, DOT);
                put(i.saturating_add(1), DOT);
                i = i.saturating_add(2);
                break;
            }
            put(i, at(src, i));
            i = i.saturating_add(1);
        }
    }
    dst.truncate(i);
    dst
}

/// Where [`is_normalized`] is in a path (`PathNormalizer::PathState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PathState {
    Start,
    Normal,
    FirstSeparator,
    Separator,
    CurrentDir,
    ParentDir,
}

/// `PathNormalizer::IsNormalized`: whether `path` is in the form [`normalize`] gives. A path that
/// does not start with `/` is [`PathError::InvalidFormat`]; a character paths may not hold,
/// after the first byte and unless `allow_all_characters`, is [`PathError::InvalidCharacter`].
/// Both are found in the order the bytes come, and a path found not normalised stops the scan.
pub fn is_normalized(path: &[u8], allow_all_characters: bool) -> Result<bool, PathError> {
    let mut state = PathState::Start;
    let mut len = 0usize;
    loop {
        let c = at(path, len);
        if c == NULL_TERMINATOR {
            break;
        }
        len = len.saturating_add(1);
        if !allow_all_characters && state != PathState::Start && is_invalid_character(c) {
            return Err(PathError::InvalidCharacter);
        }
        state = match state {
            PathState::Start if c == DIRECTORY_SEPARATOR => PathState::FirstSeparator,
            PathState::Start => return Err(PathError::InvalidFormat),
            PathState::Normal if c == DIRECTORY_SEPARATOR => PathState::Separator,
            PathState::FirstSeparator | PathState::Separator | PathState::CurrentDir
                if c == DIRECTORY_SEPARATOR =>
            {
                return Ok(false);
            }
            PathState::FirstSeparator | PathState::Separator if c == DOT => PathState::CurrentDir,
            PathState::CurrentDir if c == DOT => PathState::ParentDir,
            PathState::FirstSeparator | PathState::Separator | PathState::CurrentDir => {
                PathState::Normal
            }
            PathState::ParentDir if c == DIRECTORY_SEPARATOR => return Ok(false),
            PathState::Normal | PathState::ParentDir => PathState::Normal,
        };
    }
    match state {
        PathState::Start => Err(PathError::InvalidFormat),
        PathState::Normal | PathState::FirstSeparator => Ok(true),
        PathState::Separator | PathState::CurrentDir | PathState::ParentDir => Ok(false),
    }
}

/// The output of [`normalize`] as it is built: the reference's `dst` and `total_len`. Bytes past
/// `len` stay where earlier writes left them, as in the reference's buffer, because removing a
/// component for `..` reads them.
struct Output {
    bytes: Vec<u8>,
    len: usize,
}

impl Output {
    /// `dst[total_len++] = byte`.
    fn push(&mut self, byte: u8) {
        match self.bytes.get_mut(self.len) {
            Some(slot) => *slot = byte,
            None => self.bytes.push(byte),
        }
        self.len = self.len.saturating_add(1);
    }

    fn get(&self, index: usize) -> u8 {
        at(&self.bytes, index)
    }

    /// Removes the last component for a `..`: steps back over the separator just written and
    /// then to the separator before the component (the reference's `total_len -= 2` and its
    /// `do ... while (--total_len != 0)`).
    fn pop_component(&mut self) {
        self.len = self.len.saturating_sub(2);
        loop {
            if self.get(self.len) == DIRECTORY_SEPARATOR {
                break;
            }
            self.len = self.len.saturating_sub(1);
            if self.len == 0 {
                break;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.bytes.truncate(self.len);
        self.bytes
    }
}

/// `PathNormalizer::Normalize`: the canonical form of `path`, to fit a buffer of `max_out_size`
/// bytes with its terminator.
///
/// - A path that does not start with `/` is [`PathError::InvalidFormat`] unless
///   `is_drive_relative_path`, which puts a `/` in front of it.
/// - A component that holds a character paths may not hold is [`PathError::InvalidCharacter`]
///   unless `allow_all_characters`.
/// - Repeated separators become one; `.` components go; a `..` component removes the one before
///   it, and one at the root is [`PathError::DirectoryUnobtainable`] unless `is_windows_path`;
///   a trailing separator goes.
/// - An output that would not fit is [`PathError::TooLong`].
///
/// The errors are found in the order the path's bytes come, and the output is checked with
/// [`is_normalized`] at the end, which can fail with that function's errors.
pub fn normalize(
    path: &[u8],
    max_out_size: usize,
    is_windows_path: bool,
    is_drive_relative_path: bool,
    allow_all_characters: bool,
) -> Result<Vec<u8>, PathError> {
    let mut out = Output {
        bytes: Vec::new(),
        len: 0,
    };
    if at(path, 0) != DIRECTORY_SEPARATOR {
        if !is_drive_relative_path {
            return Err(PathError::InvalidFormat);
        }
        out.push(DIRECTORY_SEPARATOR);
    }

    let replaced;
    let cur: &[u8] = if is_parent_directory_path_replacement_needed(path) {
        replaced = replace_parent_directory_path(path);
        &replaced
    } else {
        path
    };

    let mut skip_next_sep = false;
    let mut i = 0usize;
    while at(cur, i) != NULL_TERMINATOR {
        if at(cur, i) == DIRECTORY_SEPARATOR {
            // Swallow separators.
            loop {
                i = i.saturating_add(1);
                if at(cur, i) != DIRECTORY_SEPARATOR {
                    break;
                }
            }
            if at(cur, i) == NULL_TERMINATOR {
                break;
            }
            if !skip_next_sep {
                if out.len.saturating_add(1) == max_out_size {
                    return Err(PathError::TooLong);
                }
                out.push(DIRECTORY_SEPARATOR);
            }
            skip_next_sep = false;
        }

        // The current component's length, checking its characters.
        let mut dir_len = 0usize;
        loop {
            let c = at(cur, i.saturating_add(dir_len));
            if c == DIRECTORY_SEPARATOR || c == NULL_TERMINATOR {
                break;
            }
            if !allow_all_characters && is_invalid_character(c) {
                return Err(PathError::InvalidCharacter);
            }
            dir_len = dir_len.saturating_add(1);
        }

        if is_current_directory(cur, i) {
            skip_next_sep = true;
        } else if is_parent_directory(cur, i) {
            if out.len == 1 {
                if !is_windows_path {
                    return Err(PathError::DirectoryUnobtainable);
                }
                out.len = 0;
            } else {
                out.pop_component();
            }
        } else {
            let needed = out.len.saturating_add(dir_len).saturating_add(1);
            if needed > max_out_size {
                return Err(PathError::TooLong);
            }
            for j in 0..dir_len {
                out.push(at(cur, i.saturating_add(j)));
            }
        }
        i = i.saturating_add(dir_len);
    }

    if skip_next_sep {
        out.len = out.len.saturating_sub(1);
    }
    if out.len == 0 && max_out_size != 0 {
        out.len = 0;
        out.push(DIRECTORY_SEPARATOR);
    }
    // The reference's `max_out_size >= total_len - 1`, in its unsigned arithmetic: with nothing
    // written the subtraction wraps and the check fails.
    if max_out_size < out.len.wrapping_sub(1) {
        return Err(PathError::TooLong);
    }
    let normalized = out.finish();
    if !is_normalized(&normalized, allow_all_characters)? {
        // The reference asserts this cannot happen; an assertion there only logs.
        tracing::error!("the path normaliser produced a path that is not normalised");
    }
    Ok(normalized)
}

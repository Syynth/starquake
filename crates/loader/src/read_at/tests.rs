//! Positional reads: offsets up to and past the end, the exact-read loop, the cursor adapter,
//! and a host file.

use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::sync::Mutex;

use super::*;

/// Bytes whose every value is distinct, so a read from the wrong offset shows.
fn bytes() -> Vec<u8> {
    (0u8..32).collect()
}

#[test]
fn a_read_at_any_offset_returns_the_bytes_there_and_none_at_or_past_the_end() {
    let source = bytes();
    let len = u64::try_from(source.len()).unwrap();
    assert_eq!(source.size().unwrap(), len);
    for offset in 0..=len + 1 {
        let mut buf = [0xEE; 5];
        let count = source.read_at(offset, &mut buf).unwrap();
        let start = usize::try_from(offset).unwrap().min(source.len());
        let expected = &source[start..(start + 5).min(source.len())];
        assert_eq!(&buf[..count], expected, "at {offset}");
        // Bytes past the count are untouched.
        assert!(buf[count..].iter().all(|&b| b == 0xEE));
    }
    // The last byte, exactly at the end, and one past it.
    let mut one = [0];
    assert_eq!(source.read_at(len - 1, &mut one).unwrap(), 1);
    assert_eq!(one, [31]);
    assert_eq!(source.read_at(len, &mut one).unwrap(), 0);
    assert_eq!(source.read_at(len + 1, &mut one).unwrap(), 0);
    assert_eq!(source.read_at(u64::MAX, &mut one).unwrap(), 0);
    // A slice and a shared or borrowed source read the same.
    let shared = Arc::new(source.clone());
    let mut a = [0; 4];
    let mut b = [0; 4];
    let mut c = [0; 4];
    shared.read_at(9, &mut a).unwrap();
    source.as_slice().read_at(9, &mut b).unwrap();
    source.read_at(9, &mut c).unwrap();
    assert_eq!((a, b, c), ([9, 10, 11, 12], a, a));
}

/// A source that hands out one byte per call, after an interruption, and counts its calls.
struct Trickle {
    bytes: Vec<u8>,
    interrupted: Mutex<bool>,
    calls: Mutex<usize>,
}

impl ReadAt for Trickle {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        *self.calls.lock().unwrap() += 1;
        let mut interrupted = self.interrupted.lock().unwrap();
        if !*interrupted {
            *interrupted = true;
            return Err(io::Error::from(ErrorKind::Interrupted));
        }
        let end = buf.len().min(1);
        Ok(read_slice(&self.bytes, offset, &mut buf[..end]))
    }

    fn size(&self) -> io::Result<u64> {
        Ok(slice_size(&self.bytes))
    }
}

#[test]
fn read_exact_at_loops_over_short_and_interrupted_reads_and_fails_at_the_end() {
    let trickle = Trickle {
        bytes: bytes(),
        interrupted: Mutex::new(false),
        calls: Mutex::new(0),
    };
    let mut buf = [0; 4];
    read_exact_at(&trickle, 30, &mut buf[..2]).unwrap();
    assert_eq!(buf[..2], [30, 31]);
    // One interruption and two one-byte reads.
    assert_eq!(*trickle.calls.lock().unwrap(), 3);
    // Ending exactly at the end is fine; one byte past it is not.
    read_exact_at(&bytes(), 28, &mut buf).unwrap();
    assert_eq!(buf, [28, 29, 30, 31]);
    let error = read_exact_at(&bytes(), 29, &mut buf).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::UnexpectedEof);
    assert_eq!(
        error.to_string(),
        "the source ended before the bytes asked for"
    );
    // Nothing to read is nothing to fail.
    read_exact_at(&bytes(), u64::MAX, &mut []).unwrap();
}

/// A source that claims bytes at every offset, to drive offsets to the top of the range.
struct Endless;

impl ReadAt for Endless {
    fn read_at(&self, _offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        buf.fill(7);
        Ok(buf.len())
    }

    fn size(&self) -> io::Result<u64> {
        Ok(u64::MAX)
    }
}

#[test]
fn an_offset_that_would_pass_u64_max_is_an_error_not_a_wrap() {
    let mut buf = [0; 2];
    // The read itself fits; the offset after it does not.
    let error = read_exact_at(&Endless, u64::MAX - 1, &mut buf).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    read_exact_at(&Endless, u64::MAX - 2, &mut buf).unwrap();
    let mut cursor = Positional::new(Endless);
    cursor.seek(SeekFrom::Start(u64::MAX - 1)).unwrap();
    assert_eq!(
        cursor.read(&mut buf).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
}

#[test]
fn the_cursor_adapter_reads_and_seeks_without_moving_positional_reads() {
    let mut cursor = Positional::new(bytes());
    let mut buf = [0; 3];
    cursor.read_exact(&mut buf).unwrap();
    assert_eq!(buf, [0, 1, 2]);
    assert_eq!(cursor.seek(SeekFrom::Current(5)).unwrap(), 8);
    // A positional read in between neither uses nor moves the cursor.
    assert_eq!(cursor.read_at(20, &mut buf).unwrap(), 3);
    assert_eq!(buf, [20, 21, 22]);
    cursor.read_exact(&mut buf).unwrap();
    assert_eq!(buf, [8, 9, 10]);
    assert_eq!(cursor.seek(SeekFrom::End(-2)).unwrap(), 30);
    let mut rest = Vec::new();
    cursor.read_to_end(&mut rest).unwrap();
    assert_eq!(rest, [30, 31]);
    // Past the end reads nothing; before the start is an error that leaves the cursor.
    assert_eq!(cursor.seek(SeekFrom::End(4)).unwrap(), 36);
    assert_eq!(cursor.read(&mut buf).unwrap(), 0);
    let error = cursor.seek(SeekFrom::Current(-37)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(cursor.stream_position().unwrap(), 36);
    assert_eq!(cursor.seek(SeekFrom::Current(-36)).unwrap(), 0);
    assert_eq!(cursor.size().unwrap(), 32);
    assert_eq!(cursor.get_ref().len(), 32);
    assert_eq!(cursor.into_inner(), bytes());
}

/// A per-test folder under the system's temporary folder, removed when dropped.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(test: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("starquake-loader-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(unix)]
#[test]
fn a_host_file_reads_at_offsets_without_moving_its_cursor() {
    let scratch = Scratch::new("read-at-file");
    let path = scratch.0.join("bytes");
    std::fs::write(&path, bytes()).unwrap();
    let mut file = std::fs::File::open(&path).unwrap();
    assert_eq!(ReadAt::size(&file).unwrap(), 32);
    let mut buf = [0; 4];
    assert_eq!(ReadAt::read_at(&file, 28, &mut buf).unwrap(), 4);
    assert_eq!(buf, [28, 29, 30, 31]);
    assert_eq!(ReadAt::read_at(&file, 32, &mut buf).unwrap(), 0);
    // The file's own cursor is still at the start.
    assert_eq!(file.read(&mut buf).unwrap(), 4);
    assert_eq!(buf, [0, 1, 2, 3]);
}

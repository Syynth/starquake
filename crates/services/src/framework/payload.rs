//! The explicit little-endian [`Reader`] and [`Writer`] a service method unpacks and packs its
//! raw payload with.
//!
//! A payload is the command's arguments or results as the console lays them out: a packed C
//! struct, little-endian, each field at its natural alignment. Methods read and write it field by
//! field and say where the padding is, so a method can be checked against its reference line by
//! line:
//!
//! ```
//! use starquake_services::framework::{Fault, Reader, Writer};
//!
//! // struct { u32 index; /* 4 bytes of padding */ u64 offset; }
//! let payload = [7, 0, 0, 0, 0xAA, 0xAA, 0xAA, 0xAA, 9, 0, 0, 0, 0, 0, 0, 0];
//! let mut r = Reader::new(&payload);
//! let index = r.read_u32()?;
//! r.align(8)?;
//! let offset = r.read_u64()?;
//! assert_eq!((index, offset), (7, 9));
//!
//! let mut w = Writer::new();
//! w.write_u32(index).align(8).write_u64(offset);
//! assert_eq!(w.as_bytes(), [7, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0]);
//! # Ok::<(), Fault>(())
//! ```
//!
//! Offsets and alignment are measured from the start of the payload. For CMIF that is the same
//! as measuring from the start of the message's raw data, because the payload starts on a 16-byte
//! boundary (`eden/src/core/hle/service/hle_ipc.cpp`, `ParseCommandBuffer`: "Padding to align to
//! 16 bytes", then 16-byte headers).
//!
//! Divergence from Eden, deliberate: Eden's `RequestParser::Pop` and `ResponseBuilder::Push`
//! (`eden/src/core/hle/service/ipc_helpers.h`) round every value up to a whole 32-bit word, so
//! two `u8` pops read bytes 0 and 4. Here a read or write moves by the value's own size and
//! nothing else; a method that ports an Eden handler states Eden's word rounding with `align(4)`.
//! The test `values_move_by_their_own_size_not_by_words` pins this.

use starquake_ipc::BUFFER_SIZE;

use super::result::Fault;

/// The position after `pos` that is a multiple of `to`. An alignment of 0 or 1 leaves `pos` as
/// it is. `None` if the position does not fit a `usize`.
fn aligned(pos: usize, to: usize) -> Option<usize> {
    if to <= 1 {
        Some(pos)
    } else {
        pos.checked_next_multiple_of(to)
    }
}

/// A bounds-checked cursor over a request payload. Every read is little-endian and moves the
/// cursor by exactly the bytes read; a read past the end is [`Fault::PayloadTruncated`] and
/// leaves the cursor where it was.
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

macro_rules! read_integers {
    ($($name:ident -> $int:ty;)*) => {$(
        #[doc = concat!("Reads a little-endian `", stringify!($int), "`.")]
        pub fn $name(&mut self) -> Result<$int, Fault> {
            self.read_array().map(<$int>::from_le_bytes)
        }
    )*};
}

impl<'a> Reader<'a> {
    /// A reader at the start of `data`.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// The cursor's byte offset from the start of the payload.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Bytes between the cursor and the end of the payload.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn truncated(&self, needed: usize) -> Fault {
        Fault::PayloadTruncated {
            offset: self.pos,
            needed,
            len: self.data.len(),
        }
    }

    /// The bytes from the cursor to the end.
    fn rest(&self) -> &'a [u8] {
        self.data.get(self.pos..).unwrap_or_default()
    }

    /// Reads the next `len` bytes.
    pub fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], Fault> {
        let bytes = self.rest().get(..len);
        let bytes = bytes.ok_or_else(|| self.truncated(len))?;
        self.pos = self.pos.saturating_add(len);
        Ok(bytes)
    }

    /// Reads the next `N` bytes as an array.
    pub fn read_array<const N: usize>(&mut self) -> Result<[u8; N], Fault> {
        let array = self.rest().first_chunk::<N>().copied();
        let array = array.ok_or_else(|| self.truncated(N))?;
        self.pos = self.pos.saturating_add(N);
        Ok(array)
    }

    read_integers! {
        read_u8 -> u8;
        read_u16 -> u16;
        read_u32 -> u32;
        read_u64 -> u64;
        read_i32 -> i32;
        read_i64 -> i64;
    }

    /// Skips padding so the cursor is at a multiple of `to` bytes from the start of the payload.
    /// An alignment of 0 or 1 does nothing.
    pub fn align(&mut self, to: usize) -> Result<(), Fault> {
        let target = aligned(self.pos, to).filter(|target| *target <= self.data.len());
        self.pos = target.ok_or_else(|| self.truncated(to))?;
        Ok(())
    }
}

/// Builds a reply payload. Every write is little-endian and appends exactly the value's bytes.
///
/// Writes cannot fail where they are made. A reply has to fit the caller's message buffer
/// ([`Writer::LIMIT`]); a write that would pass that drops its bytes and marks the payload
/// [overflowed](Writer::overflowed), and the framework then answers the request with
/// [`NOT_SUPPORTED`](super::result::NOT_SUPPORTED) instead of a cut-off payload.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Writer {
    data: Vec<u8>,
    overflowed: bool,
}

macro_rules! write_integers {
    ($($name:ident($int:ty);)*) => {$(
        #[doc = concat!("Appends a little-endian `", stringify!($int), "`.")]
        pub fn $name(&mut self, value: $int) -> &mut Self {
            self.write_bytes(&value.to_le_bytes())
        }
    )*};
}

impl Writer {
    /// The most bytes a payload can hold: the size of a message buffer. A payload this long never
    /// fits next to its headers; the limit only bounds what a method can make the writer hold.
    pub const LIMIT: usize = BUFFER_SIZE;

    /// An empty payload.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes written so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether nothing has been written.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The payload so far.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Whether a write was dropped because the payload would have passed [`Writer::LIMIT`].
    #[must_use]
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// The length after `more` bytes, if that stays within the limit.
    fn grown(&self, more: usize) -> Option<usize> {
        let len = self.data.len().checked_add(more)?;
        (len <= Self::LIMIT).then_some(len)
    }

    /// Appends `bytes`.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> &mut Self {
        if self.grown(bytes.len()).is_some() {
            self.data.extend_from_slice(bytes);
        } else {
            self.overflowed = true;
        }
        self
    }

    write_integers! {
        write_u8(u8);
        write_u16(u16);
        write_u32(u32);
        write_u64(u64);
        write_i32(i32);
        write_i64(i64);
    }

    /// Appends zero bytes until the length is a multiple of `to`. An alignment of 0 or 1 does
    /// nothing.
    pub fn align(&mut self, to: usize) -> &mut Self {
        match aligned(self.data.len(), to).filter(|len| *len <= Self::LIMIT) {
            Some(len) => self.data.resize(len, 0),
            None => self.overflowed = true,
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_bytes_for_every_write() {
        let mut w = Writer::new();
        w.write_u8(0x11)
            .write_u16(0x2322)
            .align(4)
            .write_u32(0x4746_4544)
            .write_i32(-2)
            .write_bytes(&[0x51, 0x52, 0x53])
            .align(8)
            .write_u64(0x6867_6665_6463_6261)
            .write_i64(-3);
        #[rustfmt::skip]
        let expected = [
            0x11,                                           // u8 at 0
            0x22, 0x23,                                     // u16 at 1
            0x00,                                           // align(4)
            0x44, 0x45, 0x46, 0x47,                         // u32 at 4
            0xFE, 0xFF, 0xFF, 0xFF,                         // i32 at 8
            0x51, 0x52, 0x53,                               // bytes at 12
            0x00,                                           // align(8)
            0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, // u64 at 16
            0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // i64 at 24
        ];
        assert_eq!(w.as_bytes(), expected);
        assert_eq!(w.len(), expected.len());
        assert!(!w.overflowed());

        let mut r = Reader::new(&expected);
        assert_eq!(r.read_u8().unwrap(), 0x11);
        assert_eq!(r.read_u16().unwrap(), 0x2322);
        r.align(4).unwrap();
        assert_eq!(r.read_u32().unwrap(), 0x4746_4544);
        assert_eq!(r.read_i32().unwrap(), -2);
        assert_eq!(r.read_bytes(3).unwrap(), [0x51, 0x52, 0x53]);
        r.align(8).unwrap();
        assert_eq!(r.read_u64().unwrap(), 0x6867_6665_6463_6261);
        assert_eq!(r.read_i64().unwrap(), -3);
        assert_eq!((r.position(), r.remaining()), (expected.len(), 0));
    }

    #[test]
    fn values_move_by_their_own_size_not_by_words() {
        // Eden's Pop<u8> would read bytes 0 and 4 (see the module docs).
        let mut r = Reader::new(&[1, 2, 3, 4, 5]);
        assert_eq!((r.read_u8().unwrap(), r.read_u8().unwrap()), (1, 2));
        assert_eq!(r.position(), 2);
        let mut w = Writer::new();
        w.write_u8(1).write_u8(2);
        assert_eq!(w.as_bytes(), [1, 2]);
    }

    /// One step of a generated payload layout.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Step {
        U8,
        U16,
        U32,
        U64,
        I32,
        I64,
        Bytes,
        Align2,
        Align4,
        Align8,
    }

    const STEPS: [Step; 10] = [
        Step::U8,
        Step::U16,
        Step::U32,
        Step::U64,
        Step::I32,
        Step::I64,
        Step::Bytes,
        Step::Align2,
        Step::Align4,
        Step::Align8,
    ];

    /// A distinct value for the field at position `at`: every byte non-zero, and no two
    /// positions share a byte, so a field read at the wrong offset or width never matches.
    fn value(at: usize) -> u64 {
        let base = u64::try_from(at).unwrap() * 8 + 0x11;
        (0..8).fold(0, |acc, i| acc | ((base + i) << (i * 8)))
    }

    fn low<const N: usize>(value: u64) -> [u8; N] {
        let mut out = [0_u8; N];
        out.copy_from_slice(&value.to_le_bytes()[..N]);
        out
    }

    fn write(w: &mut Writer, step: Step, at: usize) {
        let v = value(at);
        match step {
            Step::U8 => w.write_u8(u8::from_le_bytes(low(v))),
            Step::U16 => w.write_u16(u16::from_le_bytes(low(v))),
            Step::U32 => w.write_u32(u32::from_le_bytes(low(v))),
            Step::U64 => w.write_u64(v),
            Step::I32 => w.write_i32(i32::from_le_bytes(low(v))),
            Step::I64 => w.write_i64(i64::from_le_bytes(low(v))),
            Step::Bytes => w.write_bytes(&low::<3>(v)),
            Step::Align2 => w.align(2),
            Step::Align4 => w.align(4),
            Step::Align8 => w.align(8),
        };
    }

    /// Reads the step back and says whether it holds what `write` put there.
    fn read_matches(r: &mut Reader<'_>, step: Step, at: usize) -> Result<bool, Fault> {
        let v = value(at);
        Ok(match step {
            Step::U8 => r.read_u8()? == u8::from_le_bytes(low(v)),
            Step::U16 => r.read_u16()? == u16::from_le_bytes(low(v)),
            Step::U32 => r.read_u32()? == u32::from_le_bytes(low(v)),
            Step::U64 => r.read_u64()? == v,
            Step::I32 => r.read_i32()? == i32::from_le_bytes(low(v)),
            Step::I64 => r.read_i64()? == i64::from_le_bytes(low(v)),
            Step::Bytes => r.read_bytes(3)? == low::<3>(v),
            Step::Align2 => r.align(2).is_ok(),
            Step::Align4 => r.align(4).is_ok(),
            Step::Align8 => r.align(8).is_ok(),
        })
    }

    /// Every layout of up to four steps, built and then read back. Generated exhaustively, so
    /// the run is the same every time.
    fn layouts() -> Vec<Vec<Step>> {
        let mut all = vec![Vec::new()];
        let mut last = all.clone();
        for _ in 0..4 {
            last = last
                .iter()
                .flat_map(|prefix| {
                    STEPS.iter().map(move |step| {
                        let mut next = prefix.clone();
                        next.push(*step);
                        next
                    })
                })
                .collect();
            all.extend(last.iter().cloned());
        }
        all
    }

    #[test]
    fn every_layout_reads_back_what_was_written() {
        for layout in layouts() {
            let mut w = Writer::new();
            for (at, step) in layout.iter().enumerate() {
                write(&mut w, *step, at);
            }
            let mut r = Reader::new(w.as_bytes());
            for (at, step) in layout.iter().enumerate() {
                assert_eq!(
                    read_matches(&mut r, *step, at),
                    Ok(true),
                    "{layout:?} step {at}"
                );
            }
            assert_eq!(r.remaining(), 0, "{layout:?}");
        }
    }

    #[test]
    fn swapping_two_different_reads_is_caught() {
        // The round trip above must fail when a method reads two neighbouring fields in the
        // wrong order. Checked for every pair of distinct value steps.
        let values = &STEPS[..7];
        for a in values {
            for b in values.iter().filter(|b| *b != a) {
                let mut w = Writer::new();
                write(&mut w, *a, 0);
                write(&mut w, *b, 1);
                let mut r = Reader::new(w.as_bytes());
                // Read b's kind first, then a's: at least one must fail or not match.
                let first = read_matches(&mut r, *b, 1) == Ok(true);
                let second = read_matches(&mut r, *a, 0) == Ok(true);
                assert!(!(first && second), "{a:?} then {b:?}");
            }
        }
    }

    #[test]
    fn a_read_past_the_end_is_an_error_and_leaves_the_cursor() {
        let data = [1, 2, 3, 4, 5, 6, 7];
        let mut r = Reader::new(&data);
        assert_eq!(r.read_u32().unwrap(), 0x0403_0201);
        // Exactly at the limit: three bytes are left.
        let truncated = |needed| Fault::PayloadTruncated {
            offset: 4,
            needed,
            len: data.len(),
        };
        assert_eq!(r.read_u32(), Err(truncated(4)));
        assert_eq!(r.read_u64(), Err(truncated(8)));
        assert_eq!(r.read_bytes(4), Err(truncated(4)));
        assert_eq!(r.read_bytes(usize::MAX), Err(truncated(usize::MAX)));
        assert_eq!(r.position(), 4);
        assert_eq!(r.read_bytes(3).unwrap(), [5, 6, 7]);
        assert_eq!(
            r.read_u8(),
            Err(Fault::PayloadTruncated {
                offset: 7,
                needed: 1,
                len: data.len()
            })
        );
    }

    #[test]
    fn align_stops_at_the_end_of_the_payload() {
        let data = [0_u8; 6];
        let mut r = Reader::new(&data);
        r.read_u8().unwrap();
        r.align(2).unwrap();
        assert_eq!(r.position(), 2);
        r.align(2).unwrap();
        assert_eq!(r.position(), 2);
        r.align(0).unwrap();
        r.align(1).unwrap();
        assert_eq!(r.position(), 2);
        // The next multiple of 4 is 4, inside the payload; of 8 is 8, past its 6 bytes.
        r.align(4).unwrap();
        r.read_u8().unwrap();
        assert_eq!(
            r.align(8),
            Err(Fault::PayloadTruncated {
                offset: 5,
                needed: 8,
                len: data.len()
            })
        );
        assert_eq!(r.position(), 5);
        // Aligning exactly onto the end is allowed.
        r.align(6).unwrap();
        assert_eq!(r.remaining(), 0);
        assert!(r.align(usize::MAX).is_err());
    }

    #[test]
    fn the_writer_holds_at_most_its_limit() {
        let mut w = Writer::new();
        w.write_bytes(&vec![0xEE; Writer::LIMIT - 1]);
        assert!(!w.overflowed());
        // One more byte lands exactly on the limit.
        w.write_u8(1);
        assert_eq!(w.len(), Writer::LIMIT);
        assert!(!w.overflowed());
        // Anything further is dropped and flagged.
        w.write_u8(2);
        assert_eq!(w.len(), Writer::LIMIT);
        assert!(w.overflowed());
    }

    #[test]
    fn align_past_the_limit_overflows_instead_of_growing() {
        let mut w = Writer::new();
        w.write_u8(1).align(Writer::LIMIT);
        assert_eq!(w.len(), Writer::LIMIT);
        assert!(!w.overflowed());
        w.align(0).align(1);
        assert_eq!(w.len(), Writer::LIMIT);
        let mut w = Writer::new();
        w.write_u8(1).align(Writer::LIMIT + 1);
        assert_eq!(w.len(), 1);
        assert!(w.overflowed());
        let mut w = Writer::new();
        w.write_u8(1).align(usize::MAX);
        assert!(w.overflowed());
        assert_eq!(w.len(), 1);
    }
}

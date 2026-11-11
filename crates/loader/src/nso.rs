//! NSO executables: header, LZ4-compressed segments, SHA-256 checks.
//!
//! Parsing only: [`Nso::parse`] takes the file bytes and returns the three decompressed segments
//! with their memory offsets, the bss size and the build id. Mapping them into guest memory is the
//! kernel's job.
//!
//! Ported from `eden/src/core/loader/nso.{h,cpp}` @ cf654052 (GPL-3.0-or-later; the header layout
//! and the segment loop). Where Eden asserts on bad input (`ASSERT(r == size)`), this returns an
//! error instead.
//!
//! Facts that are not in the pinned references, taken from the SwitchBrew NSO page and marked
//! where they are used: the segment-hash check flags (flags bits 3 to 5, hash over the
//! decompressed segment) and the meaning of the first two segments' `extra` words (module name
//! offset and size). Eden never checks hashes and calls the `extra` word an alignment.

use lz4_flex::block::DecompressError;
use sha2::{Digest, Sha256};

/// `NSO0` file magic at offset 0 (`eden/src/core/loader/nso.cpp`, `IdentifyType`).
pub const MAGIC: [u8; 4] = *b"NSO0";
/// Size of the NSO header (`eden/src/core/loader/nso.h`, `static_assert(sizeof(NSOHeader) == 0x100)`).
pub const HEADER_SIZE: usize = 0x100;
/// Number of segments: text, rodata, data, in that order (`NSOHeader::segments`).
pub const SEGMENT_COUNT: usize = 3;

/// Offset of the flags word (`NSOHeader::flags`).
const FLAGS_OFFSET: usize = 0xC;
/// Offset of the first segment header; each is 0x10 bytes (`NSOSegmentHeader`).
const SEGMENT_HEADERS_OFFSET: usize = 0x10;
/// Size of one segment header.
const SEGMENT_HEADER_SIZE: usize = 0x10;
/// Offset of the 0x20-byte build id (`NSOHeader::build_id`).
const BUILD_ID_OFFSET: usize = 0x40;
/// Offset of the three compressed sizes (`NSOHeader::segments_compressed_size`).
const COMPRESSED_SIZES_OFFSET: usize = 0x60;
/// Offset of the three SHA-256 hashes (`NSOHeader::segment_hashes`).
const HASHES_OFFSET: usize = 0xA0;
/// Size of a SHA-256 digest.
const HASH_SIZE: usize = 0x20;

/// Flag bit `n` set: segment `n` is LZ4 compressed (`NSOHeader::IsSegmentCompressed`).
const FLAG_COMPRESSED_SHIFT: u32 = 0;
/// Flag bit `3 + n` set: segment `n`'s hash must be checked (SwitchBrew; not in the refs).
const FLAG_HASH_CHECK_SHIFT: u32 = 3;
/// Flag bit 7: the segments use ZBIC rather than LZ4 (`NSOHeader::IsZBICCompressed`, an Eden-only
/// extension that this crate does not implement).
const FLAG_ZBIC: u32 = 1 << 7;

/// LZ4 cannot expand data by more than about 255 times. A declared decompressed size past
/// `compressed_len * 255 + 255` is rejected before allocating, so a tiny file cannot request a
/// multi-gigabyte buffer.
const MAX_LZ4_EXPANSION: u64 = 255;

/// One decompressed NSO segment and where it belongs in the program image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NsoSegment {
    /// Offset from the module base at which the segment is loaded (`NSOSegmentHeader::location`).
    pub memory_offset: u32,
    /// The segment contents after decompression (and hash verification, if requested).
    pub data: Vec<u8>,
}

/// A parsed NSO.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nso {
    /// The `.text` segment.
    pub text: NsoSegment,
    /// The `.rodata` segment.
    pub rodata: NsoSegment,
    /// The `.data` segment.
    pub data: NsoSegment,
    /// Size of the zero-filled `.bss` that follows `.data` (the third segment header's last word).
    pub bss_size: u32,
    /// The module build id (`NSOHeader::build_id`).
    pub build_id: [u8; 0x20],
    /// The raw header flags word (compression and hash-check bits).
    pub flags: u32,
    /// Module name offset: the first segment header's last word. Eden calls this word an
    /// alignment; SwitchBrew calls it the module name offset. A guess, exposed raw and not
    /// resolved.
    pub module_name_offset: u32,
    /// Module name size: the second segment header's last word (same caveat as
    /// [`Nso::module_name_offset`]).
    pub module_name_size: u32,
}

impl Nso {
    /// The segments in header order: text, rodata, data.
    pub fn segments(&self) -> [&NsoSegment; SEGMENT_COUNT] {
        [&self.text, &self.rodata, &self.data]
    }

    /// Parses an NSO, decompressing segments and verifying hashes where the header asks for it.
    ///
    /// Every size and offset in the file is checked against the input before use, so malformed
    /// input is an error and never a panic.
    pub fn parse(bytes: &[u8]) -> Result<Self, NsoError> {
        if bytes.len() < HEADER_SIZE {
            return Err(NsoError::Truncated {
                len: bytes.len(),
                needed: HEADER_SIZE,
            });
        }
        if read_array::<4>(bytes, 0) != Some(MAGIC) {
            return Err(NsoError::BadMagic);
        }
        let flags = u32_at(bytes, FLAGS_OFFSET)?;
        if flags & FLAG_ZBIC != 0 {
            return Err(NsoError::UnsupportedCompression);
        }

        let mut segments = Vec::with_capacity(SEGMENT_COUNT);
        let mut extra = [0u32; SEGMENT_COUNT];
        for (index, extra_word) in extra.iter_mut().enumerate() {
            let header = SEGMENT_HEADERS_OFFSET + index * SEGMENT_HEADER_SIZE;
            let file_offset = u32_at(bytes, header)?;
            let memory_offset = u32_at(bytes, header + 4)?;
            let size = u32_at(bytes, header + 8)?;
            *extra_word = u32_at(bytes, header + 12)?;
            let compressed_size = u32_at(bytes, COMPRESSED_SIZES_OFFSET + index * 4)?;

            let compressed = (flags >> (FLAG_COMPRESSED_SHIFT + shift(index))) & 1 != 0;
            let hash_checked = (flags >> (FLAG_HASH_CHECK_SHIFT + shift(index))) & 1 != 0;

            // The span of the file this segment occupies. Eden reads `compressed_size` bytes for
            // every segment and copies `size` of them; an uncompressed segment is `size` bytes.
            let stored_size = if compressed { compressed_size } else { size };
            let stored = get_range(bytes, u64::from(file_offset), u64::from(stored_size)).ok_or(
                NsoError::SegmentOutOfBounds {
                    segment: index,
                    offset: file_offset,
                    size: stored_size,
                    file_len: bytes.len(),
                },
            )?;
            memory_offset
                .checked_add(size)
                .ok_or(NsoError::MemoryRangeOverflow { segment: index })?;

            let data = if compressed {
                decompress(index, stored, size)?
            } else {
                stored.to_vec()
            };

            if hash_checked {
                let expected = get_range(
                    bytes,
                    (HASHES_OFFSET + index * HASH_SIZE) as u64,
                    HASH_SIZE as u64,
                )
                .ok_or(NsoError::Truncated {
                    len: bytes.len(),
                    needed: HEADER_SIZE,
                })?;
                if Sha256::digest(&data).as_slice() != expected {
                    return Err(NsoError::HashMismatch { segment: index });
                }
            }
            segments.push(NsoSegment {
                memory_offset,
                data,
            });
        }

        let bss_size = extra[2];
        let data_segment = segments
            .get(2)
            .ok_or(NsoError::MemoryRangeOverflow { segment: 2 })?;
        let data_len = u32::try_from(data_segment.data.len())
            .map_err(|_| NsoError::MemoryRangeOverflow { segment: 2 })?;
        data_segment
            .memory_offset
            .checked_add(data_len)
            .and_then(|end| end.checked_add(bss_size))
            .ok_or(NsoError::MemoryRangeOverflow { segment: 2 })?;

        let build_id = read_array::<0x20>(bytes, BUILD_ID_OFFSET).ok_or(NsoError::Truncated {
            len: bytes.len(),
            needed: HEADER_SIZE,
        })?;

        let mut it = segments.into_iter();
        let (Some(text), Some(rodata), Some(data)) = (it.next(), it.next(), it.next()) else {
            return Err(NsoError::MemoryRangeOverflow { segment: 0 });
        };
        Ok(Self {
            text,
            rodata,
            data,
            bss_size,
            build_id,
            flags,
            module_name_offset: extra[0],
            module_name_size: extra[1],
        })
    }
}

/// Why an NSO could not be parsed.
#[derive(Debug, thiserror::Error)]
pub enum NsoError {
    /// The input is shorter than the 0x100-byte header.
    #[error("NSO is {len} bytes, shorter than the {needed}-byte header")]
    Truncated {
        /// Length of the input.
        len: usize,
        /// Bytes needed.
        needed: usize,
    },
    /// The first four bytes are not `NSO0`.
    #[error("NSO magic is not NSO0")]
    BadMagic,
    /// The header selects ZBIC compression, which is not implemented.
    #[error("NSO uses ZBIC compression, which is not supported")]
    UnsupportedCompression,
    /// A segment's stored bytes extend past the end of the file.
    #[error(
        "NSO segment {segment}: {size:#x} bytes at file offset {offset:#x} exceed the {file_len:#x}-byte file"
    )]
    SegmentOutOfBounds {
        /// Segment index (0 text, 1 rodata, 2 data).
        segment: usize,
        /// Declared file offset.
        offset: u32,
        /// Declared stored size.
        size: u32,
        /// Length of the input.
        file_len: usize,
    },
    /// A segment's memory range (or the data segment plus bss) overflows 32 bits.
    #[error("NSO segment {segment}: memory range overflows 32 bits")]
    MemoryRangeOverflow {
        /// Segment index.
        segment: usize,
    },
    /// The declared decompressed size is larger than LZ4 can produce from the stored bytes.
    #[error(
        "NSO segment {segment}: decompressed size {size:#x} is implausible for {compressed:#x} compressed bytes"
    )]
    ImplausibleSize {
        /// Segment index.
        segment: usize,
        /// Declared decompressed size.
        size: u32,
        /// Stored (compressed) size.
        compressed: usize,
    },
    /// The LZ4 stream is corrupt.
    #[error("NSO segment {segment}: LZ4 decompression failed")]
    Lz4 {
        /// Segment index.
        segment: usize,
        /// The decoder's error.
        #[source]
        source: DecompressError,
    },
    /// The LZ4 stream decoded to a different length than the header declares.
    #[error("NSO segment {segment}: decompressed to {actual:#x} bytes, header says {expected:#x}")]
    DecompressedSizeMismatch {
        /// Segment index.
        segment: usize,
        /// Declared size.
        expected: u32,
        /// Actual decoded size.
        actual: usize,
    },
    /// The segment's SHA-256 does not match the header, and its check flag is set.
    #[error("NSO segment {segment}: SHA-256 does not match the header")]
    HashMismatch {
        /// Segment index.
        segment: usize,
    },
}

/// Decompresses one LZ4 block into exactly `size` bytes.
fn decompress(segment: usize, stored: &[u8], size: u32) -> Result<Vec<u8>, NsoError> {
    let limit = (stored.len() as u64)
        .saturating_mul(MAX_LZ4_EXPANSION)
        .saturating_add(MAX_LZ4_EXPANSION);
    if u64::from(size) > limit {
        return Err(NsoError::ImplausibleSize {
            segment,
            size,
            compressed: stored.len(),
        });
    }
    let expected = usize::try_from(size).map_err(|_| NsoError::ImplausibleSize {
        segment,
        size,
        compressed: stored.len(),
    })?;
    let out = lz4_flex::block::decompress(stored, expected)
        .map_err(|source| NsoError::Lz4 { segment, source })?;
    if out.len() != expected {
        return Err(NsoError::DecompressedSizeMismatch {
            segment,
            expected: size,
            actual: out.len(),
        });
    }
    Ok(out)
}

/// `index` as a bit shift (index is always below [`SEGMENT_COUNT`]).
fn shift(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}

/// `len` bytes at `offset`, or `None` if they do not fit in `bytes`.
fn get_range(bytes: &[u8], offset: u64, len: u64) -> Option<&[u8]> {
    let end = offset.checked_add(len)?;
    bytes.get(usize::try_from(offset).ok()?..usize::try_from(end).ok()?)
}

/// `N` bytes at `offset`.
fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Option<[u8; N]> {
    let end = offset.checked_add(N)?;
    <[u8; N]>::try_from(bytes.get(offset..end)?).ok()
}

/// A little-endian `u32` at `offset`.
fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, NsoError> {
    read_array::<4>(bytes, offset)
        .map(u32::from_le_bytes)
        .ok_or(NsoError::Truncated {
            len: bytes.len(),
            needed: offset.saturating_add(4),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Fixed seed so property runs are reproducible.
    const PROPTEST_SEED: u64 = 0x5EED_0000_5143;

    /// How a test segment is stored.
    #[derive(Clone, Copy)]
    struct Spec<'a> {
        data: &'a [u8],
        compress: bool,
        check_hash: bool,
    }

    /// Builds an NSO by writing bytes at the documented offsets, with segment contents laid out
    /// back to back after the header and memory offsets at 0x1000 intervals.
    fn build(specs: [Spec<'_>; 3], bss_size: u32) -> Vec<u8> {
        let mut header = vec![0u8; HEADER_SIZE];
        header[..4].copy_from_slice(&MAGIC);
        let mut flags = 0u32;
        let mut body = Vec::new();
        for (i, spec) in specs.iter().enumerate() {
            let stored = if spec.compress {
                lz4_flex::block::compress(spec.data)
            } else {
                spec.data.to_vec()
            };
            let file_offset = u32::try_from(HEADER_SIZE + body.len()).unwrap();
            let hdr = SEGMENT_HEADERS_OFFSET + i * SEGMENT_HEADER_SIZE;
            header[hdr..hdr + 4].copy_from_slice(&file_offset.to_le_bytes());
            let memory = u32::try_from(i).unwrap() * 0x1000;
            header[hdr + 4..hdr + 8].copy_from_slice(&memory.to_le_bytes());
            let size = u32::try_from(spec.data.len()).unwrap();
            header[hdr + 8..hdr + 12].copy_from_slice(&size.to_le_bytes());
            let stored_len = u32::try_from(stored.len()).unwrap();
            let cs = COMPRESSED_SIZES_OFFSET + i * 4;
            header[cs..cs + 4].copy_from_slice(&stored_len.to_le_bytes());
            let hash = Sha256::digest(spec.data);
            let h = HASHES_OFFSET + i * HASH_SIZE;
            header[h..h + HASH_SIZE].copy_from_slice(&hash);
            let shift_i = u32::try_from(i).unwrap();
            if spec.compress {
                flags |= 1 << (FLAG_COMPRESSED_SHIFT + shift_i);
            }
            if spec.check_hash {
                flags |= 1 << (FLAG_HASH_CHECK_SHIFT + shift_i);
            }
            body.extend_from_slice(&stored);
        }
        header[FLAGS_OFFSET..FLAGS_OFFSET + 4].copy_from_slice(&flags.to_le_bytes());
        // Third segment header's last word is the bss size; the first two carry the module name.
        header[0x1C..0x20].copy_from_slice(&0x1234u32.to_le_bytes());
        header[0x2C..0x30].copy_from_slice(&0x56u32.to_le_bytes());
        header[0x3C..0x40].copy_from_slice(&bss_size.to_le_bytes());
        for (i, byte) in header[BUILD_ID_OFFSET..BUILD_ID_OFFSET + 0x20]
            .iter_mut()
            .enumerate()
        {
            *byte = u8::try_from(i).unwrap();
        }
        header.extend_from_slice(&body);
        header
    }

    fn le(n: usize) -> [u8; 4] {
        u32::try_from(n).unwrap().to_le_bytes()
    }

    fn plain(data: &[u8]) -> Spec<'_> {
        Spec {
            data,
            compress: false,
            check_hash: false,
        }
    }

    const TEXT: &[u8] = &[0xAA; 0x40];
    const RODATA: &[u8] = b"hello rodata hello rodata hello rodata";
    const DATA: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8];

    #[test]
    fn raw_offsets_match_the_documented_layout() {
        let file = build([plain(TEXT), plain(RODATA), plain(DATA)], 0x800);
        assert_eq!(&file[0..4], b"NSO0");
        // Segment 1 (rodata) header lives at 0x20; its memory offset is the second word.
        assert_eq!(&file[0x24..0x28], &0x1000u32.to_le_bytes());
        // Segment 2 (data) size at 0x38, bss size at 0x3C.
        assert_eq!(&file[0x38..0x3C], &le(DATA.len()));
        assert_eq!(&file[0x3C..0x40], &0x800u32.to_le_bytes());
        // Build id at 0x40, rodata compressed size at 0x64.
        assert_eq!(file[0x41], 1);
        assert_eq!(&file[0x64..0x68], &le(RODATA.len()));
    }

    #[test]
    fn parses_uncompressed_segments() {
        let file = build([plain(TEXT), plain(RODATA), plain(DATA)], 0x800);
        let nso = Nso::parse(&file).unwrap();
        assert_eq!(nso.text.data, TEXT);
        assert_eq!(nso.rodata.data, RODATA);
        assert_eq!(nso.data.data, DATA);
        assert_eq!(nso.text.memory_offset, 0);
        assert_eq!(nso.rodata.memory_offset, 0x1000);
        assert_eq!(nso.data.memory_offset, 0x2000);
        assert_eq!(nso.bss_size, 0x800);
        assert_eq!(nso.build_id[0x1F], 0x1F);
        assert_eq!(nso.module_name_offset, 0x1234);
        assert_eq!(nso.module_name_size, 0x56);
        assert_eq!(nso.segments()[1], &nso.rodata);
    }

    #[test]
    fn parses_compressed_segments() {
        let compressed = |data| Spec {
            data,
            compress: true,
            check_hash: false,
        };
        let file = build([compressed(TEXT), plain(RODATA), compressed(DATA)], 0);
        let nso = Nso::parse(&file).unwrap();
        assert_eq!(nso.text.data, TEXT);
        assert_eq!(nso.rodata.data, RODATA);
        assert_eq!(nso.data.data, DATA);
        assert_eq!(nso.flags & 0b111, 0b101);
    }

    #[test]
    fn verifies_hashes_when_the_check_flag_is_set() {
        let checked = |data| Spec {
            data,
            compress: true,
            check_hash: true,
        };
        let file = build([checked(TEXT), checked(RODATA), checked(DATA)], 0);
        assert!(Nso::parse(&file).is_ok());
    }

    #[test]
    fn hash_mismatch_is_an_error_only_when_checked() {
        let checked = Spec {
            data: TEXT,
            compress: false,
            check_hash: true,
        };
        let mut file = build([checked, plain(RODATA), plain(DATA)], 0);
        // Corrupt the stored hash of segment 0.
        file[HASHES_OFFSET] ^= 0xFF;
        assert!(matches!(
            Nso::parse(&file),
            Err(NsoError::HashMismatch { segment: 0 })
        ));

        // The same corruption in a segment whose check flag is clear is ignored.
        let mut file = build([plain(TEXT), plain(RODATA), plain(DATA)], 0);
        file[HASHES_OFFSET] ^= 0xFF;
        assert!(Nso::parse(&file).is_ok());
    }

    #[test]
    fn corrupt_segment_contents_fail_the_hash_check() {
        let checked = Spec {
            data: TEXT,
            compress: false,
            check_hash: true,
        };
        let mut file = build([checked, plain(RODATA), plain(DATA)], 0);
        file[HEADER_SIZE] ^= 1;
        assert!(matches!(
            Nso::parse(&file),
            Err(NsoError::HashMismatch { segment: 0 })
        ));
    }

    #[test]
    fn declared_size_past_end_of_file_is_an_error() {
        // Uncompressed: the declared size runs past the end.
        let mut file = build([plain(TEXT), plain(RODATA), plain(DATA)], 0);
        file[0x18..0x1C].copy_from_slice(&0x10_0000u32.to_le_bytes());
        assert!(matches!(
            Nso::parse(&file),
            Err(NsoError::SegmentOutOfBounds { segment: 0, .. })
        ));

        // Compressed: the compressed size runs past the end.
        let compressed = Spec {
            data: DATA,
            compress: true,
            check_hash: false,
        };
        let mut file = build([plain(TEXT), plain(RODATA), compressed], 0);
        file[COMPRESSED_SIZES_OFFSET + 8..COMPRESSED_SIZES_OFFSET + 12]
            .copy_from_slice(&0x10_0000u32.to_le_bytes());
        assert!(matches!(
            Nso::parse(&file),
            Err(NsoError::SegmentOutOfBounds { segment: 2, .. })
        ));

        // A file offset past the end.
        let mut file = build([plain(TEXT), plain(RODATA), plain(DATA)], 0);
        file[0x10..0x14].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            Nso::parse(&file),
            Err(NsoError::SegmentOutOfBounds { segment: 0, .. })
        ));
    }

    #[test]
    fn rejects_bad_magic_and_short_input() {
        let mut file = build([plain(TEXT), plain(RODATA), plain(DATA)], 0);
        assert!(matches!(
            Nso::parse(&file[..HEADER_SIZE - 1]),
            Err(NsoError::Truncated { .. })
        ));
        file[0] = b'X';
        assert!(matches!(Nso::parse(&file), Err(NsoError::BadMagic)));
        assert!(matches!(Nso::parse(&[]), Err(NsoError::Truncated { .. })));
    }

    #[test]
    fn rejects_zbic_and_wrong_decompressed_sizes() {
        let mut file = build([plain(TEXT), plain(RODATA), plain(DATA)], 0);
        file[FLAGS_OFFSET] |= u8::try_from(FLAG_ZBIC).unwrap();
        assert!(matches!(
            Nso::parse(&file),
            Err(NsoError::UnsupportedCompression)
        ));

        let compressed = Spec {
            data: TEXT,
            compress: true,
            check_hash: false,
        };
        // Header claims a larger decompressed size than the stream holds.
        let mut file = build([compressed, plain(RODATA), plain(DATA)], 0);
        file[0x18..0x1C].copy_from_slice(&le(TEXT.len() + 1));
        assert!(Nso::parse(&file).is_err());
        // Header claims an absurd size for a tiny stream: refused before allocating.
        file[0x18..0x1C].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            Nso::parse(&file),
            Err(NsoError::ImplausibleSize { segment: 0, .. })
        ));
    }

    #[test]
    fn rejects_memory_ranges_that_overflow() {
        let mut file = build([plain(TEXT), plain(RODATA), plain(DATA)], 0);
        file[0x14..0x18].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            Nso::parse(&file),
            Err(NsoError::MemoryRangeOverflow { segment: 0 })
        ));
        // Data segment plus bss overflows.
        let file = build([plain(TEXT), plain(RODATA), plain(DATA)], u32::MAX);
        assert!(matches!(
            Nso::parse(&file),
            Err(NsoError::MemoryRangeOverflow { segment: 2 })
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            rng_seed: proptest::test_runner::RngSeed::Fixed(PROPTEST_SEED),
            ..ProptestConfig::default()
        })]
        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..2048)) {
            let _ = Nso::parse(&bytes);
        }

        #[test]
        fn arbitrary_bytes_behind_a_valid_magic_never_panic(
            tail in proptest::collection::vec(any::<u8>(), 0..2048),
        ) {
            let mut bytes = MAGIC.to_vec();
            bytes.extend_from_slice(&tail);
            let _ = Nso::parse(&bytes);
        }

        #[test]
        fn mutated_valid_files_never_panic(
            index in 0usize..400,
            value in any::<u8>(),
            compress in any::<bool>(),
        ) {
            let spec = |data| Spec { data, compress, check_hash: compress };
            let mut file = build([spec(TEXT), spec(RODATA), spec(DATA)], 0x10);
            let at = index % file.len();
            file[at] = value;
            let _ = Nso::parse(&file);
        }
    }
}

//! A module's code set: what process creation puts into a module's pages, whichever format the
//! module came in. [`CodeSet::from_nro`] and [`CodeSet::from_nso`] turn a parsed executable into
//! the bytes to copy (relative to the module's base), its three segments, where its `.bss` lies,
//! the module's size in whole pages, and the ranges guest code gets permissions on
//! ([`CodeSet::protections`]). Placing a set of modules in an address space, and rewriting their
//! counter reads, is the caller's (`starquake-core`'s loading).
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - NRO: `LoadNroImpl` in `eden/src/core/loader/nro.cpp`. The image is the file's first
//!   `size` bytes; `.bss` is sized from `MOD0` when the module has one, else from the header
//!   (`bss_size = mod_header.bss_end_offset - mod_header.bss_start_offset` only when
//!   `mod_header.magic == MOD0`); the module is the image rounded up to a page plus `.bss`
//!   rounded up to a page (`PageAlignSize(program_image.size()) + PageAlignSize(bss_size)`).
//! - NSO: `AppLoader_NSO::LoadModule` in `eden/src/core/loader/nso.cpp`. Each segment is copied
//!   at its memory offset; `.bss` follows the data segment (`codeset.DataSegment().size +=
//!   bss_size`); the module is the end of the data segment plus `.bss`, rounded up to a page
//!   (`PageAlignSize(codeset.memory.size() + bss_size)`, with `memory` sized to the last
//!   segment's end).
//!
//! The NSO checks Eden does not make are Atmosphère's loader's, read as facts only
//! (`atmosphere/stratosphere/loader/source/ldr_process_creation.cpp` @ 4e3f3703,
//! `CheckAutoLoad`): every segment's memory offset is page-aligned
//! (`util::IsAligned(hdr.text_dst_offset, os::MemoryPageSize)`, and the same for rodata and
//! data), text starts the module (`hdr.text_dst_offset == 0`), text ends at or before rodata
//! starts and rodata at or before data starts (`text_end <= ro_dst_offset`,
//! `ro_end <= rw_dst_offset`). Its `DecideAddressSpaceLayout` sizes the module the same way as
//! Eden once those hold (the largest segment end plus `.bss`, aligned up), and its
//! `LoadAutoLoadModule` gives data and `.bss` one read-write range
//! (`util::AlignUp(nso_header->rw_size + nso_header->bss_size, os::MemoryPageSize)`).
//!
//! # Relocations
//!
//! Nothing here applies relocations, and neither reference does: Eden's two loaders above copy
//! the segments as they are, and Atmosphère's `LoadAutoLoadModule` only copies and
//! reprotects. A module relocates itself when it starts (libnx's `crt0` for homebrew; `rtld`
//! for a title's modules), writing only its own writable pages, so the counter reads the caller
//! rewrites in `.text` are not touched by it.
//!
//! # Divergences
//!
//! - **NSO: Atmosphère's checks**, which Eden does not make: each is an error here
//!   ([`CodeSetError::TextNotAtStart`], [`CodeSetError::MisalignedSegment`],
//!   [`CodeSetError::SegmentsOutOfOrder`]). Tests: `an_nso_whose_text_does_not_start_the_module_is_refused`,
//!   `an_nso_segment_off_a_page_boundary_is_refused`, `nso_segments_out_of_order_are_refused`.
//! - **NRO: a `MOD0` whose `.bss` ends before it starts** is [`CodeSetError::BadBss`]; Eden
//!   subtracts the two unchecked. Test: `a_mod0_whose_bss_ends_before_it_starts_is_refused`.
//! - **NRO: `.bss` is its own range**, the pages after the image ([`CodeSet::protections`]).
//!   `LoadNroImpl` adds `bss_size` to the data segment instead, so Eden's read-write range is
//!   the data segment plus `bss_size` bytes after it; the two are the same pages only when the
//!   data segment ends at the page-rounded end of the image. Test:
//!   `an_nro_gives_bss_its_own_range_after_the_image`.

use starquake_memory::{PAGE_SIZE, Perms};

use crate::nro::{Nro, NroError};
use crate::nso::Nso;

/// Size of a module's build id (`NSOHeader::build_id`, `NroHeader::m_module_id`).
pub const BUILD_ID_SIZE: usize = 0x20;

/// A byte range of a module, from its base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// Offset from the module's base.
    pub offset: u64,
    /// Size in bytes.
    pub size: u64,
}

impl Segment {
    /// One past the last byte, or `None` if that does not fit 64 bits.
    #[must_use]
    pub fn end(&self) -> Option<u64> {
        self.offset.checked_add(self.size)
    }
}

/// Bytes to copy into a module's pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    /// Offset from the module's base of the first byte.
    pub offset: u64,
    /// The bytes.
    pub bytes: Vec<u8>,
}

/// The executable format a code set came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// An NRO: one image holding its header and segments.
    Nro,
    /// An NSO: three separately stored segments.
    Nso,
}

/// A module ready to be placed (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeSet {
    /// The format it came from.
    pub format: Format,
    /// What to copy into the module, in offset order and not overlapping: an NRO's image as one
    /// piece, an NSO's segments one piece each. Every byte of the module outside them reads as
    /// zero.
    pub pieces: Vec<Piece>,
    /// The `.text` segment, as the file gives it.
    pub text: Segment,
    /// The `.rodata` segment, as the file gives it.
    pub rodata: Segment,
    /// The `.data` segment, as the file gives it.
    pub data: Segment,
    /// Where `.bss` lies (see the module docs for each format).
    pub bss: Segment,
    /// Bytes of the module from its base: a multiple of [`PAGE_SIZE`].
    pub size: u64,
    /// The build id.
    pub build_id: [u8; BUILD_ID_SIZE],
}

/// Why a parsed executable cannot be loaded as a module.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CodeSetError {
    /// The NRO's `MOD0` header describes a `.bss` whose end precedes its start.
    #[error("the NRO's MOD0 header describes a .bss that ends before it starts")]
    BadBss,
    /// The NSO's text segment does not start at the module's base.
    #[error("the NSO's text segment starts at {offset:#x}, not at the start of the module")]
    TextNotAtStart {
        /// The text segment's memory offset.
        offset: u32,
    },
    /// An NSO segment does not start on a page boundary.
    #[error("the NSO's segment at {offset:#x} does not start on a page boundary")]
    MisalignedSegment {
        /// The segment's memory offset.
        offset: u32,
    },
    /// An NSO segment starts before the one ahead of it (text, rodata, data) ends.
    #[error("the NSO's segment at {offset:#x} starts before the segment ahead of it ends")]
    SegmentsOutOfOrder {
        /// The later segment's memory offset.
        offset: u32,
    },
    /// A size, or an offset derived from one, does not fit 64 bits.
    #[error("the module's sizes overflow the address space")]
    Overflow,
}

/// `value` rounded up to a page.
fn page_up(value: u64) -> Result<u64, CodeSetError> {
    value
        .checked_next_multiple_of(PAGE_SIZE)
        .ok_or(CodeSetError::Overflow)
}

fn add(a: u64, b: u64) -> Result<u64, CodeSetError> {
    a.checked_add(b).ok_or(CodeSetError::Overflow)
}

fn len(bytes: &[u8]) -> Result<u64, CodeSetError> {
    u64::try_from(bytes.len()).map_err(|_| CodeSetError::Overflow)
}

impl CodeSet {
    /// The module of the NRO `nro`, parsed from `file` (see the module docs).
    ///
    /// # Errors
    ///
    /// [`CodeSetError::BadBss`] for a `MOD0` whose `.bss` ends before it starts, and
    /// [`CodeSetError::Overflow`] when the image does not fit `file` (which [`Nro::parse`] has
    /// already ruled out) or the sizes overflow.
    pub fn from_nro(nro: &Nro, file: &[u8]) -> Result<Self, CodeSetError> {
        let image = usize::try_from(nro.size)
            .ok()
            .and_then(|size| file.get(..size))
            .ok_or(CodeSetError::Overflow)?
            .to_vec();
        let bss_size = match &nro.mod0 {
            Some(mod0) => mod0.bss_size().ok_or(CodeSetError::BadBss)?,
            None => nro.bss_size,
        };
        let image_end = page_up(u64::from(nro.size))?;
        let size = add(image_end, page_up(u64::from(bss_size))?)?;
        let segment = |s: crate::nro::NroSegment| Segment {
            offset: u64::from(s.offset),
            size: u64::from(s.size),
        };
        Ok(Self {
            format: Format::Nro,
            pieces: vec![Piece {
                offset: 0,
                bytes: image,
            }],
            text: segment(nro.text),
            rodata: segment(nro.rodata),
            data: segment(nro.data),
            bss: Segment {
                offset: image_end,
                size: u64::from(bss_size),
            },
            size,
            build_id: nro.build_id,
        })
    }

    /// Parses `file` as an NRO and makes its module.
    ///
    /// # Errors
    ///
    /// The parser's error, or [`CodeSet::from_nro`]'s.
    pub fn parse_nro(file: &[u8]) -> Result<Self, NroOrCodeSetError> {
        let nro = Nro::parse(file)?;
        Ok(Self::from_nro(&nro, file)?)
    }

    /// The module of the NSO `nso`, taking its segments' bytes (see the module docs).
    ///
    /// # Errors
    ///
    /// Atmosphère's checks, in its order: text not at offset zero
    /// ([`CodeSetError::TextNotAtStart`]) is checked after each segment's alignment
    /// ([`CodeSetError::MisalignedSegment`], text then rodata then data), then the order of the
    /// segments ([`CodeSetError::SegmentsOutOfOrder`], rodata then data).
    pub fn from_nso(nso: Nso) -> Result<Self, CodeSetError> {
        let Nso {
            text,
            rodata,
            data,
            bss_size,
            build_id,
            ..
        } = nso;
        for segment in [&text, &rodata, &data] {
            if !u64::from(segment.memory_offset).is_multiple_of(PAGE_SIZE) {
                return Err(CodeSetError::MisalignedSegment {
                    offset: segment.memory_offset,
                });
            }
        }
        if text.memory_offset != 0 {
            return Err(CodeSetError::TextNotAtStart {
                offset: text.memory_offset,
            });
        }
        let segment = |s: &crate::nso::NsoSegment| -> Result<Segment, CodeSetError> {
            Ok(Segment {
                offset: u64::from(s.memory_offset),
                size: len(&s.data)?,
            })
        };
        let (text_at, rodata_at, data_at) = (segment(&text)?, segment(&rodata)?, segment(&data)?);
        for (ahead, next) in [(text_at, rodata_at), (rodata_at, data_at)] {
            if ahead.end().ok_or(CodeSetError::Overflow)? > next.offset {
                return Err(CodeSetError::SegmentsOutOfOrder {
                    offset: u32::try_from(next.offset).map_err(|_| CodeSetError::Overflow)?,
                });
            }
        }
        // With the segments in order, data ends last: Eden's `last_segment_it` and Atmosphère's
        // largest end are the same.
        let data_end = data_at.end().ok_or(CodeSetError::Overflow)?;
        let size = page_up(add(data_end, u64::from(bss_size))?)?;
        let pieces = [(text_at, text), (rodata_at, rodata), (data_at, data)]
            .into_iter()
            .filter(|(_, s)| !s.data.is_empty())
            .map(|(at, s)| Piece {
                offset: at.offset,
                bytes: s.data,
            })
            .collect();
        Ok(Self {
            format: Format::Nso,
            pieces,
            text: text_at,
            rodata: rodata_at,
            data: data_at,
            bss: Segment {
                offset: data_end,
                size: u64::from(bss_size),
            },
            size,
            build_id,
        })
    }

    /// The segments in order: text, rodata, data.
    #[must_use]
    pub fn segments(&self) -> [Segment; 3] {
        [self.text, self.rodata, self.data]
    }

    /// The ranges guest code gets permissions on, before rounding to pages, with those
    /// permissions: text read-execute, rodata read, data read-write, and `.bss` read-write. An
    /// NSO's `.bss` is part of its data range (Eden's and Atmosphère's one read-write range); an
    /// NRO's is its own range after the image (a divergence, see the module docs). Ranges of no
    /// bytes are left out.
    #[must_use]
    pub fn protections(&self) -> Vec<(Segment, Perms)> {
        let data = match self.format {
            Format::Nro => self.data,
            Format::Nso => Segment {
                offset: self.data.offset,
                size: self.data.size.saturating_add(self.bss.size),
            },
        };
        let mut ranges = vec![
            (self.text, Perms::ReadExecute),
            (self.rodata, Perms::Read),
            (data, Perms::ReadWrite),
        ];
        if self.format == Format::Nro {
            ranges.push((self.bss, Perms::ReadWrite));
        }
        ranges.retain(|(range, _)| range.size != 0);
        ranges
    }

    /// The bytes of `.text`, to rewrite in place: the part of the piece that holds it. `None`
    /// when no piece holds all of it.
    pub fn text_mut(&mut self) -> Option<&mut [u8]> {
        let text = self.text;
        let end = text.end()?;
        self.pieces.iter_mut().find_map(|piece| {
            let start = usize::try_from(text.offset.checked_sub(piece.offset)?).ok()?;
            let stop = usize::try_from(end.checked_sub(piece.offset)?).ok()?;
            piece.bytes.get_mut(start..stop)
        })
    }
}

/// Why an NRO file could not be made into a module.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NroOrCodeSetError {
    /// The file is not a well-formed NRO.
    #[error(transparent)]
    Nro(#[from] NroError),
    /// The NRO cannot be loaded as a module.
    #[error(transparent)]
    CodeSet(#[from] CodeSetError),
}

#[cfg(test)]
mod tests {
    use starquake_testkit::a64;
    use starquake_testkit::nro::{MOD0_FIELD_SIZE, MOD0_OFFSET, NroBuilder};

    use super::*;
    use crate::nso::{NsoSegment, RodataExtent};

    const PAGE: u64 = PAGE_SIZE;

    fn page_u32() -> u32 {
        u32::try_from(PAGE).unwrap()
    }

    /// An NSO with text at 0 (two pages and a word), rodata at three pages (0x10 bytes), data at
    /// four pages (0x24 bytes) and 0x1234 bytes of `.bss`: every value distinct.
    fn nso() -> Nso {
        let segment = |at: u64, len: usize, fill: u8| NsoSegment {
            memory_offset: u32::try_from(at).unwrap(),
            data: vec![fill; len],
        };
        Nso {
            text: segment(0, 2 * 0x1000 + 4, 0xA1),
            rodata: segment(3 * PAGE, 0x10, 0xB2),
            data: segment(4 * PAGE, 0x24, 0xC3),
            bss_size: 0x1234,
            build_id: [0x5A; BUILD_ID_SIZE],
            flags: 0,
            module_name_offset: 0,
            module_name_size: 0,
            api_info: RodataExtent { offset: 0, size: 0 },
            dynstr: RodataExtent { offset: 0, size: 0 },
            dynsym: RodataExtent { offset: 0, size: 0 },
        }
    }

    #[test]
    fn an_nso_is_its_segments_at_their_offsets_with_bss_after_data() {
        let code = CodeSet::from_nso(nso()).unwrap();
        assert_eq!(code.format, Format::Nso);
        let at = |offset: u64, size: u64| Segment { offset, size };
        assert_eq!(code.text, at(0, 2 * PAGE + 4));
        assert_eq!(code.rodata, at(3 * PAGE, 0x10));
        assert_eq!(code.data, at(4 * PAGE, 0x24));
        // `.bss` starts where data ends, unaligned (Eden: `DataSegment().size += bss_size`).
        assert_eq!(code.bss, at(4 * PAGE + 0x24, 0x1234));
        // The module is data's end plus `.bss`, rounded up: 0x4024 + 0x1234 = 0x5258 -> 0x6000.
        assert_eq!(code.size, 6 * PAGE);
        assert_eq!(code.build_id, [0x5A; BUILD_ID_SIZE]);
        let pieces: Vec<_> = code
            .pieces
            .iter()
            .map(|p| (p.offset, p.bytes.len(), p.bytes[0]))
            .collect();
        assert_eq!(
            pieces,
            [
                (0, 2 * 0x1000 + 4, 0xA1),
                (3 * PAGE, 0x10, 0xB2),
                (4 * PAGE, 0x24, 0xC3),
            ]
        );
        // Data and `.bss` are one read-write range.
        assert_eq!(
            code.protections(),
            [
                (at(0, 2 * PAGE + 4), Perms::ReadExecute),
                (at(3 * PAGE, 0x10), Perms::Read),
                (at(4 * PAGE, 0x24 + 0x1234), Perms::ReadWrite),
            ]
        );
    }

    #[test]
    fn an_nso_module_ends_on_the_page_boundary_its_bss_reaches() {
        // Data plus `.bss` ending exactly on a page boundary takes no extra page; one byte more
        // takes one.
        let mut exact = nso();
        exact.bss_size = u32::try_from(PAGE - 0x24).unwrap();
        assert_eq!(CodeSet::from_nso(exact.clone()).unwrap().size, 5 * PAGE);
        exact.bss_size += 1;
        assert_eq!(CodeSet::from_nso(exact).unwrap().size, 6 * PAGE);
        // No `.bss`: the data's own page ends the module.
        let mut none = nso();
        none.bss_size = 0;
        let code = CodeSet::from_nso(none).unwrap();
        assert_eq!(code.size, 5 * PAGE);
        assert_eq!(code.protections()[2].0.size, 0x24);
    }

    #[test]
    fn empty_nso_segments_are_neither_copied_nor_protected() {
        let mut empty = nso();
        empty.rodata.data.clear();
        let code = CodeSet::from_nso(empty).unwrap();
        assert_eq!(code.pieces.len(), 2);
        assert_eq!(code.rodata.size, 0);
        assert!(
            code.protections()
                .iter()
                .all(|(_, perms)| *perms != Perms::Read)
        );
    }

    #[test]
    fn an_nso_whose_text_does_not_start_the_module_is_refused() {
        let mut moved = nso();
        moved.text.memory_offset = page_u32();
        moved.rodata.memory_offset = 4 * page_u32();
        moved.data.memory_offset = 5 * page_u32();
        assert_eq!(
            CodeSet::from_nso(moved),
            Err(CodeSetError::TextNotAtStart { offset: page_u32() })
        );
        assert_eq!(
            CodeSetError::TextNotAtStart { offset: 0x1000 }.to_string(),
            "the NSO's text segment starts at 0x1000, not at the start of the module"
        );
    }

    #[test]
    fn an_nso_segment_off_a_page_boundary_is_refused() {
        for (index, offset) in [(0, 4), (1, 3 * page_u32() + 1), (2, 5 * page_u32() - 1)] {
            let mut file = nso();
            [&mut file.text, &mut file.rodata, &mut file.data][index].memory_offset = offset;
            assert_eq!(
                CodeSet::from_nso(file),
                Err(CodeSetError::MisalignedSegment { offset }),
                "segment {index}"
            );
        }
        // Alignment is checked before text's offset: a text at 4 is misaligned, not moved.
        assert_eq!(
            CodeSetError::MisalignedSegment { offset: 0x1001 }.to_string(),
            "the NSO's segment at 0x1001 does not start on a page boundary"
        );
    }

    #[test]
    fn nso_segments_out_of_order_are_refused() {
        // Text one word into rodata's page start; rodata into data's.
        let mut text = nso();
        text.rodata.memory_offset = 2 * page_u32();
        assert_eq!(
            CodeSet::from_nso(text),
            Err(CodeSetError::SegmentsOutOfOrder {
                offset: 2 * page_u32()
            })
        );
        let mut rodata = nso();
        rodata.rodata.data = vec![0; usize::try_from(PAGE).unwrap() + 1];
        assert_eq!(
            CodeSet::from_nso(rodata),
            Err(CodeSetError::SegmentsOutOfOrder {
                offset: 4 * page_u32()
            })
        );
        // Ending exactly where the next starts is in order.
        let mut touching = nso();
        touching.rodata.data = vec![0; usize::try_from(PAGE).unwrap()];
        assert!(CodeSet::from_nso(touching).is_ok());
        assert_eq!(
            CodeSetError::SegmentsOutOfOrder { offset: 0x2000 }.to_string(),
            "the NSO's segment at 0x2000 starts before the segment ahead of it ends"
        );
    }

    #[test]
    fn an_nro_is_one_image_with_bss_on_the_pages_after_it() {
        let file = NroBuilder::new(&[a64::nop()])
            .rodata(b"rodata")
            .data(b"data")
            .bss_size(0x1234)
            .build()
            .unwrap();
        let nro = Nro::parse(&file).unwrap();
        let code = CodeSet::from_nro(&nro, &file).unwrap();
        assert_eq!(code.format, Format::Nro);
        assert_eq!(
            code.pieces,
            [Piece {
                offset: 0,
                bytes: file.clone()
            }]
        );
        let image_end = (file.len() as u64).next_multiple_of(PAGE);
        assert_eq!(
            code.bss,
            Segment {
                offset: image_end,
                size: 0x1234
            }
        );
        // The image rounded up, plus `.bss` rounded up (two pages).
        assert_eq!(code.size, image_end + 2 * PAGE);
        assert_eq!(code.text.offset, u64::from(nro.text.offset));
        assert_eq!(code.rodata.size, u64::from(nro.rodata.size));
        assert_eq!(code.data.size, u64::from(nro.data.size));
        assert_eq!(code.build_id, nro.build_id);
        assert_eq!(CodeSet::parse_nro(&file), Ok(code));
    }

    #[test]
    fn an_nro_gives_bss_its_own_range_after_the_image() {
        // Divergence from Eden (see the module docs): `.bss` is not added to data's range.
        let file = NroBuilder::new(&[a64::nop()])
            .data(b"data")
            .bss_size(1)
            .build()
            .unwrap();
        let code = CodeSet::parse_nro(&file).unwrap();
        let protections = code.protections();
        assert_eq!(protections.last(), Some(&(code.bss, Perms::ReadWrite)));
        assert_eq!(
            protections[protections.len() - 2],
            (code.data, Perms::ReadWrite)
        );
        assert_eq!(code.bss.offset % PAGE, 0);
    }

    #[test]
    fn an_nro_prefers_mod0s_bss_size_to_the_headers() {
        let mut file = NroBuilder::new(&[a64::nop()])
            .bss_size(3 * page_u32())
            .build()
            .unwrap();
        // The builder writes both; make the header say something else.
        let header_at = starquake_testkit::nro::BSS_SIZE_OFFSET;
        file[header_at..header_at + 4].copy_from_slice(&7_u32.to_le_bytes());
        let code = CodeSet::parse_nro(&file).unwrap();
        assert_eq!(code.bss.size, 3 * PAGE);
        // Without MOD0 (its magic broken) the header's value is used.
        file[MOD0_OFFSET] ^= 0xFF;
        let nro = Nro::parse(&file).unwrap();
        assert!(nro.mod0.is_none());
        assert_eq!(CodeSet::from_nro(&nro, &file).unwrap().bss.size, 7);
    }

    #[test]
    fn a_mod0_whose_bss_ends_before_it_starts_is_refused() {
        let mut file = NroBuilder::new(&[a64::nop()]).build().unwrap();
        // bss_start_offset is MOD0's second field after the magic.
        let start_at = MOD0_OFFSET + 2 * MOD0_FIELD_SIZE;
        let end_at = start_at + MOD0_FIELD_SIZE;
        let end = u32::from_le_bytes(file[end_at..end_at + 4].try_into().unwrap());
        file[start_at..start_at + 4].copy_from_slice(&end.to_le_bytes());
        assert!(CodeSet::parse_nro(&file).is_ok(), "an empty .bss is fine");
        file[start_at..start_at + 4].copy_from_slice(&(end + 1).to_le_bytes());
        assert_eq!(
            CodeSet::parse_nro(&file),
            Err(NroOrCodeSetError::CodeSet(CodeSetError::BadBss))
        );
    }

    #[test]
    fn an_nro_image_the_file_does_not_hold_is_an_overflow() {
        let file = NroBuilder::new(&[a64::nop()]).build().unwrap();
        let nro = Nro::parse(&file).unwrap();
        assert_eq!(
            CodeSet::from_nro(&nro, &file[..file.len() - 1]),
            Err(CodeSetError::Overflow)
        );
        assert!(CodeSet::from_nro(&nro, &file).is_ok());
    }

    #[test]
    fn text_mut_is_the_text_inside_its_piece() {
        let mut code = CodeSet::from_nso(nso()).unwrap();
        let text = code.text_mut().unwrap();
        assert_eq!(text.len(), 2 * 0x1000 + 4);
        text[0] = 0x11;
        assert_eq!(code.pieces[0].bytes[0], 0x11);
        // An NRO's text is part of its one image, at the text's offset.
        let file = NroBuilder::new(&[a64::nop()]).build().unwrap();
        let mut nro = CodeSet::parse_nro(&file).unwrap();
        let at = usize::try_from(nro.text.offset).unwrap();
        let size = usize::try_from(nro.text.size).unwrap();
        assert_eq!(nro.text_mut().unwrap(), &file[at..at + size]);
        // A text no piece holds whole is `None`.
        nro.text.size = u64::MAX;
        assert_eq!(nro.text_mut(), None);
    }
}

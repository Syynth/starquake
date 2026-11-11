//! A builder for synthetic NRO files, so the loader, kernel and CPU tests share one way of
//! producing a runnable guest.
//!
//! The layout follows what the loader parses (`crates/loader/src/nro.rs`), which is ported from
//! `eden/src/core/loader/nro.cpp` (`NroHeader`, `ModHeader`) and
//! `eden/src/core/hle/service/ro/ro_types.h` (`NroHeader`, which names the first two header
//! words `m_entrypoint_insn` and `m_mod_offset` and the 0x20 bytes at 0x40 `m_module_id`). This
//! crate may not depend on the loader, so the layout is restated here and pinned by tests that
//! derive every offset from the sizes of the fields in the reference structs.
//!
//! ```text
//! 0x000  header (0x80 bytes): entry branch, MOD0 pointer, HOMEBREW, NRO0, sizes, build id
//! 0x080  MOD0 (0x1C bytes)
//! 0x0A0  dynamic section: one DT_NULL entry (0x10 bytes)
//! 0x0B0  code
//! ...    text padded to a page; then rodata, then data, each padded to a page
//! ```
//!
//! Segment sizes are page multiples, as Eden rounds them up (`PageAlignSize`) with `YUZU_PAGESIZE` = 0x1000.
//! There is no `ASET` block. `.bss` is described by the header's size and by `MOD0`, and starts
//! right after the data segment.
//!
//! Guesses, not backed by a pinned source (`docs/references.md`):
//!
//! - The text segment starts at offset 0 and so contains the header. `eden/src/core/loader/nro.cpp`
//!   maps each segment's `offset` from the image base, which is consistent with this but does not
//!   require it; no reference states it.
//! - The first word is a `b` to the code. The references only call it the entry instruction
//!   (`m_entrypoint_insn`); that it is a branch over the header is a guess, and the loader tests
//!   should not depend on it.
//! - `MOD0` offsets are relative to the `MOD0` header, as the SwitchBrew wiki documents; Eden
//!   reads them raw and only ever subtracts `bss_start_offset` from `bss_end_offset`, which is the
//!   same either way. A dynamic section holding one `DT_NULL` (ELF `d_tag` 0 ends the table) and an
//!   empty unwind range keep `MOD0` self-consistent; nothing here reads them.

use crate::a64::{self, EncodeError};

/// `NRO0` magic (`NroHeader::magic`).
pub const MAGIC: [u8; 4] = *b"NRO0";
/// `MOD0` magic (`ModHeader::magic`).
pub const MOD0_MAGIC: [u8; 4] = *b"MOD0";
/// The homebrew marker at [`HOMEBREW_OFFSET`]: `HOME` then `BREW` (`AppLoader_NRO::IsHomebrew`).
pub const HOMEBREW_MARKER: [u8; 8] = *b"HOMEBREW";

/// Size of the NRO header (`static_assert(sizeof(NroHeader) == 0x80)`).
pub const HEADER_SIZE: usize = 0x80;
/// Size of the `MOD0` header (`static_assert(sizeof(ModHeader) == 0x1c)`).
pub const MOD0_SIZE: usize = 0x1C;
/// Size of one segment header, `{offset, size}` (`NroSegmentHeader`, `0x8` bytes).
pub const SEGMENT_HEADER_SIZE: usize = 0x8;
/// Offset of the size within a segment header.
pub const SEGMENT_SIZE_OFFSET: usize = 0x4;
/// Size of one `MOD0` field (a `u32`; the magic is one as well).
pub const MOD0_FIELD_SIZE: usize = 0x4;
/// Size of the build id, `NroHeader::m_module_id` in `ro_types.h`.
pub const BUILD_ID_SIZE: usize = 0x20;
/// Size of the one dynamic entry, an ELF `Elf64_Dyn` of two `u64`s.
pub const DYNAMIC_SIZE: usize = 0x10;
/// The page size segments are padded to (`YUZU_PAGESIZE`, `eden/src/core/memory.h`).
pub const PAGE_SIZE: usize = 0x1000;

/// Offset of the entry instruction (`NroHeader::m_entrypoint_insn`).
pub const ENTRY_OFFSET: usize = 0x0;
/// Offset of the pointer to `MOD0` (`NroHeader::module_header_offset`).
pub const MOD0_POINTER_OFFSET: usize = 0x4;
/// Offset of the homebrew marker (`NroHeader::magic_ext1` and `magic_ext2`).
pub const HOMEBREW_OFFSET: usize = 0x8;
/// Offset of the magic (`NroHeader::magic`).
pub const MAGIC_OFFSET: usize = 0x10;
/// Offset of the image size (`NroHeader::file_size`).
pub const SIZE_OFFSET: usize = 0x18;
/// Offset of the three `{offset, size}` segment headers: text, rodata, data
/// (`NroHeader::segments`).
pub const SEGMENTS_OFFSET: usize = 0x20;
/// Offset of the bss size (`NroHeader::bss_size`).
pub const BSS_SIZE_OFFSET: usize = 0x38;
/// Offset of the build id (`NroHeader::m_module_id` in `ro_types.h`).
pub const BUILD_ID_OFFSET: usize = 0x40;

/// Offset of the `MOD0` header: right after the NRO header.
pub const MOD0_OFFSET: usize = HEADER_SIZE;
/// Offset of the dynamic section, 8-aligned after `MOD0`.
pub const DYNAMIC_OFFSET: usize = (MOD0_OFFSET + MOD0_SIZE).next_multiple_of(8);
/// Offset of the first instruction, which the entry branch targets.
pub const CODE_OFFSET: usize = DYNAMIC_OFFSET + DYNAMIC_SIZE;

/// Why an NRO could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NroBuildError {
    /// There is no code to run.
    #[error("an NRO needs at least one instruction")]
    EmptyCode,
    /// The image, or `.bss` placed after it, does not fit the format's 32-bit sizes.
    #[error("the image does not fit 32-bit sizes")]
    TooLarge,
    /// The entry branch could not be encoded.
    #[error("entry branch: {0}")]
    Entry(#[from] EncodeError),
    /// A field would land outside the image (a bug in this builder, reported rather than
    /// panicking).
    #[error("a header field at {offset:#x} lies outside the image")]
    OutOfImage {
        /// The field's offset.
        offset: usize,
    },
}

/// Builds the bytes of an NRO from code, read-only data and data.
///
/// ```
/// use starquake_testkit::a64;
/// use starquake_testkit::nro::{NroBuilder, MAGIC, MAGIC_OFFSET};
///
/// let nro = NroBuilder::new(&[a64::movz(0, 1, 0)?, a64::ret()]).build()?;
/// assert_eq!(nro[MAGIC_OFFSET..MAGIC_OFFSET + 4], MAGIC);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct NroBuilder {
    code: Vec<u32>,
    rodata: Vec<u8>,
    data: Vec<u8>,
    bss_size: u32,
    build_id: [u8; BUILD_ID_SIZE],
    homebrew: bool,
}

impl NroBuilder {
    /// An NRO that runs `code` (instruction words, see [`crate::a64`]) from its entry point, with
    /// no read-only data, data or `.bss`, a zero build id and the homebrew marker.
    pub fn new(code: &[u32]) -> Self {
        Self {
            code: code.to_vec(),
            rodata: Vec::new(),
            data: Vec::new(),
            bss_size: 0,
            build_id: [0; BUILD_ID_SIZE],
            homebrew: true,
        }
    }

    /// Sets the read-only data segment's contents.
    pub fn rodata(mut self, rodata: &[u8]) -> Self {
        self.rodata = rodata.to_vec();
        self
    }

    /// Sets the data segment's contents.
    pub fn data(mut self, data: &[u8]) -> Self {
        self.data = data.to_vec();
        self
    }

    /// Sets the size of `.bss`, which follows the data segment. It is written as given, to the
    /// header and to `MOD0`; the loader page-aligns it.
    pub fn bss_size(mut self, bss_size: u32) -> Self {
        self.bss_size = bss_size;
        self
    }

    /// Sets the build id (module id).
    pub fn build_id(mut self, build_id: [u8; BUILD_ID_SIZE]) -> Self {
        self.build_id = build_id;
        self
    }

    /// Sets whether the `HOMEBREW` marker is written (the bytes are zero when it is not).
    pub fn homebrew(mut self, homebrew: bool) -> Self {
        self.homebrew = homebrew;
        self
    }

    /// Lays out and returns the file.
    pub fn build(&self) -> Result<Vec<u8>, NroBuildError> {
        if self.code.is_empty() {
            return Err(NroBuildError::EmptyCode);
        }
        let code_len = self
            .code
            .len()
            .checked_mul(usize::try_from(a64::INSN_SIZE).map_err(|_| NroBuildError::TooLarge)?)
            .ok_or(NroBuildError::TooLarge)?;
        let layout = Layout::new(code_len, self.rodata.len(), self.data.len())?;
        let bss_start = layout.image_size;
        let bss_end = bss_start
            .checked_add(self.bss_size)
            .ok_or(NroBuildError::TooLarge)?;

        let mut image = Image(vec![0; usize_of(layout.image_size)?]);

        // Header.
        let entry = a64::b(
            i64::try_from(CODE_OFFSET - ENTRY_OFFSET).map_err(|_| NroBuildError::TooLarge)?,
        )?;
        image.put_u32(ENTRY_OFFSET, entry)?;
        image.put_u32(MOD0_POINTER_OFFSET, u32_of(MOD0_OFFSET)?)?;
        if self.homebrew {
            image.put(HOMEBREW_OFFSET, &HOMEBREW_MARKER)?;
        }
        image.put(MAGIC_OFFSET, &MAGIC)?;
        image.put_u32(SIZE_OFFSET, layout.image_size)?;
        for (index, (offset, size)) in layout.segments.into_iter().enumerate() {
            let at = SEGMENTS_OFFSET + index * SEGMENT_HEADER_SIZE;
            image.put_u32(at, offset)?;
            image.put_u32(at + SEGMENT_SIZE_OFFSET, size)?;
        }
        image.put_u32(BSS_SIZE_OFFSET, self.bss_size)?;
        image.put(BUILD_ID_OFFSET, &self.build_id)?;

        // MOD0, with offsets relative to the MOD0 header (see the module docs).
        let relative = |absolute: usize| u32_of(absolute - MOD0_OFFSET);
        let dynamic_end = relative(DYNAMIC_OFFSET + DYNAMIC_SIZE)?;
        let bss_start_rel = bss_start
            .checked_sub(u32_of(MOD0_OFFSET)?)
            .ok_or(NroBuildError::TooLarge)?;
        let bss_end_rel = bss_end
            .checked_sub(u32_of(MOD0_OFFSET)?)
            .ok_or(NroBuildError::TooLarge)?;
        image.put(MOD0_OFFSET, &MOD0_MAGIC)?;
        let mod0_words = [
            relative(DYNAMIC_OFFSET)?, // dynamic_offset
            bss_start_rel,             // bss_start_offset
            bss_end_rel,               // bss_end_offset
            dynamic_end,               // unwind_start_offset (empty range)
            dynamic_end,               // unwind_end_offset
            bss_start_rel,             // module_offset, "typically equal to .bss base"
        ];
        for (index, word) in mod0_words.into_iter().enumerate() {
            image.put_u32(
                MOD0_OFFSET + MOD0_FIELD_SIZE + index * MOD0_FIELD_SIZE,
                word,
            )?;
        }
        // The dynamic section is already all zero: one DT_NULL.

        // Contents.
        image.put(CODE_OFFSET, &a64::to_bytes(&self.code))?;
        let [_, (rodata_at, _), (data_at, _)] = layout.segments;
        image.put(usize_of(rodata_at)?, &self.rodata)?;
        image.put(usize_of(data_at)?, &self.data)?;
        Ok(image.0)
    }
}

/// Where each part of the image goes, with sizes checked against the format's 32-bit fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    /// `(offset, size)` of text, rodata and data; each size is a page multiple.
    segments: [(u32, u32); 3],
    /// Total size of the image (and of the file).
    image_size: u32,
}

impl Layout {
    /// Lays out `code_len` bytes of code (after the headers), `rodata_len` and `data_len`.
    fn new(code_len: usize, rodata_len: usize, data_len: usize) -> Result<Self, NroBuildError> {
        let text_len = CODE_OFFSET
            .checked_add(code_len)
            .ok_or(NroBuildError::TooLarge)?;
        let mut segments = [(0u32, 0u32); 3];
        let mut offset = 0u32;
        for (segment, len) in segments.iter_mut().zip([text_len, rodata_len, data_len]) {
            let size = len
                .checked_next_multiple_of(PAGE_SIZE)
                .ok_or(NroBuildError::TooLarge)?;
            let size = u32_of(size)?;
            *segment = (offset, size);
            offset = offset.checked_add(size).ok_or(NroBuildError::TooLarge)?;
        }
        Ok(Self {
            segments,
            image_size: offset,
        })
    }
}

/// The image being written; every write is bounds-checked.
struct Image(Vec<u8>);

impl Image {
    /// Copies `bytes` to `offset`.
    fn put(&mut self, offset: usize, bytes: &[u8]) -> Result<(), NroBuildError> {
        let out = NroBuildError::OutOfImage { offset };
        let end = offset.checked_add(bytes.len()).ok_or(out)?;
        self.0
            .get_mut(offset..end)
            .ok_or(out)?
            .copy_from_slice(bytes);
        Ok(())
    }

    /// Writes a little-endian `u32` at `offset`.
    fn put_u32(&mut self, offset: usize, value: u32) -> Result<(), NroBuildError> {
        self.put(offset, &value.to_le_bytes())
    }
}

/// `value` as a `u32` field.
fn u32_of(value: usize) -> Result<u32, NroBuildError> {
    u32::try_from(value).map_err(|_| NroBuildError::TooLarge)
}

/// `value` as an in-memory size.
fn usize_of(value: u32) -> Result<usize, NroBuildError> {
    usize::try_from(value).map_err(|_| NroBuildError::TooLarge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a64::{brk, movz, nop, ret};
    use proptest::prelude::*;

    /// Fixed seed so property runs are reproducible.
    const PROPTEST_SEED: u64 = 0x5EED_0000_0420;

    /// The fields of `NroHeader` in `eden/src/core/hle/service/ro/ro_types.h`, in order, with
    /// their sizes in bytes. Offsets are derived from this list, not restated.
    const ENTRY: &str = "m_entrypoint_insn";
    const MOD_OFFSET: &str = "m_mod_offset";
    const MARKER: &str = "padding (HOMEBREW)";
    const MAGIC_FIELD: &str = "m_magic";
    const SIZE: &str = "m_size";
    const TEXT_OFFSET: &str = "m_text_offset";
    const TEXT_SIZE: &str = "m_text_size";
    const RO_OFFSET: &str = "m_ro_offset";
    const RO_SIZE: &str = "m_ro_size";
    const RW_OFFSET: &str = "m_rw_offset";
    const RW_SIZE: &str = "m_rw_size";
    const BSS: &str = "m_bss_size";
    const MODULE_ID: &str = "m_module_id";
    const REFERENCE_HEADER: [(&str, usize); 17] = [
        (ENTRY, 4),
        (MOD_OFFSET, 4),
        (MARKER, 8),
        (MAGIC_FIELD, 4),
        ("padding", 4),
        (SIZE, 4),
        ("padding", 4),
        (TEXT_OFFSET, 4),
        (TEXT_SIZE, 4),
        (RO_OFFSET, 4),
        (RO_SIZE, 4),
        (RW_OFFSET, 4),
        (RW_SIZE, 4),
        (BSS, 4),
        ("padding", 4),
        (MODULE_ID, 0x20),
        ("padding", 0x20),
    ];
    /// The fields of `ModHeader` in `eden/src/core/loader/nro.cpp`, in order; all four bytes.
    const REFERENCE_MOD0: [&str; 7] = [
        "magic",
        "dynamic_offset",
        "bss_start_offset",
        "bss_end_offset",
        "unwind_start_offset",
        "unwind_end_offset",
        "module_offset",
    ];

    fn reference_offset(name: &str) -> usize {
        let index = REFERENCE_HEADER
            .iter()
            .position(|(field, _)| *field == name)
            .unwrap();
        REFERENCE_HEADER[..index].iter().map(|(_, size)| size).sum()
    }

    fn mod0_field_offset(name: &str) -> usize {
        MOD0_OFFSET + MOD0_FIELD_SIZE * REFERENCE_MOD0.iter().position(|f| *f == name).unwrap()
    }

    /// The size of one instruction as a byte count.
    fn insn_size() -> usize {
        usize::try_from(a64::INSN_SIZE).unwrap()
    }

    fn u32_at(file: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(file[offset..offset + 4].try_into().unwrap())
    }

    /// A pattern that differs per position and per `tag`, so swapped or shifted copies show.
    fn pattern(tag: u8, len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| tag.wrapping_add(u8::try_from(i % 251).unwrap()))
            .collect()
    }

    /// Three code words, a rodata one byte past a page, a 5-byte data, a distinctive bss size and
    /// build id.
    fn sample() -> (NroBuilder, Vec<u8>) {
        let code = [movz(0, 1, 0).unwrap(), brk(0x77).unwrap(), ret()];
        let mut build_id = [0; BUILD_ID_SIZE];
        for (i, byte) in build_id.iter_mut().enumerate() {
            *byte = u8::try_from(i).unwrap() ^ 0xA5;
        }
        let builder = NroBuilder::new(&code)
            .rodata(&pattern(0x10, PAGE_SIZE + 1))
            .data(&pattern(0x20, 5))
            .bss_size(0x1234)
            .build_id(build_id);
        let file = builder.build().unwrap();
        (builder, file)
    }

    #[test]
    fn the_offset_constants_match_the_reference_header() {
        let total: usize = REFERENCE_HEADER.iter().map(|(_, size)| size).sum();
        assert_eq!(HEADER_SIZE, total);
        assert_eq!(ENTRY_OFFSET, reference_offset(ENTRY));
        assert_eq!(MOD0_POINTER_OFFSET, reference_offset(MOD_OFFSET));
        assert_eq!(HOMEBREW_OFFSET, reference_offset(MARKER));
        assert_eq!(HOMEBREW_MARKER.len(), 8);
        assert_eq!(MAGIC_OFFSET, reference_offset(MAGIC_FIELD));
        assert_eq!(SIZE_OFFSET, reference_offset(SIZE));
        assert_eq!(SEGMENTS_OFFSET, reference_offset(TEXT_OFFSET));
        assert_eq!(BSS_SIZE_OFFSET, reference_offset(BSS));
        assert_eq!(BUILD_ID_OFFSET, reference_offset(MODULE_ID));
        assert_eq!(
            BUILD_ID_SIZE,
            REFERENCE_HEADER
                .iter()
                .find(|(field, _)| *field == MODULE_ID)
                .unwrap()
                .1
        );
        assert_eq!(MOD0_SIZE, REFERENCE_MOD0.len() * MOD0_FIELD_SIZE);
    }

    #[test]
    fn header_fields_sit_at_the_reference_offsets() {
        let (_, file) = sample();
        // text holds the headers and 3 code words; rodata is a page and a byte; data is 5 bytes.
        let (text, rodata, data) = (PAGE_SIZE, 2 * PAGE_SIZE, PAGE_SIZE);
        let image = text + rodata + data;
        assert_eq!(file.len(), image);
        let at = |name: &str| u32_at(&file, reference_offset(name));
        assert_eq!(at(MOD_OFFSET), u32::try_from(MOD0_OFFSET).unwrap());
        assert_eq!(
            &file[reference_offset(MARKER)..][..HOMEBREW_MARKER.len()],
            &HOMEBREW_MARKER
        );
        assert_eq!(
            &file[reference_offset(MAGIC_FIELD)..][..MAGIC.len()],
            &MAGIC
        );
        assert_eq!(at(SIZE), u32::try_from(image).unwrap());
        assert_eq!(at(TEXT_OFFSET), 0);
        assert_eq!(at(TEXT_SIZE), u32::try_from(text).unwrap());
        assert_eq!(at(RO_OFFSET), u32::try_from(text).unwrap());
        assert_eq!(at(RO_SIZE), u32::try_from(rodata).unwrap());
        assert_eq!(at(RW_OFFSET), u32::try_from(text + rodata).unwrap());
        assert_eq!(at(RW_SIZE), u32::try_from(data).unwrap());
        assert_eq!(at(BSS), 0x1234);
        let build_id: Vec<u8> = (0..BUILD_ID_SIZE)
            .map(|i| u8::try_from(i).unwrap() ^ 0xA5)
            .collect();
        assert_eq!(
            &file[reference_offset(MODULE_ID)..][..BUILD_ID_SIZE],
            build_id
        );
        // Everything else in the header is zero: padding, and nothing leaks across fields.
        let covered: Vec<(usize, usize)> = [
            (ENTRY, 4),
            (MOD_OFFSET, 4),
            (MARKER, 8),
            (MAGIC_FIELD, 4),
            (SIZE, 4),
            (TEXT_OFFSET, 4),
            (TEXT_SIZE, 4),
            (RO_OFFSET, 4),
            (RO_SIZE, 4),
            (RW_OFFSET, 4),
            (RW_SIZE, 4),
            (BSS, 4),
            (MODULE_ID, BUILD_ID_SIZE),
        ]
        .map(|(name, size)| (reference_offset(name), size))
        .to_vec();
        for (offset, byte) in file[..HEADER_SIZE].iter().enumerate() {
            let in_field = covered
                .iter()
                .any(|(at, size)| (*at..at + size).contains(&offset));
            assert!(in_field || *byte == 0, "stray byte at {offset:#x}");
        }
    }

    #[test]
    fn the_entry_word_branches_to_the_code() {
        let (_, file) = sample();
        let branch = crate::a64::b(i64::try_from(CODE_OFFSET).unwrap()).unwrap();
        assert_eq!(u32_at(&file, reference_offset(ENTRY)), branch);
        // And the code is there, in order.
        let code = [movz(0, 1, 0).unwrap(), brk(0x77).unwrap(), ret()];
        assert_eq!(
            file[CODE_OFFSET..CODE_OFFSET + code.len() * insn_size()],
            crate::a64::to_bytes(&code)[..]
        );
    }

    #[test]
    fn mod0_fields_follow_the_reference_order_and_are_relative_to_mod0() {
        let (_, file) = sample();
        let image = u32::try_from(4 * PAGE_SIZE).unwrap();
        let mod0 = u32::try_from(MOD0_OFFSET).unwrap();
        let dynamic = u32::try_from(DYNAMIC_OFFSET).unwrap();
        let dynamic_end = u32::try_from(DYNAMIC_OFFSET + DYNAMIC_SIZE).unwrap();
        let at = |name: &str| u32_at(&file, mod0_field_offset(name));
        assert_eq!(
            &file[mod0_field_offset("magic")..][..MOD0_MAGIC.len()],
            &MOD0_MAGIC
        );
        assert_eq!(at("dynamic_offset"), dynamic - mod0);
        assert_eq!(at("bss_start_offset"), image - mod0);
        assert_eq!(at("bss_end_offset"), image - mod0 + 0x1234);
        assert_eq!(at("unwind_start_offset"), dynamic_end - mod0);
        assert_eq!(at("unwind_end_offset"), dynamic_end - mod0);
        assert_eq!(at("module_offset"), image - mod0);
        // The loader derives .bss from the difference of the two bss fields.
        assert_eq!(at("bss_end_offset") - at("bss_start_offset"), 0x1234);
        // The dynamic section is one DT_NULL.
        assert_eq!(
            file[DYNAMIC_OFFSET..DYNAMIC_OFFSET + DYNAMIC_SIZE],
            [0; DYNAMIC_SIZE]
        );
        // The structures do not overlap.
        assert_eq!(MOD0_OFFSET, HEADER_SIZE);
        const { assert!(DYNAMIC_OFFSET >= MOD0_OFFSET + MOD0_SIZE) };
        assert_eq!(DYNAMIC_OFFSET % 8, 0);
        assert_eq!(CODE_OFFSET, DYNAMIC_OFFSET + DYNAMIC_SIZE);
    }

    #[test]
    fn segments_hold_their_own_contents_padded_with_zeros() {
        let (_, file) = sample();
        let rodata_at = PAGE_SIZE;
        let data_at = 3 * PAGE_SIZE;
        let rodata_len = PAGE_SIZE + 1;
        assert_eq!(
            file[rodata_at..][..rodata_len],
            pattern(0x10, rodata_len)[..]
        );
        assert!(
            file[rodata_at + rodata_len..data_at]
                .iter()
                .all(|b| *b == 0)
        );
        assert_eq!(file[data_at..data_at + 5], pattern(0x20, 5)[..]);
        assert!(file[data_at + 5..].iter().all(|b| *b == 0));
        // The text segment's tail after the code is zero as well.
        assert!(
            file[CODE_OFFSET + 3 * insn_size()..rodata_at]
                .iter()
                .all(|b| *b == 0)
        );
    }

    #[test]
    fn the_homebrew_marker_is_optional() {
        let file = NroBuilder::new(&[nop()]).homebrew(false).build().unwrap();
        assert_eq!(
            file[HOMEBREW_OFFSET..HOMEBREW_OFFSET + HOMEBREW_MARKER.len()],
            [0; HOMEBREW_MARKER.len()]
        );
        let file = NroBuilder::new(&[nop()]).build().unwrap();
        assert_eq!(
            file[HOMEBREW_OFFSET..HOMEBREW_OFFSET + HOMEBREW_MARKER.len()],
            HOMEBREW_MARKER
        );
    }

    #[test]
    fn empty_segments_take_no_space() {
        let file = NroBuilder::new(&[nop()]).build().unwrap();
        assert_eq!(file.len(), PAGE_SIZE);
        let text = u32::try_from(PAGE_SIZE).unwrap();
        assert_eq!(u32_at(&file, reference_offset(RO_OFFSET)), text);
        assert_eq!(u32_at(&file, reference_offset(RO_SIZE)), 0);
        assert_eq!(u32_at(&file, reference_offset(RW_OFFSET)), text);
        assert_eq!(u32_at(&file, reference_offset(RW_SIZE)), 0);
    }

    #[test]
    fn code_that_fills_the_text_page_exactly_does_not_grow_it() {
        let words = (PAGE_SIZE - CODE_OFFSET) / insn_size();
        let file = NroBuilder::new(&vec![nop(); words]).build().unwrap();
        assert_eq!(file.len(), PAGE_SIZE);
        let file = NroBuilder::new(&vec![nop(); words + 1]).build().unwrap();
        assert_eq!(file.len(), 2 * PAGE_SIZE);
    }

    #[test]
    fn no_code_is_an_error() {
        assert_eq!(NroBuilder::new(&[]).build(), Err(NroBuildError::EmptyCode));
    }

    #[test]
    fn bss_must_fit_after_the_image() {
        let image = u32::try_from(PAGE_SIZE).unwrap();
        let fits = NroBuilder::new(&[nop()]).bss_size(u32::MAX - image).build();
        assert!(fits.is_ok());
        let too_big = NroBuilder::new(&[nop()])
            .bss_size(u32::MAX - image + 1)
            .build();
        assert_eq!(too_big, Err(NroBuildError::TooLarge));
        let too_big = NroBuilder::new(&[nop()]).bss_size(u32::MAX).build();
        assert_eq!(too_big, Err(NroBuildError::TooLarge));
    }

    #[test]
    fn layout_rejects_sizes_that_overflow_32_bits() {
        let page = PAGE_SIZE;
        let max = usize::try_from(u32::MAX).unwrap();
        let last_page = max - (page - 1); // the largest page multiple that fits a u32
        // text is one page here, so the rodata can be at most `last_page - page`.
        let ok = Layout::new(0, last_page - page, 0).unwrap();
        assert_eq!(ok.image_size, u32::try_from(last_page).unwrap());
        assert_eq!(
            Layout::new(0, last_page - page + 1, 0),
            Err(NroBuildError::TooLarge)
        );
        assert_eq!(Layout::new(0, last_page, 0), Err(NroBuildError::TooLarge));
        assert_eq!(Layout::new(0, 0, last_page), Err(NroBuildError::TooLarge));
        assert_eq!(Layout::new(0, max, 0), Err(NroBuildError::TooLarge));
        assert_eq!(Layout::new(usize::MAX, 0, 0), Err(NroBuildError::TooLarge));
        assert_eq!(Layout::new(0, usize::MAX, 0), Err(NroBuildError::TooLarge));
        assert_eq!(Layout::new(0, 0, usize::MAX), Err(NroBuildError::TooLarge));
        // The sum of two segments each below the limit.
        assert_eq!(
            Layout::new(0, last_page / 2, last_page / 2 + page),
            Err(NroBuildError::TooLarge)
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            rng_seed: proptest::test_runner::RngSeed::Fixed(PROPTEST_SEED),
            cases: 64,
            ..ProptestConfig::default()
        })]

        /// Whatever goes in comes back out of the segments the header describes: a reader that
        /// uses only the reference offsets finds each part, and swapping any two reads fails
        /// because the three parts and their sizes all differ.
        #[test]
        fn segments_read_back_through_the_header(
            code_words in 1usize..2000,
            rodata_len in 0usize..9000,
            data_len in 0usize..9000,
            bss in any::<u16>(),
        ) {
            let code = vec![nop(); code_words];
            let rodata = pattern(0x31, rodata_len);
            let data = pattern(0x62, data_len);
            let file = NroBuilder::new(&code)
                .rodata(&rodata)
                .data(&data)
                .bss_size(u32::from(bss))
                .build()
                .unwrap();
            let field = |name: &str| usize::try_from(u32_at(&file, reference_offset(name))).unwrap();
            let segments = [
                (field(TEXT_OFFSET), field(TEXT_SIZE)),
                (field(RO_OFFSET), field(RO_SIZE)),
                (field(RW_OFFSET), field(RW_SIZE)),
            ];
            let mut next = 0;
            for (offset, size) in segments {
                prop_assert_eq!(offset, next);
                prop_assert_eq!(size % PAGE_SIZE, 0);
                next = offset + size;
            }
            prop_assert_eq!(next, file.len());
            prop_assert_eq!(field(SIZE), file.len());
            let (_, text_size) = segments[0];
            prop_assert!(text_size >= CODE_OFFSET + code_words * insn_size());
            prop_assert_eq!(text_size, (CODE_OFFSET + code_words * insn_size()).next_multiple_of(PAGE_SIZE));
            prop_assert_eq!(segments[1].1, rodata_len.next_multiple_of(PAGE_SIZE));
            prop_assert_eq!(segments[2].1, data_len.next_multiple_of(PAGE_SIZE));
            prop_assert_eq!(&file[segments[1].0..][..rodata_len], &rodata[..]);
            prop_assert_eq!(&file[segments[2].0..][..data_len], &data[..]);
            prop_assert_eq!(field(BSS), usize::from(bss));
        }
    }
}

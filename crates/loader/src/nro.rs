//! NRO homebrew executables.
//!
//! Parsing only: [`Nro::parse`] reads the 0x80-byte header, the `MOD0` header it points at and the
//! optional `ASET` asset section appended after the image. Mapping the image and applying
//! relocations belong to the kernel and are out of scope here.
//!
//! Ported from `eden/src/core/loader/nro.cpp` @ cf654052 (GPL-3.0-or-later): `NroHeader`,
//! `ModHeader`, `AssetHeader`, `AssetSection` and the `AppLoader_NRO` constructor. Eden reads the
//! `MOD0` header and the asset sections without bounds checks; here every offset and size is
//! checked against the input.
//!
//! The build id at header offset 0x40 is `NroHeader::m_module_id` (0x20 bytes) in
//! `eden/src/core/hle/service/ro/ro_types.h` @ cf654052, which `ro.cpp` compares to detect an
//! already-loaded module. `nro.cpp`'s own `NroHeader` skips those bytes as padding.

use std::ops::Range;

/// `NRO0` magic at offset 0x10 (`NroHeader::magic`).
pub const MAGIC: [u8; 4] = *b"NRO0";
/// Size of the NRO header (`static_assert(sizeof(NroHeader) == 0x80)`).
pub const HEADER_SIZE: usize = 0x80;
/// `MOD0` magic at the start of the module header (`ModHeader::magic`).
pub const MOD0_MAGIC: [u8; 4] = *b"MOD0";
/// Size of the `MOD0` header (`static_assert(sizeof(ModHeader) == 0x1c)`).
pub const MOD0_SIZE: usize = 0x1C;
/// `ASET` magic at the start of the asset header (`AssetHeader::magic`).
pub const ASSET_MAGIC: [u8; 4] = *b"ASET";
/// Size of the asset header (`static_assert(sizeof(AssetHeader) == 0x38)`).
pub const ASSET_HEADER_SIZE: usize = 0x38;

/// Offset of the pointer to the `MOD0` header (`NroHeader::module_header_offset`).
const MOD0_POINTER_OFFSET: usize = 0x4;
/// Offset of the homebrew marker, `HOMEBREW`, as two magics (`NroHeader::magic_ext1/2`).
const HOMEBREW_OFFSET: usize = 0x8;
/// The homebrew marker bytes (`AppLoader_NRO::IsHomebrew`: `HOME` then `BREW`).
const HOMEBREW_MARKER: [u8; 8] = *b"HOMEBREW";
/// Offset of the magic.
const MAGIC_OFFSET: usize = 0x10;
/// Offset of the image size (`NroHeader::file_size`).
const SIZE_OFFSET: usize = 0x18;
/// Offset of the three `{offset, size}` segment headers (`NroHeader::segments`).
const SEGMENTS_OFFSET: usize = 0x20;
/// Offset of the bss size (`NroHeader::bss_size`).
const BSS_SIZE_OFFSET: usize = 0x38;
/// Offset of the build id (`NroHeader::m_module_id` in `eden/src/core/hle/service/ro/ro_types.h`).
const BUILD_ID_OFFSET: usize = 0x40;

/// An NRO segment: a byte range of the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NroSegment {
    /// Offset from the start of the image (`NroSegmentHeader::offset`).
    pub offset: u32,
    /// Size in bytes (`NroSegmentHeader::size`).
    pub size: u32,
}

/// The `MOD0` module header. All fields are raw, as read from the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mod0 {
    /// Offset of the dynamic section (`ModHeader::dynamic_offset`).
    pub dynamic_offset: u32,
    /// Start of `.bss` (`ModHeader::bss_start_offset`).
    pub bss_start_offset: u32,
    /// End of `.bss` (`ModHeader::bss_end_offset`).
    pub bss_end_offset: u32,
    /// Start of the unwind (`eh_frame_hdr`) data (`ModHeader::unwind_start_offset`).
    pub unwind_start_offset: u32,
    /// End of the unwind data (`ModHeader::unwind_end_offset`).
    pub unwind_end_offset: u32,
    /// Offset of the runtime-generated module object, typically the `.bss` base
    /// (`ModHeader::module_offset`).
    pub module_offset: u32,
}

impl Mod0 {
    /// Size of `.bss` according to `MOD0`, which Eden prefers over the NRO header's value when the
    /// `MOD0` header is present. `None` if the end precedes the start.
    pub fn bss_size(&self) -> Option<u32> {
        self.bss_end_offset.checked_sub(self.bss_start_offset)
    }
}

/// A section of the `ASET` block: an offset (relative to the start of the `ASET` header) and size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssetSection {
    /// Offset from the start of the `ASET` header (`AssetSection::offset`).
    pub offset: u64,
    /// Size in bytes; zero means the section is absent (`AssetSection::size`).
    pub size: u64,
}

/// The optional `ASET` block appended to homebrew NROs. Only locations are parsed, not contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NroAssets {
    /// File offset of the `ASET` header, which is the image size.
    pub base: u32,
    /// Format version; Eden warns when it is not 0 (`AssetHeader::format_version`).
    pub format_version: u32,
    /// The icon.
    pub icon: AssetSection,
    /// The NACP control data.
    pub nacp: AssetSection,
    /// The RomFS image.
    pub romfs: AssetSection,
}

impl NroAssets {
    /// The byte range of `section` within the file, or `None` if the section is empty.
    ///
    /// [`Nro::parse`] has already checked that non-empty sections fit in the file.
    pub fn file_range(&self, section: &AssetSection) -> Option<Range<usize>> {
        if section.size == 0 {
            return None;
        }
        let start = u64::from(self.base).checked_add(section.offset)?;
        let end = start.checked_add(section.size)?;
        Some(usize::try_from(start).ok()?..usize::try_from(end).ok()?)
    }
}

/// A parsed NRO.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nro {
    /// Size of the image (`NroHeader::file_size`). Any `ASET` block follows it.
    pub size: u32,
    /// The `.text` segment.
    pub text: NroSegment,
    /// The `.rodata` segment.
    pub rodata: NroSegment,
    /// The `.data` segment.
    pub data: NroSegment,
    /// Size of `.bss` from the NRO header. Use [`Mod0::bss_size`] first when `MOD0` is present.
    pub bss_size: u32,
    /// Build id (the module id), header offset 0x40 (`NroHeader::m_module_id` in
    /// `eden/src/core/hle/service/ro/ro_types.h`).
    pub build_id: [u8; 0x20],
    /// True when the header carries the `HOMEBREW` marker (`AppLoader_NRO::IsHomebrew`).
    pub is_homebrew: bool,
    /// The `MOD0` pointer from the header (`NroHeader::module_header_offset`).
    pub mod0_offset: u32,
    /// The `MOD0` header, if the pointer lands on one. Eden treats a missing `MOD0` as "use the
    /// header's bss size", so absence is not an error.
    pub mod0: Option<Mod0>,
    /// The `ASET` block, if one follows the image.
    pub assets: Option<NroAssets>,
}

impl Nro {
    /// The segments in header order: text, rodata, data.
    pub fn segments(&self) -> [NroSegment; 3] {
        [self.text, self.rodata, self.data]
    }

    /// Parses an NRO.
    ///
    /// Every offset and size is checked against the input, so malformed input is an error and
    /// never a panic.
    pub fn parse(bytes: &[u8]) -> Result<Self, NroError> {
        if bytes.len() < HEADER_SIZE {
            return Err(NroError::Truncated {
                len: bytes.len(),
                needed: HEADER_SIZE,
            });
        }
        if read_array::<4>(bytes, MAGIC_OFFSET) != Some(MAGIC) {
            return Err(NroError::BadMagic);
        }
        let size = u32_at(bytes, SIZE_OFFSET)?;
        let size_usize = usize::try_from(size).map_err(|_| NroError::SizeTooLarge {
            size,
            len: bytes.len(),
        })?;
        if size_usize > bytes.len() || size_usize < HEADER_SIZE {
            return Err(NroError::SizeTooLarge {
                size,
                len: bytes.len(),
            });
        }

        let mut segments = [NroSegment { offset: 0, size: 0 }; 3];
        for (index, segment) in segments.iter_mut().enumerate() {
            let at = SEGMENTS_OFFSET + index * 8;
            *segment = NroSegment {
                offset: u32_at(bytes, at)?,
                size: u32_at(bytes, at + 4)?,
            };
            let end = segment
                .offset
                .checked_add(segment.size)
                .ok_or(NroError::SegmentOutOfBounds { segment: index })?;
            if end > size {
                return Err(NroError::SegmentOutOfBounds { segment: index });
            }
        }
        let [text, rodata, data] = segments;

        let mod0_offset = u32_at(bytes, MOD0_POINTER_OFFSET)?;
        let mod0 = parse_mod0(bytes, mod0_offset);

        let assets = parse_assets(bytes, size)?;

        Ok(Self {
            size,
            text,
            rodata,
            data,
            bss_size: u32_at(bytes, BSS_SIZE_OFFSET)?,
            build_id: read_array::<0x20>(bytes, BUILD_ID_OFFSET).ok_or(NroError::Truncated {
                len: bytes.len(),
                needed: HEADER_SIZE,
            })?,
            is_homebrew: read_array::<8>(bytes, HOMEBREW_OFFSET) == Some(HOMEBREW_MARKER),
            mod0_offset,
            mod0,
            assets,
        })
    }
}

/// Why an NRO could not be parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NroError {
    /// The input is shorter than a structure it must contain.
    #[error("NRO is {len} bytes, shorter than the {needed} bytes needed")]
    Truncated {
        /// Length of the input.
        len: usize,
        /// Bytes needed.
        needed: usize,
    },
    /// The magic at offset 0x10 is not `NRO0`.
    #[error("NRO magic is not NRO0")]
    BadMagic,
    /// The header's image size is smaller than the header or larger than the input.
    #[error("NRO declares a {size:#x}-byte image in a {len:#x}-byte file")]
    SizeTooLarge {
        /// Declared image size.
        size: u32,
        /// Length of the input.
        len: usize,
    },
    /// A segment's range lies outside the image.
    #[error("NRO segment {segment} lies outside the image")]
    SegmentOutOfBounds {
        /// Segment index (0 text, 1 rodata, 2 data).
        segment: usize,
    },
    /// An `ASET` section lies outside the file.
    #[error("NRO asset section {section} lies outside the file")]
    AssetOutOfBounds {
        /// Section name: `icon`, `nacp` or `romfs`.
        section: &'static str,
    },
}

/// Reads the `MOD0` header at `offset`, if one is there.
fn parse_mod0(bytes: &[u8], offset: u32) -> Option<Mod0> {
    let at = usize::try_from(offset).ok()?;
    if read_array::<4>(bytes, at)? != MOD0_MAGIC {
        return None;
    }
    let word = |index: usize| -> Option<u32> {
        read_array::<4>(bytes, at.checked_add(4 + index * 4)?).map(u32::from_le_bytes)
    };
    Some(Mod0 {
        dynamic_offset: word(0)?,
        bss_start_offset: word(1)?,
        bss_end_offset: word(2)?,
        unwind_start_offset: word(3)?,
        unwind_end_offset: word(4)?,
        module_offset: word(5)?,
    })
}

/// Reads the `ASET` block that follows an image of `size` bytes, if present.
///
/// Eden looks for it only when the file is at least `size + 0x38` bytes long and silently ignores
/// a block whose magic is wrong (`AppLoader_NRO::AppLoader_NRO`).
fn parse_assets(bytes: &[u8], size: u32) -> Result<Option<NroAssets>, NroError> {
    let Ok(base) = usize::try_from(size) else {
        return Ok(None);
    };
    let Some(header) = base
        .checked_add(ASSET_HEADER_SIZE)
        .and_then(|end| bytes.get(base..end))
    else {
        return Ok(None);
    };
    if read_array::<4>(header, 0) != Some(ASSET_MAGIC) {
        return Ok(None);
    }
    let truncated = NroError::Truncated {
        len: bytes.len(),
        needed: ASSET_HEADER_SIZE,
    };
    let word = |at: usize| u32_at(header, at).map_err(|_| truncated);
    let section = |at: usize, name: &'static str| -> Result<AssetSection, NroError> {
        let offset = read_array::<8>(header, at).map(u64::from_le_bytes);
        let len = read_array::<8>(header, at + 8).map(u64::from_le_bytes);
        let (Some(offset), Some(len)) = (offset, len) else {
            return Err(truncated);
        };
        let section = AssetSection { offset, size: len };
        if len > 0 {
            let fits = u64::from(size)
                .checked_add(offset)
                .and_then(|start| start.checked_add(len))
                .is_some_and(|end| end <= bytes.len() as u64);
            if !fits {
                return Err(NroError::AssetOutOfBounds { section: name });
            }
        }
        Ok(section)
    };
    Ok(Some(NroAssets {
        base: size,
        format_version: word(4)?,
        icon: section(8, "icon")?,
        nacp: section(0x18, "nacp")?,
        romfs: section(0x28, "romfs")?,
    }))
}

/// `N` bytes at `offset`.
fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Option<[u8; N]> {
    let end = offset.checked_add(N)?;
    <[u8; N]>::try_from(bytes.get(offset..end)?).ok()
}

/// A little-endian `u32` at `offset`.
fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, NroError> {
    read_array::<4>(bytes, offset)
        .map(u32::from_le_bytes)
        .ok_or(NroError::Truncated {
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

    const MOD0_AT: u32 = 0x80;

    /// Builds an NRO image of 0x200 bytes with a `MOD0` header at [`MOD0_AT`].
    fn image() -> Vec<u8> {
        let mut file = vec![0u8; 0x200];
        file[MOD0_POINTER_OFFSET..MOD0_POINTER_OFFSET + 4].copy_from_slice(&MOD0_AT.to_le_bytes());
        file[HOMEBREW_OFFSET..HOMEBREW_OFFSET + 8].copy_from_slice(&HOMEBREW_MARKER);
        file[MAGIC_OFFSET..MAGIC_OFFSET + 4].copy_from_slice(&MAGIC);
        file[SIZE_OFFSET..SIZE_OFFSET + 4].copy_from_slice(&0x200u32.to_le_bytes());
        // text 0x0..0x100, rodata 0x100..0x180, data 0x180..0x200.
        for (i, (offset, size)) in [(0u32, 0x100u32), (0x100, 0x80), (0x180, 0x80)]
            .into_iter()
            .enumerate()
        {
            let at = SEGMENTS_OFFSET + i * 8;
            file[at..at + 4].copy_from_slice(&offset.to_le_bytes());
            file[at + 4..at + 8].copy_from_slice(&size.to_le_bytes());
        }
        file[BSS_SIZE_OFFSET..BSS_SIZE_OFFSET + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        for (i, byte) in file[BUILD_ID_OFFSET..BUILD_ID_OFFSET + 0x20]
            .iter_mut()
            .enumerate()
        {
            *byte = u8::try_from(i).unwrap() | 0x80;
        }
        let m = MOD0_AT as usize;
        file[m..m + 4].copy_from_slice(&MOD0_MAGIC);
        for (i, value) in [0x10u32, 0x2000, 0x3000, 0x40, 0x50, 0x2000]
            .into_iter()
            .enumerate()
        {
            file[m + 4 + i * 4..m + 8 + i * 4].copy_from_slice(&value.to_le_bytes());
        }
        file
    }

    /// Appends an `ASET` block with a 4-byte icon, no NACP and an 8-byte RomFS.
    fn with_assets(mut file: Vec<u8>) -> Vec<u8> {
        let base = file.len();
        let mut aset = vec![0u8; ASSET_HEADER_SIZE];
        aset[..4].copy_from_slice(&ASSET_MAGIC);
        // icon at +0x38 size 4; nacp absent; romfs at +0x3C size 8.
        aset[0x8..0x10].copy_from_slice(&0x38u64.to_le_bytes());
        aset[0x10..0x18].copy_from_slice(&4u64.to_le_bytes());
        aset[0x28..0x30].copy_from_slice(&0x3Cu64.to_le_bytes());
        aset[0x30..0x38].copy_from_slice(&8u64.to_le_bytes());
        file.extend_from_slice(&aset);
        file.extend_from_slice(&[0xA1, 0xA2, 0xA3, 0xA4]);
        file.extend_from_slice(&[0xB0; 8]);
        assert_eq!(base, 0x200);
        file
    }

    #[test]
    fn raw_offsets_match_the_documented_layout() {
        let file = image();
        assert_eq!(&file[0x10..0x14], b"NRO0");
        assert_eq!(&file[0x8..0x10], b"HOMEBREW");
        assert_eq!(&file[0x18..0x1C], &0x200u32.to_le_bytes());
        // Rodata segment header at 0x28, bss size at 0x38.
        assert_eq!(&file[0x28..0x2C], &0x100u32.to_le_bytes());
        assert_eq!(&file[0x38..0x3C], &0x1000u32.to_le_bytes());
    }

    #[test]
    fn build_id_offset_follows_the_reference_header_layout() {
        // `NroHeader` in `eden/src/core/hle/service/ro/ro_types.h` lists, in order, the byte
        // sizes of the fields before `m_module_id`: entrypoint, mod offset, 8 padding, magic, 4
        // padding, size, 4 padding, three {offset, size} segment pairs, bss size, 4 padding.
        let before_module_id: usize = [4, 4, 8, 4, 4, 4, 4, 6 * 4, 4, 4].iter().sum();
        assert_eq!(BUILD_ID_OFFSET, before_module_id);
        // The ids bracketing the field stay out of it, and every byte of it is read in order.
        let mut file = image();
        file[BUILD_ID_OFFSET - 1] = 0xEE;
        file[BUILD_ID_OFFSET + 0x20] = 0xEE;
        let nro = Nro::parse(&file).unwrap();
        let expected: Vec<u8> = (0..0x20u8).map(|i| i | 0x80).collect();
        assert_eq!(nro.build_id.as_slice(), expected.as_slice());
    }

    #[test]
    fn parses_header_and_mod0() {
        let nro = Nro::parse(&image()).unwrap();
        assert_eq!(nro.size, 0x200);
        assert_eq!(
            nro.segments(),
            [
                NroSegment {
                    offset: 0,
                    size: 0x100
                },
                NroSegment {
                    offset: 0x100,
                    size: 0x80
                },
                NroSegment {
                    offset: 0x180,
                    size: 0x80
                },
            ]
        );
        assert_eq!(nro.bss_size, 0x1000);
        assert_eq!(nro.build_id[0], 0x80);
        assert!(nro.is_homebrew);
        assert_eq!(nro.mod0_offset, MOD0_AT);
        let mod0 = nro.mod0.unwrap();
        assert_eq!(mod0.dynamic_offset, 0x10);
        assert_eq!(mod0.bss_size(), Some(0x1000));
        assert_eq!(mod0.unwind_start_offset, 0x40);
        assert_eq!(mod0.unwind_end_offset, 0x50);
        assert_eq!(mod0.module_offset, 0x2000);
        assert!(nro.assets.is_none());
    }

    #[test]
    fn missing_or_bad_mod0_is_not_an_error() {
        let mut file = image();
        file[MOD0_AT as usize] = b'X';
        assert_eq!(Nro::parse(&file).unwrap().mod0, None);

        // A pointer past the end of the file.
        let mut file = image();
        file[MOD0_POINTER_OFFSET..MOD0_POINTER_OFFSET + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(Nro::parse(&file).unwrap().mod0, None);

        // A MOD0 that is cut off by the end of the file.
        let mut file = image();
        file[MOD0_POINTER_OFFSET..MOD0_POINTER_OFFSET + 4].copy_from_slice(&0x1F8u32.to_le_bytes());
        file[0x1F8..0x1FC].copy_from_slice(&MOD0_MAGIC);
        assert_eq!(Nro::parse(&file).unwrap().mod0, None);
    }

    #[test]
    fn mod0_with_inverted_bss_has_no_size() {
        let mut file = image();
        let m = MOD0_AT as usize;
        file[m + 12..m + 16].copy_from_slice(&0x10u32.to_le_bytes());
        assert_eq!(Nro::parse(&file).unwrap().mod0.unwrap().bss_size(), None);
    }

    #[test]
    fn non_homebrew_marker_is_reported() {
        let mut file = image();
        file[HOMEBREW_OFFSET] = 0;
        assert!(!Nro::parse(&file).unwrap().is_homebrew);
    }

    #[test]
    fn parses_the_aset_block() {
        let file = with_assets(image());
        let nro = Nro::parse(&file).unwrap();
        let assets = nro.assets.unwrap();
        assert_eq!(assets.base, 0x200);
        assert_eq!(assets.format_version, 0);
        assert_eq!(
            assets.icon,
            AssetSection {
                offset: 0x38,
                size: 4
            }
        );
        assert_eq!(assets.nacp.size, 0);
        assert_eq!(
            assets.romfs,
            AssetSection {
                offset: 0x3C,
                size: 8
            }
        );
        let icon = assets.file_range(&assets.icon).unwrap();
        assert_eq!(&file[icon], &[0xA1, 0xA2, 0xA3, 0xA4]);
        assert_eq!(assets.file_range(&assets.nacp), None);
        let romfs = assets.file_range(&assets.romfs).unwrap();
        assert_eq!(&file[romfs], &[0xB0; 8]);
    }

    #[test]
    fn wrong_aset_magic_means_no_assets() {
        let mut file = with_assets(image());
        file[0x200] = b'X';
        assert_eq!(Nro::parse(&file).unwrap().assets, None);
    }

    #[test]
    fn aset_sections_outside_the_file_are_errors() {
        let mut file = with_assets(image());
        // Grow the icon size past the end of the file (size field at ASET + 0x10).
        file[0x210..0x218].copy_from_slice(&0x1000u64.to_le_bytes());
        assert_eq!(
            Nro::parse(&file),
            Err(NroError::AssetOutOfBounds { section: "icon" })
        );

        // An offset so large that offset + size overflows.
        let mut file = with_assets(image());
        file[0x218..0x220].copy_from_slice(&u64::MAX.to_le_bytes());
        file[0x220..0x228].copy_from_slice(&1u64.to_le_bytes());
        assert_eq!(
            Nro::parse(&file),
            Err(NroError::AssetOutOfBounds { section: "nacp" })
        );
    }

    #[test]
    fn rejects_bad_headers() {
        assert!(matches!(
            Nro::parse(&image()[..HEADER_SIZE - 1]),
            Err(NroError::Truncated { .. })
        ));

        let mut file = image();
        file[MAGIC_OFFSET] = b'X';
        assert_eq!(Nro::parse(&file), Err(NroError::BadMagic));

        // Image size larger than the file, and smaller than the header.
        let mut file = image();
        file[SIZE_OFFSET..SIZE_OFFSET + 4].copy_from_slice(&0x201u32.to_le_bytes());
        assert!(matches!(
            Nro::parse(&file),
            Err(NroError::SizeTooLarge { .. })
        ));
        file[SIZE_OFFSET..SIZE_OFFSET + 4].copy_from_slice(&0x7Fu32.to_le_bytes());
        assert!(matches!(
            Nro::parse(&file),
            Err(NroError::SizeTooLarge { .. })
        ));

        // A segment running past the image, and one whose end overflows.
        let mut file = image();
        file[SEGMENTS_OFFSET + 12..SEGMENTS_OFFSET + 16].copy_from_slice(&0x101u32.to_le_bytes());
        assert_eq!(
            Nro::parse(&file),
            Err(NroError::SegmentOutOfBounds { segment: 1 })
        );
        let mut file = image();
        file[SEGMENTS_OFFSET + 16..SEGMENTS_OFFSET + 20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            Nro::parse(&file),
            Err(NroError::SegmentOutOfBounds { segment: 2 })
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            rng_seed: proptest::test_runner::RngSeed::Fixed(PROPTEST_SEED),
            ..ProptestConfig::default()
        })]
        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..1024)) {
            let _ = Nro::parse(&bytes);
        }

        #[test]
        fn arbitrary_bytes_behind_a_valid_header_never_panic(
            head in proptest::collection::vec(any::<u8>(), HEADER_SIZE),
            tail in proptest::collection::vec(any::<u8>(), 0..512),
        ) {
            let mut bytes = head;
            bytes[MAGIC_OFFSET..MAGIC_OFFSET + 4].copy_from_slice(&MAGIC);
            bytes.extend_from_slice(&tail);
            let _ = Nro::parse(&bytes);
        }

        #[test]
        fn mutated_valid_files_never_panic(index in 0usize..0x280, value in any::<u8>()) {
            let mut file = with_assets(image());
            let at = index % file.len();
            file[at] = value;
            let _ = Nro::parse(&file);
        }
    }
}

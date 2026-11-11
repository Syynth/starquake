//! NPDM program metadata: META header, ACID (Switch 1 and Switch 2 layouts), ACI0, kernel capabilities.
//!
//! Parsing only: [`Npdm::parse`] takes the bytes of a `main.npdm` and returns owned, validated
//! data. Kernel capabilities are decoded in [`capabilities`], service access control in
//! [`service_access`].
//!
//! Ported from `eden/src/core/file_sys/program_metadata.{h,cpp}` @ cf654052 (GPL-3.0-or-later):
//! the `Header`, `AcidHeader`, `AciHeader`, `FileAccessControl` and `FileAccessHeader` layouts.
//! The bounds checks and the probing of both ACID layouts are ported from
//! `eden-pr4588/src/core/file_sys/program_metadata.cpp` @ e5fb3b97 (GPL-3.0-or-later).
//! `eden-nx2/tools/nx2/npdm_info.py` is the independent reader the fixtures were cross-checked
//! against.
//!
//! # ACID layouts
//!
//! The `ACID` block starts with a signature and key area, then its `ACID` magic. On Switch 1 that
//! area is 0x200 bytes, so the magic is at block offset 0x200. On Switch 2 it is 0x300 bytes, so
//! the magic is at 0x300 (Eden-NX2 notes, blocker row 5 in `docs/switch2-eden-diff-scoping.md`,
//! "Process, memory, kernel"). [`Npdm::parse`] probes 0x200 first, then 0x300, and reports which
//! it found in [`Acid::layout`].
//!
//! The FAC, SAC and KAC offsets inside the `ACID` body, and the FAH, SAC and KAC offsets inside
//! `ACI0`, are relative to the start of their own block (the `ACID` or `ACI0` block, not the
//! file). Eden mainline reads FAC and FAH from the start of the file, which is wrong (same
//! blocker row); this port adds the block base as `eden-pr4588` does.
//!
//! # Layout facts
//!
//! `META` is 0x80 bytes: magic at 0, flags at 0xC (bit 0 is 64-bit, bits 1 to 3 the address space
//! type), main thread priority at 0xE and core at 0xF, system resource size at 0x14, process
//! category at 0x18, main thread stack size at 0x1C, a 16-byte name at 0x20, then the `ACI0`
//! offset and size at 0x70 and the `ACID` offset and size at 0x78 (`program_metadata.h`,
//! `Header`). `npdm_info.py` labels the words at 0x10 and 0x14 differently (`sys_res` at 0x10,
//! `version` at 0x14); Eden's offsets are used here, and the discrepancy is recorded in the PR.

pub mod capabilities;
pub mod service_access;

use capabilities::{CapabilityError, KernelCapabilities};
use service_access::{ServiceAccess, ServiceAccessError};

/// `META` magic at offset 0 (`Header::magic`, checked by `eden-pr4588`).
pub const META_MAGIC: [u8; 4] = *b"META";
/// Size of the `META` header (`static_assert(sizeof(Header) == 0x80)`).
pub const META_SIZE: usize = 0x80;
/// `ACID` magic.
pub const ACID_MAGIC: [u8; 4] = *b"ACID";
/// `ACI0` magic.
pub const ACI0_MAGIC: [u8; 4] = *b"ACI0";
/// Offset of the `ACID` magic within the `ACID` block, Switch 1 layout (`sizeof(signature) +
/// sizeof(nca_modulus)` in `AcidHeader`).
pub const ACID_MAGIC_OFFSET_SWITCH1: usize = 0x200;
/// Offset of the `ACID` magic within the `ACID` block, Switch 2 layout (`eden-pr4588`,
/// `read_acid(0x300)`).
pub const ACID_MAGIC_OFFSET_SWITCH2: usize = 0x300;
/// Bytes of the `ACID` body, from its magic to the end of the `AcidHeader` struct:
/// `sizeof(AcidHeader) - 0x200` (`static_assert(sizeof(AcidHeader) == 0x240)`).
const ACID_BODY_SIZE: usize = 0x40;
/// Size of the `ACI0` header (`static_assert(sizeof(AciHeader) == 0x40)`).
const ACI0_HEADER_SIZE: usize = 0x40;
/// Bytes read from a file access control block: version, 3 padding, u64 permissions, 0x20 unknown
/// (`FileAccessControl`, `fac_read_size` in `eden-pr4588`).
const FAC_SIZE: usize = 0x2C;
/// Bytes read from a file access header: version, 3 padding, u64 permissions, four u32
/// (`FileAccessHeader`, `fah_read_size` in `eden-pr4588`).
const FAH_SIZE: usize = 0x1C;

/// Which `ACID` layout a file used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcidLayout {
    /// `ACID` magic at block offset 0x200.
    Switch1,
    /// `ACID` magic at block offset 0x300.
    Switch2,
}

impl AcidLayout {
    /// Offset of the `ACID` magic within the block.
    pub fn magic_offset(self) -> usize {
        match self {
            Self::Switch1 => ACID_MAGIC_OFFSET_SWITCH1,
            Self::Switch2 => ACID_MAGIC_OFFSET_SWITCH2,
        }
    }
}

/// The address space a program runs in (`ProgramAddressSpaceType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressSpaceType {
    /// 32-bit address space (value 0).
    Is32Bit,
    /// 36-bit address space (value 1).
    Is36Bit,
    /// 32-bit address space with no map region (value 2).
    Is32BitNoMap,
    /// 39-bit address space (value 3).
    Is39Bit,
}

/// The pool partition an `ACID` assigns (`PoolPartition`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolPartition {
    /// Application pool (value 0).
    Application,
    /// Applet pool (value 1).
    Applet,
    /// System pool (value 2).
    System,
    /// System non-secure pool (value 3).
    SystemNonSecure,
    /// A value Eden does not name.
    Other(u8),
}

/// File access control from the `ACID` block (`FileAccessControl`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileAccessControl {
    /// Version byte.
    pub version: u8,
    /// Filesystem permission bits the title may be granted.
    pub permissions: u64,
    /// The 0x20 bytes after the permissions, undocumented in Eden (`unknown`).
    pub unknown: [u8; 0x20],
}

/// File access header from the `ACI0` block (`FileAccessHeader`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileAccessHeader {
    /// Version byte.
    pub version: u8,
    /// Filesystem permission bits the title requests.
    pub permissions: u64,
    /// First extent's offset (Eden: `unk_offset`).
    pub unk_offset: u32,
    /// First extent's size (Eden: `unk_size`).
    pub unk_size: u32,
    /// Second extent's offset (Eden: `unk_offset_2`).
    pub unk_offset_2: u32,
    /// Second extent's size (Eden: `unk_size_2`).
    pub unk_size_2: u32,
}

/// The signed `ACID` block: what the title is allowed to ask for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Acid {
    /// Which layout the block used.
    pub layout: AcidLayout,
    /// Production flag (flags bit 0).
    pub production: bool,
    /// Unqualified approval flag (flags bit 1).
    pub unqualified_approval: bool,
    /// Pool partition (flags bits 2 to 5).
    pub pool_partition: PoolPartition,
    /// Lowest title id this `ACID` covers.
    pub title_id_min: u64,
    /// Highest title id this `ACID` covers.
    pub title_id_max: u64,
    /// File access control.
    pub file_access: FileAccessControl,
}

/// The `ACI0` block: what the title actually asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aci0 {
    /// The title (program) id.
    pub title_id: u64,
    /// File access header.
    pub file_access: FileAccessHeader,
    /// Services the title may use or host.
    pub services: Vec<ServiceAccess>,
    /// Decoded kernel capabilities.
    pub kernel_capabilities: KernelCapabilities,
}

/// A parsed `main.npdm`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Npdm {
    /// True if the program uses 64-bit instructions (flags bit 0).
    pub is_64_bit: bool,
    /// The address space type (flags bits 1 to 3).
    pub address_space: AddressSpaceType,
    /// Main thread priority.
    pub main_thread_priority: u8,
    /// Main thread core.
    pub main_thread_core: u8,
    /// System resource size in bytes.
    pub system_resource_size: u32,
    /// Process category (Eden: `process_category`).
    pub process_category: u32,
    /// Main thread stack size in bytes.
    pub main_thread_stack_size: u32,
    /// The program name, up to the first NUL.
    pub name: String,
    /// The `ACID` block.
    pub acid: Acid,
    /// The `ACI0` block.
    pub aci: Aci0,
}

/// Why an NPDM could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NpdmError {
    /// The input is shorter than the `META` header.
    #[error("NPDM is {len} bytes, shorter than the {META_SIZE}-byte META header")]
    Truncated {
        /// Length of the input.
        len: usize,
    },
    /// The magic at offset 0 is not `META`.
    #[error("NPDM magic is not META")]
    BadMagic,
    /// The address space type field holds a value that is not defined.
    #[error("NPDM address space type {0} is not defined")]
    InvalidAddressSpace(u8),
    /// A block or sub-block lies outside the block or file that must contain it.
    #[error("NPDM {section} lies outside its containing block")]
    OutOfBounds {
        /// Which section: `ACID`, `ACI0`, `FAC`, `FAH`, `SAC` or `KAC`.
        section: &'static str,
    },
    /// A block is too small for the structure it must hold.
    #[error("NPDM {section} is {size:#x} bytes, smaller than the {needed:#x} it needs")]
    TooSmall {
        /// Which section.
        section: &'static str,
        /// Declared size.
        size: usize,
        /// Minimum size.
        needed: usize,
    },
    /// Neither the Switch 1 (0x200) nor the Switch 2 (0x300) position holds an `ACID` magic.
    #[error("NPDM ACID magic found at neither offset 0x200 nor 0x300")]
    AcidNotFound,
    /// The `ACI0` block does not start with `ACI0`.
    #[error("NPDM ACI0 magic is wrong")]
    BadAciMagic,
    /// The kernel capability block is not a whole number of words.
    #[error("NPDM KAC size {0:#x} is not a multiple of 4")]
    UnalignedKac(usize),
    /// The service access block is malformed.
    #[error("NPDM service access control is malformed")]
    ServiceAccess(#[from] ServiceAccessError),
    /// The kernel capabilities are malformed.
    #[error("NPDM kernel capabilities are malformed")]
    Capabilities(#[from] CapabilityError),
}

impl Npdm {
    /// Parses a `main.npdm`.
    ///
    /// Every offset and size is checked against the block that contains it, so malformed input is
    /// an error and never a panic.
    pub fn parse(bytes: &[u8]) -> Result<Self, NpdmError> {
        let header = bytes
            .get(..META_SIZE)
            .ok_or(NpdmError::Truncated { len: bytes.len() })?;
        if read_array::<4>(header, 0) != Some(META_MAGIC) {
            return Err(NpdmError::BadMagic);
        }
        let flags = byte_at(header, 0xC);
        let address_space = match (flags >> 1) & 0b111 {
            0 => AddressSpaceType::Is32Bit,
            1 => AddressSpaceType::Is36Bit,
            2 => AddressSpaceType::Is32BitNoMap,
            3 => AddressSpaceType::Is39Bit,
            other => return Err(NpdmError::InvalidAddressSpace(other)),
        };

        let requested_block = block(bytes, u32_at(header, 0x70), u32_at(header, 0x74), "ACI0")?;
        let signed_block = block(bytes, u32_at(header, 0x78), u32_at(header, 0x7C), "ACID")?;

        Ok(Self {
            is_64_bit: flags & 1 != 0,
            address_space,
            main_thread_priority: byte_at(header, 0xE),
            main_thread_core: byte_at(header, 0xF),
            system_resource_size: u32_at(header, 0x14),
            process_category: u32_at(header, 0x18),
            main_thread_stack_size: u32_at(header, 0x1C),
            name: read_name(header),
            acid: parse_acid(signed_block)?,
            aci: parse_aci(requested_block)?,
        })
    }
}

/// The `META` name field: 16 bytes at 0x20, cut at the first NUL.
fn read_name(header: &[u8]) -> String {
    let raw = header.get(0x20..0x30).unwrap_or_default();
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(raw.get(..end).unwrap_or_default()).into_owned()
}

/// A top-level block: at or after the end of `META`, and inside the file.
fn block<'a>(
    bytes: &'a [u8],
    offset: u32,
    size: u32,
    section: &'static str,
) -> Result<&'a [u8], NpdmError> {
    let offset = usize::try_from(offset).map_err(|_| NpdmError::OutOfBounds { section })?;
    let size = usize::try_from(size).map_err(|_| NpdmError::OutOfBounds { section })?;
    if offset < META_SIZE {
        return Err(NpdmError::OutOfBounds { section });
    }
    sub_block(bytes, offset, size, section)
}

/// `size` bytes at `offset` within `parent`.
fn sub_block<'a>(
    parent: &'a [u8],
    offset: usize,
    size: usize,
    section: &'static str,
) -> Result<&'a [u8], NpdmError> {
    offset
        .checked_add(size)
        .and_then(|end| parent.get(offset..end))
        .ok_or(NpdmError::OutOfBounds { section })
}

/// Parses an `ACID` block, probing both layouts.
fn parse_acid(block: &[u8]) -> Result<Acid, NpdmError> {
    let (layout, body) = [AcidLayout::Switch1, AcidLayout::Switch2]
        .into_iter()
        .find_map(|layout| {
            let start = layout.magic_offset();
            let body = block.get(start..start.checked_add(ACID_BODY_SIZE)?)?;
            (read_array::<4>(body, 0) == Some(ACID_MAGIC)).then_some((layout, body))
        })
        .ok_or(NpdmError::AcidNotFound)?;

    // Body layout, relative to the magic (`AcidHeader` after the 0x200-byte signature area):
    // flags at 0xC, title id range at 0x10 and 0x18, FAC offset and size at 0x20.
    let acid_flags = u32_at(body, 0xC);
    let fac_offset = to_usize(u32_at(body, 0x20), "FAC")?;
    let fac_size = to_usize(u32_at(body, 0x24), "FAC")?;
    let fac = sub_block(block, fac_offset, fac_size, "FAC")?;
    if fac.len() < FAC_SIZE {
        return Err(NpdmError::TooSmall {
            section: "FAC",
            size: fac.len(),
            needed: FAC_SIZE,
        });
    }
    Ok(Acid {
        layout,
        production: acid_flags & 1 != 0,
        unqualified_approval: acid_flags & 0b10 != 0,
        pool_partition: match (acid_flags >> 2) & 0xF {
            0 => PoolPartition::Application,
            1 => PoolPartition::Applet,
            2 => PoolPartition::System,
            3 => PoolPartition::SystemNonSecure,
            other => PoolPartition::Other(u8::try_from(other).unwrap_or(u8::MAX)),
        },
        title_id_min: u64_at(body, 0x10),
        title_id_max: u64_at(body, 0x18),
        file_access: FileAccessControl {
            version: byte_at(fac, 0),
            permissions: u64_at(fac, 4),
            unknown: read_array::<0x20>(fac, 0xC).unwrap_or([0; 0x20]),
        },
    })
}

/// Parses an `ACI0` block.
fn parse_aci(block: &[u8]) -> Result<Aci0, NpdmError> {
    if block.len() < ACI0_HEADER_SIZE {
        return Err(NpdmError::TooSmall {
            section: "ACI0",
            size: block.len(),
            needed: ACI0_HEADER_SIZE,
        });
    }
    if read_array::<4>(block, 0) != Some(ACI0_MAGIC) {
        return Err(NpdmError::BadAciMagic);
    }
    // Header layout (`AciHeader`): title id at 0x10; FAH, SAC and KAC offset/size pairs at 0x20,
    // 0x28 and 0x30, all relative to the start of this block.
    let fah_offset = to_usize(u32_at(block, 0x20), "FAH")?;
    let fah_size = to_usize(u32_at(block, 0x24), "FAH")?;
    let sac_offset = to_usize(u32_at(block, 0x28), "SAC")?;
    let sac_size = to_usize(u32_at(block, 0x2C), "SAC")?;
    let kac_offset = to_usize(u32_at(block, 0x30), "KAC")?;
    let kac_size = to_usize(u32_at(block, 0x34), "KAC")?;

    let fah = sub_block(block, fah_offset, fah_size, "FAH")?;
    if fah.len() < FAH_SIZE {
        return Err(NpdmError::TooSmall {
            section: "FAH",
            size: fah.len(),
            needed: FAH_SIZE,
        });
    }
    let sac = sub_block(block, sac_offset, sac_size, "SAC")?;
    let kac = sub_block(block, kac_offset, kac_size, "KAC")?;
    if kac.len() % 4 != 0 {
        return Err(NpdmError::UnalignedKac(kac.len()));
    }
    let words: Vec<u32> = kac
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect();

    Ok(Aci0 {
        title_id: u64_at(block, 0x10),
        file_access: FileAccessHeader {
            version: byte_at(fah, 0),
            permissions: u64_at(fah, 4),
            unk_offset: u32_at(fah, 0xC),
            unk_size: u32_at(fah, 0x10),
            unk_offset_2: u32_at(fah, 0x14),
            unk_size_2: u32_at(fah, 0x18),
        },
        services: service_access::parse(sac)?,
        kernel_capabilities: KernelCapabilities::parse(&words)?,
    })
}

fn to_usize(value: u32, section: &'static str) -> Result<usize, NpdmError> {
    usize::try_from(value).map_err(|_| NpdmError::OutOfBounds { section })
}

/// `N` bytes at `offset`.
fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Option<[u8; N]> {
    let end = offset.checked_add(N)?;
    <[u8; N]>::try_from(bytes.get(offset..end)?).ok()
}

/// A byte at `offset`, or 0 past the end. Callers have already checked the structure's size.
fn byte_at(bytes: &[u8], offset: usize) -> u8 {
    bytes.get(offset).copied().unwrap_or_default()
}

/// A little-endian `u32` at `offset`, or 0 past the end. Callers have already checked the
/// structure's size, so the fallback is unreachable for valid callers.
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    read_array::<4>(bytes, offset).map_or(0, u32::from_le_bytes)
}

/// A little-endian `u64` at `offset`, or 0 past the end (see [`u32_at`]).
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    read_array::<8>(bytes, offset).map_or(0, u64::from_le_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const TITLE_ID: u64 = 0x0400_C3F0_0006_E000;

    /// A test-side NPDM writer that places bytes at the documented offsets.
    struct Fixture {
        layout: AcidLayout,
        sac: Vec<u8>,
        kac: Vec<u32>,
    }

    impl Fixture {
        fn new(layout: AcidLayout) -> Self {
            Self {
                layout,
                // fs:ro (client), sm:m (server flag), ldr:* wildcard.
                sac: [&[0x04][..], b"fs:ro", &[0x83], b"sm:m", &[0x04], b"ldr:*"].concat(),
                kac: vec![
                    // Cores 0..=5, priorities 28..=59.
                    0x0500_73B7,
                    // SVCs 0x00..=0x16 on index 0 (mask bits 0..=22).
                    0b1111 | (0x7F_FFFF << 5),
                    // Kernel version 9.1.
                    0x3FFF | (1 << 15) | (9 << 19),
                    // Handle table size 0x200.
                    0x7FFF | (0x200 << 16),
                    u32::MAX,
                ],
            }
        }

        /// META | ACI0 block | ACID block.
        fn build(&self) -> Vec<u8> {
            let mut file = vec![0u8; META_SIZE];
            file[..4].copy_from_slice(&META_MAGIC);
            // 64-bit, 39-bit address space (value 3 in bits 1..4).
            file[0xC] = 1 | (3 << 1);
            file[0xE] = 44;
            file[0xF] = 2;
            file[0x14..0x18].copy_from_slice(&0x0010_0000u32.to_le_bytes());
            file[0x18..0x1C].copy_from_slice(&1u32.to_le_bytes());
            file[0x1C..0x20].copy_from_slice(&0x0020_0000u32.to_le_bytes());
            file[0x20..0x27].copy_from_slice(b"example");

            // ACI0: header 0x40, FAH 0x1C, SAC, KAC.
            let mut aci = vec![0u8; ACI0_HEADER_SIZE];
            aci[..4].copy_from_slice(&ACI0_MAGIC);
            aci[0x10..0x18].copy_from_slice(&TITLE_ID.to_le_bytes());
            let fah_offset = aci.len();
            let mut fah = vec![0u8; FAH_SIZE];
            fah[0] = 1;
            fah[4..12].copy_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes());
            fah[0xC..0x10].copy_from_slice(&0x1Cu32.to_le_bytes());
            fah[0x10..0x14].copy_from_slice(&0x8u32.to_le_bytes());
            fah[0x14..0x18].copy_from_slice(&0x24u32.to_le_bytes());
            fah[0x18..0x1C].copy_from_slice(&0x4u32.to_le_bytes());
            aci.extend_from_slice(&fah);
            let sac_offset = aci.len();
            aci.extend_from_slice(&self.sac);
            let kac_offset = aci.len();
            for word in &self.kac {
                aci.extend_from_slice(&word.to_le_bytes());
            }
            let put = |aci: &mut Vec<u8>, at: usize, value: usize| {
                aci[at..at + 4].copy_from_slice(&u32::try_from(value).unwrap().to_le_bytes());
            };
            put(&mut aci, 0x20, fah_offset);
            put(&mut aci, 0x24, FAH_SIZE);
            put(&mut aci, 0x28, sac_offset);
            put(&mut aci, 0x2C, self.sac.len());
            put(&mut aci, 0x30, kac_offset);
            put(&mut aci, 0x34, self.kac.len() * 4);

            // ACID: signature area, body, FAC. The block starts at a nonzero file offset, so a
            // reader that forgets the block base reads the wrong bytes.
            let magic_at = self.layout.magic_offset();
            let mut acid = vec![0xEE; magic_at];
            let mut body = vec![0u8; ACID_BODY_SIZE];
            body[..4].copy_from_slice(&ACID_MAGIC);
            // Flags: production, pool partition 2 (System) in bits 2..6.
            body[0xC..0x10].copy_from_slice(&(1u32 | (2 << 2)).to_le_bytes());
            body[0x10..0x18].copy_from_slice(&TITLE_ID.to_le_bytes());
            body[0x18..0x20].copy_from_slice(&(TITLE_ID + 0xFFF).to_le_bytes());
            let fac_offset = magic_at + ACID_BODY_SIZE;
            body[0x20..0x24].copy_from_slice(&u32::try_from(fac_offset).unwrap().to_le_bytes());
            body[0x24..0x28].copy_from_slice(&u32::try_from(FAC_SIZE).unwrap().to_le_bytes());
            acid.extend_from_slice(&body);
            let mut fac = vec![0u8; FAC_SIZE];
            fac[0] = 1;
            fac[4..12].copy_from_slice(&0x8000_0000_0000_0001u64.to_le_bytes());
            fac[0xC] = 0x5A;
            acid.extend_from_slice(&fac);

            let meta_end = file.len();
            let acid_file_offset = meta_end + aci.len();
            let put_header = |file: &mut Vec<u8>, at: usize, value: usize| {
                file[at..at + 4].copy_from_slice(&u32::try_from(value).unwrap().to_le_bytes());
            };
            put_header(&mut file, 0x70, meta_end);
            put_header(&mut file, 0x74, aci.len());
            put_header(&mut file, 0x78, acid_file_offset);
            put_header(&mut file, 0x7C, acid.len());
            file.extend_from_slice(&aci);
            file.extend_from_slice(&acid);
            file
        }
    }

    fn check_common(npdm: &Npdm) {
        assert!(npdm.is_64_bit);
        assert_eq!(npdm.address_space, AddressSpaceType::Is39Bit);
        assert_eq!(npdm.main_thread_priority, 44);
        assert_eq!(npdm.main_thread_core, 2);
        assert_eq!(npdm.system_resource_size, 0x0010_0000);
        assert_eq!(npdm.process_category, 1);
        assert_eq!(npdm.main_thread_stack_size, 0x0020_0000);
        assert_eq!(npdm.name, "example");

        assert!(npdm.acid.production);
        assert!(!npdm.acid.unqualified_approval);
        assert_eq!(npdm.acid.pool_partition, PoolPartition::System);
        assert_eq!(npdm.acid.title_id_min, TITLE_ID);
        assert_eq!(npdm.acid.title_id_max, TITLE_ID + 0xFFF);
        assert_eq!(npdm.acid.file_access.version, 1);
        assert_eq!(npdm.acid.file_access.permissions, 0x8000_0000_0000_0001);
        assert_eq!(npdm.acid.file_access.unknown[0], 0x5A);

        assert_eq!(npdm.aci.title_id, TITLE_ID);
        let fah = npdm.aci.file_access;
        assert_eq!(fah.version, 1);
        assert_eq!(fah.permissions, u64::MAX);
        assert_eq!(
            (
                fah.unk_offset,
                fah.unk_size,
                fah.unk_offset_2,
                fah.unk_size_2
            ),
            (0x1C, 8, 0x24, 4)
        );

        let services: Vec<_> = npdm
            .aci
            .services
            .iter()
            .map(|s| (s.name.as_str(), s.server, s.wildcard))
            .collect();
        assert_eq!(
            services,
            [
                ("fs:ro", false, false),
                ("sm:m", true, false),
                ("ldr:", false, true)
            ]
        );

        let caps = &npdm.aci.kernel_capabilities;
        let info = caps.thread_info.as_ref().unwrap();
        assert_eq!(info.cores, 0..=5);
        assert_eq!(info.priorities, 28..=59);
        assert_eq!(caps.svc_mask.iter().count(), 23);
        assert!(caps.svc_mask.contains(0x16));
        assert!(!caps.svc_mask.contains(0x17));
        let version = caps.kernel_version.unwrap();
        assert_eq!((version.major, version.minor), (9, 1));
        assert_eq!(caps.handle_table_size, Some(0x200));
    }

    #[test]
    fn switch1_layout_places_acid_at_0x200() {
        let file = Fixture::new(AcidLayout::Switch1).build();
        // Raw offsets: META, then ACI0 at 0x80; ACID magic 0x200 into its block.
        assert_eq!(&file[..4], b"META");
        assert_eq!(&file[0x80..0x84], b"ACI0");
        let acid_at = u32::from_le_bytes(file[0x78..0x7C].try_into().unwrap()) as usize;
        assert_eq!(&file[acid_at + ACID_MAGIC_OFFSET_SWITCH1..][..4], b"ACID");
        assert_eq!(ACID_MAGIC_OFFSET_SWITCH1, 0x200);

        let npdm = Npdm::parse(&file).unwrap();
        assert_eq!(npdm.acid.layout, AcidLayout::Switch1);
        check_common(&npdm);
    }

    #[test]
    fn switch2_layout_places_acid_at_0x300() {
        let file = Fixture::new(AcidLayout::Switch2).build();
        let acid_at = u32::from_le_bytes(file[0x78..0x7C].try_into().unwrap()) as usize;
        assert_eq!(&file[acid_at + 0x300..][..4], b"ACID");
        assert_eq!(ACID_MAGIC_OFFSET_SWITCH2, 0x300);

        let npdm = Npdm::parse(&file).unwrap();
        assert_eq!(npdm.acid.layout, AcidLayout::Switch2);
        check_common(&npdm);
    }

    #[test]
    fn fac_and_fah_offsets_are_relative_to_their_block() {
        // The FAC offset stored in the ACID body is relative to the ACID block. Read straight
        // from the start of the file (Eden's bug, blocker row 5), it would land on other bytes.
        let file = Fixture::new(AcidLayout::Switch1).build();
        let acid_start = u32::from_le_bytes(file[0x78..0x7C].try_into().unwrap()) as usize;
        let fac_offset = u32::from_le_bytes(
            file[acid_start + ACID_MAGIC_OFFSET_SWITCH1 + 0x20..][..4]
                .try_into()
                .unwrap(),
        ) as usize;
        let permissions = 0x8000_0000_0000_0001u64.to_le_bytes();
        assert_ne!(
            file.get(fac_offset + 4..fac_offset + 12),
            Some(&permissions[..])
        );
        assert_eq!(&file[acid_start + fac_offset + 4..][..8], &permissions[..]);
        let npdm = Npdm::parse(&file).unwrap();
        assert_eq!(npdm.acid.file_access.permissions, 0x8000_0000_0000_0001);
    }

    #[test]
    fn rejects_malformed_headers() {
        let good = Fixture::new(AcidLayout::Switch1).build();

        assert_eq!(
            Npdm::parse(&good[..META_SIZE - 1]),
            Err(NpdmError::Truncated { len: META_SIZE - 1 })
        );
        let mut file = good.clone();
        file[0] = b'X';
        assert_eq!(Npdm::parse(&file), Err(NpdmError::BadMagic));

        let mut file = good.clone();
        file[0xC] = 4 << 1;
        assert_eq!(Npdm::parse(&file), Err(NpdmError::InvalidAddressSpace(4)));

        // ACID offset inside META, and ACID running past the end.
        let mut file = good.clone();
        file[0x78..0x7C].copy_from_slice(&0x10u32.to_le_bytes());
        assert_eq!(
            Npdm::parse(&file),
            Err(NpdmError::OutOfBounds { section: "ACID" })
        );
        let mut file = good.clone();
        file[0x7C..0x80].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            Npdm::parse(&file),
            Err(NpdmError::OutOfBounds { section: "ACID" })
        );
        let mut file = good.clone();
        file[0x74..0x78].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            Npdm::parse(&file),
            Err(NpdmError::OutOfBounds { section: "ACI0" })
        );

        // No ACID magic at either position.
        let mut file = good.clone();
        let acid_at = u32::from_le_bytes(good[0x78..0x7C].try_into().unwrap()) as usize;
        file[acid_at + 0x200] = b'X';
        assert_eq!(Npdm::parse(&file), Err(NpdmError::AcidNotFound));

        // Bad ACI0 magic, and an ACI0 block too small for its header.
        let mut file = good.clone();
        file[0x80] = b'X';
        assert_eq!(Npdm::parse(&file), Err(NpdmError::BadAciMagic));
        let mut file = good.clone();
        file[0x74..0x78].copy_from_slice(&0x10u32.to_le_bytes());
        assert!(matches!(
            Npdm::parse(&file),
            Err(NpdmError::TooSmall {
                section: "ACI0",
                ..
            })
        ));
    }

    #[test]
    fn rejects_malformed_sub_blocks() {
        let good = Fixture::new(AcidLayout::Switch1).build();
        let acid_at = u32::from_le_bytes(good[0x78..0x7C].try_into().unwrap()) as usize;
        let body = acid_at + ACID_MAGIC_OFFSET_SWITCH1;

        // FAC size too small, and FAC running past the ACID block.
        let mut file = good.clone();
        file[body + 0x24..body + 0x28].copy_from_slice(&0x10u32.to_le_bytes());
        assert!(matches!(
            Npdm::parse(&file),
            Err(NpdmError::TooSmall { section: "FAC", .. })
        ));
        let mut file = good.clone();
        file[body + 0x20..body + 0x24].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            Npdm::parse(&file),
            Err(NpdmError::OutOfBounds { section: "FAC" })
        );

        // FAH, SAC and KAC problems in ACI0 (block starts at 0x80).
        let mut file = good.clone();
        file[0x80 + 0x24..0x80 + 0x28].copy_from_slice(&0x4u32.to_le_bytes());
        assert!(matches!(
            Npdm::parse(&file),
            Err(NpdmError::TooSmall { section: "FAH", .. })
        ));
        let mut file = good.clone();
        file[0x80 + 0x28..0x80 + 0x2C].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            Npdm::parse(&file),
            Err(NpdmError::OutOfBounds { section: "SAC" })
        );
        let mut file = good.clone();
        file[0x80 + 0x34..0x80 + 0x38].copy_from_slice(&6u32.to_le_bytes());
        assert_eq!(Npdm::parse(&file), Err(NpdmError::UnalignedKac(6)));
        let mut file = good.clone();
        file[0x80 + 0x34..0x80 + 0x38].copy_from_slice(&0x1000u32.to_le_bytes());
        assert_eq!(
            Npdm::parse(&file),
            Err(NpdmError::OutOfBounds { section: "KAC" })
        );
    }

    #[test]
    fn capability_and_service_errors_propagate() {
        let mut fixture = Fixture::new(AcidLayout::Switch2);
        fixture.kac = vec![0];
        assert!(matches!(
            Npdm::parse(&fixture.build()),
            Err(NpdmError::Capabilities(
                CapabilityError::InvalidDescriptor { .. }
            ))
        ));

        let mut fixture = Fixture::new(AcidLayout::Switch2);
        fixture.sac = vec![0x07, b'a'];
        assert!(matches!(
            Npdm::parse(&fixture.build()),
            Err(NpdmError::ServiceAccess(
                ServiceAccessError::Truncated { .. }
            ))
        ));
    }

    proptest! {
        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..1024)) {
            let _ = Npdm::parse(&bytes);
        }

        #[test]
        fn arbitrary_bytes_behind_a_valid_header_never_panic(
            head in proptest::collection::vec(any::<u8>(), META_SIZE),
            tail in proptest::collection::vec(any::<u8>(), 0..1024),
        ) {
            let mut bytes = head;
            bytes[..4].copy_from_slice(&META_MAGIC);
            bytes.extend_from_slice(&tail);
            let _ = Npdm::parse(&bytes);
        }

        #[test]
        fn mutated_valid_files_never_panic(
            switch2 in any::<bool>(),
            index in 0usize..2048,
            value in any::<u8>(),
        ) {
            let layout = if switch2 { AcidLayout::Switch2 } else { AcidLayout::Switch1 };
            let mut file = Fixture::new(layout).build();
            let at = index % file.len();
            file[at] = value;
            let _ = Npdm::parse(&file);
        }
    }
}

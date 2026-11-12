//! VMSAv8-64 translation table descriptors for the 4 KiB granule: the bits [`super::Stage1Tables`]
//! writes and [`super::walk`] reads, and the table geometry of a 39-bit address space.
//!
//! Source: Arm Architecture Reference Manual for A-profile architecture (Arm DDI 0487), the
//! AArch64 Virtual Memory System Architecture chapter: the VMSAv8-64 descriptor formats (table,
//! block and page descriptors; lower and upper attributes), the data access permissions
//! `AP[2:1]`, the execute-never controls `UXN`/`PXN`, and the hierarchical table attributes.
//! Sections are cited by name, not number, because the numbering moves between issues of the
//! manual. The manual is not one of the pinned sources in `docs/references.md`.
//!
//! Every bit below that `tools/hvprobe/src/probe.rs` also wrote (valid and type bits, `AF`,
//! `SH`, `AP[2:1]`, `PXN`, `UXN`, the table index widths with `T0SZ` = 25) was walked by a real
//! vCPU there. `nG` and the hierarchical table bits were not.

use crate::translation::PAGE_SIZE;
use crate::{Fault, GuestMemory, Perms};

/// log2 of [`PAGE_SIZE`]: the bits of a virtual address below the level-3 index.
pub(crate) const PAGE_SHIFT: u32 = PAGE_SIZE.trailing_zeros();
/// Bits of a virtual address that index one table: a 4 KiB table holds 512 eight-byte
/// descriptors.
pub(crate) const INDEX_BITS: u32 = 9;
/// Descriptors per table.
pub(crate) const ENTRIES: u64 = 1 << INDEX_BITS;
/// Bytes per descriptor: descriptors are 64-bit words.
pub(crate) const DESCRIPTOR_LEN: usize = size_of::<u64>();
/// [`DESCRIPTOR_LEN`] as an address offset.
const DESCRIPTOR_SIZE: u64 = DESCRIPTOR_LEN as u64;
/// The level the walk starts at: with a 4 KiB granule and `T0SZ` = 25 (39 bits) the first lookup
/// is at level 1 (hvprobe: "walk starts at level 1").
pub(crate) const START_LEVEL: u8 = 1;
/// The level that holds page descriptors.
pub(crate) const LEAF_LEVEL: u8 = 3;

/// Bit 0: the descriptor is valid. Clear means invalid, whatever the other bits hold.
pub(crate) const VALID: u64 = 1 << 0;
/// Bit 1: with [`VALID`], set means a table descriptor (levels 0 to 2) or a page descriptor
/// (level 3); clear means a block descriptor (levels 1 and 2) or, at level 3, a reserved
/// encoding that translates like an invalid one.
pub(crate) const TABLE_OR_PAGE: u64 = 1 << 1;
/// `AP[1]` (bit 6): EL0 may access the page.
pub(crate) const AP_EL0: u64 = 1 << 6;
/// `AP[2]` (bit 7): the page is read-only at every exception level.
pub(crate) const AP_READ_ONLY: u64 = 1 << 7;
/// `SH[1:0]` (bits 9:8) = 0b11: inner shareable.
pub(crate) const SH_INNER: u64 = 0b11 << 8;
/// `AF` (bit 10): the access flag. A block or page descriptor with it clear raises an Access
/// flag fault instead of translating.
pub(crate) const AF: u64 = 1 << 10;
/// `nG` (bit 11): not global; the TLB entry is tagged with the current ASID.
pub(crate) const NOT_GLOBAL: u64 = 1 << 11;
/// Bits 47:12: the output address of a page, or the address of the next-level table. A block's
/// output address uses the bits of this mask from the block size up.
pub(crate) const OUTPUT_ADDRESS_MASK: u64 = 0x0000_ffff_ffff_f000;
/// `PXN` (bit 53): EL1 may not execute from the page.
pub(crate) const PXN: u64 = 1 << 53;
/// `UXN` (bit 54): EL0 may not execute from the page.
pub(crate) const UXN: u64 = 1 << 54;
/// Bits 63:59 of a table descriptor: `NSTable`, `APTable[1:0]`, `UXNTable`, `PXNTable`. They
/// restrict every lookup through the table.
pub(crate) const TABLE_ATTRIBUTES: u64 = 0b1_1111 << 59;

/// The attributes every page descriptor [`super::Stage1Tables`] writes carries, before
/// permissions: a valid page with the access flag set, inner shareable, `AttrIndx` 0 (the
/// backend programs `MAIR_EL1` attribute 0; hvprobe used 0xff, normal write-back memory),
/// non-global, and never executable at EL1.
pub(crate) const PAGE_BASE: u64 = VALID | TABLE_OR_PAGE | SH_INNER | AF | NOT_GLOBAL | PXN;

/// Which bit of a virtual address a level's index starts at: 30, 21 and 12 for levels 1 to 3.
pub(crate) fn shift(level: u8) -> u32 {
    // Callers pass START_LEVEL..=LEAF_LEVEL, so the subtraction can't wrap.
    PAGE_SHIFT + INDEX_BITS * u32::from(LEAF_LEVEL.saturating_sub(level))
}

/// The descriptor index `va` selects in a table at `level`.
pub(crate) fn index(va: u64, level: u8) -> u64 {
    (va >> shift(level)) & (ENTRIES - 1)
}

/// The page descriptor for `pa` with `perms`. `pa` is page aligned and below
/// [`super::OUTPUT_ADDRESS_END`]; callers check both.
pub(crate) const fn page(pa: u64, perms: Perms) -> u64 {
    // AP[2:1]: 0b00 EL1 read/write, EL0 none; 0b01 both read/write; 0b11 both read-only.
    let ap = match perms {
        Perms::None | Perms::Execute => 0,
        Perms::Read | Perms::ReadExecute => AP_EL0 | AP_READ_ONLY,
        Perms::ReadWrite | Perms::ReadWriteExecute => AP_EL0,
    };
    let uxn = if perms.executable() { 0 } else { UXN };
    (pa & OUTPUT_ADDRESS_MASK) | PAGE_BASE | ap | uxn
}

/// The EL0 permissions a block or page descriptor grants.
pub(crate) const fn perms(descriptor: u64) -> Perms {
    let el0 = descriptor & AP_EL0 != 0;
    let read_only = descriptor & AP_READ_ONLY != 0;
    let execute = descriptor & UXN == 0;
    match (el0, read_only, execute) {
        (false, _, false) => Perms::None,
        (false, _, true) => Perms::Execute,
        (true, true, false) => Perms::Read,
        (true, false, false) => Perms::ReadWrite,
        (true, true, true) => Perms::ReadExecute,
        (true, false, true) => Perms::ReadWriteExecute,
    }
}

/// The table descriptor pointing at the table page `pa`, with no hierarchical restrictions.
pub(crate) const fn table(pa: u64) -> u64 {
    (pa & OUTPUT_ADDRESS_MASK) | VALID | TABLE_OR_PAGE
}

/// Whether a valid descriptor is a table (levels 1 and 2) or page (level 3) descriptor.
pub(crate) const fn is_table_or_page(descriptor: u64) -> bool {
    descriptor & (VALID | TABLE_OR_PAGE) == VALID | TABLE_OR_PAGE
}

/// The guest-physical address of descriptor `index` in the table at `table`.
fn entry_address(table: u64, index: u64) -> Result<u64, Fault> {
    index
        .checked_mul(DESCRIPTOR_SIZE)
        .and_then(|offset| table.checked_add(offset))
        .ok_or(Fault { address: table })
}

/// Reads descriptor `index` of the table at guest-physical `table`. Descriptors are
/// little-endian (`SCTLR_EL1.EE` = 0, as hvprobe ran).
pub(crate) fn read<M: GuestMemory + ?Sized>(
    memory: &M,
    table: u64,
    index: u64,
) -> Result<u64, Fault> {
    let mut bytes = [0; DESCRIPTOR_LEN];
    memory.read(entry_address(table, index)?, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// Writes descriptor `index` of the table at guest-physical `table`.
pub(crate) fn write<M: GuestMemory + ?Sized>(
    memory: &M,
    table: u64,
    index: u64,
    descriptor: u64,
) -> Result<(), Fault> {
    memory.write(entry_address(table, index)?, &descriptor.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values below are written from the Arm DDI 0487 bit positions (and, where noted,
    // the constants `tools/hvprobe/src/probe.rs` ran on hardware), not from the constants above,
    // so a wrong constant fails here.

    #[test]
    fn bit_positions_match_the_architecture() {
        assert_eq!(VALID, 0x1);
        assert_eq!(TABLE_OR_PAGE, 0x2);
        assert_eq!(AP_EL0, 0x40); // hvprobe AP_ALL_RW = 1 << 6
        assert_eq!(AP_EL0 | AP_READ_ONLY, 0xc0); // hvprobe AP_ALL_RO = 3 << 6
        assert_eq!(SH_INNER, 0x300); // hvprobe PAGE_SH_INNER = 3 << 8
        assert_eq!(AF, 0x400); // hvprobe PAGE_AF = 1 << 10
        assert_eq!(NOT_GLOBAL, 0x800);
        assert_eq!(PXN, 0x0020_0000_0000_0000); // hvprobe PXN = 1 << 53
        assert_eq!(UXN, 0x0040_0000_0000_0000); // hvprobe UXN = 1 << 54
        assert_eq!(TABLE_ATTRIBUTES, 0xf800_0000_0000_0000);
        assert_eq!(OUTPUT_ADDRESS_MASK.trailing_zeros(), 12);
        assert_eq!(64 - OUTPUT_ADDRESS_MASK.leading_zeros(), 48);
    }

    #[test]
    fn level_shifts_match_a_39_bit_4k_walk() {
        // hvprobe indexes level 1 with va >> 30, level 2 with va >> 21, level 3 with va >> 12,
        // each masked to 0x1ff.
        assert_eq!([shift(1), shift(2), shift(3)], [30, 21, 12], "level shifts");
        assert_eq!(ENTRIES, 0x200);
        let va = (5 << 30) | (7 << 21) | (9 << 12) | 0x123;
        assert_eq!([index(va, 1), index(va, 2), index(va, 3)], [5, 7, 9]);
        // Level 1's index ends at bit 38: a 39-bit space.
        assert_eq!(shift(START_LEVEL) + INDEX_BITS, 39);
    }

    #[test]
    fn page_descriptors_decode_to_the_perms_they_encode() {
        for perms in [
            Perms::None,
            Perms::Read,
            Perms::ReadWrite,
            Perms::Execute,
            Perms::ReadExecute,
            Perms::ReadWriteExecute,
        ] {
            assert_eq!(super::perms(page(0x1234_5000, perms)), perms);
        }
    }

    #[test]
    fn table_descriptor_is_address_and_type() {
        assert_eq!(table(0x8010_1000), 0x8010_1003);
        assert!(is_table_or_page(table(0x8010_1000)));
        assert!(!is_table_or_page(0x8010_1001));
        assert!(!is_table_or_page(0x8010_1002));
    }
}

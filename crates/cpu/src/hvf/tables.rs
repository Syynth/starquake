//! Stage-1 translation tables for the bring-up address space.
//!
//! 4 KiB granule, 39-bit virtual addresses (`TCR_EL1.T0SZ` = 25), so the walk starts at level 1
//! and every mapping is a level-3 page. Ported from `build_page_tables` in
//! `tools/hvprobe/src/probe.rs`; descriptor fields are from the Arm Architecture Reference Manual
//! (DDI 0487), "VMSAv8-64 translation table format descriptors".

use super::LayoutError;

/// Translation granule and page size.
pub(super) const PAGE: u64 = 0x1000;
/// Entries in one table.
const ENTRIES: usize = 512;
/// Highest virtual address the tables can map, plus one (`T0SZ` = 25).
pub(super) const VA_LIMIT: u64 = 1 << 39;

/// One 4 KiB translation table.
pub(super) type Table = [u64; ENTRIES];

/// Descriptor type bits for a table (levels 1-2) or a page (level 3).
const TABLE_OR_PAGE: u64 = 0b11;
/// Access flag: set, so the first access doesn't fault.
const AF: u64 = 1 << 10;
/// Inner shareable.
const SH_INNER: u64 = 3 << 8;
/// `AP[2:1]` = 0b00: read/write at EL1, no access at EL0.
pub(super) const AP_EL1_RW: u64 = 0;
/// `AP[2:1]` = 0b01: read/write at EL1 and EL0.
pub(super) const AP_ALL_RW: u64 = 1 << 6;
/// `AP[2:1]` = 0b10: read-only at EL1, no access at EL0.
pub(super) const AP_EL1_RO: u64 = 2 << 6;
/// Privileged (EL1) execute-never.
pub(super) const PXN: u64 = 1 << 53;
/// Unprivileged (EL0) execute-never.
pub(super) const UXN: u64 = 1 << 54;
/// Output address bits 47:12.
const ADDR_MASK: u64 = 0x0000_ffff_ffff_f000;

/// A run of pages to map: `len` bytes of virtual addresses from `va` onto physical `pa`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Region {
    pub va: u64,
    pub pa: u64,
    pub len: u64,
    /// Access permission and execute-never bits; the memory type is always `MAIR_EL1` index 0.
    pub attrs: u64,
}

/// Builds tables mapping `regions`. Table `i` of the result must be placed at physical address
/// `tables_pa + i * PAGE`; table 0 is the root (`TTBR0_EL1`).
pub(super) fn build(regions: &[Region], tables_pa: u64) -> Result<Vec<Table>, LayoutError> {
    let mut tables = vec![[0; ENTRIES]];
    for region in regions {
        for misaligned in [region.va, region.pa, region.len] {
            if misaligned % PAGE != 0 {
                return Err(LayoutError::Misaligned {
                    value: misaligned,
                    alignment: PAGE,
                });
            }
        }
        let va_end = region
            .va
            .checked_add(region.len)
            .ok_or(LayoutError::OutOfRange)?;
        let pa_end = region
            .pa
            .checked_add(region.len)
            .ok_or(LayoutError::OutOfRange)?;
        if va_end > VA_LIMIT || pa_end > ADDR_MASK {
            return Err(LayoutError::OutOfRange);
        }
        for page in 0..region.len / PAGE {
            let offset = page * PAGE;
            let va = region.va + offset;
            let l2 = child(&mut tables, 0, index(va, 30), tables_pa)?;
            let l3 = child(&mut tables, l2, index(va, 21), tables_pa)?;
            let entry = &mut tables[l3][index(va, 12)];
            if *entry != 0 {
                return Err(LayoutError::Overlap { va });
            }
            *entry = (region.pa + offset) | TABLE_OR_PAGE | AF | SH_INNER | region.attrs;
        }
    }
    Ok(tables)
}

/// The table index for `va` at the level that translates bits `shift + 8..=shift`.
fn index(va: u64, shift: u32) -> usize {
    // Nine bits always fit; going through bytes avoids a truncating cast.
    let [lo, hi, ..] = ((va >> shift) & 0x1ff).to_le_bytes();
    usize::from(u16::from_le_bytes([lo, hi]))
}

/// The table that entry `slot` of table `parent` points to, created if it doesn't exist yet.
fn child(
    tables: &mut Vec<Table>,
    parent: usize,
    slot: usize,
    tables_pa: u64,
) -> Result<usize, LayoutError> {
    let existing = tables[parent][slot] & ADDR_MASK;
    if existing != 0 {
        let offset = existing
            .checked_sub(tables_pa)
            .ok_or(LayoutError::OutOfRange)?;
        return usize::try_from(offset / PAGE).map_err(|_| LayoutError::OutOfRange);
    }
    let next = tables.len();
    let address = u64::try_from(next)
        .ok()
        .and_then(|n| n.checked_mul(PAGE))
        .and_then(|offset| tables_pa.checked_add(offset))
        .filter(|address| address & !ADDR_MASK == 0)
        .ok_or(LayoutError::OutOfRange)?;
    tables.push([0; ENTRIES]);
    tables[parent][slot] = address | TABLE_OR_PAGE;
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLES_PA: u64 = 0xf_0000_2000;

    /// Walks `tables` like the MMU: the leaf descriptor for `va`, if it's mapped.
    fn walk(tables: &[Table], va: u64) -> Option<u64> {
        let mut table = 0;
        for shift in [30, 21] {
            let desc = tables[table][index(va, shift)];
            if desc & TABLE_OR_PAGE != TABLE_OR_PAGE {
                return None;
            }
            table = usize::try_from(((desc & ADDR_MASK) - TABLES_PA) / PAGE).unwrap();
        }
        let leaf = tables[table][index(va, 12)];
        (leaf & TABLE_OR_PAGE == TABLE_OR_PAGE).then_some(leaf)
    }

    fn ram() -> Region {
        Region {
            va: 0x8000_0000,
            pa: 0x8000_0000,
            len: 0x40_0000, // 4 MiB: crosses a level-2 boundary
            attrs: AP_ALL_RW | PXN,
        }
    }

    #[test]
    fn maps_each_page_with_its_attributes() {
        let shim = Region {
            va: 0xf_0000_0000,
            pa: 0xf_0000_0000,
            len: PAGE,
            attrs: AP_EL1_RO | UXN,
        };
        let tables = build(&[ram(), shim], TABLES_PA).unwrap();
        // Root, one level-2 table per GiB used (2), one level-3 table per 2 MiB used (3).
        assert_eq!(tables.len(), 6);
        for va in [0x8000_0000, 0x801f_f000, 0x8020_0000, 0x803f_f000] {
            let leaf = walk(&tables, va).unwrap();
            assert_eq!(leaf & ADDR_MASK, va, "{va:#x} is identity mapped");
            assert_eq!(
                leaf & !ADDR_MASK,
                TABLE_OR_PAGE | AF | SH_INNER | AP_ALL_RW | PXN
            );
        }
        let leaf = walk(&tables, 0xf_0000_0000).unwrap();
        assert_eq!(
            leaf & !ADDR_MASK,
            TABLE_OR_PAGE | AF | SH_INNER | AP_EL1_RO | UXN
        );
        for unmapped in [0, 0x7fff_f000, 0x8040_0000, 0xf_0000_1000] {
            assert_eq!(walk(&tables, unmapped), None, "{unmapped:#x}");
        }
    }

    #[test]
    fn maps_virtual_pages_onto_other_physical_pages() {
        let remap = Region {
            va: 0x1000,
            pa: 0x8000_3000,
            len: PAGE,
            attrs: AP_EL1_RW,
        };
        let tables = build(&[remap], TABLES_PA).unwrap();
        assert_eq!(walk(&tables, 0x1000).unwrap() & ADDR_MASK, 0x8000_3000);
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(
            build(&[ram()], TABLES_PA).unwrap(),
            build(&[ram()], TABLES_PA).unwrap()
        );
    }

    #[test]
    fn rejects_bad_regions() {
        let misaligned = Region {
            va: 0x8000_0800,
            ..ram()
        };
        assert_eq!(
            build(&[misaligned], TABLES_PA),
            Err(LayoutError::Misaligned {
                value: 0x8000_0800,
                alignment: PAGE
            })
        );
        let too_high = Region {
            va: VA_LIMIT - PAGE,
            len: 2 * PAGE,
            ..ram()
        };
        assert_eq!(build(&[too_high], TABLES_PA), Err(LayoutError::OutOfRange));
        let wraps = Region {
            va: u64::MAX - (PAGE - 1),
            ..ram()
        };
        assert_eq!(build(&[wraps], TABLES_PA), Err(LayoutError::OutOfRange));
        assert_eq!(
            build(&[ram(), ram()], TABLES_PA),
            Err(LayoutError::Overlap { va: 0x8000_0000 })
        );
    }
}

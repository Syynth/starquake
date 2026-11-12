//! A software stage-1 walker: resolves a guest virtual address to a guest-physical address and
//! EL0 permissions by reading the descriptors in memory, the way the MMU would.
//!
//! It reads only the table memory and the root address, never [`super::Stage1Tables`]'s own
//! bookkeeping, so it can check tables the builder wrote, tables built by hand, and (in a
//! hardware test) tables a real vCPU walked.
//!
//! Bit layouts and walk geometry are in [`super::descriptor`], which cites Arm DDI 0487. The
//! walker assumes the translation regime the native backends set up, as hvprobe did:
//! `TCR_EL1.T0SZ` = 25 with a 4 KiB granule, `TCR_EL1.HA` = 0 (no hardware access-flag
//! management, which is what makes [`WalkError::AccessFlag`] a fault; with `HA` = 1 the
//! hardware would set `AF` instead of faulting), `SCTLR_EL1.EE` = 0 (little-endian descriptors)
//! and `SCTLR_EL1.WXN` = 0 (a writable page may be executable).
//!
//! Divergences from the hardware walk, each pinned by a test:
//!
//! - A table descriptor with any hierarchical attribute bit (63:59) set is an error,
//!   [`WalkError::TableAttributes`]. Hardware applies them as extra restrictions (unless
//!   `TCR_EL1.HPD0` is set). [`super::Stage1Tables`] never sets them, so the walker refuses them
//!   rather than modelling them.
//! - Output address bits 51:48 (only meaningful with 52-bit output addresses) and the `RES0`
//!   bits between a block's size and bit 12 are ignored, as are `AttrIndx`, `SH`, `nG`, the
//!   contiguous hint and `PXN` (which concerns EL1, not EL0).

use super::descriptor::{
    self, AF, LEAF_LEVEL, OUTPUT_ADDRESS_MASK, START_LEVEL, TABLE_ATTRIBUTES, VALID,
};
use super::{TableRoot, VA_SPACE_END};
use crate::{Access, Fault, GuestMemory, Perms};

/// Where a virtual address translates to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved {
    /// The guest-physical address, including the offset within the page or block.
    pub pa: u64,
    /// What EL0 may do there.
    pub perms: Perms,
    /// The level of the descriptor that translated it: 3 for a 4 KiB page, 2 for a 2 MiB block,
    /// 1 for a 1 GiB block.
    pub level: u8,
}

/// Why a walk did not produce a translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WalkError {
    /// The address is at or above [`VA_SPACE_END`]. Hardware would select `TTBR1_EL1` only for
    /// addresses whose top bits are all ones (as many as `TCR_EL1.T1SZ` says); an address
    /// between 2^39 and that range faults instead. Assumption, not backed by a reference: the
    /// walker translates only the `TTBR0_EL1` half, because no backend exists yet that needs
    /// the other (the EL1 shim might).
    #[error("{va:#x} is outside the 39-bit address space")]
    OutOfRange {
        /// The address.
        va: u64,
    },
    /// An invalid descriptor (a Translation fault), or the reserved level-3 encoding `0b01`,
    /// which translates the same way.
    #[error("{va:#x} is not mapped (level {level})")]
    Unmapped {
        /// The address.
        va: u64,
        /// The level of the invalid descriptor.
        level: u8,
    },
    /// A block or page descriptor with the access flag clear (an Access flag fault).
    #[error("{va:#x} has the access flag clear (level {level})")]
    AccessFlag {
        /// The address.
        va: u64,
        /// The level of the descriptor.
        level: u8,
    },
    /// A table descriptor with hierarchical attribute bits set, which the walker refuses.
    #[error(
        "table descriptor {descriptor:#x} for {va:#x} at level {level} has hierarchical attributes"
    )]
    TableAttributes {
        /// The address.
        va: u64,
        /// The level of the table descriptor.
        level: u8,
        /// The raw descriptor.
        descriptor: u64,
    },
    /// The address translates, but not for this access (a Permission fault).
    #[error("EL0 {access:?} of {va:#x} is not permitted ({perms:?})")]
    PermissionDenied {
        /// The address.
        va: u64,
        /// The access that was refused.
        access: Access,
        /// The permissions the page has.
        perms: Perms,
    },
    /// Reading a descriptor faulted: a table address outside the memory.
    #[error(transparent)]
    Memory(#[from] Fault),
}

/// Resolves `va` through the tables rooted at `root` in `memory`, whose addresses are
/// guest-physical.
///
/// Reads at most one descriptor per level, levels 1 to 3.
pub fn walk<M: GuestMemory + ?Sized>(
    memory: &M,
    root: TableRoot,
    va: u64,
) -> Result<Resolved, WalkError> {
    if va >= VA_SPACE_END {
        return Err(WalkError::OutOfRange { va });
    }
    let mut table = root.pa();
    let mut level = START_LEVEL;
    loop {
        let entry = descriptor::read(memory, table, descriptor::index(va, level))?;
        if entry & VALID == 0 {
            return Err(WalkError::Unmapped { va, level });
        }
        let table_or_page = descriptor::is_table_or_page(entry);
        if level < LEAF_LEVEL && table_or_page {
            if entry & TABLE_ATTRIBUTES != 0 {
                return Err(WalkError::TableAttributes {
                    va,
                    level,
                    descriptor: entry,
                });
            }
            table = entry & OUTPUT_ADDRESS_MASK;
            level += 1;
            continue;
        }
        if level == LEAF_LEVEL && !table_or_page {
            return Err(WalkError::Unmapped { va, level });
        }
        // A block (levels 1 and 2) or a page (level 3).
        if entry & AF == 0 {
            return Err(WalkError::AccessFlag { va, level });
        }
        let offset_mask = (1u64 << descriptor::shift(level)) - 1;
        return Ok(Resolved {
            pa: (entry & OUTPUT_ADDRESS_MASK & !offset_mask) | (va & offset_mask),
            perms: descriptor::perms(entry),
            level,
        });
    }
}

/// Resolves `va` like [`walk`] and checks that EL0 may make `access` there.
pub fn translate<M: GuestMemory + ?Sized>(
    memory: &M,
    root: TableRoot,
    va: u64,
    access: Access,
) -> Result<u64, WalkError> {
    let resolved = walk(memory, root, va)?;
    if resolved.perms.allows(access) {
        Ok(resolved.pa)
    } else {
        Err(WalkError::PermissionDenied {
            va,
            access,
            perms: resolved.perms,
        })
    }
}

#[cfg(test)]
mod tests {
    //! Tables built by hand, without [`super::super::Stage1Tables`], so the walker's index and
    //! bit decoding are checked independently of the builder.

    use super::*;
    use crate::FlatMemory;
    use crate::translation::PAGE_SIZE;

    // The layout `tools/hvprobe/src/probe.rs` (`build_page_tables`) gave a real vCPU, with its
    // constants as it wrote them. These are hvprobe's values, not this crate's.
    const GUEST_BASE: u64 = 0x8000_0000;
    const PT_BASE: u64 = 0x10_0000;
    const DATA: u64 = 0x8_0000;
    const MARKERS: u64 = 0xC_0000;
    const REMAP_VA: u64 = GUEST_BASE + 0x20_0000;
    const PAGE_AF: u64 = 1 << 10;
    const PAGE_SH_INNER: u64 = 3 << 8;
    const AP_EL1_RW: u64 = 0;
    const AP_ALL_RW: u64 = 1 << 6;
    const AP_ALL_RO: u64 = 3 << 6;
    const HV_PXN: u64 = 1 << 53;
    const HV_UXN: u64 = 1 << 54;

    fn hvprobe_tables() -> (FlatMemory, TableRoot) {
        let memory = FlatMemory::new(GUEST_BASE + PT_BASE, 0x4000);
        let put = |at: u64, value: u64| memory.write(at, &value.to_le_bytes()).unwrap();
        let (l1, l2, l3a, l3b) = (
            GUEST_BASE + PT_BASE,
            GUEST_BASE + PT_BASE + 0x1000,
            GUEST_BASE + PT_BASE + 0x2000,
            GUEST_BASE + PT_BASE + 0x3000,
        );
        let l3_entry = |table: u64, va: u64, pa_off: u64, attrs: u64| {
            put(
                table + ((va >> 12) & 0x1ff) * 8,
                (GUEST_BASE + pa_off) | 0b11 | PAGE_AF | PAGE_SH_INNER | attrs,
            );
        };
        put(l1 + ((GUEST_BASE >> 30) & 0x1ff) * 8, l2 | 0b11);
        put(l2 + ((GUEST_BASE >> 21) & 0x1ff) * 8, l3a | 0b11);
        put(l2 + ((REMAP_VA >> 21) & 0x1ff) * 8, l3b | 0b11);
        l3_entry(l3a, GUEST_BASE, 0, AP_ALL_RO);
        l3_entry(l3a, GUEST_BASE + DATA, DATA, AP_ALL_RW | HV_PXN | HV_UXN);
        l3_entry(
            l3a,
            GUEST_BASE + MARKERS,
            MARKERS,
            AP_EL1_RW | HV_PXN | HV_UXN,
        );
        l3_entry(
            l3b,
            REMAP_VA + 0x1000,
            MARKERS + 0x3000,
            AP_EL1_RW | HV_PXN | HV_UXN,
        );
        l3_entry(
            l3b,
            REMAP_VA + 0x2000,
            MARKERS + 0x1000,
            AP_EL1_RW | HV_PXN | HV_UXN,
        );
        (memory, TableRoot::new(l1).unwrap())
    }

    #[test]
    fn resolves_the_tables_hvprobe_ran_on_hardware() {
        let (memory, root) = hvprobe_tables();
        let cases = [
            (GUEST_BASE + 0x10, GUEST_BASE + 0x10, Perms::ReadExecute),
            (
                GUEST_BASE + DATA + 0x8,
                GUEST_BASE + DATA + 0x8,
                Perms::ReadWrite,
            ),
            (GUEST_BASE + MARKERS, GUEST_BASE + MARKERS, Perms::None),
            // hvprobe's remap: two 4 KiB pages inside one 16 KiB host page, in reverse order.
            (
                REMAP_VA + 0x1008,
                GUEST_BASE + MARKERS + 0x3008,
                Perms::None,
            ),
            (
                REMAP_VA + 0x2ff8,
                GUEST_BASE + MARKERS + 0x1ff8,
                Perms::None,
            ),
        ];
        for (va, pa, perms) in cases {
            assert_eq!(
                walk(&memory, root, va),
                Ok(Resolved {
                    pa,
                    perms,
                    level: 3
                }),
                "{va:#x}"
            );
        }
        // Next to the remapped pages, and in a level-2 slot with no table.
        assert_eq!(
            walk(&memory, root, REMAP_VA),
            Err(WalkError::Unmapped {
                va: REMAP_VA,
                level: 3
            })
        );
        let no_l3 = REMAP_VA + 0x20_0000;
        assert_eq!(
            walk(&memory, root, no_l3),
            Err(WalkError::Unmapped {
                va: no_l3,
                level: 2
            })
        );
        assert_eq!(
            walk(&memory, root, 0),
            Err(WalkError::Unmapped { va: 0, level: 1 })
        );
    }

    #[test]
    fn permission_denied_is_distinct_from_unmapped() {
        let (memory, root) = hvprobe_tables();
        let code = GUEST_BASE + 0x40;
        assert_eq!(translate(&memory, root, code, Access::Read), Ok(code));
        assert_eq!(translate(&memory, root, code, Access::Execute), Ok(code));
        assert_eq!(
            translate(&memory, root, code, Access::Write),
            Err(WalkError::PermissionDenied {
                va: code,
                access: Access::Write,
                perms: Perms::ReadExecute
            })
        );
        let unmapped = GUEST_BASE + PAGE_SIZE;
        assert_eq!(
            translate(&memory, root, unmapped, Access::Read),
            Err(WalkError::Unmapped {
                va: unmapped,
                level: 3
            })
        );
        assert_eq!(
            WalkError::PermissionDenied {
                va: code,
                access: Access::Write,
                perms: Perms::ReadExecute
            }
            .to_string(),
            "EL0 Write of 0x80000040 is not permitted (ReadExecute)"
        );
    }

    /// One level-1 table at `ROOT` and one level-2 table after it, in a fresh memory.
    const ROOT: u64 = 0x4000_0000;
    const L2: u64 = ROOT + 0x1000;
    const L3: u64 = ROOT + 0x2000;

    fn blank() -> (FlatMemory, TableRoot) {
        (FlatMemory::new(ROOT, 0x3000), TableRoot::new(ROOT).unwrap())
    }

    fn put(memory: &FlatMemory, at: u64, value: u64) {
        memory.write(at, &value.to_le_bytes()).unwrap();
    }

    #[test]
    fn resolves_level_1_and_level_2_blocks() {
        let (memory, root) = blank();
        // Level-1 index 2: a 1 GiB block at PA 0x1_c000_0000, read-write, never executable.
        put(&memory, ROOT + 2 * 8, 0x0040_0001_c000_0441);
        // Level-1 index 3 -> level-2 table; its index 4 is a 2 MiB block at PA 0x9_0060_0000,
        // read-execute.
        put(&memory, ROOT + 3 * 8, L2 | 0b11);
        put(&memory, L2 + 4 * 8, 0x0000_0009_0060_04c1);

        let in_gib = (2 << 30) | 0x1234_5678;
        assert_eq!(
            walk(&memory, root, in_gib),
            Ok(Resolved {
                pa: 0x1_c000_0000 | 0x1234_5678,
                perms: Perms::ReadWrite,
                level: 1
            })
        );
        let in_2mib = (3 << 30) | (4 << 21) | 0x1_2345;
        assert_eq!(
            walk(&memory, root, in_2mib),
            Ok(Resolved {
                pa: 0x9_0060_0000 | 0x1_2345,
                perms: Perms::ReadExecute,
                level: 2
            })
        );
    }

    #[test]
    fn reserved_level_3_encoding_and_clear_access_flag_fault() {
        let (memory, root) = blank();
        put(&memory, ROOT, L2 | 0b11);
        put(&memory, L2, L3 | 0b11);
        // Index 1: bits[1:0] = 0b01, a block encoding, which is reserved at level 3.
        put(&memory, L3 + 8, 0x0040_0000_1234_5441);
        // Index 2: a page with AF clear.
        put(&memory, L3 + 2 * 8, 0x0040_0000_1234_5043);
        // Index 3: the same page with AF set.
        put(&memory, L3 + 3 * 8, 0x0040_0000_1234_5443);
        let page = |n: u64| n * PAGE_SIZE;
        assert_eq!(
            walk(&memory, root, page(1)),
            Err(WalkError::Unmapped {
                va: page(1),
                level: 3
            })
        );
        assert_eq!(
            walk(&memory, root, page(2)),
            Err(WalkError::AccessFlag {
                va: page(2),
                level: 3
            })
        );
        assert_eq!(
            walk(&memory, root, page(3)),
            Ok(Resolved {
                pa: 0x1234_5000,
                perms: Perms::ReadWrite,
                level: 3
            })
        );
        // An invalid descriptor with every other bit set is still invalid.
        put(&memory, L3 + 4 * 8, !VALID);
        assert_eq!(
            walk(&memory, root, page(4)),
            Err(WalkError::Unmapped {
                va: page(4),
                level: 3
            })
        );
    }

    #[test]
    fn refuses_hierarchical_table_attributes() {
        // Divergence: hardware would apply these bits as restrictions; the walker refuses them.
        for bit in 59..64 {
            let (memory, root) = blank();
            let descriptor = L2 | 0b11 | (1 << bit);
            put(&memory, ROOT, descriptor);
            put(&memory, L2, L3 | 0b11);
            assert_eq!(
                walk(&memory, root, 0x10),
                Err(WalkError::TableAttributes {
                    va: 0x10,
                    level: 1,
                    descriptor
                }),
                "bit {bit}"
            );
        }
    }

    /// Output address bits 51:48, which a 48-bit output address ignores.
    const OUTPUT_BITS_51_48: u64 = 0b1111 << 48;
    /// The contiguous hint (bit 52).
    const CONTIGUOUS: u64 = 1 << 52;
    /// `AttrIndx` (bits 4:2), all set.
    const ATTR_INDX: u64 = 0b111 << 2;
    /// `SH` (bits 9:8) = 0b10, outer shareable; the builder writes inner.
    const SH_OUTER: u64 = 0b10 << 8;
    /// Everything the walker ignores, set at once.
    const IGNORED: u64 = OUTPUT_BITS_51_48
        | CONTIGUOUS
        | ATTR_INDX
        | SH_OUTER
        | descriptor::NOT_GLOBAL
        | descriptor::PXN;
    const TABLE: u64 = VALID | descriptor::TABLE_OR_PAGE;

    /// The bits of a block's output address field below its size, which are `RES0`.
    fn block_res0_bits(level: u8) -> u64 {
        ((1 << descriptor::shift(level)) - 1) & OUTPUT_ADDRESS_MASK
    }

    #[test]
    fn ignored_bits_do_not_change_the_translation() {
        // The documented divergences: output address bits 51:48, a block's RES0 bits between
        // its size and bit 12, and AttrIndx, SH, nG, the contiguous hint and PXN.
        let (memory, root) = blank();
        // Level-1 slot 0 -> level-2 table; slot 1 is a 1 GiB block.
        put(&memory, ROOT, L2 | TABLE | OUTPUT_BITS_51_48);
        let block1_pa = 0x1_c000_0000;
        put(
            &memory,
            ROOT + 8,
            block1_pa
                | block_res0_bits(1)
                | VALID
                | AF
                | descriptor::AP_EL0
                | descriptor::UXN
                | IGNORED,
        );
        // Level-2 slot 0 -> level-3 table; slot 1 is a 2 MiB block.
        put(&memory, L2, L3 | TABLE | OUTPUT_BITS_51_48);
        let block2_pa = 0x9_0060_0000;
        put(
            &memory,
            L2 + 8,
            block2_pa
                | block_res0_bits(2)
                | VALID
                | AF
                | descriptor::AP_EL0
                | descriptor::AP_READ_ONLY
                | IGNORED,
        );
        // Level-3 slot 1: a read-write page.
        let page_pa = 0x1234_5000;
        put(
            &memory,
            L3 + 8,
            page_pa | TABLE | AF | descriptor::AP_EL0 | descriptor::UXN | IGNORED,
        );

        let in_page = PAGE_SIZE + 0x7f8;
        assert_eq!(
            walk(&memory, root, in_page),
            Ok(Resolved {
                pa: page_pa + 0x7f8,
                perms: Perms::ReadWrite,
                level: LEAF_LEVEL
            })
        );
        let va_in_block2 = (1 << descriptor::shift(2)) | 0x1_2345;
        assert_eq!(
            walk(&memory, root, va_in_block2),
            Ok(Resolved {
                pa: block2_pa | 0x1_2345,
                perms: Perms::ReadExecute,
                level: 2
            })
        );
        let va_in_block1 = (1 << descriptor::shift(1)) | 0x1234_5678;
        assert_eq!(
            walk(&memory, root, va_in_block1),
            Ok(Resolved {
                pa: block1_pa | 0x1234_5678,
                perms: Perms::ReadWrite,
                level: 1
            })
        );
    }

    #[test]
    fn el1_read_only_el0_none_pages_are_never_el0_readable() {
        // AP[2:1] = 0b10 (EL1 read-only, EL0 no access): EL0 gets nothing, whatever UXN says
        // about execution. AP[2:1] = 0b00 is the same for EL0.
        let (memory, root) = blank();
        put(&memory, ROOT, L2 | TABLE);
        put(&memory, L2, L3 | TABLE);
        let page = |ap: u64, uxn: u64| 0x1234_5000 | TABLE | AF | ap | uxn;
        let cases = [
            (descriptor::AP_READ_ONLY, descriptor::UXN, Perms::None),
            (descriptor::AP_READ_ONLY, 0, Perms::Execute),
            (0, descriptor::UXN, Perms::None),
            (0, 0, Perms::Execute),
        ];
        for (n, (ap, uxn, perms)) in (0..).zip(cases) {
            put(&memory, L3 + n * 8, page(ap, uxn));
            assert_eq!(
                walk(&memory, root, n * PAGE_SIZE),
                Ok(Resolved {
                    pa: 0x1234_5000,
                    perms,
                    level: LEAF_LEVEL
                }),
                "AP {ap:#x} UXN {uxn:#x}"
            );
        }
    }

    #[test]
    fn blocks_with_the_access_flag_clear_fault_at_their_own_level() {
        let (memory, root) = blank();
        // Level-1 slot 2: a 1 GiB block, AF clear. Slot 3 -> level-2 table whose slot 4 is a
        // 2 MiB block, AF clear.
        put(&memory, ROOT + 2 * 8, 0x1_c000_0000 | VALID);
        put(&memory, ROOT + 3 * 8, L2 | TABLE);
        put(&memory, L2 + 4 * 8, 0x9_0060_0000 | VALID);
        let gib = 2 << descriptor::shift(1);
        assert_eq!(
            walk(&memory, root, gib | 0x10),
            Err(WalkError::AccessFlag {
                va: gib | 0x10,
                level: 1
            })
        );
        let mib = (3 << descriptor::shift(1)) | (4 << descriptor::shift(2));
        assert_eq!(
            walk(&memory, root, mib | 0x10),
            Err(WalkError::AccessFlag {
                va: mib | 0x10,
                level: 2
            })
        );
        // With AF set the same blocks translate.
        put(&memory, ROOT + 2 * 8, 0x1_c000_0000 | VALID | AF);
        put(&memory, L2 + 4 * 8, 0x9_0060_0000 | VALID | AF);
        assert!(walk(&memory, root, gib).is_ok());
        assert!(walk(&memory, root, mib).is_ok());
    }

    #[test]
    fn address_space_ends_at_bit_39() {
        let (memory, root) = blank();
        let last = VA_SPACE_END - 1;
        // The last byte walks (and is unmapped at level 1, index 511); the next one is out.
        assert_eq!(
            walk(&memory, root, last),
            Err(WalkError::Unmapped { va: last, level: 1 })
        );
        assert_eq!(VA_SPACE_END, 1 << 39);
        assert_eq!(
            walk(&memory, root, VA_SPACE_END),
            Err(WalkError::OutOfRange { va: VA_SPACE_END })
        );
        assert_eq!(
            walk(&memory, root, u64::MAX),
            Err(WalkError::OutOfRange { va: u64::MAX })
        );
    }

    #[test]
    fn a_table_outside_memory_is_a_fault_not_a_panic() {
        let (memory, root) = blank();
        let outside = ROOT + 0x10_0000;
        put(&memory, ROOT, outside | 0b11);
        assert_eq!(
            walk(&memory, root, 0),
            Err(WalkError::Memory(Fault { address: outside }))
        );
        // A table at the very top of the output address space; level-2 index 511.
        put(&memory, ROOT, OUTPUT_ADDRESS_MASK | 0b11);
        assert_eq!(
            walk(&memory, root, 0x3fff_ffff),
            Err(WalkError::Memory(Fault {
                address: OUTPUT_ADDRESS_MASK + 0x1ff * 8
            }))
        );
    }
}

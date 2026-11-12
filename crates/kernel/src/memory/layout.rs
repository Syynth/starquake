//! Where the regions of a process's address space go: the code region, and the alias, heap,
//! stack and kernel-map regions placed around the program's code.
//!
//! Ported from Eden (GPL-3.0-or-later), paths relative to the reference folder:
//!
//! - `eden/src/core/hle/kernel/k_page_table_base.cpp` @ cf654052,
//!   `KPageTableBase::InitializeForProcess`: the placement.
//! - `eden/src/core/hle/kernel/k_address_space_info.cpp` @ cf654052: the sizes of a 39-bit
//!   address space (`AddressSpaceInfos`, the rows with `bit_width = 39`, without `HAS_NCE`).
//! - `eden-pr4588/src/core/hle/kernel/k_address_space_info.cpp` @ e5fb3b97,
//!   `GetAddressSpaceSize`: for a 39-bit space the heap and alias regions are sized from the
//!   console's memory (scoping doc, "Process, memory, kernel": "The heap and alias regions scale
//!   with physical memory: 8-32 GiB heap, alias up to 8x that"). For the 4 GiB a Switch 1 has,
//!   that gives the 8 GiB and 64 GiB of Eden's table.
//!
//! Only the 39-bit address space is built: it is what a 64-bit program gets (Eden's
//! `KProcess::LoadFromMetadata` picks it from the NPDM, and its NRO loader gives a homebrew
//! program the same). The 32-bit and 36-bit layouts land with the first program that needs one.
//!
//! # Placement is deterministic by default
//!
//! Horizon moves each of the four regions by a random multiple of [`REGION_ALIGNMENT`]. Here the
//! four offsets are an input ([`RegionOffsets`]), and the default is zero for each (kernel spec:
//! "Region placement is deterministic by default"), which is also what Eden computes with
//! address-space randomisation off. A test checks the result against the addresses Eden-NX2
//! reported for the homebrew fixture (`the_default_layout_is_the_one_the_reference_reported`).
//! Deriving offsets from a seed belongs to process creation (kernel slice 8).
//!
//! Divergence from Eden: a layout that does not fit, and offsets that are not multiples of the
//! alignment or exceed the slack, are errors here. Eden returns `ResultOutOfMemory` for the
//! first and cannot produce the others (it draws the offsets itself), and it asserts, rather
//! than checks, that the regions lie inside the address space.

use starquake_memory::PAGE_SIZE;

/// One mebibyte.
const MIB: u64 = 1 << 20;
/// One gibibyte.
const GIB: u64 = 1 << 30;

/// The width of the address space this module lays out, in bits.
pub const ADDRESS_SPACE_BITS: u32 = 39;
/// One past the last address of the address space.
pub const ADDRESS_SPACE_END: u64 = 1 << ADDRESS_SPACE_BITS;
/// What regions and a program's code are aligned to (`KPageTableBase::RegionAlignment`).
pub const REGION_ALIGNMENT: u64 = 2 * MIB;
/// Where the code region starts (`Map39Bit`'s address): the region `GetInfo` calls the ASLR
/// region.
pub const CODE_REGION_START: u64 = 128 * MIB;
/// The size of the code region (`Map39Bit`'s size).
pub const CODE_REGION_SIZE: u64 = 512 * GIB - 128 * MIB;
/// The size of the stack region (`Stack`, 39 bits).
pub const STACK_REGION_SIZE: u64 = 2 * GIB;
/// The size of the kernel-map region, where thread-local pages go (`MapSmall`, 39 bits).
pub const KERNEL_MAP_REGION_SIZE: u64 = 64 * GIB;
/// The smallest heap region, and its size on a console with up to 8 GiB of memory.
pub const HEAP_REGION_SIZE_MIN: u64 = 8 * GIB;
/// The largest heap region.
pub const HEAP_REGION_SIZE_MAX: u64 = 32 * GIB;
/// How many times the heap region's size the alias region is.
pub const ALIAS_REGION_HEAP_MULTIPLE: u64 = 8;
/// The largest alias region.
pub const ALIAS_REGION_SIZE_MAX: u64 = 128 * GIB;

/// The size of the heap region of a 39-bit address space on a console with
/// `physical_memory_size` bytes of memory: that size rounded up to a gibibyte, and no less than
/// [`HEAP_REGION_SIZE_MIN`] or more than [`HEAP_REGION_SIZE_MAX`].
#[must_use]
pub fn heap_region_size(physical_memory_size: u64) -> u64 {
    physical_memory_size
        .min(HEAP_REGION_SIZE_MAX)
        .next_multiple_of(GIB)
        .clamp(HEAP_REGION_SIZE_MIN, HEAP_REGION_SIZE_MAX)
}

/// The size of the alias region that goes with a heap region of `heap_region_size` bytes.
#[must_use]
pub fn alias_region_size(heap_region_size: u64) -> u64 {
    heap_region_size
        .saturating_mul(ALIAS_REGION_HEAP_MULTIPLE)
        .min(ALIAS_REGION_SIZE_MAX)
}

/// A run of addresses: `start..end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Region {
    /// The first address.
    pub start: u64,
    /// One past the last address.
    pub end: u64,
}

impl Region {
    /// The size in bytes.
    #[must_use]
    pub const fn size(self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Whether this region and `other` share an address. An empty region shares none.
    #[must_use]
    pub const fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end && self.start < self.end
    }

    /// Whether the `size` bytes at `address` lie inside this region. Nothing lies inside an
    /// empty range, and a range that wraps lies inside nothing.
    #[must_use]
    pub fn holds(self, address: u64, size: u64) -> bool {
        size > 0
            && self.start <= address
            && address
                .checked_add(size)
                .is_some_and(|range_end| range_end <= self.end)
    }
}

/// How far each of the four placed regions is moved up from its lowest position. Each must be
/// a multiple of [`REGION_ALIGNMENT`] no larger than the slack the layout has. The default is
/// zero for each: no randomisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegionOffsets {
    /// The alias region's offset.
    pub alias: u64,
    /// The heap region's offset.
    pub heap: u64,
    /// The stack region's offset.
    pub stack: u64,
    /// The kernel-map region's offset.
    pub kernel_map: u64,
}

/// What a layout is computed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutRequest {
    /// Where the program's code starts.
    pub code_address: u64,
    /// How many bytes of code there are.
    pub code_size: u64,
    /// The console's memory, which sizes the heap and alias regions ([`heap_region_size`]).
    pub physical_memory_size: u64,
    /// Whether the alias region gets an eighth of the address space on top
    /// (`CreateProcessFlag::EnableAliasRegionExtraSize`, firmware 18 and later).
    pub alias_region_extra: bool,
    /// Where in their slack the regions go.
    pub offsets: RegionOffsets,
}

/// Why an address space could not be laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    /// The code is empty, wraps, does not lie inside the code region, or (for an address space
    /// being created) is not whole pages.
    #[error("{size:#x} bytes of code at {address:#x} are not pages inside the code region")]
    BadCode {
        /// Where the code starts.
        address: u64,
        /// Its size.
        size: u64,
    },
    /// The regions need more room than the address space has beside the code.
    #[error("the regions need {needed:#x} bytes and there are {available:#x} beside the code")]
    NoRoom {
        /// Bytes the alias, heap, stack and kernel-map regions need together.
        needed: u64,
        /// Bytes in the larger of the two gaps beside the code.
        available: u64,
    },
    /// An offset is not a multiple of the region alignment, or is larger than the slack.
    #[error("a region offset of {offset:#x} is not an aligned offset of at most {slack:#x}")]
    BadOffset {
        /// The offset.
        offset: u64,
        /// The largest offset the layout allows.
        slack: u64,
    },
}

/// The regions of one address space (the `m_*_region_start` and `_end` members of
/// `KPageTableBase`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Regions {
    /// The whole address space.
    pub address_space: Region,
    /// Where code may go: `GetInfo`'s ASLR region. For a 39-bit space the alias-code region is
    /// the same range.
    pub code: Region,
    /// The alias region: IPC buffers, and `MapPhysicalMemory`.
    pub alias: Region,
    /// The heap region.
    pub heap: Region,
    /// The stack region.
    pub stack: Region,
    /// The kernel-map region: thread-local pages.
    pub kernel_map: Region,
    /// How much of the alias region is the extra an [`LayoutRequest::alias_region_extra`]
    /// process gets; zero for any other.
    pub alias_region_extra_size: u64,
}

impl Regions {
    /// Lays out a 39-bit address space around the program's code
    /// (`KPageTableBase::InitializeForProcess`).
    pub fn new(request: &LayoutRequest) -> Result<Self, LayoutError> {
        let LayoutRequest {
            code_address,
            code_size,
            ..
        } = *request;
        let bad_code = LayoutError::BadCode {
            address: code_address,
            size: code_size,
        };
        let address_space = Region {
            start: 0,
            end: ADDRESS_SPACE_END,
        };
        let code = Region {
            start: CODE_REGION_START,
            end: CODE_REGION_START + CODE_REGION_SIZE,
        };
        // Eden asserts the code lies inside the address space; inside the code region is what
        // `KProcess::Initialize` then requires (`CanContain(.., KMemoryState::Code)`).
        if !code.holds(code_address, code_size) {
            return Err(bad_code);
        }
        let code_end = code_address.checked_add(code_size).ok_or(bad_code)?;
        let process_code_start = code_address - code_address % REGION_ALIGNMENT;
        let process_code_end = code_end
            .checked_next_multiple_of(REGION_ALIGNMENT)
            .ok_or(bad_code)?;

        let heap_size = heap_region_size(request.physical_memory_size);
        let stack_size = STACK_REGION_SIZE;
        let kernel_map_size = KERNEL_MAP_REGION_SIZE;
        // FW 18+: an eighth of the address space on top of the alias region.
        let alias_region_extra_size = if request.alias_region_extra {
            address_space.size() >> 3
        } else {
            0
        };
        let alias_size = alias_region_size(heap_size) + alias_region_extra_size;

        // Determine the region we can place our undetermineds in: the larger gap beside the
        // code.
        let below = process_code_start - code.start;
        let above = address_space.end.saturating_sub(process_code_end);
        let (alloc_start, alloc_size) = if below >= above {
            (code.start, below)
        } else {
            (process_code_end, above)
        };
        let needed = alias_size + heap_size + stack_size + kernel_map_size;
        let slack = alloc_size.checked_sub(needed).ok_or(LayoutError::NoRoom {
            needed,
            available: alloc_size,
        })?;
        let RegionOffsets {
            alias,
            heap,
            stack,
            kernel_map,
        } = request.offsets;
        for offset in [alias, heap, stack, kernel_map] {
            if !offset.is_multiple_of(REGION_ALIGNMENT) || offset > slack {
                return Err(LayoutError::BadOffset { offset, slack });
            }
        }

        let [alias_start, heap_start, stack_start, kmap_start] = place(
            alloc_start,
            [alias_size, heap_size, stack_size, kernel_map_size],
            request.offsets,
        );
        let region = |start: u64, size: u64| Region {
            start,
            end: start + size,
        };
        Ok(Self {
            address_space,
            code,
            alias: region(alias_start, alias_size),
            heap: region(heap_start, heap_size),
            stack: region(stack_start, stack_size),
            kernel_map: region(kmap_start, kernel_map_size),
            alias_region_extra_size,
        })
    }

    /// Whether the `size` bytes at `address` lie inside the address space
    /// (`KPageTableBase::Contains`).
    #[must_use]
    pub fn contains(&self, address: u64, size: u64) -> bool {
        self.address_space.holds(address, size)
    }
}

/// Where the alias, heap, stack and kernel-map regions start, in that order, given their
/// `sizes` in the same order, the lowest address `alloc_start` they may use, and their
/// offsets. Each region starts at its offset, and then moves up past every region whose offset
/// is lower: Eden's sequence of comparisons, in its order (ties included).
fn place(alloc_start: u64, sizes: [u64; 4], offsets: RegionOffsets) -> [u64; 4] {
    let [alias_size, heap_size, stack_size, kernel_map_size] = sizes;
    let RegionOffsets {
        alias: alias_rnd,
        heap: heap_rnd,
        stack: stack_rnd,
        kernel_map: kmap_rnd,
    } = offsets;
    let mut alias_start = alloc_start + alias_rnd;
    let mut heap_start = alloc_start + heap_rnd;
    if alias_rnd <= heap_rnd {
        heap_start += alias_size;
    } else {
        alias_start += heap_size;
    }
    let mut stack_start = alloc_start + stack_rnd;
    if alias_rnd < stack_rnd {
        stack_start += alias_size;
    } else {
        alias_start += stack_size;
    }
    if heap_rnd < stack_rnd {
        stack_start += heap_size;
    } else {
        heap_start += stack_size;
    }
    let mut kmap_start = alloc_start + kmap_rnd;
    if alias_rnd < kmap_rnd {
        kmap_start += alias_size;
    } else {
        alias_start += kernel_map_size;
    }
    if heap_rnd < kmap_rnd {
        kmap_start += heap_size;
    } else {
        heap_start += kernel_map_size;
    }
    if stack_rnd < kmap_rnd {
        kmap_start += stack_size;
    } else {
        stack_start += kernel_map_size;
    }
    [alias_start, heap_start, stack_start, kmap_start]
}

/// The size of a run of `pages` pages in bytes, if it fits 64 bits.
pub(super) fn pages_to_bytes(pages: u64) -> Option<u64> {
    pages.checked_mul(PAGE_SIZE)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

    use super::*;

    /// Fixed seed so a run is reproducible; failures are not persisted to disk.
    const PROPTEST_SEED: u64 = 0x4C41_594F_5554_5F31;

    fn config() -> ProptestConfig {
        ProptestConfig {
            cases: 512,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(PROPTEST_SEED),
            failure_persistence: None,
            ..Config::default()
        }
    }

    /// The memory of a Switch 1: what the reference emulator ran the fixture with.
    const FOUR_GIB: u64 = 4 * GIB;

    fn request(code_address: u64, code_size: u64) -> LayoutRequest {
        LayoutRequest {
            code_address,
            code_size,
            physical_memory_size: FOUR_GIB,
            alias_region_extra: false,
            offsets: RegionOffsets::default(),
        }
    }

    /// What Eden-NX2 @ 50fa86bb told the homebrew fixture, in the run whose code was at
    /// `0x80c01000` (seven pages): `GetInfo` 2/3 (alias), 4/5 (heap), 12/13 (ASLR) and 14/15
    /// (stack), from `tests/homebrew/hello/reference/startup.trace` before its addresses were
    /// masked. That emulator placed the four regions without randomisation, as the default
    /// here does.
    #[test]
    fn the_default_layout_is_the_one_the_reference_reported() {
        let regions = Regions::new(&request(0x80c0_1000, 7 * PAGE_SIZE)).unwrap();
        let reported = |start: u64, size: u64| Region {
            start,
            end: start + size,
        };
        assert_eq!(regions.alias, reported(0x11_00e0_0000, 0x10_0000_0000));
        assert_eq!(regions.heap, reported(0x21_00e0_0000, 0x2_0000_0000));
        assert_eq!(regions.code, reported(0x800_0000, 0x7f_f800_0000));
        assert_eq!(regions.stack, reported(0x10_80e0_0000, 0x8000_0000));
        // The kernel-map region is not reported; it is what lies between the code and the
        // stack region.
        assert_eq!(regions.kernel_map, reported(0x80e0_0000, 0x10_0000_0000));
        assert_eq!(regions.alias_region_extra_size, 0);
        // Its two other runs, with the code at 0x80682000 and 0x8075a000, round to the same
        // 2 MiB boundary and so reported one layout.
        for code in [0x8068_2000, 0x8075_a000] {
            let regions = Regions::new(&request(code, 7 * PAGE_SIZE)).unwrap();
            assert_eq!(regions.stack.start, 0x10_8080_0000, "{code:#x}");
            assert_eq!(regions.alias.start, 0x11_0080_0000, "{code:#x}");
            assert_eq!(regions.heap.start, 0x21_0080_0000, "{code:#x}");
        }
    }

    #[test]
    fn heap_and_alias_regions_scale_with_the_consoles_memory() {
        // PR #4588's rule at its edges: 8 GiB up to 8 GiB of memory, then the memory rounded
        // up to a gibibyte, and 32 GiB from 32 GiB on.
        let cases = [
            (0, HEAP_REGION_SIZE_MIN),
            (FOUR_GIB, HEAP_REGION_SIZE_MIN),
            (HEAP_REGION_SIZE_MIN, HEAP_REGION_SIZE_MIN),
            (HEAP_REGION_SIZE_MIN + 1, HEAP_REGION_SIZE_MIN + GIB),
            (12 * GIB, 12 * GIB),
            (HEAP_REGION_SIZE_MAX - 1, HEAP_REGION_SIZE_MAX),
            (HEAP_REGION_SIZE_MAX, HEAP_REGION_SIZE_MAX),
            (HEAP_REGION_SIZE_MAX + 1, HEAP_REGION_SIZE_MAX),
            (u64::MAX, HEAP_REGION_SIZE_MAX),
        ];
        for (memory, heap) in cases {
            assert_eq!(heap_region_size(memory), heap, "{memory:#x}");
        }
        // The alias region is eight heaps, capped at 128 GiB: reached at a 16 GiB heap.
        assert_eq!(alias_region_size(HEAP_REGION_SIZE_MIN), 64 * GIB);
        assert_eq!(alias_region_size(12 * GIB), 96 * GIB);
        assert_eq!(
            alias_region_size(ALIAS_REGION_SIZE_MAX / ALIAS_REGION_HEAP_MULTIPLE),
            ALIAS_REGION_SIZE_MAX
        );
        assert_eq!(
            alias_region_size(ALIAS_REGION_SIZE_MAX / ALIAS_REGION_HEAP_MULTIPLE + GIB),
            ALIAS_REGION_SIZE_MAX
        );
        assert_eq!(alias_region_size(u64::MAX), ALIAS_REGION_SIZE_MAX);

        let twelve = LayoutRequest {
            physical_memory_size: 12 * GIB,
            ..request(0x8000_0000, PAGE_SIZE)
        };
        let regions = Regions::new(&twelve).unwrap();
        assert_eq!(regions.heap.size(), 12 * GIB);
        assert_eq!(regions.alias.size(), 96 * GIB);
    }

    #[test]
    fn the_extra_alias_size_is_an_eighth_of_the_address_space() {
        let extra = LayoutRequest {
            alias_region_extra: true,
            ..request(0x8000_0000, PAGE_SIZE)
        };
        let regions = Regions::new(&extra).unwrap();
        assert_eq!(regions.alias_region_extra_size, ADDRESS_SPACE_END / 8);
        assert_eq!(
            regions.alias.size(),
            alias_region_size(HEAP_REGION_SIZE_MIN) + ADDRESS_SPACE_END / 8
        );
    }

    #[test]
    fn regions_go_below_the_code_when_there_is_more_room_there() {
        // Code in the upper half of the address space: the gap below is the larger one, so
        // the regions start at the code region's start.
        let high = ADDRESS_SPACE_END - 64 * GIB;
        let regions = Regions::new(&request(high, PAGE_SIZE)).unwrap();
        assert_eq!(regions.kernel_map.start, CODE_REGION_START);
        assert!(regions.heap.end <= high);
    }

    #[test]
    fn code_must_lie_inside_the_code_region() {
        let top = CODE_REGION_START + CODE_REGION_SIZE;
        let refused = [
            // Empty, below the region, ending past it, and wrapping.
            (CODE_REGION_START, 0),
            (CODE_REGION_START - PAGE_SIZE, PAGE_SIZE),
            (top - PAGE_SIZE, 2 * PAGE_SIZE),
            (top, PAGE_SIZE),
            (u64::MAX - PAGE_SIZE + 1, PAGE_SIZE),
            (CODE_REGION_START, u64::MAX),
        ];
        for (address, size) in refused {
            assert_eq!(
                Regions::new(&request(address, size)),
                Err(LayoutError::BadCode { address, size }),
                "{address:#x}+{size:#x}"
            );
        }
        // The first and the last page of the region are inside.
        assert!(Regions::new(&request(CODE_REGION_START, PAGE_SIZE)).is_ok());
        assert!(Regions::new(&request(top - PAGE_SIZE, PAGE_SIZE)).is_ok());
        assert_eq!(
            LayoutError::BadCode {
                address: 0x1000,
                size: 0x2000
            }
            .to_string(),
            "0x2000 bytes of code at 0x1000 are not pages inside the code region"
        );
    }

    #[test]
    fn a_layout_that_does_not_fit_beside_the_code_is_refused_exactly_at_the_limit() {
        // The largest layout: a 32 GiB heap, a 128 GiB alias region with the extra eighth of
        // the address space, the stack and the kernel map.
        let needed = ALIAS_REGION_SIZE_MAX
            + ADDRESS_SPACE_END / 8
            + HEAP_REGION_SIZE_MAX
            + STACK_REGION_SIZE
            + KERNEL_MAP_REGION_SIZE;
        let largest = |code_address: u64| LayoutRequest {
            physical_memory_size: u64::MAX,
            alias_region_extra: true,
            ..request(code_address, REGION_ALIGNMENT)
        };
        // Code that leaves exactly that much below it (and less above): the regions fill the
        // gap, from the code region's start to the code.
        let exact = CODE_REGION_START + needed;
        assert!(ADDRESS_SPACE_END - (exact + REGION_ALIGNMENT) < needed);
        let regions = Regions::new(&largest(exact)).unwrap();
        assert_eq!(regions.kernel_map.start, CODE_REGION_START);
        assert_eq!(regions.heap.end, exact);
        // One alignment unit lower, the gap is one unit short.
        assert_eq!(
            Regions::new(&largest(exact - REGION_ALIGNMENT)),
            Err(LayoutError::NoRoom {
                needed,
                available: needed - REGION_ALIGNMENT
            })
        );
        assert_eq!(
            LayoutError::NoRoom {
                needed: 0x3000,
                available: 0x2000
            }
            .to_string(),
            "the regions need 0x3000 bytes and there are 0x2000 beside the code"
        );
    }

    #[test]
    fn the_slack_is_exact_one_region_alignment_decides() {
        // Code placed so the gap above is the needed size plus one alignment unit of slack.
        let base = request(CODE_REGION_START, REGION_ALIGNMENT);
        let needed = alias_region_size(HEAP_REGION_SIZE_MIN)
            + HEAP_REGION_SIZE_MIN
            + STACK_REGION_SIZE
            + KERNEL_MAP_REGION_SIZE;
        let slack = ADDRESS_SPACE_END - (CODE_REGION_START + REGION_ALIGNMENT) - needed;
        let with = |offset: u64| LayoutRequest {
            offsets: RegionOffsets {
                heap: offset,
                ..RegionOffsets::default()
            },
            ..base
        };
        // The largest offset puts the heap region's end exactly at the end of the space.
        let regions = Regions::new(&with(slack)).unwrap();
        assert_eq!(regions.heap.end, ADDRESS_SPACE_END);
        assert_eq!(
            Regions::new(&with(slack + REGION_ALIGNMENT)),
            Err(LayoutError::BadOffset {
                offset: slack + REGION_ALIGNMENT,
                slack
            })
        );
        // An offset off the alignment is refused too.
        assert_eq!(
            Regions::new(&with(PAGE_SIZE)),
            Err(LayoutError::BadOffset {
                offset: PAGE_SIZE,
                slack
            })
        );
        assert_eq!(
            LayoutError::BadOffset {
                offset: 0x1000,
                slack: 0x20_0000
            }
            .to_string(),
            "a region offset of 0x1000 is not an aligned offset of at most 0x200000"
        );
    }

    /// Eden's placement, worked by hand for offsets in four different orders. With the code at
    /// `0x8000_0000` (one page), the regions start at `0x8020_0000` (`A`); sizes are alias 64 GiB,
    /// heap 8 GiB, stack 2 GiB, kernel map 64 GiB. Each region sits at its own offset plus the
    /// sizes of the regions with a lower offset.
    #[test]
    fn regions_with_offsets_stack_in_the_order_of_their_offsets() {
        const A: u64 = 0x8020_0000;
        const U: u64 = REGION_ALIGNMENT;
        let (alias, heap, stack, kmap) = (64 * GIB, 8 * GIB, 2 * GIB, 64 * GIB);
        let layout = |offsets: RegionOffsets| {
            let regions = Regions::new(&LayoutRequest {
                offsets,
                ..request(0x8000_0000, PAGE_SIZE)
            })
            .unwrap();
            (
                regions.alias.start,
                regions.heap.start,
                regions.stack.start,
                regions.kernel_map.start,
            )
        };
        // Offsets rising in the order heap, stack, kernel map, alias.
        assert_eq!(
            layout(RegionOffsets {
                heap: U,
                stack: 2 * U,
                kernel_map: 3 * U,
                alias: 4 * U
            }),
            (
                A + 4 * U + heap + stack + kmap,
                A + U,
                A + 2 * U + heap,
                A + 3 * U + heap + stack
            )
        );
        // The reverse order.
        assert_eq!(
            layout(RegionOffsets {
                alias: U,
                kernel_map: 2 * U,
                stack: 3 * U,
                heap: 4 * U
            }),
            (
                A + U,
                A + 4 * U + alias + kmap + stack,
                A + 3 * U + alias + kmap,
                A + 2 * U + alias
            )
        );
        // All equal: the ties go kernel map, stack, alias, heap, as with no offsets at all.
        for offset in [0, 5 * U] {
            let all = RegionOffsets {
                alias: offset,
                heap: offset,
                stack: offset,
                kernel_map: offset,
            };
            assert_eq!(
                layout(all),
                (
                    A + offset + kmap + stack,
                    A + offset + kmap + stack + alias,
                    A + offset + kmap,
                    A + offset
                )
            );
        }
    }

    #[test]
    fn a_region_holds_exactly_the_ranges_inside_it() {
        let region = Region {
            start: 0x4000,
            end: 0x8000,
        };
        assert_eq!(region.size(), 0x4000);
        assert!(region.holds(0x4000, 0x4000));
        assert!(region.holds(0x7fff, 1));
        assert!(!region.holds(0x4000, 0x4001));
        assert!(!region.holds(0x3fff, 2));
        assert!(!region.holds(0x8000, 1));
        assert!(!region.holds(0x4000, 0));
        assert!(!region.holds(0x4000, u64::MAX));
        let touching = Region {
            start: 0x8000,
            end: 0x9000,
        };
        let inside = Region {
            start: 0x7000,
            end: 0x8001,
        };
        let empty = Region {
            start: 0x5000,
            end: 0x5000,
        };
        assert!(!region.overlaps(touching) && !touching.overlaps(region));
        assert!(region.overlaps(inside) && inside.overlaps(region));
        assert!(!empty.overlaps(region));
        assert_eq!(pages_to_bytes(3), Some(3 * PAGE_SIZE));
        assert_eq!(pages_to_bytes(u64::MAX), None);
    }

    proptest! {
        #![proptest_config(config())]

        /// Whatever the code's place and size, the console's memory and the offsets (inside
        /// the slack), a layout that is accepted has what Eden asserts of its own: every
        /// region inside the address space with its size, the four placed regions aligned and
        /// disjoint from each other and from the code's aligned range.
        #[test]
        fn accepted_layouts_are_aligned_disjoint_and_inside_the_address_space(
            code_page in (CODE_REGION_START / PAGE_SIZE)..(ADDRESS_SPACE_END / PAGE_SIZE),
            code_pages in 1..0x4_0000_u64,
            memory_gib in 0..40_u64,
            extra in any::<bool>(),
            units in proptest::array::uniform4(0..0x2_0000_u64),
        ) {
            let base = LayoutRequest {
                code_address: code_page * PAGE_SIZE,
                code_size: code_pages * PAGE_SIZE,
                physical_memory_size: memory_gib * GIB,
                alias_region_extra: extra,
                offsets: RegionOffsets::default(),
            };
            let Ok(unmoved) = Regions::new(&base) else {
                return Ok(());
            };
            // The slack: how far the topmost region can still move up (or the code's start,
            // when the regions are below it).
            let top = [unmoved.alias, unmoved.heap, unmoved.stack, unmoved.kernel_map]
                .iter()
                .map(|region| region.end)
                .max()
                .unwrap();
            let limit = if unmoved.kernel_map.start == CODE_REGION_START {
                base.code_address - base.code_address % REGION_ALIGNMENT
            } else {
                ADDRESS_SPACE_END
            };
            let slack_units = (limit - top) / REGION_ALIGNMENT;
            let [alias, heap, stack, kernel_map] =
                units.map(|unit| unit % (slack_units + 1) * REGION_ALIGNMENT);
            let request = LayoutRequest {
                offsets: RegionOffsets { alias, heap, stack, kernel_map },
                ..base
            };
            let regions = Regions::new(&request).unwrap();

            let heap_size = heap_region_size(request.physical_memory_size);
            prop_assert_eq!(regions.heap.size(), heap_size);
            prop_assert_eq!(
                regions.alias.size(),
                alias_region_size(heap_size) + regions.alias_region_extra_size
            );
            prop_assert_eq!(regions.stack.size(), STACK_REGION_SIZE);
            prop_assert_eq!(regions.kernel_map.size(), KERNEL_MAP_REGION_SIZE);
            let code_start = request.code_address - request.code_address % REGION_ALIGNMENT;
            let code = Region {
                start: code_start,
                end: (request.code_address + request.code_size).next_multiple_of(REGION_ALIGNMENT),
            };
            let placed = [regions.alias, regions.heap, regions.stack, regions.kernel_map];
            for (index, region) in placed.iter().enumerate() {
                prop_assert!(region.start.is_multiple_of(REGION_ALIGNMENT));
                prop_assert!(regions.address_space.holds(region.start, region.size()));
                prop_assert!(!region.overlaps(code), "{:?} overlaps the code", region);
                for other in &placed[index + 1..] {
                    prop_assert!(!region.overlaps(*other), "{:?} {:?}", region, other);
                }
            }
        }
    }
}

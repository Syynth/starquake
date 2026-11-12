//! One guest address space as the host keeps it: the stage-1 tables guest code runs on, the RAM
//! handed out to back its pages, and a shadow of what is mapped where, through which the host
//! reads and writes guest memory by virtual address.
//!
//! Design: `docs/specs/execution-model.md`, "Guest memory". Guest code reaches memory through
//! [`Stage1Tables`], with the permissions each region was mapped with. Host code (the loader,
//! the kernel reading a system call's buffers) goes through [`AddressSpace`]'s [`GuestMemory`],
//! which "resolves addresses with a host-side shadow of the address space, not by walking the
//! hardware tables" ("Host access to guest memory").
//!
//! PROVISIONAL (walking skeleton), and what replaces each part:
//!
//! - **Backing is handed out from the start of guest RAM, in the order regions are mapped, and
//!   never returned.** Kernel slice 6 (memory blocks, regions) and the backing pools of the
//!   execution model, with their accounting, replace it.
//! - **The shadow knows addresses only.** Host access ignores permissions, which is what lets
//!   the loader write a code page that guest code may only read and execute; nothing checks
//!   that a system call's buffer is readable or writable by the guest. Kernel slice 6 keeps
//!   Horizon's memory states and permissions and checks them.
//! - **Nothing is unmapped or re-protected**, so no TLB flush is ever owed: `map` records none
//!   (`starquake_memory::stage1`, "TLB invalidation"). The kernel's memory system calls
//!   (slice 6) bring `unmap` and `protect`, and with them the backend's flush entry point.

use starquake_memory::stage1::{Stage1Tables, TableRoot};
use starquake_memory::{BackingRange, Fault, GuestMemory, MapError, Translation};

use crate::image::Region;

/// Why a region could not be mapped into a guest address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SpaceError {
    /// The translation tables refused the mapping, or could not be created.
    #[error("the guest's translation tables refused a mapping: {0}")]
    Map(#[from] MapError),
    /// Guest RAM has fewer bytes left than the region needs.
    #[error("mapping {size:#x} bytes needs more guest RAM than the {available:#x} bytes left")]
    OutOfRam {
        /// Bytes the region needs.
        size: u64,
        /// Bytes of guest RAM not yet handed out.
        available: u64,
    },
}

/// A mapped run of pages: `size` bytes at virtual `va`, backed by guest-physical `pa`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Run {
    va: u64,
    pa: u64,
    size: u64,
}

/// One guest address space (see the module docs). `M` is the guest-physical memory the tables
/// and the backing live in.
#[derive(Debug)]
pub(crate) struct AddressSpace<M> {
    tables: Stage1Tables<M>,
    /// Guest RAM: where backing comes from.
    ram: BackingRange,
    /// The first guest-physical address of RAM not handed out yet.
    next: u64,
    /// What is mapped, in virtual address order, without overlaps.
    shadow: Vec<Run>,
}

impl<M: GuestMemory> AddressSpace<M> {
    /// An empty address space whose tables take their pages from `pool` and whose pages will
    /// be backed from `ram`, both ranges of `physical`'s guest-physical addresses.
    pub(crate) fn new(
        physical: M,
        ram: BackingRange,
        pool: BackingRange,
    ) -> Result<Self, SpaceError> {
        Ok(Self {
            tables: Stage1Tables::new(physical, pool)?,
            ram,
            next: ram.base(),
            shadow: Vec::new(),
        })
    }

    /// The root of the tables, for the vCPU.
    pub(crate) fn root(&self) -> TableRoot {
        self.tables.root()
    }

    /// Maps `region` with its permissions onto the next free pages of guest RAM. A failed call
    /// maps nothing and hands out no RAM.
    pub(crate) fn map(&mut self, region: &Region) -> Result<(), SpaceError> {
        let backing = BackingRange::new(self.next, region.size)?;
        let available = self.ram.end().saturating_sub(self.next);
        if region.size > available {
            return Err(SpaceError::OutOfRam {
                size: region.size,
                available,
            });
        }
        self.tables.map(region.start, backing, region.perms)?;
        self.next = backing.end();
        let at = self.shadow.partition_point(|run| run.va < region.start);
        self.shadow.insert(
            at,
            Run {
                va: region.start,
                pa: backing.base(),
                size: region.size,
            },
        );
        Ok(())
    }

    /// The guest-physical address `va` is mapped to, and how many bytes from there on lie in
    /// the same run; `None` if `va` is not mapped.
    fn resolve(&self, va: u64) -> Option<(u64, u64)> {
        // The run that starts at or below `va`, if any, is the one before the first above it.
        let after = self.shadow.partition_point(|run| run.va <= va);
        let run = self.shadow.get(after.checked_sub(1)?)?;
        let offset = va - run.va;
        (offset < run.size).then(|| (run.pa + offset, run.size - offset))
    }

    /// Splits the access `va..va + len` into pieces that each lie in one run and hands them to
    /// `each` as (guest-physical address, offset into the access, length). Stops at the first
    /// address that is not mapped, with a fault there.
    fn pieces(
        &self,
        va: u64,
        len: usize,
        mut each: impl FnMut(u64, usize, usize) -> Result<(), Fault>,
    ) -> Result<(), Fault> {
        let mut done = 0;
        while done < len {
            // An address past the end of the 64-bit space is not mapped either.
            let at = u64::try_from(done)
                .ok()
                .and_then(|done| va.checked_add(done))
                .ok_or(Fault { address: u64::MAX })?;
            let (pa, available) = self.resolve(at).ok_or(Fault { address: at })?;
            let piece = usize::try_from(available)
                .unwrap_or(usize::MAX)
                .min(len - done);
            each(pa, done, piece).map_err(|_| Fault { address: at })?;
            done += piece;
        }
        Ok(())
    }

    /// Fails with the first address of `va..va + len` that is not mapped.
    fn check(&self, va: u64, len: usize) -> Result<(), Fault> {
        self.pieces(va, len, |_, _, _| Ok(()))
    }
}

/// Guest memory by virtual address, as host code sees it: every mapped byte can be read and
/// written, whatever guest code may do with it. An access that touches an address that is not
/// mapped is a [`Fault`] at the first such address and reads or writes nothing; an access of
/// no bytes always succeeds.
impl<M: GuestMemory> GuestMemory for AddressSpace<M> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.check(va, buf.len())?;
        let physical = self.tables.memory();
        self.pieces(va, buf.len(), |pa, done, len| {
            physical.read(pa, &mut buf[done..done + len])
        })
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        self.check(va, data.len())?;
        let physical = self.tables.memory();
        self.pieces(va, data.len(), |pa, done, len| {
            physical.write(pa, &data[done..done + len])
        })
    }
}

#[cfg(test)]
mod tests {
    use starquake_memory::stage1::{Resolved, WalkError};
    use starquake_memory::{FlatMemory, PAGE_SIZE, Perms};

    use super::*;

    /// Guest-physical layout of the tests: a table pool, then RAM, in one flat memory. Nothing
    /// here is a virtual address the tests map.
    const POOL_BASE: u64 = 0x4_0000_0000;
    const POOL_PAGES: u64 = 16;
    const RAM_BASE: u64 = POOL_BASE + POOL_PAGES * PAGE_SIZE;
    const RAM_PAGES: u64 = 8;

    /// Virtual addresses: two neighbouring pages and one far away.
    const LOW: u64 = 0x8000_0000;
    const FAR: u64 = 0x10_0000_0000;

    fn space() -> AddressSpace<FlatMemory> {
        let pages = usize::try_from((POOL_PAGES + RAM_PAGES) * PAGE_SIZE).unwrap();
        let physical = FlatMemory::new(POOL_BASE, pages);
        let pool = BackingRange::new(POOL_BASE, POOL_PAGES * PAGE_SIZE).unwrap();
        let ram = BackingRange::new(RAM_BASE, RAM_PAGES * PAGE_SIZE).unwrap();
        AddressSpace::new(physical, ram, pool).unwrap()
    }

    fn region(start: u64, pages: u64, perms: Perms) -> Region {
        Region {
            start,
            size: pages * PAGE_SIZE,
            perms,
        }
    }

    /// What guest RAM page `n` holds at `offset`, read behind the address space's back.
    fn ram_bytes<const N: usize>(space: &AddressSpace<FlatMemory>, n: u64, offset: u64) -> [u8; N] {
        let mut bytes = [0; N];
        space
            .tables
            .memory()
            .read(RAM_BASE + n * PAGE_SIZE + offset, &mut bytes)
            .unwrap();
        bytes
    }

    #[test]
    fn regions_are_backed_in_mapping_order_with_their_permissions() {
        let mut space = space();
        // Mapped out of address order: the upper neighbour first.
        let regions = [
            region(LOW + PAGE_SIZE, 1, Perms::Read),
            region(FAR, 2, Perms::ReadWrite),
            region(LOW, 1, Perms::ReadExecute),
        ];
        for region in &regions {
            space.map(region).unwrap();
        }
        // The software walker sees each page where guest code will: backed by the next pages
        // of RAM in the order they were mapped, with the region's permissions.
        let walked = |va: u64| space.tables.walk(va);
        let page = |n: u64, perms| {
            Ok(Resolved {
                pa: RAM_BASE + n * PAGE_SIZE,
                perms,
                level: 3,
            })
        };
        assert_eq!(walked(LOW + PAGE_SIZE), page(0, Perms::Read));
        assert_eq!(walked(FAR), page(1, Perms::ReadWrite));
        assert_eq!(walked(FAR + PAGE_SIZE), page(2, Perms::ReadWrite));
        assert_eq!(walked(LOW), page(3, Perms::ReadExecute));
        // The page between the neighbours and the far region is not mapped.
        assert_eq!(
            walked(LOW + 2 * PAGE_SIZE),
            Err(WalkError::Unmapped {
                va: LOW + 2 * PAGE_SIZE,
                level: 3
            })
        );
        assert_eq!(space.root().pa(), POOL_BASE);
    }

    #[test]
    fn host_access_goes_through_the_shadow_whatever_the_guest_may_do() {
        let mut space = space();
        // Neither page is writable by guest code; the host writes both. They are neighbours
        // in virtual addresses and the wrong way round in RAM.
        space.map(&region(LOW + PAGE_SIZE, 1, Perms::None)).unwrap();
        space.map(&region(LOW, 1, Perms::ReadExecute)).unwrap();
        let across = LOW + PAGE_SIZE - 3;
        space.write(across, b"abcdefgh").unwrap();
        // The first three bytes end the lower page (RAM page 1), the rest start the upper
        // (RAM page 0).
        assert_eq!(&ram_bytes::<3>(&space, 1, PAGE_SIZE - 3), b"abc");
        assert_eq!(&ram_bytes::<5>(&space, 0, 0), b"defgh");
        let mut back = [0; 8];
        space.read(across, &mut back).unwrap();
        assert_eq!(&back, b"abcdefgh");
        // A virtual address is not a physical one: the same bytes are not at the virtual
        // address in the physical memory (which does not even reach that far).
        assert!(space.tables.memory().read(across, &mut back).is_err());
    }

    #[test]
    fn an_access_that_leaves_what_is_mapped_faults_there_and_does_nothing() {
        let mut space = space();
        space.map(&region(LOW, 2, Perms::ReadWrite)).unwrap();
        let end = LOW + 2 * PAGE_SIZE;
        space.write(end - 4, b"keep").unwrap();

        // Starts mapped, ends one byte past: the fault is at the first unmapped address, and
        // the mapped part was neither written nor read.
        assert_eq!(space.write(end - 4, b"12345"), Err(Fault { address: end }));
        assert_eq!(&ram_bytes::<4>(&space, 1, PAGE_SIZE - 4), b"keep");
        let mut buf = [0xee; 5];
        assert_eq!(space.read(end - 4, &mut buf), Err(Fault { address: end }));
        assert_eq!(buf, [0xee; 5]);
        // Ending exactly at the end is inside.
        assert_eq!(space.read(end - 4, &mut buf[..4]), Ok(()));
        assert_eq!(&buf[..4], b"keep");

        // Starts unmapped: below, above, and where adding the length overflows.
        for address in [LOW - 1, end, 0, u64::MAX] {
            assert_eq!(
                space.read(address, &mut buf),
                Err(Fault { address }),
                "{address:#x}"
            );
            assert_eq!(
                space.write(address, &buf),
                Err(Fault { address }),
                "{address:#x}"
            );
        }
        // An access of no bytes touches nothing, so it cannot fault.
        assert_eq!(space.read(end, &mut []), Ok(()));
        assert_eq!(space.write(u64::MAX, &[]), Ok(()));
    }

    #[test]
    fn ram_runs_out_exactly_when_it_is_used_up() {
        let mut space = space();
        // One page too many for an empty space.
        assert_eq!(
            space.map(&region(FAR, RAM_PAGES + 1, Perms::Read)),
            Err(SpaceError::OutOfRam {
                size: (RAM_PAGES + 1) * PAGE_SIZE,
                available: RAM_PAGES * PAGE_SIZE
            })
        );
        // Exactly all of it fits, and then not one page more.
        space.map(&region(FAR, RAM_PAGES, Perms::Read)).unwrap();
        assert_eq!(
            space.map(&region(LOW, 1, Perms::Read)),
            Err(SpaceError::OutOfRam {
                size: PAGE_SIZE,
                available: 0
            })
        );
        assert_eq!(
            SpaceError::OutOfRam {
                size: 0x1000,
                available: 0
            }
            .to_string(),
            "mapping 0x1000 bytes needs more guest RAM than the 0x0 bytes left"
        );
    }

    #[test]
    fn a_refused_mapping_maps_nothing_and_uses_no_ram() {
        let mut space = space();
        space.map(&region(LOW, 1, Perms::Read)).unwrap();
        let refused = [
            // Not whole pages, at either end; no pages at all.
            (
                Region {
                    start: FAR + 1,
                    size: PAGE_SIZE,
                    perms: Perms::Read,
                },
                MapError::MisalignedAddress { address: FAR + 1 },
            ),
            (
                Region {
                    start: FAR,
                    size: PAGE_SIZE + 1,
                    perms: Perms::Read,
                },
                MapError::MisalignedSize {
                    size: PAGE_SIZE + 1,
                },
            ),
            (region(FAR, 0, Perms::Read), MapError::Empty),
            // On top of a page that is mapped.
            (
                region(LOW, 1, Perms::ReadWrite),
                MapError::AlreadyMapped { va: LOW },
            ),
        ];
        for (region, error) in refused {
            assert_eq!(
                space.map(&region),
                Err(SpaceError::Map(error)),
                "{region:?}"
            );
        }
        assert_eq!(
            SpaceError::Map(MapError::AlreadyMapped { va: LOW }).to_string(),
            "the guest's translation tables refused a mapping: 0x80000000 is already mapped"
        );
        // The next mapping gets the RAM page right after the first one's, and the page that
        // was refused a second mapping kept its permissions.
        space.map(&region(FAR, 1, Perms::ReadWrite)).unwrap();
        assert_eq!(space.tables.walk(FAR).unwrap().pa, RAM_BASE + PAGE_SIZE);
        assert_eq!(space.tables.walk(LOW).unwrap().perms, Perms::Read);
        assert_eq!(space.read(FAR + 1, &mut [0; 1]), Ok(()));
    }

    #[test]
    fn a_pool_the_memory_does_not_hold_is_refused() {
        let physical = FlatMemory::new(POOL_BASE, usize::try_from(PAGE_SIZE).unwrap());
        let pool = BackingRange::new(POOL_BASE, 2 * PAGE_SIZE).unwrap();
        let ram = BackingRange::new(RAM_BASE, PAGE_SIZE).unwrap();
        let error = AddressSpace::new(physical, ram, pool).unwrap_err();
        assert_eq!(
            error,
            SpaceError::Map(MapError::Memory(Fault {
                address: POOL_BASE + PAGE_SIZE
            }))
        );
    }
}

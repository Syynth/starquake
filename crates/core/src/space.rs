//! One guest address space as the host keeps it: the stage-1 tables guest code runs on, the RAM
//! that backs its pages, and a shadow of what is mapped where, through which the host reads and
//! writes guest memory by virtual address.
//!
//! Design: `docs/specs/execution-model.md`, "Guest memory". The kernel's address space
//! (`starquake_kernel::memory`) decides what is mapped and with what permission, and tells this
//! one through [`Mapper`]; this one makes it real:
//!
//! - **Guest code** reaches memory through [`Stage1Tables`], with the permissions the kernel
//!   gave each page.
//! - **Pages are backed when they are first touched** ("Sparse and lazy memory": "A reservation
//!   creates address-space bookkeeping but no stage-1 entries; backing pages are allocated on
//!   first touch"). A mapping the kernel makes is recorded in the shadow and gets no table
//!   entry and no RAM. The first guest access to one of its pages faults; the core loop asks
//!   [`AddressSpace::touch`], which gives the page a zeroed page of RAM and a table entry if
//!   the mapping allows the access, and the guest resumes. So a process may map far more than
//!   the host backs, as a libnx program does at start-up (a 3.2 GiB heap, of which it touches
//!   a few pages).
//! - **Host code** (the loader, the kernel reading and writing a system call's buffers) goes
//!   through [`GuestMemory`], which "resolves addresses with a host-side shadow of the address
//!   space, not by walking the hardware tables" ("Host access to guest memory"). A page that is
//!   mapped and not backed reads as zero, and is backed when the host writes to it.
//! - **After `unmap` or `protect`** the tables record a TLB flush that must run on a vCPU
//!   before the guest runs again. The tables hand it out through [`AddressSpace::tables`]; the
//!   core loop runs it after every system call (`crate::machine`).
//!
//! # Shared memory
//!
//! The kernel's [`Mapper::allocate_shared`] and [`Mapper::map_shared`] (its
//! `starquake_kernel::memory::shared` module docs, "Where the pages come from"):
//!
//! - **The pages are one run of guest RAM** ([`AddressSpace::allocate_shared`]), taken from
//!   the same place as touched pages, so they read as zero (RAM starts zeroed and is never
//!   handed out twice), or from a run a shared memory gave back, zeroed then (below). Eden
//!   takes them as one run too
//!   (`MemoryManager().AllocateAndOpenContinuous` in `KSharedMemory::Initialize`,
//!   `eden/src/core/hle/kernel/k_shared_memory.cpp` @ cf654052). The service reaches them
//!   through the returned [`SharedPages`], by guest-physical address.
//! - **A shared mapping gets every table entry at once** ([`AddressSpace::map_shared`]), with
//!   the permission the kernel asks for (read, or read and write: the kernel refuses any other,
//!   `is_shared_memory_permission`), and every page is backed in the shadow. The pages exist
//!   before the mapping, so there is nothing for a touch to add; a fault on a shared page is
//!   always [`Touch::Refused`]. A shared page is never executable through a mapping (the kernel
//!   allows no execute permission for shared memory, and its `Shared` state has none of the
//!   flags its reprotecting calls require: `CODE`, `CAN_REPROTECT`, `CAN_TRANSFER`), so a host
//!   write to it needs no instruction-cache maintenance.
//! - **Unmapping a shared mapping** is [`Mapper::unmap`], as for any mapping: the table
//!   entries go and a flush is owed. The pages keep what they hold, for the service and for
//!   any other mapping of them.
//! - **Running out** is [`SpaceError::OutOfRam`], which [`SpaceMapper`] reports as
//!   [`MapperError::OutOfMemory`], and the kernel answers `ResultOutOfMemory`, as Eden does when
//!   its page heap has no run of the size (`R_UNLESS(m_physical_address != 0,
//!   ResultOutOfMemory)` in `KSharedMemory::Initialize`,
//!   `eden/src/core/hle/kernel/k_shared_memory.cpp` @ cf654052). Every other refusal is
//!   [`MapperError::Refused`], the kernel's own failure. Tests:
//!   `running_out_of_ram_by_exactly_one_page_is_out_of_memory_and_not_a_panic`
//!   (`space/tests/kernel.rs`) and `the_kernels_mapper_reports_a_shared_memory_refusal_in_words`
//!   (`space/tests/shared.rs`).
//! - **Giving the pages back** is [`AddressSpace::release_shared`] ([`Mapper::release_shared`]),
//!   which the kernel calls when the last reference to the shared memory has gone (its
//!   `KSharedMemory::Finalize`). It is refused, with nothing changed, unless the pages are
//!   exactly a run handed out and not given back yet ([`SpaceError::NotShared`]), and then
//!   unless no mapping of them is left ([`SpaceError::StillMapped`]: the kernel's mappings each
//!   hold a reference, so it never asks while one is left, and this layer checks it anyway).
//!   The pages are not handed out again at once: a vCPU may still reach them through a TLB
//!   entry of an unmapping whose flush has not run (CLAUDE.md: a freed resource is not
//!   reusable until every pending follow-up on it has run). They wait until the tables owe no
//!   flush at all ([`Stage1Tables::has_pending_flushes`]), which covers every unmapping of
//!   them, all made before the release; the next [`AddressSpace::allocate_shared`] that finds
//!   no flush owed makes them reusable. Then they are handed out again to a shared memory
//!   whose size fits in them, zeroed first. Tests (in `space/tests/shared.rs`):
//!   `given_back_pages_wait_for_every_flush_and_come_back_zeroed`,
//!   `a_release_is_refused_unless_the_pages_are_a_run_handed_out_and_unmapped` and, through the
//!   kernel, `the_last_reference_gives_the_pages_back_and_a_close_while_mapped_does_not`
//!   (`space/tests/kernel.rs`).
//! - **A service may still hold the pages' [`SharedPages`]** after they are given back: they
//!   reach guest RAM by guest-physical address, and are not invalidated. No server lets go of a
//!   shared memory it made today (`IpcContext::create_shared_memory`: the server keeps the
//!   creator's reference for as long as it lives), so its last reference never goes while the
//!   server holds its pages; a server that let go would have to drop its [`SharedPages`] first.
//! - **Stricter: a shared mapping must name exactly a run [`AddressSpace::allocate_shared`]
//!   handed out**, or it is [`SpaceError::NotShared`]. The run is recorded in the shadow of the
//!   one address space that handed it out, so this holds only while a run has one process:
//!   there the kernel always names a shared memory's whole run (`SharedMemory::backing`) and
//!   maps it in the address space that made it (a server's `IpcContext::create_shared_memory`
//!   allocates through the kernel address space its dispatcher holds, which drives this one
//!   through [`SpaceMapper`]), so no guest reaches this. A second process mapping the same
//!   shared memory would be refused here, and the kernel would answer `SvcError::Internal`,
//!   where Eden maps the pages of its global page heap into any process (`MapPageGroup` in
//!   `KSharedMemory::Map`, `eden/src/core/hle/kernel/k_shared_memory.cpp` @ cf654052). See
//!   the PROVISIONAL list below. Test:
//!   `pages_that_are_not_one_shared_memory_are_refused_and_change_nothing`.
//!
//! # Aliases
//!
//! The kernel's [`Mapper::map_alias`], for `MapMemory` (`starquake_kernel::memory::space`'s
//! `alias` module): [`AddressSpace::map_alias`] maps a range a second time over the same RAM.
//!
//! - **Every page of the source gets backing first**, then the alias gets every table entry at
//!   once, to the source's RAM, with the permission the kernel asks for. A source page with no
//!   backing yet has none to share, and backing it later would give the two addresses
//!   different RAM, so it is backed now: a page of RAM handed out as a touch would hand it out
//!   (zeroed, never handed out twice), with the source's permission. A fault on an alias page
//!   is then always [`Touch::Refused`], as for shared memory.
//! - **The alias is read-write, never executable** (the kernel maps it `UserReadWrite`), so a
//!   host write through it needs no instruction-cache maintenance.
//! - **Unmapping the alias** is [`Mapper::unmap`], as for any mapping: its table entries go and
//!   a flush is owed; the RAM stays the source's, with what it holds.
//! - **A refused alias** changes no mapping and no table entry, and owes no flush. Source
//!   pages it backed before the refusal keep their backing: a backed page reads zero until it
//!   is written, as an unbacked one does, so no access can tell. Tests (in
//!   `space/tests/alias.rs`): `an_alias_is_refused_with_its_checks_in_order_and_maps_nothing`
//!   and `an_alias_the_tables_refuse_part_way_unmaps_what_it_mapped`.
//!
//! # The Map recording point
//!
//! The observability spec's "the address space's mutators each call the recorder once" ("The
//! event stream"): each change to the mappings the kernel asks [`SpaceMapper`] for and this
//! address space makes ([`Mapper::map`], [`Mapper::map_shared`] and [`Mapper::map_alias`] as
//! [`MapKind::Map`], [`Mapper::protect`] as [`MapKind::Protect`], [`Mapper::unmap`] as
//! [`MapKind::Unmap`]) is one [`EventKind::Map`], with the range asked for and the guest's raw
//! permission word ([`permission_word`]; zero for an unmapping), sent to the sink a run
//! attaches ([`AddressSpace::record_maps`]). A refused change sends nothing; handing out and
//! giving back shared memory ([`Mapper::allocate_shared`], [`Mapper::release_shared`]) and the
//! backing of a page on first touch change no mapping and send nothing (a touch is the core
//! loop's fault path). The sink is called once the change is made, with no lock of this
//! address space held, and under the kernel's lock, which every mapper call holds, so the
//! recorder's `seq` keeps the kernel's order. A run attaches its recorder before the kernel
//! boots the process (`crate::machine`, `GuestSpace::record_maps`): the mappings the loader
//! makes before that (the program's modules) are not recorded; the crash report's memory map
//! lists them. Tests (in `space/tests/record.rs`):
//! `each_mutation_the_kernel_asks_for_is_one_map_event`,
//! `a_refused_mutation_records_nothing`,
//! `the_sink_is_called_with_no_lock_of_the_address_space_held`; and through a run,
//! `the_thread_local_page_the_kernel_maps_at_boot_is_a_map_event_in_the_dump`
//! (`crates/core/tests/map_events.rs`).
//!
//! # Snapshots
//!
//! The pages that were touched are saved and restored by the `snapshot` module, beside the
//! kernel's address space ([`snapshot::restore`]). A restore records the shared memory runs
//! where they were, before the kernel maps them, so [`AddressSpace::map_shared`] accepts each
//! one in the new address space; and a page the snapshot restored that waits for its mapping
//! (`Shadow::pending`) is backed with its own RAM, not new RAM, by whatever backs it first
//! (an alias of it, a touch, a host write) or, for the rest, by
//! [`AddressSpace::finish_restore`].
//!
//! PROVISIONAL, and what replaces each part:
//!
//! - **Backing is handed out from the start of guest RAM, a page at a time (a run at a time
//!   for shared memory), and never returned**: an unmapped page's RAM is not reused. The one
//!   exception is a shared memory's run once given back, which only a later shared memory
//!   reuses (the first run, in address order, it fits in; what is left of the run stays
//!   free); given-back runs are not merged with their neighbours, nor with the RAM never
//!   handed out. The backing pools of the execution model replace it, sized as the kernel's
//!   pool accounting says. Until then guest RAM is sized
//!   from the program a run loads, plus a working allowance (`crate::system`'s `ram` module),
//!   which is usually far smaller than what the kernel lets a process reserve, and a guest that
//!   touches more ends its run as an emulator error.
//! - **A shared memory can be mapped only in the address space that handed it out** (or that
//!   a snapshot restored it into, which records the run the same way): its run is recorded in
//!   that [`AddressSpace`]'s shadow, and its pages are taken from that address space's RAM. System-wide guest RAM and one record of the shared runs (the execution
//!   model's backing pools), instead of per-`AddressSpace` ones, replace it, so that any
//!   process's address space can map any shared memory.
//! - **An alias backs its whole source at once**, however large: the RAM a guest can force is
//!   bounded by the source, which the kernel has already counted against the pool, and guest
//!   RAM refuses what it cannot hold ([`SpaceError::OutOfRam`], checked before anything is
//!   backed). The backing pools, which can name the RAM a page will get before it is touched,
//!   replace it with backing on first touch.
//! - **One page per fault.** A guest that touches a large mapping takes one exit per page.
//!   Backing a run of pages at once is a measurement away.
//! - **Host access ignores permissions**, which is what lets the loader write a code page that
//!   guest code may only read and execute; nothing checks that a system call's buffer is
//!   readable or writable by the guest. The kernel's block map knows; a later slice checks.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use starquake_kernel::memory::{Mapper, MapperError, MemoryPermission, SharedPages};
use starquake_memory::stage1::{Stage1Tables, TableRoot, VA_SPACE_END};
use starquake_memory::{
    Access, BackingRange, Fault, GuestMemory, MapError, PAGE_SIZE, Perms, Translation,
};
use starquake_trace::{EventKind, MapKind};

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "a run takes no snapshot yet: the run-level save and restore (observability \
                  spec, \"Order of work\", step 6) is what calls these"
    )
)]
pub(crate) mod snapshot;

/// Why a guest address space could not be built or changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SpaceError {
    /// The translation tables refused the mapping, or could not be created; or the range is
    /// not one the address space can map, unmap or reprotect.
    #[error("the guest's translation tables refused a mapping: {0}")]
    Map(#[from] MapError),
    /// Guest RAM has fewer bytes left than a page, or a shared memory, needs.
    #[error("mapping {size:#x} bytes needs more guest RAM than the {available:#x} bytes left")]
    OutOfRam {
        /// Bytes needed.
        size: u64,
        /// Bytes of guest RAM left where the request may be taken from: the RAM never handed
        /// out, and for a shared memory the largest of that and each given-back run that may
        /// be handed out again.
        available: u64,
    },
    /// Pages a shared mapping names are not the pages of one shared memory this address
    /// space handed out ([`AddressSpace::allocate_shared`]).
    #[error(
        "the {size:#x} bytes of guest RAM at {base:#x} are not a shared memory this address \
         space handed out"
    )]
    NotShared {
        /// The first guest-physical address of the pages.
        base: u64,
        /// Their size in bytes.
        size: u64,
    },
    /// Pages of a shared memory to be given back are still mapped
    /// ([`AddressSpace::release_shared`]).
    #[error(
        "the {size:#x} bytes of guest RAM at {base:#x} are still mapped: the lowest address \
         that maps a page of them is {va:#x}"
    )]
    StillMapped {
        /// The first guest-physical address of the pages.
        base: u64,
        /// Their size in bytes.
        size: u64,
        /// The lowest virtual address a page of them is mapped at.
        va: u64,
    },
    /// Guest RAM refused the write that zeroes given-back pages before they are handed out
    /// again.
    #[error("guest RAM refused the write that zeroes pages handed out again: {0}")]
    Zeroing(Fault),
}

/// What the kernel mapped from some address up to `end`: what guest code may do there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Mapped {
    /// One past the last address of the mapping.
    end: u64,
    perms: Perms,
}

/// What an access by the guest to a page it faulted on came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Touch {
    /// The page is mapped, the mapping allows the access, and the page now has backing: the
    /// guest can retry.
    Backed,
    /// The guest may not make this access: nothing is mapped there, the mapping does not allow
    /// it, or the page is backed already (so the fault is not for want of backing).
    Refused,
    /// The page should have been backed and could not be: the emulator's failure, not the
    /// guest's.
    Failed(SpaceError),
}

/// The part of an address space that changes: what is mapped, what is backed, and how much RAM
/// is handed out.
#[derive(Debug)]
struct Shadow {
    /// The first guest-physical address of RAM not handed out yet.
    next: u64,
    /// What the kernel mapped, by first address, without overlaps.
    mapped: BTreeMap<u64, Mapped>,
    /// The pages that have backing: virtual page address to guest-physical page address.
    backed: BTreeMap<u64, u64>,
    /// The runs of RAM handed out as shared memory, by first guest-physical address: one past
    /// the last address of each.
    shared: BTreeMap<u64, u64>,
    /// Runs of shared memory given back while the tables owed a flush, oldest first: not
    /// reusable until the tables owe none (module docs, "Shared memory").
    released: Vec<BackingRange>,
    /// Runs given back that a shared memory may be handed out from again, by first
    /// guest-physical address: one past the last address of each. They hold what they held;
    /// a run is zeroed when it is handed out.
    reusable: BTreeMap<u64, u64>,
    /// The pages [`AddressSpace::map_alias`] mapped: virtual page addresses whose RAM is
    /// another page's (their source's), so that a snapshot keeps each page of RAM once
    /// (`snapshot` module docs).
    aliased: BTreeSet<u64>,
    /// Pages a snapshot restored that wait for their mapping: virtual page address to the
    /// guest-physical page that holds what they held (`snapshot` module docs, "Restoring").
    /// Empty except between [`AddressSpace::begin_restore`] and
    /// [`AddressSpace::finish_restore`].
    pending: BTreeMap<u64, u64>,
}

impl Shadow {
    /// The mapping that holds `va`, with its first address.
    fn mapping(&self, va: u64) -> Option<(u64, Mapped)> {
        self.mapped
            .range(..=va)
            .next_back()
            .map(|(start, mapped)| (*start, *mapped))
            .filter(|(_, mapped)| va < mapped.end)
    }

    /// The first address in `va..end` that no mapping holds, if there is one.
    fn first_unmapped(&self, va: u64, end: u64) -> Option<u64> {
        let mut at = va;
        while at < end {
            match self.mapping(at) {
                // A mapping is never empty, so `at` moves up each round.
                Some((_, mapped)) => at = mapped.end,
                None => return Some(at),
            }
        }
        None
    }

    /// The first address in `va..end` that a mapping holds, if there is one: `va` itself, or
    /// where the first mapping above it starts.
    fn first_mapped(&self, va: u64, end: u64) -> Option<u64> {
        if self.mapping(va).is_some() {
            return Some(va);
        }
        self.mapped.range(va..end).next().map(|(start, _)| *start)
    }

    /// Makes `at` the boundary between two mappings, if it lies inside one.
    fn split(&mut self, at: u64) {
        if let Some((start, mapped)) = self.mapping(at)
            && start < at
        {
            self.mapped.insert(start, Mapped { end: at, ..mapped });
            self.mapped.insert(at, mapped);
        }
    }

    /// The runs of backed pages in `va..end`, each as (first address, size), in address order.
    fn backed_runs(&self, va: u64, end: u64) -> Vec<(u64, u64)> {
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for page in self.backed.range(va..end).map(|(page, _)| *page) {
            match runs.last_mut() {
                Some((start, size)) if *start + *size == page => *size += PAGE_SIZE,
                _ => runs.push((page, PAGE_SIZE)),
            }
        }
        runs
    }
}

/// One piece of a host access: where its bytes are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Piece {
    /// In guest RAM, at this guest-physical address.
    Backed(u64),
    /// In a page that is mapped and has no backing yet: zero.
    Unbacked,
}

/// Splits the access `va..va + len` into pieces and hands them to `each` as (where the piece
/// is, offset into the access, length). `resolve` says where the bytes at an address are and
/// how many bytes from there on are in the same place; `None` is an address that is not
/// mapped. Stops at the first such address, with a fault there.
///
/// Every piece is at least one byte: a `resolve` that answers "zero bytes here" would
/// otherwise stall the walk for ever, so it is a fault at that address instead.
fn pieces<T>(
    va: u64,
    len: usize,
    resolve: impl Fn(u64) -> Option<(T, u64)>,
    mut each: impl FnMut(T, usize, usize) -> Result<(), Fault>,
) -> Result<(), Fault> {
    let mut done = 0;
    while done < len {
        // An address past the end of the 64-bit space is not mapped either.
        let at = u64::try_from(done)
            .ok()
            .and_then(|done| va.checked_add(done))
            .ok_or(Fault { address: u64::MAX })?;
        let (place, available) = resolve(at).ok_or(Fault { address: at })?;
        let piece = usize::try_from(available)
            .unwrap_or(usize::MAX)
            .min(len - done);
        if piece == 0 {
            return Err(Fault { address: at });
        }
        each(place, done, piece).map_err(|_| Fault { address: at })?;
        done += piece;
    }
    Ok(())
}

/// [`PAGE_SIZE`] as a step between pages.
const PAGE_STEP: usize = 1 << PAGE_SIZE.trailing_zeros();

/// The first address of the page that holds `va`.
const fn page_of(va: u64) -> u64 {
    va & !(PAGE_SIZE - 1)
}

/// One guest address space (see the module docs). `M` is the guest-physical memory the tables
/// and the backing live in.
#[derive(Debug)]
pub(crate) struct AddressSpace<M> {
    tables: Stage1Tables<M>,
    /// Guest RAM: where backing comes from.
    ram: BackingRange,
    shadow: Mutex<Shadow>,
    /// Where [`SpaceMapper`]'s recording point sends its events, once a run attaches its
    /// recorder ([`AddressSpace::record_maps`]).
    maps: Mutex<Option<MapSink>>,
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
            shadow: Mutex::new(Shadow {
                next: ram.base(),
                mapped: BTreeMap::new(),
                backed: BTreeMap::new(),
                shared: BTreeMap::new(),
                released: Vec::new(),
                reusable: BTreeMap::new(),
                aliased: BTreeSet::new(),
                pending: BTreeMap::new(),
            }),
            maps: Mutex::new(None),
        })
    }

    /// The root of the tables, for the vCPU.
    pub(crate) fn root(&self) -> TableRoot {
        self.tables.root()
    }

    /// The tables, for the backend to run their pending TLB flushes (see the module docs).
    pub(crate) fn tables(&self) -> &Stage1Tables<M> {
        &self.tables
    }

    fn shadow(&self) -> MutexGuard<'_, Shadow> {
        // A poisoned lock means a panic elsewhere; the shadow is plain data, so carry on.
        self.shadow.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The end of `size` bytes at `va`, if they are a non-empty run of whole pages inside the
    /// address space the tables translate.
    fn range_end(va: u64, size: u64) -> Result<u64, MapError> {
        if !va.is_multiple_of(PAGE_SIZE) {
            return Err(MapError::MisalignedAddress { address: va });
        }
        if !size.is_multiple_of(PAGE_SIZE) {
            return Err(MapError::MisalignedSize { size });
        }
        if size == 0 {
            return Err(MapError::Empty);
        }
        va.checked_add(size)
            .filter(|end| *end <= VA_SPACE_END)
            .ok_or(MapError::VirtualRange { va, size })
    }

    /// Records that guest code may use `size` bytes at `va` as `perms` allows. Nothing may be
    /// mapped there. No page gets backing or a table entry until it is touched.
    pub(crate) fn map(&self, va: u64, size: u64, perms: Perms) -> Result<(), SpaceError> {
        let end = Self::range_end(va, size)?;
        let mut shadow = self.shadow();
        if let Some(taken) = shadow.first_mapped(va, end) {
            return Err(MapError::AlreadyMapped { va: taken }.into());
        }
        shadow.mapped.insert(va, Mapped { end, perms });
        Ok(())
    }

    /// Checks that all of `va..end` is mapped, and makes both ends boundaries between
    /// mappings.
    fn isolate(shadow: &mut Shadow, va: u64, end: u64) -> Result<(), SpaceError> {
        if let Some(unmapped) = shadow.first_unmapped(va, end) {
            return Err(MapError::NotMapped { va: unmapped }.into());
        }
        shadow.split(va);
        shadow.split(end);
        Ok(())
    }

    /// Takes `size` bytes at `va` away from guest code. All of it must be mapped. Pages that
    /// had backing lose their table entries (the tables record the flush that is owed); their
    /// RAM is not reused.
    pub(crate) fn unmap(&self, va: u64, size: u64) -> Result<(), SpaceError> {
        let end = Self::range_end(va, size)?;
        let mut shadow = self.shadow();
        Self::isolate(&mut shadow, va, end)?;
        for (run, run_size) in shadow.backed_runs(va, end) {
            self.tables.unmap(run, run_size)?;
            shadow
                .backed
                .retain(|page, _| !(run..run + run_size).contains(page));
        }
        shadow.mapped.retain(|start, _| !(va..end).contains(start));
        shadow.aliased.retain(|page| !(va..end).contains(page));
        Ok(())
    }

    /// Changes what guest code may do with `size` bytes at `va`, which keep what they hold.
    /// All of it must be mapped. Pages that have backing get new table entries (the tables
    /// record the flush that is owed).
    pub(crate) fn protect(&self, va: u64, size: u64, perms: Perms) -> Result<(), SpaceError> {
        let end = Self::range_end(va, size)?;
        let mut shadow = self.shadow();
        Self::isolate(&mut shadow, va, end)?;
        for (run, run_size) in shadow.backed_runs(va, end) {
            self.tables.protect(run, run_size, perms)?;
        }
        for (_, mapped) in shadow.mapped.range_mut(va..end) {
            mapped.perms = perms;
        }
        Ok(())
    }

    /// The next `size` bytes of RAM not handed out yet, which the caller takes by moving
    /// `shadow.next` past them once nothing else can fail. RAM starts out zeroed and is never
    /// handed out twice, so they read as zero.
    ///
    /// The size is checked before the room: a size that is not whole pages is
    /// [`MapError::MisalignedSize`] however much RAM is left; then [`SpaceError::OutOfRam`]
    /// when fewer than `size` bytes are left. A size of zero always fits, and is
    /// [`MapError::Empty`] ([`BackingRange::new`] refuses it).
    fn next_ram(&self, shadow: &Shadow, size: u64) -> Result<BackingRange, SpaceError> {
        if !size.is_multiple_of(PAGE_SIZE) {
            return Err(MapError::MisalignedSize { size }.into());
        }
        let available = self.ram.end().saturating_sub(shadow.next);
        if available < size {
            return Err(SpaceError::OutOfRam { size, available });
        }
        Ok(BackingRange::new(shadow.next, size)?)
    }

    /// Gives the page at `page` a zeroed page of RAM and a table entry with `perms`; or, when
    /// a snapshot restored what it held and it waits for its mapping (`Shadow::pending`), the
    /// page of RAM that holds it.
    fn back(&self, shadow: &mut Shadow, page: u64, perms: Perms) -> Result<u64, SpaceError> {
        if let Some(pa) = shadow.pending.get(&page).copied() {
            self.tables
                .map(page, BackingRange::new(pa, PAGE_SIZE)?, perms)?;
            shadow.pending.remove(&page);
            shadow.backed.insert(page, pa);
            return Ok(pa);
        }
        let pages = self.next_ram(shadow, PAGE_SIZE)?;
        self.tables.map(page, pages, perms)?;
        shadow.next = pages.end();
        shadow.backed.insert(page, pages.base());
        Ok(pages.base())
    }

    /// Hands out `size` bytes of RAM as the pages of one shared memory: one run, zeroed, that
    /// nothing else is handed out from, and that [`Self::map_shared`] maps. Maps nothing
    /// itself.
    ///
    /// First, when the tables owe no flush, every run given back so far becomes reusable
    /// ([`Self::release_shared`]); that changes nothing a caller can see, and stays done when
    /// the call is refused. Then the pages are the first reusable run, in address order, that
    /// holds `size` bytes (its start, zeroed; the rest of it stays reusable), or else RAM never
    /// handed out ([`Self::next_ram`]). Refused, with nothing handed out: as
    /// [`Self::next_ram`] says, where [`SpaceError::OutOfRam`]'s `available` is also the
    /// largest reusable run when that is larger; and [`SpaceError::Zeroing`] when RAM refuses
    /// the zeroing write.
    pub(crate) fn allocate_shared(&self, size: u64) -> Result<BackingRange, SpaceError> {
        let mut shadow = self.shadow();
        if !shadow.released.is_empty() && !self.tables.has_pending_flushes() {
            let released = std::mem::take(&mut shadow.released);
            shadow
                .reusable
                .extend(released.iter().map(|run| (run.base(), run.end())));
        }
        let reused = shadow
            .reusable
            .iter()
            .find(|(base, end)| **end - **base >= size)
            .map(|(base, end)| (*base, *end));
        let pages = match reused {
            Some((base, end)) if size.is_multiple_of(PAGE_SIZE) && size != 0 => {
                let pages = BackingRange::new(base, size)?;
                self.zero(pages)?;
                shadow.reusable.remove(&base);
                if pages.end() < end {
                    shadow.reusable.insert(pages.end(), end);
                }
                pages
            }
            _ => {
                let pages = self.next_ram(&shadow, size).map_err(|error| match error {
                    SpaceError::OutOfRam { size, available } => SpaceError::OutOfRam {
                        size,
                        available: shadow
                            .reusable
                            .iter()
                            .map(|(base, end)| end - base)
                            .fold(available, u64::max),
                    },
                    other => other,
                })?;
                shadow.next = pages.end();
                pages
            }
        };
        shadow.shared.insert(pages.base(), pages.end());
        Ok(pages)
    }

    /// Writes zeros over `pages`, a page at a time.
    fn zero(&self, pages: BackingRange) -> Result<(), SpaceError> {
        const ZEROS: [u8; PAGE_STEP] = [0; PAGE_STEP];
        let memory = self.tables.memory();
        (pages.base()..pages.end())
            .step_by(PAGE_STEP)
            .try_for_each(|page| memory.write(page, &ZEROS))
            .map_err(SpaceError::Zeroing)
    }

    /// Takes back the pages of a shared memory whose last reference has gone (module docs,
    /// "Shared memory"): they are handed out again ([`Self::allocate_shared`]) only once the
    /// tables owe no flush.
    ///
    /// Checked in this order, and refused with nothing changed: the pages must be exactly one
    /// run [`Self::allocate_shared`] handed out and not given back yet
    /// ([`SpaceError::NotShared`]); then no page of them may be mapped
    /// ([`SpaceError::StillMapped`], at the lowest virtual address one is). The check walks
    /// every backed page, which the release of a shared memory, once per shared memory, can
    /// afford.
    pub(crate) fn release_shared(&self, pages: BackingRange) -> Result<(), SpaceError> {
        let mut shadow = self.shadow();
        let (base, end, size) = (pages.base(), pages.end(), pages.size());
        if shadow.shared.get(&base) != Some(&end) {
            return Err(SpaceError::NotShared { base, size });
        }
        if let Some(va) = shadow
            .backed
            .iter()
            .find(|(_, pa)| (base..end).contains(*pa))
            .map(|(va, _)| *va)
        {
            return Err(SpaceError::StillMapped { base, size, va });
        }
        shadow.shared.remove(&base);
        shadow.released.push(pages);
        Ok(())
    }

    /// Records that guest code may use the pages of the shared memory `pages` at `va` as
    /// `perms` allows, and gives every page its table entry at once: the pages exist already,
    /// so nothing waits for a touch. Nothing may be mapped there.
    ///
    /// Checked in this order, and refused with nothing changed: the range
    /// ([`MapError`], as for [`Self::map`]); then the pages, which must be exactly one run
    /// [`Self::allocate_shared`] handed out ([`SpaceError::NotShared`]); then what is mapped
    /// ([`MapError::AlreadyMapped`]); then the tables ([`MapError::OutOfTablePages`] among
    /// them).
    pub(crate) fn map_shared(
        &self,
        va: u64,
        pages: BackingRange,
        perms: Perms,
    ) -> Result<(), SpaceError> {
        let end = Self::range_end(va, pages.size())?;
        let mut shadow = self.shadow();
        if shadow.shared.get(&pages.base()) != Some(&pages.end()) {
            return Err(SpaceError::NotShared {
                base: pages.base(),
                size: pages.size(),
            });
        }
        if let Some(taken) = shadow.first_mapped(va, end) {
            return Err(MapError::AlreadyMapped { va: taken }.into());
        }
        self.tables.map(va, pages, perms)?;
        shadow.mapped.insert(va, Mapped { end, perms });
        let virtual_pages = (va..end).step_by(PAGE_STEP);
        let physical_pages = (pages.base()..pages.end()).step_by(PAGE_STEP);
        shadow.backed.extend(virtual_pages.zip(physical_pages));
        Ok(())
    }

    /// Records that guest code may use `size` bytes at `va` as `perms` allows, and that they
    /// are the RAM of the `size` bytes at `source` (module docs, "Aliases"): every source page
    /// is backed, and every alias page gets its table entry to the same RAM at once. Nothing
    /// may be mapped at `va`.
    ///
    /// Checked in this order, and refused with nothing mapped: both ranges ([`MapError`], as
    /// for [`Self::map`]); then what is mapped: all of the source
    /// ([`MapError::NotMapped`] at its first address that is not), nothing at `va`
    /// ([`MapError::AlreadyMapped`]); then the RAM the source's unbacked pages need
    /// ([`SpaceError::OutOfRam`]); then the tables. A refusal by the tables part-way unmaps the
    /// alias pages it mapped (the unmap owes a flush, as any does).
    pub(crate) fn map_alias(
        &self,
        va: u64,
        source: u64,
        size: u64,
        perms: Perms,
    ) -> Result<(), SpaceError> {
        let end = Self::range_end(va, size)?;
        let source_end = Self::range_end(source, size)?;
        let mut shadow = self.shadow();
        if let Some(unmapped) = shadow.first_unmapped(source, source_end) {
            return Err(MapError::NotMapped { va: unmapped }.into());
        }
        if let Some(taken) = shadow.first_mapped(va, end) {
            return Err(MapError::AlreadyMapped { va: taken }.into());
        }
        // Back every source page, after checking guest RAM has room for all of them: a page a
        // snapshot restored has its RAM already (`Shadow::pending`).
        let unbacked: Vec<u64> = (source..source_end)
            .step_by(PAGE_STEP)
            .filter(|page| !shadow.backed.contains_key(page))
            .collect();
        let fresh = unbacked
            .iter()
            .filter(|page| !shadow.pending.contains_key(page))
            .count();
        let needed = u64::try_from(fresh)
            .ok()
            .and_then(|pages| pages.checked_mul(PAGE_SIZE))
            .unwrap_or(u64::MAX);
        let available = self.ram.end().saturating_sub(shadow.next);
        if available < needed {
            return Err(SpaceError::OutOfRam {
                size: needed,
                available,
            });
        }
        for page in unbacked {
            let page_perms = shadow
                .mapping(page)
                .map(|(_, mapped)| mapped.perms)
                .ok_or(MapError::NotMapped { va: page })?;
            self.back(&mut shadow, page, page_perms)?;
        }
        // Map each alias page to its source page's RAM.
        let mut aliased: Vec<(u64, u64)> = Vec::new();
        for (alias, from) in (va..end)
            .step_by(PAGE_STEP)
            .zip((source..source_end).step_by(PAGE_STEP))
        {
            let pa = *shadow
                .backed
                .get(&from)
                .ok_or(MapError::NotMapped { va: from })?;
            let mapped = BackingRange::new(pa, PAGE_SIZE)
                .map_err(SpaceError::from)
                .and_then(|ram| Ok(self.tables.map(alias, ram, perms)?));
            if let Err(error) = mapped {
                for (done, _) in &aliased {
                    // The page was mapped a moment ago; an unmap that fails leaves it to the
                    // error already being returned.
                    if let Err(undo) = self.tables.unmap(*done, PAGE_SIZE) {
                        tracing::error!(%undo, page = done, "an alias page could not be unmapped");
                    }
                }
                return Err(error);
            }
            aliased.push((alias, pa));
        }
        shadow.mapped.insert(va, Mapped { end, perms });
        shadow
            .aliased
            .extend(aliased.iter().map(|(alias, _)| *alias));
        shadow.backed.extend(aliased);
        Ok(())
    }

    /// Decides what a guest access that faulted at `address` comes to (see [`Touch`]), and
    /// backs the page when that is all the access was missing.
    pub(crate) fn touch(&self, address: u64, access: Access) -> Touch {
        let page = page_of(address);
        let mut shadow = self.shadow();
        let Some((_, mapped)) = shadow.mapping(page) else {
            return Touch::Refused;
        };
        if !mapped.perms.allows(access) || shadow.backed.contains_key(&page) {
            return Touch::Refused;
        }
        match self.back(&mut shadow, page, mapped.perms) {
            Ok(_) => Touch::Backed,
            Err(error) => Touch::Failed(error),
        }
    }

    /// Makes `sink` where [`SpaceMapper`]'s recording point sends its events from now on (module
    /// docs, "The Map recording point"), in place of any sink attached before.
    pub(crate) fn record_maps(&self, sink: MapSink) {
        *self.maps.lock().unwrap_or_else(PoisonError::into_inner) = Some(sink);
    }

    /// The recording point: sends one [`EventKind::Map`] to the attached sink, if there is
    /// one. The sink is called with no lock of this address space held.
    fn record_map(&self, va: u64, size: u64, perms: Option<Perms>, kind: MapKind) {
        let sink = self
            .maps
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(MapSink(sink)) = sink {
            sink(EventKind::Map {
                va,
                size,
                perms: perms.map_or(0, permission_word),
                kind,
            });
        }
    }

    /// Where the bytes at `va` are, and how many bytes from there on are in the same page.
    fn resolve(shadow: &Shadow, va: u64) -> Option<(Piece, u64)> {
        let page = page_of(va);
        let offset = va - page;
        let piece = if let Some(pa) = shadow.backed.get(&page) {
            Piece::Backed(pa + offset)
        } else {
            shadow.mapping(page)?;
            Piece::Unbacked
        };
        Some((piece, PAGE_SIZE - offset))
    }
}

/// Guest memory by virtual address, as host code sees it: every mapped byte can be read and
/// written, whatever guest code may do with it. An access that touches an address that is not
/// mapped is a [`Fault`] at the first such address and reads or writes nothing; an access of
/// no bytes always succeeds. A write that guest RAM has no room to back is a [`Fault`] at the
/// first page it could not back, and writes nothing.
impl<M: GuestMemory> GuestMemory for AddressSpace<M> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), Fault> {
        let shadow = self.shadow();
        let resolve = |at| Self::resolve(&shadow, at);
        pieces(va, buf.len(), resolve, |_, _, _| Ok(()))?;
        let physical = self.tables.memory();
        pieces(va, buf.len(), resolve, |piece, done, len| {
            let part = &mut buf[done..done + len];
            match piece {
                Piece::Backed(pa) => physical.read(pa, part),
                Piece::Unbacked => {
                    part.fill(0);
                    Ok(())
                }
            }
        })
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), Fault> {
        let mut shadow = self.shadow();
        // Every page must be mapped, and then every page backed, before a byte is written.
        pieces(
            va,
            data.len(),
            |at| Self::resolve(&shadow, at),
            |_, _, _| Ok(()),
        )?;
        let last = u64::try_from(data.len())
            .ok()
            .and_then(|len| len.checked_sub(1))
            .and_then(|offset| va.checked_add(offset));
        // No last byte: an access of no bytes, which touches no page.
        let pages = last.map_or(0..0, |last| page_of(va)..last.saturating_add(1));
        for page in pages.step_by(PAGE_STEP) {
            if shadow.backed.contains_key(&page) {
                continue;
            }
            let perms = shadow
                .mapping(page)
                .map(|(_, mapped)| mapped.perms)
                .ok_or(Fault { address: page })?;
            self.back(&mut shadow, page, perms).map_err(|error| {
                tracing::error!(%error, page, "a page the host writes could not be backed");
                Fault { address: page }
            })?;
        }
        let physical = self.tables.memory();
        pieces(
            va,
            data.len(),
            |at| Self::resolve(&shadow, at),
            |piece, done, len| match piece {
                Piece::Backed(pa) => physical.write(pa, &data[done..done + len]),
                Piece::Unbacked => Err(Fault { address: va }),
            },
        )
    }
}

/// Where the address space's recording point sends each [`EventKind::Map`] it makes
/// ([`AddressSpace::record_maps`]; module docs, "The Map recording point"). Cloned freely; every
/// clone calls the same function.
#[derive(Clone)]
pub(crate) struct MapSink(Arc<dyn Fn(EventKind) + Send + Sync>);

impl MapSink {
    /// A sink that hands each event to `record`.
    pub(crate) fn new(record: impl Fn(EventKind) + Send + Sync + 'static) -> Self {
        Self(Arc::new(record))
    }
}

impl std::fmt::Debug for MapSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MapSink")
    }
}

/// The guest's raw permission word for what `perms` lets guest code do: the bits of
/// `Svc::MemoryPermission`, read bit 0, write bit 1 and execute bit 2
/// (`eden/src/core/hle/kernel/svc_types.h` @ cf654052), as the kernel's
/// [`MemoryPermission::svc`] reports them. It is the inverse of [`MemoryPermission::guest`] on
/// the user's bits: `the_permission_word_is_the_kernels_user_bits_for_every_translation`.
pub(crate) fn permission_word(perms: Perms) -> u32 {
    let bit = |allowed: bool, permission: MemoryPermission| {
        if allowed { permission.svc() } else { 0 }
    };
    bit(perms.readable(), MemoryPermission::USER_READ)
        | bit(perms.writable(), MemoryPermission::USER_WRITE)
        | bit(perms.executable(), MemoryPermission::USER_EXECUTE)
}

/// A guest address space as the kernel's address space drives it: the [`Mapper`] a process's
/// `starquake_kernel::memory::AddressSpace` owns, while the core loop keeps the address space
/// itself for the guest's memory and its faults.
#[derive(Debug)]
pub(crate) struct SpaceMapper<M>(pub(crate) Arc<AddressSpace<M>>);

/// How the kernel hears of a refusal: running out of guest RAM is
/// [`MapperError::OutOfMemory`], and every other refusal [`MapperError::Refused`], in the
/// address space's own words.
fn refused(error: SpaceError) -> MapperError {
    match error {
        SpaceError::OutOfRam { .. } => MapperError::OutOfMemory(error.to_string()),
        SpaceError::Map(_)
        | SpaceError::NotShared { .. }
        | SpaceError::StillMapped { .. }
        | SpaceError::Zeroing(_) => MapperError::Refused(error.to_string()),
    }
}

/// The guest-physical memory an address space's tables and backing lie in, addressed by
/// guest-physical address: what the [`SharedPages`] of a shared memory it handed out read and
/// write through. [`SharedPages`] keeps each access inside its own pages.
#[derive(Debug)]
struct Physical<M>(Arc<AddressSpace<M>>);

impl<M: GuestMemory> GuestMemory for Physical<M> {
    fn read(&self, pa: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.0.tables.memory().read(pa, buf)
    }

    fn write(&self, pa: u64, data: &[u8]) -> Result<(), Fault> {
        self.0.tables.memory().write(pa, data)
    }
}

/// RECORDING POINT for the event stream: each mutation of the address space the kernel asks
/// for and this one makes is one [`EventKind::Map`] (module docs, "The Map recording point").
impl<M: GuestMemory + 'static> Mapper for SpaceMapper<M> {
    fn map(&mut self, address: u64, size: u64, perms: Perms) -> Result<(), MapperError> {
        self.0.map(address, size, perms).map_err(refused)?;
        self.0.record_map(address, size, Some(perms), MapKind::Map);
        Ok(())
    }

    fn unmap(&mut self, address: u64, size: u64) -> Result<(), MapperError> {
        self.0.unmap(address, size).map_err(refused)?;
        self.0.record_map(address, size, None, MapKind::Unmap);
        Ok(())
    }

    fn protect(&mut self, address: u64, size: u64, perms: Perms) -> Result<(), MapperError> {
        self.0.protect(address, size, perms).map_err(refused)?;
        self.0
            .record_map(address, size, Some(perms), MapKind::Protect);
        Ok(())
    }

    fn allocate_shared(&mut self, size: u64) -> Result<SharedPages, MapperError> {
        let pages = self.0.allocate_shared(size).map_err(refused)?;
        Ok(SharedPages::new(
            pages,
            Arc::new(Physical(Arc::clone(&self.0))),
        ))
    }

    fn map_shared(
        &mut self,
        address: u64,
        backing: BackingRange,
        perms: Perms,
    ) -> Result<(), MapperError> {
        self.0
            .map_shared(address, backing, perms)
            .map_err(refused)?;
        self.0
            .record_map(address, backing.size(), Some(perms), MapKind::Map);
        Ok(())
    }

    fn release_shared(&mut self, backing: BackingRange) -> Result<(), MapperError> {
        self.0.release_shared(backing).map_err(refused)
    }

    fn map_alias(
        &mut self,
        address: u64,
        source: u64,
        size: u64,
        perms: Perms,
    ) -> Result<(), MapperError> {
        self.0
            .map_alias(address, source, size, perms)
            .map_err(refused)?;
        self.0.record_map(address, size, Some(perms), MapKind::Map);
        Ok(())
    }
}

#[cfg(test)]
mod tests;

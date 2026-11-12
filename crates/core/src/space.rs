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
//! PROVISIONAL, and what replaces each part:
//!
//! - **Backing is handed out from the start of guest RAM, a page at a time, and never
//!   returned**: an unmapped page's RAM is not reused. The backing pools of the execution model
//!   replace it, sized as the kernel's pool accounting says. Until then guest RAM (16 MiB) is
//!   far smaller than what the kernel lets a process reserve, and a guest that touches more
//!   ends its run as an emulator error.
//! - **One page per fault.** A guest that touches a large mapping takes one exit per page.
//!   Backing a run of pages at once is a measurement away.
//! - **Host access ignores permissions**, which is what lets the loader write a code page that
//!   guest code may only read and execute; nothing checks that a system call's buffer is
//!   readable or writable by the guest. The kernel's block map knows; a later slice checks.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use starquake_kernel::memory::{Mapper, MapperError};
use starquake_memory::stage1::{Stage1Tables, TableRoot, VA_SPACE_END};
use starquake_memory::{
    Access, BackingRange, Fault, GuestMemory, MapError, PAGE_SIZE, Perms, Translation,
};

/// Why a guest address space could not be built or changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SpaceError {
    /// The translation tables refused the mapping, or could not be created; or the range is
    /// not one the address space can map, unmap or reprotect.
    #[error("the guest's translation tables refused a mapping: {0}")]
    Map(#[from] MapError),
    /// Guest RAM has fewer bytes left than a page needs.
    #[error("mapping {size:#x} bytes needs more guest RAM than the {available:#x} bytes left")]
    OutOfRam {
        /// Bytes needed.
        size: u64,
        /// Bytes of guest RAM not yet handed out.
        available: u64,
    },
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
            }),
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

    /// Gives the page at `page` a zeroed page of RAM and a table entry with `perms`.
    fn back(&self, shadow: &mut Shadow, page: u64, perms: Perms) -> Result<u64, SpaceError> {
        let available = self.ram.end().saturating_sub(shadow.next);
        if available < PAGE_SIZE {
            return Err(SpaceError::OutOfRam {
                size: PAGE_SIZE,
                available,
            });
        }
        // RAM starts out zeroed and is never handed out twice, so the page reads as zero.
        let pa = shadow.next;
        self.tables
            .map(page, BackingRange::new(pa, PAGE_SIZE)?, perms)?;
        shadow.next = pa + PAGE_SIZE;
        shadow.backed.insert(page, pa);
        Ok(pa)
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

    /// Where the bytes at `va` are, and how many bytes from there on are in the same page.
    fn resolve(shadow: &Shadow, va: u64) -> Option<(Piece, u64)> {
        let page = page_of(va);
        let offset = va - page;
        let piece = match shadow.backed.get(&page) {
            Some(pa) => Piece::Backed(pa + offset),
            None => {
                shadow.mapping(page)?;
                Piece::Unbacked
            }
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

/// A guest address space as the kernel's address space drives it: the [`Mapper`] a process's
/// `starquake_kernel::memory::AddressSpace` owns, while the core loop keeps the address space
/// itself for the guest's memory and its faults.
#[derive(Debug)]
pub(crate) struct SpaceMapper<M>(pub(crate) Arc<AddressSpace<M>>);

fn refused(error: SpaceError) -> MapperError {
    MapperError(error.to_string())
}

impl<M: GuestMemory + 'static> Mapper for SpaceMapper<M> {
    fn map(&mut self, address: u64, size: u64, perms: Perms) -> Result<(), MapperError> {
        self.0.map(address, size, perms).map_err(refused)
    }

    fn unmap(&mut self, address: u64, size: u64) -> Result<(), MapperError> {
        self.0.unmap(address, size).map_err(refused)
    }

    fn protect(&mut self, address: u64, size: u64, perms: Perms) -> Result<(), MapperError> {
        self.0.protect(address, size, perms).map_err(refused)
    }
}

#[cfg(test)]
mod tests;

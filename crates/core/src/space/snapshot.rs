//! The guest pages of an address space saved and restored: the observability spec's snapshot
//! of "guest memory (only pages that were touched)" ("Snapshots"; "Order of work", step 6,
//! "kernel and memory first"), beside the kernel's address-space snapshot
//! (`starquake_kernel::memory`'s `space::snapshot` module, whose blocks say what is mapped), with
//! the kernel's shared header ([`starquake_kernel::snapshot`]).
//!
//! # What is saved
//!
//! A page is touched when it has backing: guest code faulted on it, the host wrote to it, or a
//! `MapMemory` alias of it was made (module docs of `space`, "Pages are backed when they are
//! first touched"). A page that has no backing reads as zero and holds nothing to save. What is
//! saved is guest RAM as this address space hands it out, each page of RAM at most once:
//!
//! - **The first guest-physical address of RAM never handed out** (`next`).
//! - **The runs of RAM handed out as shared memory and not given back**, each its first
//!   guest-physical address and the contents of each of its pages. The kernel's object table
//!   names a shared memory by these addresses (`SharedMemory::backing`), so they are restored
//!   where they were.
//! - **The runs given back** (waiting for a flush, or reusable), each its first address and
//!   size. What they hold is not saved: a run is zeroed when it is handed out again.
//! - **Every other backed page** (a private page), by virtual address, with the guest-physical
//!   page that backs it and its contents. Left out: the pages of a shared mapping (their RAM is
//!   a shared run's, and the kernel's restore maps them again from its blocks) and the pages a
//!   `MapMemory` alias mapped (their RAM is their source's, and the kernel's restore maps them
//!   again from its alias record).
//!
//! A page's contents are [`Contents::Zero`] when every byte is zero (a page only read since it
//! was backed is one), and otherwise its bytes.
//!
//! Not saved: the translation tables (the kernel's restore makes every mapping again through a
//! new [`SpaceMapper`], and [`AddressSpace::finish_restore`] gives every restored page its
//! table entry), the permissions (the kernel's blocks hold them), and the RAM of a page backed
//! and then unmapped (it is never handed out again: module docs of `space`, PROVISIONAL).
//!
//! # Format
//!
//! The kernel's header ([`starquake_kernel::snapshot`]) with [`PAGES_MAGIC`] and
//! [`PAGES_VERSION`], then the body, in this order: `next`; the shared runs (each its base, then
//! its pages' contents); the given-back runs (each base, size); the private pages (each virtual
//! address, guest-physical address, contents). Contents are a tag (0 for zero, 1 for bytes) and,
//! for bytes, the byte count and the bytes. The layout is starquake's own: Horizon has no
//! snapshot to follow. The golden-bytes test `the_body_layout_is_pinned` pins it with every
//! variant; changing it means bumping [`PAGES_VERSION`].
//!
//! # The bound
//!
//! A title can force at most every page of guest RAM to be backed, and each page of RAM is
//! saved at most once (above), so a snapshot holds at most as many pages as guest RAM has. A
//! restore refuses bytes longer than [`max_snapshot_bytes`] of the RAM it restores into before
//! it decodes them ([`PagesError::TooLarge`]): the header at its largest, the body's fixed
//! fields at their largest, and for each page of RAM the largest entry it can add (two varints
//! of a `u64`, then its contents with a tag, a varint count and a page of bytes; a run adds its
//! two varints once, for its first page). The checks below then hold every address inside
//! that RAM, and no page twice, so a snapshot that passes them saves no more than that.
//! Tests: `a_snapshot_exactly_at_the_bound_is_decoded_and_one_byte_more_is_refused` and
//! `a_snapshot_of_every_page_of_ram_is_within_the_bound`.
//!
//! # Checks
//!
//! A snapshot is a file and is not trusted. [`AddressSpace::begin_restore`] checks, in this
//! order, and returns the first failure ([`PagesError`]):
//!
//! 1. The address space is new: nothing mapped, backed or handed out
//!    ([`PagesError::NotFresh`]).
//! 2. The length ([`PagesError::TooLarge`], above).
//! 3. The header and the body's encoding ([`starquake_kernel::snapshot::decode`]: header,
//!    magic, version, body, trailing bytes; [`PagesError::Format`]).
//! 4. `next` is page aligned and not below RAM's start ([`PagesError::Next`]), and not past
//!    RAM's end ([`PagesError::PastRam`]).
//! 5. Each shared run in turn ([`PagesError::Shared`], first failing check of the first failing
//!    run, in [`RunFault`]'s order): whole pages that do not wrap ([`RunFault::Shape`]), inside
//!    the RAM handed out (from RAM's start to `next`: [`RunFault::Ram`]), starting at or after
//!    the end of the one before ([`RunFault::Order`]), and each page's bytes a page long
//!    ([`RunFault::Length`]).
//! 6. Each given-back run in turn ([`PagesError::Free`]): the same shape, RAM and order checks,
//!    then that it shares no page with a shared run ([`RunFault::Shared`]).
//! 7. Each private page in turn ([`PagesError::Page`], in [`PageFault`]'s order): both
//!    addresses page aligned and the page inside the address space the tables translate
//!    ([`PageFault::Shape`]), above the one before ([`PageFault::Order`]), its RAM inside the
//!    RAM handed out ([`PageFault::Ram`]) and in no run ([`PageFault::Run`]), not an earlier
//!    page's RAM ([`PageFault::Duplicate`]), and its bytes a page long ([`PageFault::Length`]).
//!
//! Then, the last step that can fail, the bytes are written into RAM; a write RAM refuses is
//! [`PagesError::Ram`], and the pages written before it are zeroed again. Only then is
//! anything recorded in the address space, so a refused restore leaves it new.
//!
//! [`AddressSpace::save_pages`] runs checks 4 to 7 on what it is about to write (a failure is a
//! bug here), and first that no restore is unfinished ([`PagesError::Unfinished`]) and that each
//! alias page maps a page the snapshot keeps ([`PagesError::Alias`]).
//!
//! # Restoring
//!
//! The order is [`restore`]'s: the kernel's object table first (it holds the shared memories);
//! then [`AddressSpace::begin_restore`] on a new address space, which records the shared and
//! given-back runs and `next`, and keeps each private page waiting for its mapping; then the
//! kernel's address-space restore, which maps every block that is not free through a
//! [`SpaceMapper`] (a `Shared` one with its shared memory's pages, which
//! [`AddressSpace::map_shared`] accepts now that the run is recorded), and each alias, whose
//! source pages get the RAM their snapshot saved rather than new RAM; then
//! [`AddressSpace::finish_restore`], which gives each page still waiting its table entry, with
//! its mapping's permission, after checking, in address order, that each lies inside a mapping
//! ([`PagesError::Outside`]: the kernel's blocks hold it) and is not backed already
//! ([`PagesError::Taken`]: inside a shared mapping, or a page an alias mapped). A refusal by
//! the tables unmaps the pages it mapped and is [`PagesError::Translation`].
//!
//! # Divergences and what is not checked
//!
//! - **The runs given back while a flush was owed come back reusable.** A new address space
//!   owes no flush and no vCPU has run in it, so nothing can reach their pages through a stale
//!   TLB entry. Which run the next shared memory takes may differ from the run it would have
//!   taken; guest code cannot see a guest-physical address. Test:
//!   `a_run_given_back_while_a_flush_was_owed_comes_back_reusable`.
//! - **Not checked: that each shared run is a shared memory's.** The kernel's restore checks
//!   the other direction (each shared mapping's object names a run recorded here, or
//!   [`AddressSpace::map_shared`] refuses it: `a_shared_mapping_whose_run_is_not_saved_is_refused`).
//!   A run no shared memory names stays handed out for the rest of the run: its RAM is not
//!   reused.
//! - **Not checked: that a page's bytes are not all zero.** A page saved as bytes that are all
//!   zero restores as the zero page does; save writes [`Contents::Zero`] for it.
//! - **Not checked: the RAM between the runs and pages.** RAM handed out to a page that was
//!   unmapped is in no list (it is never handed out again), so a gap is accepted.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use starquake_kernel::id::ObjectId;
use starquake_kernel::memory::{AddressSpace as KernelSpace, SharedMemory, SnapshotError};
use starquake_kernel::snapshot::{Format, FormatError, decode, encode};
use starquake_memory::stage1::VA_SPACE_END;
use starquake_memory::{BackingRange, Fault, GuestMemory, PAGE_SIZE, Translation};

use super::{AddressSpace, PAGE_STEP, SpaceError, SpaceMapper};

#[cfg(test)]
mod tests;

/// First bytes of every guest-pages snapshot.
pub(crate) const PAGES_MAGIC: [u8; 4] = *b"SQGP";

/// Layout version written into every guest-pages snapshot and required when reading one.
pub(crate) const PAGES_VERSION: u32 = 1;

/// The guest-pages snapshot's header.
const FORMAT: Format = Format {
    name: "guest pages",
    magic: PAGES_MAGIC,
    version: PAGES_VERSION,
};

/// The most bytes `postcard` writes for a `u64` or a `usize`: a LEB128 varint of 64 bits, seven
/// bits a byte, `ceil(64 / 7)`.
const MAX_VARINT_U64: u64 = 64_u64.div_ceil(7);

/// The most bytes `postcard` writes for a `u32` (the header's version): `ceil(32 / 7)`.
const MAX_VARINT_U32: u64 = 32_u64.div_ceil(7);

/// The bytes `postcard` writes for `value` as a varint.
const fn varint_len(value: u64) -> u64 {
    let mut len = 1;
    let mut rest = value >> 7;
    while rest != 0 {
        len += 1;
        rest >>= 7;
    }
    len
}

/// The most bytes the header takes: the four magic bytes and the version.
const MAX_HEADER: u64 = PAGES_MAGIC.len() as u64 + MAX_VARINT_U32;

/// The most bytes the body's fields that do not grow with the pages take: `next` and the
/// lengths of the three lists.
const MAX_FIXED_BODY: u64 = 4 * MAX_VARINT_U64;

/// The most bytes one page's [`Contents`] take: the tag (`postcard` writes a variant index as a
/// varint, one byte for each of the two), the byte count and a page of bytes.
const MAX_CONTENTS: u64 = 1 + varint_len(PAGE_SIZE) + PAGE_SIZE;

/// The most bytes one page of RAM adds to the body: two varints (a private page's virtual and
/// guest-physical addresses; a run's base and page count or size, which a run adds once, for
/// its first page) and its contents.
const MAX_PER_PAGE: u64 = 2 * MAX_VARINT_U64 + MAX_CONTENTS;

/// The most bytes a guest-pages snapshot of an address space with guest RAM `ram` can take
/// (module docs, "The bound"), saturating.
pub(crate) fn max_snapshot_bytes(ram: BackingRange) -> u64 {
    (ram.size() / PAGE_SIZE)
        .saturating_mul(MAX_PER_PAGE)
        .saturating_add(MAX_HEADER + MAX_FIXED_BODY)
}

/// The bytes of one page, as the snapshot holds them: `postcard` writes them as a byte string
/// (their count, then the bytes). Any length decodes; check 5 and 7 refuse one that is not a
/// page ([`RunFault::Length`], [`PageFault::Length`]).
#[derive(Clone, PartialEq, Eq)]
struct PageBytes(Vec<u8>);

/// Says how many bytes there are, not what they are: a page of bytes is too long to read.
impl std::fmt::Debug for PageBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PageBytes({} bytes)", self.0.len())
    }
}

impl Serialize for PageBytes {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for PageBytes {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// Reads a byte string.
        struct Bytes;

        impl Visitor<'_> for Bytes {
            type Value = PageBytes;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("the bytes of a page")
            }

            fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<PageBytes, E> {
                Ok(PageBytes(bytes.to_vec()))
            }

            fn visit_byte_buf<E: de::Error>(self, bytes: Vec<u8>) -> Result<PageBytes, E> {
                Ok(PageBytes(bytes))
            }
        }

        deserializer.deserialize_byte_buf(Bytes)
    }
}

/// What one page holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Contents {
    /// Every byte is zero.
    Zero,
    /// Its bytes.
    Bytes(PageBytes),
}

impl Contents {
    /// What the `PAGE_SIZE` bytes in `page` come to.
    fn of(page: &[u8]) -> Self {
        if page.iter().all(|byte| *byte == 0) {
            Self::Zero
        } else {
            Self::Bytes(PageBytes(page.to_vec()))
        }
    }

    /// Whether the contents are a page long: [`Contents::Zero`] always is.
    fn whole(&self) -> bool {
        match self {
            Self::Zero => true,
            Self::Bytes(bytes) => u64::try_from(bytes.0.len()) == Ok(PAGE_SIZE),
        }
    }
}

/// A run of RAM handed out as shared memory, with what each of its pages holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SharedRun {
    /// Its first guest-physical address.
    base: u64,
    /// Its pages, in address order.
    pages: Vec<Contents>,
}

/// A run of RAM given back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct FreeRun {
    /// Its first guest-physical address.
    base: u64,
    /// Its size in bytes.
    size: u64,
}

/// A private page: a backed page that is neither in a shared mapping nor an alias's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SavedPage {
    /// Its virtual address.
    va: u64,
    /// The guest-physical address of the RAM that backs it.
    pa: u64,
    /// What it holds.
    contents: Contents,
}

/// The snapshot's body (module docs, "Format").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Body {
    next: u64,
    shared: Vec<SharedRun>,
    free: Vec<FreeRun>,
    pages: Vec<SavedPage>,
}

/// What went wrong with one run ([`PagesError::Shared`], [`PagesError::Free`]), in the order
/// they are checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum RunFault {
    /// Not page aligned, no pages, or its end overflows.
    #[error("is not a run of whole pages")]
    Shape,
    /// It does not lie inside the RAM handed out.
    #[error("lies outside the guest RAM handed out")]
    Ram,
    /// It starts before the end of the one before.
    #[error("starts before the end of the one before it")]
    Order,
    /// A page's bytes are not a page long (a shared run's only).
    #[error("has a page whose bytes are not a page long")]
    Length,
    /// It shares a page with a shared run (a given-back run's only).
    #[error("shares a page with a shared memory's run")]
    Shared,
}

/// What went wrong with one private page ([`PagesError::Page`]), in the order they are checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum PageFault {
    /// An address is not page aligned, or the page lies past the address space.
    #[error("is not a page of the address space")]
    Shape,
    /// It is not above the one before.
    #[error("is not above the one before it")]
    Order,
    /// Its RAM does not lie inside the RAM handed out.
    #[error("has RAM outside the guest RAM handed out")]
    Ram,
    /// Its RAM is in a shared or given-back run.
    #[error("has RAM inside a run")]
    Run,
    /// Its RAM is an earlier page's.
    #[error("has the RAM of an earlier page")]
    Duplicate,
    /// Its bytes are not a page long.
    #[error("has bytes that are not a page long")]
    Length,
}

/// Why guest pages could not be saved or restored.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum PagesError {
    /// The address space to restore into has something mapped, backed or handed out.
    #[error("guest pages can only be restored into a new address space")]
    NotFresh,
    /// The snapshot is longer than any of this guest RAM can be.
    #[error(
        "the guest pages snapshot is {size} bytes, more than the {limit} bytes one of this \
         guest RAM can take"
    )]
    TooLarge {
        /// Its length in bytes.
        size: u64,
        /// [`max_snapshot_bytes`] of the RAM.
        limit: u64,
    },
    /// The header or the body's encoding.
    #[error(transparent)]
    Format(#[from] FormatError),
    /// `next` is not page aligned, or lies below RAM's start.
    #[error("the end of the guest RAM handed out, {next:#x}, is not a page of guest RAM")]
    Next {
        /// The saved `next`.
        next: u64,
    },
    /// `next` lies past RAM's end.
    #[error(
        "the snapshot hands out guest RAM up to {next:#x}, past the end of guest RAM at {end:#x}"
    )]
    PastRam {
        /// The saved `next`.
        next: u64,
        /// One past the last address of guest RAM.
        end: u64,
    },
    /// A shared run.
    #[error("shared memory run {index} {fault}")]
    Shared {
        /// Its index in the snapshot.
        index: usize,
        /// What is wrong.
        fault: RunFault,
    },
    /// A given-back run.
    #[error("given-back run {index} {fault}")]
    Free {
        /// Its index in the snapshot.
        index: usize,
        /// What is wrong.
        fault: RunFault,
    },
    /// A private page.
    #[error("saved page {index} {fault}")]
    Page {
        /// Its index in the snapshot.
        index: usize,
        /// What is wrong.
        fault: PageFault,
    },
    /// RAM refused a read (save) or a write (restore).
    #[error("guest RAM refused an access while its pages were saved or restored: {0}")]
    Ram(Fault),
    /// A restored page lies where nothing is mapped: the kernel's blocks do not hold it.
    #[error("the restored page at {va:#x} lies outside every mapping the blocks made")]
    Outside {
        /// Its virtual address.
        va: u64,
    },
    /// A restored page is backed already: a shared mapping or an alias mapped it.
    #[error("the restored page at {va:#x} was mapped already, with other RAM")]
    Taken {
        /// Its virtual address.
        va: u64,
    },
    /// The tables refused a restored page's entry.
    #[error("the restored page at {va:#x} could not be mapped: {error}")]
    Translation {
        /// Its virtual address.
        va: u64,
        /// What the tables said.
        error: SpaceError,
    },
    /// Save: a restore was begun and not finished.
    #[error("the address space is part-way through a restore")]
    Unfinished,
    /// Save: an alias page maps RAM that no saved page or shared run holds.
    #[error("the alias page at {va:#x} maps RAM no saved page holds")]
    Alias {
        /// Its virtual address.
        va: u64,
    },
}

/// What [`Body::check`] found: the pages of RAM in each run, for check 7.
struct Runs {
    /// Each shared and given-back run, by first address: one past its last address.
    all: BTreeMap<u64, u64>,
}

impl Runs {
    /// Whether the page at `pa` lies in a run.
    fn holds(&self, pa: u64) -> bool {
        self.all
            .range(..=pa)
            .next_back()
            .is_some_and(|(_, end)| pa < *end)
    }
}

impl Body {
    /// Checks 4 to 7 of the module docs, against guest RAM `ram`.
    fn check(&self, ram: BackingRange) -> Result<Runs, PagesError> {
        let next = self.next;
        if !next.is_multiple_of(PAGE_SIZE) || next < ram.base() {
            return Err(PagesError::Next { next });
        }
        if next > ram.end() {
            return Err(PagesError::PastRam {
                next,
                end: ram.end(),
            });
        }
        let handed_out = ram.base()..next;
        let inside = |base: u64, end: u64| handed_out.start <= base && end <= handed_out.end;
        let mut shared = BTreeMap::new();
        let mut after = ram.base();
        for (index, run) in self.shared.iter().enumerate() {
            let fault = |fault| PagesError::Shared { index, fault };
            let end = u64::try_from(run.pages.len())
                .ok()
                .and_then(|pages| pages.checked_mul(PAGE_SIZE))
                .filter(|size| *size != 0 && run.base.is_multiple_of(PAGE_SIZE))
                .and_then(|size| run.base.checked_add(size))
                .ok_or(fault(RunFault::Shape))?;
            if !inside(run.base, end) {
                return Err(fault(RunFault::Ram));
            }
            if run.base < after {
                return Err(fault(RunFault::Order));
            }
            if !run.pages.iter().all(Contents::whole) {
                return Err(fault(RunFault::Length));
            }
            shared.insert(run.base, end);
            after = end;
        }
        let shared = Runs { all: shared };
        let mut all = shared.all.clone();
        let mut after = ram.base();
        for (index, run) in self.free.iter().enumerate() {
            let fault = |fault| PagesError::Free { index, fault };
            let end = Some(run.size)
                .filter(|size| {
                    *size != 0
                        && size.is_multiple_of(PAGE_SIZE)
                        && run.base.is_multiple_of(PAGE_SIZE)
                })
                .and_then(|size| run.base.checked_add(size))
                .ok_or(fault(RunFault::Shape))?;
            if !inside(run.base, end) {
                return Err(fault(RunFault::Ram));
            }
            if run.base < after {
                return Err(fault(RunFault::Order));
            }
            if overlaps(&shared.all, run.base, end) {
                return Err(fault(RunFault::Shared));
            }
            all.insert(run.base, end);
            after = end;
        }
        let runs = Runs { all };
        let mut taken = BTreeSet::new();
        let mut above: Option<u64> = None;
        for (index, page) in self.pages.iter().enumerate() {
            let fault = |fault| PagesError::Page { index, fault };
            let well_formed = page.va.is_multiple_of(PAGE_SIZE)
                && page.pa.is_multiple_of(PAGE_SIZE)
                && page
                    .va
                    .checked_add(PAGE_SIZE)
                    .is_some_and(|end| end <= VA_SPACE_END);
            if !well_formed {
                return Err(fault(PageFault::Shape));
            }
            if above.is_some_and(|before| page.va <= before) {
                return Err(fault(PageFault::Order));
            }
            if !page
                .pa
                .checked_add(PAGE_SIZE)
                .is_some_and(|end| inside(page.pa, end))
            {
                return Err(fault(PageFault::Ram));
            }
            if runs.holds(page.pa) {
                return Err(fault(PageFault::Run));
            }
            if !taken.insert(page.pa) {
                return Err(fault(PageFault::Duplicate));
            }
            if !page.contents.whole() {
                return Err(fault(PageFault::Length));
            }
            above = Some(page.va);
        }
        Ok(runs)
    }
}

/// Whether `base..end` shares an address with a run of `runs` (first address to one past the
/// last, none overlapping).
fn overlaps(runs: &BTreeMap<u64, u64>, base: u64, end: u64) -> bool {
    runs.range(..end)
        .next_back()
        .is_some_and(|(_, run_end)| base < *run_end)
}

impl<M: GuestMemory> AddressSpace<M> {
    /// The address space's touched pages as a snapshot (module docs, "What is saved" and
    /// "Format"), after the checks the module docs list for a save: a failure is a bug here.
    pub(crate) fn save_pages(&self) -> Result<Vec<u8>, PagesError> {
        let shadow = self.shadow();
        if !shadow.pending.is_empty() {
            return Err(PagesError::Unfinished);
        }
        let memory = self.tables.memory();
        let read = |pa: u64| -> Result<Contents, PagesError> {
            let mut page = [0; PAGE_STEP];
            memory.read(pa, &mut page).map_err(PagesError::Ram)?;
            Ok(Contents::of(&page))
        };
        let in_shared = |pa: u64| overlaps(&shadow.shared, pa, pa.saturating_add(1));
        let mut shared = Vec::with_capacity(shadow.shared.len());
        for (base, end) in &shadow.shared {
            let pages = (*base..*end)
                .step_by(PAGE_STEP)
                .map(read)
                .collect::<Result<_, _>>()?;
            shared.push(SharedRun { base: *base, pages });
        }
        let mut free: Vec<FreeRun> = shadow
            .released
            .iter()
            .map(|run| FreeRun {
                base: run.base(),
                size: run.size(),
            })
            .chain(shadow.reusable.iter().map(|(base, end)| FreeRun {
                base: *base,
                size: end - base,
            }))
            .collect();
        free.sort_by_key(|run| run.base);
        let mut pages = Vec::new();
        let mut kept = BTreeSet::new();
        for (va, pa) in &shadow.backed {
            if shadow.aliased.contains(va) || in_shared(*pa) {
                continue;
            }
            pages.push(SavedPage {
                va: *va,
                pa: *pa,
                contents: read(*pa)?,
            });
            kept.insert(*pa);
        }
        for va in &shadow.aliased {
            let held = shadow
                .backed
                .get(va)
                .is_some_and(|pa| kept.contains(pa) || in_shared(*pa));
            if !held {
                return Err(PagesError::Alias { va: *va });
            }
        }
        let body = Body {
            next: shadow.next,
            shared,
            free,
            pages,
        };
        body.check(self.ram)?;
        Ok(encode(FORMAT, &body)?)
    }

    /// Begins restoring the guest pages `bytes` hold into this address space, which must be
    /// new (module docs, "Checks", for what is checked and in what order, and "Restoring" for
    /// what comes next). Refused, it leaves the address space new.
    pub(crate) fn begin_restore(&self, bytes: &[u8]) -> Result<(), PagesError> {
        let mut shadow = self.shadow();
        let fresh = shadow.next == self.ram.base()
            && shadow.mapped.is_empty()
            && shadow.backed.is_empty()
            && shadow.shared.is_empty()
            && shadow.released.is_empty()
            && shadow.reusable.is_empty()
            && shadow.aliased.is_empty()
            && shadow.pending.is_empty();
        if !fresh {
            return Err(PagesError::NotFresh);
        }
        let limit = max_snapshot_bytes(self.ram);
        let size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if size > limit {
            return Err(PagesError::TooLarge { size, limit });
        }
        let body: Body = decode(FORMAT, bytes)?;
        body.check(self.ram)?;
        // The last step that can fail: the bytes, into RAM that is all zero (new RAM is).
        let writes = body
            .shared
            .iter()
            .flat_map(|run| {
                (run.base..)
                    .step_by(PAGE_STEP)
                    .zip(&run.pages)
                    .collect::<Vec<_>>()
            })
            .chain(body.pages.iter().map(|page| (page.pa, &page.contents)));
        let memory = self.tables.memory();
        let mut written = Vec::new();
        for (pa, contents) in writes {
            let Contents::Bytes(bytes) = contents else {
                continue;
            };
            if let Err(fault) = memory.write(pa, &bytes.0) {
                const ZEROS: [u8; PAGE_STEP] = [0; PAGE_STEP];
                for done in written {
                    if let Err(undo) = memory.write(done, &ZEROS) {
                        tracing::error!(%undo, page = done, "a restored page could not be zeroed");
                    }
                }
                return Err(PagesError::Ram(fault));
            }
            written.push(pa);
        }
        shadow.next = body.next;
        shadow.shared = body
            .shared
            .iter()
            .filter_map(|run| {
                let size = u64::try_from(run.pages.len())
                    .ok()?
                    .checked_mul(PAGE_SIZE)?;
                Some((run.base, run.base.checked_add(size)?))
            })
            .collect();
        shadow.reusable = body
            .free
            .iter()
            .filter_map(|run| Some((run.base, run.base.checked_add(run.size)?)))
            .collect();
        shadow.pending = body.pages.iter().map(|page| (page.va, page.pa)).collect();
        Ok(())
    }

    /// Ends a restore [`AddressSpace::begin_restore`] began, once the kernel's address space
    /// has been restored through this one's [`SpaceMapper`]: gives each restored page still
    /// waiting its table entry (module docs, "Restoring"). Refused with nothing mapped.
    pub(crate) fn finish_restore(&self) -> Result<(), PagesError> {
        let mut shadow = self.shadow();
        let mut entries = Vec::with_capacity(shadow.pending.len());
        for (va, pa) in &shadow.pending {
            let Some((_, mapped)) = shadow.mapping(*va) else {
                return Err(PagesError::Outside { va: *va });
            };
            if shadow.backed.contains_key(va) {
                return Err(PagesError::Taken { va: *va });
            }
            entries.push((*va, *pa, mapped.perms));
        }
        let mut done: Vec<u64> = Vec::with_capacity(entries.len());
        for (va, pa, perms) in &entries {
            let mapped = BackingRange::new(*pa, PAGE_SIZE)
                .map_err(SpaceError::from)
                .and_then(|ram| Ok(self.tables.map(*va, ram, *perms)?));
            if let Err(error) = mapped {
                for page in done {
                    if let Err(undo) = self.tables.unmap(page, PAGE_SIZE) {
                        tracing::error!(%undo, page, "a restored page could not be unmapped");
                    }
                }
                return Err(PagesError::Translation { va: *va, error });
            }
            done.push(*va);
        }
        shadow.pending.clear();
        shadow
            .backed
            .extend(entries.into_iter().map(|(va, pa, _)| (va, pa)));
        Ok(())
    }
}

/// Why an address space and its guest pages could not be restored together ([`restore`]).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum RestoreError {
    /// The guest pages.
    #[error(transparent)]
    Pages(#[from] PagesError),
    /// The kernel's address space.
    #[error(transparent)]
    Space(#[from] SnapshotError),
}

/// Restores a process's address space into `space`, which must be new: its guest pages from
/// `pages` ([`AddressSpace::save_pages`]) and the kernel's address space from `blocks`
/// (`starquake_kernel::memory::AddressSpace::save`), with `shared` giving the shared memory an
/// object id names, from the restored object table. Returns the kernel's address space, which
/// drives `space` through a [`SpaceMapper`]. See the module docs, "Restoring", for the order.
///
/// On an error, `space` may hold mappings and restored pages: discard it.
pub(crate) fn restore<M: GuestMemory + 'static>(
    space: &Arc<AddressSpace<M>>,
    pages: &[u8],
    blocks: &[u8],
    shared: impl FnMut(ObjectId) -> Option<SharedMemory>,
) -> Result<KernelSpace, RestoreError> {
    space.begin_restore(pages)?;
    let kernel = KernelSpace::restore(blocks, Box::new(SpaceMapper(Arc::clone(space))), shared)?;
    space.finish_restore()?;
    Ok(kernel)
}

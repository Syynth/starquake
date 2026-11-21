//! The GPU memory bookkeeping's snapshot ([`GpuMemory::save`], [`GpuMemory::restore`];
//! graphics-server, "What is snapshotted": "memory registrations, address spaces, mappings,
//! sparse regions" and "outstanding tickets").
//!
//! # Format
//!
//! `postcard`, after a header of [`SNAPSHOT_MAGIC`] and [`SNAPSHOT_VERSION`] that is decoded and
//! checked before the body. The body ([`State`]) is, in order: the notification events (index,
//! object), whether the worker is made, the devices (id, name, label), the address spaces (id,
//! then its 211 mappings, each start, size, handle, offset and its runs of one kind, each start,
//! size and kind byte; its sparse regions, each start, size and id; its shim page, if any), the
//! registrations (handle, start, size, device byte, the process of its lock), the ordered jobs
//! (ticket, event, the locks to release as process, start and size, the ranges to publish again
//! as address space, start and size), and the next address space id, handle, region id and
//! ticket. The golden-bytes test `the_snapshot_layout_is_pinned` pins it; changing it means
//! bumping [`SNAPSHOT_VERSION`].
//!
//! The translation tables are not in it: a restore builds each from the bookkeeping (every
//! piece published again, and every two-page registration listed for the semaphore mirror page,
//! `super::memory`, "The semaphore mirror page") and gives it to the front as a new address
//! space, and sends the front
//! a barrier for each job still waiting, in ticket order, since a restored GPU knows none of
//! them. The pages a waiting job holds are not published (`super::memory`, "Pages a waiting job
//! holds"), so they stay holes in the new table until the last job holding them is done
//! (`a_restore_leaves_a_pending_unmap_a_hole_until_it_is_done`).
//!
//! # Checks
//!
//! A snapshot is a file and is not trusted. [`GpuMemory::restore`] checks, in this order, and
//! returns the first failure: the magic, the version, that the body decodes, that no bytes
//! follow it; the counters (each at least 1); the events (indices below
//! [`NOTIFICATION_COUNT`], increasing, no object twice); the devices (ids from 1 below
//! [`DEVICE_SLOTS`], increasing); the address spaces (at most [`MAX_ADDRESS_SPACES`], ids from 1
//! below the next id, increasing); the registrations (at most [`MAX_PIECES`], handles from 1
//! below the next handle, increasing, each range a page range); then each address space's sparse
//! regions (page ranges inside NVN's window, ids from 1 below the next region id, no id twice
//! in the whole snapshot, increasing and not overlapping), its 211 mappings (page ranges inside
//! the window, increasing and not overlapping, each naming a registration, with a page-aligned
//! offset that keeps it inside the registration, and runs of one kind that cover it exactly, in
//! order), its shim (a page range) and its piece count (at most [`MAX_PIECES`]); then the jobs
//! (at most [`MAX_PENDING_JOBS`], tickets from 1 below the next ticket, increasing, each event
//! one of the notification events, each range to publish a page range of an address space).
//! Last, every device-share lock is taken back from the kernel ([`Reclaim`]): the registrations'
//! in order, then the jobs', counting earlier ones of the same exact range, so that N locks on a
//! range take back N tokens and the next is refused at its own index. [`GpuMemory::save`] runs
//! the same checks, but the kernel's, on what it is about to write.
//!
//! A restore that also restores the completion queue must count the queue's releases against
//! the same locks: two separate counts can take back more tokens than the kernel holds. That is
//! for the wiring slice (16), which composes the two.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use starquake_gpu::interface::{Kind, RegionId, Ticket};
use starquake_gpu::vm::GpuVm;
use starquake_kernel::dispatch::DeviceShareError;
use starquake_kernel::id::{ObjectId, ProcessId};
use starquake_memory::{PAGE_SIZE, PageRange};
use thiserror::Error;

use super::memory::cover::Cover;
use super::memory::{
    Counters, DEVICE_SLOTS, Device, GpuMemory, Job, MAX_ADDRESS_SPACES, MAX_PENDING_JOBS,
    MAX_PIECES, NOTIFICATION_COUNT, Record, Region, Registration, Space, in_window,
};
use super::{AsId, GpuVmFront};
use crate::completion::Reclaim;

#[cfg(test)]
mod tests;

/// First bytes of every GPU memory snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"SQGM";

/// Layout version written into every GPU memory snapshot and required when reading one.
pub const SNAPSHOT_VERSION: u32 = 1;

/// The part of a snapshot a check names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The next ids.
    Counters,
    /// A notification event.
    Event,
    /// A graphics device.
    Device,
    /// An address space.
    Space,
    /// A registration.
    Registration,
    /// A sparse region.
    Region,
    /// A 211 mapping.
    Mapping,
    /// A run of one kind of a 211 mapping.
    Span,
    /// The syncpoint shim.
    Shim,
    /// An ordered job.
    Job,
}

impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Counters => "next id",
            Self::Event => "notification event",
            Self::Device => "graphics device",
            Self::Space => "address space",
            Self::Registration => "memory registration",
            Self::Region => "sparse region",
            Self::Mapping => "GPU mapping",
            Self::Span => "kind run",
            Self::Shim => "syncpoint shim",
            Self::Job => "ordered job",
        })
    }
}

/// Why a GPU memory snapshot could not be written or read.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SnapshotError {
    /// Encoding failed.
    #[error("could not encode the GPU memory snapshot: {0}")]
    Encode(postcard::Error),
    /// The header or the body is not a well-formed encoding.
    #[error("could not decode the GPU memory snapshot: {0}")]
    Decode(postcard::Error),
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    #[error("not a GPU memory snapshot (bad magic)")]
    BadMagic,
    /// The snapshot was written with another layout version.
    #[error(
        "GPU memory snapshot layout version {found} is not supported (this build reads {SNAPSHOT_VERSION})"
    )]
    UnsupportedVersion {
        /// The version found.
        found: u32,
    },
    /// Bytes follow the body.
    #[error("{count} unexpected bytes after the GPU memory snapshot")]
    TrailingBytes {
        /// How many.
        count: usize,
    },
    /// An id, address or size is outside what the bookkeeping can hold.
    #[error("{part} {index} of the GPU memory snapshot is out of range")]
    OutOfRange {
        /// What it is.
        part: Part,
        /// Its position in its list.
        index: usize,
    },
    /// An entry is not after the one before it in its list.
    #[error("{part} {index} of the GPU memory snapshot is out of order or repeats one before it")]
    Order {
        /// What it is.
        part: Part,
        /// Its position in its list.
        index: usize,
    },
    /// An entry overlaps the one before it in its list.
    #[error("{part} {index} of the GPU memory snapshot overlaps the one before it")]
    Overlap {
        /// What it is.
        part: Part,
        /// Its position in its list.
        index: usize,
    },
    /// An entry names something the snapshot does not hold.
    #[error("{part} {index} of the GPU memory snapshot names something that does not exist")]
    Dangling {
        /// What it is.
        part: Part,
        /// Its position in its list.
        index: usize,
    },
    /// A list is longer than its bound.
    #[error("the GPU memory snapshot holds too many of {part}")]
    TooMany {
        /// What it holds too many of.
        part: Part,
    },
    /// The kernel refused a device-share lock's token ([`Reclaim`]).
    #[error("lock {index} of the GPU memory snapshot is one the kernel refused: {error}")]
    Release {
        /// The lock's position: the registrations' first, then the jobs' in order.
        index: usize,
        /// The kernel's refusal.
        error: DeviceShareError,
    },
}

/// The snapshot header.
#[derive(Serialize, Deserialize)]
struct Header {
    /// [`SNAPSHOT_MAGIC`].
    magic: [u8; 4],
    /// [`SNAPSHOT_VERSION`].
    version: u32,
}

/// A run of one kind: start, size, kind byte.
type SavedSpan = (u64, u64, u8);

/// A device-share lock: process, start, size.
type SavedLock = (ProcessId, u64, u64);

/// A 211 mapping as saved.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SavedRecord {
    start: u64,
    size: u64,
    handle: u32,
    offset: u64,
    spans: Vec<SavedSpan>,
}

/// An address space as saved.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SavedSpace {
    id: u32,
    records: Vec<SavedRecord>,
    /// Start, size, region id.
    regions: Vec<(u64, u64, u32)>,
    shim: Option<u64>,
}

/// A registration as saved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SavedRegistration {
    handle: u32,
    start: u64,
    size: u64,
    device: u8,
    process: ProcessId,
}

/// An ordered job as saved.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SavedJob {
    ticket: u64,
    event: ObjectId,
    releases: Vec<SavedLock>,
    /// Address space, start, size.
    reveal: Vec<(u32, u64, u64)>,
}

/// The snapshot body (module docs, "Format").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct State {
    events: Vec<(u32, ObjectId)>,
    worker: bool,
    /// Id, name, label.
    devices: Vec<(u8, u64, u64)>,
    spaces: Vec<SavedSpace>,
    registrations: Vec<SavedRegistration>,
    pending: Vec<SavedJob>,
    /// Address space, handle, region, ticket.
    next: (u32, u32, u32, u64),
}

/// A page range, or `fail`.
fn range(start: u64, size: u64, fail: SnapshotError) -> Result<PageRange, SnapshotError> {
    PageRange::new(start, size).map_err(|_| fail)
}

/// Checks that `keys` increase strictly, naming the first that does not.
fn increasing<T: Ord + Copy>(
    keys: impl IntoIterator<Item = T>,
    part: Part,
) -> Result<(), SnapshotError> {
    let mut previous = None;
    for (index, key) in keys.into_iter().enumerate() {
        if previous.is_some_and(|previous| key <= previous) {
            return Err(SnapshotError::Order { part, index });
        }
        previous = Some(key);
    }
    Ok(())
}

/// Checks that `ranges` increase and do not overlap, naming the first that does.
fn disjoint(ranges: &[PageRange], part: Part) -> Result<(), SnapshotError> {
    for (index, pair) in ranges.windows(2).enumerate() {
        if let [before, after] = pair {
            if after.start() <= before.start() {
                return Err(SnapshotError::Order {
                    part,
                    index: index + 1,
                });
            }
            if after.start() < before.end() {
                return Err(SnapshotError::Overlap {
                    part,
                    index: index + 1,
                });
            }
        }
    }
    Ok(())
}

impl State {
    /// The bookkeeping, as saved.
    fn of(memory: &GpuMemory) -> Self {
        let lock = |token: &starquake_kernel::memory::DeviceShareToken| {
            (token.process(), token.address(), token.size())
        };
        Self {
            events: memory.events.iter().map(|(&i, &e)| (i, e)).collect(),
            worker: memory.worker,
            devices: memory
                .devices
                .iter()
                .map(|(&id, device)| (id, device.name, device.label))
                .collect(),
            spaces: memory
                .spaces
                .iter()
                .map(|(id, space)| SavedSpace {
                    id: id.0,
                    records: space
                        .records
                        .iter()
                        .map(|(&start, record)| SavedRecord {
                            start,
                            size: record.end - start,
                            handle: record.handle,
                            offset: record.offset,
                            spans: record
                                .spans
                                .iter()
                                .map(|&(low, high, kind)| (low, high - low, kind.0))
                                .collect(),
                        })
                        .collect(),
                    regions: space
                        .regions
                        .iter()
                        .map(|(&start, region)| (start, region.end - start, region.id.0))
                        .collect(),
                    shim: space.shim,
                })
                .collect(),
            registrations: memory
                .registrations
                .iter()
                .map(|(&handle, r)| SavedRegistration {
                    handle,
                    start: r.range.start(),
                    size: r.range.size(),
                    device: r.device,
                    process: r.token.process(),
                })
                .collect(),
            pending: memory
                .pending
                .iter()
                .map(|(ticket, job)| SavedJob {
                    ticket: ticket.0,
                    event: job.event,
                    releases: job.releases.iter().map(lock).collect(),
                    reveal: job
                        .reveal
                        .iter()
                        .map(|(space, r)| (space.0, r.start(), r.size()))
                        .collect(),
                })
                .collect(),
            next: (
                memory.next.space,
                memory.next.handle,
                memory.next.region,
                memory.next.ticket,
            ),
        }
    }

    /// Every check but the kernel's, in the documented order (module docs, "Checks").
    #[expect(
        clippy::too_many_lines,
        reason = "one check per relation, in the documented order, reads best as one list"
    )]
    fn check(&self) -> Result<(), SnapshotError> {
        let (next_space, next_handle, next_region, next_ticket) = self.next;
        if next_space == 0 || next_handle == 0 || next_region == 0 || next_ticket == 0 {
            return Err(SnapshotError::OutOfRange {
                part: Part::Counters,
                index: 0,
            });
        }
        let mut objects = BTreeSet::new();
        for (index, &(at, event)) in self.events.iter().enumerate() {
            if at >= NOTIFICATION_COUNT || !objects.insert(event) {
                return Err(SnapshotError::OutOfRange {
                    part: Part::Event,
                    index,
                });
            }
        }
        increasing(self.events.iter().map(|e| e.0), Part::Event)?;
        for (index, &(id, ..)) in self.devices.iter().enumerate() {
            if id == 0 || u32::from(id) >= DEVICE_SLOTS {
                return Err(SnapshotError::OutOfRange {
                    part: Part::Device,
                    index,
                });
            }
        }
        increasing(self.devices.iter().map(|d| d.0), Part::Device)?;
        if self.spaces.len() > MAX_ADDRESS_SPACES {
            return Err(SnapshotError::TooMany { part: Part::Space });
        }
        for (index, space) in self.spaces.iter().enumerate() {
            if space.id == 0 || space.id >= next_space {
                return Err(SnapshotError::OutOfRange {
                    part: Part::Space,
                    index,
                });
            }
        }
        increasing(self.spaces.iter().map(|s| s.id), Part::Space)?;
        if self.registrations.len() > MAX_PIECES {
            return Err(SnapshotError::TooMany {
                part: Part::Registration,
            });
        }
        let mut sizes = BTreeMap::new();
        for (index, r) in self.registrations.iter().enumerate() {
            let fail = SnapshotError::OutOfRange {
                part: Part::Registration,
                index,
            };
            if r.handle == 0 || r.handle >= next_handle {
                return Err(fail);
            }
            range(r.start, r.size, fail)?;
            sizes.insert(r.handle, r.size);
        }
        increasing(
            self.registrations.iter().map(|r| r.handle),
            Part::Registration,
        )?;
        let mut region_ids = BTreeSet::new();
        for space in &self.spaces {
            let mut regions = Vec::new();
            for (index, &(start, size, id)) in space.regions.iter().enumerate() {
                let fail = SnapshotError::OutOfRange {
                    part: Part::Region,
                    index,
                };
                let held = range(start, size, fail.clone())?;
                if !in_window(start, size) || id == 0 || id >= next_region || !region_ids.insert(id)
                {
                    return Err(fail);
                }
                regions.push(held);
            }
            disjoint(&regions, Part::Region)?;
            let mut records = Vec::new();
            for (index, record) in space.records.iter().enumerate() {
                let fail = SnapshotError::OutOfRange {
                    part: Part::Mapping,
                    index,
                };
                let held = range(record.start, record.size, fail.clone())?;
                if !in_window(record.start, record.size) {
                    return Err(fail);
                }
                let Some(&size) = sizes.get(&record.handle) else {
                    return Err(SnapshotError::Dangling {
                        part: Part::Mapping,
                        index,
                    });
                };
                if !record.offset.is_multiple_of(PAGE_SIZE)
                    || record.offset > size
                    || record.size > size - record.offset
                {
                    return Err(fail);
                }
                let mut at = record.start;
                for (span_index, &(low, length, _)) in record.spans.iter().enumerate() {
                    let span = range(
                        low,
                        length,
                        SnapshotError::OutOfRange {
                            part: Part::Span,
                            index: span_index,
                        },
                    )?;
                    if span.start() != at {
                        return Err(SnapshotError::Order {
                            part: Part::Span,
                            index: span_index,
                        });
                    }
                    at = span.end();
                }
                if at != held.end() {
                    return Err(fail);
                }
                records.push(held);
            }
            disjoint(&records, Part::Mapping)?;
            if let Some(shim) = space.shim {
                range(
                    shim,
                    PAGE_SIZE,
                    SnapshotError::OutOfRange {
                        part: Part::Shim,
                        index: 0,
                    },
                )?;
            }
            let spans: usize = space.records.iter().map(|r| r.spans.len()).sum();
            if self.registrations.len() + space.regions.len() + spans > MAX_PIECES {
                return Err(SnapshotError::TooMany {
                    part: Part::Mapping,
                });
            }
        }
        if self.pending.len() > MAX_PENDING_JOBS {
            return Err(SnapshotError::TooMany { part: Part::Job });
        }
        let events: BTreeSet<ObjectId> = self.events.iter().map(|e| e.1).collect();
        let spaces: BTreeSet<u32> = self.spaces.iter().map(|s| s.id).collect();
        for (index, job) in self.pending.iter().enumerate() {
            let fail = SnapshotError::OutOfRange {
                part: Part::Job,
                index,
            };
            if job.ticket == 0 || job.ticket >= next_ticket {
                return Err(fail);
            }
            if !events.contains(&job.event) {
                return Err(SnapshotError::Dangling {
                    part: Part::Job,
                    index,
                });
            }
            for &(space, start, size) in &job.reveal {
                range(start, size, fail.clone())?;
                if !spaces.contains(&space) {
                    return Err(SnapshotError::Dangling {
                        part: Part::Job,
                        index,
                    });
                }
            }
            for &(_, start, size) in &job.releases {
                range(start, size, fail.clone())?;
            }
        }
        increasing(self.pending.iter().map(|j| j.ticket), Part::Job)
    }
}

/// The registrations and the jobs of a checked snapshot, their locks taken back.
type Reclaimed = (BTreeMap<u32, Registration>, BTreeMap<Ticket, Job>);

impl State {
    /// Takes every lock back from `kernel`: the registrations' in order, then the jobs', counting
    /// earlier ones of the same exact range (module docs, "Checks"). Run after [`State::check`].
    fn reclaim(&self, kernel: &dyn Reclaim) -> Result<Reclaimed, SnapshotError> {
        let mut counts: BTreeMap<SavedLock, u32> = BTreeMap::new();
        let mut index = 0usize;
        let mut take = |lock: SavedLock| {
            let earlier = counts.entry(lock).or_insert(0);
            let result = kernel
                .reclaim(lock.0, lock.1, lock.2, *earlier)
                .map_err(|error| SnapshotError::Release { index, error });
            *earlier = earlier.saturating_add(1);
            index = index.saturating_add(1);
            result
        };
        let mut registrations = BTreeMap::new();
        for (at, saved) in self.registrations.iter().enumerate() {
            let token = take((saved.process, saved.start, saved.size))?;
            let range = range(
                saved.start,
                saved.size,
                SnapshotError::OutOfRange {
                    part: Part::Registration,
                    index: at,
                },
            )?;
            registrations.insert(
                saved.handle,
                Registration {
                    range,
                    device: saved.device,
                    token,
                },
            );
        }
        let mut pending = BTreeMap::new();
        for job in &self.pending {
            let mut releases = Vec::new();
            for &lock in &job.releases {
                releases.push(take(lock)?);
            }
            let reveal = job
                .reveal
                .iter()
                .filter_map(|&(space, start, size)| {
                    PageRange::new(start, size).ok().map(|r| (AsId(space), r))
                })
                .collect();
            pending.insert(
                Ticket(job.ticket),
                Job {
                    event: job.event,
                    releases,
                    reveal,
                },
            );
        }
        Ok((registrations, pending))
    }
}

impl SavedSpace {
    /// The address space a checked snapshot holds, with an empty table (filled when the
    /// bookkeeping is rebuilt).
    fn space(&self) -> (AsId, Space) {
        let records = self
            .records
            .iter()
            .map(|r| {
                let spans = r
                    .spans
                    .iter()
                    .map(|&(low, size, kind)| (low, low.saturating_add(size), Kind(kind)))
                    .collect();
                let record = Record {
                    end: r.start.saturating_add(r.size),
                    handle: r.handle,
                    offset: r.offset,
                    spans,
                };
                (r.start, record)
            })
            .collect();
        let regions = self
            .regions
            .iter()
            .map(|&(start, size, id)| {
                let region = Region {
                    end: start.saturating_add(size),
                    id: RegionId(id),
                };
                (start, region)
            })
            .collect();
        let space = Space {
            vm: Arc::new(GpuVm::new()),
            records,
            regions,
            shim: self.shim,
        };
        (AsId(self.id), space)
    }
}

impl GpuMemory {
    /// Writes the bookkeeping (module docs, "Format"), after the checks a restore makes but the
    /// kernel's. Take it while nothing changes it.
    pub fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let state = State::of(self);
        state.check()?;
        let mut bytes = postcard::to_allocvec(&Header {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
        })
        .map_err(SnapshotError::Encode)?;
        bytes.extend(postcard::to_allocvec(&state).map_err(SnapshotError::Encode)?);
        Ok(bytes)
    }

    /// The bookkeeping [`save`](Self::save) wrote, checked first (module docs, "Checks"), its
    /// locks taken back from `kernel`; each address space's table is built again and given to
    /// `front`, and each waiting job's barrier sent again, in ticket order.
    pub fn restore(
        bytes: &[u8],
        kernel: &dyn Reclaim,
        front: &dyn GpuVmFront,
    ) -> Result<Self, SnapshotError> {
        let (header, rest) =
            postcard::take_from_bytes::<Header>(bytes).map_err(SnapshotError::Decode)?;
        if header.magic != SNAPSHOT_MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        if header.version != SNAPSHOT_VERSION {
            return Err(SnapshotError::UnsupportedVersion {
                found: header.version,
            });
        }
        let (state, rest) =
            postcard::take_from_bytes::<State>(rest).map_err(SnapshotError::Decode)?;
        if !rest.is_empty() {
            return Err(SnapshotError::TrailingBytes { count: rest.len() });
        }
        state.check()?;
        let (registrations, pending) = state.reclaim(kernel)?;
        let (space, handle, region, ticket) = state.next;
        let mut registered = Cover::default();
        for registration in registrations.values() {
            registered.add(registration.range.start(), registration.range.end());
        }
        let mut waiting: BTreeMap<AsId, Cover> = BTreeMap::new();
        for job in pending.values() {
            for &(space, range) in &job.reveal {
                waiting
                    .entry(space)
                    .or_default()
                    .add(range.start(), range.end());
            }
        }
        let memory = Self {
            events: state.events.iter().copied().collect(),
            worker: state.worker,
            devices: state
                .devices
                .iter()
                .map(|&(id, name, label)| (id, Device { name, label }))
                .collect(),
            spaces: state.spaces.iter().map(SavedSpace::space).collect(),
            registrations,
            pending,
            next: Counters {
                space,
                handle,
                region,
                ticket,
            },
            registered,
            waiting,
        };
        memory.rebuild(front);
        Ok(memory)
    }

    /// Lists each address space's two-page registrations in its table, gives `front` the table,
    /// publishes every piece into it, and sends a barrier for each waiting job (module docs,
    /// "Format").
    fn rebuild(&self, front: &dyn GpuVmFront) {
        for (&id, space) in &self.spaces {
            self.list_semaphore_registrations(&space.vm);
            front.create_address_space(id, Arc::clone(&space.vm));
            self.publish_all(id);
        }
        for &ticket in self.pending.keys() {
            front.barrier(ticket);
        }
    }
}

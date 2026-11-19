//! Replaying a capture (gpu-protocol, "Replay"): the seams that answer the GPU's threads from
//! their logs ([`ReplayMemory`] and the clocks and shader cache here), and [`replay`], which runs
//! the real front on its log, then the real back on the ring the front made and its own log.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use starquake_memory::{PageRange, WriteWatch};
use starquake_render::Backend;
use starquake_render::caps::Caps;
use starquake_render::recording::RecordingBackend;
use starquake_shader::lookup::{EngineState, ProgramMemory, ProgramRequest, Ticket, VariantKey};
use thiserror::Error;

use super::drive::{CollectError, Collector, Discard, MAX_COLLECTED_WORDS, Retired, entry_hash};
use super::format::{
    Access, BackRecord, Blobs, Capture, ContentHash, EngineStateKind, FrontRecord, Output,
    ShaderAnswer, StateRecord,
};
use super::record::Thread;
use crate::back::buffer::InvalidationSink;
use crate::back::draw::shaders::{SetLookup, ShaderLookup};
use crate::back::exec::{ExecConfig, ExecError, Executor};
use crate::back::fence::{
    CloseReason, FenceClock, FenceEnv, Fences, FrontWake, Report, WaiterTable,
};
use crate::front::{Front, FrontClock, FrontDeps, FrontError, FrontShared, OrderedTable};
use crate::interface::{AsId, ChannelId, Completions, Submitted};
use crate::memory::{GpuFault, GpuMemory};
use crate::ring::{RING_WORDS, RingEntry, ring};
use crate::syncpoint::{SnapshotError as SyncpointSnapshotError, Syncpoints};
use crate::vm::{GpuVm, SnapshotError as VmSnapshotError};

/// Where a replay diverged from its capture, and how: a finding (spec, "Replay": "A read that is
/// not the next one in the log ... stops the replay with the position").
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error(
    "the replay diverged at {} log record {record}: the capture has {found}, but {made}",
    .thread.name()
)]
pub struct Divergence {
    /// Whose log.
    pub thread: Thread,
    /// The record's index in that log, from 0: the record the GPU's call did not match.
    pub record: usize,
    /// The record, in plain words, or "the end of the log".
    pub found: String,
    /// What the replay did instead, in plain words.
    pub made: String,
}

/// Why a replay stopped before its end, or could not start.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ReplayError {
    /// The replay diverged.
    #[error(transparent)]
    Diverged(#[from] Divergence),
    /// The `n`-th ring entry the front pushed (from 0) differs from the capture's.
    #[error("ring entry {index} differs from the capture's")]
    RingDiffers {
        /// Its index.
        index: usize,
    },
    /// The front pushed a different number of ring entries from the capture's.
    #[error("the front pushed {replayed} ring entries; the capture has {captured}")]
    RingLength {
        /// The capture's count.
        captured: usize,
        /// The replay's.
        replayed: usize,
    },
    /// The render command stream differs from the capture's.
    #[error("the render command stream differs from the capture's")]
    RenderDiffers,
    /// A record needs a feature the replay does not have yet.
    #[error("{} log record {record} is {what}, which replay cannot feed back yet", .thread.name())]
    Unsupported {
        /// Whose log.
        thread: Thread,
        /// The record's index.
        record: usize,
        /// What it is, in plain words.
        what: &'static str,
    },
    /// The starting state holds front or back state the replay cannot restore yet.
    #[error("the capture's starting state holds {}, which replay cannot restore yet", .kind.name())]
    UnsupportedState {
        /// The kind.
        kind: EngineStateKind,
    },
    /// The starting state names an address space the header does not list, or one twice.
    #[error("the capture's starting state has a table for address space {}, which the header does not list or lists a table for twice", .space.0)]
    StateSpace {
        /// The space.
        space: AsId,
    },
    /// The starting state holds the syncpoints twice.
    #[error("the capture's starting state holds the syncpoints twice")]
    StateSyncpoints,
    /// An address table does not restore.
    #[error("the address table of space {} does not restore: {error}", .space.0)]
    AddressTable {
        /// The space.
        space: AsId,
        /// Why.
        error: VmSnapshotError,
    },
    /// The syncpoints do not restore.
    #[error("the syncpoints do not restore: {0}")]
    Syncpoints(SyncpointSnapshotError),
    /// The render backend's `Caps` do not decode.
    #[error("the render backend's capabilities do not decode: {0}")]
    Caps(String),
    /// The header's channels are not the ids a fresh front gives, in order.
    #[error("the capture's channel {expected} would be channel {made} in a fresh GPU", expected = .expected.0, made = .made.0)]
    ChannelId {
        /// The capture's id.
        expected: ChannelId,
        /// The id the front gave.
        made: ChannelId,
    },
    /// The front could not be set up or failed.
    #[error("the GPU front failed: {0}")]
    Front(#[from] FrontError),
    /// The back could not be set up: the executor's error, in words.
    #[error("the GPU back failed: {0}")]
    Back(String),
    /// The ring could not be made, or the thread that empties it failed.
    #[error("the ring between the front and the back failed: {0}")]
    Ring(String),
    /// The ring overflowed or failed.
    #[error(transparent)]
    Collect(#[from] CollectError),
}

/// What to replay up to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayOptions {
    /// Stop the back after it applied this many `Present` entries; `None` replays the whole
    /// capture.
    pub until_present: Option<u64>,
}

/// A write a replayed thread made, as the capture recorded it and the replay matched it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayedWrite {
    /// The thread.
    pub thread: Thread,
    /// The address space.
    pub space: AsId,
    /// The GPU address.
    pub va: u64,
    /// The bytes.
    pub data: Vec<u8>,
    /// The outcome the capture recorded.
    pub result: Result<(), GpuFault>,
}

/// What a replay produced.
#[derive(Clone, Debug)]
pub struct ReplayOutput {
    /// The ring entries the back applied, in order.
    pub ring: Vec<RingEntry<'static>>,
    /// The recording backend's snapshot text when the replay stopped.
    pub render: String,
    /// The writes both threads made, the front's then the back's, each in order.
    pub writes: Vec<ReplayedWrite>,
    /// `Present` entries the back applied.
    pub presents: u64,
    /// Whether the replay stopped at `--until present=N`, before the capture's end.
    pub stopped_early: bool,
}

/// One log being fed back, shared by the seams that answer from it.
struct Cursor<R> {
    thread: Thread,
    records: Vec<R>,
    next: usize,
    /// The first failure; sticky.
    failure: Option<ReplayError>,
    writes: Vec<ReplayedWrite>,
}

/// A log record as the cursor sees it.
trait Logged {
    fn access(&self) -> Option<&Access>;
    fn describe(&self) -> String;
}

fn describe_access(access: &Access) -> String {
    match *access {
        Access::Read { space, va, len, .. } => {
            format!(
                "a read of {len} bytes at {va:#x} in address space {}",
                space.0
            )
        }
        Access::Write {
            space, va, data, ..
        } => format!(
            "a write of blob {data} at {va:#x} in address space {}",
            space.0
        ),
        Access::GpuOnly { space, va, len, .. } => format!(
            "a GPU-only query of {len} bytes at {va:#x} in address space {}",
            space.0
        ),
        Access::Generation { space, .. } => {
            format!("a generation query of address space {}", space.0)
        }
    }
}

impl Logged for FrontRecord {
    fn access(&self) -> Option<&Access> {
        match self {
            Self::Access(access) => Some(access),
            _ => None,
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Turn => "a turn".to_owned(),
            Self::Submit { channel, .. } => format!("a submission to channel {}", channel.0),
            Self::Update { space, .. } => format!("an update of address space {}", space.0),
            Self::Barrier { ticket } => format!("a barrier with ticket {}", ticket.0),
            Self::Present { .. } => "a present".to_owned(),
            Self::Destroy { channel } => format!("a destroy of channel {}", channel.0),
            Self::Report { .. } => "a report from the back".to_owned(),
            Self::Syncpoint { id, value } => {
                format!("syncpoint {} at {value}", id.get())
            }
            Self::Latch { channel } => format!("a latch of channel {}", channel.0),
            Self::Clock { .. } => "a clock reading".to_owned(),
            Self::Access(access) => describe_access(access),
        }
    }
}

impl Logged for BackRecord {
    fn access(&self) -> Option<&Access> {
        match self {
            Self::Access(access) => Some(access),
            _ => None,
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Apply => "an applied ring entry".to_owned(),
            Self::Close => "a close of the open list".to_owned(),
            Self::Complete { completed } => {
                format!("completion processing at fence {completed}")
            }
            Self::Elapsed { .. } => "an elapsed-time reading".to_owned(),
            Self::Timestamp { .. } => "a timestamp reading".to_owned(),
            Self::Waiters { waiters } => format!("a waiter table of {} waiters", waiters.len()),
            Self::Drain { space, .. } => format!("a drain of address space {}", space.0),
            Self::ShaderLookup { requests, .. } => {
                format!("a shader lookup of {} programs", requests.len())
            }
            Self::ShaderWait { ticket, .. } => format!("a wait for shader ticket {ticket}"),
            Self::Access(access) => describe_access(access),
        }
    }
}

impl<R: Logged> Cursor<R> {
    fn new(thread: Thread, records: Vec<R>) -> Self {
        Self {
            thread,
            records,
            next: 0,
            failure: None,
            writes: Vec::new(),
        }
    }

    fn found(&self) -> String {
        self.records
            .get(self.next)
            .map_or_else(|| "the end of the log".to_owned(), Logged::describe)
    }

    /// Records the first failure: the GPU did `made` where the log has the next record.
    fn diverge(&mut self, made: String) {
        if self.failure.is_none() {
            self.failure = Some(ReplayError::Diverged(Divergence {
                thread: self.thread,
                record: self.next,
                found: self.found(),
                made,
            }));
        }
    }

    /// The next record if `wanted` accepts it (and steps past it); otherwise records the
    /// divergence, described by `made`.
    fn take<T>(
        &mut self,
        made: impl FnOnce() -> String,
        wanted: impl FnOnce(&R) -> Option<T>,
    ) -> Option<T> {
        if self.failure.is_some() {
            return None;
        }
        let taken = self.records.get(self.next).and_then(wanted);
        if taken.is_some() {
            self.next += 1;
        } else {
            self.diverge(made());
        }
        taken
    }
}

type Shared<R> = Arc<Mutex<Cursor<R>>>;

fn lock<R>(cursor: &Mutex<Cursor<R>>) -> MutexGuard<'_, Cursor<R>> {
    cursor.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A [`GpuMemory`] that answers one address space's calls from a thread's log, in order: a call
/// that is not the log's next record (another kind, space, address, length, or written bytes)
/// diverges, and from then on every call fails (a read or write faults at its address, a query
/// answers false or 0) and the replay stops at the next check.
pub struct ReplayMemory<R> {
    cursor: Shared<R>,
    blobs: Arc<Blobs>,
    space: AsId,
}

impl<R> ReplayMemory<R> {
    fn new(cursor: &Shared<R>, blobs: &Arc<Blobs>, space: AsId) -> Self {
        Self {
            cursor: Arc::clone(cursor),
            blobs: Arc::clone(blobs),
            space,
        }
    }
}

impl<R: Logged + Send> GpuMemory for ReplayMemory<R> {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), GpuFault> {
        let space = self.space;
        let len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
        let made = || {
            format!(
                "the GPU read {len} bytes at {va:#x} in address space {}",
                space.0
            )
        };
        let result = lock(&self.cursor).take(made, |record| match record.access() {
            Some(&Access::Read {
                space: s,
                va: v,
                len: l,
                result,
            }) if s == space && v == va && l == len => Some(result),
            _ => None,
        });
        match result {
            Some(Ok(blob)) => match self.blobs.get(blob) {
                Some(bytes) if bytes.len() == buf.len() => {
                    buf.copy_from_slice(bytes);
                    Ok(())
                }
                // `Capture::parse` checked every read's blob and length.
                _ => Err(GpuFault::Unmapped { va }),
            },
            Some(Err(fault)) => {
                buf.fill(0);
                Err(fault)
            }
            None => {
                buf.fill(0);
                Err(GpuFault::Unmapped { va })
            }
        }
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), GpuFault> {
        let space = self.space;
        let hash = ContentHash::of(data);
        let made = || {
            format!(
                "the GPU wrote blob {hash} at {va:#x} in address space {}",
                space.0
            )
        };
        let mut cursor = lock(&self.cursor);
        let result = cursor.take(made, |record| match record.access() {
            Some(&Access::Write {
                space: s,
                va: v,
                data: d,
                result,
            }) if s == space && v == va && d == hash => Some(result),
            _ => None,
        });
        match result {
            Some(result) => {
                let thread = cursor.thread;
                cursor.writes.push(ReplayedWrite {
                    thread,
                    space,
                    va,
                    data: data.to_vec(),
                    result,
                });
                result
            }
            None => Err(GpuFault::Unmapped { va }),
        }
    }

    fn generation(&self) -> u64 {
        let space = self.space;
        let made = || format!("the GPU asked address space {} for its generation", space.0);
        lock(&self.cursor)
            .take(made, |record| match record.access() {
                Some(&Access::Generation { space: s, value }) if s == space => Some(value),
                _ => None,
            })
            .unwrap_or(0)
    }

    fn is_gpu_only(&self, va: u64, len: u64) -> bool {
        let space = self.space;
        let made = || {
            format!(
                "the GPU asked whether {len} bytes at {va:#x} in address space {} are GPU-only",
                space.0
            )
        };
        lock(&self.cursor)
            .take(made, |record| match record.access() {
                Some(&Access::GpuOnly {
                    space: s,
                    va: v,
                    len: l,
                    answer,
                }) if s == space && v == va && l == len => Some(answer),
                _ => None,
            })
            .unwrap_or(false)
    }
}

/// The front's clock, answered from the front log.
struct ReplayFrontClock(Shared<FrontRecord>);

impl FrontClock for ReplayFrontClock {
    fn now(&self) -> Duration {
        lock(&self.0)
            .take(
                || "the front read its clock".to_owned(),
                |record| match record {
                    FrontRecord::Clock { now } => Some(*now),
                    _ => None,
                },
            )
            .unwrap_or(Duration::ZERO)
    }
}

/// The back's clock, answered from the back log.
struct ReplayFenceClock(Shared<BackRecord>);

impl FenceClock for ReplayFenceClock {
    fn elapsed(&self) -> Duration {
        lock(&self.0)
            .take(
                || "the back read how long its list has been open".to_owned(),
                |record| match record {
                    BackRecord::Elapsed { elapsed } => Some(*elapsed),
                    _ => None,
                },
            )
            .unwrap_or(Duration::ZERO)
    }

    fn gpu_timestamp(&self) -> u64 {
        lock(&self.0)
            .take(
                || "the back read a timestamp".to_owned(),
                |record| match record {
                    BackRecord::Timestamp { value } => Some(*value),
                    _ => None,
                },
            )
            .unwrap_or(0)
    }
}

/// The shader cache in replay: answers each lookup and wait from the back log.
///
/// The live cache's reads while it answered precede its record in the log (module docs,
/// "Shader lookups"); a lookup here steps past them (reads and queries, not writes) to the
/// lookup record. A pending answer is given back as it was. A ready or rejected answer stops the
/// replay ([`ReplayError::Unsupported`]): its shaders or reason are not in the capture.
struct ReplayShaders(Shared<BackRecord>);

impl ReplayShaders {
    fn answer(cursor: &mut Cursor<BackRecord>, answer: ShaderAnswer) -> SetLookup {
        match answer {
            ShaderAnswer::Pending { ticket } => SetLookup::Pending(Ticket::new(ticket)),
            ShaderAnswer::Ready | ShaderAnswer::Rejected => {
                if cursor.failure.is_none() {
                    cursor.failure = Some(ReplayError::Unsupported {
                        thread: Thread::Back,
                        record: cursor.next.saturating_sub(1),
                        what: match answer {
                            ShaderAnswer::Ready => "a ready shader answer",
                            _ => "a rejected shader answer",
                        },
                    });
                }
                SetLookup::Pending(Ticket::new(0))
            }
        }
    }
}

impl ShaderLookup for ReplayShaders {
    fn lookup(
        &mut self,
        requests: &[ProgramRequest],
        _variant: &VariantKey,
        _memory: &dyn ProgramMemory,
        _state: &dyn EngineState,
    ) -> SetLookup {
        let mut cursor = lock(&self.0);
        while let Some(BackRecord::Access(
            Access::Read { .. } | Access::GpuOnly { .. } | Access::Generation { .. },
        )) = cursor.records.get(cursor.next)
        {
            cursor.next += 1;
        }
        let made = || format!("the back looked up {} programs", requests.len());
        let answer = cursor.take(made, |record| match record {
            BackRecord::ShaderLookup {
                requests: logged,
                answer,
            } if logged.as_slice() == requests => Some(*answer),
            _ => None,
        });
        match answer {
            Some(answer) => Self::answer(&mut cursor, answer),
            None => SetLookup::Pending(Ticket::new(0)),
        }
    }

    fn wait(&mut self, ticket: Ticket) -> SetLookup {
        let mut cursor = lock(&self.0);
        let made = || format!("the back waited for shader ticket {}", ticket.id());
        let answer = cursor.take(made, |record| match record {
            BackRecord::ShaderWait { ticket: t, answer } if *t == ticket.id() => Some(*answer),
            _ => None,
        });
        match answer {
            Some(answer) => Self::answer(&mut cursor, answer),
            None => SetLookup::Pending(ticket),
        }
    }
}

impl InvalidationSink for ReplayShaders {
    fn invalidate(&mut self, _range: PageRange) {}
}

/// Where the back's reports go in replay: nowhere. The front was fed the reports the capture
/// recorded, before the back ran.
struct NoWake;

impl FrontWake for NoWake {
    fn wake(&self, _report: &Report) {}
}

/// The failure a cursor holds, as the error to stop with.
fn check<R>(cursor: &Shared<R>) -> Result<(), ReplayError> {
    match &lock(cursor).failure {
        Some(error) => Err(error.clone()),
        None => Ok(()),
    }
}

/// What the starting state restores: each listed address space's table, in the header's order,
/// and the syncpoints' snapshot.
struct Restored {
    tables: Vec<(AsId, Arc<GpuVm>)>,
    syncpoints: Option<Vec<u8>>,
}

fn restore_state(capture: &Capture) -> Result<Restored, ReplayError> {
    let mut saved: Vec<(AsId, Arc<GpuVm>)> = Vec::new();
    let mut syncpoints = None;
    for record in &capture.state {
        match *record {
            StateRecord::AddressTable { space, blob } => {
                if !capture.header.spaces.contains(&space)
                    || saved.iter().any(|(known, _)| *known == space)
                {
                    return Err(ReplayError::StateSpace { space });
                }
                let bytes = capture.blobs.get(blob).unwrap_or_default();
                let vm = GpuVm::restore(bytes)
                    .map_err(|error| ReplayError::AddressTable { space, error })?;
                saved.push((space, Arc::new(vm)));
            }
            StateRecord::Syncpoints { blob } => {
                if syncpoints.is_some() {
                    return Err(ReplayError::StateSyncpoints);
                }
                syncpoints = Some(capture.blobs.get(blob).unwrap_or_default().to_vec());
            }
            StateRecord::Engine { kind, .. } => {
                return Err(ReplayError::UnsupportedState { kind });
            }
        }
    }
    let tables = capture
        .header
        .spaces
        .iter()
        .map(|&space| {
            let vm = saved
                .iter()
                .find(|(known, _)| *known == space)
                .map_or_else(|| Arc::new(GpuVm::new()), |(_, vm)| Arc::clone(vm));
            (space, vm)
        })
        .collect();
    Ok(Restored { tables, syncpoints })
}

fn syncpoints(saved: Option<&[u8]>) -> Result<Arc<Syncpoints>, ReplayError> {
    let completions: Arc<dyn Completions> = Arc::new(Discard);
    Ok(Arc::new(match saved {
        Some(bytes) => Syncpoints::restore(bytes, completions).map_err(ReplayError::Syncpoints)?,
        None => Syncpoints::new(completions),
    }))
}

/// Records a divergence at the record a driver step just took.
fn diverge_at<R>(cursor: &Shared<R>, found: String, made: String) {
    let mut cursor = lock(cursor);
    if cursor.failure.is_none() {
        cursor.failure = Some(ReplayError::Diverged(Divergence {
            thread: cursor.thread,
            record: cursor.next.saturating_sub(1),
            found,
            made,
        }));
    }
}

/// The front side of a replay (module docs, "Replay").
struct FrontReplay {
    cursor: Shared<FrontRecord>,
    front: Front,
    shared: Arc<FrontShared>,
    syncpoints: Arc<Syncpoints>,
    waiters: Arc<WaiterTable>,
    ordered: Arc<OrderedTable>,
    retired: Arc<Retired>,
    collector: Collector,
}

/// The front phase's result.
struct FrontRun {
    entries: Vec<RingEntry<'static>>,
    ordered: Arc<OrderedTable>,
    retired: Arc<Retired>,
    writes: Vec<ReplayedWrite>,
}

impl FrontReplay {
    /// A fresh front over the restored state, with the header's spaces and channels made.
    fn new(
        capture: &Capture,
        restored: &Restored,
        blobs: &Arc<Blobs>,
    ) -> Result<Self, ReplayError> {
        let cursor: Shared<FrontRecord> = Arc::new(Mutex::new(Cursor::new(
            Thread::Front,
            capture.front.clone(),
        )));
        let (producer, consumer) =
            ring(RING_WORDS).map_err(|error| ReplayError::Ring(error.to_string()))?;
        let collector = Collector::start(consumer, MAX_COLLECTED_WORDS)
            .map_err(|error| ReplayError::Ring(error.to_string()))?;
        let shared = Arc::new(FrontShared::new());
        let syncpoints = syncpoints(restored.syncpoints.as_deref())?;
        let waiters = Arc::new(WaiterTable::new());
        let ordered = Arc::new(OrderedTable::new());
        let retired = Arc::new(Retired::default());
        let front = Front::new(FrontDeps {
            shared: Arc::clone(&shared),
            ring: producer,
            syncpoints: Arc::clone(&syncpoints),
            completions: Arc::new(Discard),
            waiters: Arc::clone(&waiters),
            ordered: Arc::clone(&ordered),
            clock: Arc::new(ReplayFrontClock(Arc::clone(&cursor))),
        })
        .with_retired(Arc::<Retired>::clone(&retired));
        for (space, vm) in &restored.tables {
            let memory: Arc<dyn GpuMemory> = Arc::new(ReplayMemory::new(&cursor, blobs, *space));
            shared.create_address_space(*space, Arc::clone(vm), memory);
        }
        for channel in &capture.header.channels {
            let made =
                shared
                    .create_channel(channel.config)
                    .map_err(|_| ReplayError::ChannelId {
                        expected: channel.id,
                        made: ChannelId(u32::MAX),
                    })?;
            if made != channel.id {
                return Err(ReplayError::ChannelId {
                    expected: channel.id,
                    made,
                });
            }
        }
        Ok(Self {
            cursor,
            front,
            shared,
            syncpoints,
            waiters,
            ordered,
            retired,
            collector,
        })
    }

    /// The next record for the driver, stepped past; `None` at the log's end or after a
    /// failure. A record only a turn consumes (a clock reading or an access) diverges here.
    fn next(&self) -> Option<FrontRecord> {
        let mut cursor = lock(&self.cursor);
        if cursor.failure.is_some() {
            return None;
        }
        let record = cursor.records.get(cursor.next).cloned()?;
        if matches!(record, FrontRecord::Clock { .. } | FrontRecord::Access(_)) {
            cursor.diverge("the front had finished its turn".to_owned());
            return None;
        }
        cursor.next += 1;
        Some(record)
    }

    /// Feeds one driver record to the front.
    fn apply(&mut self, record: FrontRecord) -> Result<(), ReplayError> {
        match record {
            FrontRecord::Turn => {
                self.front.turn()?;
            }
            FrontRecord::Submit {
                channel,
                submission,
            } => {
                if self.shared.submit(channel, submission) == Submitted::Full {
                    diverge_at(
                        &self.cursor,
                        format!("a submission the front took on channel {}", channel.0),
                        "the channel's queue was full".to_owned(),
                    );
                }
            }
            FrontRecord::Update {
                space,
                update,
                ticket,
            } => self.shared.update_vm(space, update, ticket),
            FrontRecord::Barrier { ticket } => self.shared.barrier(ticket),
            FrontRecord::Present { request } => self.shared.present(request),
            FrontRecord::Destroy { channel } => self.shared.destroy_channel(channel),
            FrontRecord::Report {
                markers,
                released,
                incremented,
            } => {
                self.shared.wake(&Report {
                    released: u64::from(released),
                    incremented: u64::from(incremented),
                    markers,
                    ..Report::default()
                });
            }
            FrontRecord::Syncpoint { id, value } => {
                let current = self.syncpoints.read(id);
                self.syncpoints
                    .increment_by(id, value.wrapping_sub(current));
            }
            FrontRecord::Latch { channel } => {
                if !self.waiters.latch(channel) {
                    diverge_at(
                        &self.cursor,
                        format!("a latch of channel {}'s acquire", channel.0),
                        "the channel is not parked on an acquire".to_owned(),
                    );
                }
            }
            FrontRecord::Clock { .. } | FrontRecord::Access(_) => {}
        }
        check(&self.cursor)
    }

    /// Runs the whole front log, then stops the front and collects the ring it pushed.
    fn run(mut self) -> Result<FrontRun, ReplayError> {
        while let Some(record) = self.next() {
            self.apply(record)?;
        }
        check(&self.cursor)?;
        self.shared.stop();
        // A stopping front publishes what it pushed and reads nothing (`Front::turn`).
        self.front.turn()?;
        let Self {
            cursor,
            front,
            ordered,
            retired,
            collector,
            ..
        } = self;
        drop(front);
        let (entries, ended) = collector.finish();
        ended?;
        let writes = std::mem::take(&mut lock(&cursor).writes);
        Ok(FrontRun {
            entries,
            ordered,
            retired,
            writes,
        })
    }
}

/// The executor's refusal of a setup call, as [`ReplayError::Back`].
fn back(error: &ExecError) -> ReplayError {
    ReplayError::Back(error.to_string())
}

/// Compares the entries the front pushed with the capture's ring hashes, first difference first.
fn compare_ring(capture: &Capture, entries: &[RingEntry<'static>]) -> Result<(), ReplayError> {
    let captured: Vec<ContentHash> = capture
        .outputs
        .iter()
        .filter_map(|output| match *output {
            Output::Ring { hash } => Some(hash),
            Output::Render { .. } => None,
        })
        .collect();
    for (index, (entry, hash)) in entries.iter().zip(&captured).enumerate() {
        if entry_hash(entry) != Some(*hash) {
            return Err(ReplayError::RingDiffers { index });
        }
    }
    if captured.len() != entries.len() {
        return Err(ReplayError::RingLength {
            captured: captured.len(),
            replayed: entries.len(),
        });
    }
    Ok(())
}

/// The back side of a replay (module docs, "Replay").
struct BackReplay {
    cursor: Shared<BackRecord>,
    executor: Executor,
    backend: Arc<RecordingBackend>,
    waiters: Arc<WaiterTable>,
    front: FrontRun,
    applied: Vec<RingEntry<'static>>,
    presents: u64,
}

impl BackReplay {
    /// A fresh back over the restored state, with the header's spaces and channels added.
    fn new(
        capture: &Capture,
        restored: &Restored,
        blobs: &Arc<Blobs>,
        front: FrontRun,
    ) -> Result<Self, ReplayError> {
        let caps: Caps = capture
            .blobs
            .get(capture.header.caps)
            .ok_or_else(|| ReplayError::Caps("the blob is missing".to_owned()))
            .and_then(|bytes| {
                postcard::from_bytes(bytes).map_err(|error| ReplayError::Caps(error.to_string()))
            })?;
        let cursor: Shared<BackRecord> =
            Arc::new(Mutex::new(Cursor::new(Thread::Back, capture.back.clone())));
        let backend = Arc::new(RecordingBackend::new(caps, capture.header.memory_allowed));
        let waiters = Arc::new(WaiterTable::new());
        let fences = Fences::new(
            Arc::<RecordingBackend>::clone(&backend),
            FenceEnv {
                syncpoints: syncpoints(restored.syncpoints.as_deref())?,
                waiters: Arc::clone(&waiters),
                completions: Arc::new(Discard),
                clock: Arc::new(ReplayFenceClock(Arc::clone(&cursor))),
                front: Arc::new(NoWake),
            },
            capture.header.mode,
        );
        let mut executor = Executor::new(
            Arc::<RecordingBackend>::clone(&backend),
            fences,
            Box::new(ReplayShaders(Arc::clone(&cursor))),
            Arc::clone(&front.ordered),
            ExecConfig {
                setting: capture.header.setting,
                ..ExecConfig::default()
            },
        );
        for &space in &capture.header.spaces {
            let memory: Arc<dyn GpuMemory> = Arc::new(ReplayMemory::new(&cursor, blobs, space));
            executor
                .add_space(space, memory, Arc::new(WriteWatch::new()))
                .map_err(|error| back(&error))?;
        }
        for channel in &capture.header.channels {
            executor
                .add_channel(channel.id, channel.config.space)
                .map_err(|error| back(&error))?;
        }
        if let Some(&first) = capture.header.spaces.first() {
            executor
                .set_present_space(first)
                .map_err(|error| back(&error))?;
        }
        Ok(Self {
            cursor,
            executor,
            backend,
            waiters,
            front,
            applied: Vec::new(),
            presents: 0,
        })
    }

    /// The next record for the driver, stepped past; `None` at the log's end or after a
    /// failure. A record only a step consumes diverges here.
    fn next(&self) -> Result<Option<BackRecord>, ReplayError> {
        let mut cursor = lock(&self.cursor);
        if cursor.failure.is_some() {
            return Ok(None);
        }
        let Some(record) = cursor.records.get(cursor.next).cloned() else {
            return Ok(None);
        };
        match record {
            BackRecord::Apply
            | BackRecord::Close
            | BackRecord::Complete { .. }
            | BackRecord::Waiters { .. } => {
                cursor.next += 1;
                Ok(Some(record))
            }
            BackRecord::Drain { .. } => Err(ReplayError::Unsupported {
                thread: Thread::Back,
                record: cursor.next,
                what: "a write-tracking drain",
            }),
            _ => {
                cursor.diverge("the back had finished its step".to_owned());
                Ok(None)
            }
        }
    }

    /// Takes one driver record.
    fn apply(&mut self, record: BackRecord) -> Result<(), ReplayError> {
        match record {
            BackRecord::Apply => {
                let taken = u64::try_from(self.applied.len()).unwrap_or(u64::MAX);
                for channel in self.front.retired.due(taken) {
                    // Logged and counted by the executor, as when the capture was made.
                    let _removed = self.executor.remove_channel(channel);
                }
                let Some(entry) = self.front.entries.get(self.applied.len()).cloned() else {
                    diverge_at(
                        &self.cursor,
                        "an applied ring entry".to_owned(),
                        format!("the front pushed only {} entries", self.applied.len()),
                    );
                    return check(&self.cursor);
                };
                // A refused entry is logged and counted, as it was when the capture was made.
                let _applied = self.executor.apply(&entry);
                if matches!(entry, RingEntry::Present(_)) {
                    self.presents = self.presents.saturating_add(1);
                }
                self.applied.push(entry);
            }
            BackRecord::Close => {
                // The capture's close had the same outcome, logged and counted.
                let _closed = self.executor.close(CloseReason::RingEmpty);
            }
            BackRecord::Complete { completed } => {
                let now = self.backend.completed().get();
                if now == completed {
                    self.executor.complete();
                } else {
                    diverge_at(
                        &self.cursor,
                        format!("completion processing at fence {completed}"),
                        format!("the backend had completed fence {now}"),
                    );
                }
            }
            BackRecord::Waiters { waiters } => self.waiters.replace(&waiters),
            _ => {}
        }
        check(&self.cursor)
    }
}

/// Replays `capture` (module docs, "Replay"): the front on its log, the ring against the
/// capture's hashes, then the back on that ring and its log, and, when the back ran to the end,
/// the render command stream against the capture's.
///
/// # Errors
///
/// [`ReplayError`]: the first failure, in that order.
pub fn replay(capture: &Capture, options: ReplayOptions) -> Result<ReplayOutput, ReplayError> {
    let restored = restore_state(capture)?;
    let blobs = Arc::new(capture.blobs.clone());
    let front = FrontReplay::new(capture, &restored, &blobs)?.run()?;
    compare_ring(capture, &front.entries)?;
    let mut back = BackReplay::new(capture, &restored, &blobs, front)?;
    let mut stopped_early = false;
    loop {
        if options.until_present.is_some_and(|n| back.presents >= n) {
            stopped_early = lock(&back.cursor).next < capture.back.len();
            break;
        }
        let Some(record) = back.next()? else {
            break;
        };
        back.apply(record)?;
    }
    check(&back.cursor)?;
    let render = back.backend.snapshot();
    if !stopped_early {
        let captured = capture.outputs.iter().find_map(|output| match *output {
            Output::Render { text } => capture.blobs.get(text),
            Output::Ring { .. } => None,
        });
        if captured.is_some_and(|text| text != render.as_bytes()) {
            return Err(ReplayError::RenderDiffers);
        }
    }
    let mut writes = std::mem::take(&mut back.front.writes);
    writes.append(&mut lock(&back.cursor).writes);
    Ok(ReplayOutput {
        ring: back.applied,
        render,
        writes,
        presents: back.presents,
        stopped_early,
    })
}

/// The seams alone, over one log each, for the seams' tests (`super::tests::seams`).
#[cfg(test)]
pub(crate) mod seams {
    use super::{
        Arc, BackRecord, Blobs, Cursor, FrontRecord, Mutex, ReplayError, ReplayFenceClock,
        ReplayFrontClock, ReplayMemory, ReplayShaders, ReplayedWrite, Shared, Thread, check, lock,
    };
    use crate::back::draw::shaders::ShaderLookup;
    use crate::back::fence::FenceClock;
    use crate::front::FrontClock;
    use crate::interface::AsId;

    /// A front log being fed back: its memory for `space`, its clock, and its failure.
    pub(crate) struct Front {
        cursor: Shared<FrontRecord>,
        blobs: Arc<Blobs>,
    }

    impl Front {
        pub(crate) fn new(records: Vec<FrontRecord>, blobs: Blobs) -> Self {
            Self {
                cursor: Arc::new(Mutex::new(Cursor::new(Thread::Front, records))),
                blobs: Arc::new(blobs),
            }
        }
        pub(crate) fn memory(&self, space: AsId) -> ReplayMemory<FrontRecord> {
            ReplayMemory::new(&self.cursor, &self.blobs, space)
        }
        pub(crate) fn clock(&self) -> impl FrontClock {
            ReplayFrontClock(Arc::clone(&self.cursor))
        }
        pub(crate) fn check(&self) -> Result<(), ReplayError> {
            check(&self.cursor)
        }
        pub(crate) fn next(&self) -> usize {
            lock(&self.cursor).next
        }
        pub(crate) fn writes(&self) -> Vec<ReplayedWrite> {
            lock(&self.cursor).writes.clone()
        }
    }

    /// A back log being fed back: its clock and shader cache, and its failure.
    pub(crate) struct Back {
        cursor: Shared<BackRecord>,
    }

    impl Back {
        pub(crate) fn new(records: Vec<BackRecord>) -> Self {
            Self {
                cursor: Arc::new(Mutex::new(Cursor::new(Thread::Back, records))),
            }
        }
        pub(crate) fn clock(&self) -> impl FenceClock {
            ReplayFenceClock(Arc::clone(&self.cursor))
        }
        pub(crate) fn shaders(&self) -> impl ShaderLookup {
            ReplayShaders(Arc::clone(&self.cursor))
        }
        pub(crate) fn check(&self) -> Result<(), ReplayError> {
            check(&self.cursor)
        }
        pub(crate) fn next(&self) -> usize {
            lock(&self.cursor).next
        }
    }
}

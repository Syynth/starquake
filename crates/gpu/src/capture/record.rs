//! Recording a capture: the [`Recorder`] that holds both input logs and the blob table, and the
//! seams that write to it as the GPU's threads consume things ([`RecordingMemory`],
//! [`RecordingFrontClock`], [`RecordingFenceClock`], [`RecordingShaders`]). The single-threaded
//! driver that records a whole capture from hand-written streams is the capture builder
//! (`super::builder`, in the `test-support` feature).

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use starquake_memory::PageRange;
use starquake_shader::lookup::{EngineState, ProgramMemory, ProgramRequest, Ticket, VariantKey};

use super::format::{Access, BackRecord, Blobs, ContentHash, FrontRecord, ShaderAnswer};
use crate::back::buffer::InvalidationSink;
use crate::back::draw::shaders::{SetLookup, ShaderLookup};
use crate::back::exec::ShaderCache;
use crate::back::fence::FenceClock;
use crate::front::FrontClock;
use crate::interface::AsId;
use crate::memory::{GpuFault, GpuMemory};

/// Which thread's log a record goes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Thread {
    /// The front thread.
    Front,
    /// The back thread.
    Back,
}

impl Thread {
    /// The thread's name, as messages print it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Front => "front",
            Self::Back => "back",
        }
    }
}

/// Why a recording failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RecordError {
    /// The blobs recorded passed the recorder's limit; everything after the record that passed it
    /// was dropped.
    #[error("the capture passed its limit of {limit} bytes of blobs")]
    Limit {
        /// The limit.
        limit: usize,
    },
}

/// Both input logs and the blob table, as the threads record them. Shared by every recording
/// seam of one capture; a mutex held only to push one record.
#[derive(Debug)]
pub struct Recorder {
    state: Mutex<Logs>,
}

#[derive(Debug)]
struct Logs {
    front: Vec<FrontRecord>,
    back: Vec<BackRecord>,
    blobs: Blobs,
    limit: usize,
    failed: Option<RecordError>,
}

/// What a [`Recorder`] gives when it is finished.
#[derive(Debug)]
pub struct Recorded {
    /// The front log.
    pub front: Vec<FrontRecord>,
    /// The back log.
    pub back: Vec<BackRecord>,
    /// The blob table.
    pub blobs: Blobs,
}

impl Recorder {
    /// An empty recorder that keeps at most `limit` bytes of blobs. Past it, the recording fails
    /// (sticky: nothing more is kept, and [`Recorder::finish`] reports the failure).
    pub fn new(limit: usize) -> Self {
        Self {
            state: Mutex::new(Logs {
                front: Vec::new(),
                back: Vec::new(),
                blobs: Blobs::new(),
                limit,
                failed: None,
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Logs> {
        // Every method leaves the logs whole before it could panic.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Appends `record` to the front log.
    pub fn front(&self, record: FrontRecord) {
        let mut logs = self.lock();
        if logs.failed.is_none() {
            logs.front.push(record);
        }
    }

    /// Appends `record` to the back log.
    pub fn back(&self, record: BackRecord) {
        let mut logs = self.lock();
        if logs.failed.is_none() {
            logs.back.push(record);
        }
    }

    /// Stores `bytes` in the blob table and gives its hash; past the limit the recording fails.
    pub fn blob(&self, bytes: &[u8]) -> ContentHash {
        let mut logs = self.lock();
        let hash = ContentHash::of(bytes);
        if logs.failed.is_some() || logs.blobs.get(hash).is_some() {
            return hash;
        }
        let total = logs.blobs.bytes().saturating_add(bytes.len());
        if total > logs.limit {
            logs.failed = Some(RecordError::Limit { limit: logs.limit });
            return hash;
        }
        logs.blobs.insert(bytes)
    }

    /// Appends `access` to `thread`'s log.
    pub fn access(&self, thread: Thread, access: Access) {
        match thread {
            Thread::Front => self.front(FrontRecord::Access(access)),
            Thread::Back => self.back(BackRecord::Access(access)),
        }
    }

    /// How many records the back log holds so far.
    pub fn back_len(&self) -> usize {
        self.lock().back.len()
    }

    /// The logs and blobs recorded so far, leaving the recorder empty.
    ///
    /// # Errors
    ///
    /// The first [`RecordError`]: the logs lost records.
    pub fn take(&self) -> Result<Recorded, RecordError> {
        let mut logs = self.lock();
        if let Some(error) = logs.failed {
            return Err(error);
        }
        Ok(Recorded {
            front: std::mem::take(&mut logs.front),
            back: std::mem::take(&mut logs.back),
            blobs: std::mem::take(&mut logs.blobs),
        })
    }
}

/// A [`GpuMemory`] that records every call to the memory it wraps, with the answer, in one
/// thread's log: a read's bytes or fault, a write's bytes and outcome, and the two queries.
pub struct RecordingMemory {
    inner: Arc<dyn GpuMemory>,
    recorder: Arc<Recorder>,
    thread: Thread,
    space: AsId,
}

impl RecordingMemory {
    /// Records `thread`'s calls to `inner`, the memory of address space `space`.
    pub fn new(
        inner: Arc<dyn GpuMemory>,
        recorder: Arc<Recorder>,
        thread: Thread,
        space: AsId,
    ) -> Self {
        Self {
            inner,
            recorder,
            thread,
            space,
        }
    }
}

/// A length as the file keeps it.
fn len64(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

impl GpuMemory for RecordingMemory {
    fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), GpuFault> {
        let result = self.inner.read(va, buf);
        let recorded = match result {
            Ok(()) => Ok(self.recorder.blob(buf)),
            Err(fault) => Err(fault),
        };
        self.recorder.access(
            self.thread,
            Access::Read {
                space: self.space,
                va,
                len: len64(buf.len()),
                result: recorded,
            },
        );
        result
    }

    fn write(&self, va: u64, data: &[u8]) -> Result<(), GpuFault> {
        let result = self.inner.write(va, data);
        let blob = self.recorder.blob(data);
        self.recorder.access(
            self.thread,
            Access::Write {
                space: self.space,
                va,
                data: blob,
                result,
            },
        );
        result
    }

    fn generation(&self) -> u64 {
        let value = self.inner.generation();
        self.recorder.access(
            self.thread,
            Access::Generation {
                space: self.space,
                value,
            },
        );
        value
    }

    fn is_gpu_only(&self, va: u64, len: u64) -> bool {
        let answer = self.inner.is_gpu_only(va, len);
        self.recorder.access(
            self.thread,
            Access::GpuOnly {
                space: self.space,
                va,
                len,
                answer,
            },
        );
        answer
    }
}

/// A [`FrontClock`] that records every reading of the clock it wraps (the acquire poll's ticks).
pub struct RecordingFrontClock {
    inner: Arc<dyn FrontClock>,
    recorder: Arc<Recorder>,
}

impl RecordingFrontClock {
    /// Records the readings of `inner`.
    pub fn new(inner: Arc<dyn FrontClock>, recorder: Arc<Recorder>) -> Self {
        Self { inner, recorder }
    }
}

impl FrontClock for RecordingFrontClock {
    fn now(&self) -> Duration {
        let now = self.inner.now();
        self.recorder.front(FrontRecord::Clock { now });
        now
    }
}

/// A [`FenceClock`] that records every reading of the clock it wraps.
pub struct RecordingFenceClock {
    inner: Arc<dyn FenceClock>,
    recorder: Arc<Recorder>,
}

impl RecordingFenceClock {
    /// Records the readings of `inner`.
    pub fn new(inner: Arc<dyn FenceClock>, recorder: Arc<Recorder>) -> Self {
        Self { inner, recorder }
    }
}

impl FenceClock for RecordingFenceClock {
    fn elapsed(&self) -> Duration {
        let elapsed = self.inner.elapsed();
        self.recorder.back(BackRecord::Elapsed { elapsed });
        elapsed
    }

    fn gpu_timestamp(&self) -> u64 {
        let value = self.inner.gpu_timestamp();
        self.recorder.back(BackRecord::Timestamp { value });
        value
    }
}

/// The kind of a lookup's answer, as the capture keeps it.
pub(crate) fn answer_of(answer: &SetLookup) -> ShaderAnswer {
    match answer {
        SetLookup::Ready(_) => ShaderAnswer::Ready,
        SetLookup::Pending(ticket) => ShaderAnswer::Pending {
            ticket: ticket.id(),
        },
        SetLookup::Rejected(_) => ShaderAnswer::Rejected,
    }
}

/// A shader cache that records every lookup and wait of the cache it wraps, after the cache
/// answered: the reads the cache made while answering are in the back log just before the
/// record (through the back's [`RecordingMemory`]), and the lookup record counts them (the back
/// log's growth while the cache answered), so a replay steps past exactly those.
pub struct RecordingShaders {
    inner: Box<dyn ShaderCache>,
    recorder: Arc<Recorder>,
}

impl RecordingShaders {
    /// Records the lookups and waits of `inner`.
    pub fn new(inner: Box<dyn ShaderCache>, recorder: Arc<Recorder>) -> Self {
        Self { inner, recorder }
    }
}

impl ShaderLookup for RecordingShaders {
    fn lookup(
        &mut self,
        requests: &[ProgramRequest],
        variant: &VariantKey,
        memory: &dyn ProgramMemory,
        state: &dyn EngineState,
    ) -> SetLookup {
        let before = self.recorder.back_len();
        let answer = self.inner.lookup(requests, variant, memory, state);
        let made = self.recorder.back_len().saturating_sub(before);
        self.recorder.back(BackRecord::ShaderLookup {
            cache_accesses: u32::try_from(made).unwrap_or(u32::MAX),
            requests: requests.to_vec(),
            answer: answer_of(&answer),
        });
        answer
    }

    fn wait(&mut self, ticket: Ticket) -> SetLookup {
        let answer = self.inner.wait(ticket);
        self.recorder.back(BackRecord::ShaderWait {
            ticket: ticket.id(),
            answer: answer_of(&answer),
        });
        answer
    }
}

impl InvalidationSink for RecordingShaders {
    fn invalidate(&mut self, range: PageRange) {
        self.inner.invalidate(range);
    }
}

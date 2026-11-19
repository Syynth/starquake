//! One channel on the front ([`Lane`]): its pusher, macro state and register files, its queue of
//! submissions, what it waits on, and the [`ChannelSink`] that routes what its pusher executes
//! ([`LaneSink`]).
//!
//! # Running a channel
//!
//! [`Lane::run`] runs one channel for at most a quantum of words: it gives the pusher the
//! submission's entries one by one, runs it, and when a submission's entries are all read, marks
//! the end of the submission (the pusher keeps a method still owed words for the next one),
//! pushes a `WaitForIdle` if the submission asked for one and has increments, and then its
//! increments as `Syncpoint` entries for the channel's syncpoint (row 32; the spec, "Channels and
//! submissions"). It stops when the queue is empty, the channel waits on something
//! ([`Wait`]), or the quantum is used.
//!
//! - **The quantum** counts every command word the pusher executes
//!   ([`crate::channel::Channel::words_read`]) and every increment pushed. A submission's
//!   increments are a guest-chosen count (gsv passes up to `u32::MAX`), so they are pushed a
//!   quantum at a time like words, and a channel with millions of them does not hold the others
//!   back; the time they take in all still grows with the count (an owner question: a
//!   `Syncpoint` entry with a count would make it constant).
//! - **`WaitForIdle` comes only with increments.** Eden-NX2 pushes the wait for idle inside its
//!   increment command lists only (`SubmitGPFIFOImpl`, `BuildIncrementWithWfiCommandList` and
//!   `BuildIncrementCountCommandList` in
//!   `eden-nx2/src/core/hle/service/nvdrv/devices/nvhost_gpu.cpp` @ 50fa86bb), so a submission
//!   with no increments sends none (`wait_for_idle_comes_before_the_increments_and_only_with_them`).
//! - **The fence to wait on** is checked with [`Syncpoints::reached`] before the submission's
//!   first entry; an unmet fence parks the channel ([`Wait::Fence`]). It is a front wait, not a
//!   waiter in the shared table: a syncpoint only counts up, so no value can be missed and there
//!   is nothing to latch. It is rechecked when the back reports an increment, a submission
//!   arrives, or the poll fires (gsv's video worker increments syncpoints too).
//! - **An unmet acquire** registers a waiter in the shared table (the channel, its address space,
//!   the semaphore and the condition) *before* the pusher reads the semaphore again (spec, step
//!   1; row 70), so a release performed in between is either seen by that read or latches the
//!   waiter, which the lane then looks at (`a_release_between_registering_and_the_second_read_is_not_missed`).
//! - **`SYNC_WAIT`** on a GP entry (`NVC76F_GP_ENTRY1_SYNC`, bit 31 of word 1) makes the channel
//!   wait, before it starts the entry, until the back has performed everything pushed before it:
//!   if anything pushed is unperformed, the lane pushes a `Marker` and waits for its report
//!   ([`Wait::Sync`]). **Guess** that the wait comes before the entry, from the field's name and
//!   value (`SYNC_PROCEED`, `SYNC_WAIT`) in `clc76f.h`. Eden-NX2 waits *after* an entry with
//!   the bit set, before the next, and only with its `sync_memory_operations` setting on
//!   (`DmaPusher::Step` in `eden-nx2/src/video_core/dma_pusher.cpp` @ 50fa86bb); a divergence
//!   pinned by `sync_wait_holds_the_entry_until_the_back_has_performed_earlier_work`. The wait
//!   covers everything pushed by every channel, which is at least the channel's own work.
//! - **A macro call that needs a catch-up** (`crate::macros`, "MME DMA reads") holds the pusher;
//!   the lane pushes a `Marker` and waits for its report ([`Wait::CatchUp`]), then the front runs
//!   the call again through [`ProcessMacros::resume`].
//!
//! # Routing words
//!
//! [`LaneSink`] is the pusher's [`ChannelSink`]: host ring entries go to the ring;
//! `SET_REFERENCE` goes to the ring as [`RingEntry::SetReference`]; an engine word for a method
//! [`crate::macros::takes`] names goes through [`ProcessMacros::write`], and any other through
//! the class's register file, batched: consecutive words to one method of one class go to
//! [`RegFile::write_run`] together (at most [`MAX_RUN_WORDS`] at a time), so a run of data words
//! to a trigger is one `Run` entry (spec, "Engine methods: the register file"). The batch is
//! written before anything else reaches the ring or the macro host and when the pusher returns,
//! so entries keep stream order. `holds` is [`MacroChannel::is_waiting`].
//!
//! A channel keeps one register file per class it has sent a word to. The 3D and compute files
//! are kept for the channel's life; files of every other class (which drop nothing and are read
//! by nothing, `crate::regs`, "Divergences") are at most [`MAX_OTHER_FILES`], the oldest dropped
//! first, so a guest binding class after class cannot grow the front without bound
//! (`files_of_other_classes_are_bounded`).

use std::collections::{BTreeMap, VecDeque};

use crate::back::fence::{Waiter, WaiterTable};
use crate::channel::{
    Acquire, Channel, ChannelError, ChannelSink, ChannelState, EngineWord, Note, RunOutcome,
    StateError,
};
use crate::fifo::{GpEntry, Sync};
use crate::interface::{ChannelConfig, ChannelId, Fence, GpfifoEntry, Submission, charge};
use crate::macros::{
    CallResult, Engine, MacroChannel, MacroEnv, ProcessMacros, WordOutcome, takes,
};
use crate::memory::GpuMemory;
use crate::regs::{EntrySink, RegFile};
use crate::ring::{ClassId, MAX_RUN_WORDS, MarkerToken, RingEntry, SUBCHANNEL_COUNT};
use crate::syncpoint::Syncpoints;

use super::FrontCounters;
use super::ring_sink::{RingSink, Unperformed};

/// The most register files of classes other than 3D and compute one channel keeps: one per
/// subchannel (a **guess** at what a real stream binds at once; module docs).
pub const MAX_OTHER_FILES: usize = SUBCHANNEL_COUNT as usize;

/// What a channel waits on before it can go on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Wait {
    /// A semaphore acquire, registered in the waiter table.
    Acquire(Acquire),
    /// A submission's fence.
    Fence(Fence),
    /// The back's report of this marker, before a `SYNC_WAIT` entry.
    Sync(MarkerToken),
    /// The back's report of this marker, before a macro call runs again.
    CatchUp(MarkerToken),
}

/// A submission taken by the front and not started.
pub(super) struct Queued {
    pub(super) seq: u64,
    pub(super) submission: Submission,
}

/// The submission a channel is working through.
struct Active {
    seq: u64,
    entries: Vec<GpfifoEntry>,
    /// The next entry to start.
    next: usize,
    /// The fence still to wait on.
    fence: Option<Fence>,
    /// The entry whose `SYNC_WAIT` has been met.
    synced: Option<usize>,
    /// Whether the end of the submission was marked.
    ended: bool,
    /// Whether a `WaitForIdle` is still to be pushed.
    wfi: bool,
    /// Increments still to push.
    increments: u32,
}

impl Active {
    fn new(queued: Queued) -> Self {
        let Submission {
            entries,
            wait,
            increments,
            wfi,
        } = queued.submission;
        Self {
            seq: queued.seq,
            entries,
            next: 0,
            fence: wait,
            synced: None,
            ended: false,
            wfi: wfi && increments > 0,
            increments,
        }
    }

    /// Entries not yet started.
    fn unstarted(&self) -> usize {
        self.entries.len().saturating_sub(self.next)
    }
}

/// One channel's register files (module docs).
pub(super) struct Files {
    channel: ChannelId,
    by_class: BTreeMap<ClassId, RegFile>,
    /// Classes other than 3D and compute with a file, oldest first.
    others: VecDeque<ClassId>,
}

impl Files {
    fn new(channel: ChannelId) -> Self {
        Self {
            channel,
            by_class: BTreeMap::new(),
            others: VecDeque::new(),
        }
    }

    /// The file of `class`, made if there is none.
    fn get(&mut self, class: ClassId) -> &mut RegFile {
        if !self.by_class.contains_key(&class) && Engine::of(class).is_none() {
            if self.others.len() >= MAX_OTHER_FILES
                && let Some(oldest) = self.others.pop_front()
            {
                self.by_class.remove(&oldest);
            }
            self.others.push_back(class);
        }
        let channel = self.channel;
        self.by_class
            .entry(class)
            .or_insert_with(|| RegFile::new(channel, class))
    }

    /// The file of `class`, if the channel has one.
    #[cfg(test)]
    pub(super) fn file(&self, class: ClassId) -> Option<&RegFile> {
        self.by_class.get(&class)
    }

    /// How many files the channel keeps.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.by_class.len()
    }
}

/// Words to one method of one class, not yet written (module docs, "Routing words").
struct Batch {
    class: ClassId,
    method: u16,
    words: Vec<u32>,
}

/// What a lane works with that the front owns.
pub(super) struct LaneCx<'a> {
    pub(super) ring: &'a mut RingSink,
    pub(super) process: &'a mut ProcessMacros,
    pub(super) memory: &'a dyn GpuMemory,
    pub(super) syncpoints: &'a Syncpoints,
    pub(super) waiters: &'a WaiterTable,
    pub(super) counters: &'a mut FrontCounters,
}

/// What [`Lane::step`] did.
enum Step {
    /// Something changed; go on.
    Again,
    /// Nothing more to do now: the queue is empty, the channel waits, or the quantum is used.
    Stop,
}

/// One channel on the front (module docs).
pub(super) struct Lane {
    pub(super) config: ChannelConfig,
    pub(super) pusher: Channel,
    pub(super) macros: MacroChannel,
    pub(super) files: Files,
    pub(super) queue: VecDeque<Queued>,
    active: Option<Active>,
    pub(super) wait: Option<Wait>,
    /// The sequence number of the last submission taken.
    pub(super) taken: u64,
    /// The sequence number of the last submission finished.
    pub(super) finished: u64,
    /// What the queue bound counted ([`charge`]) and the channel started or dropped since the
    /// bound was last told: each entry started or dropped, and one for each submission with no
    /// entries when it is taken off the queue or dropped.
    pub(super) started: usize,
    /// Whether a structural error stopped the channel.
    pub(super) stopped: bool,
    batch: Batch,
}

impl Lane {
    /// A fresh channel `id` with `config`.
    pub(super) fn new(id: ChannelId, config: ChannelConfig) -> Self {
        Self {
            config,
            pusher: Channel::new(id),
            macros: MacroChannel::new(id),
            files: Files::new(id),
            queue: VecDeque::new(),
            active: None,
            wait: None,
            taken: 0,
            finished: 0,
            started: 0,
            stopped: false,
            batch: Batch {
                class: ClassId(0),
                method: 0,
                words: Vec::new(),
            },
        }
    }

    /// The channel's id.
    pub(super) fn id(&self) -> ChannelId {
        self.pusher.id()
    }

    /// Takes `submission` into the channel's queue.
    pub(super) fn take(&mut self, submission: Submission) {
        self.taken = self.taken.saturating_add(1);
        self.queue.push_back(Queued {
            seq: self.taken,
            submission,
        });
    }

    /// Whether the channel has work it could do now.
    pub(super) fn is_runnable(&self) -> bool {
        !self.stopped
            && self.wait.is_none()
            && (self.active.is_some()
                || !self.queue.is_empty()
                || self.pusher.state() == ChannelState::Running)
    }

    /// Whether submission `seq` is finished, or never will run: the channel stopped.
    pub(super) fn is_past(&self, seq: u64) -> bool {
        self.stopped || self.finished >= seq
    }

    /// Stops the channel for good after `error`: drops what it still holds, counting the entries
    /// it will never start against its queue bound.
    pub(super) fn stop(&mut self, waiters: &WaiterTable) {
        self.stopped = true;
        self.wait = None;
        waiters.remove(self.id());
        let unstarted: usize = self
            .queue
            .drain(..)
            .map(|queued| charge(queued.submission.entries.len()))
            .fold(0, usize::saturating_add);
        // The active submission's charge for having no entries left the bound when it was taken
        // off the queue (`Lane::step`); only its unstarted entries are left to release.
        let active = self.active.take().map_or(0, |active| active.unstarted());
        self.started = self
            .started
            .saturating_add(unstarted)
            .saturating_add(active);
    }

    /// Runs the channel for at most `quantum` words (module docs, "Running a channel"); whether
    /// anything changed.
    pub(super) fn run(&mut self, cx: &mut LaneCx<'_>, quantum: u32) -> Result<bool, ChannelError> {
        let mut budget = quantum;
        let mut progress = false;
        loop {
            if self.stopped || self.wait.is_some() {
                return Ok(progress);
            }
            match self.pusher.state() {
                ChannelState::Stopped(error) => return Err(error),
                ChannelState::Parked(acquire) => {
                    self.wait = Some(Wait::Acquire(acquire));
                    return Ok(progress);
                }
                ChannelState::Running => {
                    if budget == 0 {
                        return Ok(progress);
                    }
                    let before = self.pusher.words_read();
                    let outcome = self.run_pusher(cx, budget);
                    let used = self.pusher.words_read().saturating_sub(before);
                    cx.counters.words = cx.counters.words.saturating_add(used);
                    budget = budget.saturating_sub(u32::try_from(used).unwrap_or(u32::MAX));
                    match outcome? {
                        RunOutcome::EntryDone | RunOutcome::Idle => progress = true,
                        RunOutcome::QuantumUsed => return Ok(true),
                        RunOutcome::Parked(acquire) => {
                            self.park(acquire, cx)?;
                            progress = true;
                        }
                        RunOutcome::Held => {
                            if self.macros.is_waiting() {
                                let token = cx.ring.marker();
                                self.wait = Some(Wait::CatchUp(token));
                                cx.counters.catch_ups = cx.counters.catch_ups.saturating_add(1);
                            }
                            return Ok(true);
                        }
                    }
                }
                ChannelState::Idle => match self.step(cx, &mut budget)? {
                    Step::Again => progress = true,
                    Step::Stop => return Ok(progress),
                },
            }
        }
    }

    /// Runs the pusher with this lane's sink, then writes what the sink batched.
    fn run_pusher(&mut self, cx: &mut LaneCx<'_>, budget: u32) -> Result<RunOutcome, ChannelError> {
        let mut sink = LaneSink {
            channel: self.pusher.id(),
            ring: cx.ring,
            files: &mut self.files,
            process: cx.process,
            macros: &mut self.macros,
            memory: cx.memory,
            counters: cx.counters,
            batch: &mut self.batch,
        };
        let outcome = self.pusher.run(cx.memory, &mut sink, budget);
        sink.flush();
        outcome
    }

    /// One step of the submission at the head of the queue, with the pusher idle.
    fn step(&mut self, cx: &mut LaneCx<'_>, budget: &mut u32) -> Result<Step, ChannelError> {
        if self.active.is_none() {
            let Some(queued) = self.queue.pop_front() else {
                return Ok(Step::Stop);
            };
            // A submission with no entries was charged one (`charge`), which no entry start
            // will release: it leaves the bound now.
            let entries = queued.submission.entries.len();
            self.started = self
                .started
                .saturating_add(charge(entries).saturating_sub(entries));
            self.active = Some(Active::new(queued));
        }
        let channel = self.pusher.id();
        let Some(active) = self.active.as_mut() else {
            return Ok(Step::Stop);
        };
        if let Some(fence) = active.fence {
            if cx.syncpoints.reached(fence) {
                active.fence = None;
            } else {
                self.wait = Some(Wait::Fence(fence));
                cx.counters.fence_waits = cx.counters.fence_waits.saturating_add(1);
                return Ok(Step::Stop);
            }
        }
        if let Some(&raw) = active.entries.get(active.next) {
            if *budget == 0 {
                // An entry is started only with words left to read in it, so a quantum of
                // one-word entries starts exactly a quantum of them.
                return Ok(Step::Stop);
            }
            let entry = GpEntry::from_raw(raw.0);
            if entry.sync == Sync::Wait
                && active.synced != Some(active.next)
                && cx.ring.has_unperformed()
            {
                let token = cx.ring.marker();
                active.synced = Some(active.next);
                self.wait = Some(Wait::Sync(token));
                cx.counters.sync_waits = cx.counters.sync_waits.saturating_add(1);
                return Ok(Step::Stop);
            }
            active.next += 1;
            self.started = self.started.saturating_add(1);
            return match self.pusher.start_entry(entry) {
                Ok(()) => Ok(Step::Again),
                Err(StateError::Stopped(error)) => Err(error),
                // The pusher is idle here, so neither can happen.
                Err(error @ (StateError::Running | StateError::Parked)) => {
                    tracing::error!(%error, "the GPU front started an entry on a busy channel");
                    Ok(Step::Stop)
                }
            };
        }
        if !active.ended {
            active.ended = true;
            let mut sink = LaneSink {
                channel,
                ring: cx.ring,
                files: &mut self.files,
                process: cx.process,
                macros: &mut self.macros,
                memory: cx.memory,
                counters: cx.counters,
                batch: &mut self.batch,
            };
            if let Err(error) = self.pusher.end_submission(&mut sink) {
                tracing::error!(%error, "the GPU front ended a submission on a busy channel");
            }
        }
        if active.wfi {
            active.wfi = false;
            cx.ring.send(RingEntry::WaitForIdle { channel });
        }
        while active.increments > 0 {
            if *budget == 0 {
                return Ok(Step::Stop);
            }
            cx.ring.send(RingEntry::Syncpoint {
                id: self.config.syncpoint,
            });
            active.increments -= 1;
            *budget -= 1;
            cx.counters.increments = cx.counters.increments.saturating_add(1);
        }
        self.finished = active.seq;
        self.active = None;
        cx.counters.submissions = cx.counters.submissions.saturating_add(1);
        Ok(Step::Again)
    }

    /// The pusher parked on `acquire`: registers the waiter, then reads the semaphore again
    /// (module docs, "Running a channel").
    fn park(&mut self, acquire: Acquire, cx: &mut LaneCx<'_>) -> Result<(), ChannelError> {
        let channel = self.pusher.id();
        cx.waiters.register(Waiter {
            channel,
            space: self.config.space,
            address: acquire.address,
            op: acquire.op,
            payload: acquire.payload,
        });
        if self.pusher.recheck(cx.memory)? {
            cx.waiters.remove(channel);
            return Ok(());
        }
        if cx.waiters.is_latched(channel) {
            self.pusher.resume_latched();
            cx.waiters.remove(channel);
            return Ok(());
        }
        self.wait = Some(Wait::Acquire(acquire));
        cx.counters.parks = cx.counters.parks.saturating_add(1);
        Ok(())
    }

    /// Looks again at what the channel waits on (a recheck: a back report, a submission, the
    /// poll); whether it can go on now.
    pub(super) fn recheck(&mut self, cx: &mut LaneCx<'_>) -> Result<bool, ChannelError> {
        let channel = self.pusher.id();
        match self.wait {
            Some(Wait::Acquire(_)) => {
                if cx.waiters.is_latched(channel) {
                    self.pusher.resume_latched();
                } else if !self.pusher.recheck(cx.memory)? {
                    return Ok(false);
                }
                cx.waiters.remove(channel);
                self.wait = None;
                Ok(true)
            }
            Some(Wait::Fence(fence)) => {
                if !cx.syncpoints.reached(fence) {
                    return Ok(false);
                }
                if let Some(active) = self.active.as_mut() {
                    active.fence = None;
                }
                self.wait = None;
                Ok(true)
            }
            Some(Wait::Sync(_) | Wait::CatchUp(_)) | None => Ok(false),
        }
    }

    /// The back reported markers up to `ring`'s newest: ends a `SYNC_WAIT` or runs a waiting
    /// macro call again; whether the channel can go on now.
    pub(super) fn reported(&mut self, cx: &mut LaneCx<'_>) -> bool {
        match self.wait {
            Some(Wait::Sync(token)) if cx.ring.is_reported(token) => {
                self.wait = None;
                true
            }
            Some(Wait::CatchUp(token)) if cx.ring.is_reported(token) => {
                self.wait = None;
                self.resume_call(cx);
                true
            }
            _ => false,
        }
    }

    /// Runs the call waiting for a catch-up again.
    fn resume_call(&mut self, cx: &mut LaneCx<'_>) {
        let Some(engine) = self.macros.waiting_engine() else {
            return;
        };
        let file = self.files.get(engine.class());
        let env = MacroEnv {
            memory: cx.memory,
            back: &Unperformed(cx.ring.has_unperformed()),
        };
        match cx.process.resume(&mut self.macros, file, &env, cx.ring) {
            Ok(report) => note_call(report.result, cx.counters),
            Err(error) => tracing::error!(%error, "the GPU front could not resume a macro call"),
        }
        if self.macros.is_waiting() {
            let token = cx.ring.marker();
            self.wait = Some(Wait::CatchUp(token));
            cx.counters.catch_ups = cx.counters.catch_ups.saturating_add(1);
        }
    }
}

/// Counts and logs what became of a macro call.
fn note_call(result: CallResult, counters: &mut FrontCounters) {
    if let CallResult::Failed {
        error,
        first_for_macro,
    } = result
    {
        counters.macro_failures = counters.macro_failures.saturating_add(1);
        if first_for_macro {
            tracing::warn!(%error, "a macro call failed and committed nothing");
        }
    }
}

/// The pusher's sink (module docs, "Routing words").
struct LaneSink<'a> {
    channel: ChannelId,
    ring: &'a mut RingSink,
    files: &'a mut Files,
    process: &'a mut ProcessMacros,
    macros: &'a mut MacroChannel,
    memory: &'a dyn GpuMemory,
    counters: &'a mut FrontCounters,
    batch: &'a mut Batch,
}

impl LaneSink<'_> {
    /// Writes the batched words, if any.
    fn flush(&mut self) {
        if self.batch.words.is_empty() {
            return;
        }
        let file = self.files.get(self.batch.class);
        if let Err(error) = file.write_run(
            u32::from(self.batch.method),
            &self.batch.words,
            &mut *self.ring,
        ) {
            tracing::error!(%error, "a register file refused a run of words");
        }
        self.batch.words.clear();
    }
}

impl ChannelSink for LaneSink<'_> {
    fn entry(&mut self, entry: RingEntry<'static>) {
        self.flush();
        self.ring.send(entry);
    }

    fn method(&mut self, word: EngineWord) {
        if takes(word.class, u32::from(word.method)) {
            self.flush();
            let back = Unperformed(self.ring.has_unperformed());
            let env = MacroEnv {
                memory: self.memory,
                back: &back,
            };
            let file = self.files.get(word.class);
            match self
                .process
                .write(self.macros, file, word, &env, &mut *self.ring)
            {
                Ok(WordOutcome::Call(report)) => note_call(report.result, self.counters),
                Ok(WordOutcome::Register | WordOutcome::Parameter) => {}
                Err(error) => tracing::error!(%error, "the macro host refused a word"),
            }
            return;
        }
        let batched = !self.batch.words.is_empty()
            && self.batch.class == word.class
            && self.batch.method == word.method
            && self.batch.words.len() < MAX_RUN_WORDS;
        if !batched {
            self.flush();
            self.batch.class = word.class;
            self.batch.method = word.method;
        }
        self.batch.words.push(word.value);
    }

    fn set_reference(&mut self, value: u32) {
        self.flush();
        self.ring.send(RingEntry::SetReference {
            channel: self.channel,
            value,
        });
    }

    fn note(&mut self, note: Note) {
        match note {
            Note::Dropped {
                va,
                word,
                first_of_kind,
            } => {
                if first_of_kind {
                    tracing::warn!(channel = self.channel.0, va, %word, "a command word was dropped");
                }
            }
            Note::MethodCarried {
                method, remaining, ..
            } => {
                tracing::debug!(
                    channel = self.channel.0,
                    method,
                    remaining,
                    "a method's data words continue in the next submission"
                );
            }
        }
    }

    fn holds(&self) -> bool {
        self.macros.is_waiting()
    }
}

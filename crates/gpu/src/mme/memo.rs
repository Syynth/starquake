//! The macro memo: replays a repeated macro call's effects instead of interpreting it again
//! (gpu-protocol spec, "Macros" and slice G6).
//!
//! Ported from `eden-nx2/src/video_core/macro_memo.h` @ 50fa86bb (GPL-3.0-or-later, Copyright
//! 2026 Eden Emulator Project), with the qualification rules of its caller
//! (`eden-nx2/src/video_core/macro.cpp`, the native path of `MacroEngine::Execute`, and the
//! access tracking in `macro_turing.h`). Eden-NX2 reports that about 96% of a title's calls
//! repeat an earlier one exactly (row 117 of its notes says 97%), at about seven instructions
//! each, so the per-call interpreter cost is what the memo saves.
//!
//! # What qualifies
//!
//! A call is remembered when its effects depend on nothing but its start, its parameters and
//! the data RAM words it read. [`MacroMemo::call`] runs the interpreter through a recording
//! host and data RAM, and does not remember the call ([`CallOutcome::NotStored`]) when the run:
//!
//! - read a register ([`MacroHost::state`]): [`NotMemoised::RegisterRead`];
//! - ran a load barrier ([`MacroHost::barrier`]): [`NotMemoised::Barrier`];
//! - asked the host for a word after its parameters were used up (words a DMA read appended):
//!   [`NotMemoised::LoadPastParameters`];
//! - read a data RAM word it had itself written earlier in the call (Eden-NX2's
//!   `reread_data_ram`): [`NotMemoised::DataRamReread`];
//! - would make a record longer than the memo's arena: [`NotMemoised::TooLarge`].
//!
//! When several apply, the first one the run reached is reported; [`NotMemoised::TooLarge`] is
//! checked last, once the run has finished. A run that fails ([`MacroError`]) is not remembered
//! either, and its error is returned as the interpreter returns it.
//!
//! # The record
//!
//! A remembered call is keyed by its start and its parameters; its guard is the data RAM words it
//! read, as `(index, value)` pairs in read order; its effects are its data RAM writes and its
//! method writes, each in order. A call whose key matches and whose guard still holds replays the
//! data RAM writes, then the method writes through [`MacroHost::method`]. Data RAM is invisible to
//! the host while a run executes, so applying the data RAM writes first gives the same final state
//! as the interpreter's interleaving. Records live in one arena of words (Eden-NX2's layout:
//! start, parameter count, read count, data RAM write count, method write count, then the
//! parameters and the three lists of pairs), found through a map from a hash of the key and
//! through a direct-mapped slot per start for the common case of a call repeating the previous
//! call of the same macro.
//!
//! The memo assumes that a start names the same code until [`MacroMemo::synchronize`] is given a
//! new code generation: the caller bumps the generation whenever macro code is uploaded.
//!
//! # Divergences from Eden-NX2
//!
//! Each is pinned by a test in `memo/tests.rs`:
//!
//! - **Any load barrier disqualifies a call.** Eden-NX2 disqualifies a call that wrote one of the
//!   MME DMA read methods (3D 0x157, `MME_DMA_READ`, and 0x158, `MME_DMA_READ_FIFOED`), so a
//!   barrier without a DMA read is still remembered there. This memo sits in front of the
//!   interpreter, not the 3D engine, and cannot tell what a host did at a barrier
//!   (`a_barrier_disqualifies_a_call_even_without_a_dma_read`).
//! - **A record longer than the arena is not stored.** Eden-NX2 appends it anyway and lets the
//!   arena grow past its limit (`a_record_longer_than_the_arena_is_not_stored`).
//! - **Compaction keeps records in arena order.** Eden-NX2 copies them in its map's iteration
//!   order; here the copy is sorted by old offset, so the arena's layout never depends on hashing.
//!   Nothing outside the memo can observe the difference.

use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Range;

use super::inst::Instruction;
use super::interp::{
    DATA_RAM_WORDS, DataRam, MacroError, MacroHost, RamAccess, StepLimit, run_with,
};

/// The most words the memo's arena holds before it compacts (Eden-NX2's `MaxArenaWords`,
/// 16 MiB). Also the most a single record may take.
pub const MAX_ARENA_WORDS: usize = 1 << 22;

/// Words of a record's header: start, parameter count, read count, data RAM write count and
/// method write count (`HeaderWords` in `macro_memo.h`).
const HEADER_WORDS: usize = 5;

/// Direct-mapped slots of "the last record of this start" (`last_by_start` in `macro_memo.h`):
/// one per value of a start's low byte.
const SLOTS: usize = 256;

/// The slot that holds no record (`NoRecord`).
const NO_RECORD: u32 = u32::MAX;

/// Why a call that ran successfully was not remembered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NotMemoised {
    /// It read a register, which the guard does not cover.
    RegisterRead,
    /// It ran a load barrier, where a host may perform MME DMA reads.
    Barrier,
    /// It loaded a word beyond its parameters.
    LoadPastParameters,
    /// It read a data RAM word it had written itself.
    DataRamReread,
    /// Its record would be longer than the arena.
    TooLarge,
}

/// What [`MacroMemo::call`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallOutcome {
    /// An earlier call's effects were replayed; the interpreter did not run.
    Replayed,
    /// The interpreter ran and the call was remembered.
    Stored,
    /// The interpreter ran and the call was not remembered.
    NotStored(NotMemoised),
}

/// The memo's counters (`Statistics` in `macro_memo.h`, plus [`Self::oversized`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MemoStatistics {
    /// Calls replayed.
    pub hits: u64,
    /// Of [`Self::hits`], those found through the start's slot without hashing.
    pub slot_hits: u64,
    /// Calls with no record under their key's hash.
    pub misses: u64,
    /// Calls whose hash had a record that was another call or whose guard no longer held.
    pub guard_failures: u64,
    /// Records written, in place or appended.
    pub stores: u64,
    /// Of [`Self::stores`], those that rewrote a record of the same shape in place.
    pub overwrites: u64,
    /// Times the arena was compacted.
    pub compactions: u64,
    /// Times a compaction emptied the memo because live records alone filled it.
    pub clears: u64,
    /// Calls not stored because their record was longer than the arena.
    pub oversized: u64,
}

/// A hasher for keys that are already well-mixed hashes: it hands the key back as it is
/// (`MacroMemoHash` in `macro_memo.h`). It is also deterministic, unlike the standard library's
/// default.
#[derive(Clone, Copy, Debug, Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // Only `write_u64` is used for the memo's keys; fold anything else in.
        for byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*byte);
        }
    }

    fn write_u64(&mut self, key: u64) {
        self.0 = key;
    }
}

/// The hash of a call's key (`Hash` in `macro_memo.h`).
fn key_hash(start: u32, parameters: &[u32]) -> u64 {
    let count = u64::try_from(parameters.len()).unwrap_or(u64::MAX);
    let mut hash = 0x9e37_79b9_7f4a_7c15_u64 ^ (u64::from(start) << 32) ^ count;
    for word in parameters {
        hash = (hash ^ u64::from(*word)).wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^= hash >> 32;
    }
    hash
}

/// The slot of `start`: its low byte, as `start % 256` in `macro_memo.h`.
fn slot_of(start: u32) -> usize {
    usize::from(start.to_le_bytes()[0])
}

/// A method index stored as a record word (always written from a `u16`).
fn method_of(word: u32) -> u16 {
    let [low, high, ..] = word.to_le_bytes();
    u16::from_le_bytes([low, high])
}

/// Notes the first reason a call cannot be remembered.
fn note(reason: &Cell<Option<NotMemoised>>, why: NotMemoised) {
    if reason.get().is_none() {
        reason.set(Some(why));
    }
}

/// Where a matched record's effects are in the arena.
struct Effects {
    data_ram_writes: Range<usize>,
    writes: Range<usize>,
}

/// A record's header, read from the arena.
#[derive(Clone, Copy)]
struct Header {
    start: u32,
    parameters: usize,
    reads: usize,
    data_ram_writes: usize,
    writes: usize,
}

impl Header {
    /// The record's length in words, or `None` if it overflows.
    fn words(self) -> Option<usize> {
        let pairs = self
            .reads
            .checked_add(self.data_ram_writes)?
            .checked_add(self.writes)?;
        HEADER_WORDS
            .checked_add(self.parameters)?
            .checked_add(pairs.checked_mul(2)?)
    }
}

/// The recording host a call is interpreted through.
struct Recorder<'a, H> {
    host: &'a mut H,
    parameters: &'a [u32],
    next: usize,
    writes: &'a mut Vec<(u32, u32)>,
    reason: &'a Cell<Option<NotMemoised>>,
}

impl<H: MacroHost> MacroHost for Recorder<'_, H> {
    fn load(&mut self) -> Option<u32> {
        if let Some(word) = self.parameters.get(self.next) {
            self.next = self.next.saturating_add(1);
            return Some(*word);
        }
        let word = self.host.load();
        if word.is_some() {
            note(self.reason, NotMemoised::LoadPastParameters);
        }
        word
    }

    fn state(&mut self, method: u16) -> u32 {
        note(self.reason, NotMemoised::RegisterRead);
        self.host.state(method)
    }

    fn method(&mut self, method: u16, value: u32) {
        self.writes.push((u32::from(method), value));
        self.host.method(method, value);
    }

    fn barrier(&mut self) {
        note(self.reason, NotMemoised::Barrier);
        self.host.barrier();
    }
}

/// The recording data RAM a call is interpreted against.
struct Tracked<'a> {
    ram: &'a mut DataRam,
    reads: &'a mut Vec<(u32, u32)>,
    writes: &'a mut Vec<(u32, u32)>,
    /// Bit `i` is set once the call has written word `i`.
    written: &'a mut [u64],
    reason: &'a Cell<Option<NotMemoised>>,
}

impl Tracked<'_> {
    /// The word and bit of `index` in `written`.
    fn bit(index: u32) -> Option<(usize, u64)> {
        let index = usize::try_from(index).ok()?;
        Some((index / 64, 1 << (index % 64)))
    }
}

impl RamAccess for Tracked<'_> {
    fn read(&mut self, index: u32) -> Option<u32> {
        let value = self.ram.get(index)?;
        let (word, bit) = Self::bit(index)?;
        if self.written.get(word).is_some_and(|bits| bits & bit != 0) {
            note(self.reason, NotMemoised::DataRamReread);
        }
        self.reads.push((index, value));
        Some(value)
    }

    fn write(&mut self, index: u32, value: u32) -> Option<()> {
        *self.ram.get_mut(index)? = value;
        let (word, bit) = Self::bit(index)?;
        if let Some(bits) = self.written.get_mut(word) {
            *bits |= bit;
        }
        self.writes.push((index, value));
        Some(())
    }
}

/// The memo of macro calls, in front of the interpreter ([`super::run`]).
///
/// One memo serves one macro code store: its records are only valid for the code generation of
/// the last [`MacroMemo::synchronize`].
#[derive(Clone, Debug)]
pub struct MacroMemo {
    entries: HashMap<u64, u32, BuildHasherDefault<KeyHasher>>,
    arena: Vec<u32>,
    /// The compaction's new arena, kept for its capacity.
    compacted: Vec<u32>,
    /// Per slot: the start and record offset last found or stored.
    last_by_start: Box<[(u32, u32)]>,
    generation: u64,
    max_arena_words: usize,
    limit: StepLimit,
    statistics: MemoStatistics,
    // Per-call scratch, kept for its capacity.
    reads: Vec<(u32, u32)>,
    data_ram_writes: Vec<(u32, u32)>,
    writes: Vec<(u32, u32)>,
    written: Box<[u64]>,
}

impl Default for MacroMemo {
    fn default() -> Self {
        Self::new()
    }
}

impl MacroMemo {
    /// An empty memo with an arena of [`MAX_ARENA_WORDS`] that interprets with
    /// [`StepLimit::DEFAULT`], at code generation 0.
    pub fn new() -> Self {
        Self::with_limits(MAX_ARENA_WORDS, StepLimit::DEFAULT)
    }

    /// An empty memo whose arena compacts at `max_arena_words` (at most [`MAX_ARENA_WORDS`]; a
    /// larger value is reduced to it) and that interprets with `limit`.
    pub fn with_limits(max_arena_words: usize, limit: StepLimit) -> Self {
        Self {
            entries: HashMap::default(),
            arena: Vec::new(),
            compacted: Vec::new(),
            last_by_start: vec![(0, NO_RECORD); SLOTS].into_boxed_slice(),
            generation: 0,
            max_arena_words: max_arena_words.min(MAX_ARENA_WORDS),
            limit,
            statistics: MemoStatistics::default(),
            reads: Vec::new(),
            data_ram_writes: Vec::new(),
            writes: Vec::new(),
            written: vec![0; DATA_RAM_WORDS.div_ceil(64)].into_boxed_slice(),
        }
    }

    /// Forgets every call if `code_generation` is not the one the memo last saw (`Synchronize`).
    pub fn synchronize(&mut self, code_generation: u64) {
        if code_generation != self.generation {
            self.generation = code_generation;
            self.clear();
        }
    }

    /// Runs the macro at `start`, whose code is `program`, with `parameters`: replays an earlier
    /// call's effects when the key matches and its guard holds, and otherwise interprets the
    /// call and remembers it if it qualifies (see the module docs).
    ///
    /// The memo serves `parameters` to the run itself; `host`'s [`MacroHost::load`] is asked
    /// only once they are used up, for words a DMA read appended. Method writes reach `host` as
    /// the interpreter makes them, or, on a replay, in their recorded order.
    ///
    /// # Errors
    ///
    /// The interpreter's [`MacroError`]; the call is then not remembered, and the host and data
    /// RAM keep what the run did before it failed, as with [`super::run`].
    pub fn call(
        &mut self,
        program: &[Instruction],
        start: u32,
        parameters: &[u32],
        data_ram: &mut DataRam,
        host: &mut impl MacroHost,
    ) -> Result<CallOutcome, MacroError> {
        if let Some(effects) = self.find(start, parameters, data_ram) {
            self.replay(&effects, data_ram, host);
            return Ok(CallOutcome::Replayed);
        }

        self.reads.clear();
        self.data_ram_writes.clear();
        self.writes.clear();
        self.written.fill(0);
        let reason = Cell::new(None);
        let mut recorder = Recorder {
            host,
            parameters,
            next: 0,
            writes: &mut self.writes,
            reason: &reason,
        };
        let mut ram = Tracked {
            ram: data_ram,
            reads: &mut self.reads,
            writes: &mut self.data_ram_writes,
            written: &mut self.written,
            reason: &reason,
        };
        run_with(program, &mut ram, &mut recorder, self.limit)?;

        if let Some(why) = reason.get() {
            return Ok(CallOutcome::NotStored(why));
        }
        if self.store(start, parameters) {
            Ok(CallOutcome::Stored)
        } else {
            Ok(CallOutcome::NotStored(NotMemoised::TooLarge))
        }
    }

    /// The memo's counters.
    pub fn statistics(&self) -> MemoStatistics {
        self.statistics
    }

    /// The number of calls remembered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no call is remembered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The words the arena holds, stale records included.
    pub fn arena_words(&self) -> usize {
        self.arena.len()
    }

    /// The record of an earlier call with this key whose guard holds (`Find`).
    fn find(&mut self, start: u32, parameters: &[u32], data_ram: &DataRam) -> Option<Effects> {
        let slot = slot_of(start);
        let (last_start, last_offset) = self.last_by_start.get(slot).copied()?;
        if last_start == start
            && last_offset != NO_RECORD
            && let Some(effects) = self.matches(last_offset, start, parameters, data_ram)
        {
            self.statistics.hits += 1;
            self.statistics.slot_hits += 1;
            return Some(effects);
        }
        let Some(offset) = self.entries.get(&key_hash(start, parameters)).copied() else {
            self.statistics.misses += 1;
            return None;
        };
        let Some(effects) = self.matches(offset, start, parameters, data_ram) else {
            self.statistics.guard_failures += 1;
            return None;
        };
        if let Some(last) = self.last_by_start.get_mut(slot) {
            *last = (start, offset);
        }
        self.statistics.hits += 1;
        Some(effects)
    }

    /// The header of the record at `offset` in `arena`.
    fn header(arena: &[u32], offset: usize) -> Option<Header> {
        let words = arena.get(offset..offset.checked_add(HEADER_WORDS)?)?;
        let &[start, parameters, reads, data_ram_writes, writes] = words else {
            return None;
        };
        Some(Header {
            start,
            parameters: usize::try_from(parameters).ok()?,
            reads: usize::try_from(reads).ok()?,
            data_ram_writes: usize::try_from(data_ram_writes).ok()?,
            writes: usize::try_from(writes).ok()?,
        })
    }

    /// Whether the record at `offset` is this call with its guard holding (`Match`).
    fn matches(
        &self,
        offset: u32,
        start: u32,
        parameters: &[u32],
        data_ram: &DataRam,
    ) -> Option<Effects> {
        let offset = usize::try_from(offset).ok()?;
        let header = Self::header(&self.arena, offset)?;
        if header.start != start || header.parameters != parameters.len() {
            return None;
        }
        let params_at = offset.checked_add(HEADER_WORDS)?;
        let reads_at = params_at.checked_add(header.parameters)?;
        if self.arena.get(params_at..reads_at)? != parameters {
            return None;
        }
        let data_ram_writes_at = reads_at.checked_add(header.reads.checked_mul(2)?)?;
        for &[index, value] in self.arena.get(reads_at..data_ram_writes_at)?.as_chunks().0 {
            if data_ram.get(index) != Some(value) {
                return None;
            }
        }
        let writes_at = data_ram_writes_at.checked_add(header.data_ram_writes.checked_mul(2)?)?;
        let end = writes_at.checked_add(header.writes.checked_mul(2)?)?;
        if end > self.arena.len() {
            return None;
        }
        Some(Effects {
            data_ram_writes: data_ram_writes_at..writes_at,
            writes: writes_at..end,
        })
    }

    /// Applies a matched record's effects.
    fn replay(&self, effects: &Effects, data_ram: &mut DataRam, host: &mut impl MacroHost) {
        let arena = &self.arena;
        let data_ram_writes = arena
            .get(effects.data_ram_writes.clone())
            .unwrap_or_default();
        for &[index, value] in data_ram_writes.as_chunks().0 {
            // Every index was recorded from a write that succeeded, so it is in range.
            if let Some(word) = data_ram.get_mut(index) {
                *word = value;
            }
        }
        let writes = arena.get(effects.writes.clone()).unwrap_or_default();
        for &[method, value] in writes.as_chunks().0 {
            host.method(method_of(method), value);
        }
    }

    /// Remembers the call just interpreted (`Store`); `false` if its record is too long.
    fn store(&mut self, start: u32, parameters: &[u32]) -> bool {
        let header = Header {
            start,
            parameters: parameters.len(),
            reads: self.reads.len(),
            data_ram_writes: self.data_ram_writes.len(),
            writes: self.writes.len(),
        };
        let counts = (
            u32::try_from(header.parameters),
            u32::try_from(header.reads),
            u32::try_from(header.data_ram_writes),
            u32::try_from(header.writes),
        );
        let (Some(words), (Ok(parameter_count), Ok(reads), Ok(data_ram_writes), Ok(writes))) =
            (header.words(), counts)
        else {
            self.statistics.oversized += 1;
            return false;
        };
        if words > self.max_arena_words {
            self.statistics.oversized += 1;
            return false;
        }
        self.statistics.stores += 1;

        let hash = key_hash(start, parameters);
        // A call stored again (its guard no longer held) usually has the same shape: rewrite its
        // record in place, or the arena fills with stale copies.
        if let Some(offset) = self.entries.get(&hash).copied()
            && let Ok(offset) = usize::try_from(offset)
            && let Some(old) = Self::header(&self.arena, offset)
            && old.start == start
            && old.parameters == header.parameters
            && old.reads == header.reads
            && old.data_ram_writes == header.data_ram_writes
            && old.writes == header.writes
            && let Some(params_at) = offset.checked_add(HEADER_WORDS)
            && let Some(pairs_at) = params_at.checked_add(header.parameters)
            && self.arena.get(params_at..pairs_at) == Some(parameters)
            && let Some(end) = offset.checked_add(words)
            && let Some(out) = self.arena.get_mut(pairs_at..end)
        {
            let pairs = self
                .reads
                .iter()
                .chain(&self.data_ram_writes)
                .chain(&self.writes)
                .flat_map(|&(address, value)| [address, value]);
            for (slot, word) in out.iter_mut().zip(pairs) {
                *slot = word;
            }
            self.statistics.overwrites += 1;
            return true;
        }

        if self.arena.len() >= self.max_arena_words {
            self.compact();
        }
        // The arena stays below twice `MAX_ARENA_WORDS`, so its offsets fit in a `u32`.
        let Ok(offset) = u32::try_from(self.arena.len()) else {
            self.statistics.oversized += 1;
            return false;
        };
        self.arena
            .extend([start, parameter_count, reads, data_ram_writes, writes]);
        self.arena.extend_from_slice(parameters);
        let pairs = self
            .reads
            .iter()
            .chain(&self.data_ram_writes)
            .chain(&self.writes)
            .flat_map(|&(address, value)| [address, value]);
        self.arena.extend(pairs);
        self.entries.insert(hash, offset);
        if let Some(last) = self.last_by_start.get_mut(slot_of(start)) {
            *last = (start, offset);
        }
        true
    }

    /// Forgets every call (`Clear`).
    fn clear(&mut self) {
        self.arena.clear();
        self.entries.clear();
        self.last_by_start.fill((0, NO_RECORD));
    }

    /// Copies the records still in the map to a new arena, dropping stale ones; empties the memo
    /// if the live records alone fill three quarters of it (`Compact`).
    fn compact(&mut self) {
        let mut live: Vec<(u32, u64)> = self
            .entries
            .iter()
            .map(|(&hash, &offset)| (offset, hash))
            .collect();
        live.sort_unstable();
        self.compacted.clear();
        for (offset, hash) in live {
            let arena = &self.arena;
            let record = usize::try_from(offset).ok().and_then(|offset| {
                let end = offset.checked_add(Self::header(arena, offset)?.words()?)?;
                arena.get(offset..end)
            });
            let new_offset = u32::try_from(self.compacted.len()).ok();
            match (record, new_offset) {
                (Some(record), Some(new_offset)) => {
                    self.compacted.extend_from_slice(record);
                    self.entries.insert(hash, new_offset);
                }
                // Unreachable for records this memo wrote; drop rather than keep a bad offset.
                _ => {
                    self.entries.remove(&hash);
                }
            }
        }
        std::mem::swap(&mut self.arena, &mut self.compacted);
        self.last_by_start.fill((0, NO_RECORD));
        self.statistics.compactions += 1;
        if self.arena.len() >= self.max_arena_words / 4 * 3 {
            self.clear();
            self.statistics.clears += 1;
        }
    }
}

#[cfg(test)]
mod tests;

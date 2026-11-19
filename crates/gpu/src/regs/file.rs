//! One class's register file on one channel ([`RegFile`]), and where its ring entries go
//! ([`EntrySink`]).

use std::borrow::Cow;

use thiserror::Error;

use crate::classes::clc797;
use crate::interface::ChannelId;
use crate::ring::{ClassId, MAX_RUN_WORDS, RingEntry};

use super::rules::{ClassRules, MethodSpan, SideEffect};

/// Where a register file sends what survives it: the ring, in production (the front loop, G14,
/// adapts its producer), or a list in tests.
pub trait EntrySink {
    /// Takes one entry, in order.
    fn send(&mut self, entry: RingEntry<'_>);
}

impl EntrySink for Vec<RingEntry<'static>> {
    fn send(&mut self, entry: RingEntry<'_>) {
        self.push(entry.into_owned());
    }
}

/// A write the register file refuses.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum RegsError {
    /// The method is not in the class's register file (for the 3D and compute classes, it is a
    /// macro method, which the macro host takes before the register file).
    #[error(
        "method {method:#x} is outside this class's register file, which holds methods below {limit:#x}"
    )]
    OutOfRange {
        /// The method.
        method: u32,
        /// The number of registers in the file.
        limit: u32,
    },
}

/// The shadow RAM's mode: the `MODE` field (1:0) of the shadow file's copy of
/// `SET_MME_SHADOW_RAM_CONTROL`, with the header's four values (`classes/clc797.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShadowMode {
    /// `METHOD_TRACK`: each write is also stored in the shadow file.
    Track,
    /// `METHOD_TRACK_WITH_FILTER`: as [`ShadowMode::Track`]. **Guess**, after Eden-NX2's
    /// `ProcessShadowRam`, which treats both alike; what the filter is would take a trace of a
    /// title that uses the mode.
    TrackWithFilter,
    /// `METHOD_PASSTHROUGH`: the shadow file is left alone.
    Passthrough,
    /// `METHOD_REPLAY`: the value written is the shadow file's, not the word sent.
    Replay,
}

impl ShadowMode {
    /// The mode a shadow-control word selects. Total: the 2-bit field has a named value for
    /// each of its four values.
    pub fn from_word(word: u32) -> Self {
        match clc797::SET_MME_SHADOW_RAM_CONTROL_MODE.get(word) {
            clc797::SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_TRACK => Self::Track,
            clc797::SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_TRACK_WITH_FILTER => {
                Self::TrackWithFilter
            }
            clc797::SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_PASSTHROUGH => Self::Passthrough,
            _ => Self::Replay,
        }
    }
}

/// The front thread's copy of one class's registers on one channel: the only authoritative copy
/// (owner's decision 1). Each data word written to it goes through the shadow RAM, the class's
/// side-effect registers and same-value dropping, and what survives is sent to an
/// [`EntrySink`] (the module docs of `regs` give the order and the rules).
#[derive(Clone, Debug)]
pub struct RegFile {
    channel: ChannelId,
    class: ClassId,
    rules: &'static ClassRules,
    values: Box<[u32]>,
    /// The shadow file: as long as `values` for a class with shadow RAM, empty otherwise.
    shadow: Box<[u32]>,
    /// One bit per register: set for a plain-state method.
    state: Box<[u64]>,
    drop_same_values: bool,
    /// The resolved words of the run being sent, at most [`MAX_RUN_WORDS`].
    run: Vec<u32>,
    /// The side-effect registers of the run being sent, with their values before it.
    before: Vec<(usize, u32)>,
}

/// The bits in a word of [`RegFile::state`].
const STATE_WORD_BITS: usize = u64::BITS as usize;

/// The value one register or shadow word held before the first write that could change it
/// since the [`Journal`] was last emptied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Undo {
    /// A shadow file word, not a register.
    shadow: bool,
    index: usize,
    old: u32,
}

/// What a sequence of writes to one [`RegFile`] changed, so that it can be undone: the macro
/// host (slice G13, `crate::macros`) applies a macro run's writes as it goes, so the run's own
/// `state` reads see them, and undoes them if the run commits nothing.
///
/// It keeps the first value of every word [`RegFile::save`] names, once per word, so it holds at
/// most twice the file's registers (a register and its shadow word), however many writes it
/// covers.
#[derive(Clone, Debug, Default)]
pub(crate) struct Journal {
    entries: Vec<Undo>,
    /// One bit per word saved: register `i` is bit `2i`, its shadow word bit `2i + 1`.
    saved: Vec<u64>,
}

impl Journal {
    /// Forgets every saved word: the writes it covered stand.
    pub(crate) fn clear(&mut self) {
        for undo in &self.entries {
            if let Some((word, bit)) = Self::bit(undo.shadow, undo.index)
                && let Some(bits) = self.saved.get_mut(word)
            {
                *bits &= !bit;
            }
        }
        self.entries.clear();
    }

    /// The word and bit of a register or shadow word in `saved`.
    fn bit(shadow: bool, index: usize) -> Option<(usize, u64)> {
        let position = index.checked_mul(2)?.checked_add(usize::from(shadow))?;
        Some((
            position / STATE_WORD_BITS,
            1 << (position % STATE_WORD_BITS),
        ))
    }

    /// Saves `old` for the word unless it was saved already.
    fn save(&mut self, shadow: bool, index: usize, old: u32) {
        let Some((word, bit)) = Self::bit(shadow, index) else {
            return;
        };
        if self.saved.len() <= word {
            self.saved.resize(word + 1, 0);
        }
        if let Some(bits) = self.saved.get_mut(word)
            && *bits & bit == 0
        {
            *bits |= bit;
            self.entries.push(Undo { shadow, index, old });
        }
    }
}

impl RegFile {
    /// A file of zeroed registers (and a zeroed shadow file, so the shadow RAM starts in
    /// [`ShadowMode::Track`], as Eden's zero-initialised `shadow_state` does) for class `class`
    /// on channel `channel`, with the rules [`ClassRules::for_class`] picks and same-value
    /// dropping on.
    pub fn new(channel: ChannelId, class: ClassId) -> Self {
        let rules = ClassRules::for_class(class);
        let registers = usize::try_from(rules.registers()).unwrap_or(0);
        let shadow = if rules.shadow_control().is_some() {
            registers
        } else {
            0
        };
        let mut state = vec![0u64; registers.div_ceil(STATE_WORD_BITS)].into_boxed_slice();
        for number in rules.state().iter().flat_map(MethodSpan::numbers) {
            let Ok(index) = usize::try_from(number) else {
                continue;
            };
            if let Some(word) = state.get_mut(index / STATE_WORD_BITS) {
                *word |= 1 << (index % STATE_WORD_BITS);
            }
        }
        Self {
            channel,
            class,
            rules,
            values: vec![0; registers].into_boxed_slice(),
            shadow: vec![0; shadow].into_boxed_slice(),
            state,
            drop_same_values: true,
            run: Vec::new(),
            before: Vec::new(),
        }
    }

    /// The channel.
    pub fn channel(&self) -> ChannelId {
        self.channel
    }

    /// The class.
    pub fn class(&self) -> ClassId {
        self.class
    }

    /// The class's rules.
    pub fn rules(&self) -> &'static ClassRules {
        self.rules
    }

    /// The register `method`'s value; `None` outside the file.
    pub fn get(&self, method: u32) -> Option<u32> {
        self.values.get(usize::try_from(method).ok()?).copied()
    }

    /// The shadow file's copy of `method`; `None` outside the file or for a class without
    /// shadow RAM.
    pub fn shadow(&self, method: u32) -> Option<u32> {
        self.shadow.get(usize::try_from(method).ok()?).copied()
    }

    /// The shadow RAM's mode, for a class with shadow RAM.
    pub fn shadow_mode(&self) -> Option<ShadowMode> {
        let control = self.rules.shadow_control()?;
        self.shadow(control).map(ShadowMode::from_word)
    }

    /// Whether `method` is a plain-state method (its same-value writes may be dropped); `false`
    /// for a trigger and outside the file.
    pub fn is_state(&self, method: u32) -> bool {
        usize::try_from(method).is_ok_and(|index| self.is_state_index(index))
    }

    /// Whether same-value writes to state methods are dropped (on for a new file).
    pub fn drop_same_values(&self) -> bool {
        self.drop_same_values
    }

    /// Turns same-value dropping on or off (off sends every write, for hunting a bug; owner's
    /// decision 12).
    pub fn set_drop_same_values(&mut self, on: bool) {
        self.drop_same_values = on;
    }

    /// Writes one data word to `method`: the same as [`RegFile::write_run`] with one word.
    pub fn write(
        &mut self,
        method: u32,
        word: u32,
        out: &mut (impl EntrySink + ?Sized),
    ) -> Result<(), RegsError> {
        self.write_run(method, &[word], out)
    }

    /// Writes data words to one method, in order (a non-incrementing run, or one word). The
    /// register file and the shadow file end as if each word had been written alone. For a
    /// state method, each word is one `Write` entry or dropped; for a trigger, the words go out
    /// as one `Run` entry (a `Write` for a single word, and a new entry every [`MAX_RUN_WORDS`]
    /// words), each followed by a `Write` of every side-effect register it changed. No words
    /// sends nothing.
    ///
    /// # Errors
    ///
    /// [`RegsError::OutOfRange`] if `method` is not in the file; nothing is then changed or
    /// sent.
    pub fn write_run(
        &mut self,
        method: u32,
        words: &[u32],
        out: &mut (impl EntrySink + ?Sized),
    ) -> Result<(), RegsError> {
        let limit = self.rules.registers();
        let out_of_range = RegsError::OutOfRange { method, limit };
        if method >= limit {
            return Err(out_of_range);
        }
        let index = usize::try_from(method).map_err(|_| out_of_range)?;
        let number = u16::try_from(method).map_err(|_| out_of_range)?;
        if self.is_state_index(index) {
            for &word in words {
                self.write_state(index, method, number, word, out);
            }
        } else {
            for chunk in words.chunks(MAX_RUN_WORDS) {
                self.write_trigger(index, method, number, chunk, out);
            }
        }
        Ok(())
    }

    /// Saves in `journal` every word a write to `method` can change, before it is made: the
    /// register itself and its shadow word (step 1, which also covers a write to the shadow
    /// control), and the register of each side effect the method triggers (step 2). A method
    /// outside the file saves nothing, as its write changes nothing.
    pub(crate) fn save(&self, method: u32, journal: &mut Journal) {
        let Ok(index) = usize::try_from(method) else {
            return;
        };
        let Some(&value) = self.values.get(index) else {
            return;
        };
        journal.save(false, index, value);
        if let Some(&shadow) = self.shadow.get(index) {
            journal.save(true, index, shadow);
        }
        for effect in self.rules.side_effects() {
            if !effect.trigger.contains(method) {
                continue;
            }
            if let Ok(target) = usize::try_from(effect.rule.register())
                && let Some(&value) = self.values.get(target)
            {
                journal.save(false, target, value);
            }
        }
    }

    /// Restores every word `journal` saved to the value it held when first saved, and empties
    /// the journal.
    pub(crate) fn undo(&mut self, journal: &mut Journal) {
        for undo in journal.entries.iter().rev() {
            let slot = if undo.shadow {
                self.shadow.get_mut(undo.index)
            } else {
                self.values.get_mut(undo.index)
            };
            if let Some(slot) = slot {
                *slot = undo.old;
            }
        }
        journal.clear();
    }

    fn is_state_index(&self, index: usize) -> bool {
        self.state
            .get(index / STATE_WORD_BITS)
            .is_some_and(|word| word & (1 << (index % STATE_WORD_BITS)) != 0)
    }

    /// Step 1: the shadow RAM. Returns the value the register takes.
    fn shadow_step(&mut self, index: usize, method: u32, word: u32) -> u32 {
        let Some(control) = self.rules.shadow_control() else {
            return word;
        };
        let mode = self
            .shadow(control)
            .map_or(ShadowMode::Track, ShadowMode::from_word);
        let value = match mode {
            ShadowMode::Track | ShadowMode::TrackWithFilter => {
                if let Some(slot) = self.shadow.get_mut(index) {
                    *slot = word;
                }
                word
            }
            ShadowMode::Replay => self.shadow.get(index).copied().unwrap_or(word),
            ShadowMode::Passthrough => word,
        };
        // The mode is the shadow file's copy of the control register, and a write to the
        // control sets it to the word sent, in every mode (row 144; Eden-NX2's `FrontWrite`).
        if method == control
            && let Some(slot) = self.shadow.get_mut(index)
        {
            *slot = word;
        }
        value
    }

    /// One word to a state method: steps 1 and 3.
    fn write_state(
        &mut self,
        index: usize,
        method: u32,
        number: u16,
        word: u32,
        out: &mut (impl EntrySink + ?Sized),
    ) {
        let value = self.shadow_step(index, method, word);
        let Some(slot) = self.values.get_mut(index) else {
            return;
        };
        if self.drop_same_values && *slot == value {
            return;
        }
        *slot = value;
        out.send(RingEntry::Write {
            channel: self.channel,
            class: self.class,
            method: number,
            value,
        });
    }

    /// At most [`MAX_RUN_WORDS`] words to a trigger: steps 1 and 2, then the entries.
    fn write_trigger(
        &mut self,
        index: usize,
        method: u32,
        number: u16,
        words: &[u32],
        out: &mut (impl EntrySink + ?Sized),
    ) {
        let side_effects = self.rules.side_effects();
        let applies = |effect: &&SideEffect| effect.trigger.contains(method);
        self.before.clear();
        for effect in side_effects.iter().filter(applies) {
            let target = usize::try_from(effect.rule.register()).ok();
            if let Some((target, &value)) =
                target.and_then(|target| Some((target, self.values.get(target)?)))
            {
                self.before.push((target, value));
            }
        }
        self.run.clear();
        for &word in words {
            let value = self.shadow_step(index, method, word);
            if let Some(slot) = self.values.get_mut(index) {
                *slot = value;
            }
            for effect in side_effects.iter().filter(applies) {
                let target = usize::try_from(effect.rule.register()).ok();
                if let Some(slot) = target.and_then(|target| self.values.get_mut(target)) {
                    *slot = effect.rule.apply(*slot);
                }
            }
            self.run.push(value);
        }
        match self.run.as_slice() {
            // Never reached: `write_run` makes no chunk of an empty run.
            [] => {}
            &[value] => out.send(RingEntry::Write {
                channel: self.channel,
                class: self.class,
                method: number,
                value,
            }),
            words => out.send(RingEntry::Run {
                channel: self.channel,
                class: self.class,
                method: number,
                words: Cow::Borrowed(words),
            }),
        }
        for &(target, before) in &self.before {
            let Some(&value) = self.values.get(target) else {
                continue;
            };
            let Ok(target) = u16::try_from(target) else {
                continue;
            };
            if value != before {
                out.send(RingEntry::Write {
                    channel: self.channel,
                    class: self.class,
                    method: target,
                    value,
                });
            }
        }
    }
}

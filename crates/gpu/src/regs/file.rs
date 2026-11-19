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

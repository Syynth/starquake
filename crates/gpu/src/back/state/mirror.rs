//! One class's register mirror on one channel ([`Mirror`]), what goes wrong reading it
//! ([`StateError`]), the constant buffer bindings it records ([`ConstantBufferBinding`]) and the
//! positions of a run's words ([`RunPositions`]).

use thiserror::Error;

use crate::classes::{Bits, Method, Value, clc797};
use crate::field::Field;
use crate::interface::ChannelId;
use crate::regs::{ClassRules, MethodSpan, SideEffect};
use crate::ring::{ClassId, RingEntry};

use super::group::{BIND_GROUPS, CONSTANT_BUFFER_SLOTS, Group, GroupSet};
use super::table::GroupMap;

/// What reading or writing a mirror refuses.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum StateError {
    /// The method is not in the class's register file.
    #[error(
        "method {method:#x} is outside this class's register file, which holds methods below {limit:#x}"
    )]
    OutOfRange {
        /// The method number.
        method: u32,
        /// The number of registers in the file.
        limit: u32,
    },
    /// The indices given do not name an element of the method: a different number of indices
    /// than the method has, or an address past 32 bits.
    #[error("{method} has no element with {} given", indices_given(*.indices))]
    NoElement {
        /// The method's name in the class header.
        method: &'static str,
        /// The number of indices given.
        indices: usize,
    },
    /// An accessor's index is past its array.
    #[error("{array} {index} does not exist: there are {count}")]
    Index {
        /// What the array holds, in words.
        array: &'static str,
        /// The index asked for.
        index: u32,
        /// The array's length.
        count: u32,
    },
    /// A field holds a value the class header does not name.
    #[error("{method} {field} holds {value:#x}, which the class header does not define")]
    UndefinedValue {
        /// The method's name in the class header.
        method: &'static str,
        /// The field's name in the class header.
        field: &'static str,
        /// The value.
        value: u32,
    },
    /// The field asked for is not one of the method's.
    #[error("{method} has no field of {width} bits at bit {low}")]
    NoField {
        /// The method's name in the class header.
        method: &'static str,
        /// The field's lowest bit.
        low: u32,
        /// The field's width.
        width: u32,
    },
    /// A typed view of one class was asked of a mirror of another.
    #[error("this mirror holds class {actual:#x}, not class {expected:#x}")]
    WrongClass {
        /// The class the view reads.
        expected: u32,
        /// The mirror's class.
        actual: u32,
    },
}

/// How [`StateError::NoElement`] reads a number of indices: "no index", "1 index", "2 indices".
fn indices_given(count: usize) -> String {
    match count {
        0 => "no index".to_owned(),
        1 => "1 index".to_owned(),
        _ => format!("{count} indices"),
    }
}

/// A constant buffer bound to a bind group's slot: what `SET_CONSTANT_BUFFER_SELECTOR_A` to `_C`
/// held when `BIND_GROUP_CONSTANT_BUFFER(j)` was written (Eden-NX2's `ProcessCBBind`,
/// `eden-nx2/src/video_core/engines/maxwell_3d.cpp`). A slot never bound reads as the default:
/// not valid, address and size zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConstantBufferBinding {
    /// The bind's `VALID` field was `TRUE`.
    pub valid: bool,
    /// The GPU address: `ADDRESS_UPPER` (`_B`, 7:0) above `ADDRESS_LOWER` (`_C`).
    pub address: u64,
    /// The size in bytes: `SIZE` (`_A`, 16:0).
    pub size: u32,
}

/// The constant buffer `SET_CONSTANT_BUFFER_SELECTOR_A` to `_C` select now
/// ([`Mirror::constant_buffer_selector`]): what a bind records and where a constant-buffer load
/// writes (Eden-NX2's `regs.const_buffer.Address()` and `regs.const_buffer.size`, read by
/// `ProcessCBBind` and `ProcessCBMultiData`, `eden-nx2/src/video_core/engines/maxwell_3d.cpp`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConstantBufferSelector {
    /// The GPU address: `ADDRESS_UPPER` (`_B`, 7:0) above `ADDRESS_LOWER` (`_C`).
    pub address: u64,
    /// The size in bytes: `SIZE` (`_A`, 16:0).
    pub size: u32,
}

/// The back thread's mirror of one class's registers on one channel (gpu-protocol spec,
/// "Mirrors and dirty flags"): written only from the front's `Write` and `Run` entries, never by
/// the back itself (owner's decision 1), with a dirty bit per state group. The `state` module
/// docs give the rules.
#[derive(Clone, Debug)]
pub struct Mirror {
    channel: ChannelId,
    class: ClassId,
    rules: &'static ClassRules,
    map: &'static GroupMap,
    values: Box<[u32]>,
    dirty: GroupSet,
    ungrouped_words: u64,
    /// For the 3D class, [`BIND_GROUPS`] times [`CONSTANT_BUFFER_SLOTS`] bindings, by bind group
    /// then slot; empty for every other class.
    bindings: Box<[ConstantBufferBinding]>,
}

/// `BIND_GROUP_CONSTANT_BUFFER(j)` for every bind group.
const BIND_SPAN: MethodSpan = MethodSpan::array(&clc797::BIND_GROUP_CONSTANT_BUFFER, BIND_GROUPS);

impl Mirror {
    /// A mirror of class `class` on channel `channel`: the class's starting values
    /// ([`ClassRules::initial_values`], the same the front's register file starts from), every
    /// group its table names dirty (as Eden's 3D engine starts with every dirty flag set,
    /// `dirty.flags.flip()` in `Maxwell3D`'s constructor), no constant buffer bound and no
    /// ungrouped words.
    pub fn new(channel: ChannelId, class: ClassId) -> Self {
        let rules = ClassRules::for_class(class);
        let map = GroupMap::for_class(class);
        let bindings = if class.0 == clc797::CLASS {
            let count = BIND_GROUPS.saturating_mul(CONSTANT_BUFFER_SLOTS);
            let count = usize::try_from(count).unwrap_or(0);
            vec![ConstantBufferBinding::default(); count].into_boxed_slice()
        } else {
            Box::default()
        };
        Self {
            channel,
            class,
            rules,
            map,
            values: rules.initial_values(),
            dirty: map.named(),
            ungrouped_words: 0,
            bindings,
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

    /// Every register, indexed by method number (what [`crate::back::qmd::QmdTable`]'s
    /// `compute_method` reads, after the write is stored).
    pub fn registers(&self) -> &[u32] {
        &self.values
    }

    /// Register `method`'s value; `None` outside the file.
    pub fn value(&self, method: u32) -> Option<u32> {
        self.values.get(usize::try_from(method).ok()?).copied()
    }

    /// The constant buffer the selector registers name now. The bits above each register's
    /// field are ignored; a register outside the file (every class but 3D) reads as zero.
    pub fn constant_buffer_selector(&self) -> ConstantBufferSelector {
        let register = |method: &Method| self.value(method.method()).unwrap_or(0);
        let upper = clc797::SET_CONSTANT_BUFFER_SELECTOR_B_ADDRESS_UPPER
            .get(register(&clc797::SET_CONSTANT_BUFFER_SELECTOR_B));
        let lower = clc797::SET_CONSTANT_BUFFER_SELECTOR_C_ADDRESS_LOWER
            .get(register(&clc797::SET_CONSTANT_BUFFER_SELECTOR_C));
        ConstantBufferSelector {
            address: join(upper, lower),
            size: clc797::SET_CONSTANT_BUFFER_SELECTOR_A_SIZE
                .get(register(&clc797::SET_CONSTANT_BUFFER_SELECTOR_A)),
        }
    }

    /// The dirty groups.
    pub fn dirty(&self) -> GroupSet {
        self.dirty
    }

    /// Whether `group` is dirty.
    pub fn is_dirty(&self, group: Group) -> bool {
        self.dirty.contains(group)
    }

    /// Clears `groups` (deriving a draw does this for the groups it re-derived). Changes no
    /// register, binding or count.
    pub fn clear(&mut self, groups: GroupSet) {
        self.dirty = self.dirty.difference(groups);
    }

    /// The dirty groups, all of them cleared.
    pub fn take_dirty(&mut self) -> GroupSet {
        std::mem::take(&mut self.dirty)
    }

    /// The words written to methods in no group, counted (saturating) since the mirror was made.
    pub fn ungrouped_words(&self) -> u64 {
        self.ungrouped_words
    }

    /// Applies a ring entry: a `Write` or a `Run` for this mirror's channel and class is
    /// [`Mirror::write`] or [`Mirror::run`], and `Ok(true)`. Any other entry, and a `Write` or
    /// `Run` for another channel or class, changes nothing and is `Ok(false)`.
    pub fn apply(&mut self, entry: &RingEntry<'_>) -> Result<bool, StateError> {
        match entry {
            RingEntry::Write {
                channel,
                class,
                method,
                value,
            } if (*channel, *class) == (self.channel, self.class) => {
                self.write(u32::from(*method), *value).map(|()| true)
            }
            RingEntry::Run {
                channel,
                class,
                method,
                words,
            } if (*channel, *class) == (self.channel, self.class) => {
                self.run(u32::from(*method), words).map(|()| true)
            }
            _ => Ok(false),
        }
    }

    /// Stores one word in register `method` and marks its groups: the same as [`Mirror::run`]
    /// with one word.
    pub fn write(&mut self, method: u32, value: u32) -> Result<(), StateError> {
        self.run(method, &[value])
    }

    /// Stores each word in register `method` in turn (the register keeps the last), records a
    /// constant buffer binding for each word to `BIND_GROUP_CONSTANT_BUFFER(j)`, and then marks
    /// the method's groups once, or, for a method in no group, adds the number of words to
    /// [`Mirror::ungrouped_words`]. A method's groups are marked whatever the value, as Eden's
    /// `ProcessDirtyRegisters` does: the front has already dropped same-value state writes. No
    /// words changes and marks nothing.
    ///
    /// The mirror never applies a side-effect register itself: the front sends the result as a
    /// `Write` after the run ([`Mirror::run_positions`] gives where each word went).
    ///
    /// # Errors
    ///
    /// [`StateError::OutOfRange`] if `method` is not in the file; nothing is then changed.
    pub fn run(&mut self, method: u32, words: &[u32]) -> Result<(), StateError> {
        let index = self.index(method)?;
        let Some(&last) = words.last() else {
            return Ok(());
        };
        // Only a 3D mirror has bindings. The emptiness test is a shortcut, not a rule: without
        // it, `bind` on another class finds no entry and changes nothing (an equivalent
        // mutation, left unpinned).
        if !self.bindings.is_empty()
            && let Some(group) = BIND_SPAN.numbers().position(|number| number == method)
        {
            for &word in words {
                self.bind(group, word);
            }
        }
        if let Some(slot) = self.values.get_mut(index) {
            *slot = last;
        }
        let groups = self.map.groups(method);
        if groups.is_empty() {
            let count = u64::try_from(words.len()).unwrap_or(u64::MAX);
            self.ungrouped_words = self.ungrouped_words.saturating_add(count);
        } else {
            self.dirty = self.dirty.union(groups);
        }
        Ok(())
    }

    /// The values register `register` takes before each of `count` words written to trigger
    /// `method`, starting from its value now: for a `Run` entry, called before the side-effect
    /// `Write`s that follow it are applied. Each step applies, in the class's list order, every
    /// side effect ([`ClassRules::side_effects`]) whose trigger is `method` and whose register is
    /// `register`, through [`crate::regs::SideEffectRule::apply`], the front's own rule; with none,
    /// every position is the value now. For a run of `LOAD_CONSTANT_BUFFER(i)` words and
    /// `LOAD_CONSTANT_BUFFER_OFFSET`, these are the byte offsets the words go to.
    ///
    /// # Errors
    ///
    /// [`StateError::OutOfRange`] if `register` is not in the file.
    pub fn run_positions(
        &self,
        method: u32,
        register: u32,
        count: usize,
    ) -> Result<RunPositions, StateError> {
        let start = self.index(register)?;
        let value = self.values.get(start).copied().unwrap_or(0);
        Ok(RunPositions {
            side_effects: self.rules.side_effects(),
            method,
            register,
            value,
            remaining: count,
        })
    }

    /// One element of a method: its register's word, through which its fields are read. `indices`
    /// has one index per stride of the method (none for a method that is not an array); they are
    /// not bounded by an array length here (the typed views bound them).
    ///
    /// # Errors
    ///
    /// [`StateError::NoElement`] if the indices do not name an element, then
    /// [`StateError::OutOfRange`] if the element is not in the file.
    pub fn reg(&self, method: &'static Method, indices: &[u32]) -> Result<Reg, StateError> {
        let number = method.method_at(indices).ok_or(StateError::NoElement {
            method: method.name,
            indices: indices.len(),
        })?;
        let index = self.index(number)?;
        let word = self.values.get(index).copied().unwrap_or(0);
        Ok(Reg {
            method,
            number,
            word,
        })
    }

    /// The binding of bind group `group`'s slot `slot` (3D only; every other class has none).
    pub(super) fn binding(&self, group: u32, slot: u32) -> Option<ConstantBufferBinding> {
        let index = group
            .checked_mul(CONSTANT_BUFFER_SLOTS)?
            .checked_add(slot)?;
        self.bindings.get(usize::try_from(index).ok()?).copied()
    }

    /// The index of register `method`, or [`StateError::OutOfRange`].
    fn index(&self, method: u32) -> Result<usize, StateError> {
        let limit = self.rules.registers();
        let out_of_range = StateError::OutOfRange { method, limit };
        if method >= limit {
            return Err(out_of_range);
        }
        usize::try_from(method).map_err(|_| out_of_range)
    }

    /// Records a bind of the selected constant buffer to bind group `group` (Eden-NX2's
    /// `ProcessCBBind`).
    fn bind(&mut self, group: usize, word: u32) {
        let selected = self.constant_buffer_selector();
        let binding = ConstantBufferBinding {
            valid: clc797::BIND_GROUP_CONSTANT_BUFFER_VALID.get(word)
                == clc797::BIND_GROUP_CONSTANT_BUFFER_VALID_TRUE,
            address: selected.address,
            size: selected.size,
        };
        let slot = clc797::BIND_GROUP_CONSTANT_BUFFER_SHADER_SLOT.get(word);
        let Some(index) = u32::try_from(group)
            .ok()
            .and_then(|group| group.checked_mul(CONSTANT_BUFFER_SLOTS))
            .and_then(|base| base.checked_add(slot))
            .and_then(|index| usize::try_from(index).ok())
        else {
            return;
        };
        if let Some(entry) = self.bindings.get_mut(index) {
            *entry = binding;
        }
    }
}

/// A 64-bit address from its upper and lower words.
pub(super) fn join(upper: u32, lower: u32) -> u64 {
    (u64::from(upper) << u32::BITS) | u64::from(lower)
}

/// The values a side-effect register takes before each word of a run ([`Mirror::run_positions`]).
#[derive(Clone, Debug)]
pub struct RunPositions {
    side_effects: &'static [SideEffect],
    method: u32,
    register: u32,
    value: u32,
    remaining: usize,
}

impl Iterator for RunPositions {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        self.remaining = self.remaining.checked_sub(1)?;
        let position = self.value;
        for effect in self.side_effects {
            if effect.trigger.contains(self.method) && effect.rule.register() == self.register {
                self.value = effect.rule.apply(self.value);
            }
        }
        Some(position)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for RunPositions {}

/// One element of a method in a mirror ([`Mirror::reg`]): its number and word, read through the
/// generated tables of [`crate::classes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reg {
    method: &'static Method,
    number: u32,
    word: u32,
}

impl Reg {
    /// The generated method.
    pub fn method(&self) -> &'static Method {
        self.method
    }

    /// The element's method number.
    pub fn number(&self) -> u32 {
        self.number
    }

    /// The whole word.
    pub fn word(&self) -> u32 {
        self.word
    }

    /// The word as an IEEE single (for the methods Mesa lists as floats, `METHOD_IS_FLOAT` in
    /// `mesa-nouveau/src/nouveau/headers/class_parser.py`).
    pub fn float(&self) -> f32 {
        f32::from_bits(self.word)
    }

    /// The value of `field`, one of the method's fields.
    ///
    /// # Errors
    ///
    /// [`StateError::NoField`] if `field` is not one of the method's.
    pub fn field(&self, field: Field) -> Result<u32, StateError> {
        self.named_field(field).map(|_| field.get(self.word))
    }

    /// The named value `field` holds, from the class header's list for the field.
    ///
    /// # Errors
    ///
    /// [`StateError::NoField`] if `field` is not one of the method's, then
    /// [`StateError::UndefinedValue`] if the header names no value equal to it (for a field
    /// with no named values, every value).
    pub fn named(&self, field: Field) -> Result<&'static Value, StateError> {
        let named = self.named_field(field)?;
        let value = field.get(self.word);
        named
            .values
            .iter()
            .find(|named| named.value == value)
            .ok_or(StateError::UndefinedValue {
                method: self.method.name,
                field: named.name,
                value,
            })
    }

    fn named_field(&self, field: Field) -> Result<&'static crate::classes::NamedField, StateError> {
        self.method
            .fields
            .iter()
            .find(|named| named.bits == Bits::Word(field))
            .ok_or(StateError::NoField {
                method: self.method.name,
                low: field.low(),
                width: field.width(),
            })
    }
}

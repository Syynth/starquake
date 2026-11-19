//! What each class's register file does with a write: its size, whether it has shadow RAM, which
//! methods are plain state, and which methods change other registers ([`ClassRules`]).
//!
//! The side-effect lists ([`SIDE_EFFECTS_3D`], [`SIDE_EFFECTS_COMPUTE`]) and the state table
//! ([`STATE_3D`], in `state_3d.rs`) are written by hand and keyed on the generated constants of
//! [`crate::classes`]. Sources are relative to the reference folder (`tools/refs-path.sh`), at
//! the commits pinned in `docs/references.md`.

use crate::classes::{Method, cl902d, clc7c0, clc797};
use crate::field::Field;
use crate::fifo::METHOD_ADDRESS;
use crate::ring::ClassId;

use super::defaults::{DEFAULTS_2D, DEFAULTS_3D, RegisterDefault};
use super::state_3d::STATE_3D;

/// The bytes in a data word: the step of every byte offset that advances by one word.
pub const WORD_BYTES: u32 = u32::BITS / u8::BITS;

/// The elements of `LOAD_CONSTANT_BUFFER(i)`: 16, from Mesa's `METHOD_ARRAY_SIZES`
/// (`mesa-nouveau/src/nouveau/headers/class_parser.py`); Eden-NX2's `IsMethodExecutable` lists
/// the same 16 (`const_buffer.buffer` to `+ 15`, `eden-nx2/src/video_core/engines/maxwell_3d.cpp`).
pub const LOAD_CONSTANT_BUFFER_COUNT: u32 = 16;

/// The value `SET_FALCON04` leaves in `SET_MME_SHADOW_SCRATCH(0)`: Eden's stub of firmware
/// call 4 (`ProcessFirmwareCall4` in `eden-nx2/src/video_core/engines/maxwell_3d.cpp`, "stubbed
/// by setting 0xd00 to 1"). **Guess**: the real call runs a firmware blob whose other effects
/// nothing here models; a trace of the registers around a `SET_FALCON04` would show them.
pub const FALCON04_SCRATCH_VALUE: u32 = 1;

/// A method, or the first `count` elements of an array method with one index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MethodSpan {
    method: &'static Method,
    count: u32,
}

impl MethodSpan {
    /// One method (element 0, for an array method).
    pub const fn one(method: &'static Method) -> Self {
        Self { method, count: 1 }
    }

    /// Elements `0..count` of an array method with one index.
    pub const fn array(method: &'static Method, count: u32) -> Self {
        Self { method, count }
    }

    /// The generated method.
    pub const fn method(&self) -> &'static Method {
        self.method
    }

    /// The number of elements.
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// The method number of element `index`; `None` from [`MethodSpan::count`] up, for a
    /// non-zero index of a method that is not an array, for an array with more than one index,
    /// or if the address overflows.
    pub fn element(&self, index: u32) -> Option<u32> {
        if index >= self.count {
            return None;
        }
        if self.method.strides.is_empty() {
            return (index == 0).then(|| self.method.method());
        }
        self.method.method_at(&[index])
    }

    /// The method numbers of every element, in index order.
    pub fn numbers(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.count).map_while(|index| self.element(index))
    }

    /// Whether method number `number` is one of the elements.
    pub fn contains(&self, number: u32) -> bool {
        let first = self.method.method();
        let Some(offset) = number.checked_sub(first) else {
            return false;
        };
        match self.method.strides {
            [] => offset == 0 && self.count > 0,
            [stride] => {
                let stride_methods = stride / WORD_BYTES;
                if stride_methods == 0 {
                    return false;
                }
                offset % stride_methods == 0 && offset / stride_methods < self.count
            }
            _ => false,
        }
    }
}

/// How a trigger method changes another register (the gpu-protocol spec's "side effects that
/// change other registers"). Applied once per data word of the trigger, after the trigger's own
/// value is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideEffectRule {
    /// Adds `step` to `field` of register `register`, wrapping within the field and leaving the
    /// register's other bits alone, when the field's value is a multiple of `align`; otherwise
    /// the register is unchanged.
    Advance {
        /// The method number of the register changed.
        register: u32,
        /// The field that advances.
        field: Field,
        /// What each word adds.
        step: u32,
        /// The field advances only from a multiple of this (1: always).
        align: u32,
    },
    /// Sets register `register` to `value`.
    Set {
        /// The method number of the register changed.
        register: u32,
        /// The value it gets.
        value: u32,
    },
}

impl SideEffectRule {
    /// The method number of the register the rule changes.
    pub const fn register(&self) -> u32 {
        match *self {
            Self::Advance { register, .. } | Self::Set { register, .. } => register,
        }
    }

    /// The register's value after one word of the trigger, given its value `old` before.
    pub fn apply(&self, old: u32) -> u32 {
        match *self {
            Self::Advance {
                field, step, align, ..
            } => {
                let current = field.get(old);
                match current.checked_rem(align) {
                    Some(0) => field.insert_truncated(old, current.wrapping_add(step)),
                    _ => old,
                }
            }
            Self::Set { value, .. } => value,
        }
    }
}

/// A trigger method (or array of them) and the register it changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SideEffect {
    /// The methods whose writes apply the rule.
    pub trigger: MethodSpan,
    /// What each word written to them does.
    pub rule: SideEffectRule,
}

/// The 3D class's side-effect registers, in the order their writes are sent (each has its own
/// fixture in `regs/tests.rs`):
///
/// 1. each `LOAD_CONSTANT_BUFFER(i)` word advances `LOAD_CONSTANT_BUFFER_OFFSET` by 4 bytes
///    (Eden-NX2's `FrontWrite` and `FrontCallMultiMethod`,
///    `eden-nx2/src/video_core/engines/maxwell_3d.cpp`), within the header's `V` field (15:0);
/// 2. each `LOAD_MME_START_ADDRESS_RAM` word advances `LOAD_MME_START_ADDRESS_RAM_POINTER` by
///    one (`FrontWrite`, `++pointer`);
/// 3. `SET_FALCON04` sets `SET_MME_SHADOW_SCRATCH(0)` to [`FALCON04_SCRATCH_VALUE`] (`FrontWrite`
///    and `ProcessFirmwareCall4`; a **guess**, see the constant);
/// 4. each `LOAD_ROOT_TABLE` word advances the `OFFSET` field (15:8) of
///    `SET_ROOT_TABLE_SELECTOR` by 4 bytes when it is a multiple of 4 (`NativeRootTables::Load`,
///    `eden-pr4588/src/video_core/native_root_table.h`, which keeps the offset in a private
///    variable instead and leaves the selector register as written, so a macro reading method
///    0x141 after a load sees the advanced offset here and the written one there; the wrap after
///    252 is a **guess**; both are divergences in the module docs of `regs`, pinned by
///    `root_table_offset_advances`).
///
/// The macro upload pointer is not here: the macro host keeps the upload as a word offset
/// (gpu-protocol, "Engine methods: the register file", step 2; decided by the owner,
/// 2026-11-18), pinned by `instruction_upload_leaves_its_pointer_alone`.
pub const SIDE_EFFECTS_3D: &[SideEffect] = &[
    SideEffect {
        trigger: MethodSpan::array(&clc797::LOAD_CONSTANT_BUFFER, LOAD_CONSTANT_BUFFER_COUNT),
        rule: SideEffectRule::Advance {
            register: clc797::LOAD_CONSTANT_BUFFER_OFFSET.method(),
            field: clc797::LOAD_CONSTANT_BUFFER_OFFSET_V,
            step: WORD_BYTES,
            align: 1,
        },
    },
    SideEffect {
        trigger: MethodSpan::one(&clc797::LOAD_MME_START_ADDRESS_RAM),
        rule: SideEffectRule::Advance {
            register: clc797::LOAD_MME_START_ADDRESS_RAM_POINTER.method(),
            field: clc797::LOAD_MME_START_ADDRESS_RAM_POINTER_V,
            step: 1,
            align: 1,
        },
    },
    SideEffect {
        trigger: MethodSpan::one(&clc797::SET_FALCON04),
        rule: SideEffectRule::Set {
            register: clc797::SET_MME_SHADOW_SCRATCH.method(),
            value: FALCON04_SCRATCH_VALUE,
        },
    },
    SideEffect {
        trigger: MethodSpan::one(&clc797::LOAD_ROOT_TABLE),
        rule: SideEffectRule::Advance {
            register: clc797::SET_ROOT_TABLE_SELECTOR.method(),
            field: clc797::SET_ROOT_TABLE_SELECTOR_OFFSET,
            step: WORD_BYTES,
            align: WORD_BYTES,
        },
    },
];

/// The compute class's side-effect registers: each `LOAD_MME_START_ADDRESS_RAM` word advances
/// `LOAD_MME_START_ADDRESS_RAM_POINTER` by one, as the 3D class's does. Eden-NX2's compute
/// engine keeps this pointer outside its registers (`FrontApply` and `ProcessNativeMethod`,
/// `eden-nx2/src/video_core/engines/kepler_compute.cpp`); see "Divergences" in the `regs`
/// module docs.
pub const SIDE_EFFECTS_COMPUTE: &[SideEffect] = &[SideEffect {
    trigger: MethodSpan::one(&clc7c0::LOAD_MME_START_ADDRESS_RAM),
    rule: SideEffectRule::Advance {
        register: clc7c0::LOAD_MME_START_ADDRESS_RAM_POINTER.method(),
        field: clc7c0::LOAD_MME_START_ADDRESS_RAM_POINTER_V,
        step: 1,
        align: 1,
    },
}];

/// The 3D class's data-load methods, which take the words written even in shadow RAM replay
/// mode: `LOAD_CONSTANT_BUFFER(0..16)` (constant-buffer loads) and `LOAD_INLINE_DATA`
/// (inline-to-memory data), as the gpu-protocol spec names them ("Engine methods: the register
/// file", step 1; decided by the owner, 2026-11-18). They are the two cases of Eden-NX2's
/// `Maxwell3D::CallMultiMethod` (`eden-nx2/src/video_core/engines/maxwell_3d.cpp`) that hand a
/// run's words to the upload without `ProcessShadowRam`: `ProcessCBMultiData(base_start,
/// amount)` and `upload_state.ProcessData(base_start, amount)`. **Guess** until a trace shows
/// replay mode during an upload; see "Divergences" in the `regs` module docs for the rest.
pub const DATA_LOADS_3D: &[MethodSpan] = &[
    MethodSpan::array(&clc797::LOAD_CONSTANT_BUFFER, LOAD_CONSTANT_BUFFER_COUNT),
    MethodSpan::one(&clc797::LOAD_INLINE_DATA),
];

/// What one class's register file does with a write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassRules {
    registers: u32,
    shadow_control: Option<u32>,
    data_loads: &'static [MethodSpan],
    state: &'static [MethodSpan],
    side_effects: &'static [SideEffect],
    defaults: &'static [RegisterDefault],
}

/// The 3D class (`AMPERE_B`, `0xC797`): registers below its macro range (`CALL_MME_MACRO`,
/// method 0xE00), shadow RAM controlled by `SET_MME_SHADOW_RAM_CONTROL` with the data loads
/// [`DATA_LOADS_3D`], the state table [`STATE_3D`], the side effects [`SIDE_EFFECTS_3D`] and
/// the reset values [`DEFAULTS_3D`].
pub const THREE_D: ClassRules = ClassRules {
    registers: clc797::CALL_MME_MACRO.method(),
    shadow_control: Some(clc797::SET_MME_SHADOW_RAM_CONTROL.method()),
    data_loads: DATA_LOADS_3D,
    state: STATE_3D,
    side_effects: SIDE_EFFECTS_3D,
    defaults: DEFAULTS_3D,
};

/// The compute class (`AMPERE_COMPUTE_B`, `0xC7C0`): registers below its macro range
/// (`CALL_MME_MACRO`, method 0xE00, as Eden-NX2's `kepler_compute.cpp` calls macros from 0xE00),
/// no shadow RAM (the gpu-protocol spec applies it "for 3D", and Eden-NX2's compute engine has
/// none, so no data loads either), no state methods (Eden-NX2's compute front sends every
/// write, `FrontCallMethod`), the side effects [`SIDE_EFFECTS_COMPUTE`], and no reset values
/// (Eden-NX2's `KeplerCompute` constructor sets no register).
pub const COMPUTE: ClassRules = ClassRules {
    registers: clc7c0::CALL_MME_MACRO.method(),
    shadow_control: None,
    data_loads: &[],
    state: &[],
    side_effects: SIDE_EFFECTS_COMPUTE,
    defaults: &[],
};

/// Every other class (DMA copy, inline-to-memory, and any class no rule set names): every
/// method a method header can address (`NVC76F_DMA_METHOD_ADDRESS`, 12 bits), no shadow RAM (so
/// no data loads), no state methods and no side effects, so every write is sent, and no reset
/// values (Eden-NX2's `MaxwellDMA` and `KeplerMemory` constructors set no register).
pub const OTHER: ClassRules = ClassRules {
    registers: METHOD_ADDRESS.max() + 1,
    shadow_control: None,
    data_loads: &[],
    state: &[],
    side_effects: &[],
    defaults: &[],
};

/// The 2D class (`FERMI_TWOD_A`, `0x902D`): as [`OTHER`] (every method a method header can
/// address, no shadow RAM, no state methods, no side effects), with the reset values
/// [`DEFAULTS_2D`].
pub const TWO_D: ClassRules = ClassRules {
    defaults: DEFAULTS_2D,
    ..OTHER
};

impl ClassRules {
    /// The rules for class `class`: [`THREE_D`] for `0xC797`, [`COMPUTE`] for `0xC7C0`,
    /// [`TWO_D`] for `0x902D`, and [`OTHER`] for every other class.
    pub fn for_class(class: ClassId) -> &'static Self {
        match class.0 {
            clc797::CLASS => &THREE_D,
            clc7c0::CLASS => &COMPUTE,
            cl902d::CLASS => &TWO_D,
            _ => &OTHER,
        }
    }

    /// The number of registers: methods `0..registers` are in the file.
    pub const fn registers(&self) -> u32 {
        self.registers
    }

    /// The method number of the shadow RAM control, for a class with shadow RAM.
    pub const fn shadow_control(&self) -> Option<u32> {
        self.shadow_control
    }

    /// The data-load methods: in shadow RAM replay mode they take the words written, where
    /// every other method takes the shadow file's value. Empty for a class without shadow RAM.
    pub const fn data_loads(&self) -> &'static [MethodSpan] {
        self.data_loads
    }

    /// Whether method number `method` is one of [`ClassRules::data_loads`].
    pub fn is_data_load(&self, method: u32) -> bool {
        self.data_loads.iter().any(|span| span.contains(method))
    }

    /// The plain-state methods. A method in none of them is a trigger.
    pub const fn state(&self) -> &'static [MethodSpan] {
        self.state
    }

    /// The side-effect registers, in the order their writes are sent.
    pub const fn side_effects(&self) -> &'static [SideEffect] {
        self.side_effects
    }

    /// The reset values: what each listed register holds when a file of the class is made.
    pub const fn defaults(&self) -> &'static [RegisterDefault] {
        self.defaults
    }

    /// The registers' values when a file of the class is made: zero, except for the registers
    /// [`ClassRules::defaults`] lists, which hold their reset values. The one starting state
    /// that both the front's [`super::RegFile`] (and its shadow file, which starts equal to
    /// the register file, as Eden's `shadow_state = regs` does at the end of
    /// `InitializeRegisterDefaults`) and the back's mirror (`crate::back::state::Mirror`) begin
    /// from, so a same-value write the front drops is never one the back has not seen. A
    /// default outside the file is skipped (none is: pinned by
    /// `reset_values_name_registers_of_the_file_once`).
    pub fn initial_values(&self) -> Box<[u32]> {
        let registers = usize::try_from(self.registers).unwrap_or(0);
        let mut values = vec![0; registers].into_boxed_slice();
        for default in self.defaults {
            for number in default.span.numbers() {
                if let Some(slot) = usize::try_from(number)
                    .ok()
                    .and_then(|index| values.get_mut(index))
                {
                    *slot = default.value;
                }
            }
        }
        values
    }
}

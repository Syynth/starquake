//! The input shared memory `hid` hands an applet: its layout, and what it holds before any
//! input arrives.
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//! `eden/src/hid_core/resources/shared_memory_format.h` (`SharedMemoryFormat` and the formats
//! inside it, with the offsets its `static_assert`s pin), `ring_lifo.h` (`Lifo`),
//! `npad/npad_types.h` (`NpadInternalState`'s fields, `NpadFullKeyColorState`,
//! `NpadJoyColorState`, `ColorAttribute`, `NpadCondition`, `NpadJoyHoldType`) and
//! `eden/src/hid_core/hid_types.h` (`SixAxisSensorState`). The layout is checked against the
//! client, libnx (`libnx/nx/include/switch/services/hid.h`: `HidSharedMemory`,
//! `HidNpadInternalState`, `HidCommonLifoHeader`; ISC).
//!
//! # What a fresh one holds
//!
//! Eden builds the memory with `std::construct_at`
//! (`eden/src/hid_core/resources/shared_memory_holder.cpp`, `SharedMemoryHolder::Initialize`) on
//! pages the kernel zeroed, so it holds the default member initialisers of `SharedMemoryFormat`
//! and nothing else until a controller is sampled; `SharedMemoryFormat::Initialize` then sets
//! `npad_condition` to its defaults again. The initialisers that are not zero are exactly these
//! ([`initial_image`] writes them, and nothing else):
//!
//! - every `Lifo`'s `total_buffer_count`, `HidEntryCount` (17): the debug pad's, the touch
//!   screen's, the mouse's, the keyboard's, the three system buttons', the gesture's and the
//!   debug mouse's, and in each of the ten npad entries its seven controller lifos, six
//!   six-axis-sensor lifos and its GameCube trigger lifo;
//! - in each npad entry, the full-key and the Joy-Con colour states' `attribute`,
//!   `ColorAttribute::NoController` (2);
//! - `npad_condition`: `is_initialized` 1, `hold_type` `NpadJoyHoldType::Horizontal` (1),
//!   `is_valid` 1.
//!
//! The other initialisers in those headers are zero (`NpadStyleSet::None`,
//! `NpadJoyAssignmentMode::Dual`, `AppletFooterUiType::None`, `GestureType::Idle`,
//! `GestureDirection::None`, `TouchScreenModeForNx::UseSystemSetting` and the `{}` ones), and the
//! digitizer, input detector and unique pad formats start with a `CommonHeader`, which has no
//! initialiser. No controller is connected: there is no host input yet, so nothing is ever
//! sampled into the memory. Eden connects one by default and starts sampling when an applet
//! resource is created (`ResourceManager::CreateAppletResource` activates the npad, six-axis,
//! touch screen and gesture resources); that is a listed divergence of [`super`].

use crate::shared_image::{blank, put};

/// The name [`put`] logs a misplaced field under.
const MEMORY: &str = "hid";

/// The size of the input shared memory: `sizeof(SharedMemoryFormat)`
/// (`static_assert(sizeof(SharedMemoryFormat) == 0x40000)`), the size libnx maps
/// (`shmemLoadRemote(&g_hidSharedmem, sharedmem_handle, 0x40000, Perm_R)` in
/// `libnx/nx/source/services/hid.c`).
pub const SIZE: u64 = 0x40000;

/// The number of entries in every ring of samples: `HidEntryCount` (`shared_memory_format.h`),
/// a `Lifo`'s initial `total_buffer_count`.
pub const LIFO_ENTRY_COUNT: u64 = 17;

/// Where a `Lifo`'s `total_buffer_count` (an `s64`) lies in it: after its `s64 timestamp`
/// (`ring_lifo.h`; libnx's `HidCommonLifoHeader` calls it `buffer_count`, after `unused`).
pub const LIFO_BUFFER_COUNT: u64 = 0x8;
/// The size of a `Lifo`'s header: four `s64`s (`timestamp`, `total_buffer_count`,
/// `buffer_tail`, `buffer_count`) before its entries.
pub const LIFO_HEADER_SIZE: u64 = 0x20;
/// The size of an entry's sampling number, before its state (`AtomicStorage<State>`: `s64
/// sampling_number; State state;`).
pub const LIFO_SAMPLING_NUMBER_SIZE: u64 = 0x8;

/// The size of a `Lifo` of [`LIFO_ENTRY_COUNT`] entries of a state of `state_size` bytes.
#[must_use]
pub const fn lifo_size(state_size: u64) -> u64 {
    LIFO_HEADER_SIZE + LIFO_ENTRY_COUNT * (LIFO_SAMPLING_NUMBER_SIZE + state_size)
}

/// Where each format lies in the memory (`shared_memory_format.h`, its `offsetof`
/// `static_assert`s).
pub mod offset {
    /// `debug_pad` (`DebugPadSharedMemoryFormat`: a lifo first).
    pub const DEBUG_PAD: u64 = 0x0;
    /// `touch_screen` (a lifo first).
    pub const TOUCH_SCREEN: u64 = 0x400;
    /// `mouse` (a lifo first).
    pub const MOUSE: u64 = 0x3400;
    /// `keyboard` (a lifo first).
    pub const KEYBOARD: u64 = 0x3800;
    /// `home_button` (a lifo first).
    pub const HOME_BUTTON: u64 = 0x4C00;
    /// `sleep_button` (a lifo first).
    pub const SLEEP_BUTTON: u64 = 0x4E00;
    /// `capture_button` (a lifo first).
    pub const CAPTURE_BUTTON: u64 = 0x5000;
    /// `npad`: [`super::NPAD_ENTRY_COUNT`] entries of [`super::NPAD_ENTRY_SIZE`] bytes.
    pub const NPAD: u64 = 0x9A00;
    /// `gesture` (a lifo first).
    pub const GESTURE: u64 = 0x3BA00;
    /// `debug_mouse` (a lifo first).
    pub const DEBUG_MOUSE: u64 = 0x3DC00;
    /// `npad_condition` (`NpadCondition`).
    pub const NPAD_CONDITION: u64 = 0x3E200;
}

/// The formats outside the npad entries that start with a `Lifo`, in the memory's order.
pub const LIFOS: [u64; 9] = [
    offset::DEBUG_PAD,
    offset::TOUCH_SCREEN,
    offset::MOUSE,
    offset::KEYBOARD,
    offset::HOME_BUTTON,
    offset::SLEEP_BUTTON,
    offset::CAPTURE_BUTTON,
    offset::GESTURE,
    offset::DEBUG_MOUSE,
];

/// The number of npad entries: `MaxSupportedNpadIdTypes` (`npad_types.h`); libnx's
/// `HidNpadSharedMemoryFormat` has as many.
pub const NPAD_ENTRY_COUNT: u64 = 10;
/// The size of one npad entry (`static_assert(sizeof(NpadSharedMemoryEntry) == 0x5000)`).
pub const NPAD_ENTRY_SIZE: u64 = 0x5000;

/// Inside an npad entry (`NpadInternalState`, whose fields libnx's `HidNpadInternalState`
/// lists in the same order).
pub mod npad {
    use super::lifo_size;

    /// `fullkey_color.attribute` (after `style_tag` and `assignment_mode`, a `u32` each).
    pub const FULL_KEY_COLOR_ATTRIBUTE: u64 = 0x8;
    /// `joycon_color.attribute` (after the 0xC-byte `NpadFullKeyColorState`).
    pub const JOY_COLOR_ATTRIBUTE: u64 = 0x14;
    /// The first controller lifo, `fullkey_lifo` (after the 0x14-byte `NpadJoyColorState`).
    pub const FIRST_CONTROLLER_LIFO: u64 = 0x28;
    /// The controller lifos: full key, handheld, dual, left, right, Palma, system ext.
    pub const CONTROLLER_LIFO_COUNT: u64 = 7;
    /// `sizeof(NPadGenericState)`.
    pub const CONTROLLER_STATE_SIZE: u64 = 0x28;
    /// The six-axis-sensor lifos, after the controller lifos: full key, handheld, dual left,
    /// dual right, left, right.
    pub const SIX_AXIS_LIFO_COUNT: u64 = 6;
    /// `sizeof(SixAxisSensorState)`.
    pub const SIX_AXIS_STATE_SIZE: u64 = 0x60;
    /// The first six-axis-sensor lifo, `sixaxis_fullkey_lifo`.
    pub const FIRST_SIX_AXIS_LIFO: u64 =
        FIRST_CONTROLLER_LIFO + CONTROLLER_LIFO_COUNT * lifo_size(CONTROLLER_STATE_SIZE);
    /// `gc_trigger_lifo`, after the device type, the system properties, the battery levels,
    /// the applet footer and 0x20 unknown bytes (libnx: `reserved_x4208`, then
    /// `gc_trigger_lifo`).
    pub const GC_TRIGGER_LIFO: u64 = 0x4228;
}

/// `ColorAttribute::NoController` (`npad_types.h`), what a colour state says with no controller.
pub const COLOR_ATTRIBUTE_NO_CONTROLLER: u32 = 2;

/// `NpadCondition`'s fields after its first `u32` (`npad_types.h`): `is_initialized` 1,
/// `hold_type` `NpadJoyHoldType::Horizontal` (1), `is_valid` 1.
pub const NPAD_CONDITION_DEFAULT: [u32; 3] = [1, 1, 1];
/// Where `is_initialized`, the first of [`NPAD_CONDITION_DEFAULT`], lies in `NpadCondition`.
pub const NPAD_CONDITION_FIELDS: u64 = 0x4;

/// The offsets of every `Lifo` in the memory, the npad entries' included, in the memory's order.
#[must_use]
pub fn lifo_offsets() -> Vec<u64> {
    let mut lifos: Vec<u64> = LIFOS
        .iter()
        .copied()
        .filter(|at| *at < offset::NPAD)
        .collect();
    for entry in 0..NPAD_ENTRY_COUNT {
        let base = offset::NPAD + entry * NPAD_ENTRY_SIZE;
        let controllers = (0..npad::CONTROLLER_LIFO_COUNT).map(|index| {
            base + npad::FIRST_CONTROLLER_LIFO + index * lifo_size(npad::CONTROLLER_STATE_SIZE)
        });
        let six_axis = (0..npad::SIX_AXIS_LIFO_COUNT).map(|index| {
            base + npad::FIRST_SIX_AXIS_LIFO + index * lifo_size(npad::SIX_AXIS_STATE_SIZE)
        });
        lifos.extend(controllers.chain(six_axis));
        lifos.push(base + npad::GC_TRIGGER_LIFO);
    }
    lifos.extend(LIFOS.iter().copied().filter(|at| *at > offset::NPAD));
    lifos
}

/// The memory's bytes before any input arrives (see the module docs): [`SIZE`] bytes, zero but
/// for the initialisers listed there.
#[must_use]
pub fn initial_image() -> Vec<u8> {
    let mut image = blank(SIZE);
    for lifo in lifo_offsets() {
        put(
            &mut image,
            lifo + LIFO_BUFFER_COUNT,
            &LIFO_ENTRY_COUNT.to_le_bytes(),
            MEMORY,
        );
    }
    let no_controller = COLOR_ATTRIBUTE_NO_CONTROLLER.to_le_bytes();
    for entry in 0..NPAD_ENTRY_COUNT {
        let base = offset::NPAD + entry * NPAD_ENTRY_SIZE;
        put(
            &mut image,
            base + npad::FULL_KEY_COLOR_ATTRIBUTE,
            &no_controller,
            MEMORY,
        );
        put(
            &mut image,
            base + npad::JOY_COLOR_ATTRIBUTE,
            &no_controller,
            MEMORY,
        );
    }
    let condition: Vec<u8> = NPAD_CONDITION_DEFAULT
        .iter()
        .flat_map(|field| field.to_le_bytes())
        .collect();
    put(
        &mut image,
        offset::NPAD_CONDITION + NPAD_CONDITION_FIELDS,
        &condition,
        MEMORY,
    );
    image
}

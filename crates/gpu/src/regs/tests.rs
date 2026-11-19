//! Tests of the register files: the shadow RAM modes, each side-effect register as its own
//! fixture, same-value dropping, runs, range errors and the hand-written tables.

use std::borrow::Cow;
use std::collections::BTreeSet;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use super::*;
use crate::classes::{Method, cl902d, cla140, clc7b5, clc7c0, clc797};
use crate::fifo::METHOD_ADDRESS;
use crate::interface::ChannelId;
use crate::ring::{ClassId, MAX_RUN_WORDS, RingEntry};

mod defaults;

/// Not channel 0, so a hard-coded channel fails.
const CHANNEL: ChannelId = ChannelId(7);
const THREE_D_CLASS: ClassId = ClassId(clc797::CLASS);
const COMPUTE_CLASS: ClassId = ClassId(clc7c0::CLASS);

const TRACK: u32 = clc797::SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_TRACK;
const TRACK_WITH_FILTER: u32 = clc797::SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_TRACK_WITH_FILTER;
const PASSTHROUGH: u32 = clc797::SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_PASSTHROUGH;
const REPLAY: u32 = clc797::SET_MME_SHADOW_RAM_CONTROL_MODE_METHOD_REPLAY;

/// `SET_MME_SHADOW_RAM_CONTROL`.
fn control() -> u32 {
    clc797::SET_MME_SHADOW_RAM_CONTROL.method()
}

/// A plain-state 3D method.
fn state() -> u32 {
    clc797::SET_ALPHA_REF.method()
}

/// A second plain-state 3D method.
fn other_state() -> u32 {
    clc797::SET_STENCIL_FUNC_REF.method()
}

/// A 3D trigger without side effects that is not a data load.
fn trigger() -> u32 {
    clc797::LAUNCH_DMA.method()
}

/// `LOAD_INLINE_DATA`: a data-load trigger without side effects.
fn inline_data() -> u32 {
    clc797::LOAD_INLINE_DATA.method()
}

/// `LOAD_CONSTANT_BUFFER(index)`.
fn constant_buffer(index: u32) -> u32 {
    clc797::LOAD_CONSTANT_BUFFER.method_at(&[index]).unwrap()
}

fn offset() -> u32 {
    clc797::LOAD_CONSTANT_BUFFER_OFFSET.method()
}

fn three_d() -> RegFile {
    RegFile::new(CHANNEL, THREE_D_CLASS)
}

fn compute() -> RegFile {
    RegFile::new(CHANNEL, COMPUTE_CLASS)
}

fn write_in(class: ClassId, method: u32, value: u32) -> RingEntry<'static> {
    RingEntry::Write {
        channel: CHANNEL,
        class,
        method: u16::try_from(method).unwrap(),
        value,
    }
}

/// A 3D `Write` entry.
fn write(method: u32, value: u32) -> RingEntry<'static> {
    write_in(THREE_D_CLASS, method, value)
}

/// A 3D `Run` entry.
fn run(method: u32, words: &[u32]) -> RingEntry<'static> {
    RingEntry::Run {
        channel: CHANNEL,
        class: THREE_D_CLASS,
        method: u16::try_from(method).unwrap(),
        words: Cow::Owned(words.to_vec()),
    }
}

/// Writes one word and returns what was sent.
fn send(file: &mut RegFile, method: u32, word: u32) -> Vec<RingEntry<'static>> {
    let mut out = Vec::new();
    file.write(method, word, &mut out).unwrap();
    out
}

/// Writes a run and returns what was sent.
fn send_run(file: &mut RegFile, method: u32, words: &[u32]) -> Vec<RingEntry<'static>> {
    let mut out = Vec::new();
    file.write_run(method, words, &mut out).unwrap();
    out
}

/// Every register and shadow value of a file, for comparing two files.
fn snapshot(file: &RegFile) -> (Vec<Option<u32>>, Vec<Option<u32>>) {
    let registers = file.rules().registers();
    (
        (0..=registers).map(|method| file.get(method)).collect(),
        (0..=registers).map(|method| file.shadow(method)).collect(),
    )
}

// ---------------------------------------------------------------------------------------------
// New files and the class rules
// ---------------------------------------------------------------------------------------------

#[test]
fn new_three_d_file_starts_from_its_reset_values_tracking_and_dropping() {
    let file = three_d();
    assert_eq!(file.channel(), CHANNEL);
    assert_eq!(file.class(), THREE_D_CLASS);
    assert_eq!(file.rules(), &THREE_D);
    // Eden's `shadow_state` starts as a copy of the registers, whose shadow control is 0, and
    // mode 0 is METHOD_TRACK.
    assert_eq!(file.shadow_mode(), Some(ShadowMode::Track));
    assert!(file.drop_same_values());
    let registers = THREE_D.registers();
    let initial = THREE_D.initial_values();
    for method in 0..registers {
        let value = initial[method as usize];
        assert_eq!(file.get(method), Some(value), "register {method:#x}");
        assert_eq!(file.shadow(method), Some(value), "shadow {method:#x}");
    }
    assert_eq!(file.get(registers), None);
    assert_eq!(file.shadow(registers), None);
}

#[test]
fn new_compute_file_has_registers_but_no_shadow_ram() {
    let file = compute();
    assert_eq!(file.rules(), &COMPUTE);
    assert_eq!(file.shadow_mode(), None);
    let registers = COMPUTE.registers();
    assert_eq!(file.get(registers - 1), Some(0));
    assert_eq!(file.get(registers), None);
    assert_eq!(file.shadow(control()), None);
    assert_eq!(file.shadow(0), None);
}

#[test]
fn rules_are_picked_by_class_id() {
    assert_eq!(ClassRules::for_class(THREE_D_CLASS), &THREE_D);
    assert_eq!(ClassRules::for_class(COMPUTE_CLASS), &COMPUTE);
    assert_eq!(ClassRules::for_class(ClassId(cl902d::CLASS)), &TWO_D);
    for class in [clc7b5::CLASS, cla140::CLASS, 0, u32::MAX] {
        assert_eq!(ClassRules::for_class(ClassId(class)), &OTHER, "{class:#x}");
    }
}

#[test]
fn other_classes_hold_every_method_a_header_can_address() {
    let file = RegFile::new(CHANNEL, ClassId(clc7b5::CLASS));
    let last = METHOD_ADDRESS.max();
    assert_eq!(file.get(last), Some(0));
    assert_eq!(file.get(last + 1), None);
    assert_eq!(file.shadow_mode(), None);
}

// ---------------------------------------------------------------------------------------------
// Shadow RAM
// ---------------------------------------------------------------------------------------------

#[test]
fn track_mode_stores_the_word_in_the_shadow_file() {
    let mut file = three_d();
    assert_eq!(send(&mut file, state(), 5), [write(state(), 5)]);
    assert_eq!(file.get(state()), Some(5));
    assert_eq!(file.shadow(state()), Some(5));
}

#[test]
fn track_with_filter_stores_like_track() {
    let mut file = three_d();
    send(&mut file, control(), TRACK_WITH_FILTER);
    assert_eq!(file.shadow_mode(), Some(ShadowMode::TrackWithFilter));
    assert_eq!(send(&mut file, state(), 9), [write(state(), 9)]);
    assert_eq!(file.shadow(state()), Some(9));
}

#[test]
fn passthrough_leaves_the_shadow_file_alone() {
    let mut file = three_d();
    send(&mut file, state(), 5);
    send(&mut file, control(), PASSTHROUGH);
    assert_eq!(file.shadow_mode(), Some(ShadowMode::Passthrough));
    assert_eq!(send(&mut file, state(), 6), [write(state(), 6)]);
    assert_eq!(file.get(state()), Some(6));
    assert_eq!(file.shadow(state()), Some(5));
}

#[test]
fn replay_writes_the_shadow_value_not_the_word() {
    let mut file = three_d();
    send(&mut file, state(), 5);
    send(&mut file, trigger(), 0xab);
    send(&mut file, control(), PASSTHROUGH);
    send(&mut file, state(), 7);
    send(&mut file, control(), REPLAY);
    assert_eq!(file.shadow_mode(), Some(ShadowMode::Replay));
    // The register holds 7, the shadow 5: the write of 6 replays 5.
    assert_eq!(send(&mut file, state(), 6), [write(state(), 5)]);
    assert_eq!(file.get(state()), Some(5));
    assert_eq!(file.shadow(state()), Some(5));
    assert_eq!(send(&mut file, trigger(), 0xcd), [write(trigger(), 0xab)]);
}

#[test]
fn replay_write_to_the_control_reads_back_and_sets_the_mode() {
    // Row 144: the mode is the shadow copy of register 0x49 itself.
    let mut file = three_d();
    send(&mut file, control(), REPLAY);
    assert_eq!(file.get(control()), Some(REPLAY));
    // In replay, the register takes the shadow copy (REPLAY), and the mode becomes the word.
    assert_eq!(
        send(&mut file, control(), PASSTHROUGH),
        [write(control(), REPLAY)]
    );
    assert_eq!(file.get(control()), Some(REPLAY));
    assert_eq!(file.shadow(control()), Some(PASSTHROUGH));
    assert_eq!(file.shadow_mode(), Some(ShadowMode::Passthrough));
    // Passthrough really is in force: the shadow file no longer follows.
    send(&mut file, state(), 3);
    assert_eq!(file.shadow(state()), Some(0));
    // From passthrough, the control's register takes the word.
    assert_eq!(
        send(&mut file, control(), TRACK_WITH_FILTER),
        [write(control(), TRACK_WITH_FILTER)]
    );
    assert_eq!(file.shadow(control()), Some(TRACK_WITH_FILTER));
}

#[test]
fn run_changing_the_mode_mid_run() {
    // Row 144: a multi-word run that changes the shadow RAM mode mid-run.
    let mut file = three_d();
    let entries = send_run(
        &mut file,
        control(),
        &[REPLAY, PASSTHROUGH, TRACK_WITH_FILTER],
    );
    // Word 0 under track: REPLAY. Word 1 under replay: the shadow copy, REPLAY. Word 2 under
    // passthrough: itself.
    assert_eq!(
        entries,
        [run(control(), &[REPLAY, REPLAY, TRACK_WITH_FILTER])]
    );
    assert_eq!(file.shadow_mode(), Some(ShadowMode::TrackWithFilter));
    assert_eq!(file.get(control()), Some(TRACK_WITH_FILTER));
}

#[test]
fn mode_is_the_mode_field() {
    let high = !clc797::SET_MME_SHADOW_RAM_CONTROL_MODE.max();
    for (value, mode) in [
        (TRACK, ShadowMode::Track),
        (TRACK_WITH_FILTER, ShadowMode::TrackWithFilter),
        (PASSTHROUGH, ShadowMode::Passthrough),
        (REPLAY, ShadowMode::Replay),
    ] {
        assert_eq!(ShadowMode::from_word(value), mode);
        assert_eq!(ShadowMode::from_word(value | high), mode);
        assert_eq!(ShadowMode::from_word(value | 0x100), mode);
    }
    // Eden compares the whole word, so this is passthrough there; replay here.
    let mut file = three_d();
    send(&mut file, state(), 5);
    send(&mut file, control(), REPLAY | 0x100);
    assert_eq!(file.shadow_mode(), Some(ShadowMode::Replay));
    assert_eq!(send(&mut file, state(), 6), []);
    assert_eq!(file.get(state()), Some(5));
}

/// The value the data-load fixtures leave in the shadow file.
const SHADOW_VALUE: u32 = 0x5a;
/// The value the data-load fixtures leave in the register file.
const REGISTER_VALUE: u32 = 0x7b;
/// The words the data-load fixtures write: distinct from each other and from both values.
const WORDS: [u32; 2] = [0x11, 0x22];

/// Every 3D data-load method, from the generated constants: `LOAD_CONSTANT_BUFFER(0..16)` and
/// `LOAD_INLINE_DATA`.
fn data_loads() -> Vec<u32> {
    let mut methods: Vec<u32> = (0..LOAD_CONSTANT_BUFFER_COUNT)
        .map(constant_buffer)
        .collect();
    methods.push(inline_data());
    methods
}

/// A 3D file in `mode` where each of `methods` holds [`REGISTER_VALUE`] and its shadow copy
/// [`SHADOW_VALUE`]: written in track mode, then again in passthrough.
fn shadowed(methods: &[u32], mode: u32) -> RegFile {
    let mut file = three_d();
    for &method in methods {
        send(&mut file, method, SHADOW_VALUE);
    }
    send(&mut file, control(), PASSTHROUGH);
    for &method in methods {
        send(&mut file, method, REGISTER_VALUE);
    }
    send(&mut file, control(), mode);
    for &method in methods {
        assert_eq!(file.get(method), Some(REGISTER_VALUE), "{method:#x}");
        assert_eq!(file.shadow(method), Some(SHADOW_VALUE), "{method:#x}");
    }
    file
}

/// What a run to the data-load method `method` sends: the run as `sent`, and for a
/// constant-buffer load the offset advanced from `offset_before` by one word per word.
fn data_load_entries(method: u32, sent: &[u32], offset_before: u32) -> Vec<RingEntry<'static>> {
    let mut entries = vec![run(method, sent)];
    if method != inline_data() {
        let words = u32::try_from(sent.len()).unwrap();
        entries.push(write(offset(), offset_before + words * WORD_BYTES));
    }
    entries
}

#[test]
fn replay_data_loads_take_the_written_words_and_state_takes_the_shadow_value() {
    // The owner's ruling of 2026-11-18 (gpu-protocol, "Engine methods: the register file",
    // step 1): in replay, data-load methods take the words written; a state method takes the
    // shadow value. Both kinds in the same file and the same mode.
    for method in data_loads() {
        let mut file = shadowed(&[method, state()], REPLAY);
        let offset_before = file.get(offset()).unwrap();
        assert_eq!(
            send_run(&mut file, method, &WORDS),
            data_load_entries(method, &WORDS, offset_before),
            "data load {method:#x}"
        );
        assert_eq!(file.get(method), Some(WORDS[1]), "{method:#x}");
        assert_eq!(file.shadow(method), Some(SHADOW_VALUE), "{method:#x}");
        // A single word likewise.
        let offset_before = file.get(offset()).unwrap();
        assert_eq!(
            send(&mut file, method, WORDS[0])[0],
            write(method, WORDS[0]),
            "single word to {method:#x}"
        );
        let advanced = if method == inline_data() {
            offset_before
        } else {
            offset_before + WORD_BYTES
        };
        assert_eq!(file.get(offset()), Some(advanced), "{method:#x}");
        // The state method in the same mode replays the shadow value (once: the second word
        // equals the register it set and is dropped).
        assert_eq!(
            send_run(&mut file, state(), &WORDS),
            [write(state(), SHADOW_VALUE)],
            "state after {method:#x}"
        );
        assert_eq!(file.get(state()), Some(SHADOW_VALUE));
        assert_eq!(file.shadow(state()), Some(SHADOW_VALUE));
    }
}

#[test]
fn replay_triggers_that_are_not_data_loads_take_the_shadow_value() {
    // Only the data loads are excepted: another trigger replays the shadow value, as
    // Eden-NX2's `CallMethod` does for every method it reaches, the inline index methods
    // among them (its `CallMultiMethod` falls through to `CallMethod` for them in replay).
    for method in [
        trigger(),
        clc797::DRAW_INLINE_INDEX.method(),
        clc797::DRAW_INLINE_INDEX2X16.method(),
        clc797::DRAW_INLINE_INDEX4X8.method(),
    ] {
        let mut file = shadowed(&[method], REPLAY);
        assert!(!file.is_state(method), "{method:#x}");
        assert_eq!(
            send_run(&mut file, method, &WORDS),
            [run(method, &[SHADOW_VALUE; 2])],
            "{method:#x}"
        );
        assert_eq!(file.get(method), Some(SHADOW_VALUE), "{method:#x}");
    }
}

#[test]
fn data_loads_end_at_their_neighbours() {
    // The first and last data-load method of each span take the written words; the methods
    // just outside them (a state method, the trigger `LAUNCH_DMA` and two methods the header
    // does not name) take the shadow value.
    let last = LOAD_CONSTANT_BUFFER_COUNT - 1;
    for method in [constant_buffer(0), constant_buffer(last), inline_data()] {
        let mut file = shadowed(&[method], REPLAY);
        let offset_before = file.get(offset()).unwrap();
        assert_eq!(
            send_run(&mut file, method, &WORDS),
            data_load_entries(method, &WORDS, offset_before),
            "{method:#x}"
        );
    }
    let neighbours = [
        constant_buffer(0) - 1,
        constant_buffer(last) + 1,
        inline_data() - 1,
        inline_data() + 1,
    ];
    assert_eq!(neighbours[0], offset());
    assert_eq!(neighbours[2], trigger());
    assert!(file_is_state(neighbours[0]));
    for method in neighbours {
        let mut file = shadowed(&[method], REPLAY);
        let expected = if file.is_state(method) {
            vec![write(method, SHADOW_VALUE)]
        } else {
            vec![run(method, &[SHADOW_VALUE; 2])]
        };
        assert_eq!(send_run(&mut file, method, &WORDS), expected, "{method:#x}");
        assert_eq!(file.get(method), Some(SHADOW_VALUE), "{method:#x}");
    }
}

#[test]
fn data_load_methods_are_the_headers_constant_buffer_and_inline_data_methods() {
    // Written out from `classes/clc797.h`: `NVC797_LOAD_CONSTANT_BUFFER(i)` is
    // `(0x2390+(i)*4)` for i in 0..16, and `NVC797_LOAD_INLINE_DATA` is 0x01b4 (byte
    // addresses; a method number is the byte address over 4).
    let mut expected: Vec<u32> = (0..16).map(|i| (0x2390 + i * 4) / 4).collect();
    expected.push(0x01b4 / 4);
    let numbers: Vec<u32> = DATA_LOADS_3D.iter().flat_map(MethodSpan::numbers).collect();
    assert_eq!(numbers, expected);
    assert_eq!(numbers, data_loads());
    assert_eq!(THREE_D.data_loads(), DATA_LOADS_3D);
    assert_eq!(COMPUTE.data_loads().len(), 0);
    assert_eq!(OTHER.data_loads().len(), 0);
    let file = three_d();
    for &method in &expected {
        assert!(THREE_D.is_data_load(method), "{method:#x}");
        assert!(!file.is_state(method), "{method:#x} is a trigger");
        assert!(!COMPUTE.is_data_load(method), "{method:#x}");
        assert!(!OTHER.is_data_load(method), "{method:#x}");
    }
    // The neighbours of each span are not data loads.
    for method in [0x238c / 4, 0x23d0 / 4, 0x01b0 / 4, 0x01b8 / 4] {
        assert!(!THREE_D.is_data_load(method), "{method:#x}");
    }
}

/// Whether `method` is a plain-state method of a new 3D file.
fn file_is_state(method: u32) -> bool {
    three_d().is_state(method)
}

#[test]
fn track_and_passthrough_treat_data_loads_and_state_alike() {
    // Unchanged by the replay rule: in track the words are sent and stored in the shadow file,
    // in passthrough they are sent and the shadow file is left alone, for both kinds.
    for (mode, shadow_after) in [
        (TRACK, Some(WORDS[1])),
        (TRACK_WITH_FILTER, Some(WORDS[1])),
        (PASSTHROUGH, Some(SHADOW_VALUE)),
    ] {
        for method in data_loads() {
            let mut file = shadowed(&[method, state()], mode);
            let offset_before = file.get(offset()).unwrap();
            assert_eq!(
                send_run(&mut file, method, &WORDS),
                data_load_entries(method, &WORDS, offset_before),
                "mode {mode} data load {method:#x}"
            );
            assert_eq!(file.get(method), Some(WORDS[1]));
            assert_eq!(file.shadow(method), shadow_after, "mode {mode} {method:#x}");
            assert_eq!(
                send_run(&mut file, state(), &WORDS),
                [write(state(), WORDS[0]), write(state(), WORDS[1])],
                "mode {mode} state"
            );
            assert_eq!(file.get(state()), Some(WORDS[1]));
            assert_eq!(file.shadow(state()), shadow_after, "mode {mode} state");
        }
    }
}

#[test]
fn compute_has_no_shadow_ram() {
    let mut file = compute();
    let method = clc7c0::SET_SHADER_SHARED_MEMORY_WINDOW_A.method();
    send(&mut file, method, 5);
    // A replay word to the compute class's 0x49 is an ordinary register value.
    send(
        &mut file,
        clc7c0::SET_MME_SHADOW_RAM_CONTROL.method(),
        REPLAY,
    );
    assert_eq!(
        send(&mut file, method, 6),
        [write_in(COMPUTE_CLASS, method, 6)]
    );
    assert_eq!(file.get(method), Some(6));
    assert_eq!(file.shadow_mode(), None);
}

// ---------------------------------------------------------------------------------------------
// Side-effect registers: one fixture per entry of each list
// ---------------------------------------------------------------------------------------------

#[test]
fn side_effect_lists_are_the_fixtures_below() {
    let names = |list: &[SideEffect]| {
        list.iter()
            .map(|effect| {
                (
                    effect.trigger.method().name,
                    effect.trigger.count(),
                    effect.rule.register(),
                )
            })
            .collect::<Vec<_>>()
    };
    // Each line has a fixture below; a new entry fails here until it gets one.
    assert_eq!(
        names(SIDE_EFFECTS_3D),
        [
            // constant_buffer_word_advances_the_offset
            ("LOAD_CONSTANT_BUFFER", LOAD_CONSTANT_BUFFER_COUNT, offset()),
            // bind_pointer_advances_and_wraps
            (
                "LOAD_MME_START_ADDRESS_RAM",
                1,
                clc797::LOAD_MME_START_ADDRESS_RAM_POINTER.method()
            ),
            // falcon04_sets_shadow_scratch_zero
            ("SET_FALCON04", 1, clc797::SET_MME_SHADOW_SCRATCH.method()),
            // root_table_offset_advances
            (
                "LOAD_ROOT_TABLE",
                1,
                clc797::SET_ROOT_TABLE_SELECTOR.method()
            ),
        ]
    );
    // compute_bind_pointer_advances
    assert_eq!(
        names(SIDE_EFFECTS_COMPUTE),
        [(
            "LOAD_MME_START_ADDRESS_RAM",
            1,
            clc7c0::LOAD_MME_START_ADDRESS_RAM_POINTER.method()
        )]
    );
    // By value, not address: each `&CONST` is its own promoted value, and whether two of them
    // share an address depends on codegen units (it failed under incremental builds on macOS).
    assert_eq!(
        *SIDE_EFFECTS_COMPUTE[0].trigger.method(),
        clc7c0::LOAD_MME_START_ADDRESS_RAM
    );
    assert_eq!(THREE_D.side_effects(), SIDE_EFFECTS_3D);
    assert_eq!(COMPUTE.side_effects(), SIDE_EFFECTS_COMPUTE);
    assert_eq!(OTHER.side_effects().len(), 0);
}

#[test]
fn constant_buffer_word_advances_the_offset() {
    let mut file = three_d();
    send(&mut file, offset(), 0x40);
    assert_eq!(
        send(&mut file, constant_buffer(0), 0x11),
        [write(constant_buffer(0), 0x11), write(offset(), 0x44)]
    );
    let last = LOAD_CONSTANT_BUFFER_COUNT - 1;
    assert_eq!(
        send(&mut file, constant_buffer(last), 0x22),
        [write(constant_buffer(last), 0x22), write(offset(), 0x48)]
    );
    // The method after the last element is no constant-buffer word.
    let after = constant_buffer(LOAD_CONSTANT_BUFFER_COUNT);
    assert_eq!(send(&mut file, after, 0x33), [write(after, 0x33)]);
    assert_eq!(file.get(offset()), Some(0x48));
    // A same-value constant-buffer word is a trigger: kept, and the offset moves again.
    assert_eq!(
        send(&mut file, constant_buffer(0), 0x11),
        [write(constant_buffer(0), 0x11), write(offset(), 0x4c)]
    );
}

#[test]
fn constant_buffer_offset_wraps_within_its_field() {
    let field = clc797::LOAD_CONSTANT_BUFFER_OFFSET_V;
    let high = 0xabcd_u32 << field.width();
    let mut file = three_d();
    send(
        &mut file,
        offset(),
        high | (field.max() - 2 * WORD_BYTES + 1),
    );
    send(&mut file, constant_buffer(3), 1);
    assert_eq!(
        file.get(offset()),
        Some(high | (field.max() - WORD_BYTES + 1))
    );
    // From the field's last word, it wraps to 0 and the bits above stay.
    assert_eq!(
        send(&mut file, constant_buffer(3), 2),
        [write(constant_buffer(3), 2), write(offset(), high)]
    );
}

#[test]
fn bind_pointer_advances_and_wraps() {
    let ram = clc797::LOAD_MME_START_ADDRESS_RAM.method();
    let pointer = clc797::LOAD_MME_START_ADDRESS_RAM_POINTER.method();
    let mut file = three_d();
    send(&mut file, pointer, 5);
    assert_eq!(
        send(&mut file, ram, 0x77),
        [write(ram, 0x77), write(pointer, 6)]
    );
    send(&mut file, pointer, u32::MAX);
    assert_eq!(
        send(&mut file, ram, 0x78),
        [write(ram, 0x78), write(pointer, 0)]
    );
}

#[test]
fn instruction_upload_leaves_its_pointer_alone() {
    // The spec's "macro upload pointer" stays in the macro host (module docs, step 2).
    let ram = clc797::LOAD_MME_INSTRUCTION_RAM.method();
    let pointer = clc797::LOAD_MME_INSTRUCTION_RAM_POINTER.method();
    let mut file = three_d();
    assert_eq!(send(&mut file, pointer, 9), [write(pointer, 9)]);
    assert_eq!(send(&mut file, ram, 0x55), [write(ram, 0x55)]);
    assert_eq!(file.get(pointer), Some(9));
    // Setting the pointer resets the upload, so a same-value write is kept.
    assert!(!file.is_state(pointer));
    assert_eq!(send(&mut file, pointer, 9), [write(pointer, 9)]);
}

#[test]
fn falcon04_sets_shadow_scratch_zero() {
    let falcon = clc797::SET_FALCON04.method();
    let scratch = clc797::SET_MME_SHADOW_SCRATCH.method();
    let mut file = three_d();
    send(&mut file, scratch, 7);
    assert_eq!(
        send(&mut file, falcon, 0x99),
        [write(falcon, 0x99), write(scratch, FALCON04_SCRATCH_VALUE)]
    );
    // Already set: the trigger is sent, the unchanged register is not.
    assert_eq!(send(&mut file, falcon, 0x99), [write(falcon, 0x99)]);
    // Other falcon methods do nothing to it.
    send(&mut file, scratch, 7);
    let falcon03 = clc797::SET_FALCON03.method();
    assert_eq!(send(&mut file, falcon03, 1), [write(falcon03, 1)]);
    assert_eq!(file.get(scratch), Some(7));
}

#[test]
fn root_table_offset_advances() {
    let load = clc797::LOAD_ROOT_TABLE.method();
    let selector = clc797::SET_ROOT_TABLE_SELECTOR.method();
    let table = clc797::SET_ROOT_TABLE_SELECTOR_ROOT_TABLE;
    let field = clc797::SET_ROOT_TABLE_SELECTOR_OFFSET;
    // Table 5, and a bit outside both fields, which must survive.
    let base = table.insert(0, 5).unwrap() | (1 << 20);
    let select = |offset: u32| field.insert(base, offset).unwrap();
    let mut file = three_d();
    send(&mut file, selector, select(8));
    assert_eq!(
        send(&mut file, load, 0x1234),
        [write(load, 0x1234), write(selector, select(12))]
    );
    // An aligned offset advances; one that is not stays (PR #4588's `Load` refuses it).
    for (from, to) in [
        (WORD_BYTES - 1, None),
        (WORD_BYTES, Some(2 * WORD_BYTES)),
        (WORD_BYTES + 1, None),
        // Even but not word-aligned: an alignment of 2 would advance these.
        (WORD_BYTES / 2, None),
        (WORD_BYTES + WORD_BYTES / 2, None),
        (field.max() + 1 - WORD_BYTES - WORD_BYTES / 2, None),
        (
            field.max() - 2 * WORD_BYTES + 1,
            Some(field.max() - WORD_BYTES + 1),
        ),
        // The last word of the table: the 8-bit field wraps to 0 (a guess; module docs).
        (field.max() - WORD_BYTES + 1, Some(0)),
    ] {
        send(&mut file, selector, select(from));
        let expected = match to {
            Some(to) => vec![write(load, 0x1234), write(selector, select(to))],
            None => vec![write(load, 0x1234)],
        };
        assert_eq!(send(&mut file, load, 0x1234), expected, "from {from}");
        assert_eq!(file.get(selector), Some(select(to.unwrap_or(from))));
    }
}

#[test]
fn compute_bind_pointer_advances() {
    let ram = clc7c0::LOAD_MME_START_ADDRESS_RAM.method();
    let pointer = clc7c0::LOAD_MME_START_ADDRESS_RAM_POINTER.method();
    let mut file = compute();
    send(&mut file, pointer, 3);
    assert_eq!(
        send(&mut file, ram, 0x42),
        [
            write_in(COMPUTE_CLASS, ram, 0x42),
            write_in(COMPUTE_CLASS, pointer, 4)
        ]
    );
    // Past Eden-NX2's 0x80 entries it keeps going.
    send(&mut file, pointer, 0x80);
    send(&mut file, ram, 0x43);
    assert_eq!(file.get(pointer), Some(0x81));
}

#[test]
fn side_effect_rules_apply_as_documented() {
    let advance = SideEffectRule::Advance {
        register: 1,
        field: crate::field::Field::new(15, 8),
        step: 4,
        align: 4,
    };
    assert_eq!(advance.register(), 1);
    assert_eq!(advance.apply(0x00ff_0800), 0x00ff_0c00);
    assert_eq!(advance.apply(0x00ff_0900), 0x00ff_0900);
    assert_eq!(advance.apply(0x00ff_fc00), 0x00ff_0000);
    // An alignment of 0 never advances (no division by zero).
    let never = SideEffectRule::Advance {
        register: 1,
        field: crate::field::Field::new(31, 0),
        step: 1,
        align: 0,
    };
    assert_eq!(never.apply(8), 8);
    let set = SideEffectRule::Set {
        register: 2,
        value: 9,
    };
    assert_eq!(set.register(), 2);
    assert_eq!(set.apply(3), 9);
}

// ---------------------------------------------------------------------------------------------
// Same-value dropping
// ---------------------------------------------------------------------------------------------

#[test]
fn same_value_state_write_is_dropped() {
    let mut file = three_d();
    // A fresh register holds 0: writing 0 changes nothing.
    assert_eq!(send(&mut file, state(), 0), []);
    assert_eq!(send(&mut file, state(), 0x100), [write(state(), 0x100)]);
    assert_eq!(send(&mut file, state(), 0x100), []);
    // Values that differ only above the low byte are different values.
    assert_eq!(
        send(&mut file, state(), 0x1_0100),
        [write(state(), 0x1_0100)]
    );
    assert_eq!(send(&mut file, state(), 0x100), [write(state(), 0x100)]);
    // Another register holding the same value is no reason to drop.
    assert_eq!(
        send(&mut file, other_state(), 0x100),
        [write(other_state(), 0x100)]
    );
}

#[test]
fn same_value_trigger_write_is_kept() {
    let mut file = three_d();
    assert!(!file.is_state(trigger()));
    assert_eq!(send(&mut file, trigger(), 5), [write(trigger(), 5)]);
    assert_eq!(send(&mut file, trigger(), 5), [write(trigger(), 5)]);
    let begin = clc797::BEGIN.method();
    assert_eq!(send(&mut file, begin, 0), [write(begin, 0)]);
}

#[test]
fn unknown_method_is_a_trigger() {
    // Byte 0x680 is no method of `clc797.h` (the gap after SET_VERTEX_STREAM_SIZE_A/B).
    let unknown = 0x680 / WORD_BYTES;
    assert!(
        clc797::HEADER
            .methods
            .iter()
            .all(|method| method.method() != unknown)
    );
    let mut file = three_d();
    assert!(!file.is_state(unknown));
    assert_eq!(send(&mut file, unknown, 0), [write(unknown, 0)]);
    assert_eq!(send(&mut file, unknown, 0), [write(unknown, 0)]);
}

#[test]
fn state_array_ends_at_its_length() {
    let span = STATE_3D
        .iter()
        // By value, not address (see `side_effect_lists_are_the_fixtures_below`).
        .find(|span| *span.method() == clc797::SET_VERTEX_STREAM_SIZE_A)
        .unwrap();
    let last = clc797::SET_VERTEX_STREAM_SIZE_A
        .method_at(&[span.count() - 1])
        .unwrap();
    let past = clc797::SET_VERTEX_STREAM_SIZE_A
        .method_at(&[span.count()])
        .unwrap();
    let mut file = three_d();
    assert!(file.is_state(last));
    assert!(!file.is_state(past));
    assert_eq!(send(&mut file, last, 0), []);
    assert_eq!(send(&mut file, past, 0), [write(past, 0)]);
}

#[test]
fn drop_switch_off_passes_everything() {
    let mut file = three_d();
    file.set_drop_same_values(false);
    assert!(!file.drop_same_values());
    assert_eq!(send(&mut file, state(), 0), [write(state(), 0)]);
    assert_eq!(send(&mut file, state(), 0), [write(state(), 0)]);
    file.set_drop_same_values(true);
    assert_eq!(send(&mut file, state(), 0), []);
}

#[test]
fn same_value_write_is_dropped_even_when_it_changes_the_shadow_file() {
    let mut file = three_d();
    send(&mut file, control(), PASSTHROUGH);
    send(&mut file, state(), 5);
    assert_eq!(file.shadow(state()), Some(0));
    send(&mut file, control(), TRACK);
    assert_eq!(send(&mut file, state(), 5), []);
    assert_eq!(file.shadow(state()), Some(5));
}

#[test]
fn replayed_value_equal_to_the_register_is_dropped() {
    let mut file = three_d();
    send(&mut file, state(), 5);
    send(&mut file, control(), REPLAY);
    assert_eq!(send(&mut file, state(), 9), []);
    assert_eq!(file.get(state()), Some(5));
}

#[test]
fn compute_and_other_classes_drop_nothing() {
    assert_eq!(COMPUTE.state().len(), 0);
    assert_eq!(OTHER.state().len(), 0);
    assert_eq!(THREE_D.state(), STATE_3D);
    let mut compute = compute();
    let mut copy = RegFile::new(CHANNEL, ClassId(clc7b5::CLASS));
    for method in 0..COMPUTE.registers() {
        assert!(!compute.is_state(method), "{method:#x}");
    }
    for method in 0..OTHER.registers() {
        assert!(!copy.is_state(method), "{method:#x}");
    }
    let method = clc7c0::SET_SHADER_SHARED_MEMORY_WINDOW_A.method();
    assert_eq!(
        send(&mut compute, method, 0),
        [write_in(COMPUTE_CLASS, method, 0)]
    );
    assert_eq!(
        send(&mut copy, 0x100, 0),
        [write_in(ClassId(clc7b5::CLASS), 0x100, 0)]
    );
}

// ---------------------------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------------------------

#[test]
fn constant_buffer_run_is_one_run_and_one_offset_write() {
    let mut file = three_d();
    send(&mut file, offset(), 0x10);
    assert_eq!(
        send_run(&mut file, constant_buffer(2), &[0xa, 0xb, 0xc]),
        [
            run(constant_buffer(2), &[0xa, 0xb, 0xc]),
            write(offset(), 0x10 + 3 * WORD_BYTES)
        ]
    );
    assert_eq!(file.get(constant_buffer(2)), Some(0xc));
    assert_eq!(file.shadow(constant_buffer(2)), Some(0xc));
}

#[test]
fn single_word_run_is_a_write() {
    let mut file = three_d();
    assert_eq!(send_run(&mut file, trigger(), &[4]), [write(trigger(), 4)]);
}

#[test]
fn empty_run_sends_and_changes_nothing() {
    let mut file = three_d();
    let before = snapshot(&file);
    for method in [trigger(), state(), constant_buffer(0), control()] {
        assert_eq!(send_run(&mut file, method, &[]), []);
    }
    assert_eq!(snapshot(&file), before);
}

#[test]
fn state_run_is_one_write_per_changed_word() {
    let mut file = three_d();
    assert_eq!(
        send_run(&mut file, state(), &[5, 5, 6, 6, 5]),
        [write(state(), 5), write(state(), 6), write(state(), 5)]
    );
}

#[test]
fn long_run_splits_at_the_entry_limit() {
    let words: Vec<u32> = (1..).take(MAX_RUN_WORDS + 1).collect();
    let limit = u32::try_from(MAX_RUN_WORDS).unwrap();
    // Exactly the limit: one entry.
    let mut file = three_d();
    assert_eq!(
        send_run(&mut file, constant_buffer(1), &words[..MAX_RUN_WORDS]),
        [
            run(constant_buffer(1), &words[..MAX_RUN_WORDS]),
            write(offset(), limit * WORD_BYTES)
        ]
    );
    // One past: a second entry (a write) with its own offset.
    let mut file = three_d();
    assert_eq!(
        send_run(&mut file, constant_buffer(1), &words),
        [
            run(constant_buffer(1), &words[..MAX_RUN_WORDS]),
            write(offset(), limit * WORD_BYTES),
            write(constant_buffer(1), words[MAX_RUN_WORDS]),
            write(offset(), (limit + 1) * WORD_BYTES),
        ]
    );
}

#[test]
fn each_chunk_reports_the_offset_it_ends_on() {
    let words: Vec<u32> = (1..).take(2 * MAX_RUN_WORDS).collect();
    let limit = u32::try_from(MAX_RUN_WORDS).unwrap();
    let field = clc797::LOAD_CONSTANT_BUFFER_OFFSET_V;
    // Two full chunks advance the offset by exactly its field's range, so the second chunk ends
    // on the offset the run started from: its `Write` shows only if each chunk compares with
    // its own starting value.
    assert_eq!(2 * limit * WORD_BYTES, field.max() + 1);
    let start = 0x10;
    let mut file = three_d();
    send(&mut file, offset(), start);
    assert_eq!(
        send_run(&mut file, constant_buffer(1), &words),
        [
            run(constant_buffer(1), &words[..MAX_RUN_WORDS]),
            write(offset(), start + limit * WORD_BYTES),
            run(constant_buffer(1), &words[MAX_RUN_WORDS..]),
            write(offset(), start),
        ]
    );
    assert_eq!(file.get(offset()), Some(start));
}

#[test]
fn run_in_replay_sends_the_written_data_words() {
    // A constant-buffer run in replay: the words as written (not the shadow copy, 2), and the
    // offset advanced by all four words.
    let mut file = three_d();
    send_run(&mut file, constant_buffer(0), &[1, 2]);
    send(&mut file, control(), REPLAY);
    assert_eq!(
        send_run(&mut file, constant_buffer(0), &[7, 8]),
        [
            run(constant_buffer(0), &[7, 8]),
            write(offset(), 4 * WORD_BYTES)
        ]
    );
    assert_eq!(file.get(constant_buffer(0)), Some(8));
    assert_eq!(file.shadow(constant_buffer(0)), Some(2));
}

#[test]
fn entries_keep_the_order_of_the_words() {
    let mut file = three_d();
    let mut out = Vec::new();
    file.write(state(), 1, &mut out).unwrap();
    file.write_run(constant_buffer(0), &[2, 3], &mut out)
        .unwrap();
    file.write(other_state(), 4, &mut out).unwrap();
    file.write(trigger(), 5, &mut out).unwrap();
    assert_eq!(
        out,
        [
            write(state(), 1),
            run(constant_buffer(0), &[2, 3]),
            write(offset(), 2 * WORD_BYTES),
            write(other_state(), 4),
            write(trigger(), 5),
        ]
    );
}

/// One write of the generated sequences: a method from a small set and its words.
fn any_write() -> impl Strategy<Value = (u32, Vec<u32>)> {
    let methods = vec![
        state(),
        other_state(),
        trigger(),
        inline_data(),
        constant_buffer(0),
        constant_buffer(5),
        constant_buffer(LOAD_CONSTANT_BUFFER_COUNT - 1),
        offset(),
        control(),
        clc797::LOAD_MME_START_ADDRESS_RAM.method(),
        clc797::LOAD_MME_START_ADDRESS_RAM_POINTER.method(),
        clc797::LOAD_ROOT_TABLE.method(),
        clc797::SET_ROOT_TABLE_SELECTOR.method(),
        clc797::SET_FALCON04.method(),
    ];
    // Small values, so modes and same values come up often.
    (
        proptest::sample::select(methods),
        proptest::collection::vec(0u32..8, 0..6),
    )
}

/// The entries with runs written out as writes, and the side-effect registers' writes left out
/// (a run sends their final values; single writes send each step).
fn flattened(entries: &[RingEntry<'static>]) -> Vec<(u16, u32)> {
    let side_effect_registers: Vec<u32> = SIDE_EFFECTS_3D
        .iter()
        .map(|effect| effect.rule.register())
        .collect();
    let mut flat = Vec::new();
    for entry in entries {
        match entry {
            RingEntry::Write { method, value, .. } => flat.push((*method, *value)),
            RingEntry::Run { method, words, .. } => {
                flat.extend(words.iter().map(|&word| (*method, word)));
            }
            other => unreachable!("a register file sent {other:?}"),
        }
    }
    flat.retain(|(method, _)| !side_effect_registers.contains(&u32::from(*method)));
    flat
}

proptest! {
    #![proptest_config(Config {
        cases: 256,
        rng_seed: RngSeed::Fixed(0x5245_4753),
        failure_persistence: None,
        ..Config::default()
    })]

    #[test]
    fn run_equals_single_writes(writes in proptest::collection::vec(any_write(), 0..24)) {
        let mut runs = three_d();
        let mut singles = three_d();
        let mut run_entries = Vec::new();
        let mut single_entries = Vec::new();
        for (method, words) in &writes {
            runs.write_run(*method, words, &mut run_entries).unwrap();
            for &word in words {
                singles.write(*method, word, &mut single_entries).unwrap();
            }
        }
        prop_assert_eq!(snapshot(&runs), snapshot(&singles));
        prop_assert_eq!(flattened(&run_entries), flattened(&single_entries));
    }
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

#[test]
fn method_outside_the_file_is_refused() {
    for (class, rules) in [
        (THREE_D_CLASS, &THREE_D),
        (COMPUTE_CLASS, &COMPUTE),
        (ClassId(clc7b5::CLASS), &OTHER),
    ] {
        let mut file = RegFile::new(CHANNEL, class);
        let limit = rules.registers();
        let last = limit - 1;
        let mut out = Vec::new();
        assert_eq!(file.write(last, 1, &mut out), Ok(()));
        assert_eq!(out.len(), 1);
        let before = snapshot(&file);
        for method in [limit, limit + 1, u32::MAX] {
            let mut out = Vec::new();
            let refused = Err(RegsError::OutOfRange { method, limit });
            assert_eq!(file.write(method, 1, &mut out), refused);
            assert_eq!(file.write_run(method, &[1, 2], &mut out), refused);
            assert_eq!(file.write_run(method, &[], &mut out), refused);
            assert_eq!(out, []);
            assert_eq!(snapshot(&file), before);
        }
    }
}

#[test]
fn range_error_message() {
    assert_eq!(
        RegsError::OutOfRange {
            method: 0xe00,
            limit: 0xe00
        }
        .to_string(),
        "method 0xe00 is outside this class's register file, which holds methods below 0xe00"
    );
    assert_eq!(
        RegsError::OutOfRange {
            method: 0,
            limit: 0
        }
        .to_string(),
        "method 0x0 is outside this class's register file, which holds methods below 0x0"
    );
}

// ---------------------------------------------------------------------------------------------
// The hand-written tables
// ---------------------------------------------------------------------------------------------

#[test]
fn every_state_entry_names_a_method_of_the_generated_table() {
    for span in STATE_3D {
        assert!(
            clc797::HEADER
                .methods
                .iter()
                .any(|method| method == span.method()),
            "{} is not a method of clc797.h",
            span.method().name
        );
    }
}

#[test]
fn state_entries_have_the_shape_of_their_method() {
    for span in STATE_3D {
        let method = span.method();
        assert!(
            method.strides.len() <= 1,
            "{} has more than one index",
            method.name
        );
        if method.strides.is_empty() {
            assert_eq!(span.count(), 1, "{}", method.name);
        } else {
            assert!(span.count() >= 1, "{}", method.name);
        }
        assert_eq!(
            span.numbers().count(),
            usize::try_from(span.count()).unwrap(),
            "{}",
            method.name
        );
    }
}

#[test]
fn state_methods_are_unique_and_engine_methods_in_the_file() {
    // The spec's facts table: methods below 0x40 are host methods.
    const HOST_METHODS: u32 = 0x40;
    let mut seen = BTreeSet::new();
    for span in STATE_3D {
        for number in span.numbers() {
            assert!(seen.insert(number), "{number:#x} listed twice");
            assert!(
                (HOST_METHODS..THREE_D.registers()).contains(&number),
                "{number:#x}"
            );
        }
    }
    let total: u32 = STATE_3D.iter().map(MethodSpan::count).sum();
    assert_eq!(seen.len(), usize::try_from(total).unwrap());
    // And the file marks exactly these.
    let file = three_d();
    for method in 0..THREE_D.registers() {
        assert_eq!(file.is_state(method), seen.contains(&method), "{method:#x}");
    }
}

/// Mesa's `METHOD_ARRAY_SIZES` (`mesa-nouveau/src/nouveau/headers/class_parser.py`), as written
/// there: a trailing `*` matches any rest of the name, and the first match wins
/// (`method_array_size`, `glob_match`).
const MESA_ARRAY_SIZES: &[(&str, u32)] = &[
    ("BIND_GROUP_CONSTANT_BUFFER", 16),
    ("CALL_MME_DATA", 256),
    ("CALL_MME_MACRO", 256),
    ("LOAD_CONSTANT_BUFFER", 16),
    ("LOAD_INLINE_QMD_DATA", 64),
    ("SET_ANTI_ALIAS_SAMPLE_POSITIONS", 4),
    ("SET_BLEND", 8),
    ("SET_BLEND_PER_TARGET_*", 8),
    ("SET_COLOR_TARGET_*", 8),
    ("SET_COLOR_COMPRESSION", 8),
    ("SET_COLOR_CLEAR_VALUE", 4),
    ("SET_CT_WRITE", 8),
    ("SET_MME_SHADOW_SCRATCH", 256),
    ("SET_MULTI_VIEW_RENDER_TARGET_ARRAY_INDEX_OFFSET", 4),
    ("SET_PIPELINE_*", 6),
    ("SET_ROOT_TABLE_VISIBILITY", 8),
    ("SET_SCG_COMPUTE_SCHEDULING_PARAMETERS", 16),
    ("SET_SCG_GRAPHICS_SCHEDULING_PARAMETERS", 16),
    ("SET_SCISSOR_*", 16),
    ("SET_SHADER_PERFORMANCE_SNAPSHOT_COUNTER_VALUE*", 8),
    ("SET_SHADER_PERFORMANCE_COUNTER_VALUE*", 8),
    ("SET_SHADER_PERFORMANCE_COUNTER_EVENT", 8),
    ("SET_SHADER_PERFORMANCE_COUNTER_CONTROL_A", 8),
    ("SET_SHADER_PERFORMANCE_COUNTER_CONTROL_B", 8),
    ("SET_SHADING_RATE_INDEX_SURFACE_*", 1),
    ("SET_SPARE_MULTI_VIEW_RENDER_TARGET_ARRAY_INDEX_OFFSET", 4),
    ("SET_STREAM_OUT_BUFFER_*", 4),
    ("SET_STREAM_OUT_CONTROL_*", 4),
    ("SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER", 13),
    ("SET_VARIABLE_PIXEL_RATE_SHADING_CONTROL*", 16),
    ("SET_VARIABLE_PIXEL_RATE_SHADING_INDEX_TO_RATE*", 16),
    ("SET_VIEWPORT_*", 16),
    ("SET_VERTEX_ATTRIBUTE_*", 16),
    ("SET_VERTEX_STREAM_*", 16),
    ("SET_WINDOW_CLIP_*", 8),
    ("SET_CLIP_ID_EXTENT_*", 4),
];

fn mesa_array_size(name: &str) -> Option<u32> {
    MESA_ARRAY_SIZES.iter().find_map(|&(glob, size)| {
        let matches = match glob.strip_suffix('*') {
            Some(prefix) => name.starts_with(prefix),
            None => name == glob,
        };
        matches.then_some(size)
    })
}

/// The rule in the module docs of `state_3d`, applied to every method of `clc797.h`, gives
/// exactly the table, with Mesa's lengths.
#[test]
fn state_table_is_the_documented_rule() {
    const PLAIN: &[&str] = &[
        "LOAD_CONSTANT_BUFFER_OFFSET",
        "LOAD_MME_START_ADDRESS_RAM_POINTER",
        "LINE_LENGTH_IN",
        "LINE_COUNT",
        "OFFSET_OUT_UPPER",
        "OFFSET_OUT",
        "PITCH_OUT",
    ];
    const EXCLUDED: &[&str] = &[
        "SET_MME_SHADOW_RAM_CONTROL",
        "SET_VERTEX_ARRAY_START",
        "SET_INDEX_BUFFER_F",
        "SET_PRIMITIVE_TOPOLOGY",
        "SET_REPORT_SEMAPHORE_D",
        "SET_RENDER_ENABLE_C",
        "SET_ROOT_TABLE_SELECTOR",
        "SET_OBJECT",
        "SET_INSTRUMENTATION_METHOD_HEADER",
        "SET_INSTRUMENTATION_METHOD_DATA",
        "SET_MME_SWITCH_STATE",
    ];
    const EXCLUDED_PREFIXES: &[&str] = &[
        "SET_FALCON",
        "SET_RESERVED_SW_METHOD",
        "SET_SPARE0",
        "SET_SPARE_NOOP",
        "SET_I2M_SPARE_NOOP",
    ];
    let mut expected = Vec::new();
    for method in clc797::HEADER.methods {
        let name = method.name;
        let candidate = name.starts_with("SET_") || PLAIN.contains(&name);
        let excluded = EXCLUDED.contains(&name)
            || EXCLUDED_PREFIXES
                .iter()
                .any(|prefix| name.starts_with(prefix));
        if !candidate || excluded {
            continue;
        }
        let count = match method.strides {
            [] => Some(1),
            [_] => mesa_array_size(name),
            _ => None,
        };
        if let Some(count) = count {
            expected.push((name, count));
        }
    }
    let table: Vec<_> = STATE_3D
        .iter()
        .map(|span| (span.method().name, span.count()))
        .collect();
    assert_eq!(table, expected);
}

#[test]
fn no_state_array_runs_into_another_method() {
    let firsts: BTreeSet<u32> = clc797::HEADER.methods.iter().map(Method::method).collect();
    for span in STATE_3D {
        for number in span.numbers().skip(1) {
            assert!(
                !firsts.contains(&number),
                "{} element at {number:#x} is another method",
                span.method().name
            );
        }
    }
}

#[test]
fn no_state_method_is_a_trigger_the_references_act_on() {
    // Eden-NX2's `IsMethodExecutable` (`eden-nx2/src/video_core/engines/maxwell_3d.cpp`), as byte
    // addresses from the `ASSERT_REG_POSITION`s and structures of `maxwell_3d.h`, and PR #4588's
    // two root-table methods (`native_root_table.h`).
    let mut bytes: Vec<u32> = vec![
        0x0110, // wait_for_idle
        0x0114, // load_mme.instruction_ptr
        0x0118, // load_mme.instruction
        0x0120, // load_mme.start_address
        0x0124, // shadow_ram_control
        0x01b0, // launch_dma
        0x01b4, // inline_data
        0x02c8, // sync_info
        0x0d74, // vertex_buffer.first
        0x0d78, // vertex_buffer.count
        0x0de0, // fragment_barrier
        0x0f74, // invalidate_texture_data_cache
        0x0f7c, // tiled_cache_barrier
        0x10ac, // draw_texture.src_y0
        0x1214, // vertex_array_instance_first
        0x1218, // vertex_array_instance_subsequent
        0x1304, // inline_index_4x8.index0
        0x1530, // clear_report_value
        0x1558, // render_enable.mode
        0x15e8, // draw_inline_index
        0x15f0, // inline_index_2x16.even
        0x1614, // draw.end
        0x1618, // draw.begin
        0x17dc, // index_buffer.first
        0x17e0, // index_buffer.count
        0x17e4, // index_buffer32_first
        0x17e8, // index_buffer16_first
        0x17ec, // index_buffer8_first
        0x17f0, // index_buffer32_subsequent
        0x17f4, // index_buffer16_subsequent
        0x17f8, // index_buffer8_subsequent
        0x1970, // topology_override
        0x19d0, // clear_surface
        0x1b0c, // report_semaphore.query
        0x2310, // falcon[4]
        0x0504, // NativeRootTables::SelectorMethod
        0x0508, // NativeRootTables::LoadMethod
    ];
    bytes.extend((0..16).map(|i| 0x2390 + 4 * i)); // const_buffer.buffer[i]
    bytes.extend((0..5).map(|i| 0x2410 + 0x20 * i)); // bind_groups[i].raw_config
    let file = three_d();
    for byte in bytes {
        assert!(!file.is_state(byte / 4), "byte {byte:#x}");
    }
    // Nor any trigger the side effects or the shadow RAM key on.
    for effect in SIDE_EFFECTS_3D {
        for number in effect.trigger.numbers() {
            assert!(!file.is_state(number), "{number:#x}");
        }
    }
    assert!(!file.is_state(control()));
}

#[test]
fn method_spans_contain_exactly_their_elements() {
    let array = MethodSpan::array(&clc797::LOAD_CONSTANT_BUFFER, 3);
    let first = clc797::LOAD_CONSTANT_BUFFER.method();
    assert_eq!(
        array.numbers().collect::<Vec<_>>(),
        [first, first + 1, first + 2]
    );
    assert!(!array.contains(first - 1));
    assert!(array.contains(first + 2));
    assert!(!array.contains(first + 3));
    assert_eq!(array.element(3), None);
    // A stride of several methods: the methods in between are not elements.
    let strided = MethodSpan::array(&clc797::SET_VERTEX_STREAM_SIZE_A, 2);
    let base = clc797::SET_VERTEX_STREAM_SIZE_A.method();
    let stride = clc797::SET_VERTEX_STREAM_SIZE_A.strides[0] / WORD_BYTES;
    assert!(strided.contains(base));
    assert!(!strided.contains(base + 1));
    assert!(strided.contains(base + stride));
    assert!(!strided.contains(base + 2 * stride));
    let one = MethodSpan::one(&clc797::SET_ALPHA_REF);
    assert!(one.contains(state()));
    assert!(!one.contains(state() + 1));
    assert_eq!(one.element(1), None);
    // A method that is not an array has no element past 0, whatever its count says.
    let scalar = MethodSpan::array(&clc797::SET_ALPHA_REF, 2);
    assert_eq!(scalar.numbers().collect::<Vec<_>>(), [state()]);
    assert!(!scalar.contains(state() + 1));
    // A count of zero: no element, for a method that is not an array and for an array.
    let empty_scalar = MethodSpan::array(&clc797::SET_ALPHA_REF, 0);
    assert_eq!(empty_scalar.numbers().count(), 0);
    assert!(!empty_scalar.contains(state()));
    let empty_array = MethodSpan::array(&clc797::LOAD_CONSTANT_BUFFER, 0);
    assert_eq!(empty_array.numbers().count(), 0);
    assert!(!empty_array.contains(first));
    // Two indices: no element.
    let two = MethodSpan::array(&clc797::SET_STREAM_OUT_LAYOUT_SELECT, 2);
    assert_eq!(two.numbers().count(), 0);
    assert!(!two.contains(clc797::SET_STREAM_OUT_LAYOUT_SELECT.method()));
}

/// The method numbers and bit fields this module relies on, against the references' own
/// numbers written out (tests that import the shipping constants cannot catch a wrong one).
#[test]
fn reference_values() {
    // gpu-protocol spec, facts table: `SET_MME_SHADOW_RAM_CONTROL` is method 0x49 (byte 0x124);
    // macros from 0xE00; root tables loaded with 0x141 and 0x142.
    assert_eq!(control(), 0x49);
    assert_eq!(THREE_D.registers(), 0xe00);
    assert_eq!(THREE_D.shadow_control(), Some(0x49));
    assert_eq!(clc797::SET_ROOT_TABLE_SELECTOR.method(), 0x141);
    assert_eq!(clc797::LOAD_ROOT_TABLE.method(), 0x142);
    // Eden-NX2 `maxwell_3d.h`: load_mme at 0x114 (instruction_ptr, instruction,
    // start_address_ptr, start_address), falcon at 0x2300, const_buffer at 0x2380 (size,
    // address_high, address_low, offset, buffer[16]), shadow_scratch at 0x3400.
    assert_eq!(clc797::LOAD_MME_INSTRUCTION_RAM_POINTER.method(), 0x114 / 4);
    assert_eq!(clc797::LOAD_MME_INSTRUCTION_RAM.method(), 0x118 / 4);
    assert_eq!(
        clc797::LOAD_MME_START_ADDRESS_RAM_POINTER.method(),
        0x11c / 4
    );
    assert_eq!(clc797::LOAD_MME_START_ADDRESS_RAM.method(), 0x120 / 4);
    assert_eq!(clc797::SET_FALCON04.method(), (0x2300 + 4 * 4) / 4);
    assert_eq!(offset(), (0x2380 + 3 * 4) / 4);
    assert_eq!(constant_buffer(0), (0x2380 + 4 * 4) / 4);
    assert_eq!(clc797::SET_MME_SHADOW_SCRATCH.method(), 0xd00);
    assert_eq!(LOAD_CONSTANT_BUFFER_COUNT, 16);
    assert_eq!(FALCON04_SCRATCH_VALUE, 1);
    assert_eq!(WORD_BYTES, 4);
    // PR #4588 `native_root_table.h`: `Select` takes the table from bits 2:0 and the offset
    // from bits 15:8 (`value & 7`, `(value >> 8) & 255`).
    let selector = clc797::SET_ROOT_TABLE_SELECTOR_OFFSET;
    assert_eq!((selector.low(), selector.max()), (8, 255));
    let table = clc797::SET_ROOT_TABLE_SELECTOR_ROOT_TABLE;
    assert_eq!((table.low(), table.max()), (0, 7));
    // `clc797.h`: LOAD_CONSTANT_BUFFER_OFFSET_V 15:0; SET_MME_SHADOW_RAM_CONTROL_MODE 1:0 with
    // METHOD_TRACK 0, METHOD_TRACK_WITH_FILTER 1, METHOD_PASSTHROUGH 2, METHOD_REPLAY 3.
    let offset_field = clc797::LOAD_CONSTANT_BUFFER_OFFSET_V;
    assert_eq!((offset_field.low(), offset_field.max()), (0, 0xffff));
    let mode = clc797::SET_MME_SHADOW_RAM_CONTROL_MODE;
    assert_eq!((mode.low(), mode.max()), (0, 3));
    assert_eq!(
        (TRACK, TRACK_WITH_FILTER, PASSTHROUGH, REPLAY),
        (0, 1, 2, 3)
    );
    // `clc7c0.h` and Eden-NX2 `kepler_compute.cpp` (methods 0x47 and 0x48, macros from 0xe00).
    assert_eq!(clc7c0::LOAD_MME_START_ADDRESS_RAM_POINTER.method(), 0x47);
    assert_eq!(clc7c0::LOAD_MME_START_ADDRESS_RAM.method(), 0x48);
    assert_eq!(COMPUTE.registers(), 0xe00);
    assert_eq!(COMPUTE.shadow_control(), None);
    // `clc76f.h`: NVC76F_DMA_METHOD_ADDRESS is 11:0.
    assert_eq!(OTHER.registers(), 0x1000);
    assert_eq!(OTHER.shadow_control(), None);
}

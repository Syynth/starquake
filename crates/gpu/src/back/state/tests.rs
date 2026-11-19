//! Tests of the back's mirrors: the group tables (each method marks exactly its groups), the
//! mirror's writes, runs and run positions, clearing, the constant buffer bindings, and the typed
//! views' bounds, enum checks and wiring.
//!
//! One mutation survives and is equivalent: `Mirror::run` skips the bind lookup for a mirror
//! with no bindings (every class but 3D); removing that check changes nothing observable, since
//! such a mirror has no binding to record into.

use std::borrow::Cow;
use std::collections::BTreeMap;

use super::*;
use crate::classes::{ClassHeader, Method, Value, cl902d, clc7b5, clc7c0, clc797};
use crate::field::Field;
use crate::interface::ChannelId;
use crate::regs::{ClassRules, LOAD_CONSTANT_BUFFER_COUNT, RegFile, STATE_3D, WORD_BYTES};
use crate::ring::{ClassId, MAX_RUN_WORDS, RingEntry};

/// Not channel 0, so a hard-coded channel fails.
const CHANNEL: ChannelId = ChannelId(7);
const THREE_D: ClassId = ClassId(clc797::CLASS);
const COMPUTE: ClassId = ClassId(clc7c0::CLASS);
const DMA: ClassId = ClassId(clc7b5::CLASS);

fn three_d() -> Mirror {
    Mirror::new(CHANNEL, THREE_D)
}

fn compute() -> Mirror {
    Mirror::new(CHANNEL, COMPUTE)
}

fn group(kind: GroupKind, index: u32) -> Group {
    Group::new(kind, index).unwrap()
}

fn one(kind: GroupKind) -> Group {
    Group::one(kind)
}

fn set(groups: &[Group]) -> GroupSet {
    groups.iter().copied().collect()
}

/// Element `index` of an array method (or the method itself with no index).
fn number(method: &Method, indices: &[u32]) -> u32 {
    method.method_at(indices).unwrap()
}

/// Writes `value` to `method` at `indices` in `mirror`.
fn put(mirror: &mut Mirror, method: &Method, indices: &[u32], value: u32) {
    mirror.write(number(method, indices), value).unwrap();
}

/// A fresh mirror of `class` with nothing dirty.
fn clean(class: ClassId) -> Mirror {
    let mut mirror = Mirror::new(CHANNEL, class);
    mirror.take_dirty();
    mirror
}

// ---------------------------------------------------------------------------------------------
// Groups.

#[test]
fn array_lengths_are_the_cited_values() {
    // Mesa's `METHOD_ARRAY_SIZES` (`mesa-nouveau/src/nouveau/headers/class_parser.py`) and the
    // header's `NVC797_BIND_GROUP_CONSTANT_BUFFER_SHADER_SLOT 8:4`, written out.
    assert_eq!(VIEWPORTS, 16);
    assert_eq!(SCISSORS, 16);
    assert_eq!(BLEND_TARGETS, 8);
    assert_eq!(COLOR_TARGETS, 8);
    assert_eq!(VERTEX_STREAMS_PER_HALF, 16);
    assert_eq!(VERTEX_STREAMS, 32);
    assert_eq!(VERTEX_STREAM_SIZES, 16);
    assert_eq!(VERTEX_ATTRIBUTES_PER_HALF, 16);
    assert_eq!(VERTEX_ATTRIBUTES, 32);
    assert_eq!(BIND_GROUPS, 16);
    assert_eq!(CONSTANT_BUFFER_SLOTS, 32);
    assert_eq!(PIPELINES, 6);
}

#[test]
fn array_lengths_agree_with_the_front_state_table() {
    // The front's state table takes its lengths from Mesa too: every method both tables list
    // has the same length in both.
    let mut checked = 0;
    for entry in GROUPS_3D {
        let method = entry.span.method();
        for span in STATE_3D.iter().filter(|span| span.method() == method) {
            assert_eq!(span.count(), entry.span.count(), "{}", method.name);
            checked += 1;
        }
    }
    assert!(checked > 50, "only {checked} spans compared");
}

#[test]
fn every_group_has_its_own_bit() {
    let mut all = GroupSet::EMPTY;
    let mut listed = Vec::new();
    for kind in GroupKind::ALL {
        for index in 0..kind.count() {
            let group = group(kind, index);
            assert!(!all.contains(group), "{group:?} shares a bit");
            all = all.with(group);
            listed.push(group);
        }
    }
    assert_eq!(all.len(), GROUP_COUNT);
    // The set iterates in bit order: the order of `GroupKind::ALL`, then by index.
    assert_eq!(all.iter().collect::<Vec<_>>(), listed);
}

#[test]
fn group_count_is_the_sum_of_the_kinds() {
    // 10 single groups, plus 8 blend targets, 8 colour targets, 32 vertex streams, 16 bind
    // groups and 6 pipelines.
    assert_eq!(GROUP_COUNT, 10 + 8 + 8 + 32 + 16 + 6);
}

#[test]
fn a_group_index_is_bounded_by_its_kind_on_both_sides() {
    for kind in GroupKind::ALL {
        let last = kind.count() - 1;
        let group = Group::new(kind, last).unwrap();
        assert_eq!((group.kind(), group.index()), (kind, last));
        assert_eq!(Group::new(kind, kind.count()), None, "{kind:?}");
    }
    assert_eq!(Group::one(GroupKind::Zt).index(), 0);
}

#[test]
fn set_operations() {
    let a = group(GroupKind::ColorTarget, 3);
    let b = group(GroupKind::VertexStream, 31);
    let c = one(GroupKind::DepthStencil);
    let ab = set(&[a, b]);
    assert!(ab.contains(a) && ab.contains(b) && !ab.contains(c));
    assert_eq!(ab.len(), 2);
    assert!(!ab.is_empty());
    assert!(GroupSet::EMPTY.is_empty());
    assert_eq!(ab.union(set(&[c])), set(&[a, b, c]));
    assert_eq!(set(&[a, b, c]).difference(set(&[b])), set(&[a, c]));
    // Neighbouring indices of one kind are distinct groups.
    assert!(!ab.contains(group(GroupKind::ColorTarget, 2)));
    assert!(!ab.contains(group(GroupKind::ColorTarget, 4)));
}

// ---------------------------------------------------------------------------------------------
// The group tables.

/// The tables and the header and register count of their class.
fn tables() -> [(ClassId, &'static [GroupEntry], ClassHeader); 2] {
    [
        (THREE_D, GROUPS_3D, clc797::HEADER),
        (COMPUTE, GROUPS_COMPUTE, clc7c0::HEADER),
    ]
}

#[test]
fn every_table_element_is_a_register_of_its_class_and_names_a_group() {
    for (class, table, header) in tables() {
        let registers = ClassRules::for_class(class).registers();
        for entry in table {
            let method = entry.span.method();
            assert!(
                header.methods.contains(method),
                "{} is not a method of {}",
                method.name,
                header.header
            );
            let numbers: Vec<u32> = entry.span.numbers().collect();
            assert_eq!(numbers.len(), usize::try_from(entry.span.count()).unwrap());
            for (index, number) in (0..).zip(numbers) {
                assert!(number < registers, "{} element {index}", method.name);
                assert!(
                    entry.marks.group(index).is_some(),
                    "{} element {index}",
                    method.name
                );
            }
        }
    }
}

/// The groups each method marks, from the table: the union over every entry that lists it.
fn expected_groups(table: &[GroupEntry]) -> BTreeMap<u32, GroupSet> {
    let mut expected: BTreeMap<u32, GroupSet> = BTreeMap::new();
    for entry in table {
        for (index, number) in (0..).zip(entry.span.numbers()) {
            let group = entry.marks.group(index).unwrap();
            let groups = expected.entry(number).or_default();
            *groups = groups.with(group);
        }
    }
    expected
}

#[test]
fn every_method_marks_exactly_its_groups_and_ungrouped_ones_are_counted() {
    for (class, table, _) in tables() {
        let expected = expected_groups(table);
        let registers = ClassRules::for_class(class).registers();
        for method in 0..registers {
            let mut mirror = clean(class);
            mirror.write(method, method ^ 0x5a5a_0000).unwrap();
            let groups = expected.get(&method).copied().unwrap_or_default();
            assert_eq!(
                mirror.dirty(),
                groups,
                "class {:#x} method {method:#x}",
                class.0
            );
            let ungrouped = u64::from(groups.is_empty());
            assert_eq!(mirror.ungrouped_words(), ungrouped, "method {method:#x}");
        }
    }
    // Every other class has no table: every method is ungrouped.
    let mut dma = clean(DMA);
    dma.write(clc7b5::LAUNCH_DMA.method(), 1).unwrap();
    assert_eq!(dma.dirty(), GroupSet::EMPTY);
    assert_eq!(dma.ungrouped_words(), 1);
}

#[test]
fn groups_by_method_number_from_the_header_text() {
    // Method numbers worked out by hand from `clc797.h` and `clc7c0.h` (byte address / 4), so
    // a wrong span, stride or element-to-group rule fails here even where the table-driven test
    // above agrees with itself.
    use GroupKind::*;
    let cases: &[(ClassId, u32, &[Group])] = &[
        // SET_VIEWPORT_SCALE_X(0): 0x0a00.
        (THREE_D, 0x280, &[one(Viewport)]),
        // SET_VIEWPORT_CLIP_MAX_Z(15): 0x0c0c + 15 * 16.
        (THREE_D, 0x33f, &[one(Viewport)]),
        // SET_SCISSOR_VERTICAL(15): 0x0e08 + 15 * 16.
        (THREE_D, 0x3be, &[one(Scissor)]),
        // SET_COLOR_TARGET_A(3): 0x0800 + 3 * 64.
        (THREE_D, 0x230, &[group(ColorTarget, 3)]),
        // SET_COLOR_TARGET_C_ROP_SLICE_MAP(7): 0x0824 + 7 * 64.
        (THREE_D, 0x279, &[group(ColorTarget, 7)]),
        // SET_BLEND_PER_TARGET_COLOR_OP(7): 0x1e04 + 7 * 32.
        (THREE_D, 0x7b9, &[group(BlendPerTarget, 7)]),
        // SET_BLEND(2): 0x1360 + 2 * 4.
        (THREE_D, 0x4da, &[group(BlendPerTarget, 2)]),
        // SET_CT_WRITE(5): 0x1a00 + 5 * 4.
        (THREE_D, 0x685, &[group(BlendPerTarget, 5)]),
        // SET_VERTEX_STREAM_A_FORMAT(2): 0x1c00 + 2 * 16.
        (THREE_D, 0x708, &[group(VertexStream, 2)]),
        // SET_VERTEX_STREAM_B_FORMAT(2): 0x1d00 + 2 * 16, stream 18.
        (THREE_D, 0x748, &[group(VertexStream, 18)]),
        // SET_VERTEX_STREAM_INSTANCE_B(1): 0x18c0 + 1 * 4, stream 17.
        (THREE_D, 0x631, &[group(VertexStream, 17)]),
        // SET_VERTEX_STREAM_SIZE_B(15): 0x0604 + 15 * 8.
        (THREE_D, 0x19f, &[group(VertexStream, 15)]),
        // SET_VERTEX_ATTRIBUTE_B(15): 0x11a0 + 15 * 4.
        (THREE_D, 0x477, &[one(VertexAttribute)]),
        // SET_INDEX_BUFFER_F: 0x17dc.
        (THREE_D, 0x5f7, &[one(IndexBuffer)]),
        // BIND_GROUP_CONSTANT_BUFFER(4): 0x2410 + 4 * 32.
        (THREE_D, 0x924, &[group(BindGroup, 4)]),
        // BIND_GROUP_CONSTANT_BUFFER(15): 0x2410 + 15 * 32.
        (THREE_D, 0x97c, &[group(BindGroup, 15)]),
        // SET_PIPELINE_PROGRAM_ADDRESS_B(5): 0x2018 + 5 * 64.
        (THREE_D, 0x856, &[group(Pipeline, 5)]),
        // SET_TEX_HEADER_POOL_A: 0x1574.
        (THREE_D, 0x55d, &[one(TexPool)]),
        // SET_ZT_SELECT: 0x1538.
        (THREE_D, 0x54e, &[one(Zt)]),
        // SET_CT_SELECT: 0x121c.
        (THREE_D, 0x487, &[one(CtSelect)]),
        // SET_SURFACE_CLIP_HORIZONTAL: 0x0ff4, two groups.
        (THREE_D, 0x3fd, &[one(Viewport), one(CtSelect)]),
        // SET_WINDOW_ORIGIN: 0x13ac, two groups.
        (THREE_D, 0x4eb, &[one(Viewport), one(Rasterizer)]),
        // SET_DEPTH_BIAS_CLAMP: 0x187c.
        (THREE_D, 0x61f, &[one(Rasterizer)]),
        // SET_DEPTH_FUNC: 0x130c.
        (THREE_D, 0x4c3, &[one(DepthStencil)]),
        // SET_BLEND_CONST_ALPHA: 0x1328.
        (THREE_D, 0x4ca, &[one(Blend)]),
        // SET_CONSTANT_BUFFER_SELECTOR_A: 0x2380, no group.
        (THREE_D, 0x8e0, &[]),
        // SET_TEX_HEADER_POOL_A of compute: 0x1574.
        (COMPUTE, 0x55d, &[one(TexPool)]),
        // SEND_PCAS_A of compute: 0x02b4, no group.
        (COMPUTE, 0xad, &[]),
    ];
    for &(class, method, groups) in cases {
        let mut mirror = clean(class);
        mirror.write(method, 1).unwrap();
        assert_eq!(
            mirror.dirty(),
            set(groups),
            "class {:#x} method {method:#x}",
            class.0
        );
    }
}

#[test]
fn the_named_constants_have_the_header_addresses_used_above() {
    // The byte addresses the previous test's comments rely on, read from the generated tables.
    assert_eq!(clc797::SET_CT_SELECT.byte_address, 0x121c);
    assert_eq!(clc797::SET_DEPTH_BIAS_CLAMP.byte_address, 0x187c);
    assert_eq!(clc797::SET_BLEND_CONST_ALPHA.byte_address, 0x1328);
    assert_eq!(clc7c0::SEND_PCAS_A.byte_address, 0x02b4);
}

// ---------------------------------------------------------------------------------------------
// A new mirror.

#[test]
fn a_new_mirror_has_every_named_group_dirty_and_nothing_else() {
    let mirror = three_d();
    assert_eq!(mirror.channel(), CHANNEL);
    assert_eq!(mirror.class(), THREE_D);
    // The 3D table names every group.
    assert_eq!(mirror.dirty().len(), GROUP_COUNT);
    assert_eq!(mirror.ungrouped_words(), 0);
    let compute = compute();
    assert_eq!(compute.dirty(), set(&[one(GroupKind::TexPool)]));
    assert!(compute.is_dirty(one(GroupKind::TexPool)));
    assert!(!compute.is_dirty(one(GroupKind::Zt)));
    assert_eq!(Mirror::new(CHANNEL, DMA).dirty(), GroupSet::EMPTY);
}

#[test]
fn a_new_mirror_starts_from_the_same_values_as_the_front_file() {
    for class in [THREE_D, COMPUTE, DMA, ClassId(cl902d::CLASS)] {
        let front = RegFile::new(CHANNEL, class);
        let mirror = Mirror::new(CHANNEL, class);
        let registers = ClassRules::for_class(class).registers();
        assert_eq!(
            mirror.registers().len(),
            usize::try_from(registers).unwrap()
        );
        for method in 0..registers {
            assert_eq!(Some(mirror.value(method).unwrap()), front.get(method));
        }
        assert_eq!(mirror.value(registers), None);
    }
}

#[test]
fn a_new_mirror_has_no_constant_buffer_bound() {
    let mirror = three_d();
    let view = mirror.three_d().unwrap();
    for group in 0..BIND_GROUPS {
        for slot in 0..CONSTANT_BUFFER_SLOTS {
            let binding = view
                .bind_group(group)
                .unwrap()
                .constant_buffer(slot)
                .unwrap();
            assert_eq!(binding, ConstantBufferBinding::default());
            assert_eq!(
                (binding.valid, binding.address, binding.size),
                (false, 0, 0)
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Writes, runs and clearing.

#[test]
fn a_write_outside_the_file_is_refused_and_changes_nothing() {
    let limit = ClassRules::for_class(THREE_D).registers();
    let mut mirror = clean(THREE_D);
    assert_eq!(
        mirror.write(limit, 1),
        Err(StateError::OutOfRange {
            method: limit,
            limit
        })
    );
    assert_eq!(
        mirror.run(limit, &[1, 2]),
        Err(StateError::OutOfRange {
            method: limit,
            limit
        })
    );
    assert_eq!(mirror.ungrouped_words(), 0);
    assert_eq!(mirror.dirty(), GroupSet::EMPTY);
    // One below the limit is the last register: stored, and counted as ungrouped.
    mirror.write(limit - 1, 0xabcd).unwrap();
    assert_eq!(mirror.value(limit - 1), Some(0xabcd));
    assert_eq!(mirror.ungrouped_words(), 1);
}

#[test]
fn a_run_stores_its_last_word_and_marks_once() {
    let mut mirror = clean(THREE_D);
    let scissor = number(&clc797::SET_SCISSOR_HORIZONTAL, &[2]);
    mirror.run(scissor, &[1, 2, 3]).unwrap();
    assert_eq!(mirror.value(scissor), Some(3));
    assert_eq!(mirror.dirty(), set(&[one(GroupKind::Scissor)]));
    assert_eq!(mirror.ungrouped_words(), 0);

    let selector = clc797::SET_CONSTANT_BUFFER_SELECTOR_C.method();
    mirror.run(selector, &[4, 5, 6]).unwrap();
    assert_eq!(mirror.value(selector), Some(6));
    assert_eq!(mirror.ungrouped_words(), 3);
    mirror.write(selector, 7).unwrap();
    assert_eq!(mirror.ungrouped_words(), 4);

    // No words: nothing changes and nothing is marked or counted.
    let mut mirror = clean(THREE_D);
    mirror.run(scissor, &[]).unwrap();
    mirror.run(selector, &[]).unwrap();
    assert_eq!(mirror.dirty(), GroupSet::EMPTY);
    assert_eq!(mirror.ungrouped_words(), 0);
    assert_eq!(mirror.value(scissor), Some(0));
}

#[test]
fn a_write_marks_its_group_even_when_the_value_is_unchanged() {
    let mut mirror = clean(THREE_D);
    let func = clc797::SET_DEPTH_FUNC.method();
    mirror.write(func, 0).unwrap();
    assert_eq!(mirror.dirty(), set(&[one(GroupKind::DepthStencil)]));
}

#[test]
fn apply_takes_only_this_mirrors_writes_and_runs() {
    let mut mirror = clean(THREE_D);
    let method = number(&clc797::SET_COLOR_TARGET_WIDTH, &[1]);
    let short = u16::try_from(method).unwrap();
    let write = |channel, class, value| RingEntry::Write {
        channel,
        class,
        method: short,
        value,
    };
    assert_eq!(mirror.apply(&write(ChannelId(8), THREE_D, 9)), Ok(false));
    assert_eq!(mirror.apply(&write(CHANNEL, COMPUTE, 9)), Ok(false));
    assert_eq!(
        mirror.apply(&RingEntry::WaitForIdle { channel: CHANNEL }),
        Ok(false)
    );
    assert_eq!(mirror.value(method), Some(0));
    assert_eq!(mirror.dirty(), GroupSet::EMPTY);

    assert_eq!(mirror.apply(&write(CHANNEL, THREE_D, 9)), Ok(true));
    assert_eq!(mirror.value(method), Some(9));
    assert_eq!(mirror.dirty(), set(&[group(GroupKind::ColorTarget, 1)]));

    let run = |channel, class| RingEntry::Run {
        channel,
        class,
        method: short,
        words: Cow::Owned(vec![10, 11]),
    };
    assert_eq!(mirror.apply(&run(ChannelId(8), THREE_D)), Ok(false));
    assert_eq!(mirror.apply(&run(CHANNEL, DMA)), Ok(false));
    assert_eq!(mirror.value(method), Some(9));
    assert_eq!(mirror.apply(&run(CHANNEL, THREE_D)), Ok(true));
    assert_eq!(mirror.value(method), Some(11));
}

/// A 3D mirror with distinct values in every register, a binding and some counts.
fn busy_mirror() -> Mirror {
    let mut mirror = three_d();
    let registers = ClassRules::for_class(THREE_D).registers();
    for method in 0..registers {
        mirror
            .write(method, method.wrapping_mul(0x9e37_79b9))
            .unwrap();
    }
    put(
        &mut mirror,
        &clc797::SET_CONSTANT_BUFFER_SELECTOR_A,
        &[],
        0x100,
    );
    put(
        &mut mirror,
        &clc797::SET_CONSTANT_BUFFER_SELECTOR_B,
        &[],
        0x12,
    );
    put(
        &mut mirror,
        &clc797::SET_CONSTANT_BUFFER_SELECTOR_C,
        &[],
        0x3400,
    );
    put(
        &mut mirror,
        &clc797::BIND_GROUP_CONSTANT_BUFFER,
        &[1],
        bind_word(true, 3),
    );
    mirror
}

fn bindings(mirror: &Mirror) -> Vec<ConstantBufferBinding> {
    let view = mirror.three_d().unwrap();
    (0..BIND_GROUPS)
        .flat_map(|group| {
            (0..CONSTANT_BUFFER_SLOTS).map(move |slot| {
                view.bind_group(group)
                    .unwrap()
                    .constant_buffer(slot)
                    .unwrap()
            })
        })
        .collect()
}

#[test]
fn clearing_after_derivation_leaves_the_mirror_unchanged() {
    let mut mirror = busy_mirror();
    let registers = mirror.registers().to_vec();
    let bound = bindings(&mirror);
    let ungrouped = mirror.ungrouped_words();
    assert_eq!(mirror.dirty().len(), GROUP_COUNT);

    // Clearing some groups clears only those.
    let some = set(&[one(GroupKind::Viewport), group(GroupKind::BindGroup, 1)]);
    mirror.clear(some);
    assert!(!mirror.is_dirty(one(GroupKind::Viewport)));
    assert!(!mirror.is_dirty(group(GroupKind::BindGroup, 1)));
    assert!(mirror.is_dirty(group(GroupKind::BindGroup, 0)));
    assert_eq!(mirror.dirty().len(), GROUP_COUNT - 2);

    // Taking the rest clears everything.
    let taken = mirror.take_dirty();
    assert_eq!(taken.len(), GROUP_COUNT - 2);
    assert!(!taken.contains(one(GroupKind::Viewport)));
    assert_eq!(mirror.dirty(), GroupSet::EMPTY);

    assert_eq!(mirror.registers(), registers.as_slice());
    assert_eq!(bindings(&mirror), bound);
    assert_eq!(mirror.ungrouped_words(), ungrouped);
}

// ---------------------------------------------------------------------------------------------
// Run positions.

fn offset() -> u32 {
    clc797::LOAD_CONSTANT_BUFFER_OFFSET.method()
}

fn constant_buffer(index: u32) -> u32 {
    number(&clc797::LOAD_CONSTANT_BUFFER, &[index])
}

#[test]
fn run_positions_advance_within_the_offset_field_and_keep_the_bits_above() {
    let mut mirror = three_d();
    mirror.write(offset(), 0x1234_fff8).unwrap();
    let positions: Vec<u32> = mirror
        .run_positions(constant_buffer(LOAD_CONSTANT_BUFFER_COUNT - 1), offset(), 4)
        .unwrap()
        .collect();
    assert_eq!(
        positions,
        [0x1234_fff8, 0x1234_fffc, 0x1234_0000, 0x1234_0004]
    );
    // The mirror itself is not advanced: the front sends that.
    assert_eq!(mirror.value(offset()), Some(0x1234_fff8));
}

#[test]
fn run_positions_without_a_rule_stay_put() {
    let mut mirror = three_d();
    mirror.write(offset(), 0x40).unwrap();
    // `LOAD_INLINE_DATA` has no side effect on the offset.
    let positions = mirror
        .run_positions(clc797::LOAD_INLINE_DATA.method(), offset(), 3)
        .unwrap();
    assert_eq!(positions.len(), 3);
    assert_eq!(positions.collect::<Vec<_>>(), [0x40; 3]);
    // Nor does a constant buffer load past the array Mesa gives.
    let past = clc797::LOAD_CONSTANT_BUFFER
        .method_at(&[LOAD_CONSTANT_BUFFER_COUNT])
        .unwrap();
    let positions: Vec<u32> = mirror.run_positions(past, offset(), 2).unwrap().collect();
    assert_eq!(positions, [0x40; 2]);
}

#[test]
fn run_positions_of_the_root_table_follow_its_alignment_rule() {
    let selector = clc797::SET_ROOT_TABLE_SELECTOR.method();
    let field = clc797::SET_ROOT_TABLE_SELECTOR_OFFSET;
    let load = clc797::LOAD_ROOT_TABLE.method();
    let mut mirror = three_d();
    mirror
        .write(selector, field.insert(0x0000_0003, 248).unwrap())
        .unwrap();
    let offsets: Vec<u32> = mirror
        .run_positions(load, selector, 3)
        .unwrap()
        .map(|word| field.get(word))
        .collect();
    assert_eq!(offsets, [248, 252, 0]);
    // An offset that is not a multiple of 4 does not advance.
    mirror.write(selector, field.insert(0, 5).unwrap()).unwrap();
    let offsets: Vec<u32> = mirror
        .run_positions(load, selector, 2)
        .unwrap()
        .map(|word| field.get(word))
        .collect();
    assert_eq!(offsets, [5, 5]);
}

#[test]
fn run_positions_of_a_register_outside_the_file_are_refused() {
    let limit = ClassRules::for_class(THREE_D).registers();
    let mirror = three_d();
    assert_eq!(
        mirror
            .run_positions(constant_buffer(0), limit, 1)
            .unwrap_err(),
        StateError::OutOfRange {
            method: limit,
            limit
        }
    );
    assert!(
        mirror
            .run_positions(constant_buffer(0), limit - 1, 1)
            .is_ok()
    );
}

/// Sends `entries` to `mirror` in order, and for each run or write to `method`, the positions
/// of its words in `register`, taken before the entries after it are applied.
fn apply_collecting(
    mirror: &mut Mirror,
    entries: &[RingEntry<'static>],
    method: u32,
    register: u32,
) -> Vec<u32> {
    let mut positions = Vec::new();
    for entry in entries {
        let words = match entry {
            RingEntry::Run {
                method: m, words, ..
            } if u32::from(*m) == method => words.len(),
            RingEntry::Write { method: m, .. } if u32::from(*m) == method => 1,
            _ => 0,
        };
        positions.extend(mirror.run_positions(method, register, words).unwrap());
        assert_eq!(mirror.apply(entry), Ok(true));
    }
    positions
}

#[test]
fn a_multi_chunk_run_that_wraps_the_offset_lands_where_the_front_put_it() {
    // A run through the front's register file of a full wrap of the 16-bit offset field and
    // three words more: three `Run` entries of `MAX_RUN_WORDS`, `MAX_RUN_WORDS` and 3 words,
    // each followed by the offset's `Write`.
    let field_bytes = clc797::LOAD_CONSTANT_BUFFER_OFFSET_V.max() + 1;
    let wrap_words = usize::try_from(field_bytes / WORD_BYTES).unwrap();
    assert_eq!(wrap_words, 2 * MAX_RUN_WORDS);
    let words: Vec<u32> = (0..u32::try_from(wrap_words + 3).unwrap()).collect();
    let start = 0x00aa_0010;

    let mut front = RegFile::new(CHANNEL, THREE_D);
    let mut entries: Vec<RingEntry<'static>> = Vec::new();
    front.write(offset(), start, &mut entries).unwrap();
    front
        .write_run(constant_buffer(5), &words, &mut entries)
        .unwrap();
    let runs = entries
        .iter()
        .filter(|entry| matches!(entry, RingEntry::Run { .. }))
        .count();
    assert_eq!(runs, 3);

    let mut mirror = three_d();
    let positions = apply_collecting(&mut mirror, &entries, constant_buffer(5), offset());
    assert_eq!(positions.len(), words.len());
    for (k, &position) in (0u32..).zip(&positions) {
        let expected = 0x00aa_0000 | (0x10u32.wrapping_add(k * WORD_BYTES) & 0xffff);
        assert_eq!(position, expected, "word {k}");
    }
    // A full wrap comes back to the start.
    assert_eq!(positions[wrap_words], start);
    // And the mirror ends equal to the front file, offset included.
    for method in 0..ClassRules::for_class(THREE_D).registers() {
        assert_eq!(
            Some(mirror.value(method).unwrap()),
            front.get(method),
            "{method:#x}"
        );
    }
    assert_eq!(mirror.value(offset()), Some(0x00aa_001c));
}

// ---------------------------------------------------------------------------------------------
// Constant buffer bindings.

fn bind_word(valid: bool, slot: u32) -> u32 {
    let valid = if valid {
        clc797::BIND_GROUP_CONSTANT_BUFFER_VALID_TRUE
    } else {
        clc797::BIND_GROUP_CONSTANT_BUFFER_VALID_FALSE
    };
    let word = clc797::BIND_GROUP_CONSTANT_BUFFER_VALID
        .insert(0, valid)
        .unwrap();
    clc797::BIND_GROUP_CONSTANT_BUFFER_SHADER_SLOT
        .insert(word, slot)
        .unwrap()
}

fn select(mirror: &mut Mirror, size: u32, upper: u32, lower: u32) {
    put(mirror, &clc797::SET_CONSTANT_BUFFER_SELECTOR_A, &[], size);
    put(mirror, &clc797::SET_CONSTANT_BUFFER_SELECTOR_B, &[], upper);
    put(mirror, &clc797::SET_CONSTANT_BUFFER_SELECTOR_C, &[], lower);
}

fn binding(mirror: &Mirror, group: u32, slot: u32) -> ConstantBufferBinding {
    mirror
        .three_d()
        .unwrap()
        .bind_group(group)
        .unwrap()
        .constant_buffer(slot)
        .unwrap()
}

#[test]
fn a_bind_records_the_selected_buffer_in_its_group_and_slot() {
    let mut mirror = clean(THREE_D);
    // Bits above each selector field are ignored: size is 16:0 and the upper address 7:0.
    select(&mut mirror, 0xfffe_0100, 0xffff_ffab, 0x1234_5600);
    let last_slot = CONSTANT_BUFFER_SLOTS - 1;
    put(
        &mut mirror,
        &clc797::BIND_GROUP_CONSTANT_BUFFER,
        &[3],
        bind_word(true, last_slot),
    );
    let expected = ConstantBufferBinding {
        valid: true,
        address: 0xab_1234_5600,
        size: 0x0_0100,
    };
    assert_eq!(binding(&mirror, 3, last_slot), expected);
    assert_eq!(mirror.dirty(), set(&[group(GroupKind::BindGroup, 3)]));
    // The neighbouring slot and the same slot of the neighbouring groups are untouched.
    assert_eq!(
        binding(&mirror, 3, last_slot - 1),
        ConstantBufferBinding::default()
    );
    assert_eq!(
        binding(&mirror, 2, last_slot),
        ConstantBufferBinding::default()
    );
    assert_eq!(
        binding(&mirror, 4, last_slot),
        ConstantBufferBinding::default()
    );

    // An invalid bind keeps the selected address and size, as Eden-NX2's `ProcessCBBind`.
    select(&mut mirror, 0x40, 0x1, 0x2000);
    put(
        &mut mirror,
        &clc797::BIND_GROUP_CONSTANT_BUFFER,
        &[3],
        bind_word(false, last_slot),
    );
    assert_eq!(
        binding(&mirror, 3, last_slot),
        ConstantBufferBinding {
            valid: false,
            address: 0x1_0000_2000,
            size: 0x40
        }
    );
}

#[test]
fn every_word_of_a_bind_run_binds() {
    let mut mirror = clean(THREE_D);
    select(&mut mirror, 0x80, 0, 0x8000);
    let bind = number(&clc797::BIND_GROUP_CONSTANT_BUFFER, &[BIND_GROUPS - 1]);
    mirror
        .run(bind, &[bind_word(true, 0), bind_word(true, 9)])
        .unwrap();
    let expected = ConstantBufferBinding {
        valid: true,
        address: 0x8000,
        size: 0x80,
    };
    assert_eq!(binding(&mirror, BIND_GROUPS - 1, 0), expected);
    assert_eq!(binding(&mirror, BIND_GROUPS - 1, 9), expected);
    assert_eq!(
        binding(&mirror, BIND_GROUPS - 1, 1),
        ConstantBufferBinding::default()
    );
}

#[test]
fn bind_groups_are_told_apart() {
    // A bind to each group with a distinct selector lands in that group only.
    let mut mirror = clean(THREE_D);
    for group in 0..BIND_GROUPS {
        select(&mut mirror, group + 1, 0, 0x100 * (group + 1));
        put(
            &mut mirror,
            &clc797::BIND_GROUP_CONSTANT_BUFFER,
            &[group],
            bind_word(true, group),
        );
    }
    for group in 0..BIND_GROUPS {
        let binding = binding(&mirror, group, group);
        assert_eq!(binding.size, group + 1);
        assert_eq!(binding.address, u64::from(0x100 * (group + 1)));
    }
}

// ---------------------------------------------------------------------------------------------
// Views: classes, bounds and enums.

#[test]
fn a_view_of_the_wrong_class_is_refused() {
    let compute = compute();
    assert_eq!(
        compute.three_d().unwrap_err(),
        StateError::WrongClass {
            expected: clc797::CLASS,
            actual: clc7c0::CLASS
        }
    );
    let three_d = three_d();
    assert_eq!(
        three_d.compute().unwrap_err(),
        StateError::WrongClass {
            expected: clc7c0::CLASS,
            actual: clc797::CLASS
        }
    );
    assert!(compute.compute().is_ok());
    assert!(three_d.three_d().is_ok());
}

fn index_error(array: &'static str, index: u32, count: u32) -> StateError {
    StateError::Index {
        array,
        index,
        count,
    }
}

/// Writes each `(method, indices, word)` in order.
fn put_all(mirror: &mut Mirror, puts: &[(&Method, &[u32], u32)]) {
    for &(method, indices, word) in puts {
        put(mirror, method, indices, word);
    }
}

#[test]
fn element_accessors_are_bounded_at_their_exact_edge() {
    use clc797::*;
    let mut mirror = three_d();
    // Each last element gets a distinct value, read back through the accessor.
    put_all(
        &mut mirror,
        &[
            (&SET_VIEWPORT_SCALE_X, &[VIEWPORTS - 1], 1.5f32.to_bits()),
            (&SET_SCISSOR_HORIZONTAL, &[SCISSORS - 1], 0x0002_0001),
            (
                &SET_BLEND_PER_TARGET_COLOR_OP,
                &[BLEND_TARGETS - 1],
                SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_MAX,
            ),
            (&SET_COLOR_TARGET_WIDTH, &[COLOR_TARGETS - 1], 640),
            (&SET_PIPELINE_REGISTER_COUNT, &[PIPELINES - 1], 24),
            (
                &SET_CT_SELECT,
                &[],
                SET_CT_SELECT_TARGET7.insert(0, 6).unwrap(),
            ),
        ],
    );
    let view = mirror.three_d().unwrap();
    let viewport = view.viewport(VIEWPORTS - 1).unwrap();
    assert_eq!(viewport.scale_x(), Ok(1.5));
    let error = index_error(arrays::VIEWPORT, VIEWPORTS, VIEWPORTS);
    assert_eq!(view.viewport(VIEWPORTS).unwrap_err(), error);
    assert_eq!(view.scissor(SCISSORS - 1).unwrap().xmax(), Ok(2));
    let error = index_error(arrays::SCISSOR, SCISSORS, SCISSORS);
    assert_eq!(view.scissor(SCISSORS).unwrap_err(), error);
    let target = view.blend_per_target(BLEND_TARGETS - 1).unwrap();
    assert_eq!(name(target.color_op()), "OGL_MAX");
    let error = index_error(arrays::BLEND_TARGET, BLEND_TARGETS, BLEND_TARGETS);
    assert_eq!(view.blend_per_target(BLEND_TARGETS).unwrap_err(), error);
    assert_eq!(
        view.color_target(COLOR_TARGETS - 1).unwrap().width(),
        Ok(640)
    );
    let error = index_error(arrays::COLOR_TARGET, COLOR_TARGETS, COLOR_TARGETS);
    assert_eq!(view.color_target(COLOR_TARGETS).unwrap_err(), error);
    assert_eq!(
        view.pipeline(PIPELINES - 1).unwrap().register_count(),
        Ok(24)
    );
    let error = index_error(arrays::PIPELINE, PIPELINES, PIPELINES);
    assert_eq!(view.pipeline(PIPELINES).unwrap_err(), error);
    assert_eq!(view.ct_select().target(COLOR_TARGETS - 1), Ok(6));
    let error = index_error(arrays::CT_SELECT_TARGET, COLOR_TARGETS, COLOR_TARGETS);
    assert_eq!(view.ct_select().target(COLOR_TARGETS).unwrap_err(), error);
}

#[test]
fn stream_attribute_and_binding_accessors_are_bounded_at_their_exact_edge() {
    use clc797::*;
    let mut mirror = three_d();
    put_all(
        &mut mirror,
        &[
            (
                &SET_VERTEX_STREAM_B_FREQUENCY,
                &[VERTEX_STREAMS_PER_HALF - 1],
                33,
            ),
            (&SET_VERTEX_STREAM_SIZE_B, &[VERTEX_STREAM_SIZES - 1], 0x44),
            (
                &SET_VERTEX_ATTRIBUTE_B,
                &[VERTEX_ATTRIBUTES_PER_HALF - 1],
                0x1f,
            ),
        ],
    );
    select(&mut mirror, 0x10, 0, 0x20);
    let last_group = BIND_GROUPS - 1;
    let last_slot = CONSTANT_BUFFER_SLOTS - 1;
    put(
        &mut mirror,
        &BIND_GROUP_CONSTANT_BUFFER,
        &[last_group],
        bind_word(true, last_slot),
    );
    let view = mirror.three_d().unwrap();

    assert_eq!(
        view.vertex_stream(VERTEX_STREAMS - 1).unwrap().frequency(),
        Ok(33)
    );
    let error = index_error(arrays::VERTEX_STREAM, VERTEX_STREAMS, VERTEX_STREAMS);
    assert_eq!(view.vertex_stream(VERTEX_STREAMS).unwrap_err(), error);
    // The size array is shorter than the streams: stream 16 exists but has no size.
    assert_eq!(
        view.vertex_stream(VERTEX_STREAM_SIZES - 1).unwrap().size(),
        Ok(0x44)
    );
    let stream = view.vertex_stream(VERTEX_STREAM_SIZES).unwrap();
    let error = index_error(
        arrays::VERTEX_STREAM_SIZE,
        VERTEX_STREAM_SIZES,
        VERTEX_STREAM_SIZES,
    );
    assert_eq!(stream.size().unwrap_err(), error);
    let attribute = view.vertex_attribute(VERTEX_ATTRIBUTES - 1).unwrap();
    assert_eq!(attribute.stream(), Ok(0x1f));
    let error = index_error(
        arrays::VERTEX_ATTRIBUTE,
        VERTEX_ATTRIBUTES,
        VERTEX_ATTRIBUTES,
    );
    assert_eq!(view.vertex_attribute(VERTEX_ATTRIBUTES).unwrap_err(), error);
    let group = view.bind_group(last_group).unwrap();
    assert_eq!(group.index(), last_group);
    assert_eq!(group.constant_buffer(last_slot).unwrap().size, 0x10);
    let error = index_error(
        arrays::CONSTANT_BUFFER_SLOT,
        CONSTANT_BUFFER_SLOTS,
        CONSTANT_BUFFER_SLOTS,
    );
    assert_eq!(
        group.constant_buffer(CONSTANT_BUFFER_SLOTS).unwrap_err(),
        error
    );
    let error = index_error(arrays::BIND_GROUP, BIND_GROUPS, BIND_GROUPS);
    assert_eq!(view.bind_group(BIND_GROUPS).unwrap_err(), error);
}

#[test]
fn an_undefined_enum_value_is_the_exact_error() {
    let func = |value| {
        let mut mirror = three_d();
        put(&mut mirror, &clc797::SET_DEPTH_FUNC, &[], value);
        mirror.three_d().unwrap().depth_stencil().depth_func()
    };
    let undefined = |value| StateError::UndefinedValue {
        method: "SET_DEPTH_FUNC",
        field: "V",
        value,
    };
    // The header names 0x200 to 0x207 (OGL) and 1 to 8 (D3D).
    assert_eq!(func(0x200).unwrap().name, "OGL_NEVER");
    assert_eq!(func(0x207).unwrap().name, "OGL_ALWAYS");
    assert_eq!(func(0x1ff), Err(undefined(0x1ff)));
    assert_eq!(func(0x208), Err(undefined(0x208)));
    assert_eq!(func(1).unwrap().name, "D3D_NEVER");
    assert_eq!(func(8).unwrap().name, "D3D_ALWAYS");
    assert_eq!(func(0), Err(undefined(0)));
    assert_eq!(func(9), Err(undefined(9)));
    // A value is compared with the field, not the word: bits outside a narrower field are
    // ignored (`SET_VIEWPORT_COORDINATE_SWIZZLE_X` is 2:0).
    let mut mirror = three_d();
    put(
        &mut mirror,
        &clc797::SET_VIEWPORT_COORDINATE_SWIZZLE,
        &[0],
        0xffff_fff8 | clc797::SET_VIEWPORT_COORDINATE_SWIZZLE_X_NEG_W,
    );
    let viewport = mirror.three_d().unwrap().viewport(0).unwrap();
    assert_eq!(
        viewport.swizzle_x().unwrap().value,
        clc797::SET_VIEWPORT_COORDINATE_SWIZZLE_X_NEG_W
    );
}

#[test]
fn the_generic_reg_checks_its_indices_and_the_file() {
    let limit = ClassRules::for_class(THREE_D).registers();
    let mut mirror = three_d();
    let scratch = &clc797::SET_MME_SHADOW_SCRATCH;
    let last = limit - scratch.method();
    mirror.write(limit - 1, 0x77).unwrap();
    let reg = mirror.reg(scratch, &[last - 1]).unwrap();
    assert_eq!(
        (reg.number(), reg.word(), reg.method().name),
        (limit - 1, 0x77, "SET_MME_SHADOW_SCRATCH")
    );
    assert_eq!(
        mirror.reg(scratch, &[last]).unwrap_err(),
        StateError::OutOfRange {
            method: limit,
            limit
        }
    );
    assert_eq!(
        mirror.reg(scratch, &[]).unwrap_err(),
        StateError::NoElement {
            method: "SET_MME_SHADOW_SCRATCH",
            indices: 0
        }
    );
    // Too many indices is refused before the element could be found past the file.
    assert_eq!(
        mirror.reg(scratch, &[last, 1]).unwrap_err(),
        StateError::NoElement {
            method: "SET_MME_SHADOW_SCRATCH",
            indices: 2
        }
    );
    // A method that is not an array takes no index.
    assert!(mirror.reg(&clc797::SET_DEPTH_FUNC, &[]).is_ok());
}

#[test]
fn a_field_of_another_method_is_refused() {
    let mirror = three_d();
    let reg = mirror.reg(&clc797::SET_DEPTH_FUNC, &[]).unwrap();
    let foreign = clc797::SET_SCISSOR_HORIZONTAL_XMAX;
    let no_field = StateError::NoField {
        method: "SET_DEPTH_FUNC",
        low: 16,
        width: 16,
    };
    assert_eq!(reg.field(foreign), Err(no_field));
    assert_eq!(reg.named(foreign), Err(no_field));
    assert_eq!(reg.field(clc797::SET_DEPTH_FUNC_V), Ok(0));
    // A field with no named values defines none.
    let clip = mirror
        .reg(&clc797::SET_VIEWPORT_CLIP_HORIZONTAL, &[0])
        .unwrap();
    assert_eq!(
        clip.named(clc797::SET_VIEWPORT_CLIP_HORIZONTAL_X0),
        Err(StateError::UndefinedValue {
            method: "SET_VIEWPORT_CLIP_HORIZONTAL",
            field: "X0",
            value: 0
        })
    );
}

#[test]
fn errors_read_as_plain_words() {
    let cases = [
        (
            StateError::OutOfRange {
                method: 0xe00,
                limit: 0xe00,
            },
            "method 0xe00 is outside this class's register file, which holds methods below 0xe00",
        ),
        (
            StateError::NoElement {
                method: "SET_BLEND",
                indices: 0,
            },
            "SET_BLEND has no element at the 0 indices given",
        ),
        (
            index_error(arrays::VIEWPORT, 16, 16),
            "viewport 16 does not exist: there are 16",
        ),
        (
            StateError::UndefinedValue {
                method: "SET_DEPTH_FUNC",
                field: "V",
                value: 0x208,
            },
            "SET_DEPTH_FUNC V holds 0x208, which the class header does not define",
        ),
        (
            StateError::NoField {
                method: "SET_DEPTH_FUNC",
                low: 16,
                width: 16,
            },
            "SET_DEPTH_FUNC has no field of 16 bits at bit 16",
        ),
        (
            StateError::WrongClass {
                expected: 0xc797,
                actual: 0xc7c0,
            },
            "this mirror holds class 0xc7c0, not class 0xc797",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
}

// ---------------------------------------------------------------------------------------------
// Views: every getter reads its own method and field. Each test writes distinct values so a
// getter wired to a neighbour fails.

/// Writes `fields` into one word of `method` at `indices`.
fn put_fields(mirror: &mut Mirror, method: &Method, indices: &[u32], fields: &[(Field, u32)]) {
    let word = fields.iter().fold(0, |word, &(field, value)| {
        field.insert(word, value).unwrap()
    });
    put(mirror, method, indices, word);
}

fn name(value: Result<&'static Value, StateError>) -> &'static str {
    value.unwrap().name
}

#[test]
fn viewport_getters() {
    use clc797::*;
    let mut mirror = three_d();
    let i = 5;
    let floats = [
        (&SET_VIEWPORT_SCALE_X, 1.0f32),
        (&SET_VIEWPORT_SCALE_Y, 2.0),
        (&SET_VIEWPORT_SCALE_Z, 3.0),
        (&SET_VIEWPORT_OFFSET_X, 4.0),
        (&SET_VIEWPORT_OFFSET_Y, 5.0),
        (&SET_VIEWPORT_OFFSET_Z, 6.0),
        (&SET_VIEWPORT_CLIP_MIN_Z, 7.0),
        (&SET_VIEWPORT_CLIP_MAX_Z, 8.0),
    ];
    for (method, value) in floats {
        put(&mut mirror, method, &[i], value.to_bits());
        // The neighbours hold other values.
        put(&mut mirror, method, &[i - 1], (-value).to_bits());
        put(&mut mirror, method, &[i + 1], (value + 0.5).to_bits());
    }
    put_fields(
        &mut mirror,
        &SET_VIEWPORT_COORDINATE_SWIZZLE,
        &[i],
        &[
            (
                SET_VIEWPORT_COORDINATE_SWIZZLE_X,
                SET_VIEWPORT_COORDINATE_SWIZZLE_X_POS_Y,
            ),
            (
                SET_VIEWPORT_COORDINATE_SWIZZLE_Y,
                SET_VIEWPORT_COORDINATE_SWIZZLE_Y_NEG_Z,
            ),
            (
                SET_VIEWPORT_COORDINATE_SWIZZLE_Z,
                SET_VIEWPORT_COORDINATE_SWIZZLE_Z_POS_W,
            ),
            (
                SET_VIEWPORT_COORDINATE_SWIZZLE_W,
                SET_VIEWPORT_COORDINATE_SWIZZLE_W_NEG_X,
            ),
        ],
    );
    put_fields(
        &mut mirror,
        &SET_VIEWPORT_CLIP_HORIZONTAL,
        &[i],
        &[
            (SET_VIEWPORT_CLIP_HORIZONTAL_X0, 11),
            (SET_VIEWPORT_CLIP_HORIZONTAL_WIDTH, 12),
        ],
    );
    put_fields(
        &mut mirror,
        &SET_VIEWPORT_CLIP_VERTICAL,
        &[i],
        &[
            (SET_VIEWPORT_CLIP_VERTICAL_Y0, 13),
            (SET_VIEWPORT_CLIP_VERTICAL_HEIGHT, 14),
        ],
    );
    let view = mirror.three_d().unwrap();
    let viewport = view.viewport(i).unwrap();
    assert_eq!(viewport.index(), i);
    let read = [
        viewport.scale_x(),
        viewport.scale_y(),
        viewport.scale_z(),
        viewport.offset_x(),
        viewport.offset_y(),
        viewport.offset_z(),
        viewport.clip_min_z(),
        viewport.clip_max_z(),
    ];
    assert_eq!(
        read.map(Result::unwrap),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
    assert_eq!(
        [
            viewport.swizzle_x(),
            viewport.swizzle_y(),
            viewport.swizzle_z(),
            viewport.swizzle_w()
        ]
        .map(name),
        ["POS_Y", "NEG_Z", "POS_W", "NEG_X"]
    );
    assert_eq!(
        [
            viewport.clip_x0(),
            viewport.clip_width(),
            viewport.clip_y0(),
            viewport.clip_height()
        ]
        .map(Result::unwrap),
        [11, 12, 13, 14]
    );
}

#[test]
fn scissor_and_ct_select_getters() {
    use clc797::*;
    let mut mirror = three_d();
    put_fields(
        &mut mirror,
        &SET_SCISSOR_ENABLE,
        &[9],
        &[(SET_SCISSOR_ENABLE_V, SET_SCISSOR_ENABLE_V_TRUE)],
    );
    put_fields(
        &mut mirror,
        &SET_SCISSOR_HORIZONTAL,
        &[9],
        &[
            (SET_SCISSOR_HORIZONTAL_XMIN, 1),
            (SET_SCISSOR_HORIZONTAL_XMAX, 2),
        ],
    );
    put_fields(
        &mut mirror,
        &SET_SCISSOR_VERTICAL,
        &[9],
        &[
            (SET_SCISSOR_VERTICAL_YMIN, 3),
            (SET_SCISSOR_VERTICAL_YMAX, 4),
        ],
    );
    put_fields(
        &mut mirror,
        &SET_CT_SELECT,
        &[],
        &[
            (SET_CT_SELECT_TARGET_COUNT, 5),
            (SET_CT_SELECT_TARGET0, 7),
            (SET_CT_SELECT_TARGET1, 6),
            (SET_CT_SELECT_TARGET2, 5),
            (SET_CT_SELECT_TARGET3, 4),
            (SET_CT_SELECT_TARGET4, 3),
            (SET_CT_SELECT_TARGET5, 2),
            (SET_CT_SELECT_TARGET6, 1),
            (SET_CT_SELECT_TARGET7, 0),
        ],
    );
    put_fields(
        &mut mirror,
        &SET_SURFACE_CLIP_HORIZONTAL,
        &[],
        &[
            (SET_SURFACE_CLIP_HORIZONTAL_X, 21),
            (SET_SURFACE_CLIP_HORIZONTAL_WIDTH, 22),
        ],
    );
    put_fields(
        &mut mirror,
        &SET_SURFACE_CLIP_VERTICAL,
        &[],
        &[
            (SET_SURFACE_CLIP_VERTICAL_Y, 23),
            (SET_SURFACE_CLIP_VERTICAL_HEIGHT, 24),
        ],
    );
    let view = mirror.three_d().unwrap();
    let scissor = view.scissor(9).unwrap();
    assert_eq!(scissor.index(), 9);
    assert_eq!(name(scissor.enable()), "TRUE");
    assert_eq!(name(view.scissor(8).unwrap().enable()), "FALSE");
    assert_eq!(
        [
            scissor.xmin(),
            scissor.xmax(),
            scissor.ymin(),
            scissor.ymax()
        ]
        .map(Result::unwrap),
        [1, 2, 3, 4]
    );
    let select = view.ct_select();
    assert_eq!(select.target_count(), Ok(5));
    let targets: Vec<u32> = (0..COLOR_TARGETS)
        .map(|k| select.target(k).unwrap())
        .collect();
    assert_eq!(targets, [7, 6, 5, 4, 3, 2, 1, 0]);
    assert_eq!(
        [
            select.surface_clip_x(),
            select.surface_clip_width(),
            select.surface_clip_y(),
            select.surface_clip_height()
        ]
        .map(Result::unwrap),
        [21, 22, 23, 24]
    );
}

#[test]
fn blend_getters() {
    use clc797::*;
    let mut mirror = three_d();
    put_all(
        &mut mirror,
        &[
            (
                &SET_BLEND_STATE_PER_TARGET,
                &[],
                SET_BLEND_STATE_PER_TARGET_ENABLE_TRUE,
            ),
            (
                &SET_BLEND_SEPARATE_FOR_ALPHA,
                &[],
                SET_BLEND_SEPARATE_FOR_ALPHA_ENABLE_TRUE,
            ),
            (
                &SET_BLEND_COLOR_OP,
                &[],
                SET_BLEND_COLOR_OP_V_OGL_FUNC_SUBTRACT,
            ),
            (
                &SET_BLEND_COLOR_SOURCE_COEFF,
                &[],
                SET_BLEND_COLOR_SOURCE_COEFF_V_OGL_SRC_ALPHA,
            ),
            (
                &SET_BLEND_COLOR_DEST_COEFF,
                &[],
                SET_BLEND_COLOR_DEST_COEFF_V_OGL_ONE_MINUS_SRC_ALPHA,
            ),
            (&SET_BLEND_ALPHA_OP, &[], SET_BLEND_ALPHA_OP_V_OGL_MIN),
            (
                &SET_BLEND_ALPHA_SOURCE_COEFF,
                &[],
                SET_BLEND_ALPHA_SOURCE_COEFF_V_OGL_DST_ALPHA,
            ),
            (
                &SET_BLEND_ALPHA_DEST_COEFF,
                &[],
                SET_BLEND_ALPHA_DEST_COEFF_V_OGL_ZERO,
            ),
            (&SET_BLEND_CONST_RED, &[], 0.25f32.to_bits()),
            (&SET_BLEND_CONST_GREEN, &[], 0.5f32.to_bits()),
            (&SET_BLEND_CONST_BLUE, &[], 0.75f32.to_bits()),
            (&SET_BLEND_CONST_ALPHA, &[], 1.0f32.to_bits()),
            (
                &SET_SINGLE_CT_WRITE_CONTROL,
                &[],
                SET_SINGLE_CT_WRITE_CONTROL_ENABLE_TRUE,
            ),
        ],
    );
    let view = mirror.three_d().unwrap();
    let blend = view.blend();
    let enables = [
        blend.state_per_target(),
        blend.separate_for_alpha(),
        blend.single_ct_write_control(),
    ];
    assert_eq!(enables.map(name), ["TRUE", "TRUE", "TRUE"]);
    let ops = [
        blend.color_op(),
        blend.color_source_coeff(),
        blend.color_dest_coeff(),
        blend.alpha_op(),
        blend.alpha_source_coeff(),
        blend.alpha_dest_coeff(),
    ];
    let expected = [
        "OGL_FUNC_SUBTRACT",
        "OGL_SRC_ALPHA",
        "OGL_ONE_MINUS_SRC_ALPHA",
        "OGL_MIN",
        "OGL_DST_ALPHA",
        "OGL_ZERO",
    ];
    assert_eq!(ops.map(name), expected);
    let constants = [
        blend.const_red(),
        blend.const_green(),
        blend.const_blue(),
        blend.const_alpha(),
    ];
    assert_eq!(constants.map(Result::unwrap), [0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn blend_per_target_getters() {
    use clc797::*;
    let mut mirror = three_d();
    let j = 6;
    let write = [
        (SET_CT_WRITE_R_ENABLE, SET_CT_WRITE_R_ENABLE_TRUE),
        (SET_CT_WRITE_G_ENABLE, SET_CT_WRITE_G_ENABLE_FALSE),
        (SET_CT_WRITE_B_ENABLE, SET_CT_WRITE_B_ENABLE_TRUE),
        (SET_CT_WRITE_A_ENABLE, SET_CT_WRITE_A_ENABLE_FALSE),
    ];
    put_fields(&mut mirror, &SET_CT_WRITE, &[j], &write);
    put_all(
        &mut mirror,
        &[
            (&SET_BLEND, &[j], SET_BLEND_ENABLE_TRUE),
            (
                &SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA,
                &[j],
                SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA_ENABLE_TRUE,
            ),
            (
                &SET_BLEND_PER_TARGET_COLOR_OP,
                &[j],
                SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_FUNC_ADD,
            ),
            (
                &SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF,
                &[j],
                SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_ONE,
            ),
            (
                &SET_BLEND_PER_TARGET_COLOR_DEST_COEFF,
                &[j],
                SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V_OGL_SRC_COLOR,
            ),
            (
                &SET_BLEND_PER_TARGET_ALPHA_OP,
                &[j],
                SET_BLEND_PER_TARGET_ALPHA_OP_V_OGL_FUNC_REVERSE_SUBTRACT,
            ),
            (
                &SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF,
                &[j],
                SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V_OGL_DST_COLOR,
            ),
            (
                &SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF,
                &[j],
                SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V_OGL_CONSTANT_ALPHA,
            ),
        ],
    );
    let view = mirror.three_d().unwrap();
    let target = view.blend_per_target(j).unwrap();
    assert_eq!(target.index(), j);
    let enables = [
        target.enable(),
        target.write_red(),
        target.write_green(),
        target.write_blue(),
        target.write_alpha(),
        target.separate_for_alpha(),
    ];
    assert_eq!(
        enables.map(name),
        ["TRUE", "TRUE", "FALSE", "TRUE", "FALSE", "TRUE"]
    );
    let ops = [
        target.color_op(),
        target.color_source_coeff(),
        target.color_dest_coeff(),
        target.alpha_op(),
        target.alpha_source_coeff(),
        target.alpha_dest_coeff(),
    ];
    let expected = [
        "OGL_FUNC_ADD",
        "OGL_ONE",
        "OGL_SRC_COLOR",
        "OGL_FUNC_REVERSE_SUBTRACT",
        "OGL_DST_COLOR",
        "OGL_CONSTANT_ALPHA",
    ];
    assert_eq!(ops.map(name), expected);
    assert_eq!(
        name(view.blend_per_target(j - 1).unwrap().enable()),
        "FALSE"
    );
}

#[test]
fn color_target_getters() {
    use clc797::*;
    let mut mirror = three_d();
    let j = 4;
    let memory = [
        (
            SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH,
            SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH_ONE_GOB,
        ),
        (
            SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT,
            SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_FOUR_GOBS,
        ),
        (
            SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH,
            SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_TWO_GOBS,
        ),
        (
            SET_COLOR_TARGET_MEMORY_LAYOUT,
            SET_COLOR_TARGET_MEMORY_LAYOUT_PITCH,
        ),
        (
            SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL,
            SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_DEPTH_SIZE,
        ),
    ];
    put_fields(&mut mirror, &SET_COLOR_TARGET_MEMORY, &[j], &memory);
    put_all(
        &mut mirror,
        &[
            (&SET_COLOR_TARGET_A, &[j], 0xffff_ff12),
            (&SET_COLOR_TARGET_B, &[j], 0x3456_7800),
            (&SET_COLOR_TARGET_WIDTH, &[j], 1280),
            (&SET_COLOR_TARGET_HEIGHT, &[j], 720),
            (
                &SET_COLOR_TARGET_FORMAT,
                &[j],
                SET_COLOR_TARGET_FORMAT_V_A8B8G8R8,
            ),
            (&SET_COLOR_TARGET_THIRD_DIMENSION, &[j], 3),
            (&SET_COLOR_TARGET_ARRAY_PITCH, &[j], 0x1000),
            (&SET_COLOR_TARGET_LAYER, &[j], 2),
            (&SET_COLOR_TARGET_WIDTH, &[j + 1], 1),
        ],
    );
    let view = mirror.three_d().unwrap();
    let target = view.color_target(j).unwrap();
    assert_eq!(target.index(), j);
    // The upper address field is 7:0; the bits above it are ignored.
    assert_eq!(target.address(), Ok(0x12_3456_7800));
    let sizes = [
        target.width(),
        target.height(),
        target.third_dimension(),
        target.array_pitch(),
        target.layer(),
    ];
    assert_eq!(sizes.map(Result::unwrap), [1280, 720, 3, 0x1000, 2]);
    let named = [
        target.format(),
        target.block_width(),
        target.block_height(),
        target.block_depth(),
        target.layout(),
        target.third_dimension_control(),
    ];
    let expected = [
        "A8B8G8R8",
        "ONE_GOB",
        "FOUR_GOBS",
        "TWO_GOBS",
        "PITCH",
        "THIRD_DIMENSION_DEFINES_DEPTH_SIZE",
    ];
    assert_eq!(named.map(name), expected);
}

#[test]
fn zt_getters() {
    use clc797::*;
    let mut mirror = three_d();
    let block = [
        (SET_ZT_BLOCK_SIZE_WIDTH, SET_ZT_BLOCK_SIZE_WIDTH_ONE_GOB),
        (
            SET_ZT_BLOCK_SIZE_HEIGHT,
            SET_ZT_BLOCK_SIZE_HEIGHT_EIGHT_GOBS,
        ),
        (SET_ZT_BLOCK_SIZE_DEPTH, SET_ZT_BLOCK_SIZE_DEPTH_ONE_GOB),
    ];
    put_fields(&mut mirror, &SET_ZT_BLOCK_SIZE, &[], &block);
    let size_c = [
        (SET_ZT_SIZE_C_THIRD_DIMENSION, 5),
        (
            SET_ZT_SIZE_C_CONTROL,
            SET_ZT_SIZE_C_CONTROL_ARRAY_SIZE_IS_ONE,
        ),
    ];
    put_fields(&mut mirror, &SET_ZT_SIZE_C, &[], &size_c);
    put_all(
        &mut mirror,
        &[
            (&SET_ZT_A, &[], 0x34),
            (&SET_ZT_B, &[], 0x5000_0000),
            (&SET_ZT_FORMAT, &[], SET_ZT_FORMAT_V_Z24S8),
            (&SET_ZT_ARRAY_PITCH, &[], 0x2000),
            (&SET_ZT_SIZE_A, &[], 800),
            (&SET_ZT_SIZE_B, &[], 600),
            (&SET_ZT_LAYER, &[], 6),
            (&SET_ZT_SELECT, &[], 1),
        ],
    );
    let view = mirror.three_d().unwrap();
    let zt = view.zt();
    assert_eq!(zt.address(), Ok(0x34_5000_0000));
    let named = [
        zt.format(),
        zt.block_width(),
        zt.block_height(),
        zt.block_depth(),
        zt.size_control(),
    ];
    assert_eq!(
        named.map(name),
        [
            "Z24S8",
            "ONE_GOB",
            "EIGHT_GOBS",
            "ONE_GOB",
            "ARRAY_SIZE_IS_ONE"
        ]
    );
    let numbers = [
        zt.array_pitch(),
        zt.width(),
        zt.height(),
        zt.third_dimension(),
        zt.layer(),
        zt.target_count(),
    ];
    assert_eq!(numbers.map(Result::unwrap), [0x2000, 800, 600, 5, 6, 1]);
}

#[test]
fn vertex_stream_getters_read_the_half_of_their_index() {
    use clc797::*;
    let mut mirror = three_d();
    let a = 3;
    let b = VERTEX_STREAMS_PER_HALF + a;
    put_fields(
        &mut mirror,
        &SET_VERTEX_STREAM_A_FORMAT,
        &[a],
        &[
            (SET_VERTEX_STREAM_A_FORMAT_STRIDE, 16),
            (
                SET_VERTEX_STREAM_A_FORMAT_ENABLE,
                SET_VERTEX_STREAM_A_FORMAT_ENABLE_TRUE,
            ),
        ],
    );
    put_fields(
        &mut mirror,
        &SET_VERTEX_STREAM_B_FORMAT,
        &[a],
        &[
            (SET_VERTEX_STREAM_B_FORMAT_STRIDE, 32),
            (
                SET_VERTEX_STREAM_B_FORMAT_ENABLE,
                SET_VERTEX_STREAM_B_FORMAT_ENABLE_FALSE,
            ),
        ],
    );
    put(&mut mirror, &SET_VERTEX_STREAM_A_LOCATION_A, &[a], 0x01);
    put(&mut mirror, &SET_VERTEX_STREAM_A_LOCATION_B, &[a], 0x1000);
    put(&mut mirror, &SET_VERTEX_STREAM_B_LOCATION_A, &[a], 0x02);
    put(&mut mirror, &SET_VERTEX_STREAM_B_LOCATION_B, &[a], 0x2000);
    put(&mut mirror, &SET_VERTEX_STREAM_A_FREQUENCY, &[a], 5);
    put(&mut mirror, &SET_VERTEX_STREAM_B_FREQUENCY, &[a], 6);
    put(
        &mut mirror,
        &SET_VERTEX_STREAM_INSTANCE_A,
        &[a],
        SET_VERTEX_STREAM_INSTANCE_A_IS_INSTANCED_TRUE,
    );
    put(
        &mut mirror,
        &SET_VERTEX_STREAM_INSTANCE_B,
        &[a],
        SET_VERTEX_STREAM_INSTANCE_B_IS_INSTANCED_FALSE,
    );
    put(&mut mirror, &SET_VERTEX_STREAM_SIZE_A, &[a], 0x07);
    put(&mut mirror, &SET_VERTEX_STREAM_SIZE_B, &[a], 0x8000);

    let view = mirror.three_d().unwrap();
    let low = view.vertex_stream(a).unwrap();
    let high = view.vertex_stream(b).unwrap();
    assert_eq!((low.index(), high.index()), (a, b));
    assert_eq!((low.stride(), high.stride()), (Ok(16), Ok(32)));
    assert_eq!((name(low.enable()), name(high.enable())), ("TRUE", "FALSE"));
    assert_eq!(
        (low.address(), high.address()),
        (Ok(0x1_0000_1000), Ok(0x2_0000_2000))
    );
    assert_eq!((low.frequency(), high.frequency()), (Ok(5), Ok(6)));
    assert_eq!(
        (name(low.instanced()), name(high.instanced())),
        ("TRUE", "FALSE")
    );
    assert_eq!(low.size(), Ok(0x7_0000_8000));
}

#[test]
fn vertex_attribute_getters_read_the_half_of_their_index() {
    use clc797::*;
    let mut mirror = three_d();
    let a = 5;
    put_fields(
        &mut mirror,
        &SET_VERTEX_ATTRIBUTE_A,
        &[a],
        &[
            (SET_VERTEX_ATTRIBUTE_A_STREAM, 1),
            (
                SET_VERTEX_ATTRIBUTE_A_SOURCE,
                SET_VERTEX_ATTRIBUTE_A_SOURCE_ACTIVE,
            ),
            (SET_VERTEX_ATTRIBUTE_A_OFFSET, 12),
            (
                SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS,
                SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_R32_G32_B32,
            ),
            (
                SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE,
                SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_FLOAT,
            ),
            (
                SET_VERTEX_ATTRIBUTE_A_SWAP_R_AND_B,
                SET_VERTEX_ATTRIBUTE_A_SWAP_R_AND_B_FALSE,
            ),
        ],
    );
    put_fields(
        &mut mirror,
        &SET_VERTEX_ATTRIBUTE_B,
        &[a],
        &[
            (SET_VERTEX_ATTRIBUTE_B_STREAM, 2),
            (
                SET_VERTEX_ATTRIBUTE_B_SOURCE,
                SET_VERTEX_ATTRIBUTE_B_SOURCE_INACTIVE,
            ),
            (SET_VERTEX_ATTRIBUTE_B_OFFSET, 24),
            (
                SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS,
                SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS_R8_G8_B8_A8,
            ),
            (
                SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE,
                SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE_NUM_UNORM,
            ),
            (
                SET_VERTEX_ATTRIBUTE_B_SWAP_R_AND_B,
                SET_VERTEX_ATTRIBUTE_B_SWAP_R_AND_B_TRUE,
            ),
        ],
    );
    let view = mirror.three_d().unwrap();
    let low = view.vertex_attribute(a).unwrap();
    let high = view
        .vertex_attribute(VERTEX_ATTRIBUTES_PER_HALF + a)
        .unwrap();
    assert_eq!(
        (low.index(), high.index()),
        (a, VERTEX_ATTRIBUTES_PER_HALF + a)
    );
    assert_eq!((low.stream(), high.stream()), (Ok(1), Ok(2)));
    assert_eq!((low.offset(), high.offset()), (Ok(12), Ok(24)));
    assert_eq!(
        [
            low.source(),
            low.component_bit_widths(),
            low.numerical_type(),
            low.swap_r_and_b()
        ]
        .map(name),
        ["ACTIVE", "R32_G32_B32", "NUM_FLOAT", "FALSE"]
    );
    assert_eq!(
        [
            high.source(),
            high.component_bit_widths(),
            high.numerical_type(),
            high.swap_r_and_b()
        ]
        .map(name),
        ["INACTIVE", "R8_G8_B8_A8", "NUM_UNORM", "TRUE"]
    );
}

#[test]
fn index_buffer_pools_and_pipeline_getters() {
    use clc797::*;
    let mut mirror = three_d();
    put(&mut mirror, &SET_INDEX_BUFFER_A, &[], 0x11);
    put(&mut mirror, &SET_INDEX_BUFFER_B, &[], 0x2222_0000);
    put(&mut mirror, &SET_INDEX_BUFFER_SIZE_A, &[], 0x3);
    put(&mut mirror, &SET_INDEX_BUFFER_SIZE_B, &[], 0x4444);
    put(
        &mut mirror,
        &SET_INDEX_BUFFER_E,
        &[],
        SET_INDEX_BUFFER_E_INDEX_SIZE_TWO_BYTES,
    );
    put(&mut mirror, &SET_INDEX_BUFFER_F, &[], 55);
    put(&mut mirror, &SET_TEX_HEADER_POOL_A, &[], 0x05);
    put(&mut mirror, &SET_TEX_HEADER_POOL_B, &[], 0x6000);
    put(&mut mirror, &SET_TEX_HEADER_POOL_C, &[], 77);
    put(&mut mirror, &SET_TEX_SAMPLER_POOL_A, &[], 0x08);
    put(&mut mirror, &SET_TEX_SAMPLER_POOL_B, &[], 0x9000);
    put(&mut mirror, &SET_TEX_SAMPLER_POOL_C, &[], 99);
    let j = 4;
    put_fields(
        &mut mirror,
        &SET_PIPELINE_SHADER,
        &[j],
        &[
            (SET_PIPELINE_SHADER_ENABLE, SET_PIPELINE_SHADER_ENABLE_TRUE),
            (SET_PIPELINE_SHADER_TYPE, SET_PIPELINE_SHADER_TYPE_GEOMETRY),
        ],
    );
    put(&mut mirror, &SET_PIPELINE_PROGRAM_ADDRESS_A, &[j], 0x0a);
    put(&mut mirror, &SET_PIPELINE_PROGRAM_ADDRESS_B, &[j], 0xb000);
    put(&mut mirror, &SET_PIPELINE_REGISTER_COUNT, &[j], 40);
    put(&mut mirror, &SET_PIPELINE_BINDING, &[j], 3);

    let view = mirror.three_d().unwrap();
    let index = view.index_buffer();
    assert_eq!(index.address(), Ok(0x11_2222_0000));
    assert_eq!(index.size(), Ok(0x3_0000_4444));
    assert_eq!(name(index.index_size()), "TWO_BYTES");
    assert_eq!(index.first(), Ok(55));
    let pools = view.tex_pools();
    assert_eq!(pools.header_address(), Ok(0x5_0000_6000));
    assert_eq!(pools.header_maximum_index(), Ok(77));
    assert_eq!(pools.sampler_address(), Ok(0x8_0000_9000));
    assert_eq!(pools.sampler_maximum_index(), Ok(99));
    let pipeline = view.pipeline(j).unwrap();
    assert_eq!(pipeline.index(), j);
    assert_eq!(
        [pipeline.enable(), pipeline.shader_type()].map(name),
        ["TRUE", "GEOMETRY"]
    );
    assert_eq!(pipeline.program_address(), Ok(0xa_0000_b000));
    assert_eq!(pipeline.register_count(), Ok(40));
    assert_eq!(pipeline.binding_group(), Ok(3));
    assert_eq!(name(view.pipeline(j - 1).unwrap().enable()), "FALSE");
}

#[test]
fn rasterizer_getters() {
    use clc797::*;
    let mut mirror = three_d();
    let origin = [
        (SET_WINDOW_ORIGIN_MODE, SET_WINDOW_ORIGIN_MODE_LOWER_LEFT),
        (SET_WINDOW_ORIGIN_FLIP_Y, SET_WINDOW_ORIGIN_FLIP_Y_TRUE),
    ];
    put_fields(&mut mirror, &SET_WINDOW_ORIGIN, &[], &origin);
    put_all(
        &mut mirror,
        &[
            (&SET_FRONT_POLYGON_MODE, &[], SET_FRONT_POLYGON_MODE_V_LINE),
            (&SET_BACK_POLYGON_MODE, &[], SET_BACK_POLYGON_MODE_V_POINT),
            (&OGL_SET_CULL, &[], OGL_SET_CULL_ENABLE_TRUE),
            (&OGL_SET_CULL_FACE, &[], OGL_SET_CULL_FACE_V_FRONT_AND_BACK),
            (&OGL_SET_FRONT_FACE, &[], OGL_SET_FRONT_FACE_V_CCW),
            (&SET_RASTER_ENABLE, &[], SET_RASTER_ENABLE_V_TRUE),
            (
                &SET_POLY_OFFSET_POINT,
                &[],
                SET_POLY_OFFSET_POINT_ENABLE_TRUE,
            ),
            (
                &SET_POLY_OFFSET_LINE,
                &[],
                SET_POLY_OFFSET_LINE_ENABLE_FALSE,
            ),
            (&SET_POLY_OFFSET_FILL, &[], SET_POLY_OFFSET_FILL_ENABLE_TRUE),
            (&SET_PROVOKING_VERTEX, &[], SET_PROVOKING_VERTEX_V_LAST),
            (&SET_DEPTH_BIAS, &[], 1.0f32.to_bits()),
            (&SET_SLOPE_SCALE_DEPTH_BIAS, &[], 2.0f32.to_bits()),
            (&SET_DEPTH_BIAS_CLAMP, &[], 3.0f32.to_bits()),
            (&SET_LINE_WIDTH_FLOAT, &[], 4.0f32.to_bits()),
            (&SET_ALIASED_LINE_WIDTH_FLOAT, &[], 5.0f32.to_bits()),
            (&SET_POINT_SIZE, &[], 6.0f32.to_bits()),
        ],
    );
    let view = mirror.three_d().unwrap();
    let raster = view.rasterizer();
    let named = [
        raster.front_polygon_mode(),
        raster.back_polygon_mode(),
        raster.cull_enable(),
        raster.cull_face(),
        raster.front_face(),
        raster.window_origin_mode(),
        raster.window_origin_flip_y(),
        raster.raster_enable(),
        raster.poly_offset_point(),
        raster.poly_offset_line(),
        raster.poly_offset_fill(),
        raster.provoking_vertex(),
    ];
    let expected = [
        "LINE",
        "POINT",
        "TRUE",
        "FRONT_AND_BACK",
        "CCW",
        "LOWER_LEFT",
        "TRUE",
        "TRUE",
        "TRUE",
        "FALSE",
        "TRUE",
        "LAST",
    ];
    assert_eq!(named.map(name), expected);
    let floats = [
        raster.depth_bias(),
        raster.slope_scale_depth_bias(),
        raster.depth_bias_clamp(),
        raster.line_width(),
        raster.aliased_line_width(),
        raster.point_size(),
    ];
    assert_eq!(floats.map(Result::unwrap), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// A 3D mirror with distinct depth and stencil state.
fn depth_stencil_mirror() -> Mirror {
    use clc797::*;
    let mut mirror = three_d();
    put_all(
        &mut mirror,
        &[
            (&SET_DEPTH_TEST, &[], SET_DEPTH_TEST_ENABLE_TRUE),
            (&SET_DEPTH_WRITE, &[], SET_DEPTH_WRITE_ENABLE_FALSE),
            (&SET_DEPTH_FUNC, &[], SET_DEPTH_FUNC_V_OGL_LEQUAL),
            (
                &SET_DEPTH_BOUNDS_TEST,
                &[],
                SET_DEPTH_BOUNDS_TEST_ENABLE_TRUE,
            ),
            (&SET_DEPTH_BOUNDS_MIN, &[], 0.125f32.to_bits()),
            (&SET_DEPTH_BOUNDS_MAX, &[], 0.875f32.to_bits()),
            (&SET_STENCIL_TEST, &[], SET_STENCIL_TEST_ENABLE_TRUE),
            (&SET_STENCIL_OP_FAIL, &[], SET_STENCIL_OP_FAIL_V_OGL_ZERO),
            (
                &SET_STENCIL_OP_ZFAIL,
                &[],
                SET_STENCIL_OP_ZFAIL_V_OGL_REPLACE,
            ),
            (&SET_STENCIL_OP_ZPASS, &[], SET_STENCIL_OP_ZPASS_V_OGL_INCR),
            (&SET_STENCIL_FUNC, &[], SET_STENCIL_FUNC_V_OGL_GREATER),
            (&SET_STENCIL_FUNC_REF, &[], 1),
            (&SET_STENCIL_FUNC_MASK, &[], 2),
            (&SET_STENCIL_MASK, &[], 3),
            (
                &SET_TWO_SIDED_STENCIL_TEST,
                &[],
                SET_TWO_SIDED_STENCIL_TEST_ENABLE_TRUE,
            ),
            (
                &SET_BACK_STENCIL_OP_FAIL,
                &[],
                SET_BACK_STENCIL_OP_FAIL_V_OGL_INVERT,
            ),
            (
                &SET_BACK_STENCIL_OP_ZFAIL,
                &[],
                SET_BACK_STENCIL_OP_ZFAIL_V_OGL_DECR,
            ),
            (
                &SET_BACK_STENCIL_OP_ZPASS,
                &[],
                SET_BACK_STENCIL_OP_ZPASS_V_OGL_KEEP,
            ),
            (
                &SET_BACK_STENCIL_FUNC,
                &[],
                SET_BACK_STENCIL_FUNC_V_OGL_NOTEQUAL,
            ),
            (&SET_BACK_STENCIL_FUNC_REF, &[], 4),
            (&SET_BACK_STENCIL_FUNC_MASK, &[], 5),
            (&SET_BACK_STENCIL_MASK, &[], 6),
        ],
    );
    mirror
}

#[test]
fn depth_stencil_named_getters() {
    let mirror = depth_stencil_mirror();
    let view = mirror.three_d().unwrap();
    let ds = view.depth_stencil();
    let named = [
        ds.depth_test(),
        ds.depth_write(),
        ds.depth_func(),
        ds.depth_bounds_test(),
        ds.stencil_test(),
        ds.stencil_op_fail(),
        ds.stencil_op_zfail(),
        ds.stencil_op_zpass(),
        ds.stencil_func(),
        ds.two_sided_stencil_test(),
        ds.back_stencil_op_fail(),
        ds.back_stencil_op_zfail(),
        ds.back_stencil_op_zpass(),
        ds.back_stencil_func(),
    ];
    let expected = [
        "TRUE",
        "FALSE",
        "OGL_LEQUAL",
        "TRUE",
        "TRUE",
        "OGL_ZERO",
        "OGL_REPLACE",
        "OGL_INCR",
        "OGL_GREATER",
        "TRUE",
        "OGL_INVERT",
        "OGL_DECR",
        "OGL_KEEP",
        "OGL_NOTEQUAL",
    ];
    assert_eq!(named.map(name), expected);
}

#[test]
fn depth_stencil_number_getters() {
    let mirror = depth_stencil_mirror();
    let view = mirror.three_d().unwrap();
    let ds = view.depth_stencil();
    let bounds = (ds.depth_bounds_min(), ds.depth_bounds_max());
    assert_eq!(bounds, (Ok(0.125), Ok(0.875)));
    let masks = [
        ds.stencil_func_ref(),
        ds.stencil_func_mask(),
        ds.stencil_mask(),
        ds.back_stencil_func_ref(),
        ds.back_stencil_func_mask(),
        ds.back_stencil_mask(),
    ];
    assert_eq!(masks.map(Result::unwrap), [1, 2, 3, 4, 5, 6]);
}

#[test]
fn compute_view_reads_the_compute_class() {
    use clc7c0::*;
    let mut mirror = compute();
    put(&mut mirror, &SET_TEX_HEADER_POOL_A, &[], 0x01);
    put(&mut mirror, &SET_TEX_HEADER_POOL_B, &[], 0x2000);
    put(&mut mirror, &SET_TEX_HEADER_POOL_C, &[], 3);
    put(&mut mirror, &SET_TEX_SAMPLER_POOL_A, &[], 0x04);
    put(&mut mirror, &SET_TEX_SAMPLER_POOL_B, &[], 0x5000);
    put(&mut mirror, &SET_TEX_SAMPLER_POOL_C, &[], 6);
    put(&mut mirror, &SEND_PCAS_A, &[], 0x1234);
    let view = mirror.compute().unwrap();
    let pools = view.tex_pools();
    assert_eq!(pools.header_address(), Ok(0x1_0000_2000));
    assert_eq!(pools.header_maximum_index(), Ok(3));
    assert_eq!(pools.sampler_address(), Ok(0x4_0000_5000));
    assert_eq!(pools.sampler_maximum_index(), Ok(6));
    // The register slice is indexed by method number, as the QMD scheduler reads it.
    let pcas = usize::try_from(SEND_PCAS_A.method()).unwrap();
    assert_eq!(view.registers()[pcas], 0x1234);
    assert_eq!(view.registers(), mirror.registers());
}

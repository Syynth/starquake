//! Tests of the register reset values (`regs::defaults`): the values against the references,
//! written out; the front file, its shadow file and a new mirror starting alike; and same-value
//! dropping against a non-zero reset value.

use std::collections::BTreeMap;

use super::*;
use crate::back::state::Mirror;

const TWO_D_CLASS: ClassId = ClassId(cl902d::CLASS);
const DMA_CLASS: ClassId = ClassId(clc7b5::CLASS);
const INLINE_CLASS: ClassId = ClassId(cla140::CLASS);

/// Every class with rules of its own, the two `OTHER` engine classes and two unknown ids.
const CLASSES: [ClassId; 7] = [
    THREE_D_CLASS,
    COMPUTE_CLASS,
    TWO_D_CLASS,
    DMA_CLASS,
    INLINE_CLASS,
    ClassId(0),
    ClassId(u32::MAX),
];

/// The non-zero registers of a new file of `class`, by byte address, read through the
/// production constructor.
fn non_zero(class: ClassId) -> BTreeMap<u32, u32> {
    let file = RegFile::new(CHANNEL, class);
    (0..file.rules().registers())
        .filter_map(|method| {
            let value = file.get(method).unwrap();
            (value != 0).then_some((method * 4, value))
        })
        .collect()
}

/// Every reset value, written out from the references rather than from the shipping table:
/// byte addresses from Eden-NX2's `ASSERT_REG_POSITION` lines and struct layouts
/// (`eden-nx2/src/video_core/engines/maxwell_3d.h`, which agree with `clc797.h` at each address),
/// values from `Maxwell3D::InitializeRegisterDefaults` (`maxwell_3d.cpp`) with Eden's enum values
/// and 1.0 as 0x3F800000, and the 2D depths from the `Fermi2D` constructor (`fermi_2d.cpp`, at
/// `cl902d.h`'s `NV902D_SET_DST_DEPTH` 0x020c and `NV902D_SET_SRC_DEPTH` 0x023c). Every other
/// register of every class is zero.
#[test]
fn reset_values_are_the_references_written_out() {
    let mut three_d = BTreeMap::new();
    let mut set = |address: u32, value: u32| {
        assert_eq!(three_d.insert(address, value), None, "{address:#x} twice");
    };
    for i in 0..16 {
        // `viewports[i].depth_range_far` (0x0C00, 0x10 apart, at +0xC): 1.0f.
        set(0x0c0c + 0x10 * i, 0x3f80_0000);
        // `viewport_transform[i].swizzle` (0x0A00, 0x20 apart, at +0x18): PositiveX (0) at
        // bit 0, PositiveY (2) at bit 4, PositiveZ (4) at bit 8, PositiveW (6) at bit 12.
        set(0x0a18 + 0x20 * i, 0x6420);
    }
    // `blend` (0x133C): color_op Add_D3D, color_source One_D3D, color_dest Zero_D3D, alpha_op
    // Add_D3D, alpha_source One_D3D, (enable_global_color_key left 0), alpha_dest Zero_D3D.
    set(0x1340, 1);
    set(0x1344, 2);
    set(0x1348, 1);
    set(0x134c, 1);
    set(0x1350, 2);
    set(0x1358, 1);
    for i in 0..8 {
        // `blend_per_target[i]` (0x1E00, 0x20 apart): the same six from +4.
        let base = 0x1e00 + 0x20 * i;
        set(base + 0x04, 1);
        set(base + 0x08, 2);
        set(base + 0x0c, 1);
        set(base + 0x10, 1);
        set(base + 0x14, 2);
        set(base + 0x18, 1);
    }
    // `stencil_front_op` (0x1384): Keep_D3D three times, then Always_GL.
    set(0x1384, 1);
    set(0x1388, 1);
    set(0x138c, 1);
    set(0x1390, 0x207);
    set(0x1398, 0xffff_ffff); // stencil_front_func_mask
    set(0x139c, 0xffff_ffff); // stencil_front_mask
    set(0x1594, 1); // stencil_two_side_enable
    // `stencil_back_op` (0x1598): Keep_D3D three times, then Always_GL.
    set(0x1598, 1);
    set(0x159c, 1);
    set(0x15a0, 1);
    set(0x15a4, 0x207);
    set(0x0f5c, 0xffff_ffff); // stencil_back_func_mask
    set(0x0f58, 0xffff_ffff); // stencil_back_mask
    set(0x130c, 0x207); // depth_test_func: Always_GL
    set(0x191c, 0x900); // gl_front_face: CounterClockWise (0x901), then ClockWise (0x900)
    set(0x1920, 0x405); // gl_cull_face: Back
    set(0x1518, 0x3f80_0000); // point_size: 1.0f
    for i in 0..8 {
        // `color_mask[i]` (0x1A00, 4 apart): R, G, B, A (bits 0, 4, 8, 12) all 1.
        set(0x1a00 + 4 * i, 0x1111);
    }
    for i in 0..32 {
        // `vertex_attrib_format[i]` (0x1160, 4 apart, 32 of them): `constant` (bit 6) 1.
        set(0x1160 + 4 * i, 0x40);
    }
    set(0x037c, 1); // rasterize_enable
    set(0x0fac, 1); // color_target_mrt_enable
    set(0x15b8, 1); // framebuffer_srgb
    set(0x13b4, 0x3f80_0000); // line_width_aliased: 1.0f
    set(0x13b0, 0x3f80_0000); // line_width_smooth: 1.0f
    set(0x0db0, 0x1b02); // polygon_mode_back: Fill
    set(0x0dac, 0x1b02); // polygon_mode_front: Fill

    assert_eq!(three_d.len(), 150);
    assert_eq!(non_zero(THREE_D_CLASS), three_d);
    assert_eq!(
        non_zero(TWO_D_CLASS),
        BTreeMap::from([(0x020c, 1), (0x023c, 1)])
    );
    // Compute, DMA copy and inline-to-memory set nothing, nor does an unknown class.
    for class in [
        COMPUTE_CLASS,
        DMA_CLASS,
        INLINE_CLASS,
        ClassId(0),
        ClassId(u32::MAX),
    ] {
        assert_eq!(non_zero(class), BTreeMap::new(), "{:#x}", class.0);
    }
}

/// The lists name each register once, inside the file, with a non-zero value (a zero would
/// repeat the file's own start), as methods of the class's generated header whose arrays have
/// Mesa's lengths and run into no other method.
#[test]
fn reset_values_name_registers_of_the_file_once() {
    let lists: [(&ClassRules, &crate::classes::ClassHeader); 2] =
        [(&THREE_D, &clc797::HEADER), (&TWO_D, &cl902d::HEADER)];
    for (rules, header) in lists {
        let firsts: BTreeSet<u32> = header.methods.iter().map(Method::method).collect();
        let mut seen = BTreeSet::new();
        for default in rules.defaults() {
            let method = default.span.method();
            assert!(header.methods.contains(method), "{}", method.name);
            assert_ne!(default.value, 0, "{}", method.name);
            let expected_count = match method.strides {
                [] => 1,
                _ => mesa_array_size(method.name).unwrap(),
            };
            assert_eq!(default.span.count(), expected_count, "{}", method.name);
            let numbers: Vec<u32> = default.span.numbers().collect();
            assert_eq!(numbers.len(), usize::try_from(expected_count).unwrap());
            for (index, &number) in numbers.iter().enumerate() {
                assert!(number < rules.registers(), "{} {number:#x}", method.name);
                assert!(seen.insert(number), "{number:#x} named twice");
                if index > 0 {
                    assert!(!firsts.contains(&number), "{} runs on", method.name);
                }
            }
        }
    }
    for rules in [&COMPUTE, &OTHER] {
        assert_eq!(rules.defaults(), &[]);
    }
}

#[test]
fn defaults_are_picked_with_the_class() {
    assert_eq!(THREE_D.defaults(), DEFAULTS_3D);
    assert_eq!(TWO_D.defaults(), DEFAULTS_2D);
    // The 2D class is otherwise the other classes' rules.
    assert_eq!(TWO_D.registers(), OTHER.registers());
    assert_eq!(TWO_D.shadow_control(), None);
    assert_eq!(TWO_D.data_loads(), &[]);
    assert_eq!(TWO_D.state(), &[]);
    assert_eq!(TWO_D.side_effects(), &[]);
}

/// The front's register file, its shadow file (for 3D) and a new mirror hold the same value in
/// every register at start, for every class.
#[test]
fn front_shadow_and_mirror_start_from_the_same_values() {
    for class in CLASSES {
        let front = RegFile::new(CHANNEL, class);
        let mirror = Mirror::new(CHANNEL, class);
        let registers = front.rules().registers();
        assert_eq!(
            mirror.registers().len(),
            usize::try_from(registers).unwrap()
        );
        for method in 0..registers {
            let value = front.get(method).unwrap();
            assert_eq!(
                mirror.value(method),
                Some(value),
                "{:#x} {method:#x}",
                class.0
            );
            if class == THREE_D_CLASS {
                assert_eq!(front.shadow(method), Some(value), "shadow {method:#x}");
            }
        }
    }
    // Not vacuous: a 3D, a 2D and a shadow value are the references' non-zero defaults.
    let front = three_d();
    let mirror = Mirror::new(CHANNEL, THREE_D_CLASS);
    let depth_func = clc797::SET_DEPTH_FUNC.method();
    assert_eq!(front.get(depth_func), Some(0x207));
    assert_eq!(front.shadow(depth_func), Some(0x207));
    assert_eq!(mirror.value(depth_func), Some(0x207));
    let src_depth = cl902d::SET_SRC_DEPTH.method();
    assert_eq!(RegFile::new(CHANNEL, TWO_D_CLASS).get(src_depth), Some(1));
    assert_eq!(Mirror::new(CHANNEL, TWO_D_CLASS).value(src_depth), Some(1));
    // No reset value touches the shadow control: the shadow RAM starts tracking.
    assert_eq!(front.shadow_mode(), Some(ShadowMode::Track));
}

/// Writes `word` to `method` of `front` and applies what it sent to `mirror`.
fn send_through(
    front: &mut RegFile,
    mirror: &mut Mirror,
    method: u32,
    word: u32,
) -> Vec<RingEntry<'static>> {
    let sent = send(front, method, word);
    for entry in &sent {
        assert!(mirror.apply(entry).unwrap());
    }
    sent
}

/// A write equal to a non-zero reset value is dropped, the back already holding it; a write of
/// zero to the same register is sent, and so is the reset value after it.
#[test]
fn a_write_equal_to_a_reset_value_is_dropped_and_zero_is_sent() {
    let depth_func = clc797::SET_DEPTH_FUNC.method();
    let always = clc797::SET_DEPTH_FUNC_V_OGL_ALWAYS;
    let mut front = three_d();
    let mut mirror = Mirror::new(CHANNEL, THREE_D_CLASS);
    assert!(front.is_state(depth_func));
    assert_eq!(
        send_through(&mut front, &mut mirror, depth_func, always),
        []
    );
    assert_eq!(mirror.value(depth_func), Some(always));
    assert_eq!(
        send_through(&mut front, &mut mirror, depth_func, 0),
        [write(depth_func, 0)]
    );
    assert_eq!(mirror.value(depth_func), Some(0));
    assert_eq!(
        send_through(&mut front, &mut mirror, depth_func, always),
        [write(depth_func, always)]
    );
    assert_eq!(mirror.value(depth_func), Some(always));
    assert_eq!(front.get(depth_func), Some(always));

    // A whole-word default (a value at or above 2^31): the same, and a write of only the
    // header's field (7:0) is not the same value.
    let mask = clc797::SET_STENCIL_MASK.method();
    assert!(front.is_state(mask));
    assert_eq!(send_through(&mut front, &mut mirror, mask, u32::MAX), []);
    let field = clc797::SET_STENCIL_MASK_V.max();
    assert_eq!(
        send_through(&mut front, &mut mirror, mask, field),
        [write(mask, field)]
    );
    assert_eq!(mirror.value(mask), front.get(mask));

    // The 2D class drops nothing: a write equal to its reset depth is still sent.
    let src_depth = cl902d::SET_SRC_DEPTH.method();
    let mut two_d = RegFile::new(CHANNEL, TWO_D_CLASS);
    assert!(!two_d.is_state(src_depth));
    assert_eq!(
        send(&mut two_d, src_depth, 1),
        [write_in(TWO_D_CLASS, src_depth, 1)]
    );
}

/// In replay mode from the start, a state write takes the shadow file's reset value, which equals
/// the register, so it is dropped; a register with no reset value takes zero.
#[test]
fn replay_at_start_takes_the_reset_value() {
    let mut file = three_d();
    send(&mut file, control(), REPLAY);
    assert_eq!(file.shadow_mode(), Some(ShadowMode::Replay));
    let depth_func = clc797::SET_DEPTH_FUNC.method();
    assert_eq!(send(&mut file, depth_func, 5), []);
    assert_eq!(file.get(depth_func), Some(0x207));
    assert_eq!(file.shadow(depth_func), Some(0x207));
    // `SET_ALPHA_REF` has no reset value: the shadow's zero, dropped too.
    assert_eq!(send(&mut file, state(), 5), []);
    assert_eq!(file.get(state()), Some(0));
}

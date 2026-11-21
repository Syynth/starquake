//! Clears through the executor (slice G24): a `CLEAR_SURFACE` word reaches the draw path's
//! clear, after the CPU-write drain, and a clear, masked or not, is a clear command that sets no
//! render targets.

use super::*;
use crate::back::draw::clear::ClearSurface;
use starquake_render::pipeline::ColorWrites;

/// A word to `CLEAR_SURFACE` on [`CHANNEL`].
fn clear(surface: ClearSurface) -> String {
    write(
        THREE_D,
        &clc797::CLEAR_SURFACE,
        &[],
        surface.encode().unwrap(),
    )
}

/// A clear of `color` channels of colour target 0 and, when `stencil`, the stencil.
fn surface(color: ColorWrites, stencil: bool) -> ClearSurface {
    ClearSurface {
        depth: false,
        stencil,
        color,
        target: 0,
        layer: 0,
    }
}

/// The 3D class bound on [`CHANNEL`] and [`target_state`]'s colour target.
fn clear_state() -> String {
    "bind channel=3 subchannel=0 class=0xc797\n".to_owned() + &target_state()
}

/// A depth and stencil target at [`DST`], [`SIZE`] square, with a partial stencil mask.
fn depth_state() -> String {
    let c = THREE_D;
    address(c, &clc797::SET_ZT_A, &clc797::SET_ZT_B, &[], DST)
        + &write(
            c,
            &clc797::SET_ZT_FORMAT,
            &[],
            clc797::SET_ZT_FORMAT_V_ZF32_X24S8,
        )
        + &write(c, &clc797::SET_ZT_SIZE_A, &[], SIZE)
        + &write(c, &clc797::SET_ZT_SIZE_B, &[], SIZE)
        + &write(
            c,
            &clc797::SET_ZT_SIZE_C,
            &[],
            fields(&[(
                clc797::SET_ZT_SIZE_C_CONTROL,
                clc797::SET_ZT_SIZE_C_CONTROL_ARRAY_SIZE_IS_ONE,
            )]),
        )
        + &write(c, &clc797::SET_ZT_SELECT, &[], 1)
        + &write(c, &clc797::SET_STENCIL_MASK, &[], 0x0f)
}

fn targets_set(commands: &[Command]) -> usize {
    commands
        .iter()
        .filter(|c| matches!(c, Command::SetRenderTargets(_)))
        .count()
}

fn clears(commands: &[Command]) -> usize {
    commands
        .iter()
        .filter(|c| matches!(c, Command::Clear(_)))
        .count()
}

#[test]
fn a_clear_surface_word_reaches_the_clear_path() {
    let mut rig = Rig::new();
    rig.run(&(clear_state() + &clear(surface(ColorWrites::ALL, false))));
    rig.submit();
    let commands = rig.commands();
    assert_eq!(clears(&commands), 1, "{commands:?}");
    let counters = rig.exec.draws(CHANNEL).unwrap().clear_counters();
    assert_eq!((counters.words, counters.cleared), (1, 1));
}

#[test]
fn a_clear_drains_cpu_writes_first() {
    let page = |va: u64| PageRange::new(va - va % PAGE_SIZE, PAGE_SIZE).unwrap();
    let mut rig = Rig::new();
    rig.run(&clear_state());
    cpu_write(&rig.watch, UNIFORMS);
    assert_eq!(rig.log().invalidated, vec![]);
    rig.run(&clear(surface(ColorWrites::ALL, false)));
    assert_eq!(rig.log().invalidated, vec![page(UNIFORMS)]);
}

/// The clear commands appended, in order.
fn clear_descs(commands: &[Command]) -> Vec<starquake_render::command::ClearDesc> {
    commands
        .iter()
        .filter_map(|c| match c {
            Command::Clear(d) => Some(*d),
            _ => None,
        })
        .collect()
}

#[test]
fn a_draw_after_a_masked_clear_keeps_its_targets() {
    let mut rig = Rig::new();
    rig.run(&(draw_state() + &depth_state() + &draw()));
    // A masked stencil clear is a clear command through the mask, not a draw.
    rig.run(&clear(surface(ColorWrites::EMPTY, true)));
    let counters = rig.exec.draws(CHANNEL).unwrap().clear_counters();
    assert_eq!(counters.cleared, 1);
    rig.run(&draw());
    rig.submit();
    let commands = rig.commands();
    assert_eq!(draws(&commands), 2);
    assert_eq!(targets_set(&commands), 1, "{commands:?}");
    let cleared = clear_descs(&commands);
    assert_eq!(cleared.len(), 1);
    assert_eq!(cleared[0].stencil_mask, 0x0f);
}

#[test]
fn a_masked_clear_on_another_channel_leaves_the_next_draws_targets() {
    let mut rig = Rig::new();
    let other_state = clear_state().replace("channel=3 ", "channel=9 ");
    rig.run(&(draw_state() + &other_state + &draw()));
    let masked = clear(surface(ColorWrites::RED, false)).replace("channel=3 ", "channel=9 ");
    rig.run(&masked);
    let other = rig.exec.draws(OTHER).unwrap().clear_counters();
    assert_eq!(other.cleared, 1);
    rig.run(&draw());
    rig.submit();
    let commands = rig.commands();
    assert_eq!(draws(&commands), 2);
    assert_eq!(targets_set(&commands), 1, "{commands:?}");
    let cleared = clear_descs(&commands);
    assert_eq!(cleared.len(), 1);
    assert_eq!(cleared[0].color_mask, ColorWrites::RED);
}

#[test]
fn a_whole_clear_between_draws_leaves_the_targets_set() {
    let mut rig = Rig::new();
    rig.run(&(draw_state() + &draw()));
    rig.run(&clear(surface(ColorWrites::ALL, false)));
    rig.run(&draw());
    rig.submit();
    let commands = rig.commands();
    assert_eq!(draws(&commands), 2);
    assert_eq!(clears(&commands), 1);
    assert_eq!(targets_set(&commands), 1, "{commands:?}");
}

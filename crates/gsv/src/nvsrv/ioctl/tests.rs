//! Ioctl numbers decoded field by field, and fixed arguments read and given back as `WrapFixed`
//! does.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};

use super::*;

/// Hand-written numbers, each field distinct, with what Eden's bit fields read from them
/// (`nvdata.h`: `union Ioctl`).
#[test]
fn hand_written_numbers_decode_field_by_field() {
    // (raw, command, group, size, in, out)
    let cases: [(u32, u32, u32, u32, bool, bool); 6] = [
        // nvmap's IocCreate as libnx sends it: `_NV_IOWR(0x01, 0x01, data)`, 8 bytes, both ways.
        (0xC008_0101, 0x01, 0x01, 0x8, true, true),
        // Each field distinct: out only, size 0x123, group 0x47, command 0x05.
        (0x8123_4705, 0x05, 0x47, 0x123, false, true),
        // In only, the widest size, group 0xFF, command 0xA6.
        (0x7FFF_FFA6, 0xA6, 0xFF, 0x3FFF, true, false),
        // Neither direction, the smallest size, group 0x48, command 0x1A.
        (0x0000_481A, 0x1A, 0x48, 0, false, false),
        // Every bit set.
        (0xFFFF_FFFF, 0xFF, 0xFF, 0x3FFF, true, true),
        // Nothing set.
        (0, 0, 0, 0, false, false),
    ];
    for (raw, command, group, size, is_in, is_out) in cases {
        let ioctl = Ioctl::from_raw(raw);
        assert_eq!(ioctl.raw(), raw);
        assert_eq!(
            (
                ioctl.command(),
                ioctl.group(),
                ioctl.size(),
                ioctl.is_in(),
                ioctl.is_out()
            ),
            (command, group, size, is_in, is_out),
            "{raw:#010x}"
        );
    }
}

/// The fields tile the 32 bits, in Eden's order, and the largest argument is the size field's
/// widest value.
#[test]
fn the_fields_tile_the_number() {
    assert_eq!(COMMAND_SHIFT + COMMAND_BITS, GROUP_SHIFT);
    assert_eq!(GROUP_SHIFT + GROUP_BITS, SIZE_SHIFT);
    assert_eq!(SIZE_SHIFT + SIZE_BITS, IN_BIT);
    assert_eq!(IN_BIT + 1, OUT_BIT);
    assert_eq!(OUT_BIT + 1, u32::BITS);
    assert_eq!(MAX_ARGUMENT_BYTES, 0x3FFF);
    assert_eq!(Ioctl::from_raw(u32::MAX).size(), MAX_ARGUMENT_BYTES);
}

fn config() -> ProptestConfig {
    Config {
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(0x5EED_0187),
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// A number laid out from five generated fields reads back as those fields: swapping any two
    /// reads (or two shifts) fails, since the fields' values differ in most cases and their
    /// widths in every case but command and group.
    #[test]
    fn a_number_reads_back_the_fields_it_was_built_from(
        command in 0..=u32::MAX >> (u32::BITS - COMMAND_BITS),
        group in 0..=u32::MAX >> (u32::BITS - GROUP_BITS),
        size in 0..=MAX_ARGUMENT_BYTES,
        is_in: bool,
        is_out: bool,
    ) {
        let raw = command << COMMAND_SHIFT
            | group << GROUP_SHIFT
            | size << SIZE_SHIFT
            | u32::from(is_in) << IN_BIT
            | u32::from(is_out) << OUT_BIT;
        let ioctl = Ioctl::from_raw(raw);
        prop_assert_eq!(
            (ioctl.command(), ioctl.group(), ioctl.size(), ioctl.is_in(), ioctl.is_out()),
            (command, group, size, is_in, is_out)
        );
    }
}

/// A command and a group that are never equal, so a swap of the two shows on every number.
#[test]
fn command_and_group_are_not_swapped() {
    for value in 0..=0xFF_u32 {
        let other = value ^ 0x5A;
        let ioctl = Ioctl::from_raw(other << GROUP_SHIFT | value << COMMAND_SHIFT);
        assert_eq!((ioctl.command(), ioctl.group()), (value, other));
    }
}

/// A short input leaves the bytes past its end zero; a long one is cut to the struct; what goes
/// back is the whole struct.
#[test]
fn a_fixed_argument_takes_what_fits_and_zeroes_the_rest() {
    let short = Fixed::<8>::from_input(&[1, 2, 3]);
    assert_eq!(short.into_output(), [1, 2, 3, 0, 0, 0, 0, 0]);
    let empty = Fixed::<4>::from_input(&[]);
    assert_eq!(empty.into_output(), [0; 4]);
    let long = Fixed::<4>::from_input(&[9, 8, 7, 6, 5, 4]);
    assert_eq!(long.into_output(), [9, 8, 7, 6]);
    let exact = Fixed::<4>::from_input(&[0xA, 0xB, 0xC, 0xD]);
    assert_eq!(exact.into_output(), [0xA, 0xB, 0xC, 0xD]);
}

/// Each accessor reads and writes little-endian at its offset and nowhere else.
#[test]
fn fields_are_little_endian_at_their_offsets() {
    let bytes: Vec<u8> = (0x10..0x20).collect();
    let mut args = Fixed::<16>::from_input(&bytes);
    assert_eq!(args.u8_at::<3>(), 0x13);
    assert_eq!(args.u32_at::<4>(), 0x1716_1514);
    assert_eq!(args.u64_at::<8>(), 0x1F1E_1D1C_1B1A_1918);
    args.set_u32::<0>(0xA3A2_A1A0);
    args.set_u64::<8>(0xB7B6_B5B4_B3B2_B1B0);
    assert_eq!(
        args.into_output(),
        [
            0xA0, 0xA1, 0xA2, 0xA3, 0x14, 0x15, 0x16, 0x17, 0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5,
            0xB6, 0xB7,
        ]
    );
}

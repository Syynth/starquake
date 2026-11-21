//! PR #4588's rows for the uniform group's forms
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97; each
//! row's four words, low word first), in [`super::FORMS`] order, with the bits each form
//! rejects and requires, written out (the module docs' divergence table). A twin's rejected bits
//! are its vector form's (`op::integer`'s rows: 62 and 63 the wide slot's modifiers, 72 and 73
//! source 0's, 74 and 75 the register slot's, 80 `LOP3`'s `.PAND`), which
//! `decode::tests::uniform::twin_rows_drop_the_vector_ops_bits` checks; its fixed bits are its
//! vector form's and bit 91.

use super::super::super::form::ReferenceRow;
use super::super::integer::bits;

/// `UMOV`'s bits 38..64, which `0xc82` rejects: past its six-bit register, and its modifiers.
const BITS_38_TO_64: u128 = ((1 << 26) - 1) << 38;

/// The rows.
pub(super) const ROWS: [ReferenceRow; 42] = [
    ReferenceRow::new(0x290, [0x3f3f_ffff, 0xc000_003f, 0x0fff_ef3f, 0xffff_fe00])
        .dropping(bits(&[62, 73]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x490, [0x3f3f_ffff, 0xffff_ffff, 0x0fff_ef3f, 0xffff_fe00])
        .dropping(bits(&[73]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x890, [0x3f3f_ffff, 0xffff_ffff, 0x0fff_ef3f, 0xffff_fe00])
        .dropping(bits(&[73]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x291, [0x3f3f_ffff, 0xc000_003f, 0x0f8f_ff3f, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x491, [0x3f3f_ffff, 0xffff_ffff, 0x0f8f_ff3f, 0xffff_fe00])
        .fixing(bits(&[91])),
    ReferenceRow::new(0x891, [0x3f3f_ffff, 0xffff_ffff, 0x0f8f_ff3f, 0xffff_fe00])
        .fixing(bits(&[91])),
    ReferenceRow::new(0x292, [0x3f3f_ffff, 0xc000_003f, 0x0f8f_ff3f, 0xffff_fe00])
        .dropping(bits(&[62, 63, 80]))
        .fixing(bits(&[87, 88, 89, 90, 91])),
    ReferenceRow::new(0x492, [0x3f3f_ffff, 0xffff_ffff, 0x0f8f_ff3f, 0xffff_fe00])
        .dropping(bits(&[80]))
        .fixing(bits(&[87, 88, 89, 90, 91])),
    ReferenceRow::new(0x892, [0x3f3f_ffff, 0xffff_ffff, 0x0f8f_ff3f, 0xffff_fe00])
        .dropping(bits(&[80]))
        .fixing(bits(&[87, 88, 89, 90, 91])),
    ReferenceRow::new(0x296, [0x3f3f_ffff, 0xc000_003f, 0x0800_0f3f, 0xffff_fe00])
        .dropping(bits(&[62, 63, 75]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x496, [0x3f3f_ffff, 0xffff_ffff, 0x0800_0f3f, 0xffff_fe00])
        .dropping(bits(&[75]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x896, [0x3f3f_ffff, 0xffff_ffff, 0x0800_0f3f, 0xffff_fe00])
        .dropping(bits(&[75]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x299, [0x3f3f_ffff, 0xc000_003f, 0x0801_1f3f, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x499, [0x3f3f_ffff, 0xffff_ffff, 0x0801_1f3f, 0xffff_fe00])
        .dropping(bits(&[72]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x899, [0x3f3f_ffff, 0xffff_ffff, 0x0801_1f3f, 0xffff_fe00])
        .dropping(bits(&[72]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x2a4, [0x3f3f_ffff, 0xc000_003f, 0x0f8e_0f3f, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(bits(&[81, 82, 83, 91])),
    ReferenceRow::new(0x4a4, [0x3f3f_ffff, 0xffff_ffff, 0x0f8e_0f3f, 0xffff_fe00])
        .fixing(bits(&[81, 82, 83, 91])),
    ReferenceRow::new(0x8a4, [0x3f3f_ffff, 0xffff_ffff, 0x0f8e_0f3f, 0xffff_fe00])
        .fixing(bits(&[81, 82, 83, 91])),
    ReferenceRow::new(0x2a5, [0x3f3f_ffff, 0xc000_003f, 0x0f8e_0f3f, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 74, 75]))
        .fixing(bits(&[81, 82, 83, 91])),
    ReferenceRow::new(0x4a5, [0x3f3f_ffff, 0xffff_ffff, 0x0f8e_0f3f, 0xffff_fe00])
        .dropping(bits(&[72, 74, 75]))
        .fixing(bits(&[81, 82, 83, 91])),
    ReferenceRow::new(0x8a5, [0x3f3f_ffff, 0xffff_ffff, 0x0f8e_0f3f, 0xffff_fe00])
        .dropping(bits(&[72, 74, 75]))
        .fixing(bits(&[81, 82, 83, 91])),
    ReferenceRow::new(0x28c, [0x3f00_ffff, 0xc000_003f, 0x0ffe_7ff0, 0xffff_fe00])
        .dropping(bits(&[62, 63]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x88c, [0x3f00_ffff, 0xffff_ffff, 0x0ffe_7ff0, 0xffff_fe00])
        .fixing(bits(&[91])),
    ReferenceRow::new(0x287, [0x3f3f_ffff, 0xc000_003f, 0x0f80_0300, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 73]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x887, [0x3f3f_ffff, 0xffff_ffff, 0x0f80_0300, 0xffff_fe00])
        .dropping(bits(&[72, 73]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x29a, [0x3f3f_ffff, 0xc000_003f, 0x0800_0b00, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x89a, [0x3f3f_ffff, 0xffff_ffff, 0x0800_0b00, 0xffff_fe00])
        .dropping(bits(&[72]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x29b, [0x3f3f_ffff, 0xc000_003f, 0x0800_0b00, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 73]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x89b, [0x3f3f_ffff, 0xffff_ffff, 0x0800_0b00, 0xffff_fe00])
        .dropping(bits(&[72, 73]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x2bd, [0x003f_ffff, 0xc000_003f, 0x080e_0600, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(bits(&[81, 82, 83, 91])),
    ReferenceRow::new(0x8bd, [0x003f_ffff, 0xffff_ffff, 0x080e_0600, 0xffff_fe00])
        .fixing(bits(&[81, 82, 83, 91])),
    ReferenceRow::new(0x2be, [0x003f_ffff, 0xc000_003f, 0x0800_0000, 0xffff_fe00])
        .dropping(bits(&[62, 63]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x8be, [0x003f_ffff, 0xffff_ffff, 0x0800_0000, 0xffff_fe00])
        .fixing(bits(&[91])),
    ReferenceRow::new(0x2bf, [0x003f_ffff, 0xc000_003f, 0x0800_0000, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(bits(&[91])),
    ReferenceRow::new(0x8bf, [0x003f_ffff, 0xffff_ffff, 0x0800_0000, 0xffff_fe00])
        .fixing(bits(&[91])),
    ReferenceRow::new(0x3c2, [0xff3f_ffff, 0x0000_0000, 0x000e_0000, 0xffff_fe00])
        .fixing(bits(&[81, 82, 83])),
    ReferenceRow::new(0x882, [0x003f_ffff, 0xffff_ffff, 0x0000_0000, 0xffff_fe00]),
    ReferenceRow::new(0xc82, [0x003f_ffff, 0xffff_ffff, 0x0800_0000, 0xffff_fe00])
        .dropping(BITS_38_TO_64)
        .fixing(bits(&[91])),
    ReferenceRow::new(0xab9, [0x3f3f_ffff, 0x07ff_ffc0, 0x0800_0e00, 0xffff_fe00]),
    ReferenceRow::new(0xabb, [0x3f3f_ffff, 0x07ff_ffc0, 0x0800_0e00, 0xffff_fe00])
        .fixing(bits(&[91])),
    ReferenceRow::new(0x8b8, [0x3f3f_ffff, 0xffff_ffc0, 0x0f80_0e3f, 0xffff_fe00])
        .fixing(bits(&[91])),
    ReferenceRow::new(0x89c, [0x00ff_ffff, 0x0000_0000, 0x07ff_fff7, 0xffff_fe00]),
];

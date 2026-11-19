//! Comparisons and predicates: every float comparison in each width, ordered and unordered,
//! with a NaN on either side; flush-to-zero on a comparison; signed zeros.

use super::{u1_of, value_of};

/// The bits of `1.0`, `2.0` and a NaN in each width, and the opcode suffix.
const WIDTHS: [(&str, [u64; 3], &str); 3] = [
    ("16", [0x3c00, 0x4000, 0x7e01], "f16"),
    ("32", [0x3f80_0000, 0x4000_0000, 0x7fc0_0001], "f32"),
    (
        "64",
        [
            0x3ff0_0000_0000_0000,
            0x4000_0000_0000_0000,
            0x7ff8_0000_0000_0001,
        ],
        "f64",
    ),
];

/// The value of `opcode` + `width` on the operands with bits `a` and `b` of type `ty`.
fn compare(opcode: &str, width: &str, ty: &str, a: u64, b: u64) -> bool {
    let body = if ty == "f16" {
        format!(
            "    %0 = BitCastF16U16 u16 {a:#x}
    %1 = BitCastF16U16 u16 {b:#x}
    %2 = {opcode}{width} %0, %1"
        )
    } else {
        format!(
            "    %0 = Identity {ty} {a:#x}
    %1 = Identity {ty} {b:#x}
    %2 = {opcode}{width} %0, %1"
        )
    };
    u1_of(value_of(&body, 2))
}

#[test]
fn every_float_comparison_in_every_width() {
    // Rows: the comparison, then its result for (1, 2), (2, 2), (2, 1), (NaN, 1) and (1, NaN),
    // written out for the ordered and the unordered form.
    let rows: [(&str, [bool; 5], [bool; 5]); 6] = [
        (
            "Equal",
            [false, true, false, false, false],
            [false, true, false, true, true],
        ),
        (
            "NotEqual",
            [true, false, true, false, false],
            [true, false, true, true, true],
        ),
        (
            "LessThan",
            [true, false, false, false, false],
            [true, false, false, true, true],
        ),
        (
            "GreaterThan",
            [false, false, true, false, false],
            [false, false, true, true, true],
        ),
        (
            "LessThanEqual",
            [true, true, false, false, false],
            [true, true, false, true, true],
        ),
        (
            "GreaterThanEqual",
            [false, true, true, false, false],
            [false, true, true, true, true],
        ),
    ];
    for (width, [one, two, nan], ty) in WIDTHS {
        let pairs = [(one, two), (two, two), (two, one), (nan, one), (one, nan)];
        for (name, ordered, unordered) in rows {
            for (prefix, expected) in [("FPOrd", ordered), ("FPUnord", unordered)] {
                let opcode = format!("{prefix}{name}");
                for (&(a, b), want) in pairs.iter().zip(expected) {
                    assert_eq!(
                        compare(&opcode, width, ty, a, b),
                        want,
                        "{opcode}{width} {a:#x} {b:#x}"
                    );
                }
            }
        }
    }
}

#[test]
fn zeros_compare_equal_and_ftz_flushes_operands() {
    let body = "    %0 = FPOrdEqual32 f32 0.0, f32 -0.0
    %1 = FPOrdEqual32 f32 0x1, f32 0.0
    %2 = FPOrdEqual32 f32 0x1, f32 0.0 {ftz}
    %3 = FPOrdLessThan32 f32 0.0, f32 0x1 {ftz}
    %4 = FPOrdLessThan32 f32 0.0, f32 0x1";
    let expected = [(0, true), (1, false), (2, true), (3, false), (4, true)];
    for (id, want) in expected {
        assert_eq!(u1_of(value_of(body, id)), want, "%{id}");
    }
}

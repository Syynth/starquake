//! Composites: construct, extract and insert for every component kind and length, with the
//! index at and past the last component; and the public constants and value helpers.

use super::{run, value_of};
use crate::interp::{Machine, PRED_SLOTS, REG_SLOTS, UNIFORM_PRED_SLOTS, UNIFORM_REG_SLOTS, Val};
use crate::ir::Type;

/// A component kind of the composite opcodes.
#[derive(Clone, Copy)]
enum Kind {
    U32,
    F16,
    F32,
    F64,
}

impl Kind {
    const ALL: [Self; 4] = [Self::U32, Self::F16, Self::F32, Self::F64];

    const fn name(self) -> &'static str {
        match self {
            Self::U32 => "U32",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::F64 => "F64",
        }
    }

    /// A distinct component for index `index`: its bits.
    fn bits(self, index: u64) -> u64 {
        match self {
            Self::U32 => 0x1000_0001 + index,
            Self::F16 => 0x3c00 + index,
            Self::F32 => 0x3f80_0000 + index,
            Self::F64 => 0x3ff0_0000_0000_0000 + index,
        }
    }

    /// The vector of `len` components with `bits`.
    fn vector(self, bits: &[u64]) -> Val {
        // Only the arm of the kind uses its narrowing, where every component fits.
        let narrow32 = |b: &u64| u32::try_from(*b).unwrap_or(u32::MAX);
        let narrow16 = |b: &u64| u16::try_from(*b).unwrap_or(u16::MAX);
        let u32s: Vec<u32> = bits.iter().map(narrow32).collect();
        let u16s: Vec<u16> = bits.iter().map(narrow16).collect();
        match (self, bits.len()) {
            (Self::U32, 2) => Val::U32x2(u32s.try_into().expect("2")),
            (Self::U32, 3) => Val::U32x3(u32s.try_into().expect("3")),
            (Self::U32, _) => Val::U32x4(u32s.try_into().expect("4")),
            (Self::F32, 2) => Val::F32x2(u32s.try_into().expect("2")),
            (Self::F32, 3) => Val::F32x3(u32s.try_into().expect("3")),
            (Self::F32, _) => Val::F32x4(u32s.try_into().expect("4")),
            (Self::F16, 2) => Val::F16x2(u16s.try_into().expect("2")),
            (Self::F16, 3) => Val::F16x3(u16s.try_into().expect("3")),
            (Self::F16, _) => Val::F16x4(u16s.try_into().expect("4")),
            (Self::F64, 2) => Val::F64x2(bits.try_into().expect("2")),
            (Self::F64, 3) => Val::F64x3(bits.try_into().expect("3")),
            (Self::F64, _) => Val::F64x4(bits.try_into().expect("4")),
        }
    }

    /// The scalar with `bits`.
    fn scalar(self, bits: u64) -> Val {
        match self {
            Self::U32 => Val::U32(u32::try_from(bits).expect("32 bits")),
            Self::F16 => Val::F16(u16::try_from(bits).expect("16 bits")),
            Self::F32 => Val::F32(u32::try_from(bits).expect("32 bits")),
            Self::F64 => Val::F64(bits),
        }
    }

    /// A line defining `%id` as the scalar with `bits`.
    fn define(self, id: usize, bits: u64) -> String {
        match self {
            Self::U32 => format!("    %{id} = Identity u32 {bits:#x}"),
            Self::F16 => format!("    %{id} = BitCastF16U16 u16 {bits:#x}"),
            Self::F32 => format!("    %{id} = Identity f32 {bits:#x}"),
            Self::F64 => format!("    %{id} = Identity f64 {bits:#x}"),
        }
    }
}

#[test]
fn construct_extract_and_insert_every_shape() {
    for kind in Kind::ALL {
        for len in 2..=4_usize {
            let ty = format!("{}x{len}", kind.name());
            let bits: Vec<u64> = (0..len)
                .map(|index| kind.bits(u64::try_from(index).expect("small")))
                .collect();
            // %0..%len-1 are the components, %len the extra one inserted, then the vector.
            let mut lines: Vec<String> = bits
                .iter()
                .enumerate()
                .map(|(id, &b)| kind.define(id, b))
                .collect();
            let extra = kind.bits(0x40);
            lines.push(kind.define(len, extra));
            let ids: Vec<String> = (0..len).map(|id| format!("%{id}")).collect();
            let vector = len + 1;
            lines.push(format!(
                "    %{vector} = CompositeConstruct{ty} {}",
                ids.join(", ")
            ));
            for index in 0..len {
                lines.push(format!(
                    "    %{} = CompositeExtract{ty} %{vector}, u32 {index}",
                    vector + 1 + index
                ));
                lines.push(format!(
                    "    %{} = CompositeInsert{ty} %{vector}, %{len}, u32 {index}",
                    vector + 1 + len + index
                ));
            }
            let body = lines.join("\n");
            let id = |n: usize| u32::try_from(n).expect("small");
            assert_eq!(value_of(&body, id(vector)), kind.vector(&bits), "{ty}");
            for index in 0..len {
                assert_eq!(
                    value_of(&body, id(vector + 1 + index)),
                    kind.scalar(bits[index]),
                    "{ty} extract {index}"
                );
                let mut inserted = bits.clone();
                inserted[index] = extra;
                assert_eq!(
                    value_of(&body, id(vector + 1 + len + index)),
                    kind.vector(&inserted),
                    "{ty} insert {index}"
                );
            }
            // The index one past the last component is refused, for extract and insert.
            let past = format!(
                "{}\n    %{vector} = CompositeConstruct{ty} {}",
                lines[..=len].join("\n"),
                ids.join(", ")
            );
            let lower = ty.to_lowercase();
            for op in [
                format!("CompositeExtract{ty} %{vector}, u32 {len}"),
                format!("CompositeInsert{ty} %{vector}, %0, u32 {len}"),
            ] {
                let text = format!("block0:\n{past}\n    %99 = {op}\n    return\n");
                let error = run(&text, &mut Machine::new()).err().map(|e| e.to_string());
                assert_eq!(
                    error,
                    Some(format!("%99: component {len} is past the end of a {lower}")),
                    "{op}"
                );
            }
        }
    }
}

#[test]
fn constants_and_value_helpers() {
    // The register files: 256 per-thread registers, 8 predicates, 64 uniform registers and 8
    // uniform predicates, each including its zero or true name.
    assert_eq!(REG_SLOTS, 256);
    assert_eq!(PRED_SLOTS, 8);
    assert_eq!(UNIFORM_REG_SLOTS, 64);
    assert_eq!(UNIFORM_PRED_SLOTS, 8);
    assert_eq!(Val::f32(1.5), Val::F32(0x3fc0_0000));
    assert_eq!(Val::f64(-2.0), Val::F64(0xc000_0000_0000_0000));
    assert_eq!(Val::f32(1.5).ty(), Type::F32);
    assert_eq!(Val::zero(Type::F32x3), Some(Val::F32x3([0; 3])));
    assert_eq!(Val::zero(Type::Reg), None);
    assert_eq!(Val::zero(Type::Void), None);
    for ty in Type::ALL {
        if let Some(zero) = Val::zero(ty) {
            assert_eq!(zero.ty(), ty);
        }
        assert_eq!(Val::zero(ty).is_some(), ty.is_value(), "{ty}");
    }
}

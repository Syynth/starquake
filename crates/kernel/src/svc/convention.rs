//! The 64-bit register convention: which register carries each argument of a system call, and
//! which register each output goes to.
//!
//! Taken from Eden's generated wrappers (`SvcWrap_<Name>64` in `eden/src/core/hle/kernel/svc.cpp`
//! @ cf654052, GPL-3.0-or-later), which read arguments from and write outputs to `x0`-`x7`: they
//! take `std::span<uint64_t, 8>`, which confirms the guess behind `starquake-trace`'s
//! [`SVC_ARG_COUNT`], the count this module uses. Two of the calls with an entry here have no
//! generated wrapper and are transcribed from their hand-written ones: `SendSyncRequestLight`
//! (`SvcWrap_LightIpc` in `eden/src/core/hle/kernel/svc/svc_light_ipc.cpp` @ cf654052) and
//! `SetGpuMemoryAttribute` (the 0x80 case of `Call64` in `eden-pr4588/src/core/hle/kernel/svc.cpp`
//! @ e5fb3b97). The tests check [`CONVENTIONS`] entry by entry against a copy of those wrappers
//! (`reference/svc_wrappers64.txt`).
//!
//! Only the calls retail titles may make have an entry ([`is_retail`]): milestones 1 and 2 need
//! no others. Later slices add rows from the same reference file.
//!
//! `convention/table.rs` and both reference files are written by `tools/gen-svc-conventions.py`
//! from Eden's `svc.cpp`; run it with `--check` to see that they are current.
//!
//! Widths follow Eden's `Convert`, a `memcpy` of the smaller of the two sizes into a zeroed value
//! (`svc.cpp`): an argument narrower than its register is the register's low bytes, and a
//! narrower output is written zero-extended, signed or not ([`Kind::truncate`]). A kind's width
//! is the size `svc.cpp` `static_assert`s for its type (`reference/svc_type_sizes.txt`): 8 bytes
//! for the 64-bit integers, 1 for `bool`, 4 for everything else. [`Kind::Word`] takes any type
//! name and is always 4 bytes, so the tests check every `Word` in the table, and every type the
//! reference wrappers name, against that file.

use starquake_cpu::GuestRegs;
use starquake_trace::SVC_ARG_COUNT;

use super::SvcId;

mod table;

pub use table::CONVENTIONS;

/// The type of one argument or output, as it sits in a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// `uint64_t`.
    U64,
    /// `int64_t`.
    I64,
    /// `uint32_t`.
    U32,
    /// `int32_t`.
    I32,
    /// A 32-bit handle (`Handle`).
    Handle,
    /// A 32-bit result word (`Result`).
    Result,
    /// `bool`: one byte. No retail call takes one; `CreateSession` and `CreatePort` do
    /// (`is_light`).
    Bool,
    /// Another 4-byte type (an enum such as `InfoType`, or `PageInfo`), named as Eden names it.
    Word(&'static str),
}

impl Kind {
    /// The value's size in bytes: 8 for the 64-bit integers, 1 for `bool`, 4 for everything
    /// else.
    #[must_use]
    pub const fn bytes(self) -> u8 {
        match self {
            Self::U64 | Self::I64 => 8,
            Self::U32 | Self::I32 | Self::Handle | Self::Result | Self::Word(_) => 4,
            Self::Bool => 1,
        }
    }

    /// The C++ type Eden's wrappers use for it.
    #[must_use]
    pub const fn eden_type(self) -> &'static str {
        match self {
            Self::U64 => "uint64_t",
            Self::I64 => "int64_t",
            Self::U32 => "uint32_t",
            Self::I32 => "int32_t",
            Self::Handle => "Handle",
            Self::Result => "Result",
            Self::Bool => "bool",
            Self::Word(name) => name,
        }
    }

    /// The part of a 64-bit register this kind occupies: its low [`Kind::bytes`] bytes,
    /// zero-extended (the whole register for an 8-byte kind). Used both to decode an argument
    /// and to write an output, as Eden's `Convert` does.
    #[must_use]
    pub fn truncate(self, register: u64) -> u64 {
        let bits = u32::from(self.bytes()) * u8::BITS;
        match 1_u64.checked_shl(bits) {
            Some(above) => register & (above - 1),
            // Eight bytes: the whole register.
            None => register,
        }
    }
}

/// One argument or output of a system call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Arg {
    /// Eden's parameter name. Elements of a struct spread over several registers are `name[k]`.
    pub name: &'static str,
    /// The register number: `x<reg>`.
    pub reg: u8,
    /// Its type.
    pub kind: Kind,
}

/// Where one system call's arguments come from and its outputs go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Convention {
    /// Arguments, in the order of Eden's function parameters.
    pub inputs: &'static [Arg],
    /// Outputs, in the order Eden's wrapper writes them. A call that returns a `Result` has it
    /// first, in `x0`.
    pub outputs: &'static [Arg],
}

/// A system call's implementation and its convention disagree. This is a bug in the dispatcher,
/// never something a guest can cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ConventionError {
    /// The call takes or produces a different number of values than the convention lists.
    #[error("the convention lists {expected} values but the call has {actual}")]
    Arity {
        /// How many the convention lists.
        expected: usize,
        /// How many the call takes or produced.
        actual: usize,
    },
    /// The convention names a register outside `x0`-`x7`.
    #[error("x{0} is not a system-call argument register")]
    Register(u8),
}

fn register_index(arg: &Arg) -> Result<usize, ConventionError> {
    let index = usize::from(arg.reg);
    if index < SVC_ARG_COUNT {
        Ok(index)
    } else {
        Err(ConventionError::Register(arg.reg))
    }
}

impl Convention {
    /// Decodes the arguments, in input order, each truncated to its [`Kind`].
    pub fn read<const N: usize>(&self, regs: &GuestRegs) -> Result<[u64; N], ConventionError> {
        let mut values = [0; N];
        self.read_into(regs, &mut values)?;
        Ok(values)
    }

    fn read_into(&self, regs: &GuestRegs, values: &mut [u64]) -> Result<(), ConventionError> {
        if values.len() != self.inputs.len() {
            return Err(ConventionError::Arity {
                expected: self.inputs.len(),
                actual: values.len(),
            });
        }
        for (value, arg) in values.iter_mut().zip(self.inputs) {
            let register = regs
                .x
                .get(register_index(arg)?)
                .ok_or(ConventionError::Register(arg.reg))?;
            *value = arg.kind.truncate(*register);
        }
        Ok(())
    }

    /// Writes the outputs, in output order, each truncated to its [`Kind`]. Nothing is written
    /// unless every output has a valid register.
    pub fn write(&self, regs: &mut GuestRegs, values: &[u64]) -> Result<(), ConventionError> {
        if values.len() != self.outputs.len() {
            return Err(ConventionError::Arity {
                expected: self.outputs.len(),
                actual: values.len(),
            });
        }
        for arg in self.outputs {
            register_index(arg)?;
        }
        for (value, arg) in values.iter().zip(self.outputs) {
            let register = regs
                .x
                .get_mut(register_index(arg)?)
                .ok_or(ConventionError::Register(arg.reg))?;
            *register = arg.kind.truncate(*value);
        }
        Ok(())
    }
}

/// Whether retail titles may make this call: 0x01-0x36 and 0x80. Kernel spec, "What is built
/// now": the Switch 2 titles Eden-NX2 examined are permitted these and no others
/// (`eden-nx2/NX2_NOTES.md` @ 50fa86bb: "SVCs 0x01-0x36 plus 0x80").
#[must_use]
pub const fn is_retail(id: SvcId) -> bool {
    let n = id.number();
    (n >= SvcId::SetHeapSize.number() && n <= SvcId::SynchronizePreemptionState.number())
        || n == SvcId::SetGpuMemoryAttribute.number()
}

impl SvcId {
    /// The call's register convention, or `None` if this table has no entry for it yet (only the
    /// [`is_retail`] calls have one).
    #[must_use]
    pub fn convention(self) -> Option<&'static Convention> {
        CONVENTIONS
            .binary_search_by_key(&self, |(id, _)| *id)
            .ok()
            .and_then(|i| CONVENTIONS.get(i))
            .map(|(_, convention)| convention)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eden's wrappers, reduced to their register use (see the file's header).
    const REFERENCE: &str = include_str!("reference/svc_wrappers64.txt");

    /// The size Eden asserts for each type the wrappers use (see the file's header).
    const TYPE_SIZES: &str = include_str!("reference/svc_type_sizes.txt");

    /// `(type, bytes)` for every line of the size file.
    fn type_sizes() -> Vec<(&'static str, u8)> {
        TYPE_SIZES
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| {
                let (name, size) = l.split_once(" = ").unwrap();
                (name.trim(), size.trim().parse().unwrap())
            })
            .collect()
    }

    /// Eden's size for a type, from the size file.
    fn eden_size(ty: &str) -> Option<u8> {
        type_sizes()
            .into_iter()
            .find(|(name, _)| *name == ty)
            .map(|(_, size)| size)
    }

    type Entry = (String, String, String);

    fn parse_args(list: &str) -> Vec<Entry> {
        list.split_whitespace()
            .map(|a| {
                let mut parts = a.split(':');
                let name = parts.next().unwrap().to_owned();
                let ty = parts.next().unwrap().to_owned();
                let reg = parts.next().unwrap().to_owned();
                assert_eq!(parts.next(), None, "{a}");
                (name, ty, reg)
            })
            .collect()
    }

    /// `(name, inputs, outputs)` for every call in the reference file.
    fn reference() -> Vec<(String, Vec<Entry>, Vec<Entry>)> {
        REFERENCE
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| {
                let mut columns = l.split('|');
                let name = columns.next().unwrap().trim().to_owned();
                let inputs = parse_args(columns.next().unwrap());
                let outputs = parse_args(columns.next().unwrap());
                assert_eq!(columns.next(), None, "{l}");
                (name, inputs, outputs)
            })
            .collect()
    }

    fn ours(args: &[Arg]) -> Vec<Entry> {
        args.iter()
            .map(|a| {
                (
                    a.name.to_owned(),
                    a.kind.eden_type().to_owned(),
                    format!("x{}", a.reg),
                )
            })
            .collect()
    }

    #[test]
    fn every_convention_matches_edens_wrapper() {
        let reference = reference();
        for (id, convention) in CONVENTIONS {
            let found = reference.iter().find(|(name, _, _)| name == id.name());
            assert!(found.is_some(), "{} has no reference wrapper", id.name());
            let (_, inputs, outputs) = found.unwrap();
            assert_eq!(&ours(convention.inputs), inputs, "{} inputs", id.name());
            assert_eq!(&ours(convention.outputs), outputs, "{} outputs", id.name());
        }
    }

    #[test]
    fn the_reference_covers_every_listed_call_once() {
        let reference = reference();
        for &id in SvcId::ALL {
            let count = reference.iter().filter(|(n, _, _)| n == id.name()).count();
            assert_eq!(count, 1, "{}", id.name());
        }
        assert_eq!(reference.len(), SvcId::ALL.len());
    }

    #[test]
    fn exactly_the_retail_calls_have_a_convention_in_number_order() {
        let with: Vec<_> = CONVENTIONS.iter().map(|(id, _)| *id).collect();
        let retail: Vec<_> = SvcId::ALL
            .iter()
            .copied()
            .filter(|&id| is_retail(id))
            .collect();
        assert_eq!(with, retail);
        for &id in SvcId::ALL {
            assert_eq!(id.convention().is_some(), is_retail(id), "{}", id.name());
        }
    }

    #[test]
    fn retail_range_is_0x01_to_0x36_plus_0x80() {
        // NX2_NOTES.md: "SVCs 0x01-0x36 plus 0x80". Each bound sits exactly at the limit.
        let first = SvcId::SetHeapSize;
        let last = SvcId::SynchronizePreemptionState;
        assert!(is_retail(first) && is_retail(last));
        assert_eq!(first.number(), 0x01);
        assert_eq!(last.number(), 0x36);
        assert!(!is_retail(SvcId::GetResourceLimitPeakValue)); // 0x37, one past the range
        assert_eq!(SvcId::GetResourceLimitPeakValue.number(), last.number() + 1);
        assert!(is_retail(SvcId::SetGpuMemoryAttribute));
        assert_eq!(SvcId::SetGpuMemoryAttribute.number(), 0x80);
        assert!(!is_retail(SvcId::CallSecureMonitor)); // 0x7f, just below 0x80
    }

    #[test]
    fn every_register_is_an_argument_register() {
        for (id, convention) in CONVENTIONS {
            for arg in convention.inputs.iter().chain(convention.outputs) {
                assert!(
                    usize::from(arg.reg) < SVC_ARG_COUNT,
                    "{} {}",
                    id.name(),
                    arg.name
                );
            }
        }
    }

    #[test]
    fn a_result_returning_call_has_its_result_first_in_x0() {
        for (id, convention) in CONVENTIONS {
            for (i, arg) in convention.outputs.iter().enumerate() {
                if arg.kind == Kind::Result {
                    assert_eq!((i, arg.reg), (0, 0), "{}", id.name());
                }
            }
        }
    }

    /// The low `bytes` bytes of a register, zero-extended, worked out bytewise so it does not
    /// repeat [`Kind::truncate`]'s arithmetic.
    fn low_bytes(register: u64, bytes: u8) -> u64 {
        let mut kept = [0; 8];
        let count = usize::from(bytes);
        kept[..count].copy_from_slice(&register.to_le_bytes()[..count]);
        u64::from_le_bytes(kept)
    }

    /// A register file whose registers are all different, in both halves, so decoding from the
    /// wrong register or keeping the wrong half shows up.
    fn distinct_registers() -> GuestRegs {
        let mut regs = GuestRegs::default();
        for (i, x) in (0u64..).zip(regs.x.iter_mut()) {
            *x = ((0xA0 + i) << 56) | ((0xB0 + i) << 32) | ((0xC0 + i) << 8) | (0xD0 + i);
        }
        regs
    }

    #[test]
    fn every_convention_decodes_each_argument_from_its_own_register() {
        let regs = distinct_registers();
        for (id, convention) in CONVENTIONS {
            let mut values = vec![0; convention.inputs.len()];
            convention.read_into(&regs, &mut values).unwrap();
            for (value, arg) in values.iter().zip(convention.inputs) {
                let register = regs.x[usize::from(arg.reg)];
                assert_eq!(
                    *value,
                    low_bytes(register, arg.kind.bytes()),
                    "{} {}",
                    id.name(),
                    arg.name
                );
            }
        }
    }

    #[test]
    fn every_convention_writes_each_output_to_its_own_register() {
        for (id, convention) in CONVENTIONS {
            let mut regs = distinct_registers();
            let before = regs.clone();
            let values: Vec<u64> = (0u64..)
                .take(convention.outputs.len())
                .map(|i| u64::MAX - i)
                .collect();
            convention.write(&mut regs, &values).unwrap();
            for (value, arg) in values.iter().zip(convention.outputs) {
                assert_eq!(
                    regs.x[usize::from(arg.reg)],
                    low_bytes(*value, arg.kind.bytes()),
                    "{} {}",
                    id.name(),
                    arg.name
                );
            }
            for (i, (now, was)) in regs.x.iter().zip(&before.x).enumerate() {
                if convention.outputs.iter().all(|a| usize::from(a.reg) != i) {
                    assert_eq!(now, was, "{} wrote x{i}", id.name());
                }
            }
        }
    }

    #[test]
    fn four_byte_outputs_are_zero_extended_even_when_signed() {
        // Eden's Convert<uint64_t>(int32_t) copies 4 bytes into a zeroed u64.
        let convention = SvcId::GetCurrentProcessorNumber.convention().unwrap();
        assert_eq!(convention.outputs[0].kind, Kind::I32);
        let mut regs = GuestRegs::default();
        convention
            .write(&mut regs, &[u64::from((-1i32).cast_unsigned())])
            .unwrap();
        assert_eq!(regs.x[0], u64::from(u32::MAX));
        convention.write(&mut regs, &[u64::MAX]).unwrap();
        assert_eq!(regs.x[0], u64::from(u32::MAX));
    }

    #[test]
    fn arity_mismatches_are_errors() {
        let convention = SvcId::OutputDebugString.convention().unwrap();
        let regs = GuestRegs::default();
        assert_eq!(
            convention.read::<3>(&regs),
            Err(ConventionError::Arity {
                expected: 2,
                actual: 3
            })
        );
        let mut regs = GuestRegs::default();
        assert_eq!(
            convention.write(&mut regs, &[1, 2]),
            Err(ConventionError::Arity {
                expected: 1,
                actual: 2
            })
        );
        assert_eq!(
            ConventionError::Arity {
                expected: 1,
                actual: 2
            }
            .to_string(),
            "the convention lists 1 values but the call has 2"
        );
    }

    #[test]
    fn registers_past_x7_are_rejected_and_nothing_is_written() {
        static OUTPUTS: &[Arg] = &[
            Arg {
                name: "ok",
                reg: 0,
                kind: Kind::U64,
            },
            Arg {
                name: "bad",
                reg: 8,
                kind: Kind::U64,
            },
        ];
        let convention = Convention {
            inputs: OUTPUTS,
            outputs: OUTPUTS,
        };
        let mut regs = GuestRegs::default();
        assert_eq!(
            convention.read::<2>(&regs),
            Err(ConventionError::Register(8))
        );
        assert_eq!(
            convention.write(&mut regs, &[1, 2]),
            Err(ConventionError::Register(8))
        );
        assert_eq!(regs, GuestRegs::default());
        assert_eq!(
            ConventionError::Register(8).to_string(),
            "x8 is not a system-call argument register"
        );
    }

    #[test]
    fn kinds_name_their_eden_type_and_width() {
        for (kind, ty, bytes) in [
            (Kind::U64, "uint64_t", 8),
            (Kind::I64, "int64_t", 8),
            (Kind::U32, "uint32_t", 4),
            (Kind::I32, "int32_t", 4),
            (Kind::Handle, "Handle", 4),
            (Kind::Result, "Result", 4),
            (Kind::Bool, "bool", 1),
            (Kind::Word("InfoType"), "InfoType", 4),
        ] {
            assert_eq!((kind.eden_type(), kind.bytes()), (ty, bytes));
        }
    }

    #[test]
    fn every_kind_in_the_table_is_as_wide_as_eden_asserts_its_type_to_be() {
        // `Kind::Word` takes any type name and is 4 bytes: a row naming a type of another size
        // would convert wrongly, and fails here.
        for (id, convention) in CONVENTIONS {
            for arg in convention.inputs.iter().chain(convention.outputs) {
                assert_eq!(
                    eden_size(arg.kind.eden_type()),
                    Some(arg.kind.bytes()),
                    "{} {} ({})",
                    id.name(),
                    arg.name,
                    arg.kind.eden_type()
                );
            }
        }
        // The kinds with a type of their own, whether or not a row uses them yet.
        for kind in [
            Kind::U64,
            Kind::I64,
            Kind::U32,
            Kind::I32,
            Kind::Handle,
            Kind::Result,
            Kind::Bool,
        ] {
            assert_eq!(eden_size(kind.eden_type()), Some(kind.bytes()), "{kind:?}");
        }
    }

    #[test]
    fn every_type_the_reference_wrappers_name_has_a_kind_of_its_width() {
        // Rows beyond the retail calls come from the same reference file. Each type in it must
        // have a size on record, and that size must be one a `Kind` can carry: its own kind for
        // the types that have one, otherwise the 4 bytes of a `Word`.
        let own: Vec<(&str, u8)> = [
            Kind::U64,
            Kind::I64,
            Kind::U32,
            Kind::I32,
            Kind::Handle,
            Kind::Result,
            Kind::Bool,
        ]
        .iter()
        .map(|kind| (kind.eden_type(), kind.bytes()))
        .collect();
        let mut seen_bool = false;
        for (call, inputs, outputs) in reference() {
            for (name, ty, _) in inputs.iter().chain(&outputs) {
                let carried = own
                    .iter()
                    .find(|(eden, _)| eden == ty)
                    .map_or(Kind::Word("").bytes(), |(_, bytes)| *bytes);
                assert_eq!(eden_size(ty), Some(carried), "{call} {name} ({ty})");
                seen_bool |= ty == Kind::Bool.eden_type();
            }
        }
        // `CreateSession` and `CreatePort` take `bool is_light`: the reason `Bool` exists.
        assert!(seen_bool);
        // The size file lists exactly the types the wrappers name.
        let mut named: Vec<String> = reference()
            .into_iter()
            .flat_map(|(_, inputs, outputs)| inputs.into_iter().chain(outputs))
            .map(|(_, ty, _)| ty)
            .collect();
        named.sort();
        named.dedup();
        let listed: Vec<String> = type_sizes()
            .into_iter()
            .map(|(name, _)| name.to_owned())
            .collect();
        assert_eq!(listed, named);
    }

    #[test]
    fn a_bool_is_the_low_byte_of_its_register_both_ways() {
        // The reference row: `CreateSession | is_light:bool:x2 ...`.
        static INPUTS: &[Arg] = &[
            Arg {
                name: "is_light",
                reg: 2,
                kind: Kind::Bool,
            },
            Arg {
                name: "name",
                reg: 3,
                kind: Kind::U64,
            },
        ];
        static OUTPUTS: &[Arg] = &[Arg {
            name: "flag",
            reg: 1,
            kind: Kind::Bool,
        }];
        let convention = Convention {
            inputs: INPUTS,
            outputs: OUTPUTS,
        };
        let (_, reference_inputs, _) = reference()
            .into_iter()
            .find(|(name, _, _)| name == SvcId::CreateSession.name())
            .unwrap();
        assert_eq!(ours(INPUTS), reference_inputs);

        let regs = distinct_registers();
        let [is_light, name] = convention.read(&regs).unwrap();
        assert_eq!(is_light, low_bytes(regs.x[2], 1));
        assert_eq!(is_light, 0xD2);
        assert_eq!(name, regs.x[3]);

        let mut regs = distinct_registers();
        convention
            .write(&mut regs, &[0xFFFF_FFFF_FFFF_FF01])
            .unwrap();
        assert_eq!(regs.x[1], 1);
    }

    #[test]
    fn truncation_keeps_exactly_the_kinds_bytes() {
        let register = 0x8877_6655_4433_2211;
        assert_eq!(Kind::Bool.truncate(register), 0x11);
        assert_eq!(Kind::U32.truncate(register), 0x4433_2211);
        assert_eq!(Kind::Word("InfoType").truncate(register), 0x4433_2211);
        assert_eq!(Kind::U64.truncate(register), register);
        assert_eq!(Kind::I64.truncate(u64::MAX), u64::MAX);
        // The bit just above each width is dropped, the one just inside is kept.
        assert_eq!(Kind::Bool.truncate(0x100), 0);
        assert_eq!(Kind::Bool.truncate(0x80), 0x80);
        assert_eq!(Kind::I32.truncate(1 << 32), 0);
        assert_eq!(Kind::I32.truncate(1 << 31), 1 << 31);
    }
}

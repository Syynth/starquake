//! The disassembler: operands and instructions as `nvdisasm` writes them, lower-cased, the way
//! NAK's `nvdisasm` tests compare them (`mesa-nouveau/src/nouveau/compiler/nak/nvdisasm_tests.rs`
//! @ 52238762, `disassemble_instrs`: address and encoding stripped, lower-cased; the expected
//! strings end in `;` with no space before it). Each operation's printer is in its group's
//! module under `op`; this module prints the operands they share and the instruction around
//! them.
//!
//! Which parts are confirmed by NAK's tests: `test_float_ops` pins `FFMA` with register and
//! immediate sources, the `.ftz`, `.rz` and `.sat` modifiers in that order, and the immediate
//! `2.0` written as `2`; `test_nop` and the other tests pin the `mnemonic operands;` shape,
//! registers as `r1` and `rz`, predicates as `p0`/`pt`, uniform registers as `ur2`, uniform
//! predicates as `up1`, and constant buffers as `c[0x5][0x100]` and `cx[ur2][0x100]`
//! (`test_ldc`, `test_nanosleep`). Of the control group (`op::control`), `test_nop`,
//! `test_redux`, `test_match` and `test_nanosleep` pin `nop`, every `redux` operation, `match`
//! with and without its predicate and `.u64`, and `nanosleep` with each source kind and a hex
//! immediate. **Everything else is a guess** in `nvdisasm`'s usual style that a developer with
//! `nvdisasm` should check (spec, "`nvdisasm` cross-check"): `urz` and `upt`, the `.rm`/`.rp`
//! rounding names (taken from NAK's own `FRndMode` printer), `.fmz` for the `dnz` bit, float
//! immediates other than small integers, `-`/`|..|` source modifiers (on registers, uniform
//! registers and constant buffers alike), the guard prefix `@p0 `/`@!p0 `, `MOV`'s hex immediate
//! and lane mask, the `ALD`/`AST` operand syntax and width suffixes, `EXIT`'s modifiers and
//! predicate, the special-register names, the text of every other control instruction (branch
//! offsets in particular print as a signed byte displacement from the next instruction, where
//! `nvdisasm` prints the absolute target: `op::control`, `BranchOffset`), and every shell's
//! mnemonic.

use std::fmt::{self, Display, Formatter};

use super::op::Instr;
use super::operand::{
    AluSrc, CBuf, CBufRef, Gpr, Pred, PredSrc, RoundMode, UniformPred, UniformPredSrc, UniformReg,
};

/// Writes `text` if `on`: a modifier that is either present or absent.
pub(crate) fn flag(f: &mut Formatter<'_>, on: bool, text: &str) -> fmt::Result {
    if on { f.write_str(text) } else { Ok(()) }
}

impl Display for Gpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            f.write_str("rz")
        } else {
            write!(f, "r{}", self.index())
        }
    }
}

impl Display for UniformReg {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            f.write_str("urz")
        } else {
            write!(f, "ur{}", self.index())
        }
    }
}

impl Display for Pred {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_true() {
            f.write_str("pt")
        } else {
            write!(f, "p{}", self.index())
        }
    }
}

impl Display for UniformPred {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_true() {
            f.write_str("upt")
        } else {
            write!(f, "up{}", self.index())
        }
    }
}

impl Display for PredSrc {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        flag(f, self.negated, "!")?;
        write!(f, "{}", self.reg)
    }
}

impl Display for UniformPredSrc {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        flag(f, self.negated, "!")?;
        write!(f, "{}", self.reg)
    }
}

impl Display for CBuf {
    /// NAK's `CBuf` printer: `c[0x5]` or `cx[ur2]`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binding(index) => write!(f, "c[{index:#x}]"),
            Self::BindlessUniform(reg) => write!(f, "cx[{reg}]"),
        }
    }
}

impl Display for CBufRef {
    /// NAK's `CBufRef` printer: the buffer, then `[offset]` in hexadecimal.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{:#x}]", self.buf, self.offset)
    }
}

impl Display for RoundMode {
    /// The modifier, empty for the default (NAK's `FRndMode` printer: `.re`, `.rm`, `.rp`,
    /// `.rz`; `nvdisasm` omits `.re`, as `test_float_ops` expects).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NearestEven => "",
            Self::NegInf => ".rm",
            Self::PosInf => ".rp",
            Self::Zero => ".rz",
        })
    }
}

/// A 32-bit float immediate. Integers of magnitude below 2^24 print without a fraction (`2`,
/// confirmed by `test_float_ops`); other finite values print as Rust's shortest round-trip
/// form, infinities as `+inf`/`-inf` and NaNs as `+qnan`/`-qnan` with their sign (guesses).
struct FloatImm(u32);

/// The largest magnitude below which every integer is exact in an `f32` (2^24).
const EXACT_INTEGER_LIMIT: f32 = 16_777_216.0;

impl Display for FloatImm {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let value = f32::from_bits(self.0);
        let sign = if value.is_sign_negative() { "-" } else { "+" };
        if value.is_nan() {
            write!(f, "{sign}qnan")
        } else if value.is_infinite() {
            write!(f, "{sign}inf")
        } else if value.fract() == 0.0 && value.abs() < EXACT_INTEGER_LIMIT {
            // `{}` writes an integral f32 without a fraction; `-0.0` keeps its sign.
            write!(f, "{value}")
        } else {
            write!(f, "{value:?}")
        }
    }
}

/// Writes `operand` under its modifiers: `-` for negation, `|..|` for the absolute value.
fn modified(f: &mut Formatter<'_>, operand: &dyn Display, abs: bool, neg: bool) -> fmt::Result {
    flag(f, neg, "-")?;
    if abs {
        write!(f, "|{operand}|")
    } else {
        write!(f, "{operand}")
    }
}

/// An ALU source of a floating-point instruction: an immediate prints as a float.
pub(crate) struct FloatOperand(pub(crate) AluSrc);

impl Display for FloatOperand {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            AluSrc::Reg { reg, abs, neg } => modified(f, &reg, abs, neg),
            AluSrc::UniformReg { reg, abs, neg } => modified(f, &reg, abs, neg),
            AluSrc::CBuf { cbuf, abs, neg } => modified(f, &cbuf, abs, neg),
            AluSrc::Imm(bits) => write!(f, "{}", FloatImm(bits)),
        }
    }
}

/// An ALU source of an integer or bit instruction: an immediate prints in hexadecimal (NAK's
/// `test_nanosleep`: `0x87654321`), the other kinds as themselves under their modifiers.
pub(crate) struct IntOperand(pub(crate) AluSrc);

impl Display for IntOperand {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            AluSrc::Reg { reg, abs, neg } => modified(f, &reg, abs, neg),
            AluSrc::UniformReg { reg, abs, neg } => modified(f, &reg, abs, neg),
            AluSrc::CBuf { cbuf, abs, neg } => modified(f, &cbuf, abs, neg),
            AluSrc::Imm(bits) => write!(f, "{bits:#x}"),
        }
    }
}

impl Display for Instr {
    /// The instruction with its guard (left out when it is `PT`) and the closing `;`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if !self.guard.is_true() {
            write!(f, "@{} ", self.guard)?;
        }
        write!(f, "{};", self.op)
    }
}

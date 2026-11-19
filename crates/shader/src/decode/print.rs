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
//! immediate. Of the attribute group (`op::attribute`), `test_isberd` pins `isberd` with every
//! output, access, skew and memory-type modifier, in that order, and the `[r2]`/`[r2+0x42]`
//! address; `test_isbewr` pins `isbewr` only with `.o` set and the map and `.attr` access types,
//! with every skew and memory-type modifier and both addresses, so `isbewr` without `.o` is a
//! guess. Of the float group (`op::float`), `test_float_ops`, `test_mufu` and `test_f2fp` pin
//! `fadd`, `fmul`, `fmnmx`, `fset.bf`, `fsetp` (both destinations and the accumulator), the
//! fourteen comparisons and three combine operations NAK names, `f2f`, `f2i`, `frnd` and `i2f`
//! with the types NAK tests (`.u32`, `.f16`, `.f64`) and `.trunc`, every `mufu` function with
//! `.f16`, and `f2fp.pack_ab`. Of the integer group (`op::integer`), `test_lea`, `test_sgxt`,
//! `test_plop3`, `test_iabs` and `test_imnmx` pin `lea` and `lea.hi` with the shifted value and
//! `b` negated and the shift in hex, `sgxt` and `sgxt.u32`, `plop3.lut` with negated predicates
//! and both truth tables in hex, `iabs`, and `imnmx` and `imnmx.u32` with a predicate. Of the
//! memory group (`op::memory`), `test_ldc` pins `ldc` with every type but `.128`, a bound and a
//! bindless buffer, with and without an offset register, and `test_ld_st_atom` pins `ldg`,
//! `stg`, `lds`, `sts`, `ldl`, `stl`, `atomg`, `red` and `atoms` for 32-bit accesses with `.ef`
//! and `.strong.sm`, every atomic operation and type (in shared memory the 32-bit integer ones),
//! and addresses with `.64`, `.x8`, a uniform register and a positive or negative offset
//! (`+-0x1`). Of the texture group (`op::texture`), `test_texture` pins the bindless `tex`,
//! `tld`, `tld4`, `tmml`, `txd` and `txq` with `.scr`, `.b`, every level-of-detail mode,
//! `tld4`'s `.g` and offset modes, `.ef` and `.nodep`, in that order, the operands
//! `p0, r2, r0, r1, r3, 2d, 0x7`, and the three `txq` queries; that module's docs list what of
//! its text is a guess.
//! **Everything else is a guess** in `nvdisasm`'s usual style that a developer with
//! `nvdisasm` should check (spec, "`nvdisasm` cross-check"): `urz` and `upt`, the `.rm`/`.rp`
//! rounding names (taken from NAK's own `FRndMode` printer), `.fmz` for the `dnz` bit, float
//! immediates other than small integers, `-`/`|..|` source modifiers (on registers, uniform
//! registers and constant buffers alike), the guard prefix `@p0 `/`@!p0 `, `MOV`'s hex immediate
//! and lane mask, the `ALD`/`AST` operand syntax and width suffixes, `EXIT`'s modifiers and
//! predicate, the special-register names, the text of every other attribute instruction
//! (`IPA`'s modes, `CS2R`'s `.32`, `S2UR`, `PIXLD`, `OUT`, `LDTRAM`, `AL2P`),
//! the text of every other control instruction (branch
//! offsets in particular print as a signed byte displacement from the next instruction, where
//! `nvdisasm` prints the absolute target: `op::control`, `BranchOffset`), the float group's
//! text outside NAK's cases (each printer in `op::float` says which parts), the text of every
//! other integer instruction (its modifier names and operand order, and `~` for a bitwise-not
//! source, which NAK's own printer writes `!`: `op::integer`), the memory group's text outside
//! NAK's cases (`op::memory`'s module docs list it), and every shell's mnemonic.

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

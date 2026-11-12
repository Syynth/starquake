//! The disassembler: an [`Instr`] as `nvdisasm` writes it, lower-cased, the way NAK's
//! `nvdisasm` tests compare it (`mesa-nouveau/src/nouveau/compiler/nak/nvdisasm_tests.rs`
//! @ 52238762, `disassemble_instrs`: address and encoding stripped, lower-cased; the expected
//! strings end in `;` with no space before it).
//!
//! Which parts are confirmed: NAK's `test_float_ops` pins `FFMA` with register and immediate
//! sources, the `.ftz`, `.rz` and `.sat` modifiers in that order, and the immediate `2.0`
//! written as `2`; `test_nop` and the other tests pin the `mnemonic operands;` shape, registers
//! as `r1`, and predicates as `p0`/`pt`. **Everything else is a guess** in `nvdisasm`'s usual
//! style that a developer with `nvdisasm` should check (spec, "`nvdisasm` cross-check"): the
//! `.rm`/`.rp` rounding names (taken from NAK's own `FRndMode` printer), `.fmz` for the `dnz`
//! bit, float immediates other than small integers, `-`/`|..|` source modifiers, the guard
//! prefix `@p0 `/`@!p0 `, `MOV`'s hex immediate and lane mask, the `ALD`/`AST` operand syntax
//! and width suffixes, `EXIT`'s modifiers and predicate, and the special-register names.

use std::fmt::{self, Display, Formatter};

use super::op::{
    FloatSrc, Gpr, Instr, MovSrc, Op, OpALd, OpASt, OpExit, OpFFma, OpMov, OpS2R, Pred, PredSrc,
    RoundMode, SpecialReg,
};

impl Display for Gpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            f.write_str("rz")
        } else {
            write!(f, "r{}", self.index())
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

impl Display for PredSrc {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.negated {
            f.write_str("!")?;
        }
        write!(f, "{}", self.reg)
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

impl Display for FloatSrc {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Reg { reg, abs, neg } => {
                let neg = if neg { "-" } else { "" };
                if abs {
                    write!(f, "{neg}|{reg}|")
                } else {
                    write!(f, "{neg}{reg}")
                }
            }
            Self::Imm(bits) => write!(f, "{}", FloatImm(bits)),
        }
    }
}

impl Display for MovSrc {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(reg) => write!(f, "{reg}"),
            Self::Imm(bits) => write!(f, "{bits:#x}"),
        }
    }
}

impl Display for SpecialReg {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let name = match *self {
            Self::LANE_ID => "sr_laneid",
            Self::TID_X => "sr_tid.x",
            Self::TID_Y => "sr_tid.y",
            Self::TID_Z => "sr_tid.z",
            Self::CTAID_X => "sr_ctaid.x",
            Self::CTAID_Y => "sr_ctaid.y",
            Self::CTAID_Z => "sr_ctaid.z",
            Self(index) => return write!(f, "sr{index}"),
        };
        f.write_str(name)
    }
}

/// The width suffix of an attribute access of `comps` words (guess: `nvdisasm`'s `.64`,
/// `.96`, `.128` style).
fn width(comps: u8) -> &'static str {
    match comps {
        2 => ".64",
        3 => ".96",
        4 => ".128",
        _ => "",
    }
}

/// `a[vtx][addr+offset]`, leaving out `RZ` parts (NAK's `OpALd`/`OpASt` printers).
struct AttrAddr {
    vtx: Gpr,
    offset: Gpr,
    addr: u16,
}

impl Display for AttrAddr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("a")?;
        if !self.vtx.is_zero() {
            write!(f, "[{}]", self.vtx)?;
        }
        write!(f, "[{:#x}", self.addr)?;
        if !self.offset.is_zero() {
            write!(f, "+{}", self.offset)?;
        }
        f.write_str("]")
    }
}

fn flag(f: &mut Formatter<'_>, on: bool, text: &str) -> fmt::Result {
    if on { f.write_str(text) } else { Ok(()) }
}

impl Display for OpMov {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "mov {}, {}", self.dst, self.src)?;
        if self.quad_lanes != Self::ALL_LANES {
            write!(f, ", {:#x}", self.quad_lanes)?;
        }
        Ok(())
    }
}

impl Display for OpFFma {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("ffma")?;
        flag(f, self.ftz, ".ftz")?;
        flag(f, self.dnz, ".fmz")?;
        write!(f, "{}", self.rnd)?;
        flag(f, self.saturate, ".sat")?;
        let [a, b, c] = self.srcs;
        write!(f, " {}, {a}, {b}, {c}", self.dst)
    }
}

impl Display for OpALd {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("ald")?;
        flag(f, self.output, ".o")?;
        flag(f, self.patch, ".p")?;
        flag(f, self.phys, ".phys")?;
        let addr = AttrAddr {
            vtx: self.vtx,
            offset: self.offset,
            addr: self.addr,
        };
        write!(f, "{} {}, {addr}", width(self.comps), self.dst)
    }
}

impl Display for OpASt {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("ast")?;
        flag(f, self.patch, ".p")?;
        flag(f, self.phys, ".phys")?;
        let addr = AttrAddr {
            vtx: self.vtx,
            offset: self.offset,
            addr: self.addr,
        };
        write!(f, "{} {addr}, {}", width(self.comps), self.data)
    }
}

impl Display for OpExit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("exit")?;
        flag(f, self.keep_ref_count, ".keeprefcount")?;
        flag(f, self.no_at_exit, ".no_atexit")?;
        if !self.pred.is_true() {
            write!(f, " {}", self.pred)?;
        }
        Ok(())
    }
}

impl Display for OpS2R {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "s2r {}, {}", self.dst, self.sr)
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mov(op) => op.fmt(f),
            Self::FFma(op) => op.fmt(f),
            Self::ALd(op) => op.fmt(f),
            Self::ASt(op) => op.fmt(f),
            Self::Exit(op) => op.fmt(f),
            Self::S2R(op) => op.fmt(f),
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

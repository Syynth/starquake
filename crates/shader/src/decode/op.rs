//! Decoded instructions: one struct of typed operands per instruction.
//!
//! The shapes follow NAK's `Op*` structs in `mesa-nouveau/src/nouveau/compiler/nak/ir.rs`
//! @ 52238762 (MIT, Copyright 2022 Collabora, Ltd.): `OpMov`, `OpFFma`, `OpALd`, `OpASt`,
//! `OpExit`, `OpS2R`. Differences: registers are plain register numbers (no SSA values or
//! register-file tags), a source is an enum of exactly what the form can encode, and `OpExit`
//! carries the mode and predicate bits NAK's encoder only writes as constants.

/// A per-thread register, R0 to R254; R255 is `RZ`, which reads zero and discards writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Gpr(u8);

impl Gpr {
    /// `RZ` (NAK: `zero_reg(RegFile::GPR)` is register 255).
    pub const RZ: Self = Self(255);

    /// Register `index`; 255 is [`Gpr::RZ`].
    pub const fn new(index: u8) -> Self {
        Self(index)
    }

    /// The register number.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Whether this is [`Gpr::RZ`].
    pub const fn is_zero(self) -> bool {
        self.0 == Self::RZ.0
    }
}

/// A predicate register, P0 to P6; P7 is `PT`, which is always true.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pred(u8);

impl Pred {
    /// `PT` (NAK: `true_reg` is register 7).
    pub const PT: Self = Self(7);

    /// Predicate register `index`, or `None` past [`Pred::PT`]; the encoding has three bits.
    pub const fn new(index: u8) -> Option<Self> {
        if index <= Self::PT.0 {
            Some(Self(index))
        } else {
            None
        }
    }

    /// The register number.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Whether this is [`Pred::PT`].
    pub const fn is_true(self) -> bool {
        self.0 == Self::PT.0
    }
}

/// A predicate operand: a register, possibly negated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PredSrc {
    /// The register.
    pub reg: Pred,
    /// Whether the value is negated.
    pub negated: bool,
}

impl PredSrc {
    /// `PT`: the guard of an instruction that always runs.
    pub const TRUE: Self = Self {
        reg: Pred::PT,
        negated: false,
    };

    /// Whether this is plain `PT`.
    pub const fn is_true(self) -> bool {
        self.reg.is_true() && !self.negated
    }
}

/// A 32-bit float source of an ALU instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatSrc {
    /// A register, with NAK's `FAbs`/`FNeg` modifiers.
    Reg {
        /// The register.
        reg: Gpr,
        /// Take the absolute value first.
        abs: bool,
        /// Then negate.
        neg: bool,
    },
    /// A 32-bit immediate: the bits of an `f32`.
    Imm(u32),
}

impl FloatSrc {
    /// An unmodified register.
    pub const fn reg(reg: Gpr) -> Self {
        Self::Reg {
            reg,
            abs: false,
            neg: false,
        }
    }
}

/// The source of a `MOV`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MovSrc {
    /// A register.
    Reg(Gpr),
    /// A 32-bit immediate.
    Imm(u32),
}

/// A floating-point rounding mode (NAK's `FRndMode`, encoded by `set_rnd_mode`: 0 to nearest
/// even, 1 toward negative infinity, 2 toward positive infinity, 3 toward zero).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RoundMode {
    /// To nearest, ties to even.
    NearestEven,
    /// Toward negative infinity.
    NegInf,
    /// Toward positive infinity.
    PosInf,
    /// Toward zero.
    Zero,
}

impl RoundMode {
    /// Every mode, in encoding order.
    pub const ALL: [Self; 4] = [Self::NearestEven, Self::NegInf, Self::PosInf, Self::Zero];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::NearestEven => 0,
            Self::NegInf => 1,
            Self::PosInf => 2,
            Self::Zero => 3,
        }
    }

    /// The mode a two-bit encoding names; only the low two bits are read.
    pub const fn from_bits(bits: u64) -> Self {
        Self::ALL[(bits & 3) as usize]
    }
}

/// `MOV dst, src`: copies a register or an immediate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpMov {
    /// The destination.
    pub dst: Gpr,
    /// The source.
    pub src: MovSrc,
    /// Which lanes of each quad write (four bits; `0xf` is every lane).
    pub quad_lanes: u8,
}

impl OpMov {
    /// The mask of a `MOV` that every lane performs.
    pub const ALL_LANES: u8 = 0xf;
}

/// `FFMA dst, a, b, c`: `a * b + c` in 32-bit float, fused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFFma {
    /// The destination.
    pub dst: Gpr,
    /// `a`, `b`, `c`. `a` is always a register; at most one of `b` and `c` is an immediate.
    pub srcs: [FloatSrc; 3],
    /// Clamp the result to [0, 1].
    pub saturate: bool,
    /// The rounding mode.
    pub rnd: RoundMode,
    /// Flush denormal inputs and results to zero.
    pub ftz: bool,
    /// NAK's `dnz` bit; PR #4588 calls it the multiply-by-zero mode
    /// (`ops_alu.cpp`, `FFMA`: "native FFMA multiply-zero mode").
    pub dnz: bool,
}

/// `ALD dst, a[vtx][addr + offset]`: reads `comps` 32-bit words of attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpALd {
    /// The first destination register; the others follow it.
    pub dst: Gpr,
    /// The vertex (for stages that read several vertices); `RZ` for none.
    pub vtx: Gpr,
    /// A register added to the address; `RZ` for none.
    pub offset: Gpr,
    /// The attribute byte address (ten bits).
    pub addr: u16,
    /// How many 32-bit words, 1 to 4.
    pub comps: u8,
    /// Read per-patch attributes.
    pub patch: bool,
    /// Treat the address as physical.
    pub phys: bool,
    /// Read the stage's outputs instead of its inputs.
    pub output: bool,
}

/// `AST a[vtx][addr + offset], data`: writes `comps` 32-bit words of attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpASt {
    /// The first data register; the others follow it.
    pub data: Gpr,
    /// The vertex; `RZ` for none.
    pub vtx: Gpr,
    /// A register added to the address; `RZ` for none.
    pub offset: Gpr,
    /// The attribute byte address (ten bits).
    pub addr: u16,
    /// How many 32-bit words, 1 to 4.
    pub comps: u8,
    /// Write per-patch attributes.
    pub patch: bool,
    /// Treat the address as physical.
    pub phys: bool,
}

/// `EXIT`: ends the thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpExit {
    /// Bit 84 (NAK: `./.KEEPREFCOUNT/.PREEMPTED/.INVALID3` with bit 85).
    pub keep_ref_count: bool,
    /// Bit 85 (NAK: `.NO_ATEXIT`).
    pub no_at_exit: bool,
    /// The exit's own predicate (NAK always writes `PT`).
    pub pred: PredSrc,
}

/// A special register index, as `S2R` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpecialReg(pub u8);

impl SpecialReg {
    /// The lane's index in its warp (`SR_LANEID = 0`:
    /// `eden/src/shader_recompiler/frontend/maxwell/translate/impl/move_special_register.cpp`
    /// @ cf654052, `SpecialRegister`).
    pub const LANE_ID: Self = Self(0);
    /// `SR_TID_X = 33` (same enum).
    pub const TID_X: Self = Self(33);
    /// `SR_TID_Y = 34`.
    pub const TID_Y: Self = Self(34);
    /// `SR_TID_Z = 35`.
    pub const TID_Z: Self = Self(35);
    /// `SR_CTAID_X = 37`.
    pub const CTAID_X: Self = Self(37);
    /// `SR_CTAID_Y = 38`.
    pub const CTAID_Y: Self = Self(38);
    /// `SR_CTAID_Z = 39`.
    pub const CTAID_Z: Self = Self(39);
}

/// `S2R dst, sr`: reads a special register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpS2R {
    /// The destination.
    pub dst: Gpr,
    /// The special register.
    pub sr: SpecialReg,
}

/// A decoded instruction's operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// `MOV`.
    Mov(OpMov),
    /// `FFMA`.
    FFma(OpFFma),
    /// `ALD`.
    ALd(OpALd),
    /// `AST`.
    ASt(OpASt),
    /// `EXIT`.
    Exit(OpExit),
    /// `S2R`.
    S2R(OpS2R),
}

/// A decoded instruction: an operation under a guard predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Instr {
    /// The guard: the instruction runs only in threads where it is true.
    pub guard: PredSrc,
    /// The operation.
    pub op: Op,
}

impl Instr {
    /// `op`, always run.
    pub const fn new(op: Op) -> Self {
        Self {
            guard: PredSrc::TRUE,
            op,
        }
    }
}

//! Operand types every instruction group shares: registers of the four files (per-thread and
//! uniform registers, predicates and uniform predicates), constant-buffer references, ALU
//! sources with their modifier bits, packed-half sources with their swizzle, and the rounding
//! mode.
//!
//! The shapes follow NAK's `RegFile`, `CBuf`, `CBufRef` and `SrcMod`
//! (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` @ 52238762, MIT, Copyright 2022 Collabora,
//! Ltd.); the register counts and zero registers follow NAK's SM70 encoder (`sm70_encode.rs`,
//! `zero_reg`, `true_reg`, `set_ureg`), as each item says. Their printed forms are in `print`.

/// A per-thread register, R0 to R254; R255 is `RZ`, which reads zero and discards writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Gpr(u8);

impl Gpr {
    /// `RZ` (NAK: `zero_reg(RegFile::GPR)` is register 255).
    pub const RZ: Self = Self(255);
    /// The width of a register field (NAK's `set_reg` asserts an 8-bit range).
    pub const BITS: u32 = 8;

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

/// A uniform register, `UR0` to `UR62`, or `URZ`, which reads zero.
///
/// SM86 has 63 uniform registers and `URZ` is number 63: NAK encodes the uniform zero register
/// as `ugpr_max()`, 63 below SM 10.0 (`sm70_encode.rs`, `zero_reg`), and gives SM 7.x and 8.x 63
/// allocatable uniform registers (`sm70.rs`, `num_regs`: `RegFile::UGPR => 63`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UniformReg(u8);

impl UniformReg {
    /// `URZ`.
    pub const URZ: Self = Self(63);
    /// The width of a uniform register field below SM 10.0 (`sm70_encode.rs`, `set_ureg`:
    /// "Before SM100 UGPRs encode in 6 bits").
    pub const BITS: u32 = 6;

    /// Uniform register `index`, if it exists (`URZ` included).
    pub const fn new(index: u8) -> Option<Self> {
        if index <= Self::URZ.0 {
            Some(Self(index))
        } else {
            None
        }
    }

    /// The register's number.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Whether this is `URZ`.
    pub const fn is_zero(self) -> bool {
        self.0 == Self::URZ.0
    }
}

/// A predicate register, P0 to P6; P7 is `PT`, which is always true.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pred(u8);

impl Pred {
    /// `PT` (NAK: `true_reg` is register 7).
    pub const PT: Self = Self(7);
    /// The width of a predicate field (NAK's `set_pred_reg` asserts a 3-bit range).
    pub const BITS: u32 = 3;

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

/// A uniform predicate, `UP0` to `UP6`, or `UPT`, which reads true.
///
/// NAK's true register is number 7 in every predicate file (`sm70_encode.rs`, `true_reg`), and
/// SM 7.x and 8.x have seven allocatable uniform predicates (`sm70.rs`, `num_regs`:
/// `RegFile::UPred => 7`). Uniform predicate fields are three bits, like predicate fields
/// (`set_pred_reg` serves both files).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UniformPred(u8);

impl UniformPred {
    /// `UPT`.
    pub const UPT: Self = Self(7);
    /// The width of a uniform predicate field (`set_pred_reg`, which serves both predicate
    /// files, asserts a 3-bit range).
    pub const BITS: u32 = 3;

    /// Uniform predicate `index`, if it exists (`UPT` included).
    pub const fn new(index: u8) -> Option<Self> {
        if index <= Self::UPT.0 {
            Some(Self(index))
        } else {
            None
        }
    }

    /// The predicate's number.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Whether this is `UPT`.
    pub const fn is_true(self) -> bool {
        self.0 == Self::UPT.0
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

/// A uniform predicate operand: a uniform predicate, possibly negated (NAK's
/// `set_upred_src`: the register and a "not" bit, like [`PredSrc`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UniformPredSrc {
    /// The uniform predicate.
    pub reg: UniformPred,
    /// Whether the value is negated.
    pub negated: bool,
}

/// Which constant buffer a [`CBufRef`] reads (NAK's `CBuf`, without its SSA form).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CBuf {
    /// A bound constant buffer, 0 to 31 (`set_src_cb`: `v.set_field(22..27, idx)`, five bits).
    Binding(u8),
    /// The constant buffer whose handle is in a uniform register (`set_src_cb`:
    /// `CBuf::BindlessUGPR`, the register in the low six bits and the slot's "cx" bit set).
    BindlessUniform(UniformReg),
}

/// A constant-buffer operand: a buffer and a byte offset into it (NAK's `CBufRef`; the offset
/// is sixteen bits, `set_src_cb`: `v.set_field(6..22, cb.offset)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CBufRef {
    /// The buffer.
    pub buf: CBuf,
    /// The byte offset. It need not be word aligned: PR #4588 reads an unaligned offset as the
    /// 32 bits starting there (`eden-pr4588/src/shader_recompiler/frontend/ampere/visitor.cpp`
    /// @ e5fb3b97, `Visitor::Cbuf`).
    pub offset: u16,
}

/// A source of an ALU instruction, as the operand form selects it, with NAK's two modifier
/// bits.
///
/// What `abs` and `neg` mean depends on the instruction (NAK's `SrcMod`: `FAbs` and `FNeg` for
/// floating point, `INeg` or `BNot` for integers, which use the `neg` bit only); an instruction
/// whose form has no modifier bits keeps both false. An immediate has no modifier bits
/// (NAK's `ALUSrc::from_src` asserts an immediate is unmodified).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AluSrc {
    /// A per-thread register.
    Reg {
        /// The register.
        reg: Gpr,
        /// The absolute-value bit.
        abs: bool,
        /// The negation bit.
        neg: bool,
    },
    /// A uniform register (operand forms 6 and 7: `encode_alu_ureg`).
    UniformReg {
        /// The register.
        reg: UniformReg,
        /// The absolute-value bit.
        abs: bool,
        /// The negation bit.
        neg: bool,
    },
    /// A constant-buffer word (operand forms 3 and 5: `encode_alu_cb`).
    CBuf {
        /// The constant-buffer reference.
        cbuf: CBufRef,
        /// The absolute-value bit.
        abs: bool,
        /// The negation bit.
        neg: bool,
    },
    /// A 32-bit immediate (operand forms 2 and 4: `encode_alu_imm`).
    Imm(u32),
}

impl AluSrc {
    /// An unmodified per-thread register.
    pub const fn reg(reg: Gpr) -> Self {
        Self::Reg {
            reg,
            abs: false,
            neg: false,
        }
    }

    /// The source's modifier bits, `(abs, neg)`; an immediate has none.
    pub const fn modifiers(self) -> (bool, bool) {
        match self {
            Self::Reg { abs, neg, .. }
            | Self::UniformReg { abs, neg, .. }
            | Self::CBuf { abs, neg, .. } => (abs, neg),
            Self::Imm(_) => (false, false),
        }
    }
}

/// Which 16-bit halves of a packed-half source an fp16 instruction reads: NAK's `SrcSwizzle`
/// for an `F16v2` source, which `set_swizzle` encodes in a two-bit field as 0 (`None`, both
/// halves as stored), 2 (`Xx`, the low half in both lanes) and 3 (`Yy`, the high half in both).
/// The names are Eden's Maxwell `Swizzle` (`eden/src/shader_recompiler/frontend/maxwell/
/// translate/impl/half_floating_point_helper.h` @ cf654052c416: `H1_H0`, `F32`, `H0_H0`,
/// `H1_H1`).
///
/// Value 1 names nothing here: NAK writes it only for a single `F16` source (`Yy`), never for a
/// packed pair, and PR #4588's translator refuses it (`ops_half.cpp`, `Visitor::HalfLanes`:
/// `Require(operand.swizzle != 1, "native half source read as float")`), so the decoder refuses
/// it. Maxwell's `F32` (a 32-bit float operand) is the guess for what it means on SM86, for an
/// `nvdisasm` check (verification E2) to settle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HalfSwizzle {
    /// 0: both halves as stored (NAK's `None`).
    H1H0,
    /// 2: the low half in both lanes (NAK's `Xx`).
    H0H0,
    /// 3: the high half in both lanes (NAK's `Yy`).
    H1H1,
}

impl HalfSwizzle {
    /// Every swizzle, in encoding order.
    pub const ALL: [Self; 3] = [Self::H1H0, Self::H0H0, Self::H1H1];
    /// The width of a swizzle field (`set_swizzle` asserts `range.len() == 2`).
    pub const BITS: u32 = 2;

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::H1H0 => 0,
            Self::H0H0 => 2,
            Self::H1H1 => 3,
        }
    }

    /// The swizzle an encoding names: none for 1 (see the type's docs), nor for a value wider
    /// than the field.
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            0 => Some(Self::H1H0),
            2 => Some(Self::H0H0),
            3 => Some(Self::H1H1),
            _ => None,
        }
    }
}

/// A source of a packed-half instruction (`HADD2`, `HFMA2`, `HMUL2`, `HSET2`, `HSETP2`,
/// `HMNMX2`): an ALU source with its modifier bits, and the halves it reads. NAK's `Src` with
/// its `src_swizzle`, encoded with `SM70SrcType::F16v2` (`encode_fp16v2_alu`).
///
/// An immediate has no swizzle bits (NAK's `ALUSrc::from_src` asserts
/// `src.src_swizzle.is_none()` for one): the decoder gives it [`HalfSwizzle::H1H0`], and the
/// test assembler refuses any other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HalfSrc {
    /// The source and its modifier bits.
    pub src: AluSrc,
    /// The halves it reads.
    pub swizzle: HalfSwizzle,
}

impl HalfSrc {
    /// A source that reads both halves as stored.
    pub const fn new(src: AluSrc) -> Self {
        Self {
            src,
            swizzle: HalfSwizzle::H1H0,
        }
    }
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
    /// The width of a rounding-mode field (`set_rnd_mode` asserts a 2-bit range).
    pub const BITS: u32 = 2;

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::NearestEven => 0,
            Self::NegInf => 1,
            Self::PosInf => 2,
            Self::Zero => 3,
        }
    }

    /// The mode an encoding names; only the low [`RoundMode::BITS`] bits are read.
    pub const fn from_bits(bits: u64) -> Self {
        Self::ALL[(bits & ((1 << Self::BITS) - 1)) as usize]
    }
}

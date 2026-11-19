//! Slice 6a, the integer ALU: `MOV` is decoded; the other instructions are shells.
//!
//! `MOV`'s shape follows NAK's `OpMov` and its encoding NAK's `impl SM70Op for OpMov`
//! (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` and `sm70_encode.rs` @ 52238762, MIT,
//! Copyright 2022 Collabora, Ltd.): `encode_alu(0x002, dst, None, src, None)`, so its one source
//! is source 1 and its forms are 1, 4, 5 and 6. PR #4588 decodes the same four
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/visitor.cpp` @ e5fb3b97, `Visitor::Run`
//! sends opcodes `0x202`, `0x802`, `0xa02` and `0xc02` to `MOV`); its table's rows for forms 2,
//! 3 and 7 (`0x402`, `0x602`, `0xe02`) are not decoded here.
//!
//! Divergence from PR #4588's table: `MOV` has no source modifiers (NAK never modifies
//! `OpMov::src`), so bits 62 and 63, which the table's rows `0x202`, `0xa02` and `0xc02` mark
//! known, are rejected in every form; PR #4588's translator does not take them either
//! (`ops_move.cpp`, `MOV`, reads the slot with `WideSlot(false)`). Pinned by
//! `decode::tests::mov_rejects_source_modifier_bits`.

use std::fmt::{self, Display, Formatter};

use super::super::DecodeError;
use super::super::alu::{self, AluForm, Source, Sources};
use super::super::fields::{DST, mask_of};
use super::super::form::{Form, Reader};
use super::super::instruction::Field;
use super::super::operand::{AluSrc, CBufRef, Gpr, UniformReg};
use super::super::print::IntOperand;
use super::{Named, Op, shells};
#[cfg(test)]
use {
    super::super::fields::{SLOT_ABS, SLOT_NEG, SLOT_UNIFORM},
    super::super::form::ReferenceRow,
    proptest::strategy::BoxedStrategy,
};

/// `MOV`'s 9-bit opcode (`impl SM70Op for OpMov`: `encode_alu(0x002, ..)`).
pub(crate) const MOV_OPCODE: u16 = 0x002;
/// `MOV`'s per-quad-lane mask (`set_field(72..76, quad_lanes)`).
pub(crate) const MOV_QUAD_LANES: Field = Field::new(72, 4);
/// `MOV`'s sources: source 1 only.
pub(crate) const MOV_SOURCES: Sources = Sources {
    src0: false,
    src1: true,
    src2: false,
};
/// `MOV`'s operand forms.
pub(crate) const MOV_FORMS: [AluForm; 4] = [
    AluForm::Reg,
    AluForm::ImmSrc1,
    AluForm::CBufSrc1,
    AluForm::UniformSrc1,
];

/// The source of a `MOV`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MovSrc {
    /// A register (form 1).
    Reg(Gpr),
    /// A 32-bit immediate (form 4).
    Imm(u32),
    /// A constant-buffer word (form 5).
    CBuf(CBufRef),
    /// A uniform register (form 6).
    UniformReg(UniformReg),
}

impl MovSrc {
    /// The source as an unmodified ALU source.
    pub const fn to_alu(self) -> AluSrc {
        match self {
            Self::Reg(reg) => AluSrc::reg(reg),
            Self::Imm(imm) => AluSrc::Imm(imm),
            Self::CBuf(cbuf) => AluSrc::CBuf {
                cbuf,
                abs: false,
                neg: false,
            },
            Self::UniformReg(reg) => AluSrc::UniformReg {
                reg,
                abs: false,
                neg: false,
            },
        }
    }

    /// The `MOV` source an unmodified ALU source is; a modified one is none.
    pub const fn from_alu(src: AluSrc) -> Option<Self> {
        match src {
            AluSrc::Reg {
                reg,
                abs: false,
                neg: false,
            } => Some(Self::Reg(reg)),
            AluSrc::Imm(imm) => Some(Self::Imm(imm)),
            AluSrc::CBuf {
                cbuf,
                abs: false,
                neg: false,
            } => Some(Self::CBuf(cbuf)),
            AluSrc::UniformReg {
                reg,
                abs: false,
                neg: false,
            } => Some(Self::UniformReg(reg)),
            _ => None,
        }
    }
}

impl Display for MovSrc {
    /// A register, a constant buffer or a uniform register as itself; an immediate in
    /// hexadecimal (a guess, see `print`).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        IntOperand(self.to_alu()).fmt(f)
    }
}

/// `MOV dst, src`: copies a register, an immediate, a constant-buffer word or a uniform register.
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

impl Named for OpMov {
    const MNEMONIC: &'static str = "mov";
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

fn decode_mov(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let src = alu::src(reader, form, Source::One, false);
    Ok(Op::Mov(OpMov {
        dst: reader.gpr(DST),
        // Cannot fail: the source was read without modifiers.
        src: MovSrc::from_alu(src).unwrap_or(MovSrc::Reg(Gpr::RZ)),
        quad_lanes: reader.byte(MOV_QUAD_LANES),
    }))
}

/// `MOV` in `form`.
const fn mov(form: AluForm) -> Form {
    Form::new(
        form.opcode(MOV_OPCODE),
        mask_of(&[DST, MOV_QUAD_LANES]) | alu::known(form, MOV_SOURCES, false),
        decode_mov,
    )
    .with_fixed(alu::fixed(form))
}

/// The integer ALU's forms.
pub(crate) const FORMS: &[Form] = &[
    mov(MOV_FORMS[0]),
    mov(MOV_FORMS[1]),
    mov(MOV_FORMS[2]),
    mov(MOV_FORMS[3]),
];

/// PR #4588's rows for the group's forms, in [`FORMS`] order
/// (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97; each
/// row's four words, low word first). The rows of the register, constant-buffer and
/// uniform-register forms drop bits 62 and 63, the divergence in the module docs; the
/// uniform-register form requires bit 91.
#[cfg(test)]
pub(crate) const REFERENCE_ROWS: &[ReferenceRow] = &[
    ReferenceRow::new(0x202, [0x00ff_ffff, 0xc000_00ff, 0x0000_0f00, 0xffff_fe00])
        .dropping(MOV_DROPPED),
    ReferenceRow::new(0x802, [0x00ff_ffff, 0xffff_ffff, 0x0000_0f00, 0xffff_fe00]),
    ReferenceRow::new(0xa02, [0x00ff_ffff, 0xc7ff_ffff, 0x0800_0f00, 0xffff_fe00])
        .dropping(MOV_DROPPED),
    ReferenceRow::new(0xc02, [0x00ff_ffff, 0xc000_003f, 0x0800_0f00, 0xffff_fe00])
        .dropping(MOV_DROPPED)
        .fixing(SLOT_UNIFORM.mask()),
];

/// The bits PR #4588's `MOV` rows mark known that `MOV` rejects: the wide slot's modifiers.
#[cfg(test)]
pub(crate) const MOV_DROPPED: u128 = mask_of(&[SLOT_ABS, SLOT_NEG]);

/// Generators of the group's decoded operations, one per operation.
#[cfg(test)]
pub(crate) fn any_ops() -> Vec<BoxedStrategy<Op>> {
    use proptest::prelude::*;

    use crate::testing::{any_cbuf, any_gpr, any_uniform_reg};

    let src = prop_oneof![
        any_gpr().prop_map(MovSrc::Reg),
        any::<u32>().prop_map(MovSrc::Imm),
        any_cbuf().prop_map(MovSrc::CBuf),
        any_uniform_reg().prop_map(MovSrc::UniformReg),
    ];
    let mov =
        (any_gpr(), src, 0_u8..(1 << MOV_QUAD_LANES.width)).prop_map(|(dst, src, quad_lanes)| {
            Op::Mov(OpMov {
                dst,
                src,
                quad_lanes,
            })
        });
    vec![mov.boxed()]
}

shells! {
    IAdd3(OpIAdd3) = "iadd3";
    IAdd3X(OpIAdd3X) = "iadd3.x";
    IMad(OpIMad) = "imad";
    IMad64(OpIMad64) = "imad.wide";
    Lop3(OpLop3) = "lop3";
    Shf(OpShf) = "shf";
    Lea(OpLea) = "lea";
    LeaX(OpLeaX) = "lea.x";
    ISetP(OpISetP) = "isetp";
    IMnMx(OpIMnMx) = "imnmx";
    Prmt(OpPrmt) = "prmt";
    PopC(OpPopC) = "popc";
    Flo(OpFlo) = "flo";
    BRev(OpBRev) = "brev";
    Sgxt(OpSgxt) = "sgxt";
    BMsk(OpBMsk) = "bmsk";
    IAbs(OpIAbs) = "iabs";
    IDp4(OpIDp4) = "idp4";
    Sel(OpSel) = "sel";
    PLop3(OpPLop3) = "plop3";
    P2R(OpP2R) = "p2r";
    R2P(OpR2P) = "r2p";
}

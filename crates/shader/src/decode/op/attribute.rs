//! Slice 6g, attributes and system values: `ALD`, `AST` and `S2R` are decoded; the other
//! instructions are shells.
//!
//! Shapes follow NAK's `OpALd`, `OpASt` and `OpS2R` and encodings NAK's `impl SM70Op` blocks
//! for them (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` and `sm70_encode.rs` @ 52238762,
//! MIT, Copyright 2022 Collabora, Ltd.), as each constant says.

use std::fmt::{self, Display, Formatter};

use super::super::fields::{DST, mask_of};
use super::super::form::{Form, Reader};
use super::super::instruction::Field;
use super::super::operand::Gpr;
use super::super::print::flag;
use super::{Named, Op, shells};

/// `ALD` (`impl SM70Op for OpALd`: `set_opcode(0x321)`).
pub(crate) const ALD_OPCODE: u16 = 0x321;
/// `AST` (`impl SM70Op for OpASt`: `set_opcode(0x322)`).
pub(crate) const AST_OPCODE: u16 = 0x322;
/// The attribute offset register of `ALD` and `AST` (`set_reg_src(24..32, offset)`).
pub(crate) const ATTR_OFFSET: Field = Field::new(24, Gpr::BITS);
/// `ALD`'s vertex register (`set_reg_src(32..40, vtx)`).
pub(crate) const ALD_VTX: Field = Field::new(32, Gpr::BITS);
/// `AST`'s data register (`set_reg_src(32..40, data)`).
pub(crate) const AST_DATA: Field = Field::new(32, Gpr::BITS);
/// `AST`'s vertex register (`set_reg_src(64..72, vtx)`).
pub(crate) const AST_VTX: Field = Field::new(64, Gpr::BITS);
/// The attribute byte address of `ALD` and `AST` (`set_field(40..50, addr)`).
pub(crate) const ATTR_ADDR: Field = Field::new(40, 10);
/// The component count minus one of `ALD` and `AST` (`set_field(74..76, comps - 1)`).
pub(crate) const ATTR_COMPS: Field = Field::new(74, 2);
/// The per-patch bit of `ALD` and `AST` (`set_field(76..77, patch)`).
pub(crate) const ATTR_PATCH: Field = Field::bit(76);
/// The physical-address bit of `ALD` and `AST` (`set_field(77..78, phys)`).
pub(crate) const ATTR_PHYS: Field = Field::bit(77);
/// `ALD`'s read-an-output bit (`set_field(79..80, output)`).
pub(crate) const ALD_OUTPUT: Field = Field::bit(79);

/// `S2R` (`impl SM70Op for OpS2R`: `set_opcode(0x919)` for the non-uniform form).
pub(crate) const S2R_OPCODE: u16 = 0x919;
/// `S2R`'s special register index (`set_field(72..80, idx)`).
pub(crate) const S2R_SR: Field = Field::new(72, 8);

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

impl Named for OpALd {
    const MNEMONIC: &'static str = "ald";
}

impl Named for OpASt {
    const MNEMONIC: &'static str = "ast";
}

impl Named for OpS2R {
    const MNEMONIC: &'static str = "s2r";
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

impl Display for OpS2R {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "s2r {}, {}", self.dst, self.sr)
    }
}

fn attr_addr(reader: &mut Reader) -> u16 {
    // Cannot truncate: the field is ten bits wide.
    u16::try_from(reader.field(ATTR_ADDR)).unwrap_or(u16::MAX)
}

fn attr_comps(reader: &mut Reader) -> u8 {
    // The field holds the count minus one (`set_field(74..76, comps - 1)`); two bits, so 1..=4.
    reader.byte(ATTR_COMPS).saturating_add(1)
}

fn decode_ald(reader: &mut Reader) -> Op {
    Op::ALd(OpALd {
        dst: reader.gpr(DST),
        vtx: reader.gpr(ALD_VTX),
        offset: reader.gpr(ATTR_OFFSET),
        addr: attr_addr(reader),
        comps: attr_comps(reader),
        patch: reader.flag(ATTR_PATCH),
        phys: reader.flag(ATTR_PHYS),
        output: reader.flag(ALD_OUTPUT),
    })
}

fn decode_ast(reader: &mut Reader) -> Op {
    Op::ASt(OpASt {
        data: reader.gpr(AST_DATA),
        vtx: reader.gpr(AST_VTX),
        offset: reader.gpr(ATTR_OFFSET),
        addr: attr_addr(reader),
        comps: attr_comps(reader),
        patch: reader.flag(ATTR_PATCH),
        phys: reader.flag(ATTR_PHYS),
    })
}

fn decode_s2r(reader: &mut Reader) -> Op {
    Op::S2R(OpS2R {
        dst: reader.gpr(DST),
        sr: SpecialReg(reader.byte(S2R_SR)),
    })
}

/// The attribute and system-value group's forms.
pub(crate) const FORMS: &[Form] = &[
    Form::new(
        ALD_OPCODE,
        mask_of(&[
            DST,
            ATTR_OFFSET,
            ALD_VTX,
            ATTR_ADDR,
            ATTR_COMPS,
            ATTR_PATCH,
            ATTR_PHYS,
            ALD_OUTPUT,
        ]),
        |reader| Ok(decode_ald(reader)),
    ),
    Form::new(
        AST_OPCODE,
        mask_of(&[
            ATTR_OFFSET,
            AST_DATA,
            AST_VTX,
            ATTR_ADDR,
            ATTR_COMPS,
            ATTR_PATCH,
            ATTR_PHYS,
        ]),
        |reader| Ok(decode_ast(reader)),
    ),
    Form::new(S2R_OPCODE, mask_of(&[DST, S2R_SR]), |reader| {
        Ok(decode_s2r(reader))
    }),
];

shells! {
    Ipa(OpIpa) = "ipa";
    AL2P(OpAL2P) = "al2p";
    CS2R(OpCS2R) = "cs2r";
    PixLd(OpPixLd) = "pixld";
    Out(OpOut) = "out";
    OutFinal(OpOutFinal) = "out.final";
    Isberd(OpIsberd) = "isberd";
    Isbewr(OpIsbewr) = "isbewr";
    LdTram(OpLdTram) = "ldtram";
}

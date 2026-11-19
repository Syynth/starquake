//! Slice 6g, attributes and system values: attribute loads and stores (`ALD`, `AST`, `AL2P`,
//! `IPA`, `LDTRAM`), system values (`S2R` with its uniform form `S2UR`, `CS2R`, `PIXLD`),
//! geometry output (`OUT`, `OUT.FINAL`) and the internal stage buffer of mesh and task programs
//! (`ISBERD`, `ISBEWR`). Every instruction of the group is decoded; none is left a shell.
//!
//! # Sources
//!
//! Shapes follow NAK's `Op*` structs and encodings NAK's `impl SM70Op` blocks
//! (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` and `sm70_encode.rs` @ 52238762, MIT,
//! Copyright 2022 Collabora, Ltd.), as each constant says. Each form's known-bit mask is checked
//! against PR #4588's row for it (`eden-pr4588/src/shader_recompiler/frontend/ampere/`
//! `ampere_isa_table.h` @ e5fb3b97, [`REFERENCE_ROWS`]); PR #4588's translator reads `IPA`,
//! `ALD`, `AST`, `S2R`, `CS2R`, `ISBERD` and `ISBEWR` (`ops_attr.cpp`, `ops_misc.cpp`) and has
//! rows only for `AL2P`, `PIXLD`, `OUT` and `LDTRAM`. Every form here has an encoder in NAK, so
//! none comes from PR #4588 alone. Where NAK writes a field as a constant (`IPA`'s and
//! `PIXLD`'s predicate destination, `LDTRAM`'s uniform register, `AL2P`'s width, `OUT.FINAL`'s
//! destination and stream), the op carries it as a field so the decoder can read any value of
//! it.
//!
//! Owner's decision 11 (2026-11-12): mesh and task programs come after milestone 3, "with the
//! decoder reading their instructions from the start". `ISBERD` and `ISBEWR` are decoded in
//! full: every field NAK's encoder writes, over every value it defines.
//!
//! # Decisions taken here
//!
//! - **`S2UR`** (`0x9c3`) is a form of `S2R` with a uniform destination, as NAK models it
//!   (`impl SM70Op for OpS2R`: `if self.is_uniform() { set_opcode(0x9c3); set_udst(..) }`), so
//!   [`OpS2R`]'s destination is an [`S2RDst`]. PR #4588's row names it `S2R`.
//! - **`OUT.FINAL`** is `OUT`'s opcode with the output type 0: NAK's `impl SM70Op for
//!   OpOutFinal` writes `encode_alu(0x124, None, handle, ZERO, None)` and leaves bits 78..80
//!   clear, and `OpOut` writes 1 to 3 there. So a word of `OUT`'s forms decodes as
//!   [`OpOutFinal`] when the type is 0 and as [`OpOut`] otherwise; both carry the destination
//!   and the stream.
//! - **The interpolation modes.** `IPA`'s location (bits 76..78) and frequency (bits 78..80) are
//!   NAK's `InterpLoc` (default 0, centroid 1, offset 2) and `InterpFreq` (pass 0, constant 1,
//!   state 2). Interpolation at a sample has no mode of its own: NAK lowers it to `.offset`
//!   with the sample's position as the offset, and per-sample shading to `.centroid`
//!   (`nak_nir_lower_fs_inputs.c`, `load_barycentric_at_sample` and
//!   `load_barycentric_sample`). Offset and sample interpolation are the modes naga cannot
//!   express (spec, the gap table: "Interpolation at an offset or a sample"); the decoder reads
//!   them all and leaves the refusal to the translator.
//! - **Special register names** are Eden's `SpecialRegister` enumerators, lower-cased
//!   ([`SpecialReg`]), and NAK's `NAK_SV_VARIABLE_RATE` (132), which Eden's list stops short
//!   of; NAK prints only the number. Index 255 is printed `srz`, a guess: PR #4588 reads
//!   `CS2R`'s 255 as zero ("native special register pair", `ops_misc.cpp`). Every name is a
//!   guess at `nvdisasm`'s spelling.
//!
//! # Divergences from PR #4588's table
//!
//! | Rows | What this decoder does | Pinned by |
//! |---|---|---|
//! | `OUT` forms 2, 3 and 7 (`0x524`, `0x724`, `0xf24`) | Unknown opcodes: NAK's `encode_alu(.., Some(handle), Some(stream), None)` puts the stream in source 1, which only forms 1, 4, 5 and 6 hold, as for `MOV` and `WARPSYNC` | `decode::tests::attribute::out_rows_left_undecoded_are_unknown_opcodes` |
//! | Bits 72 and 73 of `OUT` forms 1, 4, 5 and 6, bits 62 and 63 of forms 1, 5 and 6 | Rejected: NAK's handle and stream take no modifiers (`OpOut`: `#[src_type(SSA)]` and `#[src_type(ALU)]`, which `Src::supports_type` allows only unmodified) | `out_rejects_source_modifier_bits` |
//!
//! Every other form's mask is its row's exactly (`decode::tests::masks_match_the_reference_table`).
//!
//! # Values the decoder refuses
//!
//! A field whose value names nothing is [`DecodeError::UnknownValue`]: `IPA`'s location 3 (NAK
//! defines 0 to 2) and frequency 3 (NAK's `PassMulW` "is invalid on SM70+"), read in that
//! order; `PIXLD`'s values 5 to 7 (NAK encodes 0 to 4, `other => panic!("Unsupported PixVal")`);
//! `ISBERD`'s and `ISBEWR`'s memory type 3 (NAK: `U8` 0, `U16` 1, `B32` 2; PR #4588:
//! `Require(mem_type <= 2)`); and `ISBEWR`'s access types 1 and 2 (NAK: "Invalid ISBEWR access
//! type" for any but `Map` 0 and `Attribute` 3), read after the memory type. PR #4588's
//! translator is stricter for `ISBEWR` (`Require(mem_type == 2 && access == 3 && output)`) and
//! `IPA` (`Require(Take(76, 2) <= 1)`, frequency 0 or 1, no offset register, `PT` predicate):
//! that is a translator's refusal, which slice 12g decides, not a decoding one.
//!
//! # Not decided here
//!
//! The text of every instruction but `ISBERD` and `ISBEWR` (NAK's `test_isberd` and
//! `test_isbewr`) is a guess in `nvdisasm`'s style (`print`'s module docs).

#[cfg(test)]
mod generate;

use std::fmt::{self, Display, Formatter};

use super::super::DecodeError;
use super::super::alu::{self, AluForm, Source, Sources};
use super::super::fields::{DST, SRC0, mask_of};
use super::super::form::{Form, Reader};
use super::super::instruction::Field;
use super::super::operand::{AluSrc, Gpr, Pred, UniformReg};
use super::super::print::{IntOperand, flag};
use super::control::{PRED_DST, SOURCE_1_FORMS, UNIFORM_DST};
use super::{Named, Op, shells};
#[cfg(test)]
use {
    super::super::fields::{SLOT_ABS, SLOT_NEG, SRC0_ABS, SRC0_NEG},
    super::super::form::ReferenceRow,
    proptest::strategy::BoxedStrategy,
};

// ---------------------------------------------------------------------------------------------
// Fields several instructions share.

/// The offset register of `ALD`, `AST`, `AL2P`, `ISBERD` and `ISBEWR` (NAK: `set_reg_src(24..32,
/// offset)` in each).
pub(crate) const ATTR_OFFSET: Field = Field::new(24, Gpr::BITS);
/// The data register of `AST` and `ISBEWR` (`set_reg_src(32..40, data)`).
pub(crate) const ATTR_DATA: Field = Field::new(32, Gpr::BITS);
/// The attribute byte address of `ALD`, `AST` and `AL2P` (`set_field(40..50, addr)`).
pub(crate) const ATTR_ADDR: Field = Field::new(40, 10);
/// The component count minus one of `ALD`, `AST` and `AL2P` (`set_field(74..76, comps - 1)`;
/// `AL2P` writes 0, "comps").
pub(crate) const ATTR_COMPS: Field = Field::new(74, 2);
/// The output bit of `ALD`, `AL2P`, `ISBERD` and `ISBEWR` (`set_field(79..80, output)`,
/// `set_bit(79, output)`).
pub(crate) const ATTR_OUTPUT: Field = Field::bit(79);
/// The attribute address in 32-bit words of `IPA` and `LDTRAM` (`set_field(64..72, addr >> 2)`
/// after `assert!(self.addr % 4 == 0)`).
pub(crate) const ATTR_WORD: Field = Field::new(64, 8);
/// The size of [`ATTR_WORD`]'s unit in bytes: the shift of `addr >> 2`.
pub(crate) const ATTR_WORD_BYTES: u16 = 4;
/// The special register index of `S2R`, `S2UR` and `CS2R` (`set_field(72..80, idx)`).
pub(crate) const SR_INDEX: Field = Field::new(72, 8);

// ---------------------------------------------------------------------------------------------
// ALD, AST and AL2P.

/// `ALD` (`impl SM70Op for OpALd`: `set_opcode(0x321)`).
pub(crate) const ALD_OPCODE: u16 = 0x321;
/// `AST` (`impl SM70Op for OpASt`: `set_opcode(0x322)`).
pub(crate) const AST_OPCODE: u16 = 0x322;
/// `AL2P` (`impl SM70Op for OpAL2P`: `set_opcode(0x920)`).
pub(crate) const AL2P_OPCODE: u16 = 0x920;
/// `ALD`'s vertex register (`set_reg_src(32..40, vtx)`).
pub(crate) const ALD_VTX: Field = Field::new(32, Gpr::BITS);
/// `AST`'s vertex register (`set_reg_src(64..72, vtx)`).
pub(crate) const AST_VTX: Field = Field::new(64, Gpr::BITS);
/// The per-patch bit of `ALD` and `AST` (`set_field(76..77, patch)`).
pub(crate) const ATTR_PATCH: Field = Field::bit(76);
/// The physical-address bit of `ALD` and `AST` (`set_field(77..78, phys)`).
pub(crate) const ATTR_PHYS: Field = Field::bit(77);

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

/// `AL2P dst, a[addr + offset]`: converts an attribute address to a physical one (NAK's
/// `OpAL2P`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpAL2P {
    /// The destination.
    pub dst: Gpr,
    /// A register added to the address; `RZ` for none.
    pub offset: Gpr,
    /// The attribute byte address (ten bits).
    pub addr: u16,
    /// The width in 32-bit words, 1 to 4 (NAK always writes 1: `set_field(74..76, 0_u8)`).
    pub comps: u8,
    /// Convert an output address instead of an input one.
    pub output: bool,
}

impl Named for OpALd {
    const MNEMONIC: &'static str = "ald";
}

impl Named for OpASt {
    const MNEMONIC: &'static str = "ast";
}

impl Named for OpAL2P {
    const MNEMONIC: &'static str = "al2p";
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

/// `a[vtx][addr+offset]`, leaving out `RZ` parts (NAK's `OpALd`/`OpASt`/`OpAL2P` printers).
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

impl Display for OpAL2P {
    /// NAK's `al2p.o a[0x..+r..]`, with the destination and width as `ALD` prints them.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("al2p")?;
        flag(f, self.output, ".o")?;
        let addr = AttrAddr {
            vtx: Gpr::RZ,
            offset: self.offset,
            addr: self.addr,
        };
        write!(f, "{} {}, {addr}", width(self.comps), self.dst)
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
        output: reader.flag(ATTR_OUTPUT),
    })
}

fn decode_ast(reader: &mut Reader) -> Op {
    Op::ASt(OpASt {
        data: reader.gpr(ATTR_DATA),
        vtx: reader.gpr(AST_VTX),
        offset: reader.gpr(ATTR_OFFSET),
        addr: attr_addr(reader),
        comps: attr_comps(reader),
        patch: reader.flag(ATTR_PATCH),
        phys: reader.flag(ATTR_PHYS),
    })
}

fn decode_al2p(reader: &mut Reader) -> Op {
    Op::AL2P(OpAL2P {
        dst: reader.gpr(DST),
        offset: reader.gpr(ATTR_OFFSET),
        addr: attr_addr(reader),
        comps: attr_comps(reader),
        output: reader.flag(ATTR_OUTPUT),
    })
}

/// Reads a predicate destination at bits 81..84 ([`PRED_DST`]); `PT` writes nothing.
fn pred_dst(reader: &mut Reader) -> Pred {
    // Cannot fail: the field is `Pred::BITS` wide.
    Pred::new(reader.byte(PRED_DST)).unwrap_or(Pred::PT)
}

/// Reads [`ATTR_WORD`] as a byte address.
fn attr_word(reader: &mut Reader) -> u16 {
    // Cannot overflow: eight bits of words are at most 0x3fc bytes.
    u16::from(reader.byte(ATTR_WORD)).saturating_mul(ATTR_WORD_BYTES)
}

// ---------------------------------------------------------------------------------------------
// IPA and LDTRAM.

/// `IPA` (`impl SM70Op for OpIpa`: `set_opcode(0x326)`).
pub(crate) const IPA_OPCODE: u16 = 0x326;
/// `IPA`'s offset register (`set_reg_src(32..40, offset)`), which `.offset` reads.
pub(crate) const IPA_OFFSET: Field = Field::new(32, Gpr::BITS);
/// `IPA`'s location (`set_field(76..78, ..)`: [`InterpLoc`]).
pub(crate) const IPA_LOC: Field = Field::new(76, 2);
/// `IPA`'s frequency (`set_field(78..80, ..)`: [`InterpFreq`]).
pub(crate) const IPA_FREQ: Field = Field::new(78, 2);

/// `LDTRAM` (`impl SM70Op for OpLdTram`: `set_opcode(0x3ad)`).
pub(crate) const LDTRAM_OPCODE: u16 = 0x3ad;
/// `LDTRAM`'s uniform register (`set_ureg(24, e.zero_reg(RegFile::UGPR))`, six bits below
/// SM 10.0).
pub(crate) const LDTRAM_UREG: Field = Field::new(24, UniformReg::BITS);
/// `LDTRAM`'s `.C` bit (`set_bit(72, self.use_c)`).
pub(crate) const LDTRAM_C: Field = Field::bit(72);
/// `LDTRAM`'s fixed bit (`set_bit(91, true)`: "Unknown but required").
pub(crate) const LDTRAM_FIXED: Field = Field::bit(91);

/// Where `IPA` interpolates (NAK's `InterpLoc` and its encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InterpLoc {
    /// 0: the pixel centre.
    Default,
    /// 1: the centroid (also NAK's per-sample shading).
    Centroid,
    /// 2: at the offset in [`OpIpa::offset`] (also NAK's interpolation at a sample).
    Offset,
}

impl InterpLoc {
    /// Every location, in encoding order.
    pub const ALL: [Self; 3] = [Self::Default, Self::Centroid, Self::Offset];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::Default => 0,
            Self::Centroid => 1,
            Self::Offset => 2,
        }
    }
}

/// How `IPA` interpolates (NAK's `InterpFreq` without `PassMulW`, which "is invalid on SM70+").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InterpFreq {
    /// 0: interpolated.
    Pass,
    /// 1: flat.
    Constant,
    /// 2: a value of the rasteriser's state.
    State,
}

impl InterpFreq {
    /// Every frequency, in encoding order.
    pub const ALL: [Self; 3] = [Self::Pass, Self::Constant, Self::State];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::Pass => 0,
            Self::Constant => 1,
            Self::State => 2,
        }
    }
}

/// `IPA dst, a[addr]`: interpolates a pixel program's input attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpIpa {
    /// The destination.
    pub dst: Gpr,
    /// The attribute byte address, a multiple of 4 up to `0x3fc`.
    pub addr: u16,
    /// The frequency.
    pub freq: InterpFreq,
    /// The location.
    pub loc: InterpLoc,
    /// The offset `.offset` interpolates at; NAK writes `RZ` for the other locations.
    pub offset: Gpr,
    /// A predicate destination (NAK always writes `PT`: `set_pred_dst(81..84, &Dst::None)`,
    /// "TODO: What is this for?"; PR #4588 requires `PT`).
    pub pred: Pred,
}

/// `LDTRAM dst, a[addr]`: reads an attribute of a pixel program's triangle (NAK's `OpLdTram`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpLdTram {
    /// The destination.
    pub dst: Gpr,
    /// The attribute byte address, a multiple of 4 up to `0x3fc`.
    pub addr: u16,
    /// `.C` rather than `.AB`.
    pub use_c: bool,
    /// A uniform register (NAK always writes `URZ`).
    pub ureg: UniformReg,
}

impl Named for OpIpa {
    const MNEMONIC: &'static str = "ipa";
}

impl Named for OpLdTram {
    const MNEMONIC: &'static str = "ldtram";
}

impl Display for InterpFreq {
    /// NAK's `InterpFreq` printer.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pass => ".pass",
            Self::Constant => ".constant",
            Self::State => ".state",
        })
    }
}

impl Display for InterpLoc {
    /// NAK's `InterpLoc` printer: nothing for the default.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Default => "",
            Self::Centroid => ".centroid",
            Self::Offset => ".offset",
        })
    }
}

impl Display for OpIpa {
    /// NAK's `ipa{freq}{loc} a[addr]`, with the destination and a predicate other than `PT`
    /// before the attribute, and the offset after it for `.offset` or when not `RZ`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ipa{}{} ", self.freq, self.loc)?;
        if !self.pred.is_true() {
            write!(f, "{}, ", self.pred)?;
        }
        write!(f, "{}, a[{:#x}]", self.dst, self.addr)?;
        if self.loc == InterpLoc::Offset || !self.offset.is_zero() {
            write!(f, ", {}", self.offset)?;
        }
        Ok(())
    }
}

impl Display for OpLdTram {
    /// NAK's `ldtram.c a[addr]` or `ldtram.ab a[addr]`, with the destination, and the uniform
    /// register when not `URZ`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(if self.use_c { "ldtram.c" } else { "ldtram.ab" })?;
        write!(f, " {}, a[{:#x}", self.dst, self.addr)?;
        if !self.ureg.is_zero() {
            write!(f, "+{}", self.ureg)?;
        }
        f.write_str("]")
    }
}

fn decode_ipa(reader: &mut Reader) -> Result<Op, DecodeError> {
    let loc_bits = reader.field(IPA_LOC);
    let loc = match loc_bits {
        0 => InterpLoc::Default,
        1 => InterpLoc::Centroid,
        2 => InterpLoc::Offset,
        _ => return Err(reader.unknown_value(IPA_LOC, loc_bits)),
    };
    let freq_bits = reader.field(IPA_FREQ);
    let freq = match freq_bits {
        0 => InterpFreq::Pass,
        1 => InterpFreq::Constant,
        2 => InterpFreq::State,
        _ => return Err(reader.unknown_value(IPA_FREQ, freq_bits)),
    };
    Ok(Op::Ipa(OpIpa {
        dst: reader.gpr(DST),
        addr: attr_word(reader),
        freq,
        loc,
        offset: reader.gpr(IPA_OFFSET),
        pred: pred_dst(reader),
    }))
}

fn decode_ldtram(reader: &mut Reader) -> Op {
    // A fixed bit of the form: the decoder has checked that it is set.
    reader.flag(LDTRAM_FIXED);
    Op::LdTram(OpLdTram {
        dst: reader.gpr(DST),
        addr: attr_word(reader),
        use_c: reader.flag(LDTRAM_C),
        ureg: reader.uniform_reg(LDTRAM_UREG),
    })
}

// ---------------------------------------------------------------------------------------------
// S2R, S2UR, CS2R and PIXLD.

/// `S2R` (`impl SM70Op for OpS2R`: `set_opcode(0x919)` for the per-thread form).
pub(crate) const S2R_OPCODE: u16 = 0x919;
/// `S2UR`, `S2R` with a uniform destination (`set_opcode(0x9c3)`, `set_udst`).
pub(crate) const S2UR_OPCODE: u16 = 0x9c3;
/// `CS2R` (`impl SM70Op for OpCS2R`: `set_opcode(0x805)`).
pub(crate) const CS2R_OPCODE: u16 = 0x805;
/// `CS2R`'s 64-bit bit (`set_bit(80, comps == 2); // .64`).
pub(crate) const CS2R_64: Field = Field::bit(80);
/// `PIXLD` (`impl SM70Op for OpPixLd`: `set_opcode(0x925)`).
pub(crate) const PIXLD_OPCODE: u16 = 0x925;
/// `PIXLD`'s value (`set_field(78..81, ..)`: [`PixVal`]).
pub(crate) const PIXLD_VAL: Field = Field::new(78, 3);

/// A special register index, as `S2R` and `CS2R` name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpecialReg(pub u8);

/// The names of special registers 0 to 99: Eden's `SpecialRegister` enumerators, lower-cased,
/// with `_X`, `_Y` and `_Z` of the thread and block ids written `.x`, `.y`, `.z`
/// (`eden/src/shader_recompiler/frontend/maxwell/translate/impl/move_special_register.cpp`
/// @ cf654052). `None` where Eden's enumerator is a bare number (`SR12`). The numbering is
/// Maxwell's. NAK's SM 7.0 and later numbering (`nak_private.h`, `enum nak_sv`) agrees with it
/// at every index NAK names (0, 2, 3, 15, 16, 17, 19, 28, 29, 32 to 35, 37 to 39, 56 to 60, 80
/// and 81) and PR #4588 reads 0, 33 to 39 and 56 to 60 by it (`ops_misc.cpp`, `S2R`); that SM86
/// keeps Eden's numbers at the other indices is a guess `nvdisasm` would confirm.
const SPECIAL_REG_NAMES: [Option<&str>; 100] = [
    Some("sr_laneid"),
    Some("sr_clock"),
    Some("sr_virtcfg"),
    Some("sr_virtid"),
    Some("sr_pm0"),
    Some("sr_pm1"),
    Some("sr_pm2"),
    Some("sr_pm3"),
    Some("sr_pm4"),
    Some("sr_pm5"),
    Some("sr_pm6"),
    Some("sr_pm7"),
    None,
    None,
    None,
    Some("sr_ordering_ticket"),
    Some("sr_prim_type"),
    Some("sr_invocation_id"),
    Some("sr_y_direction"),
    Some("sr_thread_kill"),
    Some("sm_shader_type"),
    Some("sr_directcbewriteaddresslow"),
    Some("sr_directcbewriteaddresshigh"),
    Some("sr_directcbewriteenable"),
    Some("sr_machine_id_0"),
    Some("sr_machine_id_1"),
    Some("sr_machine_id_2"),
    Some("sr_machine_id_3"),
    Some("sr_affinity"),
    Some("sr_invocation_info"),
    Some("sr_wscalefactor_xy"),
    Some("sr_wscalefactor_z"),
    Some("sr_tid"),
    Some("sr_tid.x"),
    Some("sr_tid.y"),
    Some("sr_tid.z"),
    Some("sr_cta_param"),
    Some("sr_ctaid.x"),
    Some("sr_ctaid.y"),
    Some("sr_ctaid.z"),
    Some("sr_ntid"),
    Some("sr_cirqueueincrminusone"),
    Some("sr_nlatc"),
    None,
    Some("sr_sm_spa_version"),
    Some("sr_multipassshaderinfo"),
    Some("sr_lwinhi"),
    Some("sr_swinhi"),
    Some("sr_swinlo"),
    Some("sr_swinsz"),
    Some("sr_smemsz"),
    Some("sr_smembanks"),
    Some("sr_lwinlo"),
    Some("sr_lwinsz"),
    Some("sr_lmemlosz"),
    Some("sr_lmemhioff"),
    Some("sr_eqmask"),
    Some("sr_ltmask"),
    Some("sr_lemask"),
    Some("sr_gtmask"),
    Some("sr_gemask"),
    Some("sr_regalloc"),
    Some("sr_barrieralloc"),
    None,
    Some("sr_globalerrorstatus"),
    None,
    Some("sr_warperrorstatus"),
    Some("sr_warperrorstatusclear"),
    None,
    None,
    None,
    None,
    Some("sr_pm_hi0"),
    Some("sr_pm_hi1"),
    Some("sr_pm_hi2"),
    Some("sr_pm_hi3"),
    Some("sr_pm_hi4"),
    Some("sr_pm_hi5"),
    Some("sr_pm_hi6"),
    Some("sr_pm_hi7"),
    Some("sr_clocklo"),
    Some("sr_clockhi"),
    Some("sr_globaltimerlo"),
    Some("sr_globaltimerhi"),
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    Some("sr_hwtaskid"),
    Some("sr_circularqueueentryindex"),
    Some("sr_circularqueueentryaddresslow"),
    Some("sr_circularqueueentryaddresshigh"),
];

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
    /// `SR_CLOCKLO = 80`, the low word of the clock, which NAK's `CS2R` reads (`nak_private.h`:
    /// `NAK_SV_CLOCK_LO = 0x50`).
    pub const CLOCK_LO: Self = Self(80);
    /// NAK's `NAK_SV_VARIABLE_RATE = 0x84` (`nak_private.h`), past Eden's list.
    pub const VARIABLE_RATE: Self = Self(0x84);
    /// The zero register, 255, which `CS2R` reads as zero (PR #4588, `ops_misc.cpp`, `CS2R`:
    /// `Require(Take(72, 8) == 255, "native special register pair")`, then writes 0).
    pub const SRZ: Self = Self(255);

    /// The register's name, if the references name it ([`SpecialReg::SRZ`] is `srz`, a guess).
    pub fn name(self) -> Option<&'static str> {
        match self {
            Self::SRZ => return Some("srz"),
            Self::VARIABLE_RATE => return Some("sr_variable_rate"),
            _ => {}
        }
        SPECIAL_REG_NAMES
            .get(usize::from(self.0))
            .copied()
            .flatten()
    }
}

impl Display for SpecialReg {
    /// The name, or `sr` and the number when the references give none.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => f.write_str(name),
            None => write!(f, "sr{}", self.0),
        }
    }
}

/// Where `S2R` writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum S2RDst {
    /// A per-thread register (form `0x919`, `S2R`).
    Reg(Gpr),
    /// A uniform register (form `0x9c3`, `S2UR`).
    Uniform(UniformReg),
}

/// `S2R dst, sr`: reads a special register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpS2R {
    /// The destination.
    pub dst: S2RDst,
    /// The special register.
    pub sr: SpecialReg,
}

/// `CS2R dst, sr`: reads a special register, or a pair, with lower latency than `S2R`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpCS2R {
    /// The destination; with [`OpCS2R::wide`], the first of two.
    pub dst: Gpr,
    /// The special register.
    pub sr: SpecialReg,
    /// `.64`: read two registers into two.
    pub wide: bool,
}

/// What `PIXLD` reads (NAK's `PixVal`, without `Covered` and `Offset`, which NAK's SM70 encoder
/// does not encode).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixVal {
    /// 0.
    MsCount,
    /// 1.
    CovMask,
    /// 2.
    CentroidOffset,
    /// 3.
    MyIndex,
    /// 4.
    InnerCoverage,
}

impl PixVal {
    /// Every value, in encoding order.
    pub const ALL: [Self; 5] = [
        Self::MsCount,
        Self::CovMask,
        Self::CentroidOffset,
        Self::MyIndex,
        Self::InnerCoverage,
    ];

    /// The three-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::MsCount => 0,
            Self::CovMask => 1,
            Self::CentroidOffset => 2,
            Self::MyIndex => 3,
            Self::InnerCoverage => 4,
        }
    }
}

/// `PIXLD dst`: reads a value of the pixel's coverage or samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpPixLd {
    /// The destination.
    pub dst: Gpr,
    /// What it reads.
    pub val: PixVal,
    /// A predicate destination (NAK always writes `PT`: `set_pred_dst(81..84, &Dst::None)`).
    pub pred: Pred,
}

impl Named for OpS2R {
    const MNEMONIC: &'static str = "s2r";
}

impl Named for OpCS2R {
    const MNEMONIC: &'static str = "cs2r";
}

impl Named for OpPixLd {
    const MNEMONIC: &'static str = "pixld";
}

impl Display for OpS2R {
    /// `s2r r3, sr_laneid` or `s2ur ur3, sr_laneid`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.dst {
            S2RDst::Reg(dst) => write!(f, "s2r {dst}, {}", self.sr),
            S2RDst::Uniform(dst) => write!(f, "s2ur {dst}, {}", self.sr),
        }
    }
}

impl Display for OpCS2R {
    /// `cs2r r2, sr_clocklo` for the pair, `cs2r.32 r2, sr_clocklo` for one register.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("cs2r")?;
        flag(f, !self.wide, ".32")?;
        write!(f, " {}, {}", self.dst, self.sr)
    }
}

impl Display for PixVal {
    /// NAK's `PixVal` printer.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MsCount => ".mscount",
            Self::CovMask => ".covmask",
            Self::CentroidOffset => ".centroid_offset",
            Self::MyIndex => ".my_index",
            Self::InnerCoverage => ".inner_coverage",
        })
    }
}

impl Display for OpPixLd {
    /// NAK's `pixld{val}`, with a predicate other than `PT` and the destination.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "pixld{} ", self.val)?;
        if !self.pred.is_true() {
            write!(f, "{}, ", self.pred)?;
        }
        write!(f, "{}", self.dst)
    }
}

fn decode_s2r(reader: &mut Reader) -> Op {
    let dst = if reader.opcode() == u64::from(S2UR_OPCODE) {
        S2RDst::Uniform(reader.uniform_reg(UNIFORM_DST))
    } else {
        S2RDst::Reg(reader.gpr(DST))
    };
    Op::S2R(OpS2R {
        dst,
        sr: SpecialReg(reader.byte(SR_INDEX)),
    })
}

fn decode_cs2r(reader: &mut Reader) -> Op {
    Op::CS2R(OpCS2R {
        dst: reader.gpr(DST),
        sr: SpecialReg(reader.byte(SR_INDEX)),
        wide: reader.flag(CS2R_64),
    })
}

fn decode_pixld(reader: &mut Reader) -> Result<Op, DecodeError> {
    let bits = reader.field(PIXLD_VAL);
    let val = match bits {
        0 => PixVal::MsCount,
        1 => PixVal::CovMask,
        2 => PixVal::CentroidOffset,
        3 => PixVal::MyIndex,
        4 => PixVal::InnerCoverage,
        _ => return Err(reader.unknown_value(PIXLD_VAL, bits)),
    };
    Ok(Op::PixLd(OpPixLd {
        dst: reader.gpr(DST),
        val,
        pred: pred_dst(reader),
    }))
}

// ---------------------------------------------------------------------------------------------
// OUT and OUT.FINAL.

/// `OUT`'s and `OUT.FINAL`'s 9-bit ALU opcode (`encode_alu(0x124, ..)` in `impl SM70Op for
/// OpOut` and `OpOutFinal`).
pub(crate) const OUT_OPCODE: u16 = 0x124;
/// `OUT`'s output type (`set_field(78..80, ..)`: [`OutType`]); 0 is `OUT.FINAL`, which leaves
/// the field clear.
pub(crate) const OUT_TYPE: Field = Field::new(78, 2);
/// The sources of `OUT`: the handle in source 0 and the stream in source 1.
pub(crate) const OUT_SOURCES: Sources = Sources {
    src0: true,
    src1: true,
    src2: false,
};

/// What `OUT` does with the primitive (NAK's `OutType` and its encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutType {
    /// 1: emits a vertex.
    Emit,
    /// 2: ends the primitive.
    Cut,
    /// 3: both.
    EmitThenCut,
}

impl OutType {
    /// Every type, in encoding order.
    pub const ALL: [Self; 3] = [Self::Emit, Self::Cut, Self::EmitThenCut];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::Emit => 1,
            Self::Cut => 2,
            Self::EmitThenCut => 3,
        }
    }
}

/// `OUT.{type} dst, handle, stream`: a geometry program's vertex output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpOut {
    /// The destination: the next handle.
    pub dst: Gpr,
    /// The handle.
    pub handle: Gpr,
    /// The stream, unmodified: a register, an immediate, a constant-buffer word or a uniform
    /// register.
    pub stream: AluSrc,
    /// The type.
    pub out_type: OutType,
}

/// `OUT.FINAL dst, handle, stream`: ends a geometry program's output (NAK writes `RZ` for the
/// destination and the stream).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpOutFinal {
    /// The destination.
    pub dst: Gpr,
    /// The handle.
    pub handle: Gpr,
    /// The stream, unmodified.
    pub stream: AluSrc,
}

impl Named for OpOut {
    const MNEMONIC: &'static str = "out";
}

impl Named for OpOutFinal {
    const MNEMONIC: &'static str = "out.final";
}

impl Display for OutType {
    /// NAK's `OutType` printer.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Emit => "emit",
            Self::Cut => "cut",
            Self::EmitThenCut => "emit_then_cut",
        })
    }
}

impl Display for OpOut {
    /// NAK's `out.{type} handle stream`, with the destination first.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "out.{} {}, {}, {}",
            self.out_type,
            self.dst,
            self.handle,
            IntOperand(self.stream)
        )
    }
}

impl Display for OpOutFinal {
    /// `out.final dst, handle, stream`, as `OUT` prints.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "out.final {}, {}, {}",
            self.dst,
            self.handle,
            IntOperand(self.stream)
        )
    }
}

fn decode_out(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let dst = reader.gpr(DST);
    let handle = reader.gpr(SRC0);
    let stream = alu::src(reader, form, Source::One, false);
    let out_type = match reader.field(OUT_TYPE) {
        1 => OutType::Emit,
        2 => OutType::Cut,
        3 => OutType::EmitThenCut,
        // Two bits: 0 is the only other value.
        _ => {
            return Ok(Op::OutFinal(OpOutFinal {
                dst,
                handle,
                stream,
            }));
        }
    };
    Ok(Op::Out(OpOut {
        dst,
        handle,
        stream,
        out_type,
    }))
}

/// `OUT` and `OUT.FINAL` in `form`.
const fn out(form: AluForm) -> Form {
    Form::new(
        form.opcode(OUT_OPCODE),
        mask_of(&[DST, OUT_TYPE]) | alu::known(form, OUT_SOURCES, false),
        decode_out,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// ISBERD and ISBEWR.

/// `ISBERD` (`impl SM70Op for OpIsberd`: `set_opcode(0x923)`).
pub(crate) const ISBERD_OPCODE: u16 = 0x923;
/// `ISBEWR` (`impl SM70Op for OpIsbewr`: `set_opcode(0x927)`).
pub(crate) const ISBEWR_OPCODE: u16 = 0x927;
/// The immediate offset of `ISBERD` and `ISBEWR` (`set_field(40..56, self.imm_offset)`; "only
/// supported on SM86+").
pub(crate) const ISBE_IMM: Field = Field::new(40, 16);
/// The memory type of `ISBERD` and `ISBEWR` (`set_field(74..76, ..)`: [`IsbeMemType`]).
pub(crate) const ISBE_MEM: Field = Field::new(74, 2);
/// The access type of `ISBERD` and `ISBEWR` (`set_field(76..78, ..)`: [`IsbeAccess`]).
pub(crate) const ISBE_ACCESS: Field = Field::new(76, 2);
/// The `.SKEW` bit of `ISBERD` and `ISBEWR` (`set_bit(78, self.skew)`).
pub(crate) const ISBE_SKEW: Field = Field::bit(78);

/// The size of an internal stage buffer access (NAK's `MemType`, of which `ISBERD` and
/// `ISBEWR` encode three: "Invalid ISBERD mem type" for the others).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsbeMemType {
    /// 0: an unsigned byte.
    U8,
    /// 1: an unsigned 16-bit word.
    U16,
    /// 2: a 32-bit word.
    B32,
}

impl IsbeMemType {
    /// Every type, in encoding order.
    pub const ALL: [Self; 3] = [Self::U8, Self::U16, Self::B32];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::U8 => 0,
            Self::U16 => 1,
            Self::B32 => 2,
        }
    }
}

/// What an internal stage buffer access addresses (NAK's `IsbeAccessType` and its encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsbeAccess {
    /// 0: the map.
    Map,
    /// 1: per-patch data (`ISBERD` only).
    Patch,
    /// 2: per-primitive data (`ISBERD` only).
    Primitive,
    /// 3: attributes.
    Attribute,
}

impl IsbeAccess {
    /// Every access type, in encoding order.
    pub const ALL: [Self; 4] = [Self::Map, Self::Patch, Self::Primitive, Self::Attribute];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::Map => 0,
            Self::Patch => 1,
            Self::Primitive => 2,
            Self::Attribute => 3,
        }
    }

    /// Whether `ISBEWR` can write it (NAK: "Invalid ISBEWR access type" for the others).
    pub const fn writable(self) -> bool {
        matches!(self, Self::Map | Self::Attribute)
    }
}

/// `ISBERD dst, [offset+imm]`: reads the internal stage buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpIsberd {
    /// The destination.
    pub dst: Gpr,
    /// The offset register; `RZ` for none.
    pub offset: Gpr,
    /// The immediate offset.
    pub imm_offset: u16,
    /// The size.
    pub mem_type: IsbeMemType,
    /// What it addresses.
    pub access: IsbeAccess,
    /// `.O`: read the outputs.
    pub output: bool,
    /// `.SKEW`.
    pub skew: bool,
}

/// `ISBEWR [offset+imm], data`: writes the internal stage buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpIsbewr {
    /// The offset register; `RZ` for none.
    pub offset: Gpr,
    /// The data.
    pub data: Gpr,
    /// The immediate offset.
    pub imm_offset: u16,
    /// The size.
    pub mem_type: IsbeMemType,
    /// What it addresses: [`IsbeAccess::Map`] or [`IsbeAccess::Attribute`].
    pub access: IsbeAccess,
    /// `.O` (NAK's `test_isbewr` always sets it).
    pub output: bool,
    /// `.SKEW`.
    pub skew: bool,
}

impl Named for OpIsberd {
    const MNEMONIC: &'static str = "isberd";
}

impl Named for OpIsbewr {
    const MNEMONIC: &'static str = "isbewr";
}

impl Display for IsbeMemType {
    /// NAK's `test_isberd`: nothing for `U8`, `.u16`, `.32`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::U8 => "",
            Self::U16 => ".u16",
            Self::B32 => ".32",
        })
    }
}

impl Display for IsbeAccess {
    /// NAK's `test_isberd`: nothing for the map, `.patch`, `.prim`, `.attr`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Map => "",
            Self::Patch => ".patch",
            Self::Primitive => ".prim",
            Self::Attribute => ".attr",
        })
    }
}

/// `[offset]` or `[offset+0x42]` (NAK's `test_isberd`).
struct IsbeAddr {
    offset: Gpr,
    imm_offset: u16,
}

impl Display for IsbeAddr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.imm_offset == 0 {
            write!(f, "[{}]", self.offset)
        } else {
            write!(f, "[{}+{:#x}]", self.offset, self.imm_offset)
        }
    }
}

/// The modifiers `ISBERD` and `ISBEWR` print, in `test_isberd`'s order.
fn isbe_modifiers(
    f: &mut Formatter<'_>,
    output: bool,
    access: IsbeAccess,
    skew: bool,
    mem_type: IsbeMemType,
) -> fmt::Result {
    flag(f, output, ".o")?;
    write!(f, "{access}")?;
    flag(f, skew, ".skew")?;
    write!(f, "{mem_type}")
}

impl Display for OpIsberd {
    /// NAK's `test_isberd`: `isberd{.o}{access}{.skew}{type} r1, [r2+0x42];`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("isberd")?;
        isbe_modifiers(f, self.output, self.access, self.skew, self.mem_type)?;
        let addr = IsbeAddr {
            offset: self.offset,
            imm_offset: self.imm_offset,
        };
        write!(f, " {}, {addr}", self.dst)
    }
}

impl Display for OpIsbewr {
    /// NAK's `test_isbewr`: `isbewr.o{access}{.skew}{type} [r2+0x42], r1;`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("isbewr")?;
        isbe_modifiers(f, self.output, self.access, self.skew, self.mem_type)?;
        let addr = IsbeAddr {
            offset: self.offset,
            imm_offset: self.imm_offset,
        };
        write!(f, " {addr}, {}", self.data)
    }
}

fn isbe_mem(reader: &mut Reader) -> Result<IsbeMemType, DecodeError> {
    let bits = reader.field(ISBE_MEM);
    match bits {
        0 => Ok(IsbeMemType::U8),
        1 => Ok(IsbeMemType::U16),
        2 => Ok(IsbeMemType::B32),
        _ => Err(reader.unknown_value(ISBE_MEM, bits)),
    }
}

fn isbe_imm(reader: &mut Reader) -> u16 {
    // Cannot truncate: the field is sixteen bits wide.
    u16::try_from(reader.field(ISBE_IMM)).unwrap_or(u16::MAX)
}

fn decode_isberd(reader: &mut Reader) -> Result<Op, DecodeError> {
    let mem_type = isbe_mem(reader)?;
    let access = match reader.field(ISBE_ACCESS) {
        0 => IsbeAccess::Map,
        1 => IsbeAccess::Patch,
        2 => IsbeAccess::Primitive,
        // Two bits: 3 is the only other value.
        _ => IsbeAccess::Attribute,
    };
    Ok(Op::Isberd(OpIsberd {
        dst: reader.gpr(DST),
        offset: reader.gpr(ATTR_OFFSET),
        imm_offset: isbe_imm(reader),
        mem_type,
        access,
        output: reader.flag(ATTR_OUTPUT),
        skew: reader.flag(ISBE_SKEW),
    }))
}

fn decode_isbewr(reader: &mut Reader) -> Result<Op, DecodeError> {
    let mem_type = isbe_mem(reader)?;
    let access_bits = reader.field(ISBE_ACCESS);
    let access = match access_bits {
        0 => IsbeAccess::Map,
        3 => IsbeAccess::Attribute,
        _ => return Err(reader.unknown_value(ISBE_ACCESS, access_bits)),
    };
    Ok(Op::Isbewr(OpIsbewr {
        offset: reader.gpr(ATTR_OFFSET),
        data: reader.gpr(ATTR_DATA),
        imm_offset: isbe_imm(reader),
        mem_type,
        access,
        output: reader.flag(ATTR_OUTPUT),
        skew: reader.flag(ISBE_SKEW),
    }))
}

// ---------------------------------------------------------------------------------------------
// The form table.

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
            ATTR_OUTPUT,
        ]),
        |reader| Ok(decode_ald(reader)),
    ),
    Form::new(
        AST_OPCODE,
        mask_of(&[
            ATTR_OFFSET,
            ATTR_DATA,
            AST_VTX,
            ATTR_ADDR,
            ATTR_COMPS,
            ATTR_PATCH,
            ATTR_PHYS,
        ]),
        |reader| Ok(decode_ast(reader)),
    ),
    Form::new(S2R_OPCODE, mask_of(&[DST, SR_INDEX]), |reader| {
        Ok(decode_s2r(reader))
    }),
    Form::new(S2UR_OPCODE, mask_of(&[UNIFORM_DST, SR_INDEX]), |reader| {
        Ok(decode_s2r(reader))
    }),
    Form::new(CS2R_OPCODE, mask_of(&[DST, SR_INDEX, CS2R_64]), |reader| {
        Ok(decode_cs2r(reader))
    }),
    Form::new(
        AL2P_OPCODE,
        mask_of(&[DST, ATTR_OFFSET, ATTR_ADDR, ATTR_COMPS, ATTR_OUTPUT]),
        |reader| Ok(decode_al2p(reader)),
    ),
    Form::new(
        IPA_OPCODE,
        mask_of(&[DST, IPA_OFFSET, ATTR_WORD, IPA_LOC, IPA_FREQ, PRED_DST]),
        decode_ipa,
    ),
    Form::new(
        LDTRAM_OPCODE,
        mask_of(&[DST, LDTRAM_UREG, ATTR_WORD, LDTRAM_C]),
        |reader| Ok(decode_ldtram(reader)),
    )
    .with_fixed(LDTRAM_FIXED.mask()),
    Form::new(
        PIXLD_OPCODE,
        mask_of(&[DST, PIXLD_VAL, PRED_DST]),
        decode_pixld,
    ),
    out(SOURCE_1_FORMS[0]),
    out(SOURCE_1_FORMS[1]),
    out(SOURCE_1_FORMS[2]),
    out(SOURCE_1_FORMS[3]),
    Form::new(
        ISBERD_OPCODE,
        mask_of(&[
            DST,
            ATTR_OFFSET,
            ISBE_IMM,
            ISBE_MEM,
            ISBE_ACCESS,
            ISBE_SKEW,
            ATTR_OUTPUT,
        ]),
        decode_isberd,
    ),
    Form::new(
        ISBEWR_OPCODE,
        mask_of(&[
            ATTR_OFFSET,
            ATTR_DATA,
            ISBE_IMM,
            ISBE_MEM,
            ISBE_ACCESS,
            ISBE_SKEW,
            ATTR_OUTPUT,
        ]),
        decode_isbewr,
    ),
];

/// The bits PR #4588's rows of `OUT`'s register, constant-buffer and uniform-register forms
/// mark known that the forms reject: both sources' modifiers.
#[cfg(test)]
pub(crate) const OUT_DROPPED: u128 = mask_of(&[SRC0_ABS, SRC0_NEG, SLOT_ABS, SLOT_NEG]);
/// The bits PR #4588's row of `OUT`'s immediate form marks known that the form rejects: source
/// 0's modifiers (bits 62 and 63 are the immediate's).
#[cfg(test)]
pub(crate) const OUT_IMM_DROPPED: u128 = mask_of(&[SRC0_ABS, SRC0_NEG]);

/// PR #4588's rows for the group's forms, in [`FORMS`] order
/// (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97; each
/// row's four words, low word first), with the bits each form drops or requires (module docs).
#[cfg(test)]
pub(crate) const REFERENCE_ROWS: &[ReferenceRow] = &[
    ReferenceRow::new(0x321, [0xffff_ffff, 0x0003_ffff, 0x0000_bc00, 0xffff_fe00]),
    ReferenceRow::new(0x322, [0xff00_ffff, 0x0003_ffff, 0x0000_3cff, 0xffff_fe00]),
    ReferenceRow::new(0x919, [0x00ff_ffff, 0x0000_0000, 0x0000_ff00, 0xffff_fe00]),
    ReferenceRow::new(0x9c3, [0x003f_ffff, 0x0000_0000, 0x0000_ff00, 0xffff_fe00]),
    ReferenceRow::new(0x805, [0x00ff_ffff, 0x0000_0000, 0x0001_ff00, 0xffff_fe00]),
    ReferenceRow::new(0x920, [0xffff_ffff, 0x0003_ff00, 0x0000_8c00, 0xffff_fe00]),
    ReferenceRow::new(0x326, [0x00ff_ffff, 0x0000_00ff, 0x000e_f0ff, 0xffff_fe00]),
    ReferenceRow::new(0x3ad, [0x3fff_ffff, 0x0000_0000, 0x0800_01ff, 0xffff_fe00])
        .fixing(LDTRAM_FIXED.mask()),
    ReferenceRow::new(0x925, [0x00ff_ffff, 0x0000_0000, 0x000f_c000, 0xffff_fe00]),
    ReferenceRow::new(0x324, [0xffff_ffff, 0xc000_00ff, 0x0000_c300, 0xffff_fe00])
        .dropping(OUT_DROPPED),
    ReferenceRow::new(0x924, [0xffff_ffff, 0xffff_ffff, 0x0000_c300, 0xffff_fe00])
        .dropping(OUT_IMM_DROPPED),
    ReferenceRow::new(0xb24, [0xffff_ffff, 0xc7ff_ffff, 0x0800_c300, 0xffff_fe00])
        .dropping(OUT_DROPPED),
    ReferenceRow::new(0xd24, [0xffff_ffff, 0xc000_003f, 0x0800_c300, 0xffff_fe00])
        .dropping(OUT_DROPPED)
        .fixing(alu::fixed(AluForm::UniformSrc1)),
    ReferenceRow::new(0x923, [0xffff_ffff, 0x00ff_ff00, 0x0000_fc00, 0xffff_fe00]),
    ReferenceRow::new(0x927, [0xff00_ffff, 0x00ff_ffff, 0x0000_fc00, 0xffff_fe00]),
];

/// Generators of the group's decoded operations, one per operation (`generate`).
#[cfg(test)]
pub(crate) fn any_ops() -> Vec<BoxedStrategy<Op>> {
    generate::all()
}

shells! {}

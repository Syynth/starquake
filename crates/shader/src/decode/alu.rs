//! The operand forms of ALU instructions, which the decoder and the test assembler share.
//!
//! An ALU instruction has up to three sources. Source 0 is always a register; sources 1 and 2
//! share two places, the wide slot (bits 32..64) and a register slot (bits 64..72), and the
//! operand form in bits 9..12 says which source the wide slot holds and what it holds. This is
//! the inverse of NAK's `encode_alu_base` (`mesa-nouveau/src/nouveau/compiler/nak/sm70_encode.rs`
//! @ 52238762, MIT, Copyright 2022 Collabora, Ltd.), and matches PR #4588's reader
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/visitor.cpp` @ e5fb3b97,
//! `Visitor::Source` and `Visitor::WideSlot`):
//!
//! | Form | Source 1 | Source 2 |
//! |---|---|---|
//! | 1 | wide slot register | register slot |
//! | 2 | register slot | wide slot immediate |
//! | 3 | register slot | wide slot constant buffer |
//! | 4 | wide slot immediate | register slot |
//! | 5 | wide slot constant buffer | register slot |
//! | 6 | wide slot uniform register | register slot |
//! | 7 | register slot | wide slot uniform register |
//!
//! Form 0 is not an ALU form. The uniform-register forms always have bit 91 set (NAK's
//! `encode_alu_ureg`: `set_bit(91, true)`; PR #4588 requires it: "native uniform source flag"),
//! which [`fixed`] makes a fixed bit of their forms. In the constant-buffer forms bit 91 picks a
//! bindless buffer, whose handle is in a uniform register (`set_src_cb`). These are the forms of
//! the vector datapath; the uniform datapath's (NAK's `encode_ualu_base`) are slice 6d's.
//!
//! A packed-half source (NAK's `SM70SrcType::F16v2`, slice 6c) sits in the same places with a
//! two-bit swizzle beside its modifier bits, and in the register slot its modifier bits move
//! (`set_alu_reg`, `encode_alu_src2`): source 0's swizzle at 74..76, the wide slot's at 60..62
//! (not for an immediate), and the register slot's at 81..83 with its absolute value at 83 and
//! negation at 84 instead of 74 and 75 ([`half_known`], [`half_src0`], [`half_src`]).

use super::DecodeError;
use super::fields::{
    ALU_FORM, CBUF_INDEX, CBUF_OFFSET, SLOT_ABS, SLOT_IMM, SLOT_NEG, SLOT_REG, SLOT_UNIFORM,
    SLOT_UNIFORM_REG, SRC0, SRC0_ABS, SRC0_NEG, SRC2, SRC2_ABS, SRC2_NEG, mask_of,
};
use super::fields::{SLOT_SWIZZLE, SRC0_SWIZZLE, SRC2_HALF_ABS, SRC2_HALF_NEG, SRC2_HALF_SWIZZLE};
use super::form::Reader;
use super::instruction::Field;
use super::operand::{AluSrc, CBuf, CBufRef, HalfSrc, HalfSwizzle};

/// An ALU operand form (`encode_alu_base`'s `form` values).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum AluForm {
    /// 1: sources 1 and 2 are registers.
    Reg,
    /// 2: source 2 is an immediate.
    ImmSrc2,
    /// 3: source 2 is a constant buffer.
    CBufSrc2,
    /// 4: source 1 is an immediate.
    ImmSrc1,
    /// 5: source 1 is a constant buffer.
    CBufSrc1,
    /// 6: source 1 is a uniform register.
    UniformSrc1,
    /// 7: source 2 is a uniform register.
    UniformSrc2,
}

/// Source 1 or source 2 of an ALU instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    /// Source 1.
    One,
    /// Source 2.
    Two,
}

/// Where a source is, and what it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    /// A register in the wide slot.
    WideReg,
    /// An immediate in the wide slot.
    Imm,
    /// A constant buffer in the wide slot.
    CBuf,
    /// A uniform register in the wide slot.
    UniformReg,
    /// A register in the register slot.
    Reg,
}

impl AluForm {
    /// Every form, in encoding order.
    pub(crate) const ALL: [Self; 7] = [
        Self::Reg,
        Self::ImmSrc2,
        Self::CBufSrc2,
        Self::ImmSrc1,
        Self::CBufSrc1,
        Self::UniformSrc1,
        Self::UniformSrc2,
    ];

    /// The three-bit encoding.
    pub(crate) const fn bits(self) -> u16 {
        match self {
            Self::Reg => 1,
            Self::ImmSrc2 => 2,
            Self::CBufSrc2 => 3,
            Self::ImmSrc1 => 4,
            Self::CBufSrc1 => 5,
            Self::UniformSrc1 => 6,
            Self::UniformSrc2 => 7,
        }
    }

    /// The form a three-bit encoding names; 0 names none.
    pub(crate) const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            1 => Some(Self::Reg),
            2 => Some(Self::ImmSrc2),
            3 => Some(Self::CBufSrc2),
            4 => Some(Self::ImmSrc1),
            5 => Some(Self::CBufSrc1),
            6 => Some(Self::UniformSrc1),
            7 => Some(Self::UniformSrc2),
            _ => None,
        }
    }

    /// The 12-bit opcode of the instruction with 9-bit opcode `base` in this form
    /// (`encode_alu_base`: `set_field(0..9, opcode)`, `set_field(9..12, form)`).
    pub(crate) const fn opcode(self, base: u16) -> u16 {
        base | (self.bits() << ALU_FORM.start)
    }

    /// Where this form puts `source`.
    pub(crate) const fn slot(self, source: Source) -> Slot {
        match (self, source) {
            (Self::Reg, Source::One) => Slot::WideReg,
            (Self::ImmSrc1, Source::One) | (Self::ImmSrc2, Source::Two) => Slot::Imm,
            (Self::CBufSrc1, Source::One) | (Self::CBufSrc2, Source::Two) => Slot::CBuf,
            (Self::UniformSrc1, Source::One) | (Self::UniformSrc2, Source::Two) => Slot::UniformReg,
            _ => Slot::Reg,
        }
    }
}

/// The bits of `slot`, with its modifier bits if `modifiers`.
const fn slot_known(slot: Slot, modifiers: bool) -> u128 {
    let wide_modifiers = if modifiers {
        mask_of(&[SLOT_ABS, SLOT_NEG])
    } else {
        0
    };
    match slot {
        Slot::WideReg => SLOT_REG.mask() | wide_modifiers,
        Slot::Imm => SLOT_IMM.mask(),
        Slot::CBuf => {
            mask_of(&[SLOT_UNIFORM_REG, CBUF_OFFSET, CBUF_INDEX, SLOT_UNIFORM]) | wide_modifiers
        }
        Slot::UniformReg => mask_of(&[SLOT_UNIFORM_REG, SLOT_UNIFORM]) | wide_modifiers,
        Slot::Reg => {
            let modifiers = if modifiers {
                mask_of(&[SRC2_ABS, SRC2_NEG])
            } else {
                0
            };
            SRC2.mask() | modifiers
        }
    }
}

/// Which sources an ALU instruction has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sources {
    /// Source 0.
    pub(crate) src0: bool,
    /// Source 1.
    pub(crate) src1: bool,
    /// Source 2.
    pub(crate) src2: bool,
}

/// The bits the sources of an instruction in `form` occupy, with their modifier bits if
/// `modifiers`.
pub(crate) const fn known(form: AluForm, sources: Sources, modifiers: bool) -> u128 {
    let mut mask = 0;
    if sources.src0 {
        mask |= SRC0.mask();
        if modifiers {
            mask |= mask_of(&[SRC0_ABS, SRC0_NEG]);
        }
    }
    if sources.src1 {
        mask |= slot_known(form.slot(Source::One), modifiers);
    }
    if sources.src2 {
        mask |= slot_known(form.slot(Source::Two), modifiers);
    }
    mask
}

/// The bits every word of `form` has set: bit 91 for the uniform-register forms.
pub(crate) const fn fixed(form: AluForm) -> u128 {
    match form {
        AluForm::UniformSrc1 | AluForm::UniformSrc2 => SLOT_UNIFORM.mask(),
        _ => 0,
    }
}

/// The form of the word `reader` reads. Form 0 is not an ALU form; the form table never sends
/// an ALU decoder a word in form 0, so the error names the opcode as unknown.
pub(crate) fn form(reader: &Reader) -> Result<AluForm, DecodeError> {
    let opcode = reader.opcode();
    AluForm::from_bits(opcode >> ALU_FORM.start).ok_or(DecodeError::UnknownOpcode {
        // Cannot fail: the opcode field is twelve bits wide.
        opcode: u16::try_from(opcode).unwrap_or(u16::MAX),
    })
}

/// Reads a source's modifier bits, if the instruction has them.
fn modifiers(reader: &mut Reader, modifiers: bool, abs: Field, neg: Field) -> (bool, bool) {
    if modifiers {
        (reader.flag(abs), reader.flag(neg))
    } else {
        (false, false)
    }
}

/// Reads source 0, a register (`encode_alu_src0`).
pub(crate) fn src0(reader: &mut Reader, with_modifiers: bool) -> AluSrc {
    let reg = reader.gpr(SRC0);
    let (abs, neg) = modifiers(reader, with_modifiers, SRC0_ABS, SRC0_NEG);
    AluSrc::Reg { reg, abs, neg }
}

/// Reads `source` of an instruction in `form`.
pub(crate) fn src(
    reader: &mut Reader,
    form: AluForm,
    source: Source,
    with_modifiers: bool,
) -> AluSrc {
    match form.slot(source) {
        Slot::WideReg => {
            let reg = reader.gpr(SLOT_REG);
            let (abs, neg) = modifiers(reader, with_modifiers, SLOT_ABS, SLOT_NEG);
            AluSrc::Reg { reg, abs, neg }
        }
        Slot::Imm => AluSrc::Imm(reader.word32(SLOT_IMM)),
        Slot::CBuf => {
            let buf = if reader.flag(SLOT_UNIFORM) {
                CBuf::BindlessUniform(reader.uniform_reg(SLOT_UNIFORM_REG))
            } else {
                CBuf::Binding(reader.byte(CBUF_INDEX))
            };
            // Cannot truncate: the field is sixteen bits wide.
            let offset = u16::try_from(reader.field(CBUF_OFFSET)).unwrap_or(u16::MAX);
            let (abs, neg) = modifiers(reader, with_modifiers, SLOT_ABS, SLOT_NEG);
            AluSrc::CBuf {
                cbuf: CBufRef { buf, offset },
                abs,
                neg,
            }
        }
        Slot::UniformReg => {
            // A fixed bit of the form: the decoder has checked that it is set.
            reader.flag(SLOT_UNIFORM);
            let reg = reader.uniform_reg(SLOT_UNIFORM_REG);
            let (abs, neg) = modifiers(reader, with_modifiers, SLOT_ABS, SLOT_NEG);
            AluSrc::UniformReg { reg, abs, neg }
        }
        Slot::Reg => {
            let reg = reader.gpr(SRC2);
            let (abs, neg) = modifiers(reader, with_modifiers, SRC2_ABS, SRC2_NEG);
            AluSrc::Reg { reg, abs, neg }
        }
    }
}

/// The forms of an instruction that keeps its second operand in the wide slot: as source 1 when
/// it is a register (form 1), as source 2 otherwise (forms 2, 3 and 7). NAK picks between
/// `encode_alu(.., a, b, None)` and `encode_alu(.., a, None, b)` by `src_is_zero_or_gpr(b)` in
/// `OpFAdd`, `OpHAdd2`, `OpHSet2`, `OpHSetP2` and `OpDSetP` (`sm70_encode.rs` @ 52238762).
pub(crate) const WIDE_OPERAND_FORMS: [AluForm; 4] = [
    AluForm::Reg,
    AluForm::ImmSrc2,
    AluForm::CBufSrc2,
    AluForm::UniformSrc2,
];

/// Where an instruction of [`WIDE_OPERAND_FORMS`] keeps its second operand in `form`: source 1
/// in the forms that put source 1 in the wide slot, source 2 in the others.
pub(crate) const fn wide_operand_source(form: AluForm) -> Source {
    match form {
        AluForm::Reg | AluForm::ImmSrc1 | AluForm::CBufSrc1 | AluForm::UniformSrc1 => Source::One,
        AluForm::ImmSrc2 | AluForm::CBufSrc2 | AluForm::UniformSrc2 => Source::Two,
    }
}

/// The modifier and swizzle fields of a packed-half source in one place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HalfFields {
    /// The absolute-value bit.
    pub(crate) abs: Field,
    /// The negation bit.
    pub(crate) neg: Field,
    /// The swizzle.
    pub(crate) swizzle: Field,
}

/// Source 0's packed-half fields (`encode_alu_src0`: `set_alu_reg(24..32, 73, 72, 74..76, ..)`).
pub(crate) const HALF_SRC0: HalfFields = HalfFields {
    abs: SRC0_ABS,
    neg: SRC0_NEG,
    swizzle: SRC0_SWIZZLE,
};

/// The packed-half fields of a source in `slot`: the wide slot's at 62, 63 and 60..62
/// (`encode_alu_reg`, `encode_alu_cb`, `encode_alu_ureg`), the register slot's at 83, 84 and
/// 81..83 (`encode_alu_src2`), none for an immediate (`encode_alu_imm` writes only the value).
pub(crate) const fn half_fields(slot: Slot) -> Option<HalfFields> {
    match slot {
        Slot::Imm => None,
        Slot::Reg => Some(HalfFields {
            abs: SRC2_HALF_ABS,
            neg: SRC2_HALF_NEG,
            swizzle: SRC2_HALF_SWIZZLE,
        }),
        Slot::WideReg | Slot::CBuf | Slot::UniformReg => Some(HalfFields {
            abs: SLOT_ABS,
            neg: SLOT_NEG,
            swizzle: SLOT_SWIZZLE,
        }),
    }
}

/// The bits of `fields`.
const fn half_fields_known(fields: HalfFields) -> u128 {
    mask_of(&[fields.abs, fields.neg, fields.swizzle])
}

/// The bits of a packed-half source in `slot`: the slot without its 32-bit modifier bits, and
/// the packed-half fields.
const fn half_slot_known(slot: Slot) -> u128 {
    let fields = match half_fields(slot) {
        Some(fields) => half_fields_known(fields),
        None => 0,
    };
    slot_known(slot, false) | fields
}

/// The bits the packed-half sources of an instruction in `form` occupy, with their modifier and
/// swizzle fields.
pub(crate) const fn half_known(form: AluForm, sources: Sources) -> u128 {
    let mut mask = 0;
    if sources.src0 {
        mask |= SRC0.mask() | half_fields_known(HALF_SRC0);
    }
    if sources.src1 {
        mask |= half_slot_known(form.slot(Source::One));
    }
    if sources.src2 {
        mask |= half_slot_known(form.slot(Source::Two));
    }
    mask
}

/// `src` with the modifier bits `abs` and `neg`; an immediate has none and is returned as it is.
pub(crate) const fn with_modifiers(src: AluSrc, abs: bool, neg: bool) -> AluSrc {
    match src {
        AluSrc::Reg { reg, .. } => AluSrc::Reg { reg, abs, neg },
        AluSrc::UniformReg { reg, .. } => AluSrc::UniformReg { reg, abs, neg },
        AluSrc::CBuf { cbuf, .. } => AluSrc::CBuf { cbuf, abs, neg },
        AluSrc::Imm(imm) => AluSrc::Imm(imm),
    }
}

/// Reads the modifier bits and swizzle in `fields` onto `src`, refusing swizzle 1
/// ([`HalfSwizzle`]); without fields (an immediate) the source reads both halves as stored.
fn half(
    reader: &mut Reader,
    src: AluSrc,
    fields: Option<HalfFields>,
) -> Result<HalfSrc, DecodeError> {
    let Some(fields) = fields else {
        return Ok(HalfSrc::new(src));
    };
    let abs = reader.flag(fields.abs);
    let neg = reader.flag(fields.neg);
    let bits = reader.field(fields.swizzle);
    let swizzle =
        HalfSwizzle::from_bits(bits).ok_or_else(|| reader.unknown_value(fields.swizzle, bits))?;
    Ok(HalfSrc {
        src: with_modifiers(src, abs, neg),
        swizzle,
    })
}

/// Reads source 0 of a packed-half instruction, a register.
pub(crate) fn half_src0(reader: &mut Reader) -> Result<HalfSrc, DecodeError> {
    let reg = reader.gpr(SRC0);
    half(reader, AluSrc::reg(reg), Some(HALF_SRC0))
}

/// Reads `source` of a packed-half instruction in `form`.
pub(crate) fn half_src(
    reader: &mut Reader,
    form: AluForm,
    source: Source,
) -> Result<HalfSrc, DecodeError> {
    let slot = form.slot(source);
    let plain = src(reader, form, source, false);
    half(reader, plain, half_fields(slot))
}

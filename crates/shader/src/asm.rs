//! The test assembler: a decoded [`Instr`] back to its instruction word.
//!
//! Ported from `mesa-nouveau/src/nouveau/compiler/nak/sm70_encode.rs` @ 52238762 (MIT,
//! Copyright 2022 Collabora, Ltd.): the `SM70Encoder` field helpers (`set_reg`, `set_ureg`,
//! `set_pred_reg`, `set_pred_src`, `set_src_cb`, `set_pred`) as [`Writer`]'s methods,
//! `encode_alu_base` (operand forms) as `alu`, and each decoded operation's `impl SM70Op` as an
//! `Encode` impl in this module's group modules, which mirror the decoder's (`decode::op`). It
//! is the fixture generator for the crate's tests (spec, "Test assembler"): words copied from
//! real shaders cannot be committed. The scheduling bits (105..128, NAK's `set_instr_deps`) are
//! left zero, which the decoder ignores.
//!
//! An operation the decoder does not decode yet (a shell) has no encoding:
//! [`EncodeError::NoEncoding`]. Exported behind the `test-support` feature for downstream crates'
//! tests.

mod attribute;
mod control;
mod float;
mod half_double;
mod integer;
mod memory;
mod tensor;
mod texture;
mod uniform;

use thiserror::Error;

use crate::decode::{
    AluForm, AluSrc, CBUF_INDEX, CBUF_OFFSET, CBuf, CBufRef, Field, GUARD, GUARD_NOT, Gpr, Instr,
    Instruction, Named, OPCODE, Op, PredSrc, SLOT_ABS, SLOT_IMM, SLOT_NEG, SLOT_REG, SLOT_UNIFORM,
    SLOT_UNIFORM_REG, SRC0, SRC0_ABS, SRC0_NEG, SRC2, SRC2_ABS, SRC2_NEG, Slot, Source,
    UniformPredSrc, UniformReg, op_list,
};

/// Why an [`Instr`] has no encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum EncodeError {
    /// The operation is a shell: the decoder does not decode it yet.
    #[error("the test assembler cannot encode {mnemonic} yet")]
    NoEncoding {
        /// The operation's mnemonic.
        mnemonic: &'static str,
    },
    /// An ALU instruction's source 0 must be a per-thread register (`encode_alu_src0` takes
    /// only registers).
    #[error("the first source of {mnemonic} must be a register")]
    FirstSourceNotRegister {
        /// The operation's mnemonic.
        mnemonic: &'static str,
    },
    /// An ALU instruction has one wide slot, so at most one source that is an immediate, a
    /// constant buffer or a uniform register.
    #[error("{mnemonic} can take only one immediate, constant-buffer or uniform-register source")]
    TwoWideSources {
        /// The operation's mnemonic.
        mnemonic: &'static str,
    },
    /// A source has a modifier its instruction has no bits for.
    #[error("source {index} of {mnemonic} has a modifier the instruction cannot encode")]
    Modifier {
        /// The operation's mnemonic.
        mnemonic: &'static str,
        /// Which source, 0 to 2.
        index: u8,
    },
    /// A field value does not fit its bits.
    #[error("{what} {value} does not fit in {bits} bits")]
    TooWide {
        /// Which operand.
        what: &'static str,
        /// The value given.
        value: u64,
        /// The field's width.
        bits: u32,
    },
    /// A signed field value is outside the range of its bits.
    #[error("{what} {value} does not fit in {bits} signed bits")]
    OutOfRange {
        /// Which operand.
        what: &'static str,
        /// The value given.
        value: i64,
        /// The field's width.
        bits: u32,
    },
    /// `ALD`, `AST` and `AL2P` move one to four words.
    #[error("an attribute access moves 1 to 4 words, not {0}")]
    Comps(u8),
    /// The instruction has no encoding in the operand form its sources need (`IMAD.HI` has
    /// forms 1, 4 and 5 only, and `IDP.2A` form 1 only: `decode::op::integer`).
    #[error("{mnemonic} has no encoding in operand form {form}")]
    NoForm {
        /// The operation's mnemonic.
        mnemonic: &'static str,
        /// The form the sources need, 1 to 7.
        form: u16,
    },
    /// The operand form or opcode the operation's other fields pick has no bits for this
    /// operand or modifier (`decode::op::float`: `FMNMX`'s `.nan` and `FSET`'s combine
    /// operation with a uniform-register source, among others).
    #[error("{mnemonic} has no bits for {what}")]
    NoBits {
        /// The operation's mnemonic.
        mnemonic: &'static str,
        /// What cannot be encoded.
        what: &'static str,
    },
    /// `IPA` and `LDTRAM` address attributes in whole words (NAK: `assert!(self.addr % 4 ==
    /// 0)`).
    #[error("attribute address {addr:#x} is not a multiple of {align}")]
    AttrAlign {
        /// The byte address given.
        addr: u16,
        /// The alignment the field needs.
        align: u16,
    },
    /// `ISBEWR` writes only the map (0) and attributes (3) (NAK: "Invalid ISBEWR access type").
    #[error("isbewr cannot write access type {0}")]
    IsbewrAccess(u64),
    /// A value the field holds in larger units is not a whole number of them (a texture
    /// handle's constant-buffer offset, held in words: NAK's `assert!(cb.offset % 4 == 0)`).
    #[error("{what} {value:#x} is not a multiple of {align}")]
    Misaligned {
        /// Which operand.
        what: &'static str,
        /// The value given.
        value: u64,
        /// The unit the field holds.
        align: u64,
    },
}

/// Builds one instruction word field by field.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Writer {
    word: Instruction,
}

impl Writer {
    /// The word so far.
    pub(crate) const fn word(self) -> Instruction {
        self.word
    }

    /// Sets `field` to the low bits of `value`. For values that cannot be too wide by their
    /// type (a register number in its field, a flag); others go through [`Writer::checked`].
    pub(crate) fn set(&mut self, field: Field, value: u64) {
        self.word = self.word.with_field(field, value);
    }

    /// Sets a one-bit field.
    pub(crate) fn flag(&mut self, field: Field, value: bool) {
        self.word = self.word.with_flag(field, value);
    }

    /// Sets the 12-bit opcode (`set_opcode`).
    pub(crate) fn opcode(&mut self, opcode: u16) {
        self.set(OPCODE, u64::from(opcode));
    }

    /// Sets `field` to `value`, which must fit.
    pub(crate) fn checked(
        &mut self,
        field: Field,
        what: &'static str,
        value: u64,
    ) -> Result<(), EncodeError> {
        let fits = value.checked_shr(field.width).is_none_or(|rest| rest == 0);
        if !fits {
            return Err(EncodeError::TooWide {
                what,
                value,
                bits: field.width,
            });
        }
        self.set(field, value);
        Ok(())
    }

    /// Sets `field` to `value` in two's complement, which must be in the field's signed range
    /// (`width` from 1 to 64; a zero-width field holds no value).
    pub(crate) fn signed(
        &mut self,
        field: Field,
        what: &'static str,
        value: i64,
    ) -> Result<(), EncodeError> {
        let unused = 64_u32.saturating_sub(field.width);
        // In range exactly when shifting the sign bit to bit 63 and back is lossless.
        let in_range = value
            .checked_shl(unused)
            .and_then(|shifted| shifted.checked_shr(unused))
            == Some(value);
        if !in_range {
            return Err(EncodeError::OutOfRange {
                what,
                value,
                bits: field.width,
            });
        }
        self.set(field, value.cast_unsigned());
        Ok(())
    }

    /// Sets a register field (`set_reg`).
    pub(crate) fn gpr(&mut self, field: Field, reg: Gpr) {
        self.set(field, u64::from(reg.index()));
    }

    /// Sets a uniform register field (`set_ureg`).
    pub(crate) fn uniform_reg(&mut self, field: Field, reg: UniformReg) {
        self.set(field, u64::from(reg.index()));
    }

    /// Sets a predicate field and its negation bit (`set_pred_src`).
    pub(crate) fn pred_src(&mut self, reg: Field, not: Field, src: PredSrc) {
        self.set(reg, u64::from(src.reg.index()));
        self.flag(not, src.negated);
    }

    /// Sets a uniform predicate field and its negation bit (`set_upred_src`).
    pub(crate) fn uniform_pred_src(&mut self, reg: Field, not: Field, src: UniformPredSrc) {
        self.set(reg, u64::from(src.reg.index()));
        self.flag(not, src.negated);
    }

    /// Sets the wide slot to a constant buffer (`set_src_cb(32..59, 91, ..)`).
    pub(crate) fn cbuf(&mut self, cbuf: CBufRef) -> Result<(), EncodeError> {
        match cbuf.buf {
            CBuf::Binding(index) => {
                self.checked(CBUF_INDEX, "constant buffer index", u64::from(index))?;
                self.flag(SLOT_UNIFORM, false);
            }
            CBuf::BindlessUniform(reg) => {
                self.uniform_reg(SLOT_UNIFORM_REG, reg);
                self.flag(SLOT_UNIFORM, true);
            }
        }
        self.set(CBUF_OFFSET, u64::from(cbuf.offset));
        Ok(())
    }

    /// Encodes the sources of an ALU instruction with 9-bit opcode `base` and sets the opcode
    /// with the form they need (`encode_alu_base`, whose `None` sources write nothing). Without
    /// `modifiers`, a source with a modifier bit set is an error.
    pub(crate) fn alu(
        &mut self,
        mnemonic: &'static str,
        base: u16,
        sources: [Option<AluSrc>; 3],
        modifiers: bool,
    ) -> Result<(), EncodeError> {
        let [src0, src1, src2] = sources;
        for (index, src) in (0..).zip(sources) {
            if !modifiers && src.is_some_and(|src| src.modifiers() != (false, false)) {
                return Err(EncodeError::Modifier { mnemonic, index });
            }
        }
        if let Some(src0) = src0 {
            let AluSrc::Reg { reg, abs, neg } = src0 else {
                return Err(EncodeError::FirstSourceNotRegister { mnemonic });
            };
            self.gpr(SRC0, reg);
            if modifiers {
                self.flag(SRC0_ABS, abs);
                self.flag(SRC0_NEG, neg);
            }
        }
        let form = choose_form(src1, src2).ok_or(EncodeError::TwoWideSources { mnemonic })?;
        if let Some(src) = src1 {
            self.slot(form.slot(Source::One), src, modifiers)?;
        }
        if let Some(src) = src2 {
            self.slot(form.slot(Source::Two), src, modifiers)?;
        }
        self.opcode(form.opcode(base));
        Ok(())
    }

    /// Writes `src` into `slot`; [`choose_form`] has matched the slot to the source's kind.
    fn slot(&mut self, slot: Slot, src: AluSrc, modifiers: bool) -> Result<(), EncodeError> {
        let (abs_field, neg_field) = match slot {
            Slot::Reg => (SRC2_ABS, SRC2_NEG),
            _ => (SLOT_ABS, SLOT_NEG),
        };
        match (slot, src) {
            (Slot::Imm, AluSrc::Imm(imm)) => self.set(SLOT_IMM, u64::from(imm)),
            (Slot::WideReg, AluSrc::Reg { reg, .. }) => self.gpr(SLOT_REG, reg),
            (Slot::Reg, AluSrc::Reg { reg, .. }) => self.gpr(SRC2, reg),
            (Slot::CBuf, AluSrc::CBuf { cbuf, .. }) => self.cbuf(cbuf)?,
            (Slot::UniformReg, AluSrc::UniformReg { reg, .. }) => {
                self.uniform_reg(SLOT_UNIFORM_REG, reg);
                self.flag(SLOT_UNIFORM, true);
            }
            // Unreachable: `choose_form` picks the slot from the kind.
            _ => {}
        }
        if modifiers {
            let (abs, neg) = src.modifiers();
            if !matches!(src, AluSrc::Imm(_)) {
                self.flag(abs_field, abs);
                self.flag(neg_field, neg);
            }
        }
        Ok(())
    }
}

/// The operand form for sources 1 and 2 (`encode_alu_base`'s choice), or `None` when both need
/// the wide slot.
pub(crate) fn choose_form(src1: Option<AluSrc>, src2: Option<AluSrc>) -> Option<AluForm> {
    let wide = |src: Option<AluSrc>| match src {
        None | Some(AluSrc::Reg { .. }) => None,
        Some(AluSrc::Imm(_)) => Some((AluForm::ImmSrc1, AluForm::ImmSrc2)),
        Some(AluSrc::CBuf { .. }) => Some((AluForm::CBufSrc1, AluForm::CBufSrc2)),
        Some(AluSrc::UniformReg { .. }) => Some((AluForm::UniformSrc1, AluForm::UniformSrc2)),
    };
    match (wide(src1), wide(src2)) {
        (None, None) => Some(AluForm::Reg),
        (Some((form, _)), None) | (None, Some((_, form))) => Some(form),
        (Some(_), Some(_)) => None,
    }
}

/// An operation's encoding: NAK's `impl SM70Op` for it. A shell keeps the default, which
/// refuses.
pub(crate) trait Encode: Named {
    /// Writes the operation's fields and opcode.
    fn encode(&self, _writer: &mut Writer) -> Result<(), EncodeError> {
        Err(EncodeError::NoEncoding {
            mnemonic: Self::MNEMONIC,
        })
    }
}

/// Gives each listed shell the refusing default [`Encode`] impl.
macro_rules! no_encoding {
    ($($name:ident),* $(,)?) => {
        $(impl $crate::asm::Encode for $name {})*
    };
}
use no_encoding;

/// Dispatches [`Op`] to each operation's [`Encode`] impl.
macro_rules! define_encode_op {
    ($($module:ident { $($variant:ident($name:ident),)* })*) => {
        fn encode_op(op: &Op, writer: &mut Writer) -> Result<(), EncodeError> {
            match op {
                $($(Op::$variant(op) => op.encode(writer),)*)*
            }
        }
    };
}
op_list!(define_encode_op);

/// Encodes `instr`.
///
/// ```
/// use starquake_shader::asm::assemble;
/// use starquake_shader::decode::{Instr, Op, OpExit, PredSrc, decode};
///
/// let exit = Instr::new(Op::Exit(OpExit {
///     keep_ref_count: false,
///     no_at_exit: false,
///     pred: PredSrc::TRUE,
/// }));
/// let word = assemble(&exit)?;
/// assert_eq!(decode(word)?.to_string(), "exit;");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn assemble(instr: &Instr) -> Result<Instruction, EncodeError> {
    let mut writer = Writer::default();
    writer.pred_src(GUARD, GUARD_NOT, instr.guard);
    encode_op(&instr.op, &mut writer)?;
    Ok(writer.word())
}

/// Assembles a program: each instruction's word, as the little-endian bytes guest memory
/// holds.
pub fn assemble_program(instrs: &[Instr]) -> Result<Vec<u8>, EncodeError> {
    let mut bytes = Vec::with_capacity(instrs.len().saturating_mul(Instruction::SIZE));
    for instr in instrs {
        bytes.extend_from_slice(&assemble(instr)?.to_le_bytes());
    }
    Ok(bytes)
}

//! Slice 6d, the uniform datapath: `R2UR`, `UMOV`, `ULDC` (with the indexed `0xabb` and the
//! global `LDCG`), `UPLOP3`, and the uniform forms of the integer ALU instructions (the
//! *twins*: `UIADD3`, `UIADD3.X`, `ULEA`, `ULEA.X`, `ULOP3`, `UPRMT`, `USHF`, `UIMAD`,
//! `UIMAD.WIDE`, `UISETP`, `USEL`, `USGXT`, `UBMSK`, `UFLO`, `UBREV`, `UPOPC`). `UP2UR` stays a
//! shell (below).
//!
//! # Sources
//!
//! NAK's encoders (`mesa-nouveau/src/nouveau/compiler/nak/sm70_encode.rs` @ 52238762, MIT,
//! Copyright 2022 Collabora, Ltd.): `encode_ualu_base` and each `impl SM70Op` with an
//! `is_uniform()` branch, `OpR2UR`, `OpMov` (`0xc82`), `OpLdc` (`0xab9`), `OpLdcg` (`0x8b8`)
//! and `OpPLop3` (`0x89c`). PR #4588's table
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97) has a
//! row for every form decoded here ([`REFERENCE_ROWS`]); its translator reads them in
//! `visitor.cpp` (`Visitor::Run`, `Source`, `Dst`, with `uniform_op` set by `classify.h`),
//! `ops_misc.cpp` (`UMOV`, `ULDC`), `ops_memory.cpp` (`R2UR`) and `ops_alu.cpp` (`PLOP3`).
//! The text is NAK's `nvdisasm_tests.rs` where it has a case (`test_ldc`'s uniform part and
//! `test_uldc_global`); everything else printed here is a guess.
//!
//! # The model: why the uniform forms are variants of their own
//!
//! NAK models a uniform instruction as the vector op with uniform operands (`OpMov` is `UMOV`,
//! `OpLdc` is `ULDC`, `OpIAdd3` is `UIADD3`), and the `op` module's docs leave this slice free to
//! revisit that. This slice gives each uniform instruction an [`Op`] variant of its own, for
//! two reasons. A vector operation's registers are [`Gpr`]s, and turning every destination and
//! source of slices 6a, 6b and 6e into a register-file enum would change the structs whose
//! translators sibling slices are writing (slice 12e translates `LDC`). And a translator that
//! matched a vector variant and ignored a new "uniform" field would write the per-thread file
//! where the program wrote the uniform one; a variant of its own is an operation each
//! translator must name or refuse.
//!
//! **The twins reuse the vector structs.** The uniform ALU forms are NAK's `encode_ualu_base`:
//! the vector op's fields at the vector op's positions, every register field six bits wide
//! (`set_ureg`, "Before SM100 UGPRs encode in 6 bits"), bit 91 set ("All uniform ALU requires
//! bit 91 set"), a 9-bit opcode of their own, and only forms 1, 2 and 4 ("UALU never has `UReg`",
//! "UALU does not support cbufs"). So a twin is a [`Uniform`] of the vector struct: the struct's
//! [`Gpr`] operands name the uniform registers of the same numbers ([`Gpr::RZ`] names `URZ`), its
//! predicate operands the uniform predicates (`PT` is `UPT`), as PR #4588's translator reads the
//! same fields with `uniform_op` set (`Visitor::Dst`, `Source`, `PredIn`, `PredOut`). The
//! decoder reads a twin with the vector op's own decode function (the [`Reader`] reads its
//! register fields six bits wide), and the test assembler writes one with the vector op's
//! encoder, then narrows the register fields. [`Uniform::new`] refuses a vector op the uniform
//! encoding cannot hold, so every [`Uniform`] value has a word.
//!
//! | Twin | 9-bit opcode | Vector op | Forms | NAK |
//! |---|---|---|---|---|
//! | `UIADD3`, `UIADD3.X` | `0x090` | [`OpIAdd3`], [`OpIAdd3X`] | 1, 2, 4 | `OpIAdd3`, `OpIAdd3X`: `encode_ualu(0x090, ..)` |
//! | `ULEA`, `ULEA.X` | `0x091` | [`OpLea`], [`OpLeaX`] | 1, 2, 4 | `OpLea`, `OpLeaX`: `encode_ualu(0x091, ..)` |
//! | `ULOP3` | `0x092` | [`OpLop3`] | 1, 2, 4 | `OpLop3`: `encode_ualu(0x092, ..)` |
//! | `UPRMT` | `0x096` | [`OpPrmt`] | 1, 2, 4 | `OpPrmt`: `encode_ualu(0x96, ..)` |
//! | `USHF` | `0x099` | [`OpShf`] | 1, 2, 4 | `OpShf`: `encode_ualu(0x099, ..)` |
//! | `UIMAD` | `0x0a4` | [`OpIMad`] (not `.HI`) | 1, 2, 4 | `OpIMad`: `encode_ualu(0x0a4, ..)` |
//! | `UIMAD.WIDE` | `0x0a5` | [`OpIMad64`] | 1, 2, 4 | `OpIMad64`: `encode_ualu(0x0a5, ..)` |
//! | `UISETP` | `0x08c` | [`OpISetP`] | 1, 4 | `OpISetP`: `encode_ualu(0x08c, ..)` |
//! | `USEL` | `0x087` | [`OpSel`] | 1, 4 | `OpSel`: `encode_ualu(0x087, ..)` |
//! | `USGXT` | `0x09a` | [`OpSgxt`] | 1, 4 | `OpSgxt`: `encode_ualu(0x09a, ..)` |
//! | `UBMSK` | `0x09b` | [`OpBMsk`] | 1, 4 | `OpBMsk`: `encode_ualu(0x09b, ..)` |
//! | `UFLO` | `0x0bd` | [`OpFlo`] | 1, 4 | `OpFlo`: `encode_ualu(0x0bd, ..)` |
//! | `UBREV` | `0x0be` | [`OpBRev`] | 1, 4 | `OpBRev`: `encode_ualu(0x0be, ..)` |
//! | `UPOPC` | `0x0bf` | [`OpPopC`] | 1, 4 | `OpPopC`: `encode_ualu(0x0bf, ..)` |
//!
//! An instruction without source 2 has forms 1 and 4 only, as its vector op has no forms 2, 3
//! and 7 (`op::integer`). `UIMAD.WIDE` keeps form 2 though NAK asserts its source 2 is not an
//! immediate, as the vector `IMAD.WIDE` does. Every field the vector op reads keeps its
//! meaning, its divergences from PR #4588 (`op::integer`'s module docs) included.
//!
//! # The other instructions
//!
//! - **`R2UR ur, r`** (`0x3c2`, [`OpR2UR`]): NAK `OpR2UR` (`set_udst`, `set_reg_src(24..32)`,
//!   `set_pred_dst(81..84, &Dst::None)`). Its predicate destination is a fixed field, all ones:
//!   NAK writes `PT` and PR #4588 requires it (`Visitor::R2UR`: `Require(Take(81, 3) == 7)`),
//!   the integer group's rule for such fields.
//! - **`UMOV ur, imm`** (`0x882`) and **`UMOV ur, ur`** (`0xc82`, bit 91 fixed), [`OpUMov`]: NAK
//!   `OpMov` with a uniform destination ("umov is encoded like a non-uniform ALU op": forms 4 and
//!   6), PR #4588's `Visitor::UMOV`.
//! - **`ULDC`** ([`OpULdc`]), three forms, all printed `uldc` as `nvdisasm` does:
//!   - `0xab9`, a constant-buffer word, bound (bits 54..59) or bindless (bit 91 set, the handle at
//!     bits 24..30): NAK `OpLdc` with a uniform destination, which writes no offset register
//!     below SM 10.0 (`assert!(self.offset.is_zero())`). The bound layout does not read bits
//!     24..30 and the bindless one does not read bits 54..59 (the decoder's step 5).
//!   - `0xabb`, a bound buffer whose offset adds a uniform register (bits 24..30), bit 91
//!     fixed: **PR #4588 only** (`Visitor::ULDC`: `indexed = op == 0xabb`, `Take(24, 6)`,
//!     `Require(Take(91, 1) == 1)`; its lowering reads it the same way). **Conflict** on the
//!     name: PR #4588's table calls the row `ULDC.64`, while its translator and lowering read
//!     the width from the type field (bits 73..76) in both opcodes, as here, so `.64` is a
//!     type, not this opcode. **Needs an `nvdisasm` check** (verification E2).
//!   - `0x8b8`, a load from global memory at a uniform address pair plus a 32-bit offset, under
//!     a uniform predicate that makes it return zero: NAK `OpLdcg` (`set_ureg_src(24, addr)`,
//!     `set_field(38..70, offset)`, `set_upred_src(87..90, 90, pred)`, `set_bit(91, true)`),
//!     printed as NAK's `test_uldc_global` expects. The spec's ruling of 2026-11-19 records that
//!     no slice owned `LDCG`; issue #354 names `test_uldc_global`'s cases, so this slice decodes it.
//! - **`UPLOP3.LUT`** (`0x89c`, [`OpUPLop3`]): NAK `OpPLop3` with uniform predicates
//!   (`set_upred_src` for `a`, `b`, `c`; the truth tables and destinations as `PLOP3`). Bit 67,
//!   `PLOP3`'s uniform `c`, is not in this form (PR #4588: `Require(!uniform_c ||
//!   !uniform_op)`, and the row leaves it out).
//! - **`UP2UR` stays a shell.** Its encoding has no NAK encoder and no row in PR #4588's table;
//!   only the lowerings read it (`eden-pr4588/.../frontend/ampere/lower_to_maxwell.h` @ e5fb3b97
//!   and `eden-nx2/.../frontend/ampere/lower_to_maxwell.h`, `case 0x883`: destination at bits
//!   16..24, a merge source that must be `URZ` at 24..32, the predicate mask in bits 32..64 and
//!   bit 91 set). The decoder framework checks every form against a table row
//!   (`decode::tests::masks_match_the_reference_table`), so a form with no row needs a new
//!   kind of reference in the framework's tests first (an owner question; issue #354's notes).
//!
//! # Left as unknown opcodes
//!
//! PR #4588's table has rows this slice leaves undecoded
//! (`decode::tests::uniform::uniform_rows_left_undecoded_are_unknown_opcodes`):
//!
//! - forms 3, 5, 6 and 7 of every twin's 9-bit opcode, and forms 2 of the twins without
//!   source 2 (NAK's `encode_ualu_base` panics on a constant buffer or uniform register, and has
//!   no form 2 without source 2);
//! - the float, `IABS` and `IMNMX` twins (`0x04d`, `0x050` to `0x05d`, `0x085`): NAK encodes them
//!   only from SM 12.0 (`assert!(e.sm >= 120)`), and neither PR #4588's translator and lowering
//!   nor Eden-NX2's lowering has a case for them. The spec's list for this slice reads row 88's
//!   "uniform USHF/FMNMX/I2F/F2I" as `UFMNMX`, `UI2F` and `UF2I`; Eden-NX2's notes are as
//!   well read as `FMNMX`, `I2F` and `F2I` with uniform-register sources, which slice 6b
//!   decodes. An owner question;
//! - `UF2FP` (`0x0ba`): NAK's SM 8.6 encoding writes type sizes at bits 75..77 and 84..86
//!   ("Uniform form always expect the type size on SM86") that [`OpF2FP`](super::OpF2FP) has no
//!   fields for.
//!
//! # Divergences from PR #4588's table
//!
//! | Rows | What this decoder does | Pinned by |
//! |---|---|---|
//! | The twins' rows | The vector op's divergences (`op::integer`): the absolute-value bits and the negation bits of sources without one rejected, `LOP3`'s bit 80 rejected, the fixed fields (`LOP3`'s `!UPT` input, `UIMAD`'s, `UIMAD.WIDE`'s and `UFLO`'s `UPT` destination) and bit 91 required | `decode::tests::uniform::twin_rows_drop_the_vector_ops_bits` |
//! | `UMOV` `0xc82` bits 38..64 | Rejected: NAK writes a six-bit register and no modifiers (`encode_alu_ureg`), PR #4588's translator reads `Take(32, 6)` only and its lowering requires the word below 64 | `decode::tests::uniform::umov_rejects_the_bits_above_its_register` |
//! | `ULDC` `0xab9` bits 24..30 with bit 91 clear | Rejected (step 5): NAK writes nothing there for a bound buffer; PR #4588's translator ignores them (`Ignore(24, 6)`) | `decode::tests::uniform::uldc_layouts_reject_the_other_layouts_bits` |
//! | `ULDC` `0xab9` with bit 91 set | Decoded as NAK's bindless form; PR #4588's translator refuses it (`Require(Take(91, 1) == 0)`) though its row marks bit 91 | `decode::tests::uniform::hand_built_words_decode` |
//!
//! Every other row's mask is the form's exactly (`decode::tests::masks_match_the_reference_table`).
//!
//! # Guards
//!
//! PR #4588 reads the guard of a uniform datapath instruction as a uniform predicate
//! (`Visitor::Run`: `info.uniform ? UPred(..) : Pred(..)`; `classify.h` marks `UMOV`, `ULDC`,
//! `UPLOP3` and the twins' forms 1 and 4 uniform, and not `R2UR`). The decoder reads every guard
//! as [`super::Instr::guard`], a [`PredSrc`], and prints it `@p0`; that a twin's or `UMOV`'s
//! guard names `UP0` is for the translator, and its text a guess.

use std::fmt::{self, Display, Formatter};

use super::super::DecodeError;
use super::super::alu::{self, AluForm};
use super::super::fields::{
    ALU_FORM, CBUF_INDEX, CBUF_OFFSET, DST, SLOT_IMM, SLOT_REG, SLOT_UNIFORM, SLOT_UNIFORM_REG,
    SRC0, SRC2, mask_of,
};
use super::super::form::{Form, Reader};
use super::super::instruction::Field;
use super::super::operand::{AluSrc, CBuf, CBufRef, Gpr, UniformPred, UniformPredSrc, UniformReg};
use super::control::{COND, COND_NOT, PRED_DST, UNIFORM_BIT, UNIFORM_DST};
use super::integer::{
    Alu, IntSrc, OpBMsk, OpBRev, OpFlo, OpIAdd3, OpIAdd3X, OpIMad, OpIMad64, OpISetP, OpLea,
    OpLeaX, OpLop3, OpPopC, OpPrmt, OpSel, OpSgxt, OpShf, PLOP3_LUT0_HIGH, PLOP3_LUT0_LOW,
    PLOP3_LUT1, PRED_B, PRED_B_NOT, PRED_C, PRED_C_NOT, PRED_DST1,
};
use super::memory::{LDC_HANDLE, MEM_TYPE, MemType, offset_text, read_coded};
use super::{IMadMode, Named, Op, shells};
#[cfg(test)]
use {super::super::form::ReferenceRow, proptest::strategy::BoxedStrategy};

// ---------------------------------------------------------------------------------------------
// Opcodes and fields.

/// `R2UR`'s 12-bit opcode (NAK `OpR2UR`: `set_opcode(0x3c2)` below SM 10.0).
pub(crate) const R2UR_OPCODE: u16 = 0x3c2;
/// `UMOV` with an immediate (NAK `OpMov`: `set_opcode(0xc82)` and then form 4 in bits 9..12).
pub(crate) const UMOV_IMM_OPCODE: u16 = 0x882;
/// `UMOV` from a uniform register (NAK `OpMov`: `set_opcode(0xc82)`, form 6).
pub(crate) const UMOV_UR_OPCODE: u16 = 0xc82;
/// `ULDC` (NAK `OpLdc`: `set_opcode(0xab9)` for a uniform destination below SM 10.0).
pub(crate) const ULDC_OPCODE: u16 = 0xab9;
/// The indexed `ULDC` (PR #4588 only: `Visitor::ULDC`, `indexed = op == 0xabb`; its table names
/// the row `ULDC.64`, a conflict the module docs describe).
pub(crate) const ULDC64_OPCODE: u16 = 0xabb;
/// `ULDC` from global memory (NAK `OpLdcg`: `set_opcode(0x8b8)` below SM 10.0).
pub(crate) const LDCG_OPCODE: u16 = 0x8b8;
/// `UPLOP3` (NAK `OpPLop3`: `set_opcode(0x89c)` for uniform predicates).
pub(crate) const UPLOP3_OPCODE: u16 = 0x89c;

/// The indexed `ULDC`'s offset register (PR #4588 `Visitor::ULDC`: `Take(24, 6)`), where the
/// bindless `ULDC` has its handle ([`LDC_HANDLE`]).
pub(crate) const ULDC_INDEX: Field = Field::new(24, UniformReg::BITS);
/// `LDCG`'s address pair (NAK `OpLdcg`: `set_ureg_src(24, &self.addr)`).
pub(crate) const LDCG_ADDR: Field = Field::new(24, UniformReg::BITS);
/// `LDCG`'s offset (NAK `OpLdcg`: `set_field(38..70, self.offset)`, an `i64` written into 32
/// bits, so read as two's complement).
pub(crate) const LDCG_OFFSET: Field = Field::new(38, 32);

/// The 9-bit opcode of the uniform ALU forms, below the operand form (`encode_ualu_base`:
/// `set_field(0..9, opcode)`, then the form at [`ALU_FORM`]).
const BASE: Field = Field::new(0, ALU_FORM.start);

// ---------------------------------------------------------------------------------------------
// The twins.

/// `UIADD3` and `UIADD3.X` (NAK `OpIAdd3`, `OpIAdd3X`: `encode_ualu(0x090, ..)`).
pub(crate) const UIADD3_OPCODE: u16 = 0x090;
/// `ULEA` and `ULEA.X` (NAK `OpLea`, `OpLeaX`: `encode_ualu(0x091, ..)`).
pub(crate) const ULEA_OPCODE: u16 = 0x091;
/// `ULOP3` (NAK `OpLop3`: `encode_ualu(0x092, ..)`).
pub(crate) const ULOP3_OPCODE: u16 = 0x092;
/// `UPRMT` (NAK `OpPrmt`: `encode_ualu(0x96, ..)`).
pub(crate) const UPRMT_OPCODE: u16 = 0x096;
/// `USHF` (NAK `OpShf`: `encode_ualu(0x099, ..)`).
pub(crate) const USHF_OPCODE: u16 = 0x099;
/// `UIMAD` (NAK `OpIMad`: `encode_ualu(0x0a4, ..)`).
pub(crate) const UIMAD_OPCODE: u16 = 0x0a4;
/// `UIMAD.WIDE` (NAK `OpIMad64`: `encode_ualu(0x0a5, ..)`).
pub(crate) const UIMAD_WIDE_OPCODE: u16 = 0x0a5;
/// `UISETP` (NAK `OpISetP`: `encode_ualu(0x08c, ..)`).
pub(crate) const UISETP_OPCODE: u16 = 0x08c;
/// `USEL` (NAK `OpSel`: `encode_ualu(0x087, ..)`).
pub(crate) const USEL_OPCODE: u16 = 0x087;
/// `USGXT` (NAK `OpSgxt`: `encode_ualu(0x09a, ..)`).
pub(crate) const USGXT_OPCODE: u16 = 0x09a;
/// `UBMSK` (NAK `OpBMsk`: `encode_ualu(0x09b, ..)`).
pub(crate) const UBMSK_OPCODE: u16 = 0x09b;
/// `UFLO` (NAK `OpFlo`: `encode_ualu(0x0bd, ..)`).
pub(crate) const UFLO_OPCODE: u16 = 0x0bd;
/// `UBREV` (NAK `OpBRev`: `encode_ualu(0x0be, ..)`).
pub(crate) const UBREV_OPCODE: u16 = 0x0be;
/// `UPOPC` (NAK `OpPopC`: `encode_ualu(0x0bf, ..)`).
pub(crate) const UPOPC_OPCODE: u16 = 0x0bf;

/// The forms of a twin with sources 0, 1 and 2 (`encode_ualu_base`: 1, 2 and 4).
const THREE_SOURCE_FORMS: [AluForm; 3] = [AluForm::Reg, AluForm::ImmSrc2, AluForm::ImmSrc1];
/// The forms of a twin without source 2: source 1 is in the wide slot.
const TWO_SOURCE_FORMS: [AluForm; 2] = [AluForm::Reg, AluForm::ImmSrc1];

/// A uniform ALU instruction and the vector instruction whose fields it shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Twin {
    /// The twin's 9-bit opcode.
    pub(crate) base: u16,
    /// The vector instruction.
    pub(crate) vector: Alu,
}

/// Every twin, in [`FORMS`] order (the module docs' table).
pub(crate) const TWINS: [Twin; 14] = [
    Twin::new(UIADD3_OPCODE, Alu::IAdd3),
    Twin::new(ULEA_OPCODE, Alu::Lea),
    Twin::new(ULOP3_OPCODE, Alu::Lop3),
    Twin::new(UPRMT_OPCODE, Alu::Prmt),
    Twin::new(USHF_OPCODE, Alu::Shf),
    Twin::new(UIMAD_OPCODE, Alu::IMad),
    Twin::new(UIMAD_WIDE_OPCODE, Alu::IMadWide),
    Twin::new(UISETP_OPCODE, Alu::ISetP),
    Twin::new(USEL_OPCODE, Alu::Sel),
    Twin::new(USGXT_OPCODE, Alu::Sgxt),
    Twin::new(UBMSK_OPCODE, Alu::BMsk),
    Twin::new(UFLO_OPCODE, Alu::Flo),
    Twin::new(UBREV_OPCODE, Alu::BRev),
    Twin::new(UPOPC_OPCODE, Alu::PopC),
];

impl Twin {
    const fn new(base: u16, vector: Alu) -> Self {
        Self { base, vector }
    }

    /// The operand forms the twin has.
    pub(crate) const fn forms(self) -> &'static [AluForm] {
        if matches!(self.vector.srcs()[2], IntSrc::Absent) {
            &TWO_SOURCE_FORMS
        } else {
            &THREE_SOURCE_FORMS
        }
    }

    /// The twin of the 9-bit opcode `base`.
    pub(crate) fn of_base(base: u64) -> Option<Self> {
        TWINS.into_iter().find(|twin| u64::from(twin.base) == base)
    }

    /// The twin of the vector instruction with the 9-bit opcode `base` (for the test assembler).
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn of_vector_base(base: u64) -> Option<Self> {
        TWINS
            .into_iter()
            .find(|twin| u64::from(twin.vector.base()) == base)
    }

    /// The register fields of the vector instruction's form `form`, eight bits wide each: its
    /// destination, source 0, the wide slot's register in form 1, and the register slot, each
    /// where the vector form has it. The twin's form keeps the low [`UniformReg::BITS`] bits of
    /// each.
    pub(crate) const fn register_fields(self, form: AluForm) -> [Option<Field>; 4] {
        let known = self.vector.form(form).known;
        let slot = if matches!(form, AluForm::Reg) {
            Some(SLOT_REG)
        } else {
            None
        };
        let candidates = [Some(DST), Some(SRC0), slot, Some(SRC2)];
        let mut fields = [None; 4];
        let mut index = 0;
        while index < candidates.len() {
            if let Some(field) = candidates[index]
                && known & field.mask() == field.mask()
            {
                fields[index] = Some(field);
            }
            index += 1;
        }
        fields
    }

    /// The twin in `form`: the vector form's bits less the top two of each register field,
    /// with bit 91 fixed.
    pub(crate) const fn form(self, form: AluForm) -> Form {
        let vector = self.vector.form(form);
        let fields = self.register_fields(form);
        let mut narrow = 0;
        let mut index = 0;
        while index < fields.len() {
            if let Some(field) = fields[index] {
                narrow |=
                    Field::new(field.start + UniformReg::BITS, Gpr::BITS - UniformReg::BITS).mask();
            }
            index += 1;
        }
        Form {
            opcode: form.opcode(self.base),
            known: (vector.known & !narrow) | SLOT_UNIFORM.mask(),
            fixed: vector.fixed | SLOT_UNIFORM.mask(),
            decode: decode_twin,
        }
    }
}

/// A vector operation the uniform datapath also has (a twin's vector struct): its register
/// operands, which [`Uniform::new`] checks.
pub(crate) trait TwinOperands: Copy {
    /// The destination register, if the operation has one, and sources 0, 1 and 2, each where
    /// the operation has it.
    fn operands(&self) -> (Option<Gpr>, [Option<AluSrc>; 3]);

    /// Whether the operation's mode has a uniform form (`IMAD.HI` has none).
    fn has_twin(&self) -> bool {
        true
    }

    /// This operation with `dst` and `srcs` in place of its operands, for test generators.
    #[cfg(test)]
    fn with_operands(self, dst: Option<Gpr>, srcs: [Option<AluSrc>; 3]) -> Self;
}

/// Whether a uniform register field can hold `reg`: `UR0` to `UR62` as [`Gpr`]s of those
/// numbers, and [`Gpr::RZ`] as `URZ`.
pub(crate) const fn fits_uniform(reg: Gpr) -> bool {
    reg.is_zero() || reg.index() < UniformReg::URZ.index()
}

/// A twin: the vector operation `T` on the uniform datapath (the module docs). Each [`Gpr`]
/// operand of `T` names the uniform register of its number, [`Gpr::RZ`] naming `URZ`, and each
/// predicate operand the uniform predicate of its number, `PT` naming `UPT`.
///
/// Every value has a word: [`Uniform::new`] refuses a `T` whose registers are past `UR62` (other
/// than `RZ`), whose source 0 is not a register, which has a constant-buffer or uniform-register
/// source or two immediates, or whose mode has no uniform form. Only this crate makes values
/// (the decoder and the tests' generators); there is no public constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Uniform<T> {
    op: T,
}

impl<T> Uniform<T> {
    /// The vector operation, whose registers name uniform registers.
    pub const fn op(&self) -> &T {
        &self.op
    }

    /// `op` on the uniform datapath, if the uniform encoding can hold it.
    pub(crate) fn new(op: T) -> Option<Self>
    where
        T: TwinOperands,
    {
        let (dst, srcs) = op.operands();
        let mut wide = 0;
        for (index, src) in srcs.iter().enumerate() {
            match src {
                None => {}
                Some(AluSrc::Reg { reg, .. }) if fits_uniform(*reg) => {}
                Some(AluSrc::Imm(_)) if index > 0 => wide += 1,
                Some(_) => return None,
            }
        }
        let fits = op.has_twin() && dst.is_none_or(fits_uniform) && wide <= 1;
        fits.then_some(Self { op })
    }
}

/// Whether the byte at `index` of `text` starts a register (`r1`, `rz`) or predicate (`p0`,
/// `pt`) operand: a whole word, after something that is not part of a word or a modifier.
pub(crate) fn starts_operand(text: &str, index: usize) -> bool {
    let before = text[..index].chars().next_back();
    if before.is_none_or(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_') {
        return false;
    }
    let word: &str = text[index..]
        .split(|c: char| !c.is_ascii_alphanumeric())
        .next()
        .unwrap_or_default();
    let Some(rest) = word.strip_prefix(['r', 'p']) else {
        return false;
    };
    let named = (word.starts_with('r') && rest == "z") || (word.starts_with('p') && rest == "t");
    named || (!rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
}

/// Writes a vector operation's text as its twin's: `u` before the mnemonic and before every
/// register and predicate operand (`iadd3 r1, p0, ..` is `uiadd3 ur1, up0, ..`, a guess).
fn twin_text(f: &mut Formatter<'_>, text: &str) -> fmt::Result {
    f.write_str("u")?;
    for (index, c) in text.char_indices() {
        if starts_operand(text, index) {
            f.write_str("u")?;
        }
        write!(f, "{c}")?;
    }
    Ok(())
}

impl<T: Display> Display for Uniform<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        twin_text(f, &self.op.to_string())
    }
}

/// Reads a twin: the vector instruction's decode function, with the register fields six bits
/// wide.
fn decode_twin(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let opcode = reader.opcode();
    let unknown = DecodeError::UnknownOpcode {
        // Cannot fail: the field is twelve bits wide.
        opcode: u16::try_from(opcode).unwrap_or(u16::MAX),
    };
    // Cannot fail: the form table sends this function only the twins' opcodes.
    let twin = Twin::of_base(opcode & ((1 << BASE.width) - 1)).ok_or(unknown)?;
    // A fixed bit of the form: the decoder has checked that it is set.
    reader.flag(SLOT_UNIFORM);
    reader.uniform_datapath();
    let vector = (twin.vector.form(form).decode)(reader)?;
    // Cannot fail: a vector op read with six-bit registers from forms 1, 2 and 4 fits.
    to_twin(vector).ok_or(unknown)
}

/// Defines each twin's operation type (a [`Uniform`] of the vector struct) and mnemonic, and
/// the mapping from a vector operation to its twin.
macro_rules! twins {
    ($($variant:ident($alias:ident) = $vector:ident($inner:ident), $mnemonic:literal;)*) => {
        $(
            #[doc = concat!("`", $mnemonic, "`: [`", stringify!($inner), "`] on the uniform datapath.")]
            pub type $alias = Uniform<$inner>;

            impl Named for $alias {
                const MNEMONIC: &'static str = $mnemonic;
            }
        )*

        /// The twin of `op`, decoded from a uniform form, if `op` is a twin's vector operation
        /// and fits ([`Uniform::new`]).
        fn to_twin(op: Op) -> Option<Op> {
            match op {
                $(Op::$vector(op) => Uniform::new(op).map(Op::$variant),)*
                _ => None,
            }
        }

        /// `op`, a twin's vector operation, with each register and source mapped by `reg` and
        /// `src`, then as its twin if it fits: for the test generators.
        #[cfg(test)]
        pub(crate) fn twin_of(
            op: Op,
            reg: impl Fn(Gpr) -> Gpr,
            src: impl Fn(AluSrc) -> AluSrc,
        ) -> Option<Op> {
            match op {
                $(Op::$vector(op) => {
                    let (dst, srcs) = op.operands();
                    let op = op.with_operands(dst.map(&reg), srcs.map(|one| one.map(&src)));
                    Uniform::new(op).map(Op::$variant)
                })*
                _ => None,
            }
        }

        /// Each twin's mnemonic and its vector operation's.
        #[cfg(test)]
        pub(crate) const TWIN_MNEMONICS: &[(&str, &str)] =
            &[$(($mnemonic, <$inner as Named>::MNEMONIC)),*];
    };
}

twins! {
    UIAdd3(OpUIAdd3) = IAdd3(OpIAdd3), "uiadd3";
    UIAdd3X(OpUIAdd3X) = IAdd3X(OpIAdd3X), "uiadd3.x";
    ULea(OpULea) = Lea(OpLea), "ulea";
    ULeaX(OpULeaX) = LeaX(OpLeaX), "ulea.x";
    ULop3(OpULop3) = Lop3(OpLop3), "ulop3";
    UPrmt(OpUPrmt) = Prmt(OpPrmt), "uprmt";
    UShf(OpUShf) = Shf(OpShf), "ushf";
    UIMad(OpUIMad) = IMad(OpIMad), "uimad";
    UIMad64(OpUIMad64) = IMad64(OpIMad64), "uimad.wide";
    UISetP(OpUISetP) = ISetP(OpISetP), "uisetp";
    USel(OpUSel) = Sel(OpSel), "usel";
    USgxt(OpUSgxt) = Sgxt(OpSgxt), "usgxt";
    UBMsk(OpUBMsk) = BMsk(OpBMsk), "ubmsk";
    UFlo(OpUFlo) = Flo(OpFlo), "uflo";
    UBRev(OpUBRev) = BRev(OpBRev), "ubrev";
    UPopC(OpUPopC) = PopC(OpPopC), "upopc";
}

/// Implements [`TwinOperands`] for structs whose operands are a `dst` field and named source
/// fields (`None` for a source the struct does not have).
macro_rules! twin_operands {
    ($name:ident: $dst:tt, [$($src:tt),*]) => {
        impl TwinOperands for $name {
            fn operands(&self) -> (Option<Gpr>, [Option<AluSrc>; 3]) {
                (twin_operands!(@get self $dst), [$(twin_operands!(@get self $src)),*])
            }

            #[cfg(test)]
            fn with_operands(mut self, dst: Option<Gpr>, srcs: [Option<AluSrc>; 3]) -> Self {
                twin_operands!(@set self $dst dst);
                let [s0, s1, s2] = srcs;
                twin_operands!(@sets self [$($src),*] [s0, s1, s2]);
                self
            }
        }
    };
    (@get $self:ident none) => { None };
    (@get $self:ident ($($path:tt)*)) => { Some($self.$($path)*) };
    (@set $self:ident none $value:ident) => { let _ = $value; };
    (@set $self:ident ($($path:tt)*) $value:ident) => {
        if let Some(value) = $value {
            $self.$($path)* = value;
        }
    };
    (@sets $self:ident [$a:tt, $b:tt, $c:tt] [$x:ident, $y:ident, $z:ident]) => {
        twin_operands!(@set $self $a $x);
        twin_operands!(@set $self $b $y);
        twin_operands!(@set $self $c $z);
    };
}

twin_operands!(OpIAdd3: (dst), [(srcs[0]), (srcs[1]), (srcs[2])]);
twin_operands!(OpLea: (dst), [(a), (b), (a_high)]);
twin_operands!(OpLop3: (dst), [(srcs[0]), (srcs[1]), (srcs[2])]);
twin_operands!(OpPrmt: (dst), [(srcs[0]), (sel), (srcs[1])]);
twin_operands!(OpShf: (dst), [(low), (shift), (high)]);
twin_operands!(OpIMad64: (dst), [(srcs[0]), (srcs[1]), (srcs[2])]);
twin_operands!(OpISetP: none, [(srcs[0]), (srcs[1]), none]);
twin_operands!(OpSel: (dst), [(srcs[0]), (srcs[1]), none]);
twin_operands!(OpSgxt: (dst), [(a), (bits), none]);
twin_operands!(OpBMsk: (dst), [(pos), (width), none]);
twin_operands!(OpFlo: (dst), [none, (src), none]);
twin_operands!(OpBRev: (dst), [none, (src), none]);
twin_operands!(OpPopC: (dst), [none, (src), none]);

impl TwinOperands for OpIMad {
    fn operands(&self) -> (Option<Gpr>, [Option<AluSrc>; 3]) {
        (Some(self.dst), self.srcs.map(Some))
    }

    /// `IMAD.HI` has no uniform form (NAK has no op for it; PR #4588's table no row).
    fn has_twin(&self) -> bool {
        !matches!(self.mode, IMadMode::Hi)
    }

    #[cfg(test)]
    fn with_operands(mut self, dst: Option<Gpr>, srcs: [Option<AluSrc>; 3]) -> Self {
        self.dst = dst.unwrap_or(self.dst);
        for (src, new) in self.srcs.iter_mut().zip(srcs) {
            *src = new.unwrap_or(*src);
        }
        self
    }
}

impl TwinOperands for OpIAdd3X {
    fn operands(&self) -> (Option<Gpr>, [Option<AluSrc>; 3]) {
        self.add.operands()
    }

    #[cfg(test)]
    fn with_operands(self, dst: Option<Gpr>, srcs: [Option<AluSrc>; 3]) -> Self {
        Self {
            add: self.add.with_operands(dst, srcs),
        }
    }
}

impl TwinOperands for OpLeaX {
    fn operands(&self) -> (Option<Gpr>, [Option<AluSrc>; 3]) {
        self.lea.operands()
    }

    #[cfg(test)]
    fn with_operands(self, dst: Option<Gpr>, srcs: [Option<AluSrc>; 3]) -> Self {
        Self {
            lea: self.lea.with_operands(dst, srcs),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// R2UR.

/// `R2UR ur, r`: copies a register of the first active thread into a uniform register (NAK
/// `OpR2UR`; PR #4588 `Visitor::R2UR`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpR2UR {
    /// The destination.
    pub dst: UniformReg,
    /// The source register.
    pub src: Gpr,
}

impl Named for OpR2UR {
    const MNEMONIC: &'static str = "r2ur";
}

impl Display for OpR2UR {
    /// `r2ur ur1, r2` (a guess).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "r2ur {}, {}", self.dst, self.src)
    }
}

fn decode_r2ur(reader: &mut Reader) -> Op {
    // A fixed field: the decoder has checked that every bit of it is set.
    reader.field(PRED_DST);
    Op::R2UR(OpR2UR {
        dst: reader.uniform_reg(UNIFORM_DST),
        src: reader.gpr(SRC0),
    })
}

// ---------------------------------------------------------------------------------------------
// UMOV.

/// The source of a `UMOV`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UMovSrc {
    /// A 32-bit immediate (`0x882`).
    Imm(u32),
    /// A uniform register (`0xc82`).
    Reg(UniformReg),
}

/// `UMOV ur, src`: a uniform register from an immediate or another uniform register (NAK
/// `OpMov` with a uniform destination; PR #4588 `Visitor::UMOV`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpUMov {
    /// The destination.
    pub dst: UniformReg,
    /// The source.
    pub src: UMovSrc,
}

impl Named for OpUMov {
    const MNEMONIC: &'static str = "umov";
}

impl Display for OpUMov {
    /// `umov ur1, 0x5`, `umov ur1, ur2` (a guess, `MOV`'s text with uniform registers).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "umov {}, ", self.dst)?;
        match self.src {
            UMovSrc::Imm(imm) => write!(f, "{imm:#x}"),
            UMovSrc::Reg(reg) => write!(f, "{reg}"),
        }
    }
}

fn decode_umov(reader: &mut Reader) -> Op {
    let src = if reader.opcode() == u64::from(UMOV_UR_OPCODE) {
        // A fixed bit of the form: the decoder has checked that it is set.
        reader.flag(SLOT_UNIFORM);
        UMovSrc::Reg(reader.uniform_reg(SLOT_UNIFORM_REG))
    } else {
        UMovSrc::Imm(reader.word32(SLOT_IMM))
    };
    Op::UMov(OpUMov {
        dst: reader.uniform_reg(UNIFORM_DST),
        src,
    })
}

// ---------------------------------------------------------------------------------------------
// ULDC.

/// What a `ULDC` reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ULdcSrc {
    /// A constant-buffer word, bound or bindless (`0xab9`, NAK `OpLdc`).
    CBuf(CBufRef),
    /// A word of the bound buffer `index`, at `reg` plus `offset` (`0xabb`, PR #4588 only).
    Indexed {
        /// The bound buffer, 0 to 31.
        index: u8,
        /// The register added to the offset.
        reg: UniformReg,
        /// The byte offset.
        offset: u16,
    },
    /// Global memory at the uniform register pair `addr` plus `offset`, or zero where `pred`
    /// holds (`0x8b8`, NAK `OpLdcg`: "On true the load returns 0").
    Global {
        /// The address pair's first register.
        addr: UniformReg,
        /// The byte offset.
        offset: i32,
        /// The predicate under which the load returns zero.
        pred: UniformPredSrc,
    },
}

/// `ULDC dst, src`: a load into uniform registers from a constant buffer or global memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpULdc {
    /// The first destination register.
    pub dst: UniformReg,
    /// What is read.
    pub src: ULdcSrc,
    /// The type.
    pub mem_type: MemType,
}

impl Named for OpULdc {
    const MNEMONIC: &'static str = "uldc";
}

impl Display for OpULdc {
    /// NAK's `test_ldc` (`uldc.u8 ur4, c[0x5][0x248];`, `uldc ur4, cx[ur2][0x248];`) and
    /// `test_uldc_global` (`uldc ur2, [ur4+0x100], up1;`); the indexed form's text and a global
    /// load's without a positive offset are guesses.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "uldc{} {}, ", self.mem_type, self.dst)?;
        match self.src {
            ULdcSrc::CBuf(cbuf) => write!(f, "{cbuf}"),
            ULdcSrc::Indexed { index, reg, offset } => {
                write!(f, "{}[{reg}+{offset:#x}]", CBuf::Binding(index))
            }
            ULdcSrc::Global { addr, offset, pred } => {
                write!(f, "[{addr}")?;
                offset_text(f, i64::from(offset))?;
                write!(f, "], {pred}")
            }
        }
    }
}

/// Reads a constant-buffer offset (sixteen bits).
fn cbuf_offset(reader: &mut Reader) -> u16 {
    // Cannot truncate: the field is sixteen bits wide.
    u16::try_from(reader.field(CBUF_OFFSET)).unwrap_or(u16::MAX)
}

fn decode_uldc(reader: &mut Reader) -> Result<Op, DecodeError> {
    let dst = reader.uniform_reg(UNIFORM_DST);
    let opcode = reader.opcode();
    let src = if opcode == u64::from(LDCG_OPCODE) {
        // A fixed bit of the form: the decoder has checked that it is set.
        reader.flag(UNIFORM_BIT);
        ULdcSrc::Global {
            addr: reader.uniform_reg(LDCG_ADDR),
            // Cannot fail: the field is 32 bits wide.
            offset: i32::try_from(reader.signed(LDCG_OFFSET)).unwrap_or(i32::MAX),
            pred: reader.uniform_pred_src(COND, COND_NOT),
        }
    } else if opcode == u64::from(ULDC64_OPCODE) {
        // A fixed bit of the form: the decoder has checked that it is set.
        reader.flag(UNIFORM_BIT);
        ULdcSrc::Indexed {
            index: reader.byte(CBUF_INDEX),
            reg: reader.uniform_reg(ULDC_INDEX),
            offset: cbuf_offset(reader),
        }
    } else {
        let buf = if reader.flag(UNIFORM_BIT) {
            CBuf::BindlessUniform(reader.uniform_reg(LDC_HANDLE))
        } else {
            CBuf::Binding(reader.byte(CBUF_INDEX))
        };
        ULdcSrc::CBuf(CBufRef {
            buf,
            offset: cbuf_offset(reader),
        })
    };
    Ok(Op::ULdc(OpULdc {
        dst,
        src,
        mem_type: read_coded(reader, MEM_TYPE, MemType::from_bits)?,
    }))
}

// ---------------------------------------------------------------------------------------------
// UPLOP3.

/// `UPLOP3.LUT dst0, dst1, a, b, c, lut0, lut1`: two bitwise functions of three uniform
/// predicates, by their truth tables (NAK `OpPLop3` with uniform predicates).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpUPLop3 {
    /// The destinations; `UPT` writes nothing.
    pub dsts: [UniformPred; 2],
    /// `a` (87..91), `b` (77..81) and `c` (68..72).
    pub srcs: [UniformPredSrc; 3],
    /// The truth tables of `dsts[0]` and `dsts[1]` (NAK's `LogicOp3`).
    pub luts: [u8; 2],
}

impl Named for OpUPLop3 {
    const MNEMONIC: &'static str = "uplop3";
}

impl Display for OpUPLop3 {
    /// `PLOP3`'s text (NAK's `test_plop3`) with uniform predicates (a guess).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let [a, b, c] = self.srcs;
        write!(
            f,
            "uplop3.lut {}, {}, {a}, {b}, {c}, {:#x}, {:#x}",
            self.dsts[0], self.dsts[1], self.luts[0], self.luts[1]
        )
    }
}

fn decode_uplop3(reader: &mut Reader) -> Op {
    let low = reader.byte(PLOP3_LUT0_LOW);
    let high = reader.byte(PLOP3_LUT0_HIGH);
    Op::UPLop3(OpUPLop3 {
        dsts: [
            reader.uniform_pred(PRED_DST),
            reader.uniform_pred(PRED_DST1),
        ],
        srcs: [
            reader.uniform_pred_src(COND, COND_NOT),
            reader.uniform_pred_src(PRED_B, PRED_B_NOT),
            reader.uniform_pred_src(PRED_C, PRED_C_NOT),
        ],
        // Cannot truncate: three low bits and five high bits.
        luts: [
            low | (high << PLOP3_LUT0_LOW.width),
            reader.byte(PLOP3_LUT1),
        ],
    })
}

// ---------------------------------------------------------------------------------------------
// The form table.

/// How many forms the twins have.
const TWIN_FORMS: usize = {
    let mut count = 0;
    let mut index = 0;
    while index < TWINS.len() {
        count += TWINS[index].forms().len();
        index += 1;
    }
    count
};

/// The forms that are not twins: `R2UR`, `UMOV` (two), `ULDC` (three) and `UPLOP3`.
const OTHER_FORMS: [Form; 7] = [
    Form::new(R2UR_OPCODE, mask_of(&[UNIFORM_DST, SRC0]), |reader| {
        Ok(decode_r2ur(reader))
    })
    .with_fixed(PRED_DST.mask()),
    Form::new(
        UMOV_IMM_OPCODE,
        mask_of(&[UNIFORM_DST, SLOT_IMM]),
        |reader| Ok(decode_umov(reader)),
    ),
    Form::new(
        UMOV_UR_OPCODE,
        mask_of(&[UNIFORM_DST, SLOT_UNIFORM_REG]),
        |reader| Ok(decode_umov(reader)),
    )
    .with_fixed(SLOT_UNIFORM.mask()),
    Form::new(
        ULDC_OPCODE,
        mask_of(&[
            UNIFORM_DST,
            LDC_HANDLE,
            CBUF_OFFSET,
            CBUF_INDEX,
            MEM_TYPE,
            UNIFORM_BIT,
        ]),
        decode_uldc,
    ),
    Form::new(
        ULDC64_OPCODE,
        mask_of(&[UNIFORM_DST, ULDC_INDEX, CBUF_OFFSET, CBUF_INDEX, MEM_TYPE]),
        decode_uldc,
    )
    .with_fixed(UNIFORM_BIT.mask()),
    Form::new(
        LDCG_OPCODE,
        mask_of(&[
            UNIFORM_DST,
            LDCG_ADDR,
            LDCG_OFFSET,
            MEM_TYPE,
            COND,
            COND_NOT,
        ]),
        decode_uldc,
    )
    .with_fixed(UNIFORM_BIT.mask()),
    Form::new(
        UPLOP3_OPCODE,
        mask_of(&[
            PLOP3_LUT1,
            PLOP3_LUT0_LOW,
            PRED_C,
            PRED_C_NOT,
            PLOP3_LUT0_HIGH,
            PRED_B,
            PRED_B_NOT,
            PRED_DST,
            PRED_DST1,
            COND,
            COND_NOT,
        ]),
        |reader| Ok(decode_uplop3(reader)),
    ),
];

const FORM_TABLE: [Form; TWIN_FORMS + OTHER_FORMS.len()] = {
    let mut table = [OTHER_FORMS[0]; TWIN_FORMS + OTHER_FORMS.len()];
    let mut next = 0;
    let mut twin = 0;
    while twin < TWINS.len() {
        let forms = TWINS[twin].forms();
        let mut form = 0;
        while form < forms.len() {
            table[next] = TWINS[twin].form(forms[form]);
            next += 1;
            form += 1;
        }
        twin += 1;
    }
    let mut other = 0;
    while other < OTHER_FORMS.len() {
        table[next] = OTHER_FORMS[other];
        next += 1;
        other += 1;
    }
    table
};

/// The group's forms: each twin's forms in [`TWINS`] order, then `R2UR`, `UMOV`, `ULDC` and
/// `UPLOP3`.
pub(crate) const FORMS: &[Form] = &FORM_TABLE;

/// PR #4588's rows for the group's forms, in [`FORMS`] order: written out in `rows`.
#[cfg(test)]
pub(crate) const REFERENCE_ROWS: &[ReferenceRow] = &rows::ROWS;

/// Generators of the group's decoded operations, one per operation.
#[cfg(test)]
pub(crate) fn any_ops() -> Vec<BoxedStrategy<Op>> {
    generate::ops()
}

#[cfg(test)]
mod generate;
#[cfg(test)]
mod rows;

shells! {
    UP2UR(OpUP2UR) = "up2ur";
}

//! Slice 6a, the integer ALU: every instruction of the group is decoded (`IADD3`, `IADD3.X`,
//! `IMAD` with `IMAD.HI`, `IMAD.WIDE`, `LOP3`, `SHF`, `LEA`, `LEA.X`, `ISETP`, `IMNMX`, `PRMT`,
//! `POPC`, `FLO`, `BREV`, `SGXT`, `BMSK`, `IABS`, `IDP`, `MOV`, `SEL`, `PLOP3`, `P2R` and `R2P`),
//! in the vector datapath's forms; the uniform datapath's (`UIADD3` and the others, NAK's
//! `encode_ualu`) are slice 6d's.
//!
//! # Sources
//!
//! Shapes follow NAK's `Op*` structs and encodings NAK's `impl SM70Op` blocks
//! (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` and `sm70_encode.rs` @ 52238762, MIT,
//! Copyright 2022 Collabora, Ltd.), as each constant says. Every ALU-form instruction is
//! `encode_alu(opcode, dst, src0, src1, src2)` (`alu`'s module docs), so an instruction with
//! all three sources has the seven operand forms and one without source 2 (or with source 1
//! alone) has forms 1, 4, 5 and 6. PR #4588's table has a row for each of the seven forms of
//! every instruction here (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h`
//! @ e5fb3b97); its translator's field reads are in `ops_int.cpp`, `ops_alu.cpp`,
//! `ops_memory.cpp` (`BREV`) and `ops_move.cpp`, and its opcode dispatch in `visitor.cpp`,
//! `Visitor::Run`.
//!
//! `IMAD.HI` (`0x027`), `P2R` (`0x803`) and `R2P` (`0x804`) come from PR #4588 alone: NAK has
//! no op for them. **Each needs an `nvdisasm` check** (spec, verification E2), as do the fields
//! below that only PR #4588 reads (marked "PR #4588 only").
//!
//! # Source modifiers
//!
//! NAK's integer sources have one modifier bit, the negation position of their slot (bit 72
//! for source 0, 63 for the wide slot, 75 for the register slot): integer negation for an
//! `I32` source and bitwise not for a `B32` one (`ir.rs`, `Src::supports_type`), and none for
//! an `ALU` source. An integer source never has its absolute-value bit set (`sm70_encode.rs`,
//! `src_mod_has_abs` is false for `INeg` and `BNot`), and in most instructions that position
//! holds a field of the instruction (`IMAD`'s signedness at bit 73, `.X` at bit 74). So
//! [`IntSrc`] says, for each source of each instruction, whether it reads its negation bit;
//! absolute-value bits are never source modifiers here, and an [`AluSrc`] this group decodes
//! always has `abs` false. Which sources read the bit:
//!
//! | Instruction | Source 0 | Source 1 | Source 2 | Reference |
//! |---|---|---|---|---|
//! | `IADD3`, `IADD3.X` | negation | negation | negation | NAK `I32`/`B32` |
//! | `IMAD` | negation | negation | negation | PR #4588 only (`Visitor::IMAD`: `neg_a`, `b.negate`, `c.negate`); NAK's sources are `ALU` |
//! | `LEA`, `LEA.X` | (bit 72 is the shifted value's modifier) | negation in the wide slot only (bit 75 is the shift in forms 2, 3 and 7) | none | NAK `I32`/`B32` |
//! | `IDP` | negation | negation | negation | NAK `I32` |
//! | `POPC`, `FLO` | | not | | NAK: `set_field(63..64, not_mod)` |
//! | every other | none | none | none | NAK `ALU` |
//!
//! # Decisions taken here
//!
//! - **A field NAK writes as a constant and PR #4588 requires to be that constant is a fixed
//!   field** (`DecodeError::MissingBit` when clear): `IMAD`'s, `IMAD.WIDE`'s, `IMAD.HI`'s and
//!   `FLO`'s predicate destination (81..84 = `PT`), `LOP3`'s predicate input (87..91 = `!PT`)
//!   and `IMAD.HI`'s carry input (87..91 = `!PT`, PR #4588 only). Every fixed field is all
//!   ones, so it is listed through `ReferenceRow::fixing`.
//! - **A field NAK writes as a constant (or not at all) that PR #4588 reads with a meaning is a
//!   field of the op** (the control group's convention): `LOP3`'s predicate destination,
//!   `ISETP`'s second destination, `LEA`'s `.SX32` (bit 73), `IMAD`'s `.X` (bit 74) and the
//!   carry inputs of `IADD3`, `LEA` and `IMAD` without `.X` (87..91, and 77..81 for
//!   `IADD3`). **Conflict** for those carry inputs: NAK's `IADD3` writes `!PT`, its `LEA` and
//!   `IMAD` write nothing (0, `P0`), and PR #4588 requires `!PT` in all three. Kept as a field so
//!   every such word decodes and re-assembles; printed only by the `.X` forms, as `nvdisasm`
//!   leaves NAK's `P0` out of `LEA`'s text (`nvdisasm_tests.rs`, `test_lea`), so two words that
//!   differ only there print the same.
//! - **`IMAD.HI` is a mode of [`OpIMad`]** ([`IMadMode::Hi`]), not a variant of its own, so the
//!   `Op` enum is unchanged (the proposals' "Shader decoder follow-ups" left it unowned). Its
//!   forms are PR #4588's rows: 1, 4 and 5 (`0x227`, `0x827`, `0xa27`).
//! - **`SHF`'s type field** follows NAK (0 `I64`, 1 `U64`, 2 `I32`, 3 `U32`; `OpShf`'s
//!   encoder). **Conflict**: PR #4588 reads 1 as `U32`, 2 as `S64` and 3 as `U64` and refuses 0
//!   (`Visitor::SHF`). NAK's encoding is the one its compiler's shaders run with; owner question.
//! - **`IADD3.X` and `LEA.X` are bit 74 in every form** (NAK: `set_bit(74, true)`).
//!   **Conflict**: PR #4588 reads `IADD3.X` as source 2's absolute-value bit and `LEA.X` as
//!   bit 74 but then refuses source 1's absolute value (`Visitor::IADD3`, `Visitor::LEA`), which
//!   in forms 2, 3 and 7 is the same bit 74 (`IADD3`: bit 62 in forms 3 and 7, nothing in form
//!   2). NAK wins (spec: the decoder is the inverse of NAK's encoder).
//! - **`IDP`'s bit 76** is PR #4588's `.2A` mode (`Visitor::IDP`: "IDP.2A.LO.U16.U8"), read in
//!   form 1 only, the one row that marks it (`0x226`); NAK's `OpIDp4` is the `.4A` mode with the
//!   bit clear. The text `idp.2a.lo` and the type names in that mode are guesses.
//! - **`SGXT`'s bit 75** is `.W` (NAK: `set_bit(75, false); // .W (wrap vs clamp)`), a field
//!   though NAK writes it clear and PR #4588 does not read it.
//!
//! # Divergences from PR #4588's table
//!
//! | Rows | What this decoder does | Pinned by |
//! |---|---|---|
//! | Forms 2, 3 and 7 of `SEL`, `ISETP`, `IABS`, `IMNMX`, `SGXT`, `BMSK`, `FLO`, `BREV` and `POPC` | Unknown opcodes: without source 2, NAK's `encode_alu` puts source 1 in the wide slot, which only forms 1, 4, 5 and 6 have (as for `MOV`) | `decode::tests::integer::integer_rows_left_undecoded_are_unknown_opcodes` |
//! | The absolute-value bits of integer sources (62, 73, 74) where they are not a field | Rejected (see "Source modifiers"); PR #4588's translator refuses them too where it reads them (`Visitor::IADD3`, `IMAD`, `LEA`, `IMAD_WIDE`) | `integer_dropped_bits_are_refused` |
//! | The negation bits of sources that have none (see the table above) | Rejected: NAK writes them clear; PR #4588 refuses them (`SHF`, `IMNMX`, `PRMT`, `IMAD_WIDE`, `LOP3`, `SEL` and the one-source instructions read their sources without modifiers) | `integer_dropped_bits_are_refused` |
//! | `LOP3`'s bit 80 (`.PAND`) | Rejected: NAK writes it clear, PR #4588 requires it clear | `integer_dropped_bits_are_refused` |
//! | `IDP`'s bits 72 to 75 in its `.2A` mode | Read as in `.4A` (PR #4588 requires them clear) | `hand_built_integer_words_decode` |
//! | `P2R`'s and `R2P`'s bit 39, `R2P`'s bits 16..24 | Rejected: PR #4588's translator requires them clear (`Visitor::P2R`, `Visitor::R2P`) though its rows mark them known | `integer_dropped_bits_are_refused` |
//! | `MOV`'s bits 62 and 63 | Rejected (see `MOV`, below) | `decode::tests::mov_rejects_source_modifier_bits` |
//!
//! Every other row's mask is the form's exactly, fixed fields aside
//! (`decode::tests::masks_match_the_reference_table`).
//!
//! # Values the decoder refuses
//!
//! `ISETP`'s combining operation 3 (NAK: `And` 0, `Or` 1, `Xor` 2; PR #4588: `Require(set_op <=
//! 2)`) and `PRMT`'s mode 7 (NAK: `Index` 0 to `Replicate16` 6) are
//! [`DecodeError::UnknownValue`].
//!
//! # `MOV`
//!
//! `MOV` is `encode_alu(0x002, dst, None, src, None)`, so its one source is source 1 and its
//! forms are 1, 4, 5 and 6. PR #4588 decodes the same four
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/visitor.cpp` @ e5fb3b97, `Visitor::Run`
//! sends opcodes `0x202`, `0x802`, `0xa02` and `0xc02` to `MOV`); its table's rows for forms 2,
//! 3 and 7 (`0x402`, `0x602`, `0xe02`) are not decoded here. `MOV` has no source modifiers (NAK
//! never modifies `OpMov::src`), so bits 62 and 63, which the table's rows `0x202`, `0xa02` and
//! `0xc02` mark known, are rejected in every form; PR #4588's translator does not take them
//! either (`ops_move.cpp`, `MOV`, reads the slot with `WideSlot(false)`). Pinned by
//! `decode::tests::mov_rejects_source_modifier_bits`.

use std::fmt::{self, Display, Formatter};

use super::super::DecodeError;
use super::super::alu::{self, AluForm, Slot, Source, Sources};
use super::super::fields::{DST, SLOT_NEG, SRC0, SRC0_NEG, SRC2_NEG, mask_of};
use super::super::form::{DecodeFn, Form, Reader};
use super::super::instruction::Field;
use super::super::operand::{AluSrc, CBufRef, Gpr, Pred, PredSrc, UniformPredSrc, UniformReg};
use super::super::print::{IntOperand, flag};
use super::control::{COND, COND_NOT, PRED_DST};
use super::{Named, Op, shells};
#[cfg(test)]
use {
    super::super::fields::SLOT_UNIFORM, super::super::form::ReferenceRow,
    proptest::strategy::BoxedStrategy,
};

// ---------------------------------------------------------------------------------------------
// Fields several instructions share.

/// A second predicate destination (NAK: `set_pred_dst(84..87, ..)` in `OpIAdd3`, `OpISetP` and
/// `OpPLop3`); `PT` writes nothing.
pub(crate) const PRED_DST1: Field = Field::new(84, Pred::BITS);
/// A predicate source at bits 77..80 (NAK: `set_pred_src(77..80, 80, ..)`: `IADD3`'s second
/// carry, `PLOP3`'s source 1).
pub(crate) const PRED_B: Field = Field::new(77, Pred::BITS);
/// [`PRED_B`]'s negation.
pub(crate) const PRED_B_NOT: Field = Field::bit(80);
/// A predicate source at bits 68..71 (NAK: `set_pred_src(68..71, 71, ..)`: `ISETP`'s lower
/// comparison, `PLOP3`'s source 2).
pub(crate) const PRED_C: Field = Field::new(68, Pred::BITS);
/// [`PRED_C`]'s negation.
pub(crate) const PRED_C_NOT: Field = Field::bit(71);
/// The predicate input NAK writes as `!PT` (`set_pred_src(87..90, 90, &false.into())`): bits
/// 87..91 all set.
pub(crate) const NOT_PT_INPUT: u128 = COND.mask() | COND_NOT.mask();

// ---------------------------------------------------------------------------------------------
// Opcodes (9 bits for the ALU forms) and each instruction's own fields.

/// `MOV`'s 9-bit opcode (`impl SM70Op for OpMov`: `encode_alu(0x002, ..)`).
pub(crate) const MOV_OPCODE: u16 = 0x002;
/// `MOV`'s per-quad-lane mask (`set_field(72..76, quad_lanes)`).
pub(crate) const MOV_QUAD_LANES: Field = Field::new(72, 4);

/// `IADD3`'s 9-bit opcode (`impl SM70Op for OpIAdd3` and `OpIAdd3X`: `encode_alu(0x010, ..)`).
pub(crate) const IADD3_OPCODE: u16 = 0x010;
/// `IADD3.X` (`OpIAdd3X`: `set_bit(74, true); // .X`).
pub(crate) const IADD3_X: Field = Field::bit(74);

/// `LEA`'s 9-bit opcode (`impl SM70Op for OpLea` and `OpLeaX`: `encode_alu(0x011, ..)`).
pub(crate) const LEA_OPCODE: u16 = 0x011;
/// The modifier of the shifted value (`set_bit(72, self.intermediate_mod.is_ineg())`, and
/// `is_bnot()` for `LEA.X`).
pub(crate) const LEA_INTERMEDIATE: Field = Field::bit(72);
/// `.SX32`: the high word is `a`'s sign. PR #4588 only (`Visitor::LEA`: `sign_extend =
/// Take(73, 1)`); NAK writes it clear.
pub(crate) const LEA_SX32: Field = Field::bit(73);
/// `LEA.X` (`set_bit(74, true); // .X`; `false` for `LEA`).
pub(crate) const LEA_X: Field = Field::bit(74);
/// The shift (`set_field(75..80, self.shift)`).
pub(crate) const LEA_SHIFT: Field = Field::new(75, 5);
/// `.HI` (`set_bit(80, self.dst_high)`).
pub(crate) const LEA_HI: Field = Field::bit(80);

/// `LOP3`'s 9-bit opcode (`impl SM70Op for OpLop3`: `encode_alu(0x012, ..)`).
pub(crate) const LOP3_OPCODE: u16 = 0x012;
/// The truth table (`set_field(72..80, self.op.lut)`).
pub(crate) const LOP3_LUT: Field = Field::new(72, 8);

/// `IABS`'s 9-bit opcode (`impl SM70Op for OpIAbs`: `encode_alu(0x013, ..)`).
pub(crate) const IABS_OPCODE: u16 = 0x013;

/// `PRMT`'s 9-bit opcode (`impl SM70Op for OpPrmt`: `encode_alu(0x16, ..)`).
pub(crate) const PRMT_OPCODE: u16 = 0x016;
/// The mode (`set_field(72..75, ..)`).
pub(crate) const PRMT_MODE: Field = Field::new(72, 3);

/// `IMNMX`'s 9-bit opcode (`impl SM70Op for OpIMnMx`: `encode_alu(0x017, ..)`).
pub(crate) const IMNMX_OPCODE: u16 = 0x017;
/// Signed (`set_bit(73, ..)`: `I32` true, `U32` false).
pub(crate) const IMNMX_SIGNED: Field = Field::bit(73);

/// `SHF`'s 9-bit opcode (`impl SM70Op for OpShf`: `encode_alu(0x019, ..)`).
pub(crate) const SHF_OPCODE: u16 = 0x019;
/// The data type (`set_field(73..75, ..)`).
pub(crate) const SHF_TYPE: Field = Field::new(73, 2);
/// `.W` (`set_bit(75, self.wrap)`).
pub(crate) const SHF_WRAP: Field = Field::bit(75);
/// `.R` (`set_bit(76, self.right)`).
pub(crate) const SHF_RIGHT: Field = Field::bit(76);
/// `.HI` (`set_bit(80, self.dst_high)`).
pub(crate) const SHF_HI: Field = Field::bit(80);

/// `SGXT`'s 9-bit opcode (`impl SM70Op for OpSgxt`: `encode_alu(0x01a, ..)`).
pub(crate) const SGXT_OPCODE: u16 = 0x01a;
/// Signed (`set_bit(73, self.signed)`).
pub(crate) const SGXT_SIGNED: Field = Field::bit(73);
/// `.W` (`set_bit(75, false); // .W (wrap vs clamp)`).
pub(crate) const SGXT_WRAP: Field = Field::bit(75);

/// `BMSK`'s 9-bit opcode (`impl SM70Op for OpBMsk`: `encode_alu(0x01b, ..)`).
pub(crate) const BMSK_OPCODE: u16 = 0x01b;
/// `.W` (`set_bit(75, self.wrap)`).
pub(crate) const BMSK_WRAP: Field = Field::bit(75);

/// `IMAD`'s 9-bit opcode (`impl SM70Op for OpIMad`: `encode_alu(0x024, ..)`).
pub(crate) const IMAD_OPCODE: u16 = 0x024;
/// `IMAD.WIDE`'s 9-bit opcode (`impl SM70Op for OpIMad64`: `encode_alu(0x025, ..)`).
pub(crate) const IMAD_WIDE_OPCODE: u16 = 0x025;
/// `IMAD.HI`'s 9-bit opcode. PR #4588 only (`Visitor::Run`: `case 0x027: IMAD_HI()`; rows
/// `0x227`, `0x827` and `0xa27`, named `IMAD.HI`).
pub(crate) const IMAD_HI_OPCODE: u16 = 0x027;
/// Signed, in `IMAD`, `IMAD.WIDE` and `IMAD.HI` (`set_bit(73, self.signed)`; PR #4588:
/// `Take(73, 1)`).
pub(crate) const IMAD_SIGNED: Field = Field::bit(73);
/// `IMAD.X`: add the carry input. PR #4588 only (`Visitor::IMAD`: `extended = Take(74, 1)`).
pub(crate) const IMAD_X: Field = Field::bit(74);

/// `IDP`'s 9-bit opcode (`impl SM70Op for OpIDp4`: `encode_alu(0x026, ..)`).
pub(crate) const IDP_OPCODE: u16 = 0x026;
/// Whether source 0's bytes are signed (`set_bit(73, ..)`: `I8` true, `U8` false).
pub(crate) const IDP_SIGNED_A: Field = Field::bit(73);
/// Whether source 1's bytes are signed (`set_bit(74, ..)`).
pub(crate) const IDP_SIGNED_B: Field = Field::bit(74);
/// `.2A`, in form 1 only. PR #4588 only (`Visitor::IDP`: `Require(Take(76, 1) == 1 ..)`; row
/// `0x226` marks bit 76, the other forms' rows do not).
pub(crate) const IDP_2A: Field = Field::bit(76);

/// `ISETP`'s 9-bit opcode (`impl SM70Op for OpISetP`: `encode_alu(0x00c, None, ..)`).
pub(crate) const ISETP_OPCODE: u16 = 0x00c;
/// `.EX` (`set_bit(72, self.ex)`).
pub(crate) const ISETP_EX: Field = Field::bit(72);
/// Signed (`set_field(73..74, ..)`: `I32` 1, `U32` 0).
pub(crate) const ISETP_SIGNED: Field = Field::bit(73);
/// The combining operation (`set_pred_set_op(74..76, ..)`).
pub(crate) const ISETP_SET_OP: Field = Field::new(74, 2);
/// The comparison (`set_int_cmp_op(76..79, ..)`).
pub(crate) const ISETP_CMP: Field = Field::new(76, 3);

/// `SEL`'s 9-bit opcode (`impl SM70Op for OpSel`: `encode_alu(0x007, ..)`).
pub(crate) const SEL_OPCODE: u16 = 0x007;

/// `FLO`'s 9-bit opcode (`impl SM70Op for OpFlo`: `encode_alu(0x100, ..)`).
pub(crate) const FLO_OPCODE: u16 = 0x100;
/// Signed (`set_field(73..74, self.signed)`).
pub(crate) const FLO_SIGNED: Field = Field::bit(73);
/// `.SH`: return the shift amount (`set_field(74..75, self.return_shift_amount)`).
pub(crate) const FLO_SH: Field = Field::bit(74);

/// `BREV`'s 9-bit opcode (`impl SM70Op for OpBRev`: `encode_alu(0x101, ..)`).
pub(crate) const BREV_OPCODE: u16 = 0x101;

/// `POPC`'s 9-bit opcode (`impl SM70Op for OpPopC`: `encode_alu(0x109, ..)`).
pub(crate) const POPC_OPCODE: u16 = 0x109;

/// `PLOP3`'s 12-bit opcode (`impl SM70Op for OpPLop3`: `set_opcode(0x81c)`).
pub(crate) const PLOP3_OPCODE: u16 = 0x81c;
/// The second truth table (`set_field(16..24, self.ops[1].lut)`).
pub(crate) const PLOP3_LUT1: Field = Field::new(16, 8);
/// The first truth table's low three bits (`set_field(64..67, self.ops[0].lut & 0x7)`).
pub(crate) const PLOP3_LUT0_LOW: Field = Field::new(64, 3);
/// The first truth table's high five bits (`set_field(72..77, self.ops[0].lut >> 3)`).
pub(crate) const PLOP3_LUT0_HIGH: Field = Field::new(72, 5);
/// Source 2 is a uniform predicate (`set_bit(67, true)`).
pub(crate) const PLOP3_UNIFORM: Field = Field::bit(67);

/// `P2R`'s 12-bit opcode. PR #4588 only (`Visitor::Run`: `case 0x803: P2R()`).
pub(crate) const P2R_OPCODE: u16 = 0x803;
/// `R2P`'s 12-bit opcode. PR #4588 only (`Visitor::Run`: `case 0x804: R2P()`).
pub(crate) const R2P_OPCODE: u16 = 0x804;
/// The predicate mask of `P2R` and `R2P`, one bit for each of `P0` to `P6` (PR #4588:
/// `Take(32, 7)`).
pub(crate) const PR_MASK: Field = Field::new(32, 7);
/// The byte of `R2P`'s source the predicates come from (PR #4588 `Visitor::R2P`: `Take(76,
/// 2)`).
pub(crate) const R2P_BYTE: Field = Field::new(76, 2);

// ---------------------------------------------------------------------------------------------
// Sources.

/// The operand forms of an instruction whose sources are 0 and 1, or 1 alone: source 1 is in
/// the wide slot.
pub(crate) const SRC1_FORMS: [AluForm; 4] = [
    AluForm::Reg,
    AluForm::ImmSrc1,
    AluForm::CBufSrc1,
    AluForm::UniformSrc1,
];
/// `IMAD.HI`'s operand forms, PR #4588's rows (`0x227`, `0x827`, `0xa27`).
pub(crate) const IMAD_HI_FORMS: [AluForm; 3] = [AluForm::Reg, AluForm::ImmSrc1, AluForm::CBufSrc1];

/// How one source of an integer ALU instruction is encoded (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntSrc {
    /// The instruction has no such source.
    Absent,
    /// A source without a modifier bit.
    Plain,
    /// A source with its slot's negation bit.
    Neg,
    /// A source with its slot's negation bit only in the wide slot (`LEA`'s source 1, whose
    /// register-slot negation bit is the shift).
    NegWide,
}

/// The encoding of each of an instruction's three sources.
pub(crate) type IntSrcs = [IntSrc; 3];

/// The negation bit `src`, source `index` of an instruction in `form`, reads, if any.
pub(crate) const fn neg_field(form: AluForm, index: usize, src: IntSrc) -> Option<Field> {
    let wide_only = match src {
        IntSrc::Absent | IntSrc::Plain => return None,
        IntSrc::Neg => false,
        IntSrc::NegWide => true,
    };
    let source = match index {
        0 => return if wide_only { None } else { Some(SRC0_NEG) },
        1 => Source::One,
        _ => Source::Two,
    };
    match form.slot(source) {
        Slot::Imm => None,
        Slot::Reg if wide_only => None,
        Slot::Reg => Some(SRC2_NEG),
        Slot::WideReg | Slot::CBuf | Slot::UniformReg => Some(SLOT_NEG),
    }
}

/// The bits the sources `srcs` of an instruction in `form` occupy, negation bits included.
const fn srcs_known(form: AluForm, srcs: IntSrcs) -> u128 {
    let sources = Sources {
        src0: !matches!(srcs[0], IntSrc::Absent),
        src1: !matches!(srcs[1], IntSrc::Absent),
        src2: !matches!(srcs[2], IntSrc::Absent),
    };
    let mut mask = alu::known(form, sources, false);
    let mut index = 0;
    while index < srcs.len() {
        if let Some(neg) = neg_field(form, index, srcs[index]) {
            mask |= neg.mask();
        }
        index += 1;
    }
    mask
}

/// `src` with its negation bit set to `neg`.
const fn negated(src: AluSrc, neg: bool) -> AluSrc {
    match src {
        AluSrc::Reg { reg, abs, .. } => AluSrc::Reg { reg, abs, neg },
        AluSrc::UniformReg { reg, abs, .. } => AluSrc::UniformReg { reg, abs, neg },
        AluSrc::CBuf { cbuf, abs, .. } => AluSrc::CBuf { cbuf, abs, neg },
        AluSrc::Imm(imm) => AluSrc::Imm(imm),
    }
}

/// Reads source `index` of `kind` in `form`, with its negation bit if it has one.
fn read_src(reader: &mut Reader, form: AluForm, kind: Alu, index: usize) -> AluSrc {
    let src = match index {
        0 => alu::src0(reader, false),
        1 => alu::src(reader, form, Source::One, false),
        _ => alu::src(reader, form, Source::Two, false),
    };
    match neg_field(form, index, kind.srcs()[index]) {
        Some(field) => negated(src, reader.flag(field)),
        None => src,
    }
}

/// Reads sources 0, 1 and 2 of `kind` in `form`.
fn read_srcs(reader: &mut Reader, form: AluForm, kind: Alu) -> [AluSrc; 3] {
    [
        read_src(reader, form, kind, 0),
        read_src(reader, form, kind, 1),
        read_src(reader, form, kind, 2),
    ]
}

/// Reads sources 0 and 1 of `kind` in `form`.
fn read_srcs2(reader: &mut Reader, form: AluForm, kind: Alu) -> [AluSrc; 2] {
    [
        read_src(reader, form, kind, 0),
        read_src(reader, form, kind, 1),
    ]
}

/// Reads a predicate destination ([`Pred::BITS`] wide, so every value names one).
fn pred_dst(reader: &mut Reader, field: Field) -> Pred {
    Pred::new(reader.byte(field)).unwrap_or(Pred::PT)
}

/// Marks a fixed field read: the decoder has checked that every bit of it is set.
fn fixed_field(reader: &mut Reader, fixed: u128) {
    reader.field(Field::new(fixed.trailing_zeros(), fixed.count_ones()));
}

/// An integer source whose negation bit is a bitwise not: printed `~`, a guess (NAK's own
/// printer writes `!`).
struct NotOperand(AluSrc);

impl Display for NotOperand {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let (_, not) = self.0.modifiers();
        flag(f, not, "~")?;
        IntOperand(negated(self.0, false)).fmt(f)
    }
}

// ---------------------------------------------------------------------------------------------
// The form table.

/// The group's ALU-form instructions: what each form of each is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Alu {
    /// `MOV`.
    Mov,
    /// `IADD3` and `IADD3.X`.
    IAdd3,
    /// `LEA` and `LEA.X`.
    Lea,
    /// `LOP3`.
    Lop3,
    /// `IABS`.
    IAbs,
    /// `PRMT`.
    Prmt,
    /// `IMNMX`.
    IMnMx,
    /// `SHF`.
    Shf,
    /// `SGXT`.
    Sgxt,
    /// `BMSK`.
    BMsk,
    /// `IMAD`.
    IMad,
    /// `IMAD.WIDE`.
    IMadWide,
    /// `IDP`.
    IDp,
    /// `IMAD.HI`.
    IMadHi,
    /// `ISETP`.
    ISetP,
    /// `SEL`.
    Sel,
    /// `FLO`.
    Flo,
    /// `BREV`.
    BRev,
    /// `POPC`.
    PopC,
}

impl Alu {
    /// Every ALU-form instruction, in [`FORMS`] order.
    pub(crate) const ALL: [Self; 19] = [
        Self::Mov,
        Self::IAdd3,
        Self::Lea,
        Self::Lop3,
        Self::IAbs,
        Self::Prmt,
        Self::IMnMx,
        Self::Shf,
        Self::Sgxt,
        Self::BMsk,
        Self::IMad,
        Self::IMadWide,
        Self::IDp,
        Self::IMadHi,
        Self::ISetP,
        Self::Sel,
        Self::Flo,
        Self::BRev,
        Self::PopC,
    ];

    /// The 9-bit opcode.
    pub(crate) const fn base(self) -> u16 {
        match self {
            Self::Mov => MOV_OPCODE,
            Self::IAdd3 => IADD3_OPCODE,
            Self::Lea => LEA_OPCODE,
            Self::Lop3 => LOP3_OPCODE,
            Self::IAbs => IABS_OPCODE,
            Self::Prmt => PRMT_OPCODE,
            Self::IMnMx => IMNMX_OPCODE,
            Self::Shf => SHF_OPCODE,
            Self::Sgxt => SGXT_OPCODE,
            Self::BMsk => BMSK_OPCODE,
            Self::IMad => IMAD_OPCODE,
            Self::IMadWide => IMAD_WIDE_OPCODE,
            Self::IDp => IDP_OPCODE,
            Self::IMadHi => IMAD_HI_OPCODE,
            Self::ISetP => ISETP_OPCODE,
            Self::Sel => SEL_OPCODE,
            Self::Flo => FLO_OPCODE,
            Self::BRev => BREV_OPCODE,
            Self::PopC => POPC_OPCODE,
        }
    }

    /// How each source is encoded (the module docs' table).
    pub(crate) const fn srcs(self) -> IntSrcs {
        use IntSrc::{Absent, Neg, NegWide, Plain};
        match self {
            Self::IAdd3 | Self::IMad | Self::IDp => [Neg, Neg, Neg],
            Self::Lea => [Plain, NegWide, Plain],
            Self::Lop3 | Self::Prmt | Self::Shf | Self::IMadWide | Self::IMadHi => {
                [Plain, Plain, Plain]
            }
            Self::IMnMx | Self::Sgxt | Self::BMsk | Self::ISetP | Self::Sel => {
                [Plain, Plain, Absent]
            }
            Self::Mov | Self::IAbs | Self::BRev => [Absent, Plain, Absent],
            Self::Flo | Self::PopC => [Absent, Neg, Absent],
        }
    }

    /// The operand forms the instruction has.
    pub(crate) const fn forms(self) -> &'static [AluForm] {
        match self {
            Self::IMadHi => &IMAD_HI_FORMS,
            _ if matches!(self.srcs()[2], IntSrc::Absent) => &SRC1_FORMS,
            _ => &AluForm::ALL,
        }
    }

    /// The instruction's own fields in `form`, besides its sources and fixed fields.
    const fn fields(self, form: AluForm) -> u128 {
        match self {
            Self::Mov => mask_of(&[DST, MOV_QUAD_LANES]),
            Self::IAdd3 => mask_of(&[
                DST, IADD3_X, PRED_B, PRED_B_NOT, PRED_DST, PRED_DST1, COND, COND_NOT,
            ]),
            Self::Lea => mask_of(&[
                DST,
                LEA_INTERMEDIATE,
                LEA_SX32,
                LEA_X,
                LEA_SHIFT,
                LEA_HI,
                PRED_DST,
                COND,
                COND_NOT,
            ]),
            Self::Lop3 => mask_of(&[DST, LOP3_LUT, PRED_DST]),
            Self::IAbs | Self::BRev | Self::PopC => DST.mask(),
            Self::Prmt => mask_of(&[DST, PRMT_MODE]),
            Self::IMnMx => mask_of(&[DST, IMNMX_SIGNED, COND, COND_NOT]),
            Self::Sel => mask_of(&[DST, COND, COND_NOT]),
            Self::Shf => mask_of(&[DST, SHF_TYPE, SHF_WRAP, SHF_RIGHT, SHF_HI]),
            Self::Sgxt => mask_of(&[DST, SGXT_SIGNED, SGXT_WRAP]),
            Self::BMsk => mask_of(&[DST, BMSK_WRAP]),
            Self::IMad => mask_of(&[DST, IMAD_SIGNED, IMAD_X, COND, COND_NOT]),
            Self::IMadWide => mask_of(&[DST, IMAD_SIGNED, COND, COND_NOT]),
            Self::IMadHi => mask_of(&[DST, IMAD_SIGNED]),
            Self::IDp => {
                let two_a = if matches!(form, AluForm::Reg) {
                    IDP_2A.mask()
                } else {
                    0
                };
                mask_of(&[DST, IDP_SIGNED_A, IDP_SIGNED_B]) | two_a
            }
            Self::ISetP => mask_of(&[
                ISETP_EX,
                ISETP_SIGNED,
                ISETP_SET_OP,
                ISETP_CMP,
                PRED_DST,
                PRED_DST1,
                COND,
                COND_NOT,
                PRED_C,
                PRED_C_NOT,
            ]),
            Self::Flo => mask_of(&[DST, FLO_SIGNED, FLO_SH]),
        }
    }

    /// The fields every word has set, besides bit 91 of the uniform-register forms (see the
    /// module docs' first decision).
    const fn fixed(self) -> u128 {
        match self {
            Self::IMad | Self::IMadWide | Self::Flo => PRED_DST.mask(),
            Self::IMadHi => PRED_DST.mask() | NOT_PT_INPUT,
            Self::Lop3 => NOT_PT_INPUT,
            _ => 0,
        }
    }

    /// The function that reads the instruction.
    const fn decode(self) -> DecodeFn {
        match self {
            Self::Mov => decode_mov,
            Self::IAdd3 => decode_iadd3,
            Self::Lea => decode_lea,
            Self::Lop3 => decode_lop3,
            Self::IAbs => decode_iabs,
            Self::Prmt => decode_prmt,
            Self::IMnMx => decode_imnmx,
            Self::Shf => decode_shf,
            Self::Sgxt => decode_sgxt,
            Self::BMsk => decode_bmsk,
            Self::IMad | Self::IMadHi => decode_imad,
            Self::IMadWide => decode_imad_wide,
            Self::IDp => decode_idp,
            Self::ISetP => decode_isetp,
            Self::Sel => decode_sel,
            Self::Flo => decode_flo,
            Self::BRev => decode_brev,
            Self::PopC => decode_popc,
        }
    }

    /// The instruction in `form`.
    pub(crate) const fn form(self, form: AluForm) -> Form {
        Form::new(
            form.opcode(self.base()),
            self.fields(form) | srcs_known(form, self.srcs()),
            self.decode(),
        )
        .with_fixed(alu::fixed(form) | self.fixed())
    }
}

/// How many ALU forms the group has.
const ALU_FORMS: usize = {
    let mut count = 0;
    let mut index = 0;
    while index < Alu::ALL.len() {
        count += Alu::ALL[index].forms().len();
        index += 1;
    }
    count
};

/// The forms that are not ALU forms: `PLOP3`, `P2R` and `R2P`.
const OTHER_FORMS: [Form; 3] = [
    Form::new(
        PLOP3_OPCODE,
        mask_of(&[
            PLOP3_LUT1,
            PLOP3_LUT0_LOW,
            PLOP3_UNIFORM,
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
        |reader| Ok(decode_plop3(reader)),
    ),
    Form::new(P2R_OPCODE, mask_of(&[DST, SRC0, PR_MASK]), |reader| {
        Ok(decode_p2r(reader))
    }),
    Form::new(R2P_OPCODE, mask_of(&[SRC0, PR_MASK, R2P_BYTE]), |reader| {
        Ok(decode_r2p(reader))
    }),
];

const FORM_TABLE: [Form; ALU_FORMS + OTHER_FORMS.len()] = {
    let mut table = [OTHER_FORMS[0]; ALU_FORMS + OTHER_FORMS.len()];
    let mut next = 0;
    let mut kind = 0;
    while kind < Alu::ALL.len() {
        let forms = Alu::ALL[kind].forms();
        let mut form = 0;
        while form < forms.len() {
            table[next] = Alu::ALL[kind].form(forms[form]);
            next += 1;
            form += 1;
        }
        kind += 1;
    }
    let mut other = 0;
    while other < OTHER_FORMS.len() {
        table[next] = OTHER_FORMS[other];
        next += 1;
        other += 1;
    }
    table
};

/// The integer ALU's forms: each [`Alu`] instruction's forms in [`Alu::ALL`] order, then
/// `PLOP3`, `P2R` and `R2P`.
pub(crate) const FORMS: &[Form] = &FORM_TABLE;

// ---------------------------------------------------------------------------------------------
// MOV.

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
    let src = read_src(reader, form, Alu::Mov, 1);
    Ok(Op::Mov(OpMov {
        dst: reader.gpr(DST),
        // Cannot fail: the source was read without modifiers.
        src: MovSrc::from_alu(src).unwrap_or(MovSrc::Reg(Gpr::RZ)),
        quad_lanes: reader.byte(MOV_QUAD_LANES),
    }))
}

// ---------------------------------------------------------------------------------------------
// IADD3.

/// Writes `iadd3`'s or `iadd3.x`'s operands: the overflow predicates when either writes, then
/// the sources (bitwise-not sources with `not`), then the carries when `x`.
fn fmt_iadd3(f: &mut Formatter<'_>, op: &OpIAdd3, x: bool) -> fmt::Result {
    write!(f, " {}", op.dst)?;
    if op.overflow.iter().any(|pred| !pred.is_true()) {
        write!(f, ", {}, {}", op.overflow[0], op.overflow[1])?;
    }
    for src in op.srcs {
        if x {
            write!(f, ", {}", NotOperand(src))?;
        } else {
            write!(f, ", {}", IntOperand(src))?;
        }
    }
    if x {
        write!(f, ", {}, {}", op.carry[0], op.carry[1])?;
    }
    Ok(())
}

/// `IADD3 dst, a, b, c`: a three-way 32-bit sum, with up to two overflow predicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpIAdd3 {
    /// The destination.
    pub dst: Gpr,
    /// The overflow predicates (81..84 and 84..87): set when the sum carries out of bit 31, and
    /// out of bit 32 (PR #4588 `Visitor::IADD3`); `PT` writes nothing.
    pub overflow: [Pred; 2],
    /// `a`, `b`, `c`, each possibly negated. `a` is always a register; at most one of `b` and
    /// `c` is in the wide slot.
    pub srcs: [AluSrc; 3],
    /// Bits 87..91 and 77..81: the carry inputs of `IADD3.X`, which NAK writes as `!PT` here
    /// (see the module docs); not printed.
    pub carry: [PredSrc; 2],
}

impl Named for OpIAdd3 {
    const MNEMONIC: &'static str = "iadd3";
}

impl Display for OpIAdd3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("iadd3")?;
        fmt_iadd3(f, self, false)
    }
}

/// `IADD3.X dst, a, b, c, carry0, carry1`: `IADD3` plus two carry predicates, with bitwise-not
/// sources (NAK's `OpIAdd3X`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpIAdd3X {
    /// The same fields as `IADD3`: `srcs` are possibly bitwise-inverted, and `carry` holds the
    /// carry inputs (87..91 and 77..81).
    pub add: OpIAdd3,
}

impl Named for OpIAdd3X {
    const MNEMONIC: &'static str = "iadd3.x";
}

impl Display for OpIAdd3X {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("iadd3.x")?;
        fmt_iadd3(f, &self.add, true)
    }
}

fn decode_iadd3(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let op = OpIAdd3 {
        dst: reader.gpr(DST),
        overflow: [pred_dst(reader, PRED_DST), pred_dst(reader, PRED_DST1)],
        srcs: read_srcs(reader, form, Alu::IAdd3),
        carry: [
            reader.pred_src(COND, COND_NOT),
            reader.pred_src(PRED_B, PRED_B_NOT),
        ],
    };
    Ok(if reader.flag(IADD3_X) {
        Op::IAdd3X(OpIAdd3X { add: op })
    } else {
        Op::IAdd3(op)
    })
}

// ---------------------------------------------------------------------------------------------
// LEA.

/// `LEA dst, a, b`: `(a << shift) + b`, or with `.HI` the high word of `(a_high:a) << shift`
/// plus `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpLea {
    /// The destination.
    pub dst: Gpr,
    /// The overflow predicate (81..84); `PT` writes nothing.
    pub overflow: Pred,
    /// `a`, unmodified.
    pub a: AluSrc,
    /// `b`, possibly negated where it is in the wide slot.
    pub b: AluSrc,
    /// The high word of `a` with `.HI` (NAK writes `RZ` without), unmodified.
    pub a_high: AluSrc,
    /// The shift, 0 to 31.
    pub shift: u8,
    /// `.HI`.
    pub dst_high: bool,
    /// `.SX32`: the high word is `a`'s sign (PR #4588 only).
    pub sx32: bool,
    /// Bit 72: negate `a << shift` (bitwise not in `LEA.X`).
    pub intermediate: bool,
    /// Bits 87..91: the carry input of `LEA.X`, which NAK leaves `P0` here (see the module
    /// docs); not printed.
    pub carry: PredSrc,
}

/// `LEA.X`: `LEA` plus a carry predicate, with bitwise-not modifiers (NAK's `OpLeaX`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpLeaX {
    /// The same fields as `LEA`; `carry` is the carry input.
    pub lea: OpLea,
}

impl Named for OpLea {
    const MNEMONIC: &'static str = "lea";
}

impl Named for OpLeaX {
    const MNEMONIC: &'static str = "lea.x";
}

/// Writes a `LEA` with `x`: the modifiers, then `dst, [overflow,] a, b[, a_high], shift[,
/// carry]` (`test_lea`'s text, the rest a guess).
fn fmt_lea(f: &mut Formatter<'_>, op: &OpLea, x: bool) -> fmt::Result {
    f.write_str("lea")?;
    flag(f, op.dst_high, ".hi")?;
    flag(f, x, ".x")?;
    flag(f, op.sx32, ".sx32")?;
    write!(f, " {}", op.dst)?;
    if !op.overflow.is_true() {
        write!(f, ", {}", op.overflow)?;
    }
    let modifier = if x { "~" } else { "-" };
    f.write_str(", ")?;
    flag(f, op.intermediate, modifier)?;
    write!(f, "{}, ", IntOperand(op.a))?;
    if x {
        write!(f, "{}", NotOperand(op.b))?;
    } else {
        write!(f, "{}", IntOperand(op.b))?;
    }
    if op.dst_high || op.a_high != AluSrc::reg(Gpr::RZ) {
        write!(f, ", {}", IntOperand(op.a_high))?;
    }
    write!(f, ", {:#x}", op.shift)?;
    if x {
        write!(f, ", {}", op.carry)?;
    }
    Ok(())
}

impl Display for OpLea {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt_lea(f, self, false)
    }
}

impl Display for OpLeaX {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt_lea(f, &self.lea, true)
    }
}

fn decode_lea(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let [a, b, a_high] = read_srcs(reader, form, Alu::Lea);
    let lea = OpLea {
        dst: reader.gpr(DST),
        overflow: pred_dst(reader, PRED_DST),
        a,
        b,
        a_high,
        shift: reader.byte(LEA_SHIFT),
        dst_high: reader.flag(LEA_HI),
        sx32: reader.flag(LEA_SX32),
        intermediate: reader.flag(LEA_INTERMEDIATE),
        carry: reader.pred_src(COND, COND_NOT),
    };
    Ok(if reader.flag(LEA_X) {
        Op::LeaX(OpLeaX { lea })
    } else {
        Op::Lea(lea)
    })
}

// ---------------------------------------------------------------------------------------------
// LOP3.

/// `LOP3.LUT dst, a, b, c, lut`: any bitwise function of three values, by its truth table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpLop3 {
    /// The predicate destination (81..84): whether the result is not zero (PR #4588
    /// `Visitor::LOP3`; NAK writes `PT`, nothing).
    pub pred: Pred,
    /// The destination.
    pub dst: Gpr,
    /// `a`, `b`, `c`, unmodified.
    pub srcs: [AluSrc; 3],
    /// The truth table: bit `(a << 2 | b << 1 | c)` of each lane's bits, NAK's `LogicOp3`.
    pub lut: u8,
}

impl Named for OpLop3 {
    const MNEMONIC: &'static str = "lop3";
}

impl Display for OpLop3 {
    /// `lop3.lut [pred, ]dst, a, b, c, lut, !pt` (the fixed predicate input printed as `!pt`).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("lop3.lut ")?;
        if !self.pred.is_true() {
            write!(f, "{}, ", self.pred)?;
        }
        let [a, b, c] = self.srcs.map(IntOperand);
        write!(f, "{}, {a}, {b}, {c}, {:#x}, !pt", self.dst, self.lut)
    }
}

fn decode_lop3(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    fixed_field(reader, NOT_PT_INPUT);
    Ok(Op::Lop3(OpLop3 {
        pred: pred_dst(reader, PRED_DST),
        dst: reader.gpr(DST),
        srcs: read_srcs(reader, form, Alu::Lop3),
        lut: reader.byte(LOP3_LUT),
    }))
}

// ---------------------------------------------------------------------------------------------
// One-source instructions: IABS, BREV, POPC, FLO.

/// `IABS dst, src`: the absolute value of a signed integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpIAbs {
    /// The destination.
    pub dst: Gpr,
    /// The source, unmodified.
    pub src: AluSrc,
}

impl Named for OpIAbs {
    const MNEMONIC: &'static str = "iabs";
}

impl Display for OpIAbs {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "iabs {}, {}", self.dst, IntOperand(self.src))
    }
}

fn decode_iabs(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::IAbs(OpIAbs {
        dst: reader.gpr(DST),
        src: read_src(reader, form, Alu::IAbs, 1),
    }))
}

/// `BREV dst, src`: the source's bits in reverse order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpBRev {
    /// The destination.
    pub dst: Gpr,
    /// The source, unmodified.
    pub src: AluSrc,
}

impl Named for OpBRev {
    const MNEMONIC: &'static str = "brev";
}

impl Display for OpBRev {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "brev {}, {}", self.dst, IntOperand(self.src))
    }
}

fn decode_brev(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::BRev(OpBRev {
        dst: reader.gpr(DST),
        src: read_src(reader, form, Alu::BRev, 1),
    }))
}

/// `POPC dst, src`: how many bits are set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpPopC {
    /// The destination.
    pub dst: Gpr,
    /// The source; its `neg` is a bitwise not.
    pub src: AluSrc,
}

impl Named for OpPopC {
    const MNEMONIC: &'static str = "popc";
}

impl Display for OpPopC {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "popc {}, {}", self.dst, NotOperand(self.src))
    }
}

fn decode_popc(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::PopC(OpPopC {
        dst: reader.gpr(DST),
        src: read_src(reader, form, Alu::PopC, 1),
    }))
}

/// `FLO dst, src`: the position of the highest set bit (of the highest bit unlike the sign,
/// signed), or with `.SH` its distance from bit 31.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFlo {
    /// The destination.
    pub dst: Gpr,
    /// The source; its `neg` is a bitwise not.
    pub src: AluSrc,
    /// Signed.
    pub signed: bool,
    /// `.SH`.
    pub return_shift_amount: bool,
}

impl Named for OpFlo {
    const MNEMONIC: &'static str = "flo";
}

impl Display for OpFlo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("flo")?;
        flag(f, !self.signed, ".u32")?;
        flag(f, self.return_shift_amount, ".sh")?;
        write!(f, " {}, {}", self.dst, NotOperand(self.src))
    }
}

fn decode_flo(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    fixed_field(reader, PRED_DST.mask());
    Ok(Op::Flo(OpFlo {
        dst: reader.gpr(DST),
        src: read_src(reader, form, Alu::Flo, 1),
        signed: reader.flag(FLO_SIGNED),
        return_shift_amount: reader.flag(FLO_SH),
    }))
}

// ---------------------------------------------------------------------------------------------
// Two-source instructions: IMNMX, SEL, SGXT, BMSK.

/// `IMNMX dst, a, b, min`: the minimum of `a` and `b` where `min` is true, else the maximum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpIMnMx {
    /// The destination.
    pub dst: Gpr,
    /// `a` and `b`, unmodified.
    pub srcs: [AluSrc; 2],
    /// Compare as signed.
    pub signed: bool,
    /// Pick the minimum (87..91).
    pub min: PredSrc,
}

impl Named for OpIMnMx {
    const MNEMONIC: &'static str = "imnmx";
}

impl Display for OpIMnMx {
    /// NAK's `test_imnmx` text for SM 8.6: `imnmx[.u32] r1, r2, r3, p4`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("imnmx")?;
        flag(f, !self.signed, ".u32")?;
        let [a, b] = self.srcs.map(IntOperand);
        write!(f, " {}, {a}, {b}, {}", self.dst, self.min)
    }
}

fn decode_imnmx(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::IMnMx(OpIMnMx {
        dst: reader.gpr(DST),
        srcs: read_srcs2(reader, form, Alu::IMnMx),
        signed: reader.flag(IMNMX_SIGNED),
        min: reader.pred_src(COND, COND_NOT),
    }))
}

/// `SEL dst, a, b, cond`: `a` where `cond` is true, else `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpSel {
    /// The destination.
    pub dst: Gpr,
    /// `a` and `b`, unmodified.
    pub srcs: [AluSrc; 2],
    /// The condition (87..91).
    pub cond: PredSrc,
}

impl Named for OpSel {
    const MNEMONIC: &'static str = "sel";
}

impl Display for OpSel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let [a, b] = self.srcs.map(IntOperand);
        write!(f, "sel {}, {a}, {b}, {}", self.dst, self.cond)
    }
}

fn decode_sel(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::Sel(OpSel {
        dst: reader.gpr(DST),
        srcs: read_srcs2(reader, form, Alu::Sel),
        cond: reader.pred_src(COND, COND_NOT),
    }))
}

/// `SGXT dst, a, bits`: the low `bits` bits of `a`, sign- or zero-extended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpSgxt {
    /// The destination.
    pub dst: Gpr,
    /// `a`, unmodified.
    pub a: AluSrc,
    /// `bits`, unmodified.
    pub bits: AluSrc,
    /// Sign-extend.
    pub signed: bool,
    /// `.W`: wrap the bit count rather than clamp it (NAK's comment; NAK writes it clear).
    pub wrap: bool,
}

impl Named for OpSgxt {
    const MNEMONIC: &'static str = "sgxt";
}

impl Display for OpSgxt {
    /// NAK's `test_sgxt` text: `sgxt` signed, `sgxt.u32` not; `.w` before it is a guess.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("sgxt")?;
        flag(f, self.wrap, ".w")?;
        flag(f, !self.signed, ".u32")?;
        write!(
            f,
            " {}, {}, {}",
            self.dst,
            IntOperand(self.a),
            IntOperand(self.bits)
        )
    }
}

fn decode_sgxt(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let [a, bits] = read_srcs2(reader, form, Alu::Sgxt);
    Ok(Op::Sgxt(OpSgxt {
        dst: reader.gpr(DST),
        a,
        bits,
        signed: reader.flag(SGXT_SIGNED),
        wrap: reader.flag(SGXT_WRAP),
    }))
}

/// `BMSK dst, pos, width`: a mask of `width` ones from bit `pos`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpBMsk {
    /// The destination.
    pub dst: Gpr,
    /// `pos`, unmodified.
    pub pos: AluSrc,
    /// `width`, unmodified.
    pub width: AluSrc,
    /// `.W`: wrap the position and width rather than clamp them.
    pub wrap: bool,
}

impl Named for OpBMsk {
    const MNEMONIC: &'static str = "bmsk";
}

impl Display for OpBMsk {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("bmsk")?;
        flag(f, self.wrap, ".w")?;
        write!(
            f,
            " {}, {}, {}",
            self.dst,
            IntOperand(self.pos),
            IntOperand(self.width)
        )
    }
}

fn decode_bmsk(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let [pos, width] = read_srcs2(reader, form, Alu::BMsk);
    Ok(Op::BMsk(OpBMsk {
        dst: reader.gpr(DST),
        pos,
        width,
        wrap: reader.flag(BMSK_WRAP),
    }))
}

// ---------------------------------------------------------------------------------------------
// PRMT.

/// A `PRMT` mode (NAK's `PrmtMode` and its encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrmtMode {
    /// 0: each nibble of the selector picks a byte.
    Index,
    /// 1: `.F4E`.
    Forward4Extract,
    /// 2: `.B4E`.
    Backward4Extract,
    /// 3: `.RC8`.
    Replicate8,
    /// 4: `.ECL`.
    EdgeClampLeft,
    /// 5: `.ECR`.
    EdgeClampRight,
    /// 6: `.RC16`.
    Replicate16,
}

impl PrmtMode {
    /// Every mode, in encoding order.
    pub const ALL: [Self; 7] = [
        Self::Index,
        Self::Forward4Extract,
        Self::Backward4Extract,
        Self::Replicate8,
        Self::EdgeClampLeft,
        Self::EdgeClampRight,
        Self::Replicate16,
    ];

    /// The three-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::Index => 0,
            Self::Forward4Extract => 1,
            Self::Backward4Extract => 2,
            Self::Replicate8 => 3,
            Self::EdgeClampLeft => 4,
            Self::EdgeClampRight => 5,
            Self::Replicate16 => 6,
        }
    }

    /// The mode an encoding names; 7 names none.
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            0 => Some(Self::Index),
            1 => Some(Self::Forward4Extract),
            2 => Some(Self::Backward4Extract),
            3 => Some(Self::Replicate8),
            4 => Some(Self::EdgeClampLeft),
            5 => Some(Self::EdgeClampRight),
            6 => Some(Self::Replicate16),
            _ => None,
        }
    }
}

impl Display for PrmtMode {
    /// The modifier, empty for the default (`nvdisasm`'s names, a guess).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Index => "",
            Self::Forward4Extract => ".f4e",
            Self::Backward4Extract => ".b4e",
            Self::Replicate8 => ".rc8",
            Self::EdgeClampLeft => ".ecl",
            Self::EdgeClampRight => ".ecr",
            Self::Replicate16 => ".rc16",
        })
    }
}

/// `PRMT dst, a, sel, b`: four bytes picked from `b:a` by `sel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpPrmt {
    /// The destination.
    pub dst: Gpr,
    /// `a` (source 0) and `b` (source 2), unmodified.
    pub srcs: [AluSrc; 2],
    /// The selector (source 1), unmodified.
    pub sel: AluSrc,
    /// The mode.
    pub mode: PrmtMode,
}

impl Named for OpPrmt {
    const MNEMONIC: &'static str = "prmt";
}

impl Display for OpPrmt {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let [a, b] = self.srcs.map(IntOperand);
        write!(
            f,
            "prmt{} {}, {a}, {}, {b}",
            self.mode,
            self.dst,
            IntOperand(self.sel)
        )
    }
}

fn decode_prmt(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let bits = reader.field(PRMT_MODE);
    let mode = PrmtMode::from_bits(bits).ok_or_else(|| reader.unknown_value(PRMT_MODE, bits))?;
    let [a, sel, b] = read_srcs(reader, form, Alu::Prmt);
    Ok(Op::Prmt(OpPrmt {
        dst: reader.gpr(DST),
        srcs: [a, b],
        sel,
        mode,
    }))
}

// ---------------------------------------------------------------------------------------------
// SHF.

/// A funnel shift's data type (NAK's `IntType` as `OpShf` encodes it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShfType {
    /// 0: signed 64-bit.
    I64,
    /// 1: unsigned 64-bit.
    U64,
    /// 2: signed 32-bit.
    I32,
    /// 3: unsigned 32-bit.
    U32,
}

impl ShfType {
    /// Every type, in encoding order.
    pub const ALL: [Self; 4] = [Self::I64, Self::U64, Self::I32, Self::U32];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::I64 => 0,
            Self::U64 => 1,
            Self::I32 => 2,
            Self::U32 => 3,
        }
    }

    /// The type an encoding names; only the low two bits are read.
    pub const fn from_bits(bits: u64) -> Self {
        Self::ALL[(bits & 0b11) as usize]
    }
}

impl Display for ShfType {
    /// `nvdisasm`'s names (a guess).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::I64 => ".s64",
            Self::U64 => ".u64",
            Self::I32 => ".s32",
            Self::U32 => ".u32",
        })
    }
}

/// `SHF dst, low, shift, high`: a funnel shift of `high:low`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpShf {
    /// The destination.
    pub dst: Gpr,
    /// The low word (source 0), unmodified.
    pub low: AluSrc,
    /// The shift amount (source 1), unmodified.
    pub shift: AluSrc,
    /// The high word (source 2), unmodified.
    pub high: AluSrc,
    /// The data type.
    pub data_type: ShfType,
    /// `.W`: wrap the shift amount.
    pub wrap: bool,
    /// `.R`: shift right (else left).
    pub right: bool,
    /// `.HI`: write the high word of the result.
    pub dst_high: bool,
}

impl Named for OpShf {
    const MNEMONIC: &'static str = "shf";
}

impl Display for OpShf {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(if self.right { "shf.r" } else { "shf.l" })?;
        flag(f, self.wrap, ".w")?;
        write!(f, "{}", self.data_type)?;
        flag(f, self.dst_high, ".hi")?;
        write!(
            f,
            " {}, {}, {}, {}",
            self.dst,
            IntOperand(self.low),
            IntOperand(self.shift),
            IntOperand(self.high)
        )
    }
}

fn decode_shf(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let [low, shift, high] = read_srcs(reader, form, Alu::Shf);
    Ok(Op::Shf(OpShf {
        dst: reader.gpr(DST),
        low,
        shift,
        high,
        data_type: ShfType::from_bits(reader.field(SHF_TYPE)),
        wrap: reader.flag(SHF_WRAP),
        right: reader.flag(SHF_RIGHT),
        dst_high: reader.flag(SHF_HI),
    }))
}

// ---------------------------------------------------------------------------------------------
// IMAD, IMAD.HI, IMAD.WIDE.

/// Which `IMAD` an [`OpIMad`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IMadMode {
    /// `IMAD` (`0x024`): the low word of `a * b + c`.
    Lo {
        /// `.X`: add the carry (PR #4588 only).
        x: bool,
        /// Bits 87..91: the carry input with `.X`; without it NAK leaves `P0` here and PR #4588
        /// requires `!PT` (see the module docs), not printed.
        carry: PredSrc,
    },
    /// `IMAD.HI` (`0x027`): the high word of `a * b`, plus `c` (PR #4588 only).
    Hi,
}

/// `IMAD dst, a, b, c`: `a * b + c`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpIMad {
    /// The destination.
    pub dst: Gpr,
    /// `a`, `b`, `c`; negated sources only in [`IMadMode::Lo`].
    pub srcs: [AluSrc; 3],
    /// Signed.
    pub signed: bool,
    /// `IMAD` or `IMAD.HI`.
    pub mode: IMadMode,
}

impl Named for OpIMad {
    const MNEMONIC: &'static str = "imad";
}

impl Display for OpIMad {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("imad")?;
        let carry = match self.mode {
            IMadMode::Hi => {
                f.write_str(".hi")?;
                None
            }
            IMadMode::Lo { x, carry } => {
                flag(f, x, ".x")?;
                x.then_some(carry)
            }
        };
        flag(f, !self.signed, ".u32")?;
        let [a, b, c] = self.srcs.map(IntOperand);
        write!(f, " {}, {a}, {b}, {c}", self.dst)?;
        if let Some(carry) = carry {
            write!(f, ", {carry}")?;
        }
        Ok(())
    }
}

fn decode_imad(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    fixed_field(reader, PRED_DST.mask());
    let hi = reader.opcode() & 0x1ff == u64::from(IMAD_HI_OPCODE);
    let (kind, mode) = if hi {
        fixed_field(reader, NOT_PT_INPUT);
        (Alu::IMadHi, IMadMode::Hi)
    } else {
        let mode = IMadMode::Lo {
            x: reader.flag(IMAD_X),
            carry: reader.pred_src(COND, COND_NOT),
        };
        (Alu::IMad, mode)
    };
    Ok(Op::IMad(OpIMad {
        dst: reader.gpr(DST),
        srcs: read_srcs(reader, form, kind),
        signed: reader.flag(IMAD_SIGNED),
        mode,
    }))
}

/// `IMAD.WIDE dst, a, b, c`: the 64-bit `a * b + c`, into the register pair at `dst`, with `c` a
/// pair too.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpIMad64 {
    /// The destination pair's first register.
    pub dst: Gpr,
    /// `a`, `b`, `c` (the addend pair's first register, or a constant-buffer or uniform pair),
    /// unmodified.
    pub srcs: [AluSrc; 3],
    /// Signed.
    pub signed: bool,
    /// Bits 87..91, which NAK leaves `P0` and PR #4588 requires `!PT` (see the module docs);
    /// not printed.
    pub carry: PredSrc,
}

impl Named for OpIMad64 {
    const MNEMONIC: &'static str = "imad.wide";
}

impl Display for OpIMad64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("imad.wide")?;
        flag(f, !self.signed, ".u32")?;
        let [a, b, c] = self.srcs.map(IntOperand);
        write!(f, " {}, {a}, {b}, {c}", self.dst)
    }
}

fn decode_imad_wide(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    fixed_field(reader, PRED_DST.mask());
    Ok(Op::IMad64(OpIMad64 {
        dst: reader.gpr(DST),
        srcs: read_srcs(reader, form, Alu::IMadWide),
        signed: reader.flag(IMAD_SIGNED),
        carry: reader.pred_src(COND, COND_NOT),
    }))
}

// ---------------------------------------------------------------------------------------------
// IDP.

/// `IDP.4A dst, a, b, c`: `c` plus the dot product of the four bytes of `a` and of `b`; with
/// `.2A`, PR #4588's reading of form 1's bit 76 (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpIDp4 {
    /// The destination.
    pub dst: Gpr,
    /// `a`, `b`, `c`, each possibly negated.
    pub srcs: [AluSrc; 3],
    /// Whether the bytes of `a` and of `b` are signed.
    pub signed: [bool; 2],
    /// `.2A` (form 1 only).
    pub two_a: bool,
}

impl Named for OpIDp4 {
    const MNEMONIC: &'static str = "idp";
}

impl Display for OpIDp4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let [a_type, b_type] = if self.two_a {
            f.write_str("idp.2a.lo")?;
            [(".s16", ".u16"), (".s8", ".u8")]
        } else {
            f.write_str("idp.4a")?;
            [(".s8", ".u8"); 2]
        };
        for ((signed, unsigned), is_signed) in [a_type, b_type].into_iter().zip(self.signed) {
            f.write_str(if is_signed { signed } else { unsigned })?;
        }
        let [a, b, c] = self.srcs.map(IntOperand);
        write!(f, " {}, {a}, {b}, {c}", self.dst)
    }
}

fn decode_idp(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let two_a = matches!(form, AluForm::Reg) && reader.flag(IDP_2A);
    Ok(Op::IDp4(OpIDp4 {
        dst: reader.gpr(DST),
        srcs: read_srcs(reader, form, Alu::IDp),
        signed: [reader.flag(IDP_SIGNED_A), reader.flag(IDP_SIGNED_B)],
        two_a,
    }))
}

// ---------------------------------------------------------------------------------------------
// ISETP.

/// How a comparison combines with the accumulator predicate (NAK's `PredSetOp` and
/// `set_pred_set_op`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PredSetOp {
    /// 0.
    And,
    /// 1.
    Or,
    /// 2.
    Xor,
}

impl PredSetOp {
    /// Every operation, in encoding order.
    pub const ALL: [Self; 3] = [Self::And, Self::Or, Self::Xor];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::And => 0,
            Self::Or => 1,
            Self::Xor => 2,
        }
    }

    /// The operation an encoding names; 3 names none.
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            0 => Some(Self::And),
            1 => Some(Self::Or),
            2 => Some(Self::Xor),
            _ => None,
        }
    }
}

impl Display for PredSetOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::And => ".and",
            Self::Or => ".or",
            Self::Xor => ".xor",
        })
    }
}

/// An integer comparison (NAK's `IntCmpOp` and `set_int_cmp_op`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntCmpOp {
    /// 0: never.
    False,
    /// 1: less than.
    Lt,
    /// 2: equal.
    Eq,
    /// 3: less than or equal.
    Le,
    /// 4: greater than.
    Gt,
    /// 5: not equal.
    Ne,
    /// 6: greater than or equal.
    Ge,
    /// 7: always.
    True,
}

impl IntCmpOp {
    /// Every comparison, in encoding order.
    pub const ALL: [Self; 8] = [
        Self::False,
        Self::Lt,
        Self::Eq,
        Self::Le,
        Self::Gt,
        Self::Ne,
        Self::Ge,
        Self::True,
    ];

    /// The three-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::False => 0,
            Self::Lt => 1,
            Self::Eq => 2,
            Self::Le => 3,
            Self::Gt => 4,
            Self::Ne => 5,
            Self::Ge => 6,
            Self::True => 7,
        }
    }

    /// The comparison an encoding names; only the low three bits are read.
    pub const fn from_bits(bits: u64) -> Self {
        Self::ALL[(bits & 0b111) as usize]
    }
}

impl Display for IntCmpOp {
    /// `nvdisasm`'s names (a guess).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::False => ".f",
            Self::Lt => ".lt",
            Self::Eq => ".eq",
            Self::Le => ".le",
            Self::Gt => ".gt",
            Self::Ne => ".ne",
            Self::Ge => ".ge",
            Self::True => ".t",
        })
    }
}

/// `ISETP dst0, dst1, a, b, accum`: compares `a` with `b` and combines the result (and its
/// inverse, for `dst1`) with `accum`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpISetP {
    /// The destinations (81..84 and 84..87); `PT` writes nothing. NAK writes `PT` to the
    /// second; PR #4588 writes the inverse comparison there (`Visitor::ISETP`).
    pub dsts: [Pred; 2],
    /// `a` and `b`, unmodified.
    pub srcs: [AluSrc; 2],
    /// The comparison.
    pub cmp_op: IntCmpOp,
    /// Compare as signed.
    pub signed: bool,
    /// How the result combines with `accum`.
    pub set_op: PredSetOp,
    /// `.EX`: the high words of a 64-bit comparison, with `low_cmp` the low words' result.
    pub ex: bool,
    /// The accumulator (87..91).
    pub accum: PredSrc,
    /// The low words' comparison (68..72), printed with `.EX`.
    pub low_cmp: PredSrc,
}

impl Named for OpISetP {
    const MNEMONIC: &'static str = "isetp";
}

impl Display for OpISetP {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "isetp{}", self.cmp_op)?;
        flag(f, !self.signed, ".u32")?;
        write!(f, "{}", self.set_op)?;
        flag(f, self.ex, ".ex")?;
        let [a, b] = self.srcs.map(IntOperand);
        write!(
            f,
            " {}, {}, {a}, {b}, {}",
            self.dsts[0], self.dsts[1], self.accum
        )?;
        if self.ex || !self.low_cmp.is_true() {
            write!(f, ", {}", self.low_cmp)?;
        }
        Ok(())
    }
}

fn decode_isetp(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let bits = reader.field(ISETP_SET_OP);
    let set_op =
        PredSetOp::from_bits(bits).ok_or_else(|| reader.unknown_value(ISETP_SET_OP, bits))?;
    Ok(Op::ISetP(OpISetP {
        dsts: [pred_dst(reader, PRED_DST), pred_dst(reader, PRED_DST1)],
        srcs: read_srcs2(reader, form, Alu::ISetP),
        cmp_op: IntCmpOp::from_bits(reader.field(ISETP_CMP)),
        signed: reader.flag(ISETP_SIGNED),
        set_op,
        ex: reader.flag(ISETP_EX),
        accum: reader.pred_src(COND, COND_NOT),
        low_cmp: reader.pred_src(PRED_C, PRED_C_NOT),
    }))
}

// ---------------------------------------------------------------------------------------------
// PLOP3.

/// `PLOP3`'s third source: a predicate or, with bit 67, a uniform predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PLop3Src {
    /// A predicate.
    Pred(PredSrc),
    /// A uniform predicate (NAK: `set_upred_src(68..71, 71, ..)` and `set_bit(67, true)`).
    Uniform(UniformPredSrc),
}

impl Display for PLop3Src {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pred(src) => src.fmt(f),
            Self::Uniform(src) => src.fmt(f),
        }
    }
}

/// `PLOP3.LUT dst0, dst1, a, b, c, lut0, lut1`: two bitwise functions of three predicates, by
/// their truth tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpPLop3 {
    /// The destinations; `PT` writes nothing.
    pub dsts: [Pred; 2],
    /// `a` (87..91) and `b` (77..81).
    pub srcs: [PredSrc; 2],
    /// `c` (68..72).
    pub c: PLop3Src,
    /// The truth tables of `dsts[0]` and `dsts[1]` (NAK's `LogicOp3`).
    pub luts: [u8; 2],
}

impl Named for OpPLop3 {
    const MNEMONIC: &'static str = "plop3";
}

impl Display for OpPLop3 {
    /// NAK's `test_plop3` text.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "plop3.lut {}, {}, {}, {}, {}, {:#x}, {:#x}",
            self.dsts[0],
            self.dsts[1],
            self.srcs[0],
            self.srcs[1],
            self.c,
            self.luts[0],
            self.luts[1]
        )
    }
}

fn decode_plop3(reader: &mut Reader) -> Op {
    let c = if reader.flag(PLOP3_UNIFORM) {
        PLop3Src::Uniform(reader.uniform_pred_src(PRED_C, PRED_C_NOT))
    } else {
        PLop3Src::Pred(reader.pred_src(PRED_C, PRED_C_NOT))
    };
    let low = reader.byte(PLOP3_LUT0_LOW);
    let high = reader.byte(PLOP3_LUT0_HIGH);
    Op::PLop3(OpPLop3 {
        dsts: [pred_dst(reader, PRED_DST), pred_dst(reader, PRED_DST1)],
        srcs: [
            reader.pred_src(COND, COND_NOT),
            reader.pred_src(PRED_B, PRED_B_NOT),
        ],
        c,
        // Cannot overflow: the high field is five bits wide.
        luts: [
            low | high.wrapping_shl(PLOP3_LUT0_LOW.width),
            reader.byte(PLOP3_LUT1),
        ],
    })
}

// ---------------------------------------------------------------------------------------------
// P2R and R2P.

/// `P2R dst, PR, src, mask`: `src` with the bits `mask` selects replaced by `P0` to `P6`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpP2R {
    /// The destination.
    pub dst: Gpr,
    /// The value whose other bits are kept.
    pub src: Gpr,
    /// Which predicates, one bit each for `P0` to `P6`.
    pub mask: u8,
}

impl Named for OpP2R {
    const MNEMONIC: &'static str = "p2r";
}

impl Display for OpP2R {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "p2r {}, pr, {}, {:#x}", self.dst, self.src, self.mask)
    }
}

fn decode_p2r(reader: &mut Reader) -> Op {
    Op::P2R(OpP2R {
        dst: reader.gpr(DST),
        src: reader.gpr(SRC0),
        mask: reader.byte(PR_MASK),
    })
}

/// `R2P PR, src, mask`: sets the predicates `mask` selects from the bits of one byte of `src`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpR2P {
    /// The source.
    pub src: Gpr,
    /// Which byte of `src`, 0 to 3.
    pub byte: u8,
    /// Which predicates, one bit each for `P0` to `P6`.
    pub mask: u8,
}

impl Named for OpR2P {
    const MNEMONIC: &'static str = "r2p";
}

impl Display for OpR2P {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "r2p pr, {}", self.src)?;
        if self.byte != 0 {
            write!(f, ".b{}", self.byte)?;
        }
        write!(f, ", {:#x}", self.mask)
    }
}

fn decode_r2p(reader: &mut Reader) -> Op {
    Op::R2P(OpR2P {
        src: reader.gpr(SRC0),
        byte: reader.byte(R2P_BYTE),
        mask: reader.byte(PR_MASK),
    })
}

// ---------------------------------------------------------------------------------------------
// Test data.

/// Bits `bits`, as a mask.
#[cfg(test)]
const fn bits(bits: &[u32]) -> u128 {
    let mut mask = 0;
    let mut index = 0;
    while index < bits.len() {
        mask |= 1 << bits[index];
        index += 1;
    }
    mask
}

/// The bits of a uniform-register form that every word has set.
#[cfg(test)]
const UNIFORM: u128 = SLOT_UNIFORM.mask();

/// PR #4588's rows for the group's forms, in [`FORMS`] order
/// (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97; each
/// row's four words, low word first), with the bits each form rejects (the module docs'
/// divergences: 62 and 63 the wide slot's modifiers, 72 and 73 source 0's, 74 and 75 the
/// register slot's, 80 `LOP3`'s `.PAND`, 16..24 and 39 `R2P`'s and `P2R`'s) and its fixed
/// bits.
#[cfg(test)]
pub(crate) const REFERENCE_ROWS: &[ReferenceRow] = &[
    ReferenceRow::new(0x202, [0x00ff_ffff, 0xc000_00ff, 0x0000_0f00, 0xffff_fe00])
        .dropping(bits(&[62, 63])),
    ReferenceRow::new(0x802, [0x00ff_ffff, 0xffff_ffff, 0x0000_0f00, 0xffff_fe00]),
    ReferenceRow::new(0xa02, [0x00ff_ffff, 0xc7ff_ffff, 0x0800_0f00, 0xffff_fe00])
        .dropping(bits(&[62, 63])),
    ReferenceRow::new(0xc02, [0x00ff_ffff, 0xc000_003f, 0x0800_0f00, 0xffff_fe00])
        .dropping(bits(&[62, 63]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x210, [0xffff_ffff, 0xc000_00ff, 0x07ff_efff, 0xffff_fe00])
        .dropping(bits(&[62, 73])),
    ReferenceRow::new(0x410, [0xffff_ffff, 0xffff_ffff, 0x07ff_efff, 0xffff_fe00])
        .dropping(bits(&[73])),
    ReferenceRow::new(0x610, [0xffff_ffff, 0xc7ff_ffff, 0x0fff_efff, 0xffff_fe00])
        .dropping(bits(&[62, 73])),
    ReferenceRow::new(0x810, [0xffff_ffff, 0xffff_ffff, 0x07ff_efff, 0xffff_fe00])
        .dropping(bits(&[73])),
    ReferenceRow::new(0xa10, [0xffff_ffff, 0xc7ff_ffff, 0x0fff_efff, 0xffff_fe00])
        .dropping(bits(&[62, 73])),
    ReferenceRow::new(0xc10, [0xffff_ffff, 0xc000_003f, 0x0fff_efff, 0xffff_fe00])
        .dropping(bits(&[62, 73]))
        .fixing(UNIFORM),
    ReferenceRow::new(0xe10, [0xffff_ffff, 0xc000_003f, 0x0fff_efff, 0xffff_fe00])
        .dropping(bits(&[62, 73]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x211, [0xffff_ffff, 0xc000_00ff, 0x078f_ffff, 0xffff_fe00])
        .dropping(bits(&[62])),
    ReferenceRow::new(0x411, [0xffff_ffff, 0xffff_ffff, 0x078f_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x611, [0xffff_ffff, 0xc7ff_ffff, 0x0f8f_ffff, 0xffff_fe00])
        .dropping(bits(&[62, 63])),
    ReferenceRow::new(0x811, [0xffff_ffff, 0xffff_ffff, 0x078f_ffff, 0xffff_fe00]),
    ReferenceRow::new(0xa11, [0xffff_ffff, 0xc7ff_ffff, 0x0f8f_ffff, 0xffff_fe00])
        .dropping(bits(&[62])),
    ReferenceRow::new(0xc11, [0xffff_ffff, 0xc000_003f, 0x0f8f_ffff, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(UNIFORM),
    ReferenceRow::new(0xe11, [0xffff_ffff, 0xc000_003f, 0x0f8f_ffff, 0xffff_fe00])
        .dropping(bits(&[62, 63]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x212, [0xffff_ffff, 0xc000_00ff, 0x078f_ffff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 80]))
        .fixing(NOT_PT_INPUT),
    ReferenceRow::new(0x412, [0xffff_ffff, 0xffff_ffff, 0x078f_ffff, 0xffff_fe00])
        .dropping(bits(&[80]))
        .fixing(NOT_PT_INPUT),
    ReferenceRow::new(0x612, [0xffff_ffff, 0xc7ff_ffff, 0x0f8f_ffff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 80]))
        .fixing(NOT_PT_INPUT),
    ReferenceRow::new(0x812, [0xffff_ffff, 0xffff_ffff, 0x078f_ffff, 0xffff_fe00])
        .dropping(bits(&[80]))
        .fixing(NOT_PT_INPUT),
    ReferenceRow::new(0xa12, [0xffff_ffff, 0xc7ff_ffff, 0x0f8f_ffff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 80]))
        .fixing(NOT_PT_INPUT),
    ReferenceRow::new(0xc12, [0xffff_ffff, 0xc000_003f, 0x0f8f_ffff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 80]))
        .fixing(NOT_PT_INPUT | UNIFORM),
    ReferenceRow::new(0xe12, [0xffff_ffff, 0xc000_003f, 0x0f8f_ffff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 80]))
        .fixing(NOT_PT_INPUT | UNIFORM),
    ReferenceRow::new(0x213, [0x00ff_ffff, 0xc000_00ff, 0x0000_0000, 0xffff_fe00])
        .dropping(bits(&[62, 63])),
    ReferenceRow::new(0x813, [0x00ff_ffff, 0xffff_ffff, 0x0000_0000, 0xffff_fe00]),
    ReferenceRow::new(0xa13, [0x00ff_ffff, 0xc7ff_ffff, 0x0800_0000, 0xffff_fe00])
        .dropping(bits(&[62, 63])),
    ReferenceRow::new(0xc13, [0x00ff_ffff, 0xc000_003f, 0x0800_0000, 0xffff_fe00])
        .dropping(bits(&[62, 63]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x216, [0xffff_ffff, 0xc000_00ff, 0x0000_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 75])),
    ReferenceRow::new(0x416, [0xffff_ffff, 0xffff_ffff, 0x0000_0fff, 0xffff_fe00])
        .dropping(bits(&[75])),
    ReferenceRow::new(0x616, [0xffff_ffff, 0xc7ff_ffff, 0x0800_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 75])),
    ReferenceRow::new(0x816, [0xffff_ffff, 0xffff_ffff, 0x0000_0fff, 0xffff_fe00])
        .dropping(bits(&[75])),
    ReferenceRow::new(0xa16, [0xffff_ffff, 0xc7ff_ffff, 0x0800_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 75])),
    ReferenceRow::new(0xc16, [0xffff_ffff, 0xc000_003f, 0x0800_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 75]))
        .fixing(UNIFORM),
    ReferenceRow::new(0xe16, [0xffff_ffff, 0xc000_003f, 0x0800_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 75]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x217, [0xffff_ffff, 0xc000_00ff, 0x0780_0300, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72])),
    ReferenceRow::new(0x817, [0xffff_ffff, 0xffff_ffff, 0x0780_0300, 0xffff_fe00])
        .dropping(bits(&[72])),
    ReferenceRow::new(0xa17, [0xffff_ffff, 0xc7ff_ffff, 0x0f80_0300, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72])),
    ReferenceRow::new(0xc17, [0xffff_ffff, 0xc000_003f, 0x0f80_0300, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x219, [0xffff_ffff, 0xc000_00ff, 0x0001_1fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72])),
    ReferenceRow::new(0x419, [0xffff_ffff, 0xffff_ffff, 0x0001_1fff, 0xffff_fe00])
        .dropping(bits(&[72])),
    ReferenceRow::new(0x619, [0xffff_ffff, 0xc7ff_ffff, 0x0801_1fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72])),
    ReferenceRow::new(0x819, [0xffff_ffff, 0xffff_ffff, 0x0001_1fff, 0xffff_fe00])
        .dropping(bits(&[72])),
    ReferenceRow::new(0xa19, [0xffff_ffff, 0xc7ff_ffff, 0x0801_1fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72])),
    ReferenceRow::new(0xc19, [0xffff_ffff, 0xc000_003f, 0x0801_1fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72]))
        .fixing(UNIFORM),
    ReferenceRow::new(0xe19, [0xffff_ffff, 0xc000_003f, 0x0801_1fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x21a, [0xffff_ffff, 0xc000_00ff, 0x0000_0b00, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72])),
    ReferenceRow::new(0x81a, [0xffff_ffff, 0xffff_ffff, 0x0000_0b00, 0xffff_fe00])
        .dropping(bits(&[72])),
    ReferenceRow::new(0xa1a, [0xffff_ffff, 0xc7ff_ffff, 0x0800_0b00, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72])),
    ReferenceRow::new(0xc1a, [0xffff_ffff, 0xc000_003f, 0x0800_0b00, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x21b, [0xffff_ffff, 0xc000_00ff, 0x0000_0b00, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 73])),
    ReferenceRow::new(0x81b, [0xffff_ffff, 0xffff_ffff, 0x0000_0b00, 0xffff_fe00])
        .dropping(bits(&[72, 73])),
    ReferenceRow::new(0xa1b, [0xffff_ffff, 0xc7ff_ffff, 0x0800_0b00, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 73])),
    ReferenceRow::new(0xc1b, [0xffff_ffff, 0xc000_003f, 0x0800_0b00, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 73]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x224, [0xffff_ffff, 0xc000_00ff, 0x078e_0fff, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0x424, [0xffff_ffff, 0xffff_ffff, 0x078e_0fff, 0xffff_fe00])
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0x624, [0xffff_ffff, 0xc7ff_ffff, 0x0f8e_0fff, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0x824, [0xffff_ffff, 0xffff_ffff, 0x078e_0fff, 0xffff_fe00])
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0xa24, [0xffff_ffff, 0xc7ff_ffff, 0x0f8e_0fff, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0xc24, [0xffff_ffff, 0xc000_003f, 0x0f8e_0fff, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(PRED_DST.mask() | UNIFORM),
    ReferenceRow::new(0xe24, [0xffff_ffff, 0xc000_003f, 0x0f8e_0fff, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(PRED_DST.mask() | UNIFORM),
    ReferenceRow::new(0x225, [0xffff_ffff, 0xc000_00ff, 0x078e_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 74, 75]))
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0x425, [0xffff_ffff, 0xffff_ffff, 0x078e_0fff, 0xffff_fe00])
        .dropping(bits(&[72, 74, 75]))
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0x625, [0xffff_ffff, 0xc7ff_ffff, 0x0f8e_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 74, 75]))
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0x825, [0xffff_ffff, 0xffff_ffff, 0x078e_0fff, 0xffff_fe00])
        .dropping(bits(&[72, 74, 75]))
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0xa25, [0xffff_ffff, 0xc7ff_ffff, 0x0f8e_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 74, 75]))
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0xc25, [0xffff_ffff, 0xc000_003f, 0x0f8e_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 74, 75]))
        .fixing(PRED_DST.mask() | UNIFORM),
    ReferenceRow::new(0xe25, [0xffff_ffff, 0xc000_003f, 0x0f8e_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 74, 75]))
        .fixing(PRED_DST.mask() | UNIFORM),
    ReferenceRow::new(0x226, [0xffff_ffff, 0xc000_00ff, 0x0000_1fff, 0xffff_fe00])
        .dropping(bits(&[62])),
    ReferenceRow::new(0x426, [0xffff_ffff, 0xffff_ffff, 0x0000_0fff, 0xffff_fe00]),
    ReferenceRow::new(0x626, [0xffff_ffff, 0xc7ff_ffff, 0x0800_0fff, 0xffff_fe00])
        .dropping(bits(&[62])),
    ReferenceRow::new(0x826, [0xffff_ffff, 0xffff_ffff, 0x0000_0fff, 0xffff_fe00]),
    ReferenceRow::new(0xa26, [0xffff_ffff, 0xc7ff_ffff, 0x0800_0fff, 0xffff_fe00])
        .dropping(bits(&[62])),
    ReferenceRow::new(0xc26, [0xffff_ffff, 0xc000_003f, 0x0800_0fff, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(UNIFORM),
    ReferenceRow::new(0xe26, [0xffff_ffff, 0xc000_003f, 0x0800_0fff, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x227, [0xffff_ffff, 0xc000_00ff, 0x078e_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 74, 75]))
        .fixing(PRED_DST.mask() | NOT_PT_INPUT),
    ReferenceRow::new(0x827, [0xffff_ffff, 0xffff_ffff, 0x078e_0fff, 0xffff_fe00])
        .dropping(bits(&[72, 74, 75]))
        .fixing(PRED_DST.mask() | NOT_PT_INPUT),
    ReferenceRow::new(0xa27, [0xffff_ffff, 0xc7ff_ffff, 0x0f8e_0fff, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 74, 75]))
        .fixing(PRED_DST.mask() | NOT_PT_INPUT),
    ReferenceRow::new(0x20c, [0xff00_ffff, 0xc000_00ff, 0x07fe_7ff0, 0xffff_fe00])
        .dropping(bits(&[62, 63])),
    ReferenceRow::new(0x80c, [0xff00_ffff, 0xffff_ffff, 0x07fe_7ff0, 0xffff_fe00]),
    ReferenceRow::new(0xa0c, [0xff00_ffff, 0xc7ff_ffff, 0x0ffe_7ff0, 0xffff_fe00])
        .dropping(bits(&[62, 63])),
    ReferenceRow::new(0xc0c, [0xff00_ffff, 0xc000_003f, 0x0ffe_7ff0, 0xffff_fe00])
        .dropping(bits(&[62, 63]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x207, [0xffff_ffff, 0xc000_00ff, 0x0780_0300, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 73])),
    ReferenceRow::new(0x807, [0xffff_ffff, 0xffff_ffff, 0x0780_0300, 0xffff_fe00])
        .dropping(bits(&[72, 73])),
    ReferenceRow::new(0xa07, [0xffff_ffff, 0xc7ff_ffff, 0x0f80_0300, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 73])),
    ReferenceRow::new(0xc07, [0xffff_ffff, 0xc000_003f, 0x0f80_0300, 0xffff_fe00])
        .dropping(bits(&[62, 63, 72, 73]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x300, [0x00ff_ffff, 0xc000_00ff, 0x000e_0600, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0x900, [0x00ff_ffff, 0xffff_ffff, 0x000e_0600, 0xffff_fe00])
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0xb00, [0x00ff_ffff, 0xc7ff_ffff, 0x080e_0600, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(PRED_DST.mask()),
    ReferenceRow::new(0xd00, [0x00ff_ffff, 0xc000_003f, 0x080e_0600, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(PRED_DST.mask() | UNIFORM),
    ReferenceRow::new(0x301, [0x00ff_ffff, 0xc000_00ff, 0x0000_0000, 0xffff_fe00])
        .dropping(bits(&[62, 63])),
    ReferenceRow::new(0x901, [0x00ff_ffff, 0xffff_ffff, 0x0000_0000, 0xffff_fe00]),
    ReferenceRow::new(0xb01, [0x00ff_ffff, 0xc7ff_ffff, 0x0800_0000, 0xffff_fe00])
        .dropping(bits(&[62, 63])),
    ReferenceRow::new(0xd01, [0x00ff_ffff, 0xc000_003f, 0x0800_0000, 0xffff_fe00])
        .dropping(bits(&[62, 63]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x309, [0x00ff_ffff, 0xc000_00ff, 0x0000_0000, 0xffff_fe00])
        .dropping(bits(&[62])),
    ReferenceRow::new(0x909, [0x00ff_ffff, 0xffff_ffff, 0x0000_0000, 0xffff_fe00]),
    ReferenceRow::new(0xb09, [0x00ff_ffff, 0xc7ff_ffff, 0x0800_0000, 0xffff_fe00])
        .dropping(bits(&[62])),
    ReferenceRow::new(0xd09, [0x00ff_ffff, 0xc000_003f, 0x0800_0000, 0xffff_fe00])
        .dropping(bits(&[62]))
        .fixing(UNIFORM),
    ReferenceRow::new(0x81c, [0x00ff_ffff, 0x0000_0000, 0x07ff_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x803, [0xffff_ffff, 0x0000_00ff, 0x0000_0000, 0xffff_fe00])
        .dropping(bits(&[39])),
    ReferenceRow::new(0x804, [0xffff_ffff, 0x0000_00ff, 0x0000_3000, 0xffff_fe00])
        .dropping(bits(&[16, 17, 18, 19, 20, 21, 22, 23, 39])),
];

/// Generators of the group's decoded operations, one per operation.
#[cfg(test)]
pub(crate) fn any_ops() -> Vec<BoxedStrategy<Op>> {
    generate::ops()
}

#[cfg(test)]
mod generate;

shells! {}

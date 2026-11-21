//! Slice 6b, 32-bit float and conversions: every instruction of the group is decoded, in the
//! vector datapath's forms.
//!
//! # Sources
//!
//! Shapes follow NAK's `Op*` structs and encodings NAK's `impl SM70Op` blocks
//! (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` and `sm70_encode.rs` @ 52238762, MIT,
//! Copyright 2022 Collabora, Ltd.), as each constant says: `FADD`, `FMUL`, `FFMA`, `FMNMX`,
//! `FSET`, `FSETP`, `MUFU`, `FSWZADD`, `FRND`, `F2F`, `F2I`, `I2F` and `F2FP`. Where NAK writes
//! a field as a constant (`FMUL`'s scale, `F2I`'s `.NTZ`, `I2F`'s byte select, `F2FP`'s
//! `.MERGE_C` and `.H1`, the predicates of `FSET`), the op carries it as a field so the decoder
//! can read any value of it. Masks are PR #4588's rows
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97), and the
//! fields NAK does not write are read where PR #4588's translator reads them (`ops_alu.cpp`,
//! `ops_float.cpp`, `ops_convert.cpp`, `visitor.cpp` @ e5fb3b97).
//!
//! `FSEL` and `I2FP` come from PR #4588 alone (NAK has no op for them): `FSEL` from
//! `Visitor::SEL(true)` and its rows `0x208`, `0x808`, `0xa08`; `I2FP` from `Visitor::I2F`
//! (which `Visitor::Run` sends `0x045` to) and its rows `0x245` to `0xc45`. So are the fields NAK
//! never writes: `FMNMX`'s `.NAN` (bit 81, `propagate_nan`), the combine operation of `FSET`
//! (bits 74..76), `F2FP`'s third source when `.MERGE_C` is set (`Source(2, false)`) and the
//! meaning of `FMUL`'s scale values other than 4. **Each of these needs an `nvdisasm` check**
//! (spec, verification E2).
//!
//! The rounding-mode and flush-to-zero fields are decoded in full on every instruction that has
//! them, whether or not a host can honour them (owner's decision 10, 2026-11-12: the IR carries
//! them; they are emulated only on conversions).
//!
//! # Forms
//!
//! Each instruction is decoded in the operand forms NAK's encoder produces for it, a 16-bit
//! source's high-half select (bit 60) included wherever NAK writes it: forms 1, 5 and 6 of
//! `MUFU`, `F2I`, `F2F` and `FRND`, both opcodes of each conversion (the divergence table's
//! "high half" rows: the forms add the bit where PR #4588's rows leave it out). `FADD` puts
//! its second operand in the wide slot in every form: as source 1 when it is a register (form
//! 1), as source 2 otherwise (forms 2, 3 and 7; NAK: `src_is_zero_or_gpr` picks
//! `encode_alu(0x021, .., src1, None)` or `encode_alu(0x021, .., ZERO, src1)`). `FMUL`,
//! `FMNMX`, `FSET`, `FSETP`, `F2FP` and the one-source instructions (`MUFU`, `FRND`, `F2F`,
//! `F2I`, `I2F`, `I2FP`) put it in source 1: forms 1, 4, 5 and 6. `FSEL` has rows for forms 1,
//! 4 and 5 only. `FSWZADD` has one fixed form (`set_opcode(0x822)`). `FRND`, `F2F`, `F2I` and
//! `I2F` each have a second opcode for conversions with a 64-bit type (NAK: `0x113`, `0x110`,
//! `0x111` and `0x112` when either type is 64 bits wide).
//!
//! The uniform datapath's forms of these instructions (`UFADD` and the others, PR #4588's rows
//! with 9-bit opcodes `0x050` to `0x05d` and `0x0ba`) are slice 6d's, which leaves them unknown
//! opcodes (`op::uniform`'s module docs: NAK encodes them only from SM 12.0, and `UF2FP` has
//! fields `OpF2FP` lacks).
//!
//! # Decisions taken here
//!
//! - **The register slot an instruction does not read is kept.** `FADD` and `FMUL` read two
//!   sources, but their rows mark the register slot at bits 64..72 known. NAK writes nothing
//!   there in `FADD`'s form 1 (so R0) and `RZ` in every other form (`Src::ZERO`); PR #4588's
//!   translator ignores it (`Ignore(64, 8)`: "the unused zero register of a two-source
//!   encoding"). It is decoded as [`OpFAdd::ignored`] and [`OpFMul::ignored`], so any value
//!   re-assembles, and not printed: two words that differ only there print the same text.
//! - **Size fields.** The conversions' size fields (bits 75..77 and 84..86) hold the log2 of
//!   the type's byte count (NAK: `(bits / 8).ilog2()`). A float size of 0 (one byte) names no
//!   type: NAK's `FloatType` has `F16`, `F32` and `F64` only, and PR #4588's translator refuses
//!   it in every conversion, so the decoder refuses it ([`DecodeError::UnknownValue`]; slice 4b's
//!   canonical words let a form's bare word be refused). An integer size of 0 is an 8-bit
//!   integer, which both references have. The 64-bit opcode with no 64-bit type, and the narrow
//!   one with a 64-bit type, are decoded as written and keep their opcode (`wide`), though NAK
//!   emits neither.
//! - **The 64-bit opcode is not printed.** `OpFRnd`, `OpF2F`, `OpF2I` and `OpI2F` keep it as
//!   `wide`, so any word re-assembles, but their printers write only the types: a word whose
//!   opcode disagrees with its types (the 64-bit opcode with no 64-bit type, or the narrow one
//!   with a 64-bit type) prints the same text as the word with the matching opcode. That text
//!   reaches rejection messages and golden files, which therefore cannot tell the two apart
//!   (`decode::tests::float32::conversion_sizes_decode_every_type`).
//! - **`FMUL`'s scale** (bits 84..87, NAK's `set_field(84..87, 0x4); // TODO: PDIV`) is read as
//!   PR #4588's series, a power of two around 4 (`ops_float.cpp`, `ScaleFactor`: 3 halves, 5
//!   doubles; 2 and 1 divide by 4 and 8, 7 multiplies by 8). Value 0 is refused, as PR #4588
//!   refuses it (`Require(scale >= 1 && scale <= 7, "native FMUL scale")`).
//! - **The `FMNMX` bits 65..69** (NAK: `.IS_A`, "SM90+", written false, and a commented-out
//!   predicate destination at 66..69) are rejected: SM86 predates them and PR #4588's translator
//!   does not read them.
//!
//! # Divergences from PR #4588's table and translator
//!
//! | Rows | What this decoder does | Pinned by |
//! |---|---|---|
//! | `FADD` forms 4, 5 and 6 (`0x821`, `0xa21`, `0xc21`) | Unknown opcodes: NAK never puts `FADD`'s second operand in source 1 when it is not a register. PR #4588's translator accepts them (`Visitor::FADD`) | `decode::tests::float32::float_rows_left_undecoded_are_unknown_opcodes` |
//! | Forms 2, 3 and 7 of `FMUL`, `FMNMX`, `FSET`, `FSETP`, `MUFU`, `FRND`, `F2F`, `F2I`, `I2F` (both opcodes each) and `F2FP` | Unknown opcodes: NAK puts their source in source 1, which only forms 1, 4, 5 and 6 hold, as for `MOV` | `float_rows_left_undecoded_are_unknown_opcodes` |
//! | Bits 74 and 75 of every `FADD` and `FMUL` form | Rejected: modifiers of the slot the operation does not read (NAK's `Src::ZERO` is unmodified; PR #4588 takes only bits 64..72) | `float_rows_drop_exactly_the_listed_bits` |
//! | Bits 65..69 of every `FMNMX` form | Rejected (see above) | `float_rows_drop_exactly_the_listed_bits` |
//! | Bits 62 and 63 of `I2F` and `I2FP` forms 1, 5 and 6 | Rejected: an integer source has no modifiers (PR #4588: `Source(1, false)`) | `float_rows_drop_exactly_the_listed_bits` |
//! | Bit 91 of `I2FP` forms 1 and 4 (`0x245`, `0x845`) | Rejected: neither form has a constant buffer or uniform register for it to qualify, and PR #4588's translator does not read it there | `float_rows_drop_exactly_the_listed_bits` |
//! | Bits 62, 63, 73, 74 and 75 of `F2FP` (62 and 63 not in form 4) | Rejected: `F2FP`'s sources have no modifiers (NAK: `SrcType::ALU`; PR #4588: `Source(.., false)`) | `float_rows_drop_exactly_the_listed_bits` |
//! | Float sizes 1 to 3 that PR #4588's translator does not convert (every pair but `F2F`'s `F16` from `F32`, `FRND`'s `F32` to `F32`, `F2I`'s `F16` and `F32` sources and `I2F`'s `F32` destination), a 64-bit type in the narrow opcode, no 64-bit type in the wide one | Decoded (see above); PR #4588's translator refuses them | `conversion_sizes_decode_every_type` |
//! | `MUFU` functions 6, 7 and 9 (`RCP64H`, `RSQ64H`, `TANH`), form 4 (`0x908`) | Decoded (NAK encodes them); PR #4588's translator refuses them | `mufu_functions_and_refusals` |
//! | `FSWZADD` operations other than `0x5a` and `0x66`, and bits 77..81 not zero | Decoded (NAK encodes every operation, `.NDV`, the rounding mode and `.FTZ`); PR #4588 refuses them | `hand_built_words_of_each_form_decode` |
//! | `FFMA`'s and `FMUL`'s bit 76 (`.FMZ`), `F2FP`'s rounding and `.H1` | Decoded (NAK writes them); PR #4588's translator refuses them set | `hand_built_words_pin_each_modifier_bit` |
//! | High half (bit 60) of an `F16` source of `F2F`, forms 1, 5 and 6 of both opcodes (`0x304`, `0xb04`, `0xd04`, `0x310`, `0xb10`, `0xd10`) | Decoded as [`OpF2F::high`]: the forms add bit 60, which NAK writes (`encode_alu_base` passes `src_types()[0]`, `F16` for an `F16` source, to `encode_alu_reg`, `encode_alu_cb` and `encode_alu_ureg`, each of which calls `set_swizzle(60..62, ..)`; `opt_copy_prop`'s `prop_to_scalar_src` emits `SrcSwizzle::Yy`, for example in `f2f32` of `unpackHalf2x16`'s high half). PR #4588's rows leave bit 60 out | `decode::tests::float32::nak_high_halves_decode_in_every_form_with_bit_60`, `decode::tests::framework::added_bits_decode_print_and_round_trip` |
//! | High half (bit 60) of an `F16` source of `FRND`, forms 1, 5 and 6 of both opcodes (`0x307`, `0xb07`, `0xd07`, `0x313`, `0xb13`, `0xd13`) | Decoded as [`OpFRnd::high`], added as for `F2F`: NAK writes it the same way; PR #4588's rows leave it out | as above |
//! | High half (bit 60) of `MUFU.F16` in forms 5 and 6 (`0xb08`, `0xd08`) | Decoded as [`OpMuFu::high`], added: NAK writes it from `encode_alu_cb` and `encode_alu_ureg`; PR #4588's rows for those forms leave it out | as above |
//! | High half (bit 60) of the narrow `F2I`'s `F16` source in forms 5 and 6 (`0xb05`, `0xd05`) | Decoded as [`OpF2I::high`], added: NAK writes it from `encode_alu_cb` and `encode_alu_ureg`; PR #4588's rows for those forms leave it out | as above |
//! | High half (bit 60) of the 64-bit `F2I`'s `F16` source, forms 1, 5 and 6 (`0x311`, `0xb11`, `0xd11`) | Decoded as [`OpF2I::high`], added: NAK writes it for an `F16` source converted to a 64-bit integer; PR #4588's rows leave it out | as above |
//!
//! Every other row of a decoded form is matched exactly, and the rows above differ from their
//! forms by exactly the bits listed (`decode::tests::masks_match_the_reference_table`, with each
//! row's dropped and added bits; `decode::tests::framework::added_bits_are_the_listed_cases`
//! lists every added bit and the reference that names it). Bit 61, the swizzle's other bit,
//! which NAK sets only for an `F16v2` source (`set_swizzle`: 2 and 3), stays unknown in every
//! form.
//!
//! # Values and bits the decoder refuses
//!
//! A field whose value names nothing is [`DecodeError::UnknownValue`]: `MUFU`'s function 10 to
//! 63 (NAK's `MuFuOp` encodes 0 to 9), the combine operation 3 of `FSET` and `FSETP` (NAK's
//! `PredSetOp` encodes 0 to 2; PR #4588: `Require(set_op <= 2)`), `FMUL`'s scale 0 (PR #4588:
//! `Require(scale >= 1 && scale <= 7, "native FMUL scale")`), and a float size of 0 in the
//! float size fields of `FRND` and `F2F` (75 and 84), `F2I` (84), `I2F` and `I2FP` (75) (see
//! "Size fields"). Each conversion reads its float size fields in ascending bit order, so with
//! both 0 the one at bit 75 is reported (`decode::tests::float32::float_sizes_of_zero_are_refused`,
//! `fmul_scales_decode_and_zero_is_refused`). Their forms' bare words hold these zeros, so the
//! forms' rows give canonical words (scale 4; `F32` in each float size field) from which tests
//! build valid words (`bare_words_refuse_the_zeros_pr_4588_refuses`). Some bits are read only in
//! some of a form's layouts, so set in another they are unknown bits (`decode`'s step 5): the
//! high-half select (bit 60) of `MUFU`, `F2I`, `F2F` and `FRND`, read only for a 16-bit source
//! (`MUFU`'s `.F16`, a conversion's `F16` source type; NAK writes the swizzle only for an `F16`
//! source, and PR #4588 takes bit 60 only for a half source); `I2F`'s byte select (bits
//! 60..62), of which an 8-bit source reads two bits, a 16-bit one bit 60 and a wider one none
//! (PR #4588: `select < (32 >> (3 + src_size))`); and
//! `F2FP`'s `.H1` (bit 72), read only with `.MERGE_C` (NAK: "`MERGE_C` only"). PR #4588's rows
//! leave out `FMNMX`'s `.NAN` and `FSET`'s combine operation in form 6 (`0xc09`, `0xc0a`), so
//! there they are unknown bits (step 2) and the assembler refuses them.

use std::fmt::{self, Display, Formatter};

use super::super::DecodeError;
use super::super::alu::{self, AluForm, Source, Sources};
use super::super::fields::{ALU_FORM, DST, SRC0, SRC2, mask_of};
use super::super::form::{Form, Reader};
use super::super::instruction::Field;
use super::super::operand::{AluSrc, Gpr, Pred, PredSrc, RoundMode};
use super::super::print::{FloatOperand, IntOperand, flag};
use super::control::{COND, COND_NOT, PRED_DST, SOURCE_1_FORMS, SOURCE_1_ONLY};
use super::{Named, Op, shells};
#[cfg(test)]
use {
    super::super::fields::{SLOT_ABS, SLOT_NEG, SLOT_UNIFORM, SRC0_ABS, SRC2_ABS, SRC2_NEG},
    super::super::form::{Namer, ReferenceRow, field_value},
    proptest::strategy::BoxedStrategy,
};

// ---------------------------------------------------------------------------------------------
// Fields several instructions share.

/// The flush-to-zero bit (NAK: `set_bit(80, ftz)` in `OpFAdd`, `OpFFma`, `OpFMnMx`, `OpFMul`,
/// `OpFSet`, `OpFSetP`, `OpFSwzAdd`, `OpF2F`, `OpF2I` and `OpFRnd`).
pub(crate) const FTZ: Field = Field::bit(80);
/// The rounding mode (NAK: `set_rnd_mode(78..80, ..)` in the same ops but `OpFMnMx`, `OpFSet`
/// and `OpFSetP`, and in `OpI2F`).
pub(crate) const RND: Field = Field::new(78, RoundMode::BITS);
/// The saturate bit (NAK: `set_bit(77, saturate)` in `OpFAdd`, `OpFFma` and `OpFMul`).
pub(crate) const SAT: Field = Field::bit(77);
/// The "denorm/zero multiply" bit (NAK: `set_bit(76, dnz)` in `OpFFma` and `OpFMul`).
pub(crate) const DNZ: Field = Field::bit(76);
/// The high-half select of a 16-bit source in the wide slot (NAK: `set_swizzle(60..62, ..)` for
/// an `F16` source, `Xx` 0 and `Yy` 1, from `encode_alu_reg`, `encode_alu_cb` and
/// `encode_alu_ureg`, so in forms 1, 5 and 6 of every op whose `src_types()[0]` is `F16`: `MUFU`,
/// `F2I`, `F2F` and `FRND`). PR #4588's rows mark bit 60 only in `MUFU`'s and the narrow `F2I`'s
/// form 1, and its `MUFU` and `F2I` take only bit 60; this decoder reads it in every form NAK
/// writes it in, the forms adding it where the rows leave it out (module docs).
pub(crate) const HIGH_HALF: Field = Field::bit(60);
/// A conversion's destination size, log2 of its byte count (NAK: `set_field(75..77, (bits /
/// 8).ilog2())` in `OpF2F`, `OpF2I`, `OpI2F` and `OpFRnd`; PR #4588: `Take(75, 2)`, also for
/// `I2FP`).
pub(crate) const CVT_DST_SIZE: Field = Field::new(75, 2);
/// A conversion's source size (NAK: `set_field(84..86, ..)`; PR #4588: `Take(84, 2)`).
pub(crate) const CVT_SRC_SIZE: Field = Field::new(84, 2);
/// The combine operation of `FSET` and `FSETP` (NAK `OpFSetP`: `set_pred_set_op(74..76, ..)`;
/// PR #4588 `FSET` and `FSETP`: `Take(74, 2)`).
pub(crate) const SET_OP: Field = Field::new(74, PredSetOp::BITS);
/// The comparison of `FSET` and `FSETP` (NAK: `set_float_cmp_op(76..80, ..)`).
pub(crate) const CMP_OP: Field = Field::new(76, FloatCmpOp::BITS);

/// The sources of the two-source instructions in forms 1, 4, 5 and 6: 0 and 1.
const SOURCES_0_1: Sources = Sources {
    src0: true,
    src1: true,
    src2: false,
};

/// The 9-bit opcode below the operand form (NAK's `encode_alu_base`: `set_field(0..9, opcode)`).
fn base_opcode(reader: &Reader) -> u64 {
    reader.opcode() & ((1 << ALU_FORM.start) - 1)
}

// ---------------------------------------------------------------------------------------------
// Operand types.

/// A float comparison (NAK's `FloatCmpOp` and `set_float_cmp_op`, which encode 1 to 14; 0 and
/// 15, always false and always true, are Eden's Maxwell `FPCompareOp::F` and `T`, which PR
/// #4588's `FSET` and `FSETP` reuse: `frontend/maxwell/translate/impl/impl.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatCmpOp {
    /// 0: always false.
    False,
    /// 1: ordered less than.
    Lt,
    /// 2: ordered equal.
    Eq,
    /// 3: ordered less or equal.
    Le,
    /// 4: ordered greater than.
    Gt,
    /// 5: ordered not equal.
    Ne,
    /// 6: ordered greater or equal.
    Ge,
    /// 7: neither operand is NaN.
    Num,
    /// 8: either operand is NaN.
    Nan,
    /// 9: unordered or less than.
    LtU,
    /// 10: unordered or equal.
    EqU,
    /// 11: unordered or less or equal.
    LeU,
    /// 12: unordered or greater than.
    GtU,
    /// 13: unordered or not equal.
    NeU,
    /// 14: unordered or greater or equal.
    GeU,
    /// 15: always true.
    True,
}

impl FloatCmpOp {
    /// Every comparison, in encoding order.
    pub const ALL: [Self; 16] = [
        Self::False,
        Self::Lt,
        Self::Eq,
        Self::Le,
        Self::Gt,
        Self::Ne,
        Self::Ge,
        Self::Num,
        Self::Nan,
        Self::LtU,
        Self::EqU,
        Self::LeU,
        Self::GtU,
        Self::NeU,
        Self::GeU,
        Self::True,
    ];
    /// The width of the field (`set_float_cmp_op` asserts a 4-bit range).
    pub const BITS: u32 = 4;

    /// The four-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::False => 0,
            Self::Lt => 1,
            Self::Eq => 2,
            Self::Le => 3,
            Self::Gt => 4,
            Self::Ne => 5,
            Self::Ge => 6,
            Self::Num => 7,
            Self::Nan => 8,
            Self::LtU => 9,
            Self::EqU => 10,
            Self::LeU => 11,
            Self::GtU => 12,
            Self::NeU => 13,
            Self::GeU => 14,
            Self::True => 15,
        }
    }

    /// The comparison an encoding names; only the low [`FloatCmpOp::BITS`] bits are read, and
    /// every value names one.
    pub const fn from_bits(bits: u64) -> Self {
        Self::ALL[(bits & ((1 << Self::BITS) - 1)) as usize]
    }
}

impl Display for FloatCmpOp {
    /// NAK's `FloatCmpOp` printer (`.lt` to `.nan`); `.f` and `.t` for 0 and 15 are guesses.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::False => ".f",
            Self::Lt => ".lt",
            Self::Eq => ".eq",
            Self::Le => ".le",
            Self::Gt => ".gt",
            Self::Ne => ".ne",
            Self::Ge => ".ge",
            Self::Num => ".num",
            Self::Nan => ".nan",
            Self::LtU => ".ltu",
            Self::EqU => ".equ",
            Self::LeU => ".leu",
            Self::GtU => ".gtu",
            Self::NeU => ".neu",
            Self::GeU => ".geu",
            Self::True => ".t",
        })
    }
}

/// How a comparison combines with an accumulator predicate (NAK's `PredSetOp` and
/// `set_pred_set_op`: `And` 0, `Or` 1, `Xor` 2).
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
    /// The width of the field (`set_pred_set_op` asserts a 2-bit range).
    pub const BITS: u32 = 2;

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
    /// NAK's `PredSetOp` printer.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::And => ".and",
            Self::Or => ".or",
            Self::Xor => ".xor",
        })
    }
}

/// A float type, as a conversion's size field encodes it: log2 of its byte count (NAK's
/// `FloatType` and `(bits / 8).ilog2()`). Size 0, one byte, names no type (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatType {
    /// Size 1, 16 bits.
    F16,
    /// Size 2, 32 bits.
    F32,
    /// Size 3, 64 bits.
    F64,
}

impl FloatType {
    /// Every type, in encoding order.
    pub const ALL: [Self; 3] = [Self::F16, Self::F32, Self::F64];

    /// The size field's encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::F16 => 1,
            Self::F32 => 2,
            Self::F64 => 3,
        }
    }

    /// The type a size encoding names: none for 0, which NAK's `FloatType` cannot encode (it
    /// has `F16`, `F32` and `F64`) and every conversion of PR #4588's translator refuses, nor
    /// for a value wider than the two-bit field.
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            1 => Some(Self::F16),
            2 => Some(Self::F32),
            3 => Some(Self::F64),
            _ => None,
        }
    }

    /// The type's width in bits.
    pub const fn width(self) -> u32 {
        match self {
            Self::F16 => 16,
            Self::F32 => 32,
            Self::F64 => 64,
        }
    }
}

impl Display for FloatType {
    /// NAK's `FloatType` printer.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F16 => ".f16",
            Self::F32 => ".f32",
            Self::F64 => ".f64",
        })
    }
}

/// An integer type: a size field (log2 of the byte count) and a signed bit (NAK's `IntType`,
/// `(bits / 8).ilog2()` and `is_signed()`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntType {
    size: u8,
    signed: bool,
}

impl IntType {
    /// The largest size encoding, 3 (a 64-bit type): the conversions' size fields are two bits
    /// wide (`CVT_DST_SIZE`, `CVT_SRC_SIZE`).
    pub const MAX_SIZE: u8 = 3;
    /// `u32`.
    pub const U32: Self = Self::from_field(2, false);
    /// `s32`.
    pub const S32: Self = Self::from_field(2, true);

    /// The type of size encoding `size` (log2 of the byte count) and signedness, or `None` for
    /// a size above [`Self::MAX_SIZE`].
    pub const fn new(size: u8, signed: bool) -> Option<Self> {
        if size > Self::MAX_SIZE {
            None
        } else {
            Some(Self { size, signed })
        }
    }

    /// The type a two-bit size field and a signed bit name: only the low two bits of `size`
    /// are read, so every value of the field names a type.
    pub(crate) const fn from_field(size: u8, signed: bool) -> Self {
        Self {
            size: size & Self::MAX_SIZE,
            signed,
        }
    }

    /// The size field's encoding, 0 to 3.
    pub fn size_bits(self) -> u64 {
        u64::from(self.size)
    }

    /// Whether the type is signed.
    pub const fn is_signed(self) -> bool {
        self.signed
    }

    /// The type's width in bits: 8, 16, 32 or 64.
    pub const fn width(self) -> u32 {
        8 << self.size
    }
}

impl Display for IntType {
    /// `.u32`, `.s32` and the like. NAK's tests pin `.u32`; the `s` of a signed type is
    /// `nvdisasm`'s usual style and a guess (NAK's IR printer writes `.i32`).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let sign = if self.signed { 's' } else { 'u' };
        write!(f, ".{sign}{}", self.width())
    }
}

/// A rounding mode printed as an integer rounding: `.floor`, `.ceil`, `.trunc` (NAK's
/// `test_float_ops` pins `.trunc` for `FRND` and `F2I` toward zero; the other two are guesses).
struct IntRound(RoundMode);

impl Display for IntRound {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self.0 {
            RoundMode::NearestEven => "",
            RoundMode::NegInf => ".floor",
            RoundMode::PosInf => ".ceil",
            RoundMode::Zero => ".trunc",
        })
    }
}

/// Writes a wide-slot source with its high-half select (`.h1`, a guess).
fn half_source(f: &mut Formatter<'_>, src: AluSrc, high: bool) -> fmt::Result {
    write!(f, "{}", FloatOperand(src))?;
    flag(f, high, ".h1")
}

// ---------------------------------------------------------------------------------------------
// FADD.

/// `FADD`'s 9-bit opcode (`impl SM70Op for OpFAdd`: `encode_alu(0x021, ..)`).
pub(crate) const FADD_OPCODE: u16 = 0x021;
/// `FADD`'s operand forms: its second operand in the wide slot, as source 1 when it is a
/// register and as source 2 otherwise (`alu::WIDE_OPERAND_FORMS`).
pub(crate) const FADD_FORMS: [AluForm; 4] = alu::WIDE_OPERAND_FORMS;

/// `FADD dst, a, b`: `a + b` in 32-bit float.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFAdd {
    /// The destination.
    pub dst: Gpr,
    /// `a`, a register, and `b`, in the wide slot.
    pub srcs: [AluSrc; 2],
    /// The register slot at bits 64..72, which `FADD` does not read (module docs). Not printed.
    pub ignored: Gpr,
    /// Clamp the result to [0, 1].
    pub saturate: bool,
    /// The rounding mode.
    pub rnd: RoundMode,
    /// Flush denormal inputs and results to zero.
    pub ftz: bool,
}

impl Named for OpFAdd {
    const MNEMONIC: &'static str = "fadd";
}

impl Display for OpFAdd {
    /// NAK's `test_float_ops`: `fadd`, `.ftz`, the rounding mode, `.sat`, then the operands.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("fadd")?;
        flag(f, self.ftz, ".ftz")?;
        write!(f, "{}", self.rnd)?;
        flag(f, self.saturate, ".sat")?;
        let [a, b] = self.srcs.map(FloatOperand);
        write!(f, " {}, {a}, {b}", self.dst)
    }
}

fn decode_fadd(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::FAdd(OpFAdd {
        dst: reader.gpr(DST),
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, alu::wide_operand_source(form), true),
        ],
        ignored: reader.gpr(SRC2),
        saturate: reader.flag(SAT),
        rnd: RoundMode::from_bits(reader.field(RND)),
        ftz: reader.flag(FTZ),
    }))
}

/// `FADD` in `form`.
const fn fadd(form: AluForm) -> Form {
    let sources = match alu::wide_operand_source(form) {
        Source::One => SOURCES_0_1,
        Source::Two => Sources {
            src0: true,
            src1: false,
            src2: true,
        },
    };
    Form::new(
        form.opcode(FADD_OPCODE),
        mask_of(&[DST, SRC2, SAT, RND, FTZ]) | alu::known(form, sources, true),
        decode_fadd,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// FMUL.

/// `FMUL`'s 9-bit opcode (`impl SM70Op for OpFMul`: `encode_alu(0x020, ..)`).
pub(crate) const FMUL_OPCODE: u16 = 0x020;
/// `FMUL`'s scale (`set_field(84..87, 0x4_u8); // TODO: PDIV`; PR #4588: `Take(84, 3)`).
pub(crate) const FMUL_SCALE: Field = Field::new(84, 3);

/// `FMUL`'s scale of its first operand: a power of two, `2^(code - 4)` (module docs). Code 0
/// names no scale (PR #4588: `Require(scale >= 1 && scale <= 7, "native FMUL scale")`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FMulScale {
    /// 1: divide by 8.
    D8,
    /// 2: divide by 4.
    D4,
    /// 3: divide by 2.
    D2,
    /// 4: no scaling, the value NAK always writes.
    None,
    /// 5: multiply by 2.
    M2,
    /// 6: multiply by 4.
    M4,
    /// 7: multiply by 8.
    M8,
}

impl FMulScale {
    /// Every scale, in encoding order.
    pub const ALL: [Self; 7] = [
        Self::D8,
        Self::D4,
        Self::D2,
        Self::None,
        Self::M2,
        Self::M4,
        Self::M8,
    ];

    /// The three-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::D8 => 1,
            Self::D4 => 2,
            Self::D2 => 3,
            Self::None => 4,
            Self::M2 => 5,
            Self::M4 => 6,
            Self::M8 => 7,
        }
    }

    /// The scale an encoding names: none for 0, which PR #4588 refuses (`Require(scale >= 1 &&
    /// scale <= 7, "native FMUL scale")`), nor for a value wider than the three-bit field.
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            1 => Some(Self::D8),
            2 => Some(Self::D4),
            3 => Some(Self::D2),
            4 => Some(Self::None),
            5 => Some(Self::M2),
            6 => Some(Self::M4),
            7 => Some(Self::M8),
            _ => None,
        }
    }

    /// The power of two the first operand is scaled by (PR #4588's `ScaleFactor`: `code - 4`).
    pub const fn exponent(self) -> i8 {
        match self {
            Self::D8 => -3,
            Self::D4 => -2,
            Self::D2 => -1,
            Self::None => 0,
            Self::M2 => 1,
            Self::M4 => 2,
            Self::M8 => 3,
        }
    }
}

impl Display for FMulScale {
    /// Maxwell's `nvdisasm` names, `.d2` to `.m8` (a guess for SM86); nothing for no scaling.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::D8 => ".d8",
            Self::D4 => ".d4",
            Self::D2 => ".d2",
            Self::None => "",
            Self::M2 => ".m2",
            Self::M4 => ".m4",
            Self::M8 => ".m8",
        })
    }
}

/// `FMUL dst, a, b`: `a * b` in 32-bit float.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFMul {
    /// The destination.
    pub dst: Gpr,
    /// `a`, a register, and `b`, source 1.
    pub srcs: [AluSrc; 2],
    /// The register slot at bits 64..72, which `FMUL` does not read (module docs; NAK writes
    /// `RZ`). Not printed.
    pub ignored: Gpr,
    /// Clamp the result to [0, 1].
    pub saturate: bool,
    /// The rounding mode.
    pub rnd: RoundMode,
    /// Flush denormal inputs and results to zero.
    pub ftz: bool,
    /// NAK's `dnz` bit, printed `.fmz` as for `FFMA`.
    pub dnz: bool,
    /// The scale of `a`.
    pub scale: FMulScale,
}

impl Named for OpFMul {
    const MNEMONIC: &'static str = "fmul";
}

impl Display for OpFMul {
    /// NAK's `test_float_ops` pins `.ftz`, the rounding mode and `.sat` in that order; where
    /// `.fmz` and the scale go is a guess.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("fmul")?;
        flag(f, self.ftz, ".ftz")?;
        flag(f, self.dnz, ".fmz")?;
        write!(f, "{}{}", self.scale, self.rnd)?;
        flag(f, self.saturate, ".sat")?;
        let [a, b] = self.srcs.map(FloatOperand);
        write!(f, " {}, {a}, {b}", self.dst)
    }
}

fn decode_fmul(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let scale_bits = reader.field(FMUL_SCALE);
    let scale = FMulScale::from_bits(scale_bits)
        .ok_or_else(|| reader.unknown_value(FMUL_SCALE, scale_bits))?;
    Ok(Op::FMul(OpFMul {
        dst: reader.gpr(DST),
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, Source::One, true),
        ],
        ignored: reader.gpr(SRC2),
        saturate: reader.flag(SAT),
        rnd: RoundMode::from_bits(reader.field(RND)),
        ftz: reader.flag(FTZ),
        dnz: reader.flag(DNZ),
        scale,
    }))
}

/// `FMUL` in `form`.
const fn fmul(form: AluForm) -> Form {
    Form::new(
        form.opcode(FMUL_OPCODE),
        mask_of(&[DST, SRC2, DNZ, SAT, RND, FTZ, FMUL_SCALE]) | alu::known(form, SOURCES_0_1, true),
        decode_fmul,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// FFMA.

/// `FFMA`'s 9-bit opcode (`impl SM70Op for OpFFma`: `encode_alu(0x023, ..)`).
pub(crate) const FFMA_OPCODE: u16 = 0x023;
/// `FFMA`'s sources: all three.
pub(crate) const FFMA_SOURCES: Sources = Sources {
    src0: true,
    src1: true,
    src2: true,
};

/// `FFMA dst, a, b, c`: `a * b + c` in 32-bit float, fused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFFma {
    /// The destination.
    pub dst: Gpr,
    /// `a`, `b`, `c`. `a` is always a register; at most one of `b` and `c` is in the wide slot
    /// (an immediate, a constant buffer or a uniform register).
    pub srcs: [AluSrc; 3],
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

impl Named for OpFFma {
    const MNEMONIC: &'static str = "ffma";
}

impl Display for OpFFma {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("ffma")?;
        flag(f, self.ftz, ".ftz")?;
        flag(f, self.dnz, ".fmz")?;
        write!(f, "{}", self.rnd)?;
        flag(f, self.saturate, ".sat")?;
        let [a, b, c] = self.srcs.map(FloatOperand);
        write!(f, " {}, {a}, {b}, {c}", self.dst)
    }
}

fn decode_ffma(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::FFma(OpFFma {
        dst: reader.gpr(DST),
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, Source::One, true),
            alu::src(reader, form, Source::Two, true),
        ],
        saturate: reader.flag(SAT),
        rnd: RoundMode::from_bits(reader.field(RND)),
        ftz: reader.flag(FTZ),
        dnz: reader.flag(DNZ),
    }))
}

/// `FFMA` in `form`.
const fn ffma(form: AluForm) -> Form {
    Form::new(
        form.opcode(FFMA_OPCODE),
        mask_of(&[DST, DNZ, SAT, RND, FTZ]) | alu::known(form, FFMA_SOURCES, true),
        decode_ffma,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// FMNMX.

/// `FMNMX`'s 9-bit opcode (`impl SM70Op for OpFMnMx`: `encode_alu(0x009, ..)`).
pub(crate) const FMNMX_OPCODE: u16 = 0x009;
/// `FMNMX`'s `.NAN` bit: a NaN operand makes the result NaN. PR #4588 alone (`Visitor::FMNMX`:
/// `propagate_nan = Take(81, 1)`); NAK does not write it.
pub(crate) const FMNMX_NAN: Field = Field::bit(81);

/// Whether `FMNMX`'s row for `form` has [`FMNMX_NAN`] (PR #4588's `0xc09` does not).
const fn fmnmx_has_nan(form: AluForm) -> bool {
    !matches!(form, AluForm::UniformSrc1)
}

/// `FMNMX dst, a, b, min`: the smaller of `a` and `b` where `min` is true, the larger where it
/// is false.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFMnMx {
    /// The destination.
    pub dst: Gpr,
    /// `a`, a register, and `b`, source 1.
    pub srcs: [AluSrc; 2],
    /// Picks the minimum (NAK: `set_pred_src(87..90, 90, &self.min)`).
    pub min: PredSrc,
    /// Flush denormal inputs to zero.
    pub ftz: bool,
    /// `.NAN`: a NaN operand gives NaN (PR #4588 alone; never with a uniform-register `b`).
    pub nan: bool,
}

impl Named for OpFMnMx {
    const MNEMONIC: &'static str = "fmnmx";
}

impl Display for OpFMnMx {
    /// NAK's `test_float_ops`: `fmnmx`, `.ftz`, the operands and the predicate; `.nan` is a
    /// guess.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("fmnmx")?;
        flag(f, self.ftz, ".ftz")?;
        flag(f, self.nan, ".nan")?;
        let [a, b] = self.srcs.map(FloatOperand);
        write!(f, " {}, {a}, {b}, {}", self.dst, self.min)
    }
}

fn decode_fmnmx(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::FMnMx(OpFMnMx {
        dst: reader.gpr(DST),
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, Source::One, true),
        ],
        min: reader.pred_src(COND, COND_NOT),
        ftz: reader.flag(FTZ),
        nan: fmnmx_has_nan(form) && reader.flag(FMNMX_NAN),
    }))
}

/// `FMNMX` in `form`.
const fn fmnmx(form: AluForm) -> Form {
    let nan = if fmnmx_has_nan(form) {
        FMNMX_NAN.mask()
    } else {
        0
    };
    Form::new(
        form.opcode(FMNMX_OPCODE),
        mask_of(&[DST, FTZ, COND, COND_NOT]) | nan | alu::known(form, SOURCES_0_1, true),
        decode_fmnmx,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// FSET and FSETP.

/// `FSET`'s 9-bit opcode (`impl SM70Op for OpFSet`: `encode_alu(0x00a, ..)`).
pub(crate) const FSET_OPCODE: u16 = 0x00a;
/// `FSETP`'s 9-bit opcode (`impl SM70Op for OpFSetP`: `encode_alu(0x00b, ..)`).
pub(crate) const FSETP_OPCODE: u16 = 0x00b;
/// `FSETP`'s second destination, the inverted comparison combined with the accumulator (NAK:
/// `set_pred_dst(84..87, &Dst::None); // dst1`; PR #4588: `dest_b = Take(84, 3)`).
pub(crate) const FSETP_DST1: Field = Field::new(84, Pred::BITS);

/// Whether `FSET`'s row for `form` has [`SET_OP`] (PR #4588's `0xc0a` does not).
const fn fset_has_set_op(form: AluForm) -> bool {
    !matches!(form, AluForm::UniformSrc1)
}

/// Reads a combine operation, refusing 3.
fn read_set_op(reader: &mut Reader) -> Result<PredSetOp, DecodeError> {
    read_pred_set_op(reader, SET_OP)
}

/// Reads a combine operation in `field`, refusing 3 (NAK's `PredSetOp` encodes 0 to 2; PR
/// #4588: `Require(set_op <= 2)`); shared with the half and double group (`op::half_double`).
pub(crate) fn read_pred_set_op(
    reader: &mut Reader,
    field: Field,
) -> Result<PredSetOp, DecodeError> {
    let bits = reader.field(field);
    PredSetOp::from_bits(bits).ok_or_else(|| reader.unknown_value(field, bits))
}

/// `FSET.BF dst, a, b, accum`: 1.0 where the comparison, combined with `accum`, is true, else
/// 0.0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFSet {
    /// The destination.
    pub dst: Gpr,
    /// The comparison.
    pub cmp: FloatCmpOp,
    /// `a`, a register, and `b`, source 1.
    pub srcs: [AluSrc; 2],
    /// How the comparison combines with `accum` (PR #4588 alone; NAK leaves the field zero,
    /// `And`, and it is always `And` with a uniform-register `b`).
    pub set_op: PredSetOp,
    /// The accumulator (NAK writes `PT`: "TODO: src predicate").
    pub accum: PredSrc,
    /// Flush denormal inputs to zero.
    pub ftz: bool,
}

impl Named for OpFSet {
    const MNEMONIC: &'static str = "fset";
}

impl Display for OpFSet {
    /// NAK's `test_float_ops`: `fset.bf`, the comparison, `.ftz`, the combine operation, then
    /// the operands and the accumulator.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "fset.bf{}", self.cmp)?;
        flag(f, self.ftz, ".ftz")?;
        let [a, b] = self.srcs.map(FloatOperand);
        write!(f, "{} {}, {a}, {b}, {}", self.set_op, self.dst, self.accum)
    }
}

fn decode_fset(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let set_op = if fset_has_set_op(form) {
        read_set_op(reader)?
    } else {
        PredSetOp::And
    };
    Ok(Op::FSet(OpFSet {
        dst: reader.gpr(DST),
        cmp: FloatCmpOp::from_bits(reader.field(CMP_OP)),
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, Source::One, true),
        ],
        set_op,
        accum: reader.pred_src(COND, COND_NOT),
        ftz: reader.flag(FTZ),
    }))
}

/// `FSET` in `form`.
const fn fset(form: AluForm) -> Form {
    let set_op = if fset_has_set_op(form) {
        SET_OP.mask()
    } else {
        0
    };
    Form::new(
        form.opcode(FSET_OPCODE),
        mask_of(&[DST, CMP_OP, FTZ, COND, COND_NOT]) | set_op | alu::known(form, SOURCES_0_1, true),
        decode_fset,
    )
    .with_fixed(alu::fixed(form))
}

/// `FSETP dst, dst1, a, b, accum`: the comparison combined with `accum` into `dst`, and its
/// negation combined with `accum` into `dst1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFSetP {
    /// The destination (NAK: `set_pred_dst(81..84, ..)`); `PT` writes nothing.
    pub dst: Pred,
    /// The second destination (NAK writes `PT`; PR #4588 writes the inverted comparison there).
    pub dst1: Pred,
    /// The comparison.
    pub cmp: FloatCmpOp,
    /// How the comparison combines with `accum`.
    pub set_op: PredSetOp,
    /// `a`, a register, and `b`, source 1.
    pub srcs: [AluSrc; 2],
    /// The accumulator (NAK: `set_pred_src(87..90, 90, &self.accum)`).
    pub accum: PredSrc,
    /// Flush denormal inputs to zero.
    pub ftz: bool,
}

impl Named for OpFSetP {
    const MNEMONIC: &'static str = "fsetp";
}

impl Display for OpFSetP {
    /// NAK's `test_float_ops`: `fsetp`, the comparison, `.ftz`, the combine operation, then both
    /// destinations, the operands and the accumulator.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "fsetp{}", self.cmp)?;
        flag(f, self.ftz, ".ftz")?;
        let [a, b] = self.srcs.map(FloatOperand);
        write!(
            f,
            "{} {}, {}, {a}, {b}, {}",
            self.set_op, self.dst, self.dst1, self.accum
        )
    }
}

fn decode_fsetp(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let set_op = read_set_op(reader)?;
    Ok(Op::FSetP(OpFSetP {
        // Cannot fail for a three-bit field.
        dst: Pred::new(reader.byte(PRED_DST)).unwrap_or(Pred::PT),
        dst1: Pred::new(reader.byte(FSETP_DST1)).unwrap_or(Pred::PT),
        cmp: FloatCmpOp::from_bits(reader.field(CMP_OP)),
        set_op,
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, Source::One, true),
        ],
        accum: reader.pred_src(COND, COND_NOT),
        ftz: reader.flag(FTZ),
    }))
}

/// `FSETP` in `form`.
const fn fsetp(form: AluForm) -> Form {
    Form::new(
        form.opcode(FSETP_OPCODE),
        mask_of(&[SET_OP, CMP_OP, FTZ, PRED_DST, FSETP_DST1, COND, COND_NOT])
            | alu::known(form, SOURCES_0_1, true),
        decode_fsetp,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// FSEL.

/// `FSEL`'s 9-bit opcode, from PR #4588 alone (`visitor.cpp`, `Visitor::Run`: `case 0x008:
/// SEL(true)`; rows `0x208`, `0x808`, `0xa08`).
pub(crate) const FSEL_OPCODE: u16 = 0x008;
/// `FSEL`'s operand forms: PR #4588's table has no row for form 6.
pub(crate) const FSEL_FORMS: [AluForm; 3] = [AluForm::Reg, AluForm::ImmSrc1, AluForm::CBufSrc1];

/// `FSEL dst, a, b, cond`: `a` where `cond` is true, else `b`, with float source modifiers.
/// From PR #4588 alone (`ops_alu.cpp`, `Visitor::SEL`): **needs an `nvdisasm` check**.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFSel {
    /// The destination.
    pub dst: Gpr,
    /// `a`, a register, and `b`, source 1 (not a uniform register: no row has that form).
    pub srcs: [AluSrc; 2],
    /// The selector (PR #4588: `Take(87, 3)`, `Take(90, 1)`).
    pub cond: PredSrc,
}

impl Named for OpFSel {
    const MNEMONIC: &'static str = "fsel";
}

impl Display for OpFSel {
    /// A guess in `FMNMX`'s style.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let [a, b] = self.srcs.map(FloatOperand);
        write!(f, "fsel {}, {a}, {b}, {}", self.dst, self.cond)
    }
}

fn decode_fsel(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::FSel(OpFSel {
        dst: reader.gpr(DST),
        srcs: [
            alu::src0(reader, true),
            alu::src(reader, form, Source::One, true),
        ],
        cond: reader.pred_src(COND, COND_NOT),
    }))
}

/// `FSEL` in `form`.
const fn fsel(form: AluForm) -> Form {
    Form::new(
        form.opcode(FSEL_OPCODE),
        mask_of(&[DST, COND, COND_NOT]) | alu::known(form, SOURCES_0_1, true),
        decode_fsel,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// MUFU.

/// `MUFU`'s 9-bit opcode (`impl SM70Op for OpMuFu`: `encode_alu_base(0x108, ..)`).
pub(crate) const MUFU_OPCODE: u16 = 0x108;
/// `MUFU`'s 16-bit operation type below SM 9.0 (`set_bit(73, self.op_type == FloatType::F16)`).
pub(crate) const MUFU_F16: Field = Field::bit(73);
/// `MUFU`'s function (`set_field(74..80, ..)`).
pub(crate) const MUFU_OP: Field = Field::new(74, 6);

/// A `MUFU` function (NAK's `MuFuOp` and its encoding, 0 to 9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MuFuOp {
    /// 0: cosine.
    Cos,
    /// 1: sine.
    Sin,
    /// 2: base-2 exponential.
    Exp2,
    /// 3: base-2 logarithm.
    Log2,
    /// 4: reciprocal.
    Rcp,
    /// 5: reciprocal square root.
    Rsq,
    /// 6: reciprocal of a double, high word.
    Rcp64H,
    /// 7: reciprocal square root of a double, high word.
    Rsq64H,
    /// 8: square root.
    Sqrt,
    /// 9: hyperbolic tangent (SM 7.5 and later).
    Tanh,
}

impl MuFuOp {
    /// Every function, in encoding order.
    pub const ALL: [Self; 10] = [
        Self::Cos,
        Self::Sin,
        Self::Exp2,
        Self::Log2,
        Self::Rcp,
        Self::Rsq,
        Self::Rcp64H,
        Self::Rsq64H,
        Self::Sqrt,
        Self::Tanh,
    ];

    /// The six-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::Cos => 0,
            Self::Sin => 1,
            Self::Exp2 => 2,
            Self::Log2 => 3,
            Self::Rcp => 4,
            Self::Rsq => 5,
            Self::Rcp64H => 6,
            Self::Rsq64H => 7,
            Self::Sqrt => 8,
            Self::Tanh => 9,
        }
    }

    /// The function an encoding names; 10 and above name none.
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match bits {
            0 => Some(Self::Cos),
            1 => Some(Self::Sin),
            2 => Some(Self::Exp2),
            3 => Some(Self::Log2),
            4 => Some(Self::Rcp),
            5 => Some(Self::Rsq),
            6 => Some(Self::Rcp64H),
            7 => Some(Self::Rsq64H),
            8 => Some(Self::Sqrt),
            9 => Some(Self::Tanh),
            _ => None,
        }
    }
}

impl Display for MuFuOp {
    /// NAK's `test_mufu`: `.ex2` and `.lg2` for the exponential and logarithm, NAK's names for
    /// the rest.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cos => ".cos",
            Self::Sin => ".sin",
            Self::Exp2 => ".ex2",
            Self::Log2 => ".lg2",
            Self::Rcp => ".rcp",
            Self::Rsq => ".rsq",
            Self::Rcp64H => ".rcp64h",
            Self::Rsq64H => ".rsq64h",
            Self::Sqrt => ".sqrt",
            Self::Tanh => ".tanh",
        })
    }
}

/// `MUFU.op dst, src`: a special function of one operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpMuFu {
    /// The destination.
    pub dst: Gpr,
    /// The function.
    pub op: MuFuOp,
    /// The operand, source 1.
    pub src: AluSrc,
    /// The 16-bit operation type (`.f16`).
    pub f16: bool,
    /// For a 16-bit register operand, its high half (NAK's swizzle `Yy`; only with `f16` and a
    /// register source).
    pub high: bool,
}

impl Named for OpMuFu {
    const MNEMONIC: &'static str = "mufu";
}

impl Display for OpMuFu {
    /// NAK's `test_mufu`: `mufu`, the function, `.f16`, the operands; the operand's `.h1` is a
    /// guess.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "mufu{}", self.op)?;
        flag(f, self.f16, ".f16")?;
        write!(f, " {}, ", self.dst)?;
        half_source(f, self.src, self.high)
    }
}

fn decode_mufu(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let bits = reader.field(MUFU_OP);
    let op = MuFuOp::from_bits(bits).ok_or_else(|| reader.unknown_value(MUFU_OP, bits))?;
    let f16 = reader.flag(MUFU_F16);
    Ok(Op::MuFu(OpMuFu {
        dst: reader.gpr(DST),
        op,
        src: alu::src(reader, form, Source::One, true),
        f16,
        high: f16 && has_high_half(form) && reader.flag(HIGH_HALF),
    }))
}

/// Whether a one-source instruction's 16-bit source in `form` has a high-half select: in forms
/// 1, 5 and 6, whose source NAK writes with `encode_alu_reg`, `encode_alu_cb` and
/// `encode_alu_ureg`, each of which calls `set_swizzle(60..62, ..)` for an `F16` source; not in
/// form 4, whose bit 60 is the immediate's.
const fn has_high_half(form: AluForm) -> bool {
    matches!(
        form,
        AluForm::Reg | AluForm::CBufSrc1 | AluForm::UniformSrc1
    )
}

/// The high-half bit if `form` has it ([`has_high_half`]).
const fn high_half_known(form: AluForm) -> u128 {
    if has_high_half(form) {
        HIGH_HALF.mask()
    } else {
        0
    }
}

/// `MUFU` in `form`.
const fn mufu(form: AluForm) -> Form {
    Form::new(
        form.opcode(MUFU_OPCODE),
        mask_of(&[DST, MUFU_F16, MUFU_OP])
            | high_half_known(form)
            | alu::known(form, SOURCE_1_ONLY, true),
        decode_mufu,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// FSWZADD.

/// `FSWZADD`'s 12-bit opcode (`impl SM70Op for OpFSwzAdd`: `set_opcode(0x822)`).
pub(crate) const FSWZADD_OPCODE: u16 = 0x822;
/// `FSWZADD`'s second source (`set_reg_src(64..72, ..)`).
pub(crate) const FSWZADD_SRC1: Field = SRC2;
/// `FSWZADD`'s `.NDV` bit (`set_tex_ndv(77, ..)`).
pub(crate) const FSWZADD_NDV: Field = Field::bit(77);
/// The width of one lane's operation in `FSWZADD`'s operation byte (bits 32..40).
pub(crate) const FSWZADD_OP_BITS: u32 = 2;
/// The start of `FSWZADD`'s operation byte (`set_field(32..40, subop)`).
const FSWZADD_OPS_START: u32 = 32;

/// The field of lane `lane`'s operation, 0 to 3: NAK puts lane 0 in the top two bits of the byte
/// (`subop |= op << ((ops.len() - i - 1) * 2)`).
pub(crate) const fn fswzadd_op(lane: u32) -> Field {
    let lanes = OpFSwzAdd::LANES;
    let index = (lanes - 1).saturating_sub(lane);
    Field::new(FSWZADD_OPS_START + index * FSWZADD_OP_BITS, FSWZADD_OP_BITS)
}

/// One lane's operation in `FSWZADD` (NAK's `FSwzAddOp`, encoded `Add` 0, `SubLeft` 1,
/// `SubRight` 2, `MoveLeft` 3).
///
/// The references settle only the names and codes. The arithmetic each variant documents is a
/// **guess** from the names: NAK's `ir.rs` gives no semantics, and PR #4588 passes the inverted
/// operation byte (`~subop`) to Maxwell's `FSwizzleAdd`, whose sign table differs. An `nvdisasm`
/// or hardware check would confirm it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FSwzAddOp {
    /// 0: `a + b` (a guess).
    Add,
    /// 1: `a - b` (a guess).
    SubLeft,
    /// 2: `b - a` (a guess).
    SubRight,
    /// 3: `b` (a guess).
    MoveLeft,
}

impl FSwzAddOp {
    /// Every operation, in encoding order.
    pub const ALL: [Self; 4] = [Self::Add, Self::SubLeft, Self::SubRight, Self::MoveLeft];

    /// The two-bit encoding.
    pub const fn bits(self) -> u64 {
        match self {
            Self::Add => 0,
            Self::SubLeft => 1,
            Self::SubRight => 2,
            Self::MoveLeft => 3,
        }
    }

    /// The operation an encoding names; only the low two bits are read.
    pub const fn from_bits(bits: u64) -> Self {
        Self::ALL[(bits & 3) as usize]
    }
}

impl Display for FSwzAddOp {
    /// NAK's `FSwzAddOp` printer.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Add => "add",
            Self::SubLeft => "sub",
            Self::SubRight => "subr",
            Self::MoveLeft => "mov2",
        })
    }
}

/// `FSWZADD dst, a, b, [ops]`: per lane of each quad, the lane's operation on `a` and `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFSwzAdd {
    /// The destination.
    pub dst: Gpr,
    /// `a` (`set_reg_src(24..32, ..)`) and `b`.
    pub srcs: [Gpr; 2],
    /// Each quad lane's operation, lane 0 first.
    pub ops: [FSwzAddOp; 4],
    /// The rounding mode.
    pub rnd: RoundMode,
    /// Flush denormal inputs and results to zero.
    pub ftz: bool,
    /// `.NDV`: a non-divergent quad assumed (NAK's `TexDerivMode::NonDivergent`).
    pub ndv: bool,
}

impl OpFSwzAdd {
    /// How many lanes a quad has.
    pub const LANES: u32 = 4;
}

impl Named for OpFSwzAdd {
    const MNEMONIC: &'static str = "fswzadd";
}

impl Display for OpFSwzAdd {
    /// NAK's IR text with `nvdisasm`'s modifier order; a guess.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("fswzadd")?;
        flag(f, self.ftz, ".ftz")?;
        write!(f, "{}", self.rnd)?;
        flag(f, self.ndv, ".ndv")?;
        let [a, b] = self.srcs;
        let [o0, o1, o2, o3] = self.ops;
        write!(f, " {}, {a}, {b}, [{o0}, {o1}, {o2}, {o3}]", self.dst)
    }
}

fn decode_fswzadd(reader: &mut Reader) -> Op {
    let mut ops = [FSwzAddOp::Add; 4];
    for (lane, op) in (0..).zip(ops.iter_mut()) {
        *op = FSwzAddOp::from_bits(reader.field(fswzadd_op(lane)));
    }
    Op::FSwzAdd(OpFSwzAdd {
        dst: reader.gpr(DST),
        srcs: [reader.gpr(SRC0), reader.gpr(FSWZADD_SRC1)],
        ops,
        rnd: RoundMode::from_bits(reader.field(RND)),
        ftz: reader.flag(FTZ),
        ndv: reader.flag(FSWZADD_NDV),
    })
}

/// The bits of `FSWZADD`'s four operations.
const FSWZADD_OPS: Field = Field::new(FSWZADD_OPS_START, OpFSwzAdd::LANES * FSWZADD_OP_BITS);

// ---------------------------------------------------------------------------------------------
// Conversions: FRND, F2F, F2I, I2F, I2FP.

/// `FRND`'s 9-bit opcode for 32-bit types and below (`impl SM70Op for OpFRnd`: `0x107`).
pub(crate) const FRND_OPCODE: u16 = 0x107;
/// `FRND`'s 9-bit opcode with a 64-bit type (`0x113`).
pub(crate) const FRND_WIDE_OPCODE: u16 = 0x113;
/// `F2F`'s 9-bit opcode for 32-bit types and below (`impl SM70Op for OpF2F`: `0x104`).
pub(crate) const F2F_OPCODE: u16 = 0x104;
/// `F2F`'s 9-bit opcode with a 64-bit type (`0x110`).
pub(crate) const F2F_WIDE_OPCODE: u16 = 0x110;
/// `F2I`'s 9-bit opcode for 32-bit types and below (`impl SM70Op for OpF2I`: `0x105`).
pub(crate) const F2I_OPCODE: u16 = 0x105;
/// `F2I`'s 9-bit opcode with a 64-bit type (`0x111`).
pub(crate) const F2I_WIDE_OPCODE: u16 = 0x111;
/// `I2F`'s 9-bit opcode for 32-bit types and below (`impl SM70Op for OpI2F`: `0x106`).
pub(crate) const I2F_OPCODE: u16 = 0x106;
/// `I2F`'s 9-bit opcode with a 64-bit type (`0x112`).
pub(crate) const I2F_WIDE_OPCODE: u16 = 0x112;
/// `I2FP`'s 9-bit opcode, from PR #4588 alone (`visitor.cpp`: `case 0x106: case 0x045:
/// I2F()`; `classify.h`: "0x045: I2FP").
pub(crate) const I2FP_OPCODE: u16 = 0x045;
/// `F2I`'s signed destination (`set_bit(72, self.dst_type.is_signed())`).
pub(crate) const F2I_SIGNED: Field = Field::bit(72);
/// `F2I`'s `.NTZ` (`set_bit(77, false); // NTZ`; PR #4588: `Ignore(77, 1)`).
pub(crate) const F2I_NTZ: Field = Field::bit(77);
/// `I2F`'s and `I2FP`'s signed source (`set_bit(74, self.src_type.is_signed())`).
pub(crate) const I2F_SIGNED: Field = Field::bit(74);
/// `I2F`'s byte select (`set_field(60..62, 0_u8); // TODO: subop`; PR #4588: "Bits 60..61 pick
/// which byte or half of the register is converted").
pub(crate) const I2F_SELECT: Field = Field::new(60, 2);

/// The bits of `I2F`'s byte select a source of `src_type` reads: two for an 8-bit source (four
/// bytes), one for a 16-bit one (two halves), none for a wider one (PR #4588:
/// `select < (32U >> (3 + src_size))` parts, so log2 of that many bits).
pub(crate) const fn i2f_select(src_type: IntType) -> Field {
    let parts_log2 = match src_type.size {
        0 => 2,
        1 => 1,
        _ => 0,
    };
    Field::new(I2F_SELECT.start, parts_log2)
}

/// Whether the word's 9-bit opcode is `wide`, the 64-bit opcode of a conversion.
fn is_wide(reader: &Reader, wide: u16) -> bool {
    base_opcode(reader) == u64::from(wide)
}

/// Reads a float type from a size field; size 0 names none ([`FloatType::from_bits`]) and is
/// [`DecodeError::UnknownValue`].
fn float_type(reader: &mut Reader, field: Field) -> Result<FloatType, DecodeError> {
    let bits = reader.field(field);
    FloatType::from_bits(bits).ok_or_else(|| reader.unknown_value(field, bits))
}

/// Reads the high-half select of a conversion's source of `src_type` in `form`: only for an
/// `F16` source in a form that has the bit ([`has_high_half`]).
fn conversion_high(reader: &mut Reader, form: AluForm, src_type: FloatType) -> bool {
    src_type == FloatType::F16 && has_high_half(form) && reader.flag(HIGH_HALF)
}

/// Reads an integer type from a size field and a signed bit.
fn int_type(reader: &mut Reader, size: Field, signed: Field) -> IntType {
    IntType::from_field(reader.byte(size), reader.flag(signed))
}

/// The bits of a one-source conversion in `form`: its destination, source 1 (with its modifier
/// bits if `modifiers`) and `fields`.
const fn conversion_known(form: AluForm, modifiers: bool, fields: u128) -> u128 {
    mask_of(&[DST]) | alu::known(form, SOURCE_1_ONLY, modifiers) | fields
}

/// `FRND dst, src`: `src` rounded to an integral value by the rounding mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpFRnd {
    /// The destination.
    pub dst: Gpr,
    /// The operand, source 1.
    pub src: AluSrc,
    /// The destination type.
    pub dst_type: FloatType,
    /// The source type.
    pub src_type: FloatType,
    /// The rounding mode, printed as an integer rounding.
    pub rnd: RoundMode,
    /// Flush denormal inputs and results to zero.
    pub ftz: bool,
    /// For a 16-bit operand other than an immediate, its high half (NAK's swizzle `Yy`; only
    /// with an `F16` source).
    pub high: bool,
    /// The 64-bit opcode (module docs).
    pub wide: bool,
}

impl Named for OpFRnd {
    const MNEMONIC: &'static str = "frnd";
}

impl Display for OpFRnd {
    /// NAK's `test_float_ops`: `frnd`, `.ftz`, `.f64` when both types are 64-bit and nothing
    /// when both are 32-bit, then the integer rounding. Both types for other pairs, and the
    /// operand's `.h1`, are a guess.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("frnd")?;
        flag(f, self.ftz, ".ftz")?;
        if self.dst_type != self.src_type {
            write!(f, "{}{}", self.dst_type, self.src_type)?;
        } else if self.dst_type != FloatType::F32 {
            write!(f, "{}", self.dst_type)?;
        }
        write!(f, "{} {}, ", IntRound(self.rnd), self.dst)?;
        half_source(f, self.src, self.high)
    }
}

fn decode_frnd(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let dst_type = float_type(reader, CVT_DST_SIZE)?;
    let src_type = float_type(reader, CVT_SRC_SIZE)?;
    Ok(Op::FRnd(OpFRnd {
        dst: reader.gpr(DST),
        src: alu::src(reader, form, Source::One, true),
        dst_type,
        src_type,
        rnd: RoundMode::from_bits(reader.field(RND)),
        ftz: reader.flag(FTZ),
        high: conversion_high(reader, form, src_type),
        wide: is_wide(reader, FRND_WIDE_OPCODE),
    }))
}

/// The fields of `FRND` and `F2F` in `form`: both float types, the rounding mode, `.FTZ` and
/// the high-half select where the form has it.
const fn float_conversion_fields(form: AluForm) -> u128 {
    mask_of(&[CVT_DST_SIZE, RND, FTZ, CVT_SRC_SIZE]) | high_half_known(form)
}

/// `FRND` with 9-bit opcode `base` in `form`.
const fn frnd(base: u16, form: AluForm) -> Form {
    Form::new(
        form.opcode(base),
        conversion_known(form, true, float_conversion_fields(form)),
        decode_frnd,
    )
    .with_fixed(alu::fixed(form))
}

/// `F2F dst, src`: a float converted to another float type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpF2F {
    /// The destination.
    pub dst: Gpr,
    /// The operand, source 1.
    pub src: AluSrc,
    /// The destination type.
    pub dst_type: FloatType,
    /// The source type.
    pub src_type: FloatType,
    /// The rounding mode.
    pub rnd: RoundMode,
    /// Flush denormal inputs and results to zero.
    pub ftz: bool,
    /// For a 16-bit operand other than an immediate, its high half (NAK's swizzle `Yy`; only
    /// with an `F16` source).
    pub high: bool,
    /// The 64-bit opcode (module docs).
    pub wide: bool,
}

impl Named for OpF2F {
    const MNEMONIC: &'static str = "f2f";
}

impl Display for OpF2F {
    /// NAK's `test_float_ops`: `f2f`, `.ftz`, both types, the rounding mode; the operand's
    /// `.h1` is a guess.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("f2f")?;
        flag(f, self.ftz, ".ftz")?;
        write!(
            f,
            "{}{}{} {}, ",
            self.dst_type, self.src_type, self.rnd, self.dst
        )?;
        half_source(f, self.src, self.high)
    }
}

fn decode_f2f(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let dst_type = float_type(reader, CVT_DST_SIZE)?;
    let src_type = float_type(reader, CVT_SRC_SIZE)?;
    Ok(Op::F2F(OpF2F {
        dst: reader.gpr(DST),
        src: alu::src(reader, form, Source::One, true),
        dst_type,
        src_type,
        rnd: RoundMode::from_bits(reader.field(RND)),
        ftz: reader.flag(FTZ),
        high: conversion_high(reader, form, src_type),
        wide: is_wide(reader, F2F_WIDE_OPCODE),
    }))
}

/// `F2F` with 9-bit opcode `base` in `form`.
const fn f2f(base: u16, form: AluForm) -> Form {
    Form::new(
        form.opcode(base),
        conversion_known(form, true, float_conversion_fields(form)),
        decode_f2f,
    )
    .with_fixed(alu::fixed(form))
}

/// `F2I dst, src`: a float rounded to an integer type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag is an independent one-bit field of the encoding, as in NAK's OpF2I"
)]
pub struct OpF2I {
    /// The destination.
    pub dst: Gpr,
    /// The operand, source 1.
    pub src: AluSrc,
    /// The destination type.
    pub dst_type: IntType,
    /// The source type.
    pub src_type: FloatType,
    /// The rounding mode, printed as an integer rounding.
    pub rnd: RoundMode,
    /// Flush denormal inputs to zero.
    pub ftz: bool,
    /// `.NTZ` (NAK always writes it clear).
    pub ntz: bool,
    /// For a 16-bit operand other than an immediate, its high half (NAK's swizzle `Yy`; only
    /// with an `F16` source).
    pub high: bool,
    /// The 64-bit opcode (module docs).
    pub wide: bool,
}

impl Named for OpF2I {
    const MNEMONIC: &'static str = "f2i";
}

impl Display for OpF2I {
    /// NAK's `test_float_ops`: `f2i`, `.ftz`, both types, the integer rounding. Leaving out the
    /// default types (`.s32` and `.f32`) and `.ntz` after the rounding are guesses.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("f2i")?;
        flag(f, self.ftz, ".ftz")?;
        if self.dst_type != IntType::S32 {
            write!(f, "{}", self.dst_type)?;
        }
        if self.src_type != FloatType::F32 {
            write!(f, "{}", self.src_type)?;
        }
        write!(f, "{}", IntRound(self.rnd))?;
        flag(f, self.ntz, ".ntz")?;
        write!(f, " {}, ", self.dst)?;
        half_source(f, self.src, self.high)
    }
}

fn decode_f2i(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let wide = is_wide(reader, F2I_WIDE_OPCODE);
    let src_type = float_type(reader, CVT_SRC_SIZE)?;
    Ok(Op::F2I(OpF2I {
        dst: reader.gpr(DST),
        src: alu::src(reader, form, Source::One, true),
        dst_type: int_type(reader, CVT_DST_SIZE, F2I_SIGNED),
        src_type,
        rnd: RoundMode::from_bits(reader.field(RND)),
        ftz: reader.flag(FTZ),
        ntz: reader.flag(F2I_NTZ),
        high: conversion_high(reader, form, src_type),
        wide,
    }))
}

/// `F2I` with 9-bit opcode `base` in `form`: both opcodes have the high-half bit in the forms
/// that have it ([`has_high_half`]).
const fn f2i(base: u16, form: AluForm) -> Form {
    Form::new(
        form.opcode(base),
        conversion_known(
            form,
            true,
            mask_of(&[F2I_SIGNED, CVT_DST_SIZE, F2I_NTZ, RND, FTZ, CVT_SRC_SIZE])
                | high_half_known(form),
        ),
        decode_f2i,
    )
    .with_fixed(alu::fixed(form))
}

/// Writes an integer source with `I2F`'s byte select of `src_type` (`.b1` for a byte, `.h1` for
/// a half; a guess).
fn select_source(f: &mut Formatter<'_>, src: AluSrc, src_type: IntType, select: u8) -> fmt::Result {
    write!(f, "{}", IntOperand(src))?;
    if select == 0 {
        return Ok(());
    }
    let part = if src_type.size == 0 { 'b' } else { 'h' };
    write!(f, ".{part}{select}")
}

/// `I2F dst, src`: an integer converted to a float type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpI2F {
    /// The destination.
    pub dst: Gpr,
    /// The operand, source 1, never modified.
    pub src: AluSrc,
    /// The destination type.
    pub dst_type: FloatType,
    /// The source type.
    pub src_type: IntType,
    /// The rounding mode.
    pub rnd: RoundMode,
    /// Which byte (8-bit source) or half (16-bit source) of the operand is converted; zero for
    /// a wider source or an immediate (field [`i2f_select`]).
    pub select: u8,
    /// The 64-bit opcode (module docs).
    pub wide: bool,
}

impl Named for OpI2F {
    const MNEMONIC: &'static str = "i2f";
}

impl Display for OpI2F {
    /// NAK's `test_float_ops`: `i2f`, both types, the rounding mode. Leaving out the default
    /// types (`.f32` and `.s32`) and the select's text are guesses.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("i2f")?;
        if self.dst_type != FloatType::F32 {
            write!(f, "{}", self.dst_type)?;
        }
        if self.src_type != IntType::S32 {
            write!(f, "{}", self.src_type)?;
        }
        write!(f, "{} {}, ", self.rnd, self.dst)?;
        select_source(f, self.src, self.src_type, self.select)
    }
}

/// Whether `I2F`'s `form` has the byte select (every source-1 form but the immediate's).
const fn i2f_has_select(form: AluForm) -> bool {
    !matches!(form, AluForm::ImmSrc1)
}

fn decode_i2f(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let src_type = int_type(reader, CVT_SRC_SIZE, I2F_SIGNED);
    let select = if i2f_has_select(form) {
        reader.byte(i2f_select(src_type))
    } else {
        0
    };
    Ok(Op::I2F(OpI2F {
        dst: reader.gpr(DST),
        src: alu::src(reader, form, Source::One, false),
        dst_type: float_type(reader, CVT_DST_SIZE)?,
        src_type,
        rnd: RoundMode::from_bits(reader.field(RND)),
        select,
        wide: is_wide(reader, I2F_WIDE_OPCODE),
    }))
}

/// `I2F` with 9-bit opcode `base` in `form`.
const fn i2f(base: u16, form: AluForm) -> Form {
    let select = if i2f_has_select(form) {
        I2F_SELECT.mask()
    } else {
        0
    };
    Form::new(
        form.opcode(base),
        conversion_known(
            form,
            false,
            mask_of(&[I2F_SIGNED, CVT_DST_SIZE, RND, CVT_SRC_SIZE]) | select,
        ),
        decode_i2f,
    )
    .with_fixed(alu::fixed(form))
}

/// `I2FP dst, src`: an integer converted to a float type. From PR #4588 alone (`ops_convert.cpp`,
/// `Visitor::I2F`, the same fields as `I2F` without the byte select, which its rows lack):
/// **needs an `nvdisasm` check**.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpI2FP {
    /// The destination.
    pub dst: Gpr,
    /// The operand, source 1, never modified.
    pub src: AluSrc,
    /// The destination type (PR #4588 converts only to `F32`).
    pub dst_type: FloatType,
    /// The source type.
    pub src_type: IntType,
    /// The rounding mode.
    pub rnd: RoundMode,
}

impl Named for OpI2FP {
    const MNEMONIC: &'static str = "i2fp";
}

impl Display for OpI2FP {
    /// `i2fp`, both types, the rounding mode: a guess.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "i2fp{}{}{} {}, {}",
            self.dst_type,
            self.src_type,
            self.rnd,
            self.dst,
            IntOperand(self.src)
        )
    }
}

fn decode_i2fp(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    Ok(Op::I2FP(OpI2FP {
        dst: reader.gpr(DST),
        src: alu::src(reader, form, Source::One, false),
        dst_type: float_type(reader, CVT_DST_SIZE)?,
        src_type: int_type(reader, CVT_SRC_SIZE, I2F_SIGNED),
        rnd: RoundMode::from_bits(reader.field(RND)),
    }))
}

/// `I2FP` in `form`.
const fn i2fp(form: AluForm) -> Form {
    Form::new(
        form.opcode(I2FP_OPCODE),
        conversion_known(
            form,
            false,
            mask_of(&[I2F_SIGNED, CVT_DST_SIZE, RND, CVT_SRC_SIZE]),
        ),
        decode_i2fp,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// F2FP.

/// `F2FP`'s 9-bit opcode (`impl SM70Op for OpF2FP`: `encode_alu(0x03e, ..)`).
pub(crate) const F2FP_OPCODE: u16 = 0x03e;
/// `F2FP`'s third source, read with `.MERGE_C` (NAK: `Src::ZERO` as source 2, at 64..72; PR
/// #4588: `Source(2, false)` when `merge`).
pub(crate) const F2FP_C: Field = SRC2;
/// `F2FP`'s `.H1`, "`MERGE_C` only" (NAK: `set_bit(72, false)`).
pub(crate) const F2FP_H1: Field = Field::bit(72);
/// `F2FP`'s `.MERGE_C` (NAK: `set_bit(78, false); // TODO: .MERGE_C`; PR #4588: `Take(78, 1)`).
pub(crate) const F2FP_MERGE_C: Field = Field::bit(78);
/// `F2FP`'s rounding mode (NAK: `set_rnd_mode(79..81, ..)`).
pub(crate) const F2FP_RND: Field = Field::new(79, RoundMode::BITS);
/// `F2FP`'s sources: 0 and 1 (NAK writes `Src::ZERO` as source 2; it is [`F2FP_C`] here).
const F2FP_SOURCES: Sources = SOURCES_0_1;

/// `F2FP.PACK_AB dst, a, b`: `a` and `b` converted to 16-bit floats and packed, `a` in the high
/// half. With `.MERGE_C`, the high half is the low (or with `.H1`, high) half of `c` instead
/// (NAK's comment on `.MERGE_C`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpF2FP {
    /// The destination.
    pub dst: Gpr,
    /// `a`, a register, and `b`, source 1; never modified.
    pub srcs: [AluSrc; 2],
    /// The register at bits 64..72: the `.MERGE_C` source, unread otherwise (NAK writes `RZ`).
    /// Printed only with `merge_c`.
    pub c: Gpr,
    /// `.MERGE_C`.
    pub merge_c: bool,
    /// `.H1`: the high half of `c` (only with `merge_c`).
    pub h1: bool,
    /// The rounding mode.
    pub rnd: RoundMode,
}

impl Named for OpF2FP {
    const MNEMONIC: &'static str = "f2fp";
}

impl Display for OpF2FP {
    /// NAK's `test_f2fp` on SM 8.6: `f2fp.pack_ab`, the rounding mode, the operands. `.merge_c`,
    /// `.h1` and the fourth operand are guesses.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("f2fp.pack_ab")?;
        flag(f, self.merge_c, ".merge_c")?;
        flag(f, self.h1, ".h1")?;
        let [a, b] = self.srcs.map(FloatOperand);
        write!(f, "{} {}, {a}, {b}", self.rnd, self.dst)?;
        if self.merge_c {
            write!(f, ", {}", self.c)?;
        }
        Ok(())
    }
}

fn decode_f2fp(reader: &mut Reader) -> Result<Op, DecodeError> {
    let form = alu::form(reader)?;
    let merge_c = reader.flag(F2FP_MERGE_C);
    Ok(Op::F2FP(OpF2FP {
        dst: reader.gpr(DST),
        srcs: [
            alu::src0(reader, false),
            alu::src(reader, form, Source::One, false),
        ],
        c: reader.gpr(F2FP_C),
        merge_c,
        h1: merge_c && reader.flag(F2FP_H1),
        rnd: RoundMode::from_bits(reader.field(F2FP_RND)),
    }))
}

/// `F2FP` in `form`.
const fn f2fp(form: AluForm) -> Form {
    Form::new(
        form.opcode(F2FP_OPCODE),
        mask_of(&[DST, F2FP_C, F2FP_H1, F2FP_MERGE_C, F2FP_RND])
            | alu::known(form, F2FP_SOURCES, false),
        decode_f2fp,
    )
    .with_fixed(alu::fixed(form))
}

// ---------------------------------------------------------------------------------------------
// The form table.

/// `FFMA`'s, `FMUL`'s and the source-1 instructions' forms, by index.
const FORM_1: AluForm = SOURCE_1_FORMS[0];
const FORM_4: AluForm = SOURCE_1_FORMS[1];
const FORM_5: AluForm = SOURCE_1_FORMS[2];
const FORM_6: AluForm = SOURCE_1_FORMS[3];

/// The 32-bit float group's forms.
pub(crate) const FORMS: &[Form] = &[
    ffma(AluForm::ALL[0]),
    ffma(AluForm::ALL[1]),
    ffma(AluForm::ALL[2]),
    ffma(AluForm::ALL[3]),
    ffma(AluForm::ALL[4]),
    ffma(AluForm::ALL[5]),
    ffma(AluForm::ALL[6]),
    fadd(FADD_FORMS[0]),
    fadd(FADD_FORMS[1]),
    fadd(FADD_FORMS[2]),
    fadd(FADD_FORMS[3]),
    fmul(FORM_1),
    fmul(FORM_4),
    fmul(FORM_5),
    fmul(FORM_6),
    fmnmx(FORM_1),
    fmnmx(FORM_4),
    fmnmx(FORM_5),
    fmnmx(FORM_6),
    fset(FORM_1),
    fset(FORM_4),
    fset(FORM_5),
    fset(FORM_6),
    fsetp(FORM_1),
    fsetp(FORM_4),
    fsetp(FORM_5),
    fsetp(FORM_6),
    fsel(FSEL_FORMS[0]),
    fsel(FSEL_FORMS[1]),
    fsel(FSEL_FORMS[2]),
    mufu(FORM_1),
    mufu(FORM_4),
    mufu(FORM_5),
    mufu(FORM_6),
    Form::new(
        FSWZADD_OPCODE,
        mask_of(&[DST, SRC0, FSWZADD_OPS, FSWZADD_SRC1, FSWZADD_NDV, RND, FTZ]),
        |reader| Ok(decode_fswzadd(reader)),
    ),
    frnd(FRND_OPCODE, FORM_1),
    frnd(FRND_OPCODE, FORM_4),
    frnd(FRND_OPCODE, FORM_5),
    frnd(FRND_OPCODE, FORM_6),
    frnd(FRND_WIDE_OPCODE, FORM_1),
    frnd(FRND_WIDE_OPCODE, FORM_4),
    frnd(FRND_WIDE_OPCODE, FORM_5),
    frnd(FRND_WIDE_OPCODE, FORM_6),
    f2f(F2F_OPCODE, FORM_1),
    f2f(F2F_OPCODE, FORM_4),
    f2f(F2F_OPCODE, FORM_5),
    f2f(F2F_OPCODE, FORM_6),
    f2f(F2F_WIDE_OPCODE, FORM_1),
    f2f(F2F_WIDE_OPCODE, FORM_4),
    f2f(F2F_WIDE_OPCODE, FORM_5),
    f2f(F2F_WIDE_OPCODE, FORM_6),
    f2i(F2I_OPCODE, FORM_1),
    f2i(F2I_OPCODE, FORM_4),
    f2i(F2I_OPCODE, FORM_5),
    f2i(F2I_OPCODE, FORM_6),
    f2i(F2I_WIDE_OPCODE, FORM_1),
    f2i(F2I_WIDE_OPCODE, FORM_4),
    f2i(F2I_WIDE_OPCODE, FORM_5),
    f2i(F2I_WIDE_OPCODE, FORM_6),
    i2f(I2F_OPCODE, FORM_1),
    i2f(I2F_OPCODE, FORM_4),
    i2f(I2F_OPCODE, FORM_5),
    i2f(I2F_OPCODE, FORM_6),
    i2f(I2F_WIDE_OPCODE, FORM_1),
    i2f(I2F_WIDE_OPCODE, FORM_4),
    i2f(I2F_WIDE_OPCODE, FORM_5),
    i2f(I2F_WIDE_OPCODE, FORM_6),
    i2fp(FORM_1),
    i2fp(FORM_4),
    i2fp(FORM_5),
    i2fp(FORM_6),
    f2fp(FORM_1),
    f2fp(FORM_4),
    f2fp(FORM_5),
    f2fp(FORM_6),
];

// ---------------------------------------------------------------------------------------------
// Test hooks.

/// The bits PR #4588's `FADD` and `FMUL` rows mark known that the forms reject: the modifiers of
/// the register slot the operation does not read.
#[cfg(test)]
pub(crate) const IGNORED_SLOT_DROPPED: u128 = mask_of(&[SRC2_ABS, SRC2_NEG]);
/// The bits PR #4588's `FMNMX` rows mark known that the forms reject: NAK's SM 9.0 `.IS_A` (65)
/// and its commented-out predicate destination (66..69).
#[cfg(test)]
pub(crate) const FMNMX_DROPPED: u128 = Field::new(65, 4).mask();
/// The wide slot's modifier bits, which integer sources reject.
#[cfg(test)]
pub(crate) const SLOT_MODIFIERS: u128 = mask_of(&[SLOT_ABS, SLOT_NEG]);
/// The bits PR #4588's `F2FP` rows mark known besides the wide slot's that the forms reject:
/// source 0's absolute value and source 2's modifiers.
#[cfg(test)]
pub(crate) const F2FP_DROPPED: u128 = mask_of(&[SRC0_ABS, SRC2_ABS, SRC2_NEG]);
/// The bit that the forms of `F2F`, `FRND`, `F2I` and `MUFU` add where PR #4588's rows leave it
/// out: the high-half select of a 16-bit source, which NAK's encoder writes in forms 1, 5 and 6
/// ([`has_high_half`]; module docs).
#[cfg(test)]
pub(crate) const NAK_HIGH_HALF: u128 = HIGH_HALF.mask();
/// The canonical fields of `FMUL`'s forms: scale 4, no scaling (NAK's `set_field(84..87,
/// 0x4_u8)`), since the bare word's scale 0 is refused.
#[cfg(test)]
pub(crate) const FMUL_CANONICAL: u128 = field_value(FMUL_SCALE, FMulScale::None.bits());
/// The canonical destination type of a conversion: `F32` (NAK writes `(32 / 8).ilog2()`), since
/// the bare word's size 0 is refused.
#[cfg(test)]
pub(crate) const CVT_DST_F32: u128 = field_value(CVT_DST_SIZE, FloatType::F32.bits());
/// The canonical source type of a conversion: `F32`, as for [`CVT_DST_F32`].
#[cfg(test)]
pub(crate) const CVT_SRC_F32: u128 = field_value(CVT_SRC_SIZE, FloatType::F32.bits());

/// PR #4588's rows for the group's forms, in [`FORMS`] order
/// (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97; each
/// row's four words, low word first), with the bits each form drops or requires (module docs).
#[cfg(test)]
pub(crate) const REFERENCE_ROWS: &[ReferenceRow] = &[
    // FFMA.
    ReferenceRow::new(0x223, [0xffff_ffff, 0xc000_00ff, 0x0001_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x423, [0xffff_ffff, 0xffff_ffff, 0x0001_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x623, [0xffff_ffff, 0xc7ff_ffff, 0x0801_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x823, [0xffff_ffff, 0xffff_ffff, 0x0001_ffff, 0xffff_fe00]),
    ReferenceRow::new(0xa23, [0xffff_ffff, 0xc7ff_ffff, 0x0801_ffff, 0xffff_fe00]),
    ReferenceRow::new(0xc23, [0xffff_ffff, 0xc000_003f, 0x0801_ffff, 0xffff_fe00])
        .fixing(SLOT_UNIFORM.mask()),
    ReferenceRow::new(0xe23, [0xffff_ffff, 0xc000_003f, 0x0801_ffff, 0xffff_fe00])
        .fixing(SLOT_UNIFORM.mask()),
    // FADD.
    ReferenceRow::new(0x221, [0xffff_ffff, 0xc000_00ff, 0x0001_efff, 0xffff_fe00])
        .dropping(IGNORED_SLOT_DROPPED),
    ReferenceRow::new(0x421, [0xffff_ffff, 0xffff_ffff, 0x0001_efff, 0xffff_fe00])
        .dropping(IGNORED_SLOT_DROPPED),
    ReferenceRow::new(0x621, [0xffff_ffff, 0xc7ff_ffff, 0x0801_efff, 0xffff_fe00])
        .dropping(IGNORED_SLOT_DROPPED),
    ReferenceRow::new(0xe21, [0xffff_ffff, 0xc000_003f, 0x0801_efff, 0xffff_fe00])
        .dropping(IGNORED_SLOT_DROPPED)
        .fixing(SLOT_UNIFORM.mask()),
    // FMUL.
    ReferenceRow::new(0x220, [0xffff_ffff, 0xc000_00ff, 0x0071_ffff, 0xffff_fe00])
        .dropping(IGNORED_SLOT_DROPPED)
        .with_canonical(FMUL_CANONICAL),
    ReferenceRow::new(0x820, [0xffff_ffff, 0xffff_ffff, 0x0071_ffff, 0xffff_fe00])
        .dropping(IGNORED_SLOT_DROPPED)
        .with_canonical(FMUL_CANONICAL),
    ReferenceRow::new(0xa20, [0xffff_ffff, 0xc7ff_ffff, 0x0871_ffff, 0xffff_fe00])
        .dropping(IGNORED_SLOT_DROPPED)
        .with_canonical(FMUL_CANONICAL),
    ReferenceRow::new(0xc20, [0xffff_ffff, 0xc000_003f, 0x0871_ffff, 0xffff_fe00])
        .dropping(IGNORED_SLOT_DROPPED)
        .fixing(SLOT_UNIFORM.mask())
        .with_canonical(FMUL_CANONICAL),
    // FMNMX.
    ReferenceRow::new(0x209, [0xffff_ffff, 0xc000_00ff, 0x0783_031e, 0xffff_fe00])
        .dropping(FMNMX_DROPPED),
    ReferenceRow::new(0x809, [0xffff_ffff, 0xffff_ffff, 0x0783_031e, 0xffff_fe00])
        .dropping(FMNMX_DROPPED),
    ReferenceRow::new(0xa09, [0xffff_ffff, 0xc7ff_ffff, 0x0f83_031e, 0xffff_fe00])
        .dropping(FMNMX_DROPPED),
    ReferenceRow::new(0xc09, [0xffff_ffff, 0xc000_003f, 0x0f81_031e, 0xffff_fe00])
        .dropping(FMNMX_DROPPED)
        .fixing(SLOT_UNIFORM.mask()),
    // FSET.
    ReferenceRow::new(0x20a, [0xffff_ffff, 0xc000_00ff, 0x0781_ff00, 0xffff_fe00]),
    ReferenceRow::new(0x80a, [0xffff_ffff, 0xffff_ffff, 0x0781_ff00, 0xffff_fe00]),
    ReferenceRow::new(0xa0a, [0xffff_ffff, 0xc7ff_ffff, 0x0f81_ff00, 0xffff_fe00]),
    ReferenceRow::new(0xc0a, [0xffff_ffff, 0xc000_003f, 0x0f81_f300, 0xffff_fe00])
        .fixing(SLOT_UNIFORM.mask()),
    // FSETP.
    ReferenceRow::new(0x20b, [0xff00_ffff, 0xc000_00ff, 0x07ff_ff00, 0xffff_fe00]),
    ReferenceRow::new(0x80b, [0xff00_ffff, 0xffff_ffff, 0x07ff_ff00, 0xffff_fe00]),
    ReferenceRow::new(0xa0b, [0xff00_ffff, 0xc7ff_ffff, 0x0fff_ff00, 0xffff_fe00]),
    ReferenceRow::new(0xc0b, [0xff00_ffff, 0xc000_003f, 0x0fff_ff00, 0xffff_fe00])
        .fixing(SLOT_UNIFORM.mask()),
    // FSEL.
    ReferenceRow::new(0x208, [0xffff_ffff, 0xc000_00ff, 0x0780_0300, 0xffff_fe00]),
    ReferenceRow::new(0x808, [0xffff_ffff, 0xffff_ffff, 0x0780_0300, 0xffff_fe00]),
    ReferenceRow::new(0xa08, [0xffff_ffff, 0xc7ff_ffff, 0x0f80_0300, 0xffff_fe00]),
    // MUFU.
    ReferenceRow::new(0x308, [0x00ff_ffff, 0xd000_00ff, 0x0000_fe00, 0xffff_fe00]),
    ReferenceRow::new(0x908, [0x00ff_ffff, 0xffff_ffff, 0x0000_fe00, 0xffff_fe00]),
    ReferenceRow::new(0xb08, [0x00ff_ffff, 0xc7ff_ffff, 0x0800_fe00, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder),
    ReferenceRow::new(0xd08, [0x00ff_ffff, 0xc000_003f, 0x0800_fe00, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .fixing(SLOT_UNIFORM.mask()),
    // FSWZADD.
    ReferenceRow::new(0x822, [0xffff_ffff, 0x0000_00ff, 0x0001_e0ff, 0xffff_fe00]),
    // FRND, then its 64-bit opcode.
    ReferenceRow::new(0x307, [0x00ff_ffff, 0xc000_00ff, 0x0031_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0x907, [0x00ff_ffff, 0xffff_ffff, 0x0031_d800, 0xffff_fe00])
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0xb07, [0x00ff_ffff, 0xc7ff_ffff, 0x0831_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0xd07, [0x00ff_ffff, 0xc000_003f, 0x0831_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .fixing(SLOT_UNIFORM.mask())
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0x313, [0x00ff_ffff, 0xc000_00ff, 0x0031_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0x913, [0x00ff_ffff, 0xffff_ffff, 0x0031_d800, 0xffff_fe00])
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0xb13, [0x00ff_ffff, 0xc7ff_ffff, 0x0831_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0xd13, [0x00ff_ffff, 0xc000_003f, 0x0831_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .fixing(SLOT_UNIFORM.mask())
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    // F2F, then its 64-bit opcode.
    ReferenceRow::new(0x304, [0x00ff_ffff, 0xc000_00ff, 0x0031_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0x904, [0x00ff_ffff, 0xffff_ffff, 0x0031_d800, 0xffff_fe00])
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0xb04, [0x00ff_ffff, 0xc7ff_ffff, 0x0831_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0xd04, [0x00ff_ffff, 0xc000_003f, 0x0831_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .fixing(SLOT_UNIFORM.mask())
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0x310, [0x00ff_ffff, 0xc000_00ff, 0x0031_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0x910, [0x00ff_ffff, 0xffff_ffff, 0x0031_d800, 0xffff_fe00])
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0xb10, [0x00ff_ffff, 0xc7ff_ffff, 0x0831_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    ReferenceRow::new(0xd10, [0x00ff_ffff, 0xc000_003f, 0x0831_d800, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .fixing(SLOT_UNIFORM.mask())
        .with_canonical(CVT_DST_F32 | CVT_SRC_F32),
    // F2I, then its 64-bit opcode.
    ReferenceRow::new(0x305, [0x00ff_ffff, 0xd000_00ff, 0x0031_f900, 0xffff_fe00])
        .with_canonical(CVT_SRC_F32),
    ReferenceRow::new(0x905, [0x00ff_ffff, 0xffff_ffff, 0x0031_f900, 0xffff_fe00])
        .with_canonical(CVT_SRC_F32),
    ReferenceRow::new(0xb05, [0x00ff_ffff, 0xc7ff_ffff, 0x0831_f900, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .with_canonical(CVT_SRC_F32),
    ReferenceRow::new(0xd05, [0x00ff_ffff, 0xc000_003f, 0x0831_f900, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .fixing(SLOT_UNIFORM.mask())
        .with_canonical(CVT_SRC_F32),
    ReferenceRow::new(0x311, [0x00ff_ffff, 0xc000_00ff, 0x0031_f900, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .with_canonical(CVT_SRC_F32),
    ReferenceRow::new(0x911, [0x00ff_ffff, 0xffff_ffff, 0x0031_f900, 0xffff_fe00])
        .with_canonical(CVT_SRC_F32),
    ReferenceRow::new(0xb11, [0x00ff_ffff, 0xc7ff_ffff, 0x0831_f900, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .with_canonical(CVT_SRC_F32),
    ReferenceRow::new(0xd11, [0x00ff_ffff, 0xc000_003f, 0x0831_f900, 0xffff_fe00])
        .adding(NAK_HIGH_HALF, Namer::NakEncoder)
        .fixing(SLOT_UNIFORM.mask())
        .with_canonical(CVT_SRC_F32),
    // I2F, then its 64-bit opcode.
    ReferenceRow::new(0x306, [0x00ff_ffff, 0xf000_00ff, 0x0030_dc00, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS)
        .with_canonical(CVT_DST_F32),
    ReferenceRow::new(0x906, [0x00ff_ffff, 0xffff_ffff, 0x0030_dc00, 0xffff_fe00])
        .with_canonical(CVT_DST_F32),
    ReferenceRow::new(0xb06, [0x00ff_ffff, 0xf7ff_ffff, 0x0830_dc00, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS)
        .with_canonical(CVT_DST_F32),
    ReferenceRow::new(0xd06, [0x00ff_ffff, 0xf000_003f, 0x0830_dc00, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS)
        .fixing(SLOT_UNIFORM.mask())
        .with_canonical(CVT_DST_F32),
    ReferenceRow::new(0x312, [0x00ff_ffff, 0xf000_00ff, 0x0030_dc00, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS)
        .with_canonical(CVT_DST_F32),
    ReferenceRow::new(0x912, [0x00ff_ffff, 0xffff_ffff, 0x0030_dc00, 0xffff_fe00])
        .with_canonical(CVT_DST_F32),
    ReferenceRow::new(0xb12, [0x00ff_ffff, 0xf7ff_ffff, 0x0830_dc00, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS)
        .with_canonical(CVT_DST_F32),
    ReferenceRow::new(0xd12, [0x00ff_ffff, 0xf000_003f, 0x0830_dc00, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS)
        .fixing(SLOT_UNIFORM.mask())
        .with_canonical(CVT_DST_F32),
    // I2FP.
    ReferenceRow::new(0x245, [0x00ff_ffff, 0xc000_00ff, 0x0830_dc00, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS | SLOT_UNIFORM.mask())
        .with_canonical(CVT_DST_F32),
    ReferenceRow::new(0x845, [0x00ff_ffff, 0xffff_ffff, 0x0830_dc00, 0xffff_fe00])
        .dropping(SLOT_UNIFORM.mask())
        .with_canonical(CVT_DST_F32),
    ReferenceRow::new(0xa45, [0x00ff_ffff, 0xc7ff_ffff, 0x0830_dc00, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS)
        .with_canonical(CVT_DST_F32),
    ReferenceRow::new(0xc45, [0x00ff_ffff, 0xc000_003f, 0x0830_dc00, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS)
        .fixing(SLOT_UNIFORM.mask())
        .with_canonical(CVT_DST_F32),
    // F2FP.
    ReferenceRow::new(0x23e, [0xffff_ffff, 0xc000_00ff, 0x0001_cfff, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS | F2FP_DROPPED),
    ReferenceRow::new(0x83e, [0xffff_ffff, 0xffff_ffff, 0x0001_cfff, 0xffff_fe00])
        .dropping(F2FP_DROPPED),
    ReferenceRow::new(0xa3e, [0xffff_ffff, 0xc7ff_ffff, 0x0801_cfff, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS | F2FP_DROPPED),
    ReferenceRow::new(0xc3e, [0xffff_ffff, 0xc000_003f, 0x0801_cfff, 0xffff_fe00])
        .dropping(SLOT_MODIFIERS | F2FP_DROPPED)
        .fixing(SLOT_UNIFORM.mask()),
];

/// Generators of the group's decoded operations, one per operation (`generate`).
#[cfg(test)]
pub(crate) fn any_ops() -> Vec<BoxedStrategy<Op>> {
    generate::all()
}

#[cfg(test)]
mod generate;

shells! {}

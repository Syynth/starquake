//! Field positions and opcodes of the six slice-1 forms, shared by the decoder and the test
//! assembler so each position is written once.
//!
//! Ported from `mesa-nouveau/src/nouveau/compiler/nak/sm70_encode.rs` @ 52238762 (MIT,
//! Copyright 2022 Collabora, Ltd.): the positions below are those NAK's encoder writes, named
//! after the `SM70Encoder` helper or `impl SM70Op` block that writes them. The known-bit masks
//! are cross-checked against PR #4588's generated decode table
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97, rows
//! `0x202`, `0x423`, `0x802`, `0x823`, `0x223`, `0x321`, `0x322`, `0x94d`, `0x919`) by
//! `tests::masks_match_the_reference_table`.

use super::instruction::Field;

/// The 12-bit opcode of a fixed-form instruction (`set_opcode`: bits 0..12).
pub(crate) const OPCODE: Field = Field::new(0, 12);
/// The operand form of an ALU instruction (`encode_alu_base`: `set_field(9..12, form)`).
pub(crate) const ALU_FORM: Field = Field::new(9, 3);
/// The guard predicate register (`set_pred`: `set_pred_reg(12..15, ..)`).
pub(crate) const GUARD: Field = Field::new(12, 3);
/// The guard predicate's negation (`set_pred`: `set_bit(15, pred_inv)`).
pub(crate) const GUARD_NOT: Field = Field::bit(15);
/// The destination register (`set_dst`: bits 16..24).
pub(crate) const DST: Field = Field::new(16, 8);
/// Scheduling information: stall count, yield, barriers, reuse (`set_instr_deps`: bits
/// 105..126 before SM 12.0). Every PR #4588 row marks bits 105..128 known (`isa.h`,
/// `FirstUnknownBit`), so this mask covers through bit 127.
pub(crate) const SCHEDULING: Field = Field::new(105, 23);

/// ALU source 0, always a register (`encode_alu_src0`: `set_alu_reg(24..32, 73, 72, ..)`).
pub(crate) const SRC0: Field = Field::new(24, 8);
/// ALU source 0's absolute-value bit.
pub(crate) const SRC0_ABS: Field = Field::bit(73);
/// ALU source 0's negation bit.
pub(crate) const SRC0_NEG: Field = Field::bit(72);
/// The register in the wide slot (`encode_alu_reg`: `set_alu_reg(32..40, 62, 63, ..)`).
pub(crate) const SLOT_REG: Field = Field::new(32, 8);
/// The wide slot register's absolute-value bit.
pub(crate) const SLOT_ABS: Field = Field::bit(62);
/// The wide slot register's negation bit.
pub(crate) const SLOT_NEG: Field = Field::bit(63);
/// The 32-bit immediate in the wide slot (`encode_alu_imm`: `set_field(32..64, imm)`).
pub(crate) const SLOT_IMM: Field = Field::new(32, 32);
/// ALU source 2 as a 32-bit register (`encode_alu_src2`: `set_alu_reg(64..72, 74, 75, ..)` when
/// the source type is not fp16).
pub(crate) const SRC2: Field = Field::new(64, 8);
/// ALU source 2's absolute-value bit (32-bit sources).
pub(crate) const SRC2_ABS: Field = Field::bit(74);
/// ALU source 2's negation bit (32-bit sources).
pub(crate) const SRC2_NEG: Field = Field::bit(75);

/// ALU operand forms (`encode_alu_base`'s `form` values).
pub(crate) mod form {
    /// Register, register, register: sources 1 and 2 are registers.
    pub(crate) const REG: u64 = 1;
    /// Source 2 is the immediate in the wide slot; source 1 moves to source 2's register field.
    pub(crate) const IMM_SRC2: u64 = 2;
    /// Source 1 is the immediate in the wide slot.
    pub(crate) const IMM_SRC1: u64 = 4;
}

/// `MOV` (`impl SM70Op for OpMov`: `encode_alu(0x002, dst, None, src, None)`).
pub(crate) const MOV_OPCODE: u64 = 0x002;
/// `MOV`'s per-quad-lane mask (`set_field(72..76, quad_lanes)`).
pub(crate) const MOV_QUAD_LANES: Field = Field::new(72, 4);

/// `FFMA` (`impl SM70Op for OpFFma`: `encode_alu(0x023, ..)`).
pub(crate) const FFMA_OPCODE: u64 = 0x023;
/// `FFMA`'s "denorm/zero multiply" bit (`set_bit(76, dnz)`).
pub(crate) const FFMA_DNZ: Field = Field::bit(76);
/// `FFMA`'s saturate bit (`set_bit(77, saturate)`).
pub(crate) const FFMA_SAT: Field = Field::bit(77);
/// `FFMA`'s rounding mode (`set_rnd_mode(78..80, ..)`).
pub(crate) const FFMA_RND: Field = Field::new(78, 2);
/// `FFMA`'s flush-to-zero bit (`set_bit(80, ftz)`).
pub(crate) const FFMA_FTZ: Field = Field::bit(80);

/// `ALD` (`impl SM70Op for OpALd`: `set_opcode(0x321)`).
pub(crate) const ALD_OPCODE: u64 = 0x321;
/// `AST` (`impl SM70Op for OpASt`: `set_opcode(0x322)`).
pub(crate) const AST_OPCODE: u64 = 0x322;
/// The attribute offset register of `ALD` and `AST` (`set_reg_src(24..32, offset)`).
pub(crate) const ATTR_OFFSET: Field = Field::new(24, 8);
/// `ALD`'s vertex register (`set_reg_src(32..40, vtx)`).
pub(crate) const ALD_VTX: Field = Field::new(32, 8);
/// `AST`'s data register (`set_reg_src(32..40, data)`).
pub(crate) const AST_DATA: Field = Field::new(32, 8);
/// `AST`'s vertex register (`set_reg_src(64..72, vtx)`).
pub(crate) const AST_VTX: Field = Field::new(64, 8);
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

/// `EXIT` (`impl SM70Op for OpExit`: `set_opcode(0x94d)`).
pub(crate) const EXIT_OPCODE: u64 = 0x94d;
/// `EXIT`'s first mode bit; NAK's comment names it `./.KEEPREFCOUNT/.PREEMPTED/.INVALID3`
/// together with bit 85 (`set_field(84..85, false)`).
pub(crate) const EXIT_KEEP_REF_COUNT: Field = Field::bit(84);
/// `EXIT`'s `.NO_ATEXIT` bit (`set_field(85..86, false)`).
pub(crate) const EXIT_NO_AT_EXIT: Field = Field::bit(85);
/// `EXIT`'s predicate register (`set_field(87..90, 0x7)`, "TODO: Predicate").
pub(crate) const EXIT_PRED: Field = Field::new(87, 3);
/// `EXIT`'s predicate negation (`set_field(90..91, false)`, "NOT").
pub(crate) const EXIT_PRED_NOT: Field = Field::bit(90);

/// `S2R` (`impl SM70Op for OpS2R`: `set_opcode(0x919)` for the non-uniform form).
pub(crate) const S2R_OPCODE: u64 = 0x919;
/// `S2R`'s special register index (`set_field(72..80, idx)`).
pub(crate) const S2R_SR: Field = Field::new(72, 8);

/// The 12-bit opcode of an ALU instruction with `opcode` in `form`.
const fn alu(opcode: u64, form: u64) -> u64 {
    opcode | (form << ALU_FORM.start)
}

/// `MOV` with a register source.
pub(crate) const MOV_REG: u64 = alu(MOV_OPCODE, form::REG);
/// `MOV` with an immediate source.
pub(crate) const MOV_IMM: u64 = alu(MOV_OPCODE, form::IMM_SRC1);
/// `FFMA` with three registers.
pub(crate) const FFMA_REG: u64 = alu(FFMA_OPCODE, form::REG);
/// `FFMA` with an immediate source 2.
pub(crate) const FFMA_IMM_SRC2: u64 = alu(FFMA_OPCODE, form::IMM_SRC2);
/// `FFMA` with an immediate source 1.
pub(crate) const FFMA_IMM_SRC1: u64 = alu(FFMA_OPCODE, form::IMM_SRC1);

/// The bits every form explains: opcode, guard and scheduling.
const COMMON: u128 = OPCODE.mask() | GUARD.mask() | GUARD_NOT.mask() | SCHEDULING.mask();

/// The bits a form with these fields explains.
const fn known(fields: &[Field]) -> u128 {
    let mut mask = COMMON;
    let mut i = 0;
    while i < fields.len() {
        mask |= fields[i].mask();
        i += 1;
    }
    mask
}

/// `MOV` with a register source (form 1). Stricter than PR #4588's row `0x202`, which also
/// marks bits 62 and 63 (the slot's abs and neg bits) as known: `MOV` has no source modifier
/// in NAK (`OpMov::src` is never modified) and its translator ignores them, so a word with one
/// set is rejected here.
pub(crate) const MOV_REG_KNOWN: u128 = known(&[DST, SLOT_REG, MOV_QUAD_LANES]);
/// `MOV` with an immediate source (form 4).
pub(crate) const MOV_IMM_KNOWN: u128 = known(&[DST, SLOT_IMM, MOV_QUAD_LANES]);

/// The `FFMA` modifier bits every form has.
const FFMA_MODIFIERS: [Field; 7] = [DST, SRC0, SRC0_ABS, SRC0_NEG, FFMA_DNZ, FFMA_SAT, FFMA_FTZ];

/// `FFMA` with three registers (form 1).
pub(crate) const FFMA_REG_KNOWN: u128 = known(&FFMA_MODIFIERS)
    | known(&[
        FFMA_RND, SLOT_REG, SLOT_ABS, SLOT_NEG, SRC2, SRC2_ABS, SRC2_NEG,
    ]);
/// `FFMA` with an immediate source 2 (form 2): source 1 is in source 2's register field.
pub(crate) const FFMA_IMM_SRC2_KNOWN: u128 =
    known(&FFMA_MODIFIERS) | known(&[FFMA_RND, SLOT_IMM, SRC2, SRC2_ABS, SRC2_NEG]);
/// `FFMA` with an immediate source 1 (form 4).
pub(crate) const FFMA_IMM_SRC1_KNOWN: u128 =
    known(&FFMA_MODIFIERS) | known(&[FFMA_RND, SLOT_IMM, SRC2, SRC2_ABS, SRC2_NEG]);

/// `ALD`.
pub(crate) const ALD_KNOWN: u128 = known(&[
    DST,
    ATTR_OFFSET,
    ALD_VTX,
    ATTR_ADDR,
    ATTR_COMPS,
    ATTR_PATCH,
    ATTR_PHYS,
    ALD_OUTPUT,
]);
/// `AST`.
pub(crate) const AST_KNOWN: u128 = known(&[
    ATTR_OFFSET,
    AST_DATA,
    AST_VTX,
    ATTR_ADDR,
    ATTR_COMPS,
    ATTR_PATCH,
    ATTR_PHYS,
]);
/// `EXIT`.
pub(crate) const EXIT_KNOWN: u128 = known(&[
    EXIT_KEEP_REF_COUNT,
    EXIT_NO_AT_EXIT,
    EXIT_PRED,
    EXIT_PRED_NOT,
]);
/// `S2R`.
pub(crate) const S2R_KNOWN: u128 = known(&[DST, S2R_SR]);

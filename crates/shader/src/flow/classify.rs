//! What flow analysis needs to know of each opcode: whether it changes control flow, and, for
//! straight-line instructions, whether a guard on it needs no block of its own from flow
//! analysis (the reference keeps such a guard as a select on its results; the translation
//! driver makes the instruction a conditional block, `crate::translate::driver`'s "Guards").
//!
//! Ported from PR #4588: `eden-pr4588/src/shader_recompiler/frontend/ampere/classify.h` @
//! e5fb3b97 (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project), `Classify`, `FormOf` and
//! `BaseOf`. Every opcode the decoder already names is imported from its group (`decode::op`);
//! the ones it does not name yet (instructions of slices 6c, 6e and 6i, which have no decode
//! form so far) are named here with the reference's comment.
//! `tests::classify::classify_matches_the_reference` pins every row against the reference's
//! literals.
//!
//! The one change: the uniform `BRA` (`0x547`) is a branch here. The reference has no row for it
//! and so treats it as a straight-line instruction that ignores where it branches; flow analysis
//! refuses it as a uniform branch instead ([`super::FlowError::UniformBranch`]), as the reference
//! refuses a `0x947` with its `.U` bit set (`flow.cpp`, "native uniform branch").

use crate::decode::{
    ALD_OPCODE, ALU_FORM, BAR_OPCODE, BRA_OPCODE, BRA_UNIFORM_OPCODE, BREAK_OPCODE, BREV_OPCODE,
    BRX_OPCODE, BRXU_OPCODE, BSSY_OPCODE, BSYNC_OPCODE, CALL_OPCODE, CS2R_OPCODE, DEPBAR_OPCODE,
    EXIT_OPCODE, F2F_OPCODE, F2FP_OPCODE, F2I_OPCODE, FADD_OPCODE, FFMA_OPCODE, FLO_OPCODE,
    FMNMX_OPCODE, FMUL_OPCODE, FRND_OPCODE, FSEL_OPCODE, FSET_OPCODE, FSETP_OPCODE, FSWZADD_OPCODE,
    I2F_OPCODE, I2FP_OPCODE, IABS_OPCODE, IADD3_OPCODE, IMAD_HI_OPCODE, IMAD_OPCODE,
    IMAD_WIDE_OPCODE, IMNMX_OPCODE, IPA_OPCODE, ISETP_OPCODE, KILL_OPCODE, LEA_OPCODE, LOP3_OPCODE,
    MOV_OPCODE, MUFU_OPCODE, NOP_OPCODE, P2R_OPCODE, PLOP3_OPCODE, POPC_OPCODE, PRMT_OPCODE,
    R2P_OPCODE, RET_OPCODE, S2R_OPCODE, S2UR_OPCODE, SEL_OPCODE, SGXT_OPCODE, SHF_OPCODE,
    UBREV_OPCODE, UFLO_OPCODE, UIADD3_OPCODE, UIMAD_OPCODE, UIMAD_WIDE_OPCODE, UISETP_OPCODE,
    ULEA_OPCODE, ULOP3_OPCODE, UPOPC_OPCODE, USEL_OPCODE, USGXT_OPCODE, USHF_OPCODE, VOTEU_OPCODE,
    WARPSYNC_OPCODE,
};

/// The operand form of a register source (`classify.h`: "1 reg").
pub(crate) const FORM_REG: u16 = 1;
/// The operand form of an immediate source (`classify.h`: "4 imm").
pub(crate) const FORM_IMM: u16 = 4;
/// The bits of the opcode below [`ALU_FORM`]: the base operation (`BaseOf`: `op12 & 0x1ff`).
const BASE_MASK: u16 = (1 << ALU_FORM.start) - 1;
/// The bits of [`ALU_FORM`] (`FormOf`: `(op12 >> 9) & 7`).
const FORM_MASK: u16 = (1 << ALU_FORM.width) - 1;

/// `base` in operand form `form`.
pub(crate) const fn with_form(base: u16, form: u16) -> u16 {
    base | (form << ALU_FORM.start)
}

/// The operand form of a 12-bit opcode (`FormOf`).
const fn form_of(opcode: u16) -> u16 {
    (opcode >> ALU_FORM.start) & FORM_MASK
}

/// The base operation of a 12-bit opcode (`BaseOf`).
const fn base_of(opcode: u16) -> u16 {
    opcode & BASE_MASK
}

// Opcodes slice 6d's decoder names, each with `classify.h`'s comment.

// `R2UR` ("R2UR: reads the guard as data, so it is not split into a block"), `UMOV` ("UMOV
// imm", "UMOV UR"), `ULDC` ("ULDC", "ULDC.64") and `UPLOP3` ("UPLOP3"), named by slice 6d
// (`decode::op::uniform`).
pub(crate) use crate::decode::{
    R2UR_OPCODE, ULDC_OPCODE, ULDC64_OPCODE, UMOV_IMM_OPCODE, UMOV_UR_OPCODE, UPLOP3_OPCODE,
};
/// The uniform ALU bases that are pure and uniform in forms 1 and 4 (`classify.h`: "Uniform
/// ALU: UIADD3, ULEA, ULOP3, USHF, UIMAD, UISETP, USEL (register or immediate forms)"). The
/// reference lists twelve bare bases under that comment, more than it names; slice 6d's decoder
/// names each (`decode::op::uniform`): `UIADD3`, `ULEA`, `ULOP3`, `USHF`, `UIMAD`, `UIMAD.WIDE`,
/// `UBREV`, `USGXT`, `UFLO`, `UPOPC`, `UISETP` and `USEL`, in the reference's order. `UPRMT`
/// and `UBMSK` are not among them. `tests::classify::classify_matches_the_reference` keeps the
/// reference's literals.
pub(crate) const UNIFORM_ALU_BASES: [u16; 12] = [
    UIADD3_OPCODE,
    ULEA_OPCODE,
    ULOP3_OPCODE,
    USHF_OPCODE,
    UIMAD_OPCODE,
    UIMAD_WIDE_OPCODE,
    UBREV_OPCODE,
    USGXT_OPCODE,
    UFLO_OPCODE,
    UPOPC_OPCODE,
    UISETP_OPCODE,
    USEL_OPCODE,
];

// Opcodes the decoder does not name yet, each with `classify.h`'s comment.

/// `LDC` (`classify.h`: "LDC", a pure attribute read; slice 6e decodes it).
pub(crate) const LDC_OPCODE: u16 = 0xb82;
/// `HMNMX2`'s base (`classify.h`: "HMNMX2").
pub(crate) const HMNMX2_BASE: u16 = 0x040;
/// `HADD2`'s base (`classify.h`: "HADD2").
pub(crate) const HADD2_BASE: u16 = 0x030;
/// `HFMA2`'s base (`classify.h`: "HFMA2").
pub(crate) const HFMA2_BASE: u16 = 0x031;
/// `HMUL2`'s base (`classify.h`: "HMUL2").
pub(crate) const HMUL2_BASE: u16 = 0x032;
/// `HSETP2`'s base (`classify.h`: "HSETP2").
pub(crate) const HSETP2_BASE: u16 = 0x034;
/// The opcodes that need every lane of the warp, so may not run under a guard (`flow.cpp`,
/// `AnalyzeFlowImpl`: `op == 0xb1d || op == 0x83b || op == 0x23a || op == 0x23c`, "Barriers and
/// warp-wide matrix operations need every lane of the warp to execute them"). `0xb1d` is
/// `BAR`; the reference does not name the other three, which are not decoded yet (slice 6i).
pub(crate) const WARP_WIDE_OPCODES: [u16; 4] = [BAR_OPCODE, 0x83b, 0x23a, 0x23c];

/// What an instruction does to control flow (`classify.h`, `OpKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum OpKind {
    /// A straight-line instruction.
    Plain,
    /// `BRA`.
    Bra,
    /// `EXIT`.
    Exit,
    /// `KILL`.
    Kill,
    /// `BSSY`.
    Bssy,
    /// `BSYNC`.
    Bsync,
    /// `BRX` and `BRXU`.
    Brx,
    /// `CALL`.
    Call,
    /// `RET`.
    Ret,
    /// No effect on the flow graph: `BREAK`, `WARPSYNC` with an immediate, `NOP` and `DEPBAR`.
    Nop,
}

/// An opcode's class (`classify.h`, `OpInfo`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct OpInfo {
    /// What it does to control flow.
    pub kind: OpKind,
    /// Whether it only writes registers and predicates and cannot fault, so that flow analysis
    /// leaves it in its block under a guard (the reference keeps the guard as a select on its
    /// results; the translation driver makes the instruction a conditional block).
    pub pure: bool,
    /// Whether it is a uniform datapath instruction, whose guard is a uniform predicate.
    pub uniform: bool,
}

impl OpInfo {
    const fn of(kind: OpKind) -> Self {
        Self {
            kind,
            pure: false,
            uniform: false,
        }
    }

    const PLAIN: Self = Self::of(OpKind::Plain);
    const PURE: Self = Self {
        kind: OpKind::Plain,
        pure: true,
        uniform: false,
    };
    const PURE_UNIFORM: Self = Self {
        kind: OpKind::Plain,
        pure: true,
        uniform: true,
    };
}

/// The `WARPSYNC` opcode the reference counts as having no effect: its immediate form
/// (`classify.h`: `case 0x948: // WARPSYNC`).
const WARPSYNC_IMM_OPCODE: u16 = with_form(WARPSYNC_OPCODE, FORM_IMM);
/// `BREV` in its register form (`classify.h`: `case 0x301: // BREV`).
const BREV_REG_OPCODE: u16 = with_form(BREV_OPCODE, FORM_REG);

/// The class of a 12-bit opcode (`Classify`).
pub(crate) const fn classify(opcode: u16) -> OpInfo {
    match opcode {
        BRA_OPCODE | BRA_UNIFORM_OPCODE => return OpInfo::of(OpKind::Bra),
        EXIT_OPCODE => return OpInfo::of(OpKind::Exit),
        KILL_OPCODE => return OpInfo::of(OpKind::Kill),
        BSSY_OPCODE => return OpInfo::of(OpKind::Bssy),
        BSYNC_OPCODE => return OpInfo::of(OpKind::Bsync),
        BRX_OPCODE | BRXU_OPCODE => return OpInfo::of(OpKind::Brx),
        CALL_OPCODE => return OpInfo::of(OpKind::Call),
        RET_OPCODE => return OpInfo::of(OpKind::Ret),
        BREAK_OPCODE | WARPSYNC_IMM_OPCODE | NOP_OPCODE | DEPBAR_OPCODE => {
            return OpInfo::of(OpKind::Nop);
        }
        IPA_OPCODE | ALD_OPCODE | LDC_OPCODE | CS2R_OPCODE | BREV_REG_OPCODE | FSWZADD_OPCODE
        | R2UR_OPCODE | S2R_OPCODE | P2R_OPCODE | R2P_OPCODE | PLOP3_OPCODE => {
            return OpInfo::PURE;
        }
        UMOV_IMM_OPCODE | UMOV_UR_OPCODE | ULDC_OPCODE | ULDC64_OPCODE | S2UR_OPCODE
        | UPLOP3_OPCODE | VOTEU_OPCODE => return OpInfo::PURE_UNIFORM,
        _ => {}
    }
    let form = form_of(opcode);
    if form == 0 {
        return OpInfo::PLAIN;
    }
    let base = base_of(opcode);
    if form == FORM_REG || form == FORM_IMM {
        let mut i = 0;
        while i < UNIFORM_ALU_BASES.len() {
            if UNIFORM_ALU_BASES[i] == base {
                return OpInfo::PURE_UNIFORM;
            }
            i += 1;
        }
    }
    match base {
        FMUL_OPCODE | MUFU_OPCODE | MOV_OPCODE | FADD_OPCODE | FFMA_OPCODE | FMNMX_OPCODE
        | SEL_OPCODE | FSEL_OPCODE | FSETP_OPCODE | FSET_OPCODE | HMNMX2_BASE | ISETP_OPCODE
        | IADD3_OPCODE | LOP3_OPCODE | SHF_OPCODE | LEA_OPCODE | IMNMX_OPCODE | PRMT_OPCODE
        | HADD2_BASE | HFMA2_BASE | HMUL2_BASE | IMAD_OPCODE | IMAD_HI_OPCODE
        | IMAD_WIDE_OPCODE | IABS_OPCODE | FLO_OPCODE | POPC_OPCODE | SGXT_OPCODE | HSETP2_BASE
        | F2FP_OPCODE | F2I_OPCODE | I2F_OPCODE | I2FP_OPCODE | FRND_OPCODE | F2F_OPCODE => {
            OpInfo::PURE
        }
        _ => OpInfo::PLAIN,
    }
}

/// Whether `opcode` needs every lane of the warp ([`WARP_WIDE_OPCODES`]).
pub(crate) const fn is_warp_wide(opcode: u16) -> bool {
    let mut i = 0;
    while i < WARP_WIDE_OPCODES.len() {
        if WARP_WIDE_OPCODES[i] == opcode {
            return true;
        }
        i += 1;
    }
    false
}

//! Slice 12g: the attribute and system-value group, decoded by slice 6g, translated to IR.
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/` @ e5fb3b97, GPL-3.0-or-later):
//! `ops_misc.cpp` (`ALD`, `AST`, `CS2R`, `S2R`) and `ops_attr.cpp` (`IPA`), cross-checked with
//! Eden's Maxwell translators (`eden/src/shader_recompiler/frontend/maxwell/translate/impl/` @
//! cf654052, GPL-3.0-or-later): `load_store_attribute.cpp` and `move_special_register.cpp`, and
//! for the two the PR has no translator for, `attribute_memory_to_physical.cpp` (`AL2P`) and
//! `pixel_load.cpp` (`PIXLD`). Attribute numbers are Eden's (`frontend/ir/attribute.h`, the same
//! file in both trees).
//!
//! # What each instruction becomes
//!
//! | Instruction | IR | Source |
//! |---|---|---|
//! | `ALD` | `GetAttribute` per word, vertex from the vertex register (`RZ` is 0), each into its register | PR `ALD` |
//! | `AST` | `SetAttribute` per word from its register (all from `RZ` when the data register is `RZ`) | PR `AST` |
//! | `IPA` | `GetAttribute` at vertex 0, times `GetAttribute` of the position's W for a perspective input; `.OFFSET` reads `InterpolateAtOffset` instead | PR `IPA`; the offset is starquake's |
//! | `AL2P` | `IAdd32` of the offset register and the address | Eden `AL2P` |
//! | `S2R`, `S2UR` | `LaneId`, a component of `LocalInvocationId` or `WorkgroupId`, the lane masks, or `1.0` for the W scale factors, into a register or a uniform register | PR `S2R` |
//! | `CS2R` | zero into one register or two | PR `CS2R` |
//! | `PIXLD.MY_INDEX` | `SampleId` | Eden `PIXLD` |
//!
//! A write to `RZ` or `URZ` is dropped, and `S2R`, `S2UR` and `PIXLD` to them read nothing
//! (PR #4588 emits the read and drops the write; the registers come out the same).
//!
//! **`IPA`'s perspective rule** is the PR's: a legacy attribute (Eden's `IsLegacyAttribute`: the
//! front and back colours, the fog coordinate, the fixed-function texture coordinates) is
//! multiplied by the position's W; a generic one (Eden's `IsGeneric`: `Generic0X` through
//! `Generic31X`) is when the header's input map gives its first component that is not unused
//! as perspective, or every component as unused (`sph.ps.GenericInputMap`); any other attribute
//! is not. Eden's `IsGeneric` ends at `Generic31X`, so the Y, Z and W of generic 31 are not
//! multiplied; that quirk is kept (`tests::ipa_follows_edens_generic_range`). The frequency
//! (`.PASS` or `.CONSTANT`) changes nothing, as in the PR: "the flat state comes from the
//! program header". The centroid location reads the same value as the default one, as in the
//! PR ("only multisampled targets tell them apart") and in Eden, which reads no location.
//!
//! # Rejections
//!
//! Each of the PR's `Require` checks on these forms is kept with its constant as a named
//! rejection, in the PR's order, except two parts of `IPA`'s that the first divergence below
//! replaces (the sample location, and the offset register under `.OFFSET`); so is each
//! `NotImplementedException` of Eden's for the two forms ported from it:
//!
//! | Instruction | Check, in order | Rejection |
//! |---|---|---|
//! | `ALD` | offset register `RZ` (`Take(24, 8) == 255`); not patch or physical (`Take(76, 2) == 0`); not an output (`Take(79, 1) == 0`); `address % 4 == 0 && address + components * 4 <= 0x400`; `dest + components <= 255` | [`Unsupported::IndexedAttribute`], [`Unsupported::PatchOrPhysical`], [`Unsupported::OutputRead`], [`Unsupported::AttributeAddress`], [`Unsupported::AttributeRegisters`] |
//! | `AST` | the same without the output check, and `data_reg == 255 \|\| data_reg + components <= 255` | the same |
//! | `IPA` | offset register `RZ` (`Take(32, 8) == 255`) unless `.OFFSET`; frequency `.PASS` or `.CONSTANT` (`frequency == 0 \|\| frequency == 1`); no predicate output (`Take(81, 3) == 7`); the address check of `ALD` and `AST` for one word (starquake's); for a generic attribute, the header (starquake's) | [`AttributeRejection::IpaOffsetRegister`], [`AttributeRejection::IpaFrequency`], [`AttributeRejection::IpaPredicate`], [`Unsupported::AttributeAddress`], [`AttributeRejection::IpaNoHeader`] |
//! | `S2R`, `S2UR` | a special register the PR reads: 0, 30, 31, 33 to 35, 37 to 39, 56 to 60 | [`Unsupported::SpecialRegister`] |
//! | `CS2R` | `SRZ` (`Take(72, 8) == 255`); for `.64`, `dest + 1 < 255` | [`AttributeRejection::Cs2rRegister`], [`AttributeRejection::Cs2rDestination`] |
//! | `AL2P` | one word (Eden: `bitsize != BitSize::B32` throws) | [`AttributeRejection::Al2pWidth`] |
//! | `PIXLD` | no predicate output (Eden: `dest_pred != IR::Pred::PT` throws); `.MY_INDEX` (Eden's only mode) | [`AttributeRejection::PixLdPredicate`], [`AttributeRejection::PixLdValue`] |
//!
//! `OUT` and `OUT.FINAL` are rejected by name as geometry output (owner's decision 13: geometry
//! and tessellation are deferred), `ISBERD` and `ISBEWR` as the internal stage buffer of mesh
//! and task programs (decision 11: after milestone 3), both [`AttributeRejection::Deferred`].
//! `LDTRAM`, which no reference translates, is rejected by its mnemonic
//! ([`Unsupported::Instruction`]). There is no partial translation.
//!
//! # Divergences from PR #4588
//!
//! | What | Here | Reference | Pinned by |
//! |---|---|---|---|
//! | `IPA.OFFSET` (location 2), with its offset register | `InterpolateAtOffset` of the attribute at the register's bits, and of the position's W at the same offset for a perspective input (a guess, after NAK's `load_frag_w` at the same offset, `nak_nir_lower_fs_inputs.c`); the spec has the mode carried in the IR for the lowering ("Interpolation at an offset or a sample", the gap table) | Rejected: "native IPA sample location" (`Take(76, 2) <= 1`) and "native IPA offset register" | `tests::ipa_offset_is_carried_in_the_ir` |
//! | `IPA` of a generic attribute without the header | Rejected, [`AttributeRejection::IpaNoHeader`] ([`super::translate_flow`] takes the header as an option) | Always has the header (`env.SPH()`) | `tests::ipa_of_a_generic_needs_the_header` |
//! | `ISBERD`, `ISBEWR` | Rejected by name (decision 11) | Translated: `ISBERD` from global memory, an indexed input or the map; `ISBEWR` only as `.o.attr.32` | `tests::deferred_instructions_are_rejected_by_name` |
//! | `AL2P` | Translated from Eden's Maxwell `AL2P`; the output bit is read and has no effect, as Eden reads none | No translator: rejected | `tests::al2p_adds_the_offset_register` |
//! | `PIXLD` | `.MY_INDEX` translated from Eden's Maxwell `PIXLD` | No translator: rejected | `tests::pixld_my_index_reads_the_sample_id` |
//! | `IPA`'s attribute address | Checked as `ALD` and `AST` check theirs, for one word: not word aligned, or starting at or past `0x400`, is [`Unsupported::AttributeAddress`]. The decoder's byte address (an 8-bit word number times four) cannot fail it; `OpIpa`'s public fields can | No check: the PR reads an 8-bit word number (`Take(64, 8)`), which cannot leave the attribute space | `tests::ipa_addresses_outside_the_attribute_space_are_rejected` |
//! | `AL2P`'s address | Unsigned, the 10-bit SM70 field the decoder reads (NAK `sm70_encode.rs` @ 52238762, `impl SM70Op for OpAL2P`: `set_field(40..50, self.addr)`), so at most `0x3ff` and never negative | Eden's Maxwell `AL2P` reads an 11-bit signed field (`BitField<20, 11, s64> offset`) and adds it sign-extended (`Imm32(static_cast<u32>(al2p.offset.Value()))`); the PR has no `AL2P` | `tests::al2p_adds_the_offset_register` (address `0x3ff`) |
//! | `S2UR` under a guard | The guard reads a uniform predicate (`GetUniformPred`), as the PR's `classify.h` says of `S2UR` ("its guard predicate is a uniform predicate"), and the instruction is a conditional block (the translation driver's "Guards") | The guard read through `UPred`, and the uniform register selecting on its old value (`Visitor::UR`) | `super::driver::tests::a_guard_on_a_uniform_instruction_reads_a_uniform_predicate` |
//! | `S2UR` | Writes the uniform register class (`SetUniformRegister`) | Writes the uniform register packed into Maxwell's register file (`native.h`, `UniformLayout`) | `tests::s2r_and_s2ur_read_each_special_register` |
//!
//! The port follows the PR's native translator (`ops_attr.cpp`, `ops_misc.cpp`), not its
//! Maxwell lowering (`lower_to_maxwell.h`), which refuses `IPA.CENTROID` (`case 0x326`: bits
//! 76..80 must be 0 or 4) and `CS2R.32` (`case 0x805`: bit 80 must be set) where the native
//! translator accepts both (`ipa_frequency_and_centroid_change_nothing`,
//! `cs2r_reads_zero_into_one_or_two_registers`).
//!
//! Eden's Maxwell `S2R` reads more registers than the PR (the invocation id and info, the
//! packed thread id, the thread-kill flag, the Y direction, and zero for every other one); they
//! are rejected here, as in the PR. `IPA.OFFSET` follows the spec's gap table, not the PR's
//! rejection, and `AL2P` and `PIXLD` come from Eden: the owner accepted both as built (decision
//! log, 2026-11-19, "Owner rulings on the wave 44 and 45 questions", row 5); the W at the
//! offset stays a guess (the divergence table).

#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use super::{ATTRIBUTE_SPACE_END, Failure, Translator, Unsupported, imm32, register_run, require};
use crate::decode::{
    Gpr, InterpFreq, InterpLoc, Op, OpAL2P, OpALd, OpASt, OpCS2R, OpIpa, OpPixLd, OpS2R, PixVal,
    Pred, S2RDst, SpecialReg,
};
use crate::info::attribute;
use crate::ir::{
    ATTRIBUTE_COMPONENTS, ATTRIBUTE_WORD_BYTES, Attribute, GENERIC_0_X, Imm, Opcode, Operand,
    POSITION_W, Value,
};
use crate::sph::PixelImap;

/// Why a deferred instruction is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Deferral {
    /// Geometry output (owner's decision 13: geometry and tessellation are deferred).
    Geometry,
    /// The internal stage buffer of mesh and task programs (decision 11: after milestone 3).
    MeshAndTask,
}

impl Display for Deferral {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Geometry => "geometry programs are deferred",
            Self::MeshAndTask => "mesh and task programs come after milestone 3",
        })
    }
}

/// A form of the attribute and system-value group that is not translated (see the module docs
/// for the check each comes from).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AttributeRejection {
    /// An instruction of a deferred stage, by name.
    #[error("{mnemonic} ({deferral})")]
    Deferred {
        /// The instruction's mnemonic.
        mnemonic: &'static str,
        /// Why it is deferred.
        deferral: Deferral,
    },
    /// `IPA` with an offset register at a location other than `.OFFSET` (PR: "native IPA offset
    /// register", `Take(32, 8) == 255`).
    #[error("an ipa with offset register {0} outside .offset")]
    IpaOffsetRegister(Gpr),
    /// `IPA.STATE` (PR: "native IPA interpolation frequency", `frequency == 0 || frequency ==
    /// 1`).
    #[error("an ipa{0}")]
    IpaFrequency(InterpFreq),
    /// `IPA` with a predicate output (PR: "native IPA predicate output", `Take(81, 3) == 7`).
    #[error("an ipa writing predicate {0}")]
    IpaPredicate(Pred),
    /// `IPA` of a generic attribute when the translator was given no header (starquake's: the
    /// PR always has the header).
    #[error("an ipa of generic attribute {0} without the program's header")]
    IpaNoHeader(Attribute),
    /// `CS2R` of a register other than `SRZ` (PR: "native special register pair",
    /// `Take(72, 8) == 255`).
    #[error("a cs2r of {0}")]
    Cs2rRegister(SpecialReg),
    /// `CS2R.64` whose second register would be `RZ` or past it (PR: "native special register
    /// pair destination", `dest + 1 < 255`).
    #[error("a cs2r.64 to {0}")]
    Cs2rDestination(Gpr),
    /// `AL2P` of more than one word (Eden throws when `bitsize != BitSize::B32`).
    #[error("an al2p of {0} words")]
    Al2pWidth(u8),
    /// `PIXLD` with a predicate output (Eden: "Destination predicate").
    #[error("a pixld writing predicate {0}")]
    PixLdPredicate(Pred),
    /// `PIXLD` of a value other than `.MY_INDEX` (Eden: "Mode").
    #[error("a pixld{0}")]
    PixLdValue(PixVal),
}

impl From<AttributeRejection> for Failure {
    fn from(what: AttributeRejection) -> Self {
        Self::Unsupported(what.into())
    }
}

/// The last attribute word Eden's `IsGeneric` counts (`Generic31X = 156`; `IsGeneric`:
/// `attribute <= Attribute::Generic31X`).
pub(super) use crate::info::attribute::{FOG_COORDINATE, GENERIC_31_X};
/// The legacy colours, `ColorFrontDiffuseR = 160` to `ColorBackSpecularA = 175` (Eden's
/// `IsLegacyAttribute`).
pub(super) const LEGACY_COLORS: std::ops::RangeInclusive<u16> =
    attribute::COLOR_FRONT_DIFFUSE_R..=attribute::COLOR_BACK_SPECULAR_A;
/// The fixed-function texture coordinates, `FixedFncTexture0S = 192` to
/// `FixedFncTexture9Q = 231` (`IsLegacyAttribute`).
pub(super) const LEGACY_TEXTURES: std::ops::RangeInclusive<u16> =
    attribute::FIXED_FNC_TEXTURE_0_S..=attribute::FIXED_FNC_TEXTURE_9_Q;
/// How many registers `CS2R.64` writes.
pub(super) const CS2R_PAIR: u32 = 2;
/// The bits of `1.0`, which the PR writes for the W scale factors (`ops_misc.cpp`, `S2R`: "the
/// 32-bit float 1.0 is an unverified assumption").
pub(super) const ONE_F32: u32 = 0x3f80_0000;

/// `SR_WSCALEFACTOR_XY = 30` (Eden's `SpecialRegister`; the PR: "assumed from the
/// Maxwell/Volta register numbering, not in NAK").
pub(super) const WSCALEFACTOR_XY: SpecialReg = SpecialReg(30);
/// `SR_WSCALEFACTOR_Z = 31`.
pub(super) const WSCALEFACTOR_Z: SpecialReg = SpecialReg(31);
/// `SR_EQMASK = 56`.
pub(super) const EQ_MASK: SpecialReg = SpecialReg(56);
/// `SR_LTMASK = 57`.
pub(super) const LT_MASK: SpecialReg = SpecialReg(57);
/// `SR_LEMASK = 58`.
pub(super) const LE_MASK: SpecialReg = SpecialReg(58);
/// `SR_GTMASK = 59`.
pub(super) const GT_MASK: SpecialReg = SpecialReg(59);
/// `SR_GEMASK = 60`.
pub(super) const GE_MASK: SpecialReg = SpecialReg(60);

/// What `S2R` reads for a special register (PR #4588's `S2R` switch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Special {
    /// An operation with no arguments and a `u32` result.
    Op(Opcode),
    /// Component `n` of a `u32x3` operation (Eden's emitter: `CompositeExtract(.., n)`).
    Component(Opcode, u32),
    /// A constant.
    Constant(u32),
}

/// The value the PR's `S2R` reads for `sr`, or `None` when it reads none.
fn special(sr: SpecialReg) -> Option<Special> {
    let component = |opcode, base: SpecialReg| {
        Some(Special::Component(
            opcode,
            u32::from(sr.0.wrapping_sub(base.0)),
        ))
    };
    match sr {
        SpecialReg::LANE_ID => Some(Special::Op(Opcode::LaneId)),
        SpecialReg::TID_X | SpecialReg::TID_Y | SpecialReg::TID_Z => {
            component(Opcode::LocalInvocationId, SpecialReg::TID_X)
        }
        SpecialReg::CTAID_X | SpecialReg::CTAID_Y | SpecialReg::CTAID_Z => {
            component(Opcode::WorkgroupId, SpecialReg::CTAID_X)
        }
        WSCALEFACTOR_XY | WSCALEFACTOR_Z => Some(Special::Constant(ONE_F32)),
        EQ_MASK => Some(Special::Op(Opcode::SubgroupEqMask)),
        LT_MASK => Some(Special::Op(Opcode::SubgroupLtMask)),
        LE_MASK => Some(Special::Op(Opcode::SubgroupLeMask)),
        GT_MASK => Some(Special::Op(Opcode::SubgroupGtMask)),
        GE_MASK => Some(Special::Op(Opcode::SubgroupGeMask)),
        _ => None,
    }
}

/// Eden's `IsGeneric`: `Generic0X` through `Generic31X`.
fn is_generic(word: u16) -> bool {
    (GENERIC_0_X..=GENERIC_31_X).contains(&word)
}

/// Eden's `IsLegacyAttribute`.
fn is_legacy(word: u16) -> bool {
    LEGACY_COLORS.contains(&word) || word == FOG_COORDINATE || LEGACY_TEXTURES.contains(&word)
}

/// The PR's rule for a generic attribute's interpolation: the first component of the input map
/// that is not unused, perspective, or every component unused.
fn generic_is_perspective(map: [PixelImap; 4]) -> bool {
    let first = map
        .into_iter()
        .find(|&imap| imap != PixelImap::Unused)
        .unwrap_or(PixelImap::Unused);
    matches!(first, PixelImap::Perspective | PixelImap::Unused)
}

/// The group's translator: translates `op` when it is one of the group's, else `None`.
pub(super) fn translate(t: &mut Translator<'_>, op: Op) -> Option<Result<(), Failure>> {
    let deferred =
        |mnemonic, deferral| Err(AttributeRejection::Deferred { mnemonic, deferral }.into());
    Some(match op {
        Op::ALd(op) => t.ald(op),
        Op::ASt(op) => t.ast(op),
        Op::Ipa(op) => t.ipa(op),
        Op::AL2P(op) => t.al2p(op),
        Op::S2R(op) => t.s2r(op),
        Op::CS2R(op) => t.cs2r(op),
        Op::PixLd(op) => t.pixld(op),
        Op::Out(_) | Op::OutFinal(_) => deferred(op.mnemonic(), Deferral::Geometry),
        Op::Isberd(_) | Op::Isbewr(_) => deferred(op.mnemonic(), Deferral::MeshAndTask),
        Op::LdTram(_) => Err(Unsupported::Instruction(op.mnemonic()).into()),
        _ => return None,
    })
}

impl Translator<'_> {
    /// The address check `ALD`, `AST` and `IPA` share; returns the first attribute word.
    fn attribute_words(addr: u16, comps: u8) -> Result<u16, Failure> {
        let end = u32::from(comps)
            .checked_mul(ATTRIBUTE_WORD_BYTES)
            .and_then(|size| u32::from(addr).checked_add(size));
        require(
            u32::from(addr).is_multiple_of(ATTRIBUTE_WORD_BYTES)
                && end.is_some_and(|end| end <= ATTRIBUTE_SPACE_END),
            Unsupported::AttributeAddress { addr, comps },
        )?;
        Ok(word_of(addr))
    }

    /// `ALD`, with the PR's checks in its order: offset register, patch or physical, output,
    /// address, registers. The vertex register is read once, before the first word.
    fn ald(&mut self, op: OpALd) -> Result<(), Failure> {
        require(op.offset.is_zero(), Unsupported::IndexedAttribute)?;
        require(!op.patch && !op.phys, Unsupported::PatchOrPhysical)?;
        require(!op.output, Unsupported::OutputRead)?;
        let first = Self::attribute_words(op.addr, op.comps)?;
        require(
            register_run(op.dst, u32::from(op.comps)).is_some(),
            Unsupported::AttributeRegisters {
                first: op.dst,
                comps: op.comps,
            },
        )?;
        let vertex = self.get(op.vtx)?;
        for element in 0..op.comps {
            let attribute = Attribute(first.saturating_add(u16::from(element)));
            let value = self.op(
                Opcode::GetAttribute,
                &[Operand::Imm(Imm::Attribute(attribute)), vertex],
            )?;
            let reg = Gpr::new(op.dst.index().saturating_add(element));
            self.set_f32(reg, value)?;
        }
        Ok(())
    }

    /// `AST`, with the PR's checks in its order: offset register, patch or physical, address,
    /// registers. The vertex register is read once, before the first word.
    fn ast(&mut self, op: OpASt) -> Result<(), Failure> {
        require(op.offset.is_zero(), Unsupported::IndexedAttribute)?;
        require(!op.patch && !op.phys, Unsupported::PatchOrPhysical)?;
        let first = Self::attribute_words(op.addr, op.comps)?;
        // `RZ` as the data source stores zero for every component (`ops_misc.cpp`, `AST`).
        require(
            op.data.is_zero() || register_run(op.data, u32::from(op.comps)).is_some(),
            Unsupported::AttributeRegisters {
                first: op.data,
                comps: op.comps,
            },
        )?;
        let vertex = self.get(op.vtx)?;
        for element in 0..op.comps {
            let reg = if op.data.is_zero() {
                Gpr::RZ
            } else {
                Gpr::new(op.data.index().saturating_add(element))
            };
            let value = self.get_f32(reg)?;
            let attribute = Attribute(first.saturating_add(u16::from(element)));
            self.op(
                Opcode::SetAttribute,
                &[
                    Operand::Imm(Imm::Attribute(attribute)),
                    value.into(),
                    vertex,
                ],
            )?;
        }
        Ok(())
    }

    /// Whether `IPA` multiplies attribute `word` by the position's W (see the module docs).
    fn ipa_perspective(&self, word: u16) -> Result<bool, Failure> {
        if is_legacy(word) {
            return Ok(true);
        }
        if !is_generic(word) {
            return Ok(false);
        }
        let header = self
            .header
            .as_ref()
            .ok_or(AttributeRejection::IpaNoHeader(Attribute(word)))?;
        // Cannot fail: `word` is a generic word, so its index is below 32.
        let index = usize::from(word.saturating_sub(GENERIC_0_X) / ATTRIBUTE_COMPONENTS);
        let map = header
            .generic
            .get(index)
            .map_or([PixelImap::Unused; 4], |c| [c.x, c.y, c.z, c.w]);
        Ok(generic_is_perspective(map))
    }

    /// Reads attribute `word` as `IPA` does: at vertex 0, or at `offset` for `.OFFSET`.
    fn interpolate(&mut self, word: u16, offset: Option<Operand>) -> Result<Value, Failure> {
        let attribute = Operand::Imm(Imm::Attribute(Attribute(word)));
        match offset {
            None => self.op(Opcode::GetAttribute, &[attribute, imm32(0)]),
            Some(offset) => self.op(Opcode::InterpolateAtOffset, &[attribute, offset]),
        }
    }

    /// `IPA`, with the PR's checks in its order (the offset register only outside `.OFFSET`),
    /// then the address check `ALD` and `AST` share (starquake's), then the header for a generic
    /// attribute.
    fn ipa(&mut self, op: OpIpa) -> Result<(), Failure> {
        if op.loc != InterpLoc::Offset {
            require_attr(
                op.offset.is_zero(),
                AttributeRejection::IpaOffsetRegister(op.offset),
            )?;
        }
        require_attr(
            matches!(op.freq, InterpFreq::Pass | InterpFreq::Constant),
            AttributeRejection::IpaFrequency(op.freq),
        )?;
        require_attr(op.pred.is_true(), AttributeRejection::IpaPredicate(op.pred))?;
        // Starquake's: the PR reads an 8-bit word number (`Take(64, 8)`), which cannot fail
        // this; the decoder's byte address cannot either, but `OpIpa`'s fields are public.
        let word = Self::attribute_words(op.addr, 1)?;
        let perspective = self.ipa_perspective(word)?;
        let offset = if op.loc == InterpLoc::Offset {
            Some(self.get(op.offset)?)
        } else {
            None
        };
        let mut value = self.interpolate(word, offset)?;
        if perspective {
            let w = self.interpolate(POSITION_W, offset)?;
            value = self.op(Opcode::FPMul32, &[value.into(), w.into()])?;
        }
        self.set_f32(op.dst, value)
    }

    /// `AL2P`: the offset register plus the address, as Eden's Maxwell `AL2P` adds them, with
    /// the address read as the decoder reads it: unsigned, 10 bits, so never negative where
    /// Eden's is signed (see the module docs' divergence table).
    fn al2p(&mut self, op: OpAL2P) -> Result<(), Failure> {
        require_attr(op.comps == 1, AttributeRejection::Al2pWidth(op.comps))?;
        let base = self.get(op.offset)?;
        let value = self.op(Opcode::IAdd32, &[base, imm32(u32::from(op.addr))])?;
        self.set(op.dst, value.into())
    }

    /// `S2R` and `S2UR`: the special register is checked first, whatever the destination.
    fn s2r(&mut self, op: OpS2R) -> Result<(), Failure> {
        let Some(special) = special(op.sr) else {
            return Err(Unsupported::SpecialRegister(op.sr).into());
        };
        let skip = match op.dst {
            S2RDst::Reg(dst) => dst.is_zero(),
            S2RDst::Uniform(dst) => dst.is_zero(),
        };
        if skip {
            return Ok(());
        }
        let value = match special {
            Special::Op(opcode) => self.op(opcode, &[])?.into(),
            Special::Component(opcode, component) => {
                let vector = self.op(opcode, &[])?;
                let extract = self.op(
                    Opcode::CompositeExtractU32x3,
                    &[vector.into(), imm32(component)],
                )?;
                extract.into()
            }
            Special::Constant(bits) => imm32(bits),
        };
        match op.dst {
            S2RDst::Reg(dst) => self.set(dst, value),
            S2RDst::Uniform(dst) => self.set_uniform(dst, value),
        }
    }

    /// `CS2R`: only `SRZ`, which reads zero.
    fn cs2r(&mut self, op: OpCS2R) -> Result<(), Failure> {
        require_attr(
            op.sr == SpecialReg::SRZ,
            AttributeRejection::Cs2rRegister(op.sr),
        )?;
        if op.wide {
            require_attr(
                register_run(op.dst, CS2R_PAIR).is_some(),
                AttributeRejection::Cs2rDestination(op.dst),
            )?;
        }
        self.set(op.dst, imm32(0))?;
        if op.wide {
            self.set(Gpr::new(op.dst.index().saturating_add(1)), imm32(0))?;
        }
        Ok(())
    }

    /// `PIXLD.MY_INDEX`: the sample's index.
    fn pixld(&mut self, op: OpPixLd) -> Result<(), Failure> {
        require_attr(
            op.pred.is_true(),
            AttributeRejection::PixLdPredicate(op.pred),
        )?;
        require_attr(
            op.val == PixVal::MyIndex,
            AttributeRejection::PixLdValue(op.val),
        )?;
        if op.dst.is_zero() {
            return Ok(());
        }
        let value = self.op(Opcode::SampleId, &[])?;
        self.set(op.dst, value.into())
    }
}

/// The attribute word at byte address `addr`.
fn word_of(addr: u16) -> u16 {
    // Cannot truncate: a quotient of a `u16` fits one.
    u16::try_from(u32::from(addr) / ATTRIBUTE_WORD_BYTES).unwrap_or(u16::MAX)
}

/// [`require`] for this group's rejections.
fn require_attr(ok: bool, what: AttributeRejection) -> Result<(), Failure> {
    require(ok, what.into())
}

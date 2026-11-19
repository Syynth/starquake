//! Slice 6f, texture and surface: texture samples (`TEX`), texel fetches (`TLD`), gathers
//! (`TLD4`), level-of-detail queries (`TMML`), samples with explicit gradients (`TXD`), header
//! queries (`TXQ`), and surface loads, stores and atomics (`SULD`, `SUST`, `SUATOM` with its
//! reduction form `SURED`). Every instruction of the group is decoded; none is left a shell.
//!
//! # Sources
//!
//! Shapes follow NAK's `Op*` structs and encodings NAK's `impl SM70Op` blocks, for SM 8.6, which
//! is below SM 10.0 and at or above SM 8.0 wherever NAK's encoder branches on the model
//! (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs` and `sm70_encode.rs` @ 52238762, MIT,
//! Copyright 2022 Collabora, Ltd.), as each constant says. Each form's known-bit mask is checked
//! against PR #4588's row for it (`eden-pr4588/src/shader_recompiler/frontend/ampere/`
//! `ampere_isa_table.h` @ e5fb3b97, [`REFERENCE_ROWS`]).
//!
//! **From PR #4588 alone, each needing an `nvdisasm` check** (spec, "The SM86 decoder";
//! verification E2): the three `.UR` forms, `TEX.UR` (`0xf60`), `TLD4.UR` (`0xf63`) and
//! `TLD.UR` (`0xf66`), whose handle is a uniform register at bits 40..46 and which require bit
//! 91 (`ops_texture.cpp`, `TEX`, `TLD` and `TLD4`: `bindless = op == 0xf6x`, `Take(40, 6)`,
//! `Require(Take(91, 1) == 1, "native bindless texture handle")`); and the `.F16` bit 79 of
//! `TEX` and `TLD` ("bit 79 selects half precision results", "bit 79: half precision results
//! (.F16)"), which NAK's `OpTld` encoder names only in a comment ("bits 79..81: .F16"), and the
//! `.CL` bit 77 of `TLD` (`Require(Take(77, 2) == 0, "native texel fetch multisample or
//! clamp")`), which NAK's `OpTld` also names only in a comment ("bit 77: .CL", below SM 12.0).
//! NAK writes neither bit: each is a guess about the hardware until the `nvdisasm` check.
//!
//! The memory ordering, eviction priority, memory type and atomic types and operations the
//! surface instructions carry ([`MemOrder`], [`MemEvictionPriority`], [`MemType`],
//! [`AtomType`], [`AtomOp`] with [`AtomCmpSrc`]) are NAK's, which its memory instructions share
//! (`set_mem_order`, `set_eviction_priority`, `set_mem_type`, `set_atom_type`,
//! `set_atom_op`). They are defined once, in slice 6e's `op::memory`, with their fields and
//! their text, and re-exported here; so the surface instructions print them as the memory
//! instructions do (`.s8`, nothing for 32 bits and for a weak order, `.strong.sm`, nothing for
//! `.u32` atomics, `.cas` for the compare-and-swap), in `nvdisasm`'s style as NAK's
//! `test_ld_st_atom` shows it, not in NAK's own printer's (`.i8`, `.b32`, `.weak`,
//! `.strong.cta`, `.u32`, `.cmpexch.packed`). On surfaces that text is still a guess.
//!
//! # Decisions taken here
//!
//! - **Handles.** A texture instruction names its texture header in a constant buffer (the
//!   `0xb6x` forms, NAK's `TexRef::CBuf`), in its source registers (the `.B` forms `0x36x`,
//!   NAK's `TexRef::Bindless`, which set bit 59), or in a uniform register (the `.UR` forms,
//!   PR #4588's). One op covers all of an instruction's forms; [`TexRef`] tells them apart.
//!   `TMML`, `TXD` and `TXQ` have no `.UR` form in either reference.
//! - **Fields NAK writes as constants or only on one model.** `SUATOM`'s `.BA` bit 72 (NAK:
//!   `set_bit(72, false); // .BA`) is read as [`OpSuAtom::ba`]; `SURED`'s destination field,
//!   where NAK writes `RZ` (`set_dst(&self.dst)` with `Dst::None`), is read as
//!   [`OpSuAtom::dst`]. The derivative mode is one bit below SM 10.0 (`set_tex_ndv(77, ..)`:
//!   `Auto` or `NonDivergent`), read as `ndv`; the offset mode of `TEX`, `TLD` and `TXD` is one
//!   bit (`set_bit(76, offset_mode == TexOffsetMode::AddOffI)`), read as `offset`; `TLD4`'s is
//!   two bits ([`TexOffsetMode`]).
//! - **`SURED`** (`0x3a0`) is [`OpSuAtom`] with [`OpSuAtom::reduce`] set, as NAK models it
//!   (`impl SM70Op for OpSuAtom`: `if self.dst.is_none() { set_opcode(0x3a0); .. }`), and the
//!   compare-and-swap form `0x396` is [`OpSuAtom`] with [`AtomOp::CmpExch`] and
//!   [`AtomCmpSrc::Packed`], the only source NAK's encoder takes for it.
//! - **Channel masks** are read as any four bits ([`ChannelMask`]): NAK's `ChannelMask::new`
//!   refuses 0 and its surface encoder takes only 1, 3 and 15 (`set_image_channel_mask`), and
//!   PR #4588's translators refuse other masks too (`Require(mask != 0, ..)`, `SUST`'s 1, 3, 7
//!   and 15); those are compilers' and translators' limits, which slices 12f and 13 decide on,
//!   not facts of the encoding.
//! - **Texel fetches with an implicit level of detail** decode: NAK asserts `TLD`'s mode is
//!   explicit (`assert!(self.lod_mode.is_explicit_lod())`) and PR #4588's translator requires
//!   `.LZ` or `.LL`; the field still names every [`TexLodMode`], and the refusal is the
//!   translator's.
//!
//! # Divergences from PR #4588's table
//!
//! | Rows | What this decoder does | Pinned by |
//! |---|---|---|
//! | Bit 89 of `TLD4.UR` (`0xf63`) | Rejected: neither reference names it (NAK's component field is bits 87..89, `set_field(87..89, self.comp)`), and PR #4588's lowering refuses a word with it set (`lower_to_maxwell.h`, `case 0xf63`: `insn.Bits(89, 2) != 0` fails "native uniform gather mode") | `decode::tests::texture::tld4_ur_rejects_bit_89` |
//! | `SULD` `0xb97`, `SUST` `0xb9b` and `SUATOM` `0x593`, surface forms with a constant-buffer handle | Unknown opcodes: no reference gives their full layout (NAK encodes none of them; PR #4588's translator reads them, `ops_texture.cpp`, but the slice's scope is NAK's forms and the `.UR` forms) | `decode::tests::texture::surface_rows_left_undecoded_are_unknown_opcodes` |
//! | Bit 79 (`.F16`) of `TEX.B` (`0x361`) | Decoded as [`OpTex::f16`]: the form adds the bit, which PR #4588's translator reads in every `TEX` form it accepts, `0x361` included (`Visitor::TEX`: `register_handle = op == 0x361`, then `half = Take(79, 1) != 0`); its row for `0x361` leaves it out | `decode::tests::texture::f16_and_clamp_in_every_form_that_names_them`, `decode::tests::framework::added_bits_decode_print_and_round_trip` |
//! | Bit 79 (`.F16`) of `TLD.B` (`0x367`) | Decoded as [`OpTld::f16`]: the form adds the bit, which NAK's `OpTld` encoder names for every handle in a comment ("bits 79..81: .F16"); PR #4588's row for `0x367` leaves it out, and its translator does not accept `0x367` (`Visitor::TLD` requires bit 59 clear outside `0xf66`) | as above |
//! | Bit 80 of `TEX` and `TLD` (all six forms), inside NAK's `OpTld` comment "bits 79..81: .F16" | Not added: it stays unknown ([`DecodeError::UnknownBit`]). NAK's range is half-open, so it spans bits 79 and 80, but NAK writes neither bit and gives no meaning to a two-bit value; PR #4588 reads `.F16` as one bit (`Take(79, 1)` in `Visitor::TEX` and `Visitor::TLD`) and no row of its table has bit 80 in these forms. Taking bit 80 would invent a field; an `nvdisasm` check would settle it | `decode::tests::texture::f16_is_bit_79_alone_and_bit_80_stays_unknown` |
//! | Bit 77 (`.CL`) of `TLD` (`0xb66`) and `TLD.B` (`0x367`) | Decoded as [`OpTld::clamp`]: the forms add the bit, which NAK's `OpTld` encoder names for every handle in a comment ("bit 77: .CL", below SM 12.0); PR #4588's rows leave it out but for `0xf66`, though its translator names it in every form it accepts (`Require(Take(77, 2) == 0, "native texel fetch multisample or clamp")`) | as above |
//!
//! Every other form's mask is its row's exactly, and the rows above that decode a bit differ
//! from their forms by exactly the bits listed (bit 80 adds nothing: form and row agree that it
//! is unknown; `decode::tests::masks_match_the_reference_table`;
//! `decode::tests::framework::added_bits_are_the_listed_cases` lists every added bit and the
//! reference that names it). The rows themselves differ between forms of one instruction (bit
//! 79 is in the constant-buffer and `.UR` rows of `TEX` and `TLD` only, bit 77 in `TLD`'s `.UR`
//! row only: `decode::tests::texture::rows_leave_out_the_added_texture_bits`); the forms add the
//! bits so that every form of an instruction decodes the same flags.
//!
//! **Not added: `.F16` in `TLD4`.** Wave 42's note on #356 and #430's scope list bit 79 of the
//! `TLD4` forms as a bit one reference names and the other leaves out, but neither reference
//! names a `.F16` for `TLD4` on SM 7.0 and later: NAK's `OpTld4` has no such field or comment
//! (its encoder comments only "bit 77: .CL", which falls inside `TLD4`'s offset mode, bits
//! 76..78), every PR #4588 `TLD4` row leaves bit 79 out, `Visitor::TLD4` does not take it, and
//! its lowering refuses a word with any of bits 76..80 set in `0xb63` and `0xf63`
//! (`insn.Bits(76, 5)`). Adding it would invent a protocol fact, so bit 79 stays unknown in all
//! three `TLD4` forms (pinned by `f16_and_clamp_in_every_form_that_names_them`). **Owner
//! question**: whether to add it from another source (an `nvdisasm` run, or Maxwell's
//! `TLD4.F16`, a different encoding).
//!
//! # Values the decoder refuses
//!
//! A field whose value names nothing is [`DecodeError::UnknownValue`]: a texture dimension 6
//! (NAK's `set_tex_dim` uses 0 to 5 and 7), an image dimension 6 or 7 (`set_image_dim`: 0 to
//! 5), an eviction priority 6 or 7 (`set_eviction_priority`: 0 to 5), a level-of-detail mode 6
//! or 7 (`set_tex_lod_mode`: 0 to 5), `TLD4`'s offset mode 3 (0 to 2), `TXQ`'s query 3 (0 to
//! 2), a memory type 7 (`set_mem_type`: 0 to 6), an atomic type 7 (`set_atom_type` below SM
//! 9.0: 0 to 6), a memory order other than 0, 4, 5, 7 and 10 (`set_mem_order` from SM 8.0), and
//! `SUATOM`'s atomic operation 9 to 15 (`set_atom_op`: 0 to 8). Each instruction reads its
//! refusable fields in ascending bit order, so the lowest refused field is the one reported.
//!
//! # Not decided here
//!
//! The text of the bindless `TEX`, `TLD`, `TLD4`, `TMML`, `TXD` and `TXQ` cases NAK's
//! `test_texture` checks (`.scr`, `.b`, every level-of-detail mode, `TLD4`'s `.g` and offset
//! modes, `.ef`, `.nodep`, the fault predicate, both destinations, both sources, `2d`, the mask
//! in hexadecimal and the three `TXQ` queries) is `nvdisasm`'s. Everything else is a guess in
//! `nvdisasm`'s style (`print`'s module docs): the other dimension names, the constant-buffer
//! and uniform-register handle operands, `.aoffi` on `TEX`, `TLD` and `TXD`, `.ndv`, `.dc`,
//! `.f16`, `.ms`, `.cl`, the eviction priorities other than `.ef`, a fault predicate of `pt`,
//! and the text of every surface instruction.

#[cfg(test)]
mod generate;

use std::fmt::{self, Display, Formatter};

use super::super::DecodeError;
use super::super::fields::{DST, mask_of};
use super::super::form::{DecodeFn, Form, Reader};
use super::super::instruction::Field;
use super::super::operand::{Gpr, Pred, UniformReg};
use super::super::print::flag;
use super::control::PRED_DST;
pub use super::memory::{AtomCmpSrc, AtomOp, AtomType, MemEvictionPriority, MemOrder, MemType};
use super::{Named, Op, memory, shells};
#[cfg(test)]
use {
    super::super::form::{Namer, ReferenceRow},
    proptest::strategy::BoxedStrategy,
};

// ---------------------------------------------------------------------------------------------
// Fields every texture instruction shares.

/// The first register operand of every texture and surface instruction (NAK:
/// `set_reg_src(24..32, &self.srcs[0])`, `&self.src` in `OpTxq`, `&self.coord` in the surface
/// instructions).
pub(crate) const REG_A: Field = Field::new(24, Gpr::BITS);
/// The second register operand (`set_reg_src(32..40, &self.srcs[1])`, `&self.data` in `SUST`
/// and `SUATOM`).
pub(crate) const REG_B: Field = Field::new(32, Gpr::BITS);
/// A constant-buffer handle's word offset (`set_tex_cb_ref(40..59, cb)`:
/// `v.set_field(0..14, cb.offset / 4)`).
pub(crate) const TEX_CBUF_OFFSET: Field = Field::new(40, 14);
/// The size of [`TEX_CBUF_OFFSET`]'s unit in bytes: the divisor of `cb.offset / 4`, after
/// `assert!(cb.offset % 4 == 0)`.
pub(crate) const TEX_CBUF_UNIT: u16 = 4;
/// A constant-buffer handle's buffer index (`set_tex_cb_ref`: `v.set_field(14..19, cb.idx)`).
pub(crate) const TEX_CBUF_INDEX: Field = Field::new(54, 5);
/// The `.B` bit of the forms whose handle is in the source registers (`set_bit(59, true); //
/// .B`): a fixed bit of those forms.
pub(crate) const TEX_BINDLESS: Field = Field::bit(59);
/// The uniform register holding the handle of the `.UR` forms (PR #4588, `ops_texture.cpp`:
/// `handle_register = Take(40, 6)`). From PR #4588 alone: needs an `nvdisasm` check.
pub(crate) const TEX_UNIFORM_HANDLE: Field = Field::new(40, UniformReg::BITS);
/// The fixed bit of the `.UR` forms (PR #4588: `Require(Take(91, 1) == 1, "native bindless
/// texture handle")`). From PR #4588 alone.
pub(crate) const TEX_UNIFORM: Field = Field::bit(91);
/// The scalar (`.SCR`) bit of `TEX`, `TLD` and `TLD4` (`set_bit(60, self.scalar)`).
pub(crate) const TEX_SCALAR: Field = Field::bit(60);
/// The texture dimension of `TEX`, `TLD`, `TLD4`, `TMML` and `TXD` (`set_tex_dim(61..64, ..)`:
/// [`TexDim`]).
pub(crate) const TEX_DIM: Field = Field::new(61, 3);
/// The second destination of every texture instruction (`set_reg(64..72, reg)` for
/// `self.dsts[1]`, else `set_field(64..72, 255_u8)`).
pub(crate) const TEX_DST2: Field = Field::new(64, Gpr::BITS);
/// The channel mask of every texture instruction and of the formatted surface forms
/// (`set_tex_channel_mask(72..76, ..)`, `set_image_channel_mask(72..76, ..)`).
pub(crate) const CHANNEL_MASK: Field = Field::new(72, 4);
/// The `.AOFFI` bit of `TEX`, `TLD` and `TXD` below SM 10.0 (`set_bit(76, self.offset_mode ==
/// TexOffsetMode::AddOffI)`).
pub(crate) const TEX_OFFSET: Field = Field::bit(76);
/// The `.NDV` bit of `TEX` and `TMML` below SM 10.0 (`set_tex_ndv(77, self.deriv_mode)`).
pub(crate) const TEX_NDV: Field = Field::bit(77);
/// The depth-compare (`.DC`) bit of `TEX` and `TLD4` (`set_bit(78, self.z_cmpr)`).
pub(crate) const TEX_DC: Field = Field::bit(78);
/// The `.F16` bit of `TEX` and `TLD`, in every form (PR #4588, `ops_texture.cpp`: "bit 79
/// selects half precision results"; NAK's `OpTld` encoder: "bits 79..81: .F16", a comment;
/// module docs). Not written by NAK: needs an `nvdisasm` check. One bit, as PR #4588 reads it
/// (`Take(79, 1)`): NAK's comment spans bits 79 and 80 (a half-open range), but neither
/// reference reads, writes or names a meaning for bit 80, so bit 80 stays unknown (module
/// docs, "Divergences from PR #4588's table";
/// `decode::tests::texture::f16_is_bit_79_alone_and_bit_80_stays_unknown`).
pub(crate) const TEX_F16: Field = Field::bit(79);
/// The `.CL` bit of `TLD`, in every form (NAK's `OpTld` encoder: "bit 77: .CL", a comment;
/// PR #4588: `Require(Take(77, 2) == 0, "native texel fetch multisample or clamp")`). Not
/// written by NAK: needs an `nvdisasm` check.
pub(crate) const TLD_CLAMP: Field = Field::bit(77);
/// `TLD`'s multisample (`.MS`) bit below SM 12.0 (`set_bit(78, self.is_ms)`).
pub(crate) const TLD_MS: Field = Field::bit(78);
/// `TLD4`'s offset mode (`set_field(76..78, ..)`: [`TexOffsetMode`]).
pub(crate) const TLD4_OFFSET: Field = Field::new(76, 2);
/// `TLD4`'s gathered component (`set_field(87..89, self.comp)`: [`TexChannel`]).
pub(crate) const TLD4_COMP: Field = Field::new(87, 2);
/// The eviction priority of `TEX`, `TLD`, `TLD4`, `TXD` and the surface instructions
/// (`set_eviction_priority`: `set_field(84..87, ..)`: [`MemEvictionPriority`]): the memory
/// group's field, defined once there.
pub(crate) const EVICT: Field = memory::MEM_EVICTION;
/// The level-of-detail mode of `TEX` and `TLD` below SM 10.0 (`set_tex_lod_mode(87..90, ..)`:
/// [`TexLodMode`]).
pub(crate) const TEX_LOD: Field = Field::new(87, 3);
/// The `.NODEP` bit of every texture instruction (`set_bit(90, self.nodep)`).
pub(crate) const TEX_NODEP: Field = Field::bit(90);
/// `TXQ`'s query (`set_field(62..64, ..)`: [`TexQuery`]).
pub(crate) const TXQ_QUERY: Field = Field::new(62, 2);
/// The fault (`.SPARSE` residency) predicate of `TEX`, `TLD`, `TLD4`, `TXD`, `SULD` and
/// `SUATOM` (`set_pred_dst(81..84, &self.fault)`); `PT` writes nothing. The same field as
/// [`PRED_DST`].
pub(crate) const FAULT: Field = PRED_DST;

// ---------------------------------------------------------------------------------------------
// Fields of the surface instructions.

/// The image dimension of the surface instructions (`set_image_dim(61..64, ..)`:
/// [`ImageDim`]).
pub(crate) const IMAGE_DIM: Field = Field::new(61, 3);
/// The handle register of the surface instructions (`set_reg_src(64..72, &self.handle)`).
pub(crate) const SU_HANDLE: Field = Field::new(64, Gpr::BITS);
/// The memory type of the binary (`.D`) surface forms (`set_mem_type(73..76, ..)`:
/// [`MemType`]): the memory group's field, defined once there.
pub(crate) const MEM_TYPE: Field = memory::MEM_TYPE;
/// The memory order of the surface instructions from SM 8.0 (`set_mem_order`:
/// `set_field(77..81, ..)`: [`MemOrder`]): the memory group's field, defined once there.
pub(crate) const MEM_ORDER: Field = memory::MEM_ORDER;
/// `SUATOM`'s `.BA` bit (`set_bit(72, false); // .BA`).
pub(crate) const SUATOM_BA: Field = Field::bit(72);
/// `SUATOM`'s atomic type below SM 9.0 (`set_atom_type(.., true)`: `set_field(73..76, ..)`:
/// [`AtomType`]): the memory group's field, defined once there.
pub(crate) const ATOM_TYPE: Field = memory::ATOM_TYPE;
/// `SUATOM`'s atomic operation (`set_atom_op(87..91, ..)`: [`AtomOp`]): the memory group's
/// `ATOM_OP`.
pub(crate) const SUATOM_OP: Field = memory::ATOM_OP;
/// `SURED`'s atomic operation (`set_atom_op(87..90, ..)`): three bits, so not
/// [`AtomOp::Exch`] (8); the memory group's `RED_OP`.
pub(crate) const SURED_OP: Field = memory::RED_OP;

// ---------------------------------------------------------------------------------------------
// Opcodes.

/// The opcodes of one texture instruction's forms, one per kind of handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TexOpcodes {
    /// The form with a constant-buffer handle (NAK: `TexRef::CBuf`).
    pub(crate) cbuf: u16,
    /// The `.B` form, with the handle in the source registers (NAK: `TexRef::Bindless`).
    pub(crate) bindless: u16,
    /// The `.UR` form, with the handle in a uniform register (PR #4588 only), if any.
    pub(crate) uniform: Option<u16>,
}

impl TexOpcodes {
    /// The opcode of the form for `tex`, if the instruction has one (used by the assembler).
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) const fn of(self, tex: TexRef) -> Option<u16> {
        match tex {
            TexRef::CBuf(_) => Some(self.cbuf),
            TexRef::Bindless => Some(self.bindless),
            TexRef::Uniform(_) => self.uniform,
        }
    }
}

/// `TEX.UR` (PR #4588's `TEX.UR` row and `ops_texture.cpp`: `bindless = op == 0xf60`). From
/// PR #4588 alone: needs an `nvdisasm` check.
pub(crate) const TEX_UR_OPCODE: u16 = 0xf60;
/// `TLD.UR` (PR #4588's `TLD.UR` row; `ops_texture.cpp`: `op == 0xf66`). From PR #4588 alone.
pub(crate) const TLD_UR_OPCODE: u16 = 0xf66;
/// `TLD4.UR` (PR #4588's `TLD4.UR` row; `ops_texture.cpp`: `op == 0xf63`). From PR #4588 alone.
pub(crate) const TLD4_UR_OPCODE: u16 = 0xf63;

/// `TEX` (`impl SM70Op for OpTex`: `set_opcode(0xb60)` and `set_opcode(0x361)`), and
/// [`TEX_UR_OPCODE`].
pub(crate) const TEX_OPCODES: TexOpcodes = TexOpcodes {
    cbuf: 0xb60,
    bindless: 0x361,
    uniform: Some(TEX_UR_OPCODE),
};
/// `TLD` (`impl SM70Op for OpTld`: `0xb66` and `0x367`), and [`TLD_UR_OPCODE`].
pub(crate) const TLD_OPCODES: TexOpcodes = TexOpcodes {
    cbuf: 0xb66,
    bindless: 0x367,
    uniform: Some(TLD_UR_OPCODE),
};
/// `TLD4` (`impl SM70Op for OpTld4`: `0xb63` and `0x364`), and [`TLD4_UR_OPCODE`].
pub(crate) const TLD4_OPCODES: TexOpcodes = TexOpcodes {
    cbuf: 0xb63,
    bindless: 0x364,
    uniform: Some(TLD4_UR_OPCODE),
};
/// `TMML` (`impl SM70Op for OpTmml`: `0xb69` and `0x36a`).
pub(crate) const TMML_OPCODES: TexOpcodes = TexOpcodes {
    cbuf: 0xb69,
    bindless: 0x36a,
    uniform: None,
};
/// `TXD` (`impl SM70Op for OpTxd`: `0xb6c` and `0x36d`).
pub(crate) const TXD_OPCODES: TexOpcodes = TexOpcodes {
    cbuf: 0xb6c,
    bindless: 0x36d,
    uniform: None,
};
/// `TXQ` (`impl SM70Op for OpTxq`: `0xb6f` and `0x370`).
pub(crate) const TXQ_OPCODES: TexOpcodes = TexOpcodes {
    cbuf: 0xb6f,
    bindless: 0x370,
    uniform: None,
};

/// `SULD.P`, formatted (`impl SM70Op for OpSuLd`: `ImageAccess::Formatted` =>
/// `set_opcode(0x998)`).
pub(crate) const SULD_FORMATTED_OPCODE: u16 = 0x998;
/// `SULD.D`, binary (`ImageAccess::Binary` => `set_opcode(0x99a)`).
pub(crate) const SULD_BINARY_OPCODE: u16 = 0x99a;
/// `SUST.P` (`impl SM70Op for OpSuSt`: `set_opcode(0x99c)`).
pub(crate) const SUST_FORMATTED_OPCODE: u16 = 0x99c;
/// `SUST.D` (`set_opcode(0x99e)`).
pub(crate) const SUST_BINARY_OPCODE: u16 = 0x99e;
/// `SUATOM` (`impl SM70Op for OpSuAtom`: `set_opcode(0x394)` with a destination).
pub(crate) const SUATOM_OPCODE: u16 = 0x394;
/// `SUATOM.CAS` (`AtomOp::CmpExch(cmp_src)` => `set_opcode(0x396)`).
pub(crate) const SUATOM_CAS_OPCODE: u16 = 0x396;
/// `SURED` (`self.dst.is_none()` => `set_opcode(0x3a0)`).
pub(crate) const SURED_OPCODE: u16 = 0x3a0;

// ---------------------------------------------------------------------------------------------
// Encoded enumerations.

/// Defines a public enumeration whose variants each have a fixed encoding and printed text:
/// `ALL` in declaration order, `bits` and a `Display` that writes the text.
macro_rules! encoded {
    (
        $(#[$meta:meta])*
        $name:ident {
            $($(#[$vmeta:meta])* $variant:ident = $bits:literal => $text:literal,)*
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$vmeta])* $variant,)*
        }

        impl $name {
            /// Every value, in declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),*];

            /// The encoding.
            pub const fn bits(self) -> u64 {
                match self {
                    $(Self::$variant => $bits,)*
                }
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                f.write_str(match self {
                    $(Self::$variant => $text,)*
                })
            }
        }
    };
}

encoded! {
    /// A texture's dimension (NAK's `TexDim`, encoded by `set_tex_dim`). The text is
    /// `nvdisasm`'s `2d` for [`TexDim::D2`] (NAK's `test_texture`); the others are guesses.
    TexDim {
        /// 0: one-dimensional.
        D1 = 0 => "1d",
        /// 4: a one-dimensional array.
        Array1D = 4 => "array_1d",
        /// 1: two-dimensional.
        D2 = 1 => "2d",
        /// 5: a two-dimensional array.
        Array2D = 5 => "array_2d",
        /// 2: three-dimensional.
        D3 = 2 => "3d",
        /// 3: a cube.
        Cube = 3 => "cube",
        /// 7: a cube array.
        ArrayCube = 7 => "array_cube",
    }
}

encoded! {
    /// How `TEX` and `TLD` choose the level of detail (NAK's `TexLodMode`, encoded by
    /// `set_tex_lod_mode`, and its printer, which `test_texture` confirms for every mode).
    TexLodMode {
        /// 0: from the derivatives.
        Auto = 0 => "",
        /// 1: level zero (`.LZ`).
        Zero = 1 => ".lz",
        /// 2: a bias added to the automatic level (`.LB`).
        Bias = 2 => ".lb",
        /// 3: an explicit level (`.LL`).
        Lod = 3 => ".ll",
        /// 4: the automatic level, clamped below (`.LC`, the minimum-LOD clamp).
        Clamp = 4 => ".lc",
        /// 5: a bias and a clamp.
        BiasClamp = 5 => ".lb.lc",
    }
}

encoded! {
    /// `TLD4`'s texel offsets (NAK's `TexOffsetMode`, encoded by its `OpTld4` encoder, and its
    /// printer, which `test_texture` confirms).
    TexOffsetMode {
        /// 0: none.
        None = 0 => "",
        /// 1: one offset for the whole footprint (`.AOFFI`).
        AddOffI = 1 => ".aoffi",
        /// 2: an offset per gathered texel (`.PTP`).
        PerPx = 2 => ".ptp",
    }
}

encoded! {
    /// The component `TLD4` gathers (NAK's `comp`, 0 to 3; printed `.g` for 1 by
    /// `test_texture`, the others guesses).
    TexChannel {
        /// 0: red.
        R = 0 => ".r",
        /// 1: green.
        G = 1 => ".g",
        /// 2: blue.
        B = 2 => ".b",
        /// 3: alpha.
        A = 3 => ".a",
    }
}

encoded! {
    /// What `TXQ` reads (NAK's `TexQuery`, encoded by its `OpTxq` encoder; `test_texture`
    /// confirms each text).
    TexQuery {
        /// 0: the dimensions.
        Dimension = 0 => "tex_header_dimension",
        /// 1: the texture type.
        TextureType = 1 => "tex_header_texture_type",
        /// 2: the sample positions.
        SamplerPos = 2 => "tex_header_sampler_pos",
    }
}

encoded! {
    /// A surface's dimension (NAK's `ImageDim`, encoded by `set_image_dim`); every text is a
    /// guess.
    ImageDim {
        /// 0: one-dimensional.
        D1 = 0 => "1d",
        /// 1: a one-dimensional buffer.
        Buffer1D = 1 => "1d_buffer",
        /// 2: a one-dimensional array.
        Array1D = 2 => "array_1d",
        /// 3: two-dimensional.
        D2 = 3 => "2d",
        /// 4: a two-dimensional array.
        Array2D = 4 => "array_2d",
        /// 5: three-dimensional.
        D3 = 5 => "3d",
    }
}

/// The channels a texture instruction writes or a formatted surface access moves: four bits,
/// red at bit 0 to alpha at bit 3 (NAK's `ChannelMask`). Any four bits decode (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChannelMask(u8);

impl ChannelMask {
    /// Red, green, blue and alpha.
    pub const RGBA: Self = Self(0xf);

    /// The mask `bits`, if it fits in four bits.
    pub const fn new(bits: u8) -> Option<Self> {
        if bits < 1 << CHANNEL_MASK.width {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// The four bits.
    pub const fn bits(self) -> u8 {
        self.0
    }
}

impl Display for ChannelMask {
    /// `nvdisasm`'s hexadecimal mask (`0x7` in NAK's `test_texture`).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

/// A texture header's location in a constant buffer (NAK's `TexCBufRef`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TexCBufRef {
    /// The buffer, 0 to 31.
    pub index: u8,
    /// The byte offset, a multiple of [`TEX_CBUF_UNIT`] below `0x10000`.
    pub offset: u16,
}

/// Where a texture instruction finds its texture header (NAK's `TexRef`, without `Bound`,
/// which SM 7.0 and later lack, and with PR #4588's uniform-register handle).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexRef {
    /// In a constant buffer (the `0xb6x` forms).
    CBuf(TexCBufRef),
    /// In the source registers (the `.B` forms, `0x36x`).
    Bindless,
    /// In a uniform register (the `.UR` forms, `0xf6x`; PR #4588 alone).
    Uniform(UniformReg),
}

/// The kind of a texture form's handle.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Handle {
    CBuf,
    Bindless,
    Uniform,
}

impl Handle {
    /// The handle's bits.
    const fn known(self) -> u128 {
        match self {
            Self::CBuf => mask_of(&[TEX_CBUF_OFFSET, TEX_CBUF_INDEX]),
            Self::Bindless => 0,
            Self::Uniform => TEX_UNIFORM_HANDLE.mask(),
        }
    }

    /// The handle's fixed bits.
    const fn fixed(self) -> u128 {
        match self {
            Self::CBuf => 0,
            Self::Bindless => TEX_BINDLESS.mask(),
            Self::Uniform => TEX_UNIFORM.mask(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Readers.

/// Reads `field` as one of `all`, by `bits`; a value none of them has is
/// [`DecodeError::UnknownValue`].
fn read_value<T: Copy>(
    reader: &mut Reader,
    field: Field,
    all: &[T],
    bits: fn(T) -> u64,
) -> Result<T, DecodeError> {
    let value = reader.field(field);
    all.iter()
        .copied()
        .find(|&item| bits(item) == value)
        .ok_or_else(|| reader.unknown_value(field, value))
}

/// Reads the handle of a form of kind `kind`, the form's fixed bit included.
fn read_tex_ref(reader: &mut Reader, kind: Handle) -> TexRef {
    match kind {
        Handle::CBuf => TexRef::CBuf(TexCBufRef {
            index: reader.byte(TEX_CBUF_INDEX),
            // Cannot overflow: fourteen bits of words are at most 0xfffc bytes.
            offset: u16::try_from(reader.field(TEX_CBUF_OFFSET))
                .unwrap_or(u16::MAX)
                .saturating_mul(TEX_CBUF_UNIT),
        }),
        Handle::Bindless => {
            // A fixed bit of the form: the decoder has checked that it is set.
            reader.flag(TEX_BINDLESS);
            TexRef::Bindless
        }
        Handle::Uniform => {
            reader.flag(TEX_UNIFORM);
            TexRef::Uniform(reader.uniform_reg(TEX_UNIFORM_HANDLE))
        }
    }
}

/// Reads both destinations: 16..24 first, then [`TEX_DST2`].
fn read_dsts(reader: &mut Reader) -> [Gpr; 2] {
    [reader.gpr(DST), reader.gpr(TEX_DST2)]
}

/// Reads both source registers.
fn read_srcs(reader: &mut Reader) -> [Gpr; 2] {
    [reader.gpr(REG_A), reader.gpr(REG_B)]
}

/// Reads the fault predicate.
fn read_fault(reader: &mut Reader) -> Pred {
    // Cannot fail: the field is `Pred::BITS` wide.
    Pred::new(reader.byte(FAULT)).unwrap_or(Pred::PT)
}

/// Reads the channel mask.
fn read_mask(reader: &mut Reader) -> ChannelMask {
    // Cannot fail: the field is four bits wide.
    ChannelMask::new(reader.byte(CHANNEL_MASK)).unwrap_or(ChannelMask::RGBA)
}

fn read_dim(reader: &mut Reader) -> Result<TexDim, DecodeError> {
    read_value(reader, TEX_DIM, TexDim::ALL, TexDim::bits)
}

fn read_eviction(reader: &mut Reader) -> Result<MemEvictionPriority, DecodeError> {
    read_value(
        reader,
        EVICT,
        MemEvictionPriority::ALL,
        MemEvictionPriority::bits,
    )
}

fn read_lod(reader: &mut Reader) -> Result<TexLodMode, DecodeError> {
    read_value(reader, TEX_LOD, TexLodMode::ALL, TexLodMode::bits)
}

// ---------------------------------------------------------------------------------------------
// Printing.

/// The operands every texture instruction but `TXQ` prints, in `nvdisasm`'s order (NAK's
/// `test_texture`: `p0, r2, r0, r1, r3, 2d, 0x7` for the fault `p0`, `dsts` `[r0, r2]` and
/// `srcs` `[r1, r3]`): the fault predicate if the instruction has one, the second destination,
/// the first, both sources, the handle unless it is in the sources (a guess), the dimension and
/// the mask.
struct TexOperands {
    fault: Option<Pred>,
    dsts: [Gpr; 2],
    srcs: [Gpr; 2],
    tex: TexRef,
    dim: TexDim,
    channel_mask: ChannelMask,
}

impl Display for TexOperands {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(fault) = self.fault {
            write!(f, "{fault}, ")?;
        }
        let [dst, dst2] = self.dsts;
        let [a, b] = self.srcs;
        write!(f, "{dst2}, {dst}, {a}, {b}{}, ", HandleOperand(self.tex))?;
        write!(f, "{}, {}", self.dim, self.channel_mask)
    }
}

/// A handle that is not in the source registers, as an operand after the sources: NAK's
/// `TexRef` printer for a constant buffer (`c[0x1][0x40]`) and the register for `.UR`; nothing
/// for `.B`. A guess at `nvdisasm`'s text.
struct HandleOperand(TexRef);

impl Display for HandleOperand {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            TexRef::CBuf(TexCBufRef { index, offset }) => {
                write!(f, ", c[{index:#x}][{offset:#x}]")
            }
            TexRef::Bindless => Ok(()),
            TexRef::Uniform(reg) => write!(f, ", {reg}"),
        }
    }
}

/// `.b` for a handle in the source registers (`test_texture`).
fn bindless(f: &mut Formatter<'_>, tex: TexRef) -> fmt::Result {
    flag(f, tex == TexRef::Bindless, ".b")
}

// ---------------------------------------------------------------------------------------------
// TEX.

/// `TEX`: samples a texture (NAK's `OpTex`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag is an independent one-bit field of the encoding, as in NAK's OpTex"
)]
pub struct OpTex {
    /// The destinations: the first at bits 16..24, the second at [`TEX_DST2`] (`RZ` for none).
    pub dsts: [Gpr; 2],
    /// The fault predicate (`.SPARSE` residency); `PT` writes nothing.
    pub fault: Pred,
    /// The texture header.
    pub tex: TexRef,
    /// The source registers (coordinates, level of detail, depth reference).
    pub srcs: [Gpr; 2],
    /// The dimension.
    pub dim: TexDim,
    /// The level-of-detail mode.
    pub lod_mode: TexLodMode,
    /// `.NDV`: assume non-divergent derivatives.
    pub ndv: bool,
    /// `.DC`: compare with a depth reference.
    pub z_cmpr: bool,
    /// `.AOFFI`: add an immediate offset from the sources.
    pub offset: bool,
    /// `.F16`: half-precision results (module docs).
    pub f16: bool,
    /// The eviction priority.
    pub eviction: MemEvictionPriority,
    /// `.NODEP`.
    pub nodep: bool,
    /// The channels written.
    pub channel_mask: ChannelMask,
    /// `.SCR`: scalar operand layout.
    pub scalar: bool,
}

impl Named for OpTex {
    const MNEMONIC: &'static str = "tex";
}

impl Display for OpTex {
    /// `test_texture`'s `tex{.scr}.b{lod}.ef.nodep p0, r2, r0, r1, r3, 2d, 0x7`, with the guessed
    /// `.aoffi`, `.ndv`, `.dc` and `.f16` after the level-of-detail mode, in NAK's printer's
    /// order.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("tex")?;
        flag(f, self.scalar, ".scr")?;
        bindless(f, self.tex)?;
        write!(f, "{}", self.lod_mode)?;
        flag(f, self.offset, ".aoffi")?;
        flag(f, self.ndv, ".ndv")?;
        flag(f, self.z_cmpr, ".dc")?;
        flag(f, self.f16, ".f16")?;
        write!(f, "{}", self.eviction)?;
        flag(f, self.nodep, ".nodep")?;
        let operands = TexOperands {
            fault: Some(self.fault),
            dsts: self.dsts,
            srcs: self.srcs,
            tex: self.tex,
            dim: self.dim,
            channel_mask: self.channel_mask,
        };
        write!(f, " {operands}")
    }
}

fn decode_tex(reader: &mut Reader, kind: Handle) -> Result<Op, DecodeError> {
    let dim = read_dim(reader)?;
    let eviction = read_eviction(reader)?;
    let lod_mode = read_lod(reader)?;
    let tex = read_tex_ref(reader, kind);
    Ok(Op::Tex(OpTex {
        dsts: read_dsts(reader),
        fault: read_fault(reader),
        tex,
        srcs: read_srcs(reader),
        dim,
        lod_mode,
        ndv: reader.flag(TEX_NDV),
        z_cmpr: reader.flag(TEX_DC),
        offset: reader.flag(TEX_OFFSET),
        f16: reader.flag(TEX_F16),
        eviction,
        nodep: reader.flag(TEX_NODEP),
        channel_mask: read_mask(reader),
        scalar: reader.flag(TEX_SCALAR),
    }))
}

// ---------------------------------------------------------------------------------------------
// TLD.

/// `TLD`: fetches texels by integer coordinates (NAK's `OpTld`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag is an independent one-bit field of the encoding, as in NAK's OpTld"
)]
pub struct OpTld {
    /// The destinations, as [`OpTex::dsts`].
    pub dsts: [Gpr; 2],
    /// The fault predicate.
    pub fault: Pred,
    /// The texture header.
    pub tex: TexRef,
    /// The source registers.
    pub srcs: [Gpr; 2],
    /// The dimension.
    pub dim: TexDim,
    /// `.MS`: a multisampled texture.
    pub is_ms: bool,
    /// `.CL`: a level-of-detail clamp (module docs).
    pub clamp: bool,
    /// The level-of-detail mode (NAK encodes only [`TexLodMode::Zero`] and
    /// [`TexLodMode::Lod`]).
    pub lod_mode: TexLodMode,
    /// `.AOFFI`.
    pub offset: bool,
    /// `.F16` (module docs).
    pub f16: bool,
    /// The eviction priority.
    pub eviction: MemEvictionPriority,
    /// `.NODEP`.
    pub nodep: bool,
    /// The channels written.
    pub channel_mask: ChannelMask,
    /// `.SCR`.
    pub scalar: bool,
}

impl Named for OpTld {
    const MNEMONIC: &'static str = "tld";
}

impl Display for OpTld {
    /// `test_texture`'s `tld{.scr}.b{lod}.ef.nodep p0, r2, r0, r1, r3, 2d, 0x7`, with the guessed
    /// `.aoffi`, `.cl`, `.ms` and `.f16` after the mode.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("tld")?;
        flag(f, self.scalar, ".scr")?;
        bindless(f, self.tex)?;
        write!(f, "{}", self.lod_mode)?;
        flag(f, self.offset, ".aoffi")?;
        flag(f, self.clamp, ".cl")?;
        flag(f, self.is_ms, ".ms")?;
        flag(f, self.f16, ".f16")?;
        write!(f, "{}", self.eviction)?;
        flag(f, self.nodep, ".nodep")?;
        let operands = TexOperands {
            fault: Some(self.fault),
            dsts: self.dsts,
            srcs: self.srcs,
            tex: self.tex,
            dim: self.dim,
            channel_mask: self.channel_mask,
        };
        write!(f, " {operands}")
    }
}

fn decode_tld(reader: &mut Reader, kind: Handle) -> Result<Op, DecodeError> {
    let dim = read_dim(reader)?;
    let eviction = read_eviction(reader)?;
    let lod_mode = read_lod(reader)?;
    let tex = read_tex_ref(reader, kind);
    Ok(Op::Tld(OpTld {
        dsts: read_dsts(reader),
        fault: read_fault(reader),
        tex,
        srcs: read_srcs(reader),
        dim,
        is_ms: reader.flag(TLD_MS),
        clamp: reader.flag(TLD_CLAMP),
        lod_mode,
        offset: reader.flag(TEX_OFFSET),
        f16: reader.flag(TEX_F16),
        eviction,
        nodep: reader.flag(TEX_NODEP),
        channel_mask: read_mask(reader),
        scalar: reader.flag(TEX_SCALAR),
    }))
}

// ---------------------------------------------------------------------------------------------
// TLD4.

/// `TLD4`: gathers one component of four texels (NAK's `OpTld4`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpTld4 {
    /// The destinations, as [`OpTex::dsts`].
    pub dsts: [Gpr; 2],
    /// The fault predicate.
    pub fault: Pred,
    /// The texture header.
    pub tex: TexRef,
    /// The source registers.
    pub srcs: [Gpr; 2],
    /// The dimension.
    pub dim: TexDim,
    /// The component gathered.
    pub comp: TexChannel,
    /// The texel offsets, per-texel ones included.
    pub offset_mode: TexOffsetMode,
    /// `.DC`.
    pub z_cmpr: bool,
    /// The eviction priority.
    pub eviction: MemEvictionPriority,
    /// `.NODEP`.
    pub nodep: bool,
    /// The channels written.
    pub channel_mask: ChannelMask,
    /// `.SCR`.
    pub scalar: bool,
}

impl Named for OpTld4 {
    const MNEMONIC: &'static str = "tld4";
}

impl Display for OpTld4 {
    /// `test_texture`'s `tld4{.scr}.g.b{offset}.ef.nodep p0, r2, r0, r1, r3, 2d, 0x7`, with the
    /// guessed `.dc` after the offset mode.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("tld4")?;
        flag(f, self.scalar, ".scr")?;
        write!(f, "{}", self.comp)?;
        bindless(f, self.tex)?;
        write!(f, "{}", self.offset_mode)?;
        flag(f, self.z_cmpr, ".dc")?;
        write!(f, "{}", self.eviction)?;
        flag(f, self.nodep, ".nodep")?;
        let operands = TexOperands {
            fault: Some(self.fault),
            dsts: self.dsts,
            srcs: self.srcs,
            tex: self.tex,
            dim: self.dim,
            channel_mask: self.channel_mask,
        };
        write!(f, " {operands}")
    }
}

fn decode_tld4(reader: &mut Reader, kind: Handle) -> Result<Op, DecodeError> {
    let dim = read_dim(reader)?;
    let offset_mode = read_value(reader, TLD4_OFFSET, TexOffsetMode::ALL, TexOffsetMode::bits)?;
    let eviction = read_eviction(reader)?;
    // Cannot fail: the field is two bits wide and every value names a channel.
    let comp = read_value(reader, TLD4_COMP, TexChannel::ALL, TexChannel::bits)?;
    let tex = read_tex_ref(reader, kind);
    Ok(Op::Tld4(OpTld4 {
        dsts: read_dsts(reader),
        fault: read_fault(reader),
        tex,
        srcs: read_srcs(reader),
        dim,
        comp,
        offset_mode,
        z_cmpr: reader.flag(TEX_DC),
        eviction,
        nodep: reader.flag(TEX_NODEP),
        channel_mask: read_mask(reader),
        scalar: reader.flag(TEX_SCALAR),
    }))
}

// ---------------------------------------------------------------------------------------------
// TMML and TXD.

/// `TMML`: queries the level of detail a sample would use (NAK's `OpTmml`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpTmml {
    /// The destinations, as [`OpTex::dsts`].
    pub dsts: [Gpr; 2],
    /// The texture header: a constant buffer or the sources (no `.UR` form).
    pub tex: TexRef,
    /// The source registers.
    pub srcs: [Gpr; 2],
    /// The dimension.
    pub dim: TexDim,
    /// `.NDV`.
    pub ndv: bool,
    /// `.NODEP`.
    pub nodep: bool,
    /// The channels written.
    pub channel_mask: ChannelMask,
}

/// `TXD`: samples with explicit gradients (NAK's `OpTxd`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpTxd {
    /// The destinations, as [`OpTex::dsts`].
    pub dsts: [Gpr; 2],
    /// The fault predicate.
    pub fault: Pred,
    /// The texture header: a constant buffer or the sources (no `.UR` form).
    pub tex: TexRef,
    /// The source registers.
    pub srcs: [Gpr; 2],
    /// The dimension.
    pub dim: TexDim,
    /// `.AOFFI`.
    pub offset: bool,
    /// The eviction priority.
    pub eviction: MemEvictionPriority,
    /// `.NODEP`.
    pub nodep: bool,
    /// The channels written.
    pub channel_mask: ChannelMask,
}

impl Named for OpTmml {
    const MNEMONIC: &'static str = "tmml";
}

impl Named for OpTxd {
    const MNEMONIC: &'static str = "txd";
}

impl Display for OpTmml {
    /// `test_texture`'s `tmml.b.lod.nodep r2, r0, r1, r3, 2d, 0x7`, with the guessed `.ndv`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("tmml")?;
        bindless(f, self.tex)?;
        f.write_str(".lod")?;
        flag(f, self.ndv, ".ndv")?;
        flag(f, self.nodep, ".nodep")?;
        let operands = TexOperands {
            fault: None,
            dsts: self.dsts,
            srcs: self.srcs,
            tex: self.tex,
            dim: self.dim,
            channel_mask: self.channel_mask,
        };
        write!(f, " {operands}")
    }
}

impl Display for OpTxd {
    /// `test_texture`'s `txd.b.ef.nodep p0, r2, r0, r1, r3, 2d, 0x7`, with the guessed
    /// `.aoffi`.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("txd")?;
        bindless(f, self.tex)?;
        flag(f, self.offset, ".aoffi")?;
        write!(f, "{}", self.eviction)?;
        flag(f, self.nodep, ".nodep")?;
        let operands = TexOperands {
            fault: Some(self.fault),
            dsts: self.dsts,
            srcs: self.srcs,
            tex: self.tex,
            dim: self.dim,
            channel_mask: self.channel_mask,
        };
        write!(f, " {operands}")
    }
}

fn decode_tmml(reader: &mut Reader, kind: Handle) -> Result<Op, DecodeError> {
    let dim = read_dim(reader)?;
    Ok(Op::Tmml(OpTmml {
        dsts: read_dsts(reader),
        tex: read_tex_ref(reader, kind),
        srcs: read_srcs(reader),
        dim,
        ndv: reader.flag(TEX_NDV),
        nodep: reader.flag(TEX_NODEP),
        channel_mask: read_mask(reader),
    }))
}

fn decode_txd(reader: &mut Reader, kind: Handle) -> Result<Op, DecodeError> {
    let dim = read_dim(reader)?;
    let eviction = read_eviction(reader)?;
    Ok(Op::Txd(OpTxd {
        dsts: read_dsts(reader),
        fault: read_fault(reader),
        tex: read_tex_ref(reader, kind),
        srcs: read_srcs(reader),
        dim,
        offset: reader.flag(TEX_OFFSET),
        eviction,
        nodep: reader.flag(TEX_NODEP),
        channel_mask: read_mask(reader),
    }))
}

// ---------------------------------------------------------------------------------------------
// TXQ.

/// `TXQ`: reads a field of a texture header (NAK's `OpTxq`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpTxq {
    /// The destinations, as [`OpTex::dsts`].
    pub dsts: [Gpr; 2],
    /// The texture header: a constant buffer or the source (no `.UR` form).
    pub tex: TexRef,
    /// The source register (the level of detail, or the handle for `.B`).
    pub src: Gpr,
    /// What it reads.
    pub query: TexQuery,
    /// `.NODEP`.
    pub nodep: bool,
    /// The channels written.
    pub channel_mask: ChannelMask,
}

impl Named for OpTxq {
    const MNEMONIC: &'static str = "txq";
}

impl Display for OpTxq {
    /// `test_texture`'s `txq.b.nodep r2, r0, r1, tex_header_dimension, 0x7`, with the guessed
    /// handle operand of the constant-buffer form before the query.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("txq")?;
        bindless(f, self.tex)?;
        flag(f, self.nodep, ".nodep")?;
        let [dst, dst2] = self.dsts;
        write!(
            f,
            " {dst2}, {dst}, {}{}, {}, {}",
            self.src,
            HandleOperand(self.tex),
            self.query,
            self.channel_mask
        )
    }
}

fn decode_txq(reader: &mut Reader, kind: Handle) -> Result<Op, DecodeError> {
    let query = read_value(reader, TXQ_QUERY, TexQuery::ALL, TexQuery::bits)?;
    Ok(Op::Txq(OpTxq {
        dsts: read_dsts(reader),
        tex: read_tex_ref(reader, kind),
        src: reader.gpr(REG_A),
        query,
        nodep: reader.flag(TEX_NODEP),
        channel_mask: read_mask(reader),
    }))
}

// ---------------------------------------------------------------------------------------------
// SULD, SUST and SUATOM.

/// How a surface access reads or writes texels (NAK's `ImageAccess`): picked by the opcode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageAccess {
    /// `.D`: raw memory of a type (the `0x99a` and `0x99e` forms).
    Binary(MemType),
    /// `.P`: converted through the surface's format, these channels (`0x998` and `0x99c`).
    Formatted(ChannelMask),
}

/// `SULD`: loads from a surface (NAK's `OpSuLd`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpSuLd {
    /// The first destination register; the others follow it.
    pub dst: Gpr,
    /// The fault predicate.
    pub fault: Pred,
    /// Binary or formatted.
    pub access: ImageAccess,
    /// The dimension.
    pub dim: ImageDim,
    /// The memory order.
    pub order: MemOrder,
    /// The eviction priority.
    pub eviction: MemEvictionPriority,
    /// The surface handle.
    pub handle: Gpr,
    /// The first coordinate register.
    pub coord: Gpr,
}

/// `SUST`: stores to a surface (NAK's `OpSuSt`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpSuSt {
    /// Binary or formatted.
    pub access: ImageAccess,
    /// The dimension.
    pub dim: ImageDim,
    /// The memory order.
    pub order: MemOrder,
    /// The eviction priority.
    pub eviction: MemEvictionPriority,
    /// The surface handle.
    pub handle: Gpr,
    /// The first coordinate register.
    pub coord: Gpr,
    /// The first data register.
    pub data: Gpr,
}

/// `SUATOM`, and `SURED` without a result: an atomic operation on a surface texel (NAK's
/// `OpSuAtom`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpSuAtom {
    /// The destination; for `SURED`, the field NAK writes `RZ` (module docs).
    pub dst: Gpr,
    /// The fault predicate.
    pub fault: Pred,
    /// The dimension.
    pub dim: ImageDim,
    /// The operation: [`AtomOp::CmpExch`] is the `0x396` form, and `SURED` takes only the
    /// reductions [`AtomOp::Add`] to [`AtomOp::Xor`] (three bits).
    pub atom_op: AtomOp,
    /// The type.
    pub atom_type: AtomType,
    /// The memory order.
    pub order: MemOrder,
    /// The eviction priority.
    pub eviction: MemEvictionPriority,
    /// The surface handle.
    pub handle: Gpr,
    /// The first coordinate register.
    pub coord: Gpr,
    /// The first data register.
    pub data: Gpr,
    /// `.BA` (NAK always writes it clear).
    pub ba: bool,
    /// `SURED` (`0x3a0`): no result is returned.
    pub reduce: bool,
}

impl Named for OpSuLd {
    const MNEMONIC: &'static str = "suld";
}

impl Named for OpSuSt {
    const MNEMONIC: &'static str = "sust";
}

impl Named for OpSuAtom {
    const MNEMONIC: &'static str = "suatom";
}

/// The modifiers of a surface load or store, all guesses in NAK's printer's order: `.d`, the
/// dimension and the memory type for a binary access, `.p` and the dimension for a formatted
/// one, then the order and the eviction priority.
fn surface_modifiers(
    f: &mut Formatter<'_>,
    access: ImageAccess,
    dim: ImageDim,
    order: MemOrder,
    eviction: MemEvictionPriority,
) -> fmt::Result {
    match access {
        ImageAccess::Binary(mem_type) => write!(f, ".d.{dim}{mem_type}")?,
        ImageAccess::Formatted(_) => write!(f, ".p.{dim}")?,
    }
    write!(f, "{order}{eviction}")
}

/// A formatted access's mask as the last operand, as the texture instructions print theirs;
/// nothing for a binary access.
struct AccessMask(ImageAccess);

impl Display for AccessMask {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            ImageAccess::Binary(_) => Ok(()),
            ImageAccess::Formatted(mask) => write!(f, ", {mask}"),
        }
    }
}

impl Display for OpSuLd {
    /// `suld`, the modifiers, then the fault, the destination, `[coord]`, the handle and a
    /// formatted access's mask (a guess at `nvdisasm`'s text).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("suld")?;
        surface_modifiers(f, self.access, self.dim, self.order, self.eviction)?;
        write!(
            f,
            " {}, {}, [{}], {}{}",
            self.fault,
            self.dst,
            self.coord,
            self.handle,
            AccessMask(self.access)
        )
    }
}

impl Display for OpSuSt {
    /// `sust`, the modifiers, then `[coord]`, the data, the handle and a formatted access's mask
    /// (a guess).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("sust")?;
        surface_modifiers(f, self.access, self.dim, self.order, self.eviction)?;
        write!(
            f,
            " [{}], {}, {}{}",
            self.coord,
            self.data,
            self.handle,
            AccessMask(self.access)
        )
    }
}

impl Display for OpSuAtom {
    /// NAK's `suatom.p{dim}{op}{type}{order}{evict}`, `sured` for the reduction, with the
    /// guessed `.ba`, then the fault, the destination, `[coord]`, the data and the handle (a
    /// guess).
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(if self.reduce { "sured" } else { "suatom" })?;
        write!(
            f,
            ".p.{}{}{}{}{}",
            self.dim, self.atom_op, self.atom_type, self.order, self.eviction
        )?;
        flag(f, self.ba, ".ba")?;
        write!(
            f,
            " {}, {}, [{}], {}, {}",
            self.fault, self.dst, self.coord, self.data, self.handle
        )
    }
}

fn read_image_dim(reader: &mut Reader) -> Result<ImageDim, DecodeError> {
    read_value(reader, IMAGE_DIM, ImageDim::ALL, ImageDim::bits)
}

fn read_order(reader: &mut Reader) -> Result<MemOrder, DecodeError> {
    read_value(reader, MEM_ORDER, MemOrder::ALL, MemOrder::bits)
}

/// Reads the access the opcode picks: a memory type for the binary forms, a channel mask for
/// the formatted ones.
fn read_access(reader: &mut Reader, binary: bool) -> Result<ImageAccess, DecodeError> {
    if binary {
        Ok(ImageAccess::Binary(read_value(
            reader,
            MEM_TYPE,
            MemType::ALL,
            MemType::bits,
        )?))
    } else {
        Ok(ImageAccess::Formatted(read_mask(reader)))
    }
}

fn decode_suld(reader: &mut Reader) -> Result<Op, DecodeError> {
    let dim = read_image_dim(reader)?;
    let binary = reader.opcode() == u64::from(SULD_BINARY_OPCODE);
    let access = read_access(reader, binary)?;
    let order = read_order(reader)?;
    let eviction = read_eviction(reader)?;
    Ok(Op::SuLd(OpSuLd {
        dst: reader.gpr(DST),
        fault: read_fault(reader),
        access,
        dim,
        order,
        eviction,
        handle: reader.gpr(SU_HANDLE),
        coord: reader.gpr(REG_A),
    }))
}

fn decode_sust(reader: &mut Reader) -> Result<Op, DecodeError> {
    let dim = read_image_dim(reader)?;
    let binary = reader.opcode() == u64::from(SUST_BINARY_OPCODE);
    let access = read_access(reader, binary)?;
    let order = read_order(reader)?;
    let eviction = read_eviction(reader)?;
    Ok(Op::SuSt(OpSuSt {
        access,
        dim,
        order,
        eviction,
        handle: reader.gpr(SU_HANDLE),
        coord: reader.gpr(REG_A),
        data: reader.gpr(REG_B),
    }))
}

fn decode_suatom(reader: &mut Reader) -> Result<Op, DecodeError> {
    let opcode = reader.opcode();
    let dim = read_image_dim(reader)?;
    let atom_type = read_value(reader, ATOM_TYPE, AtomType::ALL, AtomType::bits)?;
    let order = read_order(reader)?;
    let eviction = read_eviction(reader)?;
    let op_bits = |op: AtomOp| op.bits().unwrap_or(u64::MAX);
    let (atom_op, reduce) = if opcode == u64::from(SUATOM_CAS_OPCODE) {
        (AtomOp::CmpExch(AtomCmpSrc::Packed), false)
    } else if opcode == u64::from(SURED_OPCODE) {
        // Three bits: every value is one of the reductions, `Add` (0) to `Xor` (7).
        (
            read_value(reader, SURED_OP, &AtomOp::ENCODED, op_bits)?,
            true,
        )
    } else {
        (
            read_value(reader, SUATOM_OP, &AtomOp::ENCODED, op_bits)?,
            false,
        )
    };
    Ok(Op::SuAtom(OpSuAtom {
        dst: reader.gpr(DST),
        fault: read_fault(reader),
        dim,
        atom_op,
        atom_type,
        order,
        eviction,
        handle: reader.gpr(SU_HANDLE),
        coord: reader.gpr(REG_A),
        data: reader.gpr(REG_B),
        ba: reader.flag(SUATOM_BA),
        reduce,
    }))
}

// ---------------------------------------------------------------------------------------------
// The form table.

/// The fields of `TEX`'s forms besides the handle, `.F16` included in every form (module docs).
const TEX_FIELDS: u128 = mask_of(&[
    DST,
    REG_A,
    REG_B,
    TEX_F16,
    TEX_SCALAR,
    TEX_DIM,
    TEX_DST2,
    CHANNEL_MASK,
    TEX_OFFSET,
    TEX_NDV,
    TEX_DC,
    FAULT,
    EVICT,
    TEX_LOD,
    TEX_NODEP,
]);
/// The fields of `TLD`'s forms besides the handle, `.F16` and `.CL` included in every form
/// (module docs).
const TLD_FIELDS: u128 = mask_of(&[
    TEX_F16,
    TLD_CLAMP,
    DST,
    REG_A,
    REG_B,
    TEX_SCALAR,
    TEX_DIM,
    TEX_DST2,
    CHANNEL_MASK,
    TEX_OFFSET,
    TLD_MS,
    FAULT,
    EVICT,
    TEX_LOD,
    TEX_NODEP,
]);
/// The fields of `TLD4`'s forms besides the handle.
const TLD4_FIELDS: u128 = mask_of(&[
    DST,
    REG_A,
    REG_B,
    TEX_SCALAR,
    TEX_DIM,
    TEX_DST2,
    CHANNEL_MASK,
    TLD4_OFFSET,
    TEX_DC,
    FAULT,
    EVICT,
    TLD4_COMP,
    TEX_NODEP,
]);
/// The fields of `TMML`'s forms besides the handle.
const TMML_FIELDS: u128 = mask_of(&[
    DST,
    REG_A,
    REG_B,
    TEX_DIM,
    TEX_DST2,
    CHANNEL_MASK,
    TEX_NDV,
    TEX_NODEP,
]);
/// The fields of `TXD`'s forms besides the handle.
const TXD_FIELDS: u128 = mask_of(&[
    DST,
    REG_A,
    REG_B,
    TEX_DIM,
    TEX_DST2,
    CHANNEL_MASK,
    TEX_OFFSET,
    FAULT,
    EVICT,
    TEX_NODEP,
]);
/// The fields of `TXQ`'s forms besides the handle.
const TXQ_FIELDS: u128 = mask_of(&[DST, REG_A, TXQ_QUERY, TEX_DST2, CHANNEL_MASK, TEX_NODEP]);
/// The fields every surface instruction has.
const SU_FIELDS: u128 = mask_of(&[REG_A, IMAGE_DIM, SU_HANDLE, MEM_ORDER, EVICT]);
/// `SUATOM`'s fields besides its operation.
const SUATOM_FIELDS: u128 = SU_FIELDS | mask_of(&[DST, REG_B, FAULT, SUATOM_BA, ATOM_TYPE]);

/// A texture instruction's form with a `kind` handle: `fields` and the handle's bits, and the
/// handle's fixed bit.
const fn tex_form(opcode: u16, kind: Handle, fields: u128, decode: DecodeFn) -> Form {
    Form::new(opcode, fields | kind.known(), decode).with_fixed(kind.fixed())
}

/// `TEX`'s form with a `kind` handle.
const fn tex(opcode: u16, kind: Handle, decode: DecodeFn) -> Form {
    tex_form(opcode, kind, TEX_FIELDS, decode)
}

/// `TLD`'s form with a `kind` handle.
const fn tld(opcode: u16, kind: Handle, decode: DecodeFn) -> Form {
    tex_form(opcode, kind, TLD_FIELDS, decode)
}

/// The texture and surface group's forms.
pub(crate) const FORMS: &[Form] = &[
    tex(TEX_OPCODES.cbuf, Handle::CBuf, |reader| {
        decode_tex(reader, Handle::CBuf)
    }),
    tex(TEX_OPCODES.bindless, Handle::Bindless, |reader| {
        decode_tex(reader, Handle::Bindless)
    }),
    tex(TEX_UR_OPCODE, Handle::Uniform, |reader| {
        decode_tex(reader, Handle::Uniform)
    }),
    tld(TLD_OPCODES.cbuf, Handle::CBuf, |reader| {
        decode_tld(reader, Handle::CBuf)
    }),
    tld(TLD_OPCODES.bindless, Handle::Bindless, |reader| {
        decode_tld(reader, Handle::Bindless)
    }),
    tld(TLD_UR_OPCODE, Handle::Uniform, |reader| {
        decode_tld(reader, Handle::Uniform)
    }),
    tex_form(TLD4_OPCODES.cbuf, Handle::CBuf, TLD4_FIELDS, |reader| {
        decode_tld4(reader, Handle::CBuf)
    }),
    tex_form(
        TLD4_OPCODES.bindless,
        Handle::Bindless,
        TLD4_FIELDS,
        |reader| decode_tld4(reader, Handle::Bindless),
    ),
    tex_form(TLD4_UR_OPCODE, Handle::Uniform, TLD4_FIELDS, |reader| {
        decode_tld4(reader, Handle::Uniform)
    }),
    tex_form(TMML_OPCODES.cbuf, Handle::CBuf, TMML_FIELDS, |reader| {
        decode_tmml(reader, Handle::CBuf)
    }),
    tex_form(
        TMML_OPCODES.bindless,
        Handle::Bindless,
        TMML_FIELDS,
        |reader| decode_tmml(reader, Handle::Bindless),
    ),
    tex_form(TXD_OPCODES.cbuf, Handle::CBuf, TXD_FIELDS, |reader| {
        decode_txd(reader, Handle::CBuf)
    }),
    tex_form(
        TXD_OPCODES.bindless,
        Handle::Bindless,
        TXD_FIELDS,
        |reader| decode_txd(reader, Handle::Bindless),
    ),
    tex_form(TXQ_OPCODES.cbuf, Handle::CBuf, TXQ_FIELDS, |reader| {
        decode_txq(reader, Handle::CBuf)
    }),
    tex_form(
        TXQ_OPCODES.bindless,
        Handle::Bindless,
        TXQ_FIELDS,
        |reader| decode_txq(reader, Handle::Bindless),
    ),
    Form::new(
        SULD_FORMATTED_OPCODE,
        SU_FIELDS | mask_of(&[DST, FAULT, CHANNEL_MASK]),
        decode_suld,
    ),
    Form::new(
        SULD_BINARY_OPCODE,
        SU_FIELDS | mask_of(&[DST, FAULT, MEM_TYPE]),
        decode_suld,
    ),
    Form::new(
        SUST_FORMATTED_OPCODE,
        SU_FIELDS | mask_of(&[REG_B, CHANNEL_MASK]),
        decode_sust,
    ),
    Form::new(
        SUST_BINARY_OPCODE,
        SU_FIELDS | mask_of(&[REG_B, MEM_TYPE]),
        decode_sust,
    ),
    Form::new(
        SUATOM_OPCODE,
        SUATOM_FIELDS | SUATOM_OP.mask(),
        decode_suatom,
    ),
    Form::new(SUATOM_CAS_OPCODE, SUATOM_FIELDS, decode_suatom),
    Form::new(SURED_OPCODE, SUATOM_FIELDS | SURED_OP.mask(), decode_suatom),
];

/// The bit PR #4588's `TLD4.UR` row marks known that the form rejects (module docs).
#[cfg(test)]
pub(crate) const TLD4_UR_DROPPED: u128 = 1 << 89;

/// PR #4588's rows for the group's forms, in [`FORMS`] order
/// (`eden-pr4588/src/shader_recompiler/frontend/ampere/ampere_isa_table.h` @ e5fb3b97; each
/// row's four words, low word first), with the bits each form drops or requires (module docs).
#[cfg(test)]
pub(crate) const REFERENCE_ROWS: &[ReferenceRow] = &[
    ReferenceRow::new(0xb60, [0xffff_ffff, 0xf7ff_ffff, 0x07fe_ffff, 0xffff_fe00]),
    ReferenceRow::new(0x361, [0xffff_ffff, 0xf800_00ff, 0x07fe_7fff, 0xffff_fe00])
        .adding(TEX_F16.mask(), Namer::Pr4588Translator)
        .fixing(TEX_BINDLESS.mask()),
    ReferenceRow::new(0xf60, [0xffff_ffff, 0xf000_3fff, 0x0ffe_ffff, 0xffff_fe00])
        .fixing(TEX_UNIFORM.mask()),
    ReferenceRow::new(0xb66, [0xffff_ffff, 0xf7ff_ffff, 0x07fe_dfff, 0xffff_fe00])
        .adding(TLD_CLAMP.mask(), Namer::NakComment),
    ReferenceRow::new(0x367, [0xffff_ffff, 0xf800_00ff, 0x07fe_5fff, 0xffff_fe00])
        .adding(TEX_F16.mask() | TLD_CLAMP.mask(), Namer::NakComment)
        .fixing(TEX_BINDLESS.mask()),
    ReferenceRow::new(0xf66, [0xffff_ffff, 0xf000_3fff, 0x0ffe_ffff, 0xffff_fe00])
        .fixing(TEX_UNIFORM.mask()),
    ReferenceRow::new(0xb63, [0xffff_ffff, 0xf7ff_ffff, 0x05fe_7fff, 0xffff_fe00]),
    ReferenceRow::new(0x364, [0xffff_ffff, 0xf800_00ff, 0x05fe_7fff, 0xffff_fe00])
        .fixing(TEX_BINDLESS.mask()),
    ReferenceRow::new(0xf63, [0xffff_ffff, 0xf000_3fff, 0x0ffe_7fff, 0xffff_fe00])
        .dropping(TLD4_UR_DROPPED)
        .fixing(TEX_UNIFORM.mask()),
    ReferenceRow::new(0xb69, [0xffff_ffff, 0xe7ff_ffff, 0x0400_2fff, 0xffff_fe00]),
    ReferenceRow::new(0x36a, [0xffff_ffff, 0xe800_00ff, 0x0400_2fff, 0xffff_fe00])
        .fixing(TEX_BINDLESS.mask()),
    ReferenceRow::new(0xb6c, [0xffff_ffff, 0xe7ff_ffff, 0x047e_1fff, 0xffff_fe00]),
    ReferenceRow::new(0x36d, [0xffff_ffff, 0xe800_00ff, 0x047e_1fff, 0xffff_fe00])
        .fixing(TEX_BINDLESS.mask()),
    ReferenceRow::new(0xb6f, [0xffff_ffff, 0xc7ff_ff00, 0x0400_0fff, 0xffff_fe00]),
    ReferenceRow::new(0x370, [0xffff_ffff, 0xc800_0000, 0x0400_0fff, 0xffff_fe00])
        .fixing(TEX_BINDLESS.mask()),
    ReferenceRow::new(0x998, [0xffff_ffff, 0xe000_0000, 0x007f_efff, 0xffff_fe00]),
    ReferenceRow::new(0x99a, [0xffff_ffff, 0xe000_0000, 0x007f_eeff, 0xffff_fe00]),
    ReferenceRow::new(0x99c, [0xff00_ffff, 0xe000_00ff, 0x0071_efff, 0xffff_fe00]),
    ReferenceRow::new(0x99e, [0xff00_ffff, 0xe000_00ff, 0x0071_eeff, 0xffff_fe00]),
    ReferenceRow::new(0x394, [0xffff_ffff, 0xe000_00ff, 0x07ff_efff, 0xffff_fe00]),
    ReferenceRow::new(0x396, [0xffff_ffff, 0xe000_00ff, 0x007f_efff, 0xffff_fe00]),
    ReferenceRow::new(0x3a0, [0xffff_ffff, 0xe000_00ff, 0x03ff_efff, 0xffff_fe00]),
];

/// Generators of the group's decoded operations, one per operation (`generate`).
#[cfg(test)]
pub(crate) fn any_ops() -> Vec<BoxedStrategy<Op>> {
    generate::all()
}

shells! {}

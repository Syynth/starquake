//! Slice 12f: the texture and surface instructions decoded by slice 6f (`TEX`, `TLD`, `TLD4`,
//! `TMML`, `TXD`, `TXQ`, `SULD`, `SUST`, `SUATOM` with `SURED`) translated into IR texture and
//! image operations with a resource reference.
//!
//! # Sources
//!
//! Ported by behaviour from PR #4588's native frontend
//! (`eden-pr4588/src/shader_recompiler/frontend/ampere/ops_texture.cpp` @ e5fb3b97,
//! GPL-3.0-or-later, Copyright 2026 Eden Emulator Project): `Visitor::TEX`, `TLD`, `TLD4`,
//! `TXQ`, `TXD`, `TMML`, `SULD`, `SUST`, `SUATOM`, `UseDescriptorBuffer`, `DimensionOf`,
//! `ImageDimensionOf` and `SurfaceCoordinates`; which opcodes reach them is `visitor.cpp`,
//! `Visitor::Run`. Cross-checked with Eden's Maxwell translators
//! (`eden/src/shader_recompiler/frontend/maxwell/translate/impl/` @ cf654052,
//! GPL-2.0-or-later): `texture_fetch.cpp`, `texture_load.cpp`, `texture_gather.cpp`,
//! `texture_gradient.cpp`, `texture_query.cpp`, `texture_mipmap_level.cpp`,
//! `surface_load_store.cpp` and `surface_atomic_operations.cpp`. Two facts neither Eden source
//! gives come from NAK (`mesa-nouveau/src/nouveau/compiler/nak_nir_lower_tex.c` @ 52238762, MIT,
//! Copyright 2022 Collabora, Ltd.; see "Level-of-detail clamp"), and the per-texel gather
//! emulation follows the GLSL definition of `textureGatherOffsets` and Mesa's
//! `nir_lower_tex.c` (`lower_tg4_offsets`, MIT): see "Per-texel gather offsets".
//!
//! # What each instruction becomes
//!
//! Each instruction's operands form one list in the PR's order (the array index first, then the
//! coordinates, the level of detail or bias, then the depth reference), spread over its two
//! source registers `A` and `B` as the PR spreads them (`ops_texture.cpp`, `TEX`: "a single
//! register each for two operands, otherwise a pair at A followed by the remainder at B" in the
//! scalar layout; every coordinate from `A` and one extra operand at `B` in the vector layout).
//! Coordinates are read as floats (`BitCastF32U32` of the register), an array index as the low
//! 16 bits converted to a float (`ConvertF32U16` of `BitFieldUExtract(raw, 0, 16)`); a texel
//! fetch reads its coordinates and array index as integers. Then:
//!
//! - `TEX`: `ImageSampleImplicitLod`, `ImageSampleExplicitLod` (`.LZ` with level 0.0, `.LL`),
//!   or their `Dref` forms with `.DC`; a depth comparison's result is one float, written to the
//!   red, green and blue channels, with alpha 1.0.
//! - `TLD`: `ImageFetch` with integer coordinates and level (0 for `.LZ`).
//! - `TLD4`: `ImageGather` or `ImageGatherDref`, with the gathered component in the texture
//!   information.
//! - `TXD`: `ImageGradient` with two derivative pairs (`num_derivatives` 2).
//! - `TMML`: `ImageQueryLod`; each selected result (0: the level relative to the base, 1: the
//!   absolute level, the host's pair in the other order) is multiplied by 256, converted to a
//!   signed (0) or unsigned (1) integer and masked to 16 bits, the 8.8 fixed point the hardware
//!   returns (`ops_texture.cpp`, `TMML`).
//! - `TXQ`: `ImageQueryDimensions` of the level in `A`, skipping the level count unless the
//!   mask selects it; the selected words go to consecutive registers.
//! - `SULD`: `ImageRead`; `SUST`: `ImageWrite`; `SUATOM`: `ImageAtomicIAdd32` and its
//!   siblings, writing the old value; `SURED`: the same atomic, its result unused.
//!
//! A sampled result's selected channels are written in red-to-alpha order to the first
//! destination and the register after it, then to the second destination and the register after
//! it (`dest + stored` for the first two, `dest_b + stored - 2` for the rest). With `.F16` each
//! pair of channels is packed into one register (`PackHalf2x16`, the low half first; a missing
//! last channel is 0.0). A destination of `RZ` is refused, as the PR refuses it.
//!
//! # Handles
//!
//! Handles stay symbolic for the texture pass (slice 13; decision 8: "texture handles are
//! traced first"), as the PR leaves them: a constant-buffer handle (the `0xb6x` forms) is its
//! byte offset as a `u32` immediate, and the buffer it is in is the program's texture
//! descriptor buffer ([`DescriptorBuffer`], the PR's `program_info->texture_buffer`); a
//! uniform-register handle (the `.UR` forms) is `GetUniformRegister` of that register; a
//! register handle (`TEX.B`, and the surface instructions) is `GetRegister` of it. Every
//! texture operation is the unbound `Image*` opcode: the pass picks the bound or bindless form.
//!
//! **For slice 12e** (the bindless-handle alignment rule the issue asks the two slices to
//! agree): a handle this module reads from a register is the register's whole 32-bit value,
//! unaligned and unmasked, exactly as the PR's `X(gpr_handle)` and Eden's `X(bindless_reg)`
//! read it; whatever alignment the texture pass needs is the pass's to check.
//!
//! # Arguments the IR leaves empty
//!
//! Eden leaves an unused optional argument empty (`IR::Value{}`: an absent offset, bias, LOD
//! clamp, second gather offset or sample index). This IR has no empty operand and the verifier
//! counts arguments, so such a slot holds [`ABSENT`], the immediate `u32 0`. It means what the
//! empty value means in every slot it fills: a zero texel offset; a bias or LOD clamp the
//! texture information says is not given (`has_bias`, `has_lod_clamp`); a second gather
//! offset that is not used (this module never passes one: "Per-texel gather offsets"); and
//! `ImageFetch`'s sample index, which no translated form uses (`.MS` is refused). **A guess
//! for the lowering to agree** (slice 17 lowers image operations): an owner question.
//!
//! # Guards
//!
//! An instruction under a guard predicate becomes a conditional block (spec, "Instruction
//! translation"), through the same helper as the float group: the guard is read, the block
//! branches to a block holding the instruction or past it, and translation continues in the
//! block after. [`translate_instr`] skips this for `PT` and when its block already tests the
//! guard (flow analysis's `guard_tested`). `@!PT` is checked and put in a block that never runs,
//! as the float group does (the PR skips it unchecked: `Visitor::Run`, `guard.Never()`).
//!
//! # Refusals
//!
//! Each `Require` the PR makes on these forms is a named [`TextureUnsupported`] whose doc
//! quotes the PR's message, checked in the PR's order, all before the first instruction is
//! appended, so a refusal leaves the function and the [`DescriptorBuffer`] as they were.
//! An instruction or form the PR does not translate is refused by name
//! ([`TextureUnsupported::HandleForm`] for the `.B` forms of `TLD`, `TLD4`, `TMML`, `TXD` and
//! `TXQ`, which `Visitor::Run` does not dispatch). The PR's checks that cannot fail on a decoded
//! instruction are not repeated: the fixed handle bits 59 and 91 (the decoder's forms fix
//! them), "native texture dimension" and "native surface dimension" (the decoder refuses
//! dimension 6, and 7 for surfaces), `TLD4`'s "native gather offset" (the decoder refuses offset
//! mode 3), `TXQ`'s bits 60 and 61 (unknown bits of its forms), `SUATOM`'s `atom_op <= 8` for
//! a decoded operation field (four bits, values 9 to 15 refused by the decoder), and `TEX`'s
//! "native texture operand count" (`count <= 6`): a list holds at most six operands, four
//! coordinates of a cube array with a level and a reference, or three coordinates with the
//! clamp word, a bias and a reference.
//!
//! # Level-of-detail clamp
//!
//! The PR refuses `.LC` and `.LB.LC` ("native texture clamp mode"), and so does Eden's Maxwell
//! `TEX` (`throw NotImplementedException("LC")`). The spec's gap table has the minimum-LOD clamp
//! carried in the IR for the lowering to emulate or ignore, so here `TEX` with a constant-buffer
//! or uniform-register handle takes both modes: `has_lod_clamp` is set and the clamp is the bias
//! slot's second component, as Eden's `MakeLodClampPair` puts it (`ir_emitter.cpp`: the clamp
//! alone when there is no bias, `CompositeConstruct(bias, lod_clamp)` with one). Where the clamp
//! is read from is NAK's: the high 16 bits of the array-index operand, as 8.8 fixed point
//! (`nak_nir_lower_tex.c`, `lower_tex`: `min_lod * 256` packed by `prmt(0x1054, min_lod,
//! arr_idx_or_zero)`, the array index or zero in the low half), an operand that a texture
//! without an array index gains at the front of its list. **From NAK alone, a guess** until an
//! `nvdisasm` or hardware check (verification E2). `TEX.B` keeps the PR's refusal
//! ([`TextureUnsupported::ClampMode`]): its operand layout is the PR's own and NAK does not
//! scalarise a bindless sample, so nothing places the clamp there. `TLD`'s `.CL` keeps the PR's
//! refusal too ([`TextureUnsupported::FetchMultisampleOrClamp`]).
//!
//! # Per-texel gather offsets
//!
//! `TLD4.PTP` gives each of the four gathered texels its own offset, two signed 6-bit fields
//! per texel in two registers (`ops_texture.cpp`, `TLD4`: `quad`, fields at bits 0, 8, 16 and
//! 24). The PR passes both registers' fields to one `ImageGather` (`offset`, `offset2`), which
//! Eden's SPIR-V backend can emit only for constant offsets (`emit_spirv_image.cpp`,
//! `ImageOperands`: "Not all arguments in PTP are immediate, ignoring") and naga not at all
//! (the gap table). Here it is emulated in the IR, as the spec's plan says: four gathers, the
//! `k`th with texel `k`'s offset alone, each giving its `(i0, j0)` texel (component 3,
//! [`GATHER_BASE_TEXEL`]) as result component `k`. That is the GLSL definition of
//! `textureGatherOffsets` ("selecting the texel i0 j0 of that footprint") and what Mesa's
//! `lower_tg4_offsets` does. Texel `k`'s offset is fields `2k` and `2k + 1` of the eight
//! (the first register's four, then the second's), as Eden's backend pairs them
//! (`ImageOperands`: `read(0, 0), read(0, 1)`, ...). A single non-constant offset (`TEX.AOFFI`,
//! `TLD4.AOFFI`) stays a value in the offset slot, for the lowering to emulate or not.
//!
//! # Surfaces
//!
//! The decoder reads `SULD`, `SUST` and `SUATOM` in NAK's forms only, whose handle is a register
//! (`0x998`, `0x99a`, `0x99c`, `0x99e`, `0x394`, `0x396`, `0x3a0`); the forms with a
//! constant-buffer handle, the only ones the PR translates (`0xb97`, `0xb9b`, `0x593`), are not
//! decoded yet (#437). So the surface translators port the PR's bodies with the register handle
//! of Eden's Maxwell `!is_bound` path (`surface_load_store.cpp`: `X(suld.bindless_reg)`), and
//! the binary (`.D`) forms, which the PR has no translator for, from Eden's Maxwell
//! `SULD`/`SUST` (`Format`, `SizeInRegs`: one register for 8 to 32 bits, two for 64, four for
//! 128, with the image format the size names). The PR's checks are kept with their messages
//! (formatted masks, register ranges, the atomic's mode, type and operation); the binary forms
//! check their register range the same way. `SURED` is `SUATOM` with its result discarded: its
//! destination field (which NAK writes `RZ`) is not written.
//!
//! # Divergences from PR #4588
//!
//! | What | PR #4588 | Here | Pinned by |
//! |---|---|---|---|
//! | A guard | a conditional block, from flow analysis (`flow.cpp`); none of these instructions is pure | the same, built here (spec, "Instruction translation") | `tests::guards::a_guard_makes_a_conditional_block` |
//! | `@!PT` | skipped unchecked (`guard.Never()`) | checked, and put in a block that never runs | `tests::guards::never_guarded_instructions_are_checked` |
//! | A fault (`.SPARSE`) predicate other than `PT` | refused ("native sparse texture residency", "native sparse surface residency") | translated; the predicate is written false, every texel reported resident (spec, gap table: "Report resident and list the divergence") | `tests::sample::sparse_results_report_resident` |
//! | `TEX.LC`, `TEX.LB.LC` with a constant-buffer or uniform handle | refused ("native texture clamp mode") | translated, the clamp carried (see "Level-of-detail clamp") | `tests::sample::lod_clamp_is_carried` |
//! | `TLD4.PTP` | one gather with both offset registers | four gathers (see "Per-texel gather offsets") | `tests::gather::per_texel_offsets_take_four_gathers` |
//! | The surface forms with a register handle (`0x998` to `0x99e`, `0x394`, `0x3a0`) | not dispatched by `Visitor::Run` | translated (see "Surfaces") | `tests::surface` |
//! | `SUATOM.CAS` (`0x396`) | not dispatched | refused by name ([`TextureUnsupported::SurfaceCompareAndSwap`]) | `tests::refusals::surface_refusals` |
//! | The descriptor buffer's conflict check | per program, set at the first handle even if the instruction is then refused (the program fails anyway) | the same per [`DescriptorBuffer`], which is set only when the instruction translates | `tests::refusals::a_refusal_leaves_the_descriptor_buffer` |
//!
//! The PR's `env.TextureBoundBuffer() == cbuf_index` check, made when it translates without a
//! `program_info`, is not repeated: there is no environment here, and [`DescriptorBuffer`] is
//! the program's record that the texture pass reads. The PR's bound on the buffer index,
//! [`DESCRIPTOR_BUFFERS`] (18, Eden's `MaxConstBuffers`), is kept as the issue asks, though the
//! spec gives the host 32 constant-buffer slots ("Passes needed for correctness", item 2):
//! whether a texture handle may sit in buffers 18 to 31 is an owner question.
//!
//! **Owner question** (`TEX.B`'s operand layout): the PR puts the level or depth reference at
//! `B` and the handle after it (`gpr_handle = source_b + (count > dim.coordinates ? 1 : 0)`),
//! from the forms it saw, and requires the scalar layout; NAK's packing puts the handle first in
//! `B` (`nak_nir_lower_tex.c`: `PUSH(src1, tex_h)` before the level) but never scalarises a
//! bindless sample, so the two do not describe the same encoding. The PR's layout is followed.

#[cfg(test)]
mod tests;

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use super::{REGISTER_LIMIT, conditional_block};
use crate::decode::{
    AtomOp, AtomType, ChannelMask, Gpr, ImageAccess, ImageDim, Instr, MemType, Op, OpSuAtom,
    OpSuLd, OpSuSt, OpTex, OpTld, OpTld4, OpTmml, OpTxd, OpTxq, Pred, TexDim, TexLodMode,
    TexOffsetMode, TexQuery, TexRef, UniformReg,
};
use crate::ir::{
    BuildError, Builder, Flags, ImageFormat, Imm, Opcode, Operand, TextureInfo, TextureType,
};

/// How many constant buffers a texture handle may be read from: the PR refuses `cbuf_index >=
/// 18` (`ops_texture.cpp`, `TEX` and `TLD`: "native texture descriptor buffer";
/// `UseDescriptorBuffer`: "native descriptor buffer"), Eden's `MaxConstBuffers`.
pub const DESCRIPTOR_BUFFERS: u8 = 18;

/// The uniform register a `.UR` handle must be below: `handle_register < 63` (`TEX`, `TLD`,
/// `TLD4`: "native bindless texture handle"), which refuses `URZ`.
pub const UNIFORM_HANDLE_LIMIT: u8 = UniformReg::URZ.index();

/// The width of an array index: its low 16 bits (`BitFieldExtract(raw, Imm32(0),
/// Imm32(16))`).
pub const ARRAY_INDEX_BITS: u32 = 16;

/// Where the minimum-LOD clamp sits in the array-index operand: its high 16 bits (NAK's
/// `prmt(0x1054, min_lod, arr_idx_or_zero)`; see the module docs, "Level-of-detail clamp").
pub const MIN_LOD_SHIFT: u32 = 16;

/// The scale of the hardware's 8.8 fixed-point levels of detail: `TMML`'s results
/// (`FPMul(level, Imm32(256.0f))`) and NAK's packed clamp (`min_lod * 256`).
pub const LOD_FIXED_SCALE: f32 = 256.0;

/// One unit of [`LOD_FIXED_SCALE`]: what a packed clamp is multiplied by to give a level.
pub const LOD_FIXED_UNIT: f32 = 1.0 / LOD_FIXED_SCALE;

/// The bits of a `TMML` result kept: `BitwiseAnd(fixed, Imm32(0xffff))`.
pub const LOD_FIXED_MASK: u32 = 0xffff;

/// The width of each of `TEX`'s two offset fields, at bits 0 and 4 of the offset register
/// (`BitFieldExtract(packed, 0, 4, true)`, `(packed, 4, 4, true)`).
pub const TEX_OFFSET_BITS: u32 = 4;

/// The width of each of `TLD4`'s offset fields (`BitFieldExtract(value, .., Imm32(6), true)`).
pub const GATHER_OFFSET_BITS: u32 = 6;

/// The distance between `TLD4`'s offset fields: bits 0, 8, 16 and 24.
pub const GATHER_OFFSET_STRIDE: u32 = 8;

/// The component of a gather's result that holds its `(i0, j0)` texel, the one a per-texel
/// offset moves (GLSL's `textureGatherOffsets`; Mesa's `lower_tg4_offsets`: `nir_get_scalar(..,
/// 3)`).
pub const GATHER_BASE_TEXEL: u32 = 3;

/// `TXQ`'s word that holds the level count: element 3 of the dimensions, skipped unless the
/// mask selects it (`ImageQueryDimension(.., Imm1(((mask >> 3) & 1) == 0))`).
pub const QUERY_LEVELS: u32 = 3;

/// The channels `TMML` may select: the low two (`Require((mask & 0xc) == 0 && mask != 0,
/// "native mipmap query channels")`).
pub const MIP_QUERY_CHANNELS: u8 = 0x3;

/// The channel masks a formatted `SUST` may have (`Require(mask == 1 || mask == 3 || mask == 7
/// || mask == 15, "native surface channel mask")`): the leading channels.
pub const SURFACE_STORE_MASKS: [u8; 4] = [0x1, 0x3, 0x7, 0xf];

/// What an argument Eden leaves empty holds here (see the module docs, "Arguments the IR leaves
/// empty").
pub const ABSENT: Operand = Operand::Imm(Imm::U32(0));

/// The channels of a texel.
const CHANNELS: u8 = 4;

/// The most operands one list holds (the PR's `std::array<u32, 8> regs`).
const LIST_CAPACITY: usize = 8;

/// A form or field value of this group that is not translated: one of PR #4588's `Require`
/// checks, named by its message, or a form the PR does not translate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TextureUnsupported {
    /// An instruction of another group, handed to this one.
    #[error("{0}, which is not a texture or surface instruction,")]
    OtherGroup(&'static str),
    /// A handle kind the PR does not translate for this instruction: the `.B` forms of `TLD`,
    /// `TLD4`, `TMML`, `TXD` and `TXQ` (`Visitor::Run` dispatches none of `0x364`, `0x367`,
    /// `0x36a`, `0x36d`, `0x370`), and a uniform-register handle on `TMML`, `TXD` or `TXQ`,
    /// which have no such form.
    #[error("{mnemonic} with its handle in {kind}")]
    HandleForm {
        /// The instruction.
        mnemonic: &'static str,
        /// Where its handle is.
        kind: HandleKind,
    },
    /// A `.UR` handle in `URZ` (`Require(Take(91, 1) == 1 && handle_register < 63, "native
    /// bindless texture handle")`; [`UNIFORM_HANDLE_LIMIT`]).
    #[error("a texture handle in {0}")]
    UniformHandle(UniformReg),
    /// `TEX.NDV` (`Require(Take(77, 1) == 0, "native non-divergent texture derivative")`).
    #[error("a texture sample with non-divergent derivatives")]
    Derivative,
    /// `TEX.B` with a level-of-detail clamp (`Require(lod_mode <= 3, "native texture clamp
    /// mode")`, kept for the `.B` form only; see the module docs, "Level-of-detail clamp").
    #[error("a texture sample with its handle in a register and a level-of-detail clamp")]
    ClampMode,
    /// An empty channel mask (`Require(mask != 0, "native empty texture channel mask")`, in
    /// `TEX`, `TLD`, `TLD4` and `TXD`).
    #[error("an empty texture channel mask")]
    EmptyMask,
    /// A constant-buffer handle in buffer [`DESCRIPTOR_BUFFERS`] or above
    /// (`Require(cbuf_index < 18, "native texture descriptor buffer")` in `TEX` and `TLD`,
    /// "native descriptor buffer" in `UseDescriptorBuffer`).
    #[error("a texture handle in constant buffer {0}")]
    DescriptorBuffer(u8),
    /// A constant-buffer handle in another buffer than the program's earlier ones ("native
    /// texture descriptor buffer conflict", "native descriptor buffer conflict").
    #[error("a texture handle in constant buffer {index} after handles in buffer {bound}")]
    DescriptorBufferConflict {
        /// The buffer the program's earlier handles are in.
        bound: u8,
        /// This handle's buffer.
        index: u8,
    },
    /// `TEX.AOFFI` outside the layout the PR handles: scalar, 32-bit results, a handle not in a
    /// register, a 2D texture, no depth comparison, no bias or clamp, and even, in-range
    /// coordinate and level pairs (`Require(.., "native texture offset")`).
    #[error("a texture offset in this operand layout")]
    Offset,
    /// `TEX.B` outside the layout the PR handles: scalar, 32-bit results, and at most one
    /// operand after the coordinates (`Require(scalar && !half && count <= dim.coordinates + 1,
    /// "native bindless texture operands")`).
    #[error("a texture sample with its handle in a register in this operand layout")]
    BindlessOperands,
    /// The vector layout with more than one operand after the coordinates (`Require(count <=
    /// dim.coordinates + 1, "native vector texture operand count")`).
    #[error("a vector texture sample with more than one operand after its coordinates")]
    VectorOperandCount,
    /// An operand register at or past `RZ` (`Require(reg < 255, "native texture operand
    /// register")`).
    #[error("texture operand register {0}")]
    OperandRegister(u32),
    /// `TEX.B`'s handle register at or past `RZ` (`Require(gpr_handle < 255, "native bindless
    /// texture handle")`).
    #[error("texture handle register {0}")]
    HandleRegister(u32),
    /// A second destination that does not match the result's size: `RZ` for one or two
    /// channels, another register for three or four (`Require(outputs <= 2 ? dest_b == 255 :
    /// dest_b != 255, "native texture result layout")`).
    #[error("a texture result layout whose second destination does not match its channel count")]
    ResultLayout,
    /// A result register at or past `RZ` (`Require(target < 255, "native texture result
    /// register")`).
    #[error("texture result register {0}")]
    ResultRegister(u32),
    /// `TLD` without `.SCR` (`Require(Take(60, 1) != 0, "native packed half-precision texel
    /// fetch result")`).
    #[error("a texel fetch in the vector operand layout")]
    FetchNotScalar,
    /// `TLD.AOFFI` (`Require(Take(76, 1) == 0, "native texel fetch offset")`).
    #[error("a texel fetch with an offset")]
    FetchOffset,
    /// `TLD.MS` or `TLD.CL` (`Require(Take(77, 2) == 0, "native texel fetch multisample or
    /// clamp")`).
    #[error("a multisampled or clamped texel fetch")]
    FetchMultisampleOrClamp,
    /// A `TLD` level mode other than `.LZ` and `.LL` (`Require(lod_mode == 1 || lod_mode == 3,
    /// "native texel fetch lod mode")`).
    #[error("a texel fetch with a level-of-detail mode other than lz and ll")]
    FetchLodMode,
    /// `TLD` of a cube or a cube array (`Unsupported("native texel fetch dimension")`).
    #[error("a texel fetch of dimension {0}")]
    FetchDimension(TexDim),
    /// `TLD4` without `.SCR` (`Require(Take(60, 1) != 0, "native packed half-precision gather
    /// result")`).
    #[error("a gather in the vector operand layout")]
    GatherNotScalar,
    /// `TLD4` offsets on a texture that is not 2D or a 2D array (`Require(.., "native gather
    /// offset dimension")`).
    #[error("gather offsets with dimension {0}")]
    GatherOffsetDimension(TexDim),
    /// `TLD4` of a texture that is not 2D, a 2D array or a cube (`Unsupported("native gather
    /// dimension")`).
    #[error("a gather of dimension {0}")]
    GatherDimension(TexDim),
    /// `TXD.AOFFI` (`Require(Take(76, 1) == 0, "native gradient texture offset")`).
    #[error("a gradient sample with an offset")]
    GradientOffset,
    /// `TXD` of a texture that is not 2D or a 2D array (`Require(.., "native gradient texture
    /// dimension")`).
    #[error("a gradient sample of dimension {0}")]
    GradientDimension(TexDim),
    /// `TXD`'s coordinate or derivative registers reaching `RZ` (`Require(coord_reg + (array ?
    /// 2 : 1) < 255 && derivative_reg + 3 < 255, "native gradient operand registers")`).
    #[error("gradient operand registers past r254")]
    GradientRegisters,
    /// `TMML` with a second source or a second destination other than `RZ` (`Require(Take(32,
    /// 8) == 255, ..)`, `Require(Take(64, 8) == 255, "native mipmap query extra register")`).
    #[error("a level-of-detail query with a second source or destination")]
    MipQueryExtraRegister,
    /// A `TMML` mask outside [`MIP_QUERY_CHANNELS`] or empty ("native mipmap query
    /// channels").
    #[error("a level-of-detail query of channels {0}")]
    MipQueryChannels(ChannelMask),
    /// `TMML`'s coordinates reaching `RZ` (`Require(coord_reg + dim.coordinates < 255, "native
    /// mipmap query registers")`).
    #[error("level-of-detail query coordinate registers past r254")]
    MipQueryRegisters,
    /// A `TMML` result register at or past `RZ` (`Require(dest + stored < 255, "native mipmap
    /// query result register")`).
    #[error("level-of-detail query result register {0}")]
    MipQueryResultRegister(u32),
    /// A `TXQ` query other than the dimensions (`Require(Take(62, 2) == 0, "native texture query
    /// kind")`).
    #[error("a texture query of {0}")]
    QueryKind(TexQuery),
    /// `TXQ` with a second destination other than `RZ` (`Require(Take(64, 8) == 255, "native
    /// texture query extra register")`).
    #[error("a texture query with a second destination")]
    QueryExtraRegister,
    /// An empty `TXQ` mask (`Require(mask != 0, "native empty texture query mask")`).
    #[error("an empty texture query mask")]
    EmptyQueryMask,
    /// A `TXQ` result register at or past `RZ` (`Require(dest + stored < 255, "native texture
    /// query result register")`).
    #[error("texture query result register {0}")]
    QueryResultRegister(u32),
    /// An empty formatted `SULD` mask (`Require(mask != 0, "native empty surface channel
    /// mask")`).
    #[error("an empty surface channel mask")]
    EmptySurfaceMask,
    /// A formatted `SUST` mask other than [`SURFACE_STORE_MASKS`] ("native surface channel
    /// mask").
    #[error("a surface store of channels {0}")]
    SurfaceChannelMask(ChannelMask),
    /// Surface coordinate or data registers reaching `RZ` (`Require(coord_reg +
    /// dim.coordinates < 255, "native surface registers")`, and `data_reg + count < 255` for
    /// `SUST`).
    #[error("surface operand registers past r254")]
    SurfaceRegisters,
    /// A `SULD` result register at or past `RZ` (`Require(dest + stored < 255, "native surface
    /// result register")`).
    #[error("surface result register {0}")]
    SurfaceResultRegister(u32),
    /// `SUATOM.BA` (`Require(Take(72, 1) == 0, "native surface atomic mode")`).
    #[error("a surface atomic with .ba")]
    SurfaceAtomicMode,
    /// A surface atomic of a type other than `U32` and `S32`, or an operation past exchange
    /// (`Require(type <= 1 && atom_op <= 8, "native surface atomic operation")`).
    #[error("a surface atomic {op} of type {ty}")]
    SurfaceAtomicOperation {
        /// The operation's name.
        op: &'static str,
        /// The type's name.
        ty: &'static str,
    },
    /// `SUATOM.CAS` (`0x396`), which `Visitor::Run` does not dispatch.
    #[error("a surface compare-and-swap")]
    SurfaceCompareAndSwap,
}

/// Where a texture instruction's handle is, for [`TextureUnsupported::HandleForm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HandleKind {
    /// In the source registers (`.B`).
    Registers,
    /// In a uniform register (`.UR`).
    Uniform,
}

impl Display for HandleKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Registers => "the source registers",
            Self::Uniform => "a uniform register",
        })
    }
}

/// Why [`translate_instr`] did not translate an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TextureError {
    /// The instruction's form is not translated; nothing was appended and the
    /// [`DescriptorBuffer`] is unchanged.
    #[error("{0} is not supported")]
    Unsupported(TextureUnsupported),
    /// The IR could not be built: an arena is full, or this module built a malformed
    /// instruction (a bug). Instructions appended before the failure stay in the function.
    #[error(transparent)]
    Build(#[from] BuildError),
}

impl From<TextureUnsupported> for TextureError {
    fn from(what: TextureUnsupported) -> Self {
        Self::Unsupported(what)
    }
}

/// The constant buffer a program's texture handles are read from: unset until the first
/// translated instruction with a constant-buffer handle sets it (PR #4588's
/// `program_info->texture_buffer`, `ops_texture.cpp`: `texture_buffer == ~0U ||
/// texture_buffer == cbuf_index`). Only [`translate_instr`] sets it, so it is always below
/// [`DESCRIPTOR_BUFFERS`]. One value serves one program: the texture pass reads it to find the
/// buffer of every constant-buffer handle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DescriptorBuffer(Option<u8>);

impl DescriptorBuffer {
    /// No handle translated yet.
    pub const fn new() -> Self {
        Self(None)
    }

    /// The buffer, once a constant-buffer handle has been translated.
    pub const fn index(self) -> Option<u8> {
        self.0
    }

    /// This record with a handle in buffer `index` added: the PR's two checks, in its order.
    const fn bind(self, index: u8) -> Result<Self, TextureUnsupported> {
        if index >= DESCRIPTOR_BUFFERS {
            return Err(TextureUnsupported::DescriptorBuffer(index));
        }
        match self.0 {
            Some(bound) if bound != index => {
                Err(TextureUnsupported::DescriptorBufferConflict { bound, index })
            }
            _ => Ok(Self(Some(index))),
        }
    }
}

/// Whether `op` is one of this group's instructions.
pub const fn in_group(op: &Op) -> bool {
    matches!(
        op,
        Op::Tex(_)
            | Op::Tld(_)
            | Op::Tld4(_)
            | Op::Tmml(_)
            | Op::Txd(_)
            | Op::Txq(_)
            | Op::SuLd(_)
            | Op::SuSt(_)
            | Op::SuAtom(_)
    )
}

/// Translates `instr`, one of this group's instructions, at the end of `builder`'s block, and
/// leaves `builder` in the block where translation continues: the same block, or for a guarded
/// instruction the block after its conditional block (module docs, "Guards"). `guard_tested`
/// says the block already runs only where the guard holds, so the guard is ignored. `buffer` is
/// the program's texture descriptor buffer, checked and then updated by a constant-buffer
/// handle. A refusal appends nothing and leaves `buffer` as it was.
pub fn translate_instr(
    builder: &mut Builder<'_>,
    instr: Instr,
    guard_tested: bool,
    buffer: &mut DescriptorBuffer,
) -> Result<(), TextureError> {
    let (plan, bound) = plan(&instr.op, *buffer)?;
    conditional_block(builder, instr.guard, guard_tested, |b| {
        Ok::<_, TextureError>(Emit { b }.plan(&plan)?)
    })?;
    *buffer = bound;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Checks: every refusal, in the PR's order, before anything is appended.

const fn require(ok: bool, what: TextureUnsupported) -> Result<(), TextureUnsupported> {
    if ok { Ok(()) } else { Err(what) }
}

/// The IR type of a texture dimension and how many coordinates it takes, the array index
/// included (`ops_texture.cpp`, `DimensionOf`).
const fn tex_dimension(dim: TexDim) -> (TextureType, u32) {
    match dim {
        TexDim::D1 => (TextureType::Color1D, 1),
        TexDim::Array1D => (TextureType::ColorArray1D, 2),
        TexDim::D2 => (TextureType::Color2D, 2),
        TexDim::Array2D => (TextureType::ColorArray2D, 3),
        TexDim::D3 => (TextureType::Color3D, 3),
        TexDim::Cube => (TextureType::ColorCube, 3),
        TexDim::ArrayCube => (TextureType::ColorArrayCube, 4),
    }
}

/// The IR type of a surface dimension and how many coordinates it takes
/// (`ImageDimensionOf`).
const fn image_dimension(dim: ImageDim) -> (TextureType, u32) {
    match dim {
        ImageDim::D1 => (TextureType::Color1D, 1),
        ImageDim::Buffer1D => (TextureType::Buffer, 1),
        ImageDim::Array1D => (TextureType::ColorArray1D, 2),
        ImageDim::D2 => (TextureType::Color2D, 2),
        ImageDim::Array2D => (TextureType::ColorArray2D, 3),
        ImageDim::D3 => (TextureType::Color3D, 3),
    }
}

/// Whether a texture dimension has an array index.
const fn is_array(dim: TexDim) -> bool {
    matches!(dim, TexDim::Array1D | TexDim::Array2D | TexDim::ArrayCube)
}

/// The register `k` places after `base`, as the PR computes it (no wrap: a `u32` sum).
fn after(base: Gpr, k: u32) -> u32 {
    u32::from(base.index()) + k
}

/// Register `n`, which a check has found below `RZ`.
fn gpr(n: u32) -> Gpr {
    // Cannot fail where it is called: every caller has checked `n < REGISTER_LIMIT`.
    Gpr::new(u8::try_from(n).unwrap_or(Gpr::RZ.index()))
}

/// Register `n`, if it is below `RZ`, else `what`.
fn below_rz(n: u32, what: TextureUnsupported) -> Result<Gpr, TextureUnsupported> {
    require(n < REGISTER_LIMIT, what)?;
    Ok(gpr(n))
}

/// A list of operand registers, in the order the instruction reads them.
#[derive(Clone, Copy, Debug)]
struct List {
    regs: [Gpr; LIST_CAPACITY],
    len: usize,
}

impl List {
    const fn new() -> Self {
        Self {
            regs: [Gpr::RZ; LIST_CAPACITY],
            len: 0,
        }
    }

    /// Appends `reg`; past the capacity it is dropped (no list holds more than six: module
    /// docs, "Refusals").
    fn push(&mut self, reg: Gpr) {
        if let Some(slot) = self.regs.get_mut(self.len) {
            *slot = reg;
            self.len += 1;
        }
    }
}

/// Reads a [`List`] front to back.
struct Cursor<'a> {
    regs: &'a [Gpr],
    next: usize,
}

impl<'a> Cursor<'a> {
    fn new(list: &'a List) -> Self {
        Self {
            regs: list.regs.get(..list.len).unwrap_or(&[]),
            next: 0,
        }
    }

    /// The next register; `RZ` past the end, which no plan reaches (each reads as many as its
    /// check counted).
    fn take(&mut self) -> Gpr {
        let reg = self.regs.get(self.next).copied().unwrap_or(Gpr::RZ);
        self.next += 1;
        reg
    }
}

/// The scalar layout's operand registers (`TEX`, `TLD`, `TLD4`: two operands one register
/// each at `A` and `B`; otherwise two at `A` and the rest from `B`), each below `RZ`.
fn scalar_list(a: Gpr, b: Gpr, count: u32) -> Result<List, TextureUnsupported> {
    let mut list = List::new();
    for k in 0..count {
        let reg = if count <= 2 {
            if k == 0 { after(a, 0) } else { after(b, 0) }
        } else if k < 2 {
            after(a, k)
        } else {
            after(b, k - 2)
        };
        list.push(below_rz(reg, TextureUnsupported::OperandRegister(reg))?);
    }
    Ok(list)
}

/// The layout with `leading` operands from `A` and the rest at `B` (the vector layout and
/// `TEX.B`'s), each below `RZ`.
fn split_list(a: Gpr, b: Gpr, leading: u32, count: u32) -> Result<List, TextureUnsupported> {
    let mut list = List::new();
    for k in 0..count {
        let reg = if k < leading {
            after(a, k)
        } else {
            after(b, 0)
        };
        list.push(below_rz(reg, TextureUnsupported::OperandRegister(reg))?);
    }
    Ok(list)
}

/// Where a texture instruction's handle comes from in the IR (module docs, "Handles").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HandleSrc {
    /// A byte offset into the descriptor buffer.
    Offset(u32),
    /// A uniform register.
    Uniform(UniformReg),
    /// A register.
    Reg(Gpr),
}

/// The handle of a `.UR` or constant-buffer form of `TEX`, `TLD` or `TLD4`; `None` for `.B`,
/// whose register the layout picks. The `.UR` register must be below `URZ`.
fn texture_handle(tex: TexRef) -> Result<Option<HandleSrc>, TextureUnsupported> {
    match tex {
        TexRef::CBuf(cbuf) => Ok(Some(HandleSrc::Offset(u32::from(cbuf.offset)))),
        TexRef::Uniform(reg) => {
            require(
                reg.index() < UNIFORM_HANDLE_LIMIT,
                TextureUnsupported::UniformHandle(reg),
            )?;
            Ok(Some(HandleSrc::Uniform(reg)))
        }
        TexRef::Bindless => Ok(None),
    }
}

/// Refuses a handle kind the PR does not translate for `mnemonic`: `.B` unless
/// `takes_registers`, `.UR` unless `takes_uniform`.
const fn handle_form(
    mnemonic: &'static str,
    tex: TexRef,
    takes_registers: bool,
    takes_uniform: bool,
) -> Result<(), TextureUnsupported> {
    match tex {
        TexRef::Bindless if !takes_registers => Err(TextureUnsupported::HandleForm {
            mnemonic,
            kind: HandleKind::Registers,
        }),
        TexRef::Uniform(_) if !takes_uniform => Err(TextureUnsupported::HandleForm {
            mnemonic,
            kind: HandleKind::Uniform,
        }),
        _ => Ok(()),
    }
}

/// `buffer` with `tex`'s buffer added when `tex` is a constant-buffer handle.
const fn bind_handle(
    buffer: DescriptorBuffer,
    tex: TexRef,
) -> Result<DescriptorBuffer, TextureUnsupported> {
    match tex {
        TexRef::CBuf(cbuf) => buffer.bind(cbuf.index),
        TexRef::Bindless | TexRef::Uniform(_) => Ok(buffer),
    }
}

/// Where a sampled result's channels go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Targets {
    /// One register per selected channel, by channel.
    Float([Option<Gpr>; CHANNELS as usize]),
    /// `.F16`: the first two selected channels packed into `low`, the other two into `high`.
    Half { low: Gpr, high: Option<Gpr> },
}

/// The registers of a sampled result (`TEX`, `TLD`, `TLD4`, `TXD`): the PR's layout check, then
/// each target below `RZ`, in channel order; with `half`, the packed registers.
fn result_targets(
    mask: ChannelMask,
    dsts: [Gpr; 2],
    half: bool,
) -> Result<Targets, TextureUnsupported> {
    let [dest, dest_b] = dsts;
    let outputs = mask.bits().count_ones();
    let layout = if outputs <= 2 {
        dest_b.is_zero()
    } else {
        !dest_b.is_zero()
    };
    require(layout, TextureUnsupported::ResultLayout)?;
    if half {
        let low = below_rz(
            after(dest, 0),
            TextureUnsupported::ResultRegister(after(dest, 0)),
        )?;
        let high = if outputs > 2 {
            Some(below_rz(
                after(dest_b, 0),
                TextureUnsupported::ResultRegister(after(dest_b, 0)),
            )?)
        } else {
            None
        };
        return Ok(Targets::Half { low, high });
    }
    let mut targets = [None; CHANNELS as usize];
    let mut stored = 0;
    for (channel, target) in (0..CHANNELS).zip(&mut targets) {
        if mask.bits() & (1 << channel) == 0 {
            continue;
        }
        let reg = if stored < 2 {
            after(dest, stored)
        } else {
            after(dest_b, stored - 2)
        };
        *target = Some(below_rz(reg, TextureUnsupported::ResultRegister(reg))?);
        stored += 1;
    }
    Ok(Targets::Float(targets))
}

/// Consecutive result registers from `dest`, one per selected channel of `mask` (`TMML`,
/// `TXQ`, a formatted `SULD`), each below `RZ` (`Require(dest + stored < 255, ..)`).
fn consecutive_targets(
    mask: u8,
    dest: Gpr,
    what: fn(u32) -> TextureUnsupported,
) -> Result<[Option<Gpr>; CHANNELS as usize], TextureUnsupported> {
    let mut targets = [None; CHANNELS as usize];
    let mut stored = 0;
    for (channel, target) in (0..CHANNELS).zip(&mut targets) {
        if mask & (1 << channel) == 0 {
            continue;
        }
        let reg = after(dest, stored);
        *target = Some(below_rz(reg, what(reg))?);
        stored += 1;
    }
    Ok(targets)
}

/// What one instruction appends, decided by its checks.
#[derive(Clone, Copy, Debug)]
enum Plan {
    Tex(TexPlan),
    Tld(TldPlan),
    Tld4(Tld4Plan),
    Txd(TxdPlan),
    Tmml(TmmlPlan),
    Txq(TxqPlan),
    SuLd(SuLdPlan),
    SuSt(SuStPlan),
    SuAtom(SuAtomPlan),
}

/// Checks `op` and decides what it appends, with the descriptor buffer it leaves.
fn plan(op: &Op, buffer: DescriptorBuffer) -> Result<(Plan, DescriptorBuffer), TextureUnsupported> {
    match *op {
        Op::Tex(op) => plan_tex(op, buffer).map(|(plan, b)| (Plan::Tex(plan), b)),
        Op::Tld(op) => plan_tld(op, buffer).map(|(plan, b)| (Plan::Tld(plan), b)),
        Op::Tld4(op) => plan_tld4(op, buffer).map(|(plan, b)| (Plan::Tld4(plan), b)),
        Op::Txd(op) => plan_txd(op, buffer).map(|(plan, b)| (Plan::Txd(plan), b)),
        Op::Tmml(op) => plan_tmml(op, buffer).map(|(plan, b)| (Plan::Tmml(plan), b)),
        Op::Txq(op) => plan_txq(op, buffer).map(|(plan, b)| (Plan::Txq(plan), b)),
        Op::SuLd(op) => plan_suld(op).map(|plan| (Plan::SuLd(plan), buffer)),
        Op::SuSt(op) => plan_sust(op).map(|plan| (Plan::SuSt(plan), buffer)),
        Op::SuAtom(op) => plan_suatom(op).map(|plan| (Plan::SuAtom(plan), buffer)),
        other => Err(TextureUnsupported::OtherGroup(other.mnemonic())),
    }
}

/// `TEX`, checked.
#[derive(Clone, Copy, Debug)]
struct TexPlan {
    op: OpTex,
    handle: HandleSrc,
    /// The operands: a clamp word for a non-array texture with a clamp, the array index, the
    /// coordinates, the level or bias, the depth reference.
    list: List,
    /// `.AOFFI`'s offset register.
    offset: Option<Gpr>,
    targets: Targets,
}

fn plan_tex(op: OpTex, buffer: DescriptorBuffer) -> Result<(TexPlan, DescriptorBuffer), TextureUnsupported> {
    let handle = texture_handle(op.tex)?;
    let register_handle = handle.is_none();
    let (ty, coordinates) = tex_dimension(op.dim);
    require(!op.ndv, TextureUnsupported::Derivative)?;
    let clamp = matches!(op.lod_mode, TexLodMode::Clamp | TexLodMode::BiasClamp);
    require(!(clamp && register_handle), TextureUnsupported::ClampMode)?;
    require(op.channel_mask.bits() != 0, TextureUnsupported::EmptyMask)?;
    let buffer = bind_handle(buffer, op.tex)?;
    let has_lod = matches!(
        op.lod_mode,
        TexLodMode::Bias | TexLodMode::Lod | TexLodMode::BiasClamp
    );
    let clamp_word = u32::from(clamp && !is_array(op.dim));
    let leading = coordinates + clamp_word;
    // At most six, so the PR's `Require(count <= 6, "native texture operand count")` cannot
    // fail (module docs, "Refusals").
    let count = leading + u32::from(has_lod) + u32::from(op.z_cmpr);
    let [a, b] = op.srcs;
    let mut offset = None;
    let mut handle = handle;
    let list = if op.offset {
        let lod = op.lod_mode == TexLodMode::Lod;
        let a_index = after(a, 0);
        let b_index = after(b, 0);
        require(
            op.scalar
                && !op.f16
                && !register_handle
                && ty == TextureType::Color2D
                && !op.z_cmpr
                && matches!(
                    op.lod_mode,
                    TexLodMode::Auto | TexLodMode::Zero | TexLodMode::Lod
                )
                && a_index % 2 == 0
                && a_index + 1 < REGISTER_LIMIT
                && b_index + 1 < REGISTER_LIMIT
                && (!lod || b_index % 2 == 0),
            TextureUnsupported::Offset,
        )?;
        let mut list = List::new();
        list.push(gpr(a_index));
        list.push(gpr(a_index + 1));
        if lod {
            list.push(gpr(b_index));
            offset = Some(gpr(b_index + 1));
        } else {
            offset = Some(b);
        }
        list
    } else if register_handle {
        require(
            op.scalar && !op.f16 && count <= coordinates + 1,
            TextureUnsupported::BindlessOperands,
        )?;
        let list = split_list(a, b, coordinates, count)?;
        let reg = after(b, u32::from(count > coordinates));
        handle = Some(HandleSrc::Reg(below_rz(
            reg,
            TextureUnsupported::HandleRegister(reg),
        )?));
        list
    } else if op.scalar {
        scalar_list(a, b, count)?
    } else {
        require(
            count <= leading + 1,
            TextureUnsupported::VectorOperandCount,
        )?;
        split_list(a, b, leading, count)?
    };
    let targets = result_targets(op.channel_mask, op.dsts, op.f16)?;
    let plan = TexPlan {
        op,
        // Cannot be `None`: a register handle is set by its layout above.
        handle: handle.unwrap_or(HandleSrc::Reg(Gpr::RZ)),
        list,
        offset,
        targets,
    };
    Ok((plan, buffer))
}

/// `TLD`, checked.
#[derive(Clone, Copy, Debug)]
struct TldPlan {
    op: OpTld,
    handle: HandleSrc,
    list: List,
    targets: Targets,
}

fn plan_tld(op: OpTld, buffer: DescriptorBuffer) -> Result<(TldPlan, DescriptorBuffer), TextureUnsupported> {
    handle_form("tld", op.tex, false, true)?;
    let handle = texture_handle(op.tex)?;
    require(op.scalar, TextureUnsupported::FetchNotScalar)?;
    let (_, coordinates) = tex_dimension(op.dim);
    require(!op.offset, TextureUnsupported::FetchOffset)?;
    require(
        !op.is_ms && !op.clamp,
        TextureUnsupported::FetchMultisampleOrClamp,
    )?;
    require(
        matches!(op.lod_mode, TexLodMode::Zero | TexLodMode::Lod),
        TextureUnsupported::FetchLodMode,
    )?;
    require(op.channel_mask.bits() != 0, TextureUnsupported::EmptyMask)?;
    let buffer = bind_handle(buffer, op.tex)?;
    let count = coordinates + u32::from(op.lod_mode == TexLodMode::Lod);
    let [a, b] = op.srcs;
    let list = scalar_list(a, b, count)?;
    require(
        !matches!(op.dim, TexDim::Cube | TexDim::ArrayCube),
        TextureUnsupported::FetchDimension(op.dim),
    )?;
    let targets = result_targets(op.channel_mask, op.dsts, op.f16)?;
    let plan = TldPlan {
        op,
        // Cannot be `None`: the `.B` form is refused above.
        handle: handle.unwrap_or(HandleSrc::Reg(Gpr::RZ)),
        list,
        targets,
    };
    Ok((plan, buffer))
}

/// `TLD4`, checked.
#[derive(Clone, Copy, Debug)]
struct Tld4Plan {
    op: OpTld4,
    handle: HandleSrc,
    /// The coordinates, the offset registers, the depth reference.
    list: List,
    targets: Targets,
}

fn plan_tld4(op: OpTld4, buffer: DescriptorBuffer) -> Result<(Tld4Plan, DescriptorBuffer), TextureUnsupported> {
    handle_form("tld4", op.tex, false, true)?;
    let handle = texture_handle(op.tex)?;
    let buffer = bind_handle(buffer, op.tex)?;
    require(op.scalar, TextureUnsupported::GatherNotScalar)?;
    let (_, coordinates) = tex_dimension(op.dim);
    require(op.channel_mask.bits() != 0, TextureUnsupported::EmptyMask)?;
    let offset_registers = match op.offset_mode {
        TexOffsetMode::None => 0,
        TexOffsetMode::AddOffI => 1,
        TexOffsetMode::PerPx => 2,
    };
    let count = coordinates + offset_registers + u32::from(op.z_cmpr);
    require(
        op.offset_mode == TexOffsetMode::None || matches!(op.dim, TexDim::D2 | TexDim::Array2D),
        TextureUnsupported::GatherOffsetDimension(op.dim),
    )?;
    let [a, b] = op.srcs;
    let list = scalar_list(a, b, count)?;
    require(
        matches!(op.dim, TexDim::D2 | TexDim::Array2D | TexDim::Cube),
        TextureUnsupported::GatherDimension(op.dim),
    )?;
    let targets = result_targets(op.channel_mask, op.dsts, false)?;
    let plan = Tld4Plan {
        op,
        // Cannot be `None`: the `.B` form is refused above.
        handle: handle.unwrap_or(HandleSrc::Reg(Gpr::RZ)),
        list,
        targets,
    };
    Ok((plan, buffer))
}

/// `TXD`, checked.
#[derive(Clone, Copy, Debug)]
struct TxdPlan {
    op: OpTxd,
    offset: u32,
    targets: Targets,
}

/// The byte offset of a constant-buffer handle, the only kind `TMML`, `TXD` and `TXQ` take
/// (their other kinds are refused by [`handle_form`] first).
fn cbuf_offset(tex: TexRef) -> u32 {
    match tex {
        TexRef::CBuf(cbuf) => u32::from(cbuf.offset),
        TexRef::Bindless | TexRef::Uniform(_) => 0,
    }
}

fn plan_txd(op: OpTxd, buffer: DescriptorBuffer) -> Result<(TxdPlan, DescriptorBuffer), TextureUnsupported> {
    handle_form("txd", op.tex, false, false)?;
    let buffer = bind_handle(buffer, op.tex)?;
    require(!op.offset, TextureUnsupported::GradientOffset)?;
    require(op.channel_mask.bits() != 0, TextureUnsupported::EmptyMask)?;
    require(
        matches!(op.dim, TexDim::D2 | TexDim::Array2D),
        TextureUnsupported::GradientDimension(op.dim),
    )?;
    let [coords, derivatives] = op.srcs;
    let last = if op.dim == TexDim::Array2D { 2 } else { 1 };
    require(
        after(coords, last) < REGISTER_LIMIT && after(derivatives, 3) < REGISTER_LIMIT,
        TextureUnsupported::GradientRegisters,
    )?;
    let targets = result_targets(op.channel_mask, op.dsts, false)?;
    let plan = TxdPlan {
        op,
        offset: cbuf_offset(op.tex),
        targets,
    };
    Ok((plan, buffer))
}

/// `TMML`, checked.
#[derive(Clone, Copy, Debug)]
struct TmmlPlan {
    op: OpTmml,
    offset: u32,
    targets: [Option<Gpr>; CHANNELS as usize],
}

fn plan_tmml(op: OpTmml, buffer: DescriptorBuffer) -> Result<(TmmlPlan, DescriptorBuffer), TextureUnsupported> {
    handle_form("tmml", op.tex, false, false)?;
    let [coords, extra] = op.srcs;
    require(extra.is_zero(), TextureUnsupported::MipQueryExtraRegister)?;
    let buffer = bind_handle(buffer, op.tex)?;
    let (_, coordinates) = tex_dimension(op.dim);
    require(
        op.dsts[1].is_zero(),
        TextureUnsupported::MipQueryExtraRegister,
    )?;
    let mask = op.channel_mask.bits();
    require(
        mask & !MIP_QUERY_CHANNELS == 0 && mask != 0,
        TextureUnsupported::MipQueryChannels(op.channel_mask),
    )?;
    require(
        after(coords, coordinates) < REGISTER_LIMIT,
        TextureUnsupported::MipQueryRegisters,
    )?;
    let targets = consecutive_targets(
        mask,
        op.dsts[0],
        TextureUnsupported::MipQueryResultRegister,
    )?;
    let plan = TmmlPlan {
        op,
        offset: cbuf_offset(op.tex),
        targets,
    };
    Ok((plan, buffer))
}

/// `TXQ`, checked.
#[derive(Clone, Copy, Debug)]
struct TxqPlan {
    op: OpTxq,
    offset: u32,
    targets: [Option<Gpr>; CHANNELS as usize],
}

fn plan_txq(op: OpTxq, buffer: DescriptorBuffer) -> Result<(TxqPlan, DescriptorBuffer), TextureUnsupported> {
    handle_form("txq", op.tex, false, false)?;
    let buffer = bind_handle(buffer, op.tex)?;
    require(
        op.query == TexQuery::Dimension,
        TextureUnsupported::QueryKind(op.query),
    )?;
    require(op.dsts[1].is_zero(), TextureUnsupported::QueryExtraRegister)?;
    let mask = op.channel_mask.bits();
    require(mask != 0, TextureUnsupported::EmptyQueryMask)?;
    let targets = consecutive_targets(
        mask,
        op.dsts[0],
        TextureUnsupported::QueryResultRegister,
    )?;
    let plan = TxqPlan {
        op,
        offset: cbuf_offset(op.tex),
        targets,
    };
    Ok((plan, buffer))
}

/// The image format and register count of a binary surface access (Eden's Maxwell `Format`
/// and `SizeInRegs`, `surface_load_store.cpp`).
const fn binary_format(mem_type: MemType) -> (ImageFormat, u32) {
    match mem_type {
        MemType::U8 => (ImageFormat::R8Uint, 1),
        MemType::I8 => (ImageFormat::R8Sint, 1),
        MemType::U16 => (ImageFormat::R16Uint, 1),
        MemType::I16 => (ImageFormat::R16Sint, 1),
        MemType::B32 => (ImageFormat::R32Uint, 1),
        MemType::B64 => (ImageFormat::R32G32Uint, 2),
        MemType::B128 => (ImageFormat::R32G32B32A32Uint, 4),
    }
}

/// The image format and channel mask of a surface access: a formatted access's mask with no
/// format, or a binary access's first `count` channels with the format its size names.
const fn access_format(access: ImageAccess) -> (ImageFormat, u8) {
    match access {
        ImageAccess::Formatted(mask) => (ImageFormat::Typeless, mask.bits()),
        ImageAccess::Binary(mem_type) => {
            let (format, count) = binary_format(mem_type);
            (format, (1_u8 << count) - 1)
        }
    }
}

/// `SULD`, checked.
#[derive(Clone, Copy, Debug)]
struct SuLdPlan {
    op: OpSuLd,
    format: ImageFormat,
    targets: [Option<Gpr>; CHANNELS as usize],
}

fn plan_suld(op: OpSuLd) -> Result<SuLdPlan, TextureUnsupported> {
    let (_, coordinates) = image_dimension(op.dim);
    let (format, mask) = access_format(op.access);
    require(mask != 0, TextureUnsupported::EmptySurfaceMask)?;
    require(
        after(op.coord, coordinates) < REGISTER_LIMIT,
        TextureUnsupported::SurfaceRegisters,
    )?;
    let targets = consecutive_targets(mask, op.dst, TextureUnsupported::SurfaceResultRegister)?;
    Ok(SuLdPlan {
        op,
        format,
        targets,
    })
}

/// `SUST`, checked.
#[derive(Clone, Copy, Debug)]
struct SuStPlan {
    op: OpSuSt,
    format: ImageFormat,
    /// How many data registers it reads.
    count: u32,
}

fn plan_sust(op: OpSuSt) -> Result<SuStPlan, TextureUnsupported> {
    let (_, coordinates) = image_dimension(op.dim);
    if let ImageAccess::Formatted(mask) = op.access {
        require(
            SURFACE_STORE_MASKS.contains(&mask.bits()),
            TextureUnsupported::SurfaceChannelMask(mask),
        )?;
    }
    let (format, mask) = access_format(op.access);
    let count = mask.count_ones();
    require(
        after(op.data, count) < REGISTER_LIMIT && after(op.coord, coordinates) < REGISTER_LIMIT,
        TextureUnsupported::SurfaceRegisters,
    )?;
    Ok(SuStPlan { op, format, count })
}

/// `SUATOM` and `SURED`, checked: the IR opcode of the operation.
#[derive(Clone, Copy, Debug)]
struct SuAtomPlan {
    op: OpSuAtom,
    opcode: Opcode,
}

/// The name of an atomic type, for [`TextureUnsupported::SurfaceAtomicOperation`].
const fn atom_type_name(ty: AtomType) -> &'static str {
    match ty {
        AtomType::U32 => "u32",
        AtomType::I32 => "s32",
        AtomType::U64 => "u64",
        AtomType::F32 => "f32",
        AtomType::F16v2 => "f16x2",
        AtomType::I64 => "s64",
        AtomType::F64 => "f64",
    }
}

/// The IR opcode of a 32-bit surface atomic (`ops_texture.cpp`, `SUATOM`'s `switch`; the
/// minimum and maximum signed for `S32`, `is_signed = type == 1`), or `None` for an operation
/// past exchange.
const fn atomic_opcode(op: AtomOp, signed: bool) -> Option<Opcode> {
    Some(match op {
        AtomOp::Add => Opcode::ImageAtomicIAdd32,
        AtomOp::Min if signed => Opcode::ImageAtomicSMin32,
        AtomOp::Min => Opcode::ImageAtomicUMin32,
        AtomOp::Max if signed => Opcode::ImageAtomicSMax32,
        AtomOp::Max => Opcode::ImageAtomicUMax32,
        AtomOp::Inc => Opcode::ImageAtomicInc32,
        AtomOp::Dec => Opcode::ImageAtomicDec32,
        AtomOp::And => Opcode::ImageAtomicAnd32,
        AtomOp::Or => Opcode::ImageAtomicOr32,
        AtomOp::Xor => Opcode::ImageAtomicXor32,
        AtomOp::Exch => Opcode::ImageAtomicExchange32,
        AtomOp::CmpExch(_) => return None,
    })
}

/// The name of an atomic operation, for [`TextureUnsupported::SurfaceAtomicOperation`].
const fn atom_op_name(op: AtomOp) -> &'static str {
    match op {
        AtomOp::Add => "add",
        AtomOp::Min => "min",
        AtomOp::Max => "max",
        AtomOp::Inc => "inc",
        AtomOp::Dec => "dec",
        AtomOp::And => "and",
        AtomOp::Or => "or",
        AtomOp::Xor => "xor",
        AtomOp::Exch => "exch",
        AtomOp::CmpExch(_) => "cas",
    }
}

fn plan_suatom(op: OpSuAtom) -> Result<SuAtomPlan, TextureUnsupported> {
    let (_, coordinates) = image_dimension(op.dim);
    require(!op.ba, TextureUnsupported::SurfaceAtomicMode)?;
    require(
        !matches!(op.atom_op, AtomOp::CmpExch(_)),
        TextureUnsupported::SurfaceCompareAndSwap,
    )?;
    let signed = op.atom_type == AtomType::I32;
    let opcode = match atomic_opcode(op.atom_op, signed) {
        Some(opcode) if matches!(op.atom_type, AtomType::U32 | AtomType::I32) => opcode,
        _ => {
            return Err(TextureUnsupported::SurfaceAtomicOperation {
                op: atom_op_name(op.atom_op),
                ty: atom_type_name(op.atom_type),
            });
        }
    };
    require(
        after(op.coord, coordinates) < REGISTER_LIMIT,
        TextureUnsupported::SurfaceRegisters,
    )?;
    Ok(SuAtomPlan { op, opcode })
}

// ---------------------------------------------------------------------------------------------
// Emission: no refusal past this point, only build errors.

/// A 32-bit immediate operand.
const fn imm32(bits: u32) -> Operand {
    Operand::Imm(Imm::U32(bits))
}

/// A 32-bit float immediate operand.
const fn f32_imm(value: f32) -> Operand {
    Operand::Imm(Imm::F32(value.to_bits()))
}

/// Appends one planned instruction.
struct Emit<'b, 'f> {
    b: &'b mut Builder<'f>,
}

impl Emit<'_, '_> {
    fn plan(&mut self, plan: &Plan) -> Result<(), BuildError> {
        match plan {
            Plan::Tex(plan) => self.tex(plan),
            Plan::Tld(plan) => self.tld(plan),
            Plan::Tld4(plan) => self.tld4(plan),
            Plan::Txd(plan) => self.txd(plan),
            Plan::Tmml(plan) => self.tmml(plan),
            Plan::Txq(plan) => self.txq(plan),
            Plan::SuLd(plan) => self.suld(plan),
            Plan::SuSt(plan) => self.sust(plan),
            Plan::SuAtom(plan) => self.suatom(plan),
        }
    }

    fn op(&mut self, opcode: Opcode, args: &[Operand]) -> Result<Operand, BuildError> {
        Ok(self.b.op(opcode, args)?.into())
    }

    /// A texture or image operation with its texture information.
    fn image(
        &mut self,
        opcode: Opcode,
        args: &[Operand],
        info: TextureInfo,
    ) -> Result<Operand, BuildError> {
        Ok(self.b.op_with(opcode, args, Flags::Texture(info))?.into())
    }

    /// A register's raw bits; `RZ` reads zero (`Visitor::X`).
    fn get(&mut self, reg: Gpr) -> Result<Operand, BuildError> {
        if reg.is_zero() {
            return Ok(imm32(0));
        }
        self.op(Opcode::GetRegister, &[Operand::Imm(Imm::Reg(reg))])
    }

    /// A register as a float (`Visitor::F`).
    fn get_f32(&mut self, reg: Gpr) -> Result<Operand, BuildError> {
        let bits = self.get(reg)?;
        self.op(Opcode::BitCastF32U32, &[bits])
    }

    /// Writes a register's raw bits; a write to `RZ` is dropped (no plan makes one).
    fn set(&mut self, reg: Gpr, value: Operand) -> Result<(), BuildError> {
        if !reg.is_zero() {
            self.op(Opcode::SetRegister, &[Operand::Imm(Imm::Reg(reg)), value])?;
        }
        Ok(())
    }

    fn set_f32(&mut self, reg: Gpr, value: Operand) -> Result<(), BuildError> {
        let bits = self.op(Opcode::BitCastU32F32, &[value])?;
        self.set(reg, bits)
    }

    /// The handle's value (module docs, "Handles").
    fn handle(&mut self, handle: HandleSrc) -> Result<Operand, BuildError> {
        match handle {
            HandleSrc::Offset(offset) => Ok(imm32(offset)),
            HandleSrc::Uniform(reg) => self.op(
                Opcode::GetUniformRegister,
                &[Operand::Imm(Imm::UniformReg(reg))],
            ),
            HandleSrc::Reg(reg) => self.get(reg),
        }
    }

    /// `count` bits of `value` from `offset`, zero- or sign-extended (Eden's
    /// `BitFieldExtract`).
    fn extract(
        &mut self,
        value: Operand,
        offset: u32,
        count: u32,
        signed: bool,
    ) -> Result<Operand, BuildError> {
        let opcode = if signed {
            Opcode::BitFieldSExtract
        } else {
            Opcode::BitFieldUExtract
        };
        self.op(opcode, &[value, imm32(offset), imm32(count)])
    }

    /// An array index: the low 16 bits of the register.
    fn layer(&mut self, reg: Gpr) -> Result<Operand, BuildError> {
        let raw = self.get(reg)?;
        self.extract(raw, 0, ARRAY_INDEX_BITS, false)
    }

    /// An array index as a float (`ConvertUToF(32, 16, ..)`).
    fn layer_f32(&mut self, reg: Gpr) -> Result<Operand, BuildError> {
        let layer = self.layer(reg)?;
        self.op(Opcode::ConvertF32U16, &[layer])
    }

    /// A vector of `parts` (a scalar for one part), with `opcodes` for two, three and four.
    fn construct(
        &mut self,
        parts: &[Operand],
        opcodes: [Opcode; 3],
    ) -> Result<Operand, BuildError> {
        match parts {
            [one] => Ok(*one),
            [_, _] => self.op(opcodes[0], parts),
            [_, _, _] => self.op(opcodes[1], parts),
            _ => self.op(opcodes[2], parts),
        }
    }

    fn construct_f32(&mut self, parts: &[Operand]) -> Result<Operand, BuildError> {
        self.construct(
            parts,
            [
                Opcode::CompositeConstructF32x2,
                Opcode::CompositeConstructF32x3,
                Opcode::CompositeConstructF32x4,
            ],
        )
    }

    fn construct_u32(&mut self, parts: &[Operand]) -> Result<Operand, BuildError> {
        self.construct(
            parts,
            [
                Opcode::CompositeConstructU32x2,
                Opcode::CompositeConstructU32x3,
                Opcode::CompositeConstructU32x4,
            ],
        )
    }

    fn extract_f32x4(&mut self, vector: Operand, index: u32) -> Result<Operand, BuildError> {
        self.op(Opcode::CompositeExtractF32x4, &[vector, imm32(index)])
    }

    fn extract_u32x4(&mut self, vector: Operand, index: u32) -> Result<Operand, BuildError> {
        self.op(Opcode::CompositeExtractU32x4, &[vector, imm32(index)])
    }

    /// Float coordinates of a sample or gather, from the list (`TEX`'s and `TLD4`'s
    /// `switch (dim.type)`): an array index first, then the coordinates.
    fn float_coords(&mut self, dim: TexDim, list: &mut Cursor<'_>) -> Result<Operand, BuildError> {
        let layer = if is_array(dim) {
            Some(self.layer_f32(list.take())?)
        } else {
            None
        };
        let (_, coordinates) = tex_dimension(dim);
        let mut parts = [imm32(0); CHANNELS as usize];
        let spatial = coordinates - u32::from(layer.is_some());
        let mut len = 0;
        for part in parts.iter_mut().take(spatial as usize) {
            *part = self.get_f32(list.take())?;
            len += 1;
        }
        if let (Some(layer), Some(part)) = (layer, parts.get_mut(len)) {
            *part = layer;
            len += 1;
        }
        self.construct_f32(parts.get(..len).unwrap_or(&[]))
    }

    /// Writes a sampled result's channels (`values`, in channel order, `None` for a channel the
    /// mask leaves out) to `targets`.
    fn write_sampled(
        &mut self,
        values: [Option<Operand>; CHANNELS as usize],
        targets: Targets,
    ) -> Result<(), BuildError> {
        match targets {
            Targets::Float(regs) => {
                for (value, reg) in values.into_iter().zip(regs) {
                    if let (Some(value), Some(reg)) = (value, reg) {
                        self.set_f32(reg, value)?;
                    }
                }
            }
            Targets::Half { low, high } => {
                let zero = f32_imm(0.0);
                let mut stored = [zero; CHANNELS as usize];
                for (slot, value) in stored.iter_mut().zip(values.into_iter().flatten()) {
                    *slot = value;
                }
                let [r, g, b, a] = stored;
                let pair = self.construct_f32(&[r, g])?;
                let packed = self.op(Opcode::PackHalf2x16, &[pair])?;
                self.set(low, packed)?;
                if let Some(high) = high {
                    let pair = self.construct_f32(&[b, a])?;
                    let packed = self.op(Opcode::PackHalf2x16, &[pair])?;
                    self.set(high, packed)?;
                }
            }
        }
        Ok(())
    }

    /// The selected channels of a four-float result.
    fn channels(
        &mut self,
        sample: Operand,
        mask: ChannelMask,
    ) -> Result<[Option<Operand>; CHANNELS as usize], BuildError> {
        let mut values = [None; CHANNELS as usize];
        for (channel, value) in (0..CHANNELS).zip(&mut values) {
            if mask.bits() & (1 << channel) != 0 {
                *value = Some(self.extract_f32x4(sample, u32::from(channel))?);
            }
        }
        Ok(values)
    }

    /// Writes "resident" to a fault predicate: false, as every texel is reported resident
    /// (module docs, the divergence table); nothing for `PT` (Eden: `SetPred(sparse_pred,
    /// LogicalNot(GetSparseFromOp(sample)))`).
    fn resident(&mut self, fault: Pred) -> Result<(), BuildError> {
        if !fault.is_true() {
            self.op(
                Opcode::SetPred,
                &[Operand::Imm(Imm::Pred(fault)), Operand::Imm(Imm::U1(false))],
            )?;
        }
        Ok(())
    }

    fn tex(&mut self, plan: &TexPlan) -> Result<(), BuildError> {
        let op = plan.op;
        let mut list = Cursor::new(&plan.list);
        let clamp_source = matches!(op.lod_mode, TexLodMode::Clamp | TexLodMode::BiasClamp);
        // A non-array texture's clamp word comes first; an array texture's shares the array
        // index register, which `float_coords` reads next.
        let clamp_reg = match (clamp_source, is_array(op.dim)) {
            (false, _) => None,
            (true, false) => Some(list.take()),
            (true, true) => list.regs.first().copied(),
        };
        let coords = self.float_coords(op.dim, &mut list)?;
        let has_lod = matches!(
            op.lod_mode,
            TexLodMode::Bias | TexLodMode::Lod | TexLodMode::BiasClamp
        );
        let lod = if has_lod {
            self.get_f32(list.take())?
        } else {
            f32_imm(0.0)
        };
        let dref = if op.z_cmpr {
            Some(self.get_f32(list.take())?)
        } else {
            None
        };
        let clamp = match clamp_reg {
            Some(reg) => {
                let raw = self.get(reg)?;
                let fixed = self.op(Opcode::ShiftRightLogical32, &[raw, imm32(MIN_LOD_SHIFT)])?;
                let level = self.op(Opcode::ConvertF32U32, &[fixed])?;
                Some(self.op(Opcode::FPMul32, &[level, f32_imm(LOD_FIXED_UNIT)])?)
            }
            None => None,
        };
        let (ty, _) = tex_dimension(op.dim);
        let info = TextureInfo {
            ty,
            is_depth: op.z_cmpr,
            has_bias: matches!(op.lod_mode, TexLodMode::Bias | TexLodMode::BiasClamp),
            has_lod_clamp: clamp.is_some(),
            relaxed_precision: op.f16,
            ..TextureInfo::default()
        };
        let handle = self.handle(plan.handle)?;
        let explicit = matches!(op.lod_mode, TexLodMode::Zero | TexLodMode::Lod);
        let bias_clamp = match (info.has_bias, clamp) {
            (true, Some(clamp)) => self.construct_f32(&[lod, clamp])?,
            (true, None) => lod,
            (false, Some(clamp)) => clamp,
            (false, None) => ABSENT,
        };
        let offset = match plan.offset {
            Some(reg) => {
                let packed = self.get(reg)?;
                let x = self.extract(packed, 0, TEX_OFFSET_BITS, true)?;
                let y = self.extract(packed, TEX_OFFSET_BITS, TEX_OFFSET_BITS, true)?;
                self.construct_u32(&[x, y])?
            }
            None => ABSENT,
        };
        let values = match dref {
            None => {
                let sample = if explicit {
                    self.image(
                        Opcode::ImageSampleExplicitLod,
                        &[handle, coords, lod, offset],
                        info,
                    )?
                } else {
                    self.image(
                        Opcode::ImageSampleImplicitLod,
                        &[handle, coords, bias_clamp, offset],
                        info,
                    )?
                };
                self.channels(sample, op.channel_mask)?
            }
            Some(dref) => {
                let sample = if explicit {
                    self.image(
                        Opcode::ImageSampleDrefExplicitLod,
                        &[handle, coords, dref, lod, ABSENT],
                        info,
                    )?
                } else {
                    self.image(
                        Opcode::ImageSampleDrefImplicitLod,
                        &[handle, coords, dref, bias_clamp, ABSENT],
                        info,
                    )?
                };
                let mut values = [None; CHANNELS as usize];
                for (channel, value) in (0..CHANNELS).zip(&mut values) {
                    if op.channel_mask.bits() & (1 << channel) != 0 {
                        *value = Some(if channel < 3 { sample } else { f32_imm(1.0) });
                    }
                }
                values
            }
        };
        self.write_sampled(values, plan.targets)?;
        self.resident(op.fault)
    }

    fn tld(&mut self, plan: &TldPlan) -> Result<(), BuildError> {
        let op = plan.op;
        let mut list = Cursor::new(&plan.list);
        let layer = if is_array(op.dim) {
            Some(self.layer(list.take())?)
        } else {
            None
        };
        let (ty, coordinates) = tex_dimension(op.dim);
        let spatial = coordinates - u32::from(layer.is_some());
        let mut parts = [imm32(0); CHANNELS as usize];
        let mut len = 0;
        for part in parts.iter_mut().take(spatial as usize) {
            *part = self.get(list.take())?;
            len += 1;
        }
        if let (Some(layer), Some(part)) = (layer, parts.get_mut(len)) {
            *part = layer;
            len += 1;
        }
        let coords = self.construct_u32(parts.get(..len).unwrap_or(&[]))?;
        let lod = if op.lod_mode == TexLodMode::Lod {
            self.get(list.take())?
        } else {
            imm32(0)
        };
        let info = TextureInfo {
            ty,
            ..TextureInfo::default()
        };
        let handle = self.handle(plan.handle)?;
        let sample = self.image(
            Opcode::ImageFetch,
            &[handle, coords, ABSENT, lod, ABSENT],
            info,
        )?;
        let values = self.channels(sample, op.channel_mask)?;
        self.write_sampled(values, plan.targets)?;
        self.resident(op.fault)
    }

    /// A `TLD4` offset register's signed fields: `count` fields from bit 0, one every
    /// [`GATHER_OFFSET_STRIDE`] bits.
    fn gather_offsets(
        &mut self,
        reg: Gpr,
        count: u32,
    ) -> Result<[Operand; CHANNELS as usize], BuildError> {
        let value = self.get(reg)?;
        let mut fields = [imm32(0); CHANNELS as usize];
        for (index, field) in (0..count).zip(&mut fields) {
            *field = self.extract(
                value,
                index * GATHER_OFFSET_STRIDE,
                GATHER_OFFSET_BITS,
                true,
            )?;
        }
        Ok(fields)
    }

    fn tld4(&mut self, plan: &Tld4Plan) -> Result<(), BuildError> {
        let op = plan.op;
        let mut list = Cursor::new(&plan.list);
        let coords = self.float_coords(op.dim, &mut list)?;
        let offsets = match op.offset_mode {
            TexOffsetMode::None => None,
            TexOffsetMode::AddOffI => {
                let [x, y, _, _] = self.gather_offsets(list.take(), 2)?;
                Some([x, y, imm32(0), imm32(0), imm32(0), imm32(0), imm32(0), imm32(0)])
            }
            TexOffsetMode::PerPx => {
                let [x0, y0, x1, y1] = self.gather_offsets(list.take(), CHANNELS.into())?;
                let [x2, y2, x3, y3] = self.gather_offsets(list.take(), CHANNELS.into())?;
                Some([x0, y0, x1, y1, x2, y2, x3, y3])
            }
        };
        let dref = if op.z_cmpr {
            Some(self.get_f32(list.take())?)
        } else {
            None
        };
        let (ty, _) = tex_dimension(op.dim);
        let info = TextureInfo {
            ty,
            is_depth: op.z_cmpr,
            gather_component: u8::try_from(op.comp.bits()).unwrap_or(0),
            ..TextureInfo::default()
        };
        let handle = self.handle(plan.handle)?;
        let sample = match offsets {
            None => self.gather(handle, coords, ABSENT, dref, info)?,
            Some([x, y, ..]) if op.offset_mode == TexOffsetMode::AddOffI => {
                let offset = self.construct_u32(&[x, y])?;
                self.gather(handle, coords, offset, dref, info)?
            }
            Some(fields) => {
                let mut texels = [imm32(0); CHANNELS as usize];
                for (texel, pair) in texels.iter_mut().zip(fields.chunks_exact(2)) {
                    let offset = self.construct_u32(pair)?;
                    let gathered = self.gather(handle, coords, offset, dref, info)?;
                    *texel = self.extract_f32x4(gathered, GATHER_BASE_TEXEL)?;
                }
                self.construct_f32(&texels)?
            }
        };
        let values = self.channels(sample, op.channel_mask)?;
        self.write_sampled(values, plan.targets)?;
        self.resident(op.fault)
    }

    /// One `ImageGather`, or `ImageGatherDref` with a depth reference; the second offset is
    /// never passed (module docs, "Per-texel gather offsets").
    fn gather(
        &mut self,
        handle: Operand,
        coords: Operand,
        offset: Operand,
        dref: Option<Operand>,
        info: TextureInfo,
    ) -> Result<Operand, BuildError> {
        match dref {
            Some(dref) => self.image(
                Opcode::ImageGatherDref,
                &[handle, coords, offset, ABSENT, dref],
                info,
            ),
            None => self.image(
                Opcode::ImageGather,
                &[handle, coords, offset, ABSENT],
                info,
            ),
        }
    }

    fn txd(&mut self, plan: &TxdPlan) -> Result<(), BuildError> {
        let op = plan.op;
        let [coord, derivative] = op.srcs;
        let x = self.get_f32(coord)?;
        let y = self.get_f32(gpr(after(coord, 1)))?;
        let coords = if op.dim == TexDim::Array2D {
            let layer = self.layer_f32(gpr(after(coord, 2)))?;
            self.construct_f32(&[x, y, layer])?
        } else {
            self.construct_f32(&[x, y])?
        };
        let mut parts = [imm32(0); CHANNELS as usize];
        for (k, part) in (0..).zip(&mut parts) {
            *part = self.get_f32(gpr(after(derivative, k)))?;
        }
        let derivatives = self.construct_f32(&parts)?;
        let (ty, _) = tex_dimension(op.dim);
        let info = TextureInfo {
            ty,
            num_derivatives: 2,
            ..TextureInfo::default()
        };
        let sample = self.image(
            Opcode::ImageGradient,
            &[imm32(plan.offset), coords, derivatives, ABSENT, ABSENT],
            info,
        )?;
        let values = self.channels(sample, op.channel_mask)?;
        self.write_sampled(values, plan.targets)?;
        self.resident(op.fault)
    }

    fn tmml(&mut self, plan: &TmmlPlan) -> Result<(), BuildError> {
        let op = plan.op;
        let base = op.srcs[0];
        let reg = |k| gpr(after(base, k));
        let mut parts = [imm32(0); 3];
        let first = u32::from(is_array(op.dim));
        let count: u32 = match op.dim {
            TexDim::D1 | TexDim::Array1D => 1,
            TexDim::D2 | TexDim::Array2D => 2,
            TexDim::D3 | TexDim::Cube | TexDim::ArrayCube => 3,
        };
        for (k, part) in (first..first + count).zip(&mut parts) {
            *part = self.get_f32(reg(k))?;
        }
        let coords = self.construct_f32(parts.get(..count as usize).unwrap_or(&[]))?;
        let (ty, _) = tex_dimension(op.dim);
        let info = TextureInfo {
            ty,
            ..TextureInfo::default()
        };
        let lod = self.image(
            Opcode::ImageQueryLod,
            &[imm32(plan.offset), coords],
            info,
        )?;
        for (element, target) in (0_u32..).zip(plan.targets) {
            let Some(target) = target else { continue };
            let level = self.extract_f32x4(lod, element ^ 1)?;
            let scaled = self.op(Opcode::FPMul32, &[level, f32_imm(LOD_FIXED_SCALE)])?;
            let fixed = if element == 0 {
                self.op(Opcode::ConvertS32F32, &[scaled])?
            } else {
                self.op(Opcode::ConvertU32F32, &[scaled])?
            };
            let masked = self.op(Opcode::BitwiseAnd32, &[fixed, imm32(LOD_FIXED_MASK)])?;
            self.set(target, masked)?;
        }
        Ok(())
    }

    fn txq(&mut self, plan: &TxqPlan) -> Result<(), BuildError> {
        let op = plan.op;
        let level = self.get(op.src)?;
        let skip_levels = op.channel_mask.bits() & (1 << QUERY_LEVELS) == 0;
        let query = self.image(
            Opcode::ImageQueryDimensions,
            &[imm32(plan.offset), level, Operand::Imm(Imm::U1(skip_levels))],
            TextureInfo::default(),
        )?;
        for (element, target) in (0_u32..).zip(plan.targets) {
            if let Some(target) = target {
                let word = self.extract_u32x4(query, element)?;
                self.set(target, word)?;
            }
        }
        Ok(())
    }

    /// A surface's integer coordinates from `reg`, an array index last
    /// (`SurfaceCoordinates`).
    fn surface_coords(&mut self, dim: ImageDim, reg: Gpr) -> Result<Operand, BuildError> {
        let (_, coordinates) = image_dimension(dim);
        let array = matches!(dim, ImageDim::Array1D | ImageDim::Array2D);
        let mut parts = [imm32(0); 3];
        let mut len = 0;
        for (k, part) in (0..coordinates).zip(&mut parts) {
            let reg = gpr(after(reg, k));
            *part = if array && k + 1 == coordinates {
                self.layer(reg)?
            } else {
                self.get(reg)?
            };
            len += 1;
        }
        self.construct_u32(parts.get(..len).unwrap_or(&[]))
    }

    fn suld(&mut self, plan: &SuLdPlan) -> Result<(), BuildError> {
        let op = plan.op;
        let coords = self.surface_coords(op.dim, op.coord)?;
        let handle = self.get(op.handle)?;
        let (ty, _) = image_dimension(op.dim);
        let info = TextureInfo {
            ty,
            image_format: plan.format,
            ..TextureInfo::default()
        };
        let result = self.image(Opcode::ImageRead, &[handle, coords], info)?;
        for (channel, target) in (0_u32..).zip(plan.targets) {
            if let Some(target) = target {
                let word = self.extract_u32x4(result, channel)?;
                self.set(target, word)?;
            }
        }
        self.resident(op.fault)
    }

    fn sust(&mut self, plan: &SuStPlan) -> Result<(), BuildError> {
        let op = plan.op;
        let coords = self.surface_coords(op.dim, op.coord)?;
        let handle = self.get(op.handle)?;
        let mut colors = [imm32(0); CHANNELS as usize];
        for (k, color) in (0..plan.count).zip(&mut colors) {
            *color = self.get(gpr(after(op.data, k)))?;
        }
        let color = self.construct_u32(&colors)?;
        let (ty, _) = image_dimension(op.dim);
        let info = TextureInfo {
            ty,
            image_format: plan.format,
            ..TextureInfo::default()
        };
        self.image(Opcode::ImageWrite, &[handle, coords, color], info)?;
        Ok(())
    }

    fn suatom(&mut self, plan: &SuAtomPlan) -> Result<(), BuildError> {
        let op = plan.op;
        let coords = self.surface_coords(op.dim, op.coord)?;
        let handle = self.get(op.handle)?;
        let value = self.get(op.data)?;
        let (ty, _) = image_dimension(op.dim);
        let info = TextureInfo {
            ty,
            image_format: ImageFormat::Typeless,
            ..TextureInfo::default()
        };
        let old = self.image(plan.opcode, &[handle, coords, value], info)?;
        if !op.reduce {
            self.set(op.dst, old)?;
        }
        self.resident(op.fault)
    }
}

//! Slice 6f's encoders: NAK's `impl SM70Op` for `OpTex`, `OpTld`, `OpTld4`, `OpTmml`, `OpTxd`,
//! `OpTxq`, `OpSuLd`, `OpSuSt` and `OpSuAtom` below SM 10.0 (`decode::op::texture`), with PR
//! #4588's `.UR` forms and `.F16` and `.CL` bits. Where NAK writes a constant (`SUATOM`'s `.BA`,
//! `SURED`'s destination), the encoder writes the op's field instead.
//!
//! An op the forms cannot hold is refused: a handle kind the instruction has no form for (a
//! uniform register for `TMML`, `TXD` and `TXQ`), `.F16` with a `.B` handle and `.CL` without a
//! `.UR` one ([`EncodeError::NoBits`]: the rows leave those bits out), a constant-buffer handle
//! past its fields ([`EncodeError::TooWide`]) or not word-aligned ([`EncodeError::Misaligned`],
//! NAK: `assert!(cb.offset % 4 == 0)`), and an atomic operation `SUATOM`'s forms do not encode
//! (a separate compare source, NAK: `assert!(cmp_src == AtomCmpSrc::Packed)`; a reduction that
//! three bits cannot hold).

use super::{Encode, EncodeError, Writer};
use crate::decode::texture::{
    ATOM_TYPE, CHANNEL_MASK, EVICT, FAULT, IMAGE_DIM, MEM_ORDER, MEM_TYPE, REG_A, REG_B, SU_HANDLE,
    SUATOM_BA, SUATOM_CAS_OPCODE, SUATOM_OP, SUATOM_OPCODE, SULD_BINARY_OPCODE,
    SULD_FORMATTED_OPCODE, SURED_OP, SURED_OPCODE, SUST_BINARY_OPCODE, SUST_FORMATTED_OPCODE,
    TEX_BINDLESS, TEX_CBUF_INDEX, TEX_CBUF_OFFSET, TEX_CBUF_UNIT, TEX_DC, TEX_DIM, TEX_DST2,
    TEX_F16, TEX_LOD, TEX_NDV, TEX_NODEP, TEX_OFFSET, TEX_OPCODES, TEX_SCALAR, TEX_UNIFORM,
    TEX_UNIFORM_HANDLE, TLD_CLAMP, TLD_MS, TLD_OPCODES, TLD4_COMP, TLD4_OFFSET, TLD4_OPCODES,
    TMML_OPCODES, TXD_OPCODES, TXQ_OPCODES, TXQ_QUERY, TexOpcodes, has_clamp, has_f16,
};
use crate::decode::{
    AtomCmpSrc, AtomOp, ChannelMask, DST, Gpr, ImageAccess, ImageDim, MemEvictionPriority,
    MemOrder, Named, OpSuAtom, OpSuLd, OpSuSt, OpTex, OpTld, OpTld4, OpTmml, OpTxd, OpTxq, Pred,
    TexDim, TexRef,
};

/// Writes the opcode of the form for `tex` and the handle's fields
/// (`set_tex_cb_ref(40..59, cb)`, `set_bit(59, true); // .B`, or PR #4588's uniform register at
/// 40..46 and bit 91).
fn tex_ref(
    writer: &mut Writer,
    mnemonic: &'static str,
    opcodes: TexOpcodes,
    tex: TexRef,
) -> Result<(), EncodeError> {
    let opcode = opcodes.of(tex).ok_or(EncodeError::NoBits {
        mnemonic,
        what: "a uniform-register texture handle",
    })?;
    match tex {
        TexRef::CBuf(cbuf) => {
            if !cbuf.offset.is_multiple_of(TEX_CBUF_UNIT) {
                return Err(EncodeError::Misaligned {
                    what: "texture handle offset",
                    value: u64::from(cbuf.offset),
                    align: u64::from(TEX_CBUF_UNIT),
                });
            }
            writer.checked(
                TEX_CBUF_INDEX,
                "texture handle buffer",
                u64::from(cbuf.index),
            )?;
            writer.set(TEX_CBUF_OFFSET, u64::from(cbuf.offset / TEX_CBUF_UNIT));
        }
        TexRef::Bindless => writer.flag(TEX_BINDLESS, true),
        TexRef::Uniform(reg) => {
            writer.uniform_reg(TEX_UNIFORM_HANDLE, reg);
            writer.flag(TEX_UNIFORM, true);
        }
    }
    writer.opcode(opcode);
    Ok(())
}

/// The fields `TEX`, `TLD`, `TLD4`, `TMML` and `TXD` share: both destinations (`set_dst`, and
/// `set_reg(64..72, ..)`), both sources, the dimension and the mask.
fn tex_common(writer: &mut Writer, dsts: [Gpr; 2], srcs: [Gpr; 2], dim: TexDim, mask: ChannelMask) {
    let [first, second] = dsts;
    let [a, b] = srcs;
    writer.gpr(DST, first);
    writer.gpr(TEX_DST2, second);
    writer.gpr(REG_A, a);
    writer.gpr(REG_B, b);
    writer.set(TEX_DIM, dim.bits());
    writer.set(CHANNEL_MASK, u64::from(mask.bits()));
}

/// The fault predicate (`set_pred_dst(81..84, &self.fault)`) and the eviction priority.
fn fault_and_eviction(writer: &mut Writer, fault: Pred, eviction: MemEvictionPriority) {
    writer.set(FAULT, u64::from(fault.index()));
    writer.set(EVICT, eviction.bits());
}

/// `.F16`, which the `.B` forms have no bit for.
fn f16(
    writer: &mut Writer,
    mnemonic: &'static str,
    tex: TexRef,
    on: bool,
) -> Result<(), EncodeError> {
    if on && !has_f16(tex) {
        return Err(EncodeError::NoBits {
            mnemonic,
            what: ".f16 with a handle in the sources",
        });
    }
    writer.flag(TEX_F16, on);
    Ok(())
}

impl Encode for OpTex {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        tex_ref(writer, Self::MNEMONIC, TEX_OPCODES, self.tex)?;
        f16(writer, Self::MNEMONIC, self.tex, self.f16)?;
        tex_common(writer, self.dsts, self.srcs, self.dim, self.channel_mask);
        fault_and_eviction(writer, self.fault, self.eviction);
        writer.flag(TEX_SCALAR, self.scalar);
        writer.flag(TEX_OFFSET, self.offset);
        writer.flag(TEX_NDV, self.ndv);
        writer.flag(TEX_DC, self.z_cmpr);
        writer.set(TEX_LOD, self.lod_mode.bits());
        writer.flag(TEX_NODEP, self.nodep);
        Ok(())
    }
}

impl Encode for OpTld {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        tex_ref(writer, Self::MNEMONIC, TLD_OPCODES, self.tex)?;
        f16(writer, Self::MNEMONIC, self.tex, self.f16)?;
        if self.clamp && !has_clamp(self.tex) {
            return Err(EncodeError::NoBits {
                mnemonic: Self::MNEMONIC,
                what: ".cl without a uniform-register handle",
            });
        }
        writer.flag(TLD_CLAMP, self.clamp);
        tex_common(writer, self.dsts, self.srcs, self.dim, self.channel_mask);
        fault_and_eviction(writer, self.fault, self.eviction);
        writer.flag(TEX_SCALAR, self.scalar);
        writer.flag(TEX_OFFSET, self.offset);
        writer.flag(TLD_MS, self.is_ms);
        writer.set(TEX_LOD, self.lod_mode.bits());
        writer.flag(TEX_NODEP, self.nodep);
        Ok(())
    }
}

impl Encode for OpTld4 {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        tex_ref(writer, Self::MNEMONIC, TLD4_OPCODES, self.tex)?;
        tex_common(writer, self.dsts, self.srcs, self.dim, self.channel_mask);
        fault_and_eviction(writer, self.fault, self.eviction);
        writer.flag(TEX_SCALAR, self.scalar);
        writer.set(TLD4_OFFSET, self.offset_mode.bits());
        writer.flag(TEX_DC, self.z_cmpr);
        writer.set(TLD4_COMP, self.comp.bits());
        writer.flag(TEX_NODEP, self.nodep);
        Ok(())
    }
}

impl Encode for OpTmml {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        tex_ref(writer, Self::MNEMONIC, TMML_OPCODES, self.tex)?;
        tex_common(writer, self.dsts, self.srcs, self.dim, self.channel_mask);
        writer.flag(TEX_NDV, self.ndv);
        writer.flag(TEX_NODEP, self.nodep);
        Ok(())
    }
}

impl Encode for OpTxd {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        tex_ref(writer, Self::MNEMONIC, TXD_OPCODES, self.tex)?;
        tex_common(writer, self.dsts, self.srcs, self.dim, self.channel_mask);
        fault_and_eviction(writer, self.fault, self.eviction);
        writer.flag(TEX_OFFSET, self.offset);
        writer.flag(TEX_NODEP, self.nodep);
        Ok(())
    }
}

impl Encode for OpTxq {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        tex_ref(writer, Self::MNEMONIC, TXQ_OPCODES, self.tex)?;
        let [dst, dst2] = self.dsts;
        writer.gpr(DST, dst);
        writer.gpr(TEX_DST2, dst2);
        writer.gpr(REG_A, self.src);
        writer.set(TXQ_QUERY, self.query.bits());
        writer.set(CHANNEL_MASK, u64::from(self.channel_mask.bits()));
        writer.flag(TEX_NODEP, self.nodep);
        Ok(())
    }
}

/// The fields every surface instruction has (`set_reg_src(24..32, &self.coord)`,
/// `set_reg_src(64..72, &self.handle)`, `set_image_dim`, `set_mem_order`,
/// `set_eviction_priority`).
fn surface(
    writer: &mut Writer,
    coord: Gpr,
    handle: Gpr,
    dim: ImageDim,
    order: MemOrder,
    eviction: MemEvictionPriority,
) {
    writer.gpr(REG_A, coord);
    writer.gpr(SU_HANDLE, handle);
    writer.set(IMAGE_DIM, dim.bits());
    writer.set(MEM_ORDER, order.bits());
    writer.set(EVICT, eviction.bits());
}

/// The opcode an access picks and its field: a memory type (`set_mem_type(73..76, ..)`) or a
/// channel mask (`set_image_channel_mask(72..76, ..)`).
fn access(writer: &mut Writer, access: ImageAccess, binary: u16, formatted: u16) {
    match access {
        ImageAccess::Binary(mem_type) => {
            writer.opcode(binary);
            writer.set(MEM_TYPE, mem_type.bits());
        }
        ImageAccess::Formatted(mask) => {
            writer.opcode(formatted);
            writer.set(CHANNEL_MASK, u64::from(mask.bits()));
        }
    }
}

impl Encode for OpSuLd {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        access(
            writer,
            self.access,
            SULD_BINARY_OPCODE,
            SULD_FORMATTED_OPCODE,
        );
        writer.gpr(DST, self.dst);
        writer.set(FAULT, u64::from(self.fault.index()));
        surface(
            writer,
            self.coord,
            self.handle,
            self.dim,
            self.order,
            self.eviction,
        );
        Ok(())
    }
}

impl Encode for OpSuSt {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        access(
            writer,
            self.access,
            SUST_BINARY_OPCODE,
            SUST_FORMATTED_OPCODE,
        );
        writer.gpr(REG_B, self.data);
        surface(
            writer,
            self.coord,
            self.handle,
            self.dim,
            self.order,
            self.eviction,
        );
        Ok(())
    }
}

impl Encode for OpSuAtom {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let mnemonic = Self::MNEMONIC;
        match (self.atom_op, self.reduce) {
            (AtomOp::CmpExch(AtomCmpSrc::Packed), false) => writer.opcode(SUATOM_CAS_OPCODE),
            (AtomOp::CmpExch(AtomCmpSrc::Separate { .. }), _) => {
                return Err(EncodeError::NoBits {
                    mnemonic,
                    what: "a separate compare source",
                });
            }
            (AtomOp::CmpExch(AtomCmpSrc::Packed), true) => {
                return Err(EncodeError::NoBits {
                    mnemonic,
                    what: "a compare-and-swap without a result",
                });
            }
            (op, reduce) => {
                let (opcode, field) = if reduce {
                    (SURED_OPCODE, SURED_OP)
                } else {
                    (SUATOM_OPCODE, SUATOM_OP)
                };
                // `bits` is `None` only for `CmpExch`, handled above.
                let bits = op.bits().unwrap_or(u64::MAX);
                writer.checked(field, "atomic operation", bits)?;
                writer.opcode(opcode);
            }
        }
        writer.gpr(DST, self.dst);
        writer.gpr(REG_B, self.data);
        writer.set(FAULT, u64::from(self.fault.index()));
        writer.set(ATOM_TYPE, self.atom_type.bits());
        writer.flag(SUATOM_BA, self.ba);
        surface(
            writer,
            self.coord,
            self.handle,
            self.dim,
            self.order,
            self.eviction,
        );
        Ok(())
    }
}

//! Slice 6e's encoders: NAK's `impl SM70Op` for `OpLd`, `OpSt`, `OpLdc`, `OpAtom`, `OpCCtl` and
//! `OpMemBar` at SM 8.6 (`has_ugpr`, both forms where NAK has two), and PR #4588's rows and field
//! reads for `OpLdGsts`, `OpLdgDepBar`, `OpErrBar` and `OpAtomsPopc` (`decode::op::memory`).
//! Where NAK writes a constant, the encoder writes the op's field instead (the decoder's module
//! docs list them). A field the chosen form has no bits for is refused
//! ([`EncodeError::NoBits`]) rather than dropped, so every encoded instruction decodes back to
//! itself.

use super::{Encode, EncodeError, Writer};
use crate::decode::control::{COND, PRED_DST, UNIFORM_BIT};
use crate::decode::memory::{
    ATOM_OP, ATOM_TYPE, ATOMG_CAS_OPCODE, ATOMG_OPCODE, ATOMG_UNIFORM, ATOMG_UNIFORM_OPCODE,
    ATOMS_CAS_OPCODE, ATOMS_OPCODE, ATOMS_POPC_FIXED_FIELDS, ATOMS_POPC_OPCODE,
    ATOMS_UNIFORM_OPCODE, AddrLayout, CAS_CMPR, CAS_DATA, CCTL_OFFSET, CCTL_OP, CCTL_OPCODE,
    ERRBAR_OPCODE, GLOBAL, LD_LOCAL_SHARED, LDC_BINDLESS_OFFSET, LDC_BINDLESS_OPCODE, LDC_HANDLE,
    LDC_MODE, LDC_OFFSET, LDC_OPCODE, LDG_OPCODE, LDG_PRED, LDG_PRED_NOT, LDG_UNIFORM,
    LDG_UNIFORM_OPCODE, LDGDEPBAR_OPCODE, LDGSTS_DESCRIPTOR, LDGSTS_ENABLE_NOT,
    LDGSTS_FIXED_FIELDS, LDGSTS_GLOBAL, LDGSTS_GLOBAL_OFFSET, LDGSTS_OPCODE, LDGSTS_SHARED,
    LDGSTS_SHARED_OFFSET, LDL_OPCODE, LDS_OPCODE, LDS_ZD, MEM_ADDR, MEM_BIT_76, MEM_DATA, MEM_E,
    MEM_EVICTION, MEM_OFFSET, MEM_ORDER, MEM_STRIDE, MEM_TYPE, MEMBAR_MMIO, MEMBAR_OPCODE,
    MEMBAR_SC, MEMBAR_SCOPE, RED_OP, RED_OPCODE, ST_GLOBAL_UNIFORM, ST_UNIFORM_PLAIN, STG_OPCODE,
    STG_UNIFORM_OPCODE, STL_OPCODE, STL_UNIFORM_OPCODE, STS_OPCODE, STS_UNIFORM_OPCODE,
};
use crate::decode::{
    AtomDst, AtomOp, CBUF_INDEX, CBUF_OFFSET, CBuf, DST, LdcMode, MemAddr, MemSpace, Named, OpAtom,
    OpAtomsPopc, OpCCtl, OpErrBar, OpLd, OpLdGsts, OpLdc, OpLdgDepBar, OpMemBar, OpSt, Pred,
    PredSrc, UniformReg,
};

/// Refuses `what` unless `absent`: the chosen form has no bits for it.
fn absent(mnemonic: &'static str, absent: bool, what: &'static str) -> Result<(), EncodeError> {
    if absent {
        Ok(())
    } else {
        Err(EncodeError::NoBits { mnemonic, what })
    }
}

/// Writes `addr` in `layout`, which the caller picked by whether `addr` has a uniform register
/// (`set_reg_addr`, `set_ureg_addr`, `set_field(40..64, offset)`, `set_bit(91, ..)`).
fn addr(
    writer: &mut Writer,
    mnemonic: &'static str,
    layout: AddrLayout,
    addr: MemAddr,
) -> Result<(), EncodeError> {
    writer.gpr(MEM_ADDR, addr.reg);
    writer.signed(MEM_OFFSET, "address offset", i64::from(addr.offset))?;
    match (layout.uniform, addr.uniform) {
        (Some(field), Some(uniform)) => {
            writer.uniform_reg(field, uniform.reg);
            writer.flag(UNIFORM_BIT, uniform.enabled);
            if layout.e {
                writer.flag(MEM_E, uniform.wide);
            } else {
                absent(mnemonic, !uniform.wide, "a 64-bit uniform address")?;
            }
            match layout.reg_64 {
                Some(field) => writer.flag(field, addr.reg_64),
                None => absent(mnemonic, !addr.reg_64, "a 64-bit address register")?,
            }
        }
        (None, None) => {
            if layout.e {
                writer.flag(MEM_E, addr.reg_64);
            } else {
                absent(mnemonic, !addr.reg_64, "a 64-bit address register")?;
            }
        }
        (Some(_), None) => absent(mnemonic, false, "an address without a uniform register")?,
        (None, Some(_)) => absent(mnemonic, false, "a uniform address register")?,
    }
    Ok(())
}

/// Writes the space's own fields (`set_mem_order`, `set_eviction_priority`,
/// `set_field(84..87, 1_u8)`, `set_field(78..80, stride)`).
fn space(writer: &mut Writer, space: MemSpace) {
    match space {
        MemSpace::Global(access) => {
            writer.set(MEM_ORDER, access.order.bits());
            writer.set(MEM_EVICTION, access.eviction.bits());
        }
        MemSpace::Local(eviction) => writer.set(MEM_EVICTION, eviction.bits()),
        MemSpace::Shared(stride) => writer.set(MEM_STRIDE, stride.bits()),
    }
}

/// The opcode and address layout of a load or store in `space`, with or without a uniform
/// register.
fn ld_st_form(space: MemSpace, uniform: bool, ld: bool) -> Option<(u16, AddrLayout)> {
    Some(match (space, uniform, ld) {
        (MemSpace::Global(_), false, true) => (LDG_OPCODE, GLOBAL),
        (MemSpace::Global(_), true, true) => (LDG_UNIFORM_OPCODE, LDG_UNIFORM),
        (MemSpace::Local(_), true, true) => (LDL_OPCODE, LD_LOCAL_SHARED),
        (MemSpace::Shared(_), true, true) => (LDS_OPCODE, LD_LOCAL_SHARED),
        (MemSpace::Local(_) | MemSpace::Shared(_), false, true) => return None,
        (MemSpace::Global(_), false, false) => (STG_OPCODE, GLOBAL),
        (MemSpace::Global(_), true, false) => (STG_UNIFORM_OPCODE, ST_GLOBAL_UNIFORM),
        (MemSpace::Local(_), false, false) => (STL_OPCODE, AddrLayout::PLAIN),
        (MemSpace::Local(_), true, false) => (STL_UNIFORM_OPCODE, ST_UNIFORM_PLAIN),
        (MemSpace::Shared(_), false, false) => (STS_OPCODE, AddrLayout::PLAIN),
        (MemSpace::Shared(_), true, false) => (STS_UNIFORM_OPCODE, ST_UNIFORM_PLAIN),
    })
}

impl Encode for OpLd {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let mnemonic = Self::MNEMONIC;
        let uniform = self.addr.uniform.is_some();
        let (opcode, layout) =
            ld_st_form(self.space, uniform, true).ok_or(EncodeError::NoBits {
                mnemonic,
                what: "an address without a uniform register",
            })?;
        writer.opcode(opcode);
        writer.gpr(DST, self.dst);
        addr(writer, mnemonic, layout, self.addr)?;
        writer.set(MEM_TYPE, self.mem_type.bits());
        space(writer, self.space);
        let global = matches!(self.space, MemSpace::Global(_));
        if global {
            writer.flag(MEM_BIT_76, self.bit76);
            writer.set(PRED_DST, u64::from(self.pred_dst.index()));
        } else {
            absent(mnemonic, !self.bit76, "bit 76")?;
            absent(mnemonic, self.pred_dst.is_true(), "a predicate destination")?;
        }
        if global && uniform {
            // `set_rev_pred_src`: the field holds 7 minus the register.
            let field = Pred::PT.index().saturating_sub(self.pred.reg.index());
            writer.set(LDG_PRED, u64::from(field));
            writer.flag(LDG_PRED_NOT, self.pred.negated);
        } else {
            absent(mnemonic, self.pred == PredSrc::TRUE, "a load predicate")?;
        }
        if matches!(self.space, MemSpace::Shared(_)) {
            writer.flag(LDS_ZD, self.zd);
        } else {
            absent(mnemonic, !self.zd, ".zd")?;
        }
        Ok(())
    }
}

impl Encode for OpSt {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let mnemonic = Self::MNEMONIC;
        let uniform = self.addr.uniform.is_some();
        let (opcode, layout) =
            ld_st_form(self.space, uniform, false).ok_or(EncodeError::NoBits {
                mnemonic,
                what: "an address without a uniform register",
            })?;
        writer.opcode(opcode);
        writer.gpr(MEM_DATA, self.data);
        addr(writer, mnemonic, layout, self.addr)?;
        writer.set(MEM_TYPE, self.mem_type.bits());
        space(writer, self.space);
        if matches!(self.space, MemSpace::Global(_)) && uniform {
            writer.flag(MEM_BIT_76, self.bit76);
        } else {
            absent(mnemonic, !self.bit76, "bit 76")?;
        }
        Ok(())
    }
}

impl Encode for OpLdc {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.gpr(DST, self.dst);
        writer.set(CBUF_OFFSET, u64::from(self.cbuf.offset));
        writer.set(MEM_TYPE, self.mem_type.bits());
        match self.cbuf.buf {
            CBuf::Binding(index) => {
                writer.opcode(LDC_OPCODE);
                writer.checked(CBUF_INDEX, "constant buffer index", u64::from(index))?;
                writer.gpr(LDC_OFFSET, self.offset);
                writer.set(LDC_MODE, self.mode.bits());
            }
            CBuf::BindlessUniform(handle) => {
                writer.opcode(LDC_BINDLESS_OPCODE);
                writer.uniform_reg(LDC_HANDLE, handle);
                writer.flag(UNIFORM_BIT, true);
                writer.gpr(LDC_BINDLESS_OFFSET, self.offset);
                absent(
                    Self::MNEMONIC,
                    self.mode == LdcMode::Indexed,
                    "an indexing mode with a bindless buffer",
                )?;
            }
        }
        Ok(())
    }
}

impl Encode for OpAtom {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        let mnemonic = Self::MNEMONIC;
        let uniform = self.addr.uniform.is_some();
        let cmpr = match self.op {
            AtomOp::CmpExch { cmpr } => Some(cmpr),
            _ => None,
        };
        let (opcode, layout, op_field) = match (self.space, self.dst, cmpr, uniform) {
            (MemSpace::Local(_), ..) => return absent(mnemonic, false, "local memory"),
            (MemSpace::Global(_), AtomDst::Reduction { .. }, None, true) => {
                (RED_OPCODE, ST_GLOBAL_UNIFORM, Some(RED_OP))
            }
            (MemSpace::Global(_), AtomDst::Reduction { .. }, None, false) => {
                return absent(mnemonic, false, "an address without a uniform register");
            }
            (MemSpace::Global(_), AtomDst::Reg(_), None, false) => {
                (ATOMG_OPCODE, GLOBAL, Some(ATOM_OP))
            }
            (MemSpace::Global(_), AtomDst::Reg(_), None, true) => {
                (ATOMG_UNIFORM_OPCODE, ATOMG_UNIFORM, Some(ATOM_OP))
            }
            (MemSpace::Global(_), AtomDst::Reg(_), Some(_), _) => (ATOMG_CAS_OPCODE, GLOBAL, None),
            (MemSpace::Shared(_), AtomDst::Reg(_), None, false) => {
                (ATOMS_OPCODE, AddrLayout::PLAIN, Some(ATOM_OP))
            }
            (MemSpace::Shared(_), AtomDst::Reg(_), None, true) => {
                (ATOMS_UNIFORM_OPCODE, ST_UNIFORM_PLAIN, Some(ATOM_OP))
            }
            (MemSpace::Shared(_), AtomDst::Reg(_), Some(_), _) => {
                (ATOMS_CAS_OPCODE, AddrLayout::PLAIN, None)
            }
            (_, AtomDst::Reduction { .. }, ..) => {
                return absent(mnemonic, false, "an atomic without a destination");
            }
        };
        writer.opcode(opcode);
        addr(writer, mnemonic, layout, self.addr)?;
        writer.set(ATOM_TYPE, self.atom_type.bits());
        space(writer, self.space);
        match self.dst {
            AtomDst::Reg(dst) | AtomDst::Reduction { ignored: dst } => writer.gpr(DST, dst),
        }
        match cmpr {
            Some(cmpr) => {
                writer.gpr(CAS_CMPR, cmpr);
                writer.gpr(CAS_DATA, self.data);
            }
            None => writer.gpr(MEM_DATA, self.data),
        }
        if let (Some(field), Some(bits)) = (op_field, self.op.bits()) {
            writer.checked(field, "atomic operation", bits)?;
        }
        let pred_dst = matches!(
            (self.space, self.dst),
            (MemSpace::Global(_), AtomDst::Reg(_))
        );
        if pred_dst {
            writer.set(PRED_DST, u64::from(self.pred_dst.index()));
        } else {
            absent(mnemonic, self.pred_dst.is_true(), "a predicate destination")?;
        }
        Ok(())
    }
}

impl Encode for OpAtomsPopc {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(ATOMS_POPC_OPCODE);
        for field in ATOMS_POPC_FIXED_FIELDS {
            writer.set(field, u64::MAX);
        }
        writer.gpr(MEM_ADDR, self.addr);
        writer.set(MEM_STRIDE, self.stride.bits());
        Ok(())
    }
}

impl Encode for OpCCtl {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(CCTL_OPCODE);
        writer.gpr(MEM_ADDR, self.addr);
        writer.signed(CCTL_OFFSET, "address offset", i64::from(self.offset))?;
        writer.set(CCTL_OP, self.op.bits());
        Ok(())
    }
}

impl Encode for OpMemBar {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(MEMBAR_OPCODE);
        writer.flag(MEMBAR_MMIO, self.mmio);
        writer.set(MEMBAR_SCOPE, self.scope.bits());
        writer.flag(MEMBAR_SC, self.sc);
        Ok(())
    }
}

impl Encode for OpErrBar {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(ERRBAR_OPCODE);
        Ok(())
    }
}

impl Encode for OpLdgDepBar {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(LDGDEPBAR_OPCODE);
        Ok(())
    }
}

impl Encode for OpLdGsts {
    fn encode(&self, writer: &mut Writer) -> Result<(), EncodeError> {
        writer.opcode(LDGSTS_OPCODE);
        writer.gpr(LDGSTS_SHARED, self.shared);
        writer.set(
            LDGSTS_SHARED_OFFSET,
            u64::from(self.shared_offset.cast_unsigned()),
        );
        writer.gpr(LDGSTS_GLOBAL, self.global);
        writer.signed(
            LDGSTS_GLOBAL_OFFSET,
            "global offset",
            i64::from(self.global_offset),
        )?;
        writer.checked(
            LDGSTS_DESCRIPTOR,
            "memory descriptor",
            u64::from(self.descriptor),
        )?;
        writer.pred_src(COND, LDGSTS_ENABLE_NOT, self.enable);
        for field in LDGSTS_FIXED_FIELDS {
            writer.set(field, u64::MAX);
        }
        Ok(())
    }
}

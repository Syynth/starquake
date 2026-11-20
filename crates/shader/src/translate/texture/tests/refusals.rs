//! Every refusal with its exact error, each limit on both sides of its edge, the PR's order of
//! checks with two faults at once, and the exact text of every error.

use super::*;
use crate::decode::AtomCmpSrc;

use TextureUnsupported as U;

/// `op` translates, after checking it encodes.
fn accepted(op: Op) {
    let instr = encodable(Instr::new(op));
    let (function, _) = translate_raw(instr, DescriptorBuffer::new()).expect("translates");
    assert_eq!(function.verify(), Ok(()), "{instr}");
}

/// A buffer bound to `index`.
fn bound(index: u8) -> DescriptorBuffer {
    DescriptorBuffer::new().bind(index).expect("binds")
}

#[test]
fn tex_refusals() {
    assert_refused(
        Op::Tex(OpTex {
            tex: TexRef::Uniform(UniformReg::URZ),
            ..tex()
        }),
        U::UniformHandle(UniformReg::URZ),
    );
    accepted(Op::Tex(OpTex {
        tex: TexRef::Uniform(ur(UNIFORM_HANDLE_LIMIT - 1)),
        ..tex()
    }));
    assert_refused(Op::Tex(OpTex { ndv: true, ..tex() }), U::Derivative);
    for lod_mode in [TexLodMode::Clamp, TexLodMode::BiasClamp] {
        assert_refused(
            Op::Tex(OpTex {
                tex: TexRef::Bindless,
                lod_mode,
                ..tex()
            }),
            U::ClampMode,
        );
    }
    assert_refused(
        Op::Tex(OpTex {
            channel_mask: mask(0),
            ..tex()
        }),
        U::EmptyMask,
    );
    assert_refused(
        Op::Tex(OpTex {
            tex: cb(DESCRIPTOR_BUFFERS, 0),
            ..tex()
        }),
        U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
    );
    accepted(Op::Tex(OpTex {
        tex: cb(DESCRIPTOR_BUFFERS - 1, 0),
        ..tex()
    }));
    assert_refused(
        Op::Tex(OpTex {
            tex: TexRef::Bindless,
            scalar: false,
            ..tex()
        }),
        U::BindlessOperands,
    );
    assert_refused(
        Op::Tex(OpTex {
            tex: TexRef::Bindless,
            lod_mode: TexLodMode::Lod,
            z_cmpr: true,
            ..tex()
        }),
        U::BindlessOperands,
    );
    // `TEX.B.F16` encodes since the decoder reads bit 79 of `0x361`.
    assert_refused(
        Op::Tex(OpTex {
            tex: TexRef::Bindless,
            f16: true,
            ..tex()
        }),
        U::BindlessOperands,
    );
    assert_refused(
        Op::Tex(OpTex {
            scalar: false,
            lod_mode: TexLodMode::Lod,
            z_cmpr: true,
            ..tex()
        }),
        U::VectorOperandCount,
    );
    accepted(Op::Tex(OpTex {
        scalar: false,
        z_cmpr: true,
        ..tex()
    }));
}

/// Each condition of the PR's offset layout refuses on its own, and the case one step inside
/// it translates.
#[test]
fn tex_offset_layout_conditions() {
    let offset = OpTex {
        offset: true,
        ..tex()
    };
    let lod = OpTex {
        lod_mode: TexLodMode::Lod,
        ..offset
    };
    let refused = [
        OpTex {
            scalar: false,
            ..offset
        },
        OpTex {
            f16: true,
            ..offset
        },
        OpTex {
            tex: TexRef::Bindless,
            ..offset
        },
        OpTex {
            dim: TexDim::Array2D,
            ..offset
        },
        OpTex {
            z_cmpr: true,
            ..offset
        },
        OpTex {
            lod_mode: TexLodMode::Bias,
            ..offset
        },
        OpTex {
            lod_mode: TexLodMode::Clamp,
            ..offset
        },
        OpTex {
            srcs: [r(11), r(20)],
            ..offset
        },
        OpTex {
            srcs: [r(254), r(20)],
            ..offset
        },
        OpTex {
            srcs: [r(10), r(254)],
            ..offset
        },
        OpTex {
            srcs: [r(10), r(21)],
            ..lod
        },
    ];
    for op in refused {
        assert_refused(Op::Tex(op), U::Offset);
    }
    for op in [
        offset,
        lod,
        OpTex {
            lod_mode: TexLodMode::Zero,
            ..offset
        },
        OpTex {
            srcs: [r(252), r(253)],
            ..offset
        },
        OpTex {
            srcs: [r(252), r(252)],
            ..lod
        },
    ] {
        accepted(Op::Tex(op));
    }
}

/// Operand and handle registers at `RZ` are refused; one below translates.
#[test]
fn tex_register_edges() {
    assert_refused(
        Op::Tex(OpTex {
            srcs: [Gpr::RZ, r(20)],
            ..tex()
        }),
        U::OperandRegister(255),
    );
    accepted(Op::Tex(OpTex {
        srcs: [r(254), r(253)],
        ..tex()
    }));
    // Three operands: A and A + 1.
    assert_refused(
        Op::Tex(OpTex {
            lod_mode: TexLodMode::Lod,
            srcs: [r(254), r(20)],
            ..tex()
        }),
        U::OperandRegister(255),
    );
    accepted(Op::Tex(OpTex {
        lod_mode: TexLodMode::Lod,
        srcs: [r(253), r(254)],
        ..tex()
    }));
    // Four operands: B and B + 1.
    assert_refused(
        Op::Tex(OpTex {
            lod_mode: TexLodMode::Lod,
            z_cmpr: true,
            srcs: [r(10), r(254)],
            ..tex()
        }),
        U::OperandRegister(255),
    );
    // The vector layout: A onwards.
    assert_refused(
        Op::Tex(OpTex {
            scalar: false,
            dim: TexDim::D3,
            srcs: [r(253), r(20)],
            ..tex()
        }),
        U::OperandRegister(255),
    );
    accepted(Op::Tex(OpTex {
        scalar: false,
        dim: TexDim::D3,
        srcs: [r(252), r(20)],
        ..tex()
    }));
    let bindless = OpTex {
        tex: TexRef::Bindless,
        lod_mode: TexLodMode::Lod,
        ..tex()
    };
    assert_refused(
        Op::Tex(OpTex {
            srcs: [r(10), r(254)],
            ..bindless
        }),
        U::HandleRegister(255),
    );
    accepted(Op::Tex(OpTex {
        srcs: [r(10), r(253)],
        ..bindless
    }));
    assert_refused(
        Op::Tex(OpTex {
            tex: TexRef::Bindless,
            srcs: [r(10), Gpr::RZ],
            ..tex()
        }),
        U::HandleRegister(255),
    );
}

/// Result registers at `RZ` are refused; one below translates.
#[test]
fn tex_result_register_edges() {
    assert_refused(
        Op::Tex(OpTex {
            channel_mask: mask(0x3),
            dsts: [r(254), Gpr::RZ],
            ..tex()
        }),
        U::ResultRegister(255),
    );
    accepted(Op::Tex(OpTex {
        channel_mask: mask(0x3),
        dsts: [r(253), Gpr::RZ],
        ..tex()
    }));
    assert_refused(
        Op::Tex(OpTex {
            channel_mask: mask(0x1),
            dsts: [Gpr::RZ, Gpr::RZ],
            ..tex()
        }),
        U::ResultRegister(255),
    );
    assert_refused(
        Op::Tex(OpTex {
            dsts: [r(0), r(254)],
            ..tex()
        }),
        U::ResultRegister(255),
    );
    accepted(Op::Tex(OpTex {
        dsts: [r(0), r(253)],
        ..tex()
    }));
    assert_refused(
        Op::Tex(OpTex {
            f16: true,
            dsts: [Gpr::RZ, r(2)],
            ..tex()
        }),
        U::ResultRegister(255),
    );
    accepted(Op::Tex(OpTex {
        f16: true,
        dsts: [r(254), r(2)],
        ..tex()
    }));
}

/// The second destination must be `RZ` for one or two channels and a register for more.
#[test]
fn result_layouts() {
    for bits in 1..16_u8 {
        let more = bits.count_ones() > 2;
        let (good, bad) = if more {
            (r(8), Gpr::RZ)
        } else {
            (Gpr::RZ, r(8))
        };
        accepted(Op::Tex(OpTex {
            channel_mask: mask(bits),
            dsts: [r(0), good],
            ..tex()
        }));
        assert_refused(
            Op::Tex(OpTex {
                channel_mask: mask(bits),
                dsts: [r(0), bad],
                ..tex()
            }),
            U::ResultLayout,
        );
    }
    for op in [
        Op::Tld(OpTld {
            dsts: [r(0), Gpr::RZ],
            ..tld()
        }),
        Op::Tld4(OpTld4 {
            dsts: [r(0), Gpr::RZ],
            ..tld4()
        }),
        Op::Txd(OpTxd {
            dsts: [r(0), Gpr::RZ],
            ..txd()
        }),
        Op::Tex(OpTex {
            f16: true,
            channel_mask: mask(0x3),
            dsts: [r(0), r(2)],
            ..tex()
        }),
    ] {
        assert_refused(op, U::ResultLayout);
    }
}

#[test]
fn tld_refusals() {
    assert_refused(
        Op::Tld(OpTld {
            tex: TexRef::Bindless,
            ..tld()
        }),
        U::HandleForm {
            mnemonic: "tld",
            kind: HandleKind::Registers,
        },
    );
    assert_refused(
        Op::Tld(OpTld {
            tex: TexRef::Uniform(UniformReg::URZ),
            ..tld()
        }),
        U::UniformHandle(UniformReg::URZ),
    );
    assert_refused(
        Op::Tld(OpTld {
            scalar: false,
            ..tld()
        }),
        U::FetchNotScalar,
    );
    assert_refused(
        Op::Tld(OpTld {
            offset: true,
            ..tld()
        }),
        U::FetchOffset,
    );
    assert_refused(
        Op::Tld(OpTld {
            is_ms: true,
            ..tld()
        }),
        U::FetchMultisampleOrClamp,
    );
    assert_refused(
        Op::Tld(OpTld {
            tex: TexRef::Uniform(ur(1)),
            clamp: true,
            ..tld()
        }),
        U::FetchMultisampleOrClamp,
    );
    for lod_mode in [
        TexLodMode::Auto,
        TexLodMode::Bias,
        TexLodMode::Clamp,
        TexLodMode::BiasClamp,
    ] {
        assert_refused(Op::Tld(OpTld { lod_mode, ..tld() }), U::FetchLodMode);
    }
    assert_refused(
        Op::Tld(OpTld {
            channel_mask: mask(0),
            ..tld()
        }),
        U::EmptyMask,
    );
    assert_refused(
        Op::Tld(OpTld {
            tex: cb(DESCRIPTOR_BUFFERS, 0),
            ..tld()
        }),
        U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
    );
    assert_refused(
        Op::Tld(OpTld {
            srcs: [r(10), Gpr::RZ],
            ..tld()
        }),
        U::OperandRegister(255),
    );
    for dim in [TexDim::Cube, TexDim::ArrayCube] {
        assert_refused(Op::Tld(OpTld { dim, ..tld() }), U::FetchDimension(dim));
    }
    assert_refused(
        Op::Tld(OpTld {
            dsts: [r(254), r(2)],
            channel_mask: mask(0x3),
            ..tld()
        }),
        U::ResultLayout,
    );
    assert_refused(
        Op::Tld(OpTld {
            dsts: [r(254), Gpr::RZ],
            channel_mask: mask(0x3),
            ..tld()
        }),
        U::ResultRegister(255),
    );
}

#[test]
fn tld4_refusals() {
    assert_refused(
        Op::Tld4(OpTld4 {
            tex: TexRef::Bindless,
            ..tld4()
        }),
        U::HandleForm {
            mnemonic: "tld4",
            kind: HandleKind::Registers,
        },
    );
    assert_refused(
        Op::Tld4(OpTld4 {
            tex: TexRef::Uniform(UniformReg::URZ),
            ..tld4()
        }),
        U::UniformHandle(UniformReg::URZ),
    );
    assert_refused(
        Op::Tld4(OpTld4 {
            tex: cb(DESCRIPTOR_BUFFERS, 0),
            ..tld4()
        }),
        U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
    );
    assert_refused(
        Op::Tld4(OpTld4 {
            scalar: false,
            ..tld4()
        }),
        U::GatherNotScalar,
    );
    assert_refused(
        Op::Tld4(OpTld4 {
            channel_mask: mask(0),
            ..tld4()
        }),
        U::EmptyMask,
    );
    for dim in [
        TexDim::D1,
        TexDim::Array1D,
        TexDim::D3,
        TexDim::Cube,
        TexDim::ArrayCube,
    ] {
        for offset_mode in [TexOffsetMode::AddOffI, TexOffsetMode::PerPx] {
            assert_refused(
                Op::Tld4(OpTld4 {
                    dim,
                    offset_mode,
                    ..tld4()
                }),
                U::GatherOffsetDimension(dim),
            );
        }
    }
    for dim in [TexDim::D1, TexDim::Array1D, TexDim::D3, TexDim::ArrayCube] {
        assert_refused(Op::Tld4(OpTld4 { dim, ..tld4() }), U::GatherDimension(dim));
    }
    assert_refused(
        Op::Tld4(OpTld4 {
            offset_mode: TexOffsetMode::PerPx,
            srcs: [r(10), r(254)],
            ..tld4()
        }),
        U::OperandRegister(255),
    );
    accepted(Op::Tld4(OpTld4 {
        offset_mode: TexOffsetMode::PerPx,
        srcs: [r(10), r(253)],
        ..tld4()
    }));
}

#[test]
fn txd_refusals() {
    assert_refused(
        Op::Txd(OpTxd {
            tex: TexRef::Bindless,
            ..txd()
        }),
        U::HandleForm {
            mnemonic: "txd",
            kind: HandleKind::Registers,
        },
    );
    assert_refused_unencodable(
        Op::Txd(OpTxd {
            tex: TexRef::Uniform(ur(1)),
            ..txd()
        }),
        U::HandleForm {
            mnemonic: "txd",
            kind: HandleKind::Uniform,
        },
    );
    assert_refused(
        Op::Txd(OpTxd {
            tex: cb(DESCRIPTOR_BUFFERS, 0),
            ..txd()
        }),
        U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
    );
    assert_refused(
        Op::Txd(OpTxd {
            offset: true,
            ..txd()
        }),
        U::GradientOffset,
    );
    assert_refused(
        Op::Txd(OpTxd {
            channel_mask: mask(0),
            ..txd()
        }),
        U::EmptyMask,
    );
    for dim in [
        TexDim::D1,
        TexDim::Array1D,
        TexDim::D3,
        TexDim::Cube,
        TexDim::ArrayCube,
    ] {
        assert_refused(Op::Txd(OpTxd { dim, ..txd() }), U::GradientDimension(dim));
    }
    for (dim, refused, fine) in [(TexDim::D2, 254, 253), (TexDim::Array2D, 253, 252)] {
        assert_refused(
            Op::Txd(OpTxd {
                dim,
                srcs: [r(refused), r(20)],
                ..txd()
            }),
            U::GradientRegisters,
        );
        accepted(Op::Txd(OpTxd {
            dim,
            srcs: [r(fine), r(20)],
            ..txd()
        }));
    }
    assert_refused(
        Op::Txd(OpTxd {
            srcs: [r(10), r(252)],
            ..txd()
        }),
        U::GradientRegisters,
    );
    accepted(Op::Txd(OpTxd {
        srcs: [r(10), r(251)],
        ..txd()
    }));
}

#[test]
fn tmml_refusals() {
    assert_refused(
        Op::Tmml(OpTmml {
            tex: TexRef::Bindless,
            ..tmml()
        }),
        U::HandleForm {
            mnemonic: "tmml",
            kind: HandleKind::Registers,
        },
    );
    assert_refused_unencodable(
        Op::Tmml(OpTmml {
            tex: TexRef::Uniform(ur(1)),
            ..tmml()
        }),
        U::HandleForm {
            mnemonic: "tmml",
            kind: HandleKind::Uniform,
        },
    );
    assert_refused(
        Op::Tmml(OpTmml {
            srcs: [r(10), r(11)],
            ..tmml()
        }),
        U::MipQueryExtraRegister,
    );
    assert_refused(
        Op::Tmml(OpTmml {
            tex: cb(DESCRIPTOR_BUFFERS, 0),
            ..tmml()
        }),
        U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
    );
    assert_refused(
        Op::Tmml(OpTmml {
            dsts: [r(4), r(5)],
            ..tmml()
        }),
        U::MipQueryExtraRegister,
    );
    for bits in [0x0, 0x4, 0x5, 0x8, 0xa, 0xc, 0xf] {
        assert_refused(
            Op::Tmml(OpTmml {
                channel_mask: mask(bits),
                ..tmml()
            }),
            U::MipQueryChannels(mask(bits)),
        );
    }
    assert_refused(
        Op::Tmml(OpTmml {
            srcs: [r(253), Gpr::RZ],
            ..tmml()
        }),
        U::MipQueryRegisters,
    );
    accepted(Op::Tmml(OpTmml {
        srcs: [r(252), Gpr::RZ],
        ..tmml()
    }));
    assert_refused(
        Op::Tmml(OpTmml {
            dsts: [r(254), Gpr::RZ],
            channel_mask: mask(0x3),
            ..tmml()
        }),
        U::MipQueryResultRegister(255),
    );
    accepted(Op::Tmml(OpTmml {
        dsts: [r(253), Gpr::RZ],
        channel_mask: mask(0x3),
        ..tmml()
    }));
}

#[test]
fn txq_refusals() {
    assert_refused(
        Op::Txq(OpTxq {
            tex: TexRef::Bindless,
            ..txq()
        }),
        U::HandleForm {
            mnemonic: "txq",
            kind: HandleKind::Registers,
        },
    );
    assert_refused_unencodable(
        Op::Txq(OpTxq {
            tex: TexRef::Uniform(ur(1)),
            ..txq()
        }),
        U::HandleForm {
            mnemonic: "txq",
            kind: HandleKind::Uniform,
        },
    );
    assert_refused(
        Op::Txq(OpTxq {
            tex: cb(DESCRIPTOR_BUFFERS, 0),
            ..txq()
        }),
        U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
    );
    for query in [TexQuery::TextureType, TexQuery::SamplerPos] {
        assert_refused(Op::Txq(OpTxq { query, ..txq() }), U::QueryKind(query));
    }
    assert_refused(
        Op::Txq(OpTxq {
            dsts: [r(4), r(8)],
            ..txq()
        }),
        U::QueryExtraRegister,
    );
    assert_refused(
        Op::Txq(OpTxq {
            channel_mask: mask(0),
            ..txq()
        }),
        U::EmptyQueryMask,
    );
    assert_refused(
        Op::Txq(OpTxq {
            dsts: [r(252), Gpr::RZ],
            ..txq()
        }),
        U::QueryResultRegister(255),
    );
    accepted(Op::Txq(OpTxq {
        dsts: [r(251), Gpr::RZ],
        ..txq()
    }));
}

#[test]
fn surface_refusals() {
    assert_refused(
        Op::SuLd(OpSuLd {
            access: ImageAccess::Formatted(mask(0)),
            ..suld()
        }),
        U::EmptySurfaceMask,
    );
    assert_refused(
        Op::SuLd(OpSuLd {
            coord: r(253),
            ..suld()
        }),
        U::SurfaceRegisters,
    );
    accepted(Op::SuLd(OpSuLd {
        coord: r(252),
        ..suld()
    }));
    assert_refused(
        Op::SuLd(OpSuLd {
            dst: r(252),
            ..suld()
        }),
        U::SurfaceResultRegister(255),
    );
    accepted(Op::SuLd(OpSuLd {
        dst: r(251),
        ..suld()
    }));
    assert_refused(
        Op::SuLd(OpSuLd {
            access: ImageAccess::Binary(MemType::B128),
            dst: r(252),
            ..suld()
        }),
        U::SurfaceResultRegister(255),
    );
    accepted(Op::SuLd(OpSuLd {
        access: ImageAccess::Binary(MemType::B64),
        dst: r(253),
        ..suld()
    }));
}

#[test]
fn surface_store_and_atomic_refusals() {
    for bits in (0..16_u8).filter(|bits| !SURFACE_STORE_MASKS.contains(bits)) {
        assert_refused(
            Op::SuSt(OpSuSt {
                access: ImageAccess::Formatted(mask(bits)),
                ..sust()
            }),
            U::SurfaceChannelMask(mask(bits)),
        );
    }
    // `data_reg + count < 255`, as the PR checks it.
    assert_refused(
        Op::SuSt(OpSuSt {
            data: r(251),
            ..sust()
        }),
        U::SurfaceRegisters,
    );
    accepted(Op::SuSt(OpSuSt {
        data: r(250),
        ..sust()
    }));
    assert_refused(
        Op::SuSt(OpSuSt {
            access: ImageAccess::Binary(MemType::B32),
            data: r(254),
            ..sust()
        }),
        U::SurfaceRegisters,
    );
    accepted(Op::SuSt(OpSuSt {
        access: ImageAccess::Binary(MemType::B32),
        data: r(253),
        ..sust()
    }));
    assert_refused(
        Op::SuSt(OpSuSt {
            coord: r(253),
            ..sust()
        }),
        U::SurfaceRegisters,
    );
    assert_refused(
        Op::SuAtom(OpSuAtom {
            ba: true,
            ..suatom()
        }),
        U::SurfaceAtomicMode,
    );
    assert_refused(
        Op::SuAtom(OpSuAtom {
            atom_op: AtomOp::CmpExch(AtomCmpSrc::Packed),
            ..suatom()
        }),
        U::SurfaceCompareAndSwap,
    );
    for (atom_type, name) in [
        (AtomType::U64, "u64"),
        (AtomType::F32, "f32"),
        (AtomType::F16v2, "f16x2"),
        (AtomType::I64, "s64"),
        (AtomType::F64, "f64"),
    ] {
        assert_refused(
            Op::SuAtom(OpSuAtom {
                atom_type,
                atom_op: AtomOp::Min,
                ..suatom()
            }),
            U::SurfaceAtomicOperation {
                op: "min",
                ty: name,
            },
        );
    }
    assert_refused(
        Op::SuAtom(OpSuAtom {
            coord: r(253),
            ..suatom()
        }),
        U::SurfaceRegisters,
    );
    accepted(Op::SuAtom(OpSuAtom {
        coord: r(252),
        ..suatom()
    }));
}

#[test]
fn other_groups_are_refused() {
    let exit = Op::Exit(crate::decode::OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    });
    assert!(!in_group(&exit));
    assert_refused(exit, U::OtherGroup("exit"));
    for op in [
        Op::Tex(tex()),
        Op::Tld(tld()),
        Op::Tld4(tld4()),
        Op::Tmml(tmml()),
        Op::Txd(txd()),
        Op::Txq(txq()),
        Op::SuLd(suld()),
        Op::SuSt(sust()),
        Op::SuAtom(suatom()),
    ] {
        assert!(in_group(&op), "{op}");
    }
}

/// A handle in another buffer than the program's is refused; the same buffer translates.
#[test]
fn descriptor_buffer_conflicts() {
    for op in [
        Op::Tex(tex()),
        Op::Tld(tld()),
        Op::Tld4(tld4()),
        Op::Txd(txd()),
        Op::Tmml(tmml()),
        Op::Txq(txq()),
    ] {
        let instr = encodable(Instr::new(op));
        assert_refused_raw(
            instr,
            bound(3),
            U::DescriptorBufferConflict { bound: 3, index: 2 },
        );
        let (_, after) = translate_raw(instr, bound(2)).expect("translates");
        assert_eq!(after, bound(2), "{instr}");
        let (_, fresh) = translate_raw(instr, DescriptorBuffer::new()).expect("translates");
        assert_eq!(fresh, bound(2), "{instr}");
    }
    // The bound is checked before the conflict.
    assert_refused_raw(
        encodable(Instr::new(Op::Tex(OpTex {
            tex: cb(DESCRIPTOR_BUFFERS, 0),
            ..tex()
        }))),
        bound(3),
        U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
    );
}

/// A refused instruction leaves the descriptor buffer unset even when its handle's buffer was
/// checked and accepted before the refusal (the module's divergence table).
#[test]
fn a_refusal_leaves_the_descriptor_buffer() {
    let op = OpTex {
        tex: cb(5, 0x10),
        dsts: [r(0), Gpr::RZ],
        ..tex()
    };
    assert_refused(Op::Tex(op), U::ResultLayout);
    let op = OpTld4 {
        tex: cb(5, 0x10),
        scalar: false,
        ..tld4()
    };
    assert_refused(Op::Tld4(op), U::GatherNotScalar);
}

/// Asserts each case's refusal: two faults at once give the first in the PR's order.
fn assert_first_faults(cases: Vec<(Op, U)>) {
    for (op, what) in cases {
        assert_refused(op, what);
    }
}

/// `TEX`'s checks come in the PR's order: two faults at once give the first.
#[test]
fn tex_checks_come_in_the_prs_order() {
    assert_first_faults(vec![
        (
            Op::Tex(OpTex {
                tex: TexRef::Uniform(UniformReg::URZ),
                ndv: true,
                ..tex()
            }),
            U::UniformHandle(UniformReg::URZ),
        ),
        (
            Op::Tex(OpTex {
                tex: TexRef::Bindless,
                ndv: true,
                lod_mode: TexLodMode::Clamp,
                ..tex()
            }),
            U::Derivative,
        ),
        (
            Op::Tex(OpTex {
                tex: TexRef::Bindless,
                lod_mode: TexLodMode::Clamp,
                channel_mask: mask(0),
                ..tex()
            }),
            U::ClampMode,
        ),
        (
            Op::Tex(OpTex {
                channel_mask: mask(0),
                tex: cb(DESCRIPTOR_BUFFERS, 0),
                ..tex()
            }),
            U::EmptyMask,
        ),
        (
            Op::Tex(OpTex {
                tex: cb(DESCRIPTOR_BUFFERS, 0),
                offset: true,
                scalar: false,
                ..tex()
            }),
            U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
        ),
        (
            Op::Tex(OpTex {
                offset: true,
                scalar: false,
                dsts: [r(0), Gpr::RZ],
                ..tex()
            }),
            U::Offset,
        ),
        (
            Op::Tex(OpTex {
                srcs: [Gpr::RZ, r(20)],
                dsts: [r(0), Gpr::RZ],
                ..tex()
            }),
            U::OperandRegister(255),
        ),
        (
            Op::Tex(OpTex {
                dsts: [r(254), Gpr::RZ],
                ..tex()
            }),
            U::ResultLayout,
        ),
    ]);
}

/// `TLD`'s checks come in the PR's order: two faults at once give the first.
#[test]
fn tld_checks_come_in_the_prs_order() {
    assert_first_faults(vec![
        (
            Op::Tld(OpTld {
                tex: TexRef::Bindless,
                scalar: false,
                ..tld()
            }),
            U::HandleForm {
                mnemonic: "tld",
                kind: HandleKind::Registers,
            },
        ),
        (
            Op::Tld(OpTld {
                scalar: false,
                offset: true,
                ..tld()
            }),
            U::FetchNotScalar,
        ),
        (
            Op::Tld(OpTld {
                offset: true,
                is_ms: true,
                ..tld()
            }),
            U::FetchOffset,
        ),
        (
            Op::Tld(OpTld {
                is_ms: true,
                lod_mode: TexLodMode::Auto,
                ..tld()
            }),
            U::FetchMultisampleOrClamp,
        ),
        (
            Op::Tld(OpTld {
                lod_mode: TexLodMode::Auto,
                channel_mask: mask(0),
                ..tld()
            }),
            U::FetchLodMode,
        ),
        (
            Op::Tld(OpTld {
                channel_mask: mask(0),
                tex: cb(DESCRIPTOR_BUFFERS, 0),
                ..tld()
            }),
            U::EmptyMask,
        ),
        (
            Op::Tld(OpTld {
                tex: cb(DESCRIPTOR_BUFFERS, 0),
                srcs: [Gpr::RZ, r(20)],
                ..tld()
            }),
            U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
        ),
        (
            Op::Tld(OpTld {
                srcs: [Gpr::RZ, r(20)],
                dim: TexDim::Cube,
                ..tld()
            }),
            U::OperandRegister(255),
        ),
        (
            Op::Tld(OpTld {
                dim: TexDim::Cube,
                dsts: [r(0), Gpr::RZ],
                ..tld()
            }),
            U::FetchDimension(TexDim::Cube),
        ),
    ]);
}

/// `TLD4`'s checks come in the PR's order: two faults at once give the first.
#[test]
fn tld4_checks_come_in_the_prs_order() {
    assert_first_faults(vec![
        (
            Op::Tld4(OpTld4 {
                tex: cb(DESCRIPTOR_BUFFERS, 0),
                scalar: false,
                ..tld4()
            }),
            U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
        ),
        (
            Op::Tld4(OpTld4 {
                scalar: false,
                channel_mask: mask(0),
                ..tld4()
            }),
            U::GatherNotScalar,
        ),
        (
            Op::Tld4(OpTld4 {
                channel_mask: mask(0),
                dim: TexDim::Cube,
                offset_mode: TexOffsetMode::AddOffI,
                ..tld4()
            }),
            U::EmptyMask,
        ),
        (
            Op::Tld4(OpTld4 {
                dim: TexDim::D3,
                offset_mode: TexOffsetMode::AddOffI,
                srcs: [Gpr::RZ, r(20)],
                ..tld4()
            }),
            U::GatherOffsetDimension(TexDim::D3),
        ),
        (
            Op::Tld4(OpTld4 {
                dim: TexDim::D3,
                srcs: [Gpr::RZ, r(20)],
                ..tld4()
            }),
            U::OperandRegister(255),
        ),
        (
            Op::Tld4(OpTld4 {
                dim: TexDim::D3,
                dsts: [r(0), Gpr::RZ],
                ..tld4()
            }),
            U::GatherDimension(TexDim::D3),
        ),
    ]);
}

/// `TXD`'s checks come in the PR's order: two faults at once give the first.
#[test]
fn txd_checks_come_in_the_prs_order() {
    assert_first_faults(vec![
        (
            Op::Txd(OpTxd {
                tex: cb(DESCRIPTOR_BUFFERS, 0),
                offset: true,
                ..txd()
            }),
            U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
        ),
        (
            Op::Txd(OpTxd {
                offset: true,
                channel_mask: mask(0),
                ..txd()
            }),
            U::GradientOffset,
        ),
        (
            Op::Txd(OpTxd {
                channel_mask: mask(0),
                dim: TexDim::D3,
                ..txd()
            }),
            U::EmptyMask,
        ),
        (
            Op::Txd(OpTxd {
                dim: TexDim::D3,
                srcs: [r(254), r(20)],
                ..txd()
            }),
            U::GradientDimension(TexDim::D3),
        ),
        (
            Op::Txd(OpTxd {
                srcs: [r(254), r(20)],
                dsts: [r(0), Gpr::RZ],
                ..txd()
            }),
            U::GradientRegisters,
        ),
    ]);
}

/// `TMML`'s checks come in the PR's order: two faults at once give the first.
#[test]
fn tmml_checks_come_in_the_prs_order() {
    assert_first_faults(vec![
        (
            Op::Tmml(OpTmml {
                srcs: [r(10), r(11)],
                tex: cb(DESCRIPTOR_BUFFERS, 0),
                ..tmml()
            }),
            U::MipQueryExtraRegister,
        ),
        (
            Op::Tmml(OpTmml {
                tex: cb(DESCRIPTOR_BUFFERS, 0),
                dsts: [r(4), r(5)],
                ..tmml()
            }),
            U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
        ),
        (
            Op::Tmml(OpTmml {
                dsts: [r(4), r(5)],
                channel_mask: mask(0),
                ..tmml()
            }),
            U::MipQueryExtraRegister,
        ),
        (
            Op::Tmml(OpTmml {
                channel_mask: mask(0),
                srcs: [r(253), Gpr::RZ],
                ..tmml()
            }),
            U::MipQueryChannels(mask(0)),
        ),
        (
            Op::Tmml(OpTmml {
                srcs: [r(253), Gpr::RZ],
                dsts: [r(254), Gpr::RZ],
                channel_mask: mask(0x3),
                ..tmml()
            }),
            U::MipQueryRegisters,
        ),
    ]);
}

/// `TXQ`'s checks come in the PR's order: two faults at once give the first.
#[test]
fn txq_checks_come_in_the_prs_order() {
    assert_first_faults(vec![
        (
            Op::Txq(OpTxq {
                tex: cb(DESCRIPTOR_BUFFERS, 0),
                query: TexQuery::SamplerPos,
                ..txq()
            }),
            U::DescriptorBuffer(DESCRIPTOR_BUFFERS),
        ),
        (
            Op::Txq(OpTxq {
                query: TexQuery::SamplerPos,
                dsts: [r(4), r(8)],
                ..txq()
            }),
            U::QueryKind(TexQuery::SamplerPos),
        ),
        (
            Op::Txq(OpTxq {
                dsts: [r(4), r(8)],
                channel_mask: mask(0),
                ..txq()
            }),
            U::QueryExtraRegister,
        ),
    ]);
}

/// The surface instructions' checks come in the PR's order: two faults at once give the first.
#[test]
fn surface_checks_come_in_the_prs_order() {
    assert_first_faults(vec![
        (
            Op::SuLd(OpSuLd {
                access: ImageAccess::Formatted(mask(0)),
                coord: r(253),
                ..suld()
            }),
            U::EmptySurfaceMask,
        ),
        (
            Op::SuLd(OpSuLd {
                coord: r(253),
                dst: r(253),
                ..suld()
            }),
            U::SurfaceRegisters,
        ),
        (
            Op::SuSt(OpSuSt {
                access: ImageAccess::Formatted(mask(0x2)),
                data: r(254),
                ..sust()
            }),
            U::SurfaceChannelMask(mask(0x2)),
        ),
        (
            Op::SuAtom(OpSuAtom {
                ba: true,
                atom_op: AtomOp::CmpExch(AtomCmpSrc::Packed),
                ..suatom()
            }),
            U::SurfaceAtomicMode,
        ),
        (
            Op::SuAtom(OpSuAtom {
                atom_op: AtomOp::CmpExch(AtomCmpSrc::Packed),
                atom_type: AtomType::F32,
                ..suatom()
            }),
            U::SurfaceCompareAndSwap,
        ),
        (
            Op::SuAtom(OpSuAtom {
                atom_type: AtomType::F32,
                coord: r(253),
                ..suatom()
            }),
            U::SurfaceAtomicOperation {
                op: "add",
                ty: "f32",
            },
        ),
    ]);
}

/// Asserts each error's exact text.
fn assert_texts(cases: &[(U, &str)]) {
    for &(what, text) in cases {
        assert_eq!(what.to_string(), text);
    }
}

/// The exact text of every error of the texture instructions, in plain words.
#[test]
fn texture_error_text() {
    assert_texts(&[
        (
            U::OtherGroup("exit"),
            "exit, which is not a texture or surface instruction,",
        ),
        (
            U::HandleForm {
                mnemonic: "tld",
                kind: HandleKind::Registers,
            },
            "tld with its handle in the source registers",
        ),
        (
            U::HandleForm {
                mnemonic: "txq",
                kind: HandleKind::Uniform,
            },
            "txq with its handle in a uniform register",
        ),
        (U::UniformHandle(UniformReg::URZ), "a texture handle in urz"),
        (
            U::Derivative,
            "a texture sample with non-divergent derivatives",
        ),
        (
            U::ClampMode,
            "a texture sample with its handle in a register and a level-of-detail clamp",
        ),
        (U::EmptyMask, "an empty texture channel mask"),
        (
            U::DescriptorBuffer(18),
            "a texture handle in constant buffer 18",
        ),
        (
            U::DescriptorBufferConflict { bound: 2, index: 3 },
            "a texture handle in constant buffer 3 after handles in buffer 2",
        ),
        (U::Offset, "a texture offset in this operand layout"),
        (
            U::BindlessOperands,
            "a texture sample with its handle in a register in this operand layout",
        ),
        (
            U::VectorOperandCount,
            "a vector texture sample with more than one operand after its coordinates",
        ),
        (U::OperandRegister(255), "texture operand register 255"),
        (U::HandleRegister(255), "texture handle register 255"),
        (
            U::ResultLayout,
            "a texture result layout whose second destination does not match its channel count",
        ),
        (U::ResultRegister(256), "texture result register 256"),
        (
            U::FetchNotScalar,
            "a texel fetch in the vector operand layout",
        ),
        (U::FetchOffset, "a texel fetch with an offset"),
        (
            U::FetchMultisampleOrClamp,
            "a multisampled or clamped texel fetch",
        ),
        (
            U::FetchLodMode,
            "a texel fetch with a level-of-detail mode other than lz and ll",
        ),
        (
            U::FetchDimension(TexDim::Cube),
            "a texel fetch of dimension cube",
        ),
    ]);
}

/// The exact text of every error of the gather, gradient, query and surface instructions.
#[test]
fn other_error_text() {
    assert_texts(&[
        (U::GatherNotScalar, "a gather in the vector operand layout"),
        (
            U::GatherOffsetDimension(TexDim::D3),
            "gather offsets with dimension 3d",
        ),
        (U::GatherDimension(TexDim::D1), "a gather of dimension 1d"),
        (U::GradientOffset, "a gradient sample with an offset"),
        (
            U::GradientDimension(TexDim::ArrayCube),
            "a gradient sample of dimension array_cube",
        ),
        (U::GradientRegisters, "gradient operand registers past r254"),
        (
            U::MipQueryExtraRegister,
            "a level-of-detail query with a second source or destination",
        ),
        (
            U::MipQueryChannels(mask(0xc)),
            "a level-of-detail query of channels 0xc",
        ),
        (
            U::MipQueryRegisters,
            "level-of-detail query coordinate registers past r254",
        ),
        (
            U::MipQueryResultRegister(255),
            "level-of-detail query result register 255",
        ),
        (
            U::QueryKind(TexQuery::SamplerPos),
            "a texture query of tex_header_sampler_pos",
        ),
        (
            U::QueryExtraRegister,
            "a texture query with a second destination",
        ),
        (U::EmptyQueryMask, "an empty texture query mask"),
        (
            U::QueryResultRegister(255),
            "texture query result register 255",
        ),
        (U::EmptySurfaceMask, "an empty surface channel mask"),
        (
            U::SurfaceChannelMask(mask(0x5)),
            "a surface store of channels 0x5",
        ),
        (U::SurfaceRegisters, "surface operand registers past r254"),
        (U::SurfaceResultRegister(255), "surface result register 255"),
        (U::SurfaceAtomicMode, "a surface atomic with .ba"),
        (U::SurfaceCompareAndSwap, "a surface compare-and-swap"),
        (
            U::SurfaceAtomicOperation {
                op: "min",
                ty: "u64",
            },
            "a surface atomic min of type u64",
        ),
    ]);
    assert_eq!(
        TextureError::Unsupported(U::DescriptorBufferConflict { bound: 2, index: 3 }).to_string(),
        "a texture handle in constant buffer 3 after handles in buffer 2 is not supported"
    );
}

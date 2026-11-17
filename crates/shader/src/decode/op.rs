//! Decoded instructions: the [`Op`] enum, with one variant per instruction, and [`Instr`].
//!
//! The variants follow NAK's `Op*` structs (`mesa-nouveau/src/nouveau/compiler/nak/ir.rs`
//! @ 52238762, MIT, Copyright 2022 Collabora, Ltd.; spec, "The SM86 decoder"), one per
//! instruction of the decoder slices 6a to 6i, grouped in one module per slice so the slices can
//! be built side by side without editing a shared file:
//!
//! | Module | Slice | Instructions |
//! |---|---|---|
//! | `integer` | 6a | `IADD3`, `IADD3.X`, `IMAD`, `IMAD.WIDE`, `LOP3`, `SHF`, `LEA`, `LEA.X`, `ISETP`, `IMNMX`, `PRMT`, `POPC`, `FLO`, `BREV`, `SGXT`, `BMSK`, `IABS`, `IDP4`, `MOV`, `SEL`, `PLOP3`, `P2R`, `R2P` |
//! | `float` | 6b | `FADD`, `FMUL`, `FFMA`, `FMNMX`, `FSET`, `FSETP`, `FSEL`, `MUFU`, `FSWZADD`, `FRND`, `F2F`, `F2I`, `I2F`, `I2FP`, `F2FP` |
//! | `half_double` | 6c | `HADD2`, `HFMA2`, `HMUL2`, `HSET2`, `HSETP2`, `HMNMX2`, `DADD`, `DMUL`, `DFMA`, `DSETP` |
//! | `uniform` | 6d | `R2UR`, `UP2UR` |
//! | `memory` | 6e | `LD`/`ST` (every space), `LDC`, `ATOM` (with `ATOMS`, `ATOMG`, `RED`), `CCTL`, `MEMBAR`, `LDGSTS`, `LDGDEPBAR` |
//! | `texture` | 6f | `TEX`, `TLD`, `TLD4`, `TMML`, `TXD`, `TXQ`, `SULD`, `SUST`, `SUATOM` (with `SURED`) |
//! | `attribute` | 6g | `ALD`, `AST`, `IPA`, `AL2P`, `S2R`, `CS2R`, `PIXLD`, `OUT`, `OUT.FINAL`, `ISBERD`, `ISBEWR`, `LDTRAM` |
//! | `control` | 6h | `BRA`, `BRX`, `JMX`, `CALL`, `RET`, `EXIT`, `KILL`, `BSSY`, `BSYNC`, `BREAK`, `BMOV`, `WARPSYNC`, `BAR`, `VOTE`, `SHFL`, `MATCH`, `REDUX`, `NOP`, `DEPBAR`, `NANOSLEEP`, `BPT` |
//! | `tensor` | 6i | `HMMA`, `IMMA`, `LDSM`, `MOVM` |
//!
//! Where NAK models an instruction as a form of another (the memory spaces of `LD` and `ST`,
//! `RED` as an `ATOM` without a destination, `SURED` as an `SUATOM` without one, the uniform
//! forms of the ALU instructions such as `UMOV`, `ULDC` and `UPLOP3` as the vector op with
//! uniform operands), the variant is NAK's and the slice adds the field that tells them apart.
//! That split of the uniform forms is NAK's model, not a fact of the encoding; slice 6d may
//! revisit it. The instructions only PR #4588 knows (`BRX`, `JMX`, `CALL`, `RET`, `DEPBAR`,
//! `P2R`, `R2P`, `FSEL`, `I2FP`, `UP2UR`, `LDGSTS`, `LDGDEPBAR`, `BPT`) get variants of their
//! own, named in NAK's style.
//!
//! An instruction its slice has not built yet is a **shell**: an empty struct that prints its
//! mnemonic, which no decode form produces and the test assembler refuses
//! (`EncodeError::NoEncoding`). A slice turns a shell into a real struct in its own module
//! (fields, printer, decode forms, encoder) without touching the enum. The mnemonics are
//! `nvdisasm`'s names lower-cased; those of shells are guesses until their slice checks them.

use std::fmt::{self, Display, Formatter};

use super::operand::PredSrc;

pub(crate) mod attribute;
pub(crate) mod control;
pub(crate) mod float;
pub(crate) mod half_double;
pub(crate) mod integer;
pub(crate) mod memory;
pub(crate) mod tensor;
pub(crate) mod texture;
pub(crate) mod uniform;

pub use attribute::SpecialReg;
pub use integer::MovSrc;

/// An operation's mnemonic, as `nvdisasm` writes it lower-cased (for a shell, a guess).
pub(crate) trait Named {
    /// The mnemonic.
    const MNEMONIC: &'static str;
}

/// Defines shell operations: for each, an empty struct, its mnemonic and a printer that writes
/// the mnemonic; and, for tests, a list of every shell of the module as an [`Op`].
macro_rules! shells {
    ($($variant:ident($name:ident) = $mnemonic:literal;)*) => {
        $(
            #[doc = concat!("`", $mnemonic, "`: a shell, not decoded yet (see the `op` module).")]
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
            pub struct $name {}

            impl $crate::decode::op::Named for $name {
                const MNEMONIC: &'static str = $mnemonic;
            }

            impl ::std::fmt::Display for $name {
                fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                    f.write_str($mnemonic)
                }
            }
        )*

        /// Every shell of this module, as an operation.
        #[cfg(test)]
        pub(crate) const SHELLS: &[$crate::decode::Op] =
            &[$($crate::decode::Op::$variant($name {})),*];
    };
}
pub(crate) use shells;

/// Calls `$callback!` with every operation, as `Variant(Struct)` pairs grouped by the module
/// that defines them, in slice order: the one list the [`Op`] enum, its printer, the
/// re-exports and the test assembler's dispatch are generated from.
macro_rules! op_list {
    ($callback:ident) => {
        $callback! {
            integer {
                IAdd3(OpIAdd3), IAdd3X(OpIAdd3X), IMad(OpIMad), IMad64(OpIMad64), Lop3(OpLop3),
                Shf(OpShf), Lea(OpLea), LeaX(OpLeaX), ISetP(OpISetP), IMnMx(OpIMnMx),
                Prmt(OpPrmt), PopC(OpPopC), Flo(OpFlo), BRev(OpBRev), Sgxt(OpSgxt),
                BMsk(OpBMsk), IAbs(OpIAbs), IDp4(OpIDp4), Mov(OpMov), Sel(OpSel),
                PLop3(OpPLop3), P2R(OpP2R), R2P(OpR2P),
            }
            float {
                FAdd(OpFAdd), FMul(OpFMul), FFma(OpFFma), FMnMx(OpFMnMx), FSet(OpFSet),
                FSetP(OpFSetP), FSel(OpFSel), MuFu(OpMuFu), FSwzAdd(OpFSwzAdd), FRnd(OpFRnd),
                F2F(OpF2F), F2I(OpF2I), I2F(OpI2F), I2FP(OpI2FP), F2FP(OpF2FP),
            }
            half_double {
                HAdd2(OpHAdd2), HFma2(OpHFma2), HMul2(OpHMul2), HSet2(OpHSet2),
                HSetP2(OpHSetP2), HMnMx2(OpHMnMx2), DAdd(OpDAdd), DMul(OpDMul), DFma(OpDFma),
                DSetP(OpDSetP),
            }
            uniform {
                R2UR(OpR2UR), UP2UR(OpUP2UR),
            }
            memory {
                Ld(OpLd), St(OpSt), Ldc(OpLdc), Atom(OpAtom), CCtl(OpCCtl), MemBar(OpMemBar),
                LdGsts(OpLdGsts), LdgDepBar(OpLdgDepBar),
            }
            texture {
                Tex(OpTex), Tld(OpTld), Tld4(OpTld4), Tmml(OpTmml), Txd(OpTxd), Txq(OpTxq),
                SuLd(OpSuLd), SuSt(OpSuSt), SuAtom(OpSuAtom),
            }
            attribute {
                ALd(OpALd), ASt(OpASt), Ipa(OpIpa), AL2P(OpAL2P), S2R(OpS2R), CS2R(OpCS2R),
                PixLd(OpPixLd), Out(OpOut), OutFinal(OpOutFinal), Isberd(OpIsberd),
                Isbewr(OpIsbewr), LdTram(OpLdTram),
            }
            control {
                Bra(OpBra), Brx(OpBrx), Jmx(OpJmx), Call(OpCall), Ret(OpRet), Exit(OpExit),
                Kill(OpKill), BSSy(OpBSSy), BSync(OpBSync), Break(OpBreak), BMov(OpBMov),
                WarpSync(OpWarpSync), Bar(OpBar), Vote(OpVote), Shfl(OpShfl), Match(OpMatch),
                Redux(OpRedux), Nop(OpNop), DepBar(OpDepBar), Nanosleep(OpNanosleep),
                Bpt(OpBpt),
            }
            tensor {
                Hmma(OpHmma), Imma(OpImma), Ldsm(OpLdsm), Movm(OpMovm),
            }
        }
    };
}
#[cfg(any(test, feature = "test-support"))]
pub(crate) use op_list;

/// Defines [`Op`], its mnemonic, its printer and the re-exports of every operation struct from
/// the list.
macro_rules! define_op {
    ($($module:ident { $($variant:ident($name:ident),)* })*) => {
        $(pub use $module::{$($name),*};)*

        /// A decoded instruction's operation.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Op {
            $($(
                #[doc = concat!("[`", stringify!($name), "`].")]
                $variant($name),
            )*)*
        }

        impl Op {
            /// The operation's mnemonic, as `nvdisasm` writes it lower-cased, without modifiers.
            pub const fn mnemonic(&self) -> &'static str {
                match self {
                    $($(Self::$variant(_) => <$name as Named>::MNEMONIC,)*)*
                }
            }
        }

        impl Display for Op {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                match self {
                    $($(Self::$variant(op) => op.fmt(f),)*)*
                }
            }
        }

        /// Every operation's mnemonic, in list order.
        #[cfg(test)]
        pub(crate) const MNEMONICS: &[&str] = &[$($(<$name as Named>::MNEMONIC),*),*];
    };
}
op_list!(define_op);

/// A decoded instruction: an operation under a guard predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Instr {
    /// The guard: the instruction runs only in threads where it is true.
    pub guard: PredSrc,
    /// The operation.
    pub op: Op,
}

impl Instr {
    /// `op`, always run.
    pub const fn new(op: Op) -> Self {
        Self {
            guard: PredSrc::TRUE,
            op,
        }
    }
}

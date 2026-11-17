//! Slice 6f's encoders: none yet, every operation is a shell.

use super::no_encoding;
use crate::decode::{OpSuAtom, OpSuLd, OpSuSt, OpTex, OpTld, OpTld4, OpTmml, OpTxd, OpTxq};

no_encoding!(
    OpTex, OpTld, OpTld4, OpTmml, OpTxd, OpTxq, OpSuLd, OpSuSt, OpSuAtom
);

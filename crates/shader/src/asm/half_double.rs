//! Slice 6c's encoders: none yet, every operation is a shell.

use super::no_encoding;
use crate::decode::{
    OpDAdd, OpDFma, OpDMul, OpDSetP, OpHAdd2, OpHFma2, OpHMnMx2, OpHMul2, OpHSet2, OpHSetP2,
};

no_encoding!(
    OpHAdd2, OpHFma2, OpHMul2, OpHSet2, OpHSetP2, OpHMnMx2, OpDAdd, OpDMul, OpDFma, OpDSetP,
);

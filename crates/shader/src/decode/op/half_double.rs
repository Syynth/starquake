//! Slice 6c, half and double precision: every instruction is a shell until the slice is built
//! (see the `op` module).

use super::super::form::Form;
use super::shells;

/// The group's forms: none yet.
pub(crate) const FORMS: &[Form] = &[];

/// PR #4588's rows for the group's forms: none yet.
#[cfg(test)]
pub(crate) const REFERENCE_ROWS: &[super::super::form::ReferenceRow] = &[];

/// Generators of the group's decoded operations: none yet.
#[cfg(test)]
pub(crate) fn any_ops() -> Vec<proptest::strategy::BoxedStrategy<super::Op>> {
    Vec::new()
}

shells! {
    HAdd2(OpHAdd2) = "hadd2";
    HFma2(OpHFma2) = "hfma2";
    HMul2(OpHMul2) = "hmul2";
    HSet2(OpHSet2) = "hset2";
    HSetP2(OpHSetP2) = "hsetp2";
    HMnMx2(OpHMnMx2) = "hmnmx2";
    DAdd(OpDAdd) = "dadd";
    DMul(OpDMul) = "dmul";
    DFma(OpDFma) = "dfma";
    DSetP(OpDSetP) = "dsetp";
}

//! Slice 6i, tensor (decode only): every instruction is a shell until the slice is built (see
//! the `op` module).

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
    Hmma(OpHmma) = "hmma";
    Imma(OpImma) = "imma";
    Ldsm(OpLdsm) = "ldsm";
    Movm(OpMovm) = "movm";
}

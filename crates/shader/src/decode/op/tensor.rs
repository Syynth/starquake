//! Slice 6i, tensor (decode only): every instruction is a shell until the slice is built (see
//! the `op` module).

use super::super::form::Form;
use super::shells;

/// The group's forms: none yet.
pub(crate) const FORMS: &[Form] = &[];

shells! {
    Hmma(OpHmma) = "hmma";
    Imma(OpImma) = "imma";
    Ldsm(OpLdsm) = "ldsm";
    Movm(OpMovm) = "movm";
}

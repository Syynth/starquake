//! Slice 6f, texture and surface: every instruction is a shell until the slice is built (see
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
    Tex(OpTex) = "tex";
    Tld(OpTld) = "tld";
    Tld4(OpTld4) = "tld4";
    Tmml(OpTmml) = "tmml";
    Txd(OpTxd) = "txd";
    Txq(OpTxq) = "txq";
    SuLd(OpSuLd) = "suld";
    SuSt(OpSuSt) = "sust";
    SuAtom(OpSuAtom) = "suatom";
}

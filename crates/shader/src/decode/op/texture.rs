//! Slice 6f, texture and surface: every instruction is a shell until the slice is built (see
//! the `op` module).

use super::super::form::Form;
use super::shells;

/// The group's forms: none yet.
pub(crate) const FORMS: &[Form] = &[];

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

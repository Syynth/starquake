//! Slice 6e, memory: every instruction is a shell until the slice is built (see the `op`
//! module, which says which mnemonics share a variant).

use super::super::form::Form;
use super::shells;

/// The group's forms: none yet.
pub(crate) const FORMS: &[Form] = &[];

shells! {
    Ld(OpLd) = "ld";
    St(OpSt) = "st";
    Ldc(OpLdc) = "ldc";
    Atom(OpAtom) = "atom";
    CCtl(OpCCtl) = "cctl";
    MemBar(OpMemBar) = "membar";
    LdGsts(OpLdGsts) = "ldgsts";
    LdgDepBar(OpLdgDepBar) = "ldgdepbar";
}

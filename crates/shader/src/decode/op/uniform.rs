//! Slice 6d, the uniform datapath: every instruction is a shell until the slice is built (see
//! the `op` module, which says how the uniform forms of the ALU instructions are modelled).

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
    R2UR(OpR2UR) = "r2ur";
    UP2UR(OpUP2UR) = "up2ur";
}

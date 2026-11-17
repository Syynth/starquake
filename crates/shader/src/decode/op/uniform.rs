//! Slice 6d, the uniform datapath: every instruction is a shell until the slice is built (see
//! the `op` module, which says how the uniform forms of the ALU instructions are modelled).

use super::super::form::Form;
use super::shells;

/// The group's forms: none yet.
pub(crate) const FORMS: &[Form] = &[];

shells! {
    R2UR(OpR2UR) = "r2ur";
    UP2UR(OpUP2UR) = "up2ur";
}

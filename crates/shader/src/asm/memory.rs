//! Slice 6e's encoders: none yet, every operation is a shell.

use super::no_encoding;
use crate::decode::{OpAtom, OpCCtl, OpLd, OpLdGsts, OpLdc, OpLdgDepBar, OpMemBar, OpSt};

no_encoding!(
    OpLd,
    OpSt,
    OpLdc,
    OpAtom,
    OpCCtl,
    OpMemBar,
    OpLdGsts,
    OpLdgDepBar
);

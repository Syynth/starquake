//! Slice 6i's encoders: none yet, every operation is a shell.

use super::no_encoding;
use crate::decode::{OpHmma, OpImma, OpLdsm, OpMovm};

no_encoding!(OpHmma, OpImma, OpLdsm, OpMovm);

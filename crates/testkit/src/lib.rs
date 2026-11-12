//! Test helpers that depend on none of the workspace's crates (`docs/specs/testing.md`):
//! A64 instruction encoders for guest test programs, a builder for synthetic NRO files, and a
//! few whole guest programs for end-to-end runs.
//! Fakes for a crate's own traits live in that crate, behind its `test-support` feature.

pub mod a64;
pub mod nro;
pub mod programs;

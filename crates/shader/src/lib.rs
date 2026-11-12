//! SM86 shader translation.
//!
//! Decodes SASS into an IR and lowers it to a validated `naga` module plus tables of its
//! interface, which the GPU crate carries to the render backend; wgpu takes the module directly
//! (`docs/specs/shader-pipeline.md`). Instruction encodings come from Mesa NAK (MIT).
//!
//! This is slice 1 of the spec's port slices, the walking skeleton: [`decode`] reads six forms
//! (`MOV`, `FFMA`, `ALD`, `AST`, `EXIT`, `S2R`), [`translate`] turns straight-line programs of
//! them into the minimal [`ir`], and [`lower`] builds and validates the naga module and its
//! [`host`] tables. [`program::translate_program`] runs the whole chain on program bytes. The
//! `asm` module (feature `test-support`) is the test assembler that builds instruction words.
//!
//! Experiment E1, run before this slice, lives in `lower`'s tests and its results in the spec.

#[cfg(feature = "test-support")]
pub mod asm;
pub mod decode;
#[cfg(test)]
mod gpu_harness;
pub mod host;
pub mod ir;
pub mod lower;
pub mod program;
#[cfg(test)]
mod testing;
pub mod translate;

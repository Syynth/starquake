//! SM86 shader translation.
//!
//! Decodes SASS into an IR and lowers it to a validated `naga` module plus tables of its
//! interface, which the GPU crate carries to the render backend; wgpu takes the module directly
//! (`docs/specs/shader-pipeline.md`). Instruction encodings come from Mesa NAK (MIT).
//!
//! Slice 1 of the spec's port slices built the walking skeleton: [`translate`] turns
//! straight-line programs of six instructions (`MOV`, `FFMA`, `ALD`, `AST`, `EXIT`, `S2R`) into
//! the [`ir`], and [`lower`] builds and validates the naga module and its [`host`] tables.
//! [`program::translate_program`] runs the whole chain on program bytes. [`decode`] is slice 4's
//! decoder framework: the form table with known-bit masks, the operand types, the full `Op`
//! enum (instructions not decoded yet are shells), the printer, and the test assembler (`asm`,
//! feature `test-support`) that builds instruction words; it decodes the six instructions in
//! every operand form NAK encodes for them. The [`ir`] is slice 5's IR core: Eden's operation
//! set in arenas, with a builder, a text form that prints and parses, and a verifier.
//!
//! Slice 9 defines the translation inputs and what the GPU crate builds against: the
//! [`exchange`] of questions about engine state and their answers, the [`variant`] key with its
//! versioned encoding, the [`driver_constants`] layout, and the [`lookup`] request and result
//! types (the shader cache that uses them is slice 19).
//!
//! Slice 2 parses the shader program header in front of every graphics program ([`sph`]): the
//! stage, the attributes and system values read and written, the pixel interpolation modes, the
//! colour targets and the local memory size.
//!
//! Slice 6h decodes the control and warp group (`decode`'s `OpBra` to `OpNanosleep`):
//! branches with their [`decode::BranchOffset`], convergence barriers, calls and returns, and
//! warp votes, shuffles, matches and reductions; `JMX` and `BPT` stay shells.
//!
//! Experiment E1, run before this slice, lives in `lower`'s tests and its results in the spec.

#[cfg(feature = "test-support")]
pub mod asm;
pub mod decode;
pub mod driver_constants;
pub mod exchange;
#[cfg(test)]
mod gpu_harness;
pub mod host;
pub mod ir;
pub mod lookup;
pub mod lower;
pub mod program;
pub mod sph;
#[cfg(test)]
mod testing;
pub mod translate;
pub mod variant;

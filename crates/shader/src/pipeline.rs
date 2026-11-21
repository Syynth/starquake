//! The shader cache's production translator: [`PipelineTranslator`] implements the cache's
//! [`Translator`] seam with the whole translation pipeline (`docs/specs/shader-pipeline.md`, "The
//! translation pipeline", "Translation inputs: questions and answers" and "Compilation, threads
//! and stutter"; issue #484).
//!
//! No reference is ported: the wiring is the spec's. Each step is its slice's, and this module
//! only runs them in the spec's order.
//!
//! # The two halves
//!
//! - **[`Translator::program`]**, on the GPU back thread at a program-entry miss, keeps the
//!   capture and parses its shader program header when it has one ([`ProgramHeader`], slice 2;
//!   a compute capture has none). Both depend on the bytes only, not on the stage the program
//!   is bound to, as the seam requires: a program entry is shared by every stage whose programs
//!   have the same bytes. A header that does not parse is [`Rejection::Header`], kept in the
//!   cache's negative cache by content hash. The entry's questions ([`Questioner`]) are its
//!   jump tables' words (slice 10's [`JumpTableQuestions`], slice 9's exchange).
//! - **[`Translator::shader`]**, on a compile worker, translates one shader of the entry, in
//!   this order, stopping at the first failure:
//!   1. the header against the stage ([`check_header`], [`Rejection::Info`]);
//!   2. the jump tables, resolved from the answers ([`resolve_jump_tables`]), so a `BRX` whose
//!      table the answers give is translated with its targets; a table naming an instruction
//!      past the capture is refused ([`PipelineRefusal::JumpTargetPastCapture`], "What it does
//!      not do yet"); a `BRX` the selector pattern does not track has no target and is refused
//!      by flow analysis ([`crate::flow::FlowError::NoTarget`], as [`Rejection::Translate`]);
//!   3. the IR half of the pipeline ([`pipeline_ir`]: the translation driver, the structurizer,
//!      the SSA rewrite and the passes), given the header's colour outputs and, for a pixel
//!      program, its input map;
//!   4. collect info ([`collect`], slice 15, [`Rejection::Info`]): the spec's pass 7 ("Dead
//!      code elimination, then collect info"), run for its refusals ([`crate::info::InfoError`],
//!      among them an unknown attribute, a colour output that does not exist and a
//!      constant-buffer bank or offset out of range) and for the constant-buffer sizes the
//!      lowering declares ([`constant_buffer_sizes`]); its interface tables are not passed on
//!      yet;
//!   5. the host lowering passes the host target's features call for ([`lower_for_host`],
//!      slice 18, [`Rejection::HostLower`]): 64-bit floats become 32-bit ones where the target
//!      lacks them, and 16-bit floats likewise; the spec's pass 8;
//!   6. the lowering ([`lower_with`], [`Rejection::Lower`]), with the variant key's workgroup
//!      and shared-memory sizes and emulation flags, the local-memory size (module docs, "The
//!      local-memory size"), the constant-buffer sizes of step 4 and the host target's
//!      features.
//!
//! # The local-memory size
//!
//! [`local_memory_bytes`] gives the size the lowering sizes and bounds the local array by
//! (`crate::lower`, "Local and shared memory"), as each reference's environment gives
//! `LocalMemorySize()`:
//!
//! - **Graphics:** the header's size ([`crate::sph::Common::local_memory_size`]) plus its CRS
//!   size ([`crate::sph::Common::local_memory_crs_size`]), as `GraphicsEnvironment` sets
//!   `local_memory_size = static_cast<u32>(local_size) +
//!   sph.common3.shader_local_memory_crs_size` (`eden/src/video_core/shader_environment.cpp`
//!   @ cf654052, `eden-nx2` @ 50fa86bb and `eden-pr4588` @ e5fb3b97 alike). The CRS field is
//!   named only in version 3 of the header layout (`SHADER_LOCAL_MEMORY_CRS_SIZE`, bits 96 to
//!   119); version 4 leaves those bits unnamed, so reading them as the CRS size on Switch 2
//!   follows PR #4588, which runs titles on it (graphics-server decision 8), and is a
//!   **guess** until a large-local-memory shader (verification E3) settles it.
//! - **Compute:** [`VariantKey::local_memory_size`], which the GPU crate fills from the QMD
//!   (`local_pos_alloc + local_crs_alloc`, `ComputeEnvironment` in the same files).
//!
//! **Divergence:** Eden asserts the header's size fits 32 bits (a soft `ASSERT`) and then
//! truncates it to `u32` and adds the CRS size in 32 bits, so a size of 2^32 or more wraps.
//! Here the sum is taken in 64 bits (it cannot wrap: the header's size is below 2^56 and the
//! CRS size below 2^32), and the lowering refuses any size past
//! [`crate::lower::MAX_LOCAL_MEMORY_BYTES`]. Pinned by
//! `tests::a_header_size_of_2_to_the_32_is_refused_not_wrapped`.
//!
//! # What it does not do yet
//!
//! - **The rest of the graphics interfaces and resources** are slice 17's (#376): constant
//!   buffers and the emulation the key's flags ask for at the epilogue are lowered, but the
//!   lowering still refuses by name, so a program that needs them is rejected, not drawn
//!   wrongly: a pixel program's generic inputs ([`crate::lower::LowerError::Input`]), a
//!   constant buffer whose bank is a register
//!   ([`crate::lower::LowerError::ConstantBufferRegisterBank`]), and every texture, image and
//!   global-memory operation ([`crate::lower::LowerError::Operation`]). Collect info's
//!   interface tables (slice 15) are not passed on yet.
//! - **Jump tables do not extend the capture** (the cache's module docs, "Where this differs
//!   from the spec"): the tables are resolved against the capture's own bytes, since
//!   [`Translator::shader`] has no guest-memory reader, so a target at or past the capture's
//!   end cannot be read and the extension is abandoned ([`crate::capture::Extension::Abandoned`]).
//!   The program is then refused ([`PipelineRefusal::JumpTargetPastCapture`]).
//!
//! # Divergences from the reference
//!
//! - **A jump target past the capture is a refusal, not a dropped target.** The reference
//!   (`eden-pr4588/src/shader_recompiler/frontend/ampere/program_capture.h` @ e5fb3b97,
//!   `ResolveCaptureIndirectBranches`) first extends the capture through guest memory to the
//!   furthest target, and drops a target only when that extension fails. Here no extension is
//!   tried, so dropping the target would end the `BRX`'s comparison chain in an unreachable
//!   terminator, and a case past the first `EXIT` (scoping-doc row 92's situation) would lower
//!   to a shader that does undefined things when that case is selected. Refusing is stricter
//!   than the reference: a program the reference translates by extending its capture is
//!   refused here until the translator gets a guest-memory reader. Pinned by
//!   `tests::a_target_past_the_capture_is_refused`, with the last captured instruction as the
//!   accepted edge.

#[cfg(test)]
mod tests;

use thiserror::Error;

use crate::cache::{ShaderInput, Translator};
use crate::capture::{Capture, Extension};
use crate::decode::Instruction;
use crate::exchange::{Answers, Question, Questioner};
use crate::flow::{BranchTables, JumpTableQuestions, resolve_jump_tables};
use crate::host::{HostCaps, HostShader, Requirements};
use crate::host_lower::lower_for_host;
use crate::info::{check_header, collect, translation_stage};
use crate::lower::{LowerInputs, constant_buffer_sizes, lower_with};
use crate::program::{Rejection, pipeline_ir};
use crate::sph::{Io, ProgramHeader};
use crate::translate::{ColorOutputs, Stage, TranslateError};
use crate::variant::VariantKey;

/// What the production translator refuses itself, beside the refusals of the steps it runs
/// (module docs, "What it does not do yet").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PipelineRefusal {
    /// A jump table names an instruction at or past the capture's end, which the translator
    /// cannot extend the capture to (module docs, "Divergences from the reference").
    #[error(
        "a jump table names an instruction past the captured program, which is not extended yet"
    )]
    JumpTargetPastCapture,
}

/// The shader cache's production translator (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineTranslator;

/// A program entry of [`PipelineTranslator`]: the capture and its parsed header.
#[derive(Clone, Debug)]
pub struct PipelineProgram {
    capture: Capture,
    header: Option<ProgramHeader>,
}

impl PipelineProgram {
    /// The program entry for `capture`: its header parsed when it has one (module docs, "The
    /// two halves").
    pub fn new(capture: &Capture) -> Result<Self, Rejection> {
        let header = if capture.header().is_empty() {
            None
        } else {
            Some(ProgramHeader::from_program(capture.bytes())?.0)
        };
        Ok(Self {
            capture: capture.clone(),
            header,
        })
    }

    /// The capture.
    pub const fn capture(&self) -> &Capture {
        &self.capture
    }

    /// The parsed header, when the capture has one.
    pub const fn header(&self) -> Option<&ProgramHeader> {
        self.header.as_ref()
    }

    /// The reader the jump tables are resolved through: the capture's own bytes, so the capture
    /// is never extended (module docs, "What it does not do yet").
    fn reader(&self) -> &[u8] {
        self.capture.bytes()
    }

    /// Each `BRX`'s targets, from the words `answers` gives its table (module docs, step 2).
    ///
    /// A missing answer is [`Rejection::Translate`]; a table naming an instruction at or past
    /// the capture's end is [`PipelineRefusal::JumpTargetPastCapture`].
    pub fn branch_tables(&self, answers: &Answers) -> Result<BranchTables, Rejection> {
        let mut copy = self.capture.clone();
        let resolved =
            resolve_jump_tables(&mut copy, self.reader(), answers).map_err(TranslateError::from)?;
        if matches!(resolved.extension, Extension::Abandoned(_)) {
            return Err(PipelineRefusal::JumpTargetPastCapture.into());
        }
        Ok(resolved.branch_tables())
    }
}

impl Questioner for PipelineProgram {
    /// The words of the capture's jump tables ([`JumpTableQuestions`] over the capture's own
    /// bytes).
    fn questions(&self, answers: &Answers) -> Vec<Question> {
        JumpTableQuestions::new(&self.capture, self.reader()).questions(answers)
    }
}

impl Translator for PipelineTranslator {
    type Program = PipelineProgram;
    type Shader = HostShader;

    fn program(&self, capture: &Capture) -> Result<PipelineProgram, Rejection> {
        PipelineProgram::new(capture)
    }

    fn shader(&self, input: &ShaderInput<'_, PipelineProgram>) -> Result<HostShader, Rejection> {
        translate_shader(input)
    }
}

/// The host target's features as the lowering and the host lowering passes take them: the
/// features of [`Requirements`] they can need ([`HostCaps`]); 64-bit atomics are refused by the
/// lowering whatever the target has.
#[must_use]
pub const fn host_caps(caps: Requirements) -> HostCaps {
    HostCaps {
        f16: caps.f16,
        f64: caps.f64,
        i64: caps.i64,
        subgroup: caps.subgroup,
        subgroup_size_32: caps.subgroup_size_32,
    }
}

/// One shader's translation (module docs, "The two halves", [`Translator::shader`]).
pub fn translate_shader(input: &ShaderInput<'_, PipelineProgram>) -> Result<HostShader, Rejection> {
    let program = input.program;
    let header = program.header();
    check_header(input.stage, header)?;
    let tables = program.branch_tables(input.answers)?;
    let stage = header.map_or(
        Stage {
            stage: input.stage,
            color_outputs: ColorOutputs::default(),
        },
        |header| translation_stage(input.stage, header),
    );
    let pixel = header.and_then(|header| match &header.io {
        Io::Pixel(io) => Some(&**io),
        Io::Vtg(_) => None,
    });
    let code: Vec<Instruction> = program.capture.instructions().collect();
    let mut function = pipeline_ir(stage, pixel, &code, &tables)?;
    let info = collect(input.stage, header, &function)?;
    let caps = host_caps(input.target.caps);
    lower_for_host(&mut function, caps)?;
    let inputs = LowerInputs {
        workgroup_size: input.variant.workgroup_size,
        local_memory_bytes: local_memory_bytes(header, input.variant),
        shared_memory_bytes: input.variant.shared_memory_size,
        caps,
        constant_buffers: constant_buffer_sizes(&info),
        emulation: input.variant.emulation,
    };
    Ok(lower_with(input.stage, &function, &inputs)?)
}

/// The local-memory size in bytes of a program with `header` (module docs, "The local-memory
/// size"): a graphics program's header size plus its CRS size, or, for a compute program
/// (which has no header; [`check_header`] refuses a graphics program without one and a
/// compute program with one before this is asked), the variant key's
/// [`VariantKey::local_memory_size`]. The key's field is not read for a graphics program.
#[must_use]
pub fn local_memory_bytes(header: Option<&ProgramHeader>, variant: &VariantKey) -> u64 {
    header.map_or(u64::from(variant.local_memory_size), |header| {
        // Below 2^56 plus below 2^32: the sum cannot wrap, and saturating keeps that a proof.
        header
            .common
            .local_memory_size()
            .saturating_add(u64::from(header.common.local_memory_crs_size))
    })
}

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
//!      table the answers give is translated with its targets; a `BRX` the selector pattern
//!      does not track has no target and is refused by flow analysis
//!      ([`crate::flow::FlowError::NoTarget`], as [`Rejection::Translate`]);
//!   3. the IR half of the pipeline ([`pipeline_ir`]: the translation driver, the structurizer,
//!      the SSA rewrite and the passes), given the header's colour outputs and, for a pixel
//!      program, its input map;
//!   4. for a compute program, its local memory ([`PipelineRefusal::ComputeLocalMemory`]);
//!   5. the lowering ([`lower_with`], [`Rejection::Lower`]), with the variant key's workgroup
//!      and shared-memory sizes, the header's local-memory size and the host target's
//!      features.
//!
//! # What it does not do yet
//!
//! - **Graphics interfaces and resources** are slice 17's (#376): the lowering refuses what 17
//!   adds by name, so a program that needs it is rejected, not drawn wrongly: a pixel
//!   program's generic inputs ([`crate::lower::LowerError::Input`]), and every constant-buffer,
//!   texture, image and global-memory operation ([`crate::lower::LowerError::Operation`]).
//!   Collect info (slice 15) is not run here: nothing the lowering takes reads it until 17.
//! - **A compute program's local memory** has no size: Eden takes it from the QMD
//!   (`local_pos_alloc + local_crs_alloc`), which no question, key field or driver constant
//!   carries yet (the exchange's module docs, "For the owner: what the questions do not
//!   cover"). A compute program that still reads or writes local memory after the passes is
//!   refused ([`PipelineRefusal::ComputeLocalMemory`]) rather than lowered over an array of no
//!   words, whose reads would give 0. Pinned by
//!   `tests::a_compute_program_with_local_memory_is_refused`.
//! - **Jump tables do not extend the capture** (the cache's module docs, "Where this differs
//!   from the spec"): the tables are resolved against the capture's own bytes, so a target past
//!   the capture's end cannot be read, the extension is abandoned and that target is dropped,
//!   as the reference drops targets outside a capture it could not extend
//!   (`eden-pr4588/src/shader_recompiler/frontend/ampere/program_capture.h` @ e5fb3b97,
//!   `ResolveCaptureIndirectBranches`). Pinned by
//!   `tests::a_target_past_the_capture_is_dropped`.

#[cfg(test)]
mod tests;

use thiserror::Error;

use crate::cache::{ShaderInput, Translator};
use crate::capture::Capture;
use crate::decode::Instruction;
use crate::exchange::{Answers, Question, Questioner};
use crate::flow::{BranchTables, FlowError, JumpTableQuestions, resolve_jump_tables};
use crate::host::{HostCaps, HostShader, Requirements};
use crate::info::{check_header, translation_stage};
use crate::ir::{Function, Inst, Opcode};
use crate::lower::{LowerInputs, lower_with};
use crate::program::{Rejection, pipeline_ir};
use crate::sph::{Io, ProgramHeader};
use crate::translate::{ColorOutputs, GuestStage, Stage, TranslateError};

/// What the production translator refuses itself, beside the refusals of the steps it runs
/// (module docs, "What it does not do yet").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PipelineRefusal {
    /// A compute program reads or writes local memory, whose size comes from the QMD, which no
    /// translation input carries yet.
    #[error(
        "a compute program that uses local memory, whose size no translation input carries yet"
    )]
    ComputeLocalMemory,
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
    pub fn branch_tables(&self, answers: &Answers) -> Result<BranchTables, FlowError> {
        let mut copy = self.capture.clone();
        Ok(resolve_jump_tables(&mut copy, self.reader(), answers)?.branch_tables())
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

/// The host target's features as the lowering takes them: the features of [`Requirements`] the
/// lowering can need ([`HostCaps`]); 64-bit floats and 64-bit atomics are refused by the
/// lowering whatever the target has.
#[must_use]
pub const fn host_caps(caps: Requirements) -> HostCaps {
    HostCaps {
        f16: caps.f16,
        i64: caps.i64,
        subgroup: caps.subgroup,
    }
}

/// One shader's translation (module docs, "The two halves", [`Translator::shader`]).
pub fn translate_shader(input: &ShaderInput<'_, PipelineProgram>) -> Result<HostShader, Rejection> {
    let program = input.program;
    let header = program.header();
    check_header(input.stage, header)?;
    let tables = program
        .branch_tables(input.answers)
        .map_err(TranslateError::from)?;
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
    let function = pipeline_ir(stage, pixel, &code, &tables)?;
    if input.stage == GuestStage::Compute && uses_local_memory(&function) {
        return Err(PipelineRefusal::ComputeLocalMemory.into());
    }
    let inputs = LowerInputs {
        workgroup_size: input.variant.workgroup_size,
        local_memory_bytes: header.map_or(0, |header| header.common.local_memory_size()),
        shared_memory_bytes: input.variant.shared_memory_size,
        caps: host_caps(input.target.caps),
    };
    Ok(lower_with(input.stage, &function, &inputs)?)
}

/// Whether `function` lists a local-memory operation.
fn uses_local_memory(function: &Function) -> bool {
    function.listed().any(|(_, inst)| {
        matches!(inst, Inst::Op(op) if matches!(op.opcode, Opcode::LoadLocal | Opcode::WriteLocal))
    })
}

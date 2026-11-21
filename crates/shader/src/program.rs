//! The walking skeleton end to end: program bytes to a validated naga module and its tables.
//!
//! Bytes are split into instructions; the translation driver ([`translate_flow`], slice 12i)
//! rebuilds their control flow, decodes the instructions its blocks hold and translates them to
//! IR; the structurizer (slice 11) structures the IR, and the IR is lowered and validated. The
//! first failure is the rejection; there is no partial translation (spec, "Instruction
//! translation"). The bytes carry no jump tables, so a `BRX` has no target and is refused
//! ([`crate::flow::FlowError::NoTarget`]); jump tables come with the questions and answers
//! (slice 9) when the shader cache drives the chain.

use thiserror::Error;

use crate::cache::CacheRefusal;
use crate::capture::CaptureError;
use crate::decode::{DecodeError, Instr, Instruction, LengthError, decode};
use crate::exchange::ExchangeError;
use crate::flow::{BranchTables, IrBlocks};
use crate::host::HostShader;
use crate::info::{InfoError, ShaderInfo, check_header, collect, translation_stage};
use crate::ir::Function;
use crate::lower::{LowerError, lower};
use crate::opt::{PassError, optimize};
use crate::sph::{HeaderError, Io, PixelIo, ProgramHeader};
use crate::structure::{MAX_STRUCTURE_WORK, StructureError};
use crate::translate::{GuestStage, Stage, TranslateError, translate_flow};

/// Why a program was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Rejection {
    /// The bytes are not whole instructions.
    #[error(transparent)]
    Length(#[from] LengthError),
    /// An instruction does not decode: from [`decode_program`], or one a flow block holds,
    /// which the translation driver decodes ([`TranslateError::Decode`] is reported as this).
    #[error("instruction {index}: {error}")]
    Decode {
        /// The instruction's index.
        index: usize,
        /// Why.
        error: DecodeError,
    },
    /// The program does not translate (every [`TranslateError`] but `Decode`, which is
    /// [`Rejection::Decode`]).
    #[error(transparent)]
    Translate(TranslateError),
    /// The translated program cannot be structured (slice 11).
    #[error(transparent)]
    Structure(#[from] StructureError),
    /// The IR does not lower.
    #[error(transparent)]
    Lower(#[from] LowerError),
    /// The program's questions about engine state were not settled (spec, "Translation
    /// inputs: questions and answers").
    #[error(transparent)]
    Questions(#[from] ExchangeError),
    /// The program could not be captured from guest memory (slice 3, as the shader cache reads
    /// it; slice 19).
    #[error(transparent)]
    Capture(#[from] CaptureError),
    /// The shader cache refused the shader itself (slice 19).
    #[error(transparent)]
    Cache(#[from] CacheRefusal),
    /// The shader program header in front of a graphics program does not parse (slice 2).
    #[error(transparent)]
    Header(#[from] HeaderError),
    /// An IR pass failed (slice 8).
    #[error(transparent)]
    Pass(#[from] PassError),
    /// Collect info refused the program, or its header does not suit its pipeline slot
    /// (slice 15).
    #[error(transparent)]
    Info(#[from] InfoError),
}

impl From<TranslateError> for Rejection {
    /// A driver's decode failure is [`Rejection::Decode`], as [`decode_program`] reports one;
    /// every other translation failure is [`Rejection::Translate`].
    fn from(error: TranslateError) -> Self {
        match error {
            TranslateError::Decode { index, error } => Self::Decode { index, error },
            other => Self::Translate(other),
        }
    }
}

/// Splits `code` into instructions and decodes each one.
pub fn decode_program(code: &[u8]) -> Result<Vec<Instr>, Rejection> {
    Instruction::from_program(code)?
        .into_iter()
        .enumerate()
        .map(|(index, word)| decode(word).map_err(|error| Rejection::Decode { index, error }))
        .collect()
}

/// Splits `code` into instructions and runs the translation driver on them, with `header` for a
/// pixel program and no jump tables.
fn translated(stage: Stage, header: Option<&PixelIo>, code: &[u8]) -> Result<IrBlocks, Rejection> {
    let words = Instruction::from_program(code)?;
    Ok(translate_flow(
        stage,
        header,
        &words,
        &BranchTables::new(0),
    )?)
}

/// Translates `code` to IR with the translation driver: the filled control-flow graph, before
/// structuring.
pub fn program_ir(stage: Stage, code: &[u8]) -> Result<Function, Rejection> {
    Ok(translated(stage, None, code)?.function)
}

/// Translates a program to a host shader: the translation driver, the structurizer with
/// [`MAX_STRUCTURE_WORK`], and the lowering (no SSA pass runs between them until slice 8b, as
/// the lowering's module docs say). A program the driver leaves as one block (no control flow
/// and no guard) is lowered as it is, as the lowering takes a one-block graph (its "What it
/// takes"), so the structurizer's goto-variable resets do not reach a straight-line program's
/// module (pinned by `lower`'s `tests::skeleton_modules_are_unchanged`).
pub fn translate_program(stage: Stage, code: &[u8]) -> Result<HostShader, Rejection> {
    let blocks = translated(stage, None, code)?;
    let function = if blocks.function.block_count() == 1 {
        blocks.function
    } else {
        blocks.structure(MAX_STRUCTURE_WORK)?.blocks.function
    };
    Ok(lower(stage.stage, &function)?)
}

/// A graphics program translated to IR, with its header and what collect info found in it.
#[derive(Clone, Debug)]
pub struct GraphicsIr {
    /// The program's header.
    pub header: ProgramHeader,
    /// The IR, after the slice 8 passes ([`optimize`]).
    pub function: Function,
    /// What collect info found ([`crate::info`]).
    pub info: ShaderInfo,
}

/// Runs a graphics program bound to pipeline slot `slot` from its bytes (the header's first
/// byte onwards) to IR and its info, stopping at the first failure, in this order: the header
/// ([`ProgramHeader::from_program`], [`Rejection::Header`]); the header against the slot
/// ([`check_header`], [`Rejection::Info`]; compute has no header, so a compute slot is refused
/// here as [`InfoError::ComputeHeader`]); the translation driver on the instructions after the
/// header ([`translate_flow`]: flow analysis, decoding, translation), given the header's colour
/// outputs ([`translation_stage`]) and, for a pixel program, its input map; the slice 8 passes
/// ([`optimize`], [`Rejection::Pass`]); and collect info ([`collect`], [`Rejection::Info`]).
/// The passes run on the control-flow graph, not structured: slice 8's SSA rewrite refuses a
/// structured function until slice 8b reworks it (the owner's ruling of 2026-11-19, row 3, puts
/// the structurizer first once it does).
pub fn graphics_program_ir(slot: GuestStage, program: &[u8]) -> Result<GraphicsIr, Rejection> {
    let (header, code) = ProgramHeader::from_program(program)?;
    check_header(slot, Some(&header))?;
    let stage = translation_stage(slot, &header);
    let pixel = match &header.io {
        Io::Pixel(io) => Some(&**io),
        Io::Vtg(_) => None,
    };
    let mut function = translated(stage, pixel, code)?.function;
    optimize(&mut function)?;
    let info = collect(slot, Some(&header), &function)?;
    Ok(GraphicsIr {
        header,
        function,
        info,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::assemble_program;
    use crate::decode::{Op, OpExit, PredSrc};
    use crate::translate::{ColorOutputs, GuestStage};

    const VERTEX: Stage = Stage {
        stage: GuestStage::Vertex,
        color_outputs: ColorOutputs {
            targets: [0; crate::translate::COLOR_TARGETS],
        },
    };

    fn exit_bytes() -> Vec<u8> {
        assemble_program(&[Instr::new(Op::Exit(OpExit {
            keep_ref_count: false,
            no_at_exit: false,
            pred: PredSrc::TRUE,
        }))])
        .expect("encodable")
    }

    #[test]
    fn each_stage_of_the_chain_reports_its_rejection() {
        let mut short = exit_bytes();
        short.pop();
        assert_eq!(
            translate_program(VERTEX, &short).err(),
            Some(Rejection::Length(LengthError {
                len: Instruction::SIZE - 1,
                size: Instruction::SIZE
            }))
        );
        // An unknown word in front of the exit: the driver decodes it and reports it as the
        // chain always has, at its index. The same word after the exit is never reached.
        let mut unknown = vec![0; Instruction::SIZE];
        unknown.extend_from_slice(&exit_bytes());
        unknown.extend_from_slice(&[0; Instruction::SIZE]);
        assert_eq!(
            translate_program(VERTEX, &unknown).err(),
            Some(Rejection::Decode {
                index: 0,
                error: DecodeError::UnknownOpcode { opcode: 0 }
            })
        );
        assert_eq!(
            translate_program(VERTEX, &unknown)
                .err()
                .map(|error| error.to_string()),
            Some("instruction 0: unknown opcode 0x000".into())
        );
        let mut unreached = exit_bytes();
        unreached.extend_from_slice(&[0; Instruction::SIZE]);
        assert!(translate_program(VERTEX, &unreached).is_ok());
        assert_eq!(
            translate_program(VERTEX, &[]).err(),
            Some(Rejection::Translate(TranslateError::Flow(
                crate::flow::FlowError::ProgramSize { len: 0 }
            )))
        );
        let ir = program_ir(VERTEX, &exit_bytes()).expect("translates");
        assert_eq!(ir.to_string(), "block0:\n    return\n");
        assert!(translate_program(VERTEX, &exit_bytes()).is_ok());
    }
}

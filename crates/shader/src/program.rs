//! The translation pipeline end to end: program bytes to a validated naga module and its tables.
//!
//! [`pipeline_ir`] is the IR half, in the spec's order ("The translation pipeline", the
//! flowchart; "Building the IR"; the owner's ruling of 2026-11-19, row 3, Eden's order): the
//! translation driver ([`translate_flow`], slice 12i) rebuilds the program's control flow with
//! the jump tables it is given, decodes the instructions its blocks hold and translates them to
//! IR; the structurizer ([`IrBlocks::structure`], slice 11) structures the IR; and the SSA
//! rewrite and the passes after it ([`optimize`], slices 8 and 8b) run on the structured
//! function. The one production path is the shader cache's production translator
//! ([`crate::pipeline`]), which lowers it with the program's jump tables from slice 9's answers.
//! [`translate_program`] (used by the GPU crate's draw tests) and [`graphics_program_ir`] (used
//! by the shader crate's tests) run it too. The first failure is the rejection; there is no
//! partial translation (spec, "Instruction translation").
//!
//! [`translate_program`] and [`graphics_program_ir`] take bytes only, which carry no jump
//! tables, so a `BRX` there has no target and is refused ([`crate::flow::FlowError::NoTarget`]);
//! the shader cache's translator gives the driver the tables the answers resolve.
//!
//! A program the driver leaves as one block (no control flow and no guard) is not structured:
//! the SSA rewrite and the lowering take a one-block graph as it is (the lowering's "What it
//! takes"), so the structurizer's goto-variable resets do not reach a straight-line program.
//! Pinned by `tests::a_straight_program_is_not_structured`.

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
    /// The shader cache's production translator refused the program itself
    /// ([`crate::pipeline`]'s module docs, "What it does not do yet").
    #[error(transparent)]
    Pipeline(#[from] crate::pipeline::PipelineRefusal),
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

/// The IR half of the translation pipeline, in the spec's order (module docs): the translation
/// driver on `code` for `stage`, with `header` for a pixel program and `tables` giving each
/// `BRX`'s targets ([`translate_flow`]); the structurizer with [`MAX_STRUCTURE_WORK`]
/// ([`IrBlocks::structure`], [`Rejection::Structure`]), unless the driver left one block; and
/// the SSA rewrite and the passes after it ([`optimize`], [`Rejection::Pass`]). The result is
/// what the lowering takes.
pub fn pipeline_ir(
    stage: Stage,
    header: Option<&PixelIo>,
    code: &[Instruction],
    tables: &BranchTables,
) -> Result<Function, Rejection> {
    let mut function = structured(translate_flow(stage, header, code, tables)?)?;
    optimize(&mut function)?;
    Ok(function)
}

/// `blocks` structured with [`MAX_STRUCTURE_WORK`], or its function as it is when it is one
/// block (module docs).
fn structured(blocks: IrBlocks) -> Result<Function, Rejection> {
    if blocks.function.block_count() == 1 {
        Ok(blocks.function)
    } else {
        Ok(blocks.structure(MAX_STRUCTURE_WORK)?.blocks.function)
    }
}

/// Splits `code` into instructions and runs [`pipeline_ir`] on them, with `header` for a pixel
/// program and no jump tables.
fn pipeline_bytes(
    stage: Stage,
    header: Option<&PixelIo>,
    code: &[u8],
) -> Result<Function, Rejection> {
    let words = Instruction::from_program(code)?;
    pipeline_ir(stage, header, &words, &BranchTables::new(0))
}

/// Splits `code` into instructions and runs the translation driver on them, with no header and
/// no jump tables.
fn translated(stage: Stage, code: &[u8]) -> Result<IrBlocks, Rejection> {
    let words = Instruction::from_program(code)?;
    Ok(translate_flow(stage, None, &words, &BranchTables::new(0))?)
}

/// Translates `code` to IR with the translation driver: the filled control-flow graph, before
/// structuring.
pub fn program_ir(stage: Stage, code: &[u8]) -> Result<Function, Rejection> {
    Ok(translated(stage, code)?.function)
}

/// Translates a program's instructions to a host shader: [`pipeline_ir`] with no header and no
/// jump tables, then the lowering ([`lower`], [`Rejection::Lower`]).
pub fn translate_program(stage: Stage, code: &[u8]) -> Result<HostShader, Rejection> {
    let function = pipeline_bytes(stage, None, code)?;
    Ok(lower(stage.stage, &function)?)
}

/// `code` lowered from the IR the translation driver and the structurizer leave, before the
/// SSA rewrite and the passes: the lowering's input for the lowering's own tests, so that each
/// IR form the translators write reaches the lowering (dead code elimination would remove an
/// unused read the lowering is tested to refuse). Not a production path: every production path
/// runs [`pipeline_ir`].
#[cfg(test)]
pub(crate) fn lowered_before_passes(stage: Stage, code: &[u8]) -> Result<HostShader, Rejection> {
    let function = structured(translated(stage, code)?)?;
    Ok(lower(stage.stage, &function)?)
}

/// A graphics program translated to IR, with its header and what collect info found in it.
#[derive(Clone, Debug)]
pub struct GraphicsIr {
    /// The program's header.
    pub header: ProgramHeader,
    /// The IR, structured and after the passes ([`pipeline_ir`]).
    pub function: Function,
    /// What collect info found ([`crate::info`]).
    pub info: ShaderInfo,
}

/// Runs a graphics program bound to pipeline slot `slot` from its bytes (the header's first
/// byte onwards) to IR and its info, stopping at the first failure, in this order: the header
/// ([`ProgramHeader::from_program`], [`Rejection::Header`]); the header against the slot
/// ([`check_header`], [`Rejection::Info`]; compute has no header, so a compute slot is refused
/// here as [`InfoError::ComputeHeader`]); the IR half of the pipeline on the instructions after
/// the header ([`pipeline_ir`]: the translation driver, given the header's colour outputs
/// ([`translation_stage`]) and, for a pixel program, its input map; the structurizer; the SSA
/// rewrite and the passes); and collect info ([`collect`], [`Rejection::Info`]).
pub fn graphics_program_ir(slot: GuestStage, program: &[u8]) -> Result<GraphicsIr, Rejection> {
    let (header, code) = ProgramHeader::from_program(program)?;
    check_header(slot, Some(&header))?;
    let stage = translation_stage(slot, &header);
    let pixel = match &header.io {
        Io::Pixel(io) => Some(&**io),
        Io::Vtg(_) => None,
    };
    let function = pipeline_bytes(stage, pixel, code)?;
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

    /// A program the driver leaves as one block is not structured, and keeps its one block; one
    /// with a guarded instruction (a conditional block) is structured before the passes, which
    /// leave no guest register behind in either.
    #[test]
    fn a_straight_program_is_not_structured() {
        let words = |instrs: &[Instr]| {
            Instruction::from_program(&assemble_program(instrs).expect("encodable"))
                .expect("whole words")
        };
        let exit = Instr::new(Op::Exit(OpExit {
            keep_ref_count: false,
            no_at_exit: false,
            pred: PredSrc::TRUE,
        }));
        let mov = Instr {
            guard: PredSrc {
                reg: crate::decode::Pred::new(0).expect("P0"),
                negated: false,
            },
            ..Instr::new(Op::Mov(crate::decode::OpMov {
                dst: crate::decode::Gpr::new(0),
                src: crate::decode::MovSrc::Imm(7),
                quad_lanes: crate::decode::OpMov::ALL_LANES,
            }))
        };
        let tables = BranchTables::new(0);
        let straight = pipeline_ir(VERTEX, None, &words(&[exit]), &tables).expect("translates");
        assert_eq!(straight.structure(), None);
        assert_eq!(straight.block_count(), 1);
        let guarded = pipeline_ir(VERTEX, None, &words(&[mov, exit]), &tables).expect("translates");
        assert!(guarded.structure().is_some(), "{guarded}");
        assert!(!guarded.to_string().contains("Register"), "{guarded}");
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

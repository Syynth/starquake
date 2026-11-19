//! The walking skeleton end to end: program bytes to a validated naga module and its tables.
//!
//! Bytes are split into instructions, each instruction is decoded, the program is translated
//! to IR and the IR is lowered and validated. The first failure is the rejection; there is no
//! partial translation (spec, "Instruction translation").

use thiserror::Error;

use crate::decode::{DecodeError, Instr, Instruction, LengthError, decode};
use crate::exchange::ExchangeError;
use crate::host::HostShader;
use crate::info::{InfoError, ShaderInfo, check_header, collect, translation_stage};
use crate::ir::Function;
use crate::lower::{LowerError, lower};
use crate::opt::{PassError, optimize};
use crate::sph::{HeaderError, Io, ProgramHeader};
use crate::translate::{GuestStage, Stage, TranslateError, translate, translate_with_header};

/// Why a program was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Rejection {
    /// The bytes are not whole instructions.
    #[error(transparent)]
    Length(#[from] LengthError),
    /// An instruction does not decode.
    #[error("instruction {index}: {error}")]
    Decode {
        /// The instruction's index.
        index: usize,
        /// Why.
        error: DecodeError,
    },
    /// The program does not translate.
    #[error(transparent)]
    Translate(#[from] TranslateError),
    /// The IR does not lower.
    #[error(transparent)]
    Lower(#[from] LowerError),
    /// The program's questions about engine state were not settled (spec, "Translation
    /// inputs: questions and answers").
    #[error(transparent)]
    Questions(#[from] ExchangeError),
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

/// Splits `code` into instructions and decodes each one.
pub fn decode_program(code: &[u8]) -> Result<Vec<Instr>, Rejection> {
    Instruction::from_program(code)?
        .into_iter()
        .enumerate()
        .map(|(index, word)| decode(word).map_err(|error| Rejection::Decode { index, error }))
        .collect()
}

/// Decodes and translates `code` to IR.
pub fn program_ir(stage: Stage, code: &[u8]) -> Result<Function, Rejection> {
    Ok(translate(stage, &decode_program(code)?)?)
}

/// Translates a straight-line program to a host shader (the lowering handles slice 1's
/// instructions; the translator, more).
pub fn translate_program(stage: Stage, code: &[u8]) -> Result<HostShader, Rejection> {
    Ok(lower(stage.stage, &program_ir(stage, code)?)?)
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
/// here as [`InfoError::ComputeHeader`]); decoding the instructions after the header; the
/// translator, given the header's colour outputs ([`translation_stage`]) and, for a pixel
/// program, its input map ([`translate_with_header`]); the slice 8 passes ([`optimize`],
/// [`Rejection::Pass`]); and collect info ([`collect`], [`Rejection::Info`]).
pub fn graphics_program_ir(slot: GuestStage, program: &[u8]) -> Result<GraphicsIr, Rejection> {
    let (header, code) = ProgramHeader::from_program(program)?;
    check_header(slot, Some(&header))?;
    let stage = translation_stage(slot, &header);
    let instrs = decode_program(code)?;
    let mut function = match &header.io {
        Io::Pixel(io) => translate_with_header(stage, io, &instrs)?,
        Io::Vtg(_) => translate(stage, &instrs)?,
    };
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
        let mut unknown = exit_bytes();
        unknown.extend_from_slice(&[0; Instruction::SIZE]);
        assert_eq!(
            translate_program(VERTEX, &unknown).err(),
            Some(Rejection::Decode {
                index: 1,
                error: DecodeError::UnknownOpcode { opcode: 0 }
            })
        );
        assert_eq!(
            translate_program(VERTEX, &unknown)
                .err()
                .map(|error| error.to_string()),
            Some("instruction 1: unknown opcode 0x000".into())
        );
        assert_eq!(
            translate_program(VERTEX, &[]).err(),
            Some(Rejection::Translate(TranslateError::NoExit))
        );
        let ir = program_ir(VERTEX, &exit_bytes()).expect("translates");
        assert_eq!(ir.to_string(), "block0:\n    return\n");
        assert!(translate_program(VERTEX, &exit_bytes()).is_ok());
    }
}

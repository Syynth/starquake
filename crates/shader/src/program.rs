//! The walking skeleton end to end: program bytes to a validated naga module and its tables.
//!
//! Bytes are split into instructions, each instruction is decoded, the program is translated
//! to IR and the IR is lowered and validated. The first failure is the rejection; there is no
//! partial translation (spec, "Instruction translation").

use thiserror::Error;

use crate::decode::{DecodeError, Instr, Instruction, LengthError, decode};
use crate::host::HostShader;
use crate::ir::Function;
use crate::lower::{LowerError, lower};
use crate::translate::{Stage, TranslateError, translate};

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

/// Translates a straight-line program of the six slice-1 instructions to a host shader.
pub fn translate_program(stage: Stage, code: &[u8]) -> Result<HostShader, Rejection> {
    Ok(lower(stage.stage, &program_ir(stage, code)?)?)
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
        assert_eq!(ir.to_string(), "block0:\n");
        assert!(translate_program(VERTEX, &exit_bytes()).is_ok());
    }
}

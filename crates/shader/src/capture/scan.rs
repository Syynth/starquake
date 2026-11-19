//! The extent rules, one instruction at a time: a port of PR #4588's
//! `ProgramCaptureState::Observe` (`eden-pr4588/src/shader_recompiler/frontend/ampere/
//! program_capture.h` @ e5fb3b97; GPL-3.0-or-later, Copyright 2026 Eden Emulator Project).
//!
//! The state is three values, so each step is constant time and the scan is linear in the
//! program's length (row 121).

use super::layout::{CALL_OPCODE, always_runs, branch_displacement, is_final_exit, is_final_ret};
use super::{CaptureError, MAX_INSTRUCTIONS};
use crate::decode::Instruction;

/// What one instruction tells the scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// The program goes on (`Result::Continue`).
    Continue,
    /// This instruction ends the program (`Result::Complete`).
    Complete,
}

/// The scan's state (`ProgramCaptureState`'s three members).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Scan {
    /// The furthest target of an unconditional `CALL`, as an instruction index
    /// (`last_call_target`). Zero means no call: a call to instruction 0 is not counted, as in
    /// the reference (`if (!last_call_target) return Result::Complete`).
    last_call_target: u32,
    /// The furthest target of any other relative branch (`last_forward_target`).
    last_forward_target: u32,
    /// Whether a final `EXIT` was seen while a call was pending (`main_exited`).
    main_exited: bool,
}

/// The size of one instruction in bytes, as a signed number for target arithmetic.
const INSTRUCTION_BYTES: i64 = 16;

/// The instruction index a branch at `index` with byte displacement `displacement` reaches, or
/// why it reaches none (`Observe`: `target = index * 16 + 16 + relative`, refused when
/// `target < 0 || target % 16 || target >= MaximumInstructions * 16`, checked in that order).
pub(crate) fn branch_target(index: u32, displacement: i64) -> Result<u32, CaptureError> {
    let outside = |target| CaptureError::TargetOutside { index, target };
    let target = i64::from(index)
        .checked_add(1)
        .and_then(|next| next.checked_mul(INSTRUCTION_BYTES))
        .and_then(|end| end.checked_add(displacement))
        // Cannot happen: the displacement is at most 50 bits.
        .ok_or(outside(i64::MAX))?;
    if target < 0 {
        return Err(outside(target));
    }
    if target % INSTRUCTION_BYTES != 0 {
        return Err(CaptureError::MisalignedTarget { index, target });
    }
    let target_index = target / INSTRUCTION_BYTES;
    if target_index >= i64::from(MAX_INSTRUCTIONS) {
        return Err(outside(target));
    }
    // Cannot fail: the index is below the cap.
    u32::try_from(target_index).map_err(|_| outside(target))
}

impl Scan {
    /// Reads the instruction at `index` (`Observe`). A refused branch target is an error and
    /// leaves the state as it was.
    pub(crate) fn observe(&mut self, word: Instruction, index: u32) -> Result<Step, CaptureError> {
        if let Some(displacement) = branch_displacement(word) {
            let target = branch_target(index, displacement)?;
            // A CALL counts as a call only when its guard is exactly `PT`; any other (the
            // predicated `CALL.REL.NOINC` loop exits of row 67 among them) is a forward branch.
            if super::layout::opcode(word) == CALL_OPCODE && always_runs(word) {
                self.last_call_target = self.last_call_target.max(target);
            } else {
                self.last_forward_target = self.last_forward_target.max(target);
            }
        }
        if is_final_exit(word) && index >= self.last_forward_target {
            if self.last_call_target == 0 {
                return Ok(Step::Complete);
            }
            // Library functions may EXIT (row 88): with a call pending, the capture goes on
            // to a RET past every call target.
            self.main_exited = true;
        }
        if is_final_ret(word)
            && self.main_exited
            && index >= self.last_call_target
            && index >= self.last_forward_target
        {
            return Ok(Step::Complete);
        }
        Ok(Step::Continue)
    }
}

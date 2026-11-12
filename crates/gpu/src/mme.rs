//! Turing MME64 macro processor: the GPU-side programs NVN2 uses to drive 3D and compute state.
//!
//! - [`inst`] owns the 96-bit instruction format: [`Instruction`], [`decode_program`] and
//!   [`encode_program`].
//! - `display` gives instructions a textual form (`Display`) for traces.
//! - `interp` owns the interpreter: [`run`] executes a decoded program against a [`MacroHost`]
//!   (parameters, 3D register reads, method calls, the load barrier) and a per-channel
//!   [`DataRam`], bounded by a [`StepLimit`].
//! - [`memo`] owns the memo in front of the interpreter: [`MacroMemo::call`] replays a repeated
//!   call's recorded effects when the data RAM words it read still hold, and interprets it
//!   otherwise.
//!
//! Not here yet: a JIT, the wiring into the 3D and compute engines,
//! the older Fermi macro processor, and MME DMA reads (see [`MacroHost::barrier`] for the hook).

mod display;
pub mod inst;
pub mod interp;
pub mod memo;

#[cfg(test)]
mod testing;
#[cfg(test)]
mod tests;

pub use inst::{
    Alu, AluOp, DecodeError, EncodeError, GPR_COUNT, INSTRUCTION_WORDS, Instruction, Out, OutOp,
    Pred, Reg, decode_program, encode_program,
};
pub use interp::{
    DATA_RAM_WORDS, DataRam, LOAD_BARRIER, METHOD_MASK, MacroError, MacroHost, StepLimit, run,
};
pub use memo::{CallOutcome, MAX_ARENA_WORDS, MacroMemo, MemoStatistics, NotMemoised};

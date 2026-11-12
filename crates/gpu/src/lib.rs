//! GPU command processing.
//!
//! Split into two threads from the start (Eden-NX2 measured 1.46x draw throughput from this):
//! the *front* parses GPFIFO/pushbuffers, waits on semaphore acquires, runs Turing MME64 macros
//! (interpreter, memo of pure calls, later a JIT) and drops same-value register writes; the
//! *back* owns engine state (3D, compute/QMD, DMA, inline-to-memory) and records draws into the
//! render backend.

pub mod field;
pub mod fifo;
pub mod mme;
pub mod ring;
pub mod sema;

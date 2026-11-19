//! The GPU's back thread (gpu-protocol spec, "The back thread: engine state"): what it does
//! with the ring entries the front sends it.
//!
//! Each engine's part lives in its own submodule. So far:
//!
//! - [`qmd`]: compute dispatch through QMD 3.0 (slice G12): decoding a QMD, the PCAS scheduler
//!   and its one QMD table for all channels, the inline QMD, and the releases a QMD makes.
//! - [`state`]: the engine mirrors, their dirty state groups and typed views (slice G10).
//!
//! Later slices add copies (G11), fences and completion (G15), draws (G18) and the executor
//! that feeds them all from the ring (G19).

pub mod qmd;
pub mod state;

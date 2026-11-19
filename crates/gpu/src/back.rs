//! The GPU's back thread (gpu-protocol spec, "The back thread: engine state"): what it does
//! with the ring entries the front sends it.
//!
//! Each engine's part lives in its own submodule:
//!
//! - [`copy`]: DMA copies (`C7B5`), inline-to-memory (`A140`, and the 3D and compute classes'
//!   inline uploads) and 2D blits (`902D`) through guest memory, on the CPU (slice G11).
//! - [`qmd`]: compute dispatch through QMD 3.0 (slice G12): decoding a QMD, the PCAS scheduler
//!   and its one QMD table for all channels, the inline QMD, and the releases a QMD makes.
//! - [`state`]: the engine mirrors, their dirty state groups and typed views (slice G10).
//! - [`buffer`]: the buffer cache (slice G16): guest ranges to host buffers, kept coherent with
//!   CPU writes through the write watch, dropped by ordered unmaps, evicted by the budget.
//! - [`fence`]: fences and completion (slice G15): the open command list, fenced releases,
//!   increments and markers performed after host completion, the acquire waiter table they
//!   latch, and the back thread's wait on its ring and on completion.
//! - [`texture`]: the texture cache (slice G17): texture headers decoded, guest images to host
//!   textures uploaded through the layout crate, render targets, write-back on demand, kept
//!   coherent with CPU writes, dropped by ordered unmaps, evicted by the budget.
//! - [`readback`]: the readback ids the caches' write-backs share.
//! - [`pipeline`]: the pipeline-key cache (slice G23): pipeline state to host pipelines, keyed
//!   only on what the device does not set per draw, and the list of keys seen for precompiling
//!   on the next boot.
//!
//! Later slices add draws (G18) and the executor that feeds them all from the ring (G19).

pub mod buffer;
pub mod copy;
pub mod fence;
pub mod pipeline;
pub mod qmd;
pub mod readback;
pub mod state;
pub mod texture;

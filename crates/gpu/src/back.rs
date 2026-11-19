//! The GPU's back thread (gpu-protocol spec, "The back thread: engine state"): what it does
//! with the ring entries the front sends it.
//!
//! Each engine's part lives in its own submodule:
//!
//! - [`copy`]: DMA copies (`C7B5`), inline-to-memory (`A140`, and the 3D and compute classes'
//!   inline uploads) and 2D blits (`902D`) through guest memory, on the CPU (slice G11).
//! - [`qmd`]: compute dispatch through QMD 3.0 (slice G12): decoding a QMD, the PCAS scheduler
//!   and its one QMD table for all channels, the inline QMD, and the releases a QMD makes.
//!
//! - [`buffer`]: the buffer cache (slice G16): guest ranges to host buffers, kept coherent with
//!   CPU writes through the write watch, dropped by ordered unmaps, evicted by the budget.
//!
//! Later slices add the engine mirrors and dirty groups (G10), fences and
//! completion (G15), draws (G18) and the executor that feeds them all from the ring (G19).

pub mod buffer;
pub mod copy;
pub mod qmd;

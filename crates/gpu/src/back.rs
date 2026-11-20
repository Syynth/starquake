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
//! - [`draw`]: draws (slice G18): 3D state to a `Command::Draw`, with shader lookups, the
//!   variant key, bindings through the caches, driver constants and conditional render; and
//!   clears (slice G24, [`draw::clear`]): `CLEAR_SURFACE` to a `Command::Clear`, or a draw for a
//!   masked clear.
//!
//! - [`exec`]: the executor (slice G19): ring entries to the engines above, compute dispatches
//!   to `Command::Dispatch`, presents to `Command::Present`, and the commands to the fence path.
//!
//! - [`cb_load`]: constant-buffer loads (slice G25): `LOAD_CONSTANT_BUFFER` words written to
//!   guest memory at the selected buffer's address plus the upload offset.
//!
//! - [`report`]: the 3D class's report semaphore: a `SET_REPORT_SEMAPHORE_D` write that reports
//!   no counter releases its payload, as a fenced release.

pub mod buffer;
pub mod cb_load;
pub mod copy;
pub mod draw;
pub mod exec;
pub mod fence;
pub mod pipeline;
pub mod qmd;
pub mod readback;
pub mod report;
pub mod state;
pub mod texture;

//! System-call numbers, their register convention, and one status per call (implemented, planned,
//! deferred, not planned), plus the system calls themselves.
//!
//! A system call is a plain function whose parameters are the tables and services it touches, and
//! nothing else (kernel spec, "Tables behind a dispatcher"):
//!
//! | Call | Function | Parameters |
//! |---|---|---|
//! | `GetThreadPriority` (0x0c) | [`thread::get_thread_priority`] | the scheduler and the handle table, both shared |
//! | `GetCurrentProcessorNumber` (0x10) | [`processor::get_current_processor_number`] | the calling core |
//! | `CloseHandle` (0x16) | [`handle::close_handle`] | the handle table and the scheduler, both exclusive |
//! | `GetSystemTick` (0x1e) | [`tick::get_system_tick`] | the dispatch's guest time |
//! | `OutputDebugString` (0x27) | [`debug::output_debug_string`] | guest memory, a [`DebugSink`] and the dispatch's guest time |
//!
//! Only the [dispatcher](crate::dispatch) calls them, decoding their arguments with the call's
//! [`Convention`]. No system call reads the clock: the dispatcher reads it once per dispatch and
//! passes the guest time to the calls that need it. `ExitProcess` (0x07) is implemented too,
//! provisionally as a dispatcher arm with no function behind it (see the dispatcher's module
//! docs).

mod convention;
pub mod debug;
pub mod handle;
mod ids;
pub mod processor;
pub mod thread;
pub mod tick;

pub use convention::{Arg, CONVENTIONS, Convention, ConventionError, Kind, is_retail};
pub use debug::DebugSink;
pub use ids::{Status, SvcId};

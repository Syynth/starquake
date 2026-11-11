//! System-call numbers, their register convention, and one status per call (implemented, deferred,
//! not planned), plus the system calls themselves.
//!
//! A system call is a plain function whose parameters are the tables and services it touches, and
//! nothing else (kernel spec, "Tables behind a dispatcher"). The calls here touch no table:
//!
//! | Call | Function | Parameters |
//! |---|---|---|
//! | `GetCurrentProcessorNumber` (0x10) | [`processor::get_current_processor_number`] | the calling core |
//! | `GetSystemTick` (0x1e) | [`tick::get_system_tick`] | the [`Env`] |
//! | `OutputDebugString` (0x27) | [`debug::output_debug_string`] | guest memory and a [`DebugSink`] |
//!
//! Only the [dispatcher](crate::dispatch) calls them, decoding their arguments with the call's
//! [`Convention`].

mod convention;
pub mod debug;
mod env;
mod ids;
pub mod processor;
pub mod tick;

pub use convention::{
    Arg, CONVENTIONS, Convention, ConventionError, Kind, SVC_ARGUMENT_REGISTERS, is_retail,
};
pub use debug::DebugSink;
pub use env::Env;
pub use ids::{Status, SvcId};

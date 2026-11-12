//! System-call numbers, their register convention, and one status per call (implemented, planned,
//! deferred, not planned), plus the system calls themselves.
//!
//! A system call is a plain function whose parameters are the tables and services it touches, and
//! nothing else (kernel spec, "Tables behind a dispatcher"):
//!
//! | Call | Function | Parameters |
//! |---|---|---|
//! | `SetHeapSize` (0x01) | [`memory::set_heap_size`] | the address space, exclusive |
//! | `SetMemoryPermission` (0x02) | [`memory::set_memory_permission`] | the address space, exclusive |
//! | `SetMemoryAttribute` (0x03) | [`memory::set_memory_attribute`] | the address space, exclusive |
//! | `QueryMemory` (0x06) | [`memory::query_memory`] | the address space, shared, and guest memory |
//! | `ExitProcess` (0x07) | [`process::exit_process`] | the handle table and the scheduler, both exclusive, and the process facts |
//! | `CreateThread` (0x08) | [`thread::create_thread`] | the handle table, the address space and the scheduler, all exclusive, the process facts, guest memory and the calling core |
//! | `StartThread` (0x09) | [`thread::start_thread`] | the handle table, shared, the scheduler, exclusive, and the calling core |
//! | `ExitThread` (0x0a) | [`thread::exit_thread`] | the handle table and the scheduler, both exclusive, and the calling core |
//! | `SleepThread` (0x0b) | [`thread::sleep_thread`] | the scheduler, exclusive, and the calling core |
//! | `GetThreadPriority` (0x0c) | [`thread::get_thread_priority`] | the scheduler and the handle table, both shared |
//! | `SetThreadPriority` (0x0d) | [`thread::set_thread_priority`] | the handle table, shared, the scheduler, exclusive, the process facts and the calling core |
//! | `GetThreadCoreMask` (0x0e) | [`thread::get_thread_core_mask`] | the scheduler and the handle table, both shared, and the calling core |
//! | `SetThreadCoreMask` (0x0f) | [`thread::set_thread_core_mask`] | the handle table, shared, the scheduler, exclusive, the process facts and the calling core |
//! | `GetCurrentProcessorNumber` (0x10) | [`processor::get_current_processor_number`] | the calling core |
//! | `SignalEvent` (0x11) | [`event::signal_event`] | the handle table, shared, and the scheduler |
//! | `ClearEvent` (0x12) | [`event::clear_event`] | the handle table, shared, and the scheduler |
//! | `CloseHandle` (0x16) | [`handle::close_handle`] | the handle table and the scheduler, both exclusive |
//! | `ResetSignal` (0x17) | [`event::reset_signal`] | the handle table, shared, and the scheduler |
//! | `WaitSynchronization` (0x18) | [`sync::wait_synchronization`] | the handle table and the address space, shared, the scheduler, guest memory, and the dispatch's clock reading |
//! | `CancelSynchronization` (0x19) | [`sync::cancel_synchronization`] | the handle table, shared, the scheduler and the calling core |
//! | `ArbitrateLock` (0x1a) | [`arbitration::arbitrate_lock`] | the handle table and the address space, shared, the scheduler, guest memory and the calling thread |
//! | `ArbitrateUnlock` (0x1b) | [`arbitration::arbitrate_unlock`] | the address space, shared, the scheduler, guest memory and the calling core |
//! | `WaitProcessWideKeyAtomic` (0x1c) | [`arbitration::wait_process_wide_key_atomic`] | the address space, shared, the scheduler, guest memory, the calling thread and the dispatch's clock reading |
//! | `SignalProcessWideKey` (0x1d) | [`arbitration::signal_process_wide_key`] | the handle table and the address space, shared, the scheduler, guest memory and the calling core |
//! | `GetSystemTick` (0x1e) | [`tick::get_system_tick`] | the dispatch's guest time |
//! | `GetProcessId` (0x24) | [`process::get_process_id`] | the handle table and the scheduler, both shared, and the process facts |
//! | `GetThreadId` (0x25) | [`thread::get_thread_id`] | the scheduler and the handle table, both shared, and the calling core |
//! | `OutputDebugString` (0x27) | [`debug::output_debug_string`] | guest memory, a [`DebugSink`] and the dispatch's guest time |
//! | `GetInfo` (0x29) | [`info::get_info`] | the handle table, the address space and the scheduler, all shared |
//! | `WaitForAddress` (0x34) | [`arbitration::wait_for_address`] | the address space, shared, the scheduler, guest memory, the calling thread and the dispatch's clock reading |
//! | `SignalToAddress` (0x35) | [`arbitration::signal_to_address`] | the address space, shared, the scheduler, guest memory and the calling core |
//! | `CreateEvent` (0x45) | [`event::create_event`] | the handle table and the scheduler, both exclusive |
//!
//! Only the [dispatcher](crate::dispatch) calls them, decoding their arguments with the call's
//! [`Convention`]. No system call reads the clock: the dispatcher reads it once per dispatch and
//! passes the guest time to the calls that need it. The process facts
//! ([`crate::tables::Processes`]) are no declared table: they never change, and the dispatcher
//! hands the calls that check them a copy. `ConnectToNamedPort` (0x1f) and `SendSyncRequest`
//! (0x21) are implemented too: they are dispatcher
//! arms over the IPC table ([`crate::ipc`]), because `SendSyncRequest` releases the kernel lock
//! while a server's handler runs (`dispatch::ipc` says how).

pub mod arbitration;
mod convention;
pub mod debug;
pub mod event;
pub mod handle;
mod ids;
pub mod info;
pub mod memory;
pub mod process;
pub mod processor;
pub mod sync;
pub mod thread;
pub mod tick;

pub use convention::{
    Arg, BEYOND_RETAIL, CONVENTIONS, Convention, ConventionError, Kind, is_retail,
};
pub use debug::DebugSink;
pub use ids::{Status, SvcId};

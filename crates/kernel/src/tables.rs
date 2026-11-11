//! The kernel's tables. Each system call declares which of these it touches; nothing else can
//! reach kernel state. The ones not yet built are placeholders until their port slices land.

/// Threads, per-core run queues, wait lists, events, sessions, the address arbiter and timers.
/// Threads and run queues are built ([`crate::sched`]); the rest arrive with their slices.
pub use crate::sched::Scheduler;

/// One process's memory blocks, regions and heap.
#[derive(Debug, Default)]
pub struct AddressSpace {
    _private: (),
}

/// Per-process facts fixed at creation (capabilities, ids).
#[derive(Debug, Default)]
pub struct Processes {
    _private: (),
}

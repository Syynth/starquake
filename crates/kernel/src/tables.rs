//! The kernel's tables. Each system call declares which of these it touches; nothing else can
//! reach kernel state. Placeholders until their port slices land.

/// Threads, per-core run queues, wait lists, events, sessions, the address arbiter and timers.
#[derive(Debug, Default)]
pub struct Scheduler {
    _private: (),
}

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

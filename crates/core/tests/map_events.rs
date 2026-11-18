//! The address space's `Map` recording point in a real run (observability spec, "The event
//! stream": the address space's mutators each call the recorder once): the run attaches its
//! recorder to the address space before the kernel boots the process, so the mapping the kernel
//! makes for the main thread's thread-local page reaches the run's event dump (`events.bin`),
//! with the guest's read-write permission word, and is left out of the golden trace.
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM fails
//! the test.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use starquake_core::ExitReason;
use starquake_memory::PAGE_SIZE;
use starquake_trace::{Dump, EventKind, MapKind};

#[cfg(test)]
mod common;

use common::{GENEROUS, run};

const HELLO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nro");

/// The guest's raw permission word for read and write: `Svc::MemoryPermission::ReadWrite`,
/// `Read | Write` (`eden/src/core/hle/kernel/svc_types.h` @ cf654052).
const READ_WRITE: u32 = 0b11;

#[test]
fn the_thread_local_page_the_kernel_maps_at_boot_is_a_map_event_in_the_dump() {
    let Some(report) = run(HELLO, GENEROUS) else {
        return;
    };
    assert_eq!(report.exit, ExitReason::ProcessExited, "{}", report.trace);
    let tls = report
        .memory_map
        .iter()
        .find(|mapping| mapping.name == "tls")
        .expect("the run's memory map names the thread-local page");
    let maps: Vec<EventKind> = Dump::from_bytes(&report.events)
        .expect("a readable dump")
        .merged()
        .iter()
        .map(|event| event.kind)
        .filter(|kind| matches!(kind, EventKind::Map { .. }))
        .collect();
    let expected = EventKind::Map {
        va: tls.start,
        size: PAGE_SIZE,
        perms: READ_WRITE,
        kind: MapKind::Map,
    };
    assert!(maps.contains(&expected), "{maps:?}");
    // Map events depend on timing, so the golden trace leaves them out.
    assert!(!report.trace.contains("map Map"), "{}", report.trace);
}

//! An ExeFS run end to end: the hello fixture's NSO and `main.npdm`
//! (`tests/homebrew/hello/`), loaded as a title's modules and metadata, run natively under the
//! hypervisor backend.
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM fails
//! the test.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::time::Duration;

use starquake_core::loader::exefs::{ExeFs, MAIN_MODULE, NPDM_FILE};
use starquake_core::{ExitReason, Mapping, RunOptions, RunReport, System, exit_code};

const NSO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nso");
const NRO: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nro");
const NPDM: &[u8] = include_bytes!("../../../tests/homebrew/hello/main.npdm");

/// Long enough for the fixture on a loaded machine; it finishes in milliseconds.
const GENEROUS: Duration = Duration::from_secs(30);

/// Where the first module goes: Eden's address for 39-bit code (`crates/core/src/image.rs`).
const CODE_ADDRESS: u64 = 0x8000_0000;

/// Runs `system`, or returns `None` on hosted CI (see the module docs).
fn run(system: &System) -> Option<RunReport> {
    if std::env::var_os("CI").is_some() {
        return None;
    }
    Some(system.run(&RunOptions {
        timeout: Some(GENEROUS),
        command_line: Vec::new(),
    }))
}

/// The ExeFS of the fixture's metadata and `modules`, each the fixture's NSO.
fn exefs(modules: &[&'static str]) -> System {
    let mut files = vec![(NPDM_FILE, NPDM)];
    files.extend(modules.iter().map(|name| (*name, NSO)));
    System::from_exefs(ExeFs::parse(&files).expect("the fixture's ExeFS parses"))
        .expect("the fixture's ExeFS loads")
}

/// The mapping called `name`.
fn mapping<'a>(report: &'a RunReport, name: &str) -> &'a Mapping {
    report
        .memory_map
        .iter()
        .find(|mapping| mapping.name == name)
        .unwrap_or_else(|| panic!("no {name} in {:?}", report.memory_map))
}

#[test]
fn the_hello_exefs_runs_to_exit_process_as_its_nro_twin_does() {
    let Some(report) = run(&exefs(&[MAIN_MODULE])) else {
        return;
    };
    assert_eq!(report.exit, ExitReason::ProcessExited, "{}", report.trace);
    assert_eq!(report.exit_code(), exit_code::STOP_CONDITION);
    // The same program as the NRO: the same lines, in the same order.
    let twin = run(&System::new(NRO).unwrap()).unwrap();
    assert_eq!(twin.exit, ExitReason::ProcessExited);
    assert_eq!(report.guest_log, twin.guest_log);
    assert_eq!(report.guest_log.len(), 3);
    // The module is `main`, at the code address, and the stack is the metadata's size.
    assert_eq!(mapping(&report, "main.text").start, CODE_ADDRESS);
    let stack = mapping(&report, "stack");
    let npdm = starquake_core::loader::npdm::Npdm::parse(NPDM).unwrap();
    assert_eq!(
        stack.end - stack.start,
        u64::from(npdm.main_thread_stack_size)
    );
}

#[test]
fn a_set_of_two_modules_runs_its_first_with_the_second_placed_after_it() {
    // `rtld` comes first and is the entry point; here it is the hello program, so the run is
    // its run, with `main` mapped and patched after it.
    let Some(report) = run(&exefs(&[MAIN_MODULE, "rtld"])) else {
        return;
    };
    assert_eq!(report.exit, ExitReason::ProcessExited, "{}", report.trace);
    assert_eq!(report.guest_log.len(), 3);
    let rtld = mapping(&report, "rtld.text");
    let main = mapping(&report, "main.text");
    assert_eq!(rtld.start, CODE_ADDRESS);
    let module = report_module_size(&report, "rtld");
    assert_eq!(main.start, CODE_ADDRESS + module);
    // The first module's data ends before the second starts.
    let rtld_end = mapping(&report, "rtld.data").end;
    assert!(rtld_end <= main.start, "{:?}", report.memory_map);
}

/// The size of module `name`: from its first mapping to the next module's or the end of its
/// last mapping, page-aligned.
fn report_module_size(report: &RunReport, name: &str) -> u64 {
    let prefix = format!("{name}.");
    let mine: Vec<_> = report
        .memory_map
        .iter()
        .filter(|mapping| mapping.name.starts_with(&prefix))
        .collect();
    let start = mine.iter().map(|m| m.start).min().unwrap();
    let end = mine.iter().map(|m| m.end).max().unwrap();
    (end - start).next_multiple_of(starquake_memory::PAGE_SIZE)
}

//! Hosts with no CPU backend: a program still loads, and running it is an emulator error that
//! says why. (On macOS on Apple Silicon the same programs run; see `skeleton.rs`.)

#![cfg(not(all(target_os = "macos", target_arch = "aarch64")))]

use starquake_core::{Error, ExitReason, RunOptions, System, exit_code};
use starquake_testkit::programs;
use starquake_trace::Dump;

#[test]
fn running_a_program_reports_that_no_cpu_backend_is_available() {
    let system = System::new(&programs::hello().unwrap()).unwrap();
    let report = system.run(&RunOptions::default());
    assert_eq!(
        report.exit,
        ExitReason::EmulatorError(Error::NoCpuBackend.to_string())
    );
    assert_eq!(report.exit_code(), exit_code::EMULATOR_ERROR);
    // Nothing ran: no output, no trace, and a flight-recorder dump with no cores.
    assert_eq!(report.guest_log, Vec::<Vec<u8>>::new());
    assert_eq!(report.trace, "");
    assert_eq!(Dump::from_bytes(&report.events), Ok(Dump::new(Vec::new())));
    assert_eq!(report.summary().exit_reason, "emulator_error");
}

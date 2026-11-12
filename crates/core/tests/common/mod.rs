//! Shared by this crate's guest-program tests: running an NRO on the hypervisor backend.

use std::time::Duration;

use starquake_core::{RunOptions, RunReport, System};

/// Long enough for any of the test programs on a loaded machine; they finish in milliseconds.
pub const GENEROUS: Duration = Duration::from_secs(30);

/// Runs an NRO to completion, or returns `None` on hosted CI: these tests need a Mac with
/// Apple Silicon, and skip themselves only when the `CI` environment variable is set (hosted
/// runners can't create VMs). Anywhere else, failing to create a VM fails the test.
pub fn run(file: &[u8], timeout: Duration) -> Option<RunReport> {
    if std::env::var_os("CI").is_some() {
        return None;
    }
    let options = RunOptions {
        timeout: Some(timeout),
        command_line: Vec::new(),
        sd_card: None,
    };
    match System::new(file) {
        Ok(system) => Some(system.run(&options)),
        Err(error) => unreachable!("the test program does not load: {error}"),
    }
}

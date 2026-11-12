//! Shared by this crate's guest-program tests: running an NRO on the hypervisor backend.
//!
//! Included with `#[cfg(test)] mod common;`, so that it may unwrap like the tests do.

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
    };
    Some(System::new(file).expect("load the program").run(&options))
}

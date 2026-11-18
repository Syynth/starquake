//! Starquake command-line frontend. The command line itself is in the library (`lib.rs`), which
//! writes to whatever streams it is given; this hands it the process's own.

use std::ffi::OsString;
use std::io::Write as _;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().collect();
    let mut stderr = std::io::stderr();
    // Before anything logs, so a run's `log.txt` holds every warning.
    if let Err(error) = starquake::install_host_log() {
        // A stream that can't be written to must not stop the run.
        let _ = writeln!(stderr, "starquake: the emulator's log is not kept: {error}");
    }
    let code = starquake::execute(&args, &mut std::io::stdout(), &mut stderr);
    ExitCode::from(code)
}

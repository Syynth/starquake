//! Starquake command-line frontend. The command line itself is in the library (`lib.rs`), which
//! writes to whatever streams it is given; this hands it the process's own.

use std::ffi::OsString;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().collect();
    let code = starquake::execute(&args, &mut std::io::stdout(), &mut std::io::stderr());
    ExitCode::from(code)
}

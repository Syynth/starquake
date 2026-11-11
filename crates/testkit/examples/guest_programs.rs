//! Writes the testkit's guest programs to a folder as NRO files, so they can be handed to
//! `starquake run` by hand:
//!
//! ```sh
//! cargo run -p starquake-testkit --example guest_programs -- <folder> [counter-hz]
//! cargo run -p starquake -- run <folder>/hello.nro --artifacts <folder>/hello-run
//! ```
//!
//! `counter-hz` is the rate the counter program expects `cntfrq_el0` to read. It defaults to
//! 19.2 MHz, the rate the kernel gives a process today (`CNTFREQ` in
//! `eden/src/core/hardware_properties.h`).

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a command-line tool: it reports what it wrote on stdout and errors on stderr"
)]

use std::path::Path;
use std::process::ExitCode;

use starquake_testkit::programs;

/// The counter rate used when none is given (see the module docs).
const DEFAULT_COUNTER_HZ: u64 = 19_200_000;

fn write_all(folder: &Path, counter_hz: u64) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(folder)?;
    let files = [
        ("hello.nro", programs::hello()?),
        ("counter.nro", programs::counter(counter_hz)?.nro),
        ("crash.nro", programs::crash()?.nro),
        ("loop.nro", programs::endless_loop()?),
    ];
    for (name, bytes) in files {
        let path = folder.join(name);
        std::fs::write(&path, bytes)?;
        println!("{}", path.display());
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let (Some(folder), counter_hz, None) = (args.next(), args.next(), args.next()) else {
        eprintln!("usage: guest_programs <folder> [counter-hz]");
        return ExitCode::FAILURE;
    };
    let counter_hz = match counter_hz.map(|text| text.to_str().and_then(|text| text.parse().ok())) {
        None => DEFAULT_COUNTER_HZ,
        Some(Some(hz)) => hz,
        Some(None) => {
            eprintln!("guest_programs: counter-hz must be a whole number of hertz");
            return ExitCode::FAILURE;
        }
    };
    match write_all(Path::new(&folder), counter_hz) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("guest_programs: {error}");
            ExitCode::FAILURE
        }
    }
}

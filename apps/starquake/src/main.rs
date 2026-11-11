//! Starquake command-line frontend.

#![expect(
    clippy::print_stdout,
    reason = "a command-line tool: its output goes to stdout"
)]

fn main() {
    println!(
        "starquake {}: nothing to run yet",
        env!("CARGO_PKG_VERSION")
    );
}

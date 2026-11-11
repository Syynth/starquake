//! hvprobe: measures the Hypervisor.framework behaviour a native-execution backend depends on.
//! See README.md for the results and what each test answers.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod guest;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod hv;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod probe;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    probe::main();
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("hvprobe needs macOS on Apple Silicon (it drives Hypervisor.framework)");
    std::process::exit(1);
}

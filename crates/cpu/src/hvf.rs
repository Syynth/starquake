//! Hypervisor.framework backend (macOS arm64).
//!
//! Not implemented yet. `tools/hvprobe` has the working pieces: VM and vCPU setup, 4 KiB stage-1
//! tables, an EL1 vector table that forwards `svc` with `hvc`, and per-process PAuth keys.

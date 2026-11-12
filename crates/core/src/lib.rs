//! Wires the emulator together.
//!
//! Owns the system object (kernel, services, graphics server, GPU, render backend), timing, and
//! configuration. Frontends talk to this crate only.
//!
//! # The walking skeleton (provisional)
//!
//! What exists today is the thinnest end-to-end run (`docs/roadmap.md`, "Route to milestone 1"):
//! [`System`] takes the bytes of an NRO, parses it with the loader, creates the process's
//! memory with the kernel's address space (its code, a stack and a TLS page; text read-execute,
//! rodata read-only, the rest read-write), patches its counter reads, and runs it as one guest
//! thread on one vCPU of the hypervisor backend. Every `svc` goes through the kernel's dispatcher
//! and is recorded in the event stream. The run ends when the guest calls `ExitProcess`, when it
//! crashes, or at a timeout, and leaves a [`RunReport`] that [`artifacts::write`] turns into the
//! artifacts folder of the headless contract (`docs/specs/observability.md`).
//!
//! It is deliberately thinner than the specs. The provisional parts, and the work that replaces
//! each, are listed where they live:
//!
//! - `image`: loading a program, which creates the process's memory with the kernel: the code
//!   at a fixed address, one stack, one TLS page (replaced by kernel slice 8, process
//!   creation).
//! - `space`: the address space as the host keeps it: stage-1 tables driven by the kernel's
//!   address space, pages backed on first touch from the start of a small guest RAM (replaced
//!   by the backing pools).
//! - `machine`: one guest thread on one core with no scheduler (kernel slices 8 and 10).
//! - `clock`: every process counts time at the Switch 1 rate (kernel slice 8).
//! - [`system`]: one VM per run, hypervisor backend only.
//! - [`crash`]: the part of the crash report that needs no backtrace or symbols.

pub mod artifacts;
#[cfg_attr(
    not(any(test, all(target_os = "macos", target_arch = "aarch64"))),
    expect(
        dead_code,
        reason = "only a CPU backend gives a run its clock; hosts without one build this for its tests"
    )
)]
mod clock;
pub mod crash;
#[cfg_attr(
    not(any(test, all(target_os = "macos", target_arch = "aarch64"))),
    expect(
        dead_code,
        reason = "only a CPU backend places a program; hosts without one build this for its tests"
    )
)]
mod image;
#[cfg_attr(
    not(any(test, all(target_os = "macos", target_arch = "aarch64"))),
    expect(
        dead_code,
        reason = "only a CPU backend drives the loop; hosts without one build this for its tests"
    )
)]
mod machine;
pub mod report;
#[cfg_attr(
    not(any(test, all(target_os = "macos", target_arch = "aarch64"))),
    expect(
        dead_code,
        reason = "only a CPU backend runs guests in an address space; hosts without one build \
                  this for its tests"
    )
)]
mod space;
pub mod system;

pub use image::LoadError;
pub use report::{ExitReason, Mapping, RunOptions, RunReport, Summary, exit_code};
pub use space::SpaceError;
pub use system::{Error, System};

/// The loader's formats (NUP containers, NPDM, NSO), for frontends, which see only this crate
/// (`docs/architecture.md`, "Crates and dependency direction"): `starquake inspect` and
/// `starquake verify` read titles through it.
pub use starquake_loader as loader;

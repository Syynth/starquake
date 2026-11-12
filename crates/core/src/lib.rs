//! Wires the emulator together.
//!
//! Owns the system object (kernel, services, graphics server, GPU, render backend), timing, and
//! configuration. Frontends talk to this crate only.
//!
//! # The walking skeleton (provisional)
//!
//! What exists today is the thinnest end-to-end run (`docs/roadmap.md`, "Route to milestone 1"):
//! [`System`] takes the bytes of an NRO, parses it with the loader, places its segments, a TLS
//! block and a stack in the hypervisor backend's bring-up memory, patches its counter reads,
//! and runs it as one guest thread on one vCPU. Every `svc` goes through the kernel's dispatcher
//! and is recorded in the event stream. The run ends when the guest calls `ExitProcess`, when it
//! crashes, or at a timeout, and leaves a [`RunReport`] that [`artifacts::write`] turns into the
//! artifacts folder of the headless contract (`docs/specs/observability.md`).
//!
//! It is deliberately thinner than the specs. The provisional parts, and the work that replaces
//! each, are listed where they live:
//!
//! - `image`: the program's place in bring-up memory, with no page tables, permissions or
//!   address-space bookkeeping (replaced by stage-1 tables and kernel slice 6).
//! - `machine`: one guest thread on one core with no scheduler (kernel slices 8 and 10).
//! - [`system`]: one VM per run, hypervisor backend only.
//! - [`crash`]: the part of the crash report that needs no backtrace or symbols.

pub mod artifacts;
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
pub mod system;

pub use image::LoadError;
pub use report::{ExitReason, Mapping, RunOptions, RunReport, Summary, exit_code};
pub use system::{Error, System};

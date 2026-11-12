//! [`System`]: one program, loaded and run from start to finish.
//!
//! PROVISIONAL (walking skeleton, #74). A `System` is one NRO run once as one guest thread on
//! one vCPU of the hypervisor backend, in its bring-up memory (the `image` and `machine`
//! modules say what is provisional about each and what replaces it). It is the
//! thinnest thing that joins the loader, the code patcher, the CPU backend, the kernel
//! dispatcher and the event stream, so that later work widens something that runs.
//!
//! On a host with no CPU backend (anything but macOS on Apple Silicon, until the KVM and JIT
//! backends exist) a program still parses, and [`System::run`] reports [`Error::NoCpuBackend`]
//! as an emulator error.

use std::time::Instant;

use starquake_memory::Fault;

use crate::image::{LoadError, Program};
use crate::report::{RunOptions, RunReport};

/// Why a program could not be loaded or started.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The program is malformed or does not fit guest memory.
    #[error(transparent)]
    Load(#[from] LoadError),
    /// This host has no way to run guest code.
    #[error(
        "no CPU backend is available on this host: the hypervisor backend needs macOS on Apple \
         Silicon"
    )]
    NoCpuBackend,
    /// The CPU backend could not create its VM or vCPU.
    #[error("the CPU backend failed: {0}")]
    Cpu(#[from] starquake_cpu::Error),
    /// The program could not be copied into guest memory.
    #[error("the program could not be written into guest memory: {0}")]
    Memory(Fault),
}

/// A loaded program, ready to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct System {
    program: Program,
}

impl System {
    /// Loads `program`, the bytes of an NRO file. Works on every host.
    pub fn new(program: &[u8]) -> Result<Self, Error> {
        Ok(Self {
            program: Program::parse(program)?,
        })
    }

    /// Runs the program until it calls `ExitProcess`, crashes, or `options.timeout` passes, and
    /// reports what happened. A failure to start (no CPU backend, no VM, a program too large
    /// for guest memory) is reported the same way, as an emulator error with nothing else in
    /// the report.
    ///
    /// The vCPU is created on, and runs on, the calling thread. Hypervisor.framework allows one
    /// VM per process, so a second `run` in the same process waits for the first to finish.
    #[must_use]
    pub fn run(&self, options: &RunOptions) -> RunReport {
        let started = Instant::now();
        host::run(&self.program, options, started).unwrap_or_else(|error| {
            tracing::error!(%error, "the run could not start");
            RunReport::emulator_error(error.to_string(), started.elapsed())
        })
    }
}

/// The hypervisor backend: a fresh VM for every run.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod host {
    use std::sync::{Mutex, PoisonError};
    use std::time::Instant;

    use starquake_cpu::Backend;
    use starquake_cpu::hvf::{HvfBackend, RamConfig};
    use starquake_loader::patch::CounterRescale;

    use super::Error;
    use crate::clock::{CounterClock, PROCESS_COUNTER_RATE, Timebase};
    use crate::image::{LoadError, Program, place};
    use crate::machine;
    use crate::report::{RunOptions, RunReport};

    /// Held for the whole of a run: Hypervisor.framework allows one VM per process.
    static ONE_VM: Mutex<()> = Mutex::new(());

    pub(super) fn run(
        program: &Program,
        options: &RunOptions,
        started: Instant,
    ) -> Result<RunReport, Error> {
        let _one_vm = ONE_VM.lock().unwrap_or_else(PoisonError::into_inner);
        // PROVISIONAL: the backend's default bring-up RAM (16 MiB), identity-mapped.
        let backend = HvfBackend::new(RamConfig::default())?;
        let ram = backend.memory();
        // One rate and one zero point for guest time (kernel spec, "Guest time"): the patcher
        // rescales the backend's counter to the process's rate, the kernel converts its clock
        // with the same rate, and that clock is the backend's counter.
        let host_hz = backend.counter_frequency();
        let time = Timebase {
            rate: PROCESS_COUNTER_RATE,
            clock: CounterClock::new(host_hz).ok_or_else(|| {
                starquake_cpu::Error::Host("the host counter reports a rate of zero".into())
            })?,
        };
        let counter = CounterRescale {
            host_hz,
            guest_hz: time.rate.hz(),
        };
        let ram_size = u64::try_from(ram.size()).map_err(|_| LoadError::Overflow)?;
        let placed = place(program, ram.base(), ram_size, Some(counter))?;
        placed.write_into(ram).map_err(Error::Memory)?;
        // Declared after the backend, so it is dropped first, as the VM requires.
        let mut vcpu = backend.create_vcpu()?;
        Ok(machine::run(
            &mut vcpu, ram, &placed, options, started, time,
        ))
    }
}

/// Hosts with no CPU backend.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod host {
    use std::time::Instant;

    use super::Error;
    use crate::image::Program;
    use crate::report::{RunOptions, RunReport};

    pub(super) fn run(
        _program: &Program,
        _options: &RunOptions,
        _started: Instant,
    ) -> Result<RunReport, Error> {
        Err(Error::NoCpuBackend)
    }
}

#[cfg(test)]
mod tests {
    use starquake_loader::nro::NroError;

    use super::*;

    #[test]
    fn a_file_that_is_not_an_nro_does_not_load() {
        assert_eq!(
            System::new(&[0; 0x80]),
            Err(Error::Load(LoadError::Nro(NroError::BadMagic)))
        );
    }

    #[test]
    fn the_missing_backend_error_says_what_is_needed() {
        assert_eq!(
            Error::NoCpuBackend.to_string(),
            "no CPU backend is available on this host: the hypervisor backend needs macOS on \
             Apple Silicon"
        );
    }
}

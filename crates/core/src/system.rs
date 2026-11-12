//! [`System`]: one program, loaded and run from start to finish.
//!
//! PROVISIONAL (walking skeleton, #74). A `System` is one program (an NRO, or an ExeFS's
//! modules and metadata) run once as one guest thread on one vCPU of the hypervisor backend, in an address space of its own that the kernel keeps
//! the books of (the `image`, `space` and `machine` modules say what is provisional about each
//! and what replaces it). It is the
//! thinnest thing that joins the loader, the code patcher, the CPU backend, the kernel
//! dispatcher and the event stream, so that later work widens something that runs.
//!
//! On a host with no CPU backend (anything but macOS on Apple Silicon, until the KVM and JIT
//! backends exist) a program still parses, and [`System::run`] reports [`Error::NoCpuBackend`]
//! as an emulator error.

use std::time::Instant;

use crate::image::{LoadError, Program};
use crate::report::{RunOptions, RunReport};
use crate::space::SpaceError;

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
    /// The program's address space could not be built.
    #[error("the program could not be mapped into guest memory: {0}")]
    AddressSpace(#[from] SpaceError),
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

    /// Loads the program an ExeFS holds: its modules, placed one after the other, and the
    /// process its `main.npdm` describes (`starquake_loader::exefs::ExeFs::parse` reads the
    /// files). Works on every host.
    pub fn from_exefs(exefs: starquake_loader::exefs::ExeFs) -> Result<Self, Error> {
        Ok(Self {
            program: Program::from_exefs(exefs)?,
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
    use std::sync::{Arc, Mutex, PoisonError};
    use std::time::Instant;

    use starquake_cpu::hvf::{HvfBackend, HvfVcpu, PacKeys, PhysicalMemory, RamConfig};
    use starquake_cpu::{Backend, Vcpu};
    use starquake_kernel::sched::CoreId;
    use starquake_loader::patch::CounterRescale;
    use starquake_memory::Access;

    use super::Error;
    use crate::clock::{CounterClock, Timebase};
    use crate::image::{Loaded, Program, load};
    use crate::machine::{self, GuestProgram, GuestSpace, Vcpus};

    use crate::report::{RunOptions, RunReport};
    use crate::space::{AddressSpace, SpaceMapper, Touch};

    /// Held for the whole of a run: Hypervisor.framework allows one VM per process.
    static ONE_VM: Mutex<()> = Mutex::new(());

    /// The address-space id of the one process: the first id the kernel's pool hands out
    /// (`starquake_kernel::process::AsidPool`). PROVISIONAL: with one process per run nothing
    /// is given back, so the loop does not keep a pool; the core loop of kernel slice 10 takes
    /// ids from one, sized by the width the backend reports (`HvfVcpu::asid_bits`), and reports
    /// each flush to it.
    const PROCESS_ASID: u16 = starquake_kernel::process::asid::FIRST_ASID;

    /// The one address space of a run, as the loop drives it on the vCPU.
    impl GuestSpace<HvfVcpu> for AddressSpace<PhysicalMemory> {
        fn touch(&self, address: u64, access: Access) -> Touch {
            AddressSpace::touch(self, address, access)
        }

        fn settle(&self, vcpu: &mut HvfVcpu) -> Result<(), starquake_cpu::Error> {
            vcpu.flush_pending(self.tables(), PROCESS_ASID)
        }
    }

    pub(super) fn run(
        program: &Program,
        options: &RunOptions,
        started: Instant,
    ) -> Result<RunReport, Error> {
        let _one_vm = ONE_VM.lock().unwrap_or_else(PoisonError::into_inner);
        // PROVISIONAL: the backend's default guest RAM (16 MiB), and all of its table pool for
        // the one address space.
        let backend = HvfBackend::new(RamConfig::default())?;
        let ram = backend.ram();
        // One rate and one zero point for guest time (kernel spec, "Guest time"): the patcher
        // rescales the backend's counter to the rate of the console the program's description
        // names, the kernel converts its clock with the same rate, and that clock is the
        // backend's counter.
        let host_hz = backend.counter_frequency();
        let time = Timebase {
            console: program.description().console,
            clock: CounterClock::new(host_hz).ok_or_else(|| {
                starquake_cpu::Error::Host("the host counter reports a rate of zero".into())
            })?,
        };
        let counter = CounterRescale {
            host_hz,
            guest_hz: time.rate().hz(),
        };
        // The kernel's address space decides what is mapped; this one makes it real, and is
        // what the loader and the kernel read and write guest memory through.
        let space = Arc::new(AddressSpace::new(
            backend.physical(),
            ram,
            backend.table_pool(),
        )?);
        let mapper = SpaceMapper(Arc::clone(&space));
        let Loaded {
            placed,
            address_space,
        } = load(program, Some(counter), Box::new(mapper), &*space)?;
        // Every core's vCPU runs in the one address space, with the same pointer-authentication
        // keys, so that a thread can move from core to core. PROVISIONAL: the keys are fixed
        // (zero), not the process's own (`crate::machine`, "Start state").
        let new_vcpu = || -> Result<HvfVcpu, starquake_cpu::Error> {
            let mut vcpu = backend.create_vcpu()?;
            vcpu.set_address_space(space.root(), PROCESS_ASID)?;
            vcpu.set_pac_keys(&PacKeys::default())?;
            Ok(vcpu)
        };
        let make = |_: CoreId| new_vcpu();
        // Core 0's vCPU, made on this thread, which runs core 0. Declared after the backend, so
        // it is dropped first, as the VM requires; the other cores' vCPUs are dropped on their
        // own threads before `machine::run` returns.
        let mut vcpu = new_vcpu()?;
        let guest = GuestProgram {
            space: &*space,
            placed: &placed,
            address_space,
        };
        let vcpus = Vcpus {
            first: &mut vcpu,
            make: &make,
        };
        Ok(machine::run(vcpus, guest, options, started, time))
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
    fn an_address_space_that_cannot_be_built_is_reported_with_its_cause() {
        let cause = SpaceError::OutOfRam {
            size: 0x2000,
            available: 0x1000,
        };
        let error = Error::from(cause);
        assert_eq!(error, Error::AddressSpace(cause));
        assert_eq!(
            error.to_string(),
            "the program could not be mapped into guest memory: mapping 0x2000 bytes needs more \
             guest RAM than the 0x1000 bytes left"
        );
    }

    #[test]
    fn an_exefs_loads_on_every_host_and_its_metadata_is_checked() {
        use starquake_loader::exefs::{ExeFs, MAIN_MODULE, NPDM_FILE};
        let nso: &[u8] = include_bytes!("../../../tests/homebrew/hello/hello.nso");
        let npdm: &[u8] = include_bytes!("../../../tests/homebrew/hello/main.npdm");
        let exefs = || ExeFs::parse(&[(NPDM_FILE, npdm), (MAIN_MODULE, nso)]).unwrap();
        assert!(System::from_exefs(exefs()).is_ok());
        let mut no_threads = exefs();
        no_threads.npdm.aci.kernel_capabilities.thread_info = None;
        assert_eq!(
            System::from_exefs(no_threads),
            Err(Error::Load(LoadError::NoThreadInfo))
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

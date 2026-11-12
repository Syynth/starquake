//! Guest CPU execution behind pluggable backends.
//!
//! | Backend | Host | Status |
//! |---|---|---|
//! | [`hvf`] | macOS arm64 (Hypervisor.framework) | runs EL0 on stage-1 tables (grown from `tools/hvprobe`) |
//! | `kvm` | Linux arm64 | planned |
//! | `jit` | x86-64 (Steam Deck class) | planned |
//!
//! A vCPU runs guest code until it needs the host, and says why with an [`Exit`]. The HLE kernel
//! drives each vCPU from its own host thread: HVF requires a vCPU to run on the thread that
//! created it, so [`Vcpu`] is deliberately not `Send`. Other threads stop a running vCPU through
//! its [`Interrupter`].
//!
//! Guest code sees the address space a vCPU was last given with [`Vcpu::set_address_space`]: the
//! root of a set of [`starquake_memory::stage1::Stage1Tables`] and the address-space id its
//! translations are cached under. An exception the guest takes comes back as an [`Exit`] that
//! carries its [`Syndrome`].
//!
//! # Divergence from the execution-model spec
//!
//! `docs/specs/execution-model.md` lists `Exit::DataAbort`, `Exit::InstructionAbort` and
//! `Exit::Undefined` without a syndrome. Here each also carries a `syndrome` field (issues #85
//! and #36: "the exception detail carried in exits"); the fields the spec lists are unchanged.
//!
//! With the `test-support` feature the crate also exports `testing::ScriptedVcpu`, a [`Vcpu`]
//! that plays back scripted exits on any host.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[cfg_attr(
    all(target_os = "macos", target_arch = "aarch64"),
    expect(
        unsafe_code,
        reason = "Hypervisor.framework is a C API, and guest RAM is shared with the running guest"
    )
)]
pub mod hvf;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[cfg_attr(
    all(target_os = "macos", target_arch = "aarch64"),
    expect(
        unsafe_code,
        reason = "the host counter is read through a C function of libSystem"
    )
)]
pub mod host_counter;

pub mod syndrome;
#[cfg(any(test, feature = "test-support"))]
pub mod testing;

use starquake_memory::stage1::TableRoot;

pub use syndrome::{ExceptionClass, FaultStatus, Syndrome};

/// Why a vCPU stopped running guest code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// The guest executed `svc #imm`: an HLE kernel call.
    Svc(u16),
    /// A load or store the guest's address space doesn't allow: the address is not mapped, or
    /// not for this access.
    DataAbort {
        /// Faulting guest virtual address.
        address: u64,
        /// Whether the access was a write.
        write: bool,
        /// The exception's syndrome; [`Syndrome::fault`] says why the access aborted.
        syndrome: Syndrome,
    },
    /// An instruction fetch the guest's address space doesn't allow: the address is not mapped,
    /// or not executable.
    InstructionAbort {
        /// The guest virtual address that could not be fetched from.
        address: u64,
        /// The exception's syndrome; [`Syndrome::fault`] says why the fetch aborted.
        syndrome: Syndrome,
    },
    /// Any other exception the guest took at an instruction: an undefined instruction, a `brk`,
    /// a failed pointer authentication, a system register access nothing emulates, a misaligned
    /// PC or SP. [`Syndrome::class`] says which.
    Undefined {
        /// Address of the instruction.
        pc: u64,
        /// The exception's syndrome.
        syndrome: Syndrome,
    },
    /// Another host thread asked this vCPU to stop (preemption, shutdown).
    Interrupted,
}

/// AArch64 user-visible register state the HLE kernel reads and writes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GuestRegs {
    /// X0-X30.
    pub x: [u64; 31],
    /// `SP_EL0`.
    pub sp: u64,
    /// Program counter.
    pub pc: u64,
    /// NZCV and the other PSTATE bits visible at EL0.
    pub pstate: u64,
    /// `TPIDR_EL0` (user thread pointer).
    pub tpidr: u64,
    /// `TPIDRRO_EL0` (read-only thread pointer: the kernel's TLS address).
    pub tpidrro: u64,
}

/// Errors from creating or driving a vCPU.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// This backend can't run on this host.
    #[error("this CPU backend is not supported on this host")]
    Unsupported,
    /// The host API failed.
    #[error("host API error: {0}")]
    Host(String),
    /// A table root the backend can't walk: its page is not inside the memory the backend
    /// keeps translation tables in.
    #[error("table root {root:#x} is not inside the backend's table pool")]
    BadTableRoot {
        /// The guest-physical address of the root.
        root: u64,
    },
    /// An address-space id with more bits than the host gives a guest.
    #[error("address-space id {asid:#x} does not fit the {bits} bits this host gives a guest")]
    AsidTooWide {
        /// The id.
        asid: u16,
        /// How many bits an id may have.
        bits: u8,
    },
    /// Guest code was about to run on a vCPU that was never given an address space.
    #[error("the vCPU has no address space: give it one before running it")]
    NoAddressSpace,
    /// A TLB flush for a range that is empty, not made of whole pages, or not inside the
    /// address space guest tables translate.
    #[error("cannot flush {va:#x}+{size:#x}: not a non-empty run of pages in the address space")]
    BadFlushRange {
        /// The first address of the range.
        va: u64,
        /// Its size in bytes.
        size: u64,
    },
}

/// Stops a running vCPU from another host thread (preemption, shutdown).
pub trait Interrupter: Send + Sync {
    /// Makes the vCPU's current [`Vcpu::run`] return [`Exit::Interrupted`]. If the vCPU isn't
    /// running, its next `run` returns `Exit::Interrupted` without running guest code. Once the
    /// vCPU is gone this does nothing.
    fn interrupt(&self) -> Result<(), Error>;
}

/// One virtual CPU.
pub trait Vcpu {
    /// The handle other threads use to interrupt this vCPU.
    type Interrupter: Interrupter + Clone + 'static;

    /// Runs guest code until the next [`Exit`].
    fn run(&mut self) -> Result<Exit, Error>;
    /// Returns a handle that can stop [`Vcpu::run`] from another thread.
    fn interrupter(&self) -> Self::Interrupter;
    /// Reads the register state.
    fn regs(&self) -> Result<GuestRegs, Error>;
    /// Replaces the register state (context switch, SVC return values).
    fn set_regs(&mut self, regs: &GuestRegs) -> Result<(), Error>;
    /// Makes guest code on this vCPU run in the address space whose level-1 table is `root`,
    /// with its translations cached under the address-space id `asid`. A backend that
    /// translates addresses refuses to [`run`](Self::run) a vCPU before the first call
    /// ([`Error::NoAddressSpace`]), and refuses an id wider than its host supports
    /// ([`Error::AsidTooWide`]).
    ///
    /// The caller keeps the pairing honest: while translations cached under an id may exist,
    /// that id must not be given to different tables without a flush in between.
    fn set_address_space(&mut self, root: TableRoot, asid: u16) -> Result<(), Error>;
}

/// A way of executing guest code on this host.
pub trait Backend {
    /// The vCPU type this backend creates.
    type Vcpu: Vcpu;
    /// Creates a vCPU bound to the calling thread.
    fn create_vcpu(&self) -> Result<Self::Vcpu, Error>;
}

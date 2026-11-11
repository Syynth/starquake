//! Guest CPU execution behind pluggable backends.
//!
//! | Backend | Host | Status |
//! |---|---|---|
//! | [`hvf`] | macOS arm64 (Hypervisor.framework) | planned: port of `tools/hvprobe` |
//! | `kvm` | Linux arm64 | planned |
//! | `jit` | x86-64 (Steam Deck class) | planned |
//!
//! A vCPU runs guest code until it needs the host, and says why with an [`Exit`]. The HLE kernel
//! drives each vCPU from its own host thread: HVF requires a vCPU to run on the thread that
//! created it, so [`Vcpu`] is deliberately not `Send`.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod hvf;

/// Why a vCPU stopped running guest code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// The guest executed `svc #imm`: an HLE kernel call.
    Svc(u16),
    /// A memory access the backend couldn't satisfy.
    DataAbort {
        /// Faulting guest virtual address.
        address: u64,
        /// Whether the access was a write.
        write: bool,
    },
    /// An instruction the backend can't execute (on HVF: UNDEFINED that reached the host).
    Undefined {
        /// Address of the instruction.
        pc: u64,
    },
    /// Another host thread asked this vCPU to stop (preemption, shutdown).
    Interrupted,
}

/// AArch64 user-visible register state the HLE kernel reads and writes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GuestRegs {
    /// X0-X30.
    pub x: [u64; 31],
    /// SP_EL0.
    pub sp: u64,
    /// Program counter.
    pub pc: u64,
    /// NZCV and the other PSTATE bits visible at EL0.
    pub pstate: u64,
    /// TPIDR_EL0 (user thread pointer).
    pub tpidr: u64,
    /// TPIDRRO_EL0 (read-only thread pointer: the kernel's TLS address).
    pub tpidrro: u64,
}

/// Errors from creating or driving a vCPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// This backend can't run on this host.
    Unsupported,
    /// The host API failed.
    Host(String),
}

/// One virtual CPU.
pub trait Vcpu {
    /// Runs guest code until the next [`Exit`].
    fn run(&mut self) -> Result<Exit, Error>;
    /// Reads the register state.
    fn regs(&self) -> Result<GuestRegs, Error>;
    /// Replaces the register state (context switch, SVC return values).
    fn set_regs(&mut self, regs: &GuestRegs) -> Result<(), Error>;
}

/// A way of executing guest code on this host.
pub trait Backend {
    /// The vCPU type this backend creates.
    type Vcpu: Vcpu;
    /// Creates a vCPU bound to the calling thread.
    fn create_vcpu(&self) -> Result<Self::Vcpu, Error>;
}

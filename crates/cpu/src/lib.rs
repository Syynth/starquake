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
//! # A guest thread's context
//!
//! [`GuestRegs`] is the whole of it: the general registers, the stack pointer, the program
//! counter, the thread pointers, the [`GUEST_PSTATE_MASK`] bits of `PSTATE`, and the FP/SIMD
//! state (V0-V31, `FPCR`, `FPSR`). [`Vcpu::regs`] and [`Vcpu::set_regs`] move all of it, so a
//! context switch is one of each.
//!
//! The `PSTATE` rule: a context keeps the condition flags, `DIT` and `SSBS`, the bits EL0 code
//! can change (`msr nzcv`, `msr dit`, `msr ssbs`) on the Apple M2 the Hypervisor.framework
//! backend was measured on. That they are also the Cortex-A78C's EL0 bits is a guess: no
//! reference in the pinned set says which features the Switch 2's core has. Not kept: `TCO`
//! (memory tagging, which no backend enables), `BTYPE` (only meaningful with branch target
//! pages, which the stage-1 tables never set), the single-step and illegal-state bits, and the
//! mode and interrupt masks, which are the backend's.
//!
//! # Wait hints
//!
//! `wfi` and `wfe` at EL0 never reach the kernel: a backend resumes the guest at the next
//! instruction, without an [`Exit`], as Eden's dynarmic backend does for both
//! (`eden/src/core/arm/dynarmic/arm_dynarmic_64.cpp` @ cf654052,
//! `DynarmicCallbacks64::ExceptionRaised`: for `WaitForInterrupt` and `WaitForEvent` it writes a
//! trace log line and returns, and the guest goes on). A spinlock that waits with `wfe` keeps
//! spinning; a thread that idles in a
//! `wfi` loop keeps its core busy until it is interrupted, as a thread spinning on a branch
//! does. The guest's interrupter still stops either.
//!
//! # Instruction cache
//!
//! Every backend keeps guest instruction fetches coherent with writes the **host** makes to
//! guest memory through the backend (the loader, the code patcher, `GuestMemory::write`): once
//! the write has returned, a vCPU that enters the guest runs the new bytes, with no step by the
//! caller. (A vCPU that stays in guest code meanwhile is like a core running code another core
//! rewrites: the architecture promises it the new bytes only after a context synchronization
//! event, such as its next exception.) Writes the **guest** makes to its own code follow the
//! architecture: the guest cleans
//! and invalidates by address at EL0 (`dc cvau`, `ic ivau`, then `dsb ish` and `isb`), and a
//! backend must let it (on Hypervisor.framework `SCTLR_EL1.UCI` is set).
//! A JIT backend implements both by invalidating what it translated from the written range.
//!
//! # Divergence from the execution-model spec
//!
//! `docs/specs/execution-model.md` lists `Exit::DataAbort`, `Exit::InstructionAbort` and
//! `Exit::Undefined` without a syndrome. Here each also carries a `syndrome` field (issues #85
//! and #36: "the exception detail carried in exits"); the fields the spec lists are unchanged.
//!
//! The spec's `Vcpu::save` and `Vcpu::load` of a `GuestContext` are [`Vcpu::regs`] and
//! [`Vcpu::set_regs`] of a [`GuestRegs`] here (issue #35): the same full state under the names
//! the kernel and the core loop already call. The spec's single-register accessors are not
//! built yet, so a system call still moves the whole context, vectors included.
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

/// `PSTATE.{N,Z,C,V}`, the condition flags, in the `SPSR_EL1` layout (bits 31:28; Arm ARM,
/// `SPSR_EL1`).
pub const PSTATE_NZCV: u64 = 0xf << 28;
/// `PSTATE.DIT`, data-independent timing (`FEAT_DIT`), in the `SPSR_EL1` layout (bit 24). EL0
/// sets it with `msr dit, #imm`.
pub const PSTATE_DIT: u64 = 1 << 24;
/// `PSTATE.SSBS`, speculative store bypass safe (`FEAT_SSBS`), in the `SPSR_EL1` layout
/// (bit 12). EL0 sets it with `msr ssbs, #imm`.
pub const PSTATE_SSBS: u64 = 1 << 12;
/// The `PSTATE` bits a guest thread's context keeps: [`PSTATE_NZCV`], [`PSTATE_DIT`] and
/// [`PSTATE_SSBS`], the bits EL0 can change on the hosts the backends run on. The mode, the
/// interrupt masks and every other bit are the backend's own: [`Vcpu::set_regs`] ignores them
/// and [`Vcpu::regs`] reads them as zero.
pub const GUEST_PSTATE_MASK: u64 = PSTATE_NZCV | PSTATE_DIT | PSTATE_SSBS;

/// The register state of one guest thread at EL0: everything a context switch saves and
/// restores (the execution model's `GuestContext`), and what the HLE kernel reads and writes.
///
/// The FP/SIMD part is the architecture's (Arm ARM, "SIMD and floating-point registers",
/// `FPCR`, `FPSR`): 32 128-bit vector registers, each as one little-endian value (byte 0 of
/// `Vn` is bits 7:0 of `v[n]`, so `Dn` is the low 64 bits and `Sn` the low 32), and the two
/// control and status registers, as the 64-bit system registers EL0 reads with `mrs`. A
/// backend keeps every bit of `v`; what it keeps of `fpcr` and `fpsr` is what the host lets a
/// guest write (see each backend).
///
/// Not here yet: the SVE and SME state the Switch 2's Cortex-A78C does not have, and
/// pointer-authentication keys, which belong to the process and are set separately (on the
/// Hypervisor.framework backend, `hvf::HvfVcpu::set_pac_keys`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GuestRegs {
    /// X0-X30.
    pub x: [u64; 31],
    /// `SP_EL0`.
    pub sp: u64,
    /// Program counter.
    pub pc: u64,
    /// The `PSTATE` bits in [`GUEST_PSTATE_MASK`] (condition flags, `DIT`, `SSBS`), in the
    /// `SPSR_EL1` layout.
    pub pstate: u64,
    /// `TPIDR_EL0` (user thread pointer).
    pub tpidr: u64,
    /// `TPIDRRO_EL0` (read-only thread pointer: the kernel's TLS address).
    pub tpidrro: u64,
    /// V0-V31, the FP/SIMD registers.
    pub v: [u128; VECTOR_REGS],
    /// `FPCR`: rounding mode, flush-to-zero, default NaN and the trap enables.
    pub fpcr: u64,
    /// `FPSR`: the cumulative exception flags and the saturation flag.
    pub fpsr: u64,
}

/// How many FP/SIMD registers a context holds: V0-V31.
pub const VECTOR_REGS: usize = 32;

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

    /// Runs guest code until the next [`Exit`]. Wait hints (`wfi`, `wfe`) never end a run: see
    /// "Wait hints" in the crate docs.
    fn run(&mut self) -> Result<Exit, Error>;
    /// Returns a handle that can stop [`Vcpu::run`] from another thread.
    fn interrupter(&self) -> Self::Interrupter;
    /// Reads the whole register state, FP/SIMD included, with `pstate` limited to
    /// [`GUEST_PSTATE_MASK`].
    fn regs(&self) -> Result<GuestRegs, Error>;
    /// Replaces the whole register state, FP/SIMD included (context switch, SVC return
    /// values). Bits of `pstate` outside [`GUEST_PSTATE_MASK`] are ignored.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_guest_pstate_bits_are_at_their_spsr_positions() {
        // Arm DDI 0487, `SPSR_EL1`: N, Z, C, V at 31:28, DIT at 24, SSBS at 12.
        assert_eq!(PSTATE_NZCV, 0xf000_0000);
        assert_eq!(PSTATE_DIT, 0x0100_0000);
        assert_eq!(PSTATE_SSBS, 0x0000_1000);
        assert_eq!(GUEST_PSTATE_MASK, 0xf100_1000);
        // None is a bit the backend keeps for itself: the mode (M, 3:0) and the D, A, I, F
        // masks (9:6).
        assert_eq!(GUEST_PSTATE_MASK & 0x3cf, 0);
    }
}

//! The slice of Hypervisor.framework (arm64) the backend uses, and its error codes.
//!
//! Ported from `tools/hvprobe/src/hv.rs`; signatures, register ids and error codes are from the
//! macOS SDK headers `Hypervisor.framework/Headers/{hv_vcpu.h,hv_vm.h,hv_vcpu_types.h,hv_error.h}`.

use std::ffi::c_void;

use crate::Error;

/// `hv_return_t`.
pub(super) type HvReturn = i32;
/// `hv_vcpu_t`.
pub(super) type VcpuId = u64;

/// `hv_vcpu_exit_exception_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct ExitException {
    /// `ESR_EL2` of the exception that reached the host.
    pub syndrome: u64,
    pub virtual_address: u64,
    pub physical_address: u64,
}

/// `hv_vcpu_exit_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct ExitInfo {
    pub reason: u32,
    pub exception: ExitException,
}

// hv_exit_reason_t
pub(super) const EXIT_REASON_CANCELED: u32 = 0;
pub(super) const EXIT_REASON_EXCEPTION: u32 = 1;

// hv_memory_flags_t
pub(super) const MEMORY_READ: u64 = 1;
pub(super) const MEMORY_WRITE: u64 = 2;
pub(super) const MEMORY_EXEC: u64 = 4;

// hv_reg_t: X0-X30 are 0-30.
pub(super) const REG_X0: u32 = 0;
pub(super) const REG_PC: u32 = 31;
pub(super) const REG_FPCR: u32 = 32;
pub(super) const REG_FPSR: u32 = 33;
pub(super) const REG_CPSR: u32 = 34;

// hv_simd_fp_reg_t: Q0-Q31 are 0-31.
pub(super) const SIMD_FP_REG_Q0: u32 = 0;

/// The bytes of an `hv_simd_fp_uchar16_t` (a 16-byte vector, 16-byte aligned) in memory,
/// byte 0 first.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct SimdFpBytes(pub [u8; 16]);

// hv_sys_reg_t: (op0 << 14) | (op1 << 11) | (CRn << 7) | (CRm << 3) | op2
pub(super) const ID_AA64MMFR0_EL1: u16 = 0xc038;
pub(super) const SCTLR_EL1: u16 = 0xc080;
pub(super) const CPACR_EL1: u16 = 0xc082;
pub(super) const TTBR0_EL1: u16 = 0xc100;
pub(super) const TTBR1_EL1: u16 = 0xc101;
pub(super) const TCR_EL1: u16 = 0xc102;
pub(super) const APIAKEYLO_EL1: u16 = 0xc108;
pub(super) const APIAKEYHI_EL1: u16 = 0xc109;
pub(super) const APIBKEYLO_EL1: u16 = 0xc10a;
pub(super) const APIBKEYHI_EL1: u16 = 0xc10b;
pub(super) const APDAKEYLO_EL1: u16 = 0xc110;
pub(super) const APDAKEYHI_EL1: u16 = 0xc111;
pub(super) const APDBKEYLO_EL1: u16 = 0xc112;
pub(super) const APDBKEYHI_EL1: u16 = 0xc113;
pub(super) const APGAKEYLO_EL1: u16 = 0xc118;
pub(super) const APGAKEYHI_EL1: u16 = 0xc119;
pub(super) const SPSR_EL1: u16 = 0xc200;
pub(super) const ELR_EL1: u16 = 0xc201;
pub(super) const SP_EL0: u16 = 0xc208;
pub(super) const ESR_EL1: u16 = 0xc290;
pub(super) const FAR_EL1: u16 = 0xc300;
pub(super) const MAIR_EL1: u16 = 0xc510;
pub(super) const VBAR_EL1: u16 = 0xc600;
pub(super) const CNTKCTL_EL1: u16 = 0xc708;
pub(super) const TPIDR_EL0: u16 = 0xde82;
pub(super) const TPIDRRO_EL0: u16 = 0xde83;
pub(super) const SP_EL1: u16 = 0xe208;

// hv_return_t error codes (hv_error.h).
const HV_BUSY: u32 = 0xfae9_4002;
const HV_DENIED: u32 = 0xfae9_4007;
const HV_UNSUPPORTED: u32 = 0xfae9_400f;

#[link(name = "Hypervisor", kind = "framework")]
unsafe extern "C" {
    pub(super) fn hv_vm_create(config: *mut c_void) -> HvReturn;
    pub(super) fn hv_vm_destroy() -> HvReturn;
    pub(super) fn hv_vm_map(addr: *mut c_void, ipa: u64, size: usize, flags: u64) -> HvReturn;
    pub(super) fn hv_vm_unmap(ipa: u64, size: usize) -> HvReturn;
    pub(super) fn hv_vcpu_create(
        vcpu: *mut VcpuId,
        exit: *mut *const ExitInfo,
        config: *mut c_void,
    ) -> HvReturn;
    pub(super) fn hv_vcpu_destroy(vcpu: VcpuId) -> HvReturn;
    pub(super) fn hv_vcpu_run(vcpu: VcpuId) -> HvReturn;
    pub(super) fn hv_vcpus_exit(vcpus: *const VcpuId, vcpu_count: u32) -> HvReturn;
    pub(super) fn hv_vcpu_get_reg(vcpu: VcpuId, reg: u32, value: *mut u64) -> HvReturn;
    pub(super) fn hv_vcpu_set_reg(vcpu: VcpuId, reg: u32, value: u64) -> HvReturn;
    pub(super) fn hv_vcpu_get_sys_reg(vcpu: VcpuId, reg: u16, value: *mut u64) -> HvReturn;
    pub(super) fn hv_vcpu_set_sys_reg(vcpu: VcpuId, reg: u16, value: u64) -> HvReturn;
    pub(super) fn hv_vcpu_get_simd_fp_reg(
        vcpu: VcpuId,
        reg: u32,
        value: *mut SimdFpBytes,
    ) -> HvReturn;
    /// Takes `(hv_vcpu_t, hv_simd_fp_reg_t, hv_simd_fp_uchar16_t)`, the vector by value. Rust
    /// cannot pass a vector to a foreign function yet (`simd_ffi` is unstable), so this is
    /// declared without parameters and called only through [`set_simd_fp_reg`].
    fn hv_vcpu_set_simd_fp_reg();
    pub(super) fn hv_vcpu_set_vtimer_mask(vcpu: VcpuId, masked: bool) -> HvReturn;
}

// libSystem is linked into every Rust binary on macOS, so no `#[link]` attribute is needed.
unsafe extern "C" {
    /// `sys_icache_invalidate` (`<libkern/OSCacheControl.h>`): makes instruction fetches on every
    /// core see the bytes just written to `start..start + len`. It cleans the data cache to the
    /// point of unification, invalidates the instruction cache by address (both broadcast to all
    /// cores) and issues the barriers.
    pub(super) fn sys_icache_invalidate(start: *mut c_void, len: usize);
}

/// `hv_vcpu_set_simd_fp_reg(vcpu, reg, value)`, with the vector passed as the C compiler
/// passes it.
///
/// The AArch64 procedure call standard (AAPCS64, "Parameter passing") passes a 128-bit short
/// vector in the next free SIMD register and integers in the next free general registers, so
/// the call is `x0` = `vcpu`, `w1` = `reg`, `q0` = the 16 bytes; the result comes back in `w0`.
///
/// # Safety
///
/// As for any `hv_vcpu_*` call: on the thread that owns the live vCPU `vcpu`.
pub(super) unsafe fn set_simd_fp_reg(vcpu: VcpuId, reg: u32, value: &SimdFpBytes) -> HvReturn {
    let function: unsafe extern "C" fn() = hv_vcpu_set_simd_fp_reg;
    let ret: u64;
    // SAFETY: the call follows AAPCS64 for the C prototype in `hv_vcpu.h`: the arguments are in
    // x0, w1 and q0 (loaded from `value`, which is live and 16 bytes), `clobber_abi("C")` tells
    // the compiler every register the callee may change, and without `nostack` the stack
    // pointer is aligned for a call. The caller's contract makes the call itself valid.
    unsafe {
        core::arch::asm!(
            "ldr q0, [{value}]",
            "blr {function}",
            value = in(reg) value.0.as_ptr(),
            function = in(reg) function,
            inout("x0") vcpu => ret,
            inout("x1") u64::from(reg) => _,
            clobber_abi("C"),
        );
    }
    // `hv_return_t` is 32 bits, returned in w0: the upper half of x0 is not part of it.
    let [b0, b1, b2, b3, ..] = ret.to_le_bytes();
    HvReturn::from_le_bytes([b0, b1, b2, b3])
}

/// The host's `CNTFRQ_EL0`: the rate, in hertz, of the counter behind `CNTVCT_EL0`.
pub(super) fn cntfrq_el0() -> u64 {
    let hz: u64;
    // SAFETY: `mrs` of `CNTFRQ_EL0` is permitted at EL0 on macOS (it is how user space times
    // things) and only copies a system register into the output register: it reads and writes no
    // memory, uses no stack and leaves the flags alone.
    unsafe {
        core::arch::asm!(
            "mrs {hz}, cntfrq_el0",
            hz = out(reg) hz,
            options(nomem, nostack, preserves_flags),
        );
    }
    hz
}

/// Turns a Hypervisor.framework return code into a [`Result`], naming the call that failed.
pub(super) fn check(what: &str, ret: HvReturn) -> Result<(), Error> {
    if ret == 0 {
        return Ok(());
    }
    let code = u32::from_ne_bytes(ret.to_ne_bytes());
    Err(match code {
        HV_UNSUPPORTED => Error::Unsupported,
        HV_DENIED => Error::Host(format!(
            "{what} was denied ({code:#x}): is the binary signed with the hypervisor entitlement?"
        )),
        HV_BUSY => Error::Host(format!(
            "{what} failed: busy ({code:#x}); only one VM can exist per process"
        )),
        _ => Error::Host(format!("{what} failed: {code:#x}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_codes_map_to_errors() {
        assert_eq!(check("x", 0), Ok(()));
        let as_ret = |code: u32| i32::from_ne_bytes(code.to_ne_bytes());
        assert_eq!(check("x", as_ret(HV_UNSUPPORTED)), Err(Error::Unsupported));
        for code in [HV_DENIED, HV_BUSY, 0xfae9_4001] {
            let result = check("hv_vm_create", as_ret(code));
            assert!(
                matches!(&result, Err(Error::Host(message))
                    if message.starts_with("hv_vm_create")
                        && message.contains(&format!("{code:#x}"))),
                "{result:?}"
            );
        }
    }
}

//! Minimal Hypervisor.framework bindings (arm64).
#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_void;

pub type hv_return_t = i32;
pub type hv_vcpu_t = u64;
pub type hv_vm_config_t = *mut c_void;

#[repr(C)]
pub struct hv_vcpu_exit_exception_t {
    pub syndrome: u64,
    pub virtual_address: u64,
    pub physical_address: u64,
}

#[repr(C)]
pub struct hv_vcpu_exit_t {
    pub reason: u32,
    pub exception: hv_vcpu_exit_exception_t,
}

pub const HV_EXIT_REASON_CANCELED: u32 = 0;
pub const HV_EXIT_REASON_EXCEPTION: u32 = 1;
pub const HV_EXIT_REASON_VTIMER_ACTIVATED: u32 = 2;

pub const HV_MEMORY_READ: u64 = 1;
pub const HV_MEMORY_WRITE: u64 = 2;
pub const HV_MEMORY_EXEC: u64 = 4;

// hv_reg_t
pub const HV_REG_X0: u32 = 0;
pub const HV_REG_PC: u32 = 31;
pub const HV_REG_CPSR: u32 = 34;

// hv_sys_reg_t: (op0 << 14) | (op1 << 11) | (CRn << 7) | (CRm << 3) | op2
pub const SCTLR_EL1: u16 = 0xc080;
pub const CPACR_EL1: u16 = 0xc082;
pub const TTBR0_EL1: u16 = 0xc100;
pub const TCR_EL1: u16 = 0xc102;
pub const APIAKEYLO_EL1: u16 = 0xc108;
pub const APIAKEYHI_EL1: u16 = 0xc109;
pub const APIBKEYLO_EL1: u16 = 0xc10a;
pub const APIBKEYHI_EL1: u16 = 0xc10b;
pub const MAIR_EL1: u16 = 0xc510;
pub const VBAR_EL1: u16 = 0xc600;
pub const CNTKCTL_EL1: u16 = 0xc708;
pub const SP_EL1: u16 = 0xe208;
pub const ID_AA64ISAR1_EL1: u16 = 0xc031;

#[link(name = "Hypervisor", kind = "framework")]
unsafe extern "C" {
    pub fn hv_vm_config_create() -> hv_vm_config_t;
    pub fn hv_vm_config_get_max_ipa_size(bits: *mut u32) -> hv_return_t;
    pub fn hv_vm_config_get_default_ipa_size(bits: *mut u32) -> hv_return_t;
    pub fn hv_vm_config_set_ipa_size(config: hv_vm_config_t, bits: u32) -> hv_return_t;
    pub fn hv_vm_create(config: hv_vm_config_t) -> hv_return_t;
    pub fn hv_vm_map(addr: *mut c_void, ipa: u64, size: usize, flags: u64) -> hv_return_t;
    pub fn hv_vcpu_create(
        vcpu: *mut hv_vcpu_t,
        exit: *mut *const hv_vcpu_exit_t,
        config: *mut c_void,
    ) -> hv_return_t;
    pub fn hv_vcpu_destroy(vcpu: hv_vcpu_t) -> hv_return_t;
    pub fn hv_vcpu_run(vcpu: hv_vcpu_t) -> hv_return_t;
    pub fn hv_vcpu_get_reg(vcpu: hv_vcpu_t, reg: u32, value: *mut u64) -> hv_return_t;
    pub fn hv_vcpu_set_reg(vcpu: hv_vcpu_t, reg: u32, value: u64) -> hv_return_t;
    pub fn hv_vcpu_get_sys_reg(vcpu: hv_vcpu_t, reg: u16, value: *mut u64) -> hv_return_t;
    pub fn hv_vcpu_set_sys_reg(vcpu: hv_vcpu_t, reg: u16, value: u64) -> hv_return_t;
    pub fn hv_vcpu_set_vtimer_mask(vcpu: hv_vcpu_t, masked: bool) -> hv_return_t;
}

unsafe extern "C" {
    pub fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
}
pub const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
pub const QOS_CLASS_BACKGROUND: u32 = 0x09;

pub fn check(what: &str, r: hv_return_t) {
    if r != 0 {
        panic!("{what} failed: {:#x}", r as u32);
    }
}

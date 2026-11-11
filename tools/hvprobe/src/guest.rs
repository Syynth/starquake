//! Guest code, assembled into the host binary and copied into guest memory at GUEST_BASE.
//! Everything is position independent; the host passes addresses and counts in registers.
//!
//! Host exits (HVC immediates):
//!   #0  EL1 test finished
//!   #1  exception taken to guest EL1 that the handler doesn't emulate (x9=ESR, x10=ELR, x11=FAR);
//!       the handler then skips the faulting instruction
//!   #2  EL0 test finished (svc #0)
//!   #4  timestamp request (host notes wall time)
//!   #6  EL0 "syscall" (svc #n, n != 0), forwarded to the host like an HLE SVC
//!
//! Registers x9-x11 are clobbered by the exception handler; tests avoid them.

use std::arch::global_asm;

global_asm!(
    ".p2align 14",
    ".globl _guest_blob_start",
    "_guest_blob_start:",
    // ---- EL1 vector table: 16 entries of 0x80 bytes, all to the common handler ----
    ".rept 16",
    "b 9f",
    ".p2align 7",
    ".endr",
    "9:",
    "mrs x9, esr_el1",
    "lsr x10, x9, #26",
    "cmp x10, #0x18", // trapped MSR/MRS from EL0 (counter access with EL0PCTEN=0)
    "b.eq 20f",
    "cmp x10, #0x15", // SVC64
    "b.eq 30f",
    "mrs x10, elr_el1",
    "mrs x11, far_el1",
    "hvc #1",
    "mrs x10, elr_el1",
    "add x10, x10, #4",
    "msr elr_el1, x10",
    "eret",
    // trapped counter read: emulate in-guest (probe assumes Rt = x0)
    "20:",
    "mrs x0, cntvct_el0",
    "mrs x10, elr_el1",
    "add x10, x10, #4",
    "msr elr_el1, x10",
    "eret",
    // svc from EL0: #0 ends the test, anything else is forwarded to the host
    "30:",
    "and x10, x9, #0xffff",
    "cbz x10, 31f",
    "hvc #6",
    "eret",
    "31:",
    "hvc #2",
    "eret",
    // ---- EL1: ID registers and counter frequency ----
    ".globl _t_ids",
    "_t_ids:",
    "mrs x0, id_aa64isar1_el1",
    "mrs x1, id_aa64pfr1_el1",
    "mrs x2, id_aa64mmfr0_el1",
    "mrs x3, cntfrq_el0",
    "mrs x4, id_aa64isar0_el1",
    "mrs x5, id_aa64mmfr1_el1",
    "hvc #0",
    // ---- EL1: physical vs virtual counter ----
    ".globl _t_cntpct_el1",
    "_t_cntpct_el1:",
    "mrs x0, cntpct_el0",
    "mrs x1, cntvct_el0",
    "hvc #0",
    // ---- EL1: counter tick rate against host wall time (x5 = spin count) ----
    ".globl _t_rate",
    "_t_rate:",
    "mrs x0, cntvct_el0",
    "hvc #4",
    "mov x2, x5",
    "1:",
    "subs x2, x2, #1",
    "b.ne 1b",
    "mrs x1, cntvct_el0",
    "mrs x3, cntfrq_el0",
    "hvc #0",
    // ---- EL0: N reads of cntpct_el0 (native or trapped, depending on CNTKCTL_EL1) ----
    ".globl _t_el0_cntpct",
    "_t_el0_cntpct:",
    "mrs x2, cntvct_el0",
    "mov x3, x5",
    "1:",
    "mrs x0, cntpct_el0",
    "subs x3, x3, #1",
    "b.ne 1b",
    "mrs x4, cntvct_el0",
    "sub x0, x4, x2",
    "svc #0",
    // ---- EL0: N counter reads through a patched-in stub that rescales to 31.25 MHz ----
    // x12 = fractional scale factor (Q64), result = t + umulh(t, x12)
    ".globl _t_el0_patched",
    "_t_el0_patched:",
    "mrs x2, cntvct_el0",
    "mov x3, x5",
    "1:",
    "bl 2f",
    "subs x3, x3, #1",
    "b.ne 1b",
    "mrs x4, cntvct_el0",
    "sub x0, x4, x2",
    "mov x1, x14",
    "svc #0",
    "2:",
    "mrs x14, cntvct_el0",
    "umulh x13, x14, x12",
    "add x14, x14, x13",
    "ret",
    // ---- EL0: N svc round trips through the host ----
    ".globl _t_el0_svc",
    "_t_el0_svc:",
    "mov x3, x5",
    "1:",
    "svc #0x34",
    "subs x3, x3, #1",
    "b.ne 1b",
    "svc #0",
    // ---- EL1: pointer authentication (x1 = pointer, x2 = modifier) ----
    ".globl _t_pauth",
    "_t_pauth:",
    "mov x0, x1",
    "pacia x0, x2", // x0 = signed pointer
    "mov x3, x0",
    "autia x3, x2", // x3 = authenticated (should equal x1)
    "eor x4, x0, #(1 << 50)",
    "autia x4, x2", // tampered: FPAC should fault here (reported via hvc #1)
    "mov x5, x1",
    "pacib x5, x2", // B key result, for cross-core comparison
    "bl 2f",        // function using pacibsp/retab, like Switch 2 code
    "hvc #0",
    "2:",
    "pacibsp",
    "mov x6, #42",
    "retab",
    // ---- EL0: ARMv8.1-8.4 instructions Switch 2 titles use heavily (x1 = data) ----
    ".globl _t_el0_atomics",
    "_t_el0_atomics:",
    "mov x2, #5",
    "str x2, [x1]",
    "ldapr x0, [x1]", // 5
    "mov x3, #5",
    "mov x4, #7",
    "cas x3, x4, [x1]", // x3 = 5, mem = 7
    "mov x5, #10",
    "ldadd x5, x6, [x1]", // x6 = 7, mem = 17
    "mov x7, #1",
    "swp x7, x12, [x1]", // x12 = 17, mem = 1
    "ldr x13, [x1]",     // 1
    "svc #0",
    // ---- EL1: MTE store-tag (one per title in the dumps) ----
    ".globl _t_mte",
    "_t_mte:",
    ".arch_extension memtag",
    "stg x1, [x1]",
    "mov x0, #1",
    "hvc #0",
    // ---- EL1: two 4 KiB pages mapped out of order inside one 16 KiB host page ----
    ".globl _t_remap",
    "_t_remap:",
    "ldr x0, [x1]",
    "ldr x3, [x2]",
    "hvc #0",
    // ---- EL1: read one word (x1 = address) ----
    ".globl _t_read",
    "_t_read:",
    "ldr x0, [x1]",
    "hvc #0",
    // ---- EL1: keep reading [x1] without leaving the guest until it equals x2 (x3 = max tries) ----
    ".globl _t_spin_read",
    "_t_spin_read:",
    "1:",
    "ldr x0, [x1]",
    "cmp x0, x2",
    "b.eq 2f",
    "subs x3, x3, #1",
    "b.ne 1b",
    "2:",
    "hvc #0",
    // ---- EL1: flush this VM's stage-1 TLB entries on every vCPU (broadcast form) ----
    ".globl _t_tlbi_is",
    "_t_tlbi_is:",
    "tlbi vmalle1is",
    "dsb ish",
    "isb",
    "hvc #0",
    // ---- EL1: flush this vCPU's stage-1 TLB entries only ----
    ".globl _t_tlbi_local",
    "_t_tlbi_local:",
    "tlbi vmalle1",
    "dsb nsh",
    "isb",
    "hvc #0",
    ".globl _guest_blob_end",
    "_guest_blob_end:",
);

unsafe extern "C" {
    static guest_blob_start: u8;
    static guest_blob_end: u8;
    static t_ids: u8;
    static t_cntpct_el1: u8;
    static t_rate: u8;
    static t_el0_cntpct: u8;
    static t_el0_patched: u8;
    static t_el0_svc: u8;
    static t_pauth: u8;
    static t_el0_atomics: u8;
    static t_mte: u8;
    static t_remap: u8;
    static t_read: u8;
    static t_spin_read: u8;
    static t_tlbi_is: u8;
    static t_tlbi_local: u8;
}

pub fn blob() -> &'static [u8] {
    unsafe {
        let start = &guest_blob_start as *const u8;
        let end = &guest_blob_end as *const u8;
        std::slice::from_raw_parts(start, end.offset_from(start) as usize)
    }
}

fn offset(sym: &u8) -> u64 {
    unsafe { (sym as *const u8).offset_from(&guest_blob_start as *const u8) as u64 }
}

#[derive(Clone, Copy, Debug)]
pub enum Test {
    Ids,
    CntpctEl1,
    Rate,
    El0Cntpct,
    El0Patched,
    El0Svc,
    Pauth,
    El0Atomics,
    Mte,
    Remap,
    Read,
    SpinRead,
    TlbiIs,
    TlbiLocal,
}

impl Test {
    pub fn offset(self) -> u64 {
        unsafe {
            offset(match self {
                Test::Ids => &t_ids,
                Test::CntpctEl1 => &t_cntpct_el1,
                Test::Rate => &t_rate,
                Test::El0Cntpct => &t_el0_cntpct,
                Test::El0Patched => &t_el0_patched,
                Test::El0Svc => &t_el0_svc,
                Test::Pauth => &t_pauth,
                Test::El0Atomics => &t_el0_atomics,
                Test::Mte => &t_mte,
                Test::Remap => &t_remap,
                Test::Read => &t_read,
                Test::SpinRead => &t_spin_read,
                Test::TlbiIs => &t_tlbi_is,
                Test::TlbiLocal => &t_tlbi_local,
            })
        }
    }

    pub fn el0(self) -> bool {
        matches!(
            self,
            Test::El0Cntpct | Test::El0Patched | Test::El0Svc | Test::El0Atomics
        )
    }
}

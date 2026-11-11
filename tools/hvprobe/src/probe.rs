//! Checks the Apple Silicon Hypervisor.framework assumptions a native-execution
//! Switch 2 backend depends on. See README.md for what each test answers.

use crate::guest::{self, Test};
use crate::hv::*;
use std::time::{Duration, Instant};

const GUEST_BASE: u64 = 0x8000_0000;
const GUEST_SIZE: usize = 16 << 20;
const PT_BASE: u64 = 0x10_0000; // offsets from GUEST_BASE
const DATA: u64 = 0x8_0000;
const MARKERS: u64 = 0xC_0000;
const STACK_TOP: u64 = 0x20_0000;
const REMAP_VA: u64 = GUEST_BASE + 0x20_0000;

const SWITCH2_CNTFRQ: f64 = 31_250_000.0;

struct Mem(*mut u8);
unsafe impl Send for Mem {}
unsafe impl Sync for Mem {}
static mut MEM: Mem = Mem(std::ptr::null_mut());

fn mem() -> *mut u8 {
    unsafe { (*std::ptr::addr_of!(MEM)).0 }
}
fn write64(off: u64, v: u64) {
    unsafe { (mem().add(off as usize) as *mut u64).write_volatile(v) }
}
fn read32(off: u64) -> u32 {
    unsafe { (mem().add(off as usize) as *const u32).read_volatile() }
}

// ---- stage-1 page tables: 4 KiB granule, 39-bit VA (T0SZ = 25), walk starts at level 1 ----
const PAGE_AF: u64 = 1 << 10;
const PAGE_SH_INNER: u64 = 3 << 8;
const AP_EL1_RW: u64 = 0 << 6;
const AP_ALL_RW: u64 = 1 << 6;
const AP_ALL_RO: u64 = 3 << 6;
const PXN: u64 = 1 << 53;
const UXN: u64 = 1 << 54;

fn l3_entry(table: u64, va: u64, pa_off: u64, attrs: u64) {
    let index = (va >> 12) & 0x1ff;
    write64(
        table + index * 8,
        (GUEST_BASE + pa_off) | 0b11 | PAGE_AF | PAGE_SH_INNER | attrs,
    );
}

fn build_page_tables() {
    let (l1, l2, l3a, l3b) = (
        PT_BASE,
        PT_BASE + 0x1000,
        PT_BASE + 0x2000,
        PT_BASE + 0x3000,
    );
    write64(
        l1 + ((GUEST_BASE >> 30) & 0x1ff) * 8,
        (GUEST_BASE + l2) | 0b11,
    );
    write64(
        l2 + ((GUEST_BASE >> 21) & 0x1ff) * 8,
        (GUEST_BASE + l3a) | 0b11,
    );
    write64(
        l2 + ((REMAP_VA >> 21) & 0x1ff) * 8,
        (GUEST_BASE + l3b) | 0b11,
    );
    for page in (0..0x1_0000u64).step_by(0x1000) {
        l3_entry(l3a, GUEST_BASE + page, page, AP_ALL_RO); // code: EL0 + EL1 executable
    }
    for page in (DATA..DATA + 0x1_0000).step_by(0x1000) {
        l3_entry(l3a, GUEST_BASE + page, page, AP_ALL_RW | PXN | UXN);
    }
    for page in (MARKERS..MARKERS + 0x4000).step_by(0x1000) {
        l3_entry(l3a, GUEST_BASE + page, page, AP_EL1_RW | PXN | UXN);
    }
    for page in (STACK_TOP - 0x2_0000..STACK_TOP).step_by(0x1000) {
        l3_entry(l3a, GUEST_BASE + page, page, AP_EL1_RW | PXN | UXN);
    }
    // Two 4 KiB pages that live inside the same 16 KiB host page, mapped in reverse order.
    l3_entry(
        l3b,
        REMAP_VA + 0x1000,
        MARKERS + 0x3000,
        AP_EL1_RW | PXN | UXN,
    );
    l3_entry(
        l3b,
        REMAP_VA + 0x2000,
        MARKERS + 0x1000,
        AP_EL1_RW | PXN | UXN,
    );
    write64(MARKERS + 0x1000, 0xC1C1_C1C1_0000_1000);
    write64(MARKERS + 0x3000, 0xC3C3_C3C3_0000_3000);
}

// ---- vCPU ----
struct Vcpu {
    id: hv_vcpu_t,
    exit: *const hv_vcpu_exit_t,
}

const PAC_KEY_A: (u64, u64) = (0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
const PAC_KEY_B: (u64, u64) = (0x1111_2222_3333_4444, 0x5555_6666_7777_8888);
const CNTKCTL_EL0_ALL: u64 = 0b11; // EL0PCTEN | EL0VCTEN
const CNTKCTL_EL0_VCT_ONLY: u64 = 0b10; // physical counter reads from EL0 trap to EL1

impl Vcpu {
    fn new() -> Vcpu {
        let mut id = 0;
        let mut exit = std::ptr::null();
        unsafe {
            check(
                "hv_vcpu_create",
                hv_vcpu_create(&mut id, &mut exit, std::ptr::null_mut()),
            )
        };
        let v = Vcpu { id, exit };
        let sctlr = 0x30D0_0800 // RES1
            | 1 | (1 << 2) | (1 << 12) // M, C, I
            | (1 << 16) | (1 << 18) // nTWI, nTWE
            | (1 << 31) | (1 << 30); // EnIA, EnIB
        let tcr = 25 | (1 << 8) | (1 << 10) | (3 << 12) | (1 << 23) | (2u64 << 32);
        v.sys(MAIR_EL1, 0xff);
        v.sys(TCR_EL1, tcr);
        v.sys(TTBR0_EL1, GUEST_BASE + PT_BASE);
        v.sys(VBAR_EL1, GUEST_BASE);
        v.sys(CPACR_EL1, 3 << 20);
        v.sys(SP_EL1, GUEST_BASE + STACK_TOP);
        v.sys(CNTKCTL_EL1, CNTKCTL_EL0_ALL);
        v.sys(APIAKEYLO_EL1, PAC_KEY_A.0);
        v.sys(APIAKEYHI_EL1, PAC_KEY_A.1);
        v.sys(APIBKEYLO_EL1, PAC_KEY_B.0);
        v.sys(APIBKEYHI_EL1, PAC_KEY_B.1);
        v.sys(SCTLR_EL1, sctlr);
        unsafe { hv_vcpu_set_vtimer_mask(id, true) };
        v
    }
    fn sys(&self, reg: u16, value: u64) {
        unsafe {
            check(
                &format!("set_sys_reg {reg:#x}"),
                hv_vcpu_set_sys_reg(self.id, reg, value),
            )
        }
    }
    fn get_sys(&self, reg: u16) -> u64 {
        let mut v = 0;
        unsafe {
            check(
                &format!("get_sys_reg {reg:#x}"),
                hv_vcpu_get_sys_reg(self.id, reg, &mut v),
            )
        };
        v
    }
    fn set(&self, reg: u32, value: u64) {
        unsafe { check("set_reg", hv_vcpu_set_reg(self.id, reg, value)) }
    }
    fn get(&self, reg: u32) -> u64 {
        let mut v = 0;
        unsafe { check("get_reg", hv_vcpu_get_reg(self.id, reg, &mut v)) };
        v
    }
}

impl Drop for Vcpu {
    fn drop(&mut self) {
        unsafe { hv_vcpu_destroy(self.id) };
    }
}

#[derive(Default, Debug)]
struct Outcome {
    x: [u64; 16],
    guest_faults: Vec<(u64, u64, u64)>, // ESR, ELR, FAR of exceptions taken to guest EL1
    host_traps: Vec<(u64, u64)>,        // syndrome, PC of exits other than our HVCs
    forwarded_svcs: u64,
    elapsed: Duration,
    since_stamp: Option<Duration>,
}

fn run(v: &Vcpu, test: Test, args: &[(u32, u64)]) -> Outcome {
    for &(reg, value) in args {
        v.set(reg, value);
    }
    v.set(HV_REG_PC, GUEST_BASE + test.offset());
    v.set(HV_REG_CPSR, if test.el0() { 0x3c0 } else { 0x3c5 }); // EL0t / EL1h, DAIF masked
    let mut out = Outcome::default();
    let start = Instant::now();
    let mut stamp = None;
    loop {
        unsafe { check("hv_vcpu_run", hv_vcpu_run(v.id)) };
        let exit = unsafe { &*v.exit };
        if exit.reason != HV_EXIT_REASON_EXCEPTION {
            continue;
        }
        let syndrome = exit.exception.syndrome;
        let ec = syndrome >> 26;
        let pc = v.get(HV_REG_PC);
        if ec == 0x16 {
            // HVC: make sure PC is past the instruction whichever convention HVF uses.
            let imm = syndrome & 0xffff;
            if pc >= GUEST_BASE && read32(pc - GUEST_BASE) == 0xd400_0002 | ((imm as u32) << 5) {
                v.set(HV_REG_PC, pc + 4);
            }
            match imm {
                0 | 2 => break,
                1 => out.guest_faults.push((v.get(9), v.get(10), v.get(11))),
                4 => stamp = Some(Instant::now()),
                6 => out.forwarded_svcs += 1,
                _ => panic!("unexpected hvc #{imm}"),
            }
        } else {
            out.host_traps.push((syndrome, pc));
            if ec == 0x18 && out.host_traps.len() < 100_000 {
                // Trapped system register access: emulate a read as 0 and skip it.
                let rt = ((syndrome >> 5) & 0x1f) as u32;
                if syndrome & 1 == 1 && rt != 31 {
                    v.set(HV_REG_X0 + rt, 0);
                }
                v.set(HV_REG_PC, pc + 4);
                continue;
            }
            break;
        }
    }
    out.elapsed = start.elapsed();
    out.since_stamp = stamp.map(|s| s.elapsed());
    for i in 0..16 {
        out.x[i] = v.get(HV_REG_X0 + i as u32);
    }
    out
}

fn bits(v: u64, lo: u32, width: u32) -> u64 {
    (v >> lo) & ((1 << width) - 1)
}

fn traps(o: &Outcome) -> String {
    if o.host_traps.is_empty() {
        return "none".into();
    }
    let (s, pc) = o.host_traps[0];
    format!(
        "{} (first: EC {:#x}, ISS {:#x}, pc {pc:#x})",
        o.host_traps.len(),
        s >> 26,
        s & 0x1ff_ffff
    )
}

pub fn main() {
    // ---- VM with the largest IPA space HVF allows ----
    let (mut default_ipa, mut max_ipa) = (0, 0);
    unsafe {
        check(
            "get_default_ipa",
            hv_vm_config_get_default_ipa_size(&mut default_ipa),
        );
        check("get_max_ipa", hv_vm_config_get_max_ipa_size(&mut max_ipa));
        let config = hv_vm_config_create();
        check("set_ipa_size", hv_vm_config_set_ipa_size(config, max_ipa));
        check("hv_vm_create", hv_vm_create(config));
    }
    println!("## VM");
    println!(
        "IPA size: default {default_ipa} bits, max {max_ipa} bits (VM created with {max_ipa})"
    );

    let layout = std::alloc::Layout::from_size_align(GUEST_SIZE, 0x4000).unwrap();
    unsafe {
        MEM = Mem(std::alloc::alloc_zeroed(layout));
        let blob = guest::blob();
        std::ptr::copy_nonoverlapping(blob.as_ptr(), mem(), blob.len());
        check(
            "hv_vm_map",
            hv_vm_map(
                mem().cast(),
                GUEST_BASE,
                GUEST_SIZE,
                HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC,
            ),
        );
    }
    build_page_tables();
    let v = Vcpu::new();

    // ---- ID registers as the guest sees them ----
    let o = run(&v, Test::Ids, &[]);
    let [isar1, pfr1, mmfr0, cntfrq, isar0, _mmfr1, ..] = o.x;
    println!("\n## Guest-visible features (MRS at EL1)");
    println!(
        "ID_AA64ISAR1_EL1 {isar1:#018x}: APA {} API {} GPA {} GPI {} LRCPC {}",
        bits(isar1, 4, 4),
        bits(isar1, 8, 4),
        bits(isar1, 24, 4),
        bits(isar1, 28, 4),
        bits(isar1, 20, 4)
    );
    println!(
        "  host view via hv_vcpu_get_sys_reg: {:#018x}",
        v.get_sys(ID_AA64ISAR1_EL1)
    );
    println!(
        "ID_AA64ISAR0_EL1 {isar0:#018x}: ATOMIC {}",
        bits(isar0, 20, 4)
    );
    println!("ID_AA64PFR1_EL1  {pfr1:#018x}: MTE {}", bits(pfr1, 8, 4));
    println!(
        "ID_AA64MMFR0_EL1 {mmfr0:#018x}: PARange {} TGran4 {:#x} TGran16 {:#x} TGran64 {:#x} (0 = supported, 0xf = not)",
        bits(mmfr0, 0, 4),
        bits(mmfr0, 28, 4),
        bits(mmfr0, 20, 4),
        bits(mmfr0, 24, 4)
    );
    println!(
        "CNTFRQ_EL0 = {cntfrq} Hz ({:.2} MHz); Switch 2 expects 31.25 MHz",
        cntfrq as f64 / 1e6
    );
    println!("host traps: {}", traps(&o));

    // ---- counters ----
    println!("\n## Counter");
    let o = run(&v, Test::CntpctEl1, &[]);
    println!(
        "EL1 CNTPCT {:#x}, CNTVCT {:#x} (diff {}), host traps: {}",
        o.x[0],
        o.x[1],
        o.x[0] as i64 - o.x[1] as i64,
        traps(&o)
    );
    let o = run(&v, Test::Rate, &[(HV_REG_X0 + 5, 500_000_000)]);
    let wall = o.since_stamp.unwrap().as_secs_f64();
    let rate = (o.x[1] - o.x[0]) as f64 / wall;
    println!(
        "CNTVCT measured rate: {:.3} MHz over {:.3} s of wall time",
        rate / 1e6,
        wall
    );

    let freq = cntfrq as f64;
    let n = 10_000_000u64;
    // With EL0PCTEN set the guest kernel allows the read, but HVF traps the physical counter to
    // the host anyway: every read becomes a VM exit. Time it host-side.
    let n_host = 50_000u64;
    v.sys(CNTKCTL_EL1, CNTKCTL_EL0_ALL);
    let o = run(&v, Test::El0Cntpct, &[(HV_REG_X0 + 5, n_host)]);
    let native_ns = if o.host_traps.is_empty() {
        o.x[0] as f64 / freq / n_host as f64 * 1e9
    } else {
        o.elapsed.as_secs_f64() / o.host_traps.len() as f64 * 1e9
    };
    println!(
        "EL0 CNTPCT read, guest allows it:       {native_ns:7.2} ns/read  (host traps: {})",
        traps(&o)
    );

    let n_trap = 1_000_000u64;
    v.sys(CNTKCTL_EL1, CNTKCTL_EL0_VCT_ONLY);
    let o = run(&v, Test::El0Cntpct, &[(HV_REG_X0 + 5, n_trap)]);
    let trap_ns = o.x[0] as f64 / freq / n_trap as f64 * 1e9;
    println!(
        "EL0 CNTPCT read, trapped to guest EL1:  {trap_ns:7.2} ns/read  (host traps: {}, guest faults: {})",
        traps(&o),
        o.guest_faults.len()
    );
    v.sys(CNTKCTL_EL1, CNTKCTL_EL0_ALL);

    let scale = ((SWITCH2_CNTFRQ / freq - 1.0) * 2f64.powi(64)) as u64;
    let o = run(
        &v,
        Test::El0Patched,
        &[(HV_REG_X0 + 5, n), (HV_REG_X0 + 12, scale)],
    );
    let patched_ns = o.x[0] as f64 / freq / n as f64 * 1e9;
    println!(
        "EL0 counter via patched stub (rescaled): {patched_ns:7.2} ns/read  (last value {:#x})",
        o.x[1]
    );
    let reads_per_sec = 17e6;
    println!(
        "At Mario Kart World's ~17M reads/s: as-is {:.0}%, trapped to guest EL1 {:.0}%, patched {:.1}% of one core",
        reads_per_sec * native_ns / 1e7,
        reads_per_sec * trap_ns / 1e7,
        reads_per_sec * patched_ns / 1e7
    );

    // ---- SVC round trip through the host (HLE kernel calls) ----
    println!("\n## SVC from EL0 to host and back");
    let n_svc = 200_000u64;
    let o = run(&v, Test::El0Svc, &[(HV_REG_X0 + 5, n_svc)]);
    let us = o.elapsed.as_secs_f64() / o.forwarded_svcs as f64 * 1e6;
    println!(
        "{} svcs forwarded: {us:.2} us each; Mario Kart World's ~1,400 svcs/frame = {:.2} ms/frame",
        o.forwarded_svcs,
        us * 1400.0 / 1000.0
    );

    // ---- pointer authentication ----
    println!("\n## Pointer authentication (keys set from the host)");
    let ptr = GUEST_BASE + DATA;
    let pac_args = [(HV_REG_X0 + 1, ptr), (HV_REG_X0 + 2, 0x1234)];
    let o = run(&v, Test::Pauth, &pac_args);
    println!(
        "pacia {ptr:#x} -> {:#x} (signed: {})",
        o.x[0],
        o.x[0] != ptr
    );
    println!("autia -> {:#x} (round trip ok: {})", o.x[3], o.x[3] == ptr);
    for (esr, elr, _) in &o.guest_faults {
        println!(
            "tampered autia: guest exception EC {:#x} at {elr:#x} (0x1c = FPAC failure)",
            esr >> 26
        );
    }
    if o.guest_faults.is_empty() {
        println!("tampered autia: no exception (x4 = {:#x})", o.x[4]);
    }
    println!("pacibsp/retab call returned normally: {}", o.x[6] == 42);
    println!("host traps: {}", traps(&o));
    let reference = (o.x[0], o.x[5]);
    let mut consistent = true;
    for qos in [
        QOS_CLASS_BACKGROUND,
        QOS_CLASS_USER_INTERACTIVE,
        QOS_CLASS_BACKGROUND,
        QOS_CLASS_USER_INTERACTIVE,
    ] {
        let got = std::thread::spawn(move || {
            unsafe { pthread_set_qos_class_self_np(qos, 0) };
            let v = Vcpu::new();
            let mut results = Vec::new();
            for _ in 0..50 {
                let o = run(&v, Test::Pauth, &pac_args);
                results.push((o.x[0], o.x[5]));
            }
            results
        })
        .join()
        .unwrap();
        consistent &= got.iter().all(|&r| r == reference);
    }
    println!(
        "same PAC on fresh vCPUs across E-core (background) and P-core (interactive) threads, 200 runs: {consistent}"
    );

    // ---- atomics and acquire loads at EL0 ----
    println!("\n## ARMv8.1+ instructions at EL0");
    let o = run(&v, Test::El0Atomics, &[(HV_REG_X0 + 1, GUEST_BASE + DATA)]);
    let ok = o.x[0] == 5 && o.x[3] == 5 && o.x[6] == 7 && o.x[12] == 17 && o.x[13] == 1;
    println!(
        "LDAPR/CAS/LDADD/SWP results correct: {ok} (faults: {}, host traps: {})",
        o.guest_faults.len(),
        traps(&o)
    );

    // ---- MTE ----
    println!("\n## MTE");
    let o = run(&v, Test::Mte, &[(HV_REG_X0 + 1, GUEST_BASE + DATA)]);
    match o.guest_faults.first() {
        Some((esr, elr, _)) => println!(
            "STG raised guest exception EC {:#x} at {elr:#x} (0x0 = undefined), execution continued: {}",
            esr >> 26,
            o.x[0] == 1
        ),
        None => println!(
            "STG executed without an exception (host traps: {})",
            traps(&o)
        ),
    }

    // ---- 4 KiB stage-1 pages on a 16 KiB host ----
    println!("\n## 4 KiB guest pages");
    let o = run(
        &v,
        Test::Remap,
        &[
            (HV_REG_X0 + 1, REMAP_VA + 0x1000),
            (HV_REG_X0 + 2, REMAP_VA + 0x2000),
        ],
    );
    let ok = o.x[0] == 0xC3C3_C3C3_0000_3000 && o.x[3] == 0xC1C1_C1C1_0000_1000;
    println!(
        "VA +0x1000 -> PA +0x3000 and VA +0x2000 -> PA +0x1000 inside one 16 KiB page: {ok} ({:#x}, {:#x})",
        o.x[0], o.x[3]
    );

    // ---- TLB invalidation across vCPUs ----
    // Both vCPUs share the same stage-1 tables. The host repoints a page-table entry without
    // flushing, and we watch when each vCPU starts seeing the new page.
    println!("\n## TLB invalidation across vCPUs");
    const TLB_VA: u64 = REMAP_VA + 0x3000;
    const OLD: u64 = 0xC1C1_C1C1_0000_1000;
    const NEW: u64 = 0xC3C3_C3C3_0000_3000;
    const SPIN_TRIES: u64 = 4_000_000_000;
    let l3b = PT_BASE + 0x3000;
    let name = |value: u64| match value {
        OLD => "old page",
        NEW => "new page",
        _ => "unexpected value",
    };
    let (tx, rx) = std::sync::mpsc::channel::<Test>();
    let (rtx, rrx) = std::sync::mpsc::channel::<u64>();
    let other = std::thread::spawn(move || {
        let v = Vcpu::new();
        let args = [
            (HV_REG_X0 + 1, TLB_VA),
            (HV_REG_X0 + 2, NEW),
            (HV_REG_X0 + 3, SPIN_TRIES),
        ];
        for test in rx {
            let o = run(&v, test, &args);
            if rtx.send(o.x[0]).is_err() {
                break;
            }
        }
    });
    let on_b = |test: Test| {
        tx.send(test).ok();
        rrx.recv().unwrap_or(0)
    };
    let on_a = |test: Test| run(&v, test, &[(HV_REG_X0 + 1, TLB_VA)]).x[0];
    let old_entry = || l3_entry(l3b, TLB_VA, MARKERS + 0x1000, AP_EL1_RW | PXN | UXN);
    let new_entry = || l3_entry(l3b, TLB_VA, MARKERS + 0x3000, AP_EL1_RW | PXN | UXN);

    // 1. Does a cached translation survive a trip out to the host and back?
    old_entry();
    on_a(Test::TlbiIs);
    on_b(Test::TlbiLocal);
    let (a0, b0) = (on_a(Test::Read), on_b(Test::Read));
    new_entry();
    let (a1, b1) = (on_a(Test::Read), on_b(Test::Read));
    println!(
        "exit to host and re-enter: before the change A {}, B {}; after it (no flush) A {}, B {}",
        name(a0),
        name(b0),
        name(a1),
        name(b1)
    );

    // 2. A vCPU that stays inside the guest while the entry changes. Control rounds never
    //    flush; the others flush from vCPU A after 10 ms. If the broadcast form works, B sees the
    //    new page within microseconds of A's flush; without a flush it takes however long the
    //    host needs to interrupt B for its own reasons.
    let spin_round = |flush: bool| -> (bool, Duration) {
        old_entry();
        on_a(Test::TlbiIs);
        on_b(Test::TlbiLocal);
        tx.send(Test::SpinRead).ok();
        std::thread::sleep(Duration::from_millis(20));
        new_entry();
        let changed = Instant::now();
        if rrx.recv_timeout(Duration::from_millis(10)).is_ok() {
            return (false, changed.elapsed());
        }
        let start = if flush {
            on_a(Test::TlbiIs);
            Instant::now()
        } else {
            changed
        };
        rrx.recv().ok();
        (flush, start.elapsed())
    };
    let report = |label: &str, times: &mut Vec<Duration>| {
        times.sort();
        if let (Some(first), Some(last)) = (times.first(), times.last()) {
            println!(
                "{label}: {} rounds, min {:?}, median {:?}, max {:?}",
                times.len(),
                first,
                times[times.len() / 2],
                last
            );
        } else {
            println!("{label}: 0 rounds");
        }
    };
    let (mut early, mut unflushed, mut flushed) = (Vec::new(), Vec::new(), Vec::new());
    for round in 0..40 {
        let flush = round % 2 == 0;
        match spin_round(flush) {
            (true, t) => flushed.push(t),
            (false, t) if t < Duration::from_millis(10) => early.push(t),
            (false, t) => unflushed.push(t),
        }
    }
    report("B saw the new page within 10 ms on its own", &mut early);
    report(
        "no flush: time from the change until B saw the new page",
        &mut unflushed,
    );
    report(
        "broadcast flush on A: time from the flush until B saw the new page",
        &mut flushed,
    );
    drop(tx);
    other.join().ok();
}

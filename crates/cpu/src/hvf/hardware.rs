//! Hardware tests that need the backend's private layout: what EL0 can do with the upper half
//! of the address space, where the shim lives, and what stage 2 refuses. The tests of the
//! public interface are in `tests/hvf.rs`.
//!
//! Like those, these need a Mac with Apple Silicon and skip themselves only when the `CI`
//! environment variable is set.

use std::sync::{Mutex, MutexGuard, PoisonError};

use starquake_memory::stage1::{Stage1Tables, VA_SPACE_END};
use starquake_memory::{BackingRange, GuestMemory, PAGE_SIZE, Perms, Translation};

use super::*;
use crate::{FaultStatus, host_counter};

/// One VM per process: every test here holds this while it has one.
static VM_LOCK: Mutex<()> = Mutex::new(());

/// Where the probes are mapped, and the id of the address space.
const CODE: u64 = 0x40_0000;
const ASID: u16 = 7;

/// Each probe takes the address to try in x0.
const PROBES: [u32; 5] = [
    0xf940_0001, // 0x00  ldr x1, [x0]      a load
    0xd400_0021, // 0x04  svc #1
    0xf900_0001, // 0x08  str x1, [x0]      a store
    0xd400_0041, // 0x0c  svc #2
    0xd61f_0000, // 0x10  br  x0            a fetch
];
const LOAD: u64 = CODE;
const STORE: u64 = CODE + 0x08;
const FETCH: u64 = CODE + 0x10;

struct Machine {
    backend: HvfBackend,
    tables: Stage1Tables<PhysicalMemory>,
    vcpu: HvfVcpu,
    _lock: MutexGuard<'static, ()>,
}

impl Machine {
    /// A VM whose one address space holds the probes, read-execute, on tables in the first
    /// half of the table pool; `None` on hosted CI.
    fn new() -> Option<Self> {
        if std::env::var_os("CI").is_some() {
            return None;
        }
        let lock = VM_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let backend = HvfBackend::new(RamConfig::default()).unwrap();
        let pool = backend.table_pool();
        let half = BackingRange::new(pool.base(), pool.size() / 2).unwrap();
        let tables = Stage1Tables::new(backend.physical(), half).unwrap();
        let code = BackingRange::new(backend.ram().base(), PAGE_SIZE).unwrap();
        tables.map(CODE, code, Perms::ReadExecute).unwrap();
        let bytes: Vec<u8> = PROBES.iter().flat_map(|w| w.to_le_bytes()).collect();
        backend.physical().write(code.base(), &bytes).unwrap();
        let mut vcpu = backend.create_vcpu().unwrap();
        vcpu.set_address_space(tables.root(), ASID).unwrap();
        Some(Self {
            backend,
            tables,
            vcpu,
            _lock: lock,
        })
    }

    /// Runs the probe at `pc` on `address`.
    fn probe(&mut self, pc: u64, address: u64) -> Result<Exit, Error> {
        let mut regs = self.vcpu.regs()?;
        regs.pc = pc;
        regs.x[0] = address;
        regs.x[1] = 0x5eed;
        self.vcpu.set_regs(&regs)?;
        self.vcpu.run()
    }

    /// What the three probes report for `address`, which EL0 must not reach: the fault status
    /// of the load, the store and the fetch. Each must be the right kind of exit at `address`.
    fn refused(&mut self, address: u64) -> [Option<FaultStatus>; 3] {
        let load = match self.probe(LOAD, address) {
            Ok(Exit::DataAbort {
                address: at,
                write: false,
                syndrome,
            }) if at == address => syndrome.fault(),
            _ => None,
        };
        let store = match self.probe(STORE, address) {
            Ok(Exit::DataAbort {
                address: at,
                write: true,
                syndrome,
            }) if at == address => syndrome.fault(),
            _ => None,
        };
        let fetch = match self.probe(FETCH, address) {
            Ok(Exit::InstructionAbort {
                address: at,
                syndrome,
            }) if at == address => syndrome.fault(),
            _ => None,
        };
        [load, store, fetch]
    }
}

#[test]
fn el0_cannot_read_write_or_execute_the_shim_or_its_scratch_page() {
    let Some(mut machine) = Machine::new() else {
        return;
    };
    // Both pages are mapped, for EL1 only: every EL0 access is a permission fault.
    let denied = Some(FaultStatus::Permission { level: 3 });
    for page in [SHIM_VA, SCRATCH_VA] {
        for address in [page, page + PAGE - 8] {
            assert_eq!(machine.refused(address), [denied; 3], "{address:#x}");
        }
    }
    // The shim still works afterwards: the probes' own `svc` reaches the host through it, and
    // the load really loaded.
    let code = machine.backend.ram().base();
    assert_eq!(machine.probe(LOAD, CODE), Ok(Exit::Svc(1)));
    let mut first_word = [0; 8];
    machine
        .backend
        .physical()
        .read(code, &mut first_word)
        .unwrap();
    assert_eq!(
        machine.vcpu.regs().unwrap().x[1],
        u64::from_le_bytes(first_word)
    );
}

#[test]
fn the_rest_of_the_upper_half_and_the_gap_between_the_halves_are_not_mapped() {
    let Some(mut machine) = Machine::new() else {
        return;
    };
    let unmapped = |level| Some(FaultStatus::Translation { level });
    // Next to the shim and the scratch page, under the same level-3 table. (The shim is that
    // table's first page; the page below it belongs to another, which does not exist.)
    assert_eq!(machine.refused(SHIM_VA - PAGE), [unmapped(2); 3]);
    for address in [SHIM_VA + PAGE, SCRATCH_VA - PAGE, SCRATCH_VA + PAGE] {
        assert_eq!(machine.refused(address), [unmapped(3); 3], "{address:#x}");
    }
    // The first address of the upper half, far from anything mapped there.
    assert_eq!(machine.refused(UPPER_HALF), [unmapped(1); 3]);
    // Between the halves no table is consulted at all. Measured here: the fault is a
    // translation fault reported at level 0, at both ends of the gap. This is what the
    // software walker calls `WalkError::OutOfRange`.
    assert_eq!(UPPER_HALF, VA_SPACE_END.wrapping_neg());
    for address in [VA_SPACE_END, UPPER_HALF - 8] {
        assert_eq!(
            machine.tables.walk(address),
            Err(starquake_memory::stage1::WalkError::OutOfRange { va: address })
        );
        assert_eq!(machine.refused(address), [unmapped(0); 3], "{address:#x}");
    }
}

/// Maps a page at guest-physical `target` into a fresh machine's address space, readable and
/// writable, and has the guest load from it and then store to it. Returns what the load and
/// the store came to, and the address the store would have written.
fn load_then_store_through(target: u64) -> Option<(Result<Exit, Error>, Result<Exit, Error>)> {
    let mut machine = Machine::new()?;
    let va = 0x10_0000_0000;
    let backing = BackingRange::new(target, PAGE_SIZE).unwrap();
    machine.tables.map(va, backing, Perms::ReadWrite).unwrap();
    let load = machine.probe(LOAD, va + 0x18);
    let store = machine.probe(STORE, va + 0x18);
    Some((load, store))
}

/// The error a guest store to guest-physical `address` must stop the vCPU with.
fn assert_stopped_by_stage_2(store: &Result<Exit, Error>, address: u64) {
    let message = match store {
        Err(Error::Host(message)) => message.as_str(),
        _ => "",
    };
    let expected = format!(
        "the guest reached guest-physical address {address:#x} in a way stage 2 does not allow ("
    );
    assert!(message.starts_with(&expected), "{store:?}");
    assert!(message.ends_with(&format!("pc {STORE:#x})")), "{store:?}");
}

#[test]
fn a_guest_store_to_a_translation_table_is_stopped_by_stage_2() {
    let Some(machine) = Machine::new() else {
        return;
    };
    // `Stage1Tables` refuses to map its own pool. A page of the *other* half of the table pool
    // it does map, writable: the tables say yes, stage 2 says no.
    let pool = machine.backend.table_pool();
    assert_eq!(
        machine.tables.map(0x10_0000_0000, pool, Perms::ReadWrite),
        Err(starquake_memory::MapError::OverlapsTables {
            pa: pool.base(),
            size: pool.size()
        })
    );
    drop(machine);
    let target = pool.base() + pool.size() / 2;
    let (load, store) = load_then_store_through(target).unwrap();
    // A load is allowed: the guest's own walker reads the tables.
    assert_eq!(load, Ok(Exit::Svc(1)));
    assert_stopped_by_stage_2(&store, target + 0x18);
}

#[test]
fn a_guest_store_to_the_shim_or_its_tables_is_stopped_by_stage_2() {
    // Even if a guest address space wrongly mapped the backend's own pages, writable, the
    // guest could not change the shim or the tables that map it.
    for offset in [CODE_OFFSET, SYSTEM_TABLES_OFFSET] {
        let target = SYSTEM_BASE + offset;
        let Some((load, store)) = load_then_store_through(target) else {
            return;
        };
        assert_eq!(load, Ok(Exit::Svc(1)), "{target:#x}");
        assert_stopped_by_stage_2(&store, target + 0x18);
    }
    // The shim's bytes are what the backend wrote.
    let [(_, shim), _] = system_image().unwrap();
    let Some(machine) = Machine::new() else {
        return;
    };
    let mut bytes = vec![0; shim.len()];
    machine
        .backend
        .vm
        .system
        .read(CODE_OFFSET, &mut bytes)
        .unwrap();
    assert_eq!(bytes, shim);
}

/// `hv_sys_reg_t` of `CNTFRQ_EL0`: `op0` 3, `op1` 3, `CRn` 14, `CRm` 0, `op2` 0, encoded as
/// `ffi`'s other system registers are. Not a register the SDK headers name (`hv_vcpu_types.h`);
/// the test below checks that Hypervisor.framework refuses it.
const CNTFRQ_EL0: u16 = (3 << 14) | (3 << 11) | (14 << 7);
/// `HV_BAD_ARGUMENT` (`hv_error.h`).
const HV_BAD_ARGUMENT: u32 = 0xfae9_4003;
/// `msr cntfrq_el0, x0`, `mrs x1, cntfrq_el0`, `hvc #HVC_FLUSH_DONE` (assembled with LLVM).
const WRITE_CNTFRQ: [u32; 3] = [0xd51b_e000, 0xd53b_e001, 0xd400_0062];
/// `mrs x1, cntfrq_el0`, `svc #1`, as EL0 code.
const READ_CNTFRQ: [u32; 2] = [0xd53b_e001, 0xd400_0021];

#[test]
fn the_guests_counter_frequency_cannot_be_changed() {
    let Some(mut machine) = Machine::new() else {
        return;
    };
    let host_hz = machine.backend.counter_frequency();
    let wanted = host_hz + 1;
    assert_eq!(WRITE_CNTFRQ[2], shim::hvc(shim::HVC_FLUSH_DONE));
    // Hypervisor.framework refuses the register outright, for reading and writing.
    let vcpu = &machine.vcpu;
    // SAFETY: called on the owning thread with a live id.
    let set = unsafe { ffi::hv_vcpu_set_sys_reg(vcpu.id, CNTFRQ_EL0, wanted) };
    assert_eq!(u32::from_ne_bytes(set.to_ne_bytes()), HV_BAD_ARGUMENT);
    assert_eq!(
        vcpu.sys(CNTFRQ_EL0),
        Err(Error::Host(format!(
            "hv_vcpu_get_sys_reg failed: {HV_BAD_ARGUMENT:#x}"
        )))
    );
    // The guest's EL1 cannot write it either: the `msr` is UNDEFINED there. The code goes in
    // the shim's page past the shim (EL1 can execute only there), and the UNDEFINED lands in
    // the shim's unused "current EL" vector, which reports itself with `hvc #1`.
    let at = shim::LEN.next_multiple_of(16);
    assert!(at + 4 * WRITE_CNTFRQ.len() <= usize::try_from(PAGE).unwrap());
    let bytes: Vec<u8> = WRITE_CNTFRQ.iter().flat_map(|w| w.to_le_bytes()).collect();
    let at = u64::try_from(at).unwrap();
    machine
        .backend
        .vm
        .system
        .write(CODE_OFFSET + at, &bytes)
        .unwrap();
    vcpu.set_reg(ffi::REG_X0, wanted).unwrap();
    vcpu.set_reg(ffi::REG_PC, SHIM_VA + at).unwrap();
    vcpu.set_reg(ffi::REG_CPSR, shim::CPSR_EL1H_MASKED).unwrap();
    let exit = vcpu.enter().unwrap();
    assert_eq!(exit.reason, ffi::EXIT_REASON_EXCEPTION);
    assert_eq!(
        hvc_immediate(exit.exception.syndrome),
        Some(shim::HVC_UNEXPECTED_VECTOR)
    );
    assert_eq!(vcpu.sys(ffi::ELR_EL1).unwrap(), SHIM_VA + at, "the msr");
    let esr = Syndrome::from_esr(vcpu.sys(ffi::ESR_EL1).unwrap());
    assert_eq!(esr.class(), ExceptionClass::Unknown);
    vcpu.park().unwrap();
    // And EL0 still reads the host's rate.
    let page = machine.backend.ram().base() + PAGE_SIZE;
    let bytes: Vec<u8> = READ_CNTFRQ.iter().flat_map(|w| w.to_le_bytes()).collect();
    machine.backend.physical().write(page, &bytes).unwrap();
    let code = CODE + PAGE_SIZE;
    machine
        .tables
        .map(
            code,
            BackingRange::new(page, PAGE_SIZE).unwrap(),
            Perms::ReadExecute,
        )
        .unwrap();
    assert_eq!(machine.probe(code, 0), Ok(Exit::Svc(1)));
    assert_eq!(machine.vcpu.regs().unwrap().x[1], host_hz);
}

/// `hvc` immediate the recovery test plants in the shim: one the shim never uses.
const HVC_PLANTED: u16 = 7;
/// `svc #0`.
const SVC_0: u32 = 0xd400_0001;

/// `mrs x<rt>, cntpct_el0` (Arm ARM, `MRS`: `op0` 3, `op1` 3, `CRn` 14, `CRm` 0, `op2` 1;
/// `Rt` in bits 4:0).
fn mrs_cntpct(rt: u32) -> u32 {
    0xd53b_e020 | rt
}

/// A distinct value for register `n`, far from any counter reading.
fn marker(n: usize) -> u64 {
    let n = u64::try_from(n).unwrap();
    0x5a5a_0000_0000_0000 | (n << 8) | n
}

/// Every offset on the shim's path for a trapped `mrs x<rt>, cntpct_el0`: the dispatcher from
/// the vector to its branch, then the stub of `rt` to its `eret`.
fn counter_path(rt: u32) -> Vec<usize> {
    let stub = shim::STUBS + usize::try_from(rt).unwrap() * shim::STUB_LEN;
    (shim::EL0_SYNC..=shim::ELR_BUMPED)
        .step_by(4)
        .chain((stub..=stub + shim::STUB_ERET).step_by(4))
        .collect()
}

#[test]
fn a_counter_read_stopped_anywhere_on_the_shims_path_is_recovered_exactly() {
    // An interrupt can stop a vCPU at any instruction of the shim. To stop it at each one in
    // turn, the test plants an `hvc` there, lets the guest's counter read trap into the shim
    // and run up to it, puts the original word back and treats the vCPU as interrupted at that
    // instruction: exactly the state an interrupt there leaves. Recovery must give back the
    // guest's registers as they were (all but `Rt` when the read is complete), and the next run
    // must finish the read with the scaled count.
    let Some(mut machine) = Machine::new() else {
        return;
    };
    let host_hz = machine.backend.counter_frequency();
    let ratio = starquake_loader::patch::CounterRescale {
        host_hz,
        guest_hz: host_hz * 4,
    }
    .ratio()
    .unwrap();
    let scale = CounterScale::new(ratio.int(), ratio.frac());
    let page = machine.backend.ram().base() + PAGE_SIZE;
    let code = CODE + PAGE_SIZE;
    machine
        .tables
        .map(
            code,
            BackingRange::new(page, PAGE_SIZE).unwrap(),
            Perms::ReadExecute,
        )
        .unwrap();
    machine.vcpu.set_counter_scale(scale).unwrap();
    let mut stopped_at_eret = 0;
    // A register of the guest's own, the three the shim borrows, and xzr.
    let registers = [1, 9, 10, 11, 31];
    for rt in registers {
        let bytes: Vec<u8> = [mrs_cntpct(rt), SVC_0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        machine.backend.physical().write(page, &bytes).unwrap();
        for offset in counter_path(rt) {
            let context = format!("x{rt}, shim offset {offset:#x}");
            let at = CODE_OFFSET + u64::try_from(offset).unwrap();
            let system = &machine.backend.vm.system;
            system
                .write(at, &shim::hvc(HVC_PLANTED).to_le_bytes())
                .unwrap();
            let vcpu = &mut machine.vcpu;
            let mut regs = vcpu.regs().unwrap();
            for (n, x) in regs.x.iter_mut().enumerate() {
                *x = marker(n);
            }
            regs.pc = code;
            vcpu.set_regs(&regs).unwrap();
            let before = host_counter::read();
            let exit = vcpu.enter().unwrap();
            assert_eq!(exit.reason, ffi::EXIT_REASON_EXCEPTION, "{context}");
            assert_eq!(
                hvc_immediate(exit.exception.syndrome),
                Some(HVC_PLANTED),
                "{context}"
            );
            system
                .write(at, &shim::WORDS[offset / 4].to_le_bytes())
                .unwrap();
            vcpu.set_reg(ffi::REG_PC, SHIM_VA + at - CODE_OFFSET)
                .unwrap();
            assert_eq!(vcpu.interrupted(), Ok(Exit::Interrupted), "{context}");
            let recovered = vcpu.regs().unwrap();
            let rt_index = usize::try_from(rt).unwrap();
            for (n, &x) in recovered.x.iter().enumerate() {
                if n != rt_index {
                    assert_eq!(x, marker(n), "x{n} recovered, {context}");
                }
            }
            // Stopped at the stub's `eret` the read is complete; anywhere else it runs again.
            let complete = offset == shim::STUBS + rt_index * shim::STUB_LEN + shim::STUB_ERET;
            stopped_at_eret += usize::from(complete);
            assert_eq!(
                recovered.pc,
                if complete { code + 4 } else { code },
                "{context}"
            );
            assert_eq!(vcpu.run(), Ok(Exit::Svc(0)), "{context}");
            let after = host_counter::read();
            let x = vcpu.regs().unwrap().x;
            for (n, &value) in x.iter().enumerate() {
                if n == rt_index {
                    assert!(
                        ratio.scale(before) <= value && value <= ratio.scale(after),
                        "x{rt} read {value}, not in {}..={}, {context}",
                        ratio.scale(before),
                        ratio.scale(after)
                    );
                } else {
                    assert_eq!(value, marker(n), "x{n} after the read, {context}");
                }
            }
        }
    }
    assert_eq!(
        stopped_at_eret,
        registers.len(),
        "each stub was stopped at its eret once"
    );
}

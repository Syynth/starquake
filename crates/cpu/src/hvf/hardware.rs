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
use crate::FaultStatus;

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

#[test]
fn a_guest_store_to_a_translation_table_is_stopped_by_stage_2() {
    let Some(mut machine) = Machine::new() else {
        return;
    };
    // `Stage1Tables` refuses to map its own pool. Map a page of the *other* half of the table
    // pool instead, writable: the tables say yes, stage 2 says no.
    let pool = machine.backend.table_pool();
    let target = pool.base() + pool.size() / 2;
    let va = 0x10_0000_0000;
    let backing = BackingRange::new(target, PAGE_SIZE).unwrap();
    assert_eq!(
        machine.tables.map(va, pool, Perms::ReadWrite),
        Err(starquake_memory::MapError::OverlapsTables {
            pa: pool.base(),
            size: pool.size()
        })
    );
    machine.tables.map(va, backing, Perms::ReadWrite).unwrap();
    // A load is allowed (the tables are readable; the guest's own walker reads them).
    assert_eq!(machine.probe(LOAD, va + 0x18), Ok(Exit::Svc(1)));
    // A store stops the vCPU with an error that names the guest-physical address.
    let error = machine.probe(STORE, va + 0x18).unwrap_err();
    let Error::Host(message) = &error else {
        unreachable!("{error:?}");
    };
    let expected = format!(
        "the guest reached guest-physical address {:#x} in a way stage 2 does not allow (",
        target + 0x18
    );
    assert!(message.starts_with(&expected), "{message}");
    assert!(message.ends_with(&format!("pc {STORE:#x})")), "{message}");
    // Nothing was written.
    let mut word = [0xff; 8];
    machine
        .backend
        .physical()
        .read(target + 0x18, &mut word)
        .unwrap();
    assert_eq!(word, [0; 8]);
}

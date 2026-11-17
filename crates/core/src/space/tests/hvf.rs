//! Shared memory on a real vCPU (Hypervisor.framework): the kernel maps a shared memory through
//! the real mapper, and guest code at EL0 reads what the service wrote and writes what the
//! service reads; a read-only mapping refuses a store; an unmapped one, once its flush has run,
//! refuses a load, and the pages keep what they hold.
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM fails
//! the test. Each holds the run's one-VM lock (`crate::system`), as a run does.

use std::sync::PoisonError;

use starquake_cpu::hvf::{HvfBackend, HvfVcpu, PacKeys, PhysicalMemory, RamConfig};
use starquake_cpu::{Backend, Exit, FaultStatus, GuestRegs, Vcpu};
use starquake_kernel::memory::{MemoryPermission, SvcPermission};
use starquake_kernel::process::asid::FIRST_ASID;
use starquake_testkit::a64;

use super::kernel::{CODE, PAGES, SHARED, SIZE, World, raw};
use super::*;
use crate::system::host::ONE_VM;

/// The test program: loads the word at `x1` into `x0`, stores `x2` eight bytes past it, and
/// makes a system call. Five words, so it fits one page.
fn program() -> Vec<u8> {
    a64::to_bytes(&[
        a64::ldr(0, 1, 0).unwrap(),
        a64::str(2, 1, 8).unwrap(),
        a64::svc(0).unwrap(),
    ])
}

/// Runs the test program from its first instruction with `x1` = `address` and `x2` = `value`.
fn run(vcpu: &mut HvfVcpu, address: u64, value: u64) -> Exit {
    let mut regs = GuestRegs {
        pc: CODE,
        ..GuestRegs::default()
    };
    regs.x[1] = address;
    regs.x[2] = value;
    vcpu.set_regs(&regs).unwrap();
    vcpu.run().unwrap()
}

/// What a data abort says about the access and the fault, or `None` for another exit.
fn abort(exit: Exit) -> Option<(u64, bool, Option<FaultStatus>)> {
    match exit {
        Exit::DataAbort {
            address,
            write,
            syndrome,
        } => Some((address, write, syndrome.fault())),
        _ => None,
    }
}

#[test]
fn guest_code_reaches_shared_memory_as_mapped_and_not_after_it_is_unmapped() {
    if std::env::var_os("CI").is_some() {
        return;
    }
    let _one_vm = ONE_VM.lock().unwrap_or_else(PoisonError::into_inner);
    let backend = HvfBackend::new(RamConfig::default()).expect("create the VM");
    let space = AddressSpace::<PhysicalMemory>::new(
        backend.physical(),
        backend.ram(),
        backend.table_pool(),
    )
    .unwrap();
    let mut world = World::over(space);
    // The code: written through the host's view, then made read-execute by the kernel.
    world.space.write(CODE, &program()).unwrap();
    world
        .kernel
        .set_process_memory_permission(CODE, PAGE_SIZE, SvcPermission::ReadExecute)
        .unwrap();
    // A read-write shared memory, and a read-only one just above it.
    let writable = world.shared(PAGES, SvcPermission::ReadWrite);
    let read_only = world.shared(1, SvcPermission::Read);
    let read_only_at = SHARED + SIZE;
    world
        .map(writable.handle, SHARED, SIZE, raw(SvcPermission::ReadWrite))
        .unwrap();
    world
        .map(
            read_only.handle,
            read_only_at,
            PAGE_SIZE,
            raw(SvcPermission::Read),
        )
        .unwrap();
    assert_eq!(
        world.kernel.query_info(read_only_at).unwrap().permission(),
        MemoryPermission::USER_READ
    );
    let first = 0x1122_3344_5566_7788_u64;
    writable
        .pages
        .write(PAGE_SIZE, &first.to_le_bytes())
        .unwrap();
    read_only.pages.write(0, &first.to_le_bytes()).unwrap();

    // Declared after the backend, so it is dropped first, as the VM requires.
    let mut vcpu = backend.create_vcpu().unwrap();
    vcpu.set_address_space(world.space.root(), FIRST_ASID)
        .unwrap();
    vcpu.set_pac_keys(&PacKeys::default()).unwrap();
    vcpu.flush_pending(world.space.tables(), FIRST_ASID)
        .unwrap();

    // Read-write: the guest reads what the service wrote, and the service reads what the guest
    // wrote, in the second page.
    let second = 0x99aa_bbcc_ddee_ff00_u64;
    assert_eq!(run(&mut vcpu, SHARED + PAGE_SIZE, second), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], first);
    let mut word = [0; 8];
    writable.pages.read(PAGE_SIZE + 8, &mut word).unwrap();
    assert_eq!(u64::from_le_bytes(word), second);

    // Read-only: the load works, the store is a permission fault and writes nothing.
    let exit = run(&mut vcpu, read_only_at, second);
    assert_eq!(vcpu.regs().unwrap().x[0], first);
    let Some((address, true, Some(FaultStatus::Permission { .. }))) = abort(exit) else {
        panic!("a store to read-only shared memory ended as {exit:?}");
    };
    assert_eq!(address, read_only_at + 8);
    read_only.pages.read(8, &mut word).unwrap();
    assert_eq!(word, [0; 8]);

    // Unmapped, and its flush run: the load is a translation fault, and the pages keep what
    // they hold.
    world.unmap(writable.handle, SHARED, SIZE).unwrap();
    vcpu.flush_pending(world.space.tables(), FIRST_ASID)
        .unwrap();
    let exit = run(&mut vcpu, SHARED + PAGE_SIZE, 0);
    let Some((address, false, Some(FaultStatus::Translation { .. }))) = abort(exit) else {
        panic!("a load from unmapped shared memory ended as {exit:?}");
    };
    assert_eq!(address, SHARED + PAGE_SIZE);
    writable.pages.read(PAGE_SIZE + 8, &mut word).unwrap();
    assert_eq!(u64::from_le_bytes(word), second);

    // Mapped again elsewhere, the guest sees the same bytes.
    let again = SHARED + 2 * SIZE;
    world
        .map(writable.handle, again, SIZE, raw(SvcPermission::ReadWrite))
        .unwrap();
    assert_eq!(run(&mut vcpu, again + PAGE_SIZE, 0), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], first);
    drop(vcpu);
}

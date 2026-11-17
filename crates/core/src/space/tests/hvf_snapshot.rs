//! A kernel address space restored from a snapshot onto a real vCPU (Hypervisor.framework):
//! saved from an address space over plain memory, restored over the VM's through the real
//! mapper, then used. Guest code at EL0 reads the restored mapped physical memory and heap; the
//! restored kernel maps more physical memory, which guest code reads, queries it, and unmaps
//! it, after which a load from it is a translation fault (`starquake_kernel::memory`'s
//! `space::snapshot` module docs, check 9: the restore maps every block that is not free
//! through the new mapper).
//!
//! What guest memory holds is not part of the address space's snapshot, so the test writes the
//! program and the values through the host's view after the restore, as the guest memory part
//! of a snapshot will.
//!
//! These need a Mac with Apple Silicon. They skip themselves only when the `CI` environment
//! variable is set (hosted runners can't create VMs); anywhere else, failing to create a VM fails
//! the test. Each holds the run's one-VM lock (`crate::system`), as a run does. The program and
//! how it is run are `space/tests/hvf.rs`'s.

use std::sync::PoisonError;

use starquake_cpu::hvf::{HvfBackend, PacKeys, PhysicalMemory, RamConfig};
use starquake_cpu::{Backend, Exit, FaultStatus, Vcpu};
use starquake_kernel::memory::{AddressSpace as KernelSpace, MemoryState, SvcPermission};
use starquake_kernel::process::asid::FIRST_ASID;

use super::hvf::{program, run};
use super::kernel::{CODE, World};
use super::*;
use crate::system::host::ONE_VM;

/// The doubleword at `address`, read through the host's view.
fn read_u64(space: &AddressSpace<PhysicalMemory>, address: u64) -> u64 {
    let mut bytes = [0; 8];
    space.read(address, &mut bytes).unwrap();
    u64::from_le_bytes(bytes)
}

#[test]
fn a_restored_address_space_runs_guest_code_and_maps_queries_and_unmaps() {
    if std::env::var_os("CI").is_some() {
        return;
    }
    // The original, over plain memory: the code's first page made read-execute, a stack, two
    // pages of heap and two pages of physical memory at the alias region's start.
    let mut original = World::over(space());
    let kernel = &mut original.kernel;
    kernel
        .set_process_memory_permission(CODE, PAGE_SIZE, SvcPermission::ReadExecute)
        .unwrap();
    kernel.create_main_thread_stack(PAGE_SIZE).unwrap();
    let heap = kernel.set_heap_size(2 * PAGE_SIZE).unwrap();
    let alias = kernel.regions().alias.start;
    kernel.map_physical_memory(alias, 2 * PAGE_SIZE).unwrap();
    let bytes = kernel.save().unwrap();

    // The restore, over the VM's memory.
    let _one_vm = ONE_VM.lock().unwrap_or_else(PoisonError::into_inner);
    let backend = HvfBackend::new(RamConfig::default()).expect("create the VM");
    let space = Arc::new(
        AddressSpace::<PhysicalMemory>::new(
            backend.physical(),
            backend.ram(),
            backend.table_pool(),
        )
        .unwrap(),
    );
    let mapper = Box::new(SpaceMapper(Arc::clone(&space)));
    let mut kernel = KernelSpace::restore(&bytes, mapper, |_| None).unwrap();
    assert_eq!(kernel.mapped_physical_memory_size(), 2 * PAGE_SIZE);
    assert_eq!(kernel.heap_size(), 2 * PAGE_SIZE);
    space.write(CODE, &program()).unwrap();
    let first = 0x1122_3344_5566_7788_u64;
    let second = 0x99aa_bbcc_ddee_ff00_u64;
    space
        .write(alias + PAGE_SIZE, &first.to_le_bytes())
        .unwrap();
    space.write(heap, &second.to_le_bytes()).unwrap();

    // Declared after the backend, so it is dropped first, as the VM requires.
    let mut vcpu = backend.create_vcpu().unwrap();
    vcpu.set_address_space(space.root(), FIRST_ASID).unwrap();
    vcpu.set_pac_keys(&PacKeys::default()).unwrap();
    vcpu.flush_pending(space.tables(), FIRST_ASID).unwrap();

    // Guest code reads the restored physical memory and heap, and writes beside each.
    assert_eq!(run(&mut vcpu, alias + PAGE_SIZE, 0x5a), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], first);
    assert_eq!(read_u64(&space, alias + PAGE_SIZE + 8), 0x5a);
    assert_eq!(run(&mut vcpu, heap, 0xa5), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], second);
    assert_eq!(read_u64(&space, heap + 8), 0xa5);

    // The restored kernel maps a third page beside the two, which guest code then reads.
    let third = alias + 2 * PAGE_SIZE;
    kernel.map_physical_memory(third, PAGE_SIZE).unwrap();
    let block = kernel.query_info(third).unwrap();
    assert_eq!(
        (block.address(), block.size(), block.state()),
        (alias, 3 * PAGE_SIZE, MemoryState::Normal)
    );
    assert_eq!(kernel.mapped_physical_memory_size(), 3 * PAGE_SIZE);
    let value = 0x0123_4567_89ab_cdef_u64;
    space.write(third, &value.to_le_bytes()).unwrap();
    assert_eq!(run(&mut vcpu, third, 0), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], value);

    // Unmapped by the restored kernel, with its flush run: a load is a translation fault.
    kernel.unmap_physical_memory(alias, 3 * PAGE_SIZE).unwrap();
    assert_eq!(kernel.query_info(alias).unwrap().state(), MemoryState::Free);
    assert_eq!(kernel.mapped_physical_memory_size(), 0);
    vcpu.flush_pending(space.tables(), FIRST_ASID).unwrap();
    let exit = run(&mut vcpu, alias + PAGE_SIZE, 0);
    assert!(
        matches!(
            exit,
            Exit::DataAbort { address, write: false, syndrome }
                if address == alias + PAGE_SIZE
                    && matches!(syndrome.fault(), Some(FaultStatus::Translation { .. }))
        ),
        "a load from unmapped physical memory ended as {exit:?}"
    );
    // The heap is still there.
    assert_eq!(run(&mut vcpu, heap, 0), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], second);
    drop(vcpu);
}

//! A kernel address space restored from a snapshot onto a real vCPU (Hypervisor.framework):
//! saved from an address space over plain memory, restored over the VM's through the real
//! mapper, then used. Guest code at EL0 reads the restored mapped physical memory and heap; the
//! restored kernel maps more physical memory, which guest code reads, queries it, and unmaps
//! it, after which a load from it is a translation fault (`starquake_kernel::memory`'s
//! `space::snapshot` module docs, check 10: the restore maps every block that is not free
//! through the new mapper). A restored pair of `MapMemory` aliases maps its sources' pages:
//! guest code reads and writes each source through its alias.
//!
//! What guest memory holds is not part of the address space's snapshot, so the first two tests
//! write the program and the values through the host's view after the restore. The last saves
//! the guest pages too (`space::snapshot`): a page guest code stored to on one VM is read back
//! by guest code on a new VM, run from the restored code page, with nothing written after the
//! restore.
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
use crate::space::snapshot::restore;
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

#[test]
fn restored_aliases_map_their_sources_pages_for_guest_code() {
    if std::env::var_os("CI").is_some() {
        return;
    }
    // The original, over plain memory: two `MapMemory` aliases near the stack region's top,
    // heap page 0 at stack page 0 and heap pages 2-3 at stack pages 4-5.
    let mut original = World::over(space());
    let kernel = &mut original.kernel;
    kernel
        .set_process_memory_permission(CODE, PAGE_SIZE, SvcPermission::ReadExecute)
        .unwrap();
    kernel.create_main_thread_stack(PAGE_SIZE).unwrap();
    let heap = kernel.set_heap_size(4 * PAGE_SIZE).unwrap();
    let stack = kernel.regions().stack.end - 16 * PAGE_SIZE;
    let (first_alias, first_source) = (stack, heap);
    let (second_alias, second_source) = (stack + 4 * PAGE_SIZE, heap + 2 * PAGE_SIZE);
    kernel
        .map_memory(first_alias, first_source, PAGE_SIZE)
        .unwrap();
    kernel
        .map_memory(second_alias, second_source, 2 * PAGE_SIZE)
        .unwrap();
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
    space.write(CODE, &program()).unwrap();
    // What each source holds, written through the host's view: the second alias's is in its
    // second page.
    let first = 0x0bad_cafe_f00d_d00d_u64;
    let second = 0x1357_9bdf_2468_ace0_u64;
    space.write(first_source, &first.to_le_bytes()).unwrap();
    space
        .write(second_source + PAGE_SIZE, &second.to_le_bytes())
        .unwrap();

    // Declared after the backend, so it is dropped first, as the VM requires.
    let mut vcpu = backend.create_vcpu().unwrap();
    vcpu.set_address_space(space.root(), FIRST_ASID).unwrap();
    vcpu.set_pac_keys(&PacKeys::default()).unwrap();
    vcpu.flush_pending(space.tables(), FIRST_ASID).unwrap();

    // Guest code reads each source through its alias, and its write through the alias reaches
    // the source.
    assert_eq!(run(&mut vcpu, first_alias, 0x5a), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], first);
    assert_eq!(read_u64(&space, first_source + 8), 0x5a);
    let second_page = second_alias + PAGE_SIZE;
    assert_eq!(run(&mut vcpu, second_page, 0xa5), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], second);
    assert_eq!(read_u64(&space, second_source + PAGE_SIZE + 8), 0xa5);

    // The source is not the guest's while it is aliased.
    let exit = run(&mut vcpu, first_source, 0);
    assert!(
        matches!(exit, Exit::DataAbort { address, write: false, .. } if address == first_source),
        "a load from an aliased source ended as {exit:?}"
    );

    // The restored kernel unmaps the second alias: its source is the guest's again and holds
    // what was written through the alias, and the alias is gone.
    kernel
        .unmap_memory(second_alias, second_source, 2 * PAGE_SIZE)
        .unwrap();
    vcpu.flush_pending(space.tables(), FIRST_ASID).unwrap();
    assert_eq!(run(&mut vcpu, second_source + PAGE_SIZE, 0), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], second);
    let exit = run(&mut vcpu, second_page, 0);
    assert!(
        matches!(
            exit,
            Exit::DataAbort { address, write: false, syndrome }
                if address == second_page
                    && matches!(syndrome.fault(), Some(FaultStatus::Translation { .. }))
        ),
        "a load from an unmapped alias ended as {exit:?}"
    );
    // The first alias still reaches its source.
    assert_eq!(run(&mut vcpu, first_alias, 0), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], first);
    drop(vcpu);
}

#[test]
fn a_page_guest_code_stored_to_is_restored_onto_a_new_vm() {
    if std::env::var_os("CI").is_some() {
        return;
    }
    let _one_vm = ONE_VM.lock().unwrap_or_else(PoisonError::into_inner);
    let value = 0x6c6f_7374_2070_6167_u64;
    // The original, on a VM: the code made read-execute and written by the host, a stack and a
    // page of heap the guest touches first by loading from it, then stores `value` beside.
    let (pages, blocks, heap) = {
        let backend = HvfBackend::new(RamConfig::default()).expect("create the VM");
        let space = AddressSpace::<PhysicalMemory>::new(
            backend.physical(),
            backend.ram(),
            backend.table_pool(),
        )
        .unwrap();
        let mut original = World::over(space);
        let kernel = &mut original.kernel;
        kernel
            .set_process_memory_permission(CODE, PAGE_SIZE, SvcPermission::ReadExecute)
            .unwrap();
        kernel.create_main_thread_stack(PAGE_SIZE).unwrap();
        let heap = kernel.set_heap_size(PAGE_SIZE).unwrap();
        original.space.write(CODE, &program()).unwrap();
        let mut vcpu = backend.create_vcpu().unwrap();
        vcpu.set_address_space(original.space.root(), FIRST_ASID)
            .unwrap();
        vcpu.set_pac_keys(&PacKeys::default()).unwrap();
        vcpu.flush_pending(original.space.tables(), FIRST_ASID)
            .unwrap();
        // The heap page has no backing: the load faults, the page is backed as the core loop
        // backs it, and the program runs again.
        let exit = run(&mut vcpu, heap, value);
        assert!(
            matches!(exit, Exit::DataAbort { address, write: false, .. } if address == heap),
            "the first load ended as {exit:?}"
        );
        assert_eq!(original.space.touch(heap, Access::Read), Touch::Backed);
        assert_eq!(run(&mut vcpu, heap, value), Exit::Svc(0));
        drop(vcpu);
        (
            original.space.save_pages().unwrap(),
            original.kernel.save().unwrap(),
            heap,
        )
    };

    // The restore, onto a new VM: guest code from the restored code page reads what the guest
    // stored.
    let backend = HvfBackend::new(RamConfig::default()).expect("create the VM");
    let space = Arc::new(
        AddressSpace::<PhysicalMemory>::new(
            backend.physical(),
            backend.ram(),
            backend.table_pool(),
        )
        .unwrap(),
    );
    let kernel = restore(&space, &pages, &blocks, |_| None).unwrap();
    assert_eq!(kernel.heap_size(), PAGE_SIZE);
    let mut vcpu = backend.create_vcpu().unwrap();
    vcpu.set_address_space(space.root(), FIRST_ASID).unwrap();
    vcpu.set_pac_keys(&PacKeys::default()).unwrap();
    vcpu.flush_pending(space.tables(), FIRST_ASID).unwrap();
    assert_eq!(run(&mut vcpu, heap + 8, 0x77), Exit::Svc(0));
    assert_eq!(vcpu.regs().unwrap().x[0], value);
    assert_eq!(read_u64(&space, heap + 16), 0x77);
    drop(vcpu);
}

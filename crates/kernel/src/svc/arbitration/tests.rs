//! The arbitration calls' own checks, in the reference's order with both faults present, and
//! user memory reached through the fixture's address space: a read-only page reads but takes
//! no write or atomic update, and memory past the data takes nothing, from the exact word on.

use starquake_cpu::GuestRegs;
use starquake_memory::{FlatMemory, GuestMemory, PAGE_SIZE};
use starquake_trace::HostNanos;

use super::*;
use crate::memory::fixture::{self, DATA, DATA_PAGES, RODATA, RODATA_PAGES};
use crate::result::{CallResult, ResultCode};
use crate::svc::SvcId;
use crate::sync::testing::{World, core, origin, regs};
use crate::sync::{HANDLE_WAIT_MASK, ProcessKey};

/// The end of the data: the guest may touch nothing from here, though the flat memory goes on.
const DATA_END: u64 = DATA + DATA_PAGES * PAGE_SIZE;
/// A lock or arbiter word in the data, and a condition-variable key.
const AT: u64 = DATA + 0x40;
const CV: u64 = DATA + 0x80;
const NOW: HostNanos = HostNanos(1_000);
/// An arbitration or signal type no reference value has.
const BAD_TYPE: u32 = 3;

/// The sync tests' world, with the fixture's address space and flat memory from the read-only
/// data to a page past the data.
struct Svc {
    world: World,
    space: AddressSpace,
    memory: FlatMemory,
}

impl Svc {
    fn new(priorities: &[u8]) -> Self {
        let size = (RODATA_PAGES + DATA_PAGES + 1) * PAGE_SIZE;
        Self {
            world: World::new(priorities),
            space: fixture::loaded().0,
            memory: FlatMemory::new(RODATA, usize::try_from(size).unwrap()),
        }
    }

    fn set(&self, address: u64, value: u32) {
        self.memory.write(address, &value.to_le_bytes()).unwrap();
    }

    fn get(&self, address: u64) -> u32 {
        let mut bytes = [0; 4];
        self.memory.read(address, &mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    }

    fn lock(&mut self, index: usize, owner: usize, address: u64) -> CallResult<Arbitrated> {
        let regs = regs(index);
        let waiter = waiter(index, &regs);
        let handle = self.world.thread_handles[owner];
        let w = &mut self.world;
        arbitrate_lock(
            (&w.handles, &self.space, &mut w.scheduler),
            &self.memory,
            waiter,
            handle,
            address,
            w.thread_handles[index].raw(),
        )
    }

    fn unlock(&mut self, index: usize, address: u64) -> CallResult<()> {
        arbitrate_unlock(
            &self.space,
            &self.memory,
            &mut self.world.scheduler,
            core(index),
            address,
        )
    }

    fn cv_wait(&mut self, index: usize, address: u64, cv_key: u64) -> CallResult<Arbitrated> {
        let regs = regs(index);
        let tag = self.world.thread_handles[index].raw();
        wait_process_wide_key_atomic(
            &self.space,
            &self.memory,
            &mut self.world.scheduler,
            waiter(index, &regs),
            (address, cv_key, tag, -1),
        )
    }

    fn cv_signal(&mut self, index: usize, cv_key: u64, count: i32) -> CallResult<()> {
        let w = &mut self.world;
        signal_process_wide_key(
            (&w.handles, &self.space, &mut w.scheduler),
            &self.memory,
            core(index),
            cv_key,
            count,
        )
    }

    fn wait(
        &mut self,
        index: usize,
        address: u64,
        kind: u32,
        value: i32,
    ) -> CallResult<Arbitrated> {
        let regs = regs(index);
        wait_for_address(
            &self.space,
            &self.memory,
            &mut self.world.scheduler,
            waiter(index, &regs),
            (address, kind, value, -1),
        )
    }

    fn signal(&mut self, index: usize, address: u64, kind: u32, value: i32) -> CallResult<()> {
        signal_to_address(
            &self.space,
            &self.memory,
            &mut self.world.scheduler,
            core(index),
            (address, kind, value, 0),
        )
    }
}

fn waiter(index: usize, regs: &GuestRegs) -> Waiter<'_> {
    Waiter {
        core: core(index),
        now: NOW,
        regs,
        origin: origin(index),
    }
}

#[test]
fn kernel_addresses_are_the_top_of_the_space_less_the_aslr_alignment() {
    assert_eq!(
        KERNEL_VIRTUAL_ADDRESS_SPACE_BASE.wrapping_add(KERNEL_VIRTUAL_ADDRESS_SPACE_WIDTH),
        0
    );
    assert_eq!(
        0_u64.wrapping_sub(KERNEL_VIRTUAL_ADDRESS_SPACE_END),
        KERNEL_ASLR_ALIGNMENT
    );
    assert!(is_kernel_address(KERNEL_VIRTUAL_ADDRESS_SPACE_BASE));
    assert!(!is_kernel_address(KERNEL_VIRTUAL_ADDRESS_SPACE_BASE - 1));
    assert!(is_kernel_address(KERNEL_VIRTUAL_ADDRESS_SPACE_END - 1));
    assert!(!is_kernel_address(KERNEL_VIRTUAL_ADDRESS_SPACE_END));
    assert!(!is_kernel_address(u64::MAX));
}

/// Addresses with the error the address checks give them: a misaligned kernel address shows
/// the kernel check comes first, and the misaligned neighbours just outside the range show
/// where the range ends.
fn address_cases() -> [(u64, KernelError); 5] {
    [
        (
            KERNEL_VIRTUAL_ADDRESS_SPACE_BASE + 1,
            KernelError::InvalidCurrentMemory,
        ),
        (
            KERNEL_VIRTUAL_ADDRESS_SPACE_END - 1,
            KernelError::InvalidCurrentMemory,
        ),
        (
            KERNEL_VIRTUAL_ADDRESS_SPACE_BASE - 1,
            KernelError::InvalidAddress,
        ),
        (
            KERNEL_VIRTUAL_ADDRESS_SPACE_END + 1,
            KernelError::InvalidAddress,
        ),
        (AT + 2, KernelError::InvalidAddress),
    ]
}

#[test]
fn every_call_checks_its_address_kernel_range_first_then_alignment() {
    for (address, error) in address_cases() {
        let mut s = Svc::new(&[30, 30]);
        let expected = Some(error.into());
        assert_eq!(
            s.lock(1, 0, address).err(),
            expected,
            "ArbitrateLock {address:#x}"
        );
        assert_eq!(
            s.unlock(0, address).err(),
            expected,
            "ArbitrateUnlock {address:#x}"
        );
        // The one call whose kernel-address error differs: Atmosphère's `InvalidPointer`
        // (the module docs' table), where the others fail with `InvalidCurrentMemory`.
        let cv_expected = match error {
            KernelError::InvalidCurrentMemory => KernelError::InvalidPointer,
            other => other,
        };
        assert_eq!(
            s.cv_wait(0, address, CV).err(),
            Some(cv_expected.into()),
            "WaitProcessWideKeyAtomic {address:#x}"
        );
        // A type that is also bad comes after the address.
        assert_eq!(
            s.wait(0, address, BAD_TYPE, 0).err(),
            expected,
            "WaitForAddress {address:#x}"
        );
        assert_eq!(
            s.signal(0, address, BAD_TYPE, 0).err(),
            expected,
            "SignalToAddress {address:#x}"
        );
        assert!(!s.world.waiting(0) && !s.world.waiting(1));
    }
}

#[test]
fn an_arbitration_or_signal_type_outside_the_references_is_invalid_enum_value() {
    let mut s = Svc::new(&[30]);
    for kind in [BAD_TYPE, u32::MAX] {
        assert_eq!(
            s.wait(0, AT, kind, 0),
            Err(KernelError::InvalidEnumValue.into())
        );
        assert_eq!(
            s.signal(0, AT, kind, 0),
            Err(KernelError::InvalidEnumValue.into())
        );
    }
    // The last valid value of each.
    s.set(AT, 0);
    assert_eq!(
        s.wait(0, AT, ArbitrationType::WaitIfEqual.raw(), 0),
        Ok(Arbitrated::Waiting(s.world.threads[0]))
    );
}

#[test]
fn condition_variable_keys_are_aligned_down_and_never_checked() {
    let mut s = Svc::new(&[30, 30]);
    assert_eq!(
        s.cv_wait(0, AT, CV + 3),
        Ok(Arbitrated::Waiting(s.world.threads[0]))
    );
    let key = ProcessKey {
        process: None,
        address: CV,
    };
    assert_eq!(
        s.world.scheduler.sync().condition_variables().waiters(key),
        [s.world.threads[0]]
    );
    assert_eq!(s.get(CV), crate::sync::HAS_WAITER_FLAG);
    s.cv_signal(1, CV + 1, 1).unwrap();
    assert!(!s.world.waiting(0));
    // A kernel key is no error: the write of the flag fails and is ignored.
    assert_eq!(
        s.cv_signal(1, KERNEL_VIRTUAL_ADDRESS_SPACE_BASE + 3, 1),
        Ok(())
    );
    assert_eq!(align_cv_key(CV + 3), CV);
    assert_eq!(align_cv_key(CV), CV);
    // The second word, at both of its edges: aligned to a word (`sizeof(u32)`), not to more.
    assert_eq!(align_cv_key(CV + WORD_SIZE), CV + WORD_SIZE);
    assert_eq!(align_cv_key(CV + 2 * WORD_SIZE - 1), CV + WORD_SIZE);
    assert_eq!(
        s.cv_wait(0, AT, CV + 2 * WORD_SIZE - 1),
        Ok(Arbitrated::Waiting(s.world.threads[0]))
    );
    let second = ProcessKey {
        process: None,
        address: CV + WORD_SIZE,
    };
    assert_eq!(
        s.world
            .scheduler
            .sync()
            .condition_variables()
            .waiters(second),
        [s.world.threads[0]]
    );
}

#[test]
fn a_read_only_word_reads_but_takes_no_write_or_atomic_update() {
    let word = RODATA + 0x10;
    let mut s = Svc::new(&[30, 30]);
    s.set(word, 5);
    assert_eq!(
        s.wait(
            0,
            word,
            ArbitrationType::DecrementAndWaitIfLessThan.raw(),
            9
        ),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert_eq!(s.get(word), 5);
    // Atomic access is checked before the word is compared, so a comparison that would write
    // nothing fails the same way (`CanAccessAtomic` comes first).
    assert_eq!(
        s.wait(
            0,
            word,
            ArbitrationType::DecrementAndWaitIfLessThan.raw(),
            1
        ),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert_eq!(
        s.signal(1, word, SignalType::SignalAndIncrementIfEqual.raw(), 4),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert_eq!(
        s.signal(1, word, SignalType::SignalAndIncrementIfEqual.raw(), 5),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert_eq!(
        s.unlock(1, word),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert_eq!(s.get(word), 5);
    // A plain read works: the lock word does not name the owner, so the lock is free...
    assert_eq!(s.lock(1, 0, word), Ok(Arbitrated::Done));
    // ...and the arbiter waits on it.
    assert_eq!(
        s.wait(0, word, ArbitrationType::WaitIfEqual.raw(), 5),
        Ok(Arbitrated::Waiting(s.world.threads[0]))
    );
}

#[test]
fn memory_past_the_data_takes_nothing_from_its_first_word() {
    let mut s = Svc::new(&[30, 30]);
    let last = DATA_END - WORD_SIZE;
    for (address, readable) in [(last, true), (DATA_END, false)] {
        s.set(address, 1);
        let result = s.wait(0, address, ArbitrationType::WaitIfLessThan.raw(), 0);
        let expected = if readable {
            Err(KernelError::InvalidState.into())
        } else {
            Err(KernelError::InvalidCurrentMemory.into())
        };
        assert_eq!(result, expected, "{address:#x}");
    }
    // A write to the last word leaves the next one alone.
    s.set(DATA_END, 0xAAAA_AAAA);
    s.set(last, s.world.thread_handles[0].raw());
    assert_eq!(s.unlock(0, last), Ok(()));
    assert_eq!(s.get(last), crate::sync::NO_OWNER);
    assert_eq!(s.get(DATA_END), 0xAAAA_AAAA);
    assert_eq!(
        s.unlock(0, DATA_END),
        Err(KernelError::InvalidCurrentMemory.into())
    );
    assert_eq!(s.get(DATA_END), 0xAAAA_AAAA);
}

#[test]
fn a_lock_and_its_condition_variable_round_trip_through_guest_memory() {
    // 0 holds the lock with 1 waiting; 0 waits on the condition variable; 1 signals and lets go.
    let mut s = Svc::new(&[30, 20]);
    s.set(AT, s.world.thread_handles[0].raw() | HANDLE_WAIT_MASK);
    assert_eq!(
        s.lock(1, 0, AT),
        Ok(Arbitrated::Waiting(s.world.threads[1]))
    );
    assert_eq!(
        s.cv_wait(0, AT, CV),
        Ok(Arbitrated::Waiting(s.world.threads[0]))
    );
    assert_eq!(s.get(AT), s.world.thread_handles[1].raw());
    assert_eq!(
        s.world.scheduler.take_wakes(),
        [s.world.wake(1, SvcId::ArbitrateLock, ResultCode::SUCCESS)]
    );
    s.cv_signal(1, CV, 1).unwrap();
    assert_eq!(
        s.get(AT),
        s.world.thread_handles[1].raw() | HANDLE_WAIT_MASK
    );
    s.unlock(1, AT).unwrap();
    assert_eq!(s.get(AT), s.world.thread_handles[0].raw());
    assert_eq!(
        s.world.scheduler.take_wakes(),
        [s.world
            .wake(0, SvcId::WaitProcessWideKeyAtomic, ResultCode::SUCCESS)]
    );
}

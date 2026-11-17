//! The device-share token: what a lock returns, and each refusal of an unlock and of a
//! reclaimed token (module docs, "Checking a token").

use starquake_memory::PAGE_SIZE;

use super::*;
use crate::handle::HandleTable;
use crate::id::SlabId;
use crate::memory::MemoryBlock;
use crate::memory::fixture::{self, DATA, DATA_PAGES};
use crate::process::{Console, ProcessDescription};
use crate::svc::process::terminate_process;

/// A dispatcher over the fixture's loaded address space, and its process.
fn world() -> (Dispatcher, ProcessId) {
    let (space, _log, _) = fixture::loaded();
    let tables =
        crate::process::create(&ProcessDescription::homebrew(Console::Switch2), space).unwrap();
    let own = tables.processes.own();
    (Dispatcher::new(tables), own)
}

/// Every memory block, in address order.
fn blocks(dispatcher: &Dispatcher) -> Vec<MemoryBlock> {
    dispatcher.with::<(&AddressSpace,), _>(|(space,)| space.blocks().copied().collect())
}

/// The device count of the block holding `address`.
fn device_count(dispatcher: &Dispatcher, address: u64) -> u16 {
    dispatcher.with::<(&AddressSpace,), _>(|(space,)| {
        space.query_info(address).unwrap().device_use_count()
    })
}

/// Ends `process`, as `ExitProcess` does.
fn end(dispatcher: &Dispatcher, process: ProcessId) {
    dispatcher
        .with::<(&mut HandleTable, &mut Scheduler), _>(|(handles, scheduler)| {
            terminate_process(handles, scheduler, process)
        })
        .unwrap();
}

/// Two pages of the data locked, with their token: the third data page is the neighbour that
/// starts exactly at the lock's end.
fn locked() -> (Dispatcher, ProcessId, DeviceShareToken) {
    let (dispatcher, own) = world();
    let token = dispatcher.lock_device_share(DATA, 2 * PAGE_SIZE).unwrap();
    (dispatcher, own, token)
}

/// The fixture has a page of data past the lock, so "one page long" is still its own data.
const _: () = assert!(DATA_PAGES >= 3);

#[test]
fn a_lock_returns_a_token_naming_the_process_and_the_exact_range_and_it_unlocks() {
    let (dispatcher, own) = world();
    let before = blocks(&dispatcher);
    let token = dispatcher.lock_device_share(DATA, 2 * PAGE_SIZE).unwrap();
    assert_eq!(
        (token.process(), token.address(), token.size()),
        (own, DATA, 2 * PAGE_SIZE)
    );
    assert_eq!(device_count(&dispatcher, DATA), 1);
    // The neighbour that starts exactly at the end of the range is unchanged.
    assert_eq!(device_count(&dispatcher, DATA + 2 * PAGE_SIZE), 0);
    assert_eq!(dispatcher.unlock_device_share(token), Ok(()));
    assert_eq!(blocks(&dispatcher), before);
}

#[test]
fn a_token_naming_another_process_is_refused_and_changes_nothing() {
    let (dispatcher, own, token) = locked();
    let before = blocks(&dispatcher);
    // Another slot, and the same slot in another generation.
    let others = [
        ProcessId::from_parts(own.index().wrapping_add(1), own.generation()),
        ProcessId::from_parts(own.index(), own.generation().wrapping_add(1)),
    ];
    for other in others {
        let forged = DeviceShareToken::new(other, token.address(), token.size());
        assert_eq!(
            dispatcher.unlock_device_share(forged),
            Err(DeviceShareError::WrongProcess)
        );
        assert_eq!(
            dispatcher.reclaim_device_share(other, token.address(), token.size()),
            Err(DeviceShareError::WrongProcess)
        );
    }
    assert_eq!(blocks(&dispatcher), before);
    // The real token still unlocks.
    assert_eq!(dispatcher.unlock_device_share(token), Ok(()));
}

#[test]
fn a_range_one_page_short_is_refused_and_changes_nothing() {
    let (dispatcher, own, token) = locked();
    let before = blocks(&dispatcher);
    // Short at the end, and short at the start (it ends where the lock ends).
    for (address, size) in [(DATA, PAGE_SIZE), (DATA + PAGE_SIZE, PAGE_SIZE)] {
        let short = DeviceShareToken::new(own, address, size);
        let refusal = DeviceShareError::NotLocked { address, size };
        assert_eq!(dispatcher.unlock_device_share(short), Err(refusal.clone()));
        assert_eq!(
            dispatcher.reclaim_device_share(own, address, size),
            Err(refusal)
        );
    }
    assert_eq!(blocks(&dispatcher), before);
    assert_eq!(dispatcher.unlock_device_share(token), Ok(()));
}

#[test]
fn a_range_one_page_long_is_refused_and_changes_nothing() {
    let (dispatcher, own, token) = locked();
    let before = blocks(&dispatcher);
    // Long at the end (over the unlocked neighbour), and long at the start.
    for (address, size) in [(DATA, 3 * PAGE_SIZE), (DATA - PAGE_SIZE, 3 * PAGE_SIZE)] {
        let long = DeviceShareToken::new(own, address, size);
        let refusal = DeviceShareError::NotLocked { address, size };
        assert_eq!(dispatcher.unlock_device_share(long), Err(refusal.clone()));
        assert_eq!(
            dispatcher.reclaim_device_share(own, address, size),
            Err(refusal)
        );
    }
    assert_eq!(blocks(&dispatcher), before);
    assert_eq!(dispatcher.unlock_device_share(token), Ok(()));
}

#[test]
fn an_unlock_after_the_process_ended_is_refused_and_changes_nothing() {
    let (dispatcher, own, token) = locked();
    // The process is live: the token checks out.
    assert_eq!(
        dispatcher.reclaim_device_share(own, token.address(), token.size()),
        Ok(token)
    );
    end(&dispatcher, own);
    let before = blocks(&dispatcher);
    assert_eq!(
        dispatcher.unlock_device_share(token),
        Err(DeviceShareError::ProcessEnded)
    );
    assert_eq!(
        dispatcher.reclaim_device_share(own, token.address(), token.size()),
        Err(DeviceShareError::ProcessEnded)
    );
    assert_eq!(blocks(&dispatcher), before);
    assert_eq!(device_count(&dispatcher, DATA), 1);
}

#[test]
fn a_second_unlock_of_one_lock_is_refused() {
    let (dispatcher, _, token) = locked();
    assert_eq!(dispatcher.unlock_device_share(token), Ok(()));
    let before = blocks(&dispatcher);
    assert_eq!(
        dispatcher.unlock_device_share(token),
        Err(DeviceShareError::NotLocked {
            address: DATA,
            size: 2 * PAGE_SIZE,
        })
    );
    assert_eq!(blocks(&dispatcher), before);
}

#[test]
fn nested_locks_give_equal_tokens_and_each_unlocks_once() {
    let (dispatcher, own, first) = locked();
    let second = dispatcher.lock_device_share(DATA, 2 * PAGE_SIZE).unwrap();
    assert_eq!(first, second);
    assert_eq!(device_count(&dispatcher, DATA), 2);
    assert_eq!(dispatcher.unlock_device_share(first), Ok(()));
    // One lock is left: the range is still locked, and its token still checks out.
    assert_eq!(device_count(&dispatcher, DATA), 1);
    assert_eq!(
        dispatcher.reclaim_device_share(own, DATA, 2 * PAGE_SIZE),
        Ok(second)
    );
    assert_eq!(dispatcher.unlock_device_share(second), Ok(()));
    assert_eq!(device_count(&dispatcher, DATA), 0);
    assert_eq!(
        dispatcher.unlock_device_share(second),
        Err(DeviceShareError::NotLocked {
            address: DATA,
            size: 2 * PAGE_SIZE,
        })
    );
}

#[test]
fn the_process_is_checked_before_the_range() {
    let (dispatcher, own, token) = locked();
    let other = ProcessId::from_parts(own.index().wrapping_add(1), own.generation());
    end(&dispatcher, own);
    // An ended process and a range no lock was taken on: the process is answered.
    let short = DeviceShareToken::new(own, DATA, PAGE_SIZE);
    assert_eq!(
        dispatcher.unlock_device_share(short),
        Err(DeviceShareError::ProcessEnded)
    );
    // Another process, while the tables' own has ended: the process named is answered first.
    let forged = DeviceShareToken::new(other, DATA, PAGE_SIZE);
    assert_eq!(
        dispatcher.unlock_device_share(forged),
        Err(DeviceShareError::WrongProcess)
    );
    assert_eq!(device_count(&dispatcher, token.address()), 1);
}

#[test]
fn a_reclaimed_token_is_the_lock_s_and_unlocks_it_and_reclaiming_releases_nothing() {
    let (dispatcher, own, token) = locked();
    let reclaimed = dispatcher
        .reclaim_device_share(own, DATA, 2 * PAGE_SIZE)
        .unwrap();
    assert_eq!(reclaimed, token);
    assert_eq!(device_count(&dispatcher, DATA), 1);
    assert_eq!(dispatcher.unlock_device_share(reclaimed), Ok(()));
    assert_eq!(device_count(&dispatcher, DATA), 0);
}

#[test]
fn each_refusal_reads_in_plain_words() {
    assert_eq!(
        DeviceShareError::WrongProcess.to_string(),
        "the device-share token names a process whose memory this kernel does not hold"
    );
    assert_eq!(
        DeviceShareError::ProcessEnded.to_string(),
        "the device-share token's process has ended"
    );
    assert_eq!(
        DeviceShareError::NotLocked {
            address: 0x8000_4000,
            size: 0x2000,
        }
        .to_string(),
        "no device-share lock is held on exactly 0x2000 bytes at 0x80004000"
    );
    assert_eq!(
        DeviceShareError::Refused(KernelError::InvalidCurrentMemory.into()).to_string(),
        format!(
            "the address space refused the device-share unlock: {}",
            KernelError::InvalidCurrentMemory
        )
    );
}

#[test]
fn a_server_sees_a_refused_token_as_invalid_current_memory() {
    for refusal in [
        DeviceShareError::WrongProcess,
        DeviceShareError::ProcessEnded,
        DeviceShareError::NotLocked {
            address: DATA,
            size: PAGE_SIZE,
        },
    ] {
        assert_eq!(
            refusal.into_svc_error(),
            SvcError::Kernel(KernelError::InvalidCurrentMemory)
        );
    }
    assert_eq!(
        DeviceShareError::Refused(KernelError::OutOfResource.into()).into_svc_error(),
        SvcError::Kernel(KernelError::OutOfResource)
    );
}

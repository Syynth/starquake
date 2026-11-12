//! `ArbitrateLock` (0x1a), `ArbitrateUnlock` (0x1b), `WaitProcessWideKeyAtomic` (0x1c),
//! `SignalProcessWideKey` (0x1d), `WaitForAddress` (0x34) and `SignalToAddress` (0x35): the
//! system calls' own checks of their arguments, then the scheduler's operation
//! ([`crate::sync::lock`], [`crate::sync::condvar`], [`crate::sync::arbiter`]).
//!
//! Ported from Eden @ cf654052 (GPL-3.0-or-later), paths relative to the reference folder:
//! `eden/src/core/hle/kernel/svc/svc_lock.cpp`, `svc_condition_variable.cpp` and
//! `svc_address_arbiter.cpp`. The checks, in Eden's order, each failing with the first error:
//!
//! | Call | Checks |
//! |---|---|
//! | `ArbitrateLock`, `ArbitrateUnlock` | `address` not a kernel address ([`KernelError::InvalidCurrentMemory`]); `address` word-aligned ([`KernelError::InvalidAddress`]) |
//! | `WaitProcessWideKeyAtomic` | `address` not a kernel address ([`KernelError::InvalidPointer`], Atmosphère's error: see below); `address` word-aligned ([`KernelError::InvalidAddress`]); `cv_key` is not checked, only aligned down to a word (`Common::AlignDown(cv_key, sizeof(u32))`) |
//! | `SignalProcessWideKey` | none; `cv_key` is aligned down to a word |
//! | `WaitForAddress` | the same two of `address`; then the arbitration type ([`KernelError::InvalidEnumValue`]) |
//! | `SignalToAddress` | the same two of `address`; then the signal type ([`KernelError::InvalidEnumValue`]) |
//!
//! A kernel address is one in `[KernelVirtualAddressSpaceBase, KernelVirtualAddressSpaceEnd)`
//! (`IsKernelAddress` in `eden/src/core/hle/kernel/k_memory_layout.h`): the top 2^39 bytes of
//! the 64-bit space, less the last `KernelAslrAlignment` (2 MiB). The timeouts are converted as
//! `WaitSynchronization`'s are ([`Timeout::from_nanos`]).
//!
//! # Where the references disagree
//!
//! - **`WaitProcessWideKeyAtomic` on a kernel address** fails with `ResultInvalidCurrentMemory`
//!   in Eden (`svc_condition_variable.cpp`, `R_UNLESS(!IsKernelAddress(address),
//!   ResultInvalidCurrentMemory)`) but with `svc::ResultInvalidPointer` in Atmosphère
//!   (`atmosphere/libraries/libmesosphere/source/svc/kern_svc_condition_variable.cpp`
//!   @ 4e3f3703, `WaitProcessWideKeyAtomic`'s first check; fact only), while both references
//!   give `InvalidCurrentMemory` for the same check in the other five calls. starquake follows
//!   Atmosphère, as `crate::sync` does for its user-memory checks: the odd one out is more
//!   likely copied from the real kernel than invented. A guest test on hardware that passes a
//!   kernel address would confirm it. Test: `every_call_checks_its_address_kernel_range_first_then_alignment`.

use starquake_memory::GuestMemory;

use super::sync::Waiter;
use crate::handle::{Handle, HandleTable};
use crate::memory::AddressSpace;
use crate::result::{CallResult, KernelError};
use crate::sched::CoreId;
use crate::sync::{
    Arbitrated, ArbitrationType, GuestWords, SignalType, Timeout, WORD_SIZE, WaitCall,
};
use crate::tables::Scheduler;

/// The width of the kernel's virtual address space: Eden's `KernelVirtualAddressSpaceWidth`
/// (`eden/src/core/hle/kernel/k_memory_layout.h` @ cf654052, `1ULL << 39`).
pub const KERNEL_VIRTUAL_ADDRESS_SPACE_WIDTH: u64 = 1 << 39;

/// The alignment of the kernel's randomised base: Eden's `KernelAslrAlignment`
/// (`k_memory_layout.h`, `2_MiB`).
pub const KERNEL_ASLR_ALIGNMENT: u64 = 2 << 20;

/// The first kernel address: Eden's `KernelVirtualAddressSpaceBase`
/// (`0ULL - KernelVirtualAddressSpaceWidth`).
pub const KERNEL_VIRTUAL_ADDRESS_SPACE_BASE: u64 =
    0_u64.wrapping_sub(KERNEL_VIRTUAL_ADDRESS_SPACE_WIDTH);

/// The address after the last kernel address: Eden's `KernelVirtualAddressSpaceEnd`
/// (`KernelVirtualAddressSpaceBase + (KernelVirtualAddressSpaceWidth - KernelAslrAlignment)`).
pub const KERNEL_VIRTUAL_ADDRESS_SPACE_END: u64 = KERNEL_VIRTUAL_ADDRESS_SPACE_BASE
    + (KERNEL_VIRTUAL_ADDRESS_SPACE_WIDTH - KERNEL_ASLR_ALIGNMENT);

/// Whether `address` is a kernel address (`IsKernelAddress`).
#[must_use]
pub const fn is_kernel_address(address: u64) -> bool {
    KERNEL_VIRTUAL_ADDRESS_SPACE_BASE <= address && address < KERNEL_VIRTUAL_ADDRESS_SPACE_END
}

/// The checks every call here makes of its lock or arbiter `address`, in Eden's order:
/// `kernel_error` for a kernel address, then [`KernelError::InvalidAddress`] for one that is
/// not word-aligned.
fn check_address_with(address: u64, kernel_error: KernelError) -> CallResult<()> {
    if is_kernel_address(address) {
        return Err(kernel_error.into());
    }
    if !address.is_multiple_of(WORD_SIZE) {
        return Err(KernelError::InvalidAddress.into());
    }
    Ok(())
}

/// [`check_address_with`] for every call but `WaitProcessWideKeyAtomic`: a kernel address is
/// [`KernelError::InvalidCurrentMemory`].
fn check_address(address: u64) -> CallResult<()> {
    check_address_with(address, KernelError::InvalidCurrentMemory)
}

/// `cv_key` aligned down to a word (`Common::AlignDown(cv_key, sizeof(u32))`).
#[must_use]
pub const fn align_cv_key(cv_key: u64) -> u64 {
    cv_key - cv_key % WORD_SIZE
}

/// The wait half of a [`Waiter`]: who waits and how the event is recorded.
const fn wait_call(waiter: Waiter<'_>) -> WaitCall<'_> {
    WaitCall {
        core: waiter.core,
        regs: waiter.regs,
        origin: waiter.origin,
    }
}

/// `ArbitrateLock(thread_handle, address, tag)`: see the module docs for the checks, then
/// [`Scheduler::arbitrate_lock`].
pub fn arbitrate_lock(
    (handles, space, scheduler): (&HandleTable, &AddressSpace, &mut Scheduler),
    memory: &dyn GuestMemory,
    waiter: Waiter<'_>,
    thread_handle: Handle,
    address: u64,
    tag: u32,
) -> CallResult<Arbitrated> {
    check_address(address)?;
    let words = GuestWords { space, memory };
    scheduler.arbitrate_lock(
        handles,
        &words,
        wait_call(waiter),
        thread_handle,
        address,
        tag,
    )
}

/// `ArbitrateUnlock(address)`: see the module docs for the checks, then
/// [`Scheduler::arbitrate_unlock`].
pub fn arbitrate_unlock(
    space: &AddressSpace,
    memory: &dyn GuestMemory,
    scheduler: &mut Scheduler,
    core: CoreId,
    address: u64,
) -> CallResult<()> {
    check_address(address)?;
    scheduler.arbitrate_unlock(&GuestWords { space, memory }, core, address)
}

/// `WaitProcessWideKeyAtomic(address, cv_key, tag, timeout_ns)`: see the module docs for the
/// checks, then [`Scheduler::wait_condition_variable`] with `cv_key` aligned down to a word.
pub fn wait_process_wide_key_atomic(
    space: &AddressSpace,
    memory: &dyn GuestMemory,
    scheduler: &mut Scheduler,
    waiter: Waiter<'_>,
    (address, cv_key, tag, timeout_ns): (u64, u64, u32, i64),
) -> CallResult<Arbitrated> {
    check_address_with(address, KernelError::InvalidPointer)?;
    let timeout = Timeout::from_nanos(waiter.now, timeout_ns);
    scheduler.wait_condition_variable(
        &GuestWords { space, memory },
        wait_call(waiter),
        address,
        align_cv_key(cv_key),
        tag,
        timeout,
    )
}

/// `SignalProcessWideKey(cv_key, count)`: [`Scheduler::signal_condition_variable`] with
/// `cv_key` aligned down to a word. The call returns nothing; a failure here is the kernel's
/// own.
pub fn signal_process_wide_key(
    (handles, space, scheduler): (&HandleTable, &AddressSpace, &mut Scheduler),
    memory: &dyn GuestMemory,
    core: CoreId,
    cv_key: u64,
    count: i32,
) -> CallResult<()> {
    scheduler.signal_condition_variable(
        handles,
        &GuestWords { space, memory },
        core,
        align_cv_key(cv_key),
        count,
    )
}

/// `WaitForAddress(address, arb_type, value, timeout_ns)`: see the module docs for the checks,
/// then [`Scheduler::wait_for_address`].
pub fn wait_for_address(
    space: &AddressSpace,
    memory: &dyn GuestMemory,
    scheduler: &mut Scheduler,
    waiter: Waiter<'_>,
    (address, arb_type, value, timeout_ns): (u64, u32, i32, i64),
) -> CallResult<Arbitrated> {
    check_address(address)?;
    let kind = ArbitrationType::from_raw(arb_type).ok_or(KernelError::InvalidEnumValue)?;
    let timeout = Timeout::from_nanos(waiter.now, timeout_ns);
    scheduler.wait_for_address(
        &GuestWords { space, memory },
        wait_call(waiter),
        address,
        kind,
        value,
        timeout,
    )
}

/// `SignalToAddress(address, signal_type, value, count)`: see the module docs for the checks,
/// then [`Scheduler::signal_to_address`].
pub fn signal_to_address(
    space: &AddressSpace,
    memory: &dyn GuestMemory,
    scheduler: &mut Scheduler,
    core: CoreId,
    (address, signal_type, value, count): (u64, u32, i32, i32),
) -> CallResult<()> {
    check_address(address)?;
    let kind = SignalType::from_raw(signal_type).ok_or(KernelError::InvalidEnumValue)?;
    scheduler.signal_to_address(
        &GuestWords { space, memory },
        core,
        address,
        kind,
        value,
        count,
    )
}

#[cfg(test)]
mod tests;

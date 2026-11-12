//! `WaitSynchronization` (0x18) and `CancelSynchronization` (0x19).
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_synchronization.cpp` @ cf654052
//! (GPL-3.0-or-later). `WaitSynchronization`'s checks, in Eden's order:
//! 1. `0 <= num_handles <= ArgumentHandleCountMax`, else [`KernelError::OutOfRange`].
//! 2. When `num_handles > 0`: the handles are read from guest memory at `handles`, four bytes
//!    each, else [`KernelError::InvalidPointer`]; then each must name a synchronisation object
//!    (`KHandleTable::GetMultipleObjects<KSynchronizationObject>` in `k_handle_table.h`, which
//!    takes no pseudo-handle), else [`KernelError::InvalidHandle`], with every reference it
//!    opened closed again.
//! 3. The timeout is converted ([`Timeout::from_nanos`]) and the thread waits
//!    ([`Scheduler::wait_synchronization`], which owns the references from there).
//!
//! `CancelSynchronization`: the handle must name a thread (`GetObject<KThread>`, which takes the
//! current-thread pseudo-handle), else [`KernelError::InvalidHandle`]; then
//! [`Scheduler::cancel_synchronization`].
//!
//! Divergence from Eden, pinned by `dispatch/sync/tests.rs`,
//! `a_handle_array_the_guest_may_not_read_is_an_invalid_pointer`: Eden's `ReadBlock` fails only
//! for memory that is not mapped. Here the whole array must also be readable by the guest in
//! the address space, as for every guest buffer the kernel reads ([`read_user_memory`]). A guess
//! at Horizon's behaviour, whose kernel copies from user memory with the user's permissions; a
//! hardware test passing an array in a mapped page without read permission would confirm it.

use starquake_cpu::GuestRegs;
use starquake_memory::GuestMemory;
use starquake_trace::HostNanos;

use super::thread::thread_from_handle;
use crate::handle::{Handle, HandleTable, ObjectRefs};
use crate::id::ObjectId;
use crate::memory::AddressSpace;
use crate::result::{CallResult, KernelError, SvcError};
use crate::sched::CoreId;
use crate::sync::{
    ARGUMENT_HANDLE_COUNT_MAX, Timeout, WaitBegun, WaitOrigin, is_synchronization_object,
};
use crate::tables::Scheduler;

/// The size of one handle in the guest's array: a handle is a 32-bit word ([`Handle::raw`]),
/// and Eden asserts `sizeof(Handle) == 4` (`reference/svc_type_sizes.txt`).
pub const HANDLE_SIZE: usize = size_of::<u32>();

/// Copies the guest's memory at `address` into `out` if the guest may read all of it in
/// `space`, and returns whether it did; nothing is copied otherwise. The one check every guest
/// buffer the kernel reads goes through.
pub(crate) fn read_user_memory(
    space: &AddressSpace,
    memory: &dyn GuestMemory,
    address: u64,
    out: &mut [u8],
) -> bool {
    let Ok(len) = u64::try_from(out.len()) else {
        return false;
    };
    space.user_can_read(address, len) && memory.read(address, out).is_ok()
}

/// `WaitSynchronization`'s guest arguments, as its convention decodes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaitArgs {
    /// Where the handle array is (`x1`).
    pub handles: u64,
    /// How many handles it has (`w2`).
    pub num_handles: i32,
    /// The timeout in nanoseconds (`x3`): zero polls, a negative one never expires.
    pub timeout_ns: i64,
}

/// The thread making a `WaitSynchronization`: the core it runs on, the dispatch's reading of
/// the environment's clock, its registers (saved as its context if it waits) and who its event
/// is recorded for.
#[derive(Clone, Copy, Debug)]
pub struct Waiter<'a> {
    /// The calling core.
    pub core: CoreId,
    /// The dispatch's clock reading, which a positive timeout counts from.
    pub now: HostNanos,
    /// The caller's registers.
    pub regs: &'a GuestRegs,
    /// Who the call's event is recorded for.
    pub origin: WaitOrigin,
}

/// `WaitSynchronization`: see the module docs for the checks and their order.
pub fn wait_synchronization(
    handles: &HandleTable,
    space: &AddressSpace,
    memory: &dyn GuestMemory,
    scheduler: &mut Scheduler,
    waiter: Waiter<'_>,
    args: WaitArgs,
) -> CallResult<WaitBegun> {
    let count = usize::try_from(args.num_handles)
        .ok()
        .filter(|count| *count <= ARGUMENT_HANDLE_COUNT_MAX)
        .ok_or(KernelError::OutOfRange)?;
    let mut raw = vec![0; count.saturating_mul(HANDLE_SIZE)];
    if count > 0 && !read_user_memory(space, memory, args.handles, &mut raw) {
        return Err(KernelError::InvalidPointer.into());
    }
    let objects = open_synchronization_objects(handles, scheduler, &raw)?;
    let timeout = Timeout::from_nanos(waiter.now, args.timeout_ns);
    scheduler.wait_synchronization(waiter.core, &objects, timeout, waiter.regs, waiter.origin)
}

/// `GetMultipleObjects<KSynchronizationObject>`: opens a reference to the object each handle in
/// `raw` (little-endian, [`HANDLE_SIZE`] bytes each) names. Stops at the first handle that names
/// no synchronisation object, closes what it opened, and fails with
/// [`KernelError::InvalidHandle`].
fn open_synchronization_objects(
    handles: &HandleTable,
    scheduler: &mut Scheduler,
    raw: &[u8],
) -> CallResult<Vec<ObjectId>> {
    // `raw` holds whole handles; a trailing part of one would be ignored.
    let (words, _) = raw.as_chunks::<HANDLE_SIZE>();
    let mut opened = Vec::with_capacity(words.len());
    let mut failure = None;
    for word in words {
        let handle = Handle::from_raw(u32::from_le_bytes(*word));
        let object = handles.get(handle).ok().filter(|object| {
            scheduler
                .objects()
                .get(*object)
                .is_ok_and(is_synchronization_object)
        });
        let Some(object) = object else {
            failure = Some(KernelError::InvalidHandle.into());
            break;
        };
        if let Err(error) = scheduler.objects_mut().open(object) {
            failure = Some(SvcError::internal(&error));
            break;
        }
        opened.push(object);
    }
    let Some(failure) = failure else {
        return Ok(opened);
    };
    scheduler.release_objects(&opened)?;
    Err(failure)
}

/// `CancelSynchronization`: cancels the wait of the thread `handle` names, for a call made on
/// `core`.
pub fn cancel_synchronization(
    handles: &HandleTable,
    scheduler: &mut Scheduler,
    core: CoreId,
    handle: Handle,
) -> CallResult<()> {
    let thread = thread_from_handle(scheduler, handles, core, handle)?;
    scheduler.cancel_synchronization(thread)
}

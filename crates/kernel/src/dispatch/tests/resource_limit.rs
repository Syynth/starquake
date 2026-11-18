//! The resource-limit calls through the production dispatcher: `GetInfo`'s `ResourceLimit` id
//! hands out a handle, and `GetResourceLimitLimitValue` and `GetResourceLimitCurrentValue`
//! answer through it, reading `x1` and `w2` and writing `x0` and `x1`.

use super::*;
use crate::resource_limit::{LimitableResource, THREAD_COUNT_LIMIT};
use crate::svc::info::InfoType;

/// Registers for a resource-limit call on `handle` and `which` over [`regs`], with the high
/// halves of both registers set, so a call that reads them whole fails.
fn limit_regs(handle: Handle, which: u32) -> GuestRegs {
    let mut regs = regs();
    regs.x[1] = 0xffff_ffff_0000_0000 | u64::from(handle.raw());
    regs.x[2] = 0xabcd_0000_0000_0000 | u64::from(which);
    regs
}

#[test]
fn a_handle_from_get_info_answers_both_calls_through_the_dispatcher() {
    let memory = FlatMemory::new(BASE, 0x1000);
    let dispatcher = running_dispatcher(&memory);
    // `GetInfo(ResourceLimit, 0, 0)`: x1 the new handle.
    let mut regs = regs();
    regs.x[1] = u64::from(InfoType::ResourceLimit.number());
    regs.x[2] = u64::from(Handle::INVALID.raw());
    regs.x[3] = 0;
    let (record, _) = dispatch_on(&dispatcher, SvcId::GetInfo.number(), &mut regs, 0);
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(regs.x[0], 0);
    let handle = Handle::from_raw(u32::try_from(regs.x[1]).unwrap());
    let limit = dispatcher.with::<(&HandleTable, &Scheduler), _>(|(handles, scheduler)| {
        let limit = handles.get(handle).unwrap();
        let object = match scheduler.objects().get(limit) {
            Ok(Object::ResourceLimit(object)) => object,
            other => unreachable!("{other:?}"),
        };
        // The limit its process's row names.
        let row = scheduler.process(object.owner()).unwrap();
        assert_eq!(row.resource_limit(), limit);
        limit
    });
    // Both calls: the main thread is the one thread.
    let threads = LimitableResource::ThreadCount.number();
    for (id, value) in [
        (SvcId::GetResourceLimitLimitValue, THREAD_COUNT_LIMIT),
        (SvcId::GetResourceLimitCurrentValue, 1),
    ] {
        let mut regs = limit_regs(handle, threads);
        let before = regs.clone();
        let (record, events) = dispatch_on(&dispatcher, id.number(), &mut regs, 0);
        assert_eq!(record.outcome, Outcome::Completed, "{}", id.name());
        assert_eq!((regs.x[0], regs.x[1]), (0, value), "{}", id.name());
        assert_untouched_from(&regs, &before, 2);
        // The event carries the handle and the resource, each cut to its four bytes.
        let mut args = [0; SVC_ARG_COUNT];
        args[1] = u64::from(handle.raw());
        args[2] = u64::from(threads);
        assert_eq!(
            kinds(&events),
            [EventKind::Svc {
                id: u32::from(id.number()),
                args,
                result: starquake_trace::ResultCode(0),
            }],
            "{}",
            id.name()
        );
    }
    // A refusal writes its result and zeroes the value.
    let mut regs = limit_regs(handle, u32::MAX);
    let (record, _) = dispatch_on(
        &dispatcher,
        SvcId::GetResourceLimitLimitValue.number(),
        &mut regs,
        0,
    );
    assert_eq!(record.outcome, Outcome::Completed);
    assert_eq!(
        (regs.x[0], regs.x[1]),
        (word(KernelError::InvalidEnumValue), 0)
    );
    let mut regs = limit_regs(Handle::CURRENT_PROCESS, threads);
    dispatch_on(
        &dispatcher,
        SvcId::GetResourceLimitCurrentValue.number(),
        &mut regs,
        0,
    );
    assert_eq!(
        (regs.x[0], regs.x[1]),
        (word(KernelError::InvalidHandle), 0)
    );
    // The handle holds a reference beside the process row's; nothing else changed.
    dispatcher.with::<(&Scheduler,), _>(|(scheduler,)| {
        assert_eq!(scheduler.objects().refs(limit), 2);
    });
}

//! The two calls the services spec added on 2026-11-16, on [`Context`] through the fake kernel:
//! clearing an event the service made, and ending the caller's process after the reply. Each
//! kernel refusal reaches the caller as the reply's result; a kernel that does not offer the
//! calls answers "not implemented"; and an end cannot be combined with a deferred reply.

use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::{KernelError, ResultCode};
use starquake_trace::ServiceId;

use super::super::fake::FakeKernel;
use super::super::result::NOT_SUPPORTED;
use super::super::{Context, Fault, Kernel, Outcome, Reply, Request, Server, UnknownPolicy};
use super::{PARKED, built, only, plain, request};
use crate::commands;

/// `MAKE`: makes and signals an event, which the session keeps; out a copy handle to it. `CLEAR`: clears it. `CLEAR_FOREIGN`:
/// clears an id the kernel made that is no event of this server's. `END`: asks to end the
/// process. `DEFER_THEN_END`: takes a reply token, then asks; out `u32` the ask's result.
/// `END_THEN_DEFER`: asks, then takes a reply token. `REFUSED_END_THEN_DEFER`: asks, checks
/// that the kernel refused with "not implemented", then takes a reply token.
const MAKE: u32 = 0;
const CLEAR: u32 = 1;
const CLEAR_FOREIGN: u32 = 2;
const END: u32 = 3;
const DEFER_THEN_END: u32 = 4;
const END_THEN_DEFER: u32 = 5;
const REFUSED_END_THEN_DEFER: u32 = 6;

/// A session's worth of the two calls.
#[derive(serde::Serialize, serde::Deserialize)]
struct Ender {
    made: Option<ObjectId>,
    foreign: ObjectId,
}

commands! {
    impl Ender {
        MAKE => make,
        CLEAR => clear,
        CLEAR_FOREIGN => clear_foreign,
        END => end,
        DEFER_THEN_END => defer_then_end,
        END_THEN_DEFER => end_then_defer,
        REFUSED_END_THEN_DEFER => refused_end_then_defer,
    }
}

/// The result word of a call's outcome.
fn result_of(asked: Result<(), Fault>) -> u32 {
    match asked {
        Ok(()) => ResultCode::SUCCESS.raw(),
        Err(Fault::Result(code)) => code.raw(),
        Err(other) => unreachable!("not a result: {other:?}"),
    }
}

impl Ender {
    fn make(&mut self, cx: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        let event = cx.create_event()?;
        self.made = Some(event);
        cx.signal_event(event)?;
        reply.copy_handle(event);
        reply.ok()
    }

    fn clear(&mut self, cx: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        let event = self.made.ok_or(Fault::Result(NOT_SUPPORTED))?;
        cx.clear_event(event)?;
        reply.ok()
    }

    fn clear_foreign(
        &mut self,
        cx: &mut Context<'_>,
        _: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        cx.clear_event(self.foreign)?;
        reply.ok()
    }

    fn end(_: &mut Self, cx: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
        cx.end_process()?;
        reply.ok()
    }

    fn defer_then_end(
        _: &mut Self,
        cx: &mut Context<'_>,
        _: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        let token = cx.defer()?;
        PARKED.with_borrow_mut(|parked| parked.push(token));
        let asked = cx.end_process();
        reply.payload().write_u32(result_of(asked));
        reply.ok()
    }

    fn end_then_defer(
        _: &mut Self,
        cx: &mut Context<'_>,
        _: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        cx.end_process()?;
        let token = cx.defer()?;
        PARKED.with_borrow_mut(|parked| parked.push(token));
        reply.deferred()
    }

    fn refused_end_then_defer(
        _: &mut Self,
        cx: &mut Context<'_>,
        _: &Request<'_>,
        reply: &mut Reply,
    ) -> Outcome {
        assert_eq!(
            cx.end_process(),
            Err(Fault::from(KernelError::NotImplemented))
        );
        let token = cx.defer()?;
        PARKED.with_borrow_mut(|parked| parked.push(token));
        reply.deferred()
    }
}

/// A kernel whose caller has an object that is no event, a server of [`Ender`]s with one open
/// session, and that session.
fn rig() -> (FakeKernel, Server, ObjectId) {
    let mut kernel = FakeKernel::new(0).unwrap();
    let session = kernel.new_object();
    let foreign = kernel.new_object();
    let mut server = Server::new("ender", UnknownPolicy::Strict).with_service_id(ServiceId(0x63));
    server
        .open(
            session,
            Box::new(Ender {
                made: None,
                foreign,
            }),
        )
        .unwrap();
    (kernel, server, session)
}

fn send(
    server: &mut Server,
    kernel: &mut dyn Kernel,
    session: ObjectId,
    command: u32,
) -> super::Answer {
    let mut buffer = built(&request(command));
    server.handle(kernel, session, &mut buffer).unwrap();
    plain(&buffer)
}

#[test]
fn clearing_an_event_it_made_leaves_it_clear_whether_or_not_it_was_signalled() {
    let (mut kernel, mut server, session) = rig();
    let ok = only(ResultCode::SUCCESS);
    let made = send(&mut server, &mut kernel, session, MAKE);
    assert_eq!(made.result, ResultCode::SUCCESS.raw());
    let event = kernel
        .resolve_handle(Handle::from_raw(made.copies[0]))
        .unwrap();
    assert_eq!(kernel.is_signalled(event), Some(true));
    assert_eq!(send(&mut server, &mut kernel, session, CLEAR), ok);
    assert_eq!(kernel.is_signalled(event), Some(false));
    assert_eq!(send(&mut server, &mut kernel, session, CLEAR), ok);
    assert_eq!(kernel.is_signalled(event), Some(false));
    // Clearing counts no signal, and a signal after it signals again.
    assert_eq!(kernel.signals(event), Some(1));
    assert_eq!(kernel.signal_event(event), Ok(()));
    assert_eq!(kernel.is_signalled(event), Some(true));
}

#[test]
fn clearing_an_id_that_is_no_event_it_made_is_an_invalid_handle() {
    let (mut kernel, mut server, session) = rig();
    assert_eq!(
        send(&mut server, &mut kernel, session, CLEAR_FOREIGN),
        only(KernelError::InvalidHandle.code())
    );
}

#[test]
fn ending_the_process_answers_success_and_asks_the_kernel_once() {
    let (mut kernel, mut server, session) = rig();
    assert_eq!(
        send(&mut server, &mut kernel, session, END),
        only(ResultCode::SUCCESS)
    );
    assert_eq!(kernel.ends(), 1);
}

#[test]
fn an_end_asked_after_deferring_is_refused_and_the_kernel_is_not_asked() {
    let (mut kernel, mut server, session) = rig();
    let reply = send(&mut server, &mut kernel, session, DEFER_THEN_END);
    assert_eq!(reply.data, NOT_SUPPORTED.raw().to_le_bytes());
    assert_eq!(kernel.ends(), 0);
    drop(super::parked());
}

#[test]
fn deferring_after_an_end_is_refused_so_the_request_is_answered_now() {
    let (mut kernel, mut server, session) = rig();
    assert_eq!(
        send(&mut server, &mut kernel, session, END_THEN_DEFER),
        only(NOT_SUPPORTED)
    );
    assert_eq!(kernel.ends(), 1);
    assert!(super::parked().is_empty());
}

/// A kernel that offers neither call: the trait's defaults.
struct Bare(FakeKernel);

impl Kernel for Bare {
    fn process_id(&self) -> u64 {
        self.0.process_id()
    }

    fn resolve_handle(&self, handle: Handle) -> starquake_kernel::result::SvcResult<ObjectId> {
        self.0.resolve_handle(handle)
    }

    fn insert_handle(&mut self, object: ObjectId) -> starquake_kernel::result::SvcResult<Handle> {
        self.0.insert_handle(object)
    }

    fn create_event(&mut self) -> starquake_kernel::result::SvcResult<ObjectId> {
        self.0.create_event()
    }

    fn signal_event(&mut self, event: ObjectId) -> starquake_kernel::result::SvcResult<()> {
        self.0.signal_event(event)
    }

    fn create_shared_memory(
        &mut self,
        size: u64,
        permission: starquake_kernel::memory::SvcPermission,
    ) -> starquake_kernel::result::SvcResult<(ObjectId, starquake_kernel::memory::SharedPages)>
    {
        self.0.create_shared_memory(size, permission)
    }

    fn create_session(&mut self) -> starquake_kernel::result::SvcResult<(ObjectId, Handle)> {
        self.0.create_session()
    }

    fn connect(
        &mut self,
        service: ServiceId,
    ) -> starquake_kernel::result::SvcResult<(ObjectId, Handle)> {
        self.0.connect(service)
    }

    fn read_memory(&self, address: u64, out: &mut [u8]) -> starquake_kernel::result::SvcResult<()> {
        self.0.read_memory(address, out)
    }

    fn write_memory(
        &mut self,
        address: u64,
        data: &[u8],
    ) -> starquake_kernel::result::SvcResult<()> {
        self.0.write_memory(address, data)
    }

    fn record(&mut self, kind: starquake_trace::EventKind) {
        self.0.record(kind);
    }
}

#[test]
fn a_kernel_that_does_not_offer_the_calls_answers_not_implemented() {
    let (kernel, mut server, session) = rig();
    let mut bare = Bare(kernel);
    let refused = only(KernelError::NotImplemented.code());
    assert_eq!(
        send(&mut server, &mut bare, session, MAKE).result,
        ResultCode::SUCCESS.raw()
    );
    assert_eq!(send(&mut server, &mut bare, session, CLEAR), refused);
    assert_eq!(send(&mut server, &mut bare, session, END), refused);
    assert_eq!(bare.0.ends(), 0);
}

#[test]
fn an_end_the_kernel_refused_leaves_the_reply_free_to_be_deferred() {
    let (kernel, mut server, session) = rig();
    let mut bare = Bare(kernel);
    let request = request(REFUSED_END_THEN_DEFER);
    let mut buffer = built(&request);
    let handled = server.handle(&mut bare, session, &mut buffer).unwrap();
    assert!(
        matches!(handled, super::super::Handled::Deferred(_)),
        "{handled:?}"
    );
    // Nothing was written: the reply waits for its token.
    assert_eq!(buffer, built(&request));
    assert_eq!(super::parked().len(), 1);
    assert_eq!(bare.0.ends(), 0);
}

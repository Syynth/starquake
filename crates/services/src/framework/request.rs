//! [`Request`]: what a service method is given, and the buffers it reaches through it.
//!
//! # Buffers by index
//!
//! A command numbers its "in" buffers and its "out" buffers from zero. Which descriptor an index
//! means is Eden's rule (`eden/src/core/hle/service/hle_ipc.cpp`):
//!
//! - **In** (`ReadBuffer`, `GetReadBufferSize`): the A descriptor with that index if there is one
//!   and its size is not zero; otherwise the X descriptor with that index.
//! - **Out** (`WriteBuffer`, `GetWriteBufferSize`): the B descriptor with that index if there is
//!   one and its size is not zero; otherwise the C descriptor with that index.
//!
//! X descriptors are taken in the order they appear, not by their counter field, as in Eden.
//!
//! Divergences from Eden, each pinned by a test in `framework/tests/buffers.rs`:
//!
//! - Where there is no X (or C) descriptor with that index to fall back on, Eden logs a failed
//!   assertion and carries on with an empty read, or writes nothing. Here the two ways of
//!   getting there are told apart. A zero-sized A (or B) descriptor with nothing behind it is an
//!   empty buffer and nothing is logged: a command may be sent nothing to read, or no room to
//!   write. No descriptor of either kind at that index is [`Fault::NoBuffer`], and the command
//!   fails.
//! - Eden's `WriteBuffer` cuts a write down to the buffer's size and carries on (marked "This
//!   needs to be HW tested" there). Here a write past the end is [`Fault::BufferOverrun`] and
//!   writes nothing; a method that means to fill what fits says so with [`OutBuffer::len`].
//!
//! Not reachable through a request: W (exchange) descriptors, which Eden parses and never reads,
//! and the inline receive list (flag 1), where there are no C descriptors and Eden's
//! `WriteBuffer` finds none either.

use starquake_ipc::Message;
use starquake_kernel::handle::Handle;
use starquake_kernel::id::ObjectId;
use starquake_kernel::result::{KernelError, SvcResult};

use super::commands::{Protocol, Session};
use super::context::Context;
use super::kernel::Kernel;
use super::payload::Reader;
use super::result::{Direction, Fault, HandleKind};
use super::targets::TargetId;

/// A request as a service method sees it.
pub struct Request<'a> {
    protocol: Protocol,
    command_id: u32,
    payload: &'a [u8],
    process_id: Option<u64>,
    copy_handles: Vec<SvcResult<ObjectId>>,
    move_handles: Vec<SvcResult<ObjectId>>,
    message: &'a Message<'a>,
    in_objects: Vec<Option<TargetId>>,
}

/// An address range in the caller's memory and a cursor inside it.
#[derive(Debug, Clone, Copy)]
struct Span {
    address: u64,
    len: u64,
    pos: u64,
}

impl Span {
    fn new((address, len): (u64, u64)) -> Self {
        Self {
            address,
            len,
            pos: 0,
        }
    }

    fn remaining(&self) -> u64 {
        self.len.saturating_sub(self.pos)
    }

    /// The address of the next `count` bytes and the cursor after them. Checks the whole extent
    /// against the buffer; the kernel checks it against the caller's memory.
    fn next(&self, count: usize) -> Result<(u64, u64), Fault> {
        let overrun = Fault::BufferOverrun {
            offset: self.pos,
            needed: count,
            len: self.len,
        };
        let end = u64::try_from(count)
            .ok()
            .and_then(|count| self.pos.checked_add(count))
            .filter(|end| *end <= self.len)
            .ok_or(overrun)?;
        let address = self
            .address
            .checked_add(self.pos)
            .ok_or(KernelError::InvalidCurrentMemory)?;
        Ok((address, end))
    }
}

/// An "in" buffer: bytes the caller sends. Reading copies them out of the caller's memory.
pub struct InBuffer<'c> {
    kernel: &'c dyn Kernel,
    span: Span,
}

impl InBuffer<'_> {
    /// The buffer's size in bytes.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.span.len
    }

    /// Whether the buffer holds no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.span.len == 0
    }

    /// Bytes not read yet.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.span.remaining()
    }

    /// Fills `out` with the next `out.len()` bytes. Reading past the end is
    /// [`Fault::BufferOverrun`]; memory the caller has not mapped is the kernel's error. Either
    /// way nothing is read.
    pub fn read(&mut self, out: &mut [u8]) -> Result<(), Fault> {
        let (address, end) = self.span.next(out.len())?;
        if !out.is_empty() {
            self.kernel.read_memory(address, out)?;
        }
        self.span.pos = end;
        Ok(())
    }

    /// Reads every byte not read yet. The size is the caller's to choose, so the method states
    /// the most it accepts: more than `limit` bytes is [`Fault::BufferTooLarge`], before
    /// anything is allocated.
    pub fn read_all(&mut self, limit: usize) -> Result<Vec<u8>, Fault> {
        let len = self.span.remaining();
        let too_large = Fault::BufferTooLarge { len, limit };
        let fits = usize::try_from(len).ok().filter(|len| *len <= limit);
        let mut bytes = vec![0; fits.ok_or(too_large)?];
        self.read(&mut bytes)?;
        Ok(bytes)
    }
}

/// An "out" buffer: room the caller offers. Writing copies into the caller's memory.
pub struct OutBuffer<'c> {
    kernel: &'c mut dyn Kernel,
    span: Span,
}

impl OutBuffer<'_> {
    /// The buffer's size in bytes.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.span.len
    }

    /// Whether the buffer has no room at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.span.len == 0
    }

    /// Bytes written so far.
    #[must_use]
    pub fn written(&self) -> u64 {
        self.span.pos
    }

    /// Appends `data`. Writing past the end is [`Fault::BufferOverrun`]; memory the caller has
    /// not mapped is the kernel's error. Either way nothing is written.
    pub fn write(&mut self, data: &[u8]) -> Result<(), Fault> {
        let (address, end) = self.span.next(data.len())?;
        if !data.is_empty() {
            self.kernel.write_memory(address, data)?;
        }
        self.span.pos = end;
        Ok(())
    }
}

/// Eden's choice between two descriptor lists: the first if it has a non-empty entry, otherwise
/// the second; and, where Eden logs a failed assertion because there is no second, the first's
/// empty entry.
fn pick(first: Option<(u64, u64)>, second: Option<(u64, u64)>) -> Option<(u64, u64)> {
    match first {
        Some((_, size)) if size != 0 => first,
        _ => second.or(first),
    }
}

impl<'a> Request<'a> {
    pub(super) fn new(
        kernel: &dyn Kernel,
        protocol: Protocol,
        command_id: u32,
        payload: &'a [u8],
        message: &'a Message<'a>,
        in_objects: Vec<Option<TargetId>>,
    ) -> Self {
        let resolve = |handles: &[u32]| {
            let resolved = handles.iter().map(|raw| Handle::from_raw(*raw));
            resolved
                .map(|handle| kernel.resolve_handle(handle))
                .collect()
        };
        Self {
            protocol,
            command_id,
            payload,
            process_id: message.pid.map(|_| kernel.process_id()),
            copy_handles: resolve(&message.copy_handles),
            move_handles: resolve(&message.move_handles),
            message,
            in_objects,
        }
    }

    /// Which protocol the request came in.
    #[must_use]
    pub fn protocol(&self) -> Protocol {
        self.protocol
    }

    /// The command id.
    #[must_use]
    pub fn command_id(&self) -> u32 {
        self.command_id
    }

    /// A reader at the start of the raw payload: the bytes after the `SFCI` header, or all of a
    /// TIPC message's raw data. On an ordinary CMIF session the payload runs to the end of the
    /// message's raw data and so can end in padding the caller added; a method reads the fields
    /// it knows and ignores the rest.
    #[must_use]
    pub fn payload(&self) -> Reader<'a> {
        Reader::new(self.payload)
    }

    /// The caller's process id, if the request asked the kernel to attach it.
    #[must_use]
    pub fn process_id(&self) -> Option<u64> {
        self.process_id
    }

    fn handle(
        list: &[SvcResult<ObjectId>],
        kind: HandleKind,
        index: usize,
    ) -> Result<ObjectId, Fault> {
        let resolved = list.get(index).ok_or(Fault::NoHandle { kind, index })?;
        Ok((*resolved)?)
    }

    /// The object behind the `index`th handle the caller copied. A handle that names nothing in
    /// the caller's table is the kernel's error, raised here and not before, so a command that
    /// never looks at a handle is not failed by it.
    pub fn copy_handle(&self, index: usize) -> Result<ObjectId, Fault> {
        Self::handle(&self.copy_handles, HandleKind::Copy, index)
    }

    /// The object behind the `index`th handle the caller moved. As [`Request::copy_handle`].
    pub fn move_handle(&self, index: usize) -> Result<ObjectId, Fault> {
        Self::handle(&self.move_handles, HandleKind::Move, index)
    }

    fn span(&self, direction: Direction, index: usize) -> Result<Span, Fault> {
        let message = self.message;
        let (first, second) = match direction {
            Direction::In => (
                message.send_buffers.get(index).map(|d| (d.address, d.size)),
                (message.static_descriptors.get(index)).map(|d| (d.address, u64::from(d.size))),
            ),
            Direction::Out => (
                (message.receive_buffers.get(index)).map(|d| (d.address, d.size)),
                (message.receive_descriptors.get(index)).map(|d| (d.address, u64::from(d.size))),
            ),
        };
        let range = pick(first, second).ok_or(Fault::NoBuffer { direction, index })?;
        Ok(Span::new(range))
    }

    /// The `index`th in buffer (an A or X descriptor; see the [module docs](self)).
    pub fn in_buffer<'c>(
        &self,
        cx: &'c mut Context<'_>,
        index: usize,
    ) -> Result<InBuffer<'c>, Fault> {
        let span = self.span(Direction::In, index)?;
        Ok(InBuffer {
            kernel: &*cx.kernel,
            span,
        })
    }

    /// The `index`th out buffer (a B or C descriptor; see the [module docs](self)).
    pub fn out_buffer<'c>(
        &self,
        cx: &'c mut Context<'_>,
        index: usize,
    ) -> Result<OutBuffer<'c>, Fault> {
        let span = self.span(Direction::Out, index)?;
        Ok(OutBuffer {
            kernel: &mut *cx.kernel,
            span,
        })
    }

    /// The `index`th input object of a domain request, as the interface type the command takes.
    ///
    /// [`Fault::NoObject`] if the request has no such input object or the id there names nothing
    /// in the domain; [`Fault::WrongObject`] if the object is not a `T`, or is the very object
    /// the command was sent to (which is busy handling it).
    pub fn in_object<'c, T: Session>(
        &self,
        cx: &'c mut Context<'_>,
        index: usize,
    ) -> Result<&'c mut T, Fault> {
        let target = self.in_objects.get(index).copied().flatten();
        let target = target.ok_or(Fault::NoObject { index })?;
        let session = cx.targets.interface_mut(target);
        let session: &mut dyn core::any::Any = session.ok_or(Fault::WrongObject { index })?;
        session.downcast_mut().ok_or(Fault::WrongObject { index })
    }
}

#[cfg(test)]
mod tests {
    use starquake_ipc::{BUFFER_SIZE, CommandType, RequestBuilder, SessionKind};

    use std::sync::Arc;

    use super::super::defer::{Deferral, Mailbox};
    use super::super::fake::FakeKernel;
    use super::super::targets::Targets;
    use super::*;
    use crate::commands;

    const BASE: u64 = 0x4000;

    /// An interface that holds a number.
    #[derive(serde::Serialize)]
    struct Probe(u32);

    commands! {
        impl Probe {}
    }

    /// A different interface.
    #[derive(serde::Serialize)]
    struct Other;

    commands! {
        impl Other {}
    }

    #[test]
    fn each_accessor_names_exactly_what_the_request_lacks() {
        let mut kernel = FakeKernel::new(0).unwrap();
        let mut targets = Targets::default();
        let probe = targets.insert_interface(Box::new(Probe(7))).unwrap();
        let other = targets.insert_interface(Box::new(Other)).unwrap();
        let busy = targets.insert_interface(Box::new(Probe(8))).unwrap();
        let checked_out = targets.check_out(busy).unwrap();

        // A request with no handles and no buffers, and four input object slots.
        let mut buffer = vec![0; BUFFER_SIZE];
        let built = RequestBuilder::cmif(CommandType::Request, 9).build(&mut buffer);
        built.unwrap();
        let message = Message::parse_request(&buffer, SessionKind::Session).unwrap();
        let in_objects = vec![Some(probe), None, Some(other), Some(busy)];
        let request = Request::new(&kernel, Protocol::Cmif, 9, &[], &message, in_objects);
        let mailbox = Arc::new(Mailbox::new(None));
        let mut next_token = 0;
        let mut cx = Context {
            kernel: &mut kernel,
            targets: &mut targets,
            family: None,
            deferral: Deferral {
                mailbox: &mailbox,
                next_token: &mut next_token,
                taken: None,
            },
        };

        assert_eq!(
            (request.protocol(), request.command_id()),
            (Protocol::Cmif, 9)
        );
        assert_eq!(request.process_id(), None);
        assert_eq!(request.payload().remaining(), 0);
        let no_handle = |kind| Fault::NoHandle { kind, index: 0 };
        assert_eq!(request.copy_handle(0), Err(no_handle(HandleKind::Copy)));
        assert_eq!(request.move_handle(0), Err(no_handle(HandleKind::Move)));
        let no_buffer = |direction| Fault::NoBuffer {
            direction,
            index: 0,
        };
        let in_buffer = request.in_buffer(&mut cx, 0).map(|b| b.len());
        assert_eq!(in_buffer, Err(no_buffer(Direction::In)));
        let out_buffer = request.out_buffer(&mut cx, 0).map(|b| b.len());
        assert_eq!(out_buffer, Err(no_buffer(Direction::Out)));

        let mut number = |index| request.in_object::<Probe>(&mut cx, index).map(|p| p.0);
        assert_eq!(number(0), Ok(7));
        // An id that named nothing, and an index past the list.
        assert_eq!(number(1), Err(Fault::NoObject { index: 1 }));
        assert_eq!(number(4), Err(Fault::NoObject { index: 4 }));
        // Another interface, and one that is out handling a request.
        assert_eq!(number(2), Err(Fault::WrongObject { index: 2 }));
        assert_eq!(number(3), Err(Fault::WrongObject { index: 3 }));
        targets.check_in(busy, checked_out).unwrap();
    }

    fn span(address: u64, len: u64) -> Span {
        Span::new((address, len))
    }

    #[test]
    fn a_span_hands_out_extents_up_to_its_end_and_no_further() {
        let mut span = span(BASE, 8);
        assert_eq!(span.next(3).unwrap(), (BASE, 3));
        span.pos = 3;
        // Exactly to the end.
        assert_eq!(span.next(5).unwrap(), (BASE + 3, 8));
        // One byte past it.
        let overrun = Fault::BufferOverrun {
            offset: 3,
            needed: 6,
            len: 8,
        };
        assert_eq!(span.next(6), Err(overrun));
        assert_eq!(span.remaining(), 5);
        // A count too large to add to the cursor is an overrun too, not a wrapped offset.
        let huge = Fault::BufferOverrun {
            offset: 3,
            needed: usize::MAX,
            len: 8,
        };
        assert_eq!(span.next(usize::MAX), Err(huge));
        // Nothing at the end is still inside.
        span.pos = 8;
        assert_eq!(span.next(0).unwrap(), (BASE + 8, 8));
        assert_eq!(span.remaining(), 0);
    }

    #[test]
    fn a_span_whose_addresses_wrap_is_a_memory_fault() {
        // The buffer claims the last two bytes of the address space and two more.
        let mut span = span(u64::MAX - 1, 4);
        assert_eq!(span.next(1).unwrap(), (u64::MAX - 1, 1));
        span.pos = 1;
        assert_eq!(span.next(1).unwrap(), (u64::MAX, 2));
        span.pos = 2;
        let fault = Fault::Kernel(KernelError::InvalidCurrentMemory);
        assert_eq!(span.next(1), Err(fault));
    }

    #[test]
    fn an_in_buffer_reads_in_pieces_and_stops_at_its_end() {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.map(BASE, vec![1, 2, 3, 4, 5, 6]);
        let mut buffer = InBuffer {
            kernel: &kernel,
            span: span(BASE + 1, 4),
        };
        assert_eq!(
            (buffer.len(), buffer.is_empty(), buffer.remaining()),
            (4, false, 4)
        );
        let mut pair = [0; 2];
        buffer.read(&mut pair).unwrap();
        assert_eq!(pair, [2, 3]);
        assert_eq!(buffer.remaining(), 2);
        // Three more would pass the buffer's end, though the memory behind it is mapped.
        let overrun = Fault::BufferOverrun {
            offset: 2,
            needed: 3,
            len: 4,
        };
        assert_eq!(buffer.read(&mut [0; 3]), Err(overrun));
        assert_eq!(buffer.remaining(), 2);
        // The rest, with a limit exactly its size; one less is too large.
        let too_large = Fault::BufferTooLarge { len: 2, limit: 1 };
        assert_eq!(buffer.read_all(1), Err(too_large));
        assert_eq!(buffer.read_all(2).unwrap(), [4, 5]);
        assert_eq!(buffer.remaining(), 0);
        assert_eq!(buffer.read_all(0).unwrap(), []);
    }

    #[test]
    fn an_in_buffer_larger_than_any_limit_is_refused_before_it_is_read() {
        let kernel = FakeKernel::new(0).unwrap();
        // The largest size an A descriptor can state; nothing is mapped, and nothing is asked of
        // the kernel.
        let len = (1 << 36) - 1;
        let mut buffer = InBuffer {
            kernel: &kernel,
            span: span(BASE, len),
        };
        let limit = 0x1000;
        assert_eq!(
            buffer.read_all(limit),
            Err(Fault::BufferTooLarge { len, limit })
        );
        assert_eq!(buffer.remaining(), len);
    }

    #[test]
    fn an_out_buffer_writes_in_pieces_and_stops_at_its_end() {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.map(BASE, vec![0; 6]);
        let mut buffer = OutBuffer {
            kernel: &mut kernel,
            span: span(BASE + 1, 4),
        };
        assert_eq!(
            (buffer.len(), buffer.is_empty(), buffer.written()),
            (4, false, 0)
        );
        buffer.write(&[7, 8]).unwrap();
        buffer.write(&[]).unwrap();
        assert_eq!(buffer.written(), 2);
        let overrun = Fault::BufferOverrun {
            offset: 2,
            needed: 3,
            len: 4,
        };
        assert_eq!(buffer.write(&[9, 9, 9]), Err(overrun));
        assert_eq!(buffer.written(), 2);
        buffer.write(&[9, 10]).unwrap();
        assert_eq!(buffer.written(), 4);
        assert_eq!(kernel.memory(BASE, 6).unwrap(), [0, 7, 8, 9, 10, 0]);
    }

    #[test]
    fn a_failed_kernel_copy_leaves_the_cursor_where_it_was() {
        let mut kernel = FakeKernel::new(0).unwrap();
        kernel.map(BASE, vec![0; 2]);
        // The buffer claims four bytes; only two are mapped.
        let fault = Fault::Kernel(KernelError::InvalidCurrentMemory);
        let mut out = OutBuffer {
            kernel: &mut kernel,
            span: span(BASE, 4),
        };
        assert_eq!(out.write(&[1, 2, 3]), Err(fault));
        assert_eq!(out.written(), 0);
        out.write(&[1, 2]).unwrap();
        let mut input = InBuffer {
            kernel: &kernel,
            span: span(BASE, 4),
        };
        assert_eq!(input.read(&mut [0; 3]), Err(fault));
        assert_eq!(input.remaining(), 4);
        assert_eq!(input.read_all(4), Err(fault));
    }

    #[test]
    fn the_first_list_wins_unless_its_entry_is_empty() {
        let (a, x) = (Some((1, 5)), Some((2, 6)));
        let (empty_a, empty_x) = (Some((1, 0)), Some((2, 0)));
        assert_eq!(pick(a, x), a);
        assert_eq!(pick(a, None), a);
        assert_eq!(pick(None, x), x);
        assert_eq!(pick(empty_a, x), x);
        // An empty second entry still stands in for an empty first one, as in Eden.
        assert_eq!(pick(empty_a, empty_x), empty_x);
        // Where Eden logs a failed assertion: the empty first entry is the buffer.
        assert_eq!(pick(empty_a, None), empty_a);
        assert_eq!(pick(None, None), None);
    }
}

//! The kernel's half of a request message: the process id and the handles of the special header
//! (kernel spec, "IPC": "the kernel handles the HIPC layer ... the process id, copy and move
//! handles resolved against the caller's handle table").
//!
//! Ported from Eden `eden/src/core/hle/kernel/k_server_session.cpp` @ cf654052
//! (GPL-3.0-or-later), `ProcessMessageSpecialData<true>` (the path `ReceiveMessage` takes for a
//! request):
//!
//! - The process id slot, when the special header asks for it, is overwritten with the sender's
//!   process id (`dst_msg.SetProcessId(offset, src_process.GetProcessId())`).
//! - Each copied handle that is not [`Handle::INVALID`] is looked up in the sender's table
//!   (`GetObjectForIpc`) while no earlier handle has failed; one that names nothing makes the
//!   request's result [`KernelError::InvalidHandle`], and the handles after it are skipped.
//!   The two pseudo-handles are answered without the table: [`Handle::CURRENT_THREAD`] names
//!   the sending thread and [`Handle::CURRENT_PROCESS`] its process ([`Current`]), and the
//!   request holds a reference to that object as it does for any other copied handle
//!   (`KHandleTable::GetObjectForIpc`, `eden/src/core/hle/kernel/k_handle_table.cpp` @
//!   cf654052: `cur_thread->GetOwnerProcess()` for `PseudoHandle::CurrentProcess`, `cur_thread`
//!   for `PseudoHandle::CurrentThread`).
//! - Each moved handle that is not [`Handle::INVALID`] is removed from the sender's table
//!   whether or not the request has failed (`src_handle_table.Remove`); while it has not, the
//!   handle is looked up first without pseudo-handles (`GetObjectForIpcWithoutPseudoHandle`,
//!   `eden/src/core/hle/kernel/k_handle_table.h` @ cf654052, which reads only the table), and
//!   one that names nothing is [`KernelError::InvalidHandle`]. A moved pseudo-handle is
//!   therefore [`KernelError::InvalidHandle`], and removing it removes nothing (`Remove`
//!   returns false for a pseudo-handle, `k_handle_table.cpp`); the moved handles after it are
//!   still removed. Test:
//!   `a_moved_pseudo_handle_is_an_invalid_handle_and_later_moves_still_leave_the_table`.
//!
//! Once the request has failed, a moved handle's object is not held, so removing the handle can
//! drop the object's last reference. [`Translation::destroyed`] lists those objects, and the
//! caller tears each one down as closing its last handle does (for a session,
//! [`crate::ipc::Ipc::forget`]: its row goes and its server is told).
//!
//! Where Horizon adds each object to the receiving process's table and writes the new handle
//! into the message, an HLE server has no process and no table. Instead the request holds one
//! reference to each object for as long as it is in flight, the handle words are left as the
//! sender wrote them, and [`Translated::object`] answers what each one named. The server reads
//! the handles from the message and asks for them by value (`IpcContext::resolve_handle`). A
//! guest cannot tell the difference: the message the server sees is not visible to it.
//!
//! Divergences from Eden, each pinned by the named test below:
//! - A copied pseudo-handle whose object the caller does not supply ([`Current`] holds `None`
//!   for it) is [`KernelError::InvalidHandle`]. Eden asserts that the current thread and its
//!   process exist (`ASSERT(cur_thread != nullptr)`, `ASSERT(cur_process != nullptr)` in
//!   `GetObjectForIpc`). Here the dispatcher supplies `None` for the thread only when the
//!   calling core runs no scheduler thread that has an object, which no guest thread is; every
//!   process has an object from its creation. Test:
//!   `a_copied_pseudo_handle_whose_object_is_missing_is_an_invalid_handle`.
//! - When a reference cannot be opened (an object's count is at its maximum), the request is
//!   [`KernelError::OutOfResource`]; Eden's `KScopedAutoObject` cannot fail. Unreachable from a
//!   guest in practice (four billion references); not separately tested.

use starquake_ipc::{BUFFER_SIZE, CommandHeader, SpecialHeader};

use crate::handle::{Closed, Handle, HandleTable, ObjectRefs, Resolved};
use crate::id::ObjectId;
use crate::object::Objects;
use crate::result::KernelError;

/// The size of a message word.
const WORD: usize = size_of::<u32>();
/// Where the second header word, which says whether a special header follows, starts.
const HEADER_WORD_1: usize = WORD;
/// Where the special header starts: after the two header words.
const SPECIAL_HEADER: usize = CommandHeader::WORDS * WORD;
/// The size of the process id slot.
pub(crate) const PROCESS_ID_SIZE: usize = size_of::<u64>();
/// Where the process id slot is, when the special header asks for one: after the special header.
pub(crate) const PROCESS_ID_SLOT: usize = SPECIAL_HEADER + WORD;

/// A request's message as an array, so that every offset below is checked against one size.
pub type MessageBuffer = [u8; BUFFER_SIZE];

/// What translating a request's special header left the request holding.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Translated {
    /// One reference to each object a copied or moved handle named, held for the request.
    pub held: Vec<ObjectId>,
    /// Each handle value in the message that was translated, and the object it named.
    pub handles: Vec<(Handle, ObjectId)>,
}

impl Translated {
    /// The object `handle` named when the request was sent, if it is one of the message's
    /// translated handles.
    #[must_use]
    pub fn object(&self, handle: Handle) -> Option<ObjectId> {
        self.handles
            .iter()
            .find(|(translated, _)| *translated == handle)
            .map(|(_, object)| *object)
    }
}

/// The objects the two pseudo-handles name for a request's sender: what Eden's
/// `GetObjectForIpc` answers for [`Handle::CURRENT_THREAD`] (`cur_thread`) and
/// [`Handle::CURRENT_PROCESS`] (`cur_thread->GetOwnerProcess()`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Current {
    /// The sending thread's object, or `None` when the calling core runs no thread with one.
    pub thread: Option<ObjectId>,
    /// The sending process's object, or `None` when the process has none.
    pub process: Option<ObjectId>,
}

/// What [`translate`] did to a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Translation {
    /// What the request holds.
    pub translated: Translated,
    /// The request's result: on an error the request does not reach the server, and its
    /// holdings are released by the caller as for any finished request.
    pub result: Result<(), KernelError>,
    /// The objects whose last reference went with a moved handle's removal, in the order they
    /// were removed. Only a request that has already failed can list any (one that has not
    /// holds each moved object first). Their rows are freed; the caller tears down what else
    /// named them.
    pub destroyed: Vec<ObjectId>,
}

/// The word at byte `offset` of the message, or `None` past its end.
fn word(message: &MessageBuffer, offset: usize) -> Option<u32> {
    let end = offset.checked_add(WORD)?;
    let bytes: [u8; WORD] = message.get(offset..end)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

/// The special header of `message`, when its second header word says one follows, and the
/// byte offset of its first handle word: after the process id slot when the special header asks
/// for one (`MessageBuffer::GetSpecialDataIndex`, `SetProcessId` and `GetHandle` in
/// `eden/src/core/hle/kernel/message_buffer.h` @ cf654052). What [`translate`] reads a request
/// with and [`reply_handles`] a reply with.
///
/// Two header words, the special header, a process id and fifteen handles of each kind (the
/// counts are four bits wide) fit well inside a message, so no handle word read from the offset
/// returned can run past its end.
fn special_data(message: &MessageBuffer) -> Option<(SpecialHeader, usize)> {
    let follows = word(message, HEADER_WORD_1).is_some_and(CommandHeader::special_header_follows);
    if !follows {
        return None;
    }
    let special = word(message, SPECIAL_HEADER).map(SpecialHeader::decode)?;
    let handles = if special.send_pid {
        PROCESS_ID_SLOT + PROCESS_ID_SIZE
    } else {
        PROCESS_ID_SLOT
    };
    Some((special, handles))
}

/// The handle words of a reply's special header that are not [`Handle::INVALID`], the copied
/// ones and then the moved ones, in the order the message carries them: the handles the reply
/// gives its receiver. Empty when no special header follows the header.
///
/// This is the walk Eden's `CleanupSpecialData` (`eden/src/core/hle/kernel/k_server_session.cpp`
/// @ cf654052) makes over a reply it has translated into the receiver's message: every copy and
/// move handle word in order, skipping `Svc::InvalidHandle`. The dispatcher keeps exactly these
/// of the handles a handler placed in the sender's table (`dispatch/ipc/placed.rs`).
pub(crate) fn reply_handles(message: &MessageBuffer) -> Vec<Handle> {
    let Some((special, first)) = special_data(message) else {
        return Vec::new();
    };
    let count = special.num_copy_handles + special.num_move_handles;
    (0..count)
        .filter_map(|n| {
            let offset = usize::try_from(n)
                .ok()?
                .checked_mul(WORD)?
                .checked_add(first)?;
            word(message, offset)
        })
        .map(Handle::from_raw)
        .filter(|handle| *handle != Handle::INVALID)
        .collect()
}

/// Translates the special header of `message`, a request sent by a thread of the process whose
/// id is `process_id` and whose handles are `handles`: writes the process id where it was asked
/// for and resolves the copied and moved handles (see the module docs).
///
/// See [`Translation`] for what it returns. The moved handles are out of the sender's table
/// whether or not the request failed.
pub fn translate(
    message: &mut MessageBuffer,
    process_id: u64,
    current: Current,
    handles: &mut HandleTable,
    objects: &mut Objects,
) -> Translation {
    let mut translation = Translation {
        translated: Translated::default(),
        result: Ok(()),
        destroyed: Vec::new(),
    };
    let Some((special, mut offset)) = special_data(message) else {
        return translation;
    };
    let Translation {
        translated,
        result,
        destroyed,
    } = &mut translation;
    if special.send_pid
        && let Some(slot) = message.get_mut(PROCESS_ID_SLOT..offset)
    {
        slot.copy_from_slice(&process_id.to_le_bytes());
    }
    for _ in 0..special.num_copy_handles {
        let handle = Handle::from_raw(word(message, offset).unwrap_or_default());
        offset += WORD;
        if result.is_err() || handle == Handle::INVALID {
            continue;
        }
        // `GetObjectForIpc`: a pseudo-handle names the sender's thread or process.
        let object = match handles.resolve(handle) {
            Ok(Resolved::CurrentThread) => current.thread,
            Ok(Resolved::CurrentProcess) => current.process,
            Ok(Resolved::Object(object)) => Some(object),
            Err(_) => None,
        };
        *result = hold(translated, objects, handle, object);
    }
    for _ in 0..special.num_move_handles {
        let handle = Handle::from_raw(word(message, offset).unwrap_or_default());
        offset += WORD;
        if handle == Handle::INVALID {
            continue;
        }
        if result.is_ok() {
            *result = hold(translated, objects, handle, handles.get(handle).ok());
        }
        // Removed whether or not the request failed; a handle that names nothing has nothing
        // to remove, which is already the request's error.
        if let Ok(removed) = handles.remove(objects, handle)
            && removed.closed == Closed::Last
        {
            destroyed.push(removed.object);
        }
    }
    translation
}

/// Takes a reference to the object a handle named for the request, or fails the request.
fn hold(
    translated: &mut Translated,
    objects: &mut Objects,
    handle: Handle,
    object: Option<ObjectId>,
) -> Result<(), KernelError> {
    let object = object.ok_or(KernelError::InvalidHandle)?;
    objects
        .open(object)
        .map_err(|_| KernelError::OutOfResource)?;
    translated.held.push(object);
    translated.handles.push((handle, object));
    Ok(())
}

#[cfg(test)]
mod tests {
    use starquake_ipc::{CommandType, ReplyBuilder, RequestBuilder};

    use super::*;
    use crate::id::{ProcessId, SlabId, ThreadId};
    use crate::object::Object;

    /// A handle table and an object table holding `count` thread objects, each with one handle.
    fn table(count: u32) -> (HandleTable, Objects, Vec<Handle>) {
        let mut handles = HandleTable::new(8).unwrap();
        let mut objects = Objects::new();
        let mut made = Vec::new();
        for n in 0..count {
            let object = objects
                .create(Object::Thread(ThreadId::from_parts(n, 1)))
                .unwrap();
            made.push(handles.add(&mut objects, object).unwrap());
            objects.close(object).unwrap();
        }
        (handles, objects, made)
    }

    fn build(builder: &RequestBuilder) -> MessageBuffer {
        let mut message = [0; BUFFER_SIZE];
        builder.build(&mut message).unwrap();
        message
    }

    fn request() -> RequestBuilder {
        RequestBuilder::cmif(CommandType::Request, 7).data(&[0xAB; 8])
    }

    /// Where a message built by [`request`] with these special-header fields keeps its handles.
    fn handle_offset(pid: bool) -> usize {
        PROCESS_ID_SLOT + if pid { PROCESS_ID_SIZE } else { 0 }
    }

    #[test]
    fn a_message_without_a_special_header_is_left_alone() {
        let (mut handles, mut objects, _) = table(1);
        let mut message = build(&request());
        let before = message;
        let Translation {
            translated,
            result,
            destroyed,
        } = translate(
            &mut message,
            0x51,
            Current::default(),
            &mut handles,
            &mut objects,
        );
        assert_eq!((translated, result), (Translated::default(), Ok(())));
        assert_eq!(destroyed, Vec::<ObjectId>::new());
        assert_eq!(message, before);
    }

    #[test]
    fn the_process_id_slot_gets_the_senders_process_id() {
        let (mut handles, mut objects, _) = table(0);
        let mut message = build(&request().send_pid(0x1111_2222_3333_4444));
        let mut expected = message;
        let slot = PROCESS_ID_SLOT;
        expected[slot..slot + PROCESS_ID_SIZE].copy_from_slice(&0x51_u64.to_le_bytes());
        let Translation {
            translated,
            result,
            destroyed,
        } = translate(
            &mut message,
            0x51,
            Current::default(),
            &mut handles,
            &mut objects,
        );
        assert_eq!((translated, result), (Translated::default(), Ok(())));
        assert_eq!(destroyed, Vec::<ObjectId>::new());
        assert_eq!(message, expected);
    }

    #[test]
    fn copied_handles_stay_in_the_table_and_the_request_holds_a_reference() {
        let (mut handles, mut objects, made) = table(2);
        let (a, b) = (made[0], made[1]);
        let (object_a, object_b) = (handles.get(a).unwrap(), handles.get(b).unwrap());
        let mut message = build(
            &request()
                .send_pid(0)
                .copy_handle(a.raw())
                .copy_handle(b.raw()),
        );
        let before = message;
        let Translation {
            translated,
            result,
            destroyed,
        } = translate(
            &mut message,
            0,
            Current::default(),
            &mut handles,
            &mut objects,
        );
        assert_eq!(result, Ok(()));
        assert_eq!(destroyed, Vec::<ObjectId>::new());
        assert_eq!(translated.held, [object_a, object_b]);
        assert_eq!(translated.object(a), Some(object_a));
        assert_eq!(translated.object(b), Some(object_b));
        assert_eq!(translated.object(Handle::INVALID), None);
        assert_eq!(objects.refs(object_a), 2);
        assert_eq!(handles.len(), 2);
        // The handle words are the sender's, after the process id.
        assert_eq!(message, before);
        assert_eq!(word(&message, handle_offset(true)), Some(a.raw()));
    }

    #[test]
    fn moved_handles_leave_the_senders_table_and_their_reference_goes_to_the_request() {
        let (mut handles, mut objects, made) = table(2);
        let (a, b) = (made[0], made[1]);
        let object_b = handles.get(b).unwrap();
        let mut message = build(&request().move_handle(b.raw()));
        let Translation {
            translated,
            result,
            destroyed,
        } = translate(
            &mut message,
            0,
            Current::default(),
            &mut handles,
            &mut objects,
        );
        assert_eq!(result, Ok(()));
        // The request holds the moved object, so removing its handle destroys nothing.
        assert_eq!(destroyed, Vec::<ObjectId>::new());
        assert_eq!(translated.held, [object_b]);
        assert_eq!(translated.object(b), Some(object_b));
        assert_eq!(objects.refs(object_b), 1);
        assert!(handles.get(b).is_err());
        assert!(handles.get(a).is_ok());
        assert_eq!(word(&message, handle_offset(false)), Some(b.raw()));
    }

    #[test]
    fn invalid_handle_words_are_skipped() {
        let (mut handles, mut objects, made) = table(1);
        let mut message = build(
            &request()
                .copy_handle(Handle::INVALID.raw())
                .move_handle(Handle::INVALID.raw()),
        );
        let Translation {
            translated,
            result,
            destroyed,
        } = translate(
            &mut message,
            0,
            Current::default(),
            &mut handles,
            &mut objects,
        );
        assert_eq!((translated, result), (Translated::default(), Ok(())));
        assert_eq!(destroyed, Vec::<ObjectId>::new());
        assert!(handles.get(made[0]).is_ok());
    }

    #[test]
    fn a_bad_copied_handle_fails_the_request_and_later_moves_still_leave_the_table() {
        let (mut handles, mut objects, made) = table(3);
        let (a, b, c) = (made[0], made[1], made[2]);
        let (object_a, object_c) = (handles.get(a).unwrap(), handles.get(c).unwrap());
        let stale = Handle::from_parts(5, 9).unwrap();
        let mut message = build(
            &request()
                .copy_handle(a.raw())
                .copy_handle(stale.raw())
                .copy_handle(b.raw())
                .move_handle(c.raw()),
        );
        let Translation {
            translated,
            result,
            destroyed,
        } = translate(
            &mut message,
            0,
            Current::default(),
            &mut handles,
            &mut objects,
        );
        assert_eq!(result, Err(KernelError::InvalidHandle));
        // The handle before the bad one is held (the caller releases it); none after.
        assert_eq!(translated.held, [object_a]);
        assert_eq!(objects.refs(handles.get(b).unwrap()), 1);
        // The moved handle left the table although the request failed, and with it its
        // object's last reference.
        assert!(handles.get(c).is_err());
        assert_eq!(handles.len(), 2);
        assert_eq!(destroyed, [object_c]);
        assert_eq!(
            objects.get(object_c),
            Err(crate::object::ObjectError::Stale { id: object_c })
        );
    }

    #[test]
    fn a_bad_moved_handle_fails_the_request_and_is_not_held() {
        let (mut handles, mut objects, _) = table(0);
        // A session whose only reference is the handle moved after the bad one.
        let session = objects.create(Object::Session).unwrap();
        let moved = handles.add(&mut objects, session).unwrap();
        objects.close(session).unwrap();
        let stale = Handle::from_parts(6, 3).unwrap();
        let mut message = build(&request().move_handle(stale.raw()).move_handle(moved.raw()));
        let Translation {
            translated,
            result,
            destroyed,
        } = translate(
            &mut message,
            0,
            Current::default(),
            &mut handles,
            &mut objects,
        );
        assert_eq!(result, Err(KernelError::InvalidHandle));
        assert_eq!(translated.held, Vec::<ObjectId>::new());
        // The good handle after it still left the table, and its session with it: the caller
        // is told, to tear the session down.
        assert!(handles.is_empty());
        assert!(objects.is_empty());
        assert_eq!(destroyed, [session]);
    }

    /// The sender's thread and process as objects, each holding one reference (its owner's:
    /// the scheduler's thread row and process row hold them so), with ids no table entry has.
    fn current(objects: &mut Objects) -> Current {
        let thread = objects
            .create(Object::Thread(ThreadId::from_parts(40, 2)))
            .unwrap();
        let process = objects
            .create(Object::Process(ProcessId::from_parts(0, 1)))
            .unwrap();
        Current {
            thread: Some(thread),
            process: Some(process),
        }
    }

    /// What `pseudo` names in `current`, and the object the other pseudo-handle names.
    fn named(current: Current, pseudo: Handle) -> (ObjectId, ObjectId) {
        let (thread, process) = (current.thread.unwrap(), current.process.unwrap());
        if pseudo == Handle::CURRENT_THREAD {
            (thread, process)
        } else {
            (process, thread)
        }
    }

    #[test]
    fn a_copied_pseudo_handle_is_the_senders_thread_or_process_and_the_request_holds_it() {
        for pseudo in [Handle::CURRENT_THREAD, Handle::CURRENT_PROCESS] {
            let (mut handles, mut objects, made) = table(1);
            let current = current(&mut objects);
            let (expected, other) = named(current, pseudo);
            let mut message = build(&request().copy_handle(pseudo.raw()));
            let before = message;
            let Translation {
                translated,
                result,
                destroyed,
            } = translate(&mut message, 0, current, &mut handles, &mut objects);
            assert_eq!(result, Ok(()), "{pseudo:?}");
            assert_eq!(destroyed, Vec::<ObjectId>::new());
            assert_eq!(translated.held, [expected]);
            assert_eq!(translated.object(pseudo), Some(expected));
            // The request opened a reference, as for an ordinary copied handle; the other
            // pseudo-handle's object and the table are untouched.
            assert_eq!((objects.refs(expected), objects.refs(other)), (2, 1));
            assert_eq!(handles.len(), 1);
            assert!(handles.get(made[0]).is_ok());
            // The handle word is the sender's.
            assert_eq!(message, before);
        }
    }

    #[test]
    fn pseudo_handles_copied_with_an_ordinary_handle_each_name_their_own_object_in_order() {
        let (mut handles, mut objects, made) = table(1);
        let a = made[0];
        let object_a = handles.get(a).unwrap();
        let current = current(&mut objects);
        let (thread, process) = (current.thread.unwrap(), current.process.unwrap());
        let mut message = build(
            &request()
                .send_pid(0)
                .copy_handle(Handle::CURRENT_PROCESS.raw())
                .copy_handle(a.raw())
                .copy_handle(Handle::CURRENT_THREAD.raw()),
        );
        let Translation {
            translated,
            result,
            destroyed,
        } = translate(&mut message, 0x51, current, &mut handles, &mut objects);
        assert_eq!(result, Ok(()));
        assert_eq!(destroyed, Vec::<ObjectId>::new());
        assert_eq!(translated.held, [process, object_a, thread]);
        assert_eq!(translated.object(Handle::CURRENT_PROCESS), Some(process));
        assert_eq!(translated.object(Handle::CURRENT_THREAD), Some(thread));
        assert_eq!(translated.object(a), Some(object_a));
        for object in [process, object_a, thread] {
            assert_eq!(objects.refs(object), 2);
        }
    }

    #[test]
    fn a_copied_pseudo_handle_whose_object_is_missing_is_an_invalid_handle() {
        for pseudo in [Handle::CURRENT_THREAD, Handle::CURRENT_PROCESS] {
            let (mut handles, mut objects, _) = table(0);
            let full = current(&mut objects);
            let (_, other) = named(full, pseudo);
            // Only the other pseudo-handle has an object, so a swap of the two shows; and
            // then neither has one.
            let only_other = if pseudo == Handle::CURRENT_THREAD {
                Current {
                    thread: None,
                    ..full
                }
            } else {
                Current {
                    process: None,
                    ..full
                }
            };
            for current in [only_other, Current::default()] {
                let mut message = build(&request().copy_handle(pseudo.raw()));
                let Translation {
                    translated,
                    result,
                    destroyed,
                } = translate(&mut message, 0, current, &mut handles, &mut objects);
                assert_eq!(
                    (translated, result),
                    (Translated::default(), Err(KernelError::InvalidHandle)),
                    "{pseudo:?} with {current:?}"
                );
                assert_eq!(destroyed, Vec::<ObjectId>::new());
                assert_eq!(objects.refs(other), 1);
            }
        }
    }

    #[test]
    fn a_copied_pseudo_handle_after_a_failed_one_is_not_held() {
        let (mut handles, mut objects, _) = table(0);
        let current = current(&mut objects);
        let process = current.process.unwrap();
        let stale = Handle::from_parts(5, 9).unwrap();
        let mut message = build(
            &request()
                .copy_handle(stale.raw())
                .copy_handle(Handle::CURRENT_PROCESS.raw()),
        );
        let Translation {
            translated, result, ..
        } = translate(&mut message, 0, current, &mut handles, &mut objects);
        assert_eq!(
            (translated, result),
            (Translated::default(), Err(KernelError::InvalidHandle))
        );
        assert_eq!(objects.refs(process), 1);
    }

    #[test]
    fn a_moved_pseudo_handle_is_an_invalid_handle_and_later_moves_still_leave_the_table() {
        for pseudo in [Handle::CURRENT_THREAD, Handle::CURRENT_PROCESS] {
            let (mut handles, mut objects, made) = table(1);
            let a = made[0];
            let object_a = handles.get(a).unwrap();
            let current = current(&mut objects);
            let (named, other) = named(current, pseudo);
            let mut message = build(&request().move_handle(pseudo.raw()).move_handle(a.raw()));
            let Translation {
                translated,
                result,
                destroyed,
            } = translate(&mut message, 0, current, &mut handles, &mut objects);
            // `GetObjectForIpcWithoutPseudoHandle` names nothing for a pseudo-handle.
            assert_eq!(result, Err(KernelError::InvalidHandle), "{pseudo:?}");
            assert_eq!(translated, Translated::default());
            // Neither the thread nor the process gained or lost a reference.
            assert_eq!((objects.refs(named), objects.refs(other)), (1, 1));
            // The ordinary handle moved after it still left the table, with its object's last
            // reference.
            assert!(handles.is_empty());
            assert_eq!(destroyed, [object_a]);
        }
    }

    #[test]
    fn the_special_header_is_found_whatever_the_message_type() {
        // A message type nothing defines: the kernel's half does not look at it.
        let (mut handles, mut objects, made) = table(1);
        let mut message = build(&request().move_handle(made[0].raw()));
        message[0] = 0xEE;
        message[1] = 0xEE;
        let Translation {
            translated,
            result,
            destroyed,
        } = translate(
            &mut message,
            0,
            Current::default(),
            &mut handles,
            &mut objects,
        );
        assert_eq!(result, Ok(()));
        assert_eq!(destroyed, Vec::<ObjectId>::new());
        assert_eq!(translated.held.len(), 1);
        assert!(handles.is_empty());
    }

    // ----- reply_handles ---------------------------------------------------------------------

    /// A reply built by the IPC crate's reply builder, the way an HLE server frames one.
    fn reply(builder: &ReplyBuilder) -> MessageBuffer {
        let mut message = [0; BUFFER_SIZE];
        builder.build(&mut message).unwrap();
        message
    }

    #[test]
    fn a_reply_without_a_special_header_gives_no_handles() {
        assert_eq!(reply_handles(&reply(&ReplyBuilder::cmif(0))), []);
        assert_eq!(reply_handles(&reply(&ReplyBuilder::tipc(3, 0))), []);
        // A message of zeros, as a reply a handler never wrote leaves it.
        assert_eq!(reply_handles(&[0; BUFFER_SIZE]), []);
    }

    #[test]
    fn a_replys_handles_are_its_copies_then_its_moves_in_order() {
        let built = ReplyBuilder::cmif(0x1234)
            .copy_handle(0x11)
            .copy_handle(0x22)
            .move_handle(0x33)
            .data(&[0xEE; 12]);
        let expected = [0x11, 0x22, 0x33].map(Handle::from_raw);
        assert_eq!(reply_handles(&reply(&built)), expected);
        // The TIPC framing carries the special header in the same place.
        let built = ReplyBuilder::tipc(9, 0)
            .copy_handle(0x44)
            .move_handle(0x55)
            .move_handle(0x66);
        let expected = [0x44, 0x55, 0x66].map(Handle::from_raw);
        assert_eq!(reply_handles(&reply(&built)), expected);
    }

    #[test]
    fn a_hand_built_reply_with_a_process_id_gives_the_handles_after_it() {
        // Words: header (type 0, no descriptors), header word 1 with only the special-header
        // flag, the special header (process id, one copy, one move), the two process id words,
        // then the copied and the moved handle. Built by hand, not by the IPC crate.
        let special = SpecialHeader {
            send_pid: true,
            num_copy_handles: 1,
            num_move_handles: 1,
        };
        let header = CommandHeader {
            command_type: CommandType::Request,
            num_x: 0,
            num_a: 0,
            num_b: 0,
            num_w: 0,
            raw_data_words: 0,
            receive_list: starquake_ipc::ReceiveList::Disabled,
            has_special_header: true,
        };
        let [w0, w1] = header.encode().unwrap();
        let words = [
            w0,
            w1,
            special.encode().unwrap(),
            0xAAAA_0001,
            0xAAAA_0002,
            0x0007_0011,
            0x0009_0022,
        ];
        let mut message = [0; BUFFER_SIZE];
        for (n, word) in words.iter().enumerate() {
            message[n * WORD..(n + 1) * WORD].copy_from_slice(&word.to_le_bytes());
        }
        assert_eq!(
            reply_handles(&message),
            [Handle::from_raw(0x0007_0011), Handle::from_raw(0x0009_0022)]
        );
        // The same words with the process id flag clear: the process id words are read as the
        // handles, so a walk that skipped the slot unconditionally would fail one of the two.
        let without = SpecialHeader {
            send_pid: false,
            ..special
        };
        message[2 * WORD..3 * WORD].copy_from_slice(&without.encode().unwrap().to_le_bytes());
        assert_eq!(
            reply_handles(&message),
            [Handle::from_raw(0xAAAA_0001), Handle::from_raw(0xAAAA_0002)]
        );
    }

    #[test]
    fn a_special_header_word_without_the_flag_is_not_read() {
        // A message whose third word would decode as a special header with a copied and a
        // moved handle, and whose handle words name real handles, but whose second header
        // word does not say a special header follows: neither a reply nor a request has one.
        let (mut handles, mut objects, made) = table(2);
        let mut message = build(
            &request()
                .copy_handle(made[0].raw())
                .move_handle(made[1].raw()),
        );
        let words = [
            word(&message, 0).unwrap(),
            word(&message, HEADER_WORD_1).unwrap(),
        ];
        let mut header = CommandHeader::decode(words).unwrap();
        assert!(header.has_special_header);
        header.has_special_header = false;
        let [_, word1] = header.encode().unwrap();
        message[HEADER_WORD_1..HEADER_WORD_1 + WORD].copy_from_slice(&word1.to_le_bytes());
        assert_eq!(reply_handles(&message), []);
        let before = message;
        let Translation {
            translated,
            result,
            destroyed,
        } = translate(
            &mut message,
            0,
            Current::default(),
            &mut handles,
            &mut objects,
        );
        assert_eq!((translated, result), (Translated::default(), Ok(())));
        assert_eq!(destroyed, Vec::<ObjectId>::new());
        assert_eq!((message, handles.len()), (before, 2));
    }

    #[test]
    fn invalid_handle_words_in_a_reply_are_not_handles() {
        let built = ReplyBuilder::cmif(0)
            .copy_handle(Handle::INVALID.raw())
            .copy_handle(0x21)
            .move_handle(Handle::INVALID.raw())
            .move_handle(0x41);
        assert_eq!(
            reply_handles(&reply(&built)),
            [Handle::from_raw(0x21), Handle::from_raw(0x41)]
        );
    }

    #[test]
    fn the_most_handles_a_special_header_can_count_are_all_read() {
        // As many copies and moves as the special header's counts can say, with a process id
        // before them.
        let most = u32::try_from(most_handles()).unwrap();
        let mut built = request().send_pid(0x5151);
        for n in 0..most {
            built = built.copy_handle(0x100 + n);
        }
        for n in 0..most {
            built = built.move_handle(0x200 + n);
        }
        let handles = reply_handles(&build(&built));
        let expected: Vec<Handle> = (0..most)
            .map(|n| 0x100 + n)
            .chain((0..most).map(|n| 0x200 + n))
            .map(Handle::from_raw)
            .collect();
        assert_eq!(handles, expected);
    }

    /// The most handles of one kind a special header can count: what its decoder reads from a
    /// count field with every bit set.
    fn most_handles() -> usize {
        let most = SpecialHeader::decode(u32::MAX);
        assert_eq!(most.num_copy_handles, most.num_move_handles);
        usize::try_from(most.num_copy_handles).unwrap()
    }

    fn reply_config() -> proptest::test_runner::Config {
        proptest::test_runner::Config {
            rng_seed: proptest::test_runner::RngSeed::Fixed(0x0209),
            ..proptest::test_runner::Config::default()
        }
    }

    proptest::proptest! {
        #![proptest_config(reply_config())]

        /// Whatever the counts and the process id flag, the handles come back as built: copies
        /// then moves, each list in order, so a swapped read of the two lists or a misplaced
        /// first word fails.
        #[test]
        fn built_handles_come_back_in_order(
            pid in proptest::option::of(proptest::prelude::any::<u64>()),
            copies in proptest::collection::vec(1_u32..=u32::MAX, 0..=most_handles()),
            moves in proptest::collection::vec(1_u32..=u32::MAX, 0..=most_handles()),
        ) {
            let mut built = request();
            if let Some(pid) = pid {
                built = built.send_pid(pid);
            }
            for copy in &copies {
                built = built.copy_handle(*copy);
            }
            for moved in &moves {
                built = built.move_handle(*moved);
            }
            let expected: Vec<Handle> =
                copies.iter().chain(&moves).copied().map(Handle::from_raw).collect();
            proptest::prop_assert_eq!(reply_handles(&build(&built)), expected);
        }
    }
}

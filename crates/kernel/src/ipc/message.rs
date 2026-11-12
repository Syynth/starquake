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
//! - Each moved handle that is not [`Handle::INVALID`] is removed from the sender's table
//!   whether or not the request has failed (`src_handle_table.Remove`); while it has not, the
//!   handle is looked up first without pseudo-handles (`GetObjectForIpcWithoutPseudoHandle`),
//!   and one that names nothing is [`KernelError::InvalidHandle`].
//!
//! Where Horizon adds each object to the receiving process's table and writes the new handle
//! into the message, an HLE server has no process and no table. Instead the request holds one
//! reference to each object for as long as it is in flight, the handle words are left as the
//! sender wrote them, and [`Translated::object`] answers what each one named. The server reads
//! the handles from the message and asks for them by value (`IpcContext::resolve_handle`). A
//! guest cannot tell the difference: the message the server sees is not visible to it.
//!
//! Divergences from Eden, each pinned by the named test below:
//! - A pseudo-handle among the copied handles is [`KernelError::InvalidHandle`]. Horizon's
//!   `GetObjectForIpc` gives the current thread or process, and the kernel here has no object
//!   for either yet (kernel slice 8 creates them). Test:
//!   `a_copied_pseudo_handle_is_an_invalid_handle_until_processes_are_objects`.
//! - When a reference cannot be opened (an object's count is at its maximum), the request is
//!   [`KernelError::OutOfResource`]; Eden's `KScopedAutoObject` cannot fail. Unreachable from a
//!   guest in practice (four billion references); not separately tested.

use starquake_ipc::{BUFFER_SIZE, CommandHeader, SpecialHeader};

use crate::handle::{Handle, HandleTable, ObjectRefs};
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
const PROCESS_ID_SIZE: usize = size_of::<u64>();

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

/// The word at byte `offset` of the message, or `None` past its end.
fn word(message: &MessageBuffer, offset: usize) -> Option<u32> {
    let end = offset.checked_add(WORD)?;
    let bytes: [u8; WORD] = message.get(offset..end)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

/// Translates the special header of `message`, a request sent by a thread of the process whose
/// id is `process_id` and whose handles are `handles`: writes the process id where it was asked
/// for and resolves the copied and moved handles (see the module docs).
///
/// Returns what the request holds, and its result: on an error the request does not reach the
/// server, and its holdings are released by the caller as for any finished request. The
/// moved handles are out of the sender's table either way.
pub fn translate(
    message: &mut MessageBuffer,
    process_id: u64,
    handles: &mut HandleTable,
    objects: &mut Objects,
) -> (Translated, Result<(), KernelError>) {
    let mut translated = Translated::default();
    let follows = word(message, HEADER_WORD_1).is_some_and(CommandHeader::special_header_follows);
    if !follows {
        return (translated, Ok(()));
    }
    // Two header words, the special header, a process id and fifteen handles of each kind fit
    // well inside a message, so none of the offsets below can run past its end.
    let Some(special) = word(message, SPECIAL_HEADER).map(SpecialHeader::decode) else {
        return (translated, Ok(()));
    };
    let mut offset = SPECIAL_HEADER + WORD;
    if special.send_pid {
        let end = offset + PROCESS_ID_SIZE;
        if let Some(slot) = message.get_mut(offset..end) {
            slot.copy_from_slice(&process_id.to_le_bytes());
        }
        offset = end;
    }
    let mut result = Ok(());
    for _ in 0..special.num_copy_handles {
        let handle = Handle::from_raw(word(message, offset).unwrap_or_default());
        offset += WORD;
        if result.is_err() || handle == Handle::INVALID {
            continue;
        }
        result = hold(&mut translated, objects, handle, handles.get(handle).ok());
    }
    for _ in 0..special.num_move_handles {
        let handle = Handle::from_raw(word(message, offset).unwrap_or_default());
        offset += WORD;
        if handle == Handle::INVALID {
            continue;
        }
        if result.is_ok() {
            result = hold(&mut translated, objects, handle, handles.get(handle).ok());
        }
        // Removed whether or not the request failed; a handle that names nothing has nothing
        // to remove, which is already the request's error.
        let _ = handles.remove(objects, handle);
    }
    (translated, result)
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
    use starquake_ipc::{CommandType, RequestBuilder};

    use super::*;
    use crate::id::{SlabId, ThreadId};
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
        SPECIAL_HEADER + WORD + if pid { PROCESS_ID_SIZE } else { 0 }
    }

    #[test]
    fn a_message_without_a_special_header_is_left_alone() {
        let (mut handles, mut objects, _) = table(1);
        let mut message = build(&request());
        let before = message;
        let (translated, result) = translate(&mut message, 0x51, &mut handles, &mut objects);
        assert_eq!((translated, result), (Translated::default(), Ok(())));
        assert_eq!(message, before);
    }

    #[test]
    fn the_process_id_slot_gets_the_senders_process_id() {
        let (mut handles, mut objects, _) = table(0);
        let mut message = build(&request().send_pid(0x1111_2222_3333_4444));
        let mut expected = message;
        let slot = SPECIAL_HEADER + WORD;
        expected[slot..slot + PROCESS_ID_SIZE].copy_from_slice(&0x51_u64.to_le_bytes());
        let (translated, result) = translate(&mut message, 0x51, &mut handles, &mut objects);
        assert_eq!((translated, result), (Translated::default(), Ok(())));
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
        let (translated, result) = translate(&mut message, 0, &mut handles, &mut objects);
        assert_eq!(result, Ok(()));
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
        let (translated, result) = translate(&mut message, 0, &mut handles, &mut objects);
        assert_eq!(result, Ok(()));
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
        let (translated, result) = translate(&mut message, 0, &mut handles, &mut objects);
        assert_eq!((translated, result), (Translated::default(), Ok(())));
        assert!(handles.get(made[0]).is_ok());
    }

    #[test]
    fn a_bad_copied_handle_fails_the_request_and_later_moves_still_leave_the_table() {
        let (mut handles, mut objects, made) = table(3);
        let (a, b, c) = (made[0], made[1], made[2]);
        let object_a = handles.get(a).unwrap();
        let stale = Handle::from_parts(5, 9).unwrap();
        let mut message = build(
            &request()
                .copy_handle(a.raw())
                .copy_handle(stale.raw())
                .copy_handle(b.raw())
                .move_handle(c.raw()),
        );
        let (translated, result) = translate(&mut message, 0, &mut handles, &mut objects);
        assert_eq!(result, Err(KernelError::InvalidHandle));
        // The handle before the bad one is held (the caller releases it); none after.
        assert_eq!(translated.held, [object_a]);
        assert_eq!(objects.refs(handles.get(b).unwrap()), 1);
        // The moved handle left the table although the request failed.
        assert!(handles.get(c).is_err());
        assert_eq!(handles.len(), 2);
    }

    #[test]
    fn a_bad_moved_handle_fails_the_request_and_is_not_held() {
        let (mut handles, mut objects, made) = table(1);
        let stale = Handle::from_parts(6, 3).unwrap();
        let mut message = build(
            &request()
                .move_handle(stale.raw())
                .move_handle(made[0].raw()),
        );
        let (translated, result) = translate(&mut message, 0, &mut handles, &mut objects);
        assert_eq!(result, Err(KernelError::InvalidHandle));
        assert_eq!(translated.held, Vec::<ObjectId>::new());
        // The good handle after it still left the table, and its object with it.
        assert!(handles.is_empty());
        assert!(objects.is_empty());
    }

    #[test]
    fn a_copied_pseudo_handle_is_an_invalid_handle_until_processes_are_objects() {
        for pseudo in [Handle::CURRENT_THREAD, Handle::CURRENT_PROCESS] {
            let (mut handles, mut objects, _) = table(0);
            let mut message = build(&request().copy_handle(pseudo.raw()));
            let (translated, result) = translate(&mut message, 0, &mut handles, &mut objects);
            assert_eq!(
                (translated, result),
                (Translated::default(), Err(KernelError::InvalidHandle))
            );
        }
    }

    #[test]
    fn the_special_header_is_found_whatever_the_message_type() {
        // A message type nothing defines: the kernel's half does not look at it.
        let (mut handles, mut objects, made) = table(1);
        let mut message = build(&request().move_handle(made[0].raw()));
        message[0] = 0xEE;
        message[1] = 0xEE;
        let (translated, result) = translate(&mut message, 0, &mut handles, &mut objects);
        assert_eq!(result, Ok(()));
        assert_eq!(translated.held.len(), 1);
        assert!(handles.is_empty());
    }
}

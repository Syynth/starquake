//! [`Reply`]: what a service method sends back.

use starquake_kernel::id::ObjectId;

use super::commands::Session;
use super::payload::Writer;
use super::result::{Completion, Outcome};

/// Something a reply moves to the caller, in the order the method added it.
pub(super) enum Moved {
    /// A kernel object: the caller gets a handle.
    Object(ObjectId),
    /// A new interface: the caller gets a session handle, or in a domain an object id.
    Session(Box<dyn Session>),
}

/// The reply to a request: a raw payload, handles, and new interfaces.
///
/// A method fills this in and returns [`Reply::ok`]. If it returns a
/// [`Fault`](super::Fault) instead, the caller gets the fault's result code and nothing that was
/// put here: the payload is not sent, no handle is added to the caller's table, and pushed
/// interfaces are dropped.
#[derive(Default)]
pub struct Reply {
    payload: Writer,
    copies: Vec<ObjectId>,
    moves: Vec<Moved>,
}

impl Reply {
    /// The writer for the raw payload. See [`Writer`] for the layout rules.
    pub fn payload(&mut self) -> &mut Writer {
        &mut self.payload
    }

    /// Gives the caller a handle to `object` in the reply's copy list.
    pub fn copy_handle(&mut self, object: ObjectId) -> &mut Self {
        self.copies.push(object);
        self
    }

    /// Gives the caller a handle to `object` in the reply's move list.
    pub fn move_handle(&mut self, object: ObjectId) -> &mut Self {
        self.moves.push(Moved::Object(object));
        self
    }

    /// Returns a new interface to the caller. On an ordinary session the caller receives a moved
    /// session handle, placed among the moved handles in the order of the calls; in a domain it
    /// receives an object id. Which of the two is the framework's business, not the service's.
    pub fn push_session(&mut self, session: Box<dyn Session>) -> &mut Self {
        self.moves.push(Moved::Session(session));
        self
    }

    /// The value a method returns once the reply is complete: success.
    pub fn ok(&self) -> Outcome {
        Ok(Completion::Done)
    }

    pub(super) fn into_parts(self) -> (Writer, Vec<ObjectId>, Vec<Moved>) {
        (self.payload, self.copies, self.moves)
    }
}

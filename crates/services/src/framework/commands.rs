//! [`Session`], the trait a service interface implements, and [`commands!`](crate::commands),
//! the macro that implements it from a table of command ids.

use core::any::Any;

use super::context::Context;
use super::reply::Reply;
use super::request::Request;
use super::result::Outcome;
use super::snapshot::SnapshotError;

/// Which protocol a request came in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Protocol {
    /// CMIF: the command id is in the `SFCI` header.
    Cmif,
    /// TIPC: the command id is in the message's type field.
    Tipc,
}

/// One row of a command table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Command {
    /// The command id.
    pub id: u32,
    /// The method's name, for logs and traces.
    pub name: &'static str,
}

/// The ids and names of an interface's commands. CMIF and TIPC ids overlap, so each protocol has
/// its own list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandTable {
    /// The CMIF commands, in the order the table lists them.
    pub cmif: &'static [Command],
    /// The TIPC commands, in the order the table lists them.
    pub tipc: &'static [Command],
}

impl CommandTable {
    /// The name of a command, or `None` if the table has no such command.
    #[must_use]
    pub fn name(&self, protocol: Protocol, id: u32) -> Option<&'static str> {
        let list = match protocol {
            Protocol::Cmif => self.cmif,
            Protocol::Tipc => self.tipc,
        };
        list.iter().find(|c| c.id == id).map(|c| c.name)
    }

    /// Whether neither list names an id twice. [`commands!`](crate::commands) refuses to compile
    /// a table for which this is false.
    #[must_use]
    pub const fn is_distinct(&self) -> bool {
        Self::distinct(self.cmif) && Self::distinct(self.tipc)
    }

    const fn distinct(mut list: &[Command]) -> bool {
        while let [first, rest @ ..] = list {
            let mut others = rest;
            while let [other, tail @ ..] = others {
                if first.id == other.id {
                    return false;
                }
                others = tail;
            }
            list = rest;
        }
        true
    }
}

/// One service interface: the object behind a session handle or a domain object id.
///
/// A service is a struct per session, a command table and a method per command
/// (`docs/specs/services.md`, "Writing a service"). Write the struct and the methods, and let
/// [`commands!`](crate::commands) implement this trait.
///
/// `Send`, because a server's requests may be handled on any core thread (one at a time).
///
/// The struct is the session's whole state, and it is plain serialisable data: the macro
/// implements [`Session::save`] with `serde`, so a struct that holds a host handle, a closure or
/// a lock does not compile as a service (`docs/specs/observability.md`, "Snapshots": "each
/// session keeps its state in a plain serialisable struct"). State that several sessions share
/// belongs to the server ([`Server::with_family`](super::Server::with_family)).
pub trait Session: Any + Send {
    /// The interface's name: what [`Session::name`] returns for every value of the type. A
    /// snapshot stores it to find the type again
    /// ([`Interfaces::with`](super::Interfaces::with)).
    fn interface() -> &'static str
    where
        Self: Sized;

    /// The interface's name in logs and traces.
    fn name(&self) -> &'static str;

    /// The session's state as bytes, for a snapshot. [`Interfaces`](super::Interfaces) turns
    /// them back into the session.
    fn save(&self) -> Result<Vec<u8>, SnapshotError>;

    /// The ids and names of the commands [`Session::dispatch`] knows.
    fn commands(&self) -> &'static CommandTable;

    /// Calls the method for a command. `None` means the interface has no such command; what the
    /// caller then sees is the server's [`UnknownPolicy`](super::UnknownPolicy).
    fn dispatch(
        &mut self,
        protocol: Protocol,
        id: u32,
        cx: &mut Context<'_>,
        request: &Request<'_>,
        reply: &mut Reply,
    ) -> Option<Outcome>;
}

/// Implements [`Session`](crate::framework::Session) for a service from its command table.
///
/// Each row is `id => method`, where the method has the signature
/// `fn(&mut self, &mut Context<'_>, &Request<'_>, &mut Reply) -> Outcome`. TIPC commands go in
/// their own `tipc { .. }` block after the CMIF rows, since the two protocols number their
/// commands separately.
///
/// A method that does not touch the session's state is written with `_: &mut Self` in place of
/// `&mut self` (`ping` below). That says so to the reader, and keeps `clippy::unused_self` quiet
/// without an `#[expect]` on every such method; the table row is the same either way.
///
/// ```
/// use starquake_services::commands;
/// use starquake_services::framework::{Context, Outcome, Protocol, Reply, Request, Session};
///
/// #[derive(serde::Serialize, serde::Deserialize)]
/// struct Counter {
///     count: u32,
/// }
///
/// commands! {
///     impl Counter {
///         0 => add,
///         1 => get,
///         2 => ping,
///         tipc {
///             0 => get,
///         }
///     }
/// }
///
/// impl Counter {
///     fn add(&mut self, _cx: &mut Context<'_>, req: &Request<'_>, reply: &mut Reply) -> Outcome {
///         self.count = self.count.wrapping_add(req.payload().read_u32()?);
///         reply.ok()
///     }
///
///     fn get(&mut self, _cx: &mut Context<'_>, _req: &Request<'_>, reply: &mut Reply) -> Outcome {
///         reply.payload().write_u32(self.count);
///         reply.ok()
///     }
///
///     fn ping(_: &mut Self, _cx: &mut Context<'_>, _req: &Request<'_>, reply: &mut Reply) -> Outcome {
///         reply.ok()
///     }
/// }
///
/// let counter = Counter { count: 0 };
/// assert_eq!(counter.name(), "Counter");
/// assert_eq!(counter.commands().name(Protocol::Cmif, 0), Some("add"));
/// assert_eq!(counter.commands().name(Protocol::Tipc, 0), Some("get"));
/// assert_eq!(counter.commands().name(Protocol::Tipc, 1), None);
/// ```
///
/// The macro generates the dispatch `match`, and the table of ids and names that logs and the
/// auto-stub use. The interface's name is the type's name as written.
///
/// The type must be `serde::Serialize`: the macro saves a session's state with it
/// ([`Session::save`]). To restore a snapshot the type must be
/// `serde::Deserialize` as well, and registered in the server's
/// [`Interfaces`](crate::framework::Interfaces).
///
/// Listing an id twice in one block is a compile error (`E0080`: the table's check fails while
/// the constant is evaluated). This compiles:
///
/// ```
/// # use starquake_services::commands;
/// # use starquake_services::framework::{Context, Outcome, Reply, Request};
/// # #[derive(serde::Serialize)]
/// # struct Counter;
/// # impl Counter {
/// #     fn get(&mut self, _: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
/// #         reply.ok()
/// #     }
/// # }
/// commands! {
///     impl Counter {
///         1 => get,
///         2 => get,
///         tipc {
///             1 => get,
///         }
///     }
/// }
/// ```
///
/// and the same table with the second id changed to 1 does not:
///
/// ```compile_fail,E0080
/// # use starquake_services::commands;
/// # use starquake_services::framework::{Context, Outcome, Reply, Request};
/// # #[derive(serde::Serialize)]
/// # struct Counter;
/// # impl Counter {
/// #     fn get(&mut self, _: &mut Context<'_>, _: &Request<'_>, reply: &mut Reply) -> Outcome {
/// #         reply.ok()
/// #     }
/// # }
/// commands! {
///     impl Counter {
///         1 => get,
///         1 => get,
///         tipc {
///             1 => get,
///         }
///     }
/// }
/// ```
#[macro_export]
macro_rules! commands {
    (
        impl $service:ty {
            $( $id:literal => $method:ident ),* $(,)?
            $( tipc { $( $tipc_id:literal => $tipc_method:ident ),* $(,)? } )?
        }
    ) => {
        impl $crate::framework::Session for $service {
            fn interface() -> &'static str {
                ::core::stringify!($service)
            }

            fn name(&self) -> &'static str {
                ::core::stringify!($service)
            }

            fn save(
                &self,
            ) -> ::core::result::Result<
                ::std::vec::Vec<u8>,
                $crate::framework::SnapshotError,
            > {
                $crate::framework::snapshot::encode(self)
            }

            fn commands(&self) -> &'static $crate::framework::CommandTable {
                const TABLE: $crate::framework::CommandTable = $crate::framework::CommandTable {
                    cmif: &[$(
                        $crate::framework::Command { id: $id, name: ::core::stringify!($method) },
                    )*],
                    tipc: &[$($(
                        $crate::framework::Command {
                            id: $tipc_id,
                            name: ::core::stringify!($tipc_method),
                        },
                    )*)?],
                };
                const _: () = ::core::assert!(
                    TABLE.is_distinct(),
                    "commands!: a command id is listed twice in one block"
                );
                &TABLE
            }

            fn dispatch(
                &mut self,
                protocol: $crate::framework::Protocol,
                id: u32,
                cx: &mut $crate::framework::Context<'_>,
                request: &$crate::framework::Request<'_>,
                reply: &mut $crate::framework::Reply,
            ) -> ::core::option::Option<$crate::framework::Outcome> {
                match (protocol, id, cx, request, reply) {
                    $(
                        ($crate::framework::Protocol::Cmif, $id, cx, request, reply) => {
                            ::core::option::Option::Some(<$service>::$method(self, cx, request, reply))
                        }
                    )*
                    $($(
                        ($crate::framework::Protocol::Tipc, $tipc_id, cx, request, reply) => {
                            ::core::option::Option::Some(<$service>::$tipc_method(self, cx, request, reply))
                        }
                    )*)?
                    _ => ::core::option::Option::None,
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: CommandTable = CommandTable {
        cmif: &[Command { id: 3, name: "a" }, Command { id: 7, name: "b" }],
        tipc: &[Command { id: 3, name: "c" }],
    };

    #[test]
    fn names_are_looked_up_per_protocol() {
        assert_eq!(TABLE.name(Protocol::Cmif, 3), Some("a"));
        assert_eq!(TABLE.name(Protocol::Cmif, 7), Some("b"));
        assert_eq!(TABLE.name(Protocol::Tipc, 3), Some("c"));
        assert_eq!(TABLE.name(Protocol::Tipc, 7), None);
        assert_eq!(TABLE.name(Protocol::Cmif, 4), None);
    }

    #[test]
    fn a_repeated_id_in_either_list_is_not_distinct() {
        const A: Command = Command { id: 3, name: "a" };
        const B: Command = Command { id: 7, name: "b" };
        let table = |cmif, tipc| CommandTable { cmif, tipc };
        // The same id in both lists is fine: the protocols number their commands separately.
        assert!(TABLE.is_distinct());
        assert!(table(&[], &[]).is_distinct());
        assert!(table(&[A, B], &[A, B]).is_distinct());
        assert!(!table(&[A, A], &[]).is_distinct());
        assert!(!table(&[A, B, A], &[]).is_distinct());
        assert!(!table(&[B, A, A], &[]).is_distinct());
        assert!(!table(&[A, B], &[B, B]).is_distinct());
    }
}

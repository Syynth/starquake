//! Declared buffer kinds: which descriptor each of a command's buffers is, as the command's row
//! in [`commands!`](crate::commands) states it, and the check of a request against that
//! declaration.
//!
//! # Declaring
//!
//! A command lists its buffers after its method, in the order the method takes them, each with
//! its direction and its kind:
//!
//! ```text
//! command::READ => read [Out(MapAlias)],
//! command::OPEN_FILE => open_file [In(Pointer)],
//! ```
//!
//! The kinds are the transport flags of Eden's `BufferAttr`
//! (`eden/src/core/hle/service/cmif_types.h`: `BufferAttr_HipcMapAlias`, `BufferAttr_HipcPointer`,
//! `BufferAttr_HipcAutoSelect`), which are libnx's `SfBufferAttr_Hipc*`
//! (`libnx/nx/include/switch/sf/service.h`), so a row can be held against the reference's
//! signature at a glance. A row with no list declares no buffers. A method reaches its `n`th in
//! (or out) buffer with [`Request::in_buffer`](super::Request::in_buffer) (or `out_buffer`),
//! where `n` counts the declared buffers of that direction; one it did not declare is
//! [`Fault::UndeclaredBuffer`].
//!
//! # Which descriptor a buffer is
//!
//! A client numbers each descriptor list on its own: a map-alias buffer is the next A (in) or B
//! (out) descriptor, a pointer buffer the next X (in) or C (out) descriptor, and an auto-select
//! buffer takes one of each, the one it does not use empty (libnx `_serviceRequestProcessBuffer`
//! in `libnx/nx/include/switch/sf/service.h`; `cmifRequestInAutoBuffer` and
//! `cmifRequestOutAutoBuffer` in `libnx/nx/include/switch/sf/cmif.h`). The console's server
//! reads them the same way (Atmosphère `sf_impl_command_serialization.hpp`, `ProcessUpdate`:
//! `send_map_alias_index`, `send_pointer_index` and so on; fact only), and so does `locate`.
//!
//! An auto-select buffer is the A (or B) descriptor if its size is not zero, and otherwise the X
//! (or C) descriptor: Eden's `HLERequestContext::ReadBuffer` and `WriteBuffer`
//! (`eden/src/core/hle/service/hle_ipc.cpp`), which `cmif_serialization.h` calls for
//! `BufferAttr_HipcAutoSelect`.
//!
//! # The check
//!
//! Before the method runs, `check` holds the request's descriptor counts against the
//! declaration, the console's check (Atmosphère `sf_impl_command_serialization.hpp`,
//! `HipcCommandProcessor::PrepareForProcess`, fact only): the number of X descriptors is the
//! number of declared in buffers that are pointer or auto-select, of A descriptors the in
//! buffers that are map-alias or auto-select, of B descriptors the out buffers that are map-alias
//! or auto-select, and there are no W descriptors. The number of C descriptors is at least the
//! number of declared out buffers that are pointer or auto-select on a CMIF request, and exactly
//! that number on a TIPC request (see "C descriptors" and "TIPC" below). A request that fails it
//! is answered `INVALID_CMIF_REQUEST` ([`Fault::DescriptorCount`]), the console's
//! `sf::hipc::ResultInvalidCmifRequest`, and the method is not called. The counts are compared in
//! the order the message header lists them, X, A, B, W, and then C, and the first that differs is
//! the one reported.
//!
//! The check runs against the row of the request's own protocol: a TIPC request is held against
//! the `tipc { .. }` row with its id, never the CMIF row with the same id. A request for an id
//! its protocol's table lacks is not checked at all: it goes to the server's unknown-command
//! policy whatever descriptors it carries.
//!
//! # Where this differs from the references
//!
//! Each is pinned by the test named, in this module's tests where no other file is given:
//!
//! - **Eden does not hold descriptors against the signature.** Its typed commands read whichever descriptor the signature names and
//!   log a failed assertion when there is none (`cmif_serialization.h`, `ReadInArgument`:
//!   `ASSERT(ctx.CanReadBuffer(..))`, then `ReadBufferA` or `ReadBufferX`), and carry on with an
//!   empty buffer or write nothing; a descriptor of the other kind is ignored. Here the request is
//!   refused before the method runs (`each_kind_accepts_its_own_descriptors_and_refuses_the_others`).
//! - **Eden's indexes.** Eden reads the `n`th in buffer from the `n`th entry of whichever list
//!   (`cmif_serialization.h`: `ReadBufferA(InBufferIndex)`, `ReadBufferX(InBufferIndex)`). That is
//!   the client's numbering unless an earlier buffer of the same direction is of the other single
//!   kind (a pointer buffer before a map-alias one, or the other way round). Here the client's
//!   numbering holds throughout (`each_list_is_numbered_on_its_own`). None of `sm:`, `set:sys` and
//!   `fsp-srv` has such a command.
//! - **C descriptors.** The console does not count them (the comparison is commented out in
//!   `PrepareForProcess`): its out pointer buffers are carved from the server's own pointer
//!   buffer, sized by a table in the request's raw data. Here an out pointer (or auto-select)
//!   buffer is written straight to the caller's C descriptor, as in Eden, so there must be at
//!   least one per such buffer; on a CMIF request more are accepted, as on the console
//!   (`c_descriptors_must_cover_the_declared_buffers_and_may_be_more`). A map-alias out buffer is
//!   always its B descriptor, never a stray C descriptor beside it (`framework/tests/buffers.rs`:
//!   `a_map_alias_out_buffer_never_reaches_a_stray_c_descriptor`).
//! - **Choosing an auto-select buffer's descriptor.** The console takes the map-alias
//!   descriptor when its address is not zero (`ProcessBufferImpl`: `is_buffer_map_alias =
//!   hipcGetBufferAddress(map_desc) != 0`; fact only); here, as in Eden, when its size is not
//!   zero. A client's unused descriptor has both zero, so the two agree on every request libnx
//!   sends; an A descriptor with an address and no size is the X descriptor here and an empty
//!   buffer on the console (`framework/tests/buffers.rs`:
//!   `a_non_empty_a_or_b_wins_and_an_empty_one_gives_way`).
//! - **Transfer modes.** The console also refuses a map-alias descriptor whose mode is not the one
//!   the command allows (`IsMapTransferModeValid`); Eden does not look at it, and neither does
//!   this check (`the_transfer_mode_is_not_checked`).
//! - **TIPC.** The console checks a TIPC request's whole message header against the command and
//!   answers `tipc::ResultInvalidMessageFormat` (Atmosphère `tipc_impl_command_serialization.hpp`,
//!   `ValidateCommandFormat`, which compares the header's first 64 bits with the command's
//!   `InMessageHeader`; fact only). That header's pointer (X) count and receive-list (C) field are
//!   always zero there, so the console refuses any X or C descriptor on TIPC. Here a TIPC
//!   request's X, A, B and W counts are checked as a CMIF request's are, and its C count must
//!   equal the declared out buffers that are pointer or auto-select: a TIPC request to a command
//!   that declares none is refused for a single C descriptor, as on the console
//!   (`c_descriptors_on_tipc_must_match_exactly`; `framework/tests/buffers.rs`:
//!   `a_tipc_request_with_a_stray_c_descriptor_is_refused`). Two differences remain: a mismatch
//!   is `INVALID_CMIF_REQUEST`, as the framework also answers a TIPC request whose payload is cut
//!   short (`sm/tests.rs`: `a_tipc_request_with_a_buffer_is_refused`, and
//!   `a_name_is_its_printable_bytes_wherever_they_are` for the short payload); and a TIPC row may
//!   declare a pointer or auto-select buffer, which the console's TIPC cannot express. No
//!   service's TIPC row declares one.

use starquake_ipc::Message;

use super::commands::Protocol;
use super::result::{Direction, Fault};

/// How a buffer travels: the transport flag of Eden's `BufferAttr` and libnx's `SfBufferAttr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferKind {
    /// `BufferAttr_HipcMapAlias`: an A (in) or B (out) descriptor.
    MapAlias,
    /// `BufferAttr_HipcPointer`: an X (in) or C (out) descriptor.
    Pointer,
    /// `BufferAttr_HipcAutoSelect`: one descriptor of each, the buffer being whichever is not
    /// empty (see the [module docs](self)).
    AutoSelect,
}

impl BufferKind {
    /// Whether the buffer takes an A or B descriptor.
    const fn uses_map_alias(self) -> bool {
        matches!(self, Self::MapAlias | Self::AutoSelect)
    }

    /// Whether the buffer takes an X or C descriptor.
    const fn uses_pointer(self) -> bool {
        matches!(self, Self::Pointer | Self::AutoSelect)
    }
}

/// One buffer of a command, as its row in [`commands!`](crate::commands) declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeclaredBuffer {
    /// Which way it carries data.
    pub direction: Direction,
    /// How it travels.
    pub kind: BufferKind,
}

/// One of a request's descriptor lists, by its letter in the HIPC message layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorList {
    /// Static (pointer) descriptors the caller sends: pointer in buffers.
    X,
    /// Send descriptors: map-alias in buffers.
    A,
    /// Receive descriptors: map-alias out buffers.
    B,
    /// Exchange descriptors: buffers both ways, which no command here takes.
    W,
    /// The receive list: pointer out buffers.
    C,
}

impl DescriptorList {
    /// The list's letter.
    #[must_use]
    pub const fn letter(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::A => "A",
            Self::B => "B",
            Self::W => "W",
            Self::C => "C",
        }
    }
}

/// How many of a direction's declared buffers take a map-alias and a pointer descriptor.
fn needs(declared: &[DeclaredBuffer], direction: Direction) -> (usize, usize) {
    needs_of(declared.iter().filter(|b| b.direction == direction))
}

/// How many of `buffers` take a map-alias and a pointer descriptor.
fn needs_of<'a>(buffers: impl Iterator<Item = &'a DeclaredBuffer>) -> (usize, usize) {
    // Neither count can pass the length of a declaration, a `'static` slice.
    buffers.fold((0, 0), |(map_alias, pointer), b| {
        (
            map_alias + usize::from(b.kind.uses_map_alias()),
            pointer + usize::from(b.kind.uses_pointer()),
        )
    })
}

/// Holds a request's descriptors against a command's declared buffers (see the
/// [module docs](self), "The check").
pub(crate) fn check(
    protocol: Protocol,
    declared: &[DeclaredBuffer],
    message: &Message<'_>,
) -> Result<(), Fault> {
    let (in_map_alias, in_pointer) = needs(declared, Direction::In);
    let (out_map_alias, out_pointer) = needs(declared, Direction::Out);
    let exact = [
        (
            DescriptorList::X,
            message.static_descriptors.len(),
            in_pointer,
        ),
        (DescriptorList::A, message.send_buffers.len(), in_map_alias),
        (
            DescriptorList::B,
            message.receive_buffers.len(),
            out_map_alias,
        ),
        (DescriptorList::W, message.exchange_buffers.len(), 0),
    ];
    for (list, sent, declared) in exact {
        if sent != declared {
            return Err(Fault::DescriptorCount {
                list,
                sent,
                declared,
            });
        }
    }
    let sent = message.receive_descriptors.len();
    // CMIF accepts more C descriptors than declared; TIPC accepts exactly as many (module docs).
    let refused = match protocol {
        Protocol::Cmif => sent < out_pointer,
        Protocol::Tipc => sent != out_pointer,
    };
    if refused {
        return Err(Fault::DescriptorCount {
            list: DescriptorList::C,
            sent,
            declared: out_pointer,
        });
    }
    Ok(())
}

/// Where a declared buffer is among a request's descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Place {
    /// The buffer's kind.
    pub(crate) kind: BufferKind,
    /// Its index in the A (or B) list, which only a map-alias or auto-select buffer uses.
    pub(crate) map_alias: usize,
    /// Its index in the X (or C) list, which only a pointer or auto-select buffer uses.
    pub(crate) pointer: usize,
}

/// Where the `index`th declared buffer of `direction` is (see the [module docs](self), "Which
/// descriptor a buffer is"), or `None` if the command declares no such buffer.
pub(crate) fn locate(
    declared: &[DeclaredBuffer],
    direction: Direction,
    index: usize,
) -> Option<Place> {
    let mine = declared.iter().filter(|b| b.direction == direction);
    let buffer = mine.clone().nth(index)?;
    let (map_alias, pointer) = needs_of(mine.take(index));
    Some(Place {
        kind: buffer.kind,
        map_alias,
        pointer,
    })
}

#[cfg(test)]
mod tests;

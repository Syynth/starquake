//! A vi2 session's saved form, and what reading one back checks (the spec's "What is
//! snapshotted": "vi2 displays, clients, request groups ..., availability masks, notification
//! bitmaps").
//!
//! # Layout
//!
//! `postcard`, field for field in [`Vi2Session`]'s order: the notification event (an `Option`
//! of an object id: its index, then its generation), the [`NOTIFICATION_WORDS`] words of the
//! bitmap, the request groups (a length, then each group's event and its monitor: the display's
//! name and the kind), the group count 51 recorded, the display's references, the clients (a
//! length, then each one's token, retire bit, available bit, whether bound, layer, free slots and
//! levels as `f32` bits), the endpoint's client, and the next layer number. The server's
//! snapshot holds it within its own header; the golden-bytes test
//! `the_saved_session_layout_is_pinned` (`vi2/tests/saved.rs`) pins it.
//!
//! # What a restore checks
//!
//! [`Vi2Session`]'s deserialising goes through [`SavedVi2Session`], which checks, in this order,
//! and refuses the first failure with its [`Vi2SessionError`]: every relation between the fields
//! that the commands keep.
//!
//! 1. [`REQUEST_GROUPS`] groups ([`Vi2SessionError::Groups`]).
//! 2. Each group in order: a monitor only with an event ([`Vi2SessionError::MonitorWithoutEvent`]),
//!    a kind below [`MONITOR_KINDS`] ([`Vi2SessionError::MonitorKind`]), a display name 261
//!    takes, not zero and in [`super::KNOWN_DISPLAYS`] ([`Vi2SessionError::MonitorDisplay`]); its
//!    event not the notification event ([`Vi2SessionError::NotificationEventShared`]) and not an
//!    earlier group's ([`Vi2SessionError::GroupEventShared`]).
//! 3. A group count below [`REQUEST_GROUPS`] ([`Vi2SessionError::GroupCount`]).
//! 4. At most [`MAX_DISPLAY_REFERENCES`] references ([`Vi2SessionError::DisplayReferences`]).
//! 5. Every word of the bitmap but the first zero: a raised bit is a client's (below
//!    [`CLIENT_NOTIFICATIONS`]) or the display's ([`super::OPEN_DISPLAY_FLAGS`])
//!    ([`Vi2SessionError::Notifications`]).
//! 6. A next layer number of at least [`FIRST_LAYER`] ([`Vi2SessionError::NextLayer`]).
//! 7. At most [`MAX_CLIENTS`] clients ([`Vi2SessionError::TooManyClients`]).
//! 8. Each client in order: a token that is not zero ([`Vi2SessionError::ZeroToken`]) and no
//!    earlier client's ([`Vi2SessionError::DuplicateToken`]); two different bits below
//!    [`CLIENT_NOTIFICATIONS`] ([`Vi2SessionError::ClientBits`]) that no earlier client holds
//!    ([`Vi2SessionError::SharedBit`]); free slots within [`ALL_SLOTS`]
//!    ([`Vi2SessionError::Availability`]); a layer from [`FIRST_LAYER`] and below the next
//!    layer number ([`Vi2SessionError::LayerAhead`]) that no earlier client has
//!    ([`Vi2SessionError::SharedLayer`]); levels 352 accepts ([`Vi2SessionError::Levels`]).
//! 9. An endpoint only for a registered client ([`Vi2SessionError::EndpointWithoutClient`]).
//!
//! Not checked here, each with its consequence:
//!
//! - That each event is one the kernel holds, made by this server. One it does not hold is
//!   refused when a command signals or clears it, and that command answers the kernel's error;
//!   the vsync clock drops a listener the kernel refuses ([`crate::shared::vsync`], "A refused
//!   signal").
//! - That a group's event listens to the vsync clock exactly when [`super::RequestGroup::listens`]
//!   says so: the listeners are the clock's snapshot. A group that should listen and does not
//!   is not ticked until a 261 (kind 0) or an unregister makes it listen; a listener with no
//!   group is ticked until the kernel refuses it.
//! - That the display notification bit and a client's bits are raised only by what raises
//!   them: any bit of the first word is accepted, and 21 hands it out.
//! - That two sessions' layers have different numbers: layers are numbered per session (module
//!   docs of [`super`], "Layers").

use serde::Deserialize;
use starquake_kernel::id::ObjectId;
use thiserror::Error;

use super::{
    ALL_SLOTS, CLIENT_NOTIFICATIONS, Client, FIRST_LAYER, MAX_CLIENTS, MAX_DISPLAY_REFERENCES,
    MONITOR_KINDS, NOTIFICATION_WORDS, REQUEST_GROUPS, RequestGroup, Vi2Session, is_known_display,
    levels_are_valid,
};

/// The serialised form of [`Vi2Session`], field for field, and what deserialising validates
/// (module docs).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SavedVi2Session {
    notification_event: Option<ObjectId>,
    notifications: [u32; NOTIFICATION_WORDS],
    groups: Vec<RequestGroup>,
    group_count: Option<u32>,
    display_references: u32,
    clients: Vec<Client>,
    endpoint: Option<u64>,
    next_layer: u64,
}

/// Why a saved [`Vi2Session`] could not be read back: it holds a combination of fields that no
/// sequence of vi2 commands leads to (module docs, "What a restore checks").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum Vi2SessionError {
    /// Another number of request groups than [`REQUEST_GROUPS`].
    #[error("a vi2 session holds {found} request groups, not {REQUEST_GROUPS}")]
    Groups {
        /// How many it holds.
        found: usize,
    },
    /// A group registered as a monitor with no event: 261 needs the event 31 makes.
    #[error("vi2 request group {group} is registered as a monitor but has no event")]
    MonitorWithoutEvent {
        /// The group.
        group: usize,
    },
    /// A monitor of a kind past the last (261 to 272 name six).
    #[error("vi2 request group {group} monitors kind {kind}, past the last kind")]
    MonitorKind {
        /// The group.
        group: usize,
        /// The kind it holds.
        kind: u32,
    },
    /// A monitor of a display name 261 refuses: zero, or not a known display.
    #[error("vi2 request group {group} monitors a display name no monitor command accepts")]
    MonitorDisplay {
        /// The group.
        group: usize,
    },
    /// A group whose event is the notification event: 11 and 31 each make their own.
    #[error("vi2 request group {group}'s event is the notification event")]
    NotificationEventShared {
        /// The group.
        group: usize,
    },
    /// Two groups with one event: 31 makes one per group.
    #[error("vi2 request groups {first} and {second} hold the same event")]
    GroupEventShared {
        /// The lower group.
        first: usize,
        /// The higher group.
        second: usize,
    },
    /// A group count 51 refuses.
    #[error("a vi2 session's request group count is {found}, not below {REQUEST_GROUPS}")]
    GroupCount {
        /// The count it holds.
        found: u32,
    },
    /// More display references than 101 counts.
    #[error("a vi2 session's display is open {found} times, more than {MAX_DISPLAY_REFERENCES}")]
    DisplayReferences {
        /// The count it holds.
        found: u32,
    },
    /// A raised notification past the first word: no client's bit or the display's is there.
    #[error("a vi2 session's notification word {word} has a bit raised that nothing raises")]
    Notifications {
        /// The first such word.
        word: usize,
    },
    /// A next layer number below [`FIRST_LAYER`].
    #[error("a vi2 session's next layer number is {found}, below the first, {FIRST_LAYER}")]
    NextLayer {
        /// The number it holds.
        found: u64,
    },
    /// More clients than [`MAX_CLIENTS`].
    #[error("a vi2 session holds {count} clients, more than {MAX_CLIENTS}")]
    TooManyClients {
        /// How many it holds.
        count: usize,
    },
    /// A client whose token is zero: 201 refuses it.
    #[error("vi2 client {index} has the token zero")]
    ZeroToken {
        /// The client's index.
        index: usize,
    },
    /// Two clients with one token: 201 refuses the second.
    #[error("vi2 clients {first} and {second} have the same token")]
    DuplicateToken {
        /// The earlier client.
        first: usize,
        /// The later client.
        second: usize,
    },
    /// A client whose notification bits 201 refuses: one at or past [`CLIENT_NOTIFICATIONS`],
    /// or the same bit twice.
    #[error(
        "vi2 client {index}'s notification bits are not two different bits below {CLIENT_NOTIFICATIONS}"
    )]
    ClientBits {
        /// The client's index.
        index: usize,
    },
    /// Two clients holding one notification bit: 201 refuses the second.
    #[error("vi2 clients {first} and {second} hold the same notification bit")]
    SharedBit {
        /// The earlier client.
        first: usize,
        /// The later client.
        second: usize,
    },
    /// A client with a free slot past the fifteen ([`ALL_SLOTS`]).
    #[error("vi2 client {index} has a free slot past the last")]
    Availability {
        /// The client's index.
        index: usize,
    },
    /// A client's layer that 212 has not numbered yet: below [`FIRST_LAYER`], or at or past
    /// the next layer number.
    #[error("vi2 client {index}'s layer is not one the session has numbered")]
    LayerAhead {
        /// The client's index.
        index: usize,
    },
    /// Two clients with one layer: 212 numbers each one anew.
    #[error("vi2 clients {first} and {second} have the same layer")]
    SharedLayer {
        /// The earlier client.
        first: usize,
        /// The later client.
        second: usize,
    },
    /// A client's levels 352 refuses.
    #[error("vi2 client {index}'s output levels are not ones the session accepts")]
    Levels {
        /// The client's index.
        index: usize,
    },
    /// An endpoint whose client is not registered: 111 needs it, and 202 closes it.
    #[error("a vi2 session's endpoint names no registered client")]
    EndpointWithoutClient,
}

impl SavedVi2Session {
    /// The group checks (module docs, steps 1 and 2).
    fn check_groups(&self) -> Result<(), Vi2SessionError> {
        let expected = usize::try_from(REQUEST_GROUPS).unwrap_or(usize::MAX);
        if self.groups.len() != expected {
            return Err(Vi2SessionError::Groups {
                found: self.groups.len(),
            });
        }
        for (group, slot) in self.groups.iter().enumerate() {
            if let Some(monitor) = slot.monitor {
                if slot.event.is_none() {
                    return Err(Vi2SessionError::MonitorWithoutEvent { group });
                }
                if monitor.kind >= MONITOR_KINDS {
                    return Err(Vi2SessionError::MonitorKind {
                        group,
                        kind: monitor.kind,
                    });
                }
                if monitor.display == 0 || !is_known_display(monitor.display) {
                    return Err(Vi2SessionError::MonitorDisplay { group });
                }
            }
            let Some(event) = slot.event else { continue };
            if self.notification_event == Some(event) {
                return Err(Vi2SessionError::NotificationEventShared { group });
            }
            let earlier = self.groups.get(..group).unwrap_or_default();
            if let Some(first) = earlier.iter().position(|g| g.event == Some(event)) {
                return Err(Vi2SessionError::GroupEventShared {
                    first,
                    second: group,
                });
            }
        }
        Ok(())
    }

    /// The client checks (module docs, step 8).
    fn check_clients(&self) -> Result<(), Vi2SessionError> {
        if self.clients.len() > MAX_CLIENTS {
            return Err(Vi2SessionError::TooManyClients {
                count: self.clients.len(),
            });
        }
        for (index, client) in self.clients.iter().enumerate() {
            let earlier = self.clients.get(..index).unwrap_or_default();
            if client.token == 0 {
                return Err(Vi2SessionError::ZeroToken { index });
            }
            if let Some(first) = earlier.iter().position(|c| c.token == client.token) {
                return Err(Vi2SessionError::DuplicateToken {
                    first,
                    second: index,
                });
            }
            if client.retire >= CLIENT_NOTIFICATIONS
                || client.available_bit >= CLIENT_NOTIFICATIONS
                || client.retire == client.available_bit
            {
                return Err(Vi2SessionError::ClientBits { index });
            }
            if let Some(first) = earlier.iter().position(|c| c.bits() & client.bits() != 0) {
                return Err(Vi2SessionError::SharedBit {
                    first,
                    second: index,
                });
            }
            if client.available & !ALL_SLOTS != 0 {
                return Err(Vi2SessionError::Availability { index });
            }
            if let Some(layer) = client.layer {
                if layer < FIRST_LAYER || layer >= self.next_layer {
                    return Err(Vi2SessionError::LayerAhead { index });
                }
                if let Some(first) = earlier.iter().position(|c| c.layer == Some(layer)) {
                    return Err(Vi2SessionError::SharedLayer {
                        first,
                        second: index,
                    });
                }
            }
            if client
                .levels
                .is_some_and(|bits| !levels_are_valid(bits.map(f32::from_bits)))
            {
                return Err(Vi2SessionError::Levels { index });
            }
        }
        Ok(())
    }
}

impl TryFrom<SavedVi2Session> for Vi2Session {
    type Error = Vi2SessionError;

    fn try_from(saved: SavedVi2Session) -> Result<Self, Self::Error> {
        saved.check_groups()?;
        if let Some(found) = saved.group_count.filter(|&count| count >= REQUEST_GROUPS) {
            return Err(Vi2SessionError::GroupCount { found });
        }
        if saved.display_references > MAX_DISPLAY_REFERENCES {
            return Err(Vi2SessionError::DisplayReferences {
                found: saved.display_references,
            });
        }
        if let Some(word) = saved
            .notifications
            .iter()
            .skip(1)
            .position(|&word| word != 0)
        {
            return Err(Vi2SessionError::Notifications { word: word + 1 });
        }
        if saved.next_layer < FIRST_LAYER {
            return Err(Vi2SessionError::NextLayer {
                found: saved.next_layer,
            });
        }
        saved.check_clients()?;
        if let Some(token) = saved.endpoint
            && !saved.clients.iter().any(|c| c.token == token)
        {
            return Err(Vi2SessionError::EndpointWithoutClient);
        }
        let SavedVi2Session {
            notification_event,
            notifications,
            groups,
            group_count,
            display_references,
            clients,
            endpoint,
            next_layer,
        } = saved;
        Ok(Self {
            notification_event,
            notifications,
            groups,
            group_count,
            display_references,
            clients,
            endpoint,
            next_layer,
        })
    }
}

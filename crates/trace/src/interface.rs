//! What an [`EventKind::Ipc`](crate::EventKind::Ipc) event says about where a request went
//! beyond its service's number: the [`InterfaceName`] it was dispatched to and the
//! [`IpcProtocol`] it came in.
//!
//! An event is `Copy` and fixed-size (the flight recorder must not allocate while recording,
//! `event.rs`), so the name is held inline, in at most [`INTERFACE_NAME_CAPACITY`] bytes, and not
//! as a string. The service framework (`starquake-services`) fills both in at its one recording
//! point; this crate only carries them.
//!
//! No reference source is ported here: the layout is this project's own design.

use core::fmt;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The most bytes an [`InterfaceName`] holds.
///
/// starquake's own bound, a guess sized to fit: the longest name of a service interface class
/// in Eden is 38 bytes (`IFinalOutputRecorderManagerForDebugger`,
/// `eden/src/core/hle/service/audio/audio.cpp` @ cf654052), and a service name is at most 8
/// (`starquake_services::sm::MAX_NAME_LEN`). An event with this many name bytes is no larger than
/// one holding a system call's eight arguments, so the name does not grow [`Event`](crate::Event).
pub const INTERFACE_NAME_CAPACITY: usize = 40;

/// What [`InterfaceName::lossy`] writes in place of a byte that is not printable ASCII.
pub const REPLACEMENT: u8 = b'?';

/// Why text is not an [`InterfaceName`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InterfaceNameError {
    /// The name is empty.
    #[error("an interface name is empty")]
    Empty,
    /// The name is longer than [`INTERFACE_NAME_CAPACITY`] bytes.
    #[error("an interface name of {len} bytes is longer than {INTERFACE_NAME_CAPACITY} bytes")]
    TooLong {
        /// The name's length in bytes.
        len: usize,
    },
    /// The name has a byte outside `' '` to `'~'`.
    #[error("an interface name has a byte that is not printable ASCII: {byte:#04x}")]
    NotPrintable {
        /// The first such byte.
        byte: u8,
    },
}

/// The name of the interface a request was dispatched to: 1 to [`INTERFACE_NAME_CAPACITY`]
/// bytes of printable ASCII (`' '` to `'~'`), held inline.
///
/// It is built with [`InterfaceName::new`], which refuses anything else, or with
/// [`InterfaceName::lossy`], which never fails. Deserialising goes through
/// [`InterfaceName::new`], so a dump cannot carry a name that could not have been built.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InterfaceName {
    /// The name's bytes, then zeros.
    bytes: [u8; INTERFACE_NAME_CAPACITY],
    /// How many of [`InterfaceName::bytes`] are the name: 1 to [`INTERFACE_NAME_CAPACITY`].
    len: u8,
}

/// Whether a byte is one a name may hold.
const fn is_printable(byte: u8) -> bool {
    matches!(byte, b' '..=b'~')
}

impl InterfaceName {
    /// The name `name`, if it is one.
    pub fn new(name: &str) -> Result<Self, InterfaceNameError> {
        if name.is_empty() {
            return Err(InterfaceNameError::Empty);
        }
        if let Some(byte) = name.bytes().find(|byte| !is_printable(*byte)) {
            return Err(InterfaceNameError::NotPrintable { byte });
        }
        if name.len() > INTERFACE_NAME_CAPACITY {
            return Err(InterfaceNameError::TooLong { len: name.len() });
        }
        Ok(Self::filled(name.as_bytes()))
    }

    /// `name` made into a name whatever it holds: each byte that is not printable ASCII becomes
    /// [`REPLACEMENT`], the name is cut after its first [`INTERFACE_NAME_CAPACITY`] bytes, and
    /// an empty name is one [`REPLACEMENT`]. A recording point uses this, so that recording
    /// never fails.
    #[must_use]
    pub fn lossy(name: &str) -> Self {
        let mut kept = [0; INTERFACE_NAME_CAPACITY];
        let mut len = 0;
        for (slot, byte) in kept.iter_mut().zip(name.bytes()) {
            *slot = if is_printable(byte) {
                byte
            } else {
                REPLACEMENT
            };
            len += 1;
        }
        if len == 0 {
            return Self::filled(&[REPLACEMENT]);
        }
        Self::filled(kept.get(..len).unwrap_or(&[REPLACEMENT]))
    }

    /// A name of `bytes`, which are 1 to [`INTERFACE_NAME_CAPACITY`] printable bytes.
    fn filled(bytes: &[u8]) -> Self {
        let mut name = Self {
            bytes: [0; INTERFACE_NAME_CAPACITY],
            len: 0,
        };
        for (slot, byte) in name.bytes.iter_mut().zip(bytes) {
            *slot = *byte;
            name.len += 1;
        }
        name
    }

    /// The name as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        let bytes = self.bytes.get(..usize::from(self.len)).unwrap_or_default();
        // Every byte is printable ASCII (the constructors keep no other), so this never fails.
        core::str::from_utf8(bytes).unwrap_or_default()
    }
}

impl fmt::Debug for InterfaceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InterfaceName({:?})", self.as_str())
    }
}

impl fmt::Display for InterfaceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for InterfaceName {
    /// As a string: in `postcard`, its length and then its bytes.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for InterfaceName {
    /// From a string, through [`InterfaceName::new`].
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(NameVisitor)
    }
}

/// Reads an [`InterfaceName`] from a string.
struct NameVisitor;

impl Visitor<'_> for NameVisitor {
    type Value = InterfaceName;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an interface name")
    }

    fn visit_str<E: de::Error>(self, name: &str) -> Result<Self::Value, E> {
        InterfaceName::new(name).map_err(E::custom)
    }
}

/// The protocol a request came in, as its event records it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum IpcProtocol {
    /// A CMIF request: its command id is in the `SFCI` header.
    Cmif,
    /// A TIPC request: its command id is in the message's type field.
    Tipc,
    /// A CMIF control request (convert to domain, copy from domain, clone, query pointer
    /// buffer size), which the service framework answers itself; its ids are the control
    /// table's, not the interface's.
    Control,
}

impl IpcProtocol {
    /// The protocol in one word, as the golden trace writes it: `cmif`, `tipc` or `control`.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            Self::Cmif => "cmif",
            Self::Tipc => "tipc",
            Self::Control => "control",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A name of `len` bytes, all `'a'`.
    fn of_len(len: usize) -> String {
        "a".repeat(len)
    }

    #[test]
    fn a_name_is_one_to_capacity_printable_bytes() {
        assert_eq!(InterfaceName::new(""), Err(InterfaceNameError::Empty));
        assert_eq!(InterfaceName::new("a").unwrap().as_str(), "a");
        let full = of_len(INTERFACE_NAME_CAPACITY);
        assert_eq!(InterfaceName::new(&full).unwrap().as_str(), full);
        assert_eq!(
            InterfaceName::new(&of_len(INTERFACE_NAME_CAPACITY + 1)),
            Err(InterfaceNameError::TooLong {
                len: INTERFACE_NAME_CAPACITY + 1
            })
        );
        // The two ends of the printable range, and one past each.
        assert_eq!(InterfaceName::new(" ~").unwrap().as_str(), " ~");
        assert_eq!(
            InterfaceName::new("a\u{1f}"),
            Err(InterfaceNameError::NotPrintable { byte: 0x1f })
        );
        assert_eq!(
            InterfaceName::new("a\u{7f}"),
            Err(InterfaceNameError::NotPrintable { byte: 0x7f })
        );
        // A byte that is not printable is reported before the length.
        let both = format!("{}\u{7f}", of_len(INTERFACE_NAME_CAPACITY));
        assert_eq!(
            InterfaceName::new(&both),
            Err(InterfaceNameError::NotPrintable { byte: 0x7f })
        );
    }

    #[test]
    fn the_errors_are_plain_words() {
        assert_eq!(
            InterfaceNameError::Empty.to_string(),
            "an interface name is empty"
        );
        assert_eq!(
            InterfaceNameError::TooLong { len: 41 }.to_string(),
            "an interface name of 41 bytes is longer than 40 bytes"
        );
        assert_eq!(
            InterfaceNameError::NotPrintable { byte: 0x7f }.to_string(),
            "an interface name has a byte that is not printable ASCII: 0x7f"
        );
    }

    #[test]
    fn a_lossy_name_replaces_cuts_and_fills() {
        assert_eq!(InterfaceName::lossy("ISession").as_str(), "ISession");
        assert_eq!(InterfaceName::lossy("a\u{1f}b\u{7f}").as_str(), "a?b?");
        // A character of more than one byte is one replacement per byte.
        assert_eq!(InterfaceName::lossy("é").as_str(), "??");
        let full = of_len(INTERFACE_NAME_CAPACITY);
        assert_eq!(InterfaceName::lossy(&full).as_str(), full);
        assert_eq!(
            InterfaceName::lossy(&of_len(INTERFACE_NAME_CAPACITY + 1)).as_str(),
            full
        );
        assert_eq!(InterfaceName::lossy("").as_str(), "?");
        // What the strict constructor accepts, the lossy one keeps as it is.
        assert_eq!(
            InterfaceName::lossy(" ~"),
            InterfaceName::new(" ~").unwrap()
        );
    }

    #[test]
    fn a_name_is_shown_as_its_text() {
        let name = InterfaceName::new("IFile").unwrap();
        assert_eq!(name.to_string(), "IFile");
        assert_eq!(format!("{name:?}"), "InterfaceName(\"IFile\")");
    }

    #[test]
    fn a_name_is_serialised_as_a_string_and_read_back_through_the_check() {
        let name = InterfaceName::new("IFile").unwrap();
        let bytes = postcard::to_allocvec(&name).unwrap();
        assert_eq!(bytes, b"\x05IFile");
        assert_eq!(postcard::from_bytes::<InterfaceName>(&bytes).unwrap(), name);
        let full = InterfaceName::new(&of_len(INTERFACE_NAME_CAPACITY)).unwrap();
        let bytes = postcard::to_allocvec(&full).unwrap();
        assert_eq!(postcard::from_bytes::<InterfaceName>(&bytes).unwrap(), full);
        // What `new` refuses, reading refuses: empty, one byte too long, not printable.
        for refused in [
            String::new(),
            of_len(INTERFACE_NAME_CAPACITY + 1),
            "a\u{7f}".to_owned(),
        ] {
            let bytes = postcard::to_allocvec(&refused).unwrap();
            assert!(
                postcard::from_bytes::<InterfaceName>(&bytes).is_err(),
                "{refused:?}"
            );
        }
    }

    #[test]
    fn each_protocol_has_its_word() {
        assert_eq!(IpcProtocol::Cmif.word(), "cmif");
        assert_eq!(IpcProtocol::Tipc.word(), "tipc");
        assert_eq!(IpcProtocol::Control.word(), "control");
    }
}

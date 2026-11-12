//! The ring's text form, for tests and captures: one entry per line, its kind's name, then its
//! fields as `key=value` in a fixed order.
//!
//! ```text
//! bind channel=1 subchannel=2 class=0xc797
//! write channel=1 class=0xc797 method=0x45 value=0x1
//! run channel=1 class=0xc797 method=0x8e3 words=0x1,0x2
//! release address=0x1000 payload=0x5 size=4 timestamp=no op=write
//! release address=0x1000 payload=0x1 size=8 timestamp=yes op=add.unsigned
//! syncpoint id=5
//! wait-for-idle channel=1
//! invalidate channel=1 what=l2-flush-dirty
//! marker token=0x7
//! present address=0x40000 width=1280 height=720 pitch=20 block-height=4 format=1 shown=0,0,1280,720 layer=1 display=0
//! capture command=start
//! ```
//!
//! [`RingEntry`]'s `Display` writes this form, with addresses, classes, methods, values, payloads,
//! tokens and run words in hexadecimal and the rest in decimal; [`parse_entry`] reads it back and
//! accepts every number in either base (`0x` marks hexadecimal). A run with no words is
//! `words=`. Blank lines are skipped by [`parse_text`].

use std::borrow::Cow;
use std::fmt::{self, Write as _};
use std::str::SplitWhitespace;

use starquake_render::command::Rect;
use thiserror::Error;

use super::encode::Kind;
use super::entry::{
    CaptureCommand, ChannelId, ClassId, DisplayId, ImageDesc, Invalidation, LayerId, MarkerToken,
    PresentRequest, Release, ReleaseOp, RingEntry, Subchannel, SyncpointId,
};
use crate::sema::{PayloadSize, Reduction, ReductionFormat, ReductionOp, ReleaseForm};

/// Every invalidation with its name in the text form.
const INVALIDATION_NAMES: [(Invalidation, &str); 10] = [
    (Invalidation::Membar, "membar"),
    (Invalidation::MmuTlbInvalidate, "mmu-tlb-invalidate"),
    (
        Invalidation::MmuTlbInvalidateTargeted,
        "mmu-tlb-invalidate-targeted",
    ),
    (Invalidation::MmuOperation, "mmu-operation"),
    (Invalidation::L2PeermemInvalidate, "l2-peermem-invalidate"),
    (Invalidation::L2SysmemInvalidate, "l2-sysmem-invalidate"),
    (Invalidation::L2CleanComptags, "l2-clean-comptags"),
    (Invalidation::L2FlushDirty, "l2-flush-dirty"),
    (
        Invalidation::L2WaitForSysPendingReads,
        "l2-wait-for-sys-pending-reads",
    ),
    (Invalidation::AccessCounterClear, "access-counter-clear"),
];

/// Every reduction operation with its name.
const REDUCTION_NAMES: [(ReductionOp, &str); 8] = [
    (ReductionOp::Add, "add"),
    (ReductionOp::Min, "min"),
    (ReductionOp::Max, "max"),
    (ReductionOp::Inc, "inc"),
    (ReductionOp::Dec, "dec"),
    (ReductionOp::And, "and"),
    (ReductionOp::Or, "or"),
    (ReductionOp::Xor, "xor"),
];

/// Every reduction format with its name.
const FORMAT_NAMES: [(ReductionFormat, &str); 2] = [
    (ReductionFormat::Unsigned, "unsigned"),
    (ReductionFormat::Signed, "signed"),
];

/// Every capture command with its name.
const CAPTURE_NAMES: [(CaptureCommand, &str); 2] = [
    (CaptureCommand::Start, "start"),
    (CaptureCommand::Stop, "stop"),
];

/// The name of `value` in `table`.
fn name_of<T: PartialEq + Copy>(table: &[(T, &'static str)], value: T) -> &'static str {
    table
        .iter()
        .find(|(item, _)| *item == value)
        .map_or("", |&(_, name)| name)
}

/// The value named `name` in `table`.
fn named<T: Copy>(table: &[(T, &'static str)], name: &str) -> Option<T> {
    table
        .iter()
        .find(|&&(_, item)| item == name)
        .map(|&(value, _)| value)
}

/// A line of the text form that is not an entry.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    /// The line has no words.
    #[error("the line is empty")]
    Empty,
    /// The first word names no entry kind.
    #[error("\"{0}\" is not an entry kind")]
    UnknownKind(String),
    /// A field is missing at the end.
    #[error("a {kind} entry needs {key} next, but the line ends")]
    MissingField {
        /// The entry's kind.
        kind: Kind,
        /// The field expected.
        key: &'static str,
    },
    /// A word is not the field expected there.
    #[error("a {kind} entry needs {key} next, not \"{found}\"")]
    UnexpectedField {
        /// The entry's kind.
        kind: Kind,
        /// The field expected.
        key: &'static str,
        /// The word found.
        found: String,
    },
    /// A word follows the last field.
    #[error("a {kind} entry has nothing after its last field, not \"{found}\"")]
    ExtraField {
        /// The entry's kind.
        kind: Kind,
        /// The word found.
        found: String,
    },
    /// A field's value cannot be read.
    #[error("\"{value}\" is not a valid {key}")]
    BadValue {
        /// The field.
        key: &'static str,
        /// The text.
        value: String,
    },
    /// A number is too large for its field.
    #[error("{key} {value} is out of range")]
    OutOfRange {
        /// The field.
        key: &'static str,
        /// The number.
        value: u64,
    },
}

/// A text that is not a list of entries: the first bad line, counted from 1.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("line {line}: {error}")]
pub struct TextError {
    /// The line.
    pub line: usize,
    /// What is wrong with it.
    pub error: ParseError,
}

impl fmt::Display for RingEntry<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Kind::of(self))?;
        match self {
            Self::Bind {
                channel,
                subchannel,
                class,
            } => write!(
                f,
                " channel={} subchannel={} class={:#x}",
                channel.0,
                subchannel.get(),
                class.0
            ),
            Self::Write {
                channel,
                class,
                method,
                value,
            } => write!(
                f,
                " channel={} class={:#x} method={method:#x} value={value:#x}",
                channel.0, class.0
            ),
            Self::Run {
                channel,
                class,
                method,
                words,
            } => {
                write!(
                    f,
                    " channel={} class={:#x} method={method:#x} words=",
                    channel.0, class.0
                )?;
                for (index, word) in words.iter().enumerate() {
                    let comma = if index == 0 { "" } else { "," };
                    write!(f, "{comma}{word:#x}")?;
                }
                Ok(())
            }
            Self::Release(release) => {
                let size = release.form.size.bytes();
                let timestamp = if release.form.timestamp { "yes" } else { "no" };
                write!(
                    f,
                    " address={:#x} payload={:#x} size={size} timestamp={timestamp} op=",
                    release.address, release.payload
                )?;
                match release.op {
                    ReleaseOp::Write => f.write_str("write"),
                    ReleaseOp::Reduce(reduction) => write!(
                        f,
                        "{}.{}",
                        name_of(&REDUCTION_NAMES, reduction.op),
                        name_of(&FORMAT_NAMES, reduction.format)
                    ),
                }
            }
            Self::Syncpoint { id } => write!(f, " id={}", id.get()),
            Self::WaitForIdle { channel } => write!(f, " channel={}", channel.0),
            Self::Invalidate { channel, what } => write!(
                f,
                " channel={} what={}",
                channel.0,
                name_of(&INVALIDATION_NAMES, *what)
            ),
            Self::Marker { token } => write!(f, " token={:#x}", token.0),
            Self::Present(present) => {
                let image = &present.image;
                let shown = &present.shown;
                write!(
                    f,
                    " address={:#x} width={} height={} pitch={} block-height={} format={} \
                     shown={},{},{},{} layer={} display={}",
                    image.address,
                    image.width,
                    image.height,
                    image.pitch,
                    image.block_height_log2,
                    image.format,
                    shown.x,
                    shown.y,
                    shown.width,
                    shown.height,
                    present.layer.0,
                    present.display.0
                )
            }
            Self::CaptureControl(command) => {
                write!(f, " command={}", name_of(&CAPTURE_NAMES, *command))
            }
        }
    }
}

/// Reads a number: `0x` and hexadecimal digits, or decimal digits.
fn number(key: &'static str, text: &str) -> Result<u64, ParseError> {
    let bad = || ParseError::BadValue {
        key,
        value: text.to_owned(),
    };
    let (digits, radix) = match text.strip_prefix("0x") {
        Some(hex) => (hex, 16),
        None => (text, 10),
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
        return Err(bad());
    }
    u64::from_str_radix(digits, radix).map_err(|_| bad())
}

/// Narrows a number to its field's type.
fn narrow<T: TryFrom<u64>>(key: &'static str, value: u64) -> Result<T, ParseError> {
    T::try_from(value).map_err(|_| ParseError::OutOfRange { key, value })
}

/// The fields of one line, read in order.
struct Fields<'a> {
    kind: Kind,
    words: SplitWhitespace<'a>,
}

impl<'a> Fields<'a> {
    /// The value of the next field, which must be `key`.
    fn text(&mut self, key: &'static str) -> Result<&'a str, ParseError> {
        let kind = self.kind;
        let word = self
            .words
            .next()
            .ok_or(ParseError::MissingField { kind, key })?;
        word.split_once('=')
            .filter(|(name, _)| *name == key)
            .map(|(_, value)| value)
            .ok_or_else(|| ParseError::UnexpectedField {
                kind,
                key,
                found: word.to_owned(),
            })
    }

    /// The next field as a number of type `T`.
    fn number<T: TryFrom<u64>>(&mut self, key: &'static str) -> Result<T, ParseError> {
        let text = self.text(key)?;
        narrow(key, number(key, text)?)
    }

    /// The next field as a name from `table`.
    fn named<T: Copy>(
        &mut self,
        key: &'static str,
        table: &[(T, &'static str)],
    ) -> Result<T, ParseError> {
        let text = self.text(key)?;
        named(table, text).ok_or_else(|| ParseError::BadValue {
            key,
            value: text.to_owned(),
        })
    }

    /// Checks that no field is left.
    fn finish(mut self) -> Result<(), ParseError> {
        match self.words.next() {
            None => Ok(()),
            Some(word) => Err(ParseError::ExtraField {
                kind: self.kind,
                found: word.to_owned(),
            }),
        }
    }
}

/// Reads comma-separated numbers of type `T` (none from an empty text).
fn list<T: TryFrom<u64>>(key: &'static str, text: &str) -> Result<Vec<T>, ParseError> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    text.split(',')
        .map(|item| narrow(key, number(key, item)?))
        .collect()
}

/// A release's `op` field.
fn release_op(text: &str) -> Result<ReleaseOp, ParseError> {
    if text == "write" {
        return Ok(ReleaseOp::Write);
    }
    text.split_once('.')
        .and_then(|(op, format)| {
            Some(ReleaseOp::Reduce(Reduction {
                op: named(&REDUCTION_NAMES, op)?,
                format: named(&FORMAT_NAMES, format)?,
            }))
        })
        .ok_or_else(|| ParseError::BadValue {
            key: "op",
            value: text.to_owned(),
        })
}

/// The fields of a release.
fn release(fields: &mut Fields<'_>) -> Result<RingEntry<'static>, ParseError> {
    let address = fields.number("address")?;
    let payload = fields.number("payload")?;
    let size = match fields.number::<u64>("size")? {
        4 => PayloadSize::Four,
        8 => PayloadSize::Eight,
        value => return Err(ParseError::OutOfRange { key: "size", value }),
    };
    let timestamp = fields.named("timestamp", &[(true, "yes"), (false, "no")])?;
    let op = release_op(fields.text("op")?)?;
    Ok(RingEntry::Release(Release {
        address,
        payload,
        form: ReleaseForm { size, timestamp },
        op,
    }))
}

/// The fields of a present.
fn present(fields: &mut Fields<'_>) -> Result<RingEntry<'static>, ParseError> {
    let image = ImageDesc {
        address: fields.number("address")?,
        width: fields.number("width")?,
        height: fields.number("height")?,
        pitch: fields.number("pitch")?,
        block_height_log2: fields.number("block-height")?,
        format: fields.number("format")?,
    };
    let shown_text = fields.text("shown")?;
    let shown = match list::<u32>("shown", shown_text)?.as_slice() {
        &[x, y, width, height] => Rect {
            x,
            y,
            width,
            height,
        },
        _ => {
            return Err(ParseError::BadValue {
                key: "shown",
                value: shown_text.to_owned(),
            });
        }
    };
    Ok(RingEntry::Present(PresentRequest {
        image,
        shown,
        layer: LayerId(fields.number("layer")?),
        display: DisplayId(fields.number("display")?),
    }))
}

/// Reads one entry from one line of the text form.
pub fn parse_entry(line: &str) -> Result<RingEntry<'static>, ParseError> {
    let mut words = line.split_whitespace();
    let name = words.next().ok_or(ParseError::Empty)?;
    let kind = super::encode::KINDS
        .iter()
        .find(|&&(.., kind_name)| kind_name == name)
        .map(|&(kind, ..)| kind)
        .ok_or_else(|| ParseError::UnknownKind(name.to_owned()))?;
    let mut fields = Fields { kind, words };
    let entry = match kind {
        Kind::Bind => {
            let channel = ChannelId(fields.number("channel")?);
            let value: u8 = fields.number("subchannel")?;
            let subchannel = Subchannel::new(value).ok_or(ParseError::OutOfRange {
                key: "subchannel",
                value: u64::from(value),
            })?;
            let class = ClassId(fields.number("class")?);
            RingEntry::Bind {
                channel,
                subchannel,
                class,
            }
        }
        Kind::Write => RingEntry::Write {
            channel: ChannelId(fields.number("channel")?),
            class: ClassId(fields.number("class")?),
            method: fields.number("method")?,
            value: fields.number("value")?,
        },
        Kind::Run => RingEntry::Run {
            channel: ChannelId(fields.number("channel")?),
            class: ClassId(fields.number("class")?),
            method: fields.number("method")?,
            words: Cow::Owned(list("words", fields.text("words")?)?),
        },
        Kind::Release => release(&mut fields)?,
        Kind::Syncpoint => {
            let value: u8 = fields.number("id")?;
            let id = SyncpointId::new(value).ok_or(ParseError::OutOfRange {
                key: "id",
                value: u64::from(value),
            })?;
            RingEntry::Syncpoint { id }
        }
        Kind::WaitForIdle => RingEntry::WaitForIdle {
            channel: ChannelId(fields.number("channel")?),
        },
        Kind::Invalidate => RingEntry::Invalidate {
            channel: ChannelId(fields.number("channel")?),
            what: fields.named("what", &INVALIDATION_NAMES)?,
        },
        Kind::Marker => RingEntry::Marker {
            token: MarkerToken(fields.number("token")?),
        },
        Kind::Present => present(&mut fields)?,
        Kind::CaptureControl => RingEntry::CaptureControl(fields.named("command", &CAPTURE_NAMES)?),
    };
    fields.finish()?;
    Ok(entry)
}

/// Reads a text of entries, one per line, skipping blank lines.
pub fn parse_text(text: &str) -> Result<Vec<RingEntry<'static>>, TextError> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            parse_entry(line).map_err(|error| TextError {
                line: index + 1,
                error,
            })
        })
        .collect()
}

/// Writes entries in the text form, one per line, each ending in a newline.
pub fn write_text(entries: &[RingEntry<'_>]) -> String {
    let mut text = String::new();
    for entry in entries {
        // Writing to a `String` cannot fail.
        let _ = writeln!(text, "{entry}");
    }
    text
}

//! The ring's word encoding: each [`RingEntry`] as one or more 64-bit words.
//!
//! The encoding is internal to the GPU (the spec: "the encoding is internal; a text form exists
//! for tests and captures"); this layout is starquake's own, modelled on Eden-NX2's ring
//! (`eden-nx2/src/video_core/gpu_split.h` @ 50fa86bb, `Kind` and its entry words), but with
//! every entry plain data: no engine pointers and no function pointers.
//!
//! The first word of every entry is a header: its low half holds the kind in bits 7:0 and the
//! kind's small fields above it, read and written through one [`Field`] each; its high half holds
//! one 32-bit value. Fields and words an entry does not use are zero, and the decoder rejects
//! anything else, so every word sequence that decodes is exactly what [`encode`] writes for the
//! entry it decodes to.
//!
//! | Kind | Header low half | Header high half | Further words |
//! |---|---|---|---|
//! | 1 bind | subchannel 15:8 | channel | class (low half) |
//! | 2 write | method 23:8 | value | channel (low), class (high) |
//! | 3 run | method 23:8 | word count | channel (low), class (high); the words, two per word, first in the low half |
//! | 4 release | reduce 8, eight bytes 9, timestamp 10, reduction 14:12, format 17:16 | 0 | address; payload |
//! | 5 syncpoint | id 15:8 | 0 | |
//! | 6 wait for idle | | channel | |
//! | 7 invalidate | `MEM_OP_D` operation 12:8 | channel | |
//! | 8 marker | | 0 | token |
//! | 9 present | | format | address; width, height; pitch, block height; shown x, y; shown width, height; layer; display |
//! | 10 capture | command 9:8 (1 start, 2 stop) | 0 | |
//!
//! A release's reduction and format use the `RED_*` encoding of `classes/clc797.h`, through the
//! same helpers as `MME_DMA_REDUCTION` ([`crate::sema`]). Kind 0 is not an entry, so zeroed ring
//! memory never decodes.

use std::borrow::Cow;
use std::fmt;

use thiserror::Error;

use super::entry::{
    CaptureCommand, ChannelId, ClassId, DisplayId, ImageDesc, Invalidation, LayerId, MarkerToken,
    PresentRequest, Release, ReleaseOp, RingEntry, Subchannel, SyncpointId,
};
use crate::field::Field;
use crate::sema::{self, PayloadSize, ReleaseForm, red_reduction, red_reduction_fields};
use starquake_render::command::Rect;

/// The most words one [`RingEntry::Run`] carries: Eden-NX2's `MaxMultiWords`, at which it splits
/// long runs into pieces "as when a method's data continues in the next GPFIFO entry". The front
/// splits longer runs the same way.
pub const MAX_RUN_WORDS: usize = 8192;

/// The kind of an entry (header bits 7:0).
pub(super) const KIND: Field = Field::new(7, 0);
/// Bind: the subchannel.
pub(super) const BIND_SUBCHANNEL: Field = Field::new(15, 8);
/// Write and run: the method.
pub(super) const METHOD: Field = Field::new(23, 8);
/// Release: set for a reduction.
pub(super) const RELEASE_REDUCE: Field = Field::new(8, 8);
/// Release: set for an 8-byte payload.
pub(super) const RELEASE_EIGHT_BYTES: Field = Field::new(9, 9);
/// Release: set when a timestamp follows the payload.
pub(super) const RELEASE_TIMESTAMP: Field = Field::new(10, 10);
/// Release: the reduction (`RED_ADD` … `RED_XOR`).
pub(super) const RELEASE_REDUCTION_OP: Field = Field::new(14, 12);
/// Release: the reduction format (`RED_FORMAT_UNSIGNED`, `RED_FORMAT_SIGNED`).
pub(super) const RELEASE_REDUCTION_FORMAT: Field = Field::new(17, 16);
/// Syncpoint: the id.
pub(super) const SYNCPOINT_ID: Field = Field::new(15, 8);
/// Invalidate: the `MEM_OP_D_OPERATION` value.
pub(super) const INVALIDATION: Field = Field::new(12, 8);
/// Capture: the command.
pub(super) const CAPTURE_COMMAND: Field = Field::new(9, 8);
/// [`CAPTURE_COMMAND`] for [`CaptureCommand::Start`].
pub(super) const CAPTURE_START: u32 = 1;
/// [`CAPTURE_COMMAND`] for [`CaptureCommand::Stop`].
pub(super) const CAPTURE_STOP: u32 = 2;

/// The word name a release's reduction errors carry.
const RING_RELEASE: &str = "ring release";

/// The kind of a ring entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// [`RingEntry::Bind`].
    Bind,
    /// [`RingEntry::Write`].
    Write,
    /// [`RingEntry::Run`].
    Run,
    /// [`RingEntry::Release`].
    Release,
    /// [`RingEntry::Syncpoint`].
    Syncpoint,
    /// [`RingEntry::WaitForIdle`].
    WaitForIdle,
    /// [`RingEntry::Invalidate`].
    Invalidate,
    /// [`RingEntry::Marker`].
    Marker,
    /// [`RingEntry::Present`].
    Present,
    /// [`RingEntry::CaptureControl`].
    CaptureControl,
}

/// Every kind with its header value and its name in the text form.
pub(super) const KINDS: [(Kind, u32, &str); 10] = [
    (Kind::Bind, 1, "bind"),
    (Kind::Write, 2, "write"),
    (Kind::Run, 3, "run"),
    (Kind::Release, 4, "release"),
    (Kind::Syncpoint, 5, "syncpoint"),
    (Kind::WaitForIdle, 6, "wait-for-idle"),
    (Kind::Invalidate, 7, "invalidate"),
    (Kind::Marker, 8, "marker"),
    (Kind::Present, 9, "present"),
    (Kind::CaptureControl, 10, "capture"),
];

impl Kind {
    /// The kind's header value.
    pub fn value(self) -> u32 {
        KINDS
            .iter()
            .find(|(kind, ..)| *kind == self)
            .map_or(0, |&(_, value, _)| value)
    }

    /// The kind with header value `value`.
    pub fn from_value(value: u32) -> Option<Self> {
        KINDS
            .iter()
            .find(|&&(_, kind_value, _)| kind_value == value)
            .map(|&(kind, ..)| kind)
    }

    /// The kind's name in the text form.
    pub fn name(self) -> &'static str {
        KINDS
            .iter()
            .find(|(kind, ..)| *kind == self)
            .map_or("", |&(.., name)| name)
    }

    /// The kind of `entry`.
    pub fn of(entry: &RingEntry<'_>) -> Self {
        match entry {
            RingEntry::Bind { .. } => Self::Bind,
            RingEntry::Write { .. } => Self::Write,
            RingEntry::Run { .. } => Self::Run,
            RingEntry::Release(_) => Self::Release,
            RingEntry::Syncpoint { .. } => Self::Syncpoint,
            RingEntry::WaitForIdle { .. } => Self::WaitForIdle,
            RingEntry::Invalidate { .. } => Self::Invalidate,
            RingEntry::Marker { .. } => Self::Marker,
            RingEntry::Present(_) => Self::Present,
            RingEntry::CaptureControl(_) => Self::CaptureControl,
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An entry that has no encoding.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum EncodeError {
    /// A run longer than [`MAX_RUN_WORDS`].
    #[error("a run of {words} words is longer than the most one entry carries, {MAX_RUN_WORDS}")]
    RunTooLong {
        /// Its length.
        words: usize,
    },
}

/// Words that are not an entry.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DecodeError {
    /// There are no words.
    #[error("there is no ring entry here")]
    Empty,
    /// The header's kind is not one of [`Kind`].
    #[error("ring entry kind {kind} is not defined")]
    UnknownKind {
        /// The kind field.
        kind: u32,
    },
    /// The entry is longer than the words left.
    #[error("a {kind} entry needs {needed} words but {available} are left")]
    Truncated {
        /// The kind.
        kind: Kind,
        /// The entry's length.
        needed: usize,
        /// The words there are.
        available: usize,
    },
    /// Bits the entry does not use are set.
    #[error("a {kind} entry sets unused bits {bits:#x} in its word {word}")]
    Reserved {
        /// The kind.
        kind: Kind,
        /// The word's index within the entry.
        word: usize,
        /// The offending bits.
        bits: u64,
    },
    /// A run's count is above [`MAX_RUN_WORDS`].
    #[error("a run of {words} words is longer than the most one entry carries, {MAX_RUN_WORDS}")]
    RunTooLong {
        /// The count field.
        words: u32,
    },
    /// A bind's subchannel is out of range.
    #[error("subchannel {value} is out of range")]
    Subchannel {
        /// The field.
        value: u32,
    },
    /// A syncpoint id is out of range.
    #[error("syncpoint {value} is out of range")]
    Syncpoint {
        /// The field.
        value: u32,
    },
    /// An invalidation's operation is not one `clc76f.h` defines.
    #[error("memory operation {value:#x} is not defined")]
    Invalidation {
        /// The field.
        value: u32,
    },
    /// A capture command is not defined.
    #[error("capture command {value} is not defined")]
    Capture {
        /// The field.
        value: u32,
    },
    /// A release's reduction is not defined.
    #[error(transparent)]
    Reduction(#[from] sema::DecodeError),
}

/// A word from its low and high halves.
fn word(low: u32, high: u32) -> u64 {
    (u64::from(high) << 32) | u64::from(low)
}

/// A word's low and high halves.
fn halves(word: u64) -> (u32, u32) {
    let bytes = word.to_le_bytes();
    (
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    )
}

/// The header's low half with `kind` and nothing else.
fn header(kind: Kind) -> u32 {
    KIND.insert_truncated(0, kind.value())
}

/// The bits of `fields`, as a mask of the low half.
fn mask(fields: &[Field]) -> u32 {
    fields.iter().fold(KIND.max(), |mask, field| {
        mask | field.insert_truncated(0, field.max())
    })
}

/// Words a run of `words` words takes: the header, the channel word, then two per word.
fn run_words(words: usize) -> usize {
    2 + words.div_ceil(2)
}

/// The number of words `entry` encodes to.
pub fn encoded_words(entry: &RingEntry<'_>) -> usize {
    match entry {
        RingEntry::Syncpoint { .. }
        | RingEntry::WaitForIdle { .. }
        | RingEntry::Invalidate { .. }
        | RingEntry::CaptureControl(_) => 1,
        RingEntry::Bind { .. } | RingEntry::Write { .. } | RingEntry::Marker { .. } => 2,
        RingEntry::Release(_) => 3,
        RingEntry::Present(_) => 8,
        RingEntry::Run { words, .. } => run_words(words.len()),
    }
}

/// The words of a channel and a class.
fn channel_class(channel: ChannelId, class: ClassId) -> u64 {
    word(channel.0, class.0)
}

/// Appends the encoding of `entry` to `out`; on an error `out` is unchanged.
pub fn encode(entry: &RingEntry<'_>, out: &mut Vec<u64>) -> Result<(), EncodeError> {
    let kind = Kind::of(entry);
    let head = header(kind);
    match entry {
        RingEntry::Bind {
            channel,
            subchannel,
            class,
        } => {
            let low = BIND_SUBCHANNEL.insert_truncated(head, u32::from(subchannel.get()));
            out.extend([word(low, channel.0), word(class.0, 0)]);
        }
        RingEntry::Write {
            channel,
            class,
            method,
            value,
        } => {
            let low = METHOD.insert_truncated(head, u32::from(*method));
            out.extend([word(low, *value), channel_class(*channel, *class)]);
        }
        RingEntry::Run {
            channel,
            class,
            method,
            words,
        } => {
            let count = u32::try_from(words.len())
                .ok()
                .filter(|_| words.len() <= MAX_RUN_WORDS)
                .ok_or(EncodeError::RunTooLong { words: words.len() })?;
            let low = METHOD.insert_truncated(head, u32::from(*method));
            out.extend([word(low, count), channel_class(*channel, *class)]);
            out.extend(words.chunks(2).map(|pair| match *pair {
                [first, second] => word(first, second),
                [first] => word(first, 0),
                _ => 0,
            }));
        }
        RingEntry::Release(release) => {
            let mut low = RELEASE_EIGHT_BYTES
                .insert_truncated(head, u32::from(release.form.size == PayloadSize::Eight));
            low = RELEASE_TIMESTAMP.insert_truncated(low, u32::from(release.form.timestamp));
            if let ReleaseOp::Reduce(reduction) = release.op {
                let (op, format) = red_reduction_fields(reduction);
                low = RELEASE_REDUCE.insert_truncated(low, 1);
                low = RELEASE_REDUCTION_OP.insert_truncated(low, op);
                low = RELEASE_REDUCTION_FORMAT.insert_truncated(low, format);
            }
            out.extend([word(low, 0), release.address, release.payload]);
        }
        RingEntry::Syncpoint { id } => {
            out.push(word(
                SYNCPOINT_ID.insert_truncated(head, u32::from(id.get())),
                0,
            ));
        }
        RingEntry::WaitForIdle { channel } => out.push(word(head, channel.0)),
        RingEntry::Invalidate { channel, what } => {
            let low = INVALIDATION.insert_truncated(head, what.operation());
            out.push(word(low, channel.0));
        }
        RingEntry::Marker { token } => out.extend([word(head, 0), token.0]),
        RingEntry::Present(present) => {
            let image = &present.image;
            let shown = &present.shown;
            out.extend([
                word(head, image.format),
                image.address,
                word(image.width, image.height),
                word(image.pitch, image.block_height_log2),
                word(shown.x, shown.y),
                word(shown.width, shown.height),
                present.layer.0,
                present.display.0,
            ]);
        }
        RingEntry::CaptureControl(command) => {
            let value = match command {
                CaptureCommand::Start => CAPTURE_START,
                CaptureCommand::Stop => CAPTURE_STOP,
            };
            out.push(word(CAPTURE_COMMAND.insert_truncated(head, value), 0));
        }
    }
    Ok(())
}

/// The kind and length of the entry whose header is `head`.
pub fn entry_words(head: u64) -> Result<(Kind, usize), DecodeError> {
    let (low, high) = halves(head);
    let kind = Kind::from_value(KIND.get(low)).ok_or(DecodeError::UnknownKind {
        kind: KIND.get(low),
    })?;
    let words = match kind {
        Kind::Run => {
            let count = usize::try_from(high)
                .ok()
                .filter(|count| *count <= MAX_RUN_WORDS)
                .ok_or(DecodeError::RunTooLong { words: high })?;
            run_words(count)
        }
        Kind::Syncpoint | Kind::WaitForIdle | Kind::Invalidate | Kind::CaptureControl => 1,
        Kind::Bind | Kind::Write | Kind::Marker => 2,
        Kind::Release => 3,
        Kind::Present => 8,
    };
    Ok((kind, words))
}

/// The header and words of one entry being decoded.
struct Encoded<'a> {
    kind: Kind,
    low: u32,
    high: u32,
    words: &'a [u64],
}

impl Encoded<'_> {
    /// Word `index` of the entry; every index a kind reads is below its length.
    fn at(&self, index: usize) -> u64 {
        self.words.get(index).copied().unwrap_or_default()
    }

    /// Checks that the header's low half sets only `fields` (and the kind).
    fn check_low(&self, fields: &[Field]) -> Result<(), DecodeError> {
        let extra = self.low & !mask(fields);
        if extra == 0 {
            Ok(())
        } else {
            Err(DecodeError::Reserved {
                kind: self.kind,
                word: 0,
                bits: u64::from(extra),
            })
        }
    }

    /// Checks that the high half of word `index` is zero.
    fn check_high_zero(&self, index: usize) -> Result<(), DecodeError> {
        let value = self.at(index);
        if halves(value).1 == 0 {
            Ok(())
        } else {
            Err(DecodeError::Reserved {
                kind: self.kind,
                word: index,
                bits: value & !u64::from(u32::MAX),
            })
        }
    }

    fn bind(&self) -> Result<RingEntry<'static>, DecodeError> {
        self.check_low(&[BIND_SUBCHANNEL])?;
        self.check_high_zero(1)?;
        let value = BIND_SUBCHANNEL.get(self.low);
        let subchannel = u8::try_from(value)
            .ok()
            .and_then(Subchannel::new)
            .ok_or(DecodeError::Subchannel { value })?;
        Ok(RingEntry::Bind {
            channel: ChannelId(self.high),
            subchannel,
            class: ClassId(halves(self.at(1)).0),
        })
    }

    fn write(&self) -> Result<RingEntry<'static>, DecodeError> {
        self.check_low(&[METHOD])?;
        let (channel, class) = halves(self.at(1));
        Ok(RingEntry::Write {
            channel: ChannelId(channel),
            class: ClassId(class),
            method: method_of(METHOD.get(self.low)),
            value: self.high,
        })
    }

    fn run(&self) -> Result<RingEntry<'static>, DecodeError> {
        self.check_low(&[METHOD])?;
        let (channel, class) = halves(self.at(1));
        let count = usize::try_from(self.high).unwrap_or_default();
        let mut data = Vec::with_capacity(count);
        for (index, &pair) in self.words.iter().enumerate().skip(2) {
            let (first, second) = halves(pair);
            data.push(first);
            if data.len() < count {
                data.push(second);
            } else {
                self.check_high_zero(index)?;
            }
        }
        Ok(RingEntry::Run {
            channel: ChannelId(channel),
            class: ClassId(class),
            method: method_of(METHOD.get(self.low)),
            words: Cow::Owned(data),
        })
    }

    fn release(&self) -> Result<RingEntry<'static>, DecodeError> {
        let reduce = RELEASE_REDUCE.is_set(self.low);
        if reduce {
            self.check_low(&[
                RELEASE_REDUCE,
                RELEASE_EIGHT_BYTES,
                RELEASE_TIMESTAMP,
                RELEASE_REDUCTION_OP,
                RELEASE_REDUCTION_FORMAT,
            ])?;
        } else {
            self.check_low(&[RELEASE_EIGHT_BYTES, RELEASE_TIMESTAMP])?;
        }
        self.check_high_zero(0)?;
        let op = if reduce {
            ReleaseOp::Reduce(red_reduction(
                RING_RELEASE,
                RELEASE_REDUCTION_OP.get(self.low),
                RELEASE_REDUCTION_FORMAT.get(self.low),
            )?)
        } else {
            ReleaseOp::Write
        };
        let size = if RELEASE_EIGHT_BYTES.is_set(self.low) {
            PayloadSize::Eight
        } else {
            PayloadSize::Four
        };
        Ok(RingEntry::Release(Release {
            address: self.at(1),
            payload: self.at(2),
            form: ReleaseForm {
                size,
                timestamp: RELEASE_TIMESTAMP.is_set(self.low),
            },
            op,
        }))
    }

    fn syncpoint(&self) -> Result<RingEntry<'static>, DecodeError> {
        self.check_low(&[SYNCPOINT_ID])?;
        self.check_high_zero(0)?;
        let value = SYNCPOINT_ID.get(self.low);
        let id = u8::try_from(value)
            .ok()
            .and_then(SyncpointId::new)
            .ok_or(DecodeError::Syncpoint { value })?;
        Ok(RingEntry::Syncpoint { id })
    }

    fn invalidate(&self) -> Result<RingEntry<'static>, DecodeError> {
        self.check_low(&[INVALIDATION])?;
        let value = INVALIDATION.get(self.low);
        let what =
            Invalidation::from_operation(value).ok_or(DecodeError::Invalidation { value })?;
        Ok(RingEntry::Invalidate {
            channel: ChannelId(self.high),
            what,
        })
    }

    fn present(&self) -> Result<RingEntry<'static>, DecodeError> {
        self.check_low(&[])?;
        let (width, height) = halves(self.at(2));
        let (pitch, block_height_log2) = halves(self.at(3));
        let (x, y) = halves(self.at(4));
        let (shown_width, shown_height) = halves(self.at(5));
        Ok(RingEntry::Present(PresentRequest {
            image: ImageDesc {
                address: self.at(1),
                width,
                height,
                pitch,
                block_height_log2,
                format: self.high,
            },
            shown: Rect {
                x,
                y,
                width: shown_width,
                height: shown_height,
            },
            layer: LayerId(self.at(6)),
            display: DisplayId(self.at(7)),
        }))
    }

    fn capture(&self) -> Result<RingEntry<'static>, DecodeError> {
        self.check_low(&[CAPTURE_COMMAND])?;
        self.check_high_zero(0)?;
        let command = match CAPTURE_COMMAND.get(self.low) {
            CAPTURE_START => CaptureCommand::Start,
            CAPTURE_STOP => CaptureCommand::Stop,
            value => return Err(DecodeError::Capture { value }),
        };
        Ok(RingEntry::CaptureControl(command))
    }
}

/// Decodes the entry at the start of `words`, returning it and the number of words it took.
/// Words after the entry are not read.
pub fn decode(words: &[u64]) -> Result<(RingEntry<'static>, usize), DecodeError> {
    let &head = words.first().ok_or(DecodeError::Empty)?;
    let (kind, len) = entry_words(head)?;
    let words = words.get(..len).ok_or(DecodeError::Truncated {
        kind,
        needed: len,
        available: words.len(),
    })?;
    let (low, high) = halves(head);
    let entry = Encoded {
        kind,
        low,
        high,
        words,
    };
    let entry = match kind {
        Kind::Bind => entry.bind(),
        Kind::Write => entry.write(),
        Kind::Run => entry.run(),
        Kind::Release => entry.release(),
        Kind::Syncpoint => entry.syncpoint(),
        Kind::WaitForIdle => entry.check_low(&[]).map(|()| RingEntry::WaitForIdle {
            channel: ChannelId(high),
        }),
        Kind::Invalidate => entry.invalidate(),
        Kind::Marker => entry
            .check_low(&[])
            .and_then(|()| entry.check_high_zero(0))
            .map(|()| RingEntry::Marker {
                token: MarkerToken(entry.at(1)),
            }),
        Kind::Present => entry.present(),
        Kind::CaptureControl => entry.capture(),
    }?;
    Ok((entry, len))
}

/// A 16-bit method from a field that holds 16 bits.
fn method_of(field: u32) -> u16 {
    let [low, high, ..] = field.to_le_bytes();
    u16::from_le_bytes([low, high])
}

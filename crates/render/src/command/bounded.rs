//! Caps on the lists nested inside one command, enforced while a command is decoded, and what a
//! decoded command costs in memory: the two bounds a render capture's reader needs so that a
//! hostile list cannot make decoding allocate without bound (docs/specs/render-backend.md,
//! "Render capture v2", "Bounds"; [`crate::capture`]).
//!
//! A draw holds four lists and a dispatch one. Their elements take more memory decoded than
//! encoded (a [`Binding`] takes `size_of::<Binding>()` bytes from as few as five encoded bytes), so
//! a single draw with a long enough list would cost several times its record's bytes before the
//! reader could charge it. Each list is therefore capped while it is decoded, before an element
//! past the cap is read:
//!
//! | List | Cap | Why |
//! |---|---|---|
//! | `DrawDesc::vertex_buffers` | [`GUEST_VERTEX_BUFFERS`] | the guest's own count |
//! | `DynamicValues::viewports`, `DynamicValues::scissors` | [`GUEST_VIEWPORTS`] | the guest's own count |
//! | `DrawDesc::bindings`, `DispatchDesc::bindings` | [`MAX_COMMAND_BINDINGS`] | a guess (below) |
//!
//! A list past its cap is a decode error (`postcard` keeps no message from it, so the reader
//! reports it as [`crate::capture::CaptureError::Decode`]); the writer checks the same caps with
//! [`check_nested`] and reports [`crate::capture::CaptureError::TooLarge`] naming the list.
//! Nothing else reads these caps: a list is still checked against the device when it is
//! submitted ([`crate::validate`]), which allows fewer viewports and vertex buffers where the
//! device has fewer.

use super::{Binding, BufferSlice, Command, Rect, Viewport};
use crate::caps::{GUEST_VERTEX_BUFFERS, GUEST_VIEWPORTS};
use serde::de::{Deserializer, Error, SeqAccess, Visitor};
use serde::Deserialize;
use std::fmt;
use std::marker::PhantomData;

/// The most bindings one draw or dispatch may carry in a capture. A guess: five stages of the
/// guest's 18 constant buffers, 16 storage buffers, 32 textures and 8 images are 370 bindings,
/// and the cap leaves ten times that. Confirmed by the largest draw in a local title capture.
pub const MAX_COMMAND_BINDINGS: u32 = 4096;

/// A nested list longer than its cap, as the writer reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NestedOver {
    /// The list, in words.
    pub(crate) what: &'static str,
    /// Its length.
    pub(crate) len: u64,
    /// Its cap.
    pub(crate) max: u64,
}

/// Checks every nested list of `command` against its cap (module docs), in field order:
/// bindings, vertex buffers, viewports, scissors.
pub(crate) fn check_nested(command: &Command) -> Result<(), NestedOver> {
    let lists: &[(&'static str, usize, u32)] = match command {
        Command::Draw(draw) => &[
            ("draw bindings", draw.bindings.len(), MAX_COMMAND_BINDINGS),
            (
                "draw vertex buffers",
                draw.vertex_buffers.len(),
                GUEST_VERTEX_BUFFERS,
            ),
            (
                "draw viewports",
                draw.dynamic.viewports.len(),
                GUEST_VIEWPORTS,
            ),
            (
                "draw scissors",
                draw.dynamic.scissors.len(),
                GUEST_VIEWPORTS,
            ),
        ],
        Command::Dispatch(dispatch) => &[(
            "dispatch bindings",
            dispatch.bindings.len(),
            MAX_COMMAND_BINDINGS,
        )],
        _ => &[],
    };
    for &(what, len, max) in lists {
        let len = u64::try_from(len).unwrap_or(u64::MAX);
        if len > u64::from(max) {
            return Err(NestedOver {
                what,
                len,
                max: max.into(),
            });
        }
    }
    Ok(())
}

/// The memory one decoded command is charged: `size_of::<Command>()` plus the bytes of the
/// elements of its nested lists and its staging bytes. A list's real allocation may be up to
/// twice its charge while it grows.
pub(crate) fn decoded_bytes(command: &Command) -> u64 {
    let heap = match command {
        Command::Upload { from, .. } => from.len(),
        Command::Draw(draw) => elements::<Binding>(draw.bindings.len())
            .saturating_add(elements::<BufferSlice>(draw.vertex_buffers.len()))
            .saturating_add(elements::<Viewport>(draw.dynamic.viewports.len()))
            .saturating_add(elements::<Rect>(draw.dynamic.scissors.len())),
        Command::Dispatch(dispatch) => elements::<Binding>(dispatch.bindings.len()),
        Command::Copy(_)
        | Command::Blit(_)
        | Command::Clear(_)
        | Command::SetRenderTargets(_)
        | Command::Readback { .. }
        | Command::Present(_)
        | Command::Marker(_) => 0,
    };
    u64::try_from(size_of::<Command>().saturating_add(heap)).unwrap_or(u64::MAX)
}

/// Bytes of `count` elements of `T`.
fn elements<T>(count: usize) -> usize {
    count.saturating_mul(size_of::<T>())
}

/// Decodes a list of at most `max` elements; one more is a decode error, raised before it is
/// read. No more than `max` elements are allocated ahead, whatever length the input claims.
fn bounded<'de, D, T>(deserializer: D, max: u32) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    deserializer.deserialize_seq(Bounded {
        max,
        element: PhantomData,
    })
}

/// The visitor behind [`bounded`].
struct Bounded<T> {
    max: u32,
    element: PhantomData<T>,
}

impl<'de, T: Deserialize<'de>> Visitor<'de> for Bounded<T> {
    type Value = Vec<T>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a list of at most {} elements", self.max)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let max = usize::try_from(self.max).unwrap_or(usize::MAX);
        if let Some(len) = seq.size_hint()
            && len > max
        {
            return Err(A::Error::invalid_length(len, &self));
        }
        let mut list = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(max));
        while list.len() < max {
            match seq.next_element()? {
                Some(element) => list.push(element),
                None => return Ok(list),
            }
        }
        // At the cap: one more element is too many. Postcard's sequences say their length up
        // front, so this is reached only when the claimed length was not believed.
        match seq.next_element::<serde::de::IgnoredAny>()? {
            None => Ok(list),
            Some(_) => Err(A::Error::invalid_length(max.saturating_add(1), &self)),
        }
    }
}

/// `DrawDesc::bindings` and `DispatchDesc::bindings`, capped at [`MAX_COMMAND_BINDINGS`].
pub(super) fn bindings<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Binding>, D::Error> {
    bounded(d, MAX_COMMAND_BINDINGS)
}

/// `DrawDesc::vertex_buffers`, capped at [`GUEST_VERTEX_BUFFERS`].
pub(super) fn vertex_buffers<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<Vec<BufferSlice>, D::Error> {
    bounded(d, GUEST_VERTEX_BUFFERS)
}

/// `DynamicValues::viewports`, capped at [`GUEST_VIEWPORTS`].
pub(super) fn viewports<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Viewport>, D::Error> {
    bounded(d, GUEST_VIEWPORTS)
}

/// `DynamicValues::scissors`, capped at [`GUEST_VIEWPORTS`].
pub(super) fn scissors<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Rect>, D::Error> {
    bounded(d, GUEST_VIEWPORTS)
}

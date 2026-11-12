//! Handles to host objects, and fence values.
//!
//! Ids are plain integers with a generation, allocated by the backend's common layer
//! (docs/specs/render-backend.md, "Handles, lifetimes and fences"). They carry no host object
//! and no reference, so a command list holding them is plain data. A stale or foreign id in a
//! command is refused when the list is encoded; this module only defines the values.

use serde::{Deserialize, Serialize};

/// Defines a generational id type.
macro_rules! ids {
    ($( $(#[$meta:meta])* $name:ident ),* $(,)?) => {$(
        $(#[$meta])*
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        pub struct $name {
            index: u32,
            generation: u32,
        }

        impl $name {
            /// The id of slot `index` in its `generation`.
            pub const fn new(index: u32, generation: u32) -> Self {
                Self { index, generation }
            }

            /// The slot the id names.
            pub const fn index(self) -> u32 {
                self.index
            }

            /// How many times the slot had been reused when this id was handed out.
            pub const fn generation(self) -> u32 {
                self.generation
            }
        }
    )*};
}

ids! {
    /// A host buffer.
    BufferId,
    /// A host texture (render targets are textures with the render-target usage).
    TextureId,
    /// A view of a texture: format, mip and layer range, swizzle.
    ViewId,
    /// A host sampler.
    SamplerId,
    /// A shader module, shared by every pipeline that uses it.
    ShaderId,
    /// A graphics or compute pipeline.
    PipelineId,
    /// The bytes a `Readback` command copies out, collected with `Backend::take_readback`.
    ReadbackId,
}

/// Any resource [`crate::Backend::destroy`] can release.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceId {
    /// A buffer.
    Buffer(BufferId),
    /// A texture.
    Texture(TextureId),
    /// A view.
    View(ViewId),
    /// A sampler.
    Sampler(SamplerId),
    /// A shader.
    Shader(ShaderId),
    /// A pipeline.
    Pipeline(PipelineId),
}

/// A point on the backend's one completion counter. A submitted list signals its value when the
/// host GPU has finished it; [`crate::Backend::completed`] is the highest value reached.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct FenceValue(u64);

impl FenceValue {
    /// The value before any list is submitted.
    pub const ZERO: Self = Self(0);

    /// The fence value `value`.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// The raw counter value.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The next value, or `None` when the counter is exhausted.
    pub const fn next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferId, FenceValue, ReadbackId};

    #[test]
    fn id_parts() {
        let id = BufferId::new(7, 3);
        assert_eq!((id.index(), id.generation()), (7, 3));
        assert_ne!(BufferId::new(7, 3), BufferId::new(7, 4));
        assert_ne!(BufferId::new(7, 3), BufferId::new(8, 3));
        assert_eq!(ReadbackId::new(1, 2).index(), 1);
    }

    #[test]
    fn fence_next_stops_at_the_end() {
        assert_eq!(FenceValue::ZERO.get(), 0);
        assert_eq!(FenceValue::ZERO.next(), Some(FenceValue::new(1)));
        assert_eq!(
            FenceValue::new(u64::MAX - 1).next(),
            Some(FenceValue::new(u64::MAX))
        );
        assert_eq!(FenceValue::new(u64::MAX).next(), None);
    }
}

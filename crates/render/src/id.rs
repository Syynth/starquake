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

/// The kind of a [`ResourceId`], without the id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IdKind {
    /// A buffer.
    Buffer,
    /// A texture.
    Texture,
    /// A view.
    View,
    /// A sampler.
    Sampler,
    /// A shader.
    Shader,
    /// A pipeline.
    Pipeline,
}

impl IdKind {
    /// How many kinds there are; [`IdKind::index`] is below it.
    pub const COUNT: usize = 6;

    /// Every kind, in declaration order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Buffer,
        Self::Texture,
        Self::View,
        Self::Sampler,
        Self::Shader,
        Self::Pipeline,
    ];

    /// The kind's position in [`IdKind::ALL`], for tables counted per kind.
    pub const fn index(self) -> usize {
        match self {
            Self::Buffer => 0,
            Self::Texture => 1,
            Self::View => 2,
            Self::Sampler => 3,
            Self::Shader => 4,
            Self::Pipeline => 5,
        }
    }

    /// The kind in words ("buffer").
    pub const fn name(self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::Texture => "texture",
            Self::View => "view",
            Self::Sampler => "sampler",
            Self::Shader => "shader",
            Self::Pipeline => "pipeline",
        }
    }
}

/// In words, as errors print it: "buffer".
impl std::fmt::Display for IdKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl ResourceId {
    /// The kind of resource.
    pub const fn kind(self) -> IdKind {
        match self {
            Self::Buffer(_) => IdKind::Buffer,
            Self::Texture(_) => IdKind::Texture,
            Self::View(_) => IdKind::View,
            Self::Sampler(_) => IdKind::Sampler,
            Self::Shader(_) => IdKind::Shader,
            Self::Pipeline(_) => IdKind::Pipeline,
        }
    }

    /// The slot and generation of the id inside, whatever its kind.
    pub const fn parts(self) -> (u32, u32) {
        match self {
            Self::Buffer(id) => (id.index(), id.generation()),
            Self::Texture(id) => (id.index(), id.generation()),
            Self::View(id) => (id.index(), id.generation()),
            Self::Sampler(id) => (id.index(), id.generation()),
            Self::Shader(id) => (id.index(), id.generation()),
            Self::Pipeline(id) => (id.index(), id.generation()),
        }
    }
}

/// In plain words, as errors print it: "buffer 3 (generation 1)".
impl std::fmt::Display for ResourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (index, generation) = self.parts();
        write!(f, "{} {index} (generation {generation})", self.kind())
    }
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
    use super::{
        BufferId, FenceValue, IdKind, PipelineId, ReadbackId, ResourceId, SamplerId, ShaderId,
        TextureId, ViewId,
    };

    #[test]
    fn resource_ids_in_words() {
        let cases = [
            (
                ResourceId::Buffer(BufferId::new(1, 2)),
                "buffer 1 (generation 2)",
            ),
            (
                ResourceId::Texture(TextureId::new(3, 4)),
                "texture 3 (generation 4)",
            ),
            (ResourceId::View(ViewId::new(5, 6)), "view 5 (generation 6)"),
            (
                ResourceId::Sampler(SamplerId::new(7, 8)),
                "sampler 7 (generation 8)",
            ),
            (
                ResourceId::Shader(ShaderId::new(9, 10)),
                "shader 9 (generation 10)",
            ),
            (
                ResourceId::Pipeline(PipelineId::new(11, 12)),
                "pipeline 11 (generation 12)",
            ),
        ];
        for (id, words) in cases {
            assert_eq!(id.to_string(), words);
        }
    }

    #[test]
    fn id_kinds() {
        // Each id's kind, its words and its place in the per-kind tables, written out.
        let cases = [
            (ResourceId::Buffer(BufferId::new(1, 2)), "buffer", 0),
            (ResourceId::Texture(TextureId::new(1, 2)), "texture", 1),
            (ResourceId::View(ViewId::new(1, 2)), "view", 2),
            (ResourceId::Sampler(SamplerId::new(1, 2)), "sampler", 3),
            (ResourceId::Shader(ShaderId::new(1, 2)), "shader", 4),
            (ResourceId::Pipeline(PipelineId::new(1, 2)), "pipeline", 5),
        ];
        assert_eq!(cases.len(), IdKind::COUNT);
        for (i, (id, words, index)) in cases.into_iter().enumerate() {
            let kind = id.kind();
            assert_eq!(kind.to_string(), words);
            assert_eq!(kind.name(), words);
            assert_eq!(kind.index(), index);
            assert_eq!(IdKind::ALL[i], kind);
        }
    }

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

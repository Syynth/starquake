//! State groups ([`GroupKind`], [`Group`]) and sets of them ([`GroupSet`]): what a write marks
//! dirty and what deriving a draw clears.
//!
//! A group is a kind and, for a kind that comes once per array element, an index. The array
//! lengths are below as constants; the accessors of `three_d` bound their indices by the same
//! constants, so a group and the accessor that reads it can never disagree.

use crate::classes::clc797;

/// Viewports: Mesa's length for `SET_VIEWPORT_*` (`METHOD_ARRAY_SIZES` in
/// `mesa-nouveau/src/nouveau/headers/class_parser.py`); Eden's `NumViewports` agrees.
pub const VIEWPORTS: u32 = 16;

/// Scissors: Mesa's length for `SET_SCISSOR_*`.
pub const SCISSORS: u32 = 16;

/// Blend targets: Mesa's length for `SET_BLEND`, `SET_BLEND_PER_TARGET_*` and `SET_CT_WRITE`.
pub const BLEND_TARGETS: u32 = 8;

/// Colour targets: Mesa's length for `SET_COLOR_TARGET_*`; Eden's `NumRenderTargets` agrees.
pub const COLOR_TARGETS: u32 = 8;

/// Vertex streams in each of the header's two halves, `SET_VERTEX_STREAM_A_*` and
/// `SET_VERTEX_STREAM_B_*`: Mesa's length for `SET_VERTEX_STREAM_*`.
pub const VERTEX_STREAMS_PER_HALF: u32 = 16;

/// Vertex streams: both halves, stream `j` below [`VERTEX_STREAMS_PER_HALF`] in
/// `SET_VERTEX_STREAM_A_*(j)` and the rest in `SET_VERTEX_STREAM_B_*(j - 16)` (and likewise
/// `SET_VERTEX_STREAM_INSTANCE_A` and `_B`). Eden-NX2 reads the two halves as one array of 32
/// (`vertex_streams` at byte 0x1C00 and `vertex_stream_instances` at 0x1880, `NumVertexArrays`,
/// `eden-nx2/src/video_core/engines/maxwell_3d.h`).
pub const VERTEX_STREAMS: u32 = 2 * VERTEX_STREAMS_PER_HALF;

/// Vertex stream sizes (`SET_VERTEX_STREAM_SIZE_A` and `_B`): Mesa's length for
/// `SET_VERTEX_STREAM_*`, which covers them. Streams from 16 up have no size register here:
/// **guess**, since the header's gap after `SET_VERTEX_STREAM_SIZE_B(15)` (bytes 0x680 to 0x6FF)
/// would hold sixteen more, and Eden-NX2 keeps its 32 limits at byte 0x1F00, a method this header
/// does not define. A trace of a title binding a seventeenth vertex buffer would settle it.
pub const VERTEX_STREAM_SIZES: u32 = 16;

/// Vertex attributes in each half, `SET_VERTEX_ATTRIBUTE_A` and `_B`: Mesa's length for
/// `SET_VERTEX_ATTRIBUTE_*`.
pub const VERTEX_ATTRIBUTES_PER_HALF: u32 = 16;

/// Vertex attributes: both halves, as Eden-NX2's `vertex_attrib_format` (byte 0x1160,
/// `NumVertexAttributes`, `maxwell_3d.h`) reads them.
pub const VERTEX_ATTRIBUTES: u32 = 2 * VERTEX_ATTRIBUTES_PER_HALF;

/// Bind groups (`BIND_GROUP_CONSTANT_BUFFER(j)`): Mesa's length. Eden-NX2 binds five
/// (`MaxShaderStage`, `ProcessCBBind(0..4)` in `maxwell_3d.cpp`); the owner's rule for
/// references that disagree (gpu-protocol decision 10: where one answer contains the other, the
/// larger) keeps Mesa's 16. A listed divergence in the `state` module docs.
pub const BIND_GROUPS: u32 = 16;

/// Constant buffer slots per bind group: every value of the bind's `SHADER_SLOT` field (8:4).
/// Eden-NX2 keeps 18 (`MaxConstBuffers`) and ignores a bind to a slot above them; a listed
/// divergence in the `state` module docs.
pub const CONSTANT_BUFFER_SLOTS: u32 = clc797::BIND_GROUP_CONSTANT_BUFFER_SHADER_SLOT.max() + 1;

/// Pipelines (shader stages by program slot): Mesa's length for `SET_PIPELINE_*`; Eden's
/// `MaxShaderProgram` agrees.
pub const PIPELINES: u32 = 6;

/// What a group covers, named after the Mesa methods it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GroupKind {
    /// Every viewport: `SET_VIEWPORT_*`, the clip control, the surface clip and the window
    /// origin. One group, as Eden's `Viewports`.
    Viewport,
    /// Every scissor: `SET_SCISSOR_*`. One group, as Eden's `Scissors`.
    Scissor,
    /// The blend state shared by every target: `SET_BLEND_*` without `PER_TARGET`, and
    /// `SET_SINGLE_CT_WRITE_CONTROL`.
    Blend,
    /// Blend target `j`: `SET_BLEND(j)`, `SET_BLEND_PER_TARGET_*(j)` and `SET_CT_WRITE(j)`.
    BlendPerTarget,
    /// Colour target `j`: `SET_COLOR_TARGET_*(j)`.
    ColorTarget,
    /// Which colour targets are drawn to: `SET_CT_SELECT` and the surface clip.
    CtSelect,
    /// The depth target: `SET_ZT_*`.
    Zt,
    /// Vertex stream `j`: its format, location, frequency, size and instancing.
    VertexStream,
    /// Every vertex attribute: `SET_VERTEX_ATTRIBUTE_A` and `_B`. One group, as Eden's
    /// `VertexInput`.
    VertexAttribute,
    /// The index buffer: `SET_INDEX_BUFFER_*`.
    IndexBuffer,
    /// Bind group `j`'s constant buffers: `BIND_GROUP_CONSTANT_BUFFER(j)`.
    BindGroup,
    /// The texture header and sampler pools: `SET_TEX_HEADER_POOL_*` and `SET_TEX_SAMPLER_POOL_*`.
    TexPool,
    /// Pipeline `j` (its program address and stage): `SET_PIPELINE_*(j)`.
    Pipeline,
    /// Rasterizer state: polygon modes, culling, front face, raster enable, polygon offset and
    /// depth bias, line and point sizes.
    Rasterizer,
    /// Depth and stencil state: `SET_DEPTH_TEST`, `_WRITE`, `_FUNC`, the depth bounds and every
    /// `SET_STENCIL_*` and `SET_BACK_STENCIL_*`.
    DepthStencil,
}

impl GroupKind {
    /// Every kind, in bit order.
    pub const ALL: [Self; 15] = [
        Self::Viewport,
        Self::Scissor,
        Self::Blend,
        Self::BlendPerTarget,
        Self::ColorTarget,
        Self::CtSelect,
        Self::Zt,
        Self::VertexStream,
        Self::VertexAttribute,
        Self::IndexBuffer,
        Self::BindGroup,
        Self::TexPool,
        Self::Pipeline,
        Self::Rasterizer,
        Self::DepthStencil,
    ];

    /// The number of groups of this kind: the array length for a kind per element, 1 otherwise.
    pub const fn count(self) -> u32 {
        match self {
            Self::BlendPerTarget => BLEND_TARGETS,
            Self::ColorTarget => COLOR_TARGETS,
            Self::VertexStream => VERTEX_STREAMS,
            Self::BindGroup => BIND_GROUPS,
            Self::Pipeline => PIPELINES,
            Self::Viewport
            | Self::Scissor
            | Self::Blend
            | Self::CtSelect
            | Self::Zt
            | Self::VertexAttribute
            | Self::IndexBuffer
            | Self::TexPool
            | Self::Rasterizer
            | Self::DepthStencil => 1,
        }
    }

    /// The bit of the kind's first group in a [`GroupSet`].
    const fn first_bit(self) -> u32 {
        let mut bit = 0;
        let mut position = 0;
        while position < Self::ALL.len() {
            let kind = Self::ALL[position];
            if kind as u8 == self as u8 {
                return bit;
            }
            bit += kind.count();
            position += 1;
        }
        bit
    }
}

/// The number of groups of every kind together: the bits a [`GroupSet`] uses.
pub const GROUP_COUNT: u32 = {
    let mut count = 0;
    let mut position = 0;
    while position < GroupKind::ALL.len() {
        count += GroupKind::ALL[position].count();
        position += 1;
    }
    count
};

// Every group has its own bit in a `GroupSet`'s `u128`.
const _: () = assert!(GROUP_COUNT <= u128::BITS);

/// One state group: a kind and, for a kind per element, the element's index (0 otherwise).
/// Made only through [`Group::new`] and [`Group::one`], so the index is always below the
/// kind's [`GroupKind::count`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Group {
    kind: GroupKind,
    index: u8,
}

impl Group {
    /// The group of kind `kind` for element `index`; `None` from the kind's
    /// [`GroupKind::count`] up.
    pub fn new(kind: GroupKind, index: u32) -> Option<Self> {
        if index >= kind.count() {
            return None;
        }
        let index = u8::try_from(index).ok()?;
        Some(Self { kind, index })
    }

    /// The only group of a kind that is not per element (element 0 of one that is).
    pub const fn one(kind: GroupKind) -> Self {
        Self { kind, index: 0 }
    }

    /// The kind.
    pub const fn kind(self) -> GroupKind {
        self.kind
    }

    /// The element's index (0 for a kind that is not per element).
    pub const fn index(self) -> u32 {
        self.index as u32
    }

    /// The group's bit in a [`GroupSet`], below [`GROUP_COUNT`].
    const fn bit(self) -> u32 {
        self.kind.first_bit() + self.index as u32
    }

    /// The group at bit `bit`; `None` from [`GROUP_COUNT`] up.
    fn from_bit(bit: u32) -> Option<Self> {
        GroupKind::ALL.iter().find_map(|&kind| {
            let index = bit.checked_sub(kind.first_bit())?;
            Self::new(kind, index)
        })
    }
}

/// A set of [`Group`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GroupSet(u128);

impl GroupSet {
    /// No groups.
    pub const EMPTY: Self = Self(0);

    /// Whether the set has no groups.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The number of groups in the set.
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// Whether `group` is in the set.
    pub const fn contains(self, group: Group) -> bool {
        match 1u128.checked_shl(group.bit()) {
            Some(mask) => self.0 & mask != 0,
            None => false,
        }
    }

    /// The set with `group` added.
    #[must_use]
    pub const fn with(self, group: Group) -> Self {
        match 1u128.checked_shl(group.bit()) {
            Some(mask) => Self(self.0 | mask),
            None => self,
        }
    }

    /// The groups in either set.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The groups in this set and not in `other`.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// The groups, in bit order (the order of [`GroupKind::ALL`], then by index).
    pub fn iter(self) -> impl Iterator<Item = Group> {
        (0..u128::BITS)
            .filter(move |&bit| self.0 >> bit & 1 != 0)
            .filter_map(Group::from_bit)
    }
}

impl FromIterator<Group> for GroupSet {
    fn from_iter<I: IntoIterator<Item = Group>>(groups: I) -> Self {
        groups.into_iter().fold(Self::EMPTY, Self::with)
    }
}

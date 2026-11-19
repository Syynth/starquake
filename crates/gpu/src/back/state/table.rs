//! The group tables: which methods mark which groups ([`GROUPS_3D`], [`GROUPS_COMPUTE`]), and
//! the per-method lookup built from them once per class ([`GroupMap`]).
//!
//! Written by hand and keyed on the generated constants of [`crate::classes`] (the generated
//! tables have no group map; the issue for slice G10 leaves the choice between extending
//! `cargo xtask gen-classes` and a table keyed on its constants, and this is the latter). Each
//! entry is a [`MethodSpan`], the same span type the front's state table uses, with Mesa's array
//! length; nothing here restates a method address or stride.
//!
//! The grouping follows Eden's dirty-flag tables as a model (`eden/src/video_core/dirty_flags.cpp`
//! and `eden/src/video_core/renderer_vulkan/vk_state_tracker.cpp`, Eden @ cf654052,
//! GPL-3.0-or-later and GPL-2.0-or-later), with the groups named after Mesa's methods. Where a
//! method marks two groups it is listed twice, as Eden lists a register in both of its tables:
//! the surface clip marks [`GroupKind::Viewport`] and [`GroupKind::CtSelect`] (Eden: `Viewports`
//! and `RenderTargets`), and the window origin marks [`GroupKind::Viewport`] and
//! [`GroupKind::Rasterizer`] (Eden: `Viewports` and `FrontFace`).

use std::sync::OnceLock;

use crate::classes::{Method, clc7c0, clc797};
use crate::regs::{ClassRules, MethodSpan};
use crate::ring::ClassId;

use super::group::{
    BIND_GROUPS, BLEND_TARGETS, COLOR_TARGETS, Group, GroupKind, GroupSet, PIPELINES, SCISSORS,
    VERTEX_ATTRIBUTES_PER_HALF, VERTEX_STREAM_SIZES, VERTEX_STREAMS_PER_HALF, VIEWPORTS,
};

/// Which group an entry's elements mark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Marks {
    /// Every element marks the one group of the kind ([`Group::one`]).
    One(GroupKind),
    /// Element `i` marks the group of the kind with index `first + i`.
    PerElement {
        /// The kind.
        kind: GroupKind,
        /// The index element 0 marks.
        first: u32,
    },
}

impl Marks {
    /// The group element `index` of the entry marks; `None` if the index is past the kind's
    /// groups.
    pub fn group(self, index: u32) -> Option<Group> {
        match self {
            Self::One(kind) => Some(Group::one(kind)),
            Self::PerElement { kind, first } => Group::new(kind, first.checked_add(index)?),
        }
    }
}

/// One line of a group table: the methods of a span and the groups they mark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupEntry {
    /// The methods.
    pub span: MethodSpan,
    /// What each element marks.
    pub marks: Marks,
}

const fn one(method: &'static Method, kind: GroupKind) -> GroupEntry {
    GroupEntry {
        span: MethodSpan::one(method),
        marks: Marks::One(kind),
    }
}

/// An array whose every element marks the one group of `kind`.
const fn all(method: &'static Method, count: u32, kind: GroupKind) -> GroupEntry {
    GroupEntry {
        span: MethodSpan::array(method, count),
        marks: Marks::One(kind),
    }
}

/// An array whose element `i` marks group `first + i` of `kind`.
const fn each(method: &'static Method, count: u32, kind: GroupKind, first: u32) -> GroupEntry {
    GroupEntry {
        span: MethodSpan::array(method, count),
        marks: Marks::PerElement { kind, first },
    }
}

use GroupKind::{
    BindGroup, Blend, BlendPerTarget, ColorTarget, CtSelect, DepthStencil, IndexBuffer, Pipeline,
    Rasterizer, Scissor, TexPool, VertexAttribute, VertexStream, Viewport, Zt,
};

/// The 3D class's group table, by group. A method in no entry marks nothing (and is counted by
/// the mirror; [`super::Mirror::ungrouped_words`]).
pub const GROUPS_3D: &[GroupEntry] = &[
    // Viewports.
    all(&clc797::SET_VIEWPORT_SCALE_X, VIEWPORTS, Viewport),
    all(&clc797::SET_VIEWPORT_SCALE_Y, VIEWPORTS, Viewport),
    all(&clc797::SET_VIEWPORT_SCALE_Z, VIEWPORTS, Viewport),
    all(&clc797::SET_VIEWPORT_OFFSET_X, VIEWPORTS, Viewport),
    all(&clc797::SET_VIEWPORT_OFFSET_Y, VIEWPORTS, Viewport),
    all(&clc797::SET_VIEWPORT_OFFSET_Z, VIEWPORTS, Viewport),
    all(
        &clc797::SET_VIEWPORT_COORDINATE_SWIZZLE,
        VIEWPORTS,
        Viewport,
    ),
    all(
        &clc797::SET_VIEWPORT_INCREASE_SNAP_GRID_PRECISION,
        VIEWPORTS,
        Viewport,
    ),
    all(&clc797::SET_VIEWPORT_CLIP_HORIZONTAL, VIEWPORTS, Viewport),
    all(&clc797::SET_VIEWPORT_CLIP_VERTICAL, VIEWPORTS, Viewport),
    all(&clc797::SET_VIEWPORT_CLIP_MIN_Z, VIEWPORTS, Viewport),
    all(&clc797::SET_VIEWPORT_CLIP_MAX_Z, VIEWPORTS, Viewport),
    one(&clc797::SET_VIEWPORT_SCALE_OFFSET, Viewport),
    one(&clc797::SET_VIEWPORT_CLIP_CONTROL, Viewport),
    one(&clc797::SET_VIEWPORT_Z_CLIP, Viewport),
    one(&clc797::SET_SURFACE_CLIP_HORIZONTAL, Viewport),
    one(&clc797::SET_SURFACE_CLIP_VERTICAL, Viewport),
    one(&clc797::SET_WINDOW_ORIGIN, Viewport),
    // Scissors.
    all(&clc797::SET_SCISSOR_ENABLE, SCISSORS, Scissor),
    all(&clc797::SET_SCISSOR_HORIZONTAL, SCISSORS, Scissor),
    all(&clc797::SET_SCISSOR_VERTICAL, SCISSORS, Scissor),
    // Blend state shared by every target.
    one(&clc797::SET_BLEND_STATE_PER_TARGET, Blend),
    one(&clc797::SET_BLEND_SEPARATE_FOR_ALPHA, Blend),
    one(&clc797::SET_BLEND_COLOR_OP, Blend),
    one(&clc797::SET_BLEND_COLOR_SOURCE_COEFF, Blend),
    one(&clc797::SET_BLEND_COLOR_DEST_COEFF, Blend),
    one(&clc797::SET_BLEND_ALPHA_OP, Blend),
    one(&clc797::SET_BLEND_ALPHA_SOURCE_COEFF, Blend),
    one(&clc797::SET_BLEND_ALPHA_DEST_COEFF, Blend),
    one(&clc797::SET_BLEND_CONST_RED, Blend),
    one(&clc797::SET_BLEND_CONST_GREEN, Blend),
    one(&clc797::SET_BLEND_CONST_BLUE, Blend),
    one(&clc797::SET_BLEND_CONST_ALPHA, Blend),
    one(&clc797::SET_SINGLE_CT_WRITE_CONTROL, Blend),
    one(&clc797::SET_GLOBAL_COLOR_KEY, Blend),
    one(&clc797::SET_SINGLE_ROP_CONTROL, Blend),
    one(&clc797::SET_ANTI_ALIAS_ALPHA_CONTROL, Blend),
    one(&clc797::SET_LOGIC_OP, Blend),
    one(&clc797::SET_LOGIC_OP_FUNC, Blend),
    // Blend per target.
    each(&clc797::SET_BLEND, BLEND_TARGETS, BlendPerTarget, 0),
    each(&clc797::SET_CT_WRITE, BLEND_TARGETS, BlendPerTarget, 0),
    each(
        &clc797::SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA,
        BLEND_TARGETS,
        BlendPerTarget,
        0,
    ),
    each(
        &clc797::SET_BLEND_PER_TARGET_COLOR_OP,
        BLEND_TARGETS,
        BlendPerTarget,
        0,
    ),
    each(
        &clc797::SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF,
        BLEND_TARGETS,
        BlendPerTarget,
        0,
    ),
    each(
        &clc797::SET_BLEND_PER_TARGET_COLOR_DEST_COEFF,
        BLEND_TARGETS,
        BlendPerTarget,
        0,
    ),
    each(
        &clc797::SET_BLEND_PER_TARGET_ALPHA_OP,
        BLEND_TARGETS,
        BlendPerTarget,
        0,
    ),
    each(
        &clc797::SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF,
        BLEND_TARGETS,
        BlendPerTarget,
        0,
    ),
    each(
        &clc797::SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF,
        BLEND_TARGETS,
        BlendPerTarget,
        0,
    ),
    // Colour targets.
    each(&clc797::SET_COLOR_TARGET_A, COLOR_TARGETS, ColorTarget, 0),
    each(&clc797::SET_COLOR_TARGET_B, COLOR_TARGETS, ColorTarget, 0),
    each(
        &clc797::SET_COLOR_TARGET_WIDTH,
        COLOR_TARGETS,
        ColorTarget,
        0,
    ),
    each(
        &clc797::SET_COLOR_TARGET_HEIGHT,
        COLOR_TARGETS,
        ColorTarget,
        0,
    ),
    each(
        &clc797::SET_COLOR_TARGET_FORMAT,
        COLOR_TARGETS,
        ColorTarget,
        0,
    ),
    each(
        &clc797::SET_COLOR_TARGET_MEMORY,
        COLOR_TARGETS,
        ColorTarget,
        0,
    ),
    each(
        &clc797::SET_COLOR_TARGET_THIRD_DIMENSION,
        COLOR_TARGETS,
        ColorTarget,
        0,
    ),
    each(
        &clc797::SET_COLOR_TARGET_ARRAY_PITCH,
        COLOR_TARGETS,
        ColorTarget,
        0,
    ),
    each(
        &clc797::SET_COLOR_TARGET_LAYER,
        COLOR_TARGETS,
        ColorTarget,
        0,
    ),
    each(
        &clc797::SET_COLOR_TARGET_C_ROP_SLICE_MAP,
        COLOR_TARGETS,
        ColorTarget,
        0,
    ),
    // Which targets are drawn to.
    one(&clc797::SET_CT_SELECT, CtSelect),
    one(&clc797::SET_SURFACE_CLIP_HORIZONTAL, CtSelect),
    one(&clc797::SET_SURFACE_CLIP_VERTICAL, CtSelect),
    // The depth target.
    one(&clc797::SET_ZT_A, Zt),
    one(&clc797::SET_ZT_B, Zt),
    one(&clc797::SET_ZT_FORMAT, Zt),
    one(&clc797::SET_ZT_BLOCK_SIZE, Zt),
    one(&clc797::SET_ZT_ARRAY_PITCH, Zt),
    one(&clc797::SET_ZT_SIZE_A, Zt),
    one(&clc797::SET_ZT_SIZE_B, Zt),
    one(&clc797::SET_ZT_SIZE_C, Zt),
    one(&clc797::SET_ZT_LAYER, Zt),
    one(&clc797::SET_ZT_SELECT, Zt),
    one(&clc797::SET_ZT_SPARSE, Zt),
    one(&clc797::SET_ZT_READ_ONLY, Zt),
    // Vertex streams: the A half is streams 0 to 15, the B half 16 to 31; the sizes are one
    // array of 32 (Eden-NX2's, decision 10; `VERTEX_STREAM_SIZES`).
    each(
        &clc797::SET_VERTEX_STREAM_A_FORMAT,
        VERTEX_STREAMS_PER_HALF,
        VertexStream,
        0,
    ),
    each(
        &clc797::SET_VERTEX_STREAM_A_LOCATION_A,
        VERTEX_STREAMS_PER_HALF,
        VertexStream,
        0,
    ),
    each(
        &clc797::SET_VERTEX_STREAM_A_LOCATION_B,
        VERTEX_STREAMS_PER_HALF,
        VertexStream,
        0,
    ),
    each(
        &clc797::SET_VERTEX_STREAM_A_FREQUENCY,
        VERTEX_STREAMS_PER_HALF,
        VertexStream,
        0,
    ),
    each(
        &clc797::SET_VERTEX_STREAM_B_FORMAT,
        VERTEX_STREAMS_PER_HALF,
        VertexStream,
        VERTEX_STREAMS_PER_HALF,
    ),
    each(
        &clc797::SET_VERTEX_STREAM_B_LOCATION_A,
        VERTEX_STREAMS_PER_HALF,
        VertexStream,
        VERTEX_STREAMS_PER_HALF,
    ),
    each(
        &clc797::SET_VERTEX_STREAM_B_LOCATION_B,
        VERTEX_STREAMS_PER_HALF,
        VertexStream,
        VERTEX_STREAMS_PER_HALF,
    ),
    each(
        &clc797::SET_VERTEX_STREAM_B_FREQUENCY,
        VERTEX_STREAMS_PER_HALF,
        VertexStream,
        VERTEX_STREAMS_PER_HALF,
    ),
    each(
        &clc797::SET_VERTEX_STREAM_SIZE_A,
        VERTEX_STREAM_SIZES,
        VertexStream,
        0,
    ),
    each(
        &clc797::SET_VERTEX_STREAM_SIZE_B,
        VERTEX_STREAM_SIZES,
        VertexStream,
        0,
    ),
    each(
        &clc797::SET_VERTEX_STREAM_INSTANCE_A,
        VERTEX_STREAMS_PER_HALF,
        VertexStream,
        0,
    ),
    each(
        &clc797::SET_VERTEX_STREAM_INSTANCE_B,
        VERTEX_STREAMS_PER_HALF,
        VertexStream,
        VERTEX_STREAMS_PER_HALF,
    ),
    // Vertex attributes.
    all(
        &clc797::SET_VERTEX_ATTRIBUTE_A,
        VERTEX_ATTRIBUTES_PER_HALF,
        VertexAttribute,
    ),
    all(
        &clc797::SET_VERTEX_ATTRIBUTE_B,
        VERTEX_ATTRIBUTES_PER_HALF,
        VertexAttribute,
    ),
    // The index buffer.
    one(&clc797::SET_INDEX_BUFFER_A, IndexBuffer),
    one(&clc797::SET_INDEX_BUFFER_B, IndexBuffer),
    one(&clc797::SET_INDEX_BUFFER_E, IndexBuffer),
    one(&clc797::SET_INDEX_BUFFER_F, IndexBuffer),
    one(&clc797::SET_INDEX_BUFFER_SIZE_A, IndexBuffer),
    one(&clc797::SET_INDEX_BUFFER_SIZE_B, IndexBuffer),
    one(&clc797::DRAW_INDEX_BUFFER, IndexBuffer),
    // Constant buffer bindings, per bind group.
    each(
        &clc797::BIND_GROUP_CONSTANT_BUFFER,
        BIND_GROUPS,
        BindGroup,
        0,
    ),
    // The texture header and sampler pools.
    one(&clc797::SET_TEX_HEADER_POOL_A, TexPool),
    one(&clc797::SET_TEX_HEADER_POOL_B, TexPool),
    one(&clc797::SET_TEX_HEADER_POOL_C, TexPool),
    one(&clc797::SET_TEX_SAMPLER_POOL_A, TexPool),
    one(&clc797::SET_TEX_SAMPLER_POOL_B, TexPool),
    one(&clc797::SET_TEX_SAMPLER_POOL_C, TexPool),
    // Pipelines: every `SET_PIPELINE_*(j)` Mesa lists, as Eden marks its whole `pipelines` block.
    each(&clc797::SET_PIPELINE_SHADER, PIPELINES, Pipeline, 0),
    each(&clc797::SET_PIPELINE_RESERVED_B, PIPELINES, Pipeline, 0),
    each(&clc797::SET_PIPELINE_RESERVED_A, PIPELINES, Pipeline, 0),
    each(&clc797::SET_PIPELINE_REGISTER_COUNT, PIPELINES, Pipeline, 0),
    each(&clc797::SET_PIPELINE_BINDING, PIPELINES, Pipeline, 0),
    each(
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_A,
        PIPELINES,
        Pipeline,
        0,
    ),
    each(
        &clc797::SET_PIPELINE_PROGRAM_ADDRESS_B,
        PIPELINES,
        Pipeline,
        0,
    ),
    each(
        &clc797::SET_PIPELINE_PROGRAM_PREFETCH,
        PIPELINES,
        Pipeline,
        0,
    ),
    each(&clc797::SET_PIPELINE_RESERVED_E, PIPELINES, Pipeline, 0),
    // Rasterizer state.
    one(&clc797::SET_FRONT_POLYGON_MODE, Rasterizer),
    one(&clc797::SET_BACK_POLYGON_MODE, Rasterizer),
    one(&clc797::OGL_SET_CULL, Rasterizer),
    one(&clc797::OGL_SET_CULL_FACE, Rasterizer),
    one(&clc797::OGL_SET_FRONT_FACE, Rasterizer),
    one(&clc797::D3D_SET_CULL_MODE, Rasterizer),
    one(&clc797::SET_WINDOW_ORIGIN, Rasterizer),
    one(&clc797::SET_RASTER_ENABLE, Rasterizer),
    one(&clc797::SET_POLY_OFFSET_POINT, Rasterizer),
    one(&clc797::SET_POLY_OFFSET_LINE, Rasterizer),
    one(&clc797::SET_POLY_OFFSET_FILL, Rasterizer),
    one(&clc797::SET_DEPTH_BIAS, Rasterizer),
    one(&clc797::SET_SLOPE_SCALE_DEPTH_BIAS, Rasterizer),
    one(&clc797::SET_DEPTH_BIAS_CLAMP, Rasterizer),
    one(&clc797::SET_LINE_WIDTH_FLOAT, Rasterizer),
    one(&clc797::SET_ALIASED_LINE_WIDTH_FLOAT, Rasterizer),
    one(&clc797::SET_LINE_STIPPLE, Rasterizer),
    one(&clc797::SET_POINT_SIZE, Rasterizer),
    one(&clc797::SET_PROVOKING_VERTEX, Rasterizer),
    one(&clc797::SET_LINE_STIPPLE_PARAMETERS, Rasterizer),
    one(&clc797::SET_ANTI_ALIASED_LINE, Rasterizer),
    one(&clc797::SET_CONSERVATIVE_RASTER, Rasterizer),
    one(&clc797::SET_DA_PRIMITIVE_RESTART, Rasterizer),
    // Depth and stencil state.
    one(&clc797::SET_DEPTH_TEST, DepthStencil),
    one(&clc797::SET_DEPTH_WRITE, DepthStencil),
    one(&clc797::SET_DEPTH_FUNC, DepthStencil),
    one(&clc797::SET_DEPTH_BOUNDS_TEST, DepthStencil),
    one(&clc797::SET_DEPTH_BOUNDS_MIN, DepthStencil),
    one(&clc797::SET_DEPTH_BOUNDS_MAX, DepthStencil),
    one(&clc797::SET_STENCIL_TEST, DepthStencil),
    one(&clc797::SET_STENCIL_OP_FAIL, DepthStencil),
    one(&clc797::SET_STENCIL_OP_ZFAIL, DepthStencil),
    one(&clc797::SET_STENCIL_OP_ZPASS, DepthStencil),
    one(&clc797::SET_STENCIL_FUNC, DepthStencil),
    one(&clc797::SET_STENCIL_FUNC_REF, DepthStencil),
    one(&clc797::SET_STENCIL_FUNC_MASK, DepthStencil),
    one(&clc797::SET_STENCIL_MASK, DepthStencil),
    one(&clc797::SET_TWO_SIDED_STENCIL_TEST, DepthStencil),
    one(&clc797::SET_BACK_STENCIL_OP_FAIL, DepthStencil),
    one(&clc797::SET_BACK_STENCIL_OP_ZFAIL, DepthStencil),
    one(&clc797::SET_BACK_STENCIL_OP_ZPASS, DepthStencil),
    one(&clc797::SET_BACK_STENCIL_FUNC, DepthStencil),
    one(&clc797::SET_BACK_STENCIL_FUNC_REF, DepthStencil),
    one(&clc797::SET_BACK_STENCIL_FUNC_MASK, DepthStencil),
    one(&clc797::SET_BACK_STENCIL_MASK, DepthStencil),
];

/// The compute class's group table: only the texture header and sampler pools. Everything else
/// a dispatch needs comes from its QMD ([`crate::back::qmd`]).
pub const GROUPS_COMPUTE: &[GroupEntry] = &[
    one(&clc7c0::SET_TEX_HEADER_POOL_A, TexPool),
    one(&clc7c0::SET_TEX_HEADER_POOL_B, TexPool),
    one(&clc7c0::SET_TEX_HEADER_POOL_C, TexPool),
    one(&clc7c0::SET_TEX_SAMPLER_POOL_A, TexPool),
    one(&clc7c0::SET_TEX_SAMPLER_POOL_B, TexPool),
    one(&clc7c0::SET_TEX_SAMPLER_POOL_C, TexPool),
];

/// The group table of class `class`: [`GROUPS_3D`] for `0xC797`, [`GROUPS_COMPUTE`] for
/// `0xC7C0`, and none for every other class.
pub fn group_table(class: ClassId) -> &'static [GroupEntry] {
    match class.0 {
        clc797::CLASS => GROUPS_3D,
        clc7c0::CLASS => GROUPS_COMPUTE,
        _ => &[],
    }
}

/// A group table turned into a lookup by method number, built once per class.
#[derive(Debug)]
pub struct GroupMap {
    /// The groups each method marks, indexed by method number, as long as the class's register
    /// file.
    by_method: Box<[GroupSet]>,
    /// Every group the table names.
    named: GroupSet,
}

impl GroupMap {
    /// Builds the map of `table` for a register file of `registers` methods. An element at or
    /// past `registers`, or one whose group index is past its kind, is left out (the tests check
    /// that no shipping table has one).
    pub fn build(table: &[GroupEntry], registers: u32) -> Self {
        let length = usize::try_from(registers).unwrap_or(0);
        let mut by_method = vec![GroupSet::EMPTY; length].into_boxed_slice();
        let mut named = GroupSet::EMPTY;
        for entry in table {
            for (index, number) in (0..entry.span.count()).zip(entry.span.numbers()) {
                let Some(group) = entry.marks.group(index) else {
                    continue;
                };
                let Some(slot) = usize::try_from(number)
                    .ok()
                    .and_then(|number| by_method.get_mut(number))
                else {
                    continue;
                };
                *slot = slot.with(group);
                named = named.with(group);
            }
        }
        Self { by_method, named }
    }

    /// The map for class `class`, built on first use.
    pub fn for_class(class: ClassId) -> &'static Self {
        static THREE_D: OnceLock<GroupMap> = OnceLock::new();
        static COMPUTE: OnceLock<GroupMap> = OnceLock::new();
        static OTHER: OnceLock<GroupMap> = OnceLock::new();
        let cell = match class.0 {
            clc797::CLASS => &THREE_D,
            clc7c0::CLASS => &COMPUTE,
            _ => &OTHER,
        };
        cell.get_or_init(|| {
            Self::build(group_table(class), ClassRules::for_class(class).registers())
        })
    }

    /// The groups method `method` marks; none outside the map.
    pub fn groups(&self, method: u32) -> GroupSet {
        usize::try_from(method)
            .ok()
            .and_then(|index| self.by_method.get(index))
            .copied()
            .unwrap_or(GroupSet::EMPTY)
    }

    /// Every group the table names.
    pub fn named(&self) -> GroupSet {
        self.named
    }
}

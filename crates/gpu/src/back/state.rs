//! WIP
mod group;
mod mirror;
mod table;
mod views;

#[cfg(test)]
mod tests;

pub use group::{
    BIND_GROUPS, BLEND_TARGETS, COLOR_TARGETS, CONSTANT_BUFFER_SLOTS, GROUP_COUNT, Group,
    GroupKind, GroupSet, PIPELINES, SCISSORS, VERTEX_ATTRIBUTES, VERTEX_ATTRIBUTES_PER_HALF,
    VERTEX_STREAM_SIZES, VERTEX_STREAMS, VERTEX_STREAMS_PER_HALF, VIEWPORTS,
};
pub use mirror::{ConstantBufferBinding, Mirror, Reg, RunPositions, StateError};
pub use table::{GROUPS_3D, GROUPS_COMPUTE, GroupEntry, GroupMap, Marks, group_table};
pub use views::{
    BindGroup, Blend, BlendPerTarget, ColorTarget, Compute, CtSelect, DepthStencil, IndexBuffer,
    Pipeline, Rasterizer, Scissor, TexPools, ThreeD, VertexAttribute, VertexStream, Viewport, Zt,
    arrays,
};

//! Bind group and pipeline layouts from each stage's `ShaderInterface` (docs/specs/render-backend.md,
//! "Shader emission": the wgpu backend "builds bind group layouts and pipeline layouts from
//! `ShaderInterface`").
//!
//! Every pipeline gets an explicit layout. wgpu could derive one from the shaders, but not for a
//! binding array ("mapping of `ResourceBinding` ... is missing", `docs/specs/shader-pipeline.md`,
//! the E1 table), and one rule for every pipeline keeps the layouts the GPU crate's bindings
//! (slice R7) are made against in one place.
//!
//! # From the interface, with one fact from the module
//!
//! Each resource row becomes one layout entry at its group and binding, visible to every stage
//! whose interface lists it. Its type follows the row's kind:
//!
//! | `ResourceKind` | wgpu binding type |
//! |---|---|
//! | `UniformBuffer` | uniform buffer, no dynamic offset, no minimum size |
//! | `StorageBuffer { writable }` | storage buffer, read-only unless `writable` |
//! | `SampledTexture { dimension, sample, multisampled }` | texture of that view dimension, sample type (`Float { filterable }`, `Depth`, `Sint`, `Uint`) and multisampling |
//! | `StorageTexture { dimension, format, writable }` | storage texture of that dimension and format: read-only unless `writable`; write-only when the module declares the global store-only, read-write otherwise ([`super::scan`]) |
//! | `Sampler { comparison }` | comparison sampler, or a filtering sampler unless one of its stages samples a texture that cannot be filtered with it (below) |
//!
//! Each row's array count ([`starquake_render::ResourceBinding::count`]) is the entry's `count`:
//! the interface carries it, so the layout does not read it from the module (render-backend,
//! "Shader emission"; decided by the owner, 2026-11-19, ruling 20). The pipeline checks that the
//! module agrees before planning (`crate::pipeline::agreement`, which refuses a binding array
//! whose size is not a constant with [`UNSIZED_BINDING_ARRAY`]). A 1D-array texture binding is
//! refused ([`ONE_D_ARRAY`]: wgpu has
//! no such view, as `crate::convert::view_dimension` says), and so is a storage texture whose
//! format has no wgpu form ([`STORAGE_FORMAT`]), which the shared checks refuse first on a device
//! built from wgpu's own format support.
//!
//! Two stages that list one group and binding must agree on its kind and its array size, or the
//! pipeline is refused ([`STAGE_CONFLICT`]). A storage texture both stages write is write-only
//! only when both declare it so. A group no stage uses is left out of the pipeline layout (wgpu
//! takes `None` there).
//!
//! # Filtering samplers
//!
//! wgpu-core refuses a pipeline whose shader samples a non-filterable float texture, or an
//! integer texture, through a sampler whose layout entry is `Filtering` (`FilteringError`,
//! wgpu-core 30.0.2 `src/validation.rs`, checked only for a layout the caller provides). So a
//! non-comparison sampler's entry is `NonFiltering` when an entry point of the pipeline samples,
//! through it, a texture whose interface row is `Float { filterable: false }`, `Sint` or `Uint`,
//! and `Filtering` otherwise. Which sampler goes with which texture is naga's analysis
//! (`FunctionInfo::sampling_set`), the one wgpu-core reads; [`non_filtering_samplers`] runs
//! naga's validator over the module on the compile worker to get it, since the `ModuleInfo` a
//! `ShaderDesc` carries cannot be checked against its module without the risk of indexing past
//! its end. That costs one more validation of the module for a pipeline whose interface has
//! such a texture and a non-comparison sampler, and none otherwise.
//!
//! # Not checked here
//!
//! - A binding array's size against wgpu's `max_binding_array_elements_per_shader_stage`, and an
//!   array's elements against the per-stage counts (`validate::validate_shader` counts one per
//!   row). wgpu checks both when it builds the layout, and the pipeline then fails.
//! - A storage texture format's read-write support: a read-write entry in a format the device
//!   cannot read and write fails the pipeline the same way.

use super::scan::ModuleFacts;
use crate::convert::view_dimension;
use crate::format::texture_format;
use starquake_render::RenderError;
use starquake_render::shader::{ResourceKind, SampleKind, ShaderInterface};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use wgpu::naga;

/// Two stages of a pipeline bind one group and binding as different resources.
pub const STAGE_CONFLICT: &str =
    "two stages of a pipeline bind one group and binding as different resources";
/// A binding array whose size is not a constant, in a shader's module (refused by the pipeline's
/// interface check, `crate::pipeline::agreement`).
pub const UNSIZED_BINDING_ARRAY: &str =
    "the wgpu backend needs a binding array's size to be a constant";
/// A 1D-array texture binding.
pub const ONE_D_ARRAY: &str = "wgpu has no 1D array texture bindings";
/// A storage texture in a format with no wgpu form.
pub const STORAGE_FORMAT: &str = "a storage texture's format has no wgpu form";

/// One stage's part of a pipeline: where it runs, its interface and what its module says (the
/// layout reads only each storage texture's access from it).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stage<'a> {
    /// The wgpu stage.
    pub(crate) visibility: wgpu::ShaderStages,
    /// The resource table.
    pub(crate) interface: &'a ShaderInterface,
    /// The module's facts.
    pub(crate) facts: &'a ModuleFacts,
}

/// One merged row.
#[derive(Clone, Copy, Debug)]
struct Row {
    kind: ResourceKind,
    count: Option<NonZeroU32>,
    visibility: wgpu::ShaderStages,
    write_only: bool,
}

/// A pipeline's bind group layouts, as entries per group (module docs); an empty group is one
/// no stage uses.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayoutPlan {
    /// The entries of groups 0 to the highest group used, in binding order.
    pub(crate) groups: Vec<Vec<wgpu::BindGroupLayoutEntry>>,
}

/// The wgpu binding type for `kind` (module docs' table).
fn binding_type(kind: ResourceKind, write_only: bool) -> Result<wgpu::BindingType, RenderError> {
    Ok(match kind {
        ResourceKind::UniformBuffer => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        ResourceKind::StorageBuffer { writable } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage {
                read_only: !writable,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        ResourceKind::SampledTexture {
            dimension,
            sample,
            multisampled,
        } => wgpu::BindingType::Texture {
            sample_type: match sample {
                SampleKind::Float { filterable } => wgpu::TextureSampleType::Float { filterable },
                SampleKind::Depth => wgpu::TextureSampleType::Depth,
                SampleKind::Sint => wgpu::TextureSampleType::Sint,
                SampleKind::Uint => wgpu::TextureSampleType::Uint,
            },
            view_dimension: view_dimension(dimension)
                .ok_or(RenderError::Unsupported(ONE_D_ARRAY))?,
            multisampled,
        },
        ResourceKind::StorageTexture {
            dimension,
            format,
            writable,
        } => wgpu::BindingType::StorageTexture {
            access: match (writable, write_only) {
                (false, _) => wgpu::StorageTextureAccess::ReadOnly,
                (true, true) => wgpu::StorageTextureAccess::WriteOnly,
                (true, false) => wgpu::StorageTextureAccess::ReadWrite,
            },
            format: texture_format(format).ok_or(RenderError::Unsupported(STORAGE_FORMAT))?,
            view_dimension: view_dimension(dimension)
                .ok_or(RenderError::Unsupported(ONE_D_ARRAY))?,
        },
        ResourceKind::Sampler { comparison } => wgpu::BindingType::Sampler(if comparison {
            wgpu::SamplerBindingType::Comparison
        } else {
            wgpu::SamplerBindingType::Filtering
        }),
    })
}

/// Merges the stages' interfaces into layouts (module docs). Rows are merged stage by stage, each
/// in table order, and a conflict between stages ([`STAGE_CONFLICT`]) is refused as it is
/// found; then each merged row's binding type is made in group and binding order, and the first
/// that cannot be ([`ONE_D_ARRAY`], [`STORAGE_FORMAT`]) is refused. The caller has checked each
/// stage's interface against its module (`crate::pipeline::agreement`).
pub(crate) fn plan(stages: &[Stage<'_>]) -> Result<LayoutPlan, RenderError> {
    let mut rows: BTreeMap<(u32, u32), Row> = BTreeMap::new();
    for stage in stages {
        for r in &stage.interface.resources {
            let slot = (r.group, r.binding);
            let facts = stage.facts.bindings.get(&slot);
            let count = r.count;
            let write_only = facts.is_some_and(|f| f.write_only);
            match rows.get_mut(&slot) {
                Some(row) if row.kind == r.kind && row.count == count => {
                    row.visibility |= stage.visibility;
                    row.write_only &= write_only;
                }
                Some(_) => return Err(RenderError::Invalid(STAGE_CONFLICT)),
                None => {
                    rows.insert(
                        slot,
                        Row {
                            kind: r.kind,
                            count,
                            visibility: stage.visibility,
                            write_only,
                        },
                    );
                }
            }
        }
    }
    let used = rows
        .last_key_value()
        .map_or(0, |(&(group, _), _)| group.saturating_add(1));
    let mut groups = vec![Vec::new(); usize::try_from(used).unwrap_or(0)];
    for ((group, binding), row) in rows {
        let entry = wgpu::BindGroupLayoutEntry {
            binding,
            visibility: row.visibility,
            ty: binding_type(row.kind, row.write_only)?,
            count: row.count,
        };
        if let Some(entries) = usize::try_from(group).ok().and_then(|g| groups.get_mut(g)) {
            entries.push(entry);
        }
    }
    Ok(LayoutPlan { groups })
}

/// Whether a texture row cannot be sampled through a filtering sampler (module docs).
const fn needs_non_filtering(kind: ResourceKind) -> bool {
    matches!(
        kind,
        ResourceKind::SampledTexture {
            sample: SampleKind::Float { filterable: false } | SampleKind::Sint | SampleKind::Uint,
            ..
        }
    )
}

/// Whether `interface` has both a texture that needs a non-filtering sampler and a
/// non-comparison sampler, the only case where [`non_filtering_samplers`] can find anything.
pub(crate) fn may_need_non_filtering(interface: &ShaderInterface) -> bool {
    let rows = &interface.resources;
    rows.iter().any(|r| needs_non_filtering(r.kind))
        && rows
            .iter()
            .any(|r| r.kind == ResourceKind::Sampler { comparison: false })
}

/// The groups and bindings of the samplers through which the entry point at `entry` samples a
/// texture that `interface` says cannot be filtered (module docs). `Err` carries naga's
/// validation error in words.
pub(crate) fn non_filtering_samplers(
    module: &naga::Module,
    entry: usize,
    interface: &ShaderInterface,
) -> Result<BTreeSet<(u32, u32)>, String> {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| e.into_inner().to_string())?;
    if entry >= module.entry_points.len() {
        return Ok(BTreeSet::new());
    }
    let slot = |global| {
        let binding = module
            .global_variables
            .try_get(global)
            .ok()?
            .binding
            .as_ref()?;
        Some((binding.group, binding.binding))
    };
    let unfilterable: BTreeSet<(u32, u32)> = interface
        .resources
        .iter()
        .filter(|r| needs_non_filtering(r.kind))
        .map(|r| (r.group, r.binding))
        .collect();
    Ok(info
        .get_entry_point(entry)
        .sampling_set
        .iter()
        .filter_map(|key| {
            let texture = slot(key.image)?;
            let sampler = slot(key.sampler)?;
            unfilterable.contains(&texture).then_some(sampler)
        })
        .collect())
}

/// `plan` with each filtering sampler entry at a slot in `slots` turned non-filtering.
pub(crate) fn with_non_filtering(mut plan: LayoutPlan, slots: &BTreeSet<(u32, u32)>) -> LayoutPlan {
    for (group, entries) in plan.groups.iter_mut().enumerate() {
        for entry in entries {
            let slot = (u32::try_from(group).unwrap_or(u32::MAX), entry.binding);
            if entry.ty == wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
                && slots.contains(&slot)
            {
                entry.ty = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering);
            }
        }
    }
    plan
}

#[cfg(test)]
mod tests;

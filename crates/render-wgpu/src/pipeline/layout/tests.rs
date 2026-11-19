use super::*;
use crate::pipeline::scan::{BindingFacts, scan};
use starquake_render::PixelFormat;
use starquake_render::resource::ViewDimension;
use starquake_render::shader::ResourceBinding;
use wgpu::ShaderStages as S;

fn row(group: u32, binding: u32, kind: ResourceKind) -> ResourceBinding {
    ResourceBinding {
        group,
        binding,
        kind,
    }
}

fn interface(resources: Vec<ResourceBinding>) -> ShaderInterface {
    ShaderInterface {
        resources,
        ..ShaderInterface::default()
    }
}

fn facts(entries: &[((u32, u32), ArrayCount, bool)]) -> ModuleFacts {
    ModuleFacts {
        subgroups: false,
        nesting: 0,
        bindings: entries
            .iter()
            .map(|&(slot, count, write_only)| {
                let size = None;
                (
                    slot,
                    BindingFacts {
                        count,
                        write_only,
                        size,
                    },
                )
            })
            .collect(),
    }
}

fn n(v: u32) -> NonZeroU32 {
    NonZeroU32::new(v).expect("non-zero")
}

const TEX2D: ResourceKind = ResourceKind::SampledTexture {
    dimension: ViewDimension::D2,
    sample: SampleKind::Float { filterable: true },
    multisampled: false,
};

/// One row of each kind, each mapped by hand to the wgpu type the module docs' table gives.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one row per kind, each written out by hand"
)]
fn every_kind_gets_its_binding_type() {
    use wgpu::{BindingType as B, BufferBindingType as Buf, TextureSampleType as T};
    let buffer = |ty| B::Buffer {
        ty,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let texture = |sample, dimension, multisampled| ResourceKind::SampledTexture {
        dimension,
        sample,
        multisampled,
    };
    let storage = |format, writable| ResourceKind::StorageTexture {
        dimension: ViewDimension::D2Array,
        format,
        writable,
    };
    let cases = [
        (ResourceKind::UniformBuffer, false, buffer(Buf::Uniform)),
        (
            ResourceKind::StorageBuffer { writable: false },
            false,
            buffer(Buf::Storage { read_only: true }),
        ),
        (
            ResourceKind::StorageBuffer { writable: true },
            false,
            buffer(Buf::Storage { read_only: false }),
        ),
        (
            texture(
                SampleKind::Float { filterable: true },
                ViewDimension::D2,
                false,
            ),
            false,
            B::Texture {
                sample_type: T::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        ),
        (
            texture(
                SampleKind::Float { filterable: false },
                ViewDimension::Cube,
                false,
            ),
            false,
            B::Texture {
                sample_type: T::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::Cube,
                multisampled: false,
            },
        ),
        (
            texture(SampleKind::Depth, ViewDimension::D2Array, false),
            false,
            B::Texture {
                sample_type: T::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
        ),
        (
            texture(SampleKind::Sint, ViewDimension::D3, false),
            false,
            B::Texture {
                sample_type: T::Sint,
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
        ),
        (
            texture(SampleKind::Uint, ViewDimension::D2, true),
            false,
            B::Texture {
                sample_type: T::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: true,
            },
        ),
        (
            texture(SampleKind::Uint, ViewDimension::CubeArray, false),
            false,
            B::Texture {
                sample_type: T::Uint,
                view_dimension: wgpu::TextureViewDimension::CubeArray,
                multisampled: false,
            },
        ),
        (
            texture(SampleKind::Uint, ViewDimension::D1, false),
            false,
            B::Texture {
                sample_type: T::Uint,
                view_dimension: wgpu::TextureViewDimension::D1,
                multisampled: false,
            },
        ),
        (
            storage(PixelFormat::R32Float, false),
            true,
            B::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadOnly,
                format: wgpu::TextureFormat::R32Float,
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
        ),
        (
            storage(PixelFormat::A8B8G8R8Unorm, true),
            true,
            B::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
        ),
        (
            storage(PixelFormat::A8B8G8R8Unorm, true),
            false,
            B::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
        ),
        (
            ResourceKind::Sampler { comparison: false },
            false,
            B::Sampler(wgpu::SamplerBindingType::Filtering),
        ),
        (
            ResourceKind::Sampler { comparison: true },
            false,
            B::Sampler(wgpu::SamplerBindingType::Comparison),
        ),
    ];
    for (kind, write_only, expected) in cases {
        let i = interface(vec![row(0, 0, kind)]);
        let f = facts(&[((0, 0), ArrayCount::Single, write_only)]);
        let plan = plan(&[Stage {
            visibility: S::COMPUTE,
            interface: &i,
            facts: &f,
        }])
        .expect("the row is accepted");
        assert_eq!(
            plan.groups,
            vec![vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: S::COMPUTE,
                ty: expected,
                count: None,
            }]],
            "{kind:?}, write-only {write_only}"
        );
    }
}

#[test]
fn bindings_wgpu_cannot_make_are_refused() {
    let refused = [
        (
            ResourceKind::SampledTexture {
                dimension: ViewDimension::D1Array,
                sample: SampleKind::Float { filterable: true },
                multisampled: false,
            },
            ONE_D_ARRAY,
        ),
        (
            ResourceKind::StorageTexture {
                dimension: ViewDimension::D1Array,
                format: PixelFormat::R32Float,
                writable: false,
            },
            ONE_D_ARRAY,
        ),
        (
            ResourceKind::StorageTexture {
                dimension: ViewDimension::D2,
                format: PixelFormat::R5G6B5Unorm,
                writable: false,
            },
            STORAGE_FORMAT,
        ),
    ];
    for (kind, message) in refused {
        let i = interface(vec![row(0, 0, kind)]);
        let f = ModuleFacts::default();
        assert_eq!(
            plan(&[Stage {
                visibility: S::COMPUTE,
                interface: &i,
                facts: &f,
            }]),
            Err(RenderError::Unsupported(message)),
            "{kind:?}"
        );
    }
}

#[test]
fn binding_arrays_get_their_size_and_unsized_ones_are_refused() {
    let i = interface(vec![row(0, 0, TEX2D), row(0, 1, TEX2D)]);
    let f = facts(&[
        ((0, 0), ArrayCount::Sized(n(4)), false),
        ((0, 1), ArrayCount::Single, false),
    ]);
    let stage = Stage {
        visibility: S::FRAGMENT,
        interface: &i,
        facts: &f,
    };
    let plan = plan(&[stage]).expect("accepted");
    assert_eq!(plan.groups[0][0].count, Some(n(4)));
    assert_eq!(plan.groups[0][1].count, None);

    let unsized_ = facts(&[((0, 1), ArrayCount::Unsized, false)]);
    assert_eq!(
        super::plan(&[Stage {
            facts: &unsized_,
            ..stage
        }]),
        Err(RenderError::Unsupported(UNSIZED_BINDING_ARRAY))
    );
    // A row the module does not declare is a single binding.
    let none = ModuleFacts::default();
    let plan = super::plan(&[Stage {
        facts: &none,
        ..stage
    }])
    .expect("accepted");
    assert_eq!(plan.groups[0][0].count, None);
}

#[test]
fn stages_merge_by_group_and_binding() {
    let vertex = interface(vec![
        row(0, 0, ResourceKind::UniformBuffer),
        row(2, 1, TEX2D),
    ]);
    let fragment = interface(vec![
        row(0, 0, ResourceKind::UniformBuffer),
        row(0, 3, ResourceKind::Sampler { comparison: false }),
    ]);
    let none = ModuleFacts::default();
    let plan = plan(&[
        Stage {
            visibility: S::VERTEX,
            interface: &vertex,
            facts: &none,
        },
        Stage {
            visibility: S::FRAGMENT,
            interface: &fragment,
            facts: &none,
        },
    ])
    .expect("accepted");
    let shape: Vec<Vec<(u32, S)>> = plan
        .groups
        .iter()
        .map(|g| g.iter().map(|e| (e.binding, e.visibility)).collect())
        .collect();
    assert_eq!(
        shape,
        vec![
            vec![(0, S::VERTEX | S::FRAGMENT), (3, S::FRAGMENT)],
            vec![],
            vec![(1, S::VERTEX)],
        ],
        "group 1 is unused; entries are in binding order"
    );
}

#[test]
fn stages_that_disagree_are_refused() {
    let none = ModuleFacts::default();
    let a = interface(vec![row(0, 0, TEX2D)]);
    let b = interface(vec![row(0, 0, ResourceKind::UniformBuffer)]);
    let stages = |second: &ShaderInterface, second_facts: &ModuleFacts| {
        plan(&[
            Stage {
                visibility: S::VERTEX,
                interface: &a,
                facts: &none,
            },
            Stage {
                visibility: S::FRAGMENT,
                interface: second,
                facts: second_facts,
            },
        ])
    };
    assert_eq!(stages(&b, &none), Err(RenderError::Invalid(STAGE_CONFLICT)));
    let arrayed = facts(&[((0, 0), ArrayCount::Sized(n(2)), false)]);
    assert_eq!(
        stages(&a, &arrayed),
        Err(RenderError::Invalid(STAGE_CONFLICT)),
        "same kind, different array size"
    );
    assert!(stages(&a, &none).is_ok(), "the same row twice");
}

#[test]
fn a_storage_texture_is_write_only_only_when_every_stage_only_writes() {
    let kind = ResourceKind::StorageTexture {
        dimension: ViewDimension::D2,
        format: PixelFormat::A8B8G8R8Unorm,
        writable: true,
    };
    let i = interface(vec![row(0, 0, kind)]);
    let w = facts(&[((0, 0), ArrayCount::Single, true)]);
    let rw = facts(&[((0, 0), ArrayCount::Single, false)]);
    let access = |first: &ModuleFacts, second: &ModuleFacts| {
        let plan = plan(&[
            Stage {
                visibility: S::VERTEX,
                interface: &i,
                facts: first,
            },
            Stage {
                visibility: S::FRAGMENT,
                interface: &i,
                facts: second,
            },
        ])
        .expect("accepted");
        match plan.groups[0][0].ty {
            wgpu::BindingType::StorageTexture { access, .. } => Ok(access),
            other => Err(other),
        }
        .expect("a storage texture")
    };
    assert_eq!(access(&w, &w), wgpu::StorageTextureAccess::WriteOnly);
    assert_eq!(access(&w, &rw), wgpu::StorageTextureAccess::ReadWrite);
    assert_eq!(access(&rw, &w), wgpu::StorageTextureAccess::ReadWrite);
    assert_eq!(access(&rw, &rw), wgpu::StorageTextureAccess::ReadWrite);
}

#[test]
fn the_first_refusal_in_stage_and_table_order_wins() {
    // Stage one has an unsized array; stage two conflicts with stage one. The array is found
    // first.
    let a = interface(vec![row(0, 0, TEX2D), row(0, 1, TEX2D)]);
    let unsized_ = facts(&[((0, 1), ArrayCount::Unsized, false)]);
    let b = interface(vec![row(0, 0, ResourceKind::UniformBuffer)]);
    let none = ModuleFacts::default();
    assert_eq!(
        plan(&[
            Stage {
                visibility: S::VERTEX,
                interface: &a,
                facts: &unsized_,
            },
            Stage {
                visibility: S::FRAGMENT,
                interface: &b,
                facts: &none,
            },
        ]),
        Err(RenderError::Unsupported(UNSIZED_BINDING_ARRAY))
    );
    // Swapped: the conflict is found while the second stage is merged, before the array row.
    assert_eq!(
        plan(&[
            Stage {
                visibility: S::FRAGMENT,
                interface: &b,
                facts: &none,
            },
            Stage {
                visibility: S::VERTEX,
                interface: &a,
                facts: &unsized_,
            },
        ]),
        Err(RenderError::Invalid(STAGE_CONFLICT))
    );
}

#[test]
fn an_interface_with_no_resources_has_no_groups() {
    let i = ShaderInterface::default();
    let f = ModuleFacts::default();
    assert_eq!(
        plan(&[Stage {
            visibility: S::VERTEX,
            interface: &i,
            facts: &f,
        }]),
        Ok(LayoutPlan { groups: vec![] })
    );
}

const SAMPLING: &str = "
    @group(0) @binding(0) var a: texture_2d<f32>;
    @group(0) @binding(1) var b: texture_2d<f32>;
    @group(0) @binding(2) var s: sampler;
    @group(0) @binding(3) var t: sampler;
    @fragment fn main() -> @location(0) vec4<f32> {
        return textureSample(a, s, vec2(0.5)) + textureSample(b, t, vec2(0.5));
    }";

fn sampling_interface(a_filterable: bool, b_filterable: bool) -> ShaderInterface {
    let float = |filterable| ResourceKind::SampledTexture {
        dimension: ViewDimension::D2,
        sample: SampleKind::Float { filterable },
        multisampled: false,
    };
    interface(vec![
        row(0, 0, float(a_filterable)),
        row(0, 1, float(b_filterable)),
        row(0, 2, ResourceKind::Sampler { comparison: false }),
        row(0, 3, ResourceKind::Sampler { comparison: false }),
    ])
}

#[test]
fn a_sampler_used_with_an_unfilterable_texture_is_non_filtering() {
    let module = naga::front::wgsl::parse_str(SAMPLING).expect("parses");
    for (a, b, expected) in [
        (true, true, vec![]),
        (false, true, vec![(0, 2)]),
        (true, false, vec![(0, 3)]),
        (false, false, vec![(0, 2), (0, 3)]),
    ] {
        let found = non_filtering_samplers(&module, 0, &sampling_interface(a, b))
            .expect("the module validates");
        assert_eq!(
            found.into_iter().collect::<Vec<_>>(),
            expected,
            "a filterable {a}, b filterable {b}"
        );
    }
    // An entry index past the module's entry points finds nothing (and does not index past
    // the validator's table).
    assert_eq!(
        non_filtering_samplers(&module, 1, &sampling_interface(false, false)),
        Ok(BTreeSet::new())
    );
}

#[test]
fn an_invalid_module_reports_naga_s_error() {
    let mut module = naga::front::wgsl::parse_str(SAMPLING).expect("parses");
    // A return of nothing from a function that returns a value.
    module.entry_points[0].function.body =
        naga::Block::from_vec(vec![naga::Statement::Return { value: None }]);
    let error = non_filtering_samplers(&module, 0, &sampling_interface(false, false))
        .expect_err("the module is invalid");
    assert_ne!(error, "");
}

#[test]
fn non_filtering_is_needed_only_with_both_kinds_of_row() {
    let unfilterable = ResourceKind::SampledTexture {
        dimension: ViewDimension::D2,
        sample: SampleKind::Float { filterable: false },
        multisampled: false,
    };
    let int = ResourceKind::SampledTexture {
        dimension: ViewDimension::D2,
        sample: SampleKind::Sint,
        multisampled: false,
    };
    let uint = ResourceKind::SampledTexture {
        dimension: ViewDimension::D2,
        sample: SampleKind::Uint,
        multisampled: false,
    };
    let depth = ResourceKind::SampledTexture {
        dimension: ViewDimension::D2,
        sample: SampleKind::Depth,
        multisampled: false,
    };
    let sampler = ResourceKind::Sampler { comparison: false };
    let shadow = ResourceKind::Sampler { comparison: true };
    for (rows, expected) in [
        (vec![unfilterable, sampler], true),
        (vec![int, sampler], true),
        (vec![uint, sampler], true),
        (vec![TEX2D, sampler], false),
        (vec![depth, sampler], false),
        (vec![unfilterable, shadow], false),
        (vec![unfilterable], false),
        (vec![sampler], false),
    ] {
        let i = interface(
            rows.iter()
                .enumerate()
                .map(|(b, &k)| row(0, u32::try_from(b).expect("small"), k))
                .collect(),
        );
        assert_eq!(may_need_non_filtering(&i), expected, "{rows:?}");
    }
}

#[test]
fn only_filtering_samplers_at_the_slots_change() {
    use wgpu::{BindingType::Sampler, SamplerBindingType as T};
    let i = interface(vec![
        row(0, 0, ResourceKind::Sampler { comparison: false }),
        row(0, 1, ResourceKind::Sampler { comparison: false }),
        row(0, 2, ResourceKind::Sampler { comparison: true }),
        row(1, 0, ResourceKind::Sampler { comparison: false }),
    ]);
    let f = ModuleFacts::default();
    let base = plan(&[Stage {
        visibility: S::FRAGMENT,
        interface: &i,
        facts: &f,
    }])
    .expect("accepted");
    let changed = with_non_filtering(base, &BTreeSet::from([(0, 1), (0, 2), (1, 0)]));
    let types: Vec<Vec<wgpu::BindingType>> = changed
        .groups
        .iter()
        .map(|g| g.iter().map(|e| e.ty).collect())
        .collect();
    assert_eq!(
        types,
        vec![
            vec![
                Sampler(T::Filtering),
                Sampler(T::NonFiltering),
                Sampler(T::Comparison)
            ],
            vec![Sampler(T::NonFiltering)],
        ]
    );
}

/// The scan and the plan together, on a hand-written module with a binding array: the plan's
/// count is the module's size.
#[test]
fn a_module_s_binding_array_reaches_the_layout() {
    let module = naga::front::wgsl::parse_str(
        "enable wgpu_binding_array;
         @group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 4>;
         @group(0) @binding(1) var s: sampler;
         @fragment fn main(@location(0) @interpolate(flat) i: u32) -> @location(0) vec4<f32> {
             return textureSample(textures[i], s, vec2(0.5));
         }",
    )
    .expect("parses");
    let f = scan(&module);
    let i = interface(vec![
        row(0, 0, TEX2D),
        row(0, 1, ResourceKind::Sampler { comparison: false }),
    ]);
    let plan = plan(&[Stage {
        visibility: S::FRAGMENT,
        interface: &i,
        facts: &f,
    }])
    .expect("accepted");
    assert_eq!(plan.groups[0][0].count, Some(n(4)));
    assert_eq!(plan.groups[0][1].count, None);
}

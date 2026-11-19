//! `Caps` from made-up adapters: every field from its source, with distinct values.

use super::{
    ByteLimits, COPY_BUFFER_ALIGNMENT, FormatSupport, WANTED_FEATURES, WGPU_MAX_ANISOTROPY,
    WGPU_VIEWPORTS, caps, device_features, format_features, sample_counts,
};
use starquake_render::caps::{DynamicState, StageSupport};
use starquake_render::{Features, FormatFeatures, PixelFormat, SampleCounts};
use wgpu::{TextureFormatFeatureFlags as Flags, TextureFormatFeatures, TextureUsages as Usages};

/// A wgpu limit set where every field the caps read has its own value.
fn distinct_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_buffer_size: 1_000_003,
        max_texture_dimension_1d: 101,
        max_texture_dimension_2d: 102,
        max_texture_dimension_3d: 103,
        max_texture_array_layers: 104,
        max_vertex_attributes: 105,
        max_vertex_buffers: 106,
        max_vertex_buffer_array_stride: 107,
        max_color_attachments: 108,
        max_bind_groups: 109,
        max_bindings_per_bind_group: 110,
        max_compute_workgroups_per_dimension: 111,
        max_uniform_buffers_per_shader_stage: 112,
        max_storage_buffers_per_shader_stage: 113,
        max_sampled_textures_per_shader_stage: 114,
        max_storage_textures_per_shader_stage: 115,
        max_samplers_per_shader_stage: 116,
        ..wgpu::Limits::default()
    }
}

fn no_formats() -> FormatSupport {
    FormatSupport::new(wgpu::Features::empty(), |_| TextureFormatFeatures {
        allowed_usages: Usages::empty(),
        flags: Flags::empty(),
    })
}

const ROOMY: ByteLimits = ByteLimits {
    memory: u64::MAX,
    staging: u64::MAX,
};

#[test]
fn every_limit_comes_from_its_wgpu_limit() {
    let c = caps(
        wgpu::Features::empty(),
        &distinct_limits(),
        wgpu::DownlevelFlags::empty(),
        &no_formats(),
        ROOMY,
    );
    let l = c.limits;
    assert_eq!(
        l.max_buffer_size, 1_000_000,
        "rounded down to the copy alignment"
    );
    assert_eq!(l.max_staging_size, 1_000_000, "capped at the buffer size");
    assert_eq!(
        [
            l.max_texture_dimension_1d,
            l.max_texture_dimension_2d,
            l.max_texture_dimension_3d,
            l.max_texture_array_layers,
            l.max_vertex_attributes,
            l.max_vertex_buffers,
            l.max_vertex_buffer_stride,
            l.max_color_attachments,
            l.max_bind_groups,
            l.max_bindings_per_bind_group,
            l.max_compute_workgroups_per_dimension,
            l.max_uniform_buffers_per_stage,
            l.max_storage_buffers_per_stage,
            l.max_sampled_textures_per_stage,
            l.max_storage_textures_per_stage,
            l.max_samplers_per_stage,
        ],
        [
            101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116
        ]
    );
    assert_eq!(l.max_viewports, WGPU_VIEWPORTS);
    assert_eq!(l.max_sampler_anisotropy, WGPU_MAX_ANISOTROPY);
    // The values wgpu fixes, written out from its sources (module docs).
    assert_eq!((WGPU_VIEWPORTS, WGPU_MAX_ANISOTROPY), (1, 16));
    assert_eq!(COPY_BUFFER_ALIGNMENT, 4);
    assert!(!c.view_swizzle);
    assert_eq!(c.stages, StageSupport::EMPTY);
    assert_eq!(
        c.dynamic_state,
        DynamicState::VIEWPORT
            .union(DynamicState::SCISSOR)
            .union(DynamicState::BLEND_CONSTANT)
            .union(DynamicState::STENCIL_REFERENCE)
    );
}

#[test]
fn byte_limits_cap_the_buffer_and_staging_sizes_at_their_edges() {
    let at = |wgpu_max: u64, memory: u64, staging: u64| {
        let limits = wgpu::Limits {
            max_buffer_size: wgpu_max,
            ..distinct_limits()
        };
        let l = caps(
            wgpu::Features::empty(),
            &limits,
            wgpu::DownlevelFlags::empty(),
            &no_formats(),
            ByteLimits { memory, staging },
        )
        .limits;
        (l.max_buffer_size, l.max_staging_size)
    };
    // wgpu's limit: a multiple of four is kept, one more is rounded back down.
    assert_eq!(at(4096, u64::MAX, u64::MAX), (4096, 4096));
    assert_eq!(at(4097, u64::MAX, u64::MAX), (4096, 4096));
    assert_eq!(at(4099, u64::MAX, u64::MAX), (4096, 4096));
    assert_eq!(at(4100, u64::MAX, u64::MAX), (4100, 4100));
    // The memory limit, one below, at and one above the device's.
    assert_eq!(at(4096, 4095, u64::MAX), (4095, 4095));
    assert_eq!(at(4096, 4096, u64::MAX), (4096, 4096));
    assert_eq!(at(4096, 4097, u64::MAX), (4096, 4096));
    // The staging limit, one below, at and one above the buffer size.
    assert_eq!(at(4096, u64::MAX, 4095), (4096, 4095));
    assert_eq!(at(4096, u64::MAX, 4096), (4096, 4096));
    assert_eq!(at(4096, u64::MAX, 4097), (4096, 4096));
}

#[test]
fn each_feature_comes_from_its_own_source() {
    let cases = [
        (
            wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER,
            wgpu::DownlevelFlags::empty(),
            Features::CLAMP_TO_BORDER,
        ),
        (
            wgpu::Features::POLYGON_MODE_LINE,
            wgpu::DownlevelFlags::empty(),
            Features::POLYGON_MODE_LINE,
        ),
        (
            wgpu::Features::POLYGON_MODE_POINT,
            wgpu::DownlevelFlags::empty(),
            Features::POLYGON_MODE_POINT,
        ),
        (
            wgpu::Features::DEPTH_CLIP_CONTROL,
            wgpu::DownlevelFlags::empty(),
            Features::DEPTH_CLAMP,
        ),
        (
            wgpu::Features::DUAL_SOURCE_BLENDING,
            wgpu::DownlevelFlags::empty(),
            Features::DUAL_SOURCE_BLEND,
        ),
        (
            wgpu::Features::empty(),
            wgpu::DownlevelFlags::INDIRECT_EXECUTION,
            Features::MULTI_DRAW_INDIRECT,
        ),
    ];
    for (wgpu_feature, downlevel, ours) in cases {
        let c = caps(
            wgpu_feature,
            &distinct_limits(),
            downlevel,
            &no_formats(),
            ROOMY,
        );
        assert_eq!(c.features, ours, "{wgpu_feature:?} {downlevel:?}");
    }
    // Everything a device can offer still never reports what wgpu lacks.
    let c = caps(
        wgpu::Features::all(),
        &distinct_limits(),
        wgpu::DownlevelFlags::all(),
        &no_formats(),
        ROOMY,
    );
    for never in [
        Features::LOGIC_OP,
        Features::ROBUST_BUFFER_ACCESS,
        Features::ROBUST_IMAGE_ACCESS,
        Features::MIRROR_CLAMP_TO_EDGE,
        Features::SAMPLER_MIN_MAX_REDUCTION,
        Features::DEPTH_RANGE_MINUS_ONE_TO_ONE,
        Features::PRIMITIVE_RESTART_OFF,
        Features::UNTYPED_FRAGMENT_OUTPUTS,
    ] {
        assert!(!c.features.intersects(never), "{never:?}");
    }
}

#[test]
fn the_device_gets_only_wanted_features_the_adapter_has() {
    assert_eq!(device_features(wgpu::Features::all()), WANTED_FEATURES);
    assert_eq!(
        device_features(wgpu::Features::SHADER_F64 | wgpu::Features::TEXTURE_COMPRESSION_BC),
        wgpu::Features::TEXTURE_COMPRESSION_BC
    );
    assert_eq!(
        device_features(wgpu::Features::empty()),
        wgpu::Features::empty()
    );
}

#[test]
fn format_features_match_a_model_for_every_combination() {
    let usages = [
        Usages::TEXTURE_BINDING,
        Usages::RENDER_ATTACHMENT,
        Usages::STORAGE_BINDING,
        Usages::COPY_SRC,
        Usages::COPY_DST,
    ];
    let flags = [Flags::FILTERABLE, Flags::BLENDABLE];
    let bits = usages.len() + flags.len();
    for mask in 0_u32..(1 << bits) {
        let on = |i: usize| mask & (1 << i) != 0;
        let mut allowed = Usages::empty();
        for (i, u) in usages.iter().enumerate() {
            if on(i) {
                allowed |= *u;
            }
        }
        let mut f = Flags::empty();
        for (i, flag) in flags.iter().enumerate() {
            if on(usages.len() + i) {
                f |= *flag;
            }
        }
        let mut expected = FormatFeatures::EMPTY;
        let model = [
            (on(0), FormatFeatures::SAMPLE),
            (on(1), FormatFeatures::RENDER),
            (on(2), FormatFeatures::STORAGE),
            (on(3) && on(4), FormatFeatures::COPY),
            (on(5), FormatFeatures::FILTER),
            (on(6), FormatFeatures::BLEND),
        ];
        for (has, feature) in model {
            if has {
                expected = expected.union(feature);
            }
        }
        let got = format_features(&TextureFormatFeatures {
            allowed_usages: allowed,
            flags: f,
        });
        assert_eq!(got, expected, "mask {mask:#b}");
    }
}

#[test]
fn sample_counts_match_a_model_for_every_combination() {
    let flags = [
        (Flags::MULTISAMPLE_X2, SampleCounts::X2),
        (Flags::MULTISAMPLE_X4, SampleCounts::X4),
        (Flags::MULTISAMPLE_X8, SampleCounts::X8),
        (Flags::MULTISAMPLE_X16, SampleCounts::X16),
    ];
    for mask in 0_u32..16 {
        let mut f = Flags::FILTERABLE;
        let mut expected = SampleCounts::X1;
        for (i, (flag, count)) in flags.iter().enumerate() {
            if mask & (1 << i) != 0 {
                f |= *flag;
                expected = expected.union(*count);
            }
        }
        assert_eq!(sample_counts(f), expected, "mask {mask:#b}");
    }
}

#[test]
fn formats_follow_the_device_features_and_the_map() {
    let all_usages =
        Usages::TEXTURE_BINDING | Usages::RENDER_ATTACHMENT | Usages::COPY_SRC | Usages::COPY_DST;
    let describe = |format: wgpu::TextureFormat| TextureFormatFeatures {
        allowed_usages: all_usages,
        flags: if format == wgpu::TextureFormat::Rgba8Unorm {
            Flags::MULTISAMPLE_X4
        } else {
            Flags::empty()
        },
    };
    let full = FormatFeatures::SAMPLE
        .union(FormatFeatures::RENDER)
        .union(FormatFeatures::COPY);
    // Without the BC feature, a BC format can do nothing; with it, what wgpu says.
    let without = FormatSupport::new(wgpu::Features::empty(), describe);
    let with = FormatSupport::new(wgpu::Features::TEXTURE_COMPRESSION_BC, describe);
    assert_eq!(
        without.table.get(PixelFormat::Bc1RgbaUnorm),
        FormatFeatures::EMPTY
    );
    assert_eq!(with.table.get(PixelFormat::Bc1RgbaUnorm), full);
    // A format that needs no feature, and one with no wgpu form.
    assert_eq!(without.table.get(PixelFormat::R8Unorm), full);
    assert_eq!(
        with.table.get(PixelFormat::R5G6B5Unorm),
        FormatFeatures::EMPTY
    );
    // Sample counts per format, and `Caps` takes RGBA8's.
    assert_eq!(
        with.samples(PixelFormat::A8B8G8R8Unorm),
        SampleCounts::X1.union(SampleCounts::X4)
    );
    assert_eq!(with.samples(PixelFormat::R8Unorm), SampleCounts::X1);
    assert_eq!(with.samples(PixelFormat::R5G6B5Unorm), SampleCounts::X1);
    assert_eq!(with.usages(PixelFormat::Bc1RgbaUnorm), all_usages);
    assert_eq!(without.usages(PixelFormat::Bc1RgbaUnorm), Usages::empty());
    assert_eq!(with.usages(PixelFormat::R5G6B5Unorm), Usages::empty());
    let c = caps(
        wgpu::Features::empty(),
        &distinct_limits(),
        wgpu::DownlevelFlags::empty(),
        &with,
        ROOMY,
    );
    assert_eq!(c.sample_counts, SampleCounts::X1.union(SampleCounts::X4));
    assert_eq!(c.formats, with.table);
}

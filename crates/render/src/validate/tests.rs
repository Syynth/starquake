//! Validation tests: every limit at its edge on both sides, every rule both ways, and exact
//! errors.

use super::*;
use crate::caps::{
    DynamicState, FormatTable, GUEST_COLOR_TARGETS, GUEST_VERTEX_ATTRIBUTES, GUEST_VERTEX_BUFFERS,
    GUEST_VIEWPORTS, Limits, SampleCounts,
};
use crate::command::DynamicValues;
use crate::command::{BindingResource, IndexBufferBinding, IndexFormat, Viewport};
use crate::id::{BufferId, PipelineId, SamplerId, ShaderId, TextureId};
use crate::pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorWrites, CullMode, DepthBias,
    DepthStencilState, EmulationFlags, FrontFace, LogicOp, RasterState, Stages, TargetFormats,
    VertexAttribute, VertexBufferLayout, VertexComponents, VertexKind, VertexLayout,
    VertexStepMode,
};
use crate::resource::{
    CompareFunction, FilterMode, Origin3d, ReductionMode, Swizzle, SwizzleSource,
};
use crate::shader::{Interpolation, SampleKind, ScalarKind, ShaderInterface, ShaderSource};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

/// A device whose limits are all different, some below the guest's and some above, so a check
/// against the wrong limit shows.
fn caps() -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: 1 << 20,
            max_staging_size: 1 << 16,
            max_texture_dimension_1d: 4096,
            max_texture_dimension_2d: 2048,
            max_texture_dimension_3d: 512,
            max_texture_array_layers: 300,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: GUEST_VERTEX_ATTRIBUTES - 12,
            max_vertex_buffers: GUEST_VERTEX_BUFFERS + 8,
            max_vertex_buffer_stride: 2040,
            max_color_attachments: u32::try_from(GUEST_COLOR_TARGETS).expect("small") - 2,
            max_viewports: GUEST_VIEWPORTS + 4,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 9,
            max_compute_workgroups_per_dimension: 65_535,
        },
        features: Features::ALL,
        stages: StageSupport::ALL,
        dynamic_state: DynamicState::ALL,
        sample_counts: SampleCounts::X1 | SampleCounts::X4,
        view_swizzle: true,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

fn out_of_range(what: &'static str, value: u64, min: u64, max: u64) -> RenderError {
    RenderError::OutOfRange {
        what,
        value,
        min,
        max,
    }
}

fn not_below(what: &'static str, index: u64, count: u64) -> RenderError {
    RenderError::NotBelow { what, index, count }
}

fn past_end(what: &'static str, end: u64, limit: u64) -> RenderError {
    RenderError::PastEnd { what, end, limit }
}

fn n(v: u32) -> u64 {
    u64::from(v)
}

// ---------------------------------------------------------------------------------------------
// Buffers and staging

fn buffer(size: u64) -> BufferDesc {
    BufferDesc {
        size,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
    }
}

#[test]
fn buffer_size_edges() {
    let c = caps();
    let max = c.limits.max_buffer_size;
    assert_eq!(validate_buffer(&c, &buffer(1)), Ok(()));
    assert_eq!(validate_buffer(&c, &buffer(max)), Ok(()));
    assert_eq!(
        validate_buffer(&c, &buffer(0)),
        Err(out_of_range("buffer size", 0, 1, max))
    );
    assert_eq!(
        validate_buffer(&c, &buffer(max + 1)),
        Err(out_of_range("buffer size", max + 1, 1, max))
    );
    assert_eq!(
        validate_buffer(&c, &buffer(max + 1))
            .map_err(|e| e.to_string())
            .unwrap_err(),
        format!("buffer size {} is outside 1..={max}", max + 1)
    );
}

#[test]
fn buffer_needs_a_usage() {
    let c = caps();
    let desc = BufferDesc {
        size: 4,
        usage: BufferUsage::EMPTY,
    };
    assert_eq!(
        validate_buffer(&c, &desc),
        Err(RenderError::Invalid("a buffer needs at least one usage"))
    );
}

#[test]
fn staging_size_edges() {
    let c = caps();
    let max = c.limits.max_staging_size;
    assert_eq!(validate_staging_size(&c, 1), Ok(()));
    assert_eq!(validate_staging_size(&c, max), Ok(()));
    assert_eq!(
        validate_staging_size(&c, 0),
        Err(out_of_range("staging size", 0, 1, max))
    );
    assert_eq!(
        validate_staging_size(&c, max + 1),
        Err(out_of_range("staging size", max + 1, 1, max))
    );
}

// ---------------------------------------------------------------------------------------------
// Textures

fn texture(dimension: TextureDimension, w: u32, h: u32, d: u32) -> TextureDesc {
    TextureDesc {
        dimension,
        format: PixelFormat::A8B8G8R8Unorm,
        size: Extent3d {
            width: w,
            height: h,
            depth_or_layers: d,
        },
        mip_levels: 1,
        sample_count: 1,
        usage: TextureUsage::SAMPLED | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
    }
}

/// Checks `make(limit)` passes and `make(limit + 1)` and `make(0)` fail with `what`.
fn texture_edge(what: &'static str, min: u32, max: u32, make: impl Fn(u32) -> TextureDesc) {
    let c = caps();
    assert_eq!(validate_texture(&c, &make(min)), Ok(()), "{what} at {min}");
    assert_eq!(validate_texture(&c, &make(max)), Ok(()), "{what} at {max}");
    assert_eq!(
        validate_texture(&c, &make(min - 1)),
        Err(out_of_range(what, n(min - 1), n(min), n(max)))
    );
    assert_eq!(
        validate_texture(&c, &make(max + 1)),
        Err(out_of_range(what, n(max + 1), n(min), n(max)))
    );
}

#[test]
fn texture_dimension_edges() {
    use TextureDimension as D;
    let l = caps().limits;
    let layers = l.max_texture_array_layers;
    texture_edge("texture width", 1, l.max_texture_dimension_1d, |v| {
        texture(D::D1, v, 1, 1)
    });
    texture_edge("texture layer count", 1, layers, |v| {
        texture(D::D1, 1, 1, v)
    });
    texture_edge("texture width", 1, l.max_texture_dimension_2d, |v| {
        texture(D::D2, v, 1, 1)
    });
    texture_edge("texture height", 1, l.max_texture_dimension_2d, |v| {
        texture(D::D2, 1, v, 1)
    });
    texture_edge("texture layer count", 1, layers, |v| {
        texture(D::D2, 1, 1, v)
    });
    texture_edge("texture width", 1, l.max_texture_dimension_3d, |v| {
        texture(D::D3, v, 1, 1)
    });
    texture_edge("texture height", 1, l.max_texture_dimension_3d, |v| {
        texture(D::D3, 1, v, 1)
    });
    texture_edge("texture depth", 1, l.max_texture_dimension_3d, |v| {
        texture(D::D3, 1, 1, v)
    });
    texture_edge("texture width", 1, l.max_texture_dimension_2d, |v| {
        texture(D::Cube, v, v, CUBE_FACES)
    });
}

#[test]
fn one_d_texture_height_is_one() {
    let c = caps();
    assert_eq!(
        validate_texture(&c, &texture(TextureDimension::D1, 4, 2, 1)),
        Err(out_of_range("1D texture height", 2, 1, 1))
    );
    assert_eq!(
        validate_texture(&c, &texture(TextureDimension::D1, 4, 0, 1)),
        Err(out_of_range("1D texture height", 0, 1, 1))
    );
}

#[test]
fn cube_texture_rules() {
    let c = caps();
    let layers = c.limits.max_texture_array_layers;
    let largest_cube_layers = layers - layers % CUBE_FACES;
    assert_eq!(
        validate_texture(
            &c,
            &texture(TextureDimension::Cube, 8, 8, largest_cube_layers)
        ),
        Ok(())
    );
    assert_eq!(
        validate_texture(&c, &texture(TextureDimension::Cube, 8, 4, CUBE_FACES)),
        Err(RenderError::Invalid("a cube texture must be square"))
    );
    for bad in [CUBE_FACES - 1, CUBE_FACES + 1] {
        assert_eq!(
            validate_texture(&c, &texture(TextureDimension::Cube, 8, 8, bad)),
            Err(RenderError::Invalid(
                "a cube texture's layer count must be a multiple of 6"
            ))
        );
    }
    assert_eq!(
        validate_texture(&c, &texture(TextureDimension::Cube, 8, 8, 0)),
        Err(out_of_range("texture layer count", 0, 1, n(layers)))
    );
}

#[test]
fn full_mip_chain_counts_halvings() {
    assert_eq!(full_mip_chain(1), 1);
    assert_eq!(full_mip_chain(2), 2);
    assert_eq!(full_mip_chain(3), 2);
    assert_eq!(full_mip_chain(4), 3);
    assert_eq!(full_mip_chain(2048), 12);
    assert_eq!(full_mip_chain(2047), 11);
    assert_eq!(full_mip_chain(u32::MAX), 32);
}

#[test]
fn mip_count_edges() {
    let c = caps();
    for (desc, largest) in [
        (texture(TextureDimension::D2, 100, 37, 1), 100),
        (texture(TextureDimension::D2, 37, 100, 1), 100),
        (texture(TextureDimension::D3, 3, 5, 64), 64),
        (texture(TextureDimension::D1, 33, 1, 7), 33),
        (texture(TextureDimension::Cube, 16, 16, 6), 16),
    ] {
        let chain = full_mip_chain(largest);
        let with = |mips| TextureDesc {
            mip_levels: mips,
            ..desc
        };
        assert_eq!(validate_texture(&c, &with(1)), Ok(()));
        assert_eq!(validate_texture(&c, &with(chain)), Ok(()));
        assert_eq!(
            validate_texture(&c, &with(0)),
            Err(out_of_range("mip level count", 0, 1, n(chain)))
        );
        assert_eq!(
            validate_texture(&c, &with(chain + 1)),
            Err(out_of_range("mip level count", n(chain + 1), 1, n(chain)))
        );
    }
}

#[test]
fn sample_counts() {
    let c = caps();
    let ms = |count, dimension, mips| TextureDesc {
        sample_count: count,
        mip_levels: mips,
        ..texture(dimension, 8, 8, 1)
    };
    assert_eq!(
        validate_texture(&c, &ms(4, TextureDimension::D2, 1)),
        Ok(())
    );
    assert_eq!(
        validate_texture(&c, &ms(2, TextureDimension::D2, 1)),
        Err(RenderError::SampleCount(2))
    );
    assert_eq!(
        validate_texture(&c, &ms(0, TextureDimension::D2, 1)),
        Err(RenderError::SampleCount(0))
    );
    assert_eq!(
        validate_texture(&c, &ms(4, TextureDimension::D3, 1)),
        Err(RenderError::Invalid("a multisampled texture must be 2D"))
    );
    assert_eq!(
        validate_texture(&c, &ms(4, TextureDimension::D2, 2)),
        Err(RenderError::Invalid(
            "a multisampled texture must have one mip level"
        ))
    );
    assert_eq!(
        validate_texture(&c, &ms(1, TextureDimension::D2, 2)),
        Ok(())
    );
    assert_eq!(
        RenderError::SampleCount(2).to_string(),
        "sample count 2 is not supported by this device"
    );
}

#[test]
fn texture_usage_needs_format_support() {
    for (usage, feature, name) in [
        (TextureUsage::SAMPLED, FormatFeatures::SAMPLE, "sampling"),
        (
            TextureUsage::STORAGE,
            FormatFeatures::STORAGE,
            "storage images",
        ),
        (
            TextureUsage::RENDER_TARGET,
            FormatFeatures::RENDER,
            "render targets",
        ),
        (TextureUsage::COPY_SRC, FormatFeatures::COPY, "copies"),
        (TextureUsage::COPY_DST, FormatFeatures::COPY, "copies"),
    ] {
        let mut c = caps();
        let format = PixelFormat::R16G16Float;
        let desc = TextureDesc {
            format,
            usage,
            ..texture(TextureDimension::D2, 8, 8, 1)
        };
        assert_eq!(validate_texture(&c, &desc), Ok(()));
        c.formats
            .set(format, FormatFeatures::ALL.difference(feature));
        assert_eq!(
            validate_texture(&c, &desc),
            Err(RenderError::FormatUnsupported {
                format,
                usage: name
            })
        );
        // Another format's support does not matter.
        assert_eq!(
            validate_texture(
                &c,
                &TextureDesc {
                    format: PixelFormat::R16G16Unorm,
                    ..desc
                }
            ),
            Ok(())
        );
    }
    assert_eq!(
        RenderError::FormatUnsupported {
            format: PixelFormat::D24UnormS8Uint,
            usage: "render targets"
        }
        .to_string(),
        "D24_UNORM_S8_UINT cannot be used for render targets on this device"
    );
}

#[test]
fn texture_needs_a_usage() {
    let desc = TextureDesc {
        usage: TextureUsage::EMPTY,
        ..texture(TextureDimension::D2, 8, 8, 1)
    };
    assert_eq!(
        validate_texture(&caps(), &desc),
        Err(RenderError::Invalid("a texture needs at least one usage"))
    );
}

// ---------------------------------------------------------------------------------------------
// Views

fn view(format: PixelFormat, dimension: ViewDimension, layers: (u32, u32)) -> ViewDesc {
    ViewDesc {
        format,
        dimension,
        base_mip: 0,
        mip_count: 1,
        base_layer: layers.0,
        layer_count: layers.1,
        swizzle: Swizzle::IDENTITY,
    }
}

#[test]
fn view_swizzle_needs_the_capability() {
    let mut c = caps();
    let t = texture(TextureDimension::D2, 8, 8, 1);
    let swizzled = ViewDesc {
        swizzle: Swizzle {
            r: SwizzleSource::B,
            b: SwizzleSource::R,
            ..Swizzle::IDENTITY
        },
        ..view(t.format, ViewDimension::D2, (0, 1))
    };
    assert_eq!(validate_view(&c, &t, &swizzled), Ok(()));
    c.view_swizzle = false;
    assert_eq!(
        validate_view(&c, &t, &swizzled),
        Err(RenderError::MissingFeature("view swizzles"))
    );
    assert_eq!(
        validate_view(&c, &t, &view(t.format, ViewDimension::D2, (0, 1))),
        Ok(()),
        "an identity view needs no capability"
    );
    assert_eq!(
        RenderError::MissingFeature("view swizzles").to_string(),
        "the device lacks view swizzles"
    );
}

#[test]
fn view_format_must_match_block_and_aspect() {
    let c = caps();
    let t = texture(TextureDimension::D2, 8, 8, 1);
    // R32_UINT has A8B8G8R8's 32 bits per texel: a reinterpreting view.
    assert_eq!(
        validate_view(
            &c,
            &t,
            &view(PixelFormat::R32Uint, ViewDimension::D2, (0, 1))
        ),
        Ok(())
    );
    assert_eq!(
        validate_view(
            &c,
            &t,
            &view(PixelFormat::R16Uint, ViewDimension::D2, (0, 1))
        ),
        Err(RenderError::Invalid(
            "a view's format must have the same block size as its texture's"
        ))
    );
    // Same bits, different block footprint.
    let bc = TextureDesc {
        format: PixelFormat::Bc2Unorm,
        ..t
    };
    assert_eq!(
        validate_view(
            &c,
            &bc,
            &view(PixelFormat::Astc2d8x8Unorm, ViewDimension::D2, (0, 1))
        ),
        Err(RenderError::Invalid(
            "a view's format must have the same block size as its texture's"
        ))
    );
    // Same bits and block height, different block width.
    let astc = TextureDesc {
        format: PixelFormat::Astc2d8x8Unorm,
        ..t
    };
    assert_eq!(
        validate_view(
            &c,
            &astc,
            &view(PixelFormat::Astc2d10x8Unorm, ViewDimension::D2, (0, 1))
        ),
        Err(RenderError::Invalid(
            "a view's format must have the same block size as its texture's"
        ))
    );
    // Same block size, different aspect.
    assert_eq!(
        validate_view(
            &c,
            &t,
            &view(PixelFormat::D32Float, ViewDimension::D2, (0, 1))
        ),
        Err(RenderError::Invalid(
            "a view's format must store the same aspect as its texture's"
        ))
    );
}

#[test]
fn view_mip_range_edges() {
    let c = caps();
    let t = TextureDesc {
        mip_levels: 4,
        ..texture(TextureDimension::D2, 8, 8, 1)
    };
    let v = |base, count| ViewDesc {
        base_mip: base,
        mip_count: count,
        ..view(t.format, ViewDimension::D2, (0, 1))
    };
    assert_eq!(validate_view(&c, &t, &v(0, 4)), Ok(()));
    assert_eq!(validate_view(&c, &t, &v(3, 1)), Ok(()));
    assert_eq!(
        validate_view(&c, &t, &v(0, 0)),
        Err(out_of_range("view mip count", 0, 1, 4))
    );
    assert_eq!(
        validate_view(&c, &t, &v(0, 5)),
        Err(out_of_range("view mip count", 5, 1, 4))
    );
    assert_eq!(
        validate_view(&c, &t, &v(1, 4)),
        Err(past_end("view mip range", 5, 4))
    );
    // No overflow at the top of the range.
    assert_eq!(
        validate_view(&c, &t, &v(u32::MAX, 1)),
        Err(past_end("view mip range", n(u32::MAX) + 1, 4))
    );
}

#[test]
fn view_layer_range_edges() {
    let c = caps();
    let t = texture(TextureDimension::D2, 8, 8, 10);
    let v = |base, count| view(t.format, ViewDimension::D2Array, (base, count));
    assert_eq!(validate_view(&c, &t, &v(0, 10)), Ok(()));
    assert_eq!(validate_view(&c, &t, &v(9, 1)), Ok(()));
    assert_eq!(
        validate_view(&c, &t, &v(0, 0)),
        Err(out_of_range("view layer count", 0, 1, 10))
    );
    assert_eq!(
        validate_view(&c, &t, &v(0, 11)),
        Err(out_of_range("view layer count", 11, 1, 10))
    );
    assert_eq!(
        validate_view(&c, &t, &v(10, 1)),
        Err(past_end("view layer range", 11, 10))
    );
    assert_eq!(
        validate_view(&c, &t, &v(u32::MAX, 1)),
        Err(past_end("view layer range", n(u32::MAX) + 1, 10))
    );
    // A 3D texture has one layer for views.
    let t3 = texture(TextureDimension::D3, 8, 8, 8);
    assert_eq!(
        validate_view(&c, &t3, &view(t3.format, ViewDimension::D3, (0, 1))),
        Ok(())
    );
    assert_eq!(
        validate_view(&c, &t3, &view(t3.format, ViewDimension::D3, (0, 2))),
        Err(out_of_range("view layer count", 2, 1, 1))
    );
}

#[test]
fn view_dimension_fits_texture() {
    use TextureDimension as T;
    use ViewDimension as V;
    let c = caps();
    let fits = Ok(());
    let wrong = Err(RenderError::Invalid(
        "the view dimension does not fit the texture's",
    ));
    let cases = [
        (T::D1, V::D1, 1, fits.clone()),
        (T::D1, V::D1Array, 2, fits.clone()),
        (T::D1, V::D2, 1, wrong.clone()),
        (T::D2, V::D2, 1, fits.clone()),
        (T::D2, V::D2Array, 12, fits.clone()),
        (T::D2, V::Cube, 6, fits.clone()),
        (T::D2, V::CubeArray, 12, fits.clone()),
        (T::D2, V::D3, 1, wrong.clone()),
        (T::D2, V::D1, 1, wrong.clone()),
        (T::Cube, V::Cube, 6, fits.clone()),
        (T::Cube, V::D2, 1, fits.clone()),
        (T::Cube, V::CubeArray, 12, fits.clone()),
        (T::D3, V::D3, 1, fits.clone()),
        (T::D3, V::D2, 1, wrong.clone()),
        (T::D3, V::D2Array, 1, wrong.clone()),
    ];
    for (td, vd, layers, expected) in cases {
        let height = if td == T::D1 { 1 } else { 8 };
        let t = texture(td, 8, height, if td == T::D3 { 8 } else { 12 });
        assert_eq!(
            validate_view(&c, &t, &view(t.format, vd, (0, layers))),
            expected,
            "{td:?} as {vd:?}"
        );
    }
}

#[test]
fn view_layer_shape_rules() {
    let c = caps();
    let t = texture(TextureDimension::D2, 8, 8, 18);
    let v = |dimension, count| view(t.format, dimension, (0, count));
    assert_eq!(
        validate_view(&c, &t, &v(ViewDimension::D2, 2)),
        Err(RenderError::Invalid(
            "a single-layer view must have one layer"
        ))
    );
    let t1 = texture(TextureDimension::D1, 8, 1, 2);
    assert_eq!(
        validate_view(&c, &t1, &view(t1.format, ViewDimension::D1, (0, 2))),
        Err(RenderError::Invalid(
            "a single-layer view must have one layer"
        ))
    );
    for bad in [CUBE_FACES - 1, CUBE_FACES + 1] {
        assert_eq!(
            validate_view(&c, &t, &v(ViewDimension::Cube, bad)),
            Err(RenderError::Invalid("a cube view must have 6 layers"))
        );
        assert_eq!(
            validate_view(&c, &t, &v(ViewDimension::CubeArray, bad)),
            Err(RenderError::Invalid(
                "a cube array view's layer count must be a multiple of 6"
            ))
        );
    }
    assert_eq!(
        validate_view(&c, &t, &v(ViewDimension::CubeArray, 3 * CUBE_FACES)),
        Ok(())
    );
    let wide = texture(TextureDimension::D2, 16, 8, 6);
    for dimension in [ViewDimension::Cube, ViewDimension::CubeArray] {
        assert_eq!(
            validate_view(&c, &wide, &view(wide.format, dimension, (0, CUBE_FACES))),
            Err(RenderError::Invalid("a cube view needs a square texture"))
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Samplers

fn sampler() -> SamplerDesc {
    SamplerDesc {
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mip_filter: FilterMode::Nearest,
        address: [AddressMode::Repeat; 3],
        border_color: BorderColor::TransparentBlack,
        lod_bias: 0.0,
        min_lod: 0.0,
        max_lod: 15.0,
        max_anisotropy: 1,
        compare: None,
        reduction: ReductionMode::Average,
    }
}

#[test]
fn sampler_anisotropy_edges() {
    let c = caps();
    let max = c.limits.max_sampler_anisotropy;
    let s = |a| SamplerDesc {
        max_anisotropy: a,
        ..sampler()
    };
    assert_eq!(validate_sampler(&c, &s(1)), Ok(()));
    assert_eq!(validate_sampler(&c, &s(max)), Ok(()));
    assert_eq!(
        validate_sampler(&c, &s(0)),
        Err(out_of_range("sampler anisotropy", 0, 1, n(max)))
    );
    assert_eq!(
        validate_sampler(&c, &s(max + 1)),
        Err(out_of_range("sampler anisotropy", n(max + 1), 1, n(max)))
    );
}

#[test]
fn sampler_lod_values() {
    let c = caps();
    let finite = Err(RenderError::Invalid(
        "sampler level-of-detail values must be finite",
    ));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for s in [
            SamplerDesc {
                lod_bias: bad,
                ..sampler()
            },
            SamplerDesc {
                min_lod: bad,
                ..sampler()
            },
            SamplerDesc {
                max_lod: bad,
                ..sampler()
            },
        ] {
            assert_eq!(validate_sampler(&c, &s), finite);
        }
    }
    let lod = |min, max| SamplerDesc {
        min_lod: min,
        max_lod: max,
        ..sampler()
    };
    assert_eq!(validate_sampler(&c, &lod(2.0, 2.0)), Ok(()));
    assert_eq!(
        validate_sampler(&c, &lod(2.0, 1.5)),
        Err(RenderError::Invalid(
            "a sampler's minimum level of detail is above its maximum"
        ))
    );
}

#[test]
fn sampler_address_modes_need_features() {
    for (mode, feature, name) in [
        (
            AddressMode::ClampToBorder,
            Features::CLAMP_TO_BORDER,
            "the clamp-to-border address mode",
        ),
        (
            AddressMode::MirrorClampToEdge,
            Features::MIRROR_CLAMP_TO_EDGE,
            "the mirror-clamp-to-edge address mode",
        ),
    ] {
        for axis in 0..3 {
            let mut s = sampler();
            s.address[axis] = mode;
            let mut c = caps();
            assert_eq!(validate_sampler(&c, &s), Ok(()));
            c.features = Features::ALL.difference(feature);
            assert_eq!(
                validate_sampler(&c, &s),
                Err(RenderError::MissingFeature(name)),
                "axis {axis}"
            );
        }
    }
    let mut c = caps();
    c.features = Features::EMPTY;
    for mode in [
        AddressMode::Repeat,
        AddressMode::MirrorRepeat,
        AddressMode::ClampToEdge,
    ] {
        let s = SamplerDesc {
            address: [mode; 3],
            ..sampler()
        };
        assert_eq!(validate_sampler(&c, &s), Ok(()));
    }
}

#[test]
fn custom_border_colour_is_finite() {
    let c = caps();
    let s = |color| SamplerDesc {
        border_color: BorderColor::Custom(color),
        ..sampler()
    };
    assert_eq!(validate_sampler(&c, &s([0.25, 0.5, 1.0, -2.0])), Ok(()));
    for i in 0..4 {
        let mut color = [0.0; 4];
        color[i] = f32::NAN;
        assert_eq!(
            validate_sampler(&c, &s(color)),
            Err(RenderError::Invalid("a border colour must be finite"))
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Shaders

fn shader(stage: ShaderStage, interface: ShaderInterface) -> ShaderDesc {
    ShaderDesc {
        stage,
        entry: "main".to_owned(),
        source: ShaderSource::SpirV(vec![0x0723_0203]),
        interface,
    }
}

fn resource(group: u32, binding: u32, kind: ResourceKind) -> ResourceBinding {
    ResourceBinding {
        group,
        binding,
        kind,
    }
}

fn var(location: u32, components: u32) -> InterfaceVariable {
    InterfaceVariable {
        location,
        scalar: ScalarKind::Float,
        components,
        interpolation: Interpolation::Perspective,
    }
}

#[test]
fn shader_stages_need_support() {
    for (stage, flag) in [
        (ShaderStage::Geometry, StageSupport::GEOMETRY),
        (ShaderStage::TessControl, StageSupport::TESSELLATION),
        (ShaderStage::TessEval, StageSupport::TESSELLATION),
    ] {
        let mut c = caps();
        let s = shader(stage, ShaderInterface::default());
        assert_eq!(validate_shader(&c, &s), Ok(()));
        c.stages = StageSupport::ALL.difference(flag);
        assert_eq!(
            validate_shader(&c, &s),
            Err(RenderError::StageUnsupported(stage))
        );
    }
    let mut c = caps();
    c.stages = StageSupport::EMPTY;
    for stage in [
        ShaderStage::Vertex,
        ShaderStage::Fragment,
        ShaderStage::Compute,
    ] {
        assert_eq!(
            validate_shader(&c, &shader(stage, ShaderInterface::default())),
            Ok(())
        );
    }
    assert_eq!(
        RenderError::StageUnsupported(ShaderStage::TessEval).to_string(),
        "the device cannot run tessellation evaluation shaders"
    );
}

#[test]
fn shader_needs_an_entry_point() {
    let mut s = shader(ShaderStage::Compute, ShaderInterface::default());
    s.entry.clear();
    assert_eq!(
        validate_shader(&caps(), &s),
        Err(RenderError::Invalid("a shader needs an entry point name"))
    );
}

#[test]
fn shader_accepts_a_naga_module() {
    let module = naga::Module::default();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("an empty module validates");
    let s = ShaderDesc {
        source: ShaderSource::Naga {
            module: Box::new(module),
            info: Box::new(info),
        },
        ..shader(ShaderStage::Compute, ShaderInterface::default())
    };
    assert_eq!(validate_shader(&caps(), &s), Ok(()));
    let msl = ShaderDesc {
        source: ShaderSource::Msl("kernel void main0() {}".to_owned()),
        ..shader(ShaderStage::Compute, ShaderInterface::default())
    };
    assert_eq!(validate_shader(&caps(), &msl), Ok(()));
}

#[test]
fn shader_binding_edges() {
    let c = caps();
    let groups = c.limits.max_bind_groups;
    let bindings = c.limits.max_bindings_per_bind_group;
    let with = |r| {
        shader(
            ShaderStage::Fragment,
            ShaderInterface {
                resources: vec![r],
                ..ShaderInterface::default()
            },
        )
    };
    let ubo = ResourceKind::UniformBuffer;
    assert_eq!(
        validate_shader(&c, &with(resource(groups - 1, bindings - 1, ubo))),
        Ok(())
    );
    assert_eq!(
        validate_shader(&c, &with(resource(groups, 0, ubo))),
        Err(not_below("bind group", n(groups), n(groups)))
    );
    assert_eq!(
        validate_shader(&c, &with(resource(0, bindings, ubo))),
        Err(not_below("binding", n(bindings), n(bindings)))
    );
    assert_eq!(
        not_below("binding", 9, 9).to_string(),
        "binding 9 is not below 9"
    );
}

#[test]
fn shader_bindings_are_distinct() {
    let c = caps();
    let s = |resources| {
        shader(
            ShaderStage::Compute,
            ShaderInterface {
                resources,
                ..ShaderInterface::default()
            },
        )
    };
    let sampler = ResourceKind::Sampler { comparison: false };
    let storage = ResourceKind::StorageBuffer { writable: true };
    assert_eq!(
        validate_shader(
            &c,
            &s(vec![
                resource(1, 2, sampler),
                resource(2, 1, storage),
                resource(1, 3, storage)
            ])
        ),
        Ok(())
    );
    assert_eq!(
        validate_shader(
            &c,
            &s(vec![
                resource(1, 2, sampler),
                resource(2, 1, storage),
                resource(1, 2, storage)
            ])
        ),
        Err(RenderError::Invalid(
            "a shader binds one group and binding twice"
        ))
    );
}

#[test]
fn storage_texture_format_needs_storage_support() {
    let mut c = caps();
    let format = PixelFormat::R32Uint;
    let s = shader(
        ShaderStage::Compute,
        ShaderInterface {
            resources: vec![resource(
                0,
                0,
                ResourceKind::StorageTexture {
                    dimension: ViewDimension::D2,
                    format,
                    writable: true,
                },
            )],
            ..ShaderInterface::default()
        },
    );
    assert_eq!(validate_shader(&c, &s), Ok(()));
    c.formats.set(
        format,
        FormatFeatures::ALL.difference(FormatFeatures::STORAGE),
    );
    assert_eq!(
        validate_shader(&c, &s),
        Err(RenderError::FormatUnsupported {
            format,
            usage: "storage images"
        })
    );
    // A sampled texture needs no storage support.
    let sampled = shader(
        ShaderStage::Compute,
        ShaderInterface {
            resources: vec![resource(
                0,
                0,
                ResourceKind::SampledTexture {
                    dimension: ViewDimension::D2,
                    sample: SampleKind::Uint,
                    multisampled: false,
                },
            )],
            ..ShaderInterface::default()
        },
    );
    assert_eq!(validate_shader(&c, &sampled), Ok(()));
}

#[test]
fn interface_components_edges() {
    let c = caps();
    for inputs in [true, false] {
        let s = |v: InterfaceVariable| {
            let mut i = ShaderInterface::default();
            if inputs {
                i.inputs.push(v);
            } else {
                i.outputs.push(v);
            }
            shader(ShaderStage::Fragment, i)
        };
        assert_eq!(validate_shader(&c, &s(var(0, 1))), Ok(()));
        assert_eq!(
            validate_shader(&c, &s(var(0, MAX_INTERFACE_COMPONENTS))),
            Ok(())
        );
        assert_eq!(
            validate_shader(&c, &s(var(0, 0))),
            Err(out_of_range(
                "interface components",
                0,
                1,
                n(MAX_INTERFACE_COMPONENTS)
            ))
        );
        assert_eq!(
            validate_shader(&c, &s(var(0, MAX_INTERFACE_COMPONENTS + 1))),
            Err(out_of_range(
                "interface components",
                n(MAX_INTERFACE_COMPONENTS + 1),
                1,
                n(MAX_INTERFACE_COMPONENTS)
            ))
        );
    }
}

#[test]
fn interface_locations_are_distinct() {
    let c = caps();
    let inputs = shader(
        ShaderStage::Fragment,
        ShaderInterface {
            inputs: vec![var(3, 4), var(1, 2), var(3, 1)],
            ..ShaderInterface::default()
        },
    );
    assert_eq!(
        validate_shader(&c, &inputs),
        Err(RenderError::Invalid(
            "a shader uses one input location twice"
        ))
    );
    let outputs = shader(
        ShaderStage::Vertex,
        ShaderInterface {
            outputs: vec![var(5, 4), var(5, 2)],
            ..ShaderInterface::default()
        },
    );
    assert_eq!(
        validate_shader(&c, &outputs),
        Err(RenderError::Invalid(
            "a shader uses one output location twice"
        ))
    );
    // An input and an output may share a location.
    let both = shader(
        ShaderStage::Vertex,
        ShaderInterface {
            inputs: vec![var(5, 4)],
            outputs: vec![var(5, 4)],
            ..ShaderInterface::default()
        },
    );
    assert_eq!(validate_shader(&c, &both), Ok(()));
}

#[test]
fn vertex_input_and_fragment_output_location_edges() {
    // The device's attribute limit is below the guest's, and its colour targets below the
    // guest's; then the other way round.
    let device_smaller = caps();
    let mut guest_smaller = caps();
    guest_smaller.limits.max_vertex_attributes = GUEST_VERTEX_ATTRIBUTES + 1;
    guest_smaller.limits.max_color_attachments = guest_color_targets() + 1;
    for c in [device_smaller, guest_smaller] {
        let attrs = vertex_attribute_limit(&c);
        let targets = color_target_limit(&c);
        assert_eq!(
            attrs,
            c.limits.max_vertex_attributes.min(GUEST_VERTEX_ATTRIBUTES)
        );
        assert_eq!(
            targets,
            c.limits.max_color_attachments.min(guest_color_targets())
        );
        let vs = |loc| {
            shader(
                ShaderStage::Vertex,
                ShaderInterface {
                    inputs: vec![var(loc, 4)],
                    ..ShaderInterface::default()
                },
            )
        };
        assert_eq!(validate_shader(&c, &vs(attrs - 1)), Ok(()));
        assert_eq!(
            validate_shader(&c, &vs(attrs)),
            Err(not_below("vertex input location", n(attrs), n(attrs)))
        );
        let fs = |loc| {
            shader(
                ShaderStage::Fragment,
                ShaderInterface {
                    outputs: vec![var(loc, 4)],
                    ..ShaderInterface::default()
                },
            )
        };
        assert_eq!(validate_shader(&c, &fs(targets - 1)), Ok(()));
        assert_eq!(
            validate_shader(&c, &fs(targets)),
            Err(not_below(
                "fragment output location",
                n(targets),
                n(targets)
            ))
        );
        // Locations of other stages' inputs and outputs are not bounded by these limits.
        let other = shader(
            ShaderStage::Fragment,
            ShaderInterface {
                inputs: vec![var(attrs + 40, 4)],
                ..ShaderInterface::default()
            },
        );
        assert_eq!(validate_shader(&c, &other), Ok(()));
    }
}

// ---------------------------------------------------------------------------------------------
// Graphics pipelines

fn sid(i: u32) -> ShaderId {
    ShaderId::new(i, 0)
}

fn blend_off() -> BlendState {
    BlendState {
        enabled: false,
        color: BlendComponent {
            src: BlendFactor::One,
            dst: BlendFactor::Zero,
            op: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src: BlendFactor::One,
            dst: BlendFactor::Zero,
            op: BlendOperation::Add,
        },
        write_mask: ColorWrites::ALL,
    }
}

fn pipeline() -> GraphicsPipelineDesc {
    let mut colors = [None; GUEST_COLOR_TARGETS];
    colors[0] = Some(PixelFormat::A8B8G8R8Unorm);
    GraphicsPipelineDesc {
        stages: Stages {
            vertex: sid(1),
            tess_control: None,
            tess_eval: None,
            geometry: None,
            fragment: Some(sid(2)),
        },
        vertex_input: VertexLayout {
            buffers: vec![VertexBufferLayout {
                stride: 16,
                step: VertexStepMode::Vertex,
            }],
            attributes: vec![VertexAttribute {
                location: 0,
                buffer: 0,
                offset: 0,
                components: VertexComponents::R32G32B32A32,
                kind: VertexKind::Float,
            }],
        },
        topology: Topology::Triangles,
        primitive_restart: false,
        raster: RasterState {
            cull: CullMode::Back,
            front_face: FrontFace::Ccw,
            polygon_mode: PolygonMode::Fill,
            depth_clamp: false,
            depth_bias: DepthBias::default(),
            line_width: 1.0,
        },
        depth_stencil: DepthStencilState {
            depth_test: Some(CompareFunction::Less),
            depth_write: true,
            stencil: None,
        },
        blend: [blend_off(); GUEST_COLOR_TARGETS],
        logic_op: None,
        targets: TargetFormats {
            colors,
            depth_stencil: Some(PixelFormat::D32Float),
            sample_count: 1,
        },
        emulation: EmulationFlags::EMPTY,
    }
}

#[test]
fn a_plain_pipeline_passes() {
    assert_eq!(validate_graphics_pipeline(&caps(), &pipeline()), Ok(()));
}

#[test]
fn pipeline_stage_rules() {
    let c = caps();
    let mut p = pipeline();
    p.stages.tess_control = Some(sid(3));
    p.topology = Topology::Patches;
    assert_eq!(
        validate_graphics_pipeline(&c, &p),
        Err(RenderError::Invalid(
            "tessellation needs both a control and an evaluation shader"
        ))
    );
    p.stages.tess_eval = Some(sid(4));
    assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
    let mut no_tess = c.clone();
    no_tess.stages = StageSupport::ALL.difference(StageSupport::TESSELLATION);
    assert_eq!(
        validate_graphics_pipeline(&no_tess, &p),
        Err(RenderError::StageUnsupported(ShaderStage::TessControl))
    );
    p.topology = Topology::Triangles;
    assert_eq!(
        validate_graphics_pipeline(&c, &p),
        Err(RenderError::Invalid(
            "tessellation shaders need the patch topology"
        ))
    );
    let mut only_eval = pipeline();
    only_eval.stages.tess_eval = Some(sid(4));
    assert_eq!(
        validate_graphics_pipeline(&c, &only_eval),
        Err(RenderError::Invalid(
            "tessellation needs both a control and an evaluation shader"
        ))
    );
    let mut patches = pipeline();
    patches.topology = Topology::Patches;
    assert_eq!(
        validate_graphics_pipeline(&c, &patches),
        Err(RenderError::Invalid("patches need tessellation shaders"))
    );
    let mut geometry = pipeline();
    geometry.stages.geometry = Some(sid(5));
    assert_eq!(validate_graphics_pipeline(&c, &geometry), Ok(()));
    let mut no_geometry = c.clone();
    no_geometry.stages = StageSupport::ALL.difference(StageSupport::GEOMETRY);
    assert_eq!(
        validate_graphics_pipeline(&no_geometry, &geometry),
        Err(RenderError::StageUnsupported(ShaderStage::Geometry))
    );
}

fn attribute(location: u32, buffer: u32) -> VertexAttribute {
    VertexAttribute {
        location,
        buffer,
        offset: 4,
        components: VertexComponents::R8G8B8A8,
        kind: VertexKind::Unorm,
    }
}

fn buffers(count: u32) -> Vec<VertexBufferLayout> {
    (0..count)
        .map(|_| VertexBufferLayout {
            stride: 8,
            step: VertexStepMode::Instance,
        })
        .collect()
}

#[test]
fn vertex_buffer_count_edges() {
    // The device allows more buffers than the guest here, so the guest's 32 is the edge; then
    // a device below the guest.
    let guest_smaller = caps();
    let mut device_smaller = caps();
    device_smaller.limits.max_vertex_buffers = GUEST_VERTEX_BUFFERS - 3;
    for c in [guest_smaller, device_smaller] {
        let max = vertex_buffer_limit(&c);
        assert_eq!(max, c.limits.max_vertex_buffers.min(GUEST_VERTEX_BUFFERS));
        let mut p = pipeline();
        p.vertex_input.buffers = buffers(max);
        assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
        p.vertex_input.buffers = buffers(max + 1);
        assert_eq!(
            validate_graphics_pipeline(&c, &p),
            Err(out_of_range("vertex buffer count", n(max + 1), 0, n(max)))
        );
    }
}

#[test]
fn vertex_attribute_count_and_location_edges() {
    let device_smaller = caps();
    let mut guest_smaller = caps();
    guest_smaller.limits.max_vertex_attributes = GUEST_VERTEX_ATTRIBUTES + 5;
    for c in [device_smaller, guest_smaller] {
        let max = vertex_attribute_limit(&c);
        let mut p = pipeline();
        p.vertex_input.attributes = (0..max).map(|l| attribute(l, 0)).collect();
        assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
        p.vertex_input.attributes = (0..=max).map(|l| attribute(l, 0)).collect();
        assert_eq!(
            validate_graphics_pipeline(&c, &p),
            Err(out_of_range(
                "vertex attribute count",
                n(max + 1),
                0,
                n(max)
            ))
        );
        p.vertex_input.attributes = vec![attribute(max - 1, 0)];
        assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
        p.vertex_input.attributes = vec![attribute(max, 0)];
        assert_eq!(
            validate_graphics_pipeline(&c, &p),
            Err(not_below("vertex attribute location", n(max), n(max)))
        );
    }
}

#[test]
fn vertex_attribute_buffer_and_duplicates() {
    let c = caps();
    let mut p = pipeline();
    p.vertex_input.buffers = buffers(3);
    p.vertex_input.attributes = vec![attribute(0, 2)];
    assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
    p.vertex_input.attributes = vec![attribute(0, 3)];
    assert_eq!(
        validate_graphics_pipeline(&c, &p),
        Err(not_below("vertex attribute buffer", 3, 3))
    );
    p.vertex_input.attributes = vec![attribute(4, 0), attribute(1, 1), attribute(4, 2)];
    assert_eq!(
        validate_graphics_pipeline(&c, &p),
        Err(RenderError::Invalid(
            "a pipeline uses one vertex attribute location twice"
        ))
    );
}

#[test]
fn vertex_stride_edges() {
    let c = caps();
    let max = c.limits.max_vertex_buffer_stride;
    let mut p = pipeline();
    p.vertex_input.buffers[0].stride = max;
    assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
    p.vertex_input.buffers[0].stride = max + 1;
    assert_eq!(
        validate_graphics_pipeline(&c, &p),
        Err(out_of_range("vertex buffer stride", n(max + 1), 0, n(max)))
    );
}

#[test]
fn raster_features() {
    for (mode, feature, name) in [
        (
            PolygonMode::Line,
            Features::POLYGON_MODE_LINE,
            "line polygon mode",
        ),
        (
            PolygonMode::Point,
            Features::POLYGON_MODE_POINT,
            "point polygon mode",
        ),
    ] {
        let mut p = pipeline();
        p.raster.polygon_mode = mode;
        let mut c = caps();
        assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
        c.features = Features::ALL.difference(feature);
        assert_eq!(
            validate_graphics_pipeline(&c, &p),
            Err(RenderError::MissingFeature(name))
        );
    }
    let mut p = pipeline();
    p.raster.depth_clamp = true;
    let mut c = caps();
    assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
    c.features = Features::ALL.difference(Features::DEPTH_CLAMP);
    assert_eq!(
        validate_graphics_pipeline(&c, &p),
        Err(RenderError::MissingFeature("depth clamping"))
    );
    c.features = Features::EMPTY;
    assert_eq!(validate_graphics_pipeline(&c, &pipeline()), Ok(()));
}

#[test]
fn raster_values_are_finite() {
    let c = caps();
    for i in 0..4 {
        let mut p = pipeline();
        let r = &mut p.raster;
        *[
            &mut r.depth_bias.constant,
            &mut r.depth_bias.slope_scale,
            &mut r.depth_bias.clamp,
            &mut r.line_width,
        ][i] = f32::INFINITY;
        assert_eq!(
            validate_graphics_pipeline(&c, &p),
            Err(RenderError::Invalid("rasteriser values must be finite")),
            "value {i}"
        );
    }
}

#[test]
fn pipeline_sample_count() {
    let c = caps();
    let mut p = pipeline();
    p.targets.sample_count = 4;
    assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
    p.targets.sample_count = 8;
    assert_eq!(
        validate_graphics_pipeline(&c, &p),
        Err(RenderError::SampleCount(8))
    );
}

#[test]
fn colour_target_count_edges() {
    let device_smaller = caps();
    let mut guest_smaller = caps();
    guest_smaller.limits.max_color_attachments = guest_color_targets() + 3;
    for c in [device_smaller, guest_smaller] {
        let max = color_target_limit(&c);
        let at = |index: u32| {
            let mut p = pipeline();
            p.targets.colors = [None; GUEST_COLOR_TARGETS];
            p.targets.colors[usize::try_from(index).expect("small")] = Some(PixelFormat::R8Unorm);
            p
        };
        assert_eq!(validate_graphics_pipeline(&c, &at(max - 1)), Ok(()));
        let mut targets = RenderTargets::default();
        targets.colors[usize::try_from(max - 1).expect("small")] = Some(crate::ViewId::new(1, 0));
        assert_eq!(validate_render_targets(&c, &targets), Ok(()));
        if max < guest_color_targets() {
            assert_eq!(
                validate_graphics_pipeline(&c, &at(max)),
                Err(out_of_range("colour target count", n(max + 1), 0, n(max)))
            );
            targets.colors[usize::try_from(max).expect("small")] = Some(crate::ViewId::new(2, 0));
            assert_eq!(
                validate_render_targets(&c, &targets),
                Err(out_of_range("colour target count", n(max + 1), 0, n(max)))
            );
        } else {
            // The guest's fixed array cannot name more targets than the limit.
            assert_eq!(max, guest_color_targets());
        }
    }
    let mut none = pipeline();
    none.targets.colors = [None; GUEST_COLOR_TARGETS];
    assert_eq!(validate_graphics_pipeline(&caps(), &none), Ok(()));
    assert_eq!(
        validate_render_targets(&caps(), &RenderTargets::default()),
        Ok(())
    );
}

#[test]
fn target_formats() {
    let c = caps();
    let mut p = pipeline();
    p.targets.colors[2] = Some(PixelFormat::D16Unorm);
    assert_eq!(
        validate_graphics_pipeline(&c, &p),
        Err(RenderError::Invalid(
            "a colour target needs a colour format"
        ))
    );
    let mut p = pipeline();
    p.targets.depth_stencil = Some(PixelFormat::R32Float);
    assert_eq!(
        validate_graphics_pipeline(&c, &p),
        Err(RenderError::Invalid(
            "a depth/stencil target needs a depth or stencil format"
        ))
    );
    for format in [PixelFormat::S8Uint, PixelFormat::D24UnormS8Uint] {
        let mut p = pipeline();
        p.targets.depth_stencil = Some(format);
        assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
    }
    let no_render = FormatFeatures::ALL.difference(FormatFeatures::RENDER);
    let mut c2 = caps();
    c2.formats.set(PixelFormat::A8B8G8R8Unorm, no_render);
    assert_eq!(
        validate_graphics_pipeline(&c2, &pipeline()),
        Err(RenderError::FormatUnsupported {
            format: PixelFormat::A8B8G8R8Unorm,
            usage: "render targets"
        })
    );
    let mut c3 = caps();
    c3.formats.set(PixelFormat::D32Float, no_render);
    assert_eq!(
        validate_graphics_pipeline(&c3, &pipeline()),
        Err(RenderError::FormatUnsupported {
            format: PixelFormat::D32Float,
            usage: "render targets"
        })
    );
}

#[test]
fn blending_rules() {
    let mut p = pipeline();
    p.blend[0].enabled = true;
    let mut c = caps();
    assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
    c.formats.set(
        PixelFormat::A8B8G8R8Unorm,
        FormatFeatures::ALL.difference(FormatFeatures::BLEND),
    );
    assert_eq!(
        validate_graphics_pipeline(&c, &p),
        Err(RenderError::FormatUnsupported {
            format: PixelFormat::A8B8G8R8Unorm,
            usage: "blending"
        })
    );
    // Blending switched off on that target needs no blend support.
    p.blend[0].enabled = false;
    assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
    // Blending on a target with no format is ignored.
    p.blend[5].enabled = true;
    p.blend[5].color.src = BlendFactor::Src1;
    let mut none = caps();
    none.features = Features::EMPTY;
    assert_eq!(validate_graphics_pipeline(&none, &p), Ok(()));

    for slot in 0..4 {
        let mut p = pipeline();
        p.blend[0].enabled = true;
        let b = &mut p.blend[0];
        *[
            &mut b.color.src,
            &mut b.color.dst,
            &mut b.alpha.src,
            &mut b.alpha.dst,
        ][slot] = BlendFactor::OneMinusSrc1Alpha;
        let mut c = caps();
        assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
        c.features = Features::ALL.difference(Features::DUAL_SOURCE_BLEND);
        assert_eq!(
            validate_graphics_pipeline(&c, &p),
            Err(RenderError::MissingFeature("dual-source blending")),
            "slot {slot}"
        );
    }
    for f in [
        BlendFactor::Src1,
        BlendFactor::OneMinusSrc1,
        BlendFactor::Src1Alpha,
        BlendFactor::OneMinusSrc1Alpha,
    ] {
        assert!(f.is_dual_source());
    }
    assert!(!BlendFactor::SrcAlpha.is_dual_source());
}

#[test]
fn logic_op_needs_the_feature() {
    let mut p = pipeline();
    p.logic_op = Some(LogicOp::Xor);
    let mut c = caps();
    assert_eq!(validate_graphics_pipeline(&c, &p), Ok(()));
    c.features = Features::ALL.difference(Features::LOGIC_OP);
    assert_eq!(
        validate_graphics_pipeline(&c, &p),
        Err(RenderError::MissingFeature("logic operations"))
    );
}

// ---------------------------------------------------------------------------------------------
// Draws and dispatches

fn viewport() -> Viewport {
    Viewport {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: -64.0,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

fn draw() -> DrawDesc {
    DrawDesc {
        pipeline: PipelineId::new(1, 0),
        bindings: Vec::new(),
        vertex_buffers: Vec::new(),
        index_buffer: None,
        dynamic: DynamicValues {
            viewports: vec![viewport()],
            scissors: Vec::new(),
            blend_constant: [0.0; 4],
            stencil_reference: 0,
            stencil_read_mask: 0xff,
            stencil_write_mask: 0xff,
            depth_bias: DepthBias::default(),
            line_width: 1.0,
        },
        params: DrawParams::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
    }
}

fn slice(size: u64) -> BufferSlice {
    BufferSlice {
        buffer: BufferId::new(1, 0),
        offset: 0,
        size,
    }
}

#[test]
fn viewport_count_edges() {
    let guest_smaller = caps();
    let mut device_smaller = caps();
    device_smaller.limits.max_viewports = 1;
    for c in [guest_smaller, device_smaller] {
        let max = viewport_limit(&c);
        assert_eq!(max, c.limits.max_viewports.min(GUEST_VIEWPORTS));
        let with = |count: u32, scissors: u32| {
            let mut d = draw();
            d.dynamic.viewports = (0..count).map(|_| viewport()).collect();
            d.dynamic.scissors = (0..scissors).map(|_| Rect::default()).collect();
            d
        };
        assert_eq!(validate_draw(&c, &with(1, 0)), Ok(()));
        assert_eq!(validate_draw(&c, &with(max, max)), Ok(()));
        assert_eq!(
            validate_draw(&c, &with(0, 0)),
            Err(out_of_range("viewport count", 0, 1, n(max)))
        );
        assert_eq!(
            validate_draw(&c, &with(max + 1, 0)),
            Err(out_of_range("viewport count", n(max + 1), 1, n(max)))
        );
    }
    let c = caps();
    let mut d = draw();
    d.dynamic.viewports = vec![viewport(), viewport()];
    d.dynamic.scissors = vec![Rect::default()];
    assert_eq!(
        validate_draw(&c, &d),
        Err(RenderError::Invalid(
            "a draw needs one scissor per viewport, or none"
        ))
    );
}

#[test]
fn draw_values_are_finite() {
    let c = caps();
    let finite = Err(RenderError::Invalid("draw state values must be finite"));
    for i in 0..6 {
        let mut d = draw();
        let v = &mut d.dynamic.viewports[0];
        *[
            &mut v.x,
            &mut v.y,
            &mut v.width,
            &mut v.height,
            &mut v.min_depth,
            &mut v.max_depth,
        ][i] = f32::NAN;
        assert_eq!(validate_draw(&c, &d), finite, "viewport value {i}");
    }
    for i in 0..8 {
        let mut d = draw();
        let dy = &mut d.dynamic;
        let [b0, b1, b2, b3] = &mut dy.blend_constant;
        *[
            b0,
            b1,
            b2,
            b3,
            &mut dy.depth_bias.constant,
            &mut dy.depth_bias.slope_scale,
            &mut dy.depth_bias.clamp,
            &mut dy.line_width,
        ][i] = f32::NEG_INFINITY;
        assert_eq!(validate_draw(&c, &d), finite, "value {i}");
    }
}

#[test]
fn draw_vertex_buffer_count_edges() {
    let guest_smaller = caps();
    let mut device_smaller = caps();
    device_smaller.limits.max_vertex_buffers = 2;
    for c in [guest_smaller, device_smaller] {
        let max = vertex_buffer_limit(&c);
        let mut d = draw();
        d.vertex_buffers = (0..max).map(|_| slice(4)).collect();
        assert_eq!(validate_draw(&c, &d), Ok(()));
        d.vertex_buffers.push(slice(4));
        assert_eq!(
            validate_draw(&c, &d),
            Err(out_of_range("vertex buffer count", n(max + 1), 0, n(max)))
        );
    }
}

fn binding(group: u32, binding: u32) -> Binding {
    Binding {
        group,
        binding,
        resource: BindingResource::Sampler(SamplerId::new(1, 0)),
    }
}

#[test]
fn draw_and_dispatch_binding_edges() {
    let c = caps();
    let groups = c.limits.max_bind_groups;
    let bindings = c.limits.max_bindings_per_bind_group;
    let dispatch = |b: Vec<Binding>| DispatchDesc {
        pipeline: PipelineId::new(1, 0),
        bindings: b,
        params: DispatchParams::Indirect {
            buffer: BufferId::new(1, 0),
            offset: 0,
        },
    };
    let draw_with = |b: Vec<Binding>| DrawDesc {
        bindings: b,
        ..draw()
    };
    let ok = vec![binding(groups - 1, bindings - 1), binding(0, 0)];
    assert_eq!(validate_draw(&c, &draw_with(ok.clone())), Ok(()));
    assert_eq!(validate_dispatch(&c, &dispatch(ok)), Ok(()));
    for (b, expected) in [
        (
            binding(groups, 0),
            not_below("bind group", n(groups), n(groups)),
        ),
        (
            binding(0, bindings),
            not_below("binding", n(bindings), n(bindings)),
        ),
    ] {
        assert_eq!(
            validate_draw(&c, &draw_with(vec![b])),
            Err(expected.clone())
        );
        assert_eq!(validate_dispatch(&c, &dispatch(vec![b])), Err(expected));
    }
    let twice = vec![binding(1, 1), binding(0, 1), binding(1, 1)];
    assert_eq!(
        validate_draw(&c, &draw_with(twice.clone())),
        Err(RenderError::Invalid(
            "a draw binds one group and binding twice"
        ))
    );
    assert_eq!(
        validate_dispatch(&c, &dispatch(twice)),
        Err(RenderError::Invalid(
            "a dispatch binds one group and binding twice"
        ))
    );
}

#[test]
fn indexed_draws_need_an_index_buffer() {
    let c = caps();
    let indexed = DrawParams::Indexed {
        index_count: 3,
        instance_count: 1,
        first_index: 0,
        base_vertex: -1,
        first_instance: 0,
    };
    let indirect = |indexed| DrawParams::Indirect {
        buffer: BufferId::new(2, 0),
        offset: 0,
        draw_count: 1,
        stride: 20,
        indexed,
    };
    for params in [indexed, indirect(true)] {
        let mut d = DrawDesc { params, ..draw() };
        assert_eq!(
            validate_draw(&c, &d),
            Err(RenderError::Invalid(
                "an indexed draw needs an index buffer"
            ))
        );
        d.index_buffer = Some(IndexBufferBinding {
            slice: slice(6),
            format: IndexFormat::U16,
        });
        assert_eq!(validate_draw(&c, &d), Ok(()));
    }
    let d = DrawDesc {
        params: indirect(false),
        ..draw()
    };
    assert_eq!(validate_draw(&c, &d), Ok(()));
}

#[test]
fn indirect_draw_count_edges() {
    let with = |count| DrawDesc {
        params: DrawParams::Indirect {
            buffer: BufferId::new(2, 0),
            offset: 0,
            draw_count: count,
            stride: 16,
            indexed: false,
        },
        ..draw()
    };
    let mut c = caps();
    assert_eq!(validate_draw(&c, &with(u32::MAX)), Ok(()));
    assert_eq!(
        validate_draw(&c, &with(0)),
        Err(out_of_range("indirect draw count", 0, 1, n(u32::MAX)))
    );
    c.features = Features::ALL.difference(Features::MULTI_DRAW_INDIRECT);
    assert_eq!(validate_draw(&c, &with(1)), Ok(()));
    assert_eq!(
        validate_draw(&c, &with(2)),
        Err(RenderError::MissingFeature("multi-draw indirect"))
    );
}

#[test]
fn workgroup_count_edges() {
    let c = caps();
    let max = c.limits.max_compute_workgroups_per_dimension;
    let with = |x, y, z| DispatchDesc {
        pipeline: PipelineId::new(1, 0),
        bindings: Vec::new(),
        params: DispatchParams::Direct { x, y, z },
    };
    assert_eq!(validate_dispatch(&c, &with(max, max, max)), Ok(()));
    assert_eq!(validate_dispatch(&c, &with(0, 0, 0)), Ok(()));
    for (x, y, z) in [(max + 1, 1, 1), (1, max + 1, 1), (1, 1, max + 1)] {
        assert_eq!(
            validate_dispatch(&c, &with(x, y, z)),
            Err(out_of_range("workgroup count", n(max + 1), 0, n(max)))
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Regions, copies, uploads and readbacks

fn region(
    texture: u32,
    mip: u32,
    origin: (u32, u32, u32),
    extent: (u32, u32, u32),
) -> TextureRegion {
    TextureRegion {
        texture: TextureId::new(texture, 0),
        mip,
        origin: Origin3d {
            x: origin.0,
            y: origin.1,
            z: origin.2,
        },
        extent: Extent3d {
            width: extent.0,
            height: extent.1,
            depth_or_layers: extent.2,
        },
    }
}

#[test]
fn buffer_slice_edges() {
    let b = buffer(100);
    let s = |offset, size| BufferSlice {
        buffer: BufferId::new(1, 0),
        offset,
        size,
    };
    assert_eq!(validate_buffer_slice(&b, &s(0, 100)), Ok(()));
    assert_eq!(validate_buffer_slice(&b, &s(99, 1)), Ok(()));
    assert_eq!(
        validate_buffer_slice(&b, &s(0, 0)),
        Err(out_of_range("buffer range size", 0, 1, 100))
    );
    assert_eq!(
        validate_buffer_slice(&b, &s(0, 101)),
        Err(out_of_range("buffer range size", 101, 1, 100))
    );
    assert_eq!(
        validate_buffer_slice(&b, &s(100, 1)),
        Err(past_end("buffer range", 101, 100))
    );
    assert_eq!(
        validate_buffer_slice(&buffer(u64::MAX), &s(u64::MAX, 1)),
        Err(RenderError::Overflow {
            what: "buffer range end"
        })
    );
    assert_eq!(
        validate_buffer_slice(&buffer(u64::MAX), &s(u64::MAX - 1, 1)),
        Ok(())
    );
    assert_eq!(
        past_end("buffer range", 101, 100).to_string(),
        "buffer range ends at 101, past the end at 100"
    );
    assert_eq!(
        RenderError::Overflow {
            what: "buffer range end"
        }
        .to_string(),
        "buffer range end overflows"
    );
}

#[test]
fn texture_region_edges() {
    let mut t = texture(TextureDimension::D2, 64, 32, 4);
    t.mip_levels = 3;
    // Mip 2 is 16 x 8, with 4 layers.
    assert_eq!(
        validate_texture_region(&t, &region(1, 2, (0, 0, 0), (16, 8, 4))),
        Ok(())
    );
    assert_eq!(
        validate_texture_region(&t, &region(1, 2, (15, 7, 3), (1, 1, 1))),
        Ok(())
    );
    for (r, expected) in [
        (
            region(1, 3, (0, 0, 0), (1, 1, 1)),
            not_below("mip level", 3, 3),
        ),
        (
            region(1, 2, (1, 0, 0), (16, 8, 4)),
            past_end("texture region width", 17, 16),
        ),
        (
            region(1, 2, (0, 1, 0), (16, 8, 4)),
            past_end("texture region height", 9, 8),
        ),
        (
            region(1, 2, (0, 0, 1), (16, 8, 4)),
            past_end("texture region depth or layers", 5, 4),
        ),
        (
            region(1, 2, (u32::MAX, 0, 0), (1, 1, 1)),
            past_end("texture region width", n(u32::MAX) + 1, 16),
        ),
        (
            region(1, 2, (0, 0, 0), (u32::MAX, 1, 1)),
            past_end("texture region width", n(u32::MAX), 16),
        ),
    ] {
        assert_eq!(validate_texture_region(&t, &r), Err(expected));
    }
    for extent in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
        assert_eq!(
            validate_texture_region(&t, &region(1, 0, (0, 0, 0), extent)),
            Err(RenderError::Invalid("a texture region must not be empty"))
        );
    }
    // A 3D texture's depth halves per mip.
    let mut t3 = texture(TextureDimension::D3, 8, 8, 8);
    t3.mip_levels = 2;
    assert_eq!(
        validate_texture_region(&t3, &region(1, 1, (0, 0, 0), (4, 4, 4))),
        Ok(())
    );
    assert_eq!(
        validate_texture_region(&t3, &region(1, 1, (0, 0, 0), (4, 4, 5))),
        Err(past_end("texture region depth or layers", 5, 4))
    );
}

#[test]
fn compressed_region_blocks() {
    // ASTC 8x5 at 20 x 13 texels: mip 0 is two whole blocks and a partial one across, two and
    // a partial one down.
    let t = TextureDesc {
        format: PixelFormat::Astc2d8x5Unorm,
        ..texture(TextureDimension::D2, 20, 13, 1)
    };
    let (bw, bh) = (t.format.block_width(), t.format.block_height());
    let ok = [
        region(1, 0, (0, 0, 0), (bw, bh, 1)),
        region(1, 0, (bw, bh, 0), (bw, bh, 1)),
        region(1, 0, (2 * bw, 2 * bh, 0), (20 - 2 * bw, 13 - 2 * bh, 1)),
        region(1, 0, (0, 0, 0), (20, 13, 1)),
    ];
    for r in ok {
        assert_eq!(validate_texture_region(&t, &r), Ok(()), "{r:?}");
    }
    let start = Err(RenderError::Invalid(
        "a compressed texture region must start on a block boundary",
    ));
    let whole = Err(RenderError::Invalid(
        "a compressed texture region must cover whole blocks or reach the mip edge",
    ));
    for (r, expected) in [
        (region(1, 0, (1, 0, 0), (bw, bh, 1)), start.clone()),
        (region(1, 0, (0, 1, 0), (bw, bh, 1)), start),
        (region(1, 0, (0, 0, 0), (bw - 1, bh, 1)), whole.clone()),
        (region(1, 0, (0, 0, 0), (bw, bh + 1, 1)), whole.clone()),
        (region(1, 0, (0, 0, 0), (19, 13, 1)), whole),
    ] {
        assert_eq!(validate_texture_region(&t, &r), expected, "{r:?}");
    }
}

#[test]
fn packed_sizes() {
    let e = |w, h, d| Extent3d {
        width: w,
        height: h,
        depth_or_layers: d,
    };
    let rgba8 = PixelFormat::A8B8G8R8Unorm;
    assert_eq!(
        packed_size(rgba8, e(3, 2, 5)),
        Ok(3 * 2 * 5 * u64::from(rgba8.bytes_per_block()))
    );
    let astc = PixelFormat::Astc2d10x6Unorm;
    // 21 x 7 texels is 3 x 2 blocks.
    assert_eq!(
        packed_size(astc, e(21, 7, 2)),
        Ok(3 * 2 * 2 * u64::from(astc.bytes_per_block()))
    );
    assert_eq!(
        packed_size(astc, e(20, 6, 1)),
        Ok(2 * u64::from(astc.bytes_per_block()))
    );
    let wide = PixelFormat::R32G32B32A32Float;
    // u32::MAX squared fits in u64; times the depth or the block bytes does not.
    assert_eq!(
        packed_size(PixelFormat::R8Unorm, e(u32::MAX, u32::MAX, 1)),
        Ok(n(u32::MAX) * n(u32::MAX))
    );
    let overflow = Err(RenderError::Overflow {
        what: "packed texture size",
    });
    assert_eq!(
        packed_size(PixelFormat::R8Unorm, e(u32::MAX, u32::MAX, 2)),
        overflow
    );
    assert_eq!(packed_size(wide, e(u32::MAX, u32::MAX, 1)), overflow);
}

fn copy_texture(format: PixelFormat, usage: TextureUsage) -> TextureDesc {
    TextureDesc {
        format,
        usage,
        ..texture(TextureDimension::D2, 16, 16, 2)
    }
}

#[test]
fn copy_usages() {
    let readable = buffer(1024);
    // Each lacks exactly one of the two copy usages, so a check of the wrong one shows.
    let unreadable = BufferDesc {
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        ..readable
    };
    let unwritable = BufferDesc {
        usage: BufferUsage::VERTEX | BufferUsage::COPY_SRC,
        ..readable
    };
    let s = slice(16);
    let sampled = copy_texture(
        PixelFormat::A8B8G8R8Unorm,
        TextureUsage::SAMPLED | TextureUsage::COPY_DST,
    );
    let read_only = copy_texture(
        PixelFormat::A8B8G8R8Unorm,
        TextureUsage::SAMPLED | TextureUsage::COPY_SRC,
    );
    let copyable = copy_texture(
        PixelFormat::A8B8G8R8Unorm,
        TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
    );
    let r = region(1, 0, (0, 0, 0), (2, 2, 1));
    let buf = |desc| Resolved::Buffer { desc, slice: &s };
    let tex = |desc| Resolved::Texture { desc, region: &r };
    assert_eq!(validate_copy(buf(&readable), buf(&readable)), Ok(()));
    assert_eq!(validate_copy(tex(&copyable), tex(&copyable)), Ok(()));
    for (src, dst, resource, usage) in [
        (
            buf(&unreadable),
            buf(&readable),
            "copy source buffer",
            "copy source",
        ),
        (
            buf(&readable),
            buf(&unwritable),
            "copy destination buffer",
            "copy destination",
        ),
        (
            tex(&sampled),
            tex(&copyable),
            "copy source texture",
            "copy source",
        ),
        (
            tex(&copyable),
            tex(&read_only),
            "copy destination texture",
            "copy destination",
        ),
    ] {
        assert_eq!(
            validate_copy(src, dst),
            Err(RenderError::MissingUsage { resource, usage })
        );
    }
    assert_eq!(
        RenderError::MissingUsage {
            resource: "copy source texture",
            usage: "copy source"
        }
        .to_string(),
        "the copy source texture lacks the copy source usage"
    );
    assert_eq!(validate_readback(tex(&copyable)), Ok(()));
    assert_eq!(
        validate_readback(tex(&sampled)),
        Err(RenderError::MissingUsage {
            resource: "copy source texture",
            usage: "copy source"
        })
    );
    assert_eq!(validate_readback(buf(&readable)), Ok(()));
    assert_eq!(
        validate_readback(buf(&unreadable)),
        Err(RenderError::MissingUsage {
            resource: "copy source buffer",
            usage: "copy source"
        })
    );
}

#[test]
fn copy_sides_are_bounds_checked() {
    let b = buffer(64);
    let t = copy_texture(
        PixelFormat::A8B8G8R8Unorm,
        TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
    );
    let past = BufferSlice {
        offset: 60,
        ..slice(8)
    };
    let fine = slice(16);
    let outside = region(1, 0, (15, 0, 0), (2, 2, 1));
    let inside = region(1, 0, (0, 0, 0), (2, 2, 1));
    assert_eq!(
        validate_copy(
            Resolved::Buffer {
                desc: &b,
                slice: &past
            },
            Resolved::Texture {
                desc: &t,
                region: &inside
            }
        ),
        Err(past_end("buffer range", 68, 64))
    );
    assert_eq!(
        validate_copy(
            Resolved::Buffer {
                desc: &b,
                slice: &fine
            },
            Resolved::Texture {
                desc: &t,
                region: &outside
            }
        ),
        Err(past_end("texture region width", 17, 16))
    );
    assert_eq!(
        validate_readback(Resolved::Texture {
            desc: &t,
            region: &outside
        }),
        Err(past_end("texture region width", 17, 16))
    );
}

#[test]
fn copy_sizes_must_match() {
    let b = buffer(1024);
    let format = PixelFormat::A8B8G8R8Unorm;
    let t = copy_texture(format, TextureUsage::COPY_SRC | TextureUsage::COPY_DST);
    let r = region(1, 0, (0, 0, 0), (2, 3, 1));
    let needed = 2 * 3 * u64::from(format.bytes_per_block());
    let exact = slice(needed);
    let short = slice(needed - 1);
    let tex = Resolved::Texture {
        desc: &t,
        region: &r,
    };
    assert_eq!(
        validate_copy(
            Resolved::Buffer {
                desc: &b,
                slice: &exact
            },
            tex
        ),
        Ok(())
    );
    assert_eq!(
        validate_copy(
            tex,
            Resolved::Buffer {
                desc: &b,
                slice: &exact
            }
        ),
        Ok(())
    );
    assert_eq!(
        validate_copy(
            Resolved::Buffer {
                desc: &b,
                slice: &short
            },
            tex
        ),
        Err(RenderError::SizeMismatch {
            what: "copy source buffer range",
            expected: needed,
            actual: needed - 1
        })
    );
    assert_eq!(
        validate_copy(
            tex,
            Resolved::Buffer {
                desc: &b,
                slice: &short
            }
        ),
        Err(RenderError::SizeMismatch {
            what: "copy destination buffer range",
            expected: needed,
            actual: needed - 1
        })
    );
    let other = slice(17);
    assert_eq!(
        validate_copy(
            Resolved::Buffer {
                desc: &b,
                slice: &exact
            },
            Resolved::Buffer {
                desc: &b,
                slice: &other
            }
        ),
        Err(RenderError::SizeMismatch {
            what: "copy destination buffer range",
            expected: needed,
            actual: 17
        })
    );
    assert_eq!(
        RenderError::SizeMismatch {
            what: "staging slice",
            expected: 24,
            actual: 23
        }
        .to_string(),
        "staging slice is 23 bytes, but 24 are needed"
    );
}

#[test]
fn texture_copies_need_matching_blocks_and_extents() {
    let usage = TextureUsage::COPY_SRC | TextureUsage::COPY_DST;
    let a = copy_texture(PixelFormat::A8B8G8R8Unorm, usage);
    let same_size = copy_texture(PixelFormat::R32Float, usage);
    let smaller = copy_texture(PixelFormat::R16Float, usage);
    let r = region(1, 0, (0, 0, 0), (4, 4, 1));
    let r2 = region(2, 0, (4, 4, 1), (4, 4, 1));
    let taller = region(2, 0, (0, 0, 0), (4, 5, 1));
    let res = |desc, region| Resolved::Texture { desc, region };
    assert_eq!(validate_copy(res(&a, &r), res(&same_size, &r2)), Ok(()));
    assert_eq!(
        validate_copy(res(&a, &r), res(&smaller, &r2)),
        Err(RenderError::Invalid(
            "a texture copy needs formats with the same block size"
        ))
    );
    assert_eq!(
        validate_copy(res(&a, &r), res(&a, &taller)),
        Err(RenderError::Invalid("a texture copy needs equal extents"))
    );
}

#[test]
fn upload_sizes() {
    let format = PixelFormat::Bc1RgbaUnorm;
    let t = copy_texture(format, TextureUsage::COPY_DST);
    // 16 x 16 texels of BC1 is 4 x 4 blocks of 8 bytes.
    let r = region(1, 0, (0, 0, 0), (16, 16, 1));
    let needed = packed_size(format, r.extent).expect("fits");
    assert_eq!(needed, 4 * 4 * u64::from(format.bytes_per_block()));
    let dst = Resolved::Texture {
        desc: &t,
        region: &r,
    };
    assert_eq!(validate_upload(needed, dst), Ok(()));
    for wrong in [needed - 1, needed + 1, 0] {
        assert_eq!(
            validate_upload(wrong, dst),
            Err(RenderError::SizeMismatch {
                what: "staging slice",
                expected: needed,
                actual: wrong
            })
        );
    }
    let b = buffer(256);
    let s = slice(40);
    let bdst = Resolved::Buffer {
        desc: &b,
        slice: &s,
    };
    assert_eq!(validate_upload(40, bdst), Ok(()));
    assert_eq!(
        validate_upload(41, bdst),
        Err(RenderError::SizeMismatch {
            what: "staging slice",
            expected: 40,
            actual: 41
        })
    );
    let read_only = copy_texture(format, TextureUsage::COPY_SRC);
    assert_eq!(
        validate_upload(
            needed,
            Resolved::Texture {
                desc: &read_only,
                region: &r
            }
        ),
        Err(RenderError::MissingUsage {
            resource: "copy destination texture",
            usage: "copy destination"
        })
    );
}

#[test]
fn rect_edges() {
    let r = |x, y, w, h| Rect {
        x,
        y,
        width: w,
        height: h,
    };
    assert_eq!(validate_rect(64, 32, &r(0, 0, 64, 32)), Ok(()));
    assert_eq!(validate_rect(64, 32, &r(63, 31, 1, 1)), Ok(()));
    assert_eq!(validate_rect(64, 32, &r(64, 32, 0, 0)), Ok(()));
    assert_eq!(
        validate_rect(64, 32, &r(1, 0, 64, 32)),
        Err(past_end("rectangle width", 65, 64))
    );
    assert_eq!(
        validate_rect(64, 32, &r(0, 1, 64, 32)),
        Err(past_end("rectangle height", 33, 32))
    );
    assert_eq!(
        validate_rect(64, 32, &r(u32::MAX, 0, u32::MAX, 1)),
        Err(past_end("rectangle width", 2 * n(u32::MAX), 64))
    );
}

#[test]
fn error_messages_are_plain_words() {
    assert_eq!(
        out_of_range("texture width", 0, 1, 2048).to_string(),
        "texture width 0 is outside 1..=2048"
    );
    assert_eq!(
        RenderError::Invalid("a cube texture must be square").to_string(),
        "a cube texture must be square"
    );
    assert_eq!(
        RenderError::DeviceLost.to_string(),
        "the host device was lost"
    );
    let unknown = crate::caps::Features::from_bits(1 << 31).expect_err("unknown");
    assert_eq!(
        RenderError::from(unknown).to_string(),
        "Features has unknown bits 0x80000000"
    );
}

proptest! {
    #![proptest_config(Config {
        rng_seed: RngSeed::Fixed(0x5ee_d132),
        cases: 512,
        ..Config::default()
    })]

    /// No input makes the region checks panic.
    #[test]
    fn region_checks_never_panic(
        w in any::<u32>(), h in any::<u32>(), d in any::<u32>(), mips in any::<u32>(),
        mip in any::<u32>(), ox in any::<u32>(), oy in any::<u32>(), oz in any::<u32>(),
        ew in any::<u32>(), eh in any::<u32>(), ed in any::<u32>(),
        format in 0..PixelFormat::ALL.len(), dim in 0..4_usize,
        bsize in any::<u64>(), boff in any::<u64>(), blen in any::<u64>(),
        rx in any::<u32>(), ry in any::<u32>(),
    ) {
        let dimension = [TextureDimension::D1, TextureDimension::D2, TextureDimension::D3,
            TextureDimension::Cube][dim];
        let t = TextureDesc {
            dimension,
            format: PixelFormat::ALL[format],
            size: Extent3d { width: w, height: h, depth_or_layers: d },
            mip_levels: mips,
            sample_count: 1,
            usage: TextureUsage::ALL,
        };
        let r = region(1, mip, (ox, oy, oz), (ew, eh, ed));
        let _ = validate_texture(&caps(), &t);
        let _ = validate_texture_region(&t, &r);
        let _ = packed_size(t.format, r.extent);
        let b = BufferDesc { size: bsize, usage: BufferUsage::ALL };
        let s = BufferSlice { buffer: BufferId::new(1, 0), offset: boff, size: blen };
        let _ = validate_buffer_slice(&b, &s);
        let _ = validate_copy(
            Resolved::Buffer { desc: &b, slice: &s },
            Resolved::Texture { desc: &t, region: &r },
        );
        let _ = validate_upload(blen, Resolved::Texture { desc: &t, region: &r });
        let _ = validate_rect(w, h, &Rect { x: rx, y: ry, width: ew, height: eh });
        let v = ViewDesc {
            format: PixelFormat::ALL[format],
            dimension: ViewDimension::D2Array,
            base_mip: mip, mip_count: ew, base_layer: oz, layer_count: ed,
            swizzle: Swizzle::IDENTITY,
        };
        let _ = validate_view(&caps(), &t, &v);
    }

    /// A region that validates lies inside its mip level, so `packed_size` of it is the bytes
    /// a tightly packed copy of it holds.
    #[test]
    fn valid_regions_fit_their_mip(
        w in 1..64_u32, h in 1..64_u32, layers in 1..8_u32, mip in 0..7_u32,
        ox in 0..64_u32, oy in 0..64_u32, oz in 0..8_u32,
        ew in 1..64_u32, eh in 1..64_u32, ed in 1..8_u32,
        format in 0..PixelFormat::ALL.len(),
    ) {
        let t = TextureDesc {
            dimension: TextureDimension::D2,
            format: PixelFormat::ALL[format],
            size: Extent3d { width: w, height: h, depth_or_layers: layers },
            mip_levels: full_mip_chain(w.max(h)),
            sample_count: 1,
            usage: TextureUsage::ALL,
        };
        let r = region(1, mip, (ox, oy, oz), (ew, eh, ed));
        if validate_texture_region(&t, &r).is_ok() {
            let m = t.mip_extent(mip).expect("validated mip");
            prop_assert!(ox + ew <= m.width && oy + eh <= m.height && oz + ed <= m.depth_or_layers);
            let bytes = packed_size(t.format, r.extent).expect("small");
            let blocks = u64::from(ew.div_ceil(t.format.block_width()))
                * u64::from(eh.div_ceil(t.format.block_height()))
                * u64::from(ed);
            prop_assert_eq!(bytes, blocks * u64::from(t.format.bytes_per_block()));
        }
    }
}

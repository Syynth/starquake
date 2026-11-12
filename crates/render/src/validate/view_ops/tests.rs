//! Clear, blit and present checks: each rule both ways, rectangles at the edge on both sides,
//! the documented order with two faults at once, and exact errors.

use super::{ResolvedView, validate_blit, validate_clear, validate_present};
use crate::command::{BlitDesc, ClearDesc, ClearValue, PresentDesc, PresentTransform, Rect};
use crate::error::RenderError;
use crate::format::PixelFormat;
use crate::id::ViewId;
use crate::resource::{
    Extent3d, FilterMode, Swizzle, TextureDesc, TextureDimension, TextureUsage, ViewDesc,
    ViewDimension,
};

const W: u32 = 16;
const H: u32 = 8;

fn texture(format: PixelFormat, usage: TextureUsage) -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        format,
        size: Extent3d {
            width: W,
            height: H,
            depth_or_layers: 2,
        },
        mip_levels: 2,
        sample_count: 1,
        usage,
    }
}

fn view(format: PixelFormat, base_mip: u32) -> ViewDesc {
    ViewDesc {
        format,
        dimension: ViewDimension::D2Array,
        base_mip,
        mip_count: 1,
        base_layer: 0,
        layer_count: 2,
        swizzle: Swizzle::IDENTITY,
    }
}

fn all_usage() -> TextureUsage {
    TextureUsage::ALL
}

fn rect(x: u32, y: u32, width: u32, height: u32) -> Rect {
    Rect {
        x,
        y,
        width,
        height,
    }
}

fn clear(value: ClearValue, r: Option<Rect>) -> ClearDesc {
    ClearDesc {
        target: ViewId::new(0, 0),
        value,
        rect: r,
    }
}

fn invalid(text: &'static str) -> Result<(), RenderError> {
    Err(RenderError::Invalid(text))
}

#[test]
fn clear_rect_at_the_base_mip_edge() {
    let t = texture(PixelFormat::A8B8G8R8Unorm, all_usage());
    let v0 = view(PixelFormat::A8B8G8R8Unorm, 0);
    let v1 = view(PixelFormat::A8B8G8R8Unorm, 1);
    let red = ClearValue::Float([1.0, 0.0, 0.0, 1.0]);
    let at = |v: &ViewDesc, r| {
        validate_clear(
            ResolvedView {
                texture: &t,
                view: v,
            },
            &clear(red, r),
        )
    };
    assert_eq!(at(&v0, None), Ok(()));
    assert_eq!(at(&v0, Some(rect(0, 0, W, H))), Ok(()));
    assert_eq!(
        at(&v0, Some(rect(1, 0, W, H))),
        Err(RenderError::PastEnd {
            what: "rectangle width",
            end: u64::from(W) + 1,
            limit: W.into()
        })
    );
    assert_eq!(
        at(&v0, Some(rect(0, 1, W, H))),
        Err(RenderError::PastEnd {
            what: "rectangle height",
            end: u64::from(H) + 1,
            limit: H.into()
        })
    );
    // Mip 1 is half the size.
    assert_eq!(at(&v1, Some(rect(0, 0, W / 2, H / 2))), Ok(()));
    assert_eq!(
        at(&v1, Some(rect(0, 0, W / 2 + 1, H / 2))),
        Err(RenderError::PastEnd {
            what: "rectangle width",
            end: u64::from(W / 2) + 1,
            limit: (W / 2).into()
        })
    );
    assert_eq!(
        at(&v0, Some(rect(u32::MAX, 0, u32::MAX, 1))),
        Err(RenderError::PastEnd {
            what: "rectangle width",
            end: u64::from(u32::MAX) * 2,
            limit: W.into()
        }),
        "no overflow"
    );
    for empty in [rect(0, 0, 0, 1), rect(0, 0, 1, 0)] {
        assert_eq!(
            at(&v0, Some(empty)),
            invalid("a rectangle must not be empty")
        );
    }
    // A view whose base mip the texture lacks.
    let past = view(PixelFormat::A8B8G8R8Unorm, 2);
    assert_eq!(
        at(&past, None),
        Err(RenderError::NotBelow {
            what: "mip level",
            index: 2,
            count: 2
        })
    );
}

#[test]
fn clear_value_must_suit_the_format() {
    let suits = [
        (PixelFormat::A8B8G8R8Unorm, ClearValue::Float([0.0; 4])),
        (PixelFormat::A8B8G8R8Snorm, ClearValue::Float([0.0; 4])),
        (PixelFormat::R32Float, ClearValue::Float([0.0; 4])),
        (PixelFormat::A8B8G8R8Uint, ClearValue::Uint([0; 4])),
        (PixelFormat::A8B8G8R8Sint, ClearValue::Sint([0; 4])),
        (PixelFormat::D32Float, ClearValue::Depth(0.0)),
        (PixelFormat::S8Uint, ClearValue::Stencil(0)),
        (PixelFormat::D24UnormS8Uint, ClearValue::Depth(0.0)),
        (PixelFormat::D24UnormS8Uint, ClearValue::Stencil(0)),
        (
            PixelFormat::D24UnormS8Uint,
            ClearValue::DepthStencil {
                depth: 0.0,
                stencil: 0,
            },
        ),
    ];
    let wrong = [
        (PixelFormat::A8B8G8R8Unorm, ClearValue::Uint([0; 4])),
        (PixelFormat::A8B8G8R8Unorm, ClearValue::Sint([0; 4])),
        (PixelFormat::A8B8G8R8Unorm, ClearValue::Depth(0.0)),
        (PixelFormat::A8B8G8R8Uint, ClearValue::Float([0.0; 4])),
        (PixelFormat::A8B8G8R8Uint, ClearValue::Sint([0; 4])),
        (PixelFormat::A8B8G8R8Sint, ClearValue::Uint([0; 4])),
        (PixelFormat::D32Float, ClearValue::Stencil(0)),
        (
            PixelFormat::D32Float,
            ClearValue::DepthStencil {
                depth: 0.0,
                stencil: 0,
            },
        ),
        (PixelFormat::D32Float, ClearValue::Float([0.0; 4])),
        (PixelFormat::S8Uint, ClearValue::Depth(0.0)),
        (PixelFormat::D24UnormS8Uint, ClearValue::Float([0.0; 4])),
    ];
    for (format, value) in suits {
        let t = texture(format, all_usage());
        let v = view(format, 0);
        assert_eq!(
            validate_clear(
                ResolvedView {
                    texture: &t,
                    view: &v
                },
                &clear(value, None)
            ),
            Ok(()),
            "{format} {value:?}"
        );
    }
    for (format, value) in wrong {
        let t = texture(format, all_usage());
        let v = view(format, 0);
        assert_eq!(
            validate_clear(
                ResolvedView {
                    texture: &t,
                    view: &v
                },
                &clear(value, None)
            ),
            invalid("the clear value does not suit the view's format"),
            "{format} {value:?}"
        );
    }
}

#[test]
fn clear_needs_render_target_usage_and_an_uncompressed_format() {
    let f = PixelFormat::A8B8G8R8Unorm;
    let v = view(f, 0);
    let value = ClearValue::Float([0.0; 4]);
    let without = texture(f, all_usage().difference(TextureUsage::RENDER_TARGET));
    assert_eq!(
        validate_clear(
            ResolvedView {
                texture: &without,
                view: &v
            },
            &clear(value, None)
        ),
        Err(RenderError::MissingUsage {
            resource: "cleared texture",
            usage: "render target"
        })
    );
    let bc = PixelFormat::Bc1RgbaUnorm;
    let t = texture(bc, all_usage());
    let bv = view(bc, 0);
    assert_eq!(
        validate_clear(
            ResolvedView {
                texture: &t,
                view: &bv
            },
            &clear(value, None)
        ),
        invalid("a compressed texture cannot be cleared")
    );
    // Order: usage before the format, the format before the value, the value before the rect.
    let t = texture(bc, all_usage().difference(TextureUsage::RENDER_TARGET));
    assert_eq!(
        validate_clear(
            ResolvedView {
                texture: &t,
                view: &bv
            },
            &clear(value, None)
        ),
        Err(RenderError::MissingUsage {
            resource: "cleared texture",
            usage: "render target"
        })
    );
    let t = texture(bc, all_usage());
    assert_eq!(
        validate_clear(
            ResolvedView {
                texture: &t,
                view: &bv
            },
            &clear(ClearValue::Uint([0; 4]), Some(rect(0, 0, 0, 0)))
        ),
        invalid("a compressed texture cannot be cleared")
    );
    let t = texture(f, all_usage());
    assert_eq!(
        validate_clear(
            ResolvedView {
                texture: &t,
                view: &v
            },
            &clear(ClearValue::Uint([0; 4]), Some(rect(0, 0, 0, 0)))
        ),
        invalid("the clear value does not suit the view's format")
    );
}

fn blit(src_rect: Rect, dst_rect: Rect, filter: FilterMode) -> BlitDesc {
    BlitDesc {
        src: ViewId::new(0, 0),
        src_rect,
        dst: ViewId::new(1, 0),
        dst_rect,
        filter,
    }
}

fn check_blit(
    (st, sv): (&TextureDesc, &ViewDesc),
    (dt, dv): (&TextureDesc, &ViewDesc),
    b: &BlitDesc,
) -> Result<(), RenderError> {
    validate_blit(
        ResolvedView {
            texture: st,
            view: sv,
        },
        ResolvedView {
            texture: dt,
            view: dv,
        },
        b,
    )
}

#[test]
fn blit_rules() {
    let f = PixelFormat::A8B8G8R8Unorm;
    let t = texture(f, all_usage());
    let v = view(f, 0);
    let whole = rect(0, 0, W, H);
    let ok = blit(whole, rect(0, 0, W / 2, H / 2), FilterMode::Linear);
    assert_eq!(check_blit((&t, &v), (&t, &v), &ok), Ok(()));

    let no_src = texture(f, all_usage().difference(TextureUsage::COPY_SRC));
    let no_dst = texture(f, all_usage().difference(TextureUsage::COPY_DST));
    assert_eq!(
        check_blit((&no_src, &v), (&no_dst, &v), &ok),
        Err(RenderError::MissingUsage {
            resource: "blit source texture",
            usage: "copy source"
        }),
        "source usage first"
    );
    assert_eq!(
        check_blit((&t, &v), (&no_dst, &v), &ok),
        Err(RenderError::MissingUsage {
            resource: "blit destination texture",
            usage: "copy destination"
        })
    );
    let ms = TextureDesc {
        sample_count: 4,
        mip_levels: 1,
        ..t
    };
    let ms_view = ViewDesc { ..v };
    assert_eq!(
        check_blit((&ms, &ms_view), (&ms, &ms_view), &ok),
        invalid("a blit source must have one sample")
    );
    assert_eq!(
        check_blit((&t, &v), (&ms, &ms_view), &ok),
        invalid("a blit destination must have one sample")
    );

    // Aspect, then integer kind.
    let d = PixelFormat::D32Float;
    let dt = texture(d, all_usage());
    let dv = view(d, 0);
    assert_eq!(
        check_blit((&t, &v), (&dt, &dv), &ok),
        invalid("a blit needs formats with the same aspect")
    );
    for (s, other) in [
        (PixelFormat::A8B8G8R8Uint, PixelFormat::A8B8G8R8Unorm),
        (PixelFormat::A8B8G8R8Unorm, PixelFormat::A8B8G8R8Sint),
        (PixelFormat::A8B8G8R8Uint, PixelFormat::A8B8G8R8Sint),
    ] {
        let (st, sv) = (texture(s, all_usage()), view(s, 0));
        let (ot, ov) = (texture(other, all_usage()), view(other, 0));
        assert_eq!(
            check_blit((&st, &sv), (&ot, &ov), &ok),
            invalid("a blit needs formats of the same integer kind"),
            "{s} to {other}"
        );
    }
    // Same integer kind, different formats: allowed.
    let (ut, uv) = (
        texture(PixelFormat::R32Uint, all_usage()),
        view(PixelFormat::R32Uint, 0),
    );
    let (u8t, u8v) = (
        texture(PixelFormat::A8B8G8R8Uint, all_usage()),
        view(PixelFormat::A8B8G8R8Uint, 0),
    );
    assert_eq!(check_blit((&ut, &uv), (&u8t, &u8v), &ok), Ok(()));
}

#[test]
fn blit_depth_and_rectangles() {
    let f = PixelFormat::A8B8G8R8Unorm;
    let t = texture(f, all_usage());
    let v = view(f, 0);
    let whole = rect(0, 0, W, H);
    let ok = blit(whole, rect(0, 0, W / 2, H / 2), FilterMode::Linear);
    let d = PixelFormat::D32Float;
    let dt = texture(d, all_usage());
    let dv = view(d, 0);
    // Depth: identical formats and the nearest filter.
    let nearest = blit(whole, whole, FilterMode::Nearest);
    assert_eq!(check_blit((&dt, &dv), (&dt, &dv), &nearest), Ok(()));
    assert_eq!(
        check_blit((&dt, &dv), (&dt, &dv), &ok),
        invalid("a depth or stencil blit needs the nearest filter")
    );
    let d16 = PixelFormat::D16Unorm;
    let (half_tex, half_view) = (texture(d16, all_usage()), view(d16, 0));
    assert_eq!(
        check_blit((&dt, &dv), (&half_tex, &half_view), &ok),
        invalid("a depth or stencil blit needs identical formats"),
        "formats before the filter"
    );

    // Rectangles: the source's is checked before the destination's.
    let past = rect(1, 0, W, H);
    assert_eq!(
        check_blit((&t, &v), (&t, &v), &blit(past, past, FilterMode::Linear)),
        Err(RenderError::PastEnd {
            what: "rectangle width",
            end: u64::from(W) + 1,
            limit: W.into()
        })
    );
    assert_eq!(
        check_blit(
            (&t, &v),
            (&t, &v),
            &blit(whole, rect(0, 0, 0, 1), FilterMode::Linear)
        ),
        invalid("a rectangle must not be empty")
    );
    assert_eq!(
        check_blit(
            (&t, &v),
            (&t, &v),
            &blit(rect(0, 0, 0, 1), past, FilterMode::Linear)
        ),
        invalid("a rectangle must not be empty"),
        "source rect first"
    );
}

fn present(src_rect: Rect) -> PresentDesc {
    PresentDesc {
        view: ViewId::new(0, 0),
        src_rect,
        transform: PresentTransform::Identity,
        frame: 1,
        dump: false,
    }
}

#[test]
fn present_rules() {
    let f = PixelFormat::B8G8R8A8Unorm;
    let t = texture(f, all_usage());
    let v = view(f, 0);
    let at = |t: &TextureDesc, v: &ViewDesc, r| {
        validate_present(
            ResolvedView {
                texture: t,
                view: v,
            },
            &present(r),
        )
    };
    assert_eq!(at(&t, &v, rect(0, 0, W, H)), Ok(()));
    assert_eq!(
        at(&t, &v, rect(0, 0, W, H + 1)),
        Err(RenderError::PastEnd {
            what: "rectangle height",
            end: u64::from(H) + 1,
            limit: H.into()
        })
    );
    let no_src = texture(f, all_usage().difference(TextureUsage::COPY_SRC));
    assert_eq!(
        at(&no_src, &v, rect(0, 0, 0, 0)),
        Err(RenderError::MissingUsage {
            resource: "presented texture",
            usage: "copy source"
        })
    );
    let ms = TextureDesc {
        sample_count: 4,
        mip_levels: 1,
        ..t
    };
    assert_eq!(
        at(&ms, &v, rect(0, 0, W, H)),
        invalid("a presented texture must have one sample")
    );
    let d = PixelFormat::D32Float;
    assert_eq!(
        at(&texture(d, all_usage()), &view(d, 0), rect(0, 0, 0, 0)),
        invalid("a presented view must store colour"),
        "aspect before the rectangle"
    );
}

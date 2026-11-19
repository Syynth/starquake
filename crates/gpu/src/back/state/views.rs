//! Typed views of a mirror: [`ThreeD`] for the 3D class (`0xC797`) and [`Compute`] for the
//! compute class (`0xC7C0`), with one view per state group (`three_d.viewport(i)?.scale_x()`).
//!
//! Every getter reads through the generated tables of [`crate::classes`]: the method's address
//! and stride from its generated [`Method`], the field's bits from its generated [`Field`]
//! constant, and, for a field with named values, the values from the generated
//! [`crate::classes::NamedField`] list ([`Reg::named`]). Nothing here restates a bit position,
//! stride or value; the getters only name which generated constants to read. Indices are bounded
//! by the array lengths in `group.rs`, the same constants the group table uses.
//!
//! Getters return `Result`: an index past its array is [`StateError::Index`], and a named field
//! holding a value the header does not define is [`StateError::UndefinedValue`]. A float getter
//! reads one of the methods Mesa lists as floats (`METHOD_IS_FLOAT` in
//! `mesa-nouveau/src/nouveau/headers/class_parser.py`): `SET_VIEWPORT_SCALE_*`,
//! `SET_VIEWPORT_OFFSET_*`, `SET_VIEWPORT_CLIP_MIN_Z`, `SET_VIEWPORT_CLIP_MAX_Z`,
//! `SET_BLEND_CONST_*`, `SET_DEPTH_BIAS`, `SET_SLOPE_SCALE_DEPTH_BIAS`, `SET_DEPTH_BIAS_CLAMP`,
//! `SET_DEPTH_BOUNDS_MIN` and `_MAX`, `SET_LINE_WIDTH_FLOAT`, `SET_ALIASED_LINE_WIDTH_FLOAT` and
//! `SET_POINT_SIZE`.

use crate::classes::{Method, Value, clc7c0, clc797};
use crate::field::Field;

use super::group::{
    BIND_GROUPS, BLEND_TARGETS, COLOR_TARGETS, CONSTANT_BUFFER_SLOTS, PIPELINES, SCISSORS,
    VERTEX_ATTRIBUTES, VERTEX_ATTRIBUTES_PER_HALF, VERTEX_STREAM_SIZES, VERTEX_STREAMS,
    VERTEX_STREAMS_PER_HALF, VIEWPORTS,
};
use super::mirror::{ConstantBufferBinding, Mirror, Reg, StateError, join};

/// The array names in [`StateError::Index`], one per bounded accessor.
pub mod arrays {
    /// [`super::ThreeD::viewport`].
    pub const VIEWPORT: &str = "viewport";
    /// [`super::ThreeD::scissor`].
    pub const SCISSOR: &str = "scissor";
    /// [`super::ThreeD::blend_per_target`].
    pub const BLEND_TARGET: &str = "blend target";
    /// [`super::ThreeD::color_target`].
    pub const COLOR_TARGET: &str = "colour target";
    /// [`super::CtSelect::target`].
    pub const CT_SELECT_TARGET: &str = "colour target select entry";
    /// [`super::ThreeD::vertex_stream`].
    pub const VERTEX_STREAM: &str = "vertex stream";
    /// [`super::VertexStream::size`].
    pub const VERTEX_STREAM_SIZE: &str = "vertex stream size";
    /// [`super::ThreeD::vertex_attribute`].
    pub const VERTEX_ATTRIBUTE: &str = "vertex attribute";
    /// [`super::ThreeD::bind_group`].
    pub const BIND_GROUP: &str = "bind group";
    /// [`super::BindGroup::constant_buffer`].
    pub const CONSTANT_BUFFER_SLOT: &str = "constant buffer slot";
    /// [`super::ThreeD::pipeline`].
    pub const PIPELINE: &str = "pipeline";
}

/// `Ok(index)` below `count`, or the index error for `array`.
fn bounded(array: &'static str, index: u32, count: u32) -> Result<u32, StateError> {
    if index < count {
        Ok(index)
    } else {
        Err(StateError::Index {
            array,
            index,
            count,
        })
    }
}

/// A 64-bit address from an upper and a lower register.
fn address(
    upper: Reg,
    upper_field: Field,
    lower: Reg,
    lower_field: Field,
) -> Result<u64, StateError> {
    Ok(join(upper.field(upper_field)?, lower.field(lower_field)?))
}

/// Getters of a view, each reading one generated method through the view's `at`.
macro_rules! getters {
    ($view:ident { $($body:tt)* }) => {
        impl $view<'_> {
            getters!(@each $($body)*);
        }
    };
    (@each) => {};
    (@each $(#[$doc:meta])* $name:ident: float $method:expr; $($rest:tt)*) => {
        $(#[$doc])*
        pub fn $name(&self) -> Result<f32, StateError> {
            Ok(self.at(&$method)?.float())
        }
        getters!(@each $($rest)*);
    };
    (@each $(#[$doc:meta])* $name:ident: field $method:expr, $field:expr; $($rest:tt)*) => {
        $(#[$doc])*
        pub fn $name(&self) -> Result<u32, StateError> {
            self.at(&$method)?.field($field)
        }
        getters!(@each $($rest)*);
    };
    (@each $(#[$doc:meta])* $name:ident: named $method:expr, $field:expr; $($rest:tt)*) => {
        $(#[$doc])*
        pub fn $name(&self) -> Result<&'static Value, StateError> {
            self.at(&$method)?.named($field)
        }
        getters!(@each $($rest)*);
    };
    (@each $(#[$doc:meta])* $name:ident: address $upper:expr, $upper_field:expr, $lower:expr, $lower_field:expr; $($rest:tt)*) => {
        $(#[$doc])*
        pub fn $name(&self) -> Result<u64, StateError> {
            address(self.at(&$upper)?, $upper_field, self.at(&$lower)?, $lower_field)
        }
        getters!(@each $($rest)*);
    };
}

/// A view of one element of the 3D class's arrays: element `index` of each method read.
macro_rules! element_view {
    ($(#[$doc:meta])* $view:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $view<'a> {
            mirror: &'a Mirror,
            index: u32,
        }

        impl<'a> $view<'a> {
            /// The element's index.
            pub fn index(&self) -> u32 {
                self.index
            }

            fn at(&self, method: &'static Method) -> Result<Reg, StateError> {
                self.mirror.reg(method, &[self.index])
            }
        }
    };
}

/// A view of registers that are not arrays.
macro_rules! single_view {
    ($(#[$doc:meta])* $view:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $view<'a> {
            mirror: &'a Mirror,
        }

        impl<'a> $view<'a> {
            fn at(&self, method: &'static Method) -> Result<Reg, StateError> {
                self.mirror.reg(method, &[])
            }
        }
    };
}

impl Mirror {
    /// The 3D view of a 3D mirror.
    ///
    /// # Errors
    ///
    /// [`StateError::WrongClass`] for a mirror of another class.
    pub fn three_d(&self) -> Result<ThreeD<'_>, StateError> {
        self.check_class(clc797::CLASS)?;
        Ok(ThreeD { mirror: self })
    }

    /// The compute view of a compute mirror.
    ///
    /// # Errors
    ///
    /// [`StateError::WrongClass`] for a mirror of another class.
    pub fn compute(&self) -> Result<Compute<'_>, StateError> {
        self.check_class(clc7c0::CLASS)?;
        Ok(Compute { mirror: self })
    }

    fn check_class(&self, expected: u32) -> Result<(), StateError> {
        let actual = self.class().0;
        if actual == expected {
            Ok(())
        } else {
            Err(StateError::WrongClass { expected, actual })
        }
    }
}

/// The 3D class's state, by group ([`Mirror::three_d`]).
#[derive(Clone, Copy, Debug)]
pub struct ThreeD<'a> {
    mirror: &'a Mirror,
}

impl<'a> ThreeD<'a> {
    /// Viewport `index`, below [`VIEWPORTS`].
    pub fn viewport(&self, index: u32) -> Result<Viewport<'a>, StateError> {
        let index = bounded(arrays::VIEWPORT, index, VIEWPORTS)?;
        Ok(Viewport {
            mirror: self.mirror,
            index,
        })
    }

    /// Scissor `index`, below [`SCISSORS`].
    pub fn scissor(&self, index: u32) -> Result<Scissor<'a>, StateError> {
        let index = bounded(arrays::SCISSOR, index, SCISSORS)?;
        Ok(Scissor {
            mirror: self.mirror,
            index,
        })
    }

    /// The blend state shared by every target.
    pub fn blend(&self) -> Blend<'a> {
        Blend {
            mirror: self.mirror,
        }
    }

    /// Blend target `index`, below [`BLEND_TARGETS`].
    pub fn blend_per_target(&self, index: u32) -> Result<BlendPerTarget<'a>, StateError> {
        let index = bounded(arrays::BLEND_TARGET, index, BLEND_TARGETS)?;
        Ok(BlendPerTarget {
            mirror: self.mirror,
            index,
        })
    }

    /// Colour target `index`, below [`COLOR_TARGETS`].
    pub fn color_target(&self, index: u32) -> Result<ColorTarget<'a>, StateError> {
        let index = bounded(arrays::COLOR_TARGET, index, COLOR_TARGETS)?;
        Ok(ColorTarget {
            mirror: self.mirror,
            index,
        })
    }

    /// Which colour targets are drawn to.
    pub fn ct_select(&self) -> CtSelect<'a> {
        CtSelect {
            mirror: self.mirror,
        }
    }

    /// The depth target.
    pub fn zt(&self) -> Zt<'a> {
        Zt {
            mirror: self.mirror,
        }
    }

    /// Vertex stream `index`, below [`VERTEX_STREAMS`].
    pub fn vertex_stream(&self, index: u32) -> Result<VertexStream<'a>, StateError> {
        let index = bounded(arrays::VERTEX_STREAM, index, VERTEX_STREAMS)?;
        Ok(VertexStream {
            mirror: self.mirror,
            index,
        })
    }

    /// Vertex attribute `index`, below [`VERTEX_ATTRIBUTES`].
    pub fn vertex_attribute(&self, index: u32) -> Result<VertexAttribute<'a>, StateError> {
        let index = bounded(arrays::VERTEX_ATTRIBUTE, index, VERTEX_ATTRIBUTES)?;
        Ok(VertexAttribute {
            mirror: self.mirror,
            index,
        })
    }

    /// The index buffer.
    pub fn index_buffer(&self) -> IndexBuffer<'a> {
        IndexBuffer {
            mirror: self.mirror,
        }
    }

    /// Bind group `index`, below [`BIND_GROUPS`].
    pub fn bind_group(&self, index: u32) -> Result<BindGroup<'a>, StateError> {
        let index = bounded(arrays::BIND_GROUP, index, BIND_GROUPS)?;
        Ok(BindGroup {
            mirror: self.mirror,
            index,
        })
    }

    /// The texture header and sampler pools.
    pub fn tex_pools(&self) -> TexPools<'a> {
        TexPools {
            mirror: self.mirror,
            methods: &TEX_POOLS_3D,
        }
    }

    /// Pipeline `index`, below [`PIPELINES`].
    pub fn pipeline(&self, index: u32) -> Result<Pipeline<'a>, StateError> {
        let index = bounded(arrays::PIPELINE, index, PIPELINES)?;
        Ok(Pipeline {
            mirror: self.mirror,
            index,
        })
    }

    /// The rasterizer state.
    pub fn rasterizer(&self) -> Rasterizer<'a> {
        Rasterizer {
            mirror: self.mirror,
        }
    }

    /// The depth and stencil state.
    pub fn depth_stencil(&self) -> DepthStencil<'a> {
        DepthStencil {
            mirror: self.mirror,
        }
    }
}

element_view! {
    /// One viewport ([`ThreeD::viewport`]).
    Viewport
}

getters!(Viewport {
    /// `SET_VIEWPORT_SCALE_X`.
    scale_x: float clc797::SET_VIEWPORT_SCALE_X;
    /// `SET_VIEWPORT_SCALE_Y`.
    scale_y: float clc797::SET_VIEWPORT_SCALE_Y;
    /// `SET_VIEWPORT_SCALE_Z`.
    scale_z: float clc797::SET_VIEWPORT_SCALE_Z;
    /// `SET_VIEWPORT_OFFSET_X`.
    offset_x: float clc797::SET_VIEWPORT_OFFSET_X;
    /// `SET_VIEWPORT_OFFSET_Y`.
    offset_y: float clc797::SET_VIEWPORT_OFFSET_Y;
    /// `SET_VIEWPORT_OFFSET_Z`.
    offset_z: float clc797::SET_VIEWPORT_OFFSET_Z;
    /// `SET_VIEWPORT_COORDINATE_SWIZZLE_X`.
    swizzle_x: named clc797::SET_VIEWPORT_COORDINATE_SWIZZLE, clc797::SET_VIEWPORT_COORDINATE_SWIZZLE_X;
    /// `SET_VIEWPORT_COORDINATE_SWIZZLE_Y`.
    swizzle_y: named clc797::SET_VIEWPORT_COORDINATE_SWIZZLE, clc797::SET_VIEWPORT_COORDINATE_SWIZZLE_Y;
    /// `SET_VIEWPORT_COORDINATE_SWIZZLE_Z`.
    swizzle_z: named clc797::SET_VIEWPORT_COORDINATE_SWIZZLE, clc797::SET_VIEWPORT_COORDINATE_SWIZZLE_Z;
    /// `SET_VIEWPORT_COORDINATE_SWIZZLE_W`.
    swizzle_w: named clc797::SET_VIEWPORT_COORDINATE_SWIZZLE, clc797::SET_VIEWPORT_COORDINATE_SWIZZLE_W;
    /// `SET_VIEWPORT_CLIP_HORIZONTAL_X0`.
    clip_x0: field clc797::SET_VIEWPORT_CLIP_HORIZONTAL, clc797::SET_VIEWPORT_CLIP_HORIZONTAL_X0;
    /// `SET_VIEWPORT_CLIP_HORIZONTAL_WIDTH`.
    clip_width: field clc797::SET_VIEWPORT_CLIP_HORIZONTAL, clc797::SET_VIEWPORT_CLIP_HORIZONTAL_WIDTH;
    /// `SET_VIEWPORT_CLIP_VERTICAL_Y0`.
    clip_y0: field clc797::SET_VIEWPORT_CLIP_VERTICAL, clc797::SET_VIEWPORT_CLIP_VERTICAL_Y0;
    /// `SET_VIEWPORT_CLIP_VERTICAL_HEIGHT`.
    clip_height: field clc797::SET_VIEWPORT_CLIP_VERTICAL, clc797::SET_VIEWPORT_CLIP_VERTICAL_HEIGHT;
    /// `SET_VIEWPORT_CLIP_MIN_Z`.
    clip_min_z: float clc797::SET_VIEWPORT_CLIP_MIN_Z;
    /// `SET_VIEWPORT_CLIP_MAX_Z`.
    clip_max_z: float clc797::SET_VIEWPORT_CLIP_MAX_Z;
});

element_view! {
    /// One scissor ([`ThreeD::scissor`]).
    Scissor
}

getters!(Scissor {
    /// `SET_SCISSOR_ENABLE_V`.
    enable: named clc797::SET_SCISSOR_ENABLE, clc797::SET_SCISSOR_ENABLE_V;
    /// `SET_SCISSOR_HORIZONTAL_XMIN`.
    xmin: field clc797::SET_SCISSOR_HORIZONTAL, clc797::SET_SCISSOR_HORIZONTAL_XMIN;
    /// `SET_SCISSOR_HORIZONTAL_XMAX`.
    xmax: field clc797::SET_SCISSOR_HORIZONTAL, clc797::SET_SCISSOR_HORIZONTAL_XMAX;
    /// `SET_SCISSOR_VERTICAL_YMIN`.
    ymin: field clc797::SET_SCISSOR_VERTICAL, clc797::SET_SCISSOR_VERTICAL_YMIN;
    /// `SET_SCISSOR_VERTICAL_YMAX`.
    ymax: field clc797::SET_SCISSOR_VERTICAL, clc797::SET_SCISSOR_VERTICAL_YMAX;
});

single_view! {
    /// The blend state shared by every target ([`ThreeD::blend`]).
    Blend
}

getters!(Blend {
    /// `SET_BLEND_STATE_PER_TARGET_ENABLE`.
    state_per_target: named clc797::SET_BLEND_STATE_PER_TARGET, clc797::SET_BLEND_STATE_PER_TARGET_ENABLE;
    /// `SET_BLEND_SEPARATE_FOR_ALPHA_ENABLE`.
    separate_for_alpha: named clc797::SET_BLEND_SEPARATE_FOR_ALPHA, clc797::SET_BLEND_SEPARATE_FOR_ALPHA_ENABLE;
    /// `SET_BLEND_COLOR_OP_V`.
    color_op: named clc797::SET_BLEND_COLOR_OP, clc797::SET_BLEND_COLOR_OP_V;
    /// `SET_BLEND_COLOR_SOURCE_COEFF_V`.
    color_source_coeff: named clc797::SET_BLEND_COLOR_SOURCE_COEFF, clc797::SET_BLEND_COLOR_SOURCE_COEFF_V;
    /// `SET_BLEND_COLOR_DEST_COEFF_V`.
    color_dest_coeff: named clc797::SET_BLEND_COLOR_DEST_COEFF, clc797::SET_BLEND_COLOR_DEST_COEFF_V;
    /// `SET_BLEND_ALPHA_OP_V`.
    alpha_op: named clc797::SET_BLEND_ALPHA_OP, clc797::SET_BLEND_ALPHA_OP_V;
    /// `SET_BLEND_ALPHA_SOURCE_COEFF_V`.
    alpha_source_coeff: named clc797::SET_BLEND_ALPHA_SOURCE_COEFF, clc797::SET_BLEND_ALPHA_SOURCE_COEFF_V;
    /// `SET_BLEND_ALPHA_DEST_COEFF_V`.
    alpha_dest_coeff: named clc797::SET_BLEND_ALPHA_DEST_COEFF, clc797::SET_BLEND_ALPHA_DEST_COEFF_V;
    /// `SET_BLEND_CONST_RED`.
    const_red: float clc797::SET_BLEND_CONST_RED;
    /// `SET_BLEND_CONST_GREEN`.
    const_green: float clc797::SET_BLEND_CONST_GREEN;
    /// `SET_BLEND_CONST_BLUE`.
    const_blue: float clc797::SET_BLEND_CONST_BLUE;
    /// `SET_BLEND_CONST_ALPHA`.
    const_alpha: float clc797::SET_BLEND_CONST_ALPHA;
    /// `SET_SINGLE_CT_WRITE_CONTROL_ENABLE`.
    single_ct_write_control: named clc797::SET_SINGLE_CT_WRITE_CONTROL, clc797::SET_SINGLE_CT_WRITE_CONTROL_ENABLE;
});

element_view! {
    /// One blend target ([`ThreeD::blend_per_target`]).
    BlendPerTarget
}

getters!(BlendPerTarget {
    /// `SET_BLEND_ENABLE` of this target.
    enable: named clc797::SET_BLEND, clc797::SET_BLEND_ENABLE;
    /// `SET_CT_WRITE_R_ENABLE`.
    write_red: named clc797::SET_CT_WRITE, clc797::SET_CT_WRITE_R_ENABLE;
    /// `SET_CT_WRITE_G_ENABLE`.
    write_green: named clc797::SET_CT_WRITE, clc797::SET_CT_WRITE_G_ENABLE;
    /// `SET_CT_WRITE_B_ENABLE`.
    write_blue: named clc797::SET_CT_WRITE, clc797::SET_CT_WRITE_B_ENABLE;
    /// `SET_CT_WRITE_A_ENABLE`.
    write_alpha: named clc797::SET_CT_WRITE, clc797::SET_CT_WRITE_A_ENABLE;
    /// `SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA_ENABLE`.
    separate_for_alpha: named clc797::SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA, clc797::SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA_ENABLE;
    /// `SET_BLEND_PER_TARGET_COLOR_OP_V`.
    color_op: named clc797::SET_BLEND_PER_TARGET_COLOR_OP, clc797::SET_BLEND_PER_TARGET_COLOR_OP_V;
    /// `SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V`.
    color_source_coeff: named clc797::SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF, clc797::SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V;
    /// `SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V`.
    color_dest_coeff: named clc797::SET_BLEND_PER_TARGET_COLOR_DEST_COEFF, clc797::SET_BLEND_PER_TARGET_COLOR_DEST_COEFF_V;
    /// `SET_BLEND_PER_TARGET_ALPHA_OP_V`.
    alpha_op: named clc797::SET_BLEND_PER_TARGET_ALPHA_OP, clc797::SET_BLEND_PER_TARGET_ALPHA_OP_V;
    /// `SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V`.
    alpha_source_coeff: named clc797::SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF, clc797::SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF_V;
    /// `SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V`.
    alpha_dest_coeff: named clc797::SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF, clc797::SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF_V;
});

element_view! {
    /// One colour target ([`ThreeD::color_target`]).
    ColorTarget
}

getters!(ColorTarget {
    /// `SET_COLOR_TARGET_A_OFFSET_UPPER` above `SET_COLOR_TARGET_B_OFFSET_LOWER`.
    address: address clc797::SET_COLOR_TARGET_A, clc797::SET_COLOR_TARGET_A_OFFSET_UPPER, clc797::SET_COLOR_TARGET_B, clc797::SET_COLOR_TARGET_B_OFFSET_LOWER;
    /// `SET_COLOR_TARGET_WIDTH_V`.
    width: field clc797::SET_COLOR_TARGET_WIDTH, clc797::SET_COLOR_TARGET_WIDTH_V;
    /// `SET_COLOR_TARGET_HEIGHT_V`.
    height: field clc797::SET_COLOR_TARGET_HEIGHT, clc797::SET_COLOR_TARGET_HEIGHT_V;
    /// `SET_COLOR_TARGET_FORMAT_V`.
    format: named clc797::SET_COLOR_TARGET_FORMAT, clc797::SET_COLOR_TARGET_FORMAT_V;
    /// `SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH`.
    block_width: named clc797::SET_COLOR_TARGET_MEMORY, clc797::SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH;
    /// `SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT`.
    block_height: named clc797::SET_COLOR_TARGET_MEMORY, clc797::SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT;
    /// `SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH`.
    block_depth: named clc797::SET_COLOR_TARGET_MEMORY, clc797::SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH;
    /// `SET_COLOR_TARGET_MEMORY_LAYOUT`.
    layout: named clc797::SET_COLOR_TARGET_MEMORY, clc797::SET_COLOR_TARGET_MEMORY_LAYOUT;
    /// `SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL`.
    third_dimension_control: named clc797::SET_COLOR_TARGET_MEMORY, clc797::SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL;
    /// `SET_COLOR_TARGET_THIRD_DIMENSION_V`.
    third_dimension: field clc797::SET_COLOR_TARGET_THIRD_DIMENSION, clc797::SET_COLOR_TARGET_THIRD_DIMENSION_V;
    /// `SET_COLOR_TARGET_ARRAY_PITCH_V`.
    array_pitch: field clc797::SET_COLOR_TARGET_ARRAY_PITCH, clc797::SET_COLOR_TARGET_ARRAY_PITCH_V;
    /// `SET_COLOR_TARGET_LAYER_OFFSET`.
    layer: field clc797::SET_COLOR_TARGET_LAYER, clc797::SET_COLOR_TARGET_LAYER_OFFSET;
});

single_view! {
    /// Which colour targets are drawn to ([`ThreeD::ct_select`]).
    CtSelect
}

/// `SET_CT_SELECT_TARGET0` to `TARGET7`: entry `k` names the colour target drawn to as output
/// `k`.
const CT_SELECT_TARGETS: [Field; 8] = [
    clc797::SET_CT_SELECT_TARGET0,
    clc797::SET_CT_SELECT_TARGET1,
    clc797::SET_CT_SELECT_TARGET2,
    clc797::SET_CT_SELECT_TARGET3,
    clc797::SET_CT_SELECT_TARGET4,
    clc797::SET_CT_SELECT_TARGET5,
    clc797::SET_CT_SELECT_TARGET6,
    clc797::SET_CT_SELECT_TARGET7,
];

// One select entry per colour target, so the entries' bound is [`COLOR_TARGETS`].
const _: () = assert!(CT_SELECT_TARGETS.len() == COLOR_TARGETS as usize);

getters!(CtSelect {
    /// `SET_CT_SELECT_TARGET_COUNT`.
    target_count: field clc797::SET_CT_SELECT, clc797::SET_CT_SELECT_TARGET_COUNT;
    /// `SET_SURFACE_CLIP_HORIZONTAL_X`.
    surface_clip_x: field clc797::SET_SURFACE_CLIP_HORIZONTAL, clc797::SET_SURFACE_CLIP_HORIZONTAL_X;
    /// `SET_SURFACE_CLIP_HORIZONTAL_WIDTH`.
    surface_clip_width: field clc797::SET_SURFACE_CLIP_HORIZONTAL, clc797::SET_SURFACE_CLIP_HORIZONTAL_WIDTH;
    /// `SET_SURFACE_CLIP_VERTICAL_Y`.
    surface_clip_y: field clc797::SET_SURFACE_CLIP_VERTICAL, clc797::SET_SURFACE_CLIP_VERTICAL_Y;
    /// `SET_SURFACE_CLIP_VERTICAL_HEIGHT`.
    surface_clip_height: field clc797::SET_SURFACE_CLIP_VERTICAL, clc797::SET_SURFACE_CLIP_VERTICAL_HEIGHT;
});

impl CtSelect<'_> {
    /// `SET_CT_SELECT_TARGET<entry>`, for `entry` below [`COLOR_TARGETS`]: the colour target
    /// drawn to as output `entry`.
    pub fn target(&self, entry: u32) -> Result<u32, StateError> {
        let field = usize::try_from(entry)
            .ok()
            .and_then(|entry| CT_SELECT_TARGETS.get(entry))
            .ok_or(StateError::Index {
                array: arrays::CT_SELECT_TARGET,
                index: entry,
                count: COLOR_TARGETS,
            })?;
        self.at(&clc797::SET_CT_SELECT)?.field(*field)
    }
}

single_view! {
    /// The depth target ([`ThreeD::zt`]).
    Zt
}

getters!(Zt {
    /// `SET_ZT_A_OFFSET_UPPER` above `SET_ZT_B_OFFSET_LOWER`.
    address: address clc797::SET_ZT_A, clc797::SET_ZT_A_OFFSET_UPPER, clc797::SET_ZT_B, clc797::SET_ZT_B_OFFSET_LOWER;
    /// `SET_ZT_FORMAT_V`.
    format: named clc797::SET_ZT_FORMAT, clc797::SET_ZT_FORMAT_V;
    /// `SET_ZT_BLOCK_SIZE_WIDTH`.
    block_width: named clc797::SET_ZT_BLOCK_SIZE, clc797::SET_ZT_BLOCK_SIZE_WIDTH;
    /// `SET_ZT_BLOCK_SIZE_HEIGHT`.
    block_height: named clc797::SET_ZT_BLOCK_SIZE, clc797::SET_ZT_BLOCK_SIZE_HEIGHT;
    /// `SET_ZT_BLOCK_SIZE_DEPTH`.
    block_depth: named clc797::SET_ZT_BLOCK_SIZE, clc797::SET_ZT_BLOCK_SIZE_DEPTH;
    /// `SET_ZT_ARRAY_PITCH_V`.
    array_pitch: field clc797::SET_ZT_ARRAY_PITCH, clc797::SET_ZT_ARRAY_PITCH_V;
    /// `SET_ZT_SIZE_A_WIDTH`.
    width: field clc797::SET_ZT_SIZE_A, clc797::SET_ZT_SIZE_A_WIDTH;
    /// `SET_ZT_SIZE_B_HEIGHT`.
    height: field clc797::SET_ZT_SIZE_B, clc797::SET_ZT_SIZE_B_HEIGHT;
    /// `SET_ZT_SIZE_C_THIRD_DIMENSION`.
    third_dimension: field clc797::SET_ZT_SIZE_C, clc797::SET_ZT_SIZE_C_THIRD_DIMENSION;
    /// `SET_ZT_SIZE_C_CONTROL`.
    size_control: named clc797::SET_ZT_SIZE_C, clc797::SET_ZT_SIZE_C_CONTROL;
    /// `SET_ZT_LAYER_OFFSET`.
    layer: field clc797::SET_ZT_LAYER, clc797::SET_ZT_LAYER_OFFSET;
    /// `SET_ZT_SELECT_TARGET_COUNT`.
    target_count: field clc797::SET_ZT_SELECT, clc797::SET_ZT_SELECT_TARGET_COUNT;
});

/// A vertex stream ([`ThreeD::vertex_stream`]): stream `j` below [`VERTEX_STREAMS_PER_HALF`]
/// reads the header's `_A` methods at `j`, the rest its `_B` methods at `j - 16`.
#[derive(Clone, Copy, Debug)]
pub struct VertexStream<'a> {
    mirror: &'a Mirror,
    index: u32,
}

/// One method and field in each half of a two-half array.
#[derive(Clone, Copy)]
struct Halves {
    a: (&'static Method, Field),
    b: (&'static Method, Field),
}

/// Element `index` of a two-half array whose halves hold `per_half` elements each: the field's
/// method and element. The callers bound `index` by both halves together first.
fn half(
    mirror: &Mirror,
    halves: Halves,
    index: u32,
    per_half: u32,
) -> Result<(Reg, Field), StateError> {
    let (method, field, element) = match index.checked_sub(per_half) {
        None => (halves.a.0, halves.a.1, index),
        Some(element) => (halves.b.0, halves.b.1, element),
    };
    Ok((mirror.reg(method, &[element])?, field))
}

impl VertexStream<'_> {
    /// The stream's index.
    pub fn index(&self) -> u32 {
        self.index
    }

    fn read(&self, halves: Halves) -> Result<(Reg, Field), StateError> {
        half(self.mirror, halves, self.index, VERTEX_STREAMS_PER_HALF)
    }

    /// `SET_VERTEX_STREAM_A_FORMAT_ENABLE` (or `_B_`).
    pub fn enable(&self) -> Result<&'static Value, StateError> {
        let (reg, field) = self.read(Halves {
            a: (
                &clc797::SET_VERTEX_STREAM_A_FORMAT,
                clc797::SET_VERTEX_STREAM_A_FORMAT_ENABLE,
            ),
            b: (
                &clc797::SET_VERTEX_STREAM_B_FORMAT,
                clc797::SET_VERTEX_STREAM_B_FORMAT_ENABLE,
            ),
        })?;
        reg.named(field)
    }

    /// `SET_VERTEX_STREAM_A_FORMAT_STRIDE` (or `_B_`).
    pub fn stride(&self) -> Result<u32, StateError> {
        let (reg, field) = self.read(Halves {
            a: (
                &clc797::SET_VERTEX_STREAM_A_FORMAT,
                clc797::SET_VERTEX_STREAM_A_FORMAT_STRIDE,
            ),
            b: (
                &clc797::SET_VERTEX_STREAM_B_FORMAT,
                clc797::SET_VERTEX_STREAM_B_FORMAT_STRIDE,
            ),
        })?;
        reg.field(field)
    }

    /// `SET_VERTEX_STREAM_A_LOCATION_A_OFFSET_UPPER` above `_LOCATION_B_OFFSET_LOWER` (or `_B_`).
    pub fn address(&self) -> Result<u64, StateError> {
        let (upper, upper_field) = self.read(Halves {
            a: (
                &clc797::SET_VERTEX_STREAM_A_LOCATION_A,
                clc797::SET_VERTEX_STREAM_A_LOCATION_A_OFFSET_UPPER,
            ),
            b: (
                &clc797::SET_VERTEX_STREAM_B_LOCATION_A,
                clc797::SET_VERTEX_STREAM_B_LOCATION_A_OFFSET_UPPER,
            ),
        })?;
        let (lower, lower_field) = self.read(Halves {
            a: (
                &clc797::SET_VERTEX_STREAM_A_LOCATION_B,
                clc797::SET_VERTEX_STREAM_A_LOCATION_B_OFFSET_LOWER,
            ),
            b: (
                &clc797::SET_VERTEX_STREAM_B_LOCATION_B,
                clc797::SET_VERTEX_STREAM_B_LOCATION_B_OFFSET_LOWER,
            ),
        })?;
        address(upper, upper_field, lower, lower_field)
    }

    /// `SET_VERTEX_STREAM_A_FREQUENCY_V` (or `_B_`).
    pub fn frequency(&self) -> Result<u32, StateError> {
        let (reg, field) = self.read(Halves {
            a: (
                &clc797::SET_VERTEX_STREAM_A_FREQUENCY,
                clc797::SET_VERTEX_STREAM_A_FREQUENCY_V,
            ),
            b: (
                &clc797::SET_VERTEX_STREAM_B_FREQUENCY,
                clc797::SET_VERTEX_STREAM_B_FREQUENCY_V,
            ),
        })?;
        reg.field(field)
    }

    /// `SET_VERTEX_STREAM_INSTANCE_A_IS_INSTANCED` (or `_B_`).
    pub fn instanced(&self) -> Result<&'static Value, StateError> {
        let (reg, field) = self.read(Halves {
            a: (
                &clc797::SET_VERTEX_STREAM_INSTANCE_A,
                clc797::SET_VERTEX_STREAM_INSTANCE_A_IS_INSTANCED,
            ),
            b: (
                &clc797::SET_VERTEX_STREAM_INSTANCE_B,
                clc797::SET_VERTEX_STREAM_INSTANCE_B_IS_INSTANCED,
            ),
        })?;
        reg.named(field)
    }

    /// `SET_VERTEX_STREAM_SIZE_A_UPPER` above `SET_VERTEX_STREAM_SIZE_B_LOWER`: only for streams
    /// below [`VERTEX_STREAM_SIZES`] (a **guess**; see that constant).
    pub fn size(&self) -> Result<u64, StateError> {
        let index = bounded(arrays::VERTEX_STREAM_SIZE, self.index, VERTEX_STREAM_SIZES)?;
        address(
            self.mirror
                .reg(&clc797::SET_VERTEX_STREAM_SIZE_A, &[index])?,
            clc797::SET_VERTEX_STREAM_SIZE_A_UPPER,
            self.mirror
                .reg(&clc797::SET_VERTEX_STREAM_SIZE_B, &[index])?,
            clc797::SET_VERTEX_STREAM_SIZE_B_LOWER,
        )
    }
}

/// A vertex attribute ([`ThreeD::vertex_attribute`]): attribute `i` below
/// [`VERTEX_ATTRIBUTES_PER_HALF`] reads `SET_VERTEX_ATTRIBUTE_A(i)`, the rest
/// `SET_VERTEX_ATTRIBUTE_B(i - 16)`.
#[derive(Clone, Copy, Debug)]
pub struct VertexAttribute<'a> {
    mirror: &'a Mirror,
    index: u32,
}

impl VertexAttribute<'_> {
    /// The attribute's index.
    pub fn index(&self) -> u32 {
        self.index
    }

    fn read(&self, a: Field, b: Field) -> Result<(Reg, Field), StateError> {
        half(
            self.mirror,
            Halves {
                a: (&clc797::SET_VERTEX_ATTRIBUTE_A, a),
                b: (&clc797::SET_VERTEX_ATTRIBUTE_B, b),
            },
            self.index,
            VERTEX_ATTRIBUTES_PER_HALF,
        )
    }

    /// `SET_VERTEX_ATTRIBUTE_A_STREAM` (or `_B_`).
    pub fn stream(&self) -> Result<u32, StateError> {
        let (reg, field) = self.read(
            clc797::SET_VERTEX_ATTRIBUTE_A_STREAM,
            clc797::SET_VERTEX_ATTRIBUTE_B_STREAM,
        )?;
        reg.field(field)
    }

    /// `SET_VERTEX_ATTRIBUTE_A_SOURCE` (or `_B_`).
    pub fn source(&self) -> Result<&'static Value, StateError> {
        let (reg, field) = self.read(
            clc797::SET_VERTEX_ATTRIBUTE_A_SOURCE,
            clc797::SET_VERTEX_ATTRIBUTE_B_SOURCE,
        )?;
        reg.named(field)
    }

    /// `SET_VERTEX_ATTRIBUTE_A_OFFSET` (or `_B_`).
    pub fn offset(&self) -> Result<u32, StateError> {
        let (reg, field) = self.read(
            clc797::SET_VERTEX_ATTRIBUTE_A_OFFSET,
            clc797::SET_VERTEX_ATTRIBUTE_B_OFFSET,
        )?;
        reg.field(field)
    }

    /// `SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS` (or `_B_`).
    pub fn component_bit_widths(&self) -> Result<&'static Value, StateError> {
        let (reg, field) = self.read(
            clc797::SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS,
            clc797::SET_VERTEX_ATTRIBUTE_B_COMPONENT_BIT_WIDTHS,
        )?;
        reg.named(field)
    }

    /// `SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE` (or `_B_`).
    pub fn numerical_type(&self) -> Result<&'static Value, StateError> {
        let (reg, field) = self.read(
            clc797::SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE,
            clc797::SET_VERTEX_ATTRIBUTE_B_NUMERICAL_TYPE,
        )?;
        reg.named(field)
    }

    /// `SET_VERTEX_ATTRIBUTE_A_SWAP_R_AND_B` (or `_B_`).
    pub fn swap_r_and_b(&self) -> Result<&'static Value, StateError> {
        let (reg, field) = self.read(
            clc797::SET_VERTEX_ATTRIBUTE_A_SWAP_R_AND_B,
            clc797::SET_VERTEX_ATTRIBUTE_B_SWAP_R_AND_B,
        )?;
        reg.named(field)
    }
}

single_view! {
    /// The index buffer ([`ThreeD::index_buffer`]).
    IndexBuffer
}

getters!(IndexBuffer {
    /// `SET_INDEX_BUFFER_A_ADDRESS_UPPER` above `SET_INDEX_BUFFER_B_ADDRESS_LOWER`.
    address: address clc797::SET_INDEX_BUFFER_A, clc797::SET_INDEX_BUFFER_A_ADDRESS_UPPER, clc797::SET_INDEX_BUFFER_B, clc797::SET_INDEX_BUFFER_B_ADDRESS_LOWER;
    /// `SET_INDEX_BUFFER_SIZE_A_UPPER` above `SET_INDEX_BUFFER_SIZE_B_LOWER`.
    size: address clc797::SET_INDEX_BUFFER_SIZE_A, clc797::SET_INDEX_BUFFER_SIZE_A_UPPER, clc797::SET_INDEX_BUFFER_SIZE_B, clc797::SET_INDEX_BUFFER_SIZE_B_LOWER;
    /// `SET_INDEX_BUFFER_E_INDEX_SIZE`.
    index_size: named clc797::SET_INDEX_BUFFER_E, clc797::SET_INDEX_BUFFER_E_INDEX_SIZE;
    /// `SET_INDEX_BUFFER_F_FIRST`.
    first: field clc797::SET_INDEX_BUFFER_F, clc797::SET_INDEX_BUFFER_F_FIRST;
});

/// A bind group's constant buffers ([`ThreeD::bind_group`]).
#[derive(Clone, Copy, Debug)]
pub struct BindGroup<'a> {
    mirror: &'a Mirror,
    index: u32,
}

impl BindGroup<'_> {
    /// The bind group's index.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// What was last bound to slot `slot`, below [`CONSTANT_BUFFER_SLOTS`].
    pub fn constant_buffer(&self, slot: u32) -> Result<ConstantBufferBinding, StateError> {
        let slot = bounded(arrays::CONSTANT_BUFFER_SLOT, slot, CONSTANT_BUFFER_SLOTS)?;
        self.mirror
            .binding(self.index, slot)
            .ok_or(StateError::Index {
                array: arrays::CONSTANT_BUFFER_SLOT,
                index: slot,
                count: CONSTANT_BUFFER_SLOTS,
            })
    }
}

/// The pool methods of one class.
#[derive(Debug)]
struct TexPoolMethods {
    header_a: (&'static Method, Field),
    header_b: (&'static Method, Field),
    header_c: (&'static Method, Field),
    sampler_a: (&'static Method, Field),
    sampler_b: (&'static Method, Field),
    sampler_c: (&'static Method, Field),
}

const TEX_POOLS_3D: TexPoolMethods = TexPoolMethods {
    header_a: (
        &clc797::SET_TEX_HEADER_POOL_A,
        clc797::SET_TEX_HEADER_POOL_A_OFFSET_UPPER,
    ),
    header_b: (
        &clc797::SET_TEX_HEADER_POOL_B,
        clc797::SET_TEX_HEADER_POOL_B_OFFSET_LOWER,
    ),
    header_c: (
        &clc797::SET_TEX_HEADER_POOL_C,
        clc797::SET_TEX_HEADER_POOL_C_MAXIMUM_INDEX,
    ),
    sampler_a: (
        &clc797::SET_TEX_SAMPLER_POOL_A,
        clc797::SET_TEX_SAMPLER_POOL_A_OFFSET_UPPER,
    ),
    sampler_b: (
        &clc797::SET_TEX_SAMPLER_POOL_B,
        clc797::SET_TEX_SAMPLER_POOL_B_OFFSET_LOWER,
    ),
    sampler_c: (
        &clc797::SET_TEX_SAMPLER_POOL_C,
        clc797::SET_TEX_SAMPLER_POOL_C_MAXIMUM_INDEX,
    ),
};

const TEX_POOLS_COMPUTE: TexPoolMethods = TexPoolMethods {
    header_a: (
        &clc7c0::SET_TEX_HEADER_POOL_A,
        clc7c0::SET_TEX_HEADER_POOL_A_OFFSET_UPPER,
    ),
    header_b: (
        &clc7c0::SET_TEX_HEADER_POOL_B,
        clc7c0::SET_TEX_HEADER_POOL_B_OFFSET_LOWER,
    ),
    header_c: (
        &clc7c0::SET_TEX_HEADER_POOL_C,
        clc7c0::SET_TEX_HEADER_POOL_C_MAXIMUM_INDEX,
    ),
    sampler_a: (
        &clc7c0::SET_TEX_SAMPLER_POOL_A,
        clc7c0::SET_TEX_SAMPLER_POOL_A_OFFSET_UPPER,
    ),
    sampler_b: (
        &clc7c0::SET_TEX_SAMPLER_POOL_B,
        clc7c0::SET_TEX_SAMPLER_POOL_B_OFFSET_LOWER,
    ),
    sampler_c: (
        &clc7c0::SET_TEX_SAMPLER_POOL_C,
        clc7c0::SET_TEX_SAMPLER_POOL_C_MAXIMUM_INDEX,
    ),
};

/// The texture header and sampler pools of the 3D or compute class ([`ThreeD::tex_pools`],
/// [`Compute::tex_pools`]).
#[derive(Clone, Copy, Debug)]
pub struct TexPools<'a> {
    mirror: &'a Mirror,
    methods: &'static TexPoolMethods,
}

impl TexPools<'_> {
    fn field(&self, (method, field): (&'static Method, Field)) -> Result<u32, StateError> {
        self.mirror.reg(method, &[])?.field(field)
    }

    /// `SET_TEX_HEADER_POOL_A_OFFSET_UPPER` above `SET_TEX_HEADER_POOL_B_OFFSET_LOWER`.
    pub fn header_address(&self) -> Result<u64, StateError> {
        Ok(join(
            self.field(self.methods.header_a)?,
            self.field(self.methods.header_b)?,
        ))
    }

    /// `SET_TEX_HEADER_POOL_C_MAXIMUM_INDEX`.
    pub fn header_maximum_index(&self) -> Result<u32, StateError> {
        self.field(self.methods.header_c)
    }

    /// `SET_TEX_SAMPLER_POOL_A_OFFSET_UPPER` above `SET_TEX_SAMPLER_POOL_B_OFFSET_LOWER`.
    pub fn sampler_address(&self) -> Result<u64, StateError> {
        Ok(join(
            self.field(self.methods.sampler_a)?,
            self.field(self.methods.sampler_b)?,
        ))
    }

    /// `SET_TEX_SAMPLER_POOL_C_MAXIMUM_INDEX`.
    pub fn sampler_maximum_index(&self) -> Result<u32, StateError> {
        self.field(self.methods.sampler_c)
    }
}

element_view! {
    /// One pipeline ([`ThreeD::pipeline`]).
    Pipeline
}

getters!(Pipeline {
    /// `SET_PIPELINE_SHADER_ENABLE`.
    enable: named clc797::SET_PIPELINE_SHADER, clc797::SET_PIPELINE_SHADER_ENABLE;
    /// `SET_PIPELINE_SHADER_TYPE`.
    shader_type: named clc797::SET_PIPELINE_SHADER, clc797::SET_PIPELINE_SHADER_TYPE;
    /// `SET_PIPELINE_PROGRAM_ADDRESS_A_UPPER` above `SET_PIPELINE_PROGRAM_ADDRESS_B_LOWER`.
    program_address: address clc797::SET_PIPELINE_PROGRAM_ADDRESS_A, clc797::SET_PIPELINE_PROGRAM_ADDRESS_A_UPPER, clc797::SET_PIPELINE_PROGRAM_ADDRESS_B, clc797::SET_PIPELINE_PROGRAM_ADDRESS_B_LOWER;
    /// `SET_PIPELINE_REGISTER_COUNT_V`.
    register_count: field clc797::SET_PIPELINE_REGISTER_COUNT, clc797::SET_PIPELINE_REGISTER_COUNT_V;
    /// `SET_PIPELINE_BINDING_GROUP`.
    binding_group: field clc797::SET_PIPELINE_BINDING, clc797::SET_PIPELINE_BINDING_GROUP;
});

single_view! {
    /// The rasterizer state ([`ThreeD::rasterizer`]).
    Rasterizer
}

getters!(Rasterizer {
    /// `SET_FRONT_POLYGON_MODE_V`.
    front_polygon_mode: named clc797::SET_FRONT_POLYGON_MODE, clc797::SET_FRONT_POLYGON_MODE_V;
    /// `SET_BACK_POLYGON_MODE_V`.
    back_polygon_mode: named clc797::SET_BACK_POLYGON_MODE, clc797::SET_BACK_POLYGON_MODE_V;
    /// `OGL_SET_CULL_ENABLE`.
    cull_enable: named clc797::OGL_SET_CULL, clc797::OGL_SET_CULL_ENABLE;
    /// `OGL_SET_CULL_FACE_V`.
    cull_face: named clc797::OGL_SET_CULL_FACE, clc797::OGL_SET_CULL_FACE_V;
    /// `OGL_SET_FRONT_FACE_V`.
    front_face: named clc797::OGL_SET_FRONT_FACE, clc797::OGL_SET_FRONT_FACE_V;
    /// `SET_WINDOW_ORIGIN_MODE`.
    window_origin_mode: named clc797::SET_WINDOW_ORIGIN, clc797::SET_WINDOW_ORIGIN_MODE;
    /// `SET_WINDOW_ORIGIN_FLIP_Y`.
    window_origin_flip_y: named clc797::SET_WINDOW_ORIGIN, clc797::SET_WINDOW_ORIGIN_FLIP_Y;
    /// `SET_RASTER_ENABLE_V`.
    raster_enable: named clc797::SET_RASTER_ENABLE, clc797::SET_RASTER_ENABLE_V;
    /// `SET_POLY_OFFSET_POINT_ENABLE`.
    poly_offset_point: named clc797::SET_POLY_OFFSET_POINT, clc797::SET_POLY_OFFSET_POINT_ENABLE;
    /// `SET_POLY_OFFSET_LINE_ENABLE`.
    poly_offset_line: named clc797::SET_POLY_OFFSET_LINE, clc797::SET_POLY_OFFSET_LINE_ENABLE;
    /// `SET_POLY_OFFSET_FILL_ENABLE`.
    poly_offset_fill: named clc797::SET_POLY_OFFSET_FILL, clc797::SET_POLY_OFFSET_FILL_ENABLE;
    /// `SET_DEPTH_BIAS`.
    depth_bias: float clc797::SET_DEPTH_BIAS;
    /// `SET_SLOPE_SCALE_DEPTH_BIAS`.
    slope_scale_depth_bias: float clc797::SET_SLOPE_SCALE_DEPTH_BIAS;
    /// `SET_DEPTH_BIAS_CLAMP`.
    depth_bias_clamp: float clc797::SET_DEPTH_BIAS_CLAMP;
    /// `SET_LINE_WIDTH_FLOAT`.
    line_width: float clc797::SET_LINE_WIDTH_FLOAT;
    /// `SET_ALIASED_LINE_WIDTH_FLOAT`.
    aliased_line_width: float clc797::SET_ALIASED_LINE_WIDTH_FLOAT;
    /// `SET_POINT_SIZE`.
    point_size: float clc797::SET_POINT_SIZE;
    /// `SET_PROVOKING_VERTEX_V`.
    provoking_vertex: named clc797::SET_PROVOKING_VERTEX, clc797::SET_PROVOKING_VERTEX_V;
});

single_view! {
    /// The depth and stencil state ([`ThreeD::depth_stencil`]).
    DepthStencil
}

getters!(DepthStencil {
    /// `SET_DEPTH_TEST_ENABLE`.
    depth_test: named clc797::SET_DEPTH_TEST, clc797::SET_DEPTH_TEST_ENABLE;
    /// `SET_DEPTH_WRITE_ENABLE`.
    depth_write: named clc797::SET_DEPTH_WRITE, clc797::SET_DEPTH_WRITE_ENABLE;
    /// `SET_DEPTH_FUNC_V`.
    depth_func: named clc797::SET_DEPTH_FUNC, clc797::SET_DEPTH_FUNC_V;
    /// `SET_DEPTH_BOUNDS_TEST_ENABLE`.
    depth_bounds_test: named clc797::SET_DEPTH_BOUNDS_TEST, clc797::SET_DEPTH_BOUNDS_TEST_ENABLE;
    /// `SET_DEPTH_BOUNDS_MIN`.
    depth_bounds_min: float clc797::SET_DEPTH_BOUNDS_MIN;
    /// `SET_DEPTH_BOUNDS_MAX`.
    depth_bounds_max: float clc797::SET_DEPTH_BOUNDS_MAX;
    /// `SET_STENCIL_TEST_ENABLE`.
    stencil_test: named clc797::SET_STENCIL_TEST, clc797::SET_STENCIL_TEST_ENABLE;
    /// `SET_STENCIL_OP_FAIL_V`.
    stencil_op_fail: named clc797::SET_STENCIL_OP_FAIL, clc797::SET_STENCIL_OP_FAIL_V;
    /// `SET_STENCIL_OP_ZFAIL_V`.
    stencil_op_zfail: named clc797::SET_STENCIL_OP_ZFAIL, clc797::SET_STENCIL_OP_ZFAIL_V;
    /// `SET_STENCIL_OP_ZPASS_V`.
    stencil_op_zpass: named clc797::SET_STENCIL_OP_ZPASS, clc797::SET_STENCIL_OP_ZPASS_V;
    /// `SET_STENCIL_FUNC_V`.
    stencil_func: named clc797::SET_STENCIL_FUNC, clc797::SET_STENCIL_FUNC_V;
    /// `SET_STENCIL_FUNC_REF_V`.
    stencil_func_ref: field clc797::SET_STENCIL_FUNC_REF, clc797::SET_STENCIL_FUNC_REF_V;
    /// `SET_STENCIL_FUNC_MASK_V`.
    stencil_func_mask: field clc797::SET_STENCIL_FUNC_MASK, clc797::SET_STENCIL_FUNC_MASK_V;
    /// `SET_STENCIL_MASK_V`.
    stencil_mask: field clc797::SET_STENCIL_MASK, clc797::SET_STENCIL_MASK_V;
    /// `SET_TWO_SIDED_STENCIL_TEST_ENABLE`.
    two_sided_stencil_test: named clc797::SET_TWO_SIDED_STENCIL_TEST, clc797::SET_TWO_SIDED_STENCIL_TEST_ENABLE;
    /// `SET_BACK_STENCIL_OP_FAIL_V`.
    back_stencil_op_fail: named clc797::SET_BACK_STENCIL_OP_FAIL, clc797::SET_BACK_STENCIL_OP_FAIL_V;
    /// `SET_BACK_STENCIL_OP_ZFAIL_V`.
    back_stencil_op_zfail: named clc797::SET_BACK_STENCIL_OP_ZFAIL, clc797::SET_BACK_STENCIL_OP_ZFAIL_V;
    /// `SET_BACK_STENCIL_OP_ZPASS_V`.
    back_stencil_op_zpass: named clc797::SET_BACK_STENCIL_OP_ZPASS, clc797::SET_BACK_STENCIL_OP_ZPASS_V;
    /// `SET_BACK_STENCIL_FUNC_V`.
    back_stencil_func: named clc797::SET_BACK_STENCIL_FUNC, clc797::SET_BACK_STENCIL_FUNC_V;
    /// `SET_BACK_STENCIL_FUNC_REF_V`.
    back_stencil_func_ref: field clc797::SET_BACK_STENCIL_FUNC_REF, clc797::SET_BACK_STENCIL_FUNC_REF_V;
    /// `SET_BACK_STENCIL_FUNC_MASK_V`.
    back_stencil_func_mask: field clc797::SET_BACK_STENCIL_FUNC_MASK, clc797::SET_BACK_STENCIL_FUNC_MASK_V;
    /// `SET_BACK_STENCIL_MASK_V`.
    back_stencil_mask: field clc797::SET_BACK_STENCIL_MASK, clc797::SET_BACK_STENCIL_MASK_V;
});

/// The compute class's state ([`Mirror::compute`]). The rest of a dispatch comes from its QMD
/// ([`crate::back::qmd`]), which reads [`Mirror::registers`].
#[derive(Clone, Copy, Debug)]
pub struct Compute<'a> {
    mirror: &'a Mirror,
}

impl<'a> Compute<'a> {
    /// The texture header and sampler pools.
    pub fn tex_pools(&self) -> TexPools<'a> {
        TexPools {
            mirror: self.mirror,
            methods: &TEX_POOLS_COMPUTE,
        }
    }

    /// Every register, indexed by method number.
    pub fn registers(&self) -> &'a [u32] {
        self.mirror.registers()
    }
}

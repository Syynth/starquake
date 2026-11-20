//! Tests of clears: register scripts written to a real 3D mirror, run through [`Draws`] with the
//! real texture cache over the recording backend. The golden lists are the recording backend's
//! text of the submitted list (`insta`); the pixel tests read the cleared texture back.

use std::sync::Arc;

use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, RngSeed};
use starquake_layout::SampleLayout;
use starquake_memory::{FlatMemory, PAGE_SIZE, PageRange, WriteWatch};
use starquake_render::caps::{DynamicState, StageSupport};
use starquake_render::command::{CopyRegion, TextureRegion};
use starquake_render::resource::{Extent3d, Origin3d};
use starquake_render::{
    Backend, Caps, CommandList, Features, FormatFeatures, FormatTable, Limits, Readback,
    RecordingBackend, SampleCounts, TextureId,
};
use starquake_shader::lookup::{EngineState, ProgramMemory, ProgramRequest, Ticket, VariantKey};

use super::super::shaders::{SetLookup, ShaderLookup};
use super::super::{Draws, Outcome, WaitWhileCompiling};
use super::*;
use crate::back::buffer::BufferCache;
use crate::back::pipeline::PipelineCache;
use crate::back::readback::ReadbackIds;
use crate::back::texture::TextureCache;
use crate::classes::clc797;
use crate::field::Field;
use crate::interface::ChannelId;
use crate::memory::TestGpuMemory;
use crate::ring::ClassId;

// ---------------------------------------------------------------------------------------------
// The rig.

/// Where test memory starts, at the same GPU and CPU address (row 30).
const BASE: u64 = 0x8000_0000;
/// Pages of test memory.
const PAGES: u64 = 64;
/// Not channel 0, so a hard-coded channel fails.
const CHANNEL: ChannelId = ChannelId(3);
/// Colour target 0's, colour target 1's and the depth target's addresses.
const TARGET0: u64 = BASE + 0x1_0000;
const TARGET1: u64 = BASE + 0x2_0000;
const DEPTH: u64 = BASE + 0x3_0000;
/// The targets' width and height (different, so a swap fails).
const WIDTH: u32 = 8;
const HEIGHT: u32 = 6;

type Memory = TestGpuMemory<FlatMemory>;

fn caps() -> Caps {
    Caps {
        limits: Limits {
            max_buffer_size: PAGES * PAGE_SIZE,
            max_staging_size: PAGES * PAGE_SIZE,
            max_texture_dimension_1d: 4096,
            max_texture_dimension_2d: 4096,
            max_texture_dimension_3d: 256,
            max_texture_array_layers: 64,
            max_sampler_anisotropy: 16,
            max_vertex_attributes: 16,
            max_vertex_buffers: 8,
            max_vertex_buffer_stride: 2048,
            max_color_attachments: 8,
            max_viewports: 16,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 16,
            max_compute_workgroups_per_dimension: 65_535,
            max_uniform_buffers_per_stage: 12,
            max_storage_buffers_per_stage: 8,
            max_sampled_textures_per_stage: 16,
            max_storage_textures_per_stage: 4,
            max_samplers_per_stage: 16,
        },
        features: Features::ALL,
        stages: StageSupport::ALL,
        dynamic_state: DynamicState::ALL,
        sample_counts: SampleCounts::X1,
        view_swizzle: true,
        formats: FormatTable::uniform(FormatFeatures::ALL),
    }
}

fn memory() -> Memory {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = TestGpuMemory::new(FlatMemory::new(BASE, size));
    mem.map(PageRange::new(BASE, PAGES * PAGE_SIZE).unwrap(), BASE)
        .unwrap();
    mem
}

/// A shader cache no clear may ask.
struct NoShaders;

impl ShaderLookup for NoShaders {
    fn lookup(
        &mut self,
        _: &[ProgramRequest],
        _: &VariantKey,
        _: &dyn ProgramMemory,
        _: &dyn EngineState,
    ) -> SetLookup {
        unreachable!("a clear looked up shaders")
    }

    fn wait(&mut self, _: Ticket) -> SetLookup {
        unreachable!("a clear waited for shaders")
    }
}

struct Rig {
    backend: Arc<RecordingBackend>,
    mem: Memory,
    buffers: BufferCache,
    textures: TextureCache,
    pipelines: PipelineCache,
    draws: Draws,
    mirror: Mirror,
    list: CommandList,
    readbacks: ReadbackIds,
}

impl Rig {
    fn new() -> Self {
        Self::with_caps(caps())
    }

    /// A rig whose backend reports `caps`.
    fn with_caps(caps: Caps) -> Self {
        let backend = Arc::new(RecordingBackend::new(caps, 1 << 40));
        let dyn_backend: Arc<dyn Backend> = Arc::<RecordingBackend>::clone(&backend);
        let watch = Arc::new(WriteWatch::new());
        Self {
            buffers: BufferCache::new(Arc::clone(&dyn_backend), Arc::clone(&watch)),
            textures: TextureCache::new(Arc::clone(&dyn_backend), watch),
            pipelines: PipelineCache::new(Arc::clone(&dyn_backend)),
            draws: Draws::new(dyn_backend, WaitWhileCompiling::Wait),
            mem: memory(),
            mirror: Mirror::new(CHANNEL, ClassId(clc797::CLASS)),
            list: CommandList::new(),
            backend,
            readbacks: ReadbackIds::new(),
        }
    }

    /// Writes `value` to `method` at `indices`, as the executor would: into the mirror, then to
    /// the draw path.
    fn put(
        &mut self,
        method: &Method,
        indices: &[u32],
        value: u32,
    ) -> Result<Option<Outcome>, DrawError> {
        let number = method.method_at(indices).unwrap();
        self.mirror.write(number, value).unwrap();
        let mut shaders = NoShaders;
        let mut ctx = DrawContext {
            memory: &self.mem,
            buffers: &mut self.buffers,
            textures: &mut self.textures,
            pipelines: &mut self.pipelines,
            shaders: &mut shaders,
            list: &mut self.list,
        };
        self.draws.method(&mut ctx, &mut self.mirror, number, value)
    }

    fn set(&mut self, method: &Method, indices: &[u32], value: u32) {
        assert_eq!(
            self.put(method, indices, value),
            Ok(None),
            "{}",
            method.name
        );
    }

    /// Writes `word` to `CLEAR_SURFACE`.
    fn clear_word(&mut self, word: u32) -> Result<Cleared, DrawError> {
        match self.put(&clc797::CLEAR_SURFACE, &[], word)? {
            Some(Outcome::Cleared(cleared)) => Ok(cleared),
            other => unreachable!("a clear reported {other:?}"),
        }
    }

    /// Writes `surface` to `CLEAR_SURFACE`, which must succeed.
    fn clear(&mut self, surface: ClearSurface) -> Cleared {
        self.clear_word(surface.encode().unwrap()).unwrap()
    }

    fn address(&mut self, upper: &Method, lower: &Method, indices: &[u32], address: u64) {
        self.set(upper, indices, u32::try_from(address >> 32).unwrap());
        self.set(
            lower,
            indices,
            u32::try_from(address & 0xffff_ffff).unwrap(),
        );
    }

    /// Colour target `index` at `address`, [`WIDTH`] by [`HEIGHT`], block-linear, of `layers`
    /// layers and header format `format`.
    fn color_target(&mut self, index: u32, address: u64, format: u32, layers: u32) {
        self.color_target_sized(index, address, format, layers, WIDTH, HEIGHT);
    }

    /// Colour target `index` as [`Rig::color_target`], `width` by `height`.
    fn color_target_sized(
        &mut self,
        index: u32,
        address: u64,
        format: u32,
        layers: u32,
        width: u32,
        height: u32,
    ) {
        self.address(
            &clc797::SET_COLOR_TARGET_A,
            &clc797::SET_COLOR_TARGET_B,
            &[index],
            address,
        );
        self.set(&clc797::SET_COLOR_TARGET_WIDTH, &[index], width);
        self.set(&clc797::SET_COLOR_TARGET_HEIGHT, &[index], height);
        self.set(&clc797::SET_COLOR_TARGET_FORMAT, &[index], format);
        self.set(&clc797::SET_COLOR_TARGET_THIRD_DIMENSION, &[index], layers);
    }

    /// `SET_CT_SELECT` with `count` targets, mapped to themselves.
    fn ct_select(&mut self, count: u32) {
        let targets = [clc797::SET_CT_SELECT_TARGET0, clc797::SET_CT_SELECT_TARGET1];
        let mut word = fields(&[(clc797::SET_CT_SELECT_TARGET_COUNT, count)]);
        for (index, field) in (0u32..).zip(targets) {
            word = field.insert(word, index).unwrap();
        }
        self.set(&clc797::SET_CT_SELECT, &[], word);
    }

    /// The depth target at [`DEPTH`], `width` by `height`, header format `format`, of `layers`
    /// layers.
    fn depth_target(&mut self, format: u32, width: u32, height: u32, layers: u32) {
        self.address(&clc797::SET_ZT_A, &clc797::SET_ZT_B, &[], DEPTH);
        self.set(&clc797::SET_ZT_FORMAT, &[], format);
        self.set(&clc797::SET_ZT_SIZE_A, &[], width);
        self.set(&clc797::SET_ZT_SIZE_B, &[], height);
        self.set(
            &clc797::SET_ZT_SIZE_C,
            &[],
            fields(&[
                (clc797::SET_ZT_SIZE_C_THIRD_DIMENSION, layers),
                (
                    clc797::SET_ZT_SIZE_C_CONTROL,
                    clc797::SET_ZT_SIZE_C_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE,
                ),
            ]),
        );
        self.set(&clc797::SET_ZT_SELECT, &[], 1);
    }

    /// One A8B8G8R8 colour target 0 and nothing else, the surface clip its size.
    fn basic(&mut self) {
        self.color_target(0, TARGET0, clc797::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8, 1);
        self.ct_select(1);
        self.surface_clip(WIDTH, HEIGHT);
    }

    fn surface_clip(&mut self, width: u32, height: u32) {
        self.set(
            &clc797::SET_SURFACE_CLIP_HORIZONTAL,
            &[],
            fields(&[(clc797::SET_SURFACE_CLIP_HORIZONTAL_WIDTH, width)]),
        );
        self.set(
            &clc797::SET_SURFACE_CLIP_VERTICAL,
            &[],
            fields(&[(clc797::SET_SURFACE_CLIP_VERTICAL_HEIGHT, height)]),
        );
    }

    fn clear_color(&mut self, color: [f32; 4]) {
        for (index, value) in (0u32..).zip(color) {
            self.set(&clc797::SET_COLOR_CLEAR_VALUE, &[index], value.to_bits());
        }
    }

    fn clear_depth(&mut self, depth: f32) {
        self.set(&clc797::SET_Z_CLEAR_VALUE, &[], depth.to_bits());
    }

    /// The stencil value word, `SET_STENCIL_MASK` and `SET_STENCIL_FUNC_MASK`.
    fn clear_stencil(&mut self, value: u32, mask: u32, func_mask: u32) {
        self.set(&clc797::SET_STENCIL_CLEAR_VALUE, &[], value);
        self.set(&clc797::SET_STENCIL_MASK, &[], mask);
        self.set(&clc797::SET_STENCIL_FUNC_MASK, &[], func_mask);
    }

    /// Scissor 0 on, from `x` to `x_end` and `y` to `y_end`, and `USE_SCISSOR0`.
    fn scissor(&mut self, x: u32, x_end: u32, y: u32, y_end: u32) {
        self.set(
            &clc797::SET_SCISSOR_ENABLE,
            &[0],
            clc797::SET_SCISSOR_ENABLE_V_TRUE,
        );
        self.set(
            &clc797::SET_SCISSOR_HORIZONTAL,
            &[0],
            fields(&[
                (clc797::SET_SCISSOR_HORIZONTAL_XMIN, x),
                (clc797::SET_SCISSOR_HORIZONTAL_XMAX, x_end),
            ]),
        );
        self.set(
            &clc797::SET_SCISSOR_VERTICAL,
            &[0],
            fields(&[
                (clc797::SET_SCISSOR_VERTICAL_YMIN, y),
                (clc797::SET_SCISSOR_VERTICAL_YMAX, y_end),
            ]),
        );
        self.control(fields(&[(
            clc797::SET_CLEAR_SURFACE_CONTROL_USE_SCISSOR0,
            1,
        )]));
    }

    fn control(&mut self, word: u32) {
        self.set(&clc797::SET_CLEAR_SURFACE_CONTROL, &[], word);
    }

    /// Submits the open list and returns the recording backend's text of everything so far.
    fn submit(&mut self) -> String {
        let list = core::mem::take(&mut self.list);
        let fence = self.backend.submit(list).unwrap();
        self.textures.submitted(fence);
        self.draws.submitted(fence);
        self.backend.snapshot()
    }

    /// Every texel of layer `layer` of `texture`, read back after submitting the open list.
    fn texels(&mut self, texture: TextureId, layer: u32) -> Vec<u8> {
        self.submit();
        let id = self.readbacks.allocate();
        let mut list = CommandList::new();
        list.push(Command::Readback {
            from: CopyRegion::Texture(TextureRegion {
                texture,
                mip: 0,
                origin: Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                extent: Extent3d {
                    width: WIDTH,
                    height: HEIGHT,
                    depth_or_layers: 1,
                },
            }),
            id,
        });
        self.backend.submit(list).unwrap();
        match self.backend.take_readback(id) {
            Some(Readback::Bytes(bytes)) => bytes,
            other => unreachable!("readback {other:?}"),
        }
    }

    /// The texture the texture cache holds at `address`.
    fn texture_at(&self, address: u64) -> TextureId {
        self.textures
            .entries()
            .find(|e| e.info.address == address)
            .unwrap()
            .texture
    }

    fn commands(&self) -> &[Command] {
        &self.list.commands
    }

    fn clears(&self) -> Vec<ClearDesc> {
        self.commands()
            .iter()
            .filter_map(|c| match c {
                Command::Clear(d) => Some(*d),
                _ => None,
            })
            .collect()
    }

    /// The commands that are not texture uploads, by kind.
    fn kinds(&self) -> Vec<&'static str> {
        self.commands()
            .iter()
            .filter_map(|c| match c {
                Command::Upload {
                    to: CopyRegion::Texture(_),
                    ..
                } => None,
                Command::Upload { .. } => Some("upload"),
                Command::Clear(_) => Some("clear"),
                Command::SetRenderTargets(_) => Some("targets"),
                Command::Draw(_) => Some("draw"),
                _ => Some("other"),
            })
            .collect()
    }
}

/// A word with `fields` set.
fn fields(fields: &[(Field, u32)]) -> u32 {
    fields.iter().fold(0, |word, &(field, value)| {
        field.insert(word, value).unwrap()
    })
}

/// A clear of `color` channels of colour target 0, layer 0.
fn color(color: ColorWrites) -> ClearSurface {
    ClearSurface {
        depth: false,
        stencil: false,
        color,
        target: 0,
        layer: 0,
    }
}

/// A clear of depth and/or stencil, layer 0.
fn depth_stencil(depth: bool, stencil: bool) -> ClearSurface {
    ClearSurface {
        depth,
        stencil,
        color: ColorWrites::EMPTY,
        target: 0,
        layer: 0,
    }
}

const ONLY_COLOR: Cleared = Cleared {
    color: Some(Part::Cleared),
    depth_stencil: None,
};

const ONLY_DEPTH_STENCIL: Cleared = Cleared {
    color: None,
    depth_stencil: Some(Part::Cleared),
};

// ---------------------------------------------------------------------------------------------
// Registers and decoding.

#[test]
fn clear_registers_are_where_the_header_puts_them() {
    // The literals of `classes/clc797.h`, written out (module docs, "Sources").
    assert_eq!(clc797::CLEAR_SURFACE.byte_address, 0x19d0);
    assert_eq!(clc797::CLEAR_SURFACE_Z_ENABLE, Field::new(0, 0));
    assert_eq!(clc797::CLEAR_SURFACE_STENCIL_ENABLE, Field::new(1, 1));
    assert_eq!(clc797::CLEAR_SURFACE_R_ENABLE, Field::new(2, 2));
    assert_eq!(clc797::CLEAR_SURFACE_G_ENABLE, Field::new(3, 3));
    assert_eq!(clc797::CLEAR_SURFACE_B_ENABLE, Field::new(4, 4));
    assert_eq!(clc797::CLEAR_SURFACE_A_ENABLE, Field::new(5, 5));
    assert_eq!(clc797::CLEAR_SURFACE_MRT_SELECT, Field::new(9, 6));
    assert_eq!(clc797::CLEAR_SURFACE_RT_ARRAY_INDEX, Field::new(25, 10));
    assert_eq!(clc797::SET_CLEAR_SURFACE_CONTROL.byte_address, 0x10f8);
    assert_eq!(
        clc797::SET_CLEAR_SURFACE_CONTROL_RESPECT_STENCIL_MASK,
        Field::new(0, 0)
    );
    assert_eq!(
        clc797::SET_CLEAR_SURFACE_CONTROL_USE_CLEAR_RECT,
        Field::new(4, 4)
    );
    assert_eq!(
        clc797::SET_CLEAR_SURFACE_CONTROL_USE_SCISSOR0,
        Field::new(8, 8)
    );
    assert_eq!(
        clc797::SET_CLEAR_SURFACE_CONTROL_USE_VIEWPORT_CLIP0,
        Field::new(12, 12)
    );
    assert_eq!(clc797::SET_COLOR_CLEAR_VALUE.byte_address, 0x0d80);
    assert_eq!(clc797::SET_COLOR_CLEAR_VALUE.strides, &[4]);
    assert_eq!(clc797::SET_Z_CLEAR_VALUE.byte_address, 0x0d90);
    assert_eq!(clc797::SET_STENCIL_CLEAR_VALUE.byte_address, 0x0da0);
    assert_eq!(clc797::SET_STENCIL_CLEAR_VALUE_V, Field::new(7, 0));
    assert_eq!(clc797::SET_STENCIL_MASK.byte_address, 0x139c);
    assert_eq!(clc797::SET_STENCIL_MASK_V, Field::new(7, 0));
    assert_eq!(clc797::SET_STENCIL_FUNC_MASK.byte_address, 0x1398);
    assert_eq!(clc797::SET_STENCIL_FUNC_MASK_V, Field::new(7, 0));
    assert_eq!(clc797::SET_SCISSOR_ENABLE.byte_address, 0x0e00);
    assert_eq!(clc797::SET_SCISSOR_HORIZONTAL.byte_address, 0x0e04);
    assert_eq!(clc797::SET_SCISSOR_VERTICAL.byte_address, 0x0e08);
    assert_eq!(clc797::SET_SCISSOR_HORIZONTAL_XMIN, Field::new(15, 0));
    assert_eq!(clc797::SET_SCISSOR_HORIZONTAL_XMAX, Field::new(31, 16));
    assert_eq!(clc797::SET_SCISSOR_VERTICAL_YMIN, Field::new(15, 0));
    assert_eq!(clc797::SET_SCISSOR_VERTICAL_YMAX, Field::new(31, 16));
    // Eden's `stencil_partial` constant.
    assert_eq!(FULL_STENCIL_MASK, 0xff);
    assert_eq!(CLEAR_METHODS[0].byte_address, 0x19d0);
}

#[test]
fn a_hand_built_clear_surface_word_decodes_field_by_field() {
    // Z, B and A set; MRT_SELECT 5 (bits 9:6); RT_ARRAY_INDEX 0x2a3 (bits 25:10).
    let word = 0b1 | (1 << 4) | (1 << 5) | (5 << 6) | (0x2a3 << 10);
    assert_eq!(
        ClearSurface::decode(word),
        ClearSurface {
            depth: true,
            stencil: false,
            color: ColorWrites::BLUE.union(ColorWrites::ALPHA),
            target: 5,
            layer: 0x2a3,
        }
    );
    // S, R and G; the highest target and layer; bits above 25 are no field.
    let word = (1 << 1) | (1 << 2) | (1 << 3) | (0xf << 6) | (0xffff << 10) | (0x3f << 26);
    assert_eq!(
        ClearSurface::decode(word),
        ClearSurface {
            depth: false,
            stencil: true,
            color: ColorWrites::RED.union(ColorWrites::GREEN),
            target: 0xf,
            layer: 0xffff,
        }
    );
}

#[test]
fn encode_refuses_a_target_or_layer_past_its_field() {
    let edge = ClearSurface {
        depth: false,
        stencil: false,
        color: ColorWrites::EMPTY,
        target: clc797::CLEAR_SURFACE_MRT_SELECT.max(),
        layer: clc797::CLEAR_SURFACE_RT_ARRAY_INDEX.max(),
    };
    assert!(edge.encode().is_some());
    let target = ClearSurface {
        target: edge.target + 1,
        ..edge
    };
    assert_eq!(target.encode(), None);
    let layer = ClearSurface {
        layer: edge.layer + 1,
        ..edge
    };
    assert_eq!(layer.encode(), None);
}

proptest! {
    #![proptest_config(ProptestConfig {
        rng_seed: RngSeed::Fixed(0x6324_0428),
        ..ProptestConfig::default()
    })]

    #[test]
    fn clear_surface_round_trips(
        depth: bool,
        stencil: bool,
        mask in 0u32..16,
        target in 0..=clc797::CLEAR_SURFACE_MRT_SELECT.max(),
        layer in 0..=clc797::CLEAR_SURFACE_RT_ARRAY_INDEX.max(),
    ) {
        let surface = ClearSurface {
            depth,
            stencil,
            color: ColorWrites::from_bits(mask).unwrap(),
            target,
            layer,
        };
        let word = surface.encode().unwrap();
        prop_assert_eq!(ClearSurface::decode(word), surface);
        prop_assert_eq!(ClearSurface::decode(word).encode(), Some(word));
    }
}

#[test]
fn each_clear_control_bit_decodes_alone() {
    type Getter = fn(ClearControl) -> bool;
    let bits: [(Field, Getter); 4] = [
        (
            clc797::SET_CLEAR_SURFACE_CONTROL_RESPECT_STENCIL_MASK,
            ClearControl::respect_stencil_mask,
        ),
        (
            clc797::SET_CLEAR_SURFACE_CONTROL_USE_CLEAR_RECT,
            ClearControl::use_clear_rect,
        ),
        (
            clc797::SET_CLEAR_SURFACE_CONTROL_USE_SCISSOR0,
            ClearControl::use_scissor0,
        ),
        (
            clc797::SET_CLEAR_SURFACE_CONTROL_USE_VIEWPORT_CLIP0,
            ClearControl::use_viewport_clip0,
        ),
    ];
    for (i, (field, _)) in bits.iter().enumerate() {
        let control = ClearControl::from_word(fields(&[(*field, 1)]));
        assert_eq!(control.word(), field.insert(0, 1).unwrap());
        for (j, (_, getter)) in bits.iter().enumerate() {
            assert_eq!(getter(control), i == j, "bit {i} read by getter {j}");
        }
    }
    for (_, getter) in bits {
        assert!(!getter(ClearControl::from_word(0)));
    }
}

// ---------------------------------------------------------------------------------------------
// Values.

#[test]
fn integer_clear_values_follow_edens_formula() {
    use PixelFormat as P;
    // Unsigned: `u32(f32(bits << 1) * value)`; signed: `s32(f32((bits - 1) << 1) * (value -
    // 0.5))`. Hand-computed from Eden's `RasterizerVulkan::Clear`.
    assert_eq!(
        color_value(P::A8B8G8R8Uint, [1.0, 0.5, 2.0, 0.0]),
        ClearValue::Uint([16, 8, 32, 0])
    );
    assert_eq!(
        color_value(P::A2B10G10R10Uint, [1.0, 0.25, 3.0, 0.1]),
        ClearValue::Uint([20, 5, 60, 2])
    );
    assert_eq!(
        color_value(P::R16G16B16A16Uint, [1.0, 0.5, 4.0, 0.75]),
        ClearValue::Uint([32, 16, 128, 24])
    );
    assert_eq!(
        color_value(P::R32G32B32A32Uint, [1.0, 0.5, 2.0, 0.0]),
        ClearValue::Uint([64, 32, 128, 0])
    );
    assert_eq!(
        color_value(P::A8B8G8R8Sint, [1.0, 0.5, 0.0, 2.0]),
        ClearValue::Sint([7, 0, -7, 21])
    );
    assert_eq!(
        color_value(P::R16G16Sint, [1.0, 0.5, 0.0, 1.5]),
        ClearValue::Sint([15, 0, -15, 30])
    );
    assert_eq!(
        color_value(P::R32Sint, [1.0, 0.5, 0.0, 3.0]),
        ClearValue::Sint([31, 0, -31, 155])
    );
    // A float or normalised format takes the floats as they are, each channel distinct.
    let floats = [0.25, -2.0, 3.5, 100.0];
    assert_eq!(
        color_value(P::R32G32B32A32Float, floats),
        ClearValue::Float(floats)
    );
    assert_eq!(
        color_value(P::A8B8G8R8Unorm, floats),
        ClearValue::Float(floats)
    );
}

#[test]
fn integer_component_bits_are_edens_table() {
    use PixelFormat as P;
    // `PixelComponentSizeBitsInteger`, row by row.
    for format in [
        P::A8B8G8R8Sint,
        P::A8B8G8R8Uint,
        P::R8Sint,
        P::R8Uint,
        P::R8G8Sint,
        P::R8G8Uint,
    ] {
        assert_eq!(integer_component_bits(format), Some(8), "{format:?}");
    }
    assert_eq!(integer_component_bits(P::A2B10G10R10Uint), Some(10));
    for format in [
        P::R16G16B16A16Sint,
        P::R16G16B16A16Uint,
        P::R16Uint,
        P::R16Sint,
        P::R16G16Uint,
        P::R16G16Sint,
    ] {
        assert_eq!(integer_component_bits(format), Some(16), "{format:?}");
    }
    for format in [
        P::R32G32B32A32Uint,
        P::R32G32B32A32Sint,
        P::R32G32Sint,
        P::R32G32Uint,
        P::R32Uint,
        P::R32Sint,
    ] {
        assert_eq!(integer_component_bits(format), Some(32), "{format:?}");
    }
    for format in [P::A8B8G8R8Unorm, P::R32Float, P::D32Float] {
        assert_eq!(integer_component_bits(format), None, "{format:?}");
    }
}

#[test]
fn integer_conversions_saturate() {
    use PixelFormat as P;
    assert_eq!(
        color_value(P::R32Uint, [-1.0, 1e20, f32::NAN, f32::INFINITY]),
        ClearValue::Uint([0, u32::MAX, 0, u32::MAX])
    );
    assert_eq!(
        color_value(P::R32Sint, [1e20, -1e20, f32::NAN, f32::NEG_INFINITY]),
        ClearValue::Sint([i32::MAX, i32::MIN, 0, i32::MIN])
    );
}

#[test]
fn clamp_to_area_keeps_what_lies_inside() {
    let rect = |x, y, width, height| Rect {
        x,
        y,
        width,
        height,
    };
    // Ending exactly at the area's end: unchanged; one past: one shorter.
    assert_eq!(
        clamp_to_area(rect(2, 1, 6, 5), 8, 6),
        Some(rect(2, 1, 6, 5))
    );
    assert_eq!(
        clamp_to_area(rect(2, 1, 7, 6), 8, 6),
        Some(rect(2, 1, 6, 5))
    );
    // Starting on the last column and row: one texel; at the end: nothing.
    assert_eq!(
        clamp_to_area(rect(7, 5, 9, 9), 8, 6),
        Some(rect(7, 5, 1, 1))
    );
    assert_eq!(clamp_to_area(rect(8, 0, 9, 9), 8, 6), None);
    assert_eq!(clamp_to_area(rect(0, 6, 9, 9), 8, 6), None);
    // An empty extent, or an empty area.
    assert_eq!(clamp_to_area(rect(0, 0, 0, 4), 8, 6), None);
    assert_eq!(clamp_to_area(rect(0, 0, 4, 0), 8, 6), None);
    assert_eq!(clamp_to_area(rect(0, 0, 4, 4), 0, 6), None);
    assert_eq!(clamp_to_area(rect(0, 0, 4, 4), 8, 0), None);
    // The whole scissor, and an end past `u32::MAX`.
    assert_eq!(clamp_to_area(WHOLE, 8, 6), Some(rect(0, 0, 8, 6)));
    assert_eq!(
        clamp_to_area(rect(u32::MAX - 1, 0, u32::MAX, 1), u32::MAX, 1),
        Some(rect(u32::MAX - 1, 0, 1, 1))
    );
}

// ---------------------------------------------------------------------------------------------
// Golden lists.

#[test]
fn golden_colour_only() {
    let mut rig = Rig::new();
    rig.basic();
    rig.clear_color([1.0, 0.0, 0.25, 0.5]);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    insta::assert_snapshot!(rig.submit());
}

#[test]
fn golden_depth_only() {
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32, WIDTH, HEIGHT, 1);
    rig.clear_depth(0.75);
    assert_eq!(rig.clear(depth_stencil(true, false)), ONLY_DEPTH_STENCIL);
    insta::assert_snapshot!(rig.submit());
}

#[test]
fn golden_stencil_only() {
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
    rig.clear_stencil(0x5a, FULL_STENCIL_MASK, 0x3c);
    assert_eq!(rig.clear(depth_stencil(false, true)), ONLY_DEPTH_STENCIL);
    insta::assert_snapshot!(rig.submit());
}

#[test]
fn golden_all() {
    let mut rig = Rig::new();
    rig.basic();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
    rig.clear_color([0.0, 1.0, 0.5, 0.125]);
    rig.clear_depth(0.25);
    rig.clear_stencil(0x81, FULL_STENCIL_MASK, 0xff);
    let surface = ClearSurface {
        depth: true,
        stencil: true,
        color: ColorWrites::ALL,
        target: 0,
        layer: 0,
    };
    assert_eq!(
        rig.clear(surface),
        Cleared {
            color: Some(Part::Cleared),
            depth_stencil: Some(Part::Cleared),
        }
    );
    // Colour first, then depth and stencil, as Eden.
    let clears = rig.clears();
    assert_eq!(clears.len(), 2);
    assert!(matches!(clears[0].value, ClearValue::Float(_)));
    assert_eq!(
        clears[1].value,
        ClearValue::DepthStencil {
            depth: 0.25,
            stencil: 0x81
        }
    );
    insta::assert_snapshot!(rig.submit());
}

#[test]
fn golden_one_layer() {
    let mut rig = Rig::new();
    rig.color_target(0, TARGET0, clc797::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8, 3);
    rig.ct_select(1);
    rig.clear_color([1.0, 1.0, 1.0, 1.0]);
    let surface = ClearSurface {
        layer: 2,
        ..color(ColorWrites::ALL)
    };
    assert_eq!(rig.clear(surface), ONLY_COLOR);
    insta::assert_snapshot!(rig.submit());
}

#[test]
fn golden_scissored() {
    let mut rig = Rig::new();
    rig.basic();
    rig.clear_color([0.0, 0.0, 1.0, 1.0]);
    rig.scissor(1, 5, 2, 4);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    assert_eq!(
        rig.clears()[0].rect,
        Some(Rect {
            x: 1,
            y: 2,
            width: 4,
            height: 2
        })
    );
    insta::assert_snapshot!(rig.submit());
}

#[test]
fn golden_masked() {
    let mut rig = Rig::new();
    rig.basic();
    rig.clear_color([1.0, 0.0, 1.0, 0.0]);
    rig.scissor(1, 5, 2, 4);
    let mask = ColorWrites::RED.union(ColorWrites::BLUE);
    assert_eq!(rig.clear(color(mask)), ONLY_COLOR);
    // One clear command through the mask, within the scissor.
    assert_eq!(rig.kinds(), ["clear"]);
    let clear = rig.clears()[0];
    assert_eq!(clear.color_mask, mask);
    assert_eq!(clear.stencil_mask, ALL_STENCIL_BITS);
    assert_eq!(
        clear.rect,
        Some(Rect {
            x: 1,
            y: 2,
            width: 4,
            height: 2
        })
    );
    insta::assert_snapshot!(rig.submit());
}

#[test]
fn golden_masked_stencil() {
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
    rig.clear_depth(0.5);
    rig.clear_stencil(0x33, 0x0f, 0xf0);
    assert_eq!(rig.clear(depth_stencil(true, true)), ONLY_DEPTH_STENCIL);
    assert_eq!(rig.kinds(), ["clear"]);
    let clear = rig.clears()[0];
    assert_eq!(clear.stencil_mask, 0x0f);
    assert_eq!(
        clear.value,
        ClearValue::DepthStencil {
            depth: 0.5,
            stencil: 0x33
        }
    );
    insta::assert_snapshot!(rig.submit());
}

// ---------------------------------------------------------------------------------------------
// Pixels.

#[test]
fn the_cleared_pixels_are_the_clear_colour() {
    let mut rig = Rig::new();
    rig.basic();
    rig.clear_color([1.0, 0.0, 1.0, 0.0]);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    let texture = rig.texture_at(TARGET0);
    let texels = rig.texels(texture, 0);
    let pixels = usize::try_from(WIDTH * HEIGHT).unwrap();
    assert_eq!(texels, [0xff, 0x00, 0xff, 0x00].repeat(pixels));
}

#[test]
fn a_float_target_is_cleared_to_each_channels_value() {
    let mut rig = Rig::new();
    rig.color_target(
        0,
        TARGET0,
        clc797::SET_COLOR_TARGET_FORMAT_V_RF32_GF32_BF32_AF32,
        1,
    );
    rig.ct_select(1);
    let values = [0.25f32, -2.0, 3.5, 100.0];
    rig.clear_color(values);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    let texture = rig.texture_at(TARGET0);
    let texel: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let pixels = usize::try_from(WIDTH * HEIGHT).unwrap();
    assert_eq!(rig.texels(texture, 0), texel.repeat(pixels));
}

#[test]
fn an_integer_target_is_cleared_to_the_converted_value() {
    let mut rig = Rig::new();
    rig.color_target(
        0,
        TARGET0,
        clc797::SET_COLOR_TARGET_FORMAT_V_RU32_GU32_BU32_AU32,
        1,
    );
    rig.ct_select(1);
    rig.clear_color([1.0, 0.5, 2.0, 0.0]);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    let texture = rig.texture_at(TARGET0);
    let texel: Vec<u8> = [64u32, 32, 128, 0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let pixels = usize::try_from(WIDTH * HEIGHT).unwrap();
    assert_eq!(rig.texels(texture, 0), texel.repeat(pixels));
}

#[test]
fn a_scissored_clear_changes_only_the_pixels_inside() {
    let mut rig = Rig::new();
    rig.basic();
    rig.clear_color([1.0, 1.0, 1.0, 1.0]);
    rig.scissor(1, 5, 2, 4);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    let texture = rig.texture_at(TARGET0);
    let texels = rig.texels(texture, 0);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let inside = (1..5).contains(&x) && (2..4).contains(&y);
            let at = usize::try_from((y * WIDTH + x) * 4).unwrap();
            let expect = if inside { [0xff; 4] } else { [0; 4] };
            assert_eq!(texels[at..at + 4], expect, "({x}, {y})");
        }
    }
}

#[test]
fn a_clear_of_layer_zero_of_a_layered_target_clears_that_layer_alone() {
    let mut rig = Rig::new();
    rig.color_target(0, TARGET0, clc797::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8, 3);
    rig.ct_select(1);
    rig.clear_color([1.0, 1.0, 1.0, 1.0]);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    let texture = rig.texture_at(TARGET0);
    let pixels = usize::try_from(WIDTH * HEIGHT * 4).unwrap();
    assert_eq!(rig.texels(texture, 0), vec![0xff; pixels]);
    assert_eq!(rig.texels(texture, 1), vec![0; pixels]);
    assert_eq!(rig.texels(texture, 2), vec![0; pixels]);
}

// ---------------------------------------------------------------------------------------------
// Each bit, both sides.

#[test]
fn each_clear_surface_bit_asks_for_its_part() {
    let mut rig = Rig::new();
    rig.basic();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
    rig.clear_color([1.0, 0.0, 1.0, 0.0]);
    rig.clear_stencil(0, FULL_STENCIL_MASK, 0);
    // No bit: nothing.
    assert_eq!(rig.clear_word(0), Ok(Cleared::NOTHING));
    assert_eq!(rig.clears(), []);
    // Each colour bit alone: a colour clear through that channel.
    for (field, bit) in CHANNELS {
        let cleared = rig.clear_word(fields(&[(field, 1)])).unwrap();
        assert_eq!(cleared, ONLY_COLOR, "{bit:?}");
        assert_eq!(rig.clears().last().unwrap().color_mask, bit);
    }
    // Z alone, S alone, both.
    for (z, s, value) in [
        (1, 0, ClearValue::Depth(0.0)),
        (0, 1, ClearValue::Stencil(0)),
        (
            1,
            1,
            ClearValue::DepthStencil {
                depth: 0.0,
                stencil: 0,
            },
        ),
    ] {
        let word = fields(&[
            (clc797::CLEAR_SURFACE_Z_ENABLE, z),
            (clc797::CLEAR_SURFACE_STENCIL_ENABLE, s),
        ]);
        assert_eq!(rig.clear_word(word), Ok(ONLY_DEPTH_STENCIL));
        assert_eq!(rig.clears().last().unwrap().value, value);
    }
}

#[test]
fn use_scissor0_bounds_the_clear_and_its_absence_does_not() {
    let mut rig = Rig::new();
    rig.basic();
    rig.scissor(1, 5, 2, 4);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    rig.control(0);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    let rects: Vec<_> = rig.clears().iter().map(|c| c.rect).collect();
    assert_eq!(
        rects,
        [
            Some(Rect {
                x: 1,
                y: 2,
                width: 4,
                height: 2
            }),
            None
        ]
    );
}

#[test]
fn a_disabled_scissor_0_is_the_whole_target() {
    let mut rig = Rig::new();
    rig.basic();
    rig.scissor(1, 5, 2, 4);
    rig.set(
        &clc797::SET_SCISSOR_ENABLE,
        &[0],
        clc797::SET_SCISSOR_ENABLE_V_FALSE,
    );
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    assert_eq!(rig.clears()[0].rect, None);
}

#[test]
fn the_unread_control_bits_change_nothing() {
    let script = |control: u32| {
        let mut rig = Rig::new();
        rig.basic();
        rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
        rig.clear_color([1.0, 0.0, 0.0, 1.0]);
        rig.clear_stencil(0x12, 0x0f, 0xff);
        // A clear rect and a viewport clip smaller than the target, which would bound the clear
        // if their bits were read.
        rig.set(
            &clc797::SET_CLEAR_RECT_HORIZONTAL,
            &[],
            fields(&[
                (clc797::SET_CLEAR_RECT_HORIZONTAL_XMIN, 1),
                (clc797::SET_CLEAR_RECT_HORIZONTAL_XMAX, 2),
            ]),
        );
        rig.set(
            &clc797::SET_CLEAR_RECT_VERTICAL,
            &[],
            fields(&[
                (clc797::SET_CLEAR_RECT_VERTICAL_YMIN, 1),
                (clc797::SET_CLEAR_RECT_VERTICAL_YMAX, 2),
            ]),
        );
        rig.set(
            &clc797::SET_VIEWPORT_CLIP_HORIZONTAL,
            &[0],
            fields(&[(clc797::SET_VIEWPORT_CLIP_HORIZONTAL_WIDTH, 1)]),
        );
        rig.control(control);
        let surface = ClearSurface {
            depth: false,
            stencil: true,
            color: ColorWrites::ALL,
            target: 0,
            layer: 0,
        };
        let cleared = rig.clear(surface);
        (cleared, rig.submit())
    };
    let plain = script(0);
    assert_eq!(
        plain.0,
        Cleared {
            color: Some(Part::Cleared),
            depth_stencil: Some(Part::Cleared),
        }
    );
    assert!(plain.1.contains("writing stencil 0x0f"), "{}", plain.1);
    for field in [
        clc797::SET_CLEAR_SURFACE_CONTROL_RESPECT_STENCIL_MASK,
        clc797::SET_CLEAR_SURFACE_CONTROL_USE_CLEAR_RECT,
        clc797::SET_CLEAR_SURFACE_CONTROL_USE_VIEWPORT_CLIP0,
    ] {
        assert_eq!(script(fields(&[(field, 1)])), plain, "{field:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// The area.

#[test]
fn a_reversed_scissor_clears_nothing() {
    for (x, x_end, y, y_end) in [(5, 3, 1, 4), (1, 4, 4, 2), (3, 3, 1, 4), (1, 4, 2, 2)] {
        let mut rig = Rig::new();
        rig.basic();
        rig.scissor(x, x_end, y, y_end);
        assert_eq!(
            rig.clear(color(ColorWrites::ALL)),
            Cleared {
                color: Some(Part::Skipped(ClearSkip::EmptyArea)),
                depth_stencil: None,
            }
        );
        assert_eq!(rig.commands().len(), 0);
    }
}

#[test]
fn a_scissor_past_the_render_area_is_clamped_to_it() {
    let mut rig = Rig::new();
    rig.basic();
    rig.scissor(2, 20, 3, 30);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    assert_eq!(
        rig.clears()[0].rect,
        Some(Rect {
            x: 2,
            y: 3,
            width: WIDTH - 2,
            height: HEIGHT - 3
        })
    );
    // A scissor starting at the area's end clears nothing.
    let mut rig = Rig::new();
    rig.basic();
    rig.scissor(WIDTH, WIDTH + 4, 0, HEIGHT);
    assert_eq!(
        rig.clear(color(ColorWrites::ALL)).color,
        Some(Part::Skipped(ClearSkip::EmptyArea))
    );
    // One before it clears one column.
    let mut rig = Rig::new();
    rig.basic();
    rig.scissor(WIDTH - 1, WIDTH + 4, 0, HEIGHT);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    assert_eq!(
        rig.clears()[0].rect,
        Some(Rect {
            x: WIDTH - 1,
            y: 0,
            width: 1,
            height: HEIGHT
        })
    );
}

#[test]
fn the_render_area_is_the_smallest_bound_target() {
    let mut rig = Rig::new();
    rig.basic();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32, 4, 3, 1);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    assert_eq!(
        rig.clears()[0].rect,
        Some(Rect {
            x: 0,
            y: 0,
            width: 4,
            height: 3
        })
    );
}

#[test]
fn a_lower_left_origin_flips_the_scissor() {
    let mut rig = Rig::new();
    rig.basic();
    rig.set(
        &clc797::SET_WINDOW_ORIGIN,
        &[],
        fields(&[(
            clc797::SET_WINDOW_ORIGIN_MODE,
            clc797::SET_WINDOW_ORIGIN_MODE_LOWER_LEFT,
        )]),
    );
    rig.scissor(1, 5, 1, 3);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    // y from `HEIGHT - 3` to `HEIGHT - 1`.
    assert_eq!(
        rig.clears()[0].rect,
        Some(Rect {
            x: 1,
            y: HEIGHT - 3,
            width: 4,
            height: 2
        })
    );
}

// ---------------------------------------------------------------------------------------------
// Targets.

#[test]
fn a_colour_target_that_is_not_bound_is_skipped() {
    let skipped = Cleared {
        color: Some(Part::Skipped(ClearSkip::NotBound)),
        depth_stencil: None,
    };
    let second = ClearSurface {
        target: 1,
        ..color(ColorWrites::ALL)
    };
    // Target 1 with a count of 1 (at the count): skipped.
    let mut rig = Rig::new();
    rig.basic();
    rig.color_target(1, TARGET1, clc797::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8, 1);
    assert_eq!(rig.clear(second), skipped);
    // With a count of 2 (below it): target 1's own image, not target 0's.
    rig.ct_select(2);
    assert_eq!(rig.clear(second), ONLY_COLOR);
    let target1 = rig.texture_at(TARGET1);
    let texture_of = |rig: &Rig, view| {
        rig.textures
            .entries()
            .find(|e| e.texture == target1)
            .map(|_| view)
    };
    assert!(texture_of(&rig, rig.clears()[0].target).is_some());
    // Below the count but `DISABLED`: skipped.
    rig.set(
        &clc797::SET_COLOR_TARGET_FORMAT,
        &[1],
        clc797::SET_COLOR_TARGET_FORMAT_V_DISABLED,
    );
    assert_eq!(rig.clear(second), skipped);
    // The highest `MRT_SELECT`, past every target: skipped, no panic.
    let last = ClearSurface {
        target: clc797::CLEAR_SURFACE_MRT_SELECT.max(),
        ..color(ColorWrites::ALL)
    };
    assert_eq!(rig.clear(last), skipped);
    // A count past the eight colour targets bounds nothing beyond them.
    rig.ct_select(clc797::SET_CT_SELECT_TARGET_COUNT.max());
    assert_eq!(rig.clear(last), skipped);
    let ninth = ClearSurface {
        target: COLOR_TARGETS,
        ..color(ColorWrites::ALL)
    };
    assert_eq!(rig.clear(ninth), skipped);
}

#[test]
fn a_word_asking_nothing_reads_no_state() {
    let mut rig = Rig::new();
    rig.basic();
    // A colour target format no draw could use: any clear that reads the targets fails.
    rig.set(
        &clc797::SET_COLOR_TARGET_FORMAT,
        &[0],
        clc797::SET_COLOR_TARGET_FORMAT_V_Z1R5G5B5,
    );
    assert_eq!(rig.clear_word(0), Ok(Cleared::NOTHING));
    assert_eq!(
        rig.clear_word(color(ColorWrites::ALL).encode().unwrap()),
        Err(DrawError::TargetFormat {
            value: clc797::SET_COLOR_TARGET_FORMAT_V_Z1R5G5B5
        })
    );
}

#[test]
fn the_cleared_colour_target_is_the_one_mrt_select_names() {
    let mut rig = Rig::new();
    rig.basic();
    rig.color_target(1, TARGET1, clc797::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8, 1);
    rig.ct_select(2);
    rig.clear_color([1.0, 1.0, 1.0, 1.0]);
    let second = ClearSurface {
        target: 1,
        ..color(ColorWrites::ALL)
    };
    assert_eq!(rig.clear(second), ONLY_COLOR);
    let t1 = rig.texture_at(TARGET1);
    let pixels = usize::try_from(WIDTH * HEIGHT * 4).unwrap();
    assert_eq!(rig.texels(t1, 0), vec![0xff; pixels]);
    // Target 0 was not even made.
    assert!(rig.textures.entries().all(|e| e.info.address != TARGET0));
}

#[test]
fn a_target_at_address_zero_is_cleared_like_any_other() {
    let mut rig = Rig::new();
    rig.color_target(0, 0, clc797::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8, 1);
    rig.ct_select(1);
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32, WIDTH, HEIGHT, 1);
    rig.address(&clc797::SET_ZT_A, &clc797::SET_ZT_B, &[], 0);
    let surface = ClearSurface {
        depth: true,
        ..color(ColorWrites::ALL)
    };
    assert_eq!(
        rig.clear(surface),
        Cleared {
            color: Some(Part::Cleared),
            depth_stencil: Some(Part::Cleared),
        }
    );
    assert_eq!(rig.clears().len(), 2);
}

#[test]
fn a_depth_clear_with_no_depth_target_is_skipped() {
    let mut rig = Rig::new();
    rig.basic();
    rig.clear_color([1.0, 1.0, 1.0, 1.0]);
    let surface = ClearSurface {
        depth: true,
        stencil: true,
        ..color(ColorWrites::ALL)
    };
    // The colour part is still done.
    assert_eq!(
        rig.clear(surface),
        Cleared {
            color: Some(Part::Cleared),
            depth_stencil: Some(Part::Skipped(ClearSkip::NotBound)),
        }
    );
    assert_eq!(rig.clears().len(), 1);
}

#[test]
fn only_the_aspects_the_format_has_are_cleared() {
    let cases = [
        // Stencil-only format, depth asked: nothing to clear.
        (
            clc797::SET_ZT_FORMAT_V_S8,
            true,
            false,
            Err(ClearSkip::NoAspect),
        ),
        // Depth-only format, stencil asked.
        (
            clc797::SET_ZT_FORMAT_V_Z16,
            false,
            true,
            Err(ClearSkip::NoAspect),
        ),
        // Depth-only format, both asked: depth.
        (
            clc797::SET_ZT_FORMAT_V_ZF32,
            true,
            true,
            Ok(ClearValue::Depth(0.5)),
        ),
        // Stencil-only format, both asked: stencil.
        (
            clc797::SET_ZT_FORMAT_V_S8,
            true,
            true,
            Ok(ClearValue::Stencil(9)),
        ),
        // Both aspects, depth asked: depth alone, the stencil kept.
        (
            clc797::SET_ZT_FORMAT_V_Z24S8,
            true,
            false,
            Ok(ClearValue::Depth(0.5)),
        ),
    ];
    for (format, depth, stencil, expect) in cases {
        let mut rig = Rig::new();
        rig.depth_target(format, WIDTH, HEIGHT, 1);
        rig.clear_depth(0.5);
        rig.clear_stencil(9, FULL_STENCIL_MASK, 0);
        let cleared = rig.clear(depth_stencil(depth, stencil));
        match expect {
            Err(skip) => {
                assert_eq!(
                    cleared.depth_stencil,
                    Some(Part::Skipped(skip)),
                    "{format:#x}"
                );
                assert_eq!(rig.commands().len(), 0);
            }
            Ok(value) => {
                assert_eq!(cleared, ONLY_DEPTH_STENCIL, "{format:#x}");
                assert_eq!(rig.clears()[0].value, value, "{format:#x}");
            }
        }
    }
}

#[test]
fn a_layer_past_the_targets_is_skipped() {
    let mut rig = Rig::new();
    rig.color_target(0, TARGET0, clc797::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8, 3);
    rig.ct_select(1);
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32, WIDTH, HEIGHT, 2);
    // The last layer of each: cleared.
    let last = |layer| ClearSurface {
        depth: true,
        layer,
        ..color(ColorWrites::ALL)
    };
    assert_eq!(
        rig.clear(last(1)),
        Cleared {
            color: Some(Part::Cleared),
            depth_stencil: Some(Part::Cleared),
        }
    );
    // One past the depth target's: depth skipped, colour cleared.
    assert_eq!(
        rig.clear(last(2)),
        Cleared {
            color: Some(Part::Cleared),
            depth_stencil: Some(Part::Skipped(ClearSkip::Layer {
                layer: 2,
                layers: 2
            })),
        }
    );
    // One past the colour target's.
    assert_eq!(
        rig.clear(last(3)).color,
        Some(Part::Skipped(ClearSkip::Layer {
            layer: 3,
            layers: 3
        }))
    );
    insta::assert_snapshot!(rig.submit());
}

#[test]
fn a_volume_target_deeper_than_one_slice_is_skipped() {
    let mut rig = Rig::new();
    rig.color_target(0, TARGET0, clc797::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8, 2);
    rig.set(
        &clc797::SET_COLOR_TARGET_MEMORY,
        &[0],
        fields(&[(
            clc797::SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL,
            clc797::SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_DEPTH_SIZE,
        )]),
    );
    rig.ct_select(1);
    // Two slices, and a layer past them too: the volume check comes first.
    let surface = ClearSurface {
        layer: 5,
        ..color(ColorWrites::ALL)
    };
    assert_eq!(
        rig.clear(surface).color,
        Some(Part::Skipped(ClearSkip::VolumeTarget))
    );
    // One slice is a 2D target.
    rig.set(&clc797::SET_COLOR_TARGET_THIRD_DIMENSION, &[0], 1);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
}

// ---------------------------------------------------------------------------------------------
// Values in the clear.

#[test]
fn a_depth_value_outside_zero_to_one_is_clamped_and_nan_is_skipped() {
    let cases = [
        (1.5f32, Some(1.0f32)),
        (-0.5, Some(0.0)),
        (f32::INFINITY, Some(1.0)),
        (f32::NEG_INFINITY, Some(0.0)),
        (1.0, Some(1.0)),
        (0.0, Some(0.0)),
        (f32::NAN, None),
    ];
    for (value, expect) in cases {
        let mut rig = Rig::new();
        rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
        rig.clear_depth(value);
        rig.clear_stencil(3, FULL_STENCIL_MASK, 0);
        let cleared = rig.clear(depth_stencil(true, true));
        if let Some(depth) = expect {
            assert_eq!(cleared, ONLY_DEPTH_STENCIL, "{value}");
            assert_eq!(
                rig.clears()[0].value,
                ClearValue::DepthStencil { depth, stencil: 3 },
                "{value}"
            );
        } else {
            assert_eq!(
                cleared.depth_stencil,
                Some(Part::Skipped(ClearSkip::DepthNotANumber))
            );
            assert_eq!(rig.commands().len(), 0);
        }
    }
    // A NaN depth value is not read when only stencil is asked.
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
    rig.clear_depth(f32::NAN);
    rig.clear_stencil(3, FULL_STENCIL_MASK, 0);
    assert_eq!(rig.clear(depth_stencil(false, true)), ONLY_DEPTH_STENCIL);
    assert_eq!(rig.clears()[0].value, ClearValue::Stencil(3));
}

#[test]
fn stencil_registers_are_their_low_byte() {
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_S8, WIDTH, HEIGHT, 1);
    // Bits above 7 of the value and of a full mask are dropped: a clear of 0xab.
    rig.clear_stencil(0x1ab, 0x1ff, 0x1ff);
    assert_eq!(rig.clear(depth_stencil(false, true)), ONLY_DEPTH_STENCIL);
    assert_eq!(rig.clears()[0].value, ClearValue::Stencil(0xab));
    assert_eq!(rig.clears()[0].stencil_mask, ALL_STENCIL_BITS);
    // A mask word of 0x1fe is the partial mask 0xfe.
    rig.clear_stencil(0x1ab, 0x1fe, 0x2c3);
    assert_eq!(rig.clear(depth_stencil(false, true)), ONLY_DEPTH_STENCIL);
    let clear = rig.clears()[1];
    assert_eq!(clear.stencil_mask, 0xfe);
    assert_eq!(clear.value, ClearValue::Stencil(0xab));
}

#[test]
fn a_partial_stencil_mask_is_written_through_and_full_or_zero_writes_every_bit() {
    for (mask, written) in [
        (FULL_STENCIL_MASK, ALL_STENCIL_BITS),
        (0, ALL_STENCIL_BITS),
        (0xfe, 0xfe),
        (0x01, 0x01),
    ] {
        let mut rig = Rig::new();
        rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
        rig.clear_stencil(7, mask, 0xff);
        assert_eq!(rig.clear(depth_stencil(false, true)), ONLY_DEPTH_STENCIL);
        assert_eq!(rig.clears()[0].stencil_mask, written, "{mask:#x}");
    }
    // A partial mask with stencil not asked changes nothing: a depth clear of every bit.
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
    rig.clear_stencil(7, 0x0f, 0xff);
    assert_eq!(rig.clear(depth_stencil(true, false)), ONLY_DEPTH_STENCIL);
    assert_eq!(rig.clears()[0].value, ClearValue::Depth(0.0));
    assert_eq!(rig.clears()[0].stencil_mask, ALL_STENCIL_BITS);
    // A partial mask on a format with no stencil changes nothing either.
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32, WIDTH, HEIGHT, 1);
    rig.clear_stencil(7, 0x0f, 0xff);
    assert_eq!(rig.clear(depth_stencil(true, true)), ONLY_DEPTH_STENCIL);
    assert_eq!(rig.clears()[0].value, ClearValue::Depth(0.0));
    assert_eq!(rig.clears()[0].stencil_mask, ALL_STENCIL_BITS);
}

/// [`stencil_write_mask`] against a model of Eden's `stencil_partial` test, for every mask and
/// both answers to whether stencil is written.
#[test]
fn the_stencil_write_mask_is_partial_only_between_its_two_full_values() {
    for use_stencil in [false, true] {
        for mask in 0..=u8::MAX {
            let partial = use_stencil && mask != 0xff && mask != 0x00;
            let expected = if partial { mask } else { 0xff };
            assert_eq!(
                stencil_write_mask(use_stencil, mask),
                expected,
                "{use_stencil} {mask:#x}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Masks: every part is one clear command, which the render backend writes through its masks.

#[test]
fn an_unmasked_part_carries_full_masks() {
    let mut rig = Rig::new();
    rig.basic();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
    rig.clear_stencil(0x44, FULL_STENCIL_MASK, 0xff);
    let surface = ClearSurface {
        depth: true,
        stencil: true,
        color: ColorWrites::ALL,
        target: 0,
        layer: 0,
    };
    rig.clear(surface);
    let clears = rig.clears();
    assert_eq!(clears.len(), 2);
    for clear in clears {
        assert_eq!(clear.color_mask, ColorWrites::ALL);
        assert_eq!(clear.stencil_mask, ALL_STENCIL_BITS);
    }
}

#[test]
fn a_masked_colour_clear_writes_through_its_mask() {
    let mut rig = Rig::new();
    rig.basic();
    rig.clear_color([0.0, 1.0, 0.0, 1.0]);
    rig.scissor(1, 5, 2, 4);
    let mask = ColorWrites::GREEN.union(ColorWrites::ALPHA);
    assert_eq!(rig.clear(color(mask)), ONLY_COLOR);
    // A clear command and nothing else: no targets, no draw, no parameter upload.
    assert_eq!(rig.kinds(), ["clear"]);
    assert_eq!(rig.draws.last_targets, None);
    assert_eq!(rig.clears()[0].color_mask, mask);
    // The recording backend writes green and alpha inside the scissor and keeps red and blue.
    let texture = rig.texture_at(TARGET0);
    let texels = rig.texels(texture, 0);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let inside = (1..5).contains(&x) && (2..4).contains(&y);
            let at = usize::try_from((y * WIDTH + x) * 4).unwrap();
            let expect = if inside { [0, 0xff, 0, 0xff] } else { [0; 4] };
            assert_eq!(texels[at..at + 4], expect, "({x}, {y})");
        }
    }
}

#[test]
fn a_masked_integer_clear_writes_the_converted_value() {
    let mut rig = Rig::new();
    rig.color_target(
        0,
        TARGET0,
        clc797::SET_COLOR_TARGET_FORMAT_V_AS8BS8GS8RS8,
        1,
    );
    rig.ct_select(1);
    rig.clear_color([1.0, 0.5, 0.0, 2.0]);
    assert_eq!(rig.clear(color(ColorWrites::RED)), ONLY_COLOR);
    // Eden's formula on 8 bits: (8 - 1) * 2 * (v - 0.5).
    assert_eq!(rig.clears()[0].value, ClearValue::Sint([7, 0, -7, 21]));
    assert_eq!(rig.clears()[0].color_mask, ColorWrites::RED);
    let texture = rig.texture_at(TARGET0);
    let pixels = usize::try_from(WIDTH * HEIGHT).unwrap();
    assert_eq!(rig.texels(texture, 0), [7, 0, 0, 0].repeat(pixels));
}

#[test]
fn a_masked_stencil_clear_writes_the_bits_of_its_mask() {
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_S8, WIDTH, HEIGHT, 1);
    // A full clear to 0xa0 first: the target is never uploaded, so its stencil is unknown
    // until a clear writes every bit.
    rig.clear_stencil(0xa0, FULL_STENCIL_MASK, 0xff);
    assert_eq!(rig.clear(depth_stencil(false, true)), ONLY_DEPTH_STENCIL);
    rig.clear_stencil(0x44, 0x0f, 0xff);
    assert_eq!(rig.clear(depth_stencil(false, true)), ONLY_DEPTH_STENCIL);
    assert_eq!(rig.kinds(), ["clear", "clear"]);
    assert_eq!(rig.clears()[1].stencil_mask, 0x0f);
    // (0xa0 & !0x0f) | (0x44 & 0x0f) is 0xa4.
    let texture = rig.texture_at(DEPTH);
    let pixels = usize::try_from(WIDTH * HEIGHT).unwrap();
    assert_eq!(rig.texels(texture, 0), vec![0xa4; pixels]);
}

#[test]
fn a_masked_stencil_clear_writes_depth_only_when_depth_is_asked() {
    for (depth, value) in [
        (false, ClearValue::Stencil(0x44)),
        (
            true,
            ClearValue::DepthStencil {
                depth: 0.5,
                stencil: 0x44,
            },
        ),
    ] {
        let mut rig = Rig::new();
        rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
        rig.clear_depth(0.5);
        rig.clear_stencil(0x44, 0x0f, 0xff);
        assert_eq!(
            rig.clear(depth_stencil(depth, true)),
            ONLY_DEPTH_STENCIL,
            "{depth}"
        );
        let clear = rig.clears()[0];
        assert_eq!(clear.value, value);
        assert_eq!(clear.stencil_mask, 0x0f);
    }
}

#[test]
fn two_masked_parts_of_one_clear_are_two_clear_commands() {
    let mut rig = Rig::new();
    rig.basic();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
    rig.clear_stencil(0x44, 0x0f, 0xff);
    let surface = ClearSurface {
        depth: false,
        stencil: true,
        color: ColorWrites::RED,
        target: 0,
        layer: 0,
    };
    assert_eq!(
        rig.clear(surface),
        Cleared {
            color: Some(Part::Cleared),
            depth_stencil: Some(Part::Cleared),
        }
    );
    assert_eq!(rig.kinds(), ["clear", "clear"]);
    let clears = rig.clears();
    assert_eq!(
        (clears[0].color_mask, clears[0].stencil_mask),
        (ColorWrites::RED, ALL_STENCIL_BITS)
    );
    assert_eq!(
        (clears[1].color_mask, clears[1].stencil_mask),
        (ColorWrites::ALL, 0x0f)
    );
}

#[test]
fn a_clear_makes_no_host_object() {
    let mut rig = Rig::new();
    rig.basic();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
    rig.clear_stencil(0x44, 0x0f, 0xff);
    for mask in [ColorWrites::RED, ColorWrites::GREEN, ColorWrites::ALL] {
        rig.clear(ClearSurface {
            stencil: true,
            ..color(mask)
        });
    }
    rig.submit();
    let Rig { draws, backend, .. } = rig;
    drop(draws);
    let snapshot = backend.snapshot();
    // The texture cache's two targets and their views, and nothing of the clears' own.
    for kind in ["create sh", "create p", "create b", "destroy"] {
        assert!(
            !snapshot.lines().any(|l| l.starts_with(kind)),
            "{kind}: {snapshot}"
        );
    }
}

#[test]
fn a_failed_part_appends_no_part_of_the_clear() {
    // The depth target is wider than the device allows, so it cannot be made; the colour target
    // fits.
    let mut rig = Rig::with_caps(Caps {
        limits: Limits {
            max_texture_dimension_2d: WIDTH,
            ..caps().limits
        },
        ..caps()
    });
    rig.basic();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH * 2, HEIGHT, 1);
    rig.clear_stencil(0x44, 0x0f, 0xff);
    let surface = ClearSurface {
        depth: false,
        stencil: true,
        color: ColorWrites::ALL,
        target: 0,
        layer: 0,
    };
    let word = surface.encode().unwrap();
    let error = rig.clear_word(word).unwrap_err();
    assert!(matches!(error, DrawError::Texture(_)), "{error:?}");
    assert_eq!(rig.clears(), []);
    assert_eq!(rig.draws.clear_counters().failed, 1);
    // The next attempt, with a depth target that fits, does both.
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
    assert_eq!(
        rig.clear_word(word),
        Ok(Cleared {
            color: Some(Part::Cleared),
            depth_stencil: Some(Part::Cleared),
        })
    );
    assert_eq!(rig.clears().len(), 2);
}

// ---------------------------------------------------------------------------------------------
// Order of checks, conditional render and counters.

#[test]
fn render_disabled_comes_before_every_other_check() {
    let mut rig = Rig::new();
    rig.basic();
    // An empty area and an unbound depth target too.
    rig.scissor(5, 3, 1, 4);
    rig.set(
        &clc797::SET_RENDER_ENABLE_C,
        &[],
        clc797::SET_RENDER_ENABLE_C_MODE_FALSE,
    );
    let surface = ClearSurface {
        depth: true,
        ..color(ColorWrites::ALL)
    };
    let disabled = Some(Part::Skipped(ClearSkip::RenderDisabled));
    assert_eq!(
        rig.clear(surface),
        Cleared {
            color: disabled,
            depth_stencil: disabled,
        }
    );
    assert_eq!(rig.commands().len(), 0);
    // Turned back on: the empty area is next.
    rig.set(
        &clc797::SET_RENDER_ENABLE_C,
        &[],
        clc797::SET_RENDER_ENABLE_C_MODE_TRUE,
    );
    let empty = Some(Part::Skipped(ClearSkip::EmptyArea));
    assert_eq!(
        rig.clear(surface),
        Cleared {
            color: empty,
            depth_stencil: empty,
        }
    );
}

#[test]
fn the_area_comes_before_the_targets() {
    // An empty area and an unbound colour target: the area.
    let mut rig = Rig::new();
    rig.basic();
    rig.scissor(5, 3, 1, 4);
    let unbound = ClearSurface {
        target: 1,
        ..color(ColorWrites::ALL)
    };
    assert_eq!(
        rig.clear(unbound).color,
        Some(Part::Skipped(ClearSkip::EmptyArea))
    );
}

#[test]
fn the_aspect_comes_before_the_depth_value_and_the_value_before_the_layer() {
    // A stencil-only format asked for depth, with a NaN value: the aspect.
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_S8, WIDTH, HEIGHT, 1);
    rig.clear_depth(f32::NAN);
    let past = ClearSurface {
        layer: 4,
        ..depth_stencil(true, false)
    };
    assert_eq!(
        rig.clear(past).depth_stencil,
        Some(Part::Skipped(ClearSkip::NoAspect))
    );
    // A depth format with a NaN value and a layer past it: the value.
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32, WIDTH, HEIGHT, 1);
    rig.clear_depth(f32::NAN);
    assert_eq!(
        rig.clear(past).depth_stencil,
        Some(Part::Skipped(ClearSkip::DepthNotANumber))
    );
    // A good value: the layer.
    rig.clear_depth(0.5);
    assert_eq!(
        rig.clear(past).depth_stencil,
        Some(Part::Skipped(ClearSkip::Layer {
            layer: 4,
            layers: 1
        }))
    );
}

#[test]
fn clears_are_counted() {
    let mut rig = Rig::new();
    rig.basic();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
    rig.clear_stencil(0x44, 0x0f, 0xff);
    rig.clear_word(0).unwrap();
    rig.clear(color(ColorWrites::ALL));
    rig.clear(color(ColorWrites::RED));
    rig.clear(ClearSurface {
        target: 1,
        ..color(ColorWrites::ALL)
    });
    rig.clear(depth_stencil(true, true));
    assert_eq!(
        rig.draws.clear_counters(),
        ClearCounters {
            words: 5,
            cleared: 3,
            skipped: 1,
            nothing: 1,
            failed: 0,
        }
    );
}

// ---------------------------------------------------------------------------------------------
// Partial-area clears the render backend clears only whole, and the render area's targets.

/// The caps of [`caps`] with four samples allowed too.
fn msaa_caps() -> Caps {
    Caps {
        sample_counts: SampleCounts::X1.union(SampleCounts::X4),
        ..caps()
    }
}

/// `SET_ANTI_ALIAS` with four samples per pixel, 2 by 2.
fn four_samples(rig: &mut Rig) {
    rig.set(
        &clc797::SET_ANTI_ALIAS,
        &[],
        fields(&[(
            clc797::SET_ANTI_ALIAS_SAMPLES,
            clc797::SET_ANTI_ALIAS_SAMPLES_MODE_2X2,
        )]),
    );
}

#[test]
fn golden_scissored_depth() {
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32, WIDTH, HEIGHT, 1);
    rig.clear_depth(0.5);
    rig.scissor(1, 5, 2, 4);
    assert_eq!(rig.clear(depth_stencil(true, false)), ONLY_DEPTH_STENCIL);
    // A clear command with a rectangle: the render backend draws what it cannot clear in part.
    assert_eq!(rig.kinds(), ["clear"]);
    assert_eq!(
        rig.clears()[0].rect,
        Some(Rect {
            x: 1,
            y: 2,
            width: 4,
            height: 2
        })
    );
    insta::assert_snapshot!(rig.submit());
}

#[test]
fn golden_depth_larger_than_the_render_area() {
    let mut rig = Rig::new();
    rig.basic();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH * 2, HEIGHT * 2, 1);
    rig.clear_color([0.0, 1.0, 0.0, 1.0]);
    rig.clear_depth(0.5);
    rig.clear_stencil(0x5a, FULL_STENCIL_MASK, 0x3c);
    let surface = ClearSurface {
        depth: true,
        stencil: true,
        color: ColorWrites::ALL,
        target: 0,
        layer: 0,
    };
    assert_eq!(
        rig.clear(surface),
        Cleared {
            color: Some(Part::Cleared),
            depth_stencil: Some(Part::Cleared),
        }
    );
    // The colour target is the render area, so its clear is whole; the depth target is twice
    // as large, so its clear has the area as its rectangle, writing every stencil bit.
    let clears = rig.clears();
    assert_eq!(clears.len(), 2);
    assert_eq!(clears[0].rect, None);
    assert!(matches!(clears[0].value, ClearValue::Float(_)));
    assert_eq!(
        clears[1].rect,
        Some(Rect {
            x: 0,
            y: 0,
            width: WIDTH,
            height: HEIGHT
        })
    );
    assert_eq!(
        clears[1].value,
        ClearValue::DepthStencil {
            depth: 0.5,
            stencil: 0x5a
        }
    );
    assert_eq!(clears[1].stencil_mask, ALL_STENCIL_BITS);
    insta::assert_snapshot!(rig.submit());
}

#[test]
fn golden_scissored_msaa() {
    let mut rig = Rig::with_caps(msaa_caps());
    four_samples(&mut rig);
    rig.basic();
    rig.clear_color([0.0, 0.0, 1.0, 1.0]);
    rig.scissor(1, 5, 2, 4);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    assert_eq!(rig.kinds(), ["clear"]);
    assert!(rig.clears()[0].rect.is_some());
    let snapshot = rig.submit();
    let samples = format!("{} samples", SampleLayout::S2x2.samples());
    let texture = snapshot
        .lines()
        .find(|l| l.starts_with("create t0:"))
        .unwrap();
    assert!(texture.contains(&samples), "{texture}");
    insta::assert_snapshot!(snapshot);
}

#[test]
fn a_whole_multisampled_clear_is_a_clear_command() {
    let mut rig = Rig::with_caps(msaa_caps());
    four_samples(&mut rig);
    rig.basic();
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    assert_eq!(rig.clears()[0].rect, None);
    assert_eq!(rig.kinds(), ["clear"]);
    let target = rig
        .textures
        .entries()
        .find(|e| e.info.address == TARGET0)
        .unwrap()
        .info;
    assert_eq!(target.samples, SampleLayout::S2x2);
}

#[test]
fn a_masked_multisampled_clear_is_a_clear_command() {
    let mut rig = Rig::with_caps(msaa_caps());
    four_samples(&mut rig);
    rig.basic();
    assert_eq!(rig.clear(color(ColorWrites::RED)), ONLY_COLOR);
    assert_eq!(rig.kinds(), ["clear"]);
    assert_eq!(rig.clears()[0].color_mask, ColorWrites::RED);
}

#[test]
fn a_partial_area_depth_stencil_clear_carries_the_aspects_asked() {
    // (depth asked, stencil asked, `SET_STENCIL_MASK`): the value and the stencil bits written.
    for (depth, stencil, mask, value, written) in [
        (
            true,
            false,
            FULL_STENCIL_MASK,
            ClearValue::Depth(0.5),
            ALL_STENCIL_BITS,
        ),
        (
            false,
            true,
            FULL_STENCIL_MASK,
            ClearValue::Stencil(0x44),
            ALL_STENCIL_BITS,
        ),
        // A mask of 0 is a full clear (Eden's `stencil_partial`), so every bit is written.
        (false, true, 0, ClearValue::Stencil(0x44), ALL_STENCIL_BITS),
        (
            true,
            true,
            0x0f,
            ClearValue::DepthStencil {
                depth: 0.5,
                stencil: 0x44,
            },
            0x0f,
        ),
    ] {
        let mut rig = Rig::new();
        rig.depth_target(clc797::SET_ZT_FORMAT_V_ZF32_X24S8, WIDTH, HEIGHT, 1);
        rig.clear_depth(0.5);
        rig.clear_stencil(0x44, mask, 0x3c);
        rig.scissor(1, 5, 2, 4);
        assert_eq!(rig.clear(depth_stencil(depth, stencil)), ONLY_DEPTH_STENCIL);
        let clear = rig.clears()[0];
        assert_eq!(clear.value, value);
        assert_eq!(clear.stencil_mask, written);
        assert!(clear.rect.is_some());
    }
}

#[test]
fn the_render_area_is_over_the_unmapped_colour_targets() {
    // `SET_CT_SELECT`: one target, output 0 drawing to colour target 1.
    let map_to_one = |rig: &mut Rig| {
        let word = fields(&[
            (clc797::SET_CT_SELECT_TARGET_COUNT, 1),
            (clc797::SET_CT_SELECT_TARGET0, 1),
        ]);
        rig.set(&clc797::SET_CT_SELECT, &[], word);
    };
    let format = clc797::SET_COLOR_TARGET_FORMAT_V_A8B8G8R8;
    // Target 0 smaller than target 1: the clear of target 0 is whole, and the list submits.
    let mut rig = Rig::new();
    rig.color_target_sized(0, TARGET0, format, 1, WIDTH / 2, HEIGHT / 2);
    rig.color_target(1, TARGET1, format, 1);
    map_to_one(&mut rig);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    assert_eq!(rig.clears()[0].rect, None);
    let list = core::mem::take(&mut rig.list);
    assert!(rig.backend.submit(list).is_ok());
    // Target 0 larger than target 1: target 1 does not bound the area, so the clear is whole.
    let mut rig = Rig::new();
    rig.color_target(0, TARGET0, format, 1);
    rig.color_target_sized(1, TARGET1, format, 1, WIDTH / 2, HEIGHT / 2);
    map_to_one(&mut rig);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    assert_eq!(rig.clears()[0].rect, None);
    // Target 1 made part of the count bounds it.
    rig.ct_select(2);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    assert_eq!(
        rig.clears()[1].rect,
        Some(Rect {
            x: 0,
            y: 0,
            width: WIDTH / 2,
            height: HEIGHT / 2
        })
    );
}

#[test]
fn the_surface_clip_does_not_bound_the_clear() {
    let mut rig = Rig::new();
    rig.basic();
    rig.surface_clip(WIDTH / 2, HEIGHT / 2);
    assert_eq!(rig.clear(color(ColorWrites::ALL)), ONLY_COLOR);
    assert_eq!(rig.clears()[0].rect, None);
}

#[test]
fn a_signed_then_unsigned_masked_clear_each_carry_their_kinds_value() {
    let mut rig = Rig::new();
    rig.color_target(
        0,
        TARGET0,
        clc797::SET_COLOR_TARGET_FORMAT_V_AS8BS8GS8RS8,
        1,
    );
    rig.ct_select(1);
    rig.clear_color([1.0, 0.0, 0.0, 0.0]);
    assert_eq!(rig.clear(color(ColorWrites::RED)), ONLY_COLOR);
    rig.set(
        &clc797::SET_COLOR_TARGET_FORMAT,
        &[0],
        clc797::SET_COLOR_TARGET_FORMAT_V_AU8BU8GU8RU8,
    );
    assert_eq!(rig.clear(color(ColorWrites::RED)), ONLY_COLOR);
    let values: Vec<_> = rig.clears().iter().map(|c| c.value).collect();
    // Eden's formula on 8 bits: signed (8 - 1) * 2 * (v - 0.5), unsigned 8 * 2 * v.
    assert_eq!(
        values,
        [
            ClearValue::Sint([7, -7, -7, -7]),
            ClearValue::Uint([16, 0, 0, 0])
        ]
    );
}

#[test]
fn a_stencil_only_word_skipped_before_the_targets_reports_its_part() {
    let stencil = depth_stencil(false, true);
    // Conditional render off.
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_S8, WIDTH, HEIGHT, 1);
    rig.set(
        &clc797::SET_RENDER_ENABLE_C,
        &[],
        clc797::SET_RENDER_ENABLE_C_MODE_FALSE,
    );
    assert_eq!(
        rig.clear(stencil),
        Cleared {
            color: None,
            depth_stencil: Some(Part::Skipped(ClearSkip::RenderDisabled)),
        }
    );
    // On again, with an empty scissor.
    rig.set(
        &clc797::SET_RENDER_ENABLE_C,
        &[],
        clc797::SET_RENDER_ENABLE_C_MODE_TRUE,
    );
    rig.scissor(5, 3, 1, 4);
    assert_eq!(
        rig.clear(stencil),
        Cleared {
            color: None,
            depth_stencil: Some(Part::Skipped(ClearSkip::EmptyArea)),
        }
    );
    assert_eq!(rig.commands().len(), 0);
    assert_eq!(
        rig.draws.clear_counters(),
        ClearCounters {
            words: 2,
            skipped: 2,
            ..ClearCounters::default()
        }
    );
}

#[test]
fn the_stencil_func_mask_changes_nothing_in_a_clear() {
    let mut rig = Rig::new();
    rig.depth_target(clc797::SET_ZT_FORMAT_V_S8, WIDTH, HEIGHT, 1);
    for func_mask in [0x00, 0x3c, FULL_STENCIL_MASK] {
        rig.clear_stencil(0x44, 0x0f, func_mask);
        assert_eq!(rig.clear(depth_stencil(false, true)), ONLY_DEPTH_STENCIL);
    }
    let clears = rig.clears();
    assert_eq!(clears.len(), 3);
    assert!(clears.iter().all(|c| *c == clears[0]), "{clears:?}");
}

#[test]
fn every_integer_colour_target_format_has_a_component_width() {
    // `color_value`'s 0-bit fallback is unreachable: every integer format a colour target can
    // have is in Eden's table.
    let mut integers = 0;
    for value in 0..=clc797::SET_COLOR_TARGET_FORMAT_V.max() {
        if let Ok(Some(format)) = targets::color_format(value)
            && format.is_integer()
        {
            integers += 1;
            assert!(
                integer_component_bits(format).is_some(),
                "{value:#x} {format:?}"
            );
        }
    }
    assert!(integers > 0);
}

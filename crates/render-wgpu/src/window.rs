//! Window presentation (docs/specs/render-backend.md, "Present targets" and "Presenting a guest
//! framebuffer"; slice R8): the wgpu surface made from the frontend's window, and the present
//! blit the render thread draws into it. The choices are [`crate::present`]'s pure functions.
//!
//! # Where the surface is made
//!
//! `set_present_target(PresentTarget::Window(..))` makes the surface on the thread that calls
//! it ([`make_surface`]), and the render thread configures it, draws into it and presents it.
//! On macOS, making the surface attaches a Metal layer to the window's view, which `AppKit` allows
//! only on the main thread, and wgpu panics if asked anywhere else
//! (`wgpu-30.0.2/src/api/surface.rs`, `SurfaceTarget`, "Panics"). So on macOS the frontend
//! makes that call on its main thread, the thread that runs its event loop; on any other thread
//! the backend refuses it with [`NOT_MAIN_THREAD`] instead of letting wgpu panic (checked with
//! `objc2::MainThreadMarker`). Linux has no such rule. Configuring, acquiring and presenting
//! happen on the render thread: wgpu's Metal layer code takes the layer's size there under its
//! own lock (`wgpu-hal-30.0.2/src/metal/surface.rs`, `dimensions`: "safe to call off of the main
//! thread"). That this whole path works in a real window is checked by hand, not by a test: the
//! tests have no window and do not run on the main thread.
//!
//! Every fallible step of making the surface happens before the target is queued: the thread
//! check, wgpu's surface creation, and the choice of a format, a present mode and an alpha mode
//! the surface lists ([`crate::present`]). A refused call changes nothing, and the target in
//! force stays.
//!
//! # The present blit
//!
//! A present that is shown ([`crate::present::frame_outcome`]) is drawn into the window where
//! it stands in its list, in two encoded commands:
//!
//! 1. a copy of the presented rectangle (rounded out to whole blocks) from the guest's texture
//!    into a texture of the backend's own that can be sampled (the guest's may not be: a present
//!    needs only the copy-source usage, `validate::validate_present`), which also takes the
//!    view's layer or slice out of an array, cube or 3D texture;
//! 2. a render pass into the window's texture that clears it to black and draws `present.wgsl`
//!    (R4) as a strip of four vertices: the image scaled to fit the window, filtered, turned by
//!    the present's transform and converted from the guest's format to the window's
//!    ([`crate::present`] for each choice).
//!
//! Both are encoded before anything after the present in the list, and lists are submitted in
//! order on one queue, so a later command that writes the presented image cannot change what
//! the window shows ("Submit"; the ordering test in `backend::tests::window` pins it). The
//! window's texture is presented after the list's last submission.
//!
//! The backend's sampled copy, the parameter buffer, the shader, the pipelines (one per window
//! format and filterability) and the samplers are made on first use and reused; the copy is made
//! again only when the presented format or size changes. They are host memory the budget does
//! not count, like the rectangle-clear scratch (`crate::thread`).
//!
//! # A frame that cannot be shown
//!
//! A present the frame queue keeps is still dropped, counted in its feedback as
//! `PresentOutcome::Dropped`, when: the window's size has a zero side (minimised); wgpu answers
//! the acquire with a timeout, an occluded window, a validation error, or an outdated surface
//! twice in a row ([`crate::present::acquire_action`]); or the surface was lost, which lets it
//! go, so every later present to it is dropped until the frontend sets a window target again
//! (on macOS only the main thread may make one).

use crate::plan::TexelBox;
use crate::present::{
    self, AcquireAction, Acquired, Crop, ScreenRect, alpha_mode, blit_params, fit,
    present_mode, shown_size, surface_format, surface_size, view_formats, write_format,
};
use starquake_render::command::PresentDesc;
use starquake_render::host_shaders::present as shader;
use starquake_render::{HostShader, RenderError, WindowHandles, WindowSize};
use std::collections::HashMap;
use std::sync::Arc;

/// A window surface asked for off the main thread on macOS (module docs).
pub const NOT_MAIN_THREAD: &str =
    "a window's surface must be made on the main thread on macOS (set the present target there)";
/// wgpu could not make a surface from the window's handles.
pub const NO_SURFACE: &str = "a surface could not be made from the window's handles";
/// The adapter cannot present to the window, or the surface lists no format, present mode or
/// alpha mode the backend uses ([`crate::present`]).
pub const SURFACE_UNSUPPORTED: &str = "the GPU adapter cannot present to this window";

/// Whether this platform makes window surfaces on the main thread only (module docs).
const NEEDS_MAIN_THREAD: bool = cfg!(target_os = "macos");

/// Whether the calling thread is the process's main thread, where that matters.
#[cfg(target_os = "macos")]
fn on_main_thread() -> bool {
    objc2::MainThreadMarker::new().is_some()
}

/// Whether the calling thread is the process's main thread, where that matters: nowhere here.
#[cfg(not(target_os = "macos"))]
fn on_main_thread() -> bool {
    true
}

/// A window surface and how it is configured, made by [`make_surface`].
#[derive(Debug)]
pub struct WindowSurface {
    surface: wgpu::Surface<'static>,
    format: wgpu::TextureFormat,
    present_mode: wgpu::PresentMode,
    alpha_mode: wgpu::CompositeAlphaMode,
    size: WindowSize,
}

/// Makes a surface for `handles` on this thread and chooses how to configure it (module docs).
/// The check order: the thread ([`NOT_MAIN_THREAD`]), wgpu's surface creation
/// ([`NO_SURFACE`]), then the adapter's support and the surface's lists
/// ([`SURFACE_UNSUPPORTED`]).
pub fn make_surface(
    instance: &wgpu::Instance,
    adapter: &wgpu::Adapter,
    handles: &WindowHandles,
) -> Result<WindowSurface, RenderError> {
    if !present::may_make_surface(NEEDS_MAIN_THREAD, on_main_thread()) {
        return Err(RenderError::Unsupported(NOT_MAIN_THREAD));
    }
    let surface = instance
        .create_surface(Arc::clone(handles.window()))
        .map_err(|error| {
            tracing::warn!("wgpu could not make a surface for the window: {error}");
            RenderError::Unsupported(NO_SURFACE)
        })?;
    let unsupported = RenderError::Unsupported(SURFACE_UNSUPPORTED);
    if !adapter.is_surface_supported(&surface) {
        return Err(unsupported);
    }
    let caps = surface.get_capabilities(adapter);
    let (Some(format), Some(present_mode), Some(alpha_mode)) = (
        surface_format(&caps.formats),
        present_mode(&caps.present_modes),
        alpha_mode(&caps.alpha_modes),
    ) else {
        return Err(unsupported);
    };
    if !caps.usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        return Err(unsupported);
    }
    tracing::info!("presenting to a window as {format:?}, {present_mode:?}, {alpha_mode:?}");
    Ok(WindowSurface {
        surface,
        format,
        present_mode,
        alpha_mode,
        size: handles.size().clone(),
    })
}

/// Where the render thread sends shown frames.
#[derive(Debug)]
pub enum Target {
    /// No window: presents are counted (and dumped on request), never drawn.
    Offscreen,
    /// A window.
    Window(WindowSurface),
    /// A window whose surface was lost: presents are dropped (module docs).
    Lost,
    /// A texture of the backend's standing in for a window's, for the tests: drawn into as a
    /// window's texture is, never presented. Its id is a live texture with the render-target
    /// usage.
    #[cfg(test)]
    Texture(starquake_render::TextureId),
}

/// The backend's sampled copy of the presented image.
#[derive(Debug)]
struct Source {
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    /// Bind groups for it, by whether the sampler filters ([`Blit::layouts`]'s order).
    groups: [Option<wgpu::BindGroup>; 2],
}

/// The objects every blit uses, made on the first one.
#[derive(Debug)]
struct Blit {
    module: wgpu::ShaderModule,
    params: wgpu::Buffer,
    /// Bind group and pipeline layouts for a format that cannot be filtered (0) and one that
    /// can (1).
    layouts: [(wgpu::BindGroupLayout, wgpu::PipelineLayout); 2],
    /// The nearest (0) and linear (1) samplers.
    samplers: [wgpu::Sampler; 2],
    /// Pipelines by the format written and filterability.
    pipelines: HashMap<(wgpu::TextureFormat, bool), wgpu::RenderPipeline>,
}

/// The render thread's present state (module docs).
#[derive(Debug)]
pub struct Presenter {
    target: Target,
    /// The size the window's surface is configured at.
    configured: Option<(u32, u32)>,
    /// Whether the surface must be configured again before the next frame.
    stale: bool,
    /// The window's texture drawn into by the list being encoded, presented after its last
    /// submission.
    pending: Option<wgpu::SurfaceTexture>,
    blit: Option<Blit>,
    source: Option<Source>,
}

/// Why a shown frame was dropped after all, for the log.
type Dropped = &'static str;

impl Presenter {
    /// A presenter for the offscreen target.
    pub fn new() -> Self {
        Self {
            target: Target::Offscreen,
            configured: None,
            stale: false,
            pending: None,
            blit: None,
            source: None,
        }
    }

    /// Whether presents go nowhere but the count and dumps.
    pub fn offscreen(&self) -> bool {
        matches!(self.target, Target::Offscreen)
    }

    /// The target, for the render thread to resolve a test texture.
    pub fn target(&self) -> &Target {
        &self.target
    }

    /// Sets where shown frames go. The old window's surface is let go.
    pub fn set_target(&mut self, target: Target) {
        self.target = target;
        self.configured = None;
        self.stale = false;
    }

    /// The window's texture for the next frame (module docs), configuring the surface first
    /// when its size changed or wgpu asked for it.
    pub fn acquire(&mut self, device: &wgpu::Device) -> Result<wgpu::Texture, Dropped> {
        let Target::Window(window) = &self.target else {
            return Err("there is no window");
        };
        let max = device.limits().max_texture_dimension_2d;
        let Some(size) = surface_size(window.size.get(), max) else {
            return Err("the window has no area");
        };
        if self.configured != Some(size) || self.stale {
            configure(device, window, size);
            self.configured = Some(size);
            self.stale = false;
        }
        for retried in [false, true] {
            let (acquired, texture) = match window.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) => (Acquired::Success, Some(t)),
                wgpu::CurrentSurfaceTexture::Suboptimal(t) => (Acquired::Suboptimal, Some(t)),
                wgpu::CurrentSurfaceTexture::Timeout => (Acquired::Timeout, None),
                wgpu::CurrentSurfaceTexture::Occluded => (Acquired::Occluded, None),
                wgpu::CurrentSurfaceTexture::Outdated => (Acquired::Outdated, None),
                wgpu::CurrentSurfaceTexture::Lost => (Acquired::Lost, None),
                wgpu::CurrentSurfaceTexture::Validation => (Acquired::Validation, None),
            };
            match (present::acquire_action(acquired, retried), texture) {
                (AcquireAction::Show { reconfigure }, Some(texture)) => {
                    self.stale = reconfigure;
                    let out = texture.texture.clone();
                    self.pending = Some(texture);
                    return Ok(out);
                }
                (AcquireAction::Retry, _) => configure(device, window, size),
                (AcquireAction::Forget, _) => {
                    tracing::warn!(
                        "the window's surface was lost; frames are dropped until a window \
                         target is set again"
                    );
                    self.set_target(Target::Lost);
                    return Err("the window's surface was lost");
                }
                (AcquireAction::Show { .. } | AcquireAction::Drop, _) => {
                    return Err("the window gave no texture");
                }
            }
        }
        Err("the window's surface stayed out of date")
    }

    /// Presents the window's texture the list drew into, if any; called after the list's last
    /// submission.
    pub fn finish(&mut self, queue: &wgpu::Queue) {
        if let Some(texture) = self.pending.take() {
            queue.present(texture);
        }
    }

    /// Encodes the blit of `at` in `src` (a present of `desc`, read as `format`) into `dst`
    /// (module docs, "The present blit").
    #[expect(
        clippy::too_many_arguments,
        reason = "the blit needs the device, the queue, the encoder, both textures and what the \
                  present names; a struct for them would only carry them here"
    )]
    pub fn blit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        at: TexelBox,
        desc: &PresentDesc,
        format: wgpu::TextureFormat,
        dst: &wgpu::Texture,
    ) -> Result<(), Dropped> {
        let shown = shown_size(desc.src_rect, desc.transform);
        let target = (dst.width(), dst.height());
        let screen: ScreenRect = fit(shown, target).ok_or("the image or the window is empty")?;
        let filterable = matches!(
            format.sample_type(None, Some(device.features())),
            Some(wgpu::TextureSampleType::Float { filterable: true })
        );
        let linear = present::filter(shown, screen, filterable) == wgpu::FilterMode::Linear;
        let written = write_format(dst.format(), format.is_srgb());
        let blit = match &mut self.blit {
            Some(blit) => blit,
            slot @ None => slot.insert(make_blit(device)?),
        };
        let source = source(&mut self.source, device, format, at);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: at.mip,
                origin: wgpu::Origin3d {
                    x: at.origin.x,
                    y: at.origin.y,
                    z: at.origin.z,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &source.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: at.extent.width,
                height: at.extent.height,
                depth_or_array_layers: 1,
            },
        );
        let params = blit_params(target, screen, Crop::new(desc.src_rect, at), desc.transform);
        queue.write_buffer(&blit.params, 0, &params);
        let kind = usize::from(filterable);
        let pipeline = blit
            .pipelines
            .entry((written, filterable))
            .or_insert_with(|| make_pipeline(device, &blit.module, &blit.layouts[kind].1, written));
        let group = source.groups[usize::from(linear)].get_or_insert_with(|| {
            let view = source
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("starquake present"),
                layout: &blit.layouts[kind].0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: shader::PARAMS_BINDING,
                        resource: blit.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: shader::TEXTURE_BINDING,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: shader::SAMPLER_BINDING,
                        resource: wgpu::BindingResource::Sampler(
                            &blit.samplers[usize::from(linear)],
                        ),
                    },
                ],
            })
        });
        let view = dst.create_view(&wgpu::TextureViewDescriptor {
            format: Some(written),
            usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: 0,
            array_layer_count: Some(1),
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..wgpu::TextureViewDescriptor::default()
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("starquake present"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(shader::GROUP, &*group, &[]);
        pass.draw(0..shader::VERTEX_COUNT, 0..1);
        drop(pass);
        Ok(())
    }
}

/// Configures `window`'s surface at `size`.
fn configure(device: &wgpu::Device, window: &WindowSurface, size: (u32, u32)) {
    window.surface.configure(
        device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: window.format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.0,
            height: size.1,
            present_mode: window.present_mode,
            desired_maximum_frame_latency: FRAME_LATENCY,
            alpha_mode: window.alpha_mode,
            view_formats: view_formats(window.format),
        },
    );
}

/// The surface's frame latency hint: wgpu's default (`Surface::get_default_config`). A guess,
/// to measure: wgpu notes that under mailbox on Metal and Vulkan, 2 limits the frame rate to
/// twice the display's (`wgpu-types-30.0.2/src/surface.rs`, `desired_maximum_frame_latency`),
/// which is more than the guest's 60 Hz on any display.
pub const FRAME_LATENCY: u32 = 2;

/// The backend's sampled copy for a present of `at` read as `format`, made again when the format
/// or size changed.
fn source<'a>(
    slot: &'a mut Option<Source>,
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    at: TexelBox,
) -> &'a mut Source {
    let (width, height) = (at.extent.width, at.extent.height);
    if let Some(s) = slot
        && (s.format, s.width, s.height) != (format, width, height)
    {
        *slot = None;
    }
    slot.get_or_insert_with(|| Source {
        format,
        width,
        height,
        texture: device.create_texture(&wgpu::TextureDescriptor {
            label: Some("starquake present source"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }),
        groups: [None, None],
    })
}

/// The shader, buffer, layouts and samplers every blit uses. Fails only when R4's shader does
/// not parse, which its own tests rule out.
fn make_blit(device: &wgpu::Device) -> Result<Blit, Dropped> {
    let module = naga::front::wgsl::parse_str(HostShader::Present.source()).map_err(|e| {
        tracing::error!("the present shader did not parse: {e}");
        "the present shader did not parse"
    })?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("starquake present"),
        source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("starquake present parameters"),
        size: u64::from(shader::PARAMS_SIZE),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = |filterable: bool| {
        let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("starquake present"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: shader::PARAMS_BINDING,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(u64::from(shader::PARAMS_SIZE)),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: shader::TEXTURE_BINDING,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: shader::SAMPLER_BINDING,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(if filterable {
                        wgpu::SamplerBindingType::Filtering
                    } else {
                        wgpu::SamplerBindingType::NonFiltering
                    }),
                    count: None,
                },
            ],
        });
        let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("starquake present"),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
        (group, pipeline)
    };
    let sampler = |filter: wgpu::FilterMode| {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("starquake present"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..wgpu::SamplerDescriptor::default()
        })
    };
    Ok(Blit {
        module,
        params,
        layouts: [layout(false), layout(true)],
        samplers: [
            sampler(wgpu::FilterMode::Nearest),
            sampler(wgpu::FilterMode::Linear),
        ],
        pipelines: HashMap::new(),
    })
}

/// The blit's pipeline writing `format`.
fn make_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("starquake present"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(shader::VERTEX),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(shader::FRAGMENT),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

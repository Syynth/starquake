//! A wgpu device for tests that need a real GPU (experiment E1 and the wgpu acceptance tests).
//!
//! These tests skip themselves, with a notice on standard error, where the host has no GPU
//! adapter (the spec's "wgpu acceptance" row: locally on a Mac, in Linux CI only once the render
//! backend's lavapipe experiment passes). Every wgpu error is caught with error scopes and
//! returned as text, so a rejected module fails the assertion that expected it instead of
//! reaching wgpu's default handler, which panics.
//!
//! Shader modules are created with `ShaderSource::Naga`, the path the owner's decision of
//! 2026-11-12 gives the wgpu backend. On Metal, wgpu writes and compiles the MSL only when a
//! pipeline is created (`wgpu-hal`'s Metal device keeps the naga module until then), so a test
//! that wants to know whether the host compiler accepts a module creates a pipeline too.

use std::borrow::Cow;
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

/// How long a test waits for a wgpu future before failing (native wgpu resolves its futures
/// at once; the bound only keeps a broken driver from hanging the gate).
const FUTURE_TIMEOUT: Duration = Duration::from_secs(30);

/// Runs `future` to completion on this thread, failing the test after [`FUTURE_TIMEOUT`].
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    let start = Instant::now();
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        assert!(
            start.elapsed() < FUTURE_TIMEOUT,
            "a wgpu future did not resolve within {FUTURE_TIMEOUT:?}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A device with every feature in the request that its adapter offers.
pub(crate) struct TestDevice {
    /// The device.
    pub(crate) device: wgpu::Device,
    /// What the adapter is, for messages.
    pub(crate) adapter: wgpu::AdapterInfo,
}

impl TestDevice {
    /// Opens the default adapter's device with the features of `wanted` it supports, or prints a
    /// notice naming `test` and returns `None` when the host has no adapter.
    pub(crate) fn open(test: &str, wanted: wgpu::Features) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) {
                Ok(adapter) => adapter,
                Err(error) => {
                    eprintln!("skipping {test}: no GPU adapter on this host ({error})");
                    return None;
                }
            };
        let features = adapter.features() & wanted;
        let (device, _queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(test),
            required_features: features,
            required_limits: adapter.limits(),
            ..wgpu::DeviceDescriptor::default()
        }))
        .expect("the adapter's own features and limits must be accepted");
        Some(Self {
            device,
            adapter: adapter.get_info(),
        })
    }

    /// Whether the device was opened with every feature in `features`.
    pub(crate) fn has(&self, features: wgpu::Features) -> bool {
        self.device.features().contains(features)
    }

    /// Runs `work` inside validation and internal error scopes and returns the first error as
    /// text.
    fn scoped<T>(&self, work: impl FnOnce(&wgpu::Device) -> T) -> Result<T, String> {
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let value = work(&self.device);
        let validation_error = block_on(validation.pop());
        let internal_error = block_on(internal.pop());
        match validation_error.or(internal_error) {
            Some(error) => Err(format!("{error}")),
            None => Ok(value),
        }
    }

    /// Creates a shader module from a naga module.
    pub(crate) fn module(&self, module: naga::Module) -> Result<wgpu::ShaderModule, String> {
        self.scoped(|device| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
            })
        })
    }

    /// Creates a compute pipeline with a layout derived from the shader.
    pub(crate) fn compute_pipeline(
        &self,
        module: &wgpu::ShaderModule,
        entry: &str,
    ) -> Result<(), String> {
        self.scoped(|device| {
            let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        })
    }

    /// Creates a pipeline layout with one bind group of `entries`.
    pub(crate) fn layout(
        &self,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<wgpu::PipelineLayout, String> {
        self.scoped(|device| {
            let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries,
            });
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&group)],
                immediate_size: 0,
            })
        })
    }

    /// Creates a render pipeline with `layout` (or one derived from the shaders), `buffers` as
    /// its vertex input and `targets` as its colour targets.
    pub(crate) fn render_pipeline(
        &self,
        layout: Option<&wgpu::PipelineLayout>,
        vertex: (&wgpu::ShaderModule, &str),
        fragment: (&wgpu::ShaderModule, &str),
        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        targets: &[Option<wgpu::ColorTargetState>],
    ) -> Result<(), String> {
        self.scoped(|device| {
            let _pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout,
                vertex: wgpu::VertexState {
                    module: vertex.0,
                    entry_point: Some(vertex.1),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: fragment.0,
                    entry_point: Some(fragment.1),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets,
                }),
                multiview_mask: None,
                cache: None,
            });
        })
    }
}

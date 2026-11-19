//! Shaders and pipelines. The first group runs with no GPU: the draw decision, the subgroup
//! rule, the build job's settling, and the wait and skip paths on pipelines that never reach a
//! device. The rest run on a real adapter, skipped with a notice in CI and where the host has
//! none (crate docs), each under a time bound. Their shaders are hand-written WGSL modules,
//! parsed and validated into naga IR by the tests (`crate::fixtures::wgsl`); the library takes
//! the IR only.

use super::*;
use crate::backend::tests::{bounded, open_with};
use crate::fixtures::{graphics_desc, shader_desc, wgsl};
use crate::{WgpuBackend, WgpuConfig};
use starquake_render::pipeline::{CullMode, Topology};
use starquake_render::resource::ViewDimension;
use starquake_render::shader::{ResourceBinding, ResourceKind, SampleKind, ShaderRequirements};
use starquake_render::{
    Backend, CompletionWake, ComputePipelineDesc, FenceValue, PipelineId, PixelFormat, ResourceId,
    ShaderId,
};
use std::num::NonZeroU32;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// A change that puts one fault into a description.
type Fault<'a> = &'a dyn Fn(&mut starquake_render::GraphicsPipelineDesc);

/// How long a test waits for a build or a blocked draw.
const SETTLE: Duration = Duration::from_secs(20);

/// How long a test lets a blocked draw sit to see that it stays blocked.
const STILL: Duration = Duration::from_millis(100);

// ---------------------------------------------------------------------------------------------
// No GPU.

#[test]
fn the_draw_decision_for_every_status_and_setting() {
    use PipelineStatus as P;
    use WhileCompiling as W;
    let cases = [
        (P::Ready, W::Wait, DrawAction::Draw),
        (P::Ready, W::Skip, DrawAction::Draw),
        (P::Compiling, W::Wait, DrawAction::Wait),
        (
            P::Compiling,
            W::Skip,
            DrawAction::Skip(SkipReason::Compiling),
        ),
        (P::Failed, W::Wait, DrawAction::Skip(SkipReason::Failed)),
        (P::Failed, W::Skip, DrawAction::Skip(SkipReason::Failed)),
    ];
    for (status, setting, action) in cases {
        assert_eq!(on_draw(status, setting), action, "{status:?} {setting:?}");
    }
}

/// An adapter as wgpu describes it, with the fields the subgroup rule reads set and the rest as
/// wgpu-types 30.0.2's `AdapterInfo::new` leaves them (the Metal backend's starting point).
fn adapter(backend: wgpu::Backend, name: &str, min: u32, max: u32) -> wgpu::AdapterInfo {
    wgpu::AdapterInfo {
        name: name.to_owned(),
        subgroup_min_size: min,
        subgroup_max_size: max,
        ..wgpu::AdapterInfo::new(wgpu::DeviceType::IntegratedGpu, backend)
    }
}

/// The reference values, written out: the guest's warp is 32 lanes (shader-pipeline, "Compute
/// shaders"); the Metal device name of this Mac's GPU, as measured through wgpu, is "Apple M2".
#[test]
fn the_subgroup_constants_match_their_sources() {
    assert_eq!(GUEST_SUBGROUP_SIZE, 32);
    assert_eq!(APPLE_GPU_NAME_PREFIX, "Apple ");
    assert!("Apple M2".starts_with(APPLE_GPU_NAME_PREFIX));
}

#[test]
fn only_a_metal_adapter_named_apple_is_the_apple_gpu_family() {
    use wgpu::Backend::{Gl, Metal, Vulkan};
    for (backend, name, apple) in [
        (Metal, "Apple M2", true), // measured on this Mac
        (Metal, "Apple M1 Max", true),
        (Metal, "Apple A14 GPU", true),
        (Metal, "Apple ", true), // the prefix alone
        (Metal, "Apple", false), // no space
        (Metal, "AppleM2", false),
        (Metal, "apple M2", false), // the case matters
        (Metal, " Apple M2", false),
        (Metal, "AMD Radeon Pro 5500M", false),
        (Metal, "Intel(R) UHD Graphics 630", false),
        (Metal, "", false),
        (Vulkan, "Apple M2", false), // MoltenVK or a Linux driver: the range check applies
        (Gl, "Apple M2", false),
    ] {
        assert_eq!(
            apple_gpu_family(&adapter(backend, name, 4, 64)),
            apple,
            "{backend:?} {name:?}"
        );
    }
}

#[test]
fn the_apple_family_meets_subgroup_size_32_and_others_keep_the_range_check() {
    use SubgroupSupport::{AppleFamily, Fixed32, NoSubgroups, Range};
    use wgpu::Backend::{Metal, Vulkan};
    let s = wgpu::Features::SUBGROUP;
    let none = wgpu::Features::empty();
    let g = GUEST_SUBGROUP_SIZE;
    for (backend, name, features, min, max, support) in [
        // wgpu's Metal report on an Apple GPU.
        (Metal, "Apple M2", s, 4, 64, AppleFamily),
        // Without the feature nothing meets it, the Apple family included.
        (Metal, "Apple M2", none, 4, 64, NoSubgroups),
        (Vulkan, "GPU", none, g, g, NoSubgroups),
        // The family is decided first: a range of 32 to 32 on it reads as the family.
        (Metal, "Apple M2", s, g, g, AppleFamily),
        // A 4 to 64 adapter outside the family does not meet it.
        (
            Metal,
            "AMD Radeon Pro 5500M",
            s,
            4,
            64,
            Range { min: 4, max: 64 },
        ),
        (Vulkan, "Apple M2", s, 4, 64, Range { min: 4, max: 64 }),
        // A fixed-32 adapter does.
        (Vulkan, "GPU", s, g, g, Fixed32),
        (Metal, "Intel(R) UHD Graphics 630", s, g, g, Fixed32),
        // The range check's edges, one off on each side of each end.
        (Vulkan, "GPU", s, g - 1, g, Range { min: g - 1, max: g }),
        (
            Vulkan,
            "GPU",
            s,
            g + 1,
            g + 1,
            Range {
                min: g + 1,
                max: g + 1,
            },
        ),
        (Vulkan, "GPU", s, g, g + 1, Range { min: g, max: g + 1 }),
        (Vulkan, "GPU", s, g, g - 1, Range { min: g, max: g - 1 }),
        (Vulkan, "GPU", s, 16, g, Range { min: 16, max: g }),
        (Vulkan, "GPU", s, g, 64, Range { min: g, max: 64 }),
    ] {
        let info = adapter(backend, name, min, max);
        let got = SubgroupSupport::of(&info, features);
        assert_eq!(
            got, support,
            "{backend:?} {name:?} {features:?} {min}..={max}"
        );
        assert_eq!(
            got.honoured(),
            matches!(support, AppleFamily | Fixed32),
            "{support:?}"
        );
    }
}

/// The backend's verdict comes from the device's features as given: without `SUBGROUP` nothing
/// meets "subgroup size 32", an Apple GPU included, and subgroups in vertex shaders follow
/// `SUBGROUP_VERTEX` alone (module docs, "Requirements").
#[test]
fn the_backend_verdict_reads_the_device_features() {
    use SubgroupSupport::{AppleFamily, Fixed32, NoSubgroups, Range};
    use wgpu::Backend::{Metal, Vulkan};
    use wgpu::Features as F;
    let g = GUEST_SUBGROUP_SIZE;
    let apple = adapter(Metal, "Apple M2", 4, 64);
    let fixed = adapter(Vulkan, "GPU", g, g);
    let wide = adapter(Vulkan, "GPU", 4, 64);
    let both = F::SUBGROUP | F::SUBGROUP_VERTEX;
    for (info, features, support, vertex) in [
        (&apple, F::empty(), NoSubgroups, false),
        (&apple, F::SUBGROUP_BARRIER, NoSubgroups, false),
        (&apple, F::SUBGROUP_VERTEX, NoSubgroups, true),
        // wgpu-hal 30.0.2's Metal backend on an Apple GPU (`src/metal/adapter.rs`, `features`).
        (
            &apple,
            F::SUBGROUP | F::SUBGROUP_BARRIER,
            AppleFamily,
            false,
        ),
        (&apple, both, AppleFamily, true),
        (&fixed, F::empty(), NoSubgroups, false),
        (&fixed, F::SUBGROUP, Fixed32, false),
        (&fixed, both, Fixed32, true),
        (&wide, F::SUBGROUP, Range { min: 4, max: 64 }, false),
        (&wide, both, Range { min: 4, max: 64 }, true),
    ] {
        assert_eq!(
            SubgroupVerdict::of(info, features),
            SubgroupVerdict { support, vertex },
            "{:?} {features:?}",
            info.name
        );
    }
}

#[test]
fn the_subgroup_verdict_reads_as_plain_words() {
    for (support, text) in [
        (
            SubgroupSupport::NoSubgroups,
            "no (the device has no subgroups)",
        ),
        (
            SubgroupSupport::AppleFamily,
            "yes (Apple GPU family, 32-wide SIMD groups)",
        ),
        (
            SubgroupSupport::Fixed32,
            "yes (the reported range is 32 to 32)",
        ),
        (
            SubgroupSupport::Range { min: 4, max: 64 },
            "no (the reported range is 4 to 64)",
        ),
    ] {
        assert_eq!(support.to_string(), text);
    }
}

#[test]
fn a_build_that_fails_panics_or_is_dropped_fails_its_pipeline() {
    let failed = Arc::new(PipelineCell::new());
    job(Arc::clone(&failed), || Err("no".into()))();
    assert_eq!(failed.status(), PipelineStatus::Failed);

    let panicked = Arc::new(PipelineCell::new());
    #[expect(clippy::panic, reason = "the test's build panics on purpose")]
    let build = || -> Result<HostPipeline, String> { panic!("a build that panics") };
    job(Arc::clone(&panicked), build)();
    assert_eq!(panicked.status(), PipelineStatus::Failed);

    let dropped = Arc::new(PipelineCell::new());
    let unrun = job(Arc::clone(&dropped), || Err("never run".into()));
    assert_eq!(
        dropped.status(),
        PipelineStatus::Compiling,
        "queued, not run"
    );
    drop(unrun);
    assert_eq!(dropped.status(), PipelineStatus::Failed);
}

#[test]
fn the_first_outcome_stands() {
    let cell = PipelineCell::new();
    cell.settle(Err("first".into()));
    assert_eq!(cell.status(), PipelineStatus::Failed);
    // A second settle (which no path makes) changes nothing.
    cell.settle(Err("second".into()));
    assert_eq!(cell.status(), PipelineStatus::Failed);
}

#[test]
fn a_scope_that_does_not_report_fails_the_build() {
    assert_eq!(scope_outcome(Ok(None)), Ok(()));
    assert_eq!(
        scope_outcome(Err(NewError::Timeout)),
        Err(SCOPE_TIMEOUT.to_owned())
    );
    let caught = wgpu::Error::OutOfMemory {
        source: Box::new(std::io::Error::other("caught")),
    };
    let message = caught.to_string();
    assert_eq!(scope_outcome(Ok(Some(caught))), Err(message));
}

#[test]
fn a_failed_pipeline_skips_every_draw_and_counts_once() {
    for setting in [WhileCompiling::Wait, WhileCompiling::Skip] {
        let counters = Counters::default();
        let a = PipelineCell::new();
        let b = PipelineCell::new();
        a.settle(Err("a".into()));
        b.settle(Err("b".into()));
        for _ in 0..3 {
            assert!(matches!(
                acquire(&a, setting, &counters),
                DrawPipeline::Skipped(SkipReason::Failed)
            ));
        }
        assert_eq!(
            counters.get().failed_pipelines,
            1,
            "{setting:?}: once for a"
        );
        assert!(matches!(
            acquire(&b, setting, &counters),
            DrawPipeline::Skipped(SkipReason::Failed)
        ));
        assert_eq!(
            counters.get(),
            PipelineCounts {
                skipped_while_compiling: 0,
                failed_pipelines: 2
            },
            "{setting:?}"
        );
    }
}

#[test]
fn under_skip_each_draw_of_a_compiling_pipeline_is_skipped_and_counted() {
    let counters = Counters::default();
    let cell = PipelineCell::new();
    for n in 1..=3 {
        assert!(matches!(
            acquire(&cell, WhileCompiling::Skip, &counters),
            DrawPipeline::Skipped(SkipReason::Compiling)
        ));
        assert_eq!(counters.get().skipped_while_compiling, n);
    }
    assert_eq!(counters.get().failed_pipelines, 0);
    assert_eq!(
        cell.status(),
        PipelineStatus::Compiling,
        "skipping does not wait"
    );
}

/// Under `Wait`, a draw blocks while the build runs and is released by the build's outcome;
/// a dropped job releases it too.
#[test]
fn under_wait_a_draw_blocks_until_the_build_settles() {
    bounded(|| {
        for drop_it in [false, true] {
            let cell = Arc::new(PipelineCell::new());
            let unrun = job(Arc::clone(&cell), || Err("failed".into()));
            let (tx, rx) = mpsc::channel();
            let waiter = Arc::clone(&cell);
            let draw = std::thread::spawn(move || {
                let counters = Counters::default();
                let got = acquire(&waiter, WhileCompiling::Wait, &counters);
                tx.send((
                    matches!(got, DrawPipeline::Skipped(SkipReason::Failed)),
                    counters.get(),
                ))
                .expect("the test is listening");
            });
            assert_eq!(
                rx.recv_timeout(STILL),
                Err(mpsc::RecvTimeoutError::Timeout),
                "the draw waits while the pipeline compiles"
            );
            if drop_it {
                drop(unrun);
            } else {
                unrun();
            }
            let (failed, counts) = rx.recv_timeout(SETTLE).expect("released");
            assert!(failed, "dropped {drop_it}");
            assert_eq!(counts.skipped_while_compiling, 0, "a wait is not a skip");
            assert_eq!(counts.failed_pipelines, 1);
            draw.join().expect("the draw thread ends");
        }
    });
}

#[test]
fn shaders_must_be_naga_modules_with_their_entry_point() {
    let module = || wgsl("@vertex fn main() -> @builtin(position) vec4<f32> { return vec4(0.0); }");
    let desc =
        |stage, entry: &str, source| shader_desc(stage, entry, source, ShaderInterface::default());
    assert!(ShaderRec::new(desc(ShaderStage::Vertex, "main", module())).is_ok());
    let refused = [
        (
            desc(
                ShaderStage::Vertex,
                "main",
                ShaderSource::SpirV(vec![0x0723_0203]),
            ),
            RenderError::Unsupported(NAGA_ONLY),
        ),
        (
            desc(
                ShaderStage::Vertex,
                "main",
                ShaderSource::Msl(String::new()),
            ),
            RenderError::Unsupported(NAGA_ONLY),
        ),
        (
            desc(ShaderStage::Vertex, "other", module()),
            RenderError::Invalid(NO_ENTRY_POINT),
        ),
        (
            desc(ShaderStage::Fragment, "main", module()),
            RenderError::Invalid(NO_ENTRY_POINT),
        ),
        (
            desc(ShaderStage::Geometry, "main", module()),
            RenderError::Invalid(NO_ENTRY_POINT),
        ),
    ];
    for (desc, error) in refused {
        let what = format!("{:?} {}", desc.stage, desc.entry);
        assert_eq!(ShaderRec::new(desc).map(|_| ()), Err(error), "{what}");
    }
}

/// A compute module named `main` whose body holds empty blocks nested `depth` deep (its
/// deepest block at level `depth`), built by hand since the WGSL front end limits its own
/// nesting. It is validated when `validate` is set; otherwise it carries a trivial module's
/// info, which the backend does not read (naga's validator recurses once per level, so a very
/// deep module cannot be validated on a test thread).
fn nested_compute(depth: usize, validate: bool) -> ShaderSource {
    let parse = || {
        naga::front::wgsl::parse_str("@compute @workgroup_size(1) fn main() { }")
            .expect("the test module parses")
    };
    let validator = || {
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
    };
    let mut module = parse();
    let mut inner = naga::Block::new();
    for _ in 0..depth {
        let mut outer = naga::Block::new();
        outer.push(naga::Statement::Block(inner), naga::Span::UNDEFINED);
        inner = outer;
    }
    module.entry_points[0].function.body = inner;
    let info = if validate {
        validator().validate(&module)
    } else {
        validator().validate(&parse())
    }
    .expect("the test module validates");
    ShaderSource::Naga {
        module: Box::new(module),
        info: Box::new(info),
    }
}

fn compute_desc(source: ShaderSource, entry: &str) -> ShaderDesc {
    shader_desc(
        ShaderStage::Compute,
        entry,
        source,
        ShaderInterface::default(),
    )
}

/// Nesting at the bound is kept and one level past it refused with [`NESTING`]; a module far
/// past it (deep enough that naga's recursive `Drop` would overflow the test thread) is refused
/// and freed without overflowing (`scan::dismantle`).
#[test]
fn shaders_nested_past_the_bound_are_refused() {
    bounded(|| {
        assert!(
            ShaderRec::new(compute_desc(
                nested_compute(MAX_SHADER_NESTING, true),
                "main"
            ))
            .is_ok()
        );
        for depth in [MAX_SHADER_NESTING + 1, 100_000] {
            assert_eq!(
                ShaderRec::new(compute_desc(nested_compute(depth, depth < 1000), "main"))
                    .map(|_| ()),
                Err(RenderError::Unsupported(NESTING)),
                "{depth}"
            );
        }
    });
}

/// The documented order (module docs, "Shaders"): with the entry point missing and the module
/// too deep, the entry point is reported; a module that is not naga is reported before either.
#[test]
fn shader_checks_run_in_order() {
    bounded(|| {
        assert_eq!(
            ShaderRec::new(compute_desc(nested_compute(100_000, false), "other")).map(|_| ()),
            Err(RenderError::Invalid(NO_ENTRY_POINT))
        );
        assert_eq!(
            ShaderRec::new(compute_desc(
                ShaderSource::SpirV(vec![0x0723_0203]),
                "other"
            ))
            .map(|_| ()),
            Err(RenderError::Unsupported(NAGA_ONLY))
        );
    });
}

const BALLOT: &str = "
    @group(0) @binding(0) var<storage, read_write> out: array<u32>;
    @compute @workgroup_size(32)
    fn main(@builtin(local_invocation_index) l: u32) { out[l] = subgroupBallot(true).x; }";

fn storage_out() -> ShaderInterface {
    ShaderInterface {
        resources: vec![ResourceBinding {
            group: 0,
            binding: 0,
            kind: ResourceKind::StorageBuffer { writable: true },
            count: None,
        }],
        ..ShaderInterface::default()
    }
}

/// `interface` with `requirements.subgroup` set, as the GPU crate's conversion fills it for a
/// shader that uses subgroups.
fn subgroups(interface: ShaderInterface) -> ShaderInterface {
    ShaderInterface {
        requirements: ShaderRequirements { subgroup: true },
        ..interface
    }
}

/// A vertex shader that calls a subgroup operation.
const VERTEX_BALLOT: &str = "
    @vertex fn main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
        return vec4(f32(subgroupBallot(i == 0u).x), 0.0, 0.0, 1.0);
    }";

/// A verdict that meets "subgroup size 32" (or not), with subgroups in vertex shaders (or not).
fn verdict(honoured: bool, vertex: bool) -> SubgroupVerdict {
    let support = if honoured {
        SubgroupSupport::AppleFamily
    } else {
        SubgroupSupport::Range { min: 4, max: 64 }
    };
    SubgroupVerdict { support, vertex }
}

#[test]
fn a_subgroup_shader_needs_a_device_that_honours_them() {
    let rec = |source, interface| {
        ShaderRec::new(shader_desc(
            ShaderStage::Compute,
            "main",
            wgsl(source),
            interface,
        ))
        .expect("accepted")
    };
    let ballot = rec(BALLOT, subgroups(storage_out()));
    let plain = rec(
        "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
        @compute @workgroup_size(1) fn main() { out[0] = 1u; }",
        storage_out(),
    );
    for vertex in [false, true] {
        assert_eq!(check_requirements([&ballot], verdict(true, vertex)), Ok(()));
        assert_eq!(
            check_requirements([&ballot], verdict(false, vertex)),
            Err(RenderError::Unsupported(SUBGROUP_SIZE))
        );
        assert_eq!(check_requirements([&plain], verdict(false, vertex)), Ok(()));
        assert_eq!(
            check_requirements([&plain, &ballot], verdict(false, vertex)),
            Err(RenderError::Unsupported(SUBGROUP_SIZE)),
            "any shader of the pipeline"
        );
    }
    // Every reason the size is not met refuses alike.
    for support in [
        SubgroupSupport::NoSubgroups,
        SubgroupSupport::Range { min: 4, max: 64 },
    ] {
        assert_eq!(
            check_requirements(
                [&ballot],
                SubgroupVerdict {
                    support,
                    vertex: true
                }
            ),
            Err(RenderError::Unsupported(SUBGROUP_SIZE)),
            "{support:?}"
        );
    }
}

/// A vertex shader that uses subgroups needs `SUBGROUP_VERTEX` as well, checked after the size;
/// a fragment or compute shader that uses them, beside a vertex shader that does not, needs only
/// the size (module docs, "Requirements").
#[test]
fn vertex_subgroups_need_the_vertex_feature() {
    let rec = |stage, source, interface| {
        ShaderRec::new(shader_desc(stage, "main", wgsl(source), interface)).expect("accepted")
    };
    let vertex_ballot = rec(
        ShaderStage::Vertex,
        VERTEX_BALLOT,
        subgroups(ShaderInterface::default()),
    );
    let plain_vertex = rec(ShaderStage::Vertex, BARE_VERTEX, ShaderInterface::default());
    let subgroup_fragment = rec(
        ShaderStage::Fragment,
        SUBGROUP_FRAGMENT,
        subgroups(storage_out()),
    );
    let compute_ballot = rec(ShaderStage::Compute, BALLOT, subgroups(storage_out()));
    assert!(vertex_ballot.uses_subgroups() && !plain_vertex.uses_subgroups());
    // The refusal's reason, or `None` for accepted.
    let (refused, size) = (Some(VERTEX_SUBGROUPS), Some(SUBGROUP_SIZE));
    for (shaders, honoured, vertex, expected, why) in [
        (
            vec![&vertex_ballot],
            true,
            false,
            refused,
            "the vertex feature is missing",
        ),
        (
            vec![&vertex_ballot],
            true,
            true,
            None,
            "the vertex feature is there",
        ),
        (
            vec![&vertex_ballot, &subgroup_fragment],
            true,
            false,
            refused,
            "with a subgroup fragment shader",
        ),
        (
            vec![&subgroup_fragment, &vertex_ballot],
            true,
            false,
            refused,
            "in either place",
        ),
        (
            vec![&plain_vertex, &subgroup_fragment],
            true,
            false,
            None,
            "a fragment shader needs only the size",
        ),
        (
            vec![&compute_ballot],
            true,
            false,
            None,
            "a compute shader needs only the size",
        ),
        (
            vec![&plain_vertex],
            false,
            false,
            None,
            "no subgroups, nothing needed",
        ),
        // Both faults at once: the size is reported first.
        (vec![&vertex_ballot], false, false, size, "the size first"),
        (vec![&vertex_ballot], false, true, size, "the size alone"),
    ] {
        assert_eq!(
            check_requirements(shaders, verdict(honoured, vertex)),
            expected.map_or(Ok(()), |r| Err(RenderError::Unsupported(r))),
            "{why}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Test pools.

/// Runs each job at once on a thread of its own.
struct ThreadPool;

impl CompilePool for ThreadPool {
    fn spawn(&self, job: CompileJob) {
        std::thread::spawn(job);
    }
}

/// Holds jobs until the test releases them: a scripted compile delay.
#[derive(Default)]
struct HeldPool {
    jobs: Mutex<Vec<CompileJob>>,
    spawned: AtomicU64,
}

impl HeldPool {
    /// Runs every held job, in order, on one new thread.
    fn release(&self) {
        let jobs = std::mem::take(&mut *self.jobs.lock().expect("not poisoned"));
        std::thread::spawn(move || {
            for job in jobs {
                job();
            }
        });
    }

    fn spawned(&self) -> u64 {
        self.spawned.load(Ordering::Relaxed)
    }
}

impl CompilePool for HeldPool {
    fn spawn(&self, job: CompileJob) {
        self.spawned.fetch_add(1, Ordering::Relaxed);
        self.jobs.lock().expect("not poisoned").push(job);
    }
}

/// `pool` as the backend takes it.
fn held(pool: &Arc<HeldPool>) -> Arc<dyn CompilePool> {
    let pool: Arc<HeldPool> = Arc::clone(pool);
    pool
}

/// Drops every job unrun, as a pool shutting down does.
struct DropPool;

impl CompilePool for DropPool {
    fn spawn(&self, job: CompileJob) {
        drop(job);
    }
}

fn gpu(test: &str, pool: Arc<dyn CompilePool>, setting: WhileCompiling) -> Option<WgpuBackend> {
    open_with(test, || {
        WgpuBackend::with_compile_pool(WgpuConfig::DEFAULT, CompletionWake::none(), pool, setting)
    })
}

/// Waits until `id` is no longer compiling.
fn settled(gpu: &WgpuBackend, id: PipelineId) -> PipelineStatus {
    let start = Instant::now();
    loop {
        let status = gpu.pipeline_status(id);
        if status != PipelineStatus::Compiling {
            return status;
        }
        assert!(start.elapsed() < SETTLE, "the build settles in time");
        std::thread::sleep(Duration::from_millis(1));
    }
}

// ---------------------------------------------------------------------------------------------
// Hand-written modules and their interfaces.

const VERTEX: &str = "
    @group(0) @binding(0) var<uniform> offset: vec4<f32>;
    @vertex fn main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
        let x = f32(i & 1u) * 2.0 - 1.0;
        let y = f32(i >> 1u) * 2.0 - 1.0;
        return vec4(x, y, 0.0, 1.0) + offset;
    }";

/// Uses the subgroup invocation id, and binds (0, 0) as a storage buffer.
const SUBGROUP_FRAGMENT: &str = "
    @group(0) @binding(0) var<storage, read_write> out: array<u32>;
    @fragment fn main(@builtin(subgroup_invocation_id) lane: u32) -> @location(0) vec4<f32> {
        out[lane] = 1u;
        return vec4(0.0);
    }";

const BARE_VERTEX: &str = "
    @vertex fn main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
        return vec4(f32(i), 0.0, 0.0, 1.0);
    }";

const FRAGMENT: &str = "
    @group(0) @binding(0) var<uniform> offset: vec4<f32>;
    @group(0) @binding(1) var t: texture_2d<f32>;
    @group(0) @binding(2) var s: sampler;
    @fragment fn main() -> @location(0) vec4<f32> {
        return textureSample(t, s, vec2(0.5)) + offset;
    }";

/// Reads a varying the vertex shader does not write: wgpu refuses the pipeline.
const MISMATCHED_FRAGMENT: &str = "
    @fragment fn main(@location(1) v: vec4<f32>) -> @location(0) vec4<f32> { return v; }";

const BINDING_ARRAY: &str = "
    enable wgpu_binding_array;
    @group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 4>;
    @group(0) @binding(1) var s: sampler;
    @fragment fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
        return textureSample(textures[u32(p.x) % 4u], s, vec2(0.5));
    }";

const UNFILTERABLE: &str = "
    @group(0) @binding(0) var t: texture_2d<f32>;
    @group(0) @binding(1) var s: sampler;
    @fragment fn main() -> @location(0) vec4<f32> {
        return textureSampleLevel(t, s, vec2(0.5), 0.0);
    }";

const COMPUTE: &str = "
    @group(0) @binding(0) var<storage, read_write> out: array<u32>;
    @group(0) @binding(1) var img: texture_storage_2d<rgba8unorm, write>;
    @compute @workgroup_size(1) fn main() {
        out[0] = 1u;
        textureStore(img, vec2(0), vec4(1.0));
    }";

fn rows(rows: &[(u32, ResourceKind)]) -> ShaderInterface {
    ShaderInterface {
        resources: rows
            .iter()
            .map(|&(binding, kind)| ResourceBinding {
                group: 0,
                binding,
                kind,
                count: None,
            })
            .collect(),
        ..ShaderInterface::default()
    }
}

fn texture(filterable: bool) -> ResourceKind {
    ResourceKind::SampledTexture {
        dimension: ViewDimension::D2,
        sample: SampleKind::Float { filterable },
        multisampled: false,
    }
}

const SAMPLER: ResourceKind = ResourceKind::Sampler { comparison: false };

const UNIFORM: ResourceKind = ResourceKind::UniformBuffer;

fn vertex(gpu: &WgpuBackend) -> ShaderId {
    gpu.create_shader(shader_desc(
        ShaderStage::Vertex,
        "main",
        wgsl(VERTEX),
        rows(&[(0, ResourceKind::UniformBuffer)]),
    ))
    .expect("the vertex shader is accepted")
}

/// A vertex shader that binds nothing, for fragment shaders that use group 0 themselves.
fn bare_vertex(gpu: &WgpuBackend) -> ShaderId {
    gpu.create_shader(shader_desc(
        ShaderStage::Vertex,
        "main",
        wgsl(BARE_VERTEX),
        ShaderInterface::default(),
    ))
    .expect("the vertex shader is accepted")
}

fn fragment(gpu: &WgpuBackend, source: &str, interface: ShaderInterface) -> ShaderId {
    gpu.create_shader(shader_desc(
        ShaderStage::Fragment,
        "main",
        wgsl(source),
        interface,
    ))
    .expect("the fragment shader is accepted")
}

fn textured(gpu: &WgpuBackend) -> ShaderId {
    fragment(
        gpu,
        FRAGMENT,
        rows(&[
            (0, ResourceKind::UniformBuffer),
            (1, texture(true)),
            (2, SAMPLER),
        ]),
    )
}

fn mismatched(gpu: &WgpuBackend) -> ShaderId {
    fragment(gpu, MISMATCHED_FRAGMENT, ShaderInterface::default())
}

fn ready(draw: Result<DrawPipeline, RenderError>) -> ReadyPipeline {
    match draw {
        Ok(DrawPipeline::Ready(p)) => Ok(p),
        other => Err(other),
    }
    .expect("a ready pipeline")
}

// ---------------------------------------------------------------------------------------------
// On a GPU.

#[test]
fn hand_written_modules_become_pipelines() {
    bounded(|| {
        let Some(gpu) = gpu("pipelines", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let vs = vertex(&gpu);
        let fs = textured(&gpu);
        let graphics = gpu
            .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
            .expect("accepted");
        let p = ready(gpu.draw_pipeline(graphics));
        assert!(p.render().is_some() && p.compute().is_none());
        assert_eq!(gpu.pipeline_status(graphics), PipelineStatus::Ready);

        // The same shaders in a second pipeline, drawing quads into two targets.
        let mut two = graphics_desc(vs, Some(fs));
        two.topology = Topology::Quads;
        two.targets.colors[1] = Some(PixelFormat::R32Float);
        let second = gpu.create_graphics_pipeline(two).expect("accepted");
        assert_eq!(settled(&gpu, second), PipelineStatus::Ready);

        let cs = gpu
            .create_shader(shader_desc(
                ShaderStage::Compute,
                "main",
                wgsl(COMPUTE),
                rows(&[
                    (0, ResourceKind::StorageBuffer { writable: true }),
                    (
                        1,
                        ResourceKind::StorageTexture {
                            dimension: ViewDimension::D2,
                            format: PixelFormat::A8B8G8R8Unorm,
                            writable: true,
                        },
                    ),
                ]),
            ))
            .expect("accepted");
        let compute = gpu
            .create_compute_pipeline(ComputePipelineDesc { shader: cs })
            .expect("accepted");
        let p = ready(gpu.draw_pipeline(compute));
        assert!(p.compute().is_some() && p.render().is_none());
        assert_eq!(gpu.pipeline_counts(), PipelineCounts::default());
        assert_eq!(gpu.host_errors(), 0);
    });
}

/// The depth and stencil states wgpu-core would refuse as given (a bias on lines, a stencil test
/// on a depth-only target, a depth test on a stencil-only one) build: `state` drops the part
/// wgpu cannot take (`state` module docs, "Rasterisation, depth and stencil").
#[test]
fn depth_and_stencil_state_wgpu_refuses_as_given_builds() {
    use starquake_render::pipeline::{DepthBias, StencilFace, StencilOperation, StencilState};
    use starquake_render::resource::CompareFunction;
    bounded(|| {
        let Some(gpu) = gpu(
            "depth and stencil",
            Arc::new(ThreadPool),
            WhileCompiling::Wait,
        ) else {
            return;
        };
        let vs = vertex(&gpu);
        let depth_only = || {
            let mut d = graphics_desc(vs, None);
            d.targets.colors = [None; starquake_render::caps::GUEST_COLOR_TARGETS];
            d
        };
        let face = StencilFace {
            compare: CompareFunction::Equal,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::IncrementClamp,
            pass: StencilOperation::Replace,
        };
        let stencil = StencilState {
            front: face,
            back: face,
            read_mask: 0xff,
            write_mask: 0xff,
        };
        let mut biased_lines = depth_only();
        biased_lines.topology = Topology::Lines;
        biased_lines.targets.depth_stencil = Some(PixelFormat::D32Float);
        biased_lines.raster.depth_bias = DepthBias {
            constant: 2.0,
            slope_scale: 1.5,
            clamp: 0.0,
        };
        let mut stencil_on_d16 = depth_only();
        stencil_on_d16.targets.depth_stencil = Some(PixelFormat::D16Unorm);
        stencil_on_d16.depth_stencil.stencil = Some(stencil);
        let mut depth_on_s8 = depth_only();
        depth_on_s8.targets.depth_stencil = Some(PixelFormat::S8Uint);
        depth_on_s8.depth_stencil.depth_test = Some(CompareFunction::Less);
        depth_on_s8.depth_stencil.depth_write = true;
        depth_on_s8.depth_stencil.stencil = Some(stencil);
        for (what, desc) in [
            ("a bias on lines", biased_lines),
            ("a stencil test on D16", stencil_on_d16),
            ("a depth test on S8", depth_on_s8),
        ] {
            let id = gpu.create_graphics_pipeline(desc).expect(what);
            assert_eq!(settled(&gpu, id), PipelineStatus::Ready, "{what}");
        }
        assert_eq!(gpu.host_errors(), 0);
    });
}

/// A cell settled with a real pipeline keeps it when a failure follows, and one settled failed
/// stays failed when a pipeline follows: the first outcome stands either way.
#[test]
fn the_first_outcome_stands_with_a_real_pipeline() {
    bounded(|| {
        let Some(gpu) = gpu("first outcome", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let vs = vertex(&gpu);
        let fs = textured(&gpu);
        let id = gpu
            .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
            .expect("accepted");
        let host = ready(gpu.draw_pipeline(id)).0;

        let ready_first = PipelineCell::new();
        ready_first.settle(Ok(host.clone()));
        ready_first.settle(Err("second".into()));
        assert_eq!(ready_first.status(), PipelineStatus::Ready);

        let failed_first = PipelineCell::new();
        failed_first.settle(Err("first".into()));
        failed_first.settle(Ok(host));
        assert_eq!(failed_first.status(), PipelineStatus::Failed);
    });
}

/// A module nested exactly at the bound builds (the deepest Metal compiles, `MAX_SHADER_NESTING`'s
/// docs); one level deeper is refused by `create_shader` with [`NESTING`], before any build can
/// overflow the pool worker's stack; and a deep module refused by the shared checks first (a
/// geometry shader) is freed without overflowing.
#[test]
fn a_module_at_the_nesting_bound_builds_and_one_past_it_is_refused() {
    bounded(|| {
        let Some(gpu) = gpu("nesting", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let at = gpu
            .create_shader(compute_desc(
                nested_compute(MAX_SHADER_NESTING, true),
                "main",
            ))
            .expect("accepted at the bound");
        let id = gpu
            .create_compute_pipeline(ComputePipelineDesc { shader: at })
            .expect("accepted");
        assert_eq!(settled(&gpu, id), PipelineStatus::Ready);
        assert_eq!(
            gpu.create_shader(compute_desc(
                nested_compute(MAX_SHADER_NESTING + 1, true),
                "main"
            )),
            Err(RenderError::Unsupported(NESTING))
        );
        let mut geometry = compute_desc(nested_compute(100_000, false), "main");
        geometry.stage = ShaderStage::Geometry;
        assert_eq!(
            gpu.create_shader(geometry),
            Err(RenderError::StageUnsupported(ShaderStage::Geometry))
        );
        assert_eq!(gpu.host_errors(), 0);
    });
}

#[test]
fn a_binding_array_becomes_a_pipeline_with_an_explicit_layout() {
    bounded(|| {
        let Some(gpu) = gpu("binding array", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let needed = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        if !gpu.device().features().contains(needed) {
            eprintln!("skipping binding array: the device has no binding arrays");
            return;
        }
        let vs = bare_vertex(&gpu);
        let mut interface = rows(&[(0, texture(true)), (1, SAMPLER)]);
        interface.resources[0].count = NonZeroU32::new(4);
        let fs = fragment(&gpu, BINDING_ARRAY, interface);
        let id = gpu
            .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
            .expect("accepted");
        assert_eq!(settled(&gpu, id), PipelineStatus::Ready);
        assert_eq!(gpu.host_errors(), 0);
    });
}

/// A texture the interface says cannot be filtered, sampled through a sampler: wgpu refuses a
/// filtering sampler entry for it, so the layout must say non-filtering.
#[test]
fn a_sampler_used_with_an_unfilterable_texture_builds() {
    bounded(|| {
        let Some(gpu) = gpu("unfilterable", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let vs = bare_vertex(&gpu);
        for filterable in [false, true] {
            let fs = fragment(
                &gpu,
                UNFILTERABLE,
                rows(&[(0, texture(filterable)), (1, SAMPLER)]),
            );
            let id = gpu
                .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
                .expect("accepted");
            assert_eq!(
                settled(&gpu, id),
                PipelineStatus::Ready,
                "filterable {filterable}"
            );
        }
    });
}

/// Scripted compile delays under `Wait`: the draw blocks until the build is released, then
/// gets the pipeline, and nothing is counted.
#[test]
fn under_wait_a_draw_waits_for_a_delayed_build() {
    bounded(|| {
        let pool = Arc::new(HeldPool::default());
        let Some(gpu) = gpu("wait", held(&pool), WhileCompiling::Wait) else {
            return;
        };
        let gpu = Arc::new(gpu);
        let vs = vertex(&gpu);
        let fs = textured(&gpu);
        let id = gpu
            .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
            .expect("accepted");
        assert_eq!(gpu.pipeline_status(id), PipelineStatus::Compiling);
        assert_eq!(
            pool.spawned(),
            1,
            "queued on the pool before the call returned"
        );
        let (tx, rx) = mpsc::channel();
        let drawer = Arc::clone(&gpu);
        let draw = std::thread::spawn(move || {
            let got = drawer.draw_pipeline(id);
            tx.send(matches!(got, Ok(DrawPipeline::Ready(_))))
                .expect("the test is listening");
        });
        assert_eq!(
            rx.recv_timeout(STILL),
            Err(mpsc::RecvTimeoutError::Timeout),
            "the draw waits"
        );
        assert_eq!(gpu.pipeline_status(id), PipelineStatus::Compiling);
        pool.release();
        assert_eq!(rx.recv_timeout(SETTLE), Ok(true));
        draw.join().expect("the draw thread ends");
        assert_eq!(gpu.pipeline_status(id), PipelineStatus::Ready);
        assert_eq!(gpu.pipeline_counts(), PipelineCounts::default());
    });
}

/// Scripted compile delays under `Skip`: each draw while compiling is skipped and counted;
/// once the build is released, draws get the pipeline and the count stays.
#[test]
fn under_skip_draws_are_skipped_and_counted_until_the_build_finishes() {
    bounded(|| {
        let pool = Arc::new(HeldPool::default());
        let Some(gpu) = gpu("skip", held(&pool), WhileCompiling::Skip) else {
            return;
        };
        let vs = vertex(&gpu);
        let fs = textured(&gpu);
        let id = gpu
            .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
            .expect("accepted");
        for n in 1..=3 {
            assert!(matches!(
                gpu.draw_pipeline(id),
                Ok(DrawPipeline::Skipped(SkipReason::Compiling))
            ));
            assert_eq!(gpu.pipeline_counts().skipped_while_compiling, n);
        }
        pool.release();
        assert_eq!(settled(&gpu, id), PipelineStatus::Ready);
        ready(gpu.draw_pipeline(id));
        assert_eq!(
            gpu.pipeline_counts(),
            PipelineCounts {
                skipped_while_compiling: 3,
                failed_pipelines: 0
            }
        );
    });
}

/// A pipeline wgpu refuses is built once, stays failed, and is counted once however many draws
/// meet it, under either setting.
#[test]
fn a_failed_pipeline_is_built_once_and_counted_once() {
    bounded(|| {
        for setting in [WhileCompiling::Wait, WhileCompiling::Skip] {
            let pool = Arc::new(HeldPool::default());
            let Some(gpu) = gpu("failed", held(&pool), setting) else {
                return;
            };
            let vs = vertex(&gpu);
            let fs = mismatched(&gpu);
            let a = gpu
                .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
                .expect("accepted: the mismatch is wgpu's to find");
            let b = gpu
                .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
                .expect("accepted");
            pool.release();
            assert_eq!(settled(&gpu, a), PipelineStatus::Failed);
            assert_eq!(settled(&gpu, b), PipelineStatus::Failed);
            for _ in 0..3 {
                assert!(matches!(
                    gpu.draw_pipeline(a),
                    Ok(DrawPipeline::Skipped(SkipReason::Failed))
                ));
            }
            assert_eq!(gpu.pipeline_counts().failed_pipelines, 1, "{setting:?}");
            assert!(matches!(
                gpu.draw_pipeline(b),
                Ok(DrawPipeline::Skipped(SkipReason::Failed))
            ));
            assert_eq!(
                gpu.pipeline_counts(),
                PipelineCounts {
                    skipped_while_compiling: 0,
                    failed_pipelines: 2
                },
                "{setting:?}"
            );
            assert_eq!(pool.spawned(), 2, "one build per pipeline, never again");
            pool.release();
            assert_eq!(gpu.pipeline_status(a), PipelineStatus::Failed);
            assert_eq!(
                gpu.host_errors(),
                0,
                "the failure was caught in the build's scopes"
            );
        }
    });
}

#[test]
fn a_build_the_pool_drops_fails_and_releases_the_draw() {
    bounded(|| {
        let Some(gpu) = gpu("dropped", Arc::new(DropPool), WhileCompiling::Wait) else {
            return;
        };
        let vs = vertex(&gpu);
        let fs = textured(&gpu);
        let id = gpu
            .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
            .expect("accepted");
        assert_eq!(gpu.pipeline_status(id), PipelineStatus::Failed);
        assert!(matches!(
            gpu.draw_pipeline(id),
            Ok(DrawPipeline::Skipped(SkipReason::Failed))
        ));
    });
}

#[test]
fn an_unsupported_stage_is_the_exact_error() {
    bounded(|| {
        let Some(gpu) = gpu("stages", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let geometry = shader_desc(
            ShaderStage::Geometry,
            "main",
            wgsl(VERTEX),
            ShaderInterface::default(),
        );
        assert_eq!(
            gpu.create_shader(geometry),
            Err(RenderError::StageUnsupported(ShaderStage::Geometry))
        );
        let vs = vertex(&gpu);
        let fs = textured(&gpu);
        let mut desc = graphics_desc(vs, Some(fs));
        desc.stages.geometry = Some(vs);
        assert_eq!(
            gpu.create_graphics_pipeline(desc),
            Err(RenderError::StageUnsupported(ShaderStage::Geometry))
        );
        let mut desc = graphics_desc(vs, Some(fs));
        desc.stages.tess_control = Some(vs);
        desc.stages.tess_eval = Some(vs);
        desc.topology = Topology::Patches;
        assert_eq!(
            gpu.create_graphics_pipeline(desc),
            Err(RenderError::StageUnsupported(ShaderStage::TessControl))
        );
    });
}

/// Each check of `create_*_pipeline` alone, and each next to the one after it, which must not
/// be the one reported (module docs, "Pipelines").
#[test]
fn pipeline_checks_run_in_order() {
    bounded(|| {
        let Some(gpu) = gpu("order", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let vs = vertex(&gpu);
        let fs = textured(&gpu);
        let stale = ShaderId::new(fs.index(), fs.generation() + 1);
        // Fragment rows that conflict with the vertex shader's uniform buffer at (0, 0).
        let conflicting = fragment(
            &gpu,
            UNFILTERABLE,
            rows(&[(0, texture(true)), (1, SAMPLER)]),
        );
        let geometry =
            |d: &mut starquake_render::GraphicsPipelineDesc| d.stages.geometry = Some(vs);
        let wrong_stage = |d: &mut starquake_render::GraphicsPipelineDesc| d.stages.vertex = fs;
        let stale_id =
            |d: &mut starquake_render::GraphicsPipelineDesc| d.stages.fragment = Some(stale);
        let conflict =
            |d: &mut starquake_render::GraphicsPipelineDesc| d.stages.fragment = Some(conflicting);
        let cull =
            |d: &mut starquake_render::GraphicsPipelineDesc| d.raster.cull = CullMode::FrontAndBack;
        let order: [(Fault<'_>, RenderError); 5] = [
            (
                &geometry,
                RenderError::StageUnsupported(ShaderStage::Geometry),
            ),
            (
                &wrong_stage,
                RenderError::Invalid("a pipeline's vertex shader is not a vertex shader"),
            ),
            (
                &stale_id,
                RenderError::Invalid(starquake_render::validate::NO_SHADER),
            ),
            (&conflict, RenderError::Invalid(layout::STAGE_CONFLICT)),
            (&cull, RenderError::Unsupported(state::CULL_BOTH)),
        ];
        for (i, (fault, error)) in order.iter().enumerate() {
            let mut d = graphics_desc(vs, Some(fs));
            fault(&mut d);
            assert_eq!(
                gpu.create_graphics_pipeline(d),
                Err(error.clone()),
                "check {i} alone"
            );
            if let Some((next, _)) = order.get(i + 1) {
                let mut d = graphics_desc(vs, Some(fs));
                next(&mut d);
                fault(&mut d);
                assert_eq!(
                    gpu.create_graphics_pipeline(d),
                    Err(error.clone()),
                    "check {i} before check {}",
                    i + 1
                );
            }
        }
        // No compile pool comes first of all.
        let Some(bare) = open_with("order, no pool", || {
            WgpuBackend::new(WgpuConfig::DEFAULT, CompletionWake::none())
        }) else {
            return;
        };
        let bvs = vertex(&bare);
        let mut d = graphics_desc(bvs, None);
        d.targets.colors = [None; starquake_render::caps::GUEST_COLOR_TARGETS];
        assert_eq!(
            bare.create_graphics_pipeline(d.clone()),
            Err(RenderError::Unsupported(NO_COMPILE_POOL))
        );
        d.stages.geometry = Some(bvs);
        assert_eq!(
            bare.create_graphics_pipeline(d),
            Err(RenderError::Unsupported(NO_COMPILE_POOL)),
            "before the shared checks"
        );
        assert_eq!(
            bare.create_compute_pipeline(ComputePipelineDesc { shader: bvs }),
            Err(RenderError::Unsupported(NO_COMPILE_POOL))
        );
    });
}

#[test]
fn compute_pipelines_check_their_shader() {
    bounded(|| {
        let Some(gpu) = gpu("compute checks", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let vs = vertex(&gpu);
        assert_eq!(
            gpu.create_compute_pipeline(ComputePipelineDesc { shader: vs }),
            Err(RenderError::Invalid(
                "a compute pipeline's shader is not a compute shader"
            ))
        );
        let stale = ShaderId::new(vs.index(), vs.generation() + 1);
        assert_eq!(
            gpu.create_compute_pipeline(ComputePipelineDesc { shader: stale }),
            Err(RenderError::Invalid(starquake_render::validate::NO_SHADER))
        );
    });
}

#[test]
fn a_subgroup_pipeline_follows_the_device() {
    bounded(|| {
        let Some(gpu) = gpu("subgroups", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let cs = gpu
            .create_shader(shader_desc(
                ShaderStage::Compute,
                "main",
                wgsl(BALLOT),
                subgroups(storage_out()),
            ))
            .expect("the shader itself is accepted");
        let result = gpu.create_compute_pipeline(ComputePipelineDesc { shader: cs });
        if gpu.honours_subgroups() {
            let id = result.expect("accepted");
            assert_eq!(settled(&gpu, id), PipelineStatus::Ready);
        } else {
            assert_eq!(result, Err(RenderError::Unsupported(SUBGROUP_SIZE)));
        }
        let info = gpu.device().adapter_info();
        let support = SubgroupSupport::of(&info, gpu.device().features());
        assert_eq!(gpu.subgroup_support(), support);
        assert_eq!(gpu.honours_subgroups(), support.honoured());
        // The adapter report records the facts that decided it, and the verdict.
        let report = gpu.adapter_report().to_string();
        for line in [
            format!("adapter: {}\n", info.name),
            format!("backend: {:?}\n", info.backend),
            format!(
                "subgroup size range: {} to {}\n",
                info.subgroup_min_size, info.subgroup_max_size
            ),
            format!("subgroups of 32: {support}\n"),
        ] {
            assert!(report.contains(&line), "{line:?} in {report}");
        }
        let family = if apple_gpu_family(&info) { "yes" } else { "no" };
        let line = format!("Apple GPU family: {family}\n");
        assert!(report.contains(&line), "{line:?} in {report}");
    });
}

/// The backend refuses a compute pipeline that uses subgroups whenever its verdict does not meet
/// "subgroup size 32", for each reason, on any GPU (the verdict is forced, since a device that
/// meets the size takes the accepting branch); a compute pipeline without subgroups is still
/// accepted.
#[test]
fn a_compute_subgroup_pipeline_is_refused_where_the_size_is_not_met() {
    bounded(|| {
        let Some(mut gpu) = gpu(
            "compute subgroup refusal",
            Arc::new(ThreadPool),
            WhileCompiling::Wait,
        ) else {
            return;
        };
        let cs = gpu
            .create_shader(shader_desc(
                ShaderStage::Compute,
                "main",
                wgsl(BALLOT),
                subgroups(storage_out()),
            ))
            .expect("the shader itself is accepted");
        let plain = gpu
            .create_shader(shader_desc(
                ShaderStage::Compute,
                "main",
                wgsl(
                    "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
                    @compute @workgroup_size(1) fn main() { out[0] = 1u; }",
                ),
                storage_out(),
            ))
            .expect("the shader itself is accepted");
        for support in [
            SubgroupSupport::Range { min: 4, max: 64 },
            SubgroupSupport::NoSubgroups,
        ] {
            gpu.set_subgroup_support(support);
            assert!(!gpu.honours_subgroups());
            assert_eq!(
                gpu.create_compute_pipeline(ComputePipelineDesc { shader: cs }),
                Err(RenderError::Unsupported(SUBGROUP_SIZE)),
                "{support:?}"
            );
            let id = gpu
                .create_compute_pipeline(ComputePipelineDesc { shader: plain })
                .expect("a pipeline without subgroups is accepted");
            assert_eq!(settled(&gpu, id), PipelineStatus::Ready);
        }
        assert_eq!(gpu.host_errors(), 0);
    });
}

/// A graphics pipeline whose vertex shader uses a subgroup operation: refused for the size where
/// the device does not meet it, refused with [`VERTEX_SUBGROUPS`] where it meets the size but
/// has no subgroups in vertex shaders (an Apple GPU under wgpu 30), and built otherwise. Forced
/// to the second case, it is refused on any GPU; and where the device has subgroups, a subgroup
/// fragment shader beside a plain vertex shader is still built.
#[test]
fn a_vertex_subgroup_pipeline_follows_the_device() {
    bounded(|| {
        let Some(mut gpu) = gpu(
            "vertex subgroups",
            Arc::new(ThreadPool),
            WhileCompiling::Wait,
        ) else {
            return;
        };
        assert_eq!(
            gpu.honours_vertex_subgroups(),
            gpu.device()
                .features()
                .contains(wgpu::Features::SUBGROUP_VERTEX)
        );
        if apple_gpu_family(&gpu.device().adapter_info()) {
            // wgpu-hal 30.0.2's Metal backend never offers it (`src/metal/adapter.rs`).
            assert!(gpu.honours_subgroups() && !gpu.honours_vertex_subgroups());
        }
        let vs = gpu
            .create_shader(shader_desc(
                ShaderStage::Vertex,
                "main",
                wgsl(VERTEX_BALLOT),
                subgroups(ShaderInterface::default()),
            ))
            .expect("the shader itself is accepted");
        let mut desc = graphics_desc(vs, None);
        desc.targets.colors = [None; starquake_render::caps::GUEST_COLOR_TARGETS];
        let result = gpu.create_graphics_pipeline(desc.clone());
        let honours = gpu.honours_subgroups();
        if !honours {
            assert_eq!(result, Err(RenderError::Unsupported(SUBGROUP_SIZE)));
        } else if !gpu.honours_vertex_subgroups() {
            assert_eq!(result, Err(RenderError::Unsupported(VERTEX_SUBGROUPS)));
        } else {
            let id = result.expect("accepted");
            assert_eq!(settled(&gpu, id), PipelineStatus::Ready);
        }
        // Forced: the size met, no subgroups in vertex shaders.
        gpu.set_subgroup_support(SubgroupSupport::AppleFamily);
        gpu.set_vertex_subgroups(false);
        assert!(gpu.honours_subgroups() && !gpu.honours_vertex_subgroups());
        assert_eq!(
            gpu.create_graphics_pipeline(desc),
            Err(RenderError::Unsupported(VERTEX_SUBGROUPS))
        );
        // Only where the device really has subgroups, so the build that follows can succeed.
        if honours {
            let fs = fragment(&gpu, SUBGROUP_FRAGMENT, subgroups(storage_out()));
            let id = gpu
                .create_graphics_pipeline(graphics_desc(bare_vertex(&gpu), Some(fs)))
                .expect("a fragment shader needs only the size");
            assert_eq!(settled(&gpu, id), PipelineStatus::Ready);
        }
        assert_eq!(gpu.host_errors(), 0);
    });
}

/// On an Apple GPU (every Apple-silicon Mac), a compute pipeline that uses a subgroup operation
/// and a graphics pipeline whose fragment shader uses the subgroup invocation id both build and
/// reach `Ready`, although wgpu reports a subgroup size range of 4 to 64 there. On an
/// Apple-silicon Mac the adapter must be found to be an Apple GPU; on other hosts an adapter that
/// is not one makes the test say so and stop, and `a_subgroup_pipeline_follows_the_device` covers
/// it.
#[test]
fn on_an_apple_gpu_subgroup_pipelines_are_ready() {
    bounded(|| {
        let Some(gpu) = gpu(
            "apple subgroups",
            Arc::new(ThreadPool),
            WhileCompiling::Wait,
        ) else {
            return;
        };
        let info = gpu.device().adapter_info();
        // Every Apple-silicon Mac's GPU is an Apple GPU, so there the name rule must find it
        // rather than let the test stop.
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert!(
                apple_gpu_family(&info),
                "an Apple-silicon Mac's adapter is the Apple GPU family: {:?} on {:?}",
                info.name,
                info.backend
            );
        } else if !apple_gpu_family(&info) {
            eprintln!(
                "skipping apple subgroups: the adapter is not an Apple GPU ({:?} on {:?})",
                info.name, info.backend
            );
            return;
        }
        assert_eq!(gpu.subgroup_support(), SubgroupSupport::AppleFamily);
        assert!(gpu.honours_subgroups());
        let cs = gpu
            .create_shader(shader_desc(
                ShaderStage::Compute,
                "main",
                wgsl(BALLOT),
                subgroups(storage_out()),
            ))
            .expect("the shader itself is accepted");
        let compute = gpu
            .create_compute_pipeline(ComputePipelineDesc { shader: cs })
            .expect("a subgroup compute pipeline is accepted on an Apple GPU");
        assert_eq!(settled(&gpu, compute), PipelineStatus::Ready);
        let vs = bare_vertex(&gpu);
        let fs = fragment(&gpu, SUBGROUP_FRAGMENT, subgroups(storage_out()));
        let graphics = gpu
            .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
            .expect("a subgroup graphics pipeline is accepted on an Apple GPU");
        assert_eq!(settled(&gpu, graphics), PipelineStatus::Ready);
        assert_eq!(gpu.host_errors(), 0);
    });
}

#[test]
fn shaders_and_pipelines_are_destroyed_by_id() {
    bounded(|| {
        let pool = Arc::new(HeldPool::default());
        let Some(gpu) = gpu("destroy", held(&pool), WhileCompiling::Wait) else {
            return;
        };
        let vs = vertex(&gpu);
        let fs = textured(&gpu);
        let id = gpu
            .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
            .expect("accepted");
        // The shaders go before the build runs; the build still has them.
        let now = FenceValue::ZERO;
        gpu.destroy(ResourceId::Shader(vs), now);
        gpu.destroy(ResourceId::Shader(fs), now);
        assert_eq!(gpu.refused_destroys(), 0);
        pool.release();
        assert_eq!(settled(&gpu, id), PipelineStatus::Ready);
        assert_eq!(
            gpu.create_graphics_pipeline(graphics_desc(vs, Some(fs))),
            Err(RenderError::Invalid(starquake_render::validate::NO_SHADER)),
            "the ids are stale"
        );
        gpu.destroy(ResourceId::Shader(vs), now);
        assert_eq!(gpu.refused_destroys(), 1, "a second destroy is refused");

        gpu.destroy(ResourceId::Pipeline(id), now);
        assert_eq!(
            gpu.pipeline_status(id),
            PipelineStatus::Failed,
            "no live pipeline"
        );
        assert_eq!(
            gpu.draw_pipeline(id).map(|_| ()),
            Err(RenderError::Invalid(
                starquake_render::validate::NO_PIPELINE
            ))
        );
        gpu.destroy(ResourceId::Pipeline(id), now);
        assert_eq!(gpu.refused_destroys(), 2);

        // A destroy that waits for a fence value not yet reached keeps the pipeline live.
        let vs = vertex(&gpu);
        let mut depth_only = graphics_desc(vs, None);
        depth_only.targets.colors = [None; starquake_render::caps::GUEST_COLOR_TARGETS];
        let later = gpu.create_graphics_pipeline(depth_only).expect("accepted");
        gpu.destroy(ResourceId::Pipeline(later), FenceValue::new(1));
        assert_eq!(
            gpu.pipeline_status(later),
            PipelineStatus::Compiling,
            "still live"
        );
    });
}

#[test]
fn stale_and_unknown_pipeline_ids() {
    bounded(|| {
        let Some(gpu) = gpu("stale", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let vs = vertex(&gpu);
        let fs = textured(&gpu);
        let id = gpu
            .create_graphics_pipeline(graphics_desc(vs, Some(fs)))
            .expect("accepted");
        assert_eq!(settled(&gpu, id), PipelineStatus::Ready);
        let stale = PipelineId::new(id.index(), id.generation() + 1);
        let unknown = PipelineId::new(id.index() + 1, id.generation());
        for other in [stale, unknown] {
            assert_eq!(gpu.pipeline_status(other), PipelineStatus::Failed);
            assert_eq!(
                gpu.draw_pipeline(other).map(|_| ()),
                Err(RenderError::Invalid(
                    starquake_render::validate::NO_PIPELINE
                ))
            );
        }
    });
}

#[test]
fn nothing_is_created_once_the_device_is_lost() {
    bounded(|| {
        let Some(gpu) = gpu("lost", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let vs = vertex(&gpu);
        let fs = textured(&gpu);
        gpu.lose_device_for_test();
        let start = Instant::now();
        while gpu.create_shader(shader_desc(
            ShaderStage::Vertex,
            "main",
            wgsl(VERTEX),
            ShaderInterface::default(),
        )) != Err(RenderError::DeviceLost)
        {
            assert!(start.elapsed() < SETTLE, "the loss is seen in time");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(
            gpu.create_graphics_pipeline(graphics_desc(vs, Some(fs))),
            Err(RenderError::DeviceLost)
        );
        // Check 3 comes after the shared checks and before the ids (module docs, "Pipelines").
        let mut geometry = graphics_desc(vs, Some(fs));
        geometry.stages.geometry = Some(vs);
        assert_eq!(
            gpu.create_graphics_pipeline(geometry),
            Err(RenderError::StageUnsupported(ShaderStage::Geometry))
        );
        let stale = ShaderId::new(fs.index(), fs.generation() + 1);
        assert_eq!(
            gpu.create_graphics_pipeline(graphics_desc(vs, Some(stale))),
            Err(RenderError::DeviceLost)
        );
        assert_eq!(
            gpu.create_compute_pipeline(ComputePipelineDesc { shader: stale }),
            Err(RenderError::DeviceLost)
        );
    });
}

/// Checks 4, 6 and 7 of the module docs' order with two faults at once. The backend is told its
/// device cannot promise subgroups of 32 (as a 4 to 64 adapter outside the Apple family), so the
/// order is checked on every device, an Apple GPU included.
#[test]
fn requirements_come_after_the_ids_and_before_the_layout() {
    bounded(|| {
        let Some(mut gpu) = gpu(
            "requirements order",
            Arc::new(ThreadPool),
            WhileCompiling::Wait,
        ) else {
            return;
        };
        gpu.set_subgroup_support(SubgroupSupport::Range { min: 4, max: 64 });
        assert!(!gpu.honours_subgroups());
        let vs = vertex(&gpu);
        // A fragment shader that uses subgroups and binds (0, 0) as a storage buffer, which the
        // vertex shader binds as a uniform buffer.
        let fs = fragment(&gpu, SUBGROUP_FRAGMENT, subgroups(storage_out()));
        assert_eq!(
            gpu.create_graphics_pipeline(graphics_desc(vs, Some(fs))),
            Err(RenderError::Unsupported(SUBGROUP_SIZE)),
            "requirements before the layout's conflict"
        );
        assert_eq!(
            gpu.create_graphics_pipeline(graphics_desc(fs, Some(fs))),
            Err(RenderError::Invalid(
                "a pipeline's vertex shader is not a vertex shader"
            )),
            "the stages before the requirements"
        );
        let bare = bare_vertex(&gpu);
        assert_eq!(
            gpu.create_graphics_pipeline(graphics_desc(bare, Some(fs))),
            Err(RenderError::Unsupported(SUBGROUP_SIZE)),
            "alone"
        );
    });
}

/// The cross-check of a fragment shader's `interface` against its module `source` (module docs,
/// "The interface and the module"), with no GPU.
fn agree(source: &str, interface: ShaderInterface) -> Result<(), RenderError> {
    let rec = ShaderRec::new(shader_desc(
        ShaderStage::Fragment,
        "main",
        wgsl(source),
        interface,
    ))
    .expect("the shader is accepted");
    check_interfaces([&rec])
}

/// [`BINDING_ARRAY`]'s rows, the texture array given a count of `n`.
fn arrayed(n: u32) -> ShaderInterface {
    let mut interface = rows(&[(0, texture(true)), (1, SAMPLER)]);
    interface.resources[0].count = NonZeroU32::new(n);
    interface
}

/// Binds a binding array whose size is not a constant at (0, 0).
const RUNTIME_SIZED: &str = "
    enable wgpu_binding_array;
    struct B { d: array<u32> }
    @group(0) @binding(0) var<storage> bufs: binding_array<B>;
    @fragment fn main() -> @location(0) vec4<f32> {
        return vec4(f32(arrayLength(&bufs[0].d)));
    }";

const READ_STORAGE: ResourceKind = ResourceKind::StorageBuffer { writable: false };

/// Each row's count against the module's, with the exact error, the edge and both sides of it.
#[test]
fn the_interface_and_module_must_agree_on_array_counts() {
    let mismatch = Err(RenderError::Invalid(ARRAY_COUNT_MISMATCH));
    // The module's array of 4 against the interface's counts: 4 (the edge), 3 and 5 on either
    // side, and a lone resource.
    assert_eq!(agree(BINDING_ARRAY, arrayed(4)), Ok(()));
    assert_eq!(agree(BINDING_ARRAY, arrayed(3)), mismatch);
    assert_eq!(agree(BINDING_ARRAY, arrayed(5)), mismatch);
    assert_eq!(
        agree(BINDING_ARRAY, rows(&[(0, texture(true)), (1, SAMPLER)])),
        mismatch,
        "an array the interface calls one resource"
    );
    // A lone resource the interface calls an array.
    let mut lone = rows(&[(0, UNIFORM), (1, texture(true)), (2, SAMPLER)]);
    assert_eq!(agree(FRAGMENT, lone.clone()), Ok(()));
    lone.resources[1].count = NonZeroU32::new(2);
    assert_eq!(agree(FRAGMENT, lone), mismatch);
    // A row the module does not bind is not compared, count or none.
    let mut extra = rows(&[(0, UNIFORM), (1, texture(true)), (2, SAMPLER), (3, SAMPLER)]);
    extra.resources[3].count = NonZeroU32::new(2);
    assert_eq!(agree(FRAGMENT, extra), Ok(()));
    // A binding array whose size is not a constant: no interface can describe it.
    for count in [None, NonZeroU32::new(1)] {
        let mut i = rows(&[(0, READ_STORAGE)]);
        i.resources[0].count = count;
        assert_eq!(
            agree(RUNTIME_SIZED, i),
            Err(RenderError::Unsupported(UNSIZED_BINDING_ARRAY)),
            "{count:?}"
        );
    }
}

/// A module that uses subgroups must declare them; one that does not may, and a quad operation
/// alone is no subgroup use to the scan, while a declared requirement is honoured as declared
/// (module docs, "The interface and the module").
#[test]
fn the_interface_must_declare_the_subgroups_its_module_uses() {
    assert_eq!(
        agree(SUBGROUP_FRAGMENT, storage_out()),
        Err(RenderError::Invalid(UNDECLARED_SUBGROUPS))
    );
    assert_eq!(agree(SUBGROUP_FRAGMENT, subgroups(storage_out())), Ok(()));
    let plain = rows(&[(0, UNIFORM), (1, texture(true)), (2, SAMPLER)]);
    assert_eq!(agree(FRAGMENT, subgroups(plain.clone())), Ok(()));
    assert_eq!(agree(FRAGMENT, plain), Ok(()));
    let quad = "
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;
        @compute @workgroup_size(32) fn main(@builtin(local_invocation_index) l: u32) {
            out[l] = quadBroadcast(l, 1u);
        }";
    let quad_rec = |interface| {
        ShaderRec::new(shader_desc(
            ShaderStage::Compute,
            "main",
            wgsl(quad),
            interface,
        ))
        .expect("accepted")
    };
    let undeclared = quad_rec(storage_out());
    let declared = quad_rec(subgroups(storage_out()));
    assert_eq!(check_interfaces([&undeclared]), Ok(()));
    assert_eq!(check_interfaces([&declared]), Ok(()));
    let unmet = verdict(false, true);
    assert_eq!(check_requirements([&undeclared], unmet), Ok(()));
    assert_eq!(
        check_requirements([&declared], unmet),
        Err(RenderError::Unsupported(SUBGROUP_SIZE)),
        "the declared requirement is honoured"
    );
}

/// Two faults in one shader: the rows come before the subgroups, and the rows are checked in
/// table order (the first fault found is reported, either way round).
#[test]
fn a_shaders_rows_are_checked_in_table_order_before_its_subgroups() {
    let mismatch = Err(RenderError::Invalid(ARRAY_COUNT_MISMATCH));
    let subgroup_array = "
        enable wgpu_binding_array;
        @group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 4>;
        @group(0) @binding(1) var s: sampler;
        @fragment fn main(@builtin(subgroup_invocation_id) lane: u32) -> @location(0) vec4<f32> {
            return textureSample(textures[lane % 4u], s, vec2(0.5));
        }";
    assert_eq!(
        agree(subgroup_array, rows(&[(0, texture(true)), (1, SAMPLER)])),
        mismatch,
        "both faults: the rows first"
    );
    assert_eq!(
        agree(subgroup_array, arrayed(4)),
        Err(RenderError::Invalid(UNDECLARED_SUBGROUPS)),
        "the subgroups alone"
    );
    assert_eq!(agree(subgroup_array, subgroups(arrayed(4))), Ok(()));
    // Two rows that disagree: the unsized one first in the table, then last.
    let two_faults = "
        enable wgpu_binding_array;
        struct B { d: array<u32> }
        @group(0) @binding(0) var<storage> bufs: binding_array<B>;
        @group(0) @binding(1) var textures: binding_array<texture_2d<f32>, 4>;
        @fragment fn main() -> @location(0) vec4<f32> {
            return vec4(f32(arrayLength(&bufs[0].d))) + textureLoad(textures[0], vec2(0), 0);
        }";
    let forward = rows(&[(0, READ_STORAGE), (1, texture(true))]);
    let mut backward = forward.clone();
    backward.resources.reverse();
    assert_eq!(
        agree(two_faults, forward),
        Err(RenderError::Unsupported(UNSIZED_BINDING_ARRAY))
    );
    assert_eq!(agree(two_faults, backward), mismatch);
}

/// Shaders are cross-checked in the order given (stage order for a pipeline): of two shaders
/// with different faults, the first's is reported, whichever it is.
#[test]
fn the_first_shader_that_disagrees_is_reported() {
    let rec = |stage, source, interface| {
        ShaderRec::new(shader_desc(stage, "main", wgsl(source), interface)).expect("accepted")
    };
    let vertex_ballot = rec(
        ShaderStage::Vertex,
        VERTEX_BALLOT,
        ShaderInterface::default(),
    );
    let lone_array = rec(
        ShaderStage::Fragment,
        BINDING_ARRAY,
        rows(&[(0, texture(true)), (1, SAMPLER)]),
    );
    assert_eq!(
        check_interfaces([&vertex_ballot, &lone_array]),
        Err(RenderError::Invalid(UNDECLARED_SUBGROUPS))
    );
    assert_eq!(
        check_interfaces([&lone_array, &vertex_ballot]),
        Err(RenderError::Invalid(ARRAY_COUNT_MISMATCH))
    );
}

/// Check 5 of the module docs' order on a device: after the ids (check 4) and before the
/// requirements (check 6), with two faults at once each way. The backend is told its device
/// cannot promise subgroups of 32, so the requirement fails on every device.
#[test]
fn the_interface_check_comes_after_the_ids_and_before_the_requirements() {
    bounded(|| {
        let Some(mut gpu) = gpu(
            "interface order",
            Arc::new(ThreadPool),
            WhileCompiling::Wait,
        ) else {
            return;
        };
        gpu.set_subgroup_support(SubgroupSupport::Range { min: 4, max: 64 });
        let bare = bare_vertex(&gpu);
        // Uses subgroups without declaring them: refused by the cross-check before the size.
        let undeclared = fragment(&gpu, SUBGROUP_FRAGMENT, storage_out());
        assert_eq!(
            gpu.create_graphics_pipeline(graphics_desc(bare, Some(undeclared))),
            Err(RenderError::Invalid(UNDECLARED_SUBGROUPS)),
            "the interface before the requirements"
        );
        assert_eq!(
            gpu.create_graphics_pipeline(graphics_desc(undeclared, Some(undeclared))),
            Err(RenderError::Invalid(
                "a pipeline's vertex shader is not a vertex shader"
            )),
            "the stages before the interface"
        );
        let declared = fragment(&gpu, SUBGROUP_FRAGMENT, subgroups(storage_out()));
        assert_eq!(
            gpu.create_graphics_pipeline(graphics_desc(bare, Some(declared))),
            Err(RenderError::Unsupported(SUBGROUP_SIZE)),
            "the declared requirement, refused for the size"
        );
        // Both faults: a declared requirement the device cannot meet, and an array count the
        // module does not have. The interface is reported.
        let both = fragment(
            &gpu,
            BINDING_ARRAY,
            subgroups(rows(&[(0, texture(true)), (1, SAMPLER)])),
        );
        assert_eq!(
            gpu.create_graphics_pipeline(graphics_desc(bare, Some(both))),
            Err(RenderError::Invalid(ARRAY_COUNT_MISMATCH)),
            "both faults: the interface first"
        );
        let cs = gpu
            .create_shader(shader_desc(
                ShaderStage::Compute,
                "main",
                wgsl(BALLOT),
                storage_out(),
            ))
            .expect("the shader itself is accepted");
        assert_eq!(
            gpu.create_compute_pipeline(ComputePipelineDesc { shader: cs }),
            Err(RenderError::Invalid(UNDECLARED_SUBGROUPS)),
            "a compute pipeline too"
        );
        let both = gpu
            .create_shader(shader_desc(
                ShaderStage::Compute,
                "main",
                wgsl(
                    "enable wgpu_binding_array;
                     @group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 4>;
                     @compute @workgroup_size(32) fn main() {
                         let b = subgroupBallot(true);
                         let d = textureDimensions(textures[0]);
                     }",
                ),
                subgroups(rows(&[(0, texture(true))])),
            ))
            .expect("the shader itself is accepted");
        assert_eq!(
            gpu.create_compute_pipeline(ComputePipelineDesc { shader: both }),
            Err(RenderError::Invalid(ARRAY_COUNT_MISMATCH)),
            "a compute pipeline with both faults: the interface first"
        );
    });
}

/// A shader's wgpu module is made once, by the first build that needs it, and kept: the second
/// call finds it although the naga module was handed over.
#[test]
fn a_shader_is_handed_to_wgpu_once() {
    bounded(|| {
        let Some(gpu) = gpu("module once", Arc::new(ThreadPool), WhileCompiling::Wait) else {
            return;
        };
        let rec = ShaderRec::new(shader_desc(
            ShaderStage::Vertex,
            "main",
            wgsl(VERTEX),
            ShaderInterface::default(),
        ))
        .expect("accepted");
        let first = rec.host(gpu.device()).expect("made").module.clone();
        assert!(
            rec.source.lock().expect("not poisoned").is_none(),
            "handed over"
        );
        let second = rec.host(gpu.device()).expect("kept").module.clone();
        assert_eq!(first, second, "the same wgpu module");
    });
}

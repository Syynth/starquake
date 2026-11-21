//! The shader cache behind the seam, with a scripted executor and a fake translator whose
//! shaders are real host shaders.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use starquake_memory::{PAGE_SIZE, PageRange};
use starquake_render::shader::ResourceKind;
use starquake_shader::asm::assemble_program;
use starquake_shader::cache::{
    CacheLookup, CodeForm, Executor, HostTarget, ShaderCache, ShaderInput, TranslateJob,
    Translated, Translator,
};
use starquake_shader::capture::Capture;
use starquake_shader::decode::{Instr, Op, OpExit, PredSrc};
use starquake_shader::exchange::{Answers, Question, Questioner, TextureKind};
use starquake_shader::host::{HostShader, Requirements};
use starquake_shader::lookup::{
    EngineState, GuestStage, ProgramMemory, ProgramRequest, Rejection, Unmapped, VariantKey,
};
use starquake_shader::sph::HEADER_BYTES;
use starquake_shader::translate::TranslateError;

use super::{CachedShaders, MAX_READY_SETS, SeamShader};
use crate::back::buffer::InvalidationSink;
use crate::back::draw::shaders::{
    ResourceSource, ResourceUse, SetLookup, ShaderLookup, Slot, bind_group_of,
};
use crate::back::draw::tests::host_shader;
use crate::back::pipeline::ShaderStamp;

/// The header byte that makes the fake translator reject a program.
const REJECTED_HEADER: u8 = 0xee;
/// The first byte of the fake guest memory: page-aligned.
const BASE: u64 = 0x10_0000;
/// Two programs on two pages.
const A: u64 = BASE;
const B: u64 = BASE + PAGE_SIZE;

/// A program the fake makes into one constant buffer of `bank` 1.
fn resources(stage: GuestStage) -> Vec<ResourceUse> {
    vec![ResourceUse {
        slot: Slot {
            group: bind_group_of(stage).unwrap_or(0),
            binding: 3,
        },
        kind: ResourceKind::UniformBuffer,
        count: None,
        source: ResourceSource::ConstantBuffer {
            bank: 1,
            size: 0x40,
        },
    }]
}

struct Program;

impl Questioner for Program {
    fn questions(&self, _answers: &Answers) -> Vec<Question> {
        Vec::new()
    }
}

/// A real host shader and the resources the fake gives it.
#[derive(Debug)]
struct Seam {
    host: HostShader,
    resources: Vec<ResourceUse>,
}

impl Translated for Seam {
    fn requirements(&self) -> Requirements {
        self.host.requirements
    }
}

impl SeamShader for Seam {
    fn host(&self) -> &HostShader {
        &self.host
    }

    fn resources(&self) -> Vec<ResourceUse> {
        self.resources.clone()
    }
}

/// Rejects programs whose header starts with [`REJECTED_HEADER`]; translates the rest to the
/// exit-only shader of their stage.
struct Fake;

impl Translator for Fake {
    type Program = Program;
    type Shader = Seam;

    fn program(&self, capture: &Capture) -> Result<Program, Rejection> {
        if capture.header().first() == Some(&REJECTED_HEADER) {
            return Err(Rejection::Translate(TranslateError::TooManyInstructions));
        }
        Ok(Program)
    }

    fn shader(&self, input: &ShaderInput<'_, Program>) -> Result<Seam, Rejection> {
        Ok(Seam {
            host: host_shader(input.stage),
            resources: resources(input.stage),
        })
    }
}

#[derive(Default)]
struct Scripted(Mutex<VecDeque<TranslateJob>>);

impl Scripted {
    fn run_all(&self) {
        loop {
            let job = self
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .pop_front();
            let Some(job) = job else { break };
            job();
        }
    }
}

impl Executor for Scripted {
    fn spawn(&self, job: TranslateJob) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push_back(job);
    }
}

/// Guest memory: two pages at [`BASE`].
struct Memory(Vec<u8>);

impl Memory {
    fn new() -> Self {
        Self(vec![0; usize::try_from(2 * PAGE_SIZE).expect("small")])
    }

    fn place(&mut self, address: u64, header: u8) {
        let mut bytes = vec![header; HEADER_BYTES];
        bytes.extend(
            assemble_program(&[Instr::new(Op::Exit(OpExit {
                keep_ref_count: false,
                no_at_exit: false,
                pred: PredSrc::TRUE,
            }))])
            .expect("encodable"),
        );
        let at = usize::try_from(address - BASE).expect("mapped");
        self.0[at..at + bytes.len()].copy_from_slice(&bytes);
    }
}

impl ProgramMemory for Memory {
    fn read(&self, address: u64, out: &mut [u8]) -> Result<(), Unmapped> {
        let at = address
            .checked_sub(BASE)
            .and_then(|at| usize::try_from(at).ok())
            .ok_or(Unmapped { address })?;
        let bytes = at
            .checked_add(out.len())
            .and_then(|end| self.0.get(at..end))
            .ok_or(Unmapped { address })?;
        out.copy_from_slice(bytes);
        Ok(())
    }
}

struct NoState;

impl EngineState for NoState {
    fn cbuf_word(&self, _stage: GuestStage, _bank: u8, _offset: u32) -> Option<u32> {
        None
    }

    fn texture_kind(&self, _stage: GuestStage, _handle: u32) -> Option<TextureKind> {
        None
    }
}

const TARGET: HostTarget = HostTarget {
    form: CodeForm::NagaModule,
    caps: Requirements {
        subgroup: true,
        f16: false,
        f64: false,
        i64: false,
        i64_atomics: false,
    },
};

struct Rig {
    seam: CachedShaders<Fake>,
    executor: Arc<Scripted>,
    memory: Memory,
}

impl Rig {
    fn new(limit: usize) -> Self {
        let executor = Arc::new(Scripted::default());
        let cache = ShaderCache::new(Arc::new(Fake), Arc::<Scripted>::clone(&executor), TARGET);
        let mut memory = Memory::new();
        memory.place(A, 1);
        memory.place(B, 2);
        Self {
            seam: CachedShaders::with_limit(cache, limit),
            executor,
            memory,
        }
    }

    fn lookup(&mut self, requests: &[ProgramRequest]) -> SetLookup {
        self.seam
            .lookup(requests, &VariantKey::default(), &self.memory, &NoState)
    }

    /// The cache's own answer, for its stamps.
    fn cache_stamps(&mut self, requests: &[ProgramRequest]) -> Vec<ShaderStamp> {
        match self
            .seam
            .cache
            .lookup(requests, &VariantKey::default(), &self.memory, &NoState)
        {
            CacheLookup::Ready(shaders) => shaders
                .iter()
                .map(|shader| ShaderStamp::new(shader.stamp.value()))
                .collect(),
            other => unreachable!("expected ready, got {other:?}"),
        }
    }
}

fn vertex(address: u64) -> ProgramRequest {
    ProgramRequest {
        stage: GuestStage::Vertex,
        address,
    }
}

fn pixel(address: u64) -> ProgramRequest {
    ProgramRequest {
        stage: GuestStage::Pixel,
        address,
    }
}

#[test]
fn a_ready_set_carries_the_caches_stamps_and_the_shaders_resources() {
    let mut rig = Rig::new(MAX_READY_SETS);
    let requests = [vertex(A), pixel(B)];
    let SetLookup::Pending(ticket) = rig.lookup(&requests) else {
        unreachable!("a first lookup is pending");
    };
    rig.executor.run_all();
    let SetLookup::Ready(set) = rig.seam.wait(ticket) else {
        unreachable!("the set is ready once its jobs ran");
    };
    assert_eq!(set.shaders.stages.len(), 2);
    assert_eq!(set.stages.len(), 2);
    let stamps = rig.cache_stamps(&requests);
    assert_ne!(stamps[0], stamps[1]);
    assert_eq!(
        set.stages
            .iter()
            .map(|stage| stage.stamp)
            .collect::<Vec<_>>(),
        stamps
    );
    assert_eq!(set.stages[0].resources, resources(GuestStage::Vertex));
    assert_eq!(set.stages[1].resources, resources(GuestStage::Pixel));
    assert_eq!(
        set.shaders.stages[0].entry,
        host_shader(GuestStage::Vertex).entry
    );
    assert_eq!(
        set.shaders.stages[1].entry,
        host_shader(GuestStage::Pixel).entry
    );
    // Found again: the kept conversion, no module cloned.
    let SetLookup::Ready(again) = rig.lookup(&requests) else {
        unreachable!("still ready");
    };
    assert!(Arc::ptr_eq(&again.shaders, &set.shaders));
    assert_eq!(rig.seam.sets.len(), 1);
}

#[test]
fn a_rejection_passes_its_reason_through() {
    let mut rig = Rig::new(MAX_READY_SETS);
    rig.memory.place(B, REJECTED_HEADER);
    let SetLookup::Rejected(reason) = rig.lookup(&[vertex(A), pixel(B)]) else {
        unreachable!("the pixel program is rejected");
    };
    assert_eq!(
        *reason,
        Rejection::Translate(TranslateError::TooManyInstructions)
    );
}

#[test]
fn an_unknown_ticket_waits_to_pending() {
    let mut rig = Rig::new(MAX_READY_SETS);
    let ticket = starquake_shader::lookup::Ticket::new(42);
    assert!(matches!(rig.seam.wait(ticket), SetLookup::Pending(t) if t == ticket));
}

#[test]
fn an_invalidated_page_drops_only_the_programs_on_it() {
    let mut rig = Rig::new(MAX_READY_SETS);
    rig.lookup(&[vertex(A), pixel(B)]);
    assert_eq!(rig.seam.cache().counters().captures, 2);
    // B's page: A stays, B is captured again.
    rig.seam
        .invalidate(PageRange::new(B, PAGE_SIZE).expect("a page"));
    rig.lookup(&[vertex(A), pixel(B)]);
    assert_eq!(rig.seam.cache().counters().captures, 3);
    // The page before A's touches nothing.
    rig.seam
        .invalidate(PageRange::new(A - PAGE_SIZE, PAGE_SIZE).expect("a page"));
    rig.lookup(&[vertex(A), pixel(B)]);
    assert_eq!(rig.seam.cache().counters().captures, 3);
    assert_eq!(rig.seam.cache().tracked_ranges().len(), 2);
}

/// How many bytes the cache reads for the program [`Memory::place`] writes.
fn program_extent() -> u64 {
    let mut rig = Rig::new(MAX_READY_SETS);
    rig.lookup(&[vertex(A)]);
    let ranges = rig.seam.cache().tracked_ranges();
    assert_eq!(ranges.len(), 1);
    ranges[0].end - ranges[0].start
}

/// A rig whose memory holds one vertex program at `address`, looked up once.
fn rig_with_program_at(address: u64) -> Rig {
    let mut rig = Rig::new(MAX_READY_SETS);
    rig.memory = Memory::new();
    rig.memory.place(address, 1);
    rig.lookup(&[vertex(address)]);
    assert_eq!(rig.seam.cache().counters().captures, 1);
    rig
}

#[test]
fn a_program_starting_on_a_pages_last_byte_is_dropped_with_that_page() {
    let address = B - 1;
    let mut rig = rig_with_program_at(address);
    // The page after the one it starts on, which it also reads: dropped too.
    rig.seam
        .invalidate(PageRange::new(B, PAGE_SIZE).expect("a page"));
    rig.lookup(&[vertex(address)]);
    assert_eq!(rig.seam.cache().counters().captures, 2);
    // The page it starts on, by its last byte only.
    rig.seam
        .invalidate(PageRange::new(A, PAGE_SIZE).expect("a page"));
    rig.lookup(&[vertex(address)]);
    assert_eq!(rig.seam.cache().counters().captures, 3);
}

#[test]
fn a_program_ending_one_byte_into_a_page_is_dropped_with_that_page() {
    let extent = program_extent();
    let address = B + 1 - extent;
    let mut rig = rig_with_program_at(address);
    let tracked = rig.seam.cache().tracked_ranges();
    assert_eq!(tracked.len(), 1);
    assert_eq!(
        (tracked[0].start, tracked[0].end),
        (address, B + 1),
        "the program's last byte is the page's first"
    );
    // The page after B's touches nothing.
    rig.seam
        .invalidate(PageRange::new(B + PAGE_SIZE, PAGE_SIZE).expect("a page"));
    rig.lookup(&[vertex(address)]);
    assert_eq!(rig.seam.cache().counters().captures, 1);
    // B's page, by its first byte only.
    rig.seam
        .invalidate(PageRange::new(B, PAGE_SIZE).expect("a page"));
    rig.lookup(&[vertex(address)]);
    assert_eq!(rig.seam.cache().counters().captures, 2);
}

#[test]
fn the_kept_sets_empty_at_their_cap() {
    for (limit, kept) in [(2, 2), (1, 1)] {
        let mut rig = Rig::new(limit);
        rig.lookup(&[vertex(A)]);
        rig.lookup(&[pixel(B)]);
        rig.executor.run_all();
        let SetLookup::Ready(first) = rig.lookup(&[vertex(A)]) else {
            unreachable!("ready");
        };
        let SetLookup::Ready(_) = rig.lookup(&[pixel(B)]) else {
            unreachable!("ready");
        };
        assert_eq!(rig.seam.sets.len(), kept, "limit {limit}");
        let SetLookup::Ready(again) = rig.lookup(&[vertex(A)]) else {
            unreachable!("ready");
        };
        // Kept under the larger cap; converted afresh, the same stamps, under the smaller.
        assert_eq!(Arc::ptr_eq(&again.shaders, &first.shaders), kept == 2);
        assert_eq!(again.stages, first.stages);
    }
}

#[test]
fn the_seam_is_the_executors_shader_cache() {
    let rig = Rig::new(MAX_READY_SETS);
    let boxed: Box<dyn crate::back::exec::ShaderCache + Send> = Box::new(rig.seam);
    drop(boxed);
}

#[test]
fn the_kept_set_cap_is_the_documented_guess() {
    assert_eq!(MAX_READY_SETS, 4_096);
}

#[test]
fn a_wait_through_the_seam_returns_once_a_worker_runs_the_job() {
    let mut rig = Rig::new(MAX_READY_SETS);
    let SetLookup::Pending(ticket) = rig.lookup(&[vertex(A)]) else {
        unreachable!("a first lookup is pending");
    };
    let executor = Arc::clone(&rig.executor);
    let mut seam = rig.seam;
    let (sender, receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let _sent = sender.send(matches!(seam.wait(ticket), SetLookup::Ready(_)));
    });
    std::thread::sleep(Duration::from_millis(20));
    executor.run_all();
    // Bounded: the waiter answers within the watchdog, or the test fails.
    assert_eq!(receiver.recv_timeout(Duration::from_secs(10)), Ok(true));
    waiter.join().expect("the waiter ends");
}

/// The production translator's shaders go through the seam: a real vertex program (a parsed
/// header and an `EXIT`) translated by `PipelineTranslator` comes back as a ready set of its
/// host shader, which binds no resources until slice 17 (module docs, "The production
/// translator's shaders").
#[test]
fn a_host_shader_binds_nothing_yet() {
    use starquake_shader::pipeline::PipelineTranslator;
    use starquake_shader::sph::encode::encode_bytes;
    use starquake_shader::sph::{Common, Io, ProgramHeader, VtgIo, VtgMap, VtgStage};

    let executor = Arc::new(Scripted::default());
    let cache = ShaderCache::new(
        Arc::new(PipelineTranslator),
        Arc::<Scripted>::clone(&executor),
        TARGET,
    );
    let mut seam = CachedShaders::new(cache);
    let header = ProgramHeader {
        common: Common::default(),
        io: Io::Vtg(Box::new(VtgIo {
            stage: VtgStage::Vertex,
            input: VtgMap::default(),
            output: VtgMap::default(),
            vmap: VtgMap::default(),
        })),
    };
    let mut memory = Memory::new();
    let mut bytes = encode_bytes(&header).expect("encodable").to_vec();
    bytes.extend(
        assemble_program(&[Instr::new(Op::Exit(OpExit {
            keep_ref_count: false,
            no_at_exit: false,
            pred: PredSrc::TRUE,
        }))])
        .expect("encodable"),
    );
    memory.0[..bytes.len()].copy_from_slice(&bytes);
    let requests = [vertex(A)];
    let SetLookup::Pending(ticket) =
        seam.lookup(&requests, &VariantKey::default(), &memory, &NoState)
    else {
        unreachable!("a first lookup is pending");
    };
    executor.run_all();
    let SetLookup::Ready(set) = seam.wait(ticket) else {
        unreachable!("the translation is done");
    };
    assert_eq!(set.stages.len(), 1);
    assert_eq!(set.stages[0].resources, Vec::new());
    assert_eq!(
        set.shaders.stages[0].stage,
        starquake_shader::host::HostStage::Vertex
    );
    let shader = host_shader(GuestStage::Vertex);
    assert!(SeamShader::resources(&shader).is_empty());
    assert_eq!(SeamShader::host(&shader).stage, shader.stage);
}

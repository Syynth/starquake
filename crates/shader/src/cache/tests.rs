//! The shader cache with a scripted executor and a fake translator: miss, pending, ready,
//! coalescing, rejection, the negative cache, invalidation edges, the limits, and races on a
//! fixed schedule.

mod invalidate;
mod limits;
mod races;
mod rejections;

use std::collections::{HashMap, HashSet, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::thread;
use std::time::Duration;

use super::{
    CacheCounters, CacheLookup, CacheRefusal, CodeForm, Executor, HostTarget, Limits,
    MAX_ADDRESSES, MAX_EXTENT, MAX_PROGRAM_BYTES, MAX_PROGRAMS, MAX_SHADER_BYTES, MAX_SHADERS,
    MAX_TICKETS, ProgramRejection, ShaderCache, ShaderInput, ShaderStamp, TRANSLATOR_VERSION,
    TranslateJob, Translated, Translator,
};
use crate::capture::{Capture, ContentHash, capture, header_bytes};
use crate::decode::{COND, EXIT_OPCODE, Field, GUARD, GUARD_NOT, Instruction, OPCODE, Pred};
use crate::exchange::{Answers, EngineState, Question, Questioner, TextureKind};
use crate::host::Requirements;
use crate::lookup::{ProgramMemory, ProgramRequest, Rejection, Ticket, Unmapped};
use crate::translate::{GuestStage, TranslateError};
use crate::variant::VariantKey;

// ---------------------------------------------------------------------------------------------
// Program words.

/// An opcode the capture does not read (as in the capture's tests).
const FILLER_OPCODE: u16 = 0x202;
/// Bits of a filler word the capture ignores, where a test puts a tag to make programs differ.
const TAG: Field = Field::new(32, 32);
/// The guard `PT`.
const PT: u8 = Pred::PT.index();

fn guarded(opcode: u16) -> Instruction {
    Instruction::default()
        .with_field(OPCODE, u64::from(opcode))
        .with_field(GUARD, u64::from(PT & 7))
        .with_flag(GUARD_NOT, false)
}

/// A word the capture reads past, tagged `tag`.
fn filler(tag: u32) -> Instruction {
    guarded(FILLER_OPCODE).with_field(TAG, u64::from(tag))
}

/// An `EXIT` that ends the program.
fn exit() -> Instruction {
    guarded(EXIT_OPCODE).with_field(COND, u64::from(PT))
}

/// A program of `stage`: its header (bytes `header_tag`), a filler per tag, then `EXIT`.
fn program_with_header(stage: GuestStage, header_tag: u8, tags: &[u32]) -> Vec<u8> {
    let mut bytes = vec![header_tag; header_bytes(stage)];
    for &tag in tags {
        bytes.extend_from_slice(&filler(tag).to_le_bytes());
    }
    bytes.extend_from_slice(&exit().to_le_bytes());
    bytes
}

/// A program of `stage` with a zero header.
fn program(stage: GuestStage, tags: &[u32]) -> Vec<u8> {
    program_with_header(stage, 0, tags)
}

/// The content hash of `bytes` captured as `stage`.
fn hash_of(stage: GuestStage, bytes: &[u8]) -> ContentHash {
    capture(bytes, stage)
        .expect("a test program captures")
        .hash()
}

// ---------------------------------------------------------------------------------------------
// Guest memory and engine state.

/// The first byte of the fake guest memory.
const BASE: u64 = 0x1_0000_0000;
/// The default size of the fake guest memory.
const MEMORY_SIZE: usize = 0x1_0000;

/// One mapped run of guest memory at [`BASE`]; everything else is unmapped.
struct Memory {
    base: u64,
    bytes: Vec<u8>,
}

impl Memory {
    fn new(size: usize) -> Self {
        Self {
            base: BASE,
            bytes: vec![0; size],
        }
    }

    /// The first address past the mapped run.
    fn end(&self) -> u64 {
        self.base + self.bytes.len() as u64
    }

    /// Writes `bytes` at `address`, which must be mapped.
    fn write(&mut self, address: u64, bytes: &[u8]) {
        let at = usize::try_from(address - self.base).expect("mapped");
        self.bytes[at..at + bytes.len()].copy_from_slice(bytes);
    }
}

impl ProgramMemory for Memory {
    fn read(&self, address: u64, out: &mut [u8]) -> Result<(), Unmapped> {
        if address < self.base {
            return Err(Unmapped { address });
        }
        let len = out.len() as u64;
        if address.checked_add(len).is_none_or(|end| end > self.end()) {
            return Err(Unmapped {
                address: address.max(self.end()),
            });
        }
        let at = usize::try_from(address - self.base).expect("mapped");
        out.copy_from_slice(&self.bytes[at..at + out.len()]);
        Ok(())
    }
}

/// Constant-buffer words by bank and offset; no textures.
#[derive(Default)]
struct State {
    words: HashMap<(u8, u32), u32>,
    asked: Mutex<Vec<(GuestStage, u8, u32)>>,
}

impl EngineState for State {
    fn cbuf_word(&self, stage: GuestStage, bank: u8, offset: u32) -> Option<u32> {
        self.asked
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((stage, bank, offset));
        self.words.get(&(bank, offset)).copied()
    }

    fn texture_kind(&self, _stage: GuestStage, _handle: u32) -> Option<TextureKind> {
        None
    }
}

// ---------------------------------------------------------------------------------------------
// The fake translator.

/// What the fake translator does with a program, by content hash.
#[derive(Default)]
struct Script {
    /// [`Translator::program`] rejects these.
    program_rejections: HashMap<ContentHash, Rejection>,
    /// These programs ask these questions.
    questions: HashMap<ContentHash, Vec<Question>>,
    /// These programs ask a new question after every answer, for ever.
    endless: HashSet<ContentHash>,
    /// [`Translator::shader`] rejects these.
    shader_rejections: HashMap<ContentHash, Rejection>,
    /// [`Translator::shader`] panics on these.
    panics: HashSet<ContentHash>,
    /// Shaders of these programs need these features.
    needs: HashMap<ContentHash, Requirements>,
}

/// The fake translator: scripted by content hash, counting its calls.
#[derive(Default)]
struct Fake {
    script: Mutex<Script>,
    programs: AtomicU64,
    shaders: AtomicU64,
}

impl Fake {
    fn script(&self) -> std::sync::MutexGuard<'_, Script> {
        self.script.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn program_calls(&self) -> u64 {
        self.programs.load(Ordering::SeqCst)
    }

    fn shader_calls(&self) -> u64 {
        self.shaders.load(Ordering::SeqCst)
    }
}

/// A fake program entry.
#[derive(Debug)]
struct FakeProgram {
    hash: ContentHash,
    /// The stage of the capture it was made from.
    captured_as: GuestStage,
    questions: Vec<Question>,
    endless: bool,
}

impl Questioner for FakeProgram {
    fn questions(&self, answers: &Answers) -> Vec<Question> {
        let mut questions = self.questions.clone();
        if self.endless {
            let next = u32::try_from(answers.len()).expect("few answers") * 4;
            questions.push(Question::CbufWord {
                bank: 7,
                offset: next,
            });
        }
        questions
    }
}

/// A fake translated shader: what it was translated from.
#[derive(Debug, PartialEq, Eq)]
struct FakeShader {
    hash: ContentHash,
    stage: GuestStage,
    answers: Answers,
    variant: VariantKey,
    target: HostTarget,
    needs: Requirements,
}

impl Translated for FakeShader {
    fn requirements(&self) -> Requirements {
        self.needs
    }
}

impl Translator for Fake {
    type Program = FakeProgram;
    type Shader = FakeShader;

    fn program(&self, capture: &Capture) -> Result<FakeProgram, Rejection> {
        self.programs.fetch_add(1, Ordering::SeqCst);
        let hash = capture.hash();
        let script = self.script();
        if let Some(rejection) = script.program_rejections.get(&hash) {
            return Err(rejection.clone());
        }
        Ok(FakeProgram {
            hash,
            captured_as: capture.stage(),
            questions: script.questions.get(&hash).cloned().unwrap_or_default(),
            endless: script.endless.contains(&hash),
        })
    }

    fn shader(&self, input: &ShaderInput<'_, FakeProgram>) -> Result<FakeShader, Rejection> {
        self.shaders.fetch_add(1, Ordering::SeqCst);
        let hash = input.program.hash;
        let (rejection, panics, needs) = {
            let script = self.script();
            (
                script.shader_rejections.get(&hash).cloned(),
                script.panics.contains(&hash),
                script.needs.get(&hash).copied().unwrap_or_default(),
            )
        };
        assert!(!panics, "the scripted translator panics on this program");
        if let Some(rejection) = rejection {
            return Err(rejection);
        }
        Ok(FakeShader {
            hash,
            stage: input.stage,
            answers: input.answers.clone(),
            variant: input.variant.clone(),
            target: *input.target,
            needs,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Executors.

/// Holds jobs until the test runs or drops them: the fixed schedule of every test.
#[derive(Default)]
struct Scripted {
    jobs: Mutex<VecDeque<TranslateJob>>,
}

impl Scripted {
    fn jobs(&self) -> std::sync::MutexGuard<'_, VecDeque<TranslateJob>> {
        self.jobs.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn queued(&self) -> usize {
        self.jobs().len()
    }

    /// Runs the oldest job; a panic in it is caught, as a worker thread would end.
    fn run_next(&self) -> bool {
        let job = self.jobs().pop_front();
        job.is_some_and(|job| {
            let _caught = catch_unwind(AssertUnwindSafe(job));
            true
        })
    }

    /// Runs the newest job.
    fn run_last(&self) -> bool {
        let job = self.jobs().pop_back();
        job.is_some_and(|job| {
            let _caught = catch_unwind(AssertUnwindSafe(job));
            true
        })
    }

    fn run_all(&self) {
        while self.run_next() {}
    }

    /// Drops every job unrun, as a pool shutting down does.
    fn drop_all(&self) {
        let jobs: Vec<TranslateJob> = self.jobs().drain(..).collect();
        drop(jobs);
    }
}

impl Executor for Scripted {
    fn spawn(&self, job: TranslateJob) {
        self.jobs().push_back(job);
    }
}

/// Runs every job on a thread of its own: a real pool's shape.
struct Threads;

impl Executor for Threads {
    fn spawn(&self, job: TranslateJob) {
        thread::spawn(job);
    }
}

// ---------------------------------------------------------------------------------------------
// The rig.

/// A host target with subgroup operations.
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

/// How long a test lets a wait run before calling it hung.
const WATCHDOG: Duration = Duration::from_secs(10);

/// A vertex program's request at `address`.
fn vertex(address: u64) -> ProgramRequest {
    ProgramRequest {
        stage: GuestStage::Vertex,
        address,
    }
}

/// A pixel program's request at `address`.
fn pixel(address: u64) -> ProgramRequest {
    ProgramRequest {
        stage: GuestStage::Pixel,
        address,
    }
}

/// The addresses tests put programs at: distinct, and away from the run's ends.
const A: u64 = BASE + 0x1000;
const B: u64 = BASE + 0x3000;
const C: u64 = BASE + 0x5000;

struct Rig {
    cache: ShaderCache<Fake>,
    fake: Arc<Fake>,
    executor: Arc<Scripted>,
    memory: Memory,
    state: State,
    variant: VariantKey,
}

impl Rig {
    fn new() -> Self {
        Self::with_limits(Limits::SHIPPING)
    }

    fn with_limits(limits: Limits) -> Self {
        Self::with_memory(limits, Memory::new(MEMORY_SIZE))
    }

    fn with_memory(limits: Limits, memory: Memory) -> Self {
        let fake = Arc::new(Fake::default());
        let executor = Arc::new(Scripted::default());
        let cache = ShaderCache::with_limits(
            Arc::clone(&fake),
            Arc::<Scripted>::clone(&executor),
            TARGET,
            limits,
        );
        Self {
            cache,
            fake,
            executor,
            memory,
            state: State::default(),
            variant: VariantKey::default(),
        }
    }

    /// Writes `bytes` at `address` and returns their hash as `stage` captures them.
    fn place(&mut self, address: u64, stage: GuestStage, bytes: &[u8]) -> ContentHash {
        self.memory.write(address, bytes);
        hash_of(stage, bytes)
    }

    fn lookup(&mut self, requests: &[ProgramRequest]) -> CacheLookup<FakeShader> {
        self.cache
            .lookup(requests, &self.variant, &self.memory, &self.state)
    }

    fn counters(&self) -> CacheCounters {
        self.cache.counters()
    }
}

/// The shaders of a ready answer.
#[track_caller]
fn ready(lookup: CacheLookup<FakeShader>) -> Vec<(ShaderStamp, Arc<FakeShader>)> {
    match lookup {
        CacheLookup::Ready(shaders) => shaders
            .into_iter()
            .map(|shader| (shader.stamp, shader.shader))
            .collect(),
        other => unreachable!("expected ready, got {other:?}"),
    }
}

/// The ticket of a pending answer.
#[track_caller]
fn pending(lookup: CacheLookup<FakeShader>) -> Ticket {
    match lookup {
        CacheLookup::Pending(ticket) => ticket,
        other => unreachable!("expected pending, got {other:?}"),
    }
}

/// The rejection of a rejected answer.
#[track_caller]
fn rejected(lookup: CacheLookup<FakeShader>) -> Arc<ProgramRejection> {
    match lookup {
        CacheLookup::Rejected(rejection) => rejection,
        other => unreachable!("expected rejected, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Miss, pending, ready.

#[test]
fn a_miss_is_pending_until_its_translation_runs_then_ready() {
    let mut rig = Rig::new();
    let bytes = program(GuestStage::Vertex, &[1]);
    let hash = rig.place(A, GuestStage::Vertex, &bytes);

    let ticket = pending(rig.lookup(&[vertex(A)]));
    // The lookup queued the translation and did not run it.
    assert_eq!(rig.executor.queued(), 1);
    assert_eq!(rig.fake.shader_calls(), 0);
    assert_eq!(rig.fake.program_calls(), 1);
    // Still pending before the job runs.
    assert_eq!(pending(rig.lookup(&[vertex(A)])), ticket);

    assert!(rig.executor.run_next());
    let shaders = ready(rig.lookup(&[vertex(A)]));
    assert_eq!(shaders.len(), 1);
    let (stamp, shader) = &shaders[0];
    assert_eq!(
        **shader,
        FakeShader {
            hash,
            stage: GuestStage::Vertex,
            answers: Answers::new(),
            variant: VariantKey::default(),
            target: TARGET,
            needs: Requirements::default(),
        }
    );
    // Found again: the same entry, no new work.
    let again = ready(rig.lookup(&[vertex(A)]));
    assert_eq!(again[0].0, *stamp);
    assert!(Arc::ptr_eq(&again[0].1, shader));
    assert_eq!(
        rig.counters(),
        CacheCounters {
            captures: 1,
            programs: 1,
            translations: 1,
            ..CacheCounters::default()
        }
    );
    assert_eq!(rig.fake.shader_calls(), 1);
}

#[test]
fn a_set_is_ready_in_request_order_with_a_stamp_per_shader() {
    let mut rig = Rig::new();
    let vertex_hash = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let pixel_hash = rig.place(B, GuestStage::Pixel, &program(GuestStage::Pixel, &[2]));
    let ticket = pending(rig.lookup(&[vertex(A), pixel(B)]));
    assert_eq!(rig.executor.queued(), 2);
    // One of the two finished: still pending, on the same ticket.
    assert!(rig.executor.run_next());
    assert_eq!(pending(rig.lookup(&[vertex(A), pixel(B)])), ticket);
    assert!(rig.executor.run_next());
    let shaders = ready(rig.lookup(&[vertex(A), pixel(B)]));
    assert_eq!(
        shaders
            .iter()
            .map(|(_, shader)| (shader.hash, shader.stage))
            .collect::<Vec<_>>(),
        [
            (vertex_hash, GuestStage::Vertex),
            (pixel_hash, GuestStage::Pixel)
        ]
    );
    assert_ne!(shaders[0].0, shaders[1].0);
    // The other order gives the other order.
    let swapped = ready(rig.lookup(&[pixel(B), vertex(A)]));
    assert_eq!(swapped[0].0, shaders[1].0);
    assert_eq!(swapped[1].0, shaders[0].0);
}

#[test]
fn no_requests_are_ready_with_no_shaders() {
    let mut rig = Rig::new();
    assert_eq!(ready(rig.lookup(&[])).len(), 0);
    assert_eq!(rig.counters(), CacheCounters::default());
    assert!(rig.cache.tickets.is_empty());
}

#[test]
fn a_fresh_cache_holds_nothing_and_uses_the_shipping_limits() {
    let fake = Arc::new(Fake::default());
    let cache = ShaderCache::new(fake, Arc::new(Scripted::default()), TARGET);
    assert_eq!(cache.limits, Limits::SHIPPING);
    assert_eq!(cache.target(), TARGET);
    assert_eq!(cache.counters(), CacheCounters::default());
    assert_eq!(cache.tracked_ranges().len(), 0);
    assert_eq!(cache.next_ticket, 1);
}

#[test]
fn the_shipping_constants_are_the_documented_values() {
    // Guesses, written out so a changed value is seen (module docs, "Limits").
    assert_eq!(MAX_ADDRESSES, 65_536);
    assert_eq!(MAX_PROGRAMS, 65_536);
    assert_eq!(MAX_PROGRAM_BYTES, 268_435_456);
    assert_eq!(MAX_SHADERS, 65_536);
    assert_eq!(MAX_SHADER_BYTES, 268_435_456);
    assert_eq!(MAX_TICKETS, 4_096);
    assert_eq!(TRANSLATOR_VERSION, 1);
    // 128 bytes of header and 65,536 instructions of 16 bytes (`program_capture.h`,
    // `MaximumInstructions`; PR #4588's 128-byte header skip).
    assert_eq!(MAX_EXTENT, 1_048_704);
    assert_eq!(
        Limits::SHIPPING,
        Limits {
            addresses: MAX_ADDRESSES,
            programs: MAX_PROGRAMS,
            program_bytes: MAX_PROGRAM_BYTES,
            shaders: MAX_SHADERS,
            shader_bytes: MAX_SHADER_BYTES,
            tickets: MAX_TICKETS,
        }
    );
}

// ---------------------------------------------------------------------------------------------
// Coalescing.

#[test]
fn lookups_of_a_pending_set_coalesce_on_one_ticket_and_one_job() {
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let first = pending(rig.lookup(&[vertex(A)]));
    let second = pending(rig.lookup(&[vertex(A)]));
    assert_eq!(first, second);
    assert_eq!(rig.executor.queued(), 1);
    assert_eq!(rig.counters().translations, 1);
    assert_eq!(rig.counters().captures, 1);
}

#[test]
fn two_sets_sharing_a_shader_queue_it_once_and_get_their_own_tickets() {
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    rig.place(B, GuestStage::Pixel, &program(GuestStage::Pixel, &[2]));
    rig.place(C, GuestStage::Pixel, &program(GuestStage::Pixel, &[3]));
    let one = pending(rig.lookup(&[vertex(A), pixel(B)]));
    let two = pending(rig.lookup(&[vertex(A), pixel(C)]));
    assert_ne!(one, two);
    // The vertex shader once, each pixel shader once.
    assert_eq!(rig.executor.queued(), 3);
    assert_eq!(rig.counters().translations, 3);
}

#[test]
fn the_same_bytes_at_two_addresses_share_the_program_and_the_shader() {
    let mut rig = Rig::new();
    let bytes = program(GuestStage::Vertex, &[1]);
    rig.place(A, GuestStage::Vertex, &bytes);
    rig.place(B, GuestStage::Vertex, &bytes);
    let at_a = pending(rig.lookup(&[vertex(A)]));
    let at_b = pending(rig.lookup(&[vertex(B)]));
    // Two requests, two sets: two tickets, but one program entry and one translation.
    assert_ne!(at_a, at_b);
    assert_eq!(rig.counters().captures, 2);
    assert_eq!(rig.counters().programs, 1);
    assert_eq!(rig.counters().translations, 1);
    rig.executor.run_all();
    let a = ready(rig.lookup(&[vertex(A)]));
    let b = ready(rig.lookup(&[vertex(B)]));
    assert_eq!(a[0].0, b[0].0);
    assert!(Arc::ptr_eq(&a[0].1, &b[0].1));
}

#[test]
fn one_program_as_two_stages_shares_the_program_entry_not_the_shader() {
    let mut rig = Rig::new();
    // Both graphics stages have a header, so the bytes and the hash are the same.
    let bytes = program(GuestStage::Vertex, &[1]);
    let hash = rig.place(A, GuestStage::Vertex, &bytes);
    assert_eq!(hash, hash_of(GuestStage::Pixel, &bytes));
    pending(rig.lookup(&[vertex(A)]));
    pending(rig.lookup(&[pixel(A)]));
    assert_eq!(rig.counters().programs, 1);
    assert_eq!(rig.counters().translations, 2);
    rig.executor.run_all();
    let as_vertex = ready(rig.lookup(&[vertex(A)]));
    let as_pixel = ready(rig.lookup(&[pixel(A)]));
    assert_ne!(as_vertex[0].0, as_pixel[0].0);
    assert_eq!(as_vertex[0].1.stage, GuestStage::Vertex);
    assert_eq!(as_pixel[0].1.stage, GuestStage::Pixel);
}

#[test]
fn a_program_entry_is_made_from_the_first_capture_of_its_bytes() {
    let mut rig = Rig::new();
    let bytes = program(GuestStage::Vertex, &[1]);
    rig.place(A, GuestStage::Vertex, &bytes);
    pending(rig.lookup(&[pixel(A)]));
    pending(rig.lookup(&[vertex(A)]));
    let hash = hash_of(GuestStage::Pixel, &bytes);
    let entry = rig.cache.programs.get(&hash).expect("one program entry");
    let super::ProgramSlot::Ready { program, .. } = &entry.slot else {
        unreachable!("the program is not rejected");
    };
    assert_eq!(program.captured_as, GuestStage::Pixel);
}

// ---------------------------------------------------------------------------------------------
// What the key holds.

#[test]
fn a_changed_answer_or_variant_is_another_shader_and_an_unchanged_one_is_not() {
    let mut rig = Rig::new();
    let hash = rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let question = Question::CbufWord { bank: 3, offset: 8 };
    rig.fake.script().questions.insert(hash, vec![question]);
    rig.state.words.insert((3, 8), 0x1234);
    pending(rig.lookup(&[vertex(A)]));
    rig.executor.run_all();
    let first = ready(rig.lookup(&[vertex(A)]));
    assert_eq!(first[0].1.answers.cbuf_word(3, 8), Some(Some(0x1234)));
    // The question is answered for the request's stage.
    assert!(
        rig.state
            .asked
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .all(|&(stage, bank, offset)| (stage, bank, offset) == (GuestStage::Vertex, 3, 8))
    );

    // The same word again: the same shader.
    rig.state.words.insert((3, 8), 0x1234);
    assert_eq!(ready(rig.lookup(&[vertex(A)]))[0].0, first[0].0);

    // Another word: another shader.
    rig.state.words.insert((3, 8), 0x1235);
    pending(rig.lookup(&[vertex(A)]));
    rig.executor.run_all();
    let second = ready(rig.lookup(&[vertex(A)]));
    assert_ne!(second[0].0, first[0].0);
    assert_eq!(second[0].1.answers.cbuf_word(3, 8), Some(Some(0x1235)));

    // Another variant: another shader, with that variant.
    rig.variant.dual_source_blend = true;
    pending(rig.lookup(&[vertex(A)]));
    rig.executor.run_all();
    let third = ready(rig.lookup(&[vertex(A)]));
    assert_ne!(third[0].0, second[0].0);
    assert!(third[0].1.variant.dual_source_blend);
    // Captured and made once: answers and variants are layer 3 only.
    assert_eq!(rig.counters().captures, 1);
    assert_eq!(rig.counters().programs, 1);
    assert_eq!(rig.counters().translations, 3);
}

#[test]
fn questions_are_answered_for_the_requests_stage() {
    let mut rig = Rig::new();
    let hash = rig.place(A, GuestStage::Pixel, &program(GuestStage::Pixel, &[1]));
    rig.fake
        .script()
        .questions
        .insert(hash, vec![Question::CbufWord { bank: 2, offset: 4 }]);
    pending(rig.lookup(&[pixel(A)]));
    assert_eq!(
        *rig.state
            .asked
            .lock()
            .unwrap_or_else(PoisonError::into_inner),
        [(GuestStage::Pixel, 2, 4)]
    );
}

// ---------------------------------------------------------------------------------------------
// Waiting.

#[test]
fn wait_gives_the_set_once_a_worker_has_finished_it() {
    let fake = Arc::new(Fake::default());
    let mut cache = ShaderCache::new(Arc::clone(&fake), Arc::new(Threads), TARGET);
    let mut memory = Memory::new(MEMORY_SIZE);
    let bytes = program(GuestStage::Vertex, &[1]);
    memory.write(A, &bytes);
    let state = State::default();
    let lookup = cache.lookup(&[vertex(A)], &VariantKey::default(), &memory, &state);
    let ticket = match lookup {
        CacheLookup::Pending(ticket) => ticket,
        // The worker may already have finished, but the lookup only queues.
        other => unreachable!("expected pending, got {other:?}"),
    };
    let (sender, receiver) = mpsc::channel();
    let waiter = thread::spawn(move || {
        let result = cache.wait(ticket);
        let _sent =
            sender.send(matches!(result, CacheLookup::Ready(ref shaders) if shaders.len() == 1));
        cache
    });
    assert_eq!(receiver.recv_timeout(WATCHDOG), Ok(true));
    let mut cache = waiter.join().expect("the waiter returns");
    // The ticket is spent; the set is ready to a lookup.
    assert!(cache.tickets.is_empty());
    assert!(matches!(
        cache.lookup(&[vertex(A)], &VariantKey::default(), &memory, &state),
        CacheLookup::Ready(_)
    ));
    assert_eq!(fake.shader_calls(), 1);
}

#[test]
fn wait_returns_at_once_when_the_job_already_ran() {
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let ticket = pending(rig.lookup(&[vertex(A)]));
    rig.executor.run_all();
    let shaders = ready(rig.cache.wait(ticket));
    assert_eq!(shaders.len(), 1);
    // Spent: waiting again on it is an unknown ticket.
    assert_eq!(pending(rig.cache.wait(ticket)), ticket);
    assert!(rig.cache.tickets.is_empty());
    assert!(rig.cache.set_tickets.is_empty());
}

#[test]
fn an_unknown_ticket_answers_pending_at_once() {
    let mut rig = Rig::new();
    let ticket = Ticket::new(99);
    assert_eq!(pending(rig.cache.wait(ticket)), ticket);
}

#[test]
fn a_wait_that_times_out_keeps_its_ticket() {
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let ticket = pending(rig.lookup(&[vertex(A)]));
    assert_eq!(
        pending(rig.cache.wait_timeout(ticket, Duration::from_millis(20))),
        ticket
    );
    assert!(rig.cache.tickets.contains_key(&ticket));
    rig.executor.run_all();
    assert_eq!(ready(rig.cache.wait_timeout(ticket, WATCHDOG)).len(), 1);
}

#[test]
fn a_wait_blocks_until_a_worker_posts() {
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let ticket = pending(rig.lookup(&[vertex(A)]));
    let executor = Arc::clone(&rig.executor);
    let worker = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        executor.run_all();
    });
    let started = std::time::Instant::now();
    assert_eq!(ready(rig.cache.wait_timeout(ticket, WATCHDOG)).len(), 1);
    // Woken by the post, not by the deadline.
    assert!(started.elapsed() < WATCHDOG / 2, "{:?}", started.elapsed());
    worker.join().expect("the worker ends");
}

#[test]
fn a_ready_lookup_spends_the_sets_ticket() {
    let mut rig = Rig::new();
    rig.place(A, GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let ticket = pending(rig.lookup(&[vertex(A)]));
    rig.executor.run_all();
    ready(rig.lookup(&[vertex(A)]));
    assert!(!rig.cache.tickets.contains_key(&ticket));
    assert!(rig.cache.set_tickets.is_empty());
    // A later pending lookup of another set gets a new ticket.
    rig.place(B, GuestStage::Vertex, &program(GuestStage::Vertex, &[2]));
    assert_eq!(
        pending(rig.lookup(&[vertex(B)])),
        Ticket::new(ticket.id() + 1)
    );
}

// ---------------------------------------------------------------------------------------------
// The stamp.

#[test]
fn the_stamp_changes_with_every_field_of_the_key() {
    let bytes = program(GuestStage::Vertex, &[1]);
    let hash = hash_of(GuestStage::Vertex, &bytes);
    let other_hash = hash_of(GuestStage::Vertex, &program(GuestStage::Vertex, &[2]));
    let answers = Answers::new();
    let variant = VariantKey::default();
    let base = super::ShaderKey {
        version: TRANSLATOR_VERSION,
        target: &TARGET,
        stage: GuestStage::Vertex,
        hash,
        answers: &answers,
        variant: &variant,
    };
    let stamp = base.stamp();
    // The same key, the same stamp.
    assert_eq!(
        super::ShaderKey {
            answers: &Answers::new(),
            variant: &VariantKey::default(),
            ..base
        }
        .stamp(),
        stamp
    );
    let other_target = HostTarget {
        caps: Requirements::default(),
        ..TARGET
    };
    let other_variant = VariantKey {
        dual_source_blend: true,
        ..VariantKey::default()
    };
    let mut rig = Rig::new();
    let question_hash = rig.place(A, GuestStage::Vertex, &bytes);
    rig.fake.script().questions.insert(
        question_hash,
        vec![Question::CbufWord { bank: 1, offset: 0 }],
    );
    pending(rig.lookup(&[vertex(A)]));
    rig.executor.run_all();
    let other_answers = ready(rig.lookup(&[vertex(A)]))[0].1.answers.clone();
    assert!(!other_answers.is_empty());
    let changed = [
        super::ShaderKey {
            version: TRANSLATOR_VERSION + 1,
            ..base
        },
        super::ShaderKey {
            target: &other_target,
            ..base
        },
        super::ShaderKey {
            stage: GuestStage::Pixel,
            ..base
        },
        super::ShaderKey {
            hash: other_hash,
            ..base
        },
        super::ShaderKey {
            answers: &other_answers,
            ..base
        },
        super::ShaderKey {
            variant: &other_variant,
            ..base
        },
    ];
    for (field, key) in changed.iter().enumerate() {
        assert_ne!(key.stamp(), stamp, "key field {field}");
    }
}

#[test]
fn the_key_writer_writes_integers_little_endian_at_a_fixed_width() {
    use std::hash::Hasher;
    let mut writer = super::stamp::KeyWriter::default();
    writer.write_u8(0x01);
    writer.write_u16(0x0302);
    writer.write_u32(0x0706_0504);
    writer.write_u64(0x0f0e_0d0c_0b0a_0908);
    writer.write_usize(0x11);
    writer.write_isize(-2);
    writer.write_i32(-3);
    writer.write_u128(0x22);
    let mut expected = vec![
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    ];
    expected.extend_from_slice(&[0x11, 0, 0, 0, 0, 0, 0, 0]);
    expected.extend_from_slice(&[0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
    expected.extend_from_slice(&[0xfd, 0xff, 0xff, 0xff]);
    expected.extend_from_slice(&[0x22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(writer.bytes, expected);
}

#[test]
fn a_stamp_prints_as_thirty_two_hex_digits() {
    let stamp = super::ShaderStamp(0xabc);
    assert_eq!(stamp.to_string(), "00000000000000000000000000000abc");
    assert_eq!(stamp.value(), 0xabc);
}

// ---------------------------------------------------------------------------------------------
// A real-thread executor through a production constructor.

#[test]
fn a_translation_on_a_worker_thread_reaches_a_lookup() {
    let fake = Arc::new(Fake::default());
    let mut cache = ShaderCache::new(Arc::clone(&fake), Arc::new(Threads), TARGET);
    let mut memory = Memory::new(MEMORY_SIZE);
    memory.write(A, &program(GuestStage::Vertex, &[1]));
    let state = State::default();
    let variant = VariantKey::default();
    let CacheLookup::Pending(ticket) = cache.lookup(&[vertex(A)], &variant, &memory, &state) else {
        unreachable!("a first lookup is pending");
    };
    // Bounded: the worker posts within the watchdog, or the test fails.
    assert!(matches!(
        cache.wait_timeout(ticket, WATCHDOG),
        CacheLookup::Ready(_)
    ));
    assert_eq!(cache.counters().translations, 1);
    assert!(matches!(
        cache.lookup(&[vertex(A)], &variant, &memory, &state),
        CacheLookup::Ready(_)
    ));
}

/// Rejection test helpers shared by the submodules.
fn decode_rejection(index: usize) -> Rejection {
    Rejection::Decode {
        index,
        error: crate::decode::DecodeError::UnknownOpcode { opcode: 0 },
    }
}

/// A translator's rejection with no instruction.
fn no_exit() -> Rejection {
    Rejection::Translate(TranslateError::NoExit)
}

#[test]
fn a_stamp_does_not_depend_on_what_the_scratch_held_before() {
    let hash = hash_of(GuestStage::Vertex, &program(GuestStage::Vertex, &[1]));
    let answers = Answers::new();
    let variant = VariantKey::default();
    let key = super::ShaderKey {
        version: TRANSLATOR_VERSION,
        target: &TARGET,
        stage: GuestStage::Vertex,
        hash,
        answers: &answers,
        variant: &variant,
    };
    let fresh = key.stamp_with(&mut Vec::new());
    let mut scratch = vec![0xa5; 64];
    assert_eq!(key.stamp_with(&mut scratch), fresh);
    // Reused: the same stamp again.
    assert_eq!(key.stamp_with(&mut scratch), fresh);
}

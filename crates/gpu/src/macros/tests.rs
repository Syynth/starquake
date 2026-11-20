//! Tests for the front macro host, in the recorded-call style: words go in through
//! `ProcessMacros::write` (or through a real `Channel`), what reaches the sink is recorded in
//! order, and the back is a fake that answers `BackWork` and counts the questions.

use std::cell::Cell;

use starquake_memory::{FlatMemory, PAGE_SIZE, PageRange};

use super::*;
use crate::channel::{Channel, ChannelSink, Note, RunOutcome};
use crate::fifo::{GpEntry, MethodHeader, MethodMode, PushBuffer};
use crate::memory::TestGpuMemory;
use crate::mme::testing::{finish, inst0, mov, mov_reg, r, store, with_imm, with_out};
use crate::mme::{AluOp, LOAD_BARRIER, NotMemoised, OutOp, Reg, encode_program};
use crate::regs::{ShadowMode, THREE_D};
use crate::ring::Subchannel;
use crate::test_log::captured;

// ---------------------------------------------------------------------------------------------
// Rig
// ---------------------------------------------------------------------------------------------

/// The CPU memory behind the mapping.
const CPU_BASE: u64 = 0x8000_0000;
/// Pages mapped from `GPU_BASE`.
const PAGES: u64 = 8;
/// Above 32 bits, so `SET_MME_MEM_ADDRESS_A` matters.
const GPU_BASE: u64 = 0x12_3400_0000;
/// Where DMA reads read from: not page aligned.
const SOURCE: u64 = GPU_BASE + 0x140;
/// Where command words go.
const CMD: u64 = GPU_BASE + 4 * PAGE_SIZE;
/// Channel ids other than 0, so a hard-coded id fails.
const A: ChannelId = ChannelId(7);
const B: ChannelId = ChannelId(9);

fn three_d() -> ClassId {
    ClassId(clc797::CLASS)
}

fn compute() -> ClassId {
    ClassId(clc7c0::CLASS)
}

/// `SET_MME_SHADOW_SCRATCH(i)`: registers macros write their results to in these tests.
fn out_method(index: u32) -> u16 {
    u16::try_from(clc797::SET_MME_SHADOW_SCRATCH.method_at(&[index]).unwrap()).unwrap()
}

fn m16(method: u32) -> u16 {
    u16::try_from(method).unwrap()
}

/// The macro method of call `j`, and its data method.
fn call_method(engine: Engine, j: u32) -> u16 {
    m16(engine.first_macro_method() + 2 * j)
}

/// The back: says whether it has work, and counts the questions.
#[derive(Default)]
struct FakeBack {
    busy: Cell<bool>,
    queries: Cell<u32>,
}

impl BackWork for FakeBack {
    fn has_unperformed_work(&self) -> bool {
        self.queries.set(self.queries.get() + 1);
        self.busy.get()
    }
}

/// One channel: its macro state and its 3D and compute register files.
struct Chan {
    macros: MacroChannel,
    three_d: RegFile,
    compute: RegFile,
}

impl Chan {
    fn new(id: ChannelId) -> Self {
        Self {
            macros: MacroChannel::new(id),
            three_d: RegFile::new(id, three_d()),
            compute: RegFile::new(id, compute()),
        }
    }
}

struct Rig {
    mem: TestGpuMemory<FlatMemory>,
    back: FakeBack,
    process: ProcessMacros,
    out: Vec<RingEntry<'static>>,
}

fn make_rig() -> Rig {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = TestGpuMemory::new(FlatMemory::new(CPU_BASE, size));
    mem.map(
        PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(),
        CPU_BASE,
    )
    .unwrap();
    Rig {
        mem,
        back: FakeBack::default(),
        process: ProcessMacros::new(),
        out: Vec::new(),
    }
}

fn word(engine: Engine, method: u16, value: u32, remaining: u16) -> EngineWord {
    EngineWord {
        subchannel: Subchannel::new(3).unwrap(),
        class: engine.class(),
        method,
        value,
        remaining,
    }
}

impl Rig {
    fn write_memory(&self, va: u64, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.mem.write(va, &bytes).unwrap();
    }

    /// One word, through the host.
    fn word(
        &mut self,
        chan: &mut Chan,
        engine: Engine,
        method: u16,
        value: u32,
        remaining: u16,
    ) -> Result<WordOutcome, HostError> {
        let Rig {
            mem,
            back,
            process,
            out,
        } = self;
        send(
            process,
            out,
            mem,
            back,
            chan,
            word(engine, method, value, remaining),
        )
    }

    /// Stream words to one method, as one header would send them.
    fn stream(&mut self, chan: &mut Chan, engine: Engine, method: u32, values: &[u32]) {
        for value in values {
            assert_eq!(
                self.word(chan, engine, m16(method), *value, 0),
                Ok(WordOutcome::Register)
            );
        }
    }

    /// Uploads `program` at instruction `at` of the store `engine` writes to, and binds it to
    /// start table entry `entry`, all as stream words.
    fn upload(
        &mut self,
        chan: &mut Chan,
        engine: Engine,
        at: u32,
        program: &[Instruction],
        entry: u32,
    ) {
        let methods = engine.methods();
        self.stream(chan, engine, methods.instruction_pointer, &[at]);
        self.stream(
            chan,
            engine,
            methods.instruction,
            &encode_program(program).unwrap(),
        );
        self.stream(chan, engine, methods.start_pointer, &[entry]);
        self.stream(chan, engine, methods.start, &[at]);
    }

    /// Calls macro `j` with `parameters`, as one header of incrementing-once words would, and
    /// returns the call's report.
    fn call(&mut self, chan: &mut Chan, engine: Engine, j: u32, parameters: &[u32]) -> CallReport {
        let count = parameters.len();
        let mut report = None;
        for (index, value) in parameters.iter().enumerate() {
            let method = call_method(engine, j) + u16::from(index > 0);
            let remaining = u16::try_from(count - index - 1).unwrap();
            let outcome = self.word(chan, engine, method, *value, remaining).unwrap();
            if let WordOutcome::Call(call) = outcome {
                assert_eq!(remaining, 0);
                report = Some(call);
            } else {
                assert_eq!((outcome, remaining > 0), (WordOutcome::Parameter, true));
            }
        }
        report.unwrap()
    }

    fn resume(&mut self, chan: &mut Chan, engine: Engine) -> Result<CallReport, HostError> {
        let env = MacroEnv {
            memory: &self.mem,
            back: &self.back,
        };
        let Chan {
            macros,
            three_d,
            compute,
        } = chan;
        let file = match engine {
            Engine::ThreeD => three_d,
            Engine::Compute => compute,
        };
        self.process.resume(macros, file, &env, &mut self.out)
    }

    fn take(&mut self) -> Vec<RingEntry<'static>> {
        std::mem::take(&mut self.out)
    }
}

fn write_entry(channel: ChannelId, engine: Engine, method: u16, value: u32) -> RingEntry<'static> {
    RingEntry::Write {
        channel,
        class: engine.class(),
        method,
        value,
    }
}

/// What the memo did, for a call that committed.
fn committed(report: &CallReport) -> Option<CallOutcome> {
    match report.result {
        CallResult::Committed(outcome) => Some(outcome),
        CallResult::Failed { .. } | CallResult::CatchUp => None,
    }
}

/// Every register and shadow word of a file.
fn snapshot(file: &RegFile) -> Vec<(Option<u32>, Option<u32>)> {
    (0..file.rules().registers())
        .map(|method| (file.get(method), file.shadow(method)))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------------------------

/// Emits the next parameter to `method`.
fn emit_param(method: u16) -> Instruction {
    with_out(
        with_imm(Instruction::default(), 0, method),
        0,
        OutOp::Imm1,
        OutOp::Load0,
    )
}

/// Emits `value` to `method`.
fn emit_imm(method: u16, value: u16) -> Instruction {
    with_out(
        with_imm(Instruction::default(), value, method),
        0,
        OutOp::Imm1,
        OutOp::Imm0,
    )
}

/// Emits `R<reg>` to `method`.
fn emit_reg(method: u16, reg: u8) -> Instruction {
    with_out(
        with_imm(inst0(AluOp::Add, Reg::Zero, r(reg), Reg::Zero), 0, method),
        0,
        OutOp::Imm1,
        OutOp::Alu0,
    )
}

/// `R<dst>` = register `method` (`STATE`).
fn state_to(dst: u8, method: u16) -> Instruction {
    with_imm(inst0(AluOp::State, r(dst), Reg::Imm, Reg::Zero), method, 0)
}

/// `R<dst>` = data RAM `[index]`.
fn dread_to(dst: u8, index: u16) -> Instruction {
    with_imm(inst0(AluOp::Dread, r(dst), Reg::Imm, Reg::Zero), index, 0)
}

/// The load barrier.
fn barrier() -> Instruction {
    with_imm(
        inst0(AluOp::Extended, Reg::Zero, Reg::Imm, Reg::Zero),
        u16::try_from(LOAD_BARRIER).unwrap(),
        0,
    )
}

/// Emits each parameter, in order, to `SET_MME_SHADOW_SCRATCH(0..count)`.
fn echo(count: u32) -> Vec<Instruction> {
    finish((0..count).map(|i| emit_param(out_method(i))).collect())
}

/// The conditional-render style DMA macro (row 58): parameters `[upper, lower, length]` go to
/// `SET_MME_MEM_ADDRESS_A/B` and `MME_DMA_READ_FIFOED`, then data RAM word 5 and scratch 9 are
/// written (to show they are not committed by an abandoned run), then a barrier, then the two
/// words read are emitted to scratch 0 and 1.
fn dma_macro() -> Vec<Instruction> {
    finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_param(m16(DMA_READ_FIFOED)),
        mov(3, 0x55),
        store(5, 3),
        emit_imm(out_method(9), 0x99),
        barrier(),
        mov_reg(0, Reg::Load0),
        mov_reg(1, Reg::Load0),
        emit_reg(out_method(0), 0),
        emit_reg(out_method(1), 1),
    ])
}

fn source_parameters(address: u64, words: u32) -> [u32; 3] {
    [
        u32::try_from(address >> 32).unwrap(),
        u32::try_from(address & 0xffff_ffff).unwrap(),
        words,
    ]
}

// ---------------------------------------------------------------------------------------------
// Constants and method numbers
// ---------------------------------------------------------------------------------------------

#[test]
fn constants_match_the_references() {
    // Written out from the cited sources, so a wrong shipping constant fails here.
    assert_eq!(CODE_WORDS, 0x3000); // eden-nx2 macro.h: std::array<u32, 0x3000> native_code
    assert_eq!(CODE_INSTRUCTIONS, 0x1000); // ClearCode: method < 0x1000 ? method * 3
    // class_parser.py: 'CALL_MME_MACRO': 256 (owner's ruling, 2026-11-18; Eden-NX2 has 0x80).
    assert_eq!(START_ENTRIES, 256);
    assert_eq!(DMA_READ_MAX_WORDS, 0x1000); // macro.cpp: std::min<u32>(value, 0x1000)
    for engine in Engine::ALL {
        let methods = engine.methods();
        // clc797.h and clc7c0.h: 0x0114, 0x0118, 0x011c, 0x0120, CALL_MME_MACRO(j) 0x3800+8j.
        assert_eq!(methods.instruction_pointer, 0x114 / 4);
        assert_eq!(methods.instruction, 0x118 / 4);
        assert_eq!(methods.start_pointer, 0x11c / 4);
        assert_eq!(methods.start, 0x120 / 4);
        assert_eq!(engine.first_macro_method(), 0x3800 / 4);
        assert_eq!(engine.call_stride(), 8 / 4);
    }
    // An independent derivation: every call method a header can name has its own entry.
    for engine in Engine::ALL {
        let calls =
            (crate::channel::METHOD_MAX + 1 - engine.first_macro_method()) / engine.call_stride();
        assert_eq!(usize::try_from(calls).unwrap(), START_ENTRIES);
    }
    // clc797.h: SET_MME_MEM_ADDRESS_A 0x0550 (UPPER 16:0), _B 0x0554, SET_MME_DATA_RAM_ADDRESS
    // 0x0558 (WORD 31:0), MME_DMA_READ 0x055c (LENGTH 31:0), MME_DMA_READ_FIFOED 0x0560.
    assert_eq!(MEM_ADDRESS_A, 0x550 / 4);
    assert_eq!(MEM_ADDRESS_B, 0x554 / 4);
    assert_eq!(DATA_RAM_ADDRESS, 0x558 / 4);
    assert_eq!(DMA_READ, 0x55c / 4);
    assert_eq!(DMA_READ_FIFOED, 0x560 / 4);
    assert_eq!(DATA_RAM_ADDRESS_WORD, Field::new(31, 0));
    assert_eq!(DMA_READ_LENGTH, Field::new(31, 0));
    // mme_tu104.h: MME_TU104_DRAM_COUNT; eden-nx2 maxwell_3d.h: std::array<u32, 0xc00>.
    assert_eq!(DATA_RAM_WORDS, 0xc00);
    assert_eq!(MEM_ADDRESS_A_UPPER, Field::new(16, 0));
    assert_eq!(MEM_ADDRESS_B_LOWER, Field::new(31, 0));
    assert_eq!(DMA_READ_FIFOED_LENGTH, Field::new(31, 0));
    assert_eq!(Engine::of(ClassId(0xc797)), Some(Engine::ThreeD));
    assert_eq!(Engine::of(ClassId(0xc7c0)), Some(Engine::Compute));
    assert_eq!(Engine::of(ClassId(0xa140)), None);
    assert_eq!(Engine::ThreeD.class(), ClassId(0xc797));
    assert_eq!(Engine::Compute.class(), ClassId(0xc7c0));
}

#[test]
fn entries_map_calls_to_the_start_table() {
    for engine in Engine::ALL {
        let first = engine.first_macro_method();
        assert_eq!(engine.entry(first - 1), None);
        assert_eq!(engine.entry(first), Some(0));
        assert_eq!(engine.entry(first + 1), Some(0));
        assert_eq!(engine.entry(first + 2), Some(1));
        assert_eq!(engine.entry(first + 2 * 0x7f + 1), Some(0x7f));
        // Call 0x80 has its own entry (Eden-NX2 wraps it onto entry 0).
        assert_eq!(engine.entry(first + 2 * 0x80), Some(0x80));
        // The last entry, from both its methods, and none past it.
        let entries = u32::try_from(START_ENTRIES).unwrap();
        assert_eq!(
            engine.entry(first + 2 * (entries - 1)),
            Some(START_ENTRIES - 1)
        );
        assert_eq!(
            engine.entry(first + 2 * (entries - 1) + 1),
            Some(START_ENTRIES - 1)
        );
        assert_eq!(engine.entry(first + 2 * entries), None);
        assert_eq!(engine.entry(first + 2 * (entries + 3)), None);
        assert_eq!(engine.entry(u32::MAX), None);
    }
}

#[test]
fn takes_names_the_macro_and_upload_methods_of_3d_and_compute() {
    for engine in Engine::ALL {
        let class = engine.class();
        for method in [0x45, 0x46, 0x47, 0x48, 0xe00, 0xe01, 0xfff] {
            assert!(takes(class, method), "{engine:?} {method:#x}");
        }
        for method in [0x44, 0x49, 0x154, 0x158, 0xdff] {
            assert!(!takes(class, method), "{engine:?} {method:#x}");
        }
    }
    for method in [0x45, 0x48, 0xe00] {
        assert!(!takes(ClassId(0xa140), method));
    }
}

#[test]
fn new_state_starts_empty() {
    // Eden-NX2's NativeMacroContext and MacroEngine are value-initialised (`{}`): code, starts,
    // offset and generation all zero; Maxwell3D's data RAM is zeroed.
    let process = ProcessMacros::new();
    for store in Engine::ALL {
        assert!((0..CODE_WORDS).all(|offset| process.code_word(store, offset) == Some(0)));
        assert_eq!(process.code_writer(store, 0), None);
        assert_eq!(process.starts(store), [0; START_ENTRIES]);
        assert_eq!(process.upload_offset(store), 0);
        assert_eq!(process.code_generation(store), 0);
        assert_eq!(process.uploads(A, store), ChannelUploads::default());
    }
    assert_eq!(process.cross_channel_rewrites(), 0);
    let channel = MacroChannel::new(A);
    assert_eq!(channel.id(), A);
    assert_eq!(channel.data_ram(), &DataRam::new());
    assert!(!channel.is_waiting());
    assert!(!channel.compute_uploaded());
    for engine in Engine::ALL {
        assert_eq!(channel.collected(engine), []);
    }
}

// ---------------------------------------------------------------------------------------------
// Uploads and calls
// ---------------------------------------------------------------------------------------------

#[test]
fn an_uploaded_macro_runs_with_its_parameters() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0x20, &echo(2), 5);
    // The upload words also went to the register file and on.
    assert_ne!(rig.take(), []);
    assert_eq!(rig.process.starts(Engine::ThreeD)[5], 0x20);
    let report = rig.call(&mut chan, Engine::ThreeD, 5, &[0x1234_5678, 0x9abc_def0]);
    assert_eq!(
        report,
        CallReport {
            engine: Engine::ThreeD,
            store: Engine::ThreeD,
            entry: 5,
            start: 0x20,
            result: CallResult::Committed(CallOutcome::Stored),
        }
    );
    assert_eq!(
        rig.take(),
        [
            write_entry(A, Engine::ThreeD, out_method(0), 0x1234_5678),
            write_entry(A, Engine::ThreeD, out_method(1), 0x9abc_def0),
        ]
    );
}

#[test]
fn a_call_runs_when_its_last_parameter_arrives() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &echo(3), 1);
    rig.take();
    let method = call_method(Engine::ThreeD, 1);
    assert_eq!(
        rig.word(&mut chan, Engine::ThreeD, method, 10, 2),
        Ok(WordOutcome::Parameter)
    );
    assert_eq!(
        rig.word(&mut chan, Engine::ThreeD, method + 1, 11, 1),
        Ok(WordOutcome::Parameter)
    );
    assert_eq!(chan.macros.collected(Engine::ThreeD), [10, 11]);
    assert_eq!(rig.take(), []);
    let outcome = rig
        .word(&mut chan, Engine::ThreeD, method + 1, 12, 0)
        .unwrap();
    assert!(matches!(
        outcome,
        WordOutcome::Call(CallReport { entry: 1, .. })
    ));
    assert_eq!(rig.take().len(), 3);
    assert_eq!(chan.macros.collected(Engine::ThreeD), []);
    // An immediate (remaining 0) is a one-parameter call.
    rig.upload(&mut chan, Engine::ThreeD, 0x40, &echo(1), 2);
    rig.take();
    let report = rig.call(&mut chan, Engine::ThreeD, 2, &[77]);
    assert_eq!(report.start, 0x40);
    assert_eq!(
        rig.take(),
        [write_entry(A, Engine::ThreeD, out_method(0), 77)]
    );
}

#[test]
fn a_call_may_start_on_a_data_method() {
    // Divergence: Eden-NX2's compute class drops such a word; its 3D front runs it.
    for engine in Engine::ALL {
        let mut rig = make_rig();
        let mut chan = Chan::new(A);
        rig.upload(&mut chan, engine, 0x10, &echo(1), 4);
        rig.take();
        let method = call_method(engine, 4) + 1;
        let outcome = rig.word(&mut chan, engine, method, 5, 0).unwrap();
        assert!(matches!(
            outcome,
            WordOutcome::Call(CallReport {
                entry: 4,
                start: 0x10,
                ..
            })
        ));
        assert_eq!(rig.take(), [write_entry(A, engine, out_method(0), 5)]);
    }
}

#[test]
fn every_word_of_a_header_is_a_parameter_of_the_first_words_macro() {
    // An incrementing header from CALL_MME_MACRO(0) reaches CALL_MME_MACRO(1) at its third word;
    // that word is still a parameter of macro 0 (Eden's executing_macro is kept).
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &echo(3), 0);
    rig.upload(&mut chan, Engine::ThreeD, 0x30, &echo(1), 1);
    rig.take();
    let first = call_method(Engine::ThreeD, 0);
    rig.word(&mut chan, Engine::ThreeD, first, 1, 2).unwrap();
    rig.word(&mut chan, Engine::ThreeD, first + 1, 2, 1)
        .unwrap();
    let outcome = rig
        .word(&mut chan, Engine::ThreeD, first + 2, 3, 0)
        .unwrap();
    assert!(matches!(
        outcome,
        WordOutcome::Call(CallReport {
            entry: 0,
            start: 0,
            ..
        })
    ));
    assert_eq!(rig.take().len(), 3);
}

#[test]
fn call_0x80_has_its_own_entry() {
    // Divergence: Eden-NX2's 0x80-entry tables run call 0x80 from entry 0.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0x10, &echo(1), 0);
    rig.upload(&mut chan, Engine::ThreeD, 0x20, &echo(2), 0x80);
    rig.take();
    let report = rig.call(&mut chan, Engine::ThreeD, 0x80, &[8, 9]);
    assert_eq!((report.entry, report.start), (0x80, 0x20));
    assert_eq!(
        rig.take(),
        [
            write_entry(A, Engine::ThreeD, out_method(0), 8),
            write_entry(A, Engine::ThreeD, out_method(1), 9),
        ]
    );
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &[7]);
    assert_eq!((report.entry, report.start), (0, 0x10));
    assert_eq!(rig.process.starts(Engine::ThreeD)[0x80], 0x20);
    assert_eq!(rig.process.starts(Engine::ThreeD)[0], 0x10);
}

#[test]
fn a_call_past_the_start_table_is_refused() {
    for engine in Engine::ALL {
        let mut rig = make_rig();
        let mut chan = Chan::new(A);
        let last = u32::try_from(START_ENTRIES - 1).unwrap();
        rig.upload(&mut chan, engine, 0x10, &echo(1), last);
        rig.take();
        // The last entry runs.
        let report = rig.call(&mut chan, engine, last, &[0x31]);
        assert_eq!(
            (report.entry, report.start, report.result),
            (
                START_ENTRIES - 1,
                0x10,
                CallResult::Committed(CallOutcome::Stored)
            )
        );
        rig.take();
        // The first method past it is refused and changes nothing.
        let past = call_method(engine, last + 1);
        let file = match engine {
            Engine::ThreeD => snapshot(&chan.three_d),
            Engine::Compute => snapshot(&chan.compute),
        };
        assert_eq!(
            rig.word(&mut chan, engine, past, 0x32, 1),
            Err(HostError::PastStartTable { method: past })
        );
        assert_eq!(chan.macros.collected(engine), []);
        assert_eq!(rig.take(), []);
        let after = match engine {
            Engine::ThreeD => snapshot(&chan.three_d),
            Engine::Compute => snapshot(&chan.compute),
        };
        assert_eq!(after, file);
        // A later parameter word on that method belongs to the call begun before it.
        assert_eq!(
            rig.word(&mut chan, engine, call_method(engine, last), 0x41, 1),
            Ok(WordOutcome::Parameter)
        );
        let outcome = rig.word(&mut chan, engine, past, 0x42, 0).unwrap();
        assert!(matches!(
            outcome,
            WordOutcome::Call(CallReport { entry, .. }) if entry == START_ENTRIES - 1
        ));
    }
}

#[test]
fn a_call_past_the_start_table_is_checked_after_the_waiting_and_file_checks() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let past = call_method(Engine::ThreeD, u32::try_from(START_ENTRIES).unwrap());
    // The wrong file and a call past the table: the wrong file is reported.
    let env = MacroEnv {
        memory: &rig.mem,
        back: &rig.back,
    };
    let error = rig.process.write(
        &mut chan.macros,
        &mut chan.compute,
        word(Engine::ThreeD, past, 1, 0),
        &env,
        &mut rig.out,
    );
    assert_eq!(
        error,
        Err(HostError::WrongFile {
            file_channel: A.0,
            file_class: clc7c0::CLASS,
            channel: A.0,
            class: clc797::CLASS,
        })
    );
    // Waiting and a call past the table: waiting is reported.
    rig.upload(&mut chan, Engine::ThreeD, 0, &dma_macro(), 0);
    rig.back.busy.set(true);
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &source_parameters(SOURCE, 2));
    assert_eq!(report.result, CallResult::CatchUp);
    assert_eq!(
        rig.word(&mut chan, Engine::ThreeD, past, 1, 0),
        Err(HostError::Waiting)
    );
}

#[test]
fn a_call_whose_parameters_span_two_entries_runs_once_the_last_arrives() {
    // Through the real pusher: the header and one word in the first GP entry, two words in the
    // second.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &echo(3), 6);
    rig.take();
    let mut channel = Channel::new(A);
    let mut first = PushBuffer::new();
    first
        .methods(MethodMode::Incrementing, 0, 0, &[clc797::CLASS])
        .unwrap();
    first
        .header(&MethodHeader::Methods {
            mode: MethodMode::IncrementOnce,
            subchannel: 0,
            method: call_method(Engine::ThreeD, 6),
            count: 3,
        })
        .unwrap();
    first.data(&[0xa1]);
    let mut second = PushBuffer::new();
    second.data(&[0xa2, 0xa3]);
    let second_at = CMD + PAGE_SIZE;
    rig.write_memory(CMD, first.words());
    rig.write_memory(second_at, second.words());

    channel
        .start_entry(entry(CMD, first.words().len()))
        .unwrap();
    let (outcome, reports) = rig.pump(&mut channel, &mut chan);
    assert_eq!(outcome, Ok(RunOutcome::EntryDone));
    assert_eq!(reports, [WordOutcome::Parameter]);
    assert_eq!(chan.macros.collected(Engine::ThreeD), [0xa1]);
    // Only the bind went out.
    assert!(matches!(rig.take().as_slice(), [RingEntry::Bind { .. }]));

    channel.start_entry(entry(second_at, 2)).unwrap();
    let (outcome, reports) = rig.pump(&mut channel, &mut chan);
    assert_eq!(outcome, Ok(RunOutcome::EntryDone));
    assert!(matches!(
        reports.as_slice(),
        [WordOutcome::Parameter, WordOutcome::Call(_)]
    ));
    assert_eq!(
        rig.take(),
        [
            write_entry(A, Engine::ThreeD, out_method(0), 0xa1),
            write_entry(A, Engine::ThreeD, out_method(1), 0xa2),
            write_entry(A, Engine::ThreeD, out_method(2), 0xa3),
        ]
    );
}

fn entry(address: u64, length: usize) -> GpEntry {
    GpEntry {
        address,
        length: u32::try_from(length).unwrap(),
        ..GpEntry::default()
    }
}

/// One word through `process`, with `chan`'s file of the word's class.
fn send(
    process: &mut ProcessMacros,
    out: &mut Vec<RingEntry<'static>>,
    mem: &TestGpuMemory<FlatMemory>,
    back: &FakeBack,
    chan: &mut Chan,
    word: EngineWord,
) -> Result<WordOutcome, HostError> {
    let env = MacroEnv { memory: mem, back };
    let Chan {
        macros,
        three_d,
        compute,
    } = chan;
    let file = match Engine::of(word.class) {
        Some(Engine::Compute) => compute,
        _ => three_d,
    };
    process.write(macros, file, word, &env, out)
}

/// The front's sink, as slice G14 would wire it: engine words through the host, the rest to the
/// ring, and the pusher held while a call waits.
struct FrontSink<'a> {
    process: &'a mut ProcessMacros,
    out: &'a mut Vec<RingEntry<'static>>,
    mem: &'a TestGpuMemory<FlatMemory>,
    back: &'a FakeBack,
    chan: &'a mut Chan,
    reports: Vec<WordOutcome>,
    notes: Vec<Note>,
}

impl ChannelSink for FrontSink<'_> {
    fn entry(&mut self, entry: RingEntry<'static>) {
        self.out.push(entry);
    }
    fn method(&mut self, word: EngineWord) {
        let outcome = send(self.process, self.out, self.mem, self.back, self.chan, word).unwrap();
        self.reports.push(outcome);
    }
    fn set_reference(&mut self, _: u32) {}
    fn note(&mut self, note: Note) {
        self.notes.push(note);
    }
    fn holds(&self) -> bool {
        self.chan.macros.is_waiting()
    }
}

type Pumped = (
    Result<RunOutcome, crate::channel::ChannelError>,
    Vec<WordOutcome>,
);

impl Rig {
    /// Runs `channel` through the front's sink; returns the outcome and the words' outcomes.
    fn pump(&mut self, channel: &mut Channel, chan: &mut Chan) -> Pumped {
        let Rig {
            mem,
            back,
            process,
            out,
        } = self;
        let mem: &TestGpuMemory<FlatMemory> = mem;
        let mut sink = FrontSink {
            process,
            out,
            mem,
            back,
            chan,
            reports: Vec::new(),
            notes: Vec::new(),
        };
        let outcome = channel.run(mem, &mut sink, u32::MAX);
        assert_eq!(sink.notes, []);
        (outcome, sink.reports)
    }
}

// ---------------------------------------------------------------------------------------------
// The code store and start tables
// ---------------------------------------------------------------------------------------------

#[test]
fn the_upload_offset_lives_in_the_code_store() {
    // The spec's step 2 (module docs): the offset is a word offset in the store, three words
    // per instruction, as both references keep it, and method 0x45 stays as written.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let pointer = THREE_D_METHODS.instruction_pointer;
    let instruction = THREE_D_METHODS.instruction;
    rig.stream(&mut chan, Engine::ThreeD, pointer, &[0x21]);
    assert_eq!(
        rig.process.upload_offset(Engine::ThreeD),
        0x21 * INSTRUCTION_WORDS
    );
    rig.stream(
        &mut chan,
        Engine::ThreeD,
        instruction,
        &[0xa, 0xb, 0xc, 0xd],
    );
    assert_eq!(
        rig.process.upload_offset(Engine::ThreeD),
        0x21 * INSTRUCTION_WORDS + 4
    );
    for (index, value) in [0xa, 0xb, 0xc, 0xd].into_iter().enumerate() {
        assert_eq!(
            rig.process.code_word(Engine::ThreeD, 0x63 + index),
            Some(value)
        );
    }
    assert_eq!(rig.process.code_word(Engine::ThreeD, 0x62), Some(0));
    assert_eq!(rig.process.code_word(Engine::ThreeD, 0x67), Some(0));
    assert_eq!(chan.three_d.get(pointer), Some(0x21));
    // Each word moved the generation.
    assert_eq!(rig.process.code_generation(Engine::ThreeD), 4);
    assert_eq!(rig.process.code_generation(Engine::Compute), 0);
    // The words went to the register file and the ring as any other write.
    let sent = rig.take();
    assert_eq!(
        sent.first(),
        Some(&write_entry(A, Engine::ThreeD, m16(pointer), 0x21))
    );
    assert_eq!(sent.len(), 5);
}

#[test]
fn an_upload_pointer_at_the_end_drops_the_words_after_it() {
    let last = u32::try_from(CODE_INSTRUCTIONS - 1).unwrap();
    let pointer = THREE_D_METHODS.instruction_pointer;
    let instruction = THREE_D_METHODS.instruction;

    // The last instruction: three words stored, the fourth dropped.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.stream(&mut chan, Engine::ThreeD, pointer, &[last]);
    rig.stream(&mut chan, Engine::ThreeD, instruction, &[1, 2, 3, 4]);
    assert_eq!(
        rig.process.code_word(Engine::ThreeD, CODE_WORDS - 3),
        Some(1)
    );
    assert_eq!(
        rig.process.code_word(Engine::ThreeD, CODE_WORDS - 1),
        Some(3)
    );
    assert_eq!(rig.process.code_word(Engine::ThreeD, CODE_WORDS), None);
    assert_eq!(rig.process.upload_offset(Engine::ThreeD), CODE_WORDS);
    assert_eq!(rig.process.uploads(A, Engine::ThreeD).code_words, 3);
    // The dropped word still moved the generation (AddCode bumps it first).
    assert_eq!(rig.process.code_generation(Engine::ThreeD), 4);

    // One past: nothing is stored.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.stream(&mut chan, Engine::ThreeD, pointer, &[last + 1]);
    assert_eq!(rig.process.upload_offset(Engine::ThreeD), CODE_WORDS);
    rig.stream(&mut chan, Engine::ThreeD, instruction, &[9]);
    assert!((0..CODE_WORDS).all(|offset| rig.process.code_word(Engine::ThreeD, offset) == Some(0)));
    assert_eq!(
        rig.process.uploads(A, Engine::ThreeD),
        ChannelUploads::default()
    );
    assert_eq!(rig.process.code_generation(Engine::ThreeD), 1);
}

#[test]
fn a_bind_past_the_start_table_is_dropped() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let last = u32::try_from(START_ENTRIES - 1).unwrap();
    let start_pointer = THREE_D_METHODS.start_pointer;
    let start = THREE_D_METHODS.start;
    rig.stream(&mut chan, Engine::ThreeD, start_pointer, &[last]);
    rig.stream(&mut chan, Engine::ThreeD, start, &[0x11, 0x22]);
    let starts = rig.process.starts(Engine::ThreeD);
    assert_eq!(starts.len(), START_ENTRIES);
    assert_eq!(starts[START_ENTRIES - 1], 0x11);
    assert!(starts[..START_ENTRIES - 1].iter().all(|&start| start == 0));
    // The register still advanced past the table (slice G9; Eden-NX2's FrontWrite too).
    assert_eq!(chan.three_d.get(start_pointer), Some(last + 2));
    assert_eq!(rig.process.uploads(A, Engine::ThreeD).start_entries, 1);
}

#[test]
fn the_shadow_ram_applies_to_upload_words() {
    // Eden-NX2's FrontWrite uploads the register's value, so a replayed word uploads the
    // shadow copy.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let instruction = THREE_D_METHODS.instruction;
    let control = THREE_D.shadow_control().unwrap();
    // Track: the shadow file keeps 0x77 for the instruction method.
    rig.stream(&mut chan, Engine::ThreeD, instruction, &[0x77]);
    rig.stream(
        &mut chan,
        Engine::ThreeD,
        THREE_D_METHODS.instruction_pointer,
        &[0],
    );
    rig.stream(&mut chan, Engine::ThreeD, control, &[3]);
    assert_eq!(chan.three_d.shadow_mode(), Some(ShadowMode::Replay));
    rig.stream(&mut chan, Engine::ThreeD, instruction, &[0x55]);
    assert_eq!(rig.process.code_word(Engine::ThreeD, 0), Some(0x77));
}

#[test]
fn compute_uses_the_3d_store_until_it_uploads() {
    let mut rig = make_rig();
    let mut a = Chan::new(A);
    let mut b = Chan::new(B);
    // 3D code at entry 2.
    rig.upload(&mut a, Engine::ThreeD, 0x10, &echo(1), 2);
    rig.take();
    // A compute call before the channel's compute class uploads runs the 3D store's macro.
    let report = rig.call(&mut a, Engine::Compute, 2, &[0x31]);
    assert_eq!((report.store, report.start), (Engine::ThreeD, 0x10));
    assert_eq!(
        rig.take(),
        [write_entry(A, Engine::Compute, out_method(0), 0x31)]
    );
    assert!(!a.macros.compute_uploaded());

    // Channel A's compute class uploads: from its pointer write on, it uses the compute store.
    let program = finish(vec![emit_imm(out_method(1), 0x44)]);
    rig.upload(&mut a, Engine::Compute, 0x10, &program, 2);
    rig.take();
    assert!(a.macros.compute_uploaded());
    assert_eq!(rig.process.starts(Engine::Compute)[2], 0x10);
    let words = encode_program(&program).unwrap().len();
    assert_eq!(
        rig.process.uploads(A, Engine::Compute).code_words,
        u64::try_from(words).unwrap()
    );
    // The 3D store is unchanged.
    assert_eq!(rig.process.starts(Engine::ThreeD)[2], 0x10);
    assert_eq!(
        rig.process.code_word(Engine::ThreeD, 0x30),
        Some(encode_program(&echo(1)).unwrap()[0])
    );
    let report = rig.call(&mut a, Engine::Compute, 2, &[0x32]);
    assert_eq!(report.store, Engine::Compute);
    assert_eq!(
        rig.take(),
        [write_entry(A, Engine::Compute, out_method(1), 0x44)]
    );

    // The switch is per channel: B's compute class still uses the 3D store.
    let report = rig.call(&mut b, Engine::Compute, 2, &[0x33]);
    assert_eq!(report.store, Engine::ThreeD);
    assert_eq!(
        rig.take(),
        [write_entry(B, Engine::Compute, out_method(0), 0x33)]
    );

    // A start pointer write switches too; a 3D upload never does.
    let mut c = Chan::new(ChannelId(11));
    rig.stream(
        &mut c,
        Engine::ThreeD,
        THREE_D_METHODS.instruction_pointer,
        &[0],
    );
    rig.stream(&mut c, Engine::ThreeD, THREE_D_METHODS.start_pointer, &[0]);
    assert!(!c.macros.compute_uploaded());
    let generation = rig.process.code_generation(Engine::ThreeD);
    rig.stream(&mut c, Engine::Compute, COMPUTE_METHODS.instruction, &[0]);
    assert!(!c.macros.compute_uploaded());
    // Before the switch, a compute upload word goes to the 3D store.
    assert_eq!(rig.process.code_generation(Engine::ThreeD), generation + 1);
    assert_eq!(
        rig.process.code_writer(Engine::ThreeD, 0),
        Some(ChannelId(11))
    );
    rig.stream(&mut c, Engine::Compute, COMPUTE_METHODS.start_pointer, &[0]);
    assert!(c.macros.compute_uploaded());
}

#[test]
fn data_ram_is_per_channel_and_the_code_store_is_shared() {
    let mut rig = make_rig();
    let mut a = Chan::new(A);
    let mut b = Chan::new(B);
    // Uploaded through A only: data RAM word 0x27 = the parameter.
    let writer = finish(vec![mov_reg(0, Reg::Load0), store(0x27, 0)]);
    rig.upload(&mut a, Engine::ThreeD, 0, &writer, 0);
    // Reads word 0x27 and emits it.
    let reader = finish(vec![dread_to(1, 0x27), emit_reg(out_method(0), 1)]);
    rig.upload(&mut a, Engine::ThreeD, 0x10, &reader, 1);
    rig.take();
    // B calls code only A uploaded.
    rig.call(&mut b, Engine::ThreeD, 0, &[0xbbbb]);
    rig.call(&mut a, Engine::ThreeD, 0, &[0xaaaa]);
    assert_eq!(a.macros.data_ram().get(0x27), Some(0xaaaa));
    assert_eq!(b.macros.data_ram().get(0x27), Some(0xbbbb));
    rig.take();
    // A's compute class shares A's data RAM (it runs the 3D store's reader, entry 1).
    rig.call(&mut a, Engine::Compute, 1, &[0]);
    assert_eq!(
        rig.take(),
        [write_entry(A, Engine::Compute, out_method(0), 0xaaaa)]
    );
    rig.call(&mut b, Engine::ThreeD, 1, &[0]);
    assert_eq!(
        rig.take(),
        [write_entry(B, Engine::ThreeD, out_method(0), 0xbbbb)]
    );
}

#[test]
fn uploads_are_recorded_per_channel() {
    let mut rig = make_rig();
    let mut a = Chan::new(A);
    let mut b = Chan::new(B);
    let pointer = THREE_D_METHODS.instruction_pointer;
    let instruction = THREE_D_METHODS.instruction;
    rig.stream(&mut a, Engine::ThreeD, pointer, &[0]);
    rig.stream(&mut a, Engine::ThreeD, instruction, &[1, 2, 3]);
    rig.stream(&mut a, Engine::ThreeD, THREE_D_METHODS.start_pointer, &[4]);
    rig.stream(&mut a, Engine::ThreeD, THREE_D_METHODS.start, &[0x50]);
    assert_eq!(
        rig.process.uploads(A, Engine::ThreeD),
        ChannelUploads {
            code_words: 3,
            start_entries: 1,
        }
    );
    assert_eq!(rig.process.code_writer(Engine::ThreeD, 1), Some(A));
    assert_eq!(rig.process.code_writer(Engine::ThreeD, 3), None);

    // B writes the same values over A's: not a rewrite.
    let ((), events) = captured(|| {
        rig.stream(&mut b, Engine::ThreeD, pointer, &[0]);
        rig.stream(&mut b, Engine::ThreeD, instruction, &[1]);
        rig.stream(&mut b, Engine::ThreeD, THREE_D_METHODS.start_pointer, &[4]);
        rig.stream(&mut b, Engine::ThreeD, THREE_D_METHODS.start, &[0x50]);
    });
    assert_eq!(events, []);
    assert_eq!(rig.process.cross_channel_rewrites(), 0);
    assert_eq!(rig.process.code_writer(Engine::ThreeD, 0), Some(B));
    // A rewriting its own word with a new value: not a rewrite either.
    rig.stream(&mut a, Engine::ThreeD, pointer, &[0]);
    rig.stream(&mut a, Engine::ThreeD, instruction, &[1, 9]);
    assert_eq!(rig.process.cross_channel_rewrites(), 0);
    // B changes A's word 2, then A changes the start B stored last: two rewrites, logged
    // once, naming the first.
    let ((), events) = captured(|| {
        rig.stream(&mut b, Engine::ThreeD, pointer, &[0]);
        rig.stream(&mut b, Engine::ThreeD, instruction, &[1, 9, 7]);
        rig.stream(&mut a, Engine::ThreeD, THREE_D_METHODS.start_pointer, &[4]);
        rig.stream(&mut a, Engine::ThreeD, THREE_D_METHODS.start, &[0x60]);
    });
    assert_eq!(rig.process.cross_channel_rewrites(), 2);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, tracing::Level::WARN);
    assert!(events[0].1.contains("channel=9;"), "{}", events[0].1);
    assert_eq!(
        rig.process.uploads(B, Engine::ThreeD),
        ChannelUploads {
            code_words: 4,
            start_entries: 1,
        }
    );
    assert_eq!(
        rig.process.uploads(A, Engine::ThreeD),
        ChannelUploads {
            code_words: 5,
            start_entries: 2,
        }
    );
    rig.process.forget_channel(B);
    assert_eq!(
        rig.process.uploads(B, Engine::ThreeD),
        ChannelUploads::default()
    );
    assert_eq!(rig.process.uploads(A, Engine::ThreeD).code_words, 5);
}

#[test]
fn a_channel_rebinding_its_own_start_is_not_a_rewrite() {
    let mut rig = make_rig();
    let mut a = Chan::new(A);
    let mut b = Chan::new(B);
    let start_pointer = THREE_D_METHODS.start_pointer;
    let start = THREE_D_METHODS.start;
    rig.stream(&mut a, Engine::ThreeD, start_pointer, &[4]);
    rig.stream(&mut a, Engine::ThreeD, start, &[0x50]);
    // A binds its own entry again with a new value: not a rewrite, nothing logged.
    let ((), events) = captured(|| {
        rig.stream(&mut a, Engine::ThreeD, start_pointer, &[4]);
        rig.stream(&mut a, Engine::ThreeD, start, &[0x60]);
    });
    assert_eq!(events, []);
    assert_eq!(rig.process.cross_channel_rewrites(), 0);
    assert_eq!(rig.process.starts(Engine::ThreeD)[4], 0x60);
    // B binding A's entry with another value is one.
    rig.stream(&mut b, Engine::ThreeD, start_pointer, &[4]);
    rig.stream(&mut b, Engine::ThreeD, start, &[0x70]);
    assert_eq!(rig.process.cross_channel_rewrites(), 1);
    assert_eq!(rig.process.starts(Engine::ThreeD)[4], 0x70);
}

#[test]
fn forgetting_a_channel_clears_its_record_in_both_stores() {
    let mut rig = make_rig();
    let mut a = Chan::new(A);
    let mut b = Chan::new(B);
    for chan in [&mut a, &mut b] {
        // Into the 3D store, then, after its own pointer write, into the compute store.
        rig.stream(
            chan,
            Engine::ThreeD,
            THREE_D_METHODS.instruction_pointer,
            &[0],
        );
        rig.stream(chan, Engine::ThreeD, THREE_D_METHODS.instruction, &[1, 2]);
        rig.stream(
            chan,
            Engine::Compute,
            COMPUTE_METHODS.instruction_pointer,
            &[0],
        );
        rig.stream(
            chan,
            Engine::Compute,
            COMPUTE_METHODS.instruction,
            &[3, 4, 5],
        );
        rig.stream(chan, Engine::Compute, COMPUTE_METHODS.start_pointer, &[1]);
        rig.stream(chan, Engine::Compute, COMPUTE_METHODS.start, &[0]);
        assert!(chan.macros.compute_uploaded());
    }
    let three_d = ChannelUploads {
        code_words: 2,
        start_entries: 0,
    };
    let compute = ChannelUploads {
        code_words: 3,
        start_entries: 1,
    };
    for channel in [A, B] {
        assert_eq!(rig.process.uploads(channel, Engine::ThreeD), three_d);
        assert_eq!(rig.process.uploads(channel, Engine::Compute), compute);
    }
    rig.process.forget_channel(B);
    assert_eq!(
        rig.process.uploads(B, Engine::ThreeD),
        ChannelUploads::default()
    );
    assert_eq!(
        rig.process.uploads(B, Engine::Compute),
        ChannelUploads::default()
    );
    // A's record stays.
    assert_eq!(rig.process.uploads(A, Engine::ThreeD), three_d);
    assert_eq!(rig.process.uploads(A, Engine::Compute), compute);
}

// ---------------------------------------------------------------------------------------------
// Runs: state reads, buffered writes, failures
// ---------------------------------------------------------------------------------------------

#[test]
fn state_reads_see_the_file_and_the_runs_own_writes() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let program = finish(vec![
        state_to(0, out_method(3)),
        emit_reg(out_method(4), 0),
        emit_param(out_method(3)),
        state_to(1, out_method(3)),
        emit_reg(out_method(5), 1),
        // Past the register file: 0.
        state_to(2, call_method(Engine::ThreeD, 0)),
        emit_reg(out_method(6), 2),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    rig.stream(&mut chan, Engine::ThreeD, u32::from(out_method(3)), &[0x33]);
    rig.stream(&mut chan, Engine::ThreeD, u32::from(out_method(6)), &[0x66]);
    rig.take();
    rig.call(&mut chan, Engine::ThreeD, 0, &[0x44]);
    assert_eq!(
        rig.take(),
        [
            write_entry(A, Engine::ThreeD, out_method(4), 0x33),
            write_entry(A, Engine::ThreeD, out_method(3), 0x44),
            write_entry(A, Engine::ThreeD, out_method(5), 0x44),
            write_entry(A, Engine::ThreeD, out_method(6), 0),
        ]
    );
}

#[test]
fn a_failing_run_commits_nothing() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let control = THREE_D.shadow_control().unwrap();
    let cbuf = clc797::LOAD_CONSTANT_BUFFER.method_at(&[0]).unwrap();
    // Writes a register, a side-effect trigger, the shadow control, a bind and data RAM, then
    // asks for a parameter it was not given.
    let program = finish(vec![
        emit_param(out_method(0)),
        emit_imm(m16(cbuf), 0x1234),
        emit_imm(m16(control), 2),
        emit_imm(m16(THREE_D_METHODS.start_pointer), 9),
        emit_imm(m16(THREE_D_METHODS.start), 0x123),
        mov(2, 0x77),
        store(0x10, 2),
        store(0x10, 2),
        mov_reg(3, Reg::Load0),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0x40, &program, 3);
    rig.take();
    let before = snapshot(&chan.three_d);
    let starts = rig.process.starts(Engine::ThreeD).to_vec();
    let ram = chan.macros.data_ram().clone();

    let (report, events) = captured(|| rig.call(&mut chan, Engine::ThreeD, 3, &[0xabc]));
    let error = CallFailure::Macro(MacroError::ParameterUnderflow { pc: 8 });
    assert_eq!(
        report.result,
        CallResult::Failed {
            error,
            first_for_macro: true,
        }
    );
    assert_eq!(rig.take(), []);
    assert_eq!(snapshot(&chan.three_d), before);
    assert_eq!(rig.process.starts(Engine::ThreeD), starts);
    assert_eq!(chan.macros.data_ram(), &ram);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, tracing::Level::WARN);
    assert!(events[0].1.contains("start=64;"), "{}", events[0].1);
    assert!(events[0].1.contains("parameters=0xabc;"), "{}", events[0].1);
    assert!(
        events[0].1.contains(&format!("error={error};")),
        "{}",
        events[0].1
    );

    // The same macro failing again is not logged; it still commits nothing.
    let (report, events) = captured(|| rig.call(&mut chan, Engine::ThreeD, 3, &[0xdef]));
    assert_eq!(
        report.result,
        CallResult::Failed {
            error,
            first_for_macro: false,
        }
    );
    assert_eq!(events, []);
    assert_eq!(snapshot(&chan.three_d), before);

    // With its parameter, it runs and commits everything.
    let report = rig.call(&mut chan, Engine::ThreeD, 3, &[0xabc, 0xd]);
    assert_eq!(committed(&report), Some(CallOutcome::Stored));
    assert_eq!(chan.three_d.get(u32::from(out_method(0))), Some(0xabc));
    assert_eq!(chan.three_d.shadow_mode(), Some(ShadowMode::Passthrough));
    assert_eq!(rig.process.starts(Engine::ThreeD)[9], 0x123);
    assert_eq!(chan.macros.data_ram().get(0x10), Some(0x77));
    assert_eq!(rig.take().len(), 7);
}

#[test]
fn failure_messages_are_plain_words() {
    let cases = [
        (
            CallFailure::Macro(MacroError::ParameterUnderflow { pc: 3 }),
            "the macro failed: parameter FIFO underflow at instruction 3",
        ),
        (
            CallFailure::MethodOutOfRange { method: 0xe00 },
            "the macro wrote method 0xe00, which is outside its class's register file",
        ),
        (
            CallFailure::DmaRead {
                address: 0x1_0000,
                words: 2,
                fault: GpuFault::Unmapped { va: 0x1_0000 },
            },
            "the macro's DMA read of 2 words at GPU address 0x10000 failed: GPU address 0x10000 is not mapped",
        ),
        (
            CallFailure::DmaTooLarge { words: 0x1001 },
            "the macro's DMA reads asked for 4097 words in all, more than the 4096 one call may read",
        ),
        (
            CallFailure::TooManyWrites,
            "the macro made more than 262144 method writes",
        ),
        (
            CallFailure::DataRamOutOfRange {
                index: 0xbff,
                words: 2,
            },
            "the macro's DMA read of length 2 into data RAM at word 0xbff runs past the end of data RAM (0xc00 words)",
        ),
        (
            CallFailure::DataRamOutOfRange {
                index: 0xc00,
                words: 1,
            },
            "the macro's DMA read of length 1 into data RAM at word 0xc00 runs past the end of data RAM (0xc00 words)",
        ),
    ];
    for (failure, text) in cases {
        assert_eq!(failure.to_string(), text);
    }
    let cases = [
        (
            HostError::Waiting,
            "the channel's macro call waits for the back to catch up",
        ),
        (
            HostError::NotWaiting,
            "the channel has no macro call waiting",
        ),
        (
            HostError::WrongFile {
                file_channel: 9,
                file_class: 0xc7c0,
                channel: 7,
                class: 0xc797,
            },
            "the register file of class 0xc7c0 on channel 9 does not belong to class 0xc797 on channel 7",
        ),
        (
            HostError::PastStartTable { method: 0x1000 },
            "method 0x1000 would call a macro past the start table's 256 entries",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
}

#[test]
fn a_write_outside_the_file_fails_the_run_and_is_kept_over_a_later_error() {
    // Eden-NX2: "unsupported emitted method". The host's failure comes first, so it is the one
    // reported, not the underflow that ends the starved run.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let program = finish(vec![
        emit_imm(out_method(0), 1),
        emit_imm(call_method(Engine::ThreeD, 0), 2),
        emit_imm(out_method(1), 3),
        // The first load takes the call's one parameter; the second underflows.
        mov_reg(0, Reg::Load0),
        mov_reg(0, Reg::Load0),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    rig.take();
    let before = snapshot(&chan.three_d);
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &[1]);
    assert_eq!(
        report.result,
        CallResult::Failed {
            error: CallFailure::MethodOutOfRange {
                method: call_method(Engine::ThreeD, 0),
            },
            first_for_macro: true,
        }
    );
    assert_eq!(rig.take(), []);
    assert_eq!(snapshot(&chan.three_d), before);
}

#[test]
fn a_failed_replay_commits_nothing() {
    // A run with no register reads, barrier or loads past its parameters is remembered even
    // when its write leaves the file (the memo cannot see the host's failure). Its replay fails
    // the same way, and the data RAM words the replay wrote are put back.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let program = finish(vec![
        mov_reg(0, Reg::Load0),
        store(4, 0),
        emit_imm(call_method(Engine::ThreeD, 1), 2),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    rig.take();
    let first = rig.call(&mut chan, Engine::ThreeD, 0, &[0x4444]);
    assert!(matches!(first.result, CallResult::Failed { .. }));
    assert_eq!(chan.macros.data_ram().get(4), Some(0));
    let again = rig.call(&mut chan, Engine::ThreeD, 0, &[0x4444]);
    assert_eq!(
        again.result,
        CallResult::Failed {
            error: CallFailure::MethodOutOfRange {
                method: call_method(Engine::ThreeD, 1),
            },
            first_for_macro: false,
        }
    );
    assert_eq!(chan.macros.data_ram().get(4), Some(0));
    assert_eq!(rig.take(), []);
}

#[test]
fn failures_are_logged_once_per_store_and_start() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let failing = finish(vec![mov_reg(0, Reg::Load0), mov_reg(0, Reg::Load0)]);
    rig.upload(&mut chan, Engine::ThreeD, 0x10, &failing, 0);
    rig.upload(&mut chan, Engine::ThreeD, 0x20, &failing, 1);
    rig.take();
    let first =
        |rig: &mut Rig, chan: &mut Chan, engine, j| match rig.call(chan, engine, j, &[0]).result {
            CallResult::Failed {
                first_for_macro, ..
            } => Some(first_for_macro),
            CallResult::Committed(_) | CallResult::CatchUp => None,
        };
    assert_eq!(first(&mut rig, &mut chan, Engine::ThreeD, 0), Some(true));
    assert_eq!(first(&mut rig, &mut chan, Engine::ThreeD, 0), Some(false));
    // Another start of the same store.
    assert_eq!(first(&mut rig, &mut chan, Engine::ThreeD, 1), Some(true));
    // The same start through another call entry is the same macro.
    rig.upload(&mut chan, Engine::ThreeD, 0x10, &failing, 2);
    rig.take();
    assert_eq!(first(&mut rig, &mut chan, Engine::ThreeD, 2), Some(false));
    // Starts past the code store count as one macro.
    rig.stream(
        &mut chan,
        Engine::ThreeD,
        THREE_D_METHODS.start_pointer,
        &[5],
    );
    let past = u32::try_from(CODE_INSTRUCTIONS).unwrap();
    rig.stream(
        &mut chan,
        Engine::ThreeD,
        THREE_D_METHODS.start,
        &[past, past + 7],
    );
    let report = rig.call(&mut chan, Engine::ThreeD, 5, &[0]);
    assert_eq!(
        report.result,
        CallResult::Failed {
            error: CallFailure::Macro(MacroError::PcOutOfRange { pc: 0 }),
            first_for_macro: true,
        }
    );
    assert_eq!(first(&mut rig, &mut chan, Engine::ThreeD, 6), Some(false));
    // The last instruction is still inside: its own key.
    rig.stream(
        &mut chan,
        Engine::ThreeD,
        THREE_D_METHODS.start_pointer,
        &[7],
    );
    rig.stream(
        &mut chan,
        Engine::ThreeD,
        THREE_D_METHODS.start,
        &[past - 1],
    );
    assert_eq!(first(&mut rig, &mut chan, Engine::ThreeD, 7), Some(true));
    // A compute call on the 3D store is the 3D store's macro: already logged.
    assert_eq!(first(&mut rig, &mut chan, Engine::Compute, 0), Some(false));
    // The compute store's macro at the same start is another macro.
    rig.upload(&mut chan, Engine::Compute, 0x10, &failing, 0);
    rig.take();
    assert_eq!(first(&mut rig, &mut chan, Engine::Compute, 0), Some(true));
    assert_eq!(first(&mut rig, &mut chan, Engine::Compute, 0), Some(false));
}

#[test]
fn a_branch_before_the_start_fails() {
    // Divergence: Eden-NX2 runs within the whole code RAM, where this would reach the
    // instruction before the start.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let back = crate::mme::testing::branch(AluOp::Beq, Reg::Zero, Reg::Zero, true, -1);
    rig.upload(&mut chan, Engine::ThreeD, 0x10, &finish(vec![back]), 0);
    rig.take();
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &[0]);
    assert_eq!(
        report.result,
        CallResult::Failed {
            error: CallFailure::Macro(MacroError::BranchOutOfRange { pc: 0, offset: -1 }),
            first_for_macro: true,
        }
    );
}

#[test]
fn writes_are_bounded_per_run() {
    // `count` same-value writes to a state register (dropped from the ring), in a loop.
    let program = |count: u32| {
        let state = u32::from(out_method(2));
        assert!(chan_file().is_state(state));
        finish(vec![
            mov_reg(0, Reg::Load0),
            crate::mme::testing::loop_(r(0), 1),
            emit_imm(m16(state), 0),
            emit_imm(out_method(3), u16::try_from(count & 0xff).unwrap()),
        ])
    };
    let limit = u32::try_from(MAX_RUN_WRITES).unwrap();
    // `limit - 1` loop writes and the last: exactly at the limit.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program(1), 0);
    rig.take();
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &[limit - 1]);
    assert_eq!(committed(&report), Some(CallOutcome::Stored));
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &[limit]);
    assert_eq!(
        report.result,
        CallResult::Failed {
            error: CallFailure::TooManyWrites,
            first_for_macro: true,
        }
    );
}

fn chan_file() -> RegFile {
    RegFile::new(A, three_d())
}

// ---------------------------------------------------------------------------------------------
// The memo
// ---------------------------------------------------------------------------------------------

#[test]
fn a_memo_hit_equals_the_interpreters_result_and_an_upload_clears_the_memo() {
    let mut rig = make_rig();
    let mut a = Chan::new(A);
    let mut b = Chan::new(B);
    // Reads data RAM 3 (the guard), writes data RAM 4, and emits from both.
    let program = finish(vec![
        mov_reg(0, Reg::Load0),
        dread_to(1, 3),
        store(4, 0),
        emit_reg(out_method(0), 0),
        emit_reg(out_method(1), 1),
        emit_imm(out_method(2), 0x22),
    ]);
    rig.upload(&mut a, Engine::ThreeD, 0, &program, 0);
    rig.take();
    let interpreted = rig.call(&mut a, Engine::ThreeD, 0, &[0x5150]);
    assert_eq!(committed(&interpreted), Some(CallOutcome::Stored));
    let from_interpreter = rig.take();
    // B's identical call is replayed from the memo the two channels share.
    let replayed = rig.call(&mut b, Engine::ThreeD, 0, &[0x5150]);
    assert_eq!(committed(&replayed), Some(CallOutcome::Replayed));
    let from_memo = rig.take();
    let as_a: Vec<_> = from_memo
        .into_iter()
        .map(|entry| match entry {
            RingEntry::Write {
                class,
                method,
                value,
                ..
            } => RingEntry::Write {
                channel: A,
                class,
                method,
                value,
            },
            other => other,
        })
        .collect();
    assert_eq!(as_a, from_interpreter);
    assert_eq!(a.macros.data_ram(), b.macros.data_ram());
    for index in 0..3 {
        let method = u32::from(out_method(index));
        assert_eq!(a.three_d.get(method), b.three_d.get(method));
    }

    // An instruction word uploaded clears the memo: the same call is interpreted again, with
    // the new code.
    let changed = finish(vec![mov_reg(0, Reg::Load0), emit_imm(out_method(5), 0x55)]);
    rig.upload(&mut a, Engine::ThreeD, 0, &changed, 0);
    rig.take();
    let mut c = Chan::new(ChannelId(13));
    let report = rig.call(&mut c, Engine::ThreeD, 0, &[0x5150]);
    assert_eq!(committed(&report), Some(CallOutcome::Stored));
    assert_eq!(
        rig.take(),
        [write_entry(
            ChannelId(13),
            Engine::ThreeD,
            out_method(5),
            0x55
        )]
    );
    // A bind alone does not clear it.
    rig.stream(&mut a, Engine::ThreeD, THREE_D_METHODS.start_pointer, &[1]);
    rig.stream(&mut a, Engine::ThreeD, THREE_D_METHODS.start, &[0]);
    rig.take();
    let mut d = Chan::new(ChannelId(15));
    let report = rig.call(&mut d, Engine::ThreeD, 0, &[0x5150]);
    assert_eq!(committed(&report), Some(CallOutcome::Replayed));
}

// ---------------------------------------------------------------------------------------------
// MME DMA reads and the catch-up
// ---------------------------------------------------------------------------------------------

#[test]
fn a_dma_read_with_nothing_pending_reads_at_once() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &dma_macro(), 0);
    rig.take();
    rig.write_memory(SOURCE, &[0xdead_0001, 0xdead_0002, 0xdead_0003]);
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &source_parameters(SOURCE, 2));
    assert_eq!(
        committed(&report),
        Some(CallOutcome::NotStored(NotMemoised::Barrier))
    );
    assert_eq!(rig.back.queries.get(), 1);
    let sent = rig.take();
    assert_eq!(
        sent[sent.len() - 2..],
        [
            write_entry(A, Engine::ThreeD, out_method(0), 0xdead_0001),
            write_entry(A, Engine::ThreeD, out_method(1), 0xdead_0002),
        ]
    );
    assert_eq!(chan.macros.data_ram().get(5), Some(0x55));
    assert!(!chan.macros.is_waiting());
}

#[test]
fn a_dma_read_with_pending_back_work_commits_nothing_and_reruns_after_the_catch_up() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0x30, &dma_macro(), 4);
    rig.take();
    rig.write_memory(SOURCE, &[1, 2]);
    let before = snapshot(&chan.three_d);
    rig.back.busy.set(true);
    let parameters = source_parameters(SOURCE, 2);
    let report = rig.call(&mut chan, Engine::ThreeD, 4, &parameters);
    assert_eq!(report.result, CallResult::CatchUp);
    assert_eq!((report.entry, report.start), (4, 0x30));
    assert_eq!(rig.back.queries.get(), 1);
    // Nothing committed: no entries, the file and data RAM as before.
    assert_eq!(rig.take(), []);
    assert_eq!(snapshot(&chan.three_d), before);
    assert_eq!(chan.macros.data_ram().get(5), Some(0));
    assert!(chan.macros.is_waiting());
    assert_eq!(chan.macros.waiting_engine(), Some(Engine::ThreeD));

    // The channel takes no word while the call waits.
    assert_eq!(
        rig.word(&mut chan, Engine::ThreeD, out_method(7), 1, 0),
        Err(HostError::Waiting)
    );
    assert_eq!(snapshot(&chan.three_d), before);

    // The back performs its work, which writes the words the macro reads.
    rig.write_memory(SOURCE, &[0x1111, 0x2222]);
    let report = rig.resume(&mut chan, Engine::ThreeD).unwrap();
    assert_eq!(
        report.result,
        CallResult::Committed(CallOutcome::NotStored(NotMemoised::Barrier))
    );
    // The rerun did not ask the back again, even though it still says it is busy.
    assert_eq!(rig.back.queries.get(), 1);
    assert!(!chan.macros.is_waiting());
    assert_eq!(chan.macros.waiting_engine(), None);
    let sent = rig.take();
    assert_eq!(
        sent[sent.len() - 2..],
        [
            write_entry(A, Engine::ThreeD, out_method(0), 0x1111),
            write_entry(A, Engine::ThreeD, out_method(1), 0x2222),
        ]
    );
    assert!(sent.contains(&write_entry(A, Engine::ThreeD, out_method(9), 0x99)));
    assert_eq!(chan.macros.data_ram().get(5), Some(0x55));
    // A second resume has nothing to run.
    assert_eq!(
        rig.resume(&mut chan, Engine::ThreeD),
        Err(HostError::NotWaiting)
    );
}

#[test]
fn a_waiting_call_reruns_the_start_it_was_called_with() {
    let mut rig = make_rig();
    let mut a = Chan::new(A);
    let mut b = Chan::new(B);
    rig.upload(&mut a, Engine::ThreeD, 0x30, &dma_macro(), 4);
    // Another macro, uploaded before A's call; B binds it to entry 5 for now.
    rig.upload(&mut b, Engine::ThreeD, 0x80, &echo(3), 5);
    rig.take();
    let generation = rig.process.code_generation(Engine::ThreeD);
    rig.write_memory(SOURCE, &[1, 2]);
    rig.back.busy.set(true);
    let parameters = source_parameters(SOURCE, 2);
    let report = rig.call(&mut a, Engine::ThreeD, 4, &parameters);
    assert_eq!(report.result, CallResult::CatchUp);
    assert_eq!((report.entry, report.start), (4, 0x30));

    // While A waits, B binds entry 4 to its own macro (no code word is uploaded).
    rig.stream(&mut b, Engine::ThreeD, THREE_D_METHODS.start_pointer, &[4]);
    rig.stream(&mut b, Engine::ThreeD, THREE_D_METHODS.start, &[0x80]);
    assert_eq!(rig.process.starts(Engine::ThreeD)[4], 0x80);
    assert_eq!(rig.process.code_generation(Engine::ThreeD), generation);
    rig.take();

    // The rerun runs the macro A called, at the start its first run used.
    rig.write_memory(SOURCE, &[0x1111, 0x2222]);
    let report = rig.resume(&mut a, Engine::ThreeD).unwrap();
    assert_eq!(
        (report.store, report.entry, report.start, report.result),
        (
            Engine::ThreeD,
            4,
            0x30,
            CallResult::Committed(CallOutcome::NotStored(NotMemoised::Barrier)),
        )
    );
    let sent = rig.take();
    assert_eq!(
        sent[sent.len() - 2..],
        [
            write_entry(A, Engine::ThreeD, out_method(0), 0x1111),
            write_entry(A, Engine::ThreeD, out_method(1), 0x2222),
        ]
    );
    // B's bind stays: A's next call on entry 4 runs B's macro.
    rig.back.busy.set(false);
    let report = rig.call(&mut a, Engine::ThreeD, 4, &[0xa, 0xb, 0xc]);
    assert_eq!((report.entry, report.start), (4, 0x80));
    assert_eq!(
        rig.take(),
        [
            write_entry(A, Engine::ThreeD, out_method(0), 0xa),
            write_entry(A, Engine::ThreeD, out_method(1), 0xb),
            write_entry(A, Engine::ThreeD, out_method(2), 0xc),
        ]
    );
}

#[test]
fn a_waiting_call_reruns_the_code_as_it_is_at_the_resume() {
    let mut rig = make_rig();
    let mut a = Chan::new(A);
    let mut b = Chan::new(B);
    rig.upload(&mut a, Engine::ThreeD, 0x30, &dma_macro(), 4);
    rig.take();
    rig.back.busy.set(true);
    let parameters = source_parameters(SOURCE, 2);
    let report = rig.call(&mut a, Engine::ThreeD, 4, &parameters);
    assert_eq!(report.result, CallResult::CatchUp);

    // While A waits, B uploads other code over A's macro, bound to another entry.
    rig.upload(&mut b, Engine::ThreeD, 0x30, &echo(3), 5);
    assert_eq!(rig.process.starts(Engine::ThreeD)[4], 0x30);
    rig.take();

    // The rerun runs the code now at A's start (divergence: Eden-NX2 ran the call at once).
    let report = rig.resume(&mut a, Engine::ThreeD).unwrap();
    assert_eq!((report.entry, report.start), (4, 0x30));
    assert_eq!(report.result, CallResult::Committed(CallOutcome::Stored));
    assert_eq!(
        rig.take(),
        [
            write_entry(A, Engine::ThreeD, out_method(0), parameters[0]),
            write_entry(A, Engine::ThreeD, out_method(1), parameters[1]),
            write_entry(A, Engine::ThreeD, out_method(2), parameters[2]),
        ]
    );
}

#[test]
fn the_back_is_asked_only_at_a_barrier_with_reads_queued() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    // A barrier with no read queued, then a read of 0 words and a barrier.
    let program = finish(vec![
        barrier(),
        emit_imm(m16(DMA_READ_FIFOED), 0),
        barrier(),
        emit_imm(out_method(0), 1),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    rig.take();
    rig.back.busy.set(true);
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &[0]);
    assert!(matches!(report.result, CallResult::Committed(_)));
    assert_eq!(rig.back.queries.get(), 0);
    // A read queued without a barrier: never asked, never read, and the run commits.
    let program = finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_param(m16(DMA_READ_FIFOED)),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0x10, &program, 1);
    rig.take();
    let report = rig.call(&mut chan, Engine::ThreeD, 1, &source_parameters(0, 2));
    assert!(matches!(report.result, CallResult::Committed(_)));
    assert_eq!(rig.back.queries.get(), 0);
}

#[test]
fn dma_reads_are_3d_only() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(
        &mut chan,
        Engine::Compute,
        0,
        &finish(vec![
            emit_param(m16(MEM_ADDRESS_A)),
            emit_param(m16(MEM_ADDRESS_B)),
            emit_param(m16(DMA_READ_FIFOED)),
            barrier(),
            emit_imm(out_method(0), 1),
        ]),
        0,
    );
    rig.take();
    rig.back.busy.set(true);
    let report = rig.call(&mut chan, Engine::Compute, 0, &source_parameters(SOURCE, 2));
    assert!(matches!(report.result, CallResult::Committed(_)));
    assert_eq!(rig.back.queries.get(), 0);
}

#[test]
fn dma_words_arrive_only_at_a_barrier() {
    // Divergence: Eden-NX2 reads at the MME_DMA_READ_FIFOED write.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let program = finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_param(m16(DMA_READ_FIFOED)),
        mov_reg(0, Reg::Load0),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    rig.take();
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &source_parameters(SOURCE, 1));
    assert_eq!(
        report.result,
        CallResult::Failed {
            error: CallFailure::Macro(MacroError::ParameterUnderflow { pc: 3 }),
            first_for_macro: true,
        }
    );
}

#[test]
fn the_dma_address_uses_the_headers_upper_field() {
    // Divergence: Eden-NX2 masks SET_MME_MEM_ADDRESS_A with 0x1ffffff; the header's UPPER is
    // 16:0, so bit 17 is not part of the address.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &dma_macro(), 0);
    rig.take();
    rig.write_memory(SOURCE, &[0x51, 0x52]);
    let mut parameters = source_parameters(SOURCE, 2);
    parameters[0] |= 1 << 17;
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &parameters);
    assert!(matches!(report.result, CallResult::Committed(_)));
    let sent = rig.take();
    assert_eq!(
        sent.last(),
        Some(&write_entry(A, Engine::ThreeD, out_method(1), 0x52))
    );
    // Bit 16 is: the address leaves the mapping.
    let mut parameters = source_parameters(SOURCE, 2);
    parameters[0] |= 1 << 16;
    let address = SOURCE | (1 << 48);
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &parameters);
    assert_eq!(
        report.result,
        CallResult::Failed {
            error: CallFailure::DmaRead {
                address,
                words: 2,
                fault: GpuFault::Unmapped { va: address },
            },
            first_for_macro: true,
        }
    );
}

#[test]
fn an_unreadable_dma_read_fails_the_call() {
    // Divergence: Eden-NX2's reads have no failure result.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &dma_macro(), 0);
    rig.take();
    let before = snapshot(&chan.three_d);
    // The second word is the first unmapped one.
    let end = GPU_BASE + PAGES * PAGE_SIZE;
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &source_parameters(end - 4, 2));
    assert_eq!(
        report.result,
        CallResult::Failed {
            error: CallFailure::DmaRead {
                address: end - 4,
                words: 2,
                fault: GpuFault::Unmapped { va: end },
            },
            first_for_macro: true,
        }
    );
    assert_eq!(rig.take(), []);
    assert_eq!(snapshot(&chan.three_d), before);
    assert_eq!(chan.macros.data_ram().get(5), Some(0));
    // Ending exactly at the mapping's end reads.
    rig.write_memory(end - 8, &[0x81, 0x82]);
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &source_parameters(end - 8, 2));
    assert!(matches!(report.result, CallResult::Committed(_)));
}

#[test]
fn dma_reads_are_bounded_per_run() {
    // Two reads: `first` words, then `second`; the run then loads one word and emits it.
    let program = finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_param(m16(DMA_READ_FIFOED)),
        emit_param(m16(DMA_READ_FIFOED)),
        barrier(),
        mov_reg(0, Reg::Load0),
        emit_reg(out_method(0), 0),
    ]);
    let run = |first: u32, second: u32| {
        let mut rig = make_rig();
        let mut chan = Chan::new(A);
        rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
        rig.write_memory(GPU_BASE, &[0x7e57]);
        let [upper, lower, _] = source_parameters(GPU_BASE, 0);
        rig.call(&mut chan, Engine::ThreeD, 0, &[upper, lower, first, second])
            .result
    };
    let half = DMA_RUN_MAX_WORDS / 2;
    assert!(matches!(run(half, half), CallResult::Committed(_)));
    assert_eq!(
        run(half, half + 1),
        CallResult::Failed {
            error: CallFailure::DmaTooLarge {
                words: DMA_RUN_MAX_WORDS + 1,
            },
            first_for_macro: true,
        }
    );
    // One read past DMA_READ_MAX_WORDS is cut to it, and so fits the run's bound alone.
    assert!(matches!(
        run(DMA_READ_MAX_WORDS + 1, 0),
        CallResult::Committed(_)
    ));
    assert_eq!(
        run(DMA_READ_MAX_WORDS + 1, 1),
        CallResult::Failed {
            error: CallFailure::DmaTooLarge {
                words: DMA_RUN_MAX_WORDS + 1,
            },
            first_for_macro: true,
        }
    );
}

#[test]
fn a_dma_read_is_cut_to_its_maximum() {
    // Exactly DMA_READ_MAX_WORDS words are mapped before an unmapped page: a read of one more
    // would fault if it were not cut.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &dma_macro(), 0);
    rig.take();
    let bytes = u64::from(DMA_READ_MAX_WORDS) * u64::from(WORD_BYTES);
    let end = GPU_BASE + PAGES * PAGE_SIZE;
    let start = end - bytes;
    rig.write_memory(start, &[0x61, 0x62]);
    let report = rig.call(
        &mut chan,
        Engine::ThreeD,
        0,
        &source_parameters(start, DMA_READ_MAX_WORDS + 1),
    );
    assert!(matches!(report.result, CallResult::Committed(_)));
    let sent = rig.take();
    assert_eq!(
        sent.last(),
        Some(&write_entry(A, Engine::ThreeD, out_method(1), 0x62))
    );
    // One word earlier and one word more is past the mapping only if not cut: still fine.
    let report = rig.call(
        &mut chan,
        Engine::ThreeD,
        0,
        &source_parameters(start + 4, DMA_READ_MAX_WORDS),
    );
    assert!(matches!(
        report.result,
        CallResult::Failed {
            error: CallFailure::DmaRead { .. },
            ..
        }
    ));
}

#[test]
fn a_waiting_call_holds_its_channel_at_the_word_after_it() {
    // Through the real pusher: the call, then a write. The pusher stops after the call; the
    // write is read again after the catch-up, so a word rewritten meanwhile is seen.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &dma_macro(), 0);
    rig.take();
    rig.write_memory(SOURCE, &[0x31, 0x32]);
    let mut channel = Channel::new(A);
    let mut pb = PushBuffer::new();
    pb.methods(MethodMode::Incrementing, 0, 0, &[clc797::CLASS])
        .unwrap();
    pb.methods(
        MethodMode::IncrementOnce,
        0,
        call_method(Engine::ThreeD, 0),
        &source_parameters(SOURCE, 2),
    )
    .unwrap();
    let after = pb.words().len();
    pb.immediate(0, out_method(7), 0x70).unwrap();
    rig.write_memory(CMD, pb.words());
    channel.start_entry(entry(CMD, pb.words().len())).unwrap();
    rig.back.busy.set(true);
    let (outcome, reports) = rig.pump(&mut channel, &mut chan);
    assert_eq!(outcome, Ok(RunOutcome::Held));
    assert!(matches!(
        reports.as_slice(),
        [
            WordOutcome::Parameter,
            WordOutcome::Parameter,
            WordOutcome::Call(CallReport {
                result: CallResult::CatchUp,
                ..
            }),
        ]
    ));
    assert!(matches!(rig.take().as_slice(), [RingEntry::Bind { .. }]));
    // Still held: nothing is read.
    let (outcome, reports) = rig.pump(&mut channel, &mut chan);
    assert_eq!(outcome, Ok(RunOutcome::Held));
    assert_eq!(reports, []);

    let mut new = PushBuffer::new();
    new.immediate(0, out_method(7), 0x71).unwrap();
    rig.write_memory(CMD + 4 * u64::try_from(after).unwrap(), new.words());
    let report = rig.resume(&mut chan, Engine::ThreeD).unwrap();
    assert!(matches!(report.result, CallResult::Committed(_)));
    let (outcome, reports) = rig.pump(&mut channel, &mut chan);
    assert_eq!(outcome, Ok(RunOutcome::EntryDone));
    assert_eq!(reports, [WordOutcome::Register]);
    let sent = rig.take();
    assert_eq!(
        sent[sent.len() - 3..],
        [
            write_entry(A, Engine::ThreeD, out_method(0), 0x31),
            write_entry(A, Engine::ThreeD, out_method(1), 0x32),
            write_entry(A, Engine::ThreeD, out_method(7), 0x71),
        ]
    );
}

// ---------------------------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------------------------

#[test]
fn a_word_with_the_wrong_file_is_refused_after_the_waiting_check() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let env = MacroEnv {
        memory: &rig.mem,
        back: &rig.back,
    };
    // Another channel's file, and the compute file for a 3D word.
    let mut other = RegFile::new(B, three_d());
    let error = rig.process.write(
        &mut chan.macros,
        &mut other,
        word(Engine::ThreeD, out_method(0), 1, 0),
        &env,
        &mut rig.out,
    );
    assert_eq!(
        error,
        Err(HostError::WrongFile {
            file_channel: B.0,
            file_class: clc797::CLASS,
            channel: A.0,
            class: clc797::CLASS,
        })
    );
    let error = rig.process.write(
        &mut chan.macros,
        &mut chan.compute,
        word(Engine::ThreeD, out_method(0), 1, 0),
        &env,
        &mut rig.out,
    );
    assert_eq!(
        error,
        Err(HostError::WrongFile {
            file_channel: A.0,
            file_class: clc7c0::CLASS,
            channel: A.0,
            class: clc797::CLASS,
        })
    );
    assert_eq!(rig.out, []);
    assert_eq!(chan.compute.get(u32::from(out_method(0))), Some(0));

    // Waiting and the wrong file at once: waiting is reported.
    let mut rig = make_rig();
    rig.upload(&mut chan, Engine::ThreeD, 0, &dma_macro(), 0);
    rig.back.busy.set(true);
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &source_parameters(SOURCE, 2));
    assert_eq!(report.result, CallResult::CatchUp);
    let env = MacroEnv {
        memory: &rig.mem,
        back: &rig.back,
    };
    let error = rig.process.write(
        &mut chan.macros,
        &mut other,
        word(Engine::ThreeD, out_method(0), 1, 0),
        &env,
        &mut rig.out,
    );
    assert_eq!(error, Err(HostError::Waiting));
    // Resume with the wrong file: refused, and the call still waits.
    let error = rig
        .process
        .resume(&mut chan.macros, &mut chan.compute, &env, &mut rig.out);
    assert_eq!(
        error,
        Err(HostError::WrongFile {
            file_channel: A.0,
            file_class: clc7c0::CLASS,
            channel: A.0,
            class: clc797::CLASS,
        })
    );
    assert!(chan.macros.is_waiting());
    // Not waiting and the wrong file at once: not waiting is reported.
    let mut fresh = MacroChannel::new(A);
    let error = rig
        .process
        .resume(&mut fresh, &mut other, &env, &mut rig.out);
    assert_eq!(error, Err(HostError::NotWaiting));
}

#[test]
fn other_classes_write_their_file_and_never_call() {
    let mut rig = make_rig();
    let mut macros = MacroChannel::new(A);
    let class = ClassId(crate::classes::cla140::CLASS);
    let mut file = RegFile::new(A, class);
    let env = MacroEnv {
        memory: &rig.mem,
        back: &rig.back,
    };
    let word = EngineWord {
        subchannel: Subchannel::new(2).unwrap(),
        class,
        method: 0xe00,
        value: 5,
        remaining: 3,
    };
    let outcome = rig
        .process
        .write(&mut macros, &mut file, word, &env, &mut rig.out);
    assert_eq!(outcome, Ok(WordOutcome::Register));
    assert_eq!(file.get(0xe00), Some(5));
    let upload = EngineWord {
        method: m16(THREE_D_METHODS.instruction),
        remaining: 0,
        ..word
    };
    rig.process
        .write(&mut macros, &mut file, upload, &env, &mut rig.out)
        .unwrap();
    assert_eq!(rig.process.code_generation(Engine::ThreeD), 0);
    assert_eq!(rig.out.len(), 2);
}

#[test]
fn the_dma_address_is_the_one_held_at_the_read_write() {
    // A write to SET_MME_MEM_ADDRESS_B after MME_DMA_READ_FIFOED does not move the read.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let program = finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_param(m16(DMA_READ_FIFOED)),
        emit_param(m16(MEM_ADDRESS_B)),
        barrier(),
        mov_reg(0, Reg::Load0),
        emit_reg(out_method(0), 0),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    rig.take();
    rig.write_memory(SOURCE, &[0x1010]);
    rig.write_memory(SOURCE + 0x100, &[0x2020]);
    let [upper, lower, _] = source_parameters(SOURCE, 0);
    let report = rig.call(
        &mut chan,
        Engine::ThreeD,
        0,
        &[upper, lower, 1, lower + 0x100],
    );
    assert!(matches!(report.result, CallResult::Committed(_)));
    assert_eq!(
        rig.take().last(),
        Some(&write_entry(A, Engine::ThreeD, out_method(0), 0x1010))
    );
}

#[test]
fn dma_reads_fill_the_fifo_in_queue_order() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let program = finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_param(m16(DMA_READ_FIFOED)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_param(m16(DMA_READ_FIFOED)),
        barrier(),
        mov_reg(0, Reg::Load0),
        mov_reg(1, Reg::Load0),
        mov_reg(2, Reg::Load0),
        emit_reg(out_method(0), 0),
        emit_reg(out_method(1), 1),
        emit_reg(out_method(2), 2),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    rig.take();
    rig.write_memory(SOURCE, &[0xa1, 0xa2]);
    rig.write_memory(SOURCE + 0x100, &[0xb1]);
    let [upper, lower, _] = source_parameters(SOURCE, 0);
    let report = rig.call(
        &mut chan,
        Engine::ThreeD,
        0,
        &[upper, lower, 2, lower + 0x100, 1],
    );
    assert!(matches!(report.result, CallResult::Committed(_)));
    let sent = rig.take();
    assert_eq!(
        sent[sent.len() - 3..],
        [
            write_entry(A, Engine::ThreeD, out_method(0), 0xa1),
            write_entry(A, Engine::ThreeD, out_method(1), 0xa2),
            write_entry(A, Engine::ThreeD, out_method(2), 0xb1),
        ]
    );
}

#[test]
fn a_macros_uploads_and_binds_apply_in_order_when_it_commits() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let methods = &THREE_D_METHODS;
    let program = finish(vec![
        emit_imm(m16(methods.instruction_pointer), 0x50),
        emit_param(m16(methods.instruction)),
        emit_param(m16(methods.instruction)),
        emit_param(m16(methods.instruction)),
        emit_imm(m16(methods.start_pointer), 6),
        emit_imm(m16(methods.start), 0x50),
        // Fails when a fourth parameter is missing.
        mov_reg(0, Reg::Load0),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    rig.take();
    let generation = rig.process.code_generation(Engine::ThreeD);
    let mut b = Chan::new(B);
    let report = rig.call(&mut b, Engine::ThreeD, 0, &[0xa, 0xb, 0xc]);
    assert!(matches!(report.result, CallResult::Failed { .. }));
    assert_eq!(
        rig.process
            .code_word(Engine::ThreeD, 0x50 * INSTRUCTION_WORDS),
        Some(0)
    );
    assert_eq!(rig.process.starts(Engine::ThreeD)[6], 0);
    assert_eq!(rig.process.code_generation(Engine::ThreeD), generation);
    assert_eq!(
        rig.process.uploads(B, Engine::ThreeD),
        ChannelUploads::default()
    );

    let report = rig.call(&mut b, Engine::ThreeD, 0, &[0xa, 0xb, 0xc, 0]);
    assert!(matches!(report.result, CallResult::Committed(_)));
    let base = 0x50 * INSTRUCTION_WORDS;
    for (offset, value) in [(base, 0xa), (base + 1, 0xb), (base + 2, 0xc), (base + 3, 0)] {
        assert_eq!(rig.process.code_word(Engine::ThreeD, offset), Some(value));
    }
    assert_eq!(rig.process.starts(Engine::ThreeD)[6], 0x50);
    assert_eq!(rig.process.code_generation(Engine::ThreeD), generation + 3);
    assert_eq!(
        rig.process.uploads(B, Engine::ThreeD),
        ChannelUploads {
            code_words: 3,
            start_entries: 1,
        }
    );
}

#[test]
fn the_journal_keeps_one_value_per_word() {
    // However many writes it covers, the journal holds each word's first value once.
    let mut file = RegFile::new(A, three_d());
    let mut journal = Journal::default();
    let cbuf = clc797::LOAD_CONSTANT_BUFFER.method_at(&[0]).unwrap();
    let mut sink = Vec::new();
    for value in 1..100 {
        file.save(cbuf, &mut journal);
        file.write(cbuf, value, &mut sink).unwrap();
    }
    // The register, its shadow word and LOAD_CONSTANT_BUFFER_OFFSET.
    assert_eq!(journal.len(), 3);
    file.undo(&mut journal);
    assert_eq!(snapshot(&file), snapshot(&RegFile::new(A, three_d())));
    assert_eq!(journal.len(), 0);
}

#[test]
fn the_3d_and_compute_classes_collect_calls_separately() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &echo(2), 0);
    rig.upload(&mut chan, Engine::ThreeD, 0x10, &echo(1), 1);
    rig.take();
    // A 3D call is half collected when a compute call comes and goes.
    let three_d = call_method(Engine::ThreeD, 0);
    assert_eq!(
        rig.word(&mut chan, Engine::ThreeD, three_d, 0x31, 1),
        Ok(WordOutcome::Parameter)
    );
    let report = rig.call(&mut chan, Engine::Compute, 1, &[0x41]);
    assert_eq!((report.engine, report.entry), (Engine::Compute, 1));
    assert_eq!(chan.macros.collected(Engine::ThreeD), [0x31]);
    assert_eq!(chan.macros.collected(Engine::Compute), []);
    let outcome = rig
        .word(&mut chan, Engine::ThreeD, three_d + 1, 0x32, 0)
        .unwrap();
    assert!(matches!(
        outcome,
        WordOutcome::Call(CallReport {
            engine: Engine::ThreeD,
            entry: 0,
            ..
        })
    ));
    assert_eq!(
        rig.take(),
        [
            write_entry(A, Engine::Compute, out_method(0), 0x41),
            write_entry(A, Engine::ThreeD, out_method(0), 0x31),
            write_entry(A, Engine::ThreeD, out_method(1), 0x32),
        ]
    );
}

// ---------------------------------------------------------------------------------------------
// MME_DMA_READ into data RAM
// ---------------------------------------------------------------------------------------------

/// A data RAM word the read macros write before their read, away from the words they read.
const SCRATCH_WORD: u16 = 0x10;
/// Where the read macros read to in most tests.
const RAM_AT: u32 = 0x20;

/// `R<dst>` = data RAM `[R<index>]`.
fn dread_reg(dst: u8, index: u8) -> Instruction {
    inst0(AluOp::Dread, r(dst), r(index), Reg::Zero)
}

/// The data RAM read macro: parameters `[upper, lower, ram index, length]` go to
/// `SET_MME_MEM_ADDRESS_A/B`, `SET_MME_DATA_RAM_ADDRESS` and `MME_DMA_READ`, after a data RAM
/// write to `SCRATCH_WORD` and a write to scratch 9 (to show they are not committed by a run
/// that is not). With `read_back`, the word at the read's index is then read with `DREAD`, with
/// no barrier, and emitted to scratch 0.
fn ram_read_macro(read_back: bool) -> Vec<Instruction> {
    let mut program = vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        mov_reg(2, Reg::Load0),
        emit_reg(m16(DATA_RAM_ADDRESS), 2),
        mov(3, 0x55),
        store(SCRATCH_WORD, 3),
        emit_imm(out_method(9), 0x99),
        emit_param(m16(DMA_READ)),
    ];
    if read_back {
        program.extend([dread_reg(0, 2), emit_reg(out_method(0), 0)]);
    }
    finish(program)
}

fn ram_parameters(address: u64, index: u32, words: u32) -> [u32; 4] {
    let [upper, lower, _] = source_parameters(address, 0);
    [upper, lower, index, words]
}

/// Words that differ in every byte, so a swapped or narrowed read shows.
const WORDS: [u32; 3] = [0xa1b2_c3d4, 0x5566_7788, 0x0bad_f00d];

/// The data RAM words from `from` for `count` words.
fn ram_words(chan: &Chan, from: u32, count: u32) -> Vec<u32> {
    (from..from + count)
        .map(|index| chan.macros.data_ram().get(index).unwrap())
        .collect()
}

/// Remembered calls in `store`'s memo.
fn memo_len(rig: &Rig, store: Engine) -> usize {
    rig.process.store(store).memo.len()
}

#[test]
fn a_data_ram_read_with_nothing_pending_lands_at_once() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &ram_read_macro(true), 0);
    rig.take();
    rig.write_memory(SOURCE, &WORDS);
    let report = rig.call(
        &mut chan,
        Engine::ThreeD,
        0,
        &ram_parameters(SOURCE, RAM_AT, 3),
    );
    assert_eq!(
        report.result,
        CallResult::Committed(CallOutcome::NotStored(NotMemoised::DataRamLoad))
    );
    assert_eq!(rig.back.queries.get(), 1);
    // The exact words at the exact place, and nothing on either side.
    assert_eq!(
        ram_words(&chan, RAM_AT - 1, 5),
        [0, WORDS[0], WORDS[1], WORDS[2], 0]
    );
    assert_eq!(
        chan.macros.data_ram().get(u32::from(SCRATCH_WORD)),
        Some(0x55)
    );
    // The macro's DREAD right after the write saw the first word, without a barrier.
    let sent = rig.take();
    assert_eq!(
        sent.last(),
        Some(&write_entry(A, Engine::ThreeD, out_method(0), WORDS[0]))
    );
    // The read method goes to the ring as any write.
    assert!(sent.contains(&write_entry(A, Engine::ThreeD, m16(DMA_READ), 3)));
    assert!(!chan.macros.is_waiting());
    assert_eq!(memo_len(&rig, Engine::ThreeD), 0);
}

#[test]
fn a_data_ram_read_with_pending_back_work_commits_nothing_and_reruns_after_the_catch_up() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0x30, &ram_read_macro(true), 4);
    rig.take();
    rig.write_memory(SOURCE, &[1, 2, 3]);
    let before = snapshot(&chan.three_d);
    let ram_before = chan.macros.data_ram().clone();
    rig.back.busy.set(true);
    let parameters = ram_parameters(SOURCE, RAM_AT, 3);
    let report = rig.call(&mut chan, Engine::ThreeD, 4, &parameters);
    assert_eq!(report.result, CallResult::CatchUp);
    assert_eq!((report.entry, report.start), (4, 0x30));
    assert_eq!(rig.back.queries.get(), 1);
    // Nothing committed: no entries, the file and data RAM (the scratch word too) as before.
    assert_eq!(rig.take(), []);
    assert_eq!(snapshot(&chan.three_d), before);
    assert_eq!(chan.macros.data_ram(), &ram_before);
    assert!(chan.macros.is_waiting());
    // The abandoned run was not remembered either.
    assert_eq!(memo_len(&rig, Engine::ThreeD), 0);

    // The back performs its work, which writes the words the macro reads.
    rig.write_memory(SOURCE, &WORDS);
    let report = rig.resume(&mut chan, Engine::ThreeD).unwrap();
    assert_eq!(
        report.result,
        CallResult::Committed(CallOutcome::NotStored(NotMemoised::DataRamLoad))
    );
    // The rerun read without asking the back again, although it still says it is busy.
    assert_eq!(rig.back.queries.get(), 1);
    assert_eq!(ram_words(&chan, RAM_AT, 3), WORDS);
    assert_eq!(
        chan.macros.data_ram().get(u32::from(SCRATCH_WORD)),
        Some(0x55)
    );
    let sent = rig.take();
    assert_eq!(
        sent.last(),
        Some(&write_entry(A, Engine::ThreeD, out_method(0), WORDS[0]))
    );
    assert!(sent.contains(&write_entry(A, Engine::ThreeD, out_method(9), 0x99)));
    assert!(!chan.macros.is_waiting());
}

#[test]
fn a_call_that_reads_into_data_ram_is_never_replayed() {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &ram_read_macro(true), 0);
    rig.take();
    let parameters = ram_parameters(SOURCE, RAM_AT, 1);
    for (round, word) in WORDS.into_iter().enumerate() {
        // The same call, with the same data RAM as the round before left it, but new memory: a
        // replay would leave the round before's word.
        rig.write_memory(SOURCE, &[word]);
        let report = rig.call(&mut chan, Engine::ThreeD, 0, &parameters);
        assert_eq!(
            report.result,
            CallResult::Committed(CallOutcome::NotStored(NotMemoised::DataRamLoad)),
            "round {round}"
        );
        assert_eq!(chan.macros.data_ram().get(RAM_AT), Some(word));
        assert_eq!(
            rig.take().last(),
            Some(&write_entry(A, Engine::ThreeD, out_method(0), word))
        );
        assert_eq!(memo_len(&rig, Engine::ThreeD), 0);
    }
}

#[test]
fn an_empty_data_ram_read_reads_nothing_asks_nothing_and_is_not_remembered() {
    // As Eden-NX2's `read_memory`, any MME_DMA_READ write keeps the call out of the memo; a read
    // of 0 words is not checked against data RAM's end, so the index may be anything.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &ram_read_macro(false), 0);
    rig.take();
    rig.back.busy.set(true);
    for index in [RAM_AT, u32::MAX] {
        let unmapped = GPU_BASE + PAGES * PAGE_SIZE;
        let report = rig.call(
            &mut chan,
            Engine::ThreeD,
            0,
            &ram_parameters(unmapped, index, 0),
        );
        assert_eq!(
            report.result,
            CallResult::Committed(CallOutcome::NotStored(NotMemoised::DataRamLoad))
        );
        assert_eq!(rig.back.queries.get(), 0);
        assert_eq!(chan.macros.data_ram().get(RAM_AT), Some(0));
        assert_eq!(memo_len(&rig, Engine::ThreeD), 0);
    }
}

#[test]
fn a_failed_run_that_writes_the_read_method_is_not_remembered() {
    // The run fails at a write outside the file, then writes MME_DMA_READ while starved: the
    // read is not made, and the memo still does not remember the call (its replay would make
    // the read without storing its words).
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let program = finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_imm(call_method(Engine::ThreeD, 1), 2),
        emit_imm(m16(DATA_RAM_ADDRESS), 0x20),
        emit_imm(m16(DMA_READ), 1),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    rig.take();
    rig.write_memory(SOURCE, &WORDS);
    let [upper, lower, _] = source_parameters(SOURCE, 0);
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &[upper, lower]);
    assert_eq!(
        report.result,
        CallResult::Failed {
            error: CallFailure::MethodOutOfRange {
                method: call_method(Engine::ThreeD, 1),
            },
            first_for_macro: true,
        }
    );
    assert_eq!(memo_len(&rig, Engine::ThreeD), 0);
    assert_eq!(rig.back.queries.get(), 0);
    assert_eq!(chan.macros.data_ram().get(0x20), Some(0));
}

/// One data RAM read of `words` at `index` on a fresh rig, from a source with `WORDS` at its
/// start; returns the result and the channel.
fn ram_read(index: u32, words: u32) -> (CallResult, Chan, Vec<RingEntry<'static>>) {
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &ram_read_macro(false), 0);
    rig.take();
    rig.write_memory(GPU_BASE, &WORDS);
    let report = rig.call(
        &mut chan,
        Engine::ThreeD,
        0,
        &ram_parameters(GPU_BASE, index, words),
    );
    (report.result, chan, rig.take())
}

fn out_of_range(index: u32, words: u32) -> CallResult {
    CallResult::Failed {
        error: CallFailure::DataRamOutOfRange { index, words },
        first_for_macro: true,
    }
}

#[test]
fn a_data_ram_read_past_the_end_fails_the_call() {
    // Divergence: Eden-NX2 stores the words below the end and drops the rest.
    let end = u32::try_from(DATA_RAM_WORDS).unwrap();
    let loaded = CallResult::Committed(CallOutcome::NotStored(NotMemoised::DataRamLoad));
    // Ending exactly at the end: stored, the last word in data RAM's last word.
    let (result, chan, _) = ram_read(end - 3, 3);
    assert_eq!(result, loaded);
    assert_eq!(
        ram_words(&chan, end - 4, 4),
        [0, WORDS[0], WORDS[1], WORDS[2]]
    );
    // One word more: refused, and nothing stored or sent, the scratch word included.
    let (result, chan, sent) = ram_read(end - 3, 4);
    assert_eq!(result, out_of_range(end - 3, 4));
    assert_eq!(ram_words(&chan, end - 4, 4), [0; 4]);
    assert_eq!(chan.macros.data_ram().get(u32::from(SCRATCH_WORD)), Some(0));
    assert_eq!(sent, []);
    assert_eq!(chan.three_d.get(u32::from(out_method(9))), Some(0));
    assert_eq!(chan.three_d.get(DATA_RAM_ADDRESS), Some(0));
    // The index at the edge: the last word is accepted, the end refused.
    let (result, chan, _) = ram_read(end - 1, 1);
    assert_eq!(result, loaded);
    assert_eq!(chan.macros.data_ram().get(end - 1), Some(WORDS[0]));
    assert_eq!(ram_read(end, 1).0, out_of_range(end, 1));
    // A length with a high byte, at the edge on both sides.
    let length = 0x102;
    assert_eq!(ram_read(end - length, length).0, loaded);
    assert_eq!(
        ram_read(end - length + 1, length).0,
        out_of_range(end - length + 1, length)
    );
    // The length is cut to DMA_READ_MAX_WORDS before the check, as for a load-FIFO read.
    assert_eq!(ram_read(0, u32::MAX).0, out_of_range(0, DMA_READ_MAX_WORDS));
}

#[test]
fn a_data_ram_read_index_is_32_bits_wide() {
    // `SET_MME_DATA_RAM_ADDRESS_WORD` is 31:0: an index differing only above its low byte
    // lands elsewhere, and one at or above 2^31 or near u32::MAX is past the end (Eden-NX2's
    // `ram + i` would wrap that last one onto data RAM's start).
    let (_, chan, _) = ram_read(0xb20, 1);
    assert_eq!(chan.macros.data_ram().get(0xb20), Some(WORDS[0]));
    assert_eq!(chan.macros.data_ram().get(0x20), Some(0));
    assert_eq!(ram_read(1 << 31, 1).0, out_of_range(1 << 31, 1));
    assert_eq!(ram_read(u32::MAX, 2).0, out_of_range(u32::MAX, 2));
}

#[test]
fn an_unreadable_data_ram_read_fails_the_call() {
    // Divergence: Eden-NX2's reads have no failure result.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &ram_read_macro(true), 0);
    rig.take();
    let before = snapshot(&chan.three_d);
    let end = GPU_BASE + PAGES * PAGE_SIZE;
    let report = rig.call(
        &mut chan,
        Engine::ThreeD,
        0,
        &ram_parameters(end - 4, RAM_AT, 2),
    );
    assert_eq!(
        report.result,
        CallResult::Failed {
            error: CallFailure::DmaRead {
                address: end - 4,
                words: 2,
                fault: GpuFault::Unmapped { va: end },
            },
            first_for_macro: true,
        }
    );
    assert_eq!(rig.take(), []);
    assert_eq!(snapshot(&chan.three_d), before);
    assert_eq!(ram_words(&chan, RAM_AT, 2), [0, 0]);
    assert_eq!(chan.macros.data_ram().get(u32::from(SCRATCH_WORD)), Some(0));
    assert_eq!(memo_len(&rig, Engine::ThreeD), 0);
    // Ending exactly at the mapping's end reads.
    rig.write_memory(end - 8, &[0x81, 0x82]);
    let report = rig.call(
        &mut chan,
        Engine::ThreeD,
        0,
        &ram_parameters(end - 8, RAM_AT, 2),
    );
    assert!(matches!(report.result, CallResult::Committed(_)));
    assert_eq!(ram_words(&chan, RAM_AT, 2), [0x81, 0x82]);
}

/// A load-FIFO read of `fifo` words (queued, never performed: no barrier), then a data RAM read
/// of `words` at `index`.
fn fifo_then_ram_macro() -> Vec<Instruction> {
    finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_param(m16(DMA_READ_FIFOED)),
        mov_reg(2, Reg::Load0),
        emit_reg(m16(DATA_RAM_ADDRESS), 2),
        emit_param(m16(DMA_READ)),
    ])
}

fn fifo_then_ram(rig: &mut Rig, address: u64, fifo: u32, index: u32, words: u32) -> CallResult {
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &fifo_then_ram_macro(), 0);
    rig.take();
    let [upper, lower, _] = source_parameters(address, 0);
    rig.call(
        &mut chan,
        Engine::ThreeD,
        0,
        &[upper, lower, fifo, index, words],
    )
    .result
}

#[test]
fn data_ram_reads_count_toward_the_runs_dma_bound() {
    let loaded = CallResult::Committed(CallOutcome::NotStored(NotMemoised::DataRamLoad));
    let ram = 0x10;
    let fifo = DMA_RUN_MAX_WORDS - ram;
    assert_eq!(
        fifo_then_ram(&mut make_rig(), GPU_BASE, fifo, 0, ram),
        loaded
    );
    assert_eq!(
        fifo_then_ram(&mut make_rig(), GPU_BASE, fifo, 0, ram + 1),
        CallResult::Failed {
            error: CallFailure::DmaTooLarge {
                words: DMA_RUN_MAX_WORDS + 1,
            },
            first_for_macro: true,
        }
    );
    // A data RAM read first, then a load-FIFO read: the same total.
    let program = finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_param(m16(DMA_READ)),
        emit_param(m16(DMA_READ_FIFOED)),
    ]);
    let run = |fifo: u32| {
        let mut rig = make_rig();
        let mut chan = Chan::new(A);
        rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
        let [upper, lower, _] = source_parameters(GPU_BASE, 0);
        rig.call(&mut chan, Engine::ThreeD, 0, &[upper, lower, ram, fifo])
            .result
    };
    assert_eq!(run(DMA_RUN_MAX_WORDS - ram), loaded);
    assert_eq!(
        run(DMA_RUN_MAX_WORDS - ram + 1),
        CallResult::Failed {
            error: CallFailure::DmaTooLarge {
                words: DMA_RUN_MAX_WORDS + 1,
            },
            first_for_macro: true,
        }
    );
}

#[test]
fn data_ram_read_checks_come_in_their_documented_order() {
    let end = u32::try_from(DATA_RAM_WORDS).unwrap();
    // The run's bound before data RAM's end: both faults at once.
    let mut rig = make_rig();
    assert_eq!(
        fifo_then_ram(&mut rig, GPU_BASE, DMA_RUN_MAX_WORDS - 1, end - 1, 2),
        CallResult::Failed {
            error: CallFailure::DmaTooLarge {
                words: DMA_RUN_MAX_WORDS + 1,
            },
            first_for_macro: true,
        }
    );
    // Data RAM's end before the catch-up: the back is not asked.
    let mut rig = make_rig();
    rig.back.busy.set(true);
    assert_eq!(
        fifo_then_ram(&mut rig, GPU_BASE, 0, end - 1, 2),
        out_of_range(end - 1, 2)
    );
    assert_eq!(rig.back.queries.get(), 0);
    // The catch-up before the read: an unmapped read waits, then fails at the rerun.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &ram_read_macro(false), 0);
    rig.take();
    rig.back.busy.set(true);
    let unmapped = GPU_BASE + PAGES * PAGE_SIZE;
    let report = rig.call(
        &mut chan,
        Engine::ThreeD,
        0,
        &ram_parameters(unmapped, 0, 1),
    );
    assert_eq!(report.result, CallResult::CatchUp);
    assert_eq!(rig.back.queries.get(), 1);
    let report = rig.resume(&mut chan, Engine::ThreeD).unwrap();
    assert_eq!(
        report.result,
        CallResult::Failed {
            error: CallFailure::DmaRead {
                address: unmapped,
                words: 1,
                fault: GpuFault::Unmapped { va: unmapped },
            },
            first_for_macro: true,
        }
    );
}

#[test]
fn the_data_ram_read_uses_the_registers_held_at_its_write() {
    // The registers come from stream words before the call; the macro rewrites them after its
    // read, which changes nothing.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let program = finish(vec![
        emit_param(m16(DMA_READ)),
        emit_imm(m16(MEM_ADDRESS_B), 0),
        emit_imm(m16(DATA_RAM_ADDRESS), 0x7),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    let [upper, lower, _] = source_parameters(SOURCE, 0);
    rig.stream(&mut chan, Engine::ThreeD, MEM_ADDRESS_A, &[upper]);
    rig.stream(&mut chan, Engine::ThreeD, MEM_ADDRESS_B, &[lower]);
    rig.stream(&mut chan, Engine::ThreeD, DATA_RAM_ADDRESS, &[RAM_AT]);
    rig.take();
    rig.write_memory(SOURCE, &WORDS);
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &[2]);
    assert!(matches!(report.result, CallResult::Committed(_)));
    assert_eq!(ram_words(&chan, RAM_AT, 3), [WORDS[0], WORDS[1], 0]);
    assert_eq!(chan.macros.data_ram().get(0x7), Some(0));
}

#[test]
fn data_ram_reads_are_3d_only_and_land_in_the_data_ram_compute_shares() {
    // Eden-NX2's compute front reads no MME memory; `clc7c0.h` has the method, and a compute
    // write to it is a register write only here. The call is not remembered (divergence:
    // Eden-NX2 stores it), as the 3D call it could stand for would read.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let program = finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_imm(m16(DATA_RAM_ADDRESS), 0x30),
        emit_imm(m16(DMA_READ), 1),
    ]);
    rig.upload(&mut chan, Engine::Compute, 0, &program, 0);
    rig.take();
    rig.write_memory(SOURCE, &WORDS);
    rig.back.busy.set(true);
    let [upper, lower, _] = source_parameters(SOURCE, 0);
    let report = rig.call(&mut chan, Engine::Compute, 0, &[upper, lower]);
    assert_eq!(
        report.result,
        CallResult::Committed(CallOutcome::NotStored(NotMemoised::DataRamLoad))
    );
    assert_eq!(memo_len(&rig, Engine::Compute), 0);
    assert_eq!(rig.back.queries.get(), 0);
    assert_eq!(chan.macros.data_ram().get(0x30), Some(0));
    assert_eq!(
        rig.take().last(),
        Some(&write_entry(A, Engine::Compute, m16(DMA_READ), 1))
    );

    // A 3D read lands where a compute macro reads it.
    rig.back.busy.set(false);
    rig.upload(&mut chan, Engine::ThreeD, 0, &ram_read_macro(false), 0);
    rig.upload(
        &mut chan,
        Engine::Compute,
        0x10,
        &finish(vec![dread_to(0, 0x31), emit_reg(out_method(0), 0)]),
        1,
    );
    rig.take();
    rig.call(
        &mut chan,
        Engine::ThreeD,
        0,
        &ram_parameters(SOURCE, 0x30, 2),
    );
    let report = rig.call(&mut chan, Engine::Compute, 1, &[0]);
    assert!(matches!(report.result, CallResult::Committed(_)));
    assert_eq!(
        rig.take().last(),
        Some(&write_entry(A, Engine::Compute, out_method(0), WORDS[1]))
    );
}

#[test]
fn a_compute_call_on_the_3d_store_is_not_replayed_for_a_3d_data_ram_read() {
    // Until a channel uploads compute code, its compute calls run from the 3D store and memo,
    // whose key (start, parameters) does not name the engine. A compute call that writes
    // `MME_DMA_READ` reads nothing, but it is not remembered either, so a 3D call with the
    // same entry and parameters runs and reads rather than replaying the compute call.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    rig.upload(&mut chan, Engine::ThreeD, 0, &ram_read_macro(false), 0);
    rig.take();
    rig.write_memory(SOURCE, &WORDS);
    let parameters = ram_parameters(SOURCE, RAM_AT, 3);
    let loaded = CallResult::Committed(CallOutcome::NotStored(NotMemoised::DataRamLoad));

    let report = rig.call(&mut chan, Engine::Compute, 0, &parameters);
    assert_eq!(report.result, loaded);
    assert_eq!(memo_len(&rig, Engine::ThreeD), 0);
    assert_eq!(ram_words(&chan, RAM_AT, 3), [0, 0, 0]);

    let report = rig.call(&mut chan, Engine::ThreeD, 0, &parameters);
    assert_eq!(report.result, loaded);
    assert_eq!(rig.back.queries.get(), 1);
    assert_eq!(
        ram_words(&chan, RAM_AT - 1, 5),
        [0, WORDS[0], WORDS[1], WORDS[2], 0]
    );
}

#[test]
fn a_data_ram_read_stored_over_by_a_later_dwrite_keeps_the_dwrite() {
    // The words land at the write, in order with the macro's own data RAM writes.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let program = finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        mov(3, 0x66),
        store(0x21, 3),
        emit_imm(m16(DATA_RAM_ADDRESS), 0x20),
        emit_imm(m16(DMA_READ), 3),
        mov(4, 0x77),
        store(0x22, 4),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    rig.take();
    rig.write_memory(SOURCE, &WORDS);
    let [upper, lower, _] = source_parameters(SOURCE, 0);
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &[upper, lower]);
    assert!(matches!(report.result, CallResult::Committed(_)));
    assert_eq!(ram_words(&chan, 0x20, 3), [WORDS[0], WORDS[1], 0x77]);
}

#[test]
fn two_data_ram_reads_in_one_run_each_land_where_they_were_sent() {
    // Each read hands the interpreter its own words only.
    let mut rig = make_rig();
    let mut chan = Chan::new(A);
    let program = finish(vec![
        emit_param(m16(MEM_ADDRESS_A)),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_imm(m16(DATA_RAM_ADDRESS), 0x20),
        emit_imm(m16(DMA_READ), 2),
        emit_param(m16(MEM_ADDRESS_B)),
        emit_imm(m16(DATA_RAM_ADDRESS), 0x40),
        emit_imm(m16(DMA_READ), 1),
    ]);
    rig.upload(&mut chan, Engine::ThreeD, 0, &program, 0);
    rig.take();
    rig.write_memory(SOURCE, &WORDS);
    let [upper, lower, _] = source_parameters(SOURCE, 0);
    let report = rig.call(&mut chan, Engine::ThreeD, 0, &[upper, lower, lower + 8]);
    assert!(matches!(report.result, CallResult::Committed(_)));
    assert_eq!(ram_words(&chan, 0x20, 3), [WORDS[0], WORDS[1], 0]);
    assert_eq!(ram_words(&chan, 0x40, 3), [WORDS[2], 0, 0]);
    assert_eq!(rig.back.queries.get(), 2);
}

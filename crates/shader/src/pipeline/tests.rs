//! The production translator end to end through the shader cache's production constructor
//! ([`ShaderCache::new`] with [`PipelineTranslator`] and an executor of threads): a vertex
//! program with a branch, a loop and a guarded instruction, translated, structured, put in SSA
//! form, optimised and lowered, the IR run in the interpreter; jump tables from the answers;
//! the program entry; the named refusals and the order of the checks.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use super::*;
use crate::asm::assemble;
use crate::cache::{CacheLookup, CodeForm, Executor, HostTarget, ShaderCache, TranslateJob};
use crate::capture::capture;
use crate::decode::{
    AluSrc, BraCond, BranchOffset, BrxSrc, Gpr, Instr, IntCmpOp, MemAddr, MemEvictionPriority,
    MemSpace, MemType, MovSrc, OffsetStride, Op, OpASt, OpBra, OpBrx, OpExit, OpIAdd3, OpISetP,
    OpLd, OpMov, OpSt, Pred, PredSetOp, PredSrc, UniformAddr, UniformReg,
};
use crate::exchange::{EngineState, TextureKind, exchange};
use crate::flow::FlowError;
use crate::host::{HostCode, HostStage};
use crate::info::InfoError;
use crate::interp::Machine;
use crate::ir::{ATTRIBUTE_WORD_BYTES, Attribute, Function, Inst, Node, Opcode};
use crate::lookup::{ProgramMemory, ProgramRequest, Unmapped};
use crate::lower::{LowerError, MAX_LOCAL_MEMORY_BYTES, MAX_SHARED_MEMORY_BYTES};
use crate::sph::encode::encode_bytes;
use crate::sph::{Common, HEADER_BYTES, HeaderError, VtgIo, VtgMap, VtgStage};
use crate::testing::{imnmx, ldc, shf};
use crate::translate::GuestStage;
use crate::variant::VariantKey;

// ---------------------------------------------------------------------------------------------
// Instructions.

/// `!PT`, the carry input that adds nothing.
const NOT_PT: PredSrc = PredSrc {
    reg: Pred::PT,
    negated: true,
};

fn r(index: u8) -> Gpr {
    Gpr::new(index)
}

fn p(index: u8) -> PredSrc {
    PredSrc {
        reg: Pred::new(index).expect("a predicate"),
        negated: false,
    }
}

fn mov(dst: u8, bits: u32) -> Instr {
    Instr::new(Op::Mov(OpMov {
        dst: r(dst),
        src: MovSrc::Imm(bits),
        quad_lanes: OpMov::ALL_LANES,
    }))
}

/// `IADD3 dst, a, imm, RZ`.
fn add_imm(dst: u8, a: u8, imm: u32) -> Instr {
    Instr::new(Op::IAdd3(OpIAdd3 {
        dst: r(dst),
        overflow: [Pred::PT; 2],
        srcs: [AluSrc::reg(r(a)), AluSrc::Imm(imm), AluSrc::reg(Gpr::RZ)],
        carry: [NOT_PT; 2],
    }))
}

/// `ISETP.LT.U32.AND dst, PT, a, imm, PT`.
fn less_than(dst: u8, a: u8, imm: u32) -> Instr {
    Instr::new(Op::ISetP(OpISetP {
        dsts: [Pred::new(dst).expect("a predicate"), Pred::PT],
        srcs: [AluSrc::reg(r(a)), AluSrc::Imm(imm)],
        cmp_op: IntCmpOp::Lt,
        signed: false,
        set_op: PredSetOp::And,
        ex: false,
        accum: PredSrc::TRUE,
        low_cmp: PredSrc::TRUE,
    }))
}

/// `AST a[addr], data`, one word.
fn ast(data: u8, addr: u16) -> Instr {
    Instr::new(Op::ASt(OpASt {
        data: r(data),
        vtx: Gpr::RZ,
        offset: Gpr::RZ,
        addr,
        comps: 1,
        patch: false,
        phys: false,
    }))
}

/// `STL [RZ + offset], data`, one word.
fn stl(data: u8, offset: i32) -> Instr {
    Instr::new(Op::St(OpSt {
        data: r(data),
        addr: rz_plus(offset),
        mem_type: MemType::B32,
        space: MemSpace::Local(MemEvictionPriority::Normal),
        bit76: false,
    }))
}

/// `[RZ + offset]`, the address the local and shared accesses below use.
const fn rz_plus(offset: i32) -> MemAddr {
    MemAddr {
        reg: Gpr::RZ,
        reg_64: false,
        uniform: None,
        offset,
    }
}

/// `LDL dst, [RZ + offset]`, one word.
fn ldl(dst: u8, offset: i32) -> Instr {
    Instr::new(Op::Ld(OpLd {
        dst: r(dst),
        // `LDL` encodes the uniform register field: register 0, off, as PR #4588 takes it.
        addr: MemAddr {
            uniform: Some(UniformAddr {
                reg: UniformReg::new(0).expect("a uniform register"),
                wide: false,
                enabled: false,
            }),
            ..rz_plus(offset)
        },
        mem_type: MemType::B32,
        space: MemSpace::Local(MemEvictionPriority::Normal),
        pred: PredSrc::TRUE,
        pred_dst: Pred::PT,
        zd: false,
        bit76: false,
    }))
}

/// `STS [RZ + offset], data`, one word.
fn sts(data: u8, offset: i32) -> Instr {
    Instr::new(Op::St(OpSt {
        data: r(data),
        addr: rz_plus(offset),
        mem_type: MemType::B32,
        space: MemSpace::Shared(OffsetStride::X1),
        bit76: false,
    }))
}

fn exit() -> Instr {
    Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    }))
}

/// [`crate::decode::Instruction::SIZE`] as an `i64`.
fn size() -> i64 {
    i64::try_from(Instruction::SIZE).expect("16 bytes")
}

/// The offset from the end of instruction `from` to instruction `to`.
fn offset(from: u32, to: u32) -> BranchOffset {
    let bytes = (i64::from(to) - i64::from(from) - 1) * size();
    BranchOffset::from_words(bytes / BranchOffset::WORD_BYTES)
}

/// `@guard BRA` from instruction `from` to instruction `to`.
fn bra(guard: PredSrc, from: u32, to: u32) -> Instr {
    Instr {
        guard,
        op: Op::Bra(OpBra {
            cond: BraCond::Pred(PredSrc::TRUE),
            uniform: false,
            offset: offset(from, to),
        }),
    }
}

/// `BRX reg` at instruction `at`, its table values counting from the program's start.
fn brx(reg: u8, at: u32) -> Instr {
    Instr::new(Op::Brx(OpBrx {
        src: BrxSrc::Reg(r(reg)),
        offset: offset(at, 0),
        cond: PredSrc::TRUE,
    }))
}

fn words(instrs: &[Instr]) -> Vec<Instruction> {
    instrs
        .iter()
        .map(|instr| assemble(instr).expect("encodable"))
        .collect()
}

/// A vertex program's header, with `local` bytes of local memory.
fn vertex_header(local: u32) -> ProgramHeader {
    ProgramHeader {
        common: Common {
            local_memory_low_size: local,
            ..Common::default()
        },
        io: Io::Vtg(Box::new(VtgIo {
            stage: VtgStage::Vertex,
            input: VtgMap::default(),
            output: VtgMap::default(),
            vmap: VtgMap::default(),
        })),
    }
}

/// `header` (when given) followed by `code`.
fn bytes(header: Option<&ProgramHeader>, code: &[Instruction]) -> Vec<u8> {
    let mut bytes = header.map_or_else(Vec::new, |header| {
        encode_bytes(header).expect("encodable").to_vec()
    });
    bytes.extend(code.iter().flat_map(|word| word.to_le_bytes()));
    bytes
}

/// The generic 0 X and Y attribute words (`0x80`, `0x84`; attribute words 32 and 33).
const GENERIC_0_X: u16 = 0x80;
const GENERIC_0_Y: u16 = 0x84;

/// The attribute word at byte address `address`.
fn attribute(address: u16) -> Attribute {
    Attribute(address / u16::try_from(ATTRIBUTE_WORD_BYTES).expect("4"))
}

/// The loop's count, its step, and the guarded instruction's value.
const COUNT: u32 = 5;
const STEP: u32 = 3;
const GUARDED: u32 = 0x2a;

/// The end-to-end program: a loop adding [`STEP`] to R0 [`COUNT`] times (a backward branch), a
/// guarded `MOV` of [`GUARDED`] into R2 under `P1`, set when R0 is below `bound`, and R0 and R2
/// stored to generic 0's X and Y.
fn loop_program(bound: u32) -> Vec<Instruction> {
    words(&[
        mov(0, 0),
        mov(1, 0),
        add_imm(0, 0, STEP),
        add_imm(1, 1, 1),
        less_than(0, 1, COUNT),
        bra(p(0), 5, 2),
        less_than(1, 0, bound),
        Instr {
            guard: p(1),
            ..mov(2, GUARDED)
        },
        ast(0, GENERIC_0_X),
        ast(2, GENERIC_0_Y),
        exit(),
    ])
}

/// The vertex stage, with no colour outputs.
const VERTEX: Stage = Stage {
    stage: GuestStage::Vertex,
    color_outputs: ColorOutputs {
        targets: [0; crate::translate::COLOR_TARGETS],
    },
};

fn has_loop(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| match node {
        Node::Loop { .. } => true,
        Node::If { body, .. } => has_loop(body),
        _ => false,
    })
}

fn has_if(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| match node {
        Node::If { .. } => true,
        Node::Loop { body, .. } => has_if(body),
        _ => false,
    })
}

/// How many operations of `opcode` `function` lists.
fn count(function: &Function, opcode: Opcode) -> usize {
    function
        .listed()
        .filter(|(_, inst)| matches!(inst, Inst::Op(op) if op.opcode == opcode))
        .count()
}

// ---------------------------------------------------------------------------------------------
// The cache's rig.

/// Runs every job on a thread of its own: a real pool's shape (the contract: `spawn` returns at
/// once and never runs the job on the calling thread).
struct Threads;

impl Executor for Threads {
    fn spawn(&self, job: TranslateJob) {
        thread::spawn(job);
    }
}

/// The first byte of the fake guest memory.
const BASE: u64 = 0x1_0000_0000;

/// One mapped run of guest memory at [`BASE`].
struct Memory(Vec<u8>);

impl ProgramMemory for Memory {
    fn read(&self, address: u64, out: &mut [u8]) -> Result<(), Unmapped> {
        let start = address
            .checked_sub(BASE)
            .and_then(|at| usize::try_from(at).ok())
            .ok_or(Unmapped { address })?;
        let bytes = start
            .checked_add(out.len())
            .and_then(|end| self.0.get(start..end))
            .ok_or(Unmapped { address })?;
        out.copy_from_slice(bytes);
        Ok(())
    }
}

/// Constant-buffer bank [`TABLE_BANK`]'s words from [`TABLE_OFFSET`] on; no textures.
#[derive(Default)]
struct State {
    table: Vec<u32>,
}

/// The bank and offset of the jump table the selector reads.
const TABLE_BANK: u8 = 2;
const TABLE_OFFSET: u32 = 0x40;

impl EngineState for State {
    fn cbuf_word(&self, _stage: GuestStage, bank: u8, offset: u32) -> Option<u32> {
        let entry = offset.checked_sub(TABLE_OFFSET)? / 4;
        (bank == TABLE_BANK)
            .then(|| self.table.get(usize::try_from(entry).ok()?).copied())
            .flatten()
    }

    fn texture_kind(&self, _stage: GuestStage, _handle: u32) -> Option<TextureKind> {
        None
    }
}

/// A target with no optional features.
const TARGET: HostTarget = HostTarget {
    form: CodeForm::NagaModule,
    caps: Requirements {
        subgroup: false,
        subgroup_size_32: false,
        f16: false,
        f64: false,
        i64: false,
        i64_atomics: false,
    },
};

/// How long a test lets a wait run before calling it hung.
const WATCHDOG: Duration = Duration::from_secs(30);

/// The shader the production cache gives for `program`, bound as `stage`, with `engine` and
/// `variant`: looked up, then waited for within [`WATCHDOG`].
fn through_the_cache(
    stage: GuestStage,
    program: &[u8],
    engine: &State,
    variant: &VariantKey,
) -> Result<Arc<HostShader>, Arc<Rejection>> {
    let mut cache = ShaderCache::new(Arc::new(PipelineTranslator), Arc::new(Threads), TARGET);
    let memory = Memory(program.to_vec());
    let request = [ProgramRequest {
        stage,
        address: BASE,
    }];
    let mut lookup = cache.lookup(&request, variant, &memory, engine);
    if let CacheLookup::Pending(ticket) = lookup {
        lookup = cache.wait_timeout(ticket, WATCHDOG);
    }
    match lookup {
        CacheLookup::Ready(shaders) => Ok(Arc::clone(&shaders[0].shader)),
        CacheLookup::Rejected(rejection) => Err(Arc::clone(&rejection.reason)),
        CacheLookup::Pending(_) => {
            unreachable!("the translation did not finish within the watchdog")
        }
    }
}

// ---------------------------------------------------------------------------------------------
// End to end.

/// The production translator runs the whole pipeline in the spec's order through the cache's
/// production constructor: the shader is ready, a vertex shader writing generic 0, validated by
/// the lowering (its last step). The IR the pipeline gives the lowering is structured (a loop
/// and an `if`, so the structurizer ran) and in SSA form with no guest register, predicate or
/// goto variable left (so the SSA rewrite ran after the structurizer, on its variables too),
/// and the interpreter runs it to the expected outputs: R0 is `COUNT * STEP`, and R2 is
/// `GUARDED` exactly when the guard holds.
#[test]
fn the_production_translator_runs_the_whole_pipeline() {
    let header = vertex_header(0);
    let bound_hit = COUNT * STEP + 1;
    for (bound, y) in [(bound_hit, GUARDED), (COUNT * STEP, 0)] {
        let code = loop_program(bound);
        let program = bytes(Some(&header), &code);
        let shader = through_the_cache(
            GuestStage::Vertex,
            &program,
            &State::default(),
            &VariantKey::default(),
        )
        .expect("the program translates");
        assert_eq!(shader.stage, HostStage::Vertex);
        assert_eq!(shader.requirements, Requirements::default());

        let pixel = None;
        let function =
            pipeline_ir(VERTEX, pixel, &code, &BranchTables::new(0)).expect("the IR half runs");
        let tree = function.structure().expect("structured");
        assert!(has_loop(tree), "{function}");
        assert!(has_if(tree), "{function}");
        for opcode in [
            Opcode::GetRegister,
            Opcode::SetRegister,
            Opcode::GetPred,
            Opcode::SetPred,
            Opcode::GetGotoVariable,
            Opcode::SetGotoVariable,
        ] {
            assert_eq!(count(&function, opcode), 0, "{opcode:?} left: {function}");
        }
        let mut machine = Machine::default();
        machine.run(&function).expect("the IR runs");
        assert_eq!(
            machine.outputs.get(&(attribute(GENERIC_0_X), 0)),
            Some(&(COUNT * STEP)),
            "bound {bound}"
        );
        assert_eq!(
            machine
                .outputs
                .get(&(attribute(GENERIC_0_Y), 0))
                .copied()
                .unwrap_or(0),
            y,
            "bound {bound}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Jump tables.

/// The `BRX`'s instruction index in [`selector_program`].
const BRX_AT: u32 = 4;
/// The instructions the table's two entries name in [`selector_program`].
const FIRST: u32 = 5;
const SECOND: u32 = 7;

/// The selector program: R3 is `selector`; the table of `BRX R6` is `c[2][0x40]`, two entries,
/// selected by R3; instructions [`FIRST`] and [`SECOND`] store 10 and 20 to generic 0's X.
/// Registers are not inputs (the SSA rewrite reads an unwritten one as undefined), so the
/// selector is set by the program.
fn selector_program(selector: u32) -> Vec<Instruction> {
    let mut code = words(&[mov(3, selector)]);
    code.extend([
        imnmx(4, 3, 1),
        shf(5, 4),
        ldc(6, 5, TABLE_BANK, TABLE_OFFSET),
    ]);
    code.extend(words(&[
        brx(6, BRX_AT),
        mov(1, 10),
        bra(PredSrc::TRUE, FIRST + 1, SECOND + 1),
        mov(1, 20),
        ast(1, GENERIC_0_X),
        exit(),
    ]));
    code
}

/// The table value naming instruction `index` of a vertex program: the header's length plus 16
/// bytes per instruction (`Capture::resolve_jump_tables`).
fn table_value(index: u32) -> u32 {
    let header = u32::try_from(HEADER_BYTES).expect("128");
    let size = u32::try_from(Instruction::SIZE).expect("16");
    header + index * size
}

/// A `BRX` whose table the answers give is translated with its targets: the program entry asks
/// for the table's two words, the exchange answers them, flow analysis takes the targets, and
/// the interpreter goes where the selector says. Through the cache the same program gets past
/// flow analysis and is refused only by the lowering, at the table's constant-buffer load,
/// which slice 13 and 17 lower.
#[test]
fn a_brx_the_answers_resolve_translates() {
    let header = vertex_header(0);
    let program = bytes(Some(&header), &selector_program(0));
    let state = State {
        table: vec![table_value(FIRST), table_value(SECOND)],
    };
    let captured = capture(program.as_slice(), GuestStage::Vertex).expect("captures");
    let entry = PipelineProgram::new(&captured).expect("an entry");
    let answers = exchange(GuestStage::Vertex, &entry, &state)
        .expect("answered")
        .answers;
    assert_eq!(
        answers
            .iter()
            .map(|(question, _)| question)
            .collect::<Vec<_>>(),
        vec![
            Question::CbufWord {
                bank: TABLE_BANK,
                offset: TABLE_OFFSET
            },
            Question::CbufWord {
                bank: TABLE_BANK,
                offset: TABLE_OFFSET + 4
            },
        ]
    );
    let tables = entry.branch_tables(&answers).expect("resolved");
    assert_eq!(tables.targets(BRX_AT), [FIRST, SECOND]);
    for (selector, wanted) in [(0, 10), (1, 20)] {
        let function =
            pipeline_ir(VERTEX, None, &selector_program(selector), &tables).expect("translates");
        let mut machine = Machine::default();
        let mut bank = vec![0; usize::try_from(TABLE_OFFSET).expect("small") + 8];
        for (at, value) in [(0, table_value(FIRST)), (4, table_value(SECOND))] {
            let at = usize::try_from(TABLE_OFFSET).expect("small") + at;
            bank[at..at + 4].copy_from_slice(&u32::to_le_bytes(value));
        }
        machine.cbufs.insert(u32::from(TABLE_BANK), bank);
        machine.run(&function).expect("runs");
        assert_eq!(
            machine.outputs.get(&(attribute(GENERIC_0_X), 0)),
            Some(&wanted),
            "selector {selector}"
        );
    }
    let refused = through_the_cache(GuestStage::Vertex, &program, &state, &VariantKey::default())
        .expect_err("the constant-buffer load is not lowered yet");
    assert!(
        matches!(
            &*refused,
            Rejection::Lower(LowerError::Operation {
                at,
                opcode: Opcode::GetCbufU32,
            }) if at.index() == 22
        ),
        "{refused:?}"
    );
    assert_eq!(refused.to_string(), "%22: GetCbufU32 is not lowered yet");
}

/// A `BRX` whose table words the engine does not have reads them as 0, which names no
/// instruction: flow analysis refuses the branch, with the exact error.
#[test]
fn a_brx_the_answers_do_not_resolve_is_refused() {
    let header = vertex_header(0);
    let program = bytes(Some(&header), &selector_program(0));
    let refused = through_the_cache(
        GuestStage::Vertex,
        &program,
        &State::default(),
        &VariantKey::default(),
    )
    .expect_err("refused");
    assert_eq!(
        *refused,
        Rejection::Translate(TranslateError::Flow(FlowError::NoTarget { index: BRX_AT }))
    );
    assert_eq!(
        refused.to_string(),
        "the indirect branch at instruction 4 has no target in the program"
    );
}

/// A table value naming an instruction past the capture is refused by name, since the capture
/// is not extended (module docs, "Divergences from the reference"): the first instruction past
/// the capture's end is refused, both from the program entry and through the cache, and the
/// last captured instruction is the accepted edge, kept as a target.
#[test]
fn a_target_past_the_capture_is_refused() {
    let header = vertex_header(0);
    let program = bytes(Some(&header), &selector_program(0));
    let captured = capture(program.as_slice(), GuestStage::Vertex).expect("captures");
    let count = captured.instruction_count();
    let entry = PipelineProgram::new(&captured).expect("an entry");
    let answers = |second: u32| {
        let state = State {
            table: vec![table_value(FIRST), table_value(second)],
        };
        let answers = exchange(GuestStage::Vertex, &entry, &state)
            .expect("answered")
            .answers;
        (state, answers)
    };
    let (_, last) = answers(count - 1);
    let tables = entry.branch_tables(&last).expect("resolved");
    assert_eq!(tables.targets(BRX_AT), [FIRST, count - 1]);
    let (state, past) = answers(count);
    assert_eq!(
        entry.branch_tables(&past).map(|_| ()),
        Err(Rejection::Pipeline(PipelineRefusal::JumpTargetPastCapture))
    );
    let refused = through_the_cache(GuestStage::Vertex, &program, &state, &VariantKey::default())
        .expect_err("refused");
    assert_eq!(
        *refused,
        Rejection::Pipeline(PipelineRefusal::JumpTargetPastCapture)
    );
    assert_eq!(
        refused.to_string(),
        "a jump table names an instruction past the captured program, which is not extended yet"
    );
}

// ---------------------------------------------------------------------------------------------
// The program entry, the refusals and their order.

/// A graphics capture's entry parses its header, a compute capture's has none; a header that
/// does not parse refuses the entry with the parser's exact error.
#[test]
fn the_program_entry_parses_the_header() {
    let code = words(&[exit()]);
    let header = vertex_header(0x40);
    let vertex =
        capture(bytes(Some(&header), &code).as_slice(), GuestStage::Vertex).expect("captures");
    let entry = PipelineTranslator.program(&vertex).expect("an entry");
    assert_eq!(entry.header(), Some(&header));
    assert_eq!(entry.capture(), &vertex);
    let compute = capture(bytes(None, &code).as_slice(), GuestStage::Compute).expect("captures");
    assert_eq!(
        PipelineTranslator
            .program(&compute)
            .expect("an entry")
            .header(),
        None
    );
    let mut zero = vec![0; HEADER_BYTES];
    zero.extend(code.iter().flat_map(|word| word.to_le_bytes()));
    let unparsed = capture(zero.as_slice(), GuestStage::Vertex).expect("captures");
    let refused = PipelineTranslator
        .program(&unparsed)
        .expect_err("version 0");
    assert_eq!(
        refused,
        Rejection::Header(HeaderError::Version {
            version: 0,
            supported: ProgramHeader::VERSION
        })
    );
}

/// The length in words of `shader`'s local-memory array, `None` when it has none.
fn local_words(shader: &HostShader) -> Option<u32> {
    let HostCode::Naga { module, .. } = &shader.code;
    module
        .global_variables
        .iter()
        .find(|(_, global)| global.name.as_deref() == Some("local_memory"))
        .map(|(_, global)| match module.types[global.ty].inner {
            naga::TypeInner::Array {
                size: naga::ArraySize::Constant(length),
                ..
            } => length.get(),
            ref other => panic!("local memory is a sized array, not {other:?}"),
        })
}

/// A vertex program's header with `low` bytes in the local-memory size's low field, `high` in
/// its high field and `crs` bytes of CRS size.
fn vertex_header_crs(low: u32, high: u32, crs: u32) -> ProgramHeader {
    let mut header = vertex_header(low);
    header.common.local_memory_high_size = high;
    header.common.local_memory_crs_size = crs;
    header
}

/// A compute program that writes local memory lowers over an array sized from the variant
/// key's local-memory size (the QMD's, filled by the GPU crate): 0x13 bytes are five words.
/// With a key size of zero it lowers with no array (the store is dropped, `crate::lower`'s
/// "Local and shared memory"); a compute program without local memory has no array whatever
/// the key says; and a vertex program with the same store takes its header's size, not the
/// key's. Each condition is the only one that varies between two of the cases.
#[test]
fn a_compute_program_with_local_memory_takes_the_qmd_size() {
    let store = words(&[mov(0, 7), stl(0, 0), exit()]);
    let plain = words(&[mov(0, 7), exit()]);
    let compute = |code: &[Instruction], local: u32| {
        through_the_cache(
            GuestStage::Compute,
            &bytes(None, code),
            &State::default(),
            &VariantKey {
                local_memory_size: local,
                ..compute_variant()
            },
        )
        .map(|shader| local_words(&shader))
    };
    assert_eq!(compute(&store, 0x13), Ok(Some(5)));
    assert_eq!(compute(&store, 0), Ok(None));
    assert_eq!(compute(&plain, 0x13), Ok(None));
    let vertex = through_the_cache(
        GuestStage::Vertex,
        &bytes(Some(&vertex_header(0x10)), &store),
        &State::default(),
        &VariantKey {
            local_memory_size: 0x40,
            ..VariantKey::default()
        },
    )
    .map(|shader| local_words(&shader));
    assert_eq!(vertex, Ok(Some(4)));
}

/// A compute program that only reads local memory (`LDL R0, [RZ]; STS [RZ], R0; EXIT`)
/// lowers over the key's size too: the load alone, kept alive by the shared store, makes the
/// array. Without the load the same program has no local array, so the load is the only thing
/// that varies.
#[test]
fn a_compute_program_that_only_reads_local_memory_takes_the_qmd_size() {
    let variant = VariantKey {
        shared_memory_size: 16,
        local_memory_size: 8,
        ..compute_variant()
    };
    let lowered = |code: &[Instruction]| {
        through_the_cache(
            GuestStage::Compute,
            &bytes(None, code),
            &State::default(),
            &variant,
        )
        .map(|shader| local_words(&shader))
    };
    assert_eq!(
        lowered(&words(&[ldl(0, 0), sts(0, 0), exit()])),
        Ok(Some(2))
    );
    assert_eq!(lowered(&words(&[mov(0, 7), sts(0, 0), exit()])), Ok(None));
}

/// A graphics program's local memory is its header's size plus its CRS size, as every
/// reference's `GraphicsEnvironment` adds `shader_local_memory_crs_size`: each field alone,
/// both (0x10 + 0x9 = 0x19 bytes, seven words, where either alone gives four or three), none,
/// and the high field, which adds 2^24.
#[test]
fn a_graphics_program_adds_its_crs_size() {
    let store = words(&[mov(0, 7), stl(0, 0), exit()]);
    let vertex = |header: &ProgramHeader| {
        through_the_cache(
            GuestStage::Vertex,
            &bytes(Some(header), &store),
            &State::default(),
            &VariantKey::default(),
        )
        .map(|shader| local_words(&shader))
    };
    assert_eq!(vertex(&vertex_header_crs(0x10, 0, 0)), Ok(Some(4)));
    assert_eq!(vertex(&vertex_header_crs(0, 0, 0x9)), Ok(Some(3)));
    assert_eq!(vertex(&vertex_header_crs(0x10, 0, 0x9)), Ok(Some(7)));
    assert_eq!(vertex(&vertex_header_crs(0, 0, 0)), Ok(None));
    // The high field is bits 24 up (Eden's formula, `Common::local_memory_size`): 2^24 bytes
    // are past the lowering's cap, so it is refused with that exact size plus the CRS size.
    assert_eq!(
        vertex(&vertex_header_crs(0, 1, 0x9)),
        Err(Arc::new(Rejection::Lower(LowerError::LocalMemory {
            bytes: (1 << 24) + 0x9
        })))
    );
}

/// [`local_memory_bytes`] for every combination of its inputs: no header takes the key's size
/// (zero, non-zero and the widest), a header takes its own size plus its CRS size and ignores
/// the key's (zero and non-zero), and the header's three fields each count.
#[test]
fn local_memory_bytes_for_each_input() {
    let key = |local| VariantKey {
        local_memory_size: local,
        ..VariantKey::default()
    };
    assert_eq!(local_memory_bytes(None, &key(0)), 0);
    assert_eq!(local_memory_bytes(None, &key(0x123)), 0x123);
    assert_eq!(
        local_memory_bytes(None, &key(u32::MAX)),
        u64::from(u32::MAX)
    );
    let header = vertex_header_crs(0x10, 0x2, 0x300);
    let expected = 0x10 + (0x2 << 24) + 0x300;
    assert_eq!(local_memory_bytes(Some(&header), &key(0)), expected);
    assert_eq!(local_memory_bytes(Some(&header), &key(0x123)), expected);
    assert_eq!(
        local_memory_bytes(Some(&vertex_header_crs(0, 0, 0)), &key(0x123)),
        0
    );
    // The widest fields a caller can build (the type does not hold them to 24 bits) still sum
    // without wrapping.
    let widest = vertex_header_crs(u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(
        local_memory_bytes(Some(&widest), &key(0)),
        0x00ff_ffff_ffff_ffff + u64::from(u32::MAX)
    );
}

/// Divergence (module docs, "The local-memory size"): a header size of 2^32 bytes (high field
/// 0x100) is refused with that exact size, where Eden truncates it to 32 bits and sizes local
/// memory as the CRS size alone; a header one byte below, plus a CRS size that takes the sum
/// to 2^32, is refused with the same size, where Eden's 32-bit sum wraps to 0.
#[test]
fn a_header_size_of_2_to_the_32_is_refused_not_wrapped() {
    let plain = words(&[mov(0, 7), exit()]);
    let vertex = |header: &ProgramHeader| {
        through_the_cache(
            GuestStage::Vertex,
            &bytes(Some(header), &plain),
            &State::default(),
            &VariantKey::default(),
        )
        .map(|_| ())
    };
    let refused = Err(Arc::new(Rejection::Lower(LowerError::LocalMemory {
        bytes: 1 << 32,
    })));
    assert_eq!(vertex(&vertex_header_crs(0, 0x100, 0)), refused);
    assert_eq!(
        vertex(&vertex_header_crs(0xff_ffff, 0xff, 1)),
        refused,
        "0xff_ffff_ffff + 1"
    );
}

/// Collect info runs between the passes and the lowering (module docs, step 4): a vertex
/// program that stores to attribute word 0, which no slot names, is refused with collect
/// info's exact error; the same store to generic 0's X lowers, so the attribute is the only
/// thing that varies.
#[test]
fn collect_info_refuses_before_the_lowering() {
    let unknown = through_the_cache(
        GuestStage::Vertex,
        &bytes(
            Some(&vertex_header(0)),
            &words(&[mov(0, 7), ast(0, 0), exit()]),
        ),
        &State::default(),
        &VariantKey::default(),
    )
    .expect_err("refused");
    assert!(
        matches!(
            &*unknown,
            Rejection::Info(InfoError::UnknownAttribute { attribute, .. })
                if *attribute == Attribute(0)
        ),
        "{unknown:?}"
    );
    assert!(
        through_the_cache(
            GuestStage::Vertex,
            &bytes(
                Some(&vertex_header(0)),
                &words(&[mov(0, 7), ast(0, GENERIC_0_X), exit()])
            ),
            &State::default(),
            &VariantKey::default(),
        )
        .is_ok()
    );
}

/// A compute variant: a workgroup of 8 by 4 by 2 invocations, each size distinct.
fn compute_variant() -> VariantKey {
    VariantKey {
        workgroup_size: [8, 4, 2],
        ..VariantKey::default()
    }
}

/// The variant key's workgroup, shared-memory and local-memory sizes and the header's
/// local-memory size (alone, and split with its CRS size) reach the lowering: the shader's
/// workgroup size is the key's, and a size one past each lowering bound is refused with the
/// lowering's error, at the bound accepted.
#[test]
fn the_variant_and_the_header_reach_the_lowering() {
    let plain = words(&[mov(0, 7), exit()]);
    let compute = |variant: &VariantKey| {
        through_the_cache(
            GuestStage::Compute,
            &bytes(None, &plain),
            &State::default(),
            variant,
        )
    };
    let shader = compute(&compute_variant()).expect("lowers");
    assert_eq!(shader.workgroup_size, Some([8, 4, 2]));
    for (shared, refused) in [
        (MAX_SHARED_MEMORY_BYTES, None),
        (
            MAX_SHARED_MEMORY_BYTES + 1,
            Some(LowerError::SharedMemory {
                bytes: MAX_SHARED_MEMORY_BYTES + 1,
            }),
        ),
    ] {
        let variant = VariantKey {
            shared_memory_size: shared,
            ..compute_variant()
        };
        assert_eq!(
            compute(&variant).err(),
            refused.map(|error| Arc::new(Rejection::Lower(error))),
            "{shared}"
        );
    }
    let max = u32::try_from(MAX_LOCAL_MEMORY_BYTES).expect("fits");
    for (local, refused) in [
        (max, None),
        (
            max + 1,
            Some(LowerError::LocalMemory {
                bytes: u64::from(max) + 1,
            }),
        ),
    ] {
        let refused = refused.map(|error| Arc::new(Rejection::Lower(error)));
        let vertex = through_the_cache(
            GuestStage::Vertex,
            &bytes(Some(&vertex_header(local)), &plain),
            &State::default(),
            &VariantKey::default(),
        );
        assert_eq!(vertex.err(), refused.clone(), "{local}");
        // The same size split between the header's size and its CRS size.
        let split = through_the_cache(
            GuestStage::Vertex,
            &bytes(Some(&vertex_header_crs(local - 0x40, 0, 0x40)), &plain),
            &State::default(),
            &VariantKey::default(),
        );
        assert_eq!(split.err(), refused.clone(), "{local} split");
        // And a compute program's, from the variant key.
        let from_key = compute(&VariantKey {
            local_memory_size: local,
            ..compute_variant()
        });
        assert_eq!(from_key.err(), refused.clone(), "{local} from the key");
    }
}

/// The header is checked against the stage before the jump tables are resolved: a vertex
/// program translated as a pixel shader, with no answers for its table, is refused for its
/// header; with the right stage, for the missing answer.
#[test]
fn the_header_is_checked_before_the_jump_tables() {
    let header = vertex_header(0);
    let program = bytes(Some(&header), &selector_program(0));
    let captured = capture(program.as_slice(), GuestStage::Vertex).expect("captures");
    let entry = PipelineProgram::new(&captured).expect("an entry");
    let answers = Answers::new();
    let variant = VariantKey::default();
    let input = |stage| ShaderInput {
        stage,
        program: &entry,
        answers: &answers,
        variant: &variant,
        target: &TARGET,
    };
    assert_eq!(
        translate_shader(&input(GuestStage::Pixel)).err(),
        Some(Rejection::Info(InfoError::HeaderStage {
            slot: GuestStage::Pixel,
            header: GuestStage::Vertex,
        }))
    );
    assert_eq!(
        translate_shader(&input(GuestStage::Vertex)).err(),
        Some(Rejection::Translate(TranslateError::Flow(
            FlowError::MissingAnswer {
                bank: TABLE_BANK,
                offset: TABLE_OFFSET
            }
        )))
    );
}

/// The host target's features reach the lowering field by field.
#[test]
fn host_caps_take_each_feature() {
    let all_off = Requirements::default();
    assert_eq!(host_caps(all_off), HostCaps::NONE);
    let each = [
        (
            Requirements {
                f16: true,
                ..all_off
            },
            HostCaps {
                f16: true,
                ..HostCaps::NONE
            },
        ),
        (
            Requirements {
                i64: true,
                ..all_off
            },
            HostCaps {
                i64: true,
                ..HostCaps::NONE
            },
        ),
        (
            Requirements {
                subgroup: true,
                ..all_off
            },
            HostCaps {
                subgroup: true,
                ..HostCaps::NONE
            },
        ),
        (
            Requirements {
                subgroup_size_32: true,
                ..all_off
            },
            HostCaps {
                subgroup_size_32: true,
                ..HostCaps::NONE
            },
        ),
        (
            Requirements {
                f64: true,
                i64_atomics: true,
                ..all_off
            },
            HostCaps::NONE,
        ),
    ];
    for (caps, wanted) in each {
        assert_eq!(host_caps(caps), wanted, "{caps:?}");
    }
}

/// A pixel program's header reaches the driver: its colour outputs make the `EXIT` write
/// colour target 0, which the lowered shader outputs; and its input map tells `IPA` how
/// generic 0 is interpolated, so the program gets past the driver (which refuses an `IPA` of a
/// generic attribute without the map) and is refused only by the lowering, which takes pixel
/// generic inputs with slice 17.
#[test]
fn a_pixel_program_s_header_reaches_the_driver() {
    use crate::decode::{InterpFreq, InterpLoc, OpIpa};
    use crate::host::{GuestSlot, SlotBinding, SlotType};
    use crate::sph::{ComponentMask, Components, PixelImap, PixelIo};

    let mut io = PixelIo::default();
    io.targets[0] = ComponentMask {
        x: true,
        y: false,
        z: false,
        w: false,
    };
    io.generic[0] = Components {
        x: PixelImap::Perspective,
        ..Components::default()
    };
    let header = ProgramHeader {
        common: Common::default(),
        io: Io::Pixel(Box::new(io)),
    };
    let one = 1.0_f32.to_bits();
    let colour = through_the_cache(
        GuestStage::Pixel,
        &bytes(Some(&header), &words(&[mov(0, one), exit()])),
        &State::default(),
        &VariantKey::default(),
    )
    .expect("lowers");
    assert_eq!(
        colour.outputs,
        [crate::host::InterfaceSlot {
            binding: SlotBinding::Location(0),
            ty: SlotType::F32x4,
            guest: GuestSlot::RenderTarget(0),
        }]
    );
    let ipa = Instr::new(Op::Ipa(OpIpa {
        dst: r(0),
        addr: GENERIC_0_X,
        freq: InterpFreq::Pass,
        loc: InterpLoc::Default,
        offset: Gpr::RZ,
        pred: Pred::PT,
    }));
    let refused = through_the_cache(
        GuestStage::Pixel,
        &bytes(Some(&header), &words(&[ipa, exit()])),
        &State::default(),
        &VariantKey::default(),
    )
    .expect_err("pixel generic inputs are slice 17's");
    assert!(
        matches!(*refused, Rejection::Lower(LowerError::Input { .. })),
        "{refused:?}"
    );
    assert_eq!(
        refused.to_string(),
        "a pixel program cannot read attribute a0x80"
    );
}

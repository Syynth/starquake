//! The walk over a captured program: reachability, block leaders, the reconvergence point of
//! each `BSYNC` under either barrier model, the subroutines `CALL` names and the targets of each
//! `BRX` (PR #4588's `AnalyzeFlowImpl`).
//!
//! Ported from `eden-pr4588/src/shader_recompiler/frontend/ampere/flow.cpp` @ e5fb3b97
//! (GPL-3.0-or-later, Copyright 2026 Eden Emulator Project): `AnalyzeFlowImpl`,
//! `MergeBarrierRegisters`, `CallLinkRegister`, `IsSkipCall`, `ReturnRegisterCopiesLink`,
//! `NeedsVirtualBlock`, `BranchTarget`, `GuardOf` and `SecondOf`. The checks run in the
//! reference's order; the parent module's docs list where this walk differs.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use super::classify::{OpInfo, OpKind, classify, is_warp_wide};
use super::{BarrierModel, BranchTables, FlowError, MAX_PROGRAM};
use crate::decode::{BRA_OPCODE, BRANCH_OFFSET, BSSY_OFFSET, BSSY_OPCODE, CALL_OPCODE};
use crate::decode::{
    BraCond, BranchOffset, COND, COND_NOT, DST, GUARD, GUARD_NOT, Gpr, Instr, Instruction, MovSrc,
    OPCODE, Op, OpBSSy, OpBSync, OpBra, OpBrx, OpCall, OpExit, OpMov, OpRet, Pred, PredSrc, decode,
};

/// How many barrier registers there are, so how deep the stack model's stack may grow
/// (`flow.cpp`: `stack.size() >= 16`; NAK: `RegFile::Bar` has 16 registers).
const BARRIERS: usize = crate::decode::Barrier::COUNT as usize;
/// How far before a `CALL` its return-address `MOV` may be (`CallLinkRegister`: `back <= 6`).
pub(crate) const LINK_LOOKBACK: u32 = 6;
/// How far before a `RET` the copy of the link register may be (`ReturnRegisterCopiesLink`:
/// `back <= 8`).
pub(crate) const RETURN_COPY_LOOKBACK: u32 = 8;
/// The first register `ReturnRegisterCopiesLink` does not accept as a return or link register
/// (`if (reg >= 250 || link >= 250) return false;`; the reference does not say why).
pub(crate) const RETURN_COPY_REGISTER_LIMIT: u8 = 250;
/// [`Instruction::SIZE`] as an offset.
fn instruction_bytes() -> i64 {
    // Cannot fail: an instruction is 16 bytes.
    i64::try_from(Instruction::SIZE).unwrap_or(i64::MAX)
}

/// The reconvergence point a barrier register holds: an instruction, or one that paths set
/// differently (`CONFLICTING_TARGET`, `~0U` in the reference).
type Target = Option<u32>;

/// The barrier registers set on a path, as (barrier, reconvergence point) pairs: sorted by
/// barrier in the register model, in `BSSY` order (a stack) in the stack model.
type Barriers = Vec<(u8, Target)>;

/// What [`analyze`] found (`FlowAnalysis`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Analysis {
    reachable: Vec<bool>,
    leader: Vec<bool>,
    sync_target: Vec<Option<u32>>,
    function_of: Vec<u32>,
    function_entry: Vec<u32>,
    call_function: Vec<Option<u32>>,
    indirect_tables: BTreeMap<u32, Vec<u32>>,
    link_register: Vec<u8>,
    visits: u64,
}

impl Analysis {
    /// How many instructions the program has.
    pub fn len(&self) -> u32 {
        // Cannot fail: the program has at most `MAX_PROGRAM` instructions.
        u32::try_from(self.reachable.len()).unwrap_or(u32::MAX)
    }

    /// Whether the program has no instructions (never: [`analyze`] refuses an empty one).
    pub fn is_empty(&self) -> bool {
        self.reachable.is_empty()
    }

    /// Whether instruction `index` is reachable from the entry.
    pub fn is_reachable(&self, index: u32) -> bool {
        at(&self.reachable, index).copied().unwrap_or(false)
    }

    /// Whether a block starts at instruction `index`.
    pub fn is_leader(&self, index: u32) -> bool {
        at(&self.leader, index).copied().unwrap_or(false)
    }

    /// Where the `BSYNC` at `index` reconverges, if it is a reachable one.
    pub fn sync_target(&self, index: u32) -> Option<u32> {
        at(&self.sync_target, index).copied().flatten()
    }

    /// The function a reachable instruction belongs to: 0 for the program, `n` for the `n`th
    /// subroutine found.
    pub fn function_of(&self, index: u32) -> Option<u32> {
        self.is_reachable(index)
            .then(|| at(&self.function_of, index).copied())
            .flatten()
    }

    /// The entry instruction of each function, the program's (0) first.
    pub fn function_entries(&self) -> &[u32] {
        &self.function_entry
    }

    /// The function the `CALL` at `index` calls, if it is a reachable call.
    pub fn callee(&self, index: u32) -> Option<u32> {
        at(&self.call_function, index).copied().flatten()
    }

    /// The targets of the `BRX` at `index`, if it is reachable.
    pub fn indirect_targets(&self, index: u32) -> Option<&[u32]> {
        self.indirect_tables.get(&index).map(Vec::as_slice)
    }

    /// How many times the walk applied an instruction: once when a path first reaches it and
    /// once more each time a path changes the barrier state recorded there. A path that
    /// arrives with a state already covered stops without applying it. The measure of the
    /// walk's work, which the linear-time tests bound.
    pub fn visits(&self) -> u64 {
        self.visits
    }

    /// The analysis with its work counter cleared, to compare two models' results.
    #[must_use]
    pub fn without_visits(mut self) -> Self {
        self.visits = 0;
        self
    }
}

/// Element `index` of `items`.
fn at<T>(items: &[T], index: u32) -> Option<&T> {
    usize::try_from(index).ok().and_then(|slot| items.get(slot))
}

/// Element `index` of `items`, to change it.
fn at_mut<T>(items: &mut [T], index: u32) -> Option<&mut T> {
    usize::try_from(index)
        .ok()
        .and_then(|slot| items.get_mut(slot))
}

/// A predicate field as a source.
fn pred_src(word: Instruction, reg: crate::decode::Field, not: crate::decode::Field) -> PredSrc {
    // Cannot fail: the field is three bits wide.
    PredSrc {
        reg: Pred::new(u8::try_from(word.field(reg)).unwrap_or(0)).unwrap_or(Pred::PT),
        negated: word.flag(not),
    }
}

/// The instruction's guard (`GuardOf`: bits 12..15 and 15).
pub(crate) fn guard_of(word: Instruction) -> PredSrc {
    pred_src(word, GUARD, GUARD_NOT)
}

/// The instruction's second predicate, bits 87..91 (`SecondOf`), which `EXIT`, `BRA`, `BSSY`,
/// `BSYNC`, `KILL`, `BRX`, `CALL` and `RET` carry.
pub(crate) fn second_of(word: Instruction) -> PredSrc {
    pred_src(word, COND, COND_NOT)
}

/// Whether a predicate is `!PT`: never true (`Guard::Never`).
pub(crate) const fn is_never(pred: PredSrc) -> bool {
    pred.reg.is_true() && pred.negated
}

/// The 12-bit opcode.
pub(crate) fn opcode_of(word: Instruction) -> u16 {
    // Cannot fail: the field is 12 bits wide.
    u16::try_from(word.field(OPCODE)).unwrap_or(u16::MAX)
}

/// Where a relative branch at `index` lands, in bytes from instruction 0: the next
/// instruction's address plus the offset (`BranchTarget`: `index * 16 + 16 + relative`).
pub(crate) fn target_byte(index: u32, offset: BranchOffset) -> Option<i64> {
    i64::from(index)
        .checked_add(1)?
        .checked_mul(instruction_bytes())?
        .checked_add(offset.bytes()?)
}

/// The instruction a relative branch at `index` lands on, or why it lands on none: a byte that
/// does not start an instruction, or one before instruction 0 (`BranchTarget` returns no target
/// for either).
pub(crate) fn branch_target(index: u32, offset: BranchOffset) -> Result<u32, FlowError> {
    let byte = target_byte(index, offset).ok_or(FlowError::TargetOutside {
        index,
        target: i64::MAX,
    })?;
    if byte.rem_euclid(instruction_bytes()) != 0 {
        return Err(FlowError::MisalignedTarget { index, byte });
    }
    let target = byte.div_euclid(instruction_bytes());
    u32::try_from(target).map_err(|_| FlowError::TargetOutside { index, target })
}

/// Whether a relative offset at `index` points at instruction 0's address: the base `BRX` and
/// `RET` must use (`s64{index} * 16 + 16 + RelativeOffset48(insn) != 0`).
fn points_at_start(index: u32, offset: BranchOffset) -> bool {
    target_byte(index, offset) == Some(0)
}

/// The raw relative target of a `BRA`, `BSSY` or `CALL` word, whatever its other bits, as
/// `ReturnRegisterCopiesLink` reads every instruction of the program (`BranchTarget` of
/// `RelativeOffset48`, or of `RelativeOffset30` for `BSSY`).
fn raw_branch_target(word: Instruction, index: u32) -> Option<u32> {
    let field = match opcode_of(word) {
        BSSY_OPCODE => BSSY_OFFSET,
        BRA_OPCODE | CALL_OPCODE => BRANCH_OFFSET,
        _ => return None,
    };
    branch_target(index, BranchOffset::from_words(word.signed(field))).ok()
}

/// A control instruction's decoded form, by what it does to control flow.
#[derive(Clone, Copy, Debug)]
enum Control {
    /// A straight-line instruction, not decoded.
    Plain,
    /// No effect on the flow graph (`OpKind::Nop`).
    Nop,
    /// `BRA`.
    Bra(OpBra),
    /// `EXIT`.
    Exit(OpExit),
    /// `KILL`.
    Kill,
    /// `BSSY`.
    Bssy(OpBSSy),
    /// `BSYNC`.
    Bsync(OpBSync),
    /// `BRX`.
    Brx(OpBrx),
    /// `CALL`.
    Call(OpCall),
    /// `RET`.
    Ret(OpRet),
}

/// An instruction as the walk sees it: its class, guard and second predicate, and, for a
/// control instruction, its decoded form (the reference's `LookupRow` and `FirstUnknownBit`).
struct Seen {
    info: OpInfo,
    guard: PredSrc,
    second: PredSrc,
    control: Control,
}

/// Reads instruction `index`: control instructions must decode, others are not decoded (see the
/// parent module's divergences).
fn read(code: &[Instruction], index: u32) -> Result<(Instruction, Seen), FlowError> {
    let word = *at(code, index).ok_or(FlowError::RunsPastEnd {
        last: last_index(code),
    })?;
    let info = classify(opcode_of(word));
    let control = if info.kind == OpKind::Plain {
        Control::Plain
    } else {
        // The opcode picks the variant: a control opcode decodes to its own operation.
        match decode(word)
            .map_err(|error| FlowError::Decode { index, error })?
            .op
        {
            Op::Bra(bra) => Control::Bra(bra),
            Op::Exit(exit) => Control::Exit(exit),
            Op::Kill(_) => Control::Kill,
            Op::BSSy(bssy) => Control::Bssy(bssy),
            Op::BSync(bsync) => Control::Bsync(bsync),
            Op::Brx(brx) => Control::Brx(brx),
            Op::Call(call) => Control::Call(call),
            Op::Ret(ret) => Control::Ret(ret),
            _ => Control::Nop,
        }
    };
    Ok((
        word,
        Seen {
            info,
            guard: guard_of(word),
            second: second_of(word),
            control,
        },
    ))
}

/// The last instruction's index.
fn last_index(code: &[Instruction]) -> u32 {
    u32::try_from(code.len().saturating_sub(1)).unwrap_or(u32::MAX)
}

/// The register a `MOV` of the return address sets up for the `CALL` at `index`: an
/// unguarded `MOV Rn, (index + 1) * 16` of every lane at most [`LINK_LOOKBACK`] instructions
/// before it, with only straight-line instructions in between (`CallLinkRegister`).
fn call_link_register(code: &[Instruction], index: u32) -> Option<u8> {
    let return_address = u32::try_from(Instruction::SIZE)
        .ok()?
        .checked_mul(index.checked_add(1)?)?;
    for back in 1..=LINK_LOOKBACK.min(index) {
        let word = *at(code, index.checked_sub(back)?)?;
        let kind = classify(opcode_of(word)).kind;
        if kind != OpKind::Plain && kind != OpKind::Nop {
            return None;
        }
        if let Ok(Instr {
            guard: PredSrc::TRUE,
            op:
                Op::Mov(OpMov {
                    dst,
                    src: MovSrc::Imm(value),
                    quad_lanes: OpMov::ALL_LANES,
                }),
        }) = decode(word)
            && value == return_address
        {
            return Some(dst.index());
        }
    }
    None
}

/// Whether the `CALL` at `index` is a predicated jump over the unconditional `BRA` after it:
/// `@P CALL.REL.NOINC L; BRA back; L:` with no return address set up, which never returns
/// (`IsSkipCall`; scoping doc row 67).
fn is_skip_call(code: &[Instruction], index: u32, offset: BranchOffset) -> bool {
    let Some(after_next) = index.checked_add(2) else {
        return false;
    };
    if usize::try_from(after_next).map_or(true, |slot| slot >= code.len()) {
        return false;
    }
    let Some(&next) = index.checked_add(1).and_then(|next| at(code, next)) else {
        return false;
    };
    opcode_of(next) == BRA_OPCODE
        && guard_of(next).is_true()
        && second_of(next).is_true()
        && branch_target(index, offset).ok() == Some(after_next)
        && call_link_register(code, index).is_none()
}

/// Whether translating `word` at the start of a block puts a predicate test of its own in front
/// of it (`NeedsVirtualBlock`).
fn needs_virtual_block(word: Instruction) -> bool {
    let info = classify(opcode_of(word));
    let guard = guard_of(word);
    let second = second_of(word);
    if is_never(guard) {
        return false;
    }
    match info.kind {
        OpKind::Plain => !info.pure && !guard.is_true(),
        OpKind::Kill => true,
        OpKind::Call => !guard.is_true(),
        OpKind::Exit => !(guard.is_true() && second.is_true()),
        _ => false,
    }
}

/// The register-model join of two barrier states: the union, with a register two paths set to
/// different points marked conflicting (`MergeBarrierRegisters`).
fn merge(a: &Barriers, b: &Barriers) -> Barriers {
    let mut merged = Barriers::with_capacity(BARRIERS);
    let (mut i, mut j) = (0, 0);
    loop {
        match (a.get(i), b.get(j)) {
            (None, None) => return merged,
            (Some(&left), None) => {
                merged.push(left);
                i += 1;
            }
            (None, Some(&right)) => {
                merged.push(right);
                j += 1;
            }
            (Some(&left), Some(&right)) => match left.0.cmp(&right.0) {
                Ordering::Less => {
                    merged.push(left);
                    i += 1;
                }
                Ordering::Greater => {
                    merged.push(right);
                    j += 1;
                }
                Ordering::Equal => {
                    let target = if left.1 == right.1 { left.1 } else { None };
                    merged.push((left.0, target));
                    i += 1;
                    j += 1;
                }
            },
        }
    }
}

/// Whether `shorter` is a prefix of `longer`.
fn is_prefix(shorter: &Barriers, longer: &Barriers) -> bool {
    longer.starts_with(shorter)
}

/// A path to walk: where it starts, the barrier state it carries and its function.
struct Visit {
    index: u32,
    barriers: Barriers,
    function: u32,
}

/// The walk's state.
struct Walk<'a> {
    code: &'a [Instruction],
    tables: &'a BranchTables,
    model: BarrierModel,
    count: u32,
    result: Analysis,
    seen: Vec<Option<Barriers>>,
    /// Function by entry instruction, to find a callee without a scan.
    function_by_entry: BTreeMap<u32, u32>,
    /// Per function: each `RET`'s index and the register it returns through.
    returns: Vec<Vec<(u32, u8)>>,
    work: Vec<Visit>,
}

/// Walks `code` from instruction 0 under `model` (`AnalyzeFlowImpl`). `tables` gives the
/// targets of each `BRX`.
pub fn analyze(
    code: &[Instruction],
    tables: &BranchTables,
    model: BarrierModel,
) -> Result<Analysis, FlowError> {
    let count = u32::try_from(code.len())
        .ok()
        .filter(|&count| count != 0 && count <= MAX_PROGRAM)
        .ok_or(FlowError::ProgramSize { len: code.len() })?;
    let mut walk = Walk {
        code,
        tables,
        model,
        count,
        result: Analysis {
            reachable: vec![false; code.len()],
            leader: vec![false; code.len()],
            sync_target: vec![None; code.len()],
            function_of: vec![0; code.len()],
            function_entry: vec![0],
            call_function: vec![None; code.len()],
            indirect_tables: BTreeMap::new(),
            link_register: vec![Gpr::RZ.index()],
            visits: 0,
        },
        seen: vec![None; code.len()],
        function_by_entry: BTreeMap::new(),
        returns: vec![Vec::new()],
        work: vec![Visit {
            index: 0,
            barriers: Barriers::new(),
            function: 0,
        }],
    };
    walk.set_leader(0);
    while let Some(visit) = walk.work.pop() {
        walk.path(visit)?;
    }
    walk.check_syncs()?;
    walk.check_functions()?;
    Ok(walk.result)
}

impl Walk<'_> {
    fn set_leader(&mut self, index: u32) {
        if let Some(leader) = at_mut(&mut self.result.leader, index) {
            *leader = true;
        }
    }

    /// Marks `index` as a block's start; it must be inside the program (`mark_leader`).
    fn mark_leader(&mut self, index: u32, from: u32) -> Result<(), FlowError> {
        if index >= self.count {
            return Err(FlowError::TargetOutside {
                index: from,
                target: i64::from(index),
            });
        }
        self.set_leader(index);
        Ok(())
    }

    /// Walks one path until it ends or joins a path already walked with the same state.
    fn path(&mut self, visit: Visit) -> Result<(), FlowError> {
        let Visit {
            mut index,
            mut barriers,
            function,
        } = visit;
        loop {
            if index >= self.count {
                return Err(FlowError::RunsPastEnd {
                    last: self.count.saturating_sub(1),
                });
            }
            if !self.join(index, &mut barriers, function)? {
                return Ok(());
            }
            self.result.visits = self.result.visits.saturating_add(1);
            if let Some(reachable) = at_mut(&mut self.result.reachable, index) {
                *reachable = true;
            }
            if let Some(owner) = at_mut(&mut self.result.function_of, index) {
                *owner = function;
            }
            match self.step(index, &mut barriers, function)? {
                Some(next) => index = next,
                None => return Ok(()),
            }
        }
    }

    /// Records the state arriving at `index`; returns whether the walk goes on from it (it does
    /// not when the state there already covers this one).
    fn join(
        &mut self,
        index: u32,
        barriers: &mut Barriers,
        function: u32,
    ) -> Result<bool, FlowError> {
        let owner = at(&self.result.function_of, index).copied().unwrap_or(0);
        let Some(slot) = at_mut(&mut self.seen, index) else {
            return Err(FlowError::RunsPastEnd {
                last: self.count.saturating_sub(1),
            });
        };
        if slot.is_none() {
            *slot = Some(barriers.clone());
            return Ok(true);
        }
        let Some(previous) = slot.as_mut() else {
            return Ok(true);
        };
        if owner != function {
            return Err(FlowError::SharedCode { index });
        }
        match self.model {
            BarrierModel::Registers => {
                let merged = merge(previous, barriers);
                if merged == *previous {
                    return Ok(false);
                }
                previous.clone_from(&merged);
                *barriers = merged;
            }
            BarrierModel::Stack => {
                if *previous == *barriers {
                    return Ok(false);
                }
                // Paths that left a convergence region early arrive with a longer stack; the
                // join is outside those regions, so the shorter stack (a common prefix) wins.
                if is_prefix(previous, barriers) {
                    return Ok(false);
                }
                if !is_prefix(barriers, previous) {
                    return Err(FlowError::StacksDiffer { index });
                }
                previous.clone_from(barriers);
            }
        }
        Ok(true)
    }

    fn push(&mut self, index: u32, barriers: Barriers, function: u32) {
        self.work.push(Visit {
            index,
            barriers,
            function,
        });
    }

    /// Applies instruction `index`; returns the next instruction on the path, or `None` where
    /// the path ends.
    fn step(
        &mut self,
        index: u32,
        barriers: &mut Barriers,
        function: u32,
    ) -> Result<Option<u32>, FlowError> {
        let (word, seen) = read(self.code, index)?;
        let mut control = seen.control;
        if let Control::Call(call) = control
            && is_skip_call(self.code, index, call.offset)
        {
            // A jump over the next instruction: a branch to the call's own target.
            control = Control::Bra(OpBra {
                cond: BraCond::Pred(call.cond),
                uniform: false,
                offset: call.offset,
            });
        }
        let guard = seen.guard;
        let second = seen.second;
        // Cannot overflow: the index is below the cap.
        let next = index.saturating_add(1);
        if is_never(guard) {
            // Statically disabled: no effect, and it ends no block.
            return Ok(Some(next));
        }
        match control {
            Control::Plain => {
                if is_warp_wide(opcode_of(word)) && !guard.is_true() {
                    return Err(FlowError::PredicatedWarpWide { index });
                }
                if !seen.info.pure && !guard.is_true() {
                    // A guarded side effect: its own conditional block.
                    self.set_leader(index);
                    self.mark_leader(next, index)?;
                }
                Ok(Some(next))
            }
            Control::Nop => Ok(Some(next)),
            Control::Kill => {
                if !second.is_true() {
                    return Err(FlowError::KillPredicate { index });
                }
                self.set_leader(index);
                self.mark_leader(next, index)?;
                Ok(Some(next))
            }
            Control::Exit(exit) => {
                if exit.keep_ref_count || exit.no_at_exit {
                    return Err(FlowError::SpecialExit { index });
                }
                if guard.is_true() && second.is_true() {
                    return Ok(None);
                }
                if !guard.is_true()
                    && !second.is_true()
                    && at(self.code, next).is_none_or(|&word| needs_virtual_block(word))
                {
                    // The second test needs the virtual location in front of the next
                    // instruction, which that instruction would also use.
                    return Err(FlowError::TwoPredicateExit { index });
                }
                self.set_leader(index);
                self.mark_leader(next, index)?;
                Ok(Some(next))
            }
            Control::Bra(bra) => self.bra(index, bra, barriers, function, guard, second),
            Control::Call(call) => self.call(index, call, barriers, function, guard),
            Control::Ret(ret) => self.ret(index, ret, function, guard, second),
            Control::Brx(brx) => self.brx(index, brx, barriers, function, guard),
            Control::Bssy(bssy) => self.bssy(index, bssy, barriers, guard, second),
            Control::Bsync(bsync) => self.bsync(index, bsync, barriers, function, guard, second),
        }
    }

    fn bra(
        &mut self,
        index: u32,
        bra: OpBra,
        barriers: &Barriers,
        function: u32,
        guard: PredSrc,
        second: PredSrc,
    ) -> Result<Option<u32>, FlowError> {
        let target = branch_target(index, bra.offset)?;
        if bra.uniform || matches!(bra.cond, BraCond::UniformPred(_)) {
            return Err(FlowError::UniformBranch { index });
        }
        self.mark_leader(target, index)?;
        self.push(target, barriers.clone(), function);
        if guard.is_true() && second.is_true() {
            return Ok(None);
        }
        let next = index.saturating_add(1);
        self.mark_leader(next, index)?;
        Ok(Some(next))
    }

    fn call(
        &mut self,
        index: u32,
        call: OpCall,
        barriers: &Barriers,
        function: u32,
        guard: PredSrc,
    ) -> Result<Option<u32>, FlowError> {
        if !call.no_inc || !call.cond.is_true() {
            return Err(FlowError::CallMode { index });
        }
        if !guard.is_true() {
            // The predicate test gets its own block.
            self.set_leader(index);
        }
        let target = branch_target(index, call.offset).ok();
        let link = call_link_register(self.code, index);
        let (Some(target), Some(link)) = (target.filter(|&target| target < self.count), link)
        else {
            return Err(FlowError::CallLinkage { index });
        };
        let callee = if let Some(&callee) = self.function_by_entry.get(&target) {
            if at(&self.result.link_register, callee).copied() != Some(link) {
                return Err(FlowError::InconsistentLinkage { index });
            }
            callee
        } else {
            let callee = u32::try_from(self.result.function_entry.len())
                .map_err(|_| FlowError::CallLinkage { index })?;
            self.result.function_entry.push(target);
            self.result.link_register.push(link);
            self.returns.push(Vec::new());
            self.function_by_entry.insert(target, callee);
            self.mark_leader(target, index)?;
            self.push(target, barriers.clone(), callee);
            callee
        };
        if callee == function {
            return Err(FlowError::Recursion { index });
        }
        if let Some(slot) = at_mut(&mut self.result.call_function, index) {
            *slot = Some(callee);
        }
        let next = index.saturating_add(1);
        self.mark_leader(next, index)?;
        Ok(Some(next))
    }

    fn ret(
        &mut self,
        index: u32,
        ret: OpRet,
        function: u32,
        guard: PredSrc,
        second: PredSrc,
    ) -> Result<Option<u32>, FlowError> {
        if function == 0
            || !second.is_true()
            || !ret.no_dec
            || !ret.cond.is_true()
            || !points_at_start(index, ret.offset)
        {
            return Err(FlowError::ReturnLinkage { index });
        }
        if let Some(returns) = at_mut(&mut self.returns, function) {
            returns.push((index, ret.src.index()));
        }
        if guard.is_true() {
            return Ok(None);
        }
        let next = index.saturating_add(1);
        self.mark_leader(next, index)?;
        Ok(Some(next))
    }

    fn brx(
        &mut self,
        index: u32,
        brx: OpBrx,
        barriers: &Barriers,
        function: u32,
        guard: PredSrc,
    ) -> Result<Option<u32>, FlowError> {
        if !guard.is_true() || !brx.cond.is_true() || !points_at_start(index, brx.offset) {
            return Err(FlowError::BrxMode { index });
        }
        // Table entries that are not instructions of the capture are ignored (spec, "Control-flow
        // recovery").
        let targets: Vec<u32> = self
            .tables
            .targets(index)
            .iter()
            .copied()
            .filter(|&target| target < self.count)
            .collect();
        if targets.is_empty() {
            return Err(FlowError::NoTarget { index });
        }
        for &target in &targets {
            self.mark_leader(target, index)?;
            self.push(target, barriers.clone(), function);
        }
        self.result.indirect_tables.insert(index, targets);
        Ok(None)
    }

    fn bssy(
        &mut self,
        index: u32,
        bssy: OpBSSy,
        barriers: &mut Barriers,
        guard: PredSrc,
        second: PredSrc,
    ) -> Result<Option<u32>, FlowError> {
        let target = branch_target(index, bssy.offset)?;
        if !guard.is_true() || !second.is_true() {
            return Err(FlowError::PredicatedBssy { index });
        }
        let id = bssy.bar.index();
        if target >= self.count {
            return Err(FlowError::TargetOutside {
                index,
                target: i64::from(target),
            });
        }
        match self.model {
            BarrierModel::Registers => match barriers.binary_search_by_key(&id, |entry| entry.0) {
                Ok(slot) => {
                    if let Some(entry) = barriers.get_mut(slot) {
                        entry.1 = Some(target);
                    }
                }
                Err(slot) => barriers.insert(slot, (id, Some(target))),
            },
            BarrierModel::Stack => {
                if barriers.len() >= BARRIERS || barriers.iter().any(|entry| entry.0 == id) {
                    return Err(FlowError::NestedBarrier { index });
                }
                barriers.push((id, Some(target)));
            }
        }
        Ok(Some(index.saturating_add(1)))
    }

    fn bsync(
        &mut self,
        index: u32,
        bsync: OpBSync,
        barriers: &mut Barriers,
        function: u32,
        guard: PredSrc,
        second: PredSrc,
    ) -> Result<Option<u32>, FlowError> {
        if !guard.is_true() || !second.is_true() {
            return Err(FlowError::PredicatedBsync { index });
        }
        let id = bsync.bar.index();
        let target = match self.model {
            BarrierModel::Registers => {
                let Some(slot) = barriers.iter().position(|entry| entry.0 == id) else {
                    // Not known on this path; a later join may supply it.
                    return Ok(None);
                };
                let Some(target) = barriers.get(slot).and_then(|entry| entry.1) else {
                    return Err(FlowError::TargetsDiffer { index });
                };
                barriers.remove(slot);
                target
            }
            BarrierModel::Stack => {
                // The barrier may sit below regions that were left early; those are abandoned.
                let Some(slot) = barriers.iter().rposition(|entry| entry.0 == id) else {
                    return Err(FlowError::MissingBarrier { index });
                };
                let target = barriers.get(slot).and_then(|entry| entry.1);
                barriers.truncate(slot);
                target.ok_or(FlowError::MissingBarrier { index })?
            }
        };
        self.mark_leader(target, index)?;
        let slot = at_mut(&mut self.result.sync_target, index);
        if let Some(slot) = slot {
            if slot.is_some_and(|known| known != target) {
                return Err(FlowError::TargetsDiffer { index });
            }
            *slot = Some(target);
        }
        self.push(target, std::mem::take(barriers), function);
        Ok(None)
    }

    /// Register model: every reachable `BSYNC` that runs must have found its barrier on some
    /// path (`AnalyzeFlowImpl`'s check after the walk).
    fn check_syncs(&self) -> Result<(), FlowError> {
        if self.model != BarrierModel::Registers {
            return Ok(());
        }
        for (index, word) in (0_u32..).zip(self.code) {
            if self.result.is_reachable(index)
                && opcode_of(*word) == crate::decode::BSYNC_OPCODE
                && !is_never(guard_of(*word))
                && self.result.sync_target(index).is_none()
            {
                return Err(FlowError::MissingBarrier { index });
            }
        }
        Ok(())
    }

    /// Each subroutine returns through its link register (or a copy of it), and nothing in it
    /// overwrites the link register. Checked per function in order, its returns before its
    /// code, as the reference does; the first overwrite of each function is found in one pass
    /// over the program, not one pass per function.
    fn check_functions(&self) -> Result<(), FlowError> {
        let functions = self.result.function_entry.len();
        if functions <= 1 {
            return Ok(());
        }
        let targets = self.branch_targets();
        let mut first_overwrite: Vec<Option<u32>> = vec![None; functions];
        for (index, word) in (0_u32..).zip(self.code) {
            let Some(function) = self.result.function_of(index).filter(|&f| f != 0) else {
                continue;
            };
            let info = classify(opcode_of(*word));
            let writes = info.pure && !info.uniform;
            let link = at(&self.result.link_register, function).copied();
            if writes
                && Some(dst_of(*word)) == link
                && let Some(slot) = at_mut(&mut first_overwrite, function)
                && slot.is_none()
            {
                *slot = Some(index);
            }
        }
        for function in 1..functions {
            let link = self
                .result
                .link_register
                .get(function)
                .copied()
                .unwrap_or(0);
            for &(ret, reg) in self.returns.get(function).map_or(&[][..], Vec::as_slice) {
                if reg != link && !self.copies_link(&targets, ret, reg, link) {
                    return Err(FlowError::ReturnRegister { index: ret });
                }
            }
            if let Some(index) = first_overwrite.get(function).copied().flatten() {
                return Err(FlowError::LinkOverwritten { index });
            }
        }
        Ok(())
    }

    /// Which instructions some branch lands on: a target of any `BRA`, `BSSY` or `CALL` word
    /// in the program, or of any `BRX` table (`ReturnRegisterCopiesLink`'s `is_branch_target`,
    /// computed once for the program).
    fn branch_targets(&self) -> Vec<bool> {
        let mut targets = vec![false; self.code.len()];
        for (index, word) in (0_u32..).zip(self.code) {
            if let Some(target) = raw_branch_target(*word, index)
                && let Some(slot) = at_mut(&mut targets, target)
            {
                *slot = true;
            }
        }
        for table in self.result.indirect_tables.values() {
            for &target in table {
                if let Some(slot) = at_mut(&mut targets, target) {
                    *slot = true;
                }
            }
        }
        targets
    }

    /// Whether the `RET` at `ret` may return through `reg` instead of `link`: within
    /// [`RETURN_COPY_LOOKBACK`] straight-line instructions before it, the last write to `reg`
    /// is an unguarded `MOV reg, link` of every lane, and no branch lands after that copy
    /// (`ReturnRegisterCopiesLink`).
    fn copies_link(&self, targets: &[bool], ret: u32, reg: u8, link: u8) -> bool {
        if reg >= RETURN_COPY_REGISTER_LIMIT || link >= RETURN_COPY_REGISTER_LIMIT {
            return false;
        }
        for back in 1..=RETURN_COPY_LOOKBACK.min(ret) {
            let index = ret - back;
            let Some(&word) = at(self.code, index) else {
                return false;
            };
            let info = classify(opcode_of(word));
            if info.kind != OpKind::Plain && info.kind != OpKind::Nop {
                return false;
            }
            if info.pure && !info.uniform && dst_of(word) == reg {
                let copies = matches!(
                    decode(word),
                    Ok(Instr {
                        guard: PredSrc::TRUE,
                        op: Op::Mov(OpMov {
                            src: MovSrc::Reg(src),
                            quad_lanes: OpMov::ALL_LANES,
                            ..
                        }),
                    }) if src.index() == link
                );
                return copies
                    && (index.saturating_add(1)..=ret)
                        .all(|between| !at(targets, between).copied().unwrap_or(false));
            }
        }
        false
    }
}

/// The register at bits 16..24, the destination of the instructions that have one (the
/// reference's `Bits(16, 8)`).
fn dst_of(word: Instruction) -> u8 {
    // Cannot fail: the field is eight bits wide.
    u8::try_from(word.field(DST)).unwrap_or(u8::MAX)
}

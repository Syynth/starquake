//! Control flow in both forms and SSA: branches, loops, `if`, `break`, `return`,
//! `unreachable`, phis taking their values together, the step limit at its edge, and the
//! run-level errors.

use super::{eval_error, run};
use crate::interp::{DEFAULT_STEP_LIMIT, InterpError, Machine};

/// Sums 5 + 4 + 3 + 2 + 1 into r0 with a loop in a control-flow graph, through registers.
const SUM_GRAPH: &str = "block0:
    %0 = SetRegister r0, u32 0
    %1 = SetRegister r1, u32 5
    jump block1
block1:
    %2 = GetRegister r0
    %3 = GetRegister r1
    %4 = IAdd32 %2, %3
    %5 = SetRegister r0, %4
    %6 = ISub32 %3, u32 1
    %7 = SetRegister r1, %6
    %8 = INotEqual %6, u32 0
    branch %8, block1, block2
block2:
    return
";

/// [`SUM_GRAPH`]'s steps: block0's two operations and jump, five passes through block1's seven
/// operations and branch, and block2's return.
const SUM_GRAPH_STEPS: u64 = 3 + 5 * 8 + 1;

/// The same sum in SSA form: the registers become phis.
const SUM_SSA: &str = "block0:
    jump block1
block1:
    %0: u32 = Phi [block0: u32 0], [block1: %2]
    %1: u32 = Phi [block0: u32 5], [block1: %3]
    %2 = IAdd32 %0, %1
    %3 = ISub32 %1, u32 1
    %4 = INotEqual %3, u32 0
    branch %4, block1, block2
block2:
    %5 = SetRegister r0, %2
    return
";

/// The same sum, structured: r1 counts up from 1 to 4 here, adding 1 + 2 + 3 + 4 = 10, and
/// block3, after the `return`, never runs.
const SUM_TREE: &str = "block0:
    %0 = SetRegister r0, u32 0
    %1 = SetRegister r1, u32 0
block1:
    %2 = GetRegister r1
    %3 = IAdd32 %2, u32 1
    %4 = SetRegister r1, %3
    %5 = GetRegister r0
    %6 = IAdd32 %5, %3
    %7 = SetRegister r0, %6
    %8 = IEqual %3, u32 4
block2:
    %9 = SetRegister r2, u32 99
block3:
    %10 = SetRegister r3, u32 1
structure:
    block0
    loop {
        block1
        break %8
    } repeat u1 true
    block2
    return
    block3
";

/// [`SUM_TREE`]'s steps: block0's two operations; three passes of block1's seven operations,
/// the break and the repeat condition; a fourth pass that breaks; block2 and the return.
const SUM_TREE_STEPS: u64 = 2 + 3 * (7 + 1 + 1) + (7 + 1) + 1 + 1;

#[test]
fn a_control_flow_graph_loops_through_registers() {
    let mut machine = Machine::new();
    let finished = run(SUM_GRAPH, &mut machine).expect("runs");
    assert_eq!((machine.regs[0], machine.regs[1]), (15, 0));
    assert_eq!(finished.steps, SUM_GRAPH_STEPS);
}

#[test]
fn ssa_form_gives_the_same_result() {
    let mut machine = Machine::new();
    run(SUM_SSA, &mut machine).expect("runs");
    assert_eq!(machine.regs[0], 15);
}

#[test]
fn a_structured_function_loops_breaks_and_returns() {
    let mut machine = Machine::new();
    let finished = run(SUM_TREE, &mut machine).expect("runs");
    assert_eq!(&machine.regs[..4], [10, 4, 99, 0]);
    assert_eq!(finished.steps, SUM_TREE_STEPS);
}

#[test]
fn the_step_limit_at_its_edge_in_both_forms() {
    for (text, steps) in [(SUM_GRAPH, SUM_GRAPH_STEPS), (SUM_TREE, SUM_TREE_STEPS)] {
        let mut machine = Machine::new();
        machine.step_limit = steps;
        assert_eq!(run(text, &mut machine).map(|f| f.steps), Ok(steps));
        let mut machine = Machine::new();
        machine.step_limit = steps - 1;
        assert_eq!(
            run(text, &mut machine),
            Err(InterpError::StepLimit { limit: steps - 1 })
        );
    }
    let mut machine = Machine::new();
    machine.step_limit = SUM_GRAPH_STEPS - 1;
    let error = run(SUM_GRAPH, &mut machine).err().map(|e| e.to_string());
    assert_eq!(
        error.as_deref(),
        Some("the run stopped at its limit of 43 steps")
    );
    // The machine keeps what the run did before it stopped: the last step refused was the
    // return, after the last pass had stored the sum.
    assert_eq!((machine.regs[0], machine.regs[1]), (15, 0));
}

#[test]
fn infinite_loops_end_at_the_limit() {
    let mut machine = Machine::new();
    machine.step_limit = 10;
    assert_eq!(
        run("block0:\n    jump block0\n", &mut machine),
        Err(InterpError::StepLimit { limit: 10 })
    );
    let empty_loop = "block0:\nstructure:\n    block0\n    loop {\n    } repeat u1 true\n";
    assert_eq!(
        run(empty_loop, &mut machine),
        Err(InterpError::StepLimit { limit: 10 })
    );
    // A function that takes no steps runs under a limit of zero.
    machine.step_limit = 0;
    assert_eq!(
        run("block0:\nstructure:\n    block0\n", &mut machine).map(|f| f.steps),
        Ok(0)
    );
    assert_eq!(
        run("block0:\n    return\n", &mut machine),
        Err(InterpError::StepLimit { limit: 0 })
    );
}

#[test]
fn the_default_step_limit() {
    assert_eq!(DEFAULT_STEP_LIMIT, 1_048_576);
    assert_eq!(Machine::new().step_limit, DEFAULT_STEP_LIMIT);
}

/// Runs `text` with p0 set to `p0` and returns r0 and r1.
fn with_p0(text: &str, p0: bool) -> (u32, u32) {
    let mut machine = Machine::new();
    machine.preds[0] = p0;
    run(text, &mut machine).expect("runs");
    (machine.regs[0], machine.regs[1])
}

#[test]
fn if_runs_its_body_only_when_true() {
    let text = "block0:
    %0 = GetPred p0
    %1 = LogicalNot %0
block1:
    %2 = SetRegister r0, u32 0x1
block2:
    %3 = SetRegister r1, u32 0x2
structure:
    block0
    if %0 {
        block1
    }
    if %1 {
        block2
    }
";
    assert_eq!(with_p0(text, true), (1, 0));
    assert_eq!(with_p0(text, false), (0, 2));
}

#[test]
fn a_branch_takes_each_side() {
    let text = "block0:
    %0 = GetPred p0
    branch %0, block1, block2
block1:
    %1 = SetRegister r0, u32 0x1
    return
block2:
    %2 = SetRegister r1, u32 0x2
    return
";
    assert_eq!(with_p0(text, true), (1, 0));
    assert_eq!(with_p0(text, false), (0, 2));
}

/// Counts r0 up in a loop; at 3, the `if` body ends the loop with `{exit}`; block2, after the
/// `if`, copies the count to r1; block3, after the loop, sets r2.
fn exit_from_if(exit: &str) -> String {
    format!(
        "block0:
    %0 = SetRegister r0, u32 0
block1:
    %1 = GetRegister r0
    %2 = IAdd32 %1, u32 1
    %3 = SetRegister r0, %2
    %4 = IEqual %2, u32 3
block2:
    %5 = SetRegister r1, %2
block3:
    %6 = SetRegister r2, u32 7
structure:
    block0
    loop {{
        block1
        if %4 {{
            {exit}
        }}
        block2
    }} repeat u1 true
    block3
"
    )
}

#[test]
fn break_and_return_leave_through_an_if() {
    let mut machine = Machine::new();
    run(&exit_from_if("break u1 true"), &mut machine).expect("runs");
    assert_eq!(&machine.regs[..3], [3, 2, 7]);
    let mut machine = Machine::new();
    run(&exit_from_if("return"), &mut machine).expect("runs");
    assert_eq!(&machine.regs[..3], [3, 2, 0]);
    // A false break does not leave.
    let mut machine = Machine::new();
    machine.step_limit = 100;
    assert_eq!(
        run(&exit_from_if("break u1 false"), &mut machine),
        Err(InterpError::StepLimit { limit: 100 })
    );
}

#[test]
fn a_loop_with_a_false_repeat_runs_once() {
    let text = "block0:
    %0 = GetRegister r0
    %1 = IAdd32 %0, u32 1
    %2 = SetRegister r0, %1
structure:
    loop {
        block0
    } repeat u1 false
";
    let mut machine = Machine::new();
    run(text, &mut machine).expect("runs");
    assert_eq!(machine.regs[0], 1);
}

#[test]
fn phis_take_their_values_together() {
    // Each pass swaps %0 and %1; a phi that read a value written by an earlier phi of the
    // same block would copy instead.
    let text = "block0:
    jump block1
block1:
    %0: u32 = Phi [block0: u32 1], [block1: %1]
    %1: u32 = Phi [block0: u32 2], [block1: %0]
    %2: u32 = Phi [block0: u32 0], [block1: %3]
    %3 = IAdd32 %2, u32 1
    %4 = ULessThan %3, u32 2
    branch %4, block1, block2
block2:
    %5 = SetRegister r0, %0
    %6 = SetRegister r1, %1
    return
";
    let mut machine = Machine::new();
    run(text, &mut machine).expect("runs");
    assert_eq!((machine.regs[0], machine.regs[1]), (2, 1));
}

#[test]
fn a_structured_phi_takes_the_value_of_the_block_run_last() {
    let text = "block0:
    %0 = GetPred p0
block1:
    %1 = SetRegister r1, u32 0x1
block2:
    %2: u32 = Phi [block0: u32 10], [block1: u32 20]
    %3 = SetRegister r0, %2
structure:
    block0
    if %0 {
        block1
    }
    block2
";
    assert_eq!(with_p0(text, true), (20, 1));
    assert_eq!(with_p0(text, false), (10, 0));
}

#[test]
fn phi_errors() {
    let at_start = "block0:
    %0: u32 = Phi [block1: u32 1]
    jump block1
block1:
    jump block0
";
    assert_eq!(
        run(at_start, &mut Machine::new())
            .err()
            .map(|e| e.to_string()),
        Some("%0 is a phi in the first block run, so no value flows into it".to_owned())
    );
    let no_incoming = "block0:
    %0 = LaneId
block1:
    %1: u32 = Phi [block2: u32 7]
block2:
structure:
    block0
    block1
    block2
";
    assert_eq!(
        run(no_incoming, &mut Machine::new())
            .err()
            .map(|e| e.to_string()),
        Some("%1 has no incoming value from block0".to_owned())
    );
    let not_computed = "block0:
    %0 = LaneId
block1:
    %1: u32 = Phi [block0: %2]
block2:
    %2 = LaneId
structure:
    block0
    block1
    block2
";
    assert_eq!(
        run(not_computed, &mut Machine::new())
            .err()
            .map(|e| e.to_string()),
        Some("%1 reads %2, which the run has not computed".to_owned())
    );
}

#[test]
fn unreachable_is_an_error_in_both_forms() {
    assert_eq!(
        run("block0:\n    unreachable\n", &mut Machine::new())
            .err()
            .map(|e| e.to_string()),
        Some("control reached the unreachable point in the terminator of block0".to_owned())
    );
    assert_eq!(
        run(
            "block0:\nstructure:\n    block0\n    unreachable\n",
            &mut Machine::new()
        )
        .err()
        .map(|e| e.to_string()),
        Some("control reached the unreachable point in the structure".to_owned())
    );
}

#[test]
fn run_level_errors_name_what_went_wrong() {
    assert_eq!(
        run("block0:\n    %0 = LaneId\n", &mut Machine::new())
            .err()
            .map(|e| e.to_string()),
        Some("the function is malformed: block0 has no terminator".to_owned())
    );
    for opcode in ["VoteAll u1 true", "PhiMove u32 0x1, u32 0x2", "Join"] {
        let name = opcode.split(' ').next().unwrap_or_default();
        assert_eq!(
            eval_error(opcode),
            format!("%0: the interpreter does not support {name}")
        );
    }
    // An opaque slot takes any type, but the operation still needs the right one.
    assert_eq!(
        eval_error("LoadGlobal32 u32 0x1000"),
        "%0: argument 0 is u32, not u64"
    );
}

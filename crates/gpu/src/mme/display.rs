//! Disassembly of MME64 instructions, for traces.
//!
//! Ported from `mesa-nouveau/src/nouveau/mme/mme_tu104.c` (`mme_tu104_print_inst`) @ 52238762
//! (MIT, Copyright 2022 Collabora Ltd.). Differences from Mesa's text: method indices are dword
//! indices with the increment shown as `+n` (matching [`MacroHost`](super::MacroHost)), branches
//! print the offset and the comparison outcome that takes them as the interpreter reads them, no
//! 3D method names are looked up, and an `end_next` line marks the last-but-one instruction.

use std::fmt::{self, Display, Formatter};

use super::inst::{AluOp, Instruction, OutOp, Pred, Reg, branch_offset, is_exit_offset};

impl Display for Pred {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.letters())
    }
}

impl Display for AluOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A register as written without an instruction around it: immediates show as `$imm`,
/// `$immpair` and `$imm32`.
impl Display for Reg {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpr(n) => write!(f, "$r{n}"),
            Self::Zero => f.write_str("$zero"),
            Self::Imm => f.write_str("$imm"),
            Self::ImmPair => f.write_str("$immpair"),
            Self::Imm32 => f.write_str("$imm32"),
            Self::Load0 => f.write_str("$load0"),
            Self::Load1 => f.write_str("$load1"),
            Self::Invalid(n) => write!(f, "$invalid{n}"),
        }
    }
}

fn signed16(value: u16) -> i16 {
    value.cast_signed()
}

fn imm32(inst: &Instruction) -> u32 {
    (u32::from(inst.imm[0]) << 16) | u32::from(inst.imm[1])
}

/// A source operand, with immediates resolved against `slot`'s instruction.
fn src(inst: &Instruction, slot: usize, reg: Reg) -> String {
    match reg {
        Reg::Imm => format!("{} /* {:#06x} */", signed16(inst.imm[slot]), inst.imm[slot]),
        Reg::ImmPair => {
            let imm = inst.imm[1 - slot];
            format!("{} /* {imm:#06x} */", signed16(imm))
        }
        Reg::Imm32 => format!("{:#x}", imm32(inst)),
        reg => reg.to_string(),
    }
}

/// The value an output source takes from the immediates, if it is one.
fn out_imm(inst: &Instruction, op: OutOp) -> Option<u32> {
    match op {
        OutOp::Imm0 => Some(u32::from(inst.imm[0])),
        OutOp::Imm1 => Some(u32::from(inst.imm[1])),
        OutOp::ImmHigh0 => Some(u32::from(inst.imm[0] >> 12)),
        OutOp::ImmHigh1 => Some(u32::from(inst.imm[1] >> 12)),
        OutOp::Imm32 => Some(imm32(inst)),
        _ => None,
    }
}

fn out_src(inst: &Instruction, op: OutOp) -> String {
    match op {
        OutOp::Alu0 => "$alu0".to_owned(),
        OutOp::Alu1 => "$alu1".to_owned(),
        OutOp::Load0 => "$load0".to_owned(),
        OutOp::Load1 => "$load1".to_owned(),
        OutOp::Imm0 | OutOp::Imm1 | OutOp::Imm32 => {
            format!("{:#x}", out_imm(inst, op).unwrap_or(0))
        }
        OutOp::ImmHigh0 | OutOp::ImmHigh1 => format!("{}", out_imm(inst, op).unwrap_or(0)),
        OutOp::Reserved => "RESERVED".to_owned(),
        OutOp::None => "none".to_owned(),
        OutOp::Invalid(n) => format!("invalid{n}"),
    }
}

fn alu_line(inst: &Instruction, slot: usize) -> Option<String> {
    let alu = &inst.alu[slot];
    let wanted = if slot == 0 { OutOp::Alu0 } else { OutOp::Alu1 };
    let used_by_out = inst
        .out
        .iter()
        .any(|out| out.mthd == wanted || out.emit == wanted);
    if !used_by_out && alu.dst == Reg::Zero && !alu.op.has_side_effects() {
        return None;
    }

    let prefix = match (used_by_out, alu.dst) {
        (true, Reg::Zero) => format!("$alu{slot} = "),
        (true, dst) => format!("$alu{slot}, {dst} = "),
        (false, Reg::Zero) => String::new(),
        (false, dst) => format!("{dst} = "),
    };

    let imm = inst.imm[slot];
    let a = src(inst, slot, alu.src[0]);
    let b = src(inst, slot, alu.src[1]);
    let body = match alu.op {
        AluOp::Addc => format!("ADDC {a} {b} $carry"),
        AluOp::Subb => format!("SUBB {a} {b} $borrow"),
        AluOp::Merge => format!(
            "MERGE {a} {b} ({}, {}, {})",
            (imm >> 10) & 0x3f,
            (imm >> 5) & 0x1f,
            imm & 0x1f
        ),
        AluOp::State if alu.src[1] == Reg::Zero => format!("STATE {a}"),
        AluOp::State => format!("STATE {a} + {b}"),
        AluOp::Jal => format!("JAL ({:#06x})", imm & 0x7fff),
        AluOp::Loop => format!("LOOP {a} ({imm})"),
        AluOp::Dread => format!("DREAD {a}"),
        AluOp::Dwrite => format!("DWRITE {a} {b}"),
        AluOp::Mulh | AluOp::Extended if alu.src == [Reg::Zero; 2] => alu.op.name().to_owned(),
        AluOp::Blt | AluOp::Bltu | AluOp::Ble | AluOp::Bleu | AluOp::Beq => {
            let target = if is_exit_offset(imm) {
                "exit".to_owned()
            } else {
                branch_offset(imm).to_string()
            };
            format!("{} {a} {b} ({target}, {})", alu.op, imm & 0x8000 != 0)
        }
        op => format!("{op} {a} {b}"),
    };
    Some(format!("{prefix}{body}"))
}

fn out_lines(inst: &Instruction, slot: usize, lines: &mut Vec<String>) {
    let out = &inst.out[slot];
    if out.mthd != OutOp::None {
        lines.push(match out_imm(inst, out.mthd) {
            Some(value) => format!("mthd({:#06x}, +{})", value & 0xfff, (value >> 12) & 0xf),
            None => format!("mthd({})", out_src(inst, out.mthd)),
        });
    }
    if out.emit != OutOp::None {
        lines.push(format!("emit({})", out_src(inst, out.emit)));
    }
}

/// Mesa's textual form of one instruction: one line per active ALU slot, method selector and
/// emit, wrapped in `pred MODE $reg { ... }` when the instruction is predicated. An instruction
/// that does nothing prints `nop`. There is no trailing newline.
impl Display for Instruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut lines = Vec::new();
        lines.extend((0..2).filter_map(|slot| alu_line(self, slot)));
        for slot in 0..2 {
            out_lines(self, slot, &mut lines);
        }
        if self.end_next {
            lines.push("end_next".to_owned());
        }
        if lines.is_empty() {
            lines.push("nop".to_owned());
        }

        if self.pred_mode == Pred::Uuuu {
            return f.write_str(&lines.join("\n"));
        }
        writeln!(f, "pred {} {} {{", self.pred_mode, self.pred)?;
        for line in &lines {
            writeln!(f, "    {line}")?;
        }
        f.write_str("}")
    }
}

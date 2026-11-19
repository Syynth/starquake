//! Lookup-type tests: a fake cache built from the types, the way G18's fake and slice 19's
//! cache will use them.

use super::*;
use crate::asm::assemble_program;
use crate::decode::{Instr, Op, OpExit, PredSrc};
use crate::exchange::{
    Answers, ExchangeError, MAX_QUESTION_ROUNDS, Question, Questioner, TextureKind, exchange,
};
use crate::program::translate_program;
use crate::translate::{COLOR_TARGETS, ColorOutputs, Stage};

/// Guest memory with one mapped range.
struct FakeMemory {
    base: u64,
    bytes: Vec<u8>,
}

impl ProgramMemory for FakeMemory {
    fn read(&self, address: u64, out: &mut [u8]) -> Result<(), Unmapped> {
        let len = u64::try_from(out.len()).map_err(|_| Unmapped { address })?;
        let mapped_end = self.base + u64::try_from(self.bytes.len()).expect("small");
        if address < self.base || address >= mapped_end {
            return Err(Unmapped { address });
        }
        let end = address.checked_add(len).ok_or(Unmapped {
            address: mapped_end,
        })?;
        if end > mapped_end {
            return Err(Unmapped {
                address: mapped_end,
            });
        }
        let start = usize::try_from(address - self.base).expect("small");
        out.copy_from_slice(&self.bytes[start..start + out.len()]);
        Ok(())
    }
}

/// Engine state whose constant-buffer word is its offset plus the bank.
struct FakeState;

impl EngineState for FakeState {
    fn cbuf_word(&self, _stage: GuestStage, bank: u8, offset: u32) -> Option<u32> {
        Some(offset + u32::from(bank))
    }

    fn texture_kind(&self, _stage: GuestStage, _handle: u32) -> Option<TextureKind> {
        None
    }
}

/// A program whose questions chain `depth` deep.
struct Chain(u32);

impl Questioner for Chain {
    fn questions(&self, answers: &Answers) -> Vec<Question> {
        let mut out = Vec::new();
        let mut offset = 0;
        for _ in 0..self.0 {
            out.push(Question::CbufWord { bank: 4, offset });
            match answers.cbuf_word(4, offset) {
                Some(Some(next)) => offset = next,
                _ => break,
            }
        }
        out
    }
}

fn exit_program() -> Vec<u8> {
    assemble_program(&[Instr::new(Op::Exit(OpExit {
        keep_ref_count: false,
        no_at_exit: false,
        pred: PredSrc::TRUE,
    }))])
    .expect("encodable")
}

/// A stand-in for slice 19's cache: reads each requested program, runs its exchange, and
/// translates it. The request's stage reaches the exchange.
fn fake_lookup(
    requests: &[ProgramRequest],
    mem: &dyn ProgramMemory,
    engine: &dyn EngineState,
    questioner: &dyn Questioner,
) -> Lookup {
    let mut stages = Vec::new();
    for request in requests {
        let mut code = vec![0; exit_program().len()];
        if let Err(unmapped) = mem.read(request.address, &mut code) {
            return Lookup::Pending(Ticket::new(unmapped.address));
        }
        if let Err(error) = exchange(request.stage, questioner, engine) {
            return Lookup::Rejected(Arc::new(Rejection::from(error)));
        }
        let stage = Stage {
            stage: request.stage,
            color_outputs: ColorOutputs {
                targets: [0; COLOR_TARGETS],
            },
        };
        match translate_program(stage, &code) {
            Ok(shader) => stages.push(shader),
            Err(rejection) => return Lookup::Rejected(Arc::new(rejection)),
        }
    }
    Lookup::Ready(Arc::new(ShaderSet { stages }))
}

const BASE: u64 = 0x1_0000_2000;

#[test]
fn a_lookup_is_ready_pending_or_rejected() {
    let mem = FakeMemory {
        base: BASE,
        bytes: exit_program(),
    };
    let vertex = ProgramRequest {
        stage: GuestStage::Vertex,
        address: BASE,
    };
    let ready = fake_lookup(&[vertex], &mem, &FakeState, &Chain(MAX_QUESTION_ROUNDS));
    assert!(
        matches!(&ready, Lookup::Ready(set) if set.stages.len() == 1),
        "{ready:?}"
    );
    let rejected = fake_lookup(&[vertex], &mem, &FakeState, &Chain(MAX_QUESTION_ROUNDS + 1));
    let expected = Rejection::Questions(ExchangeError::TooManyRounds {
        rounds: MAX_QUESTION_ROUNDS,
    });
    assert!(
        matches!(&rejected, Lookup::Rejected(rejection) if **rejection == expected),
        "{rejected:?}"
    );
    assert_eq!(
        expected.to_string(),
        format!(
            "the program still asks new questions after {MAX_QUESTION_ROUNDS} rounds of answers"
        )
    );
    let unmapped = ProgramRequest {
        stage: GuestStage::Vertex,
        address: BASE - 0x10,
    };
    let pending = fake_lookup(&[unmapped], &mem, &FakeState, &Chain(0));
    assert!(
        matches!(pending, Lookup::Pending(ticket) if ticket == Ticket::new(BASE - 0x10)),
        "{pending:?}"
    );
}

#[test]
fn unmapped_reads_name_the_first_unmapped_address() {
    let mem = FakeMemory {
        base: BASE,
        bytes: vec![0xa5; 0x20],
    };
    let mut out = [0; 0x10];
    assert_eq!(mem.read(BASE + 0x10, &mut out), Ok(()));
    assert_eq!(out, [0xa5; 0x10]);
    // A read that runs one byte past the mapped end.
    assert_eq!(
        mem.read(BASE + 0x11, &mut out),
        Err(Unmapped {
            address: BASE + 0x20
        })
    );
    assert_eq!(
        Unmapped {
            address: BASE + 0x20
        }
        .to_string(),
        "guest address 0x100002020 is not mapped"
    );
}

#[test]
fn tickets_keep_their_number() {
    let first = Ticket::new(3);
    let second = Ticket::new(0x1_0000_0000);
    assert_eq!(first.id(), 3);
    assert_eq!(second.id(), 0x1_0000_0000);
    assert_ne!(first, second);
    assert!(first < second);
}

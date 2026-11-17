//! Slice 6h, control and warp: `EXIT` is decoded; the other instructions are shells.
//!
//! `EXIT`'s encoding is NAK's `impl SM70Op for OpExit`
//! (`mesa-nouveau/src/nouveau/compiler/nak/sm70_encode.rs` @ 52238762, MIT, Copyright 2022
//! Collabora, Ltd.), which writes the mode and predicate bits as constants; [`OpExit`] carries
//! them as fields so the decoder can read any value of them.

use std::fmt::{self, Display, Formatter};

use super::super::fields::mask_of;
use super::super::form::{Form, Reader};
use super::super::instruction::Field;
use super::super::operand::{Pred, PredSrc};
use super::super::print::flag;
use super::{Named, Op, shells};

/// `EXIT` (`impl SM70Op for OpExit`: `set_opcode(0x94d)`).
pub(crate) const EXIT_OPCODE: u16 = 0x94d;
/// `EXIT`'s first mode bit; NAK's comment names it `./.KEEPREFCOUNT/.PREEMPTED/.INVALID3`
/// together with bit 85 (`set_field(84..85, false)`).
pub(crate) const EXIT_KEEP_REF_COUNT: Field = Field::bit(84);
/// `EXIT`'s `.NO_ATEXIT` bit (`set_field(85..86, false)`).
pub(crate) const EXIT_NO_AT_EXIT: Field = Field::bit(85);
/// `EXIT`'s predicate register (`set_field(87..90, 0x7)`, "TODO: Predicate").
pub(crate) const EXIT_PRED: Field = Field::new(87, Pred::BITS);
/// `EXIT`'s predicate negation (`set_field(90..91, false)`, "NOT").
pub(crate) const EXIT_PRED_NOT: Field = Field::bit(90);

/// `EXIT`: ends the thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpExit {
    /// Bit 84 (NAK: `./.KEEPREFCOUNT/.PREEMPTED/.INVALID3` with bit 85).
    pub keep_ref_count: bool,
    /// Bit 85 (NAK: `.NO_ATEXIT`).
    pub no_at_exit: bool,
    /// The exit's own predicate (NAK always writes `PT`).
    pub pred: PredSrc,
}

impl Named for OpExit {
    const MNEMONIC: &'static str = "exit";
}

impl Display for OpExit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("exit")?;
        flag(f, self.keep_ref_count, ".keeprefcount")?;
        flag(f, self.no_at_exit, ".no_atexit")?;
        if !self.pred.is_true() {
            write!(f, " {}", self.pred)?;
        }
        Ok(())
    }
}

fn decode_exit(reader: &mut Reader) -> Op {
    Op::Exit(OpExit {
        keep_ref_count: reader.flag(EXIT_KEEP_REF_COUNT),
        no_at_exit: reader.flag(EXIT_NO_AT_EXIT),
        pred: reader.pred_src(EXIT_PRED, EXIT_PRED_NOT),
    })
}

/// The control group's forms.
pub(crate) const FORMS: &[Form] = &[Form::new(
    EXIT_OPCODE,
    mask_of(&[
        EXIT_KEEP_REF_COUNT,
        EXIT_NO_AT_EXIT,
        EXIT_PRED,
        EXIT_PRED_NOT,
    ]),
    |reader| Ok(decode_exit(reader)),
)];

shells! {
    Bra(OpBra) = "bra";
    Brx(OpBrx) = "brx";
    Jmx(OpJmx) = "jmx";
    Call(OpCall) = "call";
    Ret(OpRet) = "ret";
    Kill(OpKill) = "kill";
    BSSy(OpBSSy) = "bssy";
    BSync(OpBSync) = "bsync";
    Break(OpBreak) = "break";
    BMov(OpBMov) = "bmov";
    WarpSync(OpWarpSync) = "warpsync";
    Bar(OpBar) = "bar";
    Vote(OpVote) = "vote";
    Shfl(OpShfl) = "shfl";
    Match(OpMatch) = "match";
    Redux(OpRedux) = "redux";
    Nop(OpNop) = "nop";
    DepBar(OpDepBar) = "depbar";
    Nanosleep(OpNanosleep) = "nanosleep";
    Bpt(OpBpt) = "bpt";
}

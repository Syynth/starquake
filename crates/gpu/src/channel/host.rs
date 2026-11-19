//! The host methods of class `C76F` (`classes/clc76f.h`): the table in the parent module's
//! docs, and what each one does to the channel.
//!
//! Method numbers are the generated tables' byte addresses / 4 ([`crate::classes::Method::method`]),
//! never restated here. Where Eden-NX2 (`eden-nx2/src/video_core/engines/puller.cpp` @ 50fa86bb,
//! `CallPullerMethod`) does something else, the parent module's docs list it.

use crate::classes::clc76f;
use crate::fifo::GP_ENTRY_GET_HI_SHIFT;
use crate::memory::GpuMemory;
use crate::ring::{ClassId, Invalidation, Release, ReleaseOp, RingEntry, Subchannel};
use crate::sema::{self, HostSemaphore, PayloadSize, ReleaseForm};

use super::{
    Acquire, AcquireMethod, Channel, ChannelError, ChannelSink, DroppedWord, HostRegs, Step,
};

/// Methods below this are host methods, whatever the subchannel (Eden-NX2's
/// `BufferMethods::NonPullerMethods`, `dma_pusher.h`; spec, "Protocol facts").
pub const HOST_METHODS_END: u32 = 0x40;

/// Where a semaphore's upper address register lands in the address: bit 32. `SEMAPHOREA`
/// (`OFFSET_UPPER`, 7:0) and `SEM_ADDR_HI` (`OFFSET`, 7:0) hold address bits 39:32, as the GP
/// entry's `GET_HI` does (Eden-NX2's `SemaphoreAddress()` in `puller.h` and
/// `ProcessSemaphoreExecute` shift by 32).
const SEMAPHORE_UPPER_SHIFT: u32 = GP_ENTRY_GET_HI_SHIFT;

pub(super) const SET_OBJECT: u32 = clc76f::SET_OBJECT.method();
pub(super) const ILLEGAL: u32 = clc76f::ILLEGAL.method();
pub(super) const NOP: u32 = clc76f::NOP.method();
pub(super) const SEMAPHOREA: u32 = clc76f::SEMAPHOREA.method();
pub(super) const SEMAPHOREB: u32 = clc76f::SEMAPHOREB.method();
pub(super) const SEMAPHOREC: u32 = clc76f::SEMAPHOREC.method();
pub(super) const SEMAPHORED: u32 = clc76f::SEMAPHORED.method();
pub(super) const NON_STALL_INTERRUPT: u32 = clc76f::NON_STALL_INTERRUPT.method();
pub(super) const FB_FLUSH: u32 = clc76f::FB_FLUSH.method();
pub(super) const MEM_OP_A: u32 = clc76f::MEM_OP_A.method();
pub(super) const MEM_OP_B: u32 = clc76f::MEM_OP_B.method();
pub(super) const MEM_OP_C: u32 = clc76f::MEM_OP_C.method();
pub(super) const MEM_OP_D: u32 = clc76f::MEM_OP_D.method();
pub(super) const SET_REFERENCE: u32 = clc76f::SET_REFERENCE.method();
pub(super) const SEM_ADDR_LO: u32 = clc76f::SEM_ADDR_LO.method();
pub(super) const SEM_ADDR_HI: u32 = clc76f::SEM_ADDR_HI.method();
pub(super) const SEM_PAYLOAD_LO: u32 = clc76f::SEM_PAYLOAD_LO.method();
pub(super) const SEM_PAYLOAD_HI: u32 = clc76f::SEM_PAYLOAD_HI.method();
pub(super) const SEM_EXECUTE: u32 = clc76f::SEM_EXECUTE.method();
pub(super) const WFI: u32 = clc76f::WFI.method();
pub(super) const YIELD: u32 = clc76f::YIELD.method();
pub(super) const CLEAR_FAULTED: u32 = clc76f::CLEAR_FAULTED.method();

impl HostRegs {
    /// The `SEMAPHOREA`/`B` address: bits 39:32 from `OFFSET_UPPER`, bits 31:2 from
    /// `OFFSET_LOWER` in place.
    fn semaphored_address(&self) -> u64 {
        semaphore_address(
            clc76f::SEMAPHOREA_OFFSET_UPPER.get(self.semaphore_a),
            self.semaphore_b,
            clc76f::SEMAPHOREB_OFFSET_LOWER,
        )
    }

    /// The `SEM_ADDR_HI`/`LO` address, laid out as [`HostRegs::semaphored_address`].
    fn sem_execute_address(&self) -> u64 {
        semaphore_address(
            clc76f::SEM_ADDR_HI_OFFSET.get(self.sem_addr_hi),
            self.sem_addr_lo,
            clc76f::SEM_ADDR_LO_OFFSET,
        )
    }

    /// The `SEM_PAYLOAD_LO`/`HI` payload at `size`: only the low word for a 32-bit payload.
    fn sem_execute_payload(&self, size: PayloadSize) -> u64 {
        let low = u64::from(self.sem_payload_lo);
        match size {
            PayloadSize::Four => low,
            PayloadSize::Eight => (u64::from(self.sem_payload_hi) << u32::BITS) | low,
        }
    }
}

/// An address from an upper register's field value and a lower register whose `lower` field
/// holds bits 31:`lower.low()` in place.
fn semaphore_address(upper: u32, lower_word: u32, lower: crate::field::Field) -> u64 {
    let low = u64::from(lower.get(lower_word)) << lower.low();
    (u64::from(upper) << SEMAPHORE_UPPER_SHIFT) | low
}

impl Channel {
    /// Runs one host method word.
    pub(super) fn host_method<M, S>(
        &mut self,
        va: u64,
        subchannel: Subchannel,
        method: u16,
        value: u32,
        memory: &M,
        sink: &mut S,
    ) -> Result<Step, ChannelError>
    where
        M: GpuMemory + ?Sized,
        S: ChannelSink + ?Sized,
    {
        match u32::from(method) {
            SET_OBJECT => self.bind(subchannel, value, sink),
            SEMAPHOREA => self.regs.semaphore_a = value,
            SEMAPHOREB => self.regs.semaphore_b = value,
            SEMAPHOREC => self.regs.semaphore_c = value,
            SEMAPHORED => match sema::decode_semaphored(value) {
                Ok(op) => {
                    let address = self.regs.semaphored_address();
                    let payload = self.regs.semaphore_c;
                    return self.semaphore(
                        op,
                        address,
                        u64::from(payload),
                        AcquireMethod::Semaphored,
                        memory,
                        sink,
                    );
                }
                Err(error) => self.drop_word(va, DroppedWord::BadSemaphore(error), sink),
            },
            SEM_ADDR_LO => self.regs.sem_addr_lo = value,
            SEM_ADDR_HI => self.regs.sem_addr_hi = value,
            SEM_PAYLOAD_LO => self.regs.sem_payload_lo = value,
            SEM_PAYLOAD_HI => self.regs.sem_payload_hi = value,
            SEM_EXECUTE => match sema::decode_sem_execute(value) {
                Ok(op) => {
                    let size = match op {
                        HostSemaphore::Acquire { size, .. }
                        | HostSemaphore::Reduce { size, .. } => size,
                        HostSemaphore::Release { form, .. } => form.size,
                    };
                    let address = self.regs.sem_execute_address();
                    let payload = self.regs.sem_execute_payload(size);
                    return self.semaphore(
                        op,
                        address,
                        payload,
                        AcquireMethod::SemExecute,
                        memory,
                        sink,
                    );
                }
                Err(error) => self.drop_word(va, DroppedWord::BadSemaphore(error), sink),
            },
            MEM_OP_D => match Invalidation::from_mem_op_d(value) {
                Some(what) => sink.entry(RingEntry::Invalidate {
                    channel: self.id,
                    what,
                }),
                None => self.drop_word(
                    va,
                    DroppedWord::UndefinedMemOp {
                        operation: clc76f::MEM_OP_D_OPERATION.get(value),
                    },
                    sink,
                ),
            },
            SET_REFERENCE => {
                self.regs.reference = value;
                sink.set_reference(value);
            }
            WFI => sink.entry(RingEntry::WaitForIdle { channel: self.id }),
            // Operands of MEM_OP_D the back does not use yet, and the methods that do nothing
            // here (**guess** for ILLEGAL, NON_STALL_INTERRUPT, YIELD and CLEAR_FAULTED; module
            // docs).
            MEM_OP_A | MEM_OP_B | MEM_OP_C | ILLEGAL | NOP | NON_STALL_INTERRUPT | FB_FLUSH
            | YIELD | CLEAR_FAULTED => {}
            _ => self.drop_word(va, DroppedWord::UnknownHostMethod { method, value }, sink),
        }
        Ok(Step::Continue)
    }

    /// `SET_OBJECT`: binds `NVCLASS` on the subchannel, or unbinds it for class 0, and tells
    /// the back.
    fn bind<S>(&mut self, subchannel: Subchannel, value: u32, sink: &mut S)
    where
        S: ChannelSink + ?Sized,
    {
        let class = ClassId(clc76f::SET_OBJECT_NVCLASS.get(value));
        if let Some(slot) = self.bindings.get_mut(usize::from(subchannel.get())) {
            *slot = (class.0 != 0).then_some(class);
        }
        sink.entry(RingEntry::Bind {
            channel: self.id,
            subchannel,
            class,
        });
    }

    /// A decoded `SEMAPHORED` or `SEM_EXECUTE`: an acquire is checked here, a release or
    /// reduction goes to the back.
    fn semaphore<M, S>(
        &mut self,
        op: HostSemaphore,
        address: u64,
        payload: u64,
        method: AcquireMethod,
        memory: &M,
        sink: &mut S,
    ) -> Result<Step, ChannelError>
    where
        M: GpuMemory + ?Sized,
        S: ChannelSink + ?Sized,
    {
        match op {
            // `size` and `switch` are not used: the acquire compares the 32-bit word with the
            // payload's low word (**guess** for a 64-bit `SEM_EXECUTE` payload; module docs).
            HostSemaphore::Acquire { op, .. } => {
                let acquire = Acquire {
                    address,
                    op,
                    payload: low_word(payload),
                    method,
                };
                return self.acquire(acquire, memory);
            }
            // `wfi` changes nothing (module docs, "Guesses").
            HostSemaphore::Release { form, .. } => sink.entry(RingEntry::Release(Release {
                address,
                payload,
                form,
                op: ReleaseOp::Write,
            })),
            HostSemaphore::Reduce {
                reduction, size, ..
            } => sink.entry(RingEntry::Release(Release {
                address,
                payload,
                form: ReleaseForm {
                    size,
                    timestamp: false,
                },
                op: ReleaseOp::Reduce(reduction),
            })),
        }
        Ok(Step::Continue)
    }
}

/// The low 32 bits of `value`.
fn low_word(value: u64) -> u32 {
    u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX)
}

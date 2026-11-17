//! The dispatch arm of `Break` (0x26): decodes the call ([`exception::break_execution`]), and
//! then either returns to the guest or ends the dispatch in [`Outcome::Broke`].
//!
//! - **A notification-only break** ([`exception::NOTIFICATION_ONLY_FLAG`] set) returns to the
//!   guest, as Eden's `Break` does. `Break` has no outputs (its convention lists none, and Eden's
//!   wrapper `SvcWrap_Break64`, `eden/src/core/hle/kernel/svc.cpp` @ cf654052, lines
//!   1518-1523, writes no register), so no register changes. It ends in
//!   [`Outcome::Completed`] and is recorded like any call with no `Result` output: as success.
//! - **Any other break** ends the dispatch in [`Outcome::Broke`], with the decoded call. No
//!   register is written. The call is recorded (as success, its convention having no `Result`)
//!   before the caller sees the outcome, so the event stream shows the break the run ended on.
//!   The caller ends the run as a guest crash; the process's tables are left as they are, as
//!   for any other guest crash. Horizon exits the process instead (`svc/exception.rs` lists
//!   the divergence).

use super::{Call, Dispatcher, Outcome, complete, fault, mismatch, word};
use crate::svc::{Convention, SvcId, exception};

impl Dispatcher {
    /// `Break`'s arm. It declares no table: it reads only the caller's memory.
    pub(super) fn break_call(
        &self,
        call: &mut Call<'_, '_>,
        id: SvcId,
        convention: &Convention,
    ) -> Outcome {
        self.call::<()>(call, |(), call, _| {
            let [reason, address, size] = match convention.read::<3>(call.regs) {
                Ok(inputs) => inputs,
                Err(error) => return mismatch(id, error, call.regs),
            };
            let reason = match word(reason) {
                Ok(reason) => reason,
                Err(error) => return fault(id, &error.to_string(), call.regs),
            };
            let broke = exception::break_execution(call.caller.memory, reason, address, size);
            if broke.notification_only {
                complete(id, convention, call.regs, &[])
            } else {
                Outcome::Broke(broke)
            }
        })
    }
}

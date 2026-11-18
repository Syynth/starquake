//! Events a service keeps and nothing signals: the one rule `appletOE` and `aoc:u` share for
//! handing one out.

use starquake_kernel::id::ObjectId;

use crate::framework::{Context, Fault};

/// The event a service keeps (`kept`), checked, or a new one the kernel makes when it keeps
/// none. The caller keeps what this returns.
///
/// A kept event is checked first with [`Context::clear_event`], which the kernel answers
/// `InvalidHandle` for any id that is not an event the calling server made (the kernel's
/// `IpcContext::clear_event`), so a restored id that names another object, or nothing, is
/// refused before a handle to it is copied. On an event the server made the clear changes
/// nothing: nothing signals these events, and the guest is given only their readable end,
/// which it cannot signal.
pub(crate) fn unsignalled_event(
    cx: &mut Context<'_>,
    kept: Option<ObjectId>,
) -> Result<ObjectId, Fault> {
    if let Some(event) = kept {
        cx.clear_event(event)?;
        return Ok(event);
    }
    cx.create_event()
}

//! Conditional rendering: whether the guest lets draws render (gpu-protocol spec, "3D: from
//! state to a draw": "Conditional rendering (`SET_RENDER_ENABLE_*`) is evaluated here from the
//! query value in guest memory").
//!
//! A port of Eden-NX2's `ProcessQueryCondition`
//! (`eden-nx2/src/video_core/engines/maxwell_3d.cpp` @ 50fa86bb, GPL-3.0-or-later), with the
//! header's names (`classes/clc797.h`): `SET_RENDER_ENABLE_OVERRIDE`'s `ALWAYS_RENDER` and
//! `NEVER_RENDER` decide alone; with `USE_RENDER_ENABLE`, `SET_RENDER_ENABLE_C`'s mode does:
//! `TRUE` renders, `FALSE` does not, and the other three read the six-word report Eden calls
//! `ReportSemaphore::Compare` at `SET_RENDER_ENABLE_A/B`'s address (`initial_sequence`,
//! `initial_mode`, two unknown words, `current_sequence`, `current_mode`):
//!
//! - `CONDITIONAL`: renders when the initial sequence and the initial mode are both non-zero;
//! - `RENDER_IF_EQUAL`: when the initial sequence and mode equal the current ones;
//! - `RENDER_IF_NOT_EQUAL`: when either differs.
//!
//! **When.** As in Eden, the answer is worked out when `SET_RENDER_ENABLE_C` is written, in
//! stream order on the back thread, and kept for every draw until the next such write
//! ([`super::Draws::method`] calls [`render_enabled`] for each such word); before the first
//! write every draw renders (Eden's `execute_on{true}`), although the register then holds
//! `FALSE`. A change to the override, the address or the report after the mode write changes
//! nothing until the mode is written again, as in Eden.
//!
//! # Divergences from Eden-NX2
//!
//! - **An undefined mode or override is an error** ([`DrawError::State`]), where Eden renders
//!   for an undefined mode and logs; the kept answer is then unchanged. The mode field is three
//!   bits and the header names five values; the override field is two bits and names three.
//!   `tests::an_undefined_render_enable_mode_is_an_error`.
//! - **A report that cannot be read is an error** ([`DrawError::Fault`]), and the kept answer is
//!   unchanged; Eden's `ReadBlock` reads whatever the memory manager gives.
//!
//! The catch-up the spec asks for "when that value is still pending" (a release not yet
//! performed will write it) is the executor's (slice G19, `back::exec`): before it hands the
//! draw path a `SET_RENDER_ENABLE_C` word, it makes every pending release of [`report_range`]
//! visible. This reads guest memory as it is.

use core::ops::Range;

use super::DrawError;
use super::shaders::read_words;
use crate::back::state::Mirror;
use crate::classes::clc797;
use crate::memory::GpuMemory;

/// Words in the report a conditional mode reads (Eden's `ReportSemaphore::Compare`).
pub const REPORT_WORDS: usize = 6;

/// The bytes of the report `SET_RENDER_ENABLE_C`'s mode reads ([`REPORT_WORDS`] words at
/// `SET_RENDER_ENABLE_A/B`), for a mode that reads one; `None` when the override or the mode
/// decides alone, when a field holds a value the header does not define (the draw path then
/// refuses the mode), or when the report would run past the end of the address space.
pub fn report_range(mirror: &Mirror) -> Option<Range<u64>> {
    let Ok(Decision::Report { address, .. }) = decide(mirror) else {
        return None;
    };
    let bytes = u64::try_from(REPORT_WORDS * size_of::<u32>()).ok()?;
    Some(address..address.checked_add(bytes)?)
}

/// What decides the next draw: the override or the mode alone, or a report in memory.
enum Decision {
    /// The override or the mode decides.
    Decided(bool),
    /// The report at `address` decides, compared as `mode` says.
    Report { mode: u32, address: u64 },
}

/// The override and the mode, read from the mirror (module docs).
fn decide(mirror: &Mirror) -> Result<Decision, DrawError> {
    let override_mode = mirror
        .reg(&clc797::SET_RENDER_ENABLE_OVERRIDE, &[])?
        .named(clc797::SET_RENDER_ENABLE_OVERRIDE_MODE)?
        .value;
    match override_mode {
        clc797::SET_RENDER_ENABLE_OVERRIDE_MODE_ALWAYS_RENDER => {
            return Ok(Decision::Decided(true));
        }
        clc797::SET_RENDER_ENABLE_OVERRIDE_MODE_NEVER_RENDER => {
            return Ok(Decision::Decided(false));
        }
        _ => {}
    }
    let mode = mirror
        .reg(&clc797::SET_RENDER_ENABLE_C, &[])?
        .named(clc797::SET_RENDER_ENABLE_C_MODE)?
        .value;
    match mode {
        clc797::SET_RENDER_ENABLE_C_MODE_TRUE => return Ok(Decision::Decided(true)),
        clc797::SET_RENDER_ENABLE_C_MODE_FALSE => return Ok(Decision::Decided(false)),
        _ => {}
    }
    let upper = mirror
        .reg(&clc797::SET_RENDER_ENABLE_A, &[])?
        .field(clc797::SET_RENDER_ENABLE_A_OFFSET_UPPER)?;
    let lower = mirror
        .reg(&clc797::SET_RENDER_ENABLE_B, &[])?
        .field(clc797::SET_RENDER_ENABLE_B_OFFSET_LOWER)?;
    let address = (u64::from(upper) << u32::BITS) | u64::from(lower);
    Ok(Decision::Report { mode, address })
}

/// Whether the next draw renders (module docs).
pub fn render_enabled<M: GpuMemory + ?Sized>(
    mirror: &Mirror,
    memory: &M,
) -> Result<bool, DrawError> {
    let (mode, address) = match decide(mirror)? {
        Decision::Decided(renders) => return Ok(renders),
        Decision::Report { mode, address } => (mode, address),
    };
    let [
        initial_sequence,
        initial_mode,
        _,
        _,
        current_sequence,
        current_mode,
    ] = read_words::<REPORT_WORDS, M>(memory, address)?;
    Ok(match mode {
        clc797::SET_RENDER_ENABLE_C_MODE_CONDITIONAL => initial_sequence != 0 && initial_mode != 0,
        clc797::SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL => {
            initial_sequence == current_sequence && initial_mode == current_mode
        }
        // `RENDER_IF_NOT_EQUAL`, the last value the getter accepts.
        _ => initial_sequence != current_sequence || initial_mode != current_mode,
    })
}

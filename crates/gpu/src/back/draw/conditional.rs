//! Conditional rendering: whether the guest let the next draw render (gpu-protocol spec, "3D:
//! from state to a draw": "Conditional rendering (`SET_RENDER_ENABLE_*`) is evaluated here from
//! the query value in guest memory").
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
//! # Divergences from Eden-NX2
//!
//! - **Evaluated at each draw**, as the spec says, where Eden evaluates once when the mode
//!   register is written and keeps the answer: a change to the report between the mode write and
//!   a draw changes the answer here. `tests::conditional_render_reads_the_report_at_the_draw`.
//! - **An undefined mode or override is an error** ([`DrawError::State`]), where Eden renders
//!   for an undefined mode and logs. The mode field is three bits and the header names five
//!   values; the override field is two bits and names three.
//!   `tests::an_undefined_render_enable_mode_is_an_error`.
//! - **A report that cannot be read is an error** ([`DrawError::Fault`]); Eden's `ReadBlock`
//!   reads whatever the memory manager gives.
//!
//! The catch-up the spec asks for "when that value is still pending" (a release not yet
//! performed will write it) is the executor's (slice G19): this reads guest memory as it is.

use super::DrawError;
use super::shaders::read_words;
use crate::back::state::Mirror;
use crate::classes::clc797;
use crate::memory::GpuMemory;

/// Words in the report a conditional mode reads (Eden's `ReportSemaphore::Compare`).
pub const REPORT_WORDS: usize = 6;

/// Whether the next draw renders (module docs).
pub fn render_enabled<M: GpuMemory + ?Sized>(
    mirror: &Mirror,
    memory: &M,
) -> Result<bool, DrawError> {
    let override_mode = mirror
        .reg(&clc797::SET_RENDER_ENABLE_OVERRIDE, &[])?
        .named(clc797::SET_RENDER_ENABLE_OVERRIDE_MODE)?
        .value;
    match override_mode {
        clc797::SET_RENDER_ENABLE_OVERRIDE_MODE_ALWAYS_RENDER => return Ok(true),
        clc797::SET_RENDER_ENABLE_OVERRIDE_MODE_NEVER_RENDER => return Ok(false),
        _ => {}
    }
    let mode = mirror
        .reg(&clc797::SET_RENDER_ENABLE_C, &[])?
        .named(clc797::SET_RENDER_ENABLE_C_MODE)?
        .value;
    match mode {
        clc797::SET_RENDER_ENABLE_C_MODE_TRUE => return Ok(true),
        clc797::SET_RENDER_ENABLE_C_MODE_FALSE => return Ok(false),
        _ => {}
    }
    let upper = mirror
        .reg(&clc797::SET_RENDER_ENABLE_A, &[])?
        .field(clc797::SET_RENDER_ENABLE_A_OFFSET_UPPER)?;
    let lower = mirror
        .reg(&clc797::SET_RENDER_ENABLE_B, &[])?
        .field(clc797::SET_RENDER_ENABLE_B_OFFSET_LOWER)?;
    let address = (u64::from(upper) << u32::BITS) | u64::from(lower);
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

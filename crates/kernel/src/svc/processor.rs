//! `GetCurrentProcessorNumber` (0x10).
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_processor.cpp` @ cf654052
//! (GPL-3.0-or-later): the call returns the index of the core running the calling thread, as
//! `int32_t`, in `x0`.

/// Returns the calling core's index. The dispatcher knows the core (each core thread dispatches
/// its own calls) and passes it in.
#[must_use]
pub fn get_current_processor_number(core: u8) -> i32 {
    i32::from(core)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_the_calling_core() {
        for core in [0, 5, u8::MAX] {
            assert_eq!(get_current_processor_number(core), i32::from(core));
        }
    }
}

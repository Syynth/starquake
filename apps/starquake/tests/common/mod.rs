//! Shared by this crate's `starquake run` tests: the date every run pins.

use std::ffi::OsString;

use starquake_core::{RtcAtBoot, UnixSeconds};

/// `--date <seconds>` with the date the runs here pin, so that none depends on the host's
/// (`docs/specs/observability.md`, rule 1): Eden's, which a run's settings also pin when they do
/// not say ([`RtcAtBoot::eden_default`]).
pub fn pinned_date() -> [OsString; 2] {
    let RtcAtBoot::Pinned(UnixSeconds(seconds)) = RtcAtBoot::eden_default() else {
        unreachable!("Eden's default date is pinned");
    };
    ["--date".into(), seconds.to_string().into()]
}

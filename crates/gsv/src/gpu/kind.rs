//! Which page-table kind a 211, 212 or 216 request gets: PR's `SelectNativeMemoryKind`.
//!
//! Ported from Eden PR #4588,
//! `eden-pr4588/src/core/hle/service/nvdrv/devices/native_gpu_kind.h` @ e5fb3b97
//! (`TranslateNativeUncompressedKind`, `SelectNativeMemoryKind`), GPL-3.0-or-later, Copyright 2026
//! Eden Emulator Project. The kind values are Eden's `Tegra::PTEKind`
//! (`eden-pr4588/src/video_core/pte_kind.h` @ e5fb3b97).
//!
//! A request names two bytes, the kind it wants and a fallback. PR turns them into two bytes:
//! the backend kind it maps the pages with (a [`Kind`], what [`starquake_gpu::vm::GpuVm`]
//! stores), and the selected kind it sends back in the reply. NX2 keeps no kinds and echoes the
//! wanted one (`eden-nx2/src/core/hle/service/nx2_gsv.h` @ 50fa86bb, `SubModifyKind`); the
//! table keeps PR's (`starquake_gpu::vm`, "Conflicts between the references": kinds).

use starquake_gpu::interface::Kind;

/// `Tegra::PTEKind::PITCH`.
pub const PITCH: Kind = Kind(0x00);
/// `Tegra::PTEKind::Z16`.
pub const Z16: Kind = Kind(0x01);
/// `Tegra::PTEKind::S8Z24`.
pub const S8Z24: Kind = Kind(0x11);
/// `Tegra::PTEKind::ZF32_X24S8`.
pub const ZF32_X24S8: Kind = Kind(0xC3);
/// `Tegra::PTEKind::GENERIC_16BX2`: generic memory.
pub const GENERIC_16BX2: Kind = Kind(0xFE);

/// The two bytes PR selects for a request (PR `NativeMemoryKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selected {
    /// The kind the pages are mapped with.
    pub backend: Kind,
    /// The kind the reply names.
    pub selected: u8,
}

/// PR `TranslateNativeUncompressedKind`: the backend kind for an uncompressed request (`kind`
/// equal to `fallback`), for the kinds PR lists; `None` for any other pair.
fn uncompressed(kind: u8, fallback: u8) -> Option<Kind> {
    if kind != fallback {
        return None;
    }
    match kind {
        0 => Some(PITCH),
        1 => Some(Z16),
        3 => Some(S8Z24),
        4 => Some(ZF32_X24S8),
        // 6, and 0xf: PR's "CUDA scratch mapping ... requests kind 0xf with the same fallback".
        6 | 0xF => Some(GENERIC_16BX2),
        _ => None,
    }
}

/// PR `SelectNativeMemoryKind`: the kinds for a request that wants `requested` and offers
/// `fallback`, or `None` for a pair PR refuses.
#[must_use]
pub fn select(requested: u8, fallback: u8) -> Option<Selected> {
    if let Some(backend) = uncompressed(requested, fallback) {
        return Some(Selected {
            backend,
            selected: requested,
        });
    }
    let (backend, selected) = match (requested, fallback) {
        // Generic compression (8, or 9 with PLC disabled) with the generic fallback.
        (8 | 9, 6) => (GENERIC_16BX2, 6),
        (0xB, 1) => (Z16, 1),
        (0xC, 3) => (S8Z24, 3),
        (0xD, 4) => (ZF32_X24S8, 4),
        _ => return None,
    };
    Some(Selected { backend, selected })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pair PR accepts, laid out by hand from `native_gpu_kind.h`: (requested, fallback,
    /// backend, selected).
    const ACCEPTED: [(u8, u8, u8, u8); 11] = [
        (0, 0, 0x00, 0),
        (1, 1, 0x01, 1),
        (3, 3, 0x11, 3),
        (4, 4, 0xC3, 4),
        (6, 6, 0xFE, 6),
        (0xF, 0xF, 0xFE, 0xF),
        (8, 6, 0xFE, 6),
        (9, 6, 0xFE, 6),
        (0xB, 1, 0x01, 1),
        (0xC, 3, 0x11, 3),
        (0xD, 4, 0xC3, 4),
    ];

    #[test]
    fn every_pair_selects_as_pr_does() {
        for requested in 0..=u8::MAX {
            for fallback in 0..=u8::MAX {
                let expected = ACCEPTED
                    .iter()
                    .find(|row| row.0 == requested && row.1 == fallback)
                    .map(|row| Selected {
                        backend: Kind(row.2),
                        selected: row.3,
                    });
                assert_eq!(
                    select(requested, fallback),
                    expected,
                    "{requested:#x}/{fallback:#x}"
                );
            }
        }
    }
}

//! Load-time code patching: rewrites instructions a CPU backend cannot run efficiently.
//!
//! Under Hypervisor.framework every `mrs xN, cntpct_el0` exits to the host (~760 ns), and titles
//! issue them millions of times a second; the virtual counter does not exit but ticks at the
//! host's rate, not the console's (`tools/hvprobe/README.md`). This module finds the counter
//! reads in a program's text and replaces each with a `B` to a small trampoline that reads
//! `CNTVCT_EL0`, rescales it to the guest's rate and branches back.
//!
//! Recognised by mask and value (no general decoder): `MRS Xt, CNTPCT_EL0`,
//! `MRS Xt, CNTVCT_EL0` and `MRS Xt, CNTFRQ_EL0`. A site whose destination is `XZR` becomes a
//! `NOP`.
//!
//! # Trampolines
//!
//! Each trampoline preserves every register except `Xt`, and the flags. Scratch registers are
//! saved with a pre-indexed store of 16 bytes at `sp` and restored with a post-indexed load, so
//! `sp` stays 16-byte aligned. This writes just below the guest's `sp` for a moment, so it assumes
//! the guest ABI has no red zone below `sp` (the same assumption Eden's patcher makes). The
//! shapes, chosen by the integer and fractional parts of the rate ratio:
//!
//! | Ratio | Shape |
//! |---|---|
//! | `CNTFRQ_EL0` | `ldr xt, =guest_hz` |
//! | equal rates | `mrs xt, cntvct_el0` |
//! | below 1 | one scratch: `xt = umulh(xt, frac)` |
//! | between 1 and 2 | one scratch: `xt = xt + umulh(xt, frac)` |
//! | 2 or more | two scratches: `xt = xt * int + umulh(xt, frac)` (Eden's shape) |
//!
//! Scratch registers are `x0`/`x1`, or `x2`/`x3` when `Xt` is `x0` or `x1`.
//!
//! # Known limit
//!
//! Patching is by encoding alone, so a data word embedded in text that happens to equal one of
//! the recognised encodings is patched too. Eden's patcher has the same limit.
//!
//! Ported from Eden `eden/src/core/arm/nce/patcher.cpp` (`WriteCntpctHandler`,
//! `WriteCntfrqHandler`) @ cf654052 (GPL-3.0-or-later): the scratch choice, the `umulh`/`madd`
//! rescale and the literal-pool trampoline layout. The one-scratch shapes follow the measured stub
//! `_t_el0_patched` in `tools/hvprobe/src/guest.rs` (`t + umulh(t, frac)`).

use thiserror::Error;

/// The `NOP` instruction, which replaces reads into `XZR`.
const NOP: u32 = 0xD503_201F;

/// Mask selecting everything but the destination register of an `MRS Xt, <sysreg>`.
const MRS_MASK: u32 = 0xFFFF_FFE0;
/// `MRS Xt, CNTFRQ_EL0` with `Rt` = 0 (op0=3, op1=3, CRn=14, CRm=0, op2=0).
const MRS_CNTFRQ: u32 = 0xD53B_E000;
/// `MRS Xt, CNTPCT_EL0` with `Rt` = 0 (op2=1).
const MRS_CNTPCT: u32 = 0xD53B_E020;
/// `MRS Xt, CNTVCT_EL0` with `Rt` = 0 (op2=2).
const MRS_CNTVCT: u32 = 0xD53B_E040;

/// Register number 31, `XZR` as a destination.
const XZR: u32 = 31;

/// `B` reaches +-128 MiB: a signed 26-bit word offset.
const BRANCH_RANGE: i128 = 1 << 27;

/// Why a text could not be patched.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PatchError {
    /// The text is not a whole number of 4-byte instructions.
    #[error("text length {len} is not a multiple of 4")]
    UnalignedTextLength {
        /// The text length in bytes.
        len: usize,
    },
    /// The text's guest address is not 4-byte aligned.
    #[error("text address {address:#x} is not 4-byte aligned")]
    UnalignedTextAddress {
        /// The offending guest address.
        address: u64,
    },
    /// The trampoline area's guest address is not 8-byte aligned (its literals are 8 bytes).
    #[error("trampoline address {address:#x} is not 8-byte aligned")]
    UnalignedTrampolineAddress {
        /// The offending guest address.
        address: u64,
    },
    /// A counter rate is zero, so no rescale exists.
    #[error("counter rates must be non-zero (host {host_hz} Hz, guest {guest_hz} Hz)")]
    ZeroRate {
        /// The host counter rate in Hz.
        host_hz: u64,
        /// The guest counter rate in Hz.
        guest_hz: u64,
    },
    /// The text, or the trampolines for it, run past the end of the 64-bit address space.
    #[error("patching at {address:#x} overflows the 64-bit address space")]
    AddressOverflow {
        /// The address at which the arithmetic overflowed.
        address: u64,
    },
    /// A branch between text and a trampoline cannot reach (limit +-128 MiB).
    #[error("branch from {from:#x} to {to:#x} is outside the +-128 MiB range")]
    BranchOutOfRange {
        /// Address of the branch instruction.
        from: u64,
        /// Address it would have to reach.
        to: u64,
    },
}

/// The rate pair a counter read is rescaled between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CounterRescale {
    /// The host's virtual counter rate in Hz (24 MHz on Apple Silicon).
    pub host_hz: u64,
    /// The rate the guest expects in Hz (`CNTFRQ_EL0` as the guest reads it).
    pub guest_hz: u64,
}

impl CounterRescale {
    /// The fixed-point form of `guest_hz / host_hz`.
    pub fn ratio(&self) -> Result<FixedRatio, PatchError> {
        FixedRatio::new(self.host_hz, self.guest_hz)
    }
}

/// What to patch. The default patches nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PatchConfig {
    /// Rewrite counter reads (`CNTPCT_EL0`, `CNTVCT_EL0`, `CNTFRQ_EL0`) with this rescale, or
    /// leave them alone.
    pub counter: Option<CounterRescale>,
}

/// `guest_hz / host_hz` as an integer part and a 64-bit binary fraction, so
/// `guest = host * guest_hz / host_hz` becomes `host * int + umulh(host, frac)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedRatio {
    int: u64,
    frac: u64,
}

impl FixedRatio {
    /// Computes the ratio exactly (the fraction is truncated, never rounded).
    pub fn new(host_hz: u64, guest_hz: u64) -> Result<Self, PatchError> {
        if host_hz == 0 || guest_hz == 0 {
            return Err(PatchError::ZeroRate { host_hz, guest_hz });
        }
        let int = guest_hz / host_hz;
        let rem = u128::from(guest_hz % host_hz);
        // rem < host_hz, so the quotient is below 2^64 and the conversion cannot saturate.
        let frac = u64::try_from((rem << 64) / u128::from(host_hz)).unwrap_or(u64::MAX);
        Ok(Self { int, frac })
    }

    /// The integer part of the ratio.
    pub fn int(&self) -> u64 {
        self.int
    }

    /// The fractional part, in units of 2^-64.
    pub fn frac(&self) -> u64 {
        self.frac
    }

    /// Rescales host ticks to guest ticks, exactly as the trampolines do: wrapping in 64 bits,
    /// and at most one tick below `host * guest_hz / host_hz` (the fraction is truncated).
    pub fn scale(&self, host_ticks: u64) -> u64 {
        let high = (u128::from(host_ticks) * u128::from(self.frac)) >> 64;
        // `high` is below `host_ticks`, so it fits.
        let high = u64::try_from(high).unwrap_or(u64::MAX);
        host_ticks.wrapping_mul(self.int).wrapping_add(high)
    }
}

/// Which instruction a site was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteKind {
    /// `MRS Xt, CNTPCT_EL0`.
    CntPct,
    /// `MRS Xt, CNTVCT_EL0`.
    CntVct,
    /// `MRS Xt, CNTFRQ_EL0`.
    CntFrq,
}

/// One patched instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Site {
    /// Guest address of the patched instruction.
    pub address: u64,
    /// The instruction that was there.
    pub kind: SiteKind,
    /// Its destination register number (0-30, or 31 for `XZR`).
    pub register: u32,
    /// Guest address of the trampoline the site now branches to; `None` if the site became a
    /// `NOP` because it wrote to `XZR`.
    pub trampoline: Option<u64>,
}

/// The result of patching: code for the caller to map at `trampoline_address`, and a record of
/// every site rewritten.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Patched {
    /// Trampoline code and literals, little-endian, to be mapped at the `trampoline_address`
    /// passed to [`patch_text`].
    pub trampolines: Vec<u8>,
    /// Every rewritten site, in address order.
    pub sites: Vec<Site>,
}

/// Rewrites the counter reads in `text` in place.
///
/// `text` is little-endian A64 code whose first byte lives at guest address `text_address`; the
/// caller will map the returned `trampolines` at `trampoline_address`. Each recognised site is
/// replaced by `B <trampoline>`, or by `NOP` if it writes `XZR`. On error `text` is untouched.
/// With no counter rescale configured, or no matching instructions, `text` is left unchanged and
/// no trampolines are produced.
///
/// Both addresses must be aligned (4 bytes for text, 8 for trampolines) and every branch must
/// reach (+-128 MiB).
pub fn patch_text(
    text: &mut [u8],
    text_address: u64,
    trampoline_address: u64,
    config: &PatchConfig,
) -> Result<Patched, PatchError> {
    if !text.len().is_multiple_of(4) {
        return Err(PatchError::UnalignedTextLength { len: text.len() });
    }
    if !text_address.is_multiple_of(4) {
        return Err(PatchError::UnalignedTextAddress {
            address: text_address,
        });
    }
    if !trampoline_address.is_multiple_of(8) {
        return Err(PatchError::UnalignedTrampolineAddress {
            address: trampoline_address,
        });
    }
    let Some(rescale) = config.counter else {
        return Ok(Patched::default());
    };
    let ratio = rescale.ratio()?;
    let overflow = |address| PatchError::AddressOverflow { address };

    let mut patched = Patched::default();
    // Edits are applied only once every site has succeeded, so an error leaves `text` untouched.
    let mut edits: Vec<(usize, u32)> = Vec::new();
    for (index, chunk) in text.as_chunks::<4>().0.iter().enumerate() {
        let word = u32::from_le_bytes(*chunk);
        let Some((kind, register)) = classify(word) else {
            continue;
        };
        let address = u64::try_from(index)
            .ok()
            .and_then(|i| i.checked_mul(4))
            .and_then(|offset| text_address.checked_add(offset))
            .ok_or_else(|| overflow(text_address))?;
        if register == XZR {
            edits.push((index, NOP));
            patched.sites.push(Site {
                address,
                kind,
                register,
                trampoline: None,
            });
            continue;
        }
        let trampoline = u64::try_from(patched.trampolines.len())
            .ok()
            .and_then(|used| trampoline_address.checked_add(used))
            .ok_or_else(|| overflow(trampoline_address))?;
        let resume = address.checked_add(4).ok_or_else(|| overflow(address))?;
        let body = build_trampoline(kind, register, &rescale, &ratio, trampoline, resume)?;
        edits.push((index, branch(address, trampoline)?));
        patched.trampolines.extend_from_slice(&body);
        patched.sites.push(Site {
            address,
            kind,
            register,
            trampoline: Some(trampoline),
        });
    }
    let (words, _) = text.as_chunks_mut::<4>();
    for (index, word) in edits {
        if let Some(slot) = words.get_mut(index) {
            *slot = word.to_le_bytes();
        }
    }
    Ok(patched)
}

/// Recognises a counter read, returning its kind and destination register.
fn classify(word: u32) -> Option<(SiteKind, u32)> {
    let kind = match word & MRS_MASK {
        MRS_CNTPCT => SiteKind::CntPct,
        MRS_CNTVCT => SiteKind::CntVct,
        MRS_CNTFRQ => SiteKind::CntFrq,
        _ => return None,
    };
    Some((kind, word & 0x1F))
}

/// `B target`, as seen from `from`.
fn branch(from: u64, to: u64) -> Result<u32, PatchError> {
    let delta = i128::from(to) - i128::from(from);
    if !(-BRANCH_RANGE..BRANCH_RANGE).contains(&delta) || delta % 4 != 0 {
        return Err(PatchError::BranchOutOfRange { from, to });
    }
    // The range check bounds `delta >> 2` to 26 signed bits.
    let imm26 = u32::try_from((delta >> 2) & 0x03FF_FFFF)
        .map_err(|_| PatchError::BranchOutOfRange { from, to })?;
    Ok(0x1400_0000 | imm26)
}

/// One trampoline instruction before literal offsets are known.
enum Op {
    /// A finished instruction word.
    Word(u32),
    /// `ldr x<rt>, <literal n>`.
    LdrLit { rt: u32, literal: usize },
}

fn build_trampoline(
    kind: SiteKind,
    rt: u32,
    rescale: &CounterRescale,
    ratio: &FixedRatio,
    address: u64,
    resume: u64,
) -> Result<Vec<u8>, PatchError> {
    // Scratch registers: x0/x1, or x2/x3 when the destination is one of those.
    let (a, b) = if rt < 2 { (2, 3) } else { (0, 1) };
    let mrs = 0xD53B_E040 | rt;
    let (mut ops, literals): (Vec<Op>, Vec<u64>) = match kind {
        SiteKind::CntFrq => (vec![Op::LdrLit { rt, literal: 0 }], vec![rescale.guest_hz]),
        SiteKind::CntPct | SiteKind::CntVct => match (ratio.int, ratio.frac) {
            (1, 0) => (vec![Op::Word(mrs)], vec![]),
            (0, frac) => (
                vec![
                    Op::Word(str_pre(a)),
                    Op::Word(mrs),
                    Op::LdrLit { rt: a, literal: 0 },
                    Op::Word(umulh(rt, rt, a)),
                    Op::Word(ldr_post(a)),
                ],
                vec![frac],
            ),
            (1, frac) => (
                vec![
                    Op::Word(str_pre(a)),
                    Op::Word(mrs),
                    Op::LdrLit { rt: a, literal: 0 },
                    Op::Word(umulh(a, rt, a)),
                    Op::Word(add(rt, rt, a)),
                    Op::Word(ldr_post(a)),
                ],
                vec![frac],
            ),
            (int, frac) => (
                vec![
                    Op::Word(stp_pre(a, b)),
                    Op::Word(mrs),
                    Op::LdrLit { rt: a, literal: 0 },
                    Op::LdrLit { rt: b, literal: 1 },
                    Op::Word(umulh(a, rt, a)),
                    Op::Word(madd(rt, rt, b, a)),
                    Op::Word(ldp_post(a, b)),
                ],
                vec![frac, int],
            ),
        },
    };
    // The branch back to the instruction after the site closes the code.
    let branch_at = address
        .checked_add(u64::try_from(ops.len() * 4).unwrap_or(u64::MAX))
        .ok_or(PatchError::AddressOverflow { address })?;
    ops.push(Op::Word(branch(branch_at, resume)?));

    // Literals follow the code, 8-byte aligned (the code is padded with a NOP).
    let code_bytes = (ops.len() * 4).next_multiple_of(8);
    let mut out = Vec::with_capacity(code_bytes + literals.len() * 8);
    for (index, op) in ops.iter().enumerate() {
        let word = match *op {
            Op::Word(word) => word,
            Op::LdrLit { rt, literal } => {
                let distance = code_bytes + literal * 8 - index * 4;
                // Trampolines are a few dozen bytes, far inside the +-1 MiB reach.
                let imm19 = u32::try_from(distance / 4).unwrap_or(0);
                0x5800_0000 | (imm19 << 5) | rt
            }
        };
        out.extend_from_slice(&word.to_le_bytes());
    }
    while out.len() < code_bytes {
        out.extend_from_slice(&NOP.to_le_bytes());
    }
    for literal in literals {
        out.extend_from_slice(&literal.to_le_bytes());
    }
    Ok(out)
}

/// `str x<rt>, [sp, #-16]!`
const fn str_pre(rt: u32) -> u32 {
    0xF81F_0FE0 | rt
}

/// `ldr x<rt>, [sp], #16`
const fn ldr_post(rt: u32) -> u32 {
    0xF841_07E0 | rt
}

/// `stp x<a>, x<b>, [sp, #-16]!`
const fn stp_pre(a: u32, b: u32) -> u32 {
    0xA9BF_03E0 | (b << 10) | a
}

/// `ldp x<a>, x<b>, [sp], #16`
const fn ldp_post(a: u32, b: u32) -> u32 {
    0xA8C1_03E0 | (b << 10) | a
}

/// `umulh x<rd>, x<rn>, x<rm>`
const fn umulh(rd: u32, rn: u32, rm: u32) -> u32 {
    0x9BC0_7C00 | (rm << 16) | (rn << 5) | rd
}

/// `add x<rd>, x<rn>, x<rm>`
const fn add(rd: u32, rn: u32, rm: u32) -> u32 {
    0x8B00_0000 | (rm << 16) | (rn << 5) | rd
}

/// `madd x<rd>, x<rn>, x<rm>, x<ra>`
const fn madd(rd: u32, rn: u32, rm: u32, ra: u32) -> u32 {
    0x9B00_0000 | (rm << 16) | (ra << 10) | (rn << 5) | rd
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    // Rates from `tools/hvprobe/README.md`: Apple Silicon's counter and the console's.
    const HOST_HZ: u64 = 24_000_000;
    const CONSOLE_HZ: u64 = 31_250_000;
    // A rate below the host's, and ones at least twice it (the two-scratch shape).
    const SLOW_HZ: u64 = 19_200_000;
    const FAST_HZ: u64 = 100_000_000;

    const TEXT_ADDRESS: u64 = 0x1000;
    const TRAMPOLINE_ADDRESS: u64 = 0x2000;
    /// `b` from the site at `TEXT_ADDRESS + 8` to `TRAMPOLINE_ADDRESS`: `b .+0xff8`.
    const SITE_BRANCH: u32 = 0x1400_03FE;

    fn config(guest_hz: u64) -> PatchConfig {
        PatchConfig {
            counter: Some(CounterRescale {
                host_hz: HOST_HZ,
                guest_hz,
            }),
        }
    }

    fn mrs(base: u32, rt: u32) -> u32 {
        base | rt
    }

    fn text_of(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn words_of(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    }

    /// `frac` computed independently of the shipping code, with `u128` math.
    fn frac_of(host: u64, guest: u64) -> u64 {
        u64::try_from((u128::from(guest % host) << 64) / u128::from(host)).unwrap()
    }

    /// Patches `[nop, nop, <site>, nop]` and checks the text and the trampoline bytes.
    fn golden(site: u32, guest_hz: u64, code: &[u32], literals: &[u64]) {
        let mut text = text_of(&[NOP, NOP, site, NOP]);
        let patched = patch_text(
            &mut text,
            TEXT_ADDRESS,
            TRAMPOLINE_ADDRESS,
            &config(guest_hz),
        )
        .unwrap();
        assert_eq!(words_of(&text), [NOP, NOP, SITE_BRANCH, NOP]);
        let mut expected = text_of(code);
        for literal in literals {
            expected.extend_from_slice(&literal.to_le_bytes());
        }
        assert_eq!(patched.trampolines, expected);
        let kind = match site & MRS_MASK {
            MRS_CNTFRQ => SiteKind::CntFrq,
            MRS_CNTPCT => SiteKind::CntPct,
            _ => SiteKind::CntVct,
        };
        assert_eq!(
            patched.sites,
            [Site {
                address: TEXT_ADDRESS + 8,
                kind,
                register: site & 0x1F,
                trampoline: Some(TRAMPOLINE_ADDRESS),
            }]
        );
    }

    // The expected words below were assembled with `clang -target arm64-apple-macos -c` and read
    // back with `otool -t -X`; the assembly is in the comment above each. The site is at 0x1008
    // and the trampoline at 0x2000, so the return branch goes from 0x2000 + offset to 0x100c.

    #[test]
    fn golden_rate_above_host_one_scratch() {
        // str   x0, [sp, #-16]!
        // mrs   x5, cntvct_el0
        // ldr   x0, .+24          ; frac
        // umulh x0, x5, x0
        // add   x5, x5, x0
        // ldr   x0, [sp], #16
        // b     .-0x100c          ; back to 0x100c
        // nop                     ; pad so the literal is 8-byte aligned
        let code = [
            0xF81F_0FE0,
            0xD53B_E045,
            0x5800_00C0,
            0x9BC0_7CA0,
            0x8B00_00A5,
            0xF841_07E0,
            0x17FF_FBFD,
            0xD503_201F,
        ];
        let literals = [frac_of(HOST_HZ, CONSOLE_HZ)];
        golden(mrs(MRS_CNTVCT, 5), CONSOLE_HZ, &code, &literals);
        // The physical counter is read through the virtual one, so the trampoline is the same.
        golden(mrs(MRS_CNTPCT, 5), CONSOLE_HZ, &code, &literals);
    }

    #[test]
    fn golden_destination_x0_moves_the_scratch_to_x2() {
        // str   x2, [sp, #-16]!
        // mrs   x0, cntvct_el0
        // ldr   x2, .+24
        // umulh x2, x0, x2
        // add   x0, x0, x2
        // ldr   x2, [sp], #16
        // b     .-0x100c
        // nop
        golden(
            mrs(MRS_CNTVCT, 0),
            CONSOLE_HZ,
            &[
                0xF81F_0FE2,
                0xD53B_E040,
                0x5800_00C2,
                0x9BC2_7C02,
                0x8B02_0000,
                0xF841_07E2,
                0x17FF_FBFD,
                0xD503_201F,
            ],
            &[frac_of(HOST_HZ, CONSOLE_HZ)],
        );
    }

    #[test]
    fn golden_rate_below_host() {
        // str   x0, [sp, #-16]!
        // mrs   x5, cntvct_el0
        // ldr   x0, .+16          ; frac
        // umulh x5, x5, x0
        // ldr   x0, [sp], #16
        // b     .-0x1008
        golden(
            mrs(MRS_CNTVCT, 5),
            SLOW_HZ,
            &[
                0xF81F_0FE0,
                0xD53B_E045,
                0x5800_0080,
                0x9BC0_7CA5,
                0xF841_07E0,
                0x17FF_FBFE,
            ],
            &[frac_of(HOST_HZ, SLOW_HZ)],
        );
    }

    #[test]
    fn golden_rate_at_least_double_two_scratches() {
        // stp   x0, x1, [sp, #-16]!
        // mrs   x5, cntvct_el0
        // ldr   x0, .+24          ; frac
        // ldr   x1, .+28          ; int
        // umulh x0, x5, x0
        // madd  x5, x5, x1, x0
        // ldp   x0, x1, [sp], #16
        // b     .-0x1010
        golden(
            mrs(MRS_CNTVCT, 5),
            FAST_HZ,
            &[
                0xA9BF_07E0,
                0xD53B_E045,
                0x5800_00C0,
                0x5800_00E1,
                0x9BC0_7CA0,
                0x9B01_00A5,
                0xA8C1_07E0,
                0x17FF_FBFC,
            ],
            &[frac_of(HOST_HZ, FAST_HZ), FAST_HZ / HOST_HZ],
        );
    }

    #[test]
    fn golden_destination_x1_moves_both_scratches() {
        // stp   x2, x3, [sp, #-16]!
        // mrs   x1, cntvct_el0
        // ldr   x2, .+24
        // ldr   x3, .+28
        // umulh x2, x1, x2
        // madd  x1, x1, x3, x2
        // ldp   x2, x3, [sp], #16
        // b     .-0x1010
        golden(
            mrs(MRS_CNTVCT, 1),
            FAST_HZ,
            &[
                0xA9BF_0FE2,
                0xD53B_E041,
                0x5800_00C2,
                0x5800_00E3,
                0x9BC2_7C22,
                0x9B03_0821,
                0xA8C1_0FE2,
                0x17FF_FBFC,
            ],
            &[frac_of(HOST_HZ, FAST_HZ), FAST_HZ / HOST_HZ],
        );
    }

    #[test]
    fn golden_equal_rates_read_the_counter_directly() {
        // mrs   x5, cntvct_el0
        // b     .-0xff8
        golden(
            mrs(MRS_CNTVCT, 5),
            HOST_HZ,
            &[0xD53B_E045, 0x17FF_FC02],
            &[],
        );
    }

    #[test]
    fn golden_frequency_read_loads_the_guest_rate() {
        // ldr   x5, .+8           ; guest_hz
        // b     .-0xff8
        golden(
            mrs(MRS_CNTFRQ, 5),
            CONSOLE_HZ,
            &[0x5800_0045, 0x17FF_FC02],
            &[CONSOLE_HZ],
        );
    }

    #[test]
    fn recognised_encodings_match_the_assembler() {
        // mrs x5, cntfrq_el0 / cntpct_el0 / cntvct_el0, and nop
        assert_eq!(mrs(MRS_CNTFRQ, 5), 0xD53B_E005);
        assert_eq!(mrs(MRS_CNTPCT, 5), 0xD53B_E025);
        assert_eq!(mrs(MRS_CNTVCT, 5), 0xD53B_E045);
        assert_eq!(NOP, 0xD503_201F);
    }

    // ---- a tiny A64 interpreter for exactly the instructions the trampolines use ----

    struct Machine {
        x: [u64; 31],
        sp: u64,
        pc: u64,
        counter: u64,
        regions: Vec<(u64, Vec<u8>)>,
        stack: BTreeMap<u64, u8>,
        written: Vec<u64>,
    }

    impl Machine {
        fn code_bytes(&self, address: u64, len: u64) -> &[u8] {
            for (base, data) in &self.regions {
                let end = base + u64::try_from(data.len()).unwrap();
                if address >= *base && address + len <= end {
                    let start = usize::try_from(address - base).unwrap();
                    return &data[start..start + usize::try_from(len).unwrap()];
                }
            }
            unreachable!("fetch outside mapped code at {address:#x}");
        }

        fn code_word(&self, address: u64) -> u32 {
            u32::from_le_bytes(self.code_bytes(address, 4).try_into().unwrap())
        }

        fn reg(&self, r: u32) -> u64 {
            if r == 31 {
                0
            } else {
                self.x[usize::try_from(r).unwrap()]
            }
        }

        fn set(&mut self, r: u32, value: u64) {
            if r < 31 {
                self.x[usize::try_from(r).unwrap()] = value;
            }
        }

        fn store(&mut self, address: u64, value: u64) {
            for (i, byte) in value.to_le_bytes().into_iter().enumerate() {
                let a = address + u64::try_from(i).unwrap();
                self.stack.insert(a, byte);
                self.written.push(a);
            }
        }

        fn load(&self, address: u64) -> u64 {
            let mut bytes = [0u8; 8];
            for (i, byte) in bytes.iter_mut().enumerate() {
                *byte = self.stack[&(address + u64::try_from(i).unwrap())];
            }
            u64::from_le_bytes(bytes)
        }

        fn step(&mut self) {
            let w = self.code_word(self.pc);
            let rt = w & 31;
            let rn = (w >> 5) & 31;
            let rm = (w >> 16) & 31;
            let mut next = self.pc + 4;
            if w == NOP {
                // nothing
            } else if w & 0xFC00_0000 == 0x1400_0000 {
                // b
                let imm = i64::from(w & 0x03FF_FFFF);
                let imm = if imm & 0x0200_0000 != 0 {
                    imm - 0x0400_0000
                } else {
                    imm
                };
                next = self.pc.wrapping_add_signed(imm * 4);
            } else if w & MRS_MASK == MRS_CNTVCT {
                self.set(rt, self.counter);
            } else if w & 0xFF00_0000 == 0x5800_0000 {
                // ldr xt, <literal>
                let imm = i64::from((w >> 5) & 0x7_FFFF);
                let imm = if imm & 0x4_0000 != 0 {
                    imm - 0x8_0000
                } else {
                    imm
                };
                let at = self.pc.wrapping_add_signed(imm * 4);
                let value = u64::from_le_bytes(self.code_bytes(at, 8).try_into().unwrap());
                self.set(rt, value);
            } else if w & 0xFFE0_0C00 == 0xF800_0C00 {
                // str xt, [sp, #imm]!
                assert_eq!(rn, 31);
                let imm = i64::from((w >> 12) & 0x1FF);
                let imm = if imm & 0x100 != 0 { imm - 0x200 } else { imm };
                self.sp = self.sp.wrapping_add_signed(imm);
                self.store(self.sp, self.reg(rt));
            } else if w & 0xFFE0_0C00 == 0xF840_0400 {
                // ldr xt, [sp], #imm
                assert_eq!(rn, 31);
                let imm = i64::from((w >> 12) & 0x1FF);
                let value = self.load(self.sp);
                self.set(rt, value);
                self.sp = self.sp.wrapping_add_signed(imm);
            } else if w & 0xFFC0_0000 == 0xA980_0000 {
                // stp xt, xt2, [sp, #imm]!
                assert_eq!(rn, 31);
                let rt2 = (w >> 10) & 31;
                let imm = i64::from((w >> 15) & 0x7F);
                let imm = if imm & 0x40 != 0 { imm - 0x80 } else { imm } * 8;
                self.sp = self.sp.wrapping_add_signed(imm);
                self.store(self.sp, self.reg(rt));
                self.store(self.sp + 8, self.reg(rt2));
            } else if w & 0xFFC0_0000 == 0xA8C0_0000 {
                // ldp xt, xt2, [sp], #imm
                assert_eq!(rn, 31);
                let rt2 = (w >> 10) & 31;
                let imm = i64::from((w >> 15) & 0x7F) * 8;
                let (lo, hi) = (self.load(self.sp), self.load(self.sp + 8));
                self.set(rt, lo);
                self.set(rt2, hi);
                self.sp = self.sp.wrapping_add_signed(imm);
            } else if w & 0xFFE0_FC00 == 0x9BC0_7C00 {
                // umulh
                let wide = u128::from(self.reg(rn)) * u128::from(self.reg(rm));
                self.set(rt, u64::try_from(wide >> 64).unwrap());
            } else if w & 0xFFE0_FC00 == 0x8B00_0000 {
                // add (shifted register, no shift)
                self.set(rt, self.reg(rn).wrapping_add(self.reg(rm)));
            } else if w & 0xFFE0_8000 == 0x9B00_0000 {
                // madd
                let ra = (w >> 10) & 31;
                let value = self
                    .reg(rn)
                    .wrapping_mul(self.reg(rm))
                    .wrapping_add(self.reg(ra));
                self.set(rt, value);
            } else {
                unreachable!("interpreter does not know {w:#010x} at {:#x}", self.pc);
            }
            self.pc = next;
        }
    }

    /// Patches a one-site text, runs it from the site until it falls through to the next
    /// instruction, and returns the machine before and after.
    fn run_site(site: u32, guest_hz: u64, counter: u64) -> (Machine, Machine) {
        let mut text = text_of(&[NOP, site, NOP]);
        let patched = patch_text(
            &mut text,
            TEXT_ADDRESS,
            TRAMPOLINE_ADDRESS,
            &config(guest_hz),
        )
        .unwrap();
        let mut x = [0u64; 31];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = 0xA5A5_0000_0000_0000 | (u64::try_from(i).unwrap() * 0x0101_0101);
        }
        let regions = vec![
            (TEXT_ADDRESS, text),
            (TRAMPOLINE_ADDRESS, patched.trampolines),
        ];
        let machine = || Machine {
            x,
            sp: 0x7FFF_0000,
            pc: TEXT_ADDRESS + 4,
            counter,
            regions: regions.clone(),
            stack: BTreeMap::new(),
            written: Vec::new(),
        };
        let before = machine();
        let mut after = machine();
        for _ in 0..64 {
            if after.pc == TEXT_ADDRESS + 8 {
                return (before, after);
            }
            after.step();
        }
        unreachable!("trampoline did not return to the instruction after the site");
    }

    fn check_site(site: u32, guest_hz: u64, counter: u64, expected: u64) {
        let rt = site & 0x1F;
        let (before, after) = run_site(site, guest_hz, counter);
        assert_eq!(after.pc, TEXT_ADDRESS + 8, "resumes after the site");
        assert_eq!(after.sp, before.sp, "sp restored (site x{rt})");
        assert_eq!(before.sp % 16, 0);
        assert!(
            after
                .written
                .iter()
                .all(|a| (before.sp - 16..before.sp).contains(a)),
            "stack writes stay in the 16 bytes below sp (site x{rt})"
        );
        for r in 0..31u32 {
            if r != rt {
                assert_eq!(
                    after.reg(r),
                    before.reg(r),
                    "x{r} preserved (site x{rt}, {HOST_HZ} -> {guest_hz})"
                );
            }
        }
        assert_eq!(
            after.reg(rt),
            expected,
            "x{rt} result ({HOST_HZ} -> {guest_hz})"
        );
    }

    #[test]
    fn counter_sites_rescale_and_preserve_everything_else() {
        let counters = [0, 1, 0x1234_5678_9ABC, u64::MAX / 8];
        for guest_hz in [HOST_HZ, CONSOLE_HZ, SLOW_HZ, FAST_HZ, HOST_HZ * 2] {
            let ratio = FixedRatio::new(HOST_HZ, guest_hz).unwrap();
            for rt in 0..=30 {
                for base in [MRS_CNTVCT, MRS_CNTPCT] {
                    for counter in counters {
                        check_site(mrs(base, rt), guest_hz, counter, ratio.scale(counter));
                    }
                }
            }
        }
    }

    #[test]
    fn frequency_sites_load_the_guest_rate() {
        for guest_hz in [HOST_HZ, CONSOLE_HZ, FAST_HZ] {
            for rt in 0..=30 {
                check_site(mrs(MRS_CNTFRQ, rt), guest_hz, 99, guest_hz);
            }
        }
    }

    #[test]
    fn scaled_counter_matches_the_exact_value() {
        let ratio = FixedRatio::new(HOST_HZ, CONSOLE_HZ).unwrap();
        assert_eq!(ratio.int(), 1);
        assert_eq!(ratio.frac(), frac_of(HOST_HZ, CONSOLE_HZ));
        // One second of host ticks is one second of guest ticks, to within the truncated tick.
        assert!([CONSOLE_HZ - 1, CONSOLE_HZ].contains(&ratio.scale(HOST_HZ)));
    }

    // ---- XZR, no-ops, errors ----

    #[test]
    fn writes_to_xzr_become_nops() {
        for base in [MRS_CNTPCT, MRS_CNTVCT, MRS_CNTFRQ] {
            let mut text = text_of(&[NOP, mrs(base, 31), NOP]);
            let patched = patch_text(
                &mut text,
                TEXT_ADDRESS,
                TRAMPOLINE_ADDRESS,
                &config(CONSOLE_HZ),
            )
            .unwrap();
            assert_eq!(words_of(&text), [NOP, NOP, NOP]);
            assert_eq!(patched.trampolines, Vec::<u8>::new());
            assert_eq!(patched.sites.len(), 1);
            assert_eq!(patched.sites[0].register, 31);
            assert_eq!(patched.sites[0].trampoline, None);
            assert_eq!(patched.sites[0].address, TEXT_ADDRESS + 4);
        }
    }

    #[test]
    fn text_without_matching_instructions_is_unchanged() {
        let words = [
            NOP,
            0xD53B_D040, // mrs x0, tpidr_el0
            0xD53B_E200, // mrs x0, cntp_tval_el0
            0xD53B_E320, // mrs x0, cntv_ctl_el0
            0xD51B_E040, // the write form (L = 0) of the cntvct encoding
            0xD400_0001, // svc #0
            0,
            u32::MAX,
        ];
        let mut text = text_of(&words);
        let original = text.clone();
        let patched = patch_text(
            &mut text,
            TEXT_ADDRESS,
            TRAMPOLINE_ADDRESS,
            &config(CONSOLE_HZ),
        )
        .unwrap();
        assert_eq!(text, original);
        assert_eq!(patched, Patched::default());
    }

    #[test]
    fn default_config_patches_nothing() {
        let mut text = text_of(&[mrs(MRS_CNTPCT, 3), mrs(MRS_CNTFRQ, 4)]);
        let original = text.clone();
        let patched = patch_text(
            &mut text,
            TEXT_ADDRESS,
            TRAMPOLINE_ADDRESS,
            &PatchConfig::default(),
        )
        .unwrap();
        assert_eq!(text, original);
        assert_eq!(patched, Patched::default());
    }

    #[test]
    fn empty_text_is_fine() {
        let patched = patch_text(
            &mut [],
            TEXT_ADDRESS,
            TRAMPOLINE_ADDRESS,
            &config(CONSOLE_HZ),
        )
        .unwrap();
        assert_eq!(patched, Patched::default());
    }

    #[test]
    fn sites_are_reported_in_address_order_with_aligned_trampolines() {
        let words = [
            mrs(MRS_CNTVCT, 1),
            NOP,
            mrs(MRS_CNTFRQ, 2),
            mrs(MRS_CNTPCT, 3),
            mrs(MRS_CNTVCT, 31),
        ];
        for guest_hz in [HOST_HZ, CONSOLE_HZ, SLOW_HZ, FAST_HZ] {
            let mut text = text_of(&words);
            let patched = patch_text(
                &mut text,
                TEXT_ADDRESS,
                TRAMPOLINE_ADDRESS,
                &config(guest_hz),
            )
            .unwrap();
            let summary: Vec<_> = patched
                .sites
                .iter()
                .map(|s| (s.address, s.kind, s.register))
                .collect();
            assert_eq!(
                summary,
                [
                    (TEXT_ADDRESS, SiteKind::CntVct, 1),
                    (TEXT_ADDRESS + 8, SiteKind::CntFrq, 2),
                    (TEXT_ADDRESS + 12, SiteKind::CntPct, 3),
                    (TEXT_ADDRESS + 16, SiteKind::CntVct, 31),
                ]
            );
            let trampolines: Vec<u64> = patched.sites.iter().filter_map(|s| s.trampoline).collect();
            assert_eq!(trampolines.len(), 3);
            assert_eq!(trampolines[0], TRAMPOLINE_ADDRESS);
            let end = TRAMPOLINE_ADDRESS + u64::try_from(patched.trampolines.len()).unwrap();
            for pair in trampolines.windows(2) {
                assert!(pair[0] < pair[1]);
            }
            for t in &trampolines {
                assert_eq!(t % 8, 0, "trampolines start 8-byte aligned");
                assert!(*t < end);
            }
            assert_eq!(end % 8, 0);
            // Every patched site is now a branch to its trampoline.
            let patched_words = words_of(&text);
            assert_eq!(patched_words[1], NOP);
            assert_eq!(patched_words[4], NOP);
            for (site, t) in [
                (0usize, trampolines[0]),
                (2, trampolines[1]),
                (3, trampolines[2]),
            ] {
                let from = TEXT_ADDRESS + u64::try_from(site).unwrap() * 4;
                assert_eq!(patched_words[site], branch(from, t).unwrap());
            }
        }
    }

    #[test]
    fn unaligned_or_bad_input_is_an_error() {
        let cfg = config(CONSOLE_HZ);
        let site = mrs(MRS_CNTVCT, 5);
        let mut text = text_of(&[site]);
        text.push(0);
        let original = text.clone();
        assert_eq!(
            patch_text(&mut text, TEXT_ADDRESS, TRAMPOLINE_ADDRESS, &cfg),
            Err(PatchError::UnalignedTextLength { len: 5 })
        );
        assert_eq!(text, original);

        let mut text = text_of(&[site]);
        assert_eq!(
            patch_text(&mut text, TEXT_ADDRESS + 2, TRAMPOLINE_ADDRESS, &cfg),
            Err(PatchError::UnalignedTextAddress {
                address: TEXT_ADDRESS + 2
            })
        );
        assert_eq!(
            patch_text(&mut text, TEXT_ADDRESS, TRAMPOLINE_ADDRESS + 4, &cfg),
            Err(PatchError::UnalignedTrampolineAddress {
                address: TRAMPOLINE_ADDRESS + 4
            })
        );
        for (host_hz, guest_hz) in [(0, CONSOLE_HZ), (HOST_HZ, 0), (0, 0)] {
            let zero = PatchConfig {
                counter: Some(CounterRescale { host_hz, guest_hz }),
            };
            assert_eq!(
                patch_text(&mut text, TEXT_ADDRESS, TRAMPOLINE_ADDRESS, &zero),
                Err(PatchError::ZeroRate { host_hz, guest_hz })
            );
        }
        assert_eq!(words_of(&text), [site]);
    }

    #[test]
    fn address_space_overflow_is_an_error() {
        let site = mrs(MRS_CNTVCT, 5);
        let mut text = text_of(&[site]);
        // The instruction after the site would be at 2^64.
        let result = patch_text(
            &mut text,
            u64::MAX - 3,
            TRAMPOLINE_ADDRESS,
            &config(CONSOLE_HZ),
        );
        assert!(matches!(result, Err(PatchError::AddressOverflow { .. })));
        assert_eq!(words_of(&text), [site]);
    }

    /// Where the range tests put their site: far from zero so both directions fit.
    const FAR_SITE: u64 = 0x1000_0000;

    fn range_result(trampoline_address: u64) -> Result<Patched, PatchError> {
        // A frequency read is the shortest trampoline: `ldr` at +0, `b` back at +4.
        let mut text = text_of(&[mrs(MRS_CNTFRQ, 5)]);
        patch_text(&mut text, FAR_SITE, trampoline_address, &config(CONSOLE_HZ))
    }

    #[test]
    fn branch_range_is_plus_or_minus_128_mib() {
        let reach = u64::try_from(BRANCH_RANGE).unwrap();
        // Forward: the last reachable trampoline, then one past it.
        assert!(range_result(FAR_SITE + reach - 8).is_ok());
        assert!(matches!(
            range_result(FAR_SITE + reach),
            Err(PatchError::BranchOutOfRange { .. })
        ));
        // Backward: the branch out may reach -128 MiB, but the branch back (from +4) then needs
        // +128 MiB, which it cannot.
        assert!(range_result(FAR_SITE - reach + 8).is_ok());
        assert_eq!(
            range_result(FAR_SITE - reach),
            Err(PatchError::BranchOutOfRange {
                from: FAR_SITE - reach + 4,
                to: FAR_SITE + 4
            })
        );
        assert!(matches!(
            range_result(FAR_SITE - reach - 8),
            Err(PatchError::BranchOutOfRange { .. })
        ));
    }

    #[test]
    fn a_failing_site_leaves_the_text_untouched() {
        // Site 0 just reaches its trampoline; site 1, whose trampoline is further on, does not.
        let reach = u64::try_from(BRANCH_RANGE).unwrap();
        let mut text = text_of(&[mrs(MRS_CNTFRQ, 5), mrs(MRS_CNTFRQ, 6)]);
        let original = text.clone();
        let result = patch_text(
            &mut text,
            FAR_SITE,
            FAR_SITE + reach - 8,
            &config(CONSOLE_HZ),
        );
        assert!(matches!(result, Err(PatchError::BranchOutOfRange { .. })));
        assert_eq!(text, original);
    }

    // ---- properties ----

    /// Maps arbitrary bits to a tick count whose exact rescale still fits in 64 bits.
    fn clamp_ticks(raw: u64, host_hz: u64, guest_hz: u64) -> u64 {
        let max = u128::from(u64::MAX) * u128::from(host_hz) / u128::from(guest_hz);
        match u64::try_from(max) {
            Ok(max) if max < u64::MAX => raw % (max + 1),
            _ => raw,
        }
    }

    fn within_one_tick(host_hz: u64, guest_hz: u64, ticks: u64) {
        let ratio = FixedRatio::new(host_hz, guest_hz).unwrap();
        let exact = u128::from(ticks) * u128::from(guest_hz) / u128::from(host_hz);
        let got = u128::from(ratio.scale(ticks));
        assert!(
            got <= exact && exact - got <= 1,
            "{ticks} ticks, {host_hz} -> {guest_hz}: got {got}, exact {exact}"
        );
    }

    proptest! {
        #[test]
        fn rescale_matches_u128_for_the_console_rate(raw in any::<u64>()) {
            within_one_tick(HOST_HZ, CONSOLE_HZ, clamp_ticks(raw, HOST_HZ, CONSOLE_HZ));
        }

        #[test]
        fn rescale_matches_u128_for_a_rate_below_the_host(raw in any::<u64>()) {
            within_one_tick(HOST_HZ, SLOW_HZ, clamp_ticks(raw, HOST_HZ, SLOW_HZ));
        }

        #[test]
        fn rescale_matches_u128_for_arbitrary_rates(
            host_hz in 1u64..=10_000_000_000,
            guest_hz in 1u64..=10_000_000_000,
            raw in any::<u64>(),
        ) {
            within_one_tick(host_hz, guest_hz, clamp_ticks(raw, host_hz, guest_hz));
        }

        #[test]
        fn patching_arbitrary_text_never_panics(
            words in proptest::collection::vec(
                prop_oneof![
                    any::<u32>(),
                    (0u32..32).prop_map(|rt| mrs(MRS_CNTPCT, rt)),
                    (0u32..32).prop_map(|rt| mrs(MRS_CNTVCT, rt)),
                    (0u32..32).prop_map(|rt| mrs(MRS_CNTFRQ, rt)),
                ],
                0..64,
            ),
            text_address in any::<u64>(),
            trampoline_address in any::<u64>(),
            host_hz in any::<u64>(),
            guest_hz in any::<u64>(),
        ) {
            let mut text = text_of(&words);
            let original = text.clone();
            let cfg = PatchConfig { counter: Some(CounterRescale { host_hz, guest_hz }) };
            if patch_text(&mut text, text_address, trampoline_address, &cfg).is_err() {
                prop_assert_eq!(text, original);
            }
        }
    }
}

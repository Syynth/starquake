//! Tests for acquire predicates, release forms and reductions.
//!
//! Hand-laid words are written out from the class headers' bit ranges, not built from this
//! module's constants, so a wrong constant cannot agree with itself.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use super::*;

fn config() -> Config {
    Config {
        cases: 2048,
        rng_seed: RngSeed::Fixed(0x7365_6d61),
        failure_persistence: None,
        ..Config::default()
    }
}

const ALL_OPS: [ReductionOp; 8] = [
    ReductionOp::Add,
    ReductionOp::Min,
    ReductionOp::Max,
    ReductionOp::Inc,
    ReductionOp::Dec,
    ReductionOp::And,
    ReductionOp::Or,
    ReductionOp::Xor,
];

const ALL_FORMATS: [ReductionFormat; 2] = [ReductionFormat::Unsigned, ReductionFormat::Signed];

fn red(op: ReductionOp) -> Reduction {
    Reduction {
        op,
        format: ReductionFormat::Unsigned,
    }
}

fn signed(op: ReductionOp) -> Reduction {
    Reduction {
        op,
        format: ReductionFormat::Signed,
    }
}

#[test]
fn every_field_constant_is_a_non_empty_field() {
    for field in [
        SEMAPHORED_OPERATION,
        SEMAPHORED_ACQUIRE_SWITCH,
        SEMAPHORED_RELEASE_WFI,
        SEMAPHORED_RELEASE_SIZE,
        SEMAPHORED_REDUCTION,
        SEMAPHORED_FORMAT,
        SEM_EXECUTE_OPERATION,
        SEM_EXECUTE_ACQUIRE_SWITCH_TSG,
        SEM_EXECUTE_RELEASE_WFI,
        SEM_EXECUTE_PAYLOAD_SIZE,
        SEM_EXECUTE_RELEASE_TIMESTAMP,
        SEM_EXECUTE_REDUCTION,
        SEM_EXECUTE_REDUCTION_FORMAT,
        MME_DMA_REDUCTION_OP,
        MME_DMA_REDUCTION_FORMAT,
        MME_DMA_REDUCTION_SIZE,
        QMD_RELEASE_REDUCTION_OP,
        QMD_RELEASE_REDUCTION_FORMAT,
        QMD_RELEASE_REDUCTION_ENABLE,
    ] {
        assert_ne!(field.width(), 0, "{field:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Acquire predicates: a table per operation, boundary values on both sides
// ---------------------------------------------------------------------------------------------

/// (value, payload, met)
fn check(op: AcquireOp, table: &[(u32, u32, bool)]) {
    for &(value, payload, met) in table {
        assert_eq!(
            op.is_met(value, payload),
            met,
            "{op:?} value {value:#x} payload {payload:#x}"
        );
    }
}

#[test]
fn acquire_equal() {
    check(
        AcquireOp::Equal,
        &[
            (5, 5, true),
            (4, 5, false),
            (6, 5, false),
            (0, 0, true),
            (u32::MAX, u32::MAX, true),
            (0, u32::MAX, false),
            (u32::MAX, 0, false),
        ],
    );
}

#[test]
fn acquire_greater_or_equal_is_unsigned_and_does_not_wrap() {
    check(
        AcquireOp::GreaterOrEqual,
        &[
            (4, 5, false),
            (5, 5, true),
            (6, 5, true),
            (0, 0, true),
            (u32::MAX - 1, u32::MAX, false),
            (u32::MAX, u32::MAX, true),
            // Past the wrap the value is small again: not met.
            (0, u32::MAX, false),
            (1, u32::MAX, false),
            // Unsigned: 0x8000_0000 is above 0x7fff_ffff.
            (0x8000_0000, 0x7fff_ffff, true),
        ],
    );
}

#[test]
fn acquire_circular_greater_or_equal_wraps() {
    let half = i32::MAX.cast_unsigned();
    check(
        AcquireOp::CircularGreaterOrEqual,
        &[
            (4, 5, false),
            (5, 5, true),
            (6, 5, true),
            // Across the wrap: 0 and 1 are just past u32::MAX.
            (0, u32::MAX, true),
            (1, u32::MAX, true),
            (u32::MAX, 0, false),
            // The boundary: up to i32::MAX ahead is met, one more is behind.
            (5u32.wrapping_add(half), 5, true),
            (5u32.wrapping_add(half).wrapping_add(1), 5, false),
            (half, 0, true),
            (half + 1, 0, false),
        ],
    );
}

#[test]
fn acquire_and() {
    check(
        AcquireOp::And,
        &[
            (0b1010, 0b0101, false),
            (0b1010, 0b0010, true),
            (0, u32::MAX, false),
            (u32::MAX, 0, false),
            (0x8000_0000, 0x8000_0000, true),
            (1, 1, true),
        ],
    );
}

#[test]
fn acquire_nor() {
    check(
        AcquireOp::Nor,
        &[
            (0b1010, 0b0101, true),
            (0b1010, 0b0010, false),
            (0, u32::MAX, true),
            (u32::MAX, 0, true),
            (0x8000_0000, 0x8000_0000, false),
            (1, 1, false),
        ],
    );
}

// ---------------------------------------------------------------------------------------------
// Reductions: a table per operation
// ---------------------------------------------------------------------------------------------

/// (reduction, old, payload, new)
fn reduce32(table: &[(Reduction, u32, u32, u32)]) {
    for &(reduction, old, payload, new) in table {
        assert_eq!(
            reduction.apply_u32(old, payload),
            new,
            "{reduction:?} old {old:#x} payload {payload:#x}"
        );
    }
}

#[test]
fn reduce_add_wraps() {
    let add = red(ReductionOp::Add);
    reduce32(&[
        (add, 1, 2, 3),
        (add, u32::MAX, 1, 0),
        (add, u32::MAX - 1, 1, u32::MAX),
        (add, u32::MAX, u32::MAX, u32::MAX - 1),
    ]);
    assert_eq!(add.apply_u64(u64::MAX, 1), 0);
    assert_eq!(add.apply_u64(u64::from(u32::MAX), 1), 1 << 32);
}

#[test]
fn reduce_min_and_max_unsigned() {
    let (min, max) = (red(ReductionOp::Min), red(ReductionOp::Max));
    reduce32(&[
        (min, 3, 4, 3),
        (min, 4, 3, 3),
        (min, 4, 4, 4),
        (min, 0x8000_0000, 1, 1),
        (max, 3, 4, 4),
        (max, 4, 3, 4),
        (max, 4, 4, 4),
        (max, 0x8000_0000, 1, 0x8000_0000),
    ]);
}

#[test]
fn signed_format_changes_min_and_max_only() {
    // Divergence from Eden (`ProcessNativeReportSemaphore` compares unsigned).
    let neg = 0x8000_0000; // i32::MIN
    let minus_one = u32::MAX; // -1
    reduce32(&[
        (signed(ReductionOp::Min), neg, 1, neg),
        (signed(ReductionOp::Min), 1, neg, neg),
        (signed(ReductionOp::Min), minus_one, 0, minus_one),
        (signed(ReductionOp::Max), neg, 1, 1),
        (signed(ReductionOp::Max), minus_one, 0, 0),
        (signed(ReductionOp::Max), 0, minus_one, 0),
    ]);
    assert_eq!(signed(ReductionOp::Min).apply_u64(u64::MAX, 0), u64::MAX);
    assert_eq!(signed(ReductionOp::Max).apply_u64(u64::MAX, 0), 0);
    assert_eq!(red(ReductionOp::Max).apply_u64(u64::MAX, 0), u64::MAX);
    // Every other operation gives the same answer in both formats.
    for op in ALL_OPS {
        if matches!(op, ReductionOp::Min | ReductionOp::Max) {
            continue;
        }
        for (old, payload) in [(neg, 1), (1, neg), (minus_one, 0), (0, minus_one), (5, 5)] {
            assert_eq!(
                signed(op).apply_u32(old, payload),
                red(op).apply_u32(old, payload),
                "{op:?} {old:#x} {payload:#x}"
            );
        }
    }
}

#[test]
fn reduce_inc_wraps_to_zero_at_the_payload() {
    let inc = red(ReductionOp::Inc);
    reduce32(&[
        (inc, 3, 5, 4),
        (inc, 4, 5, 5),
        (inc, 5, 5, 0),
        (inc, 6, 5, 0),
        (inc, 0, 0, 0),
        (inc, u32::MAX - 1, u32::MAX, u32::MAX),
        (inc, u32::MAX, u32::MAX, 0),
    ]);
    assert_eq!(inc.apply_u64(u64::MAX - 1, u64::MAX), u64::MAX);
    assert_eq!(inc.apply_u64(u64::MAX, u64::MAX), 0);
}

#[test]
fn reduce_dec_reloads_the_payload_from_zero_or_above() {
    let dec = red(ReductionOp::Dec);
    reduce32(&[
        (dec, 0, 5, 5),
        (dec, 1, 5, 0),
        (dec, 5, 5, 4),
        (dec, 6, 5, 5),
        (dec, u32::MAX, 5, 5),
        (dec, 0, 0, 0),
        (dec, 1, 0, 0),
    ]);
    assert_eq!(dec.apply_u64(0, u64::MAX), u64::MAX);
    assert_eq!(dec.apply_u64(u64::MAX, u64::MAX), u64::MAX - 1);
}

#[test]
fn reduce_bitwise() {
    reduce32(&[
        (red(ReductionOp::And), 0b1100, 0b1010, 0b1000),
        (red(ReductionOp::Or), 0b1100, 0b1010, 0b1110),
        (red(ReductionOp::Xor), 0b1100, 0b1010, 0b0110),
    ]);
    assert_eq!(
        red(ReductionOp::Xor).apply_u64(u64::MAX, 1 << 40),
        u64::MAX ^ (1 << 40)
    );
}

#[test]
fn every_reduction_matches_edens_arithmetic() {
    // `ProcessNativeReportSemaphore`, indexed by the `RED_*` value.
    let eden = |op: u32, old: u32, payload: u32| -> u32 {
        match op {
            0 => old.wrapping_add(payload),
            1 => old.min(payload),
            2 => old.max(payload),
            3 => {
                if old >= payload {
                    0
                } else {
                    old + 1
                }
            }
            4 => {
                if old == 0 || old > payload {
                    payload
                } else {
                    old - 1
                }
            }
            5 => old & payload,
            6 => old | payload,
            _ => old ^ payload,
        }
    };
    let values = [0, 1, 2, 5, 0x7fff_ffff, 0x8000_0000, u32::MAX - 1, u32::MAX];
    for value in 0..=MME_DMA_REDUCTION_OP.max() {
        let reduction = decode_mme_dma_reduction(value).unwrap().reduction;
        for old in values {
            for payload in values {
                assert_eq!(
                    reduction.apply_u32(old, payload),
                    eden(value, old, payload),
                    "{reduction:?} {old:#x} {payload:#x}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// SEMAPHORED
// ---------------------------------------------------------------------------------------------

#[test]
fn semaphored_acquires() {
    let acquire = |op, switch| {
        Ok(HostSemaphore::Acquire {
            op,
            size: PayloadSize::Four,
            switch,
        })
    };
    assert_eq!(
        decode_semaphored(0x0000_0001),
        acquire(AcquireOp::Equal, false)
    );
    assert_eq!(
        decode_semaphored(0x0000_0004),
        acquire(AcquireOp::GreaterOrEqual, false)
    );
    assert_eq!(
        decode_semaphored(0x0000_0008),
        acquire(AcquireOp::And, false)
    );
    // ACQUIRE_SWITCH is bit 12.
    assert_eq!(
        decode_semaphored(0x0000_1001),
        acquire(AcquireOp::Equal, true)
    );
    // Release-only fields do not change an acquire.
    assert_eq!(
        decode_semaphored(0xf910_0004),
        acquire(AcquireOp::GreaterOrEqual, false)
    );
}

#[test]
fn semaphored_release_fields() {
    let release = |timestamp, wfi| {
        Ok(HostSemaphore::Release {
            form: ReleaseForm {
                size: PayloadSize::Four,
                timestamp,
            },
            wfi,
        })
    };
    // RELEASE_SIZE (bit 24): 0 is 16 bytes with a timestamp, 1 is 4 bytes (row 52).
    assert_eq!(decode_semaphored(0x0000_0002), release(true, true));
    assert_eq!(decode_semaphored(0x0100_0002), release(false, true));
    // RELEASE_WFI (bit 20): 0 is enabled, 1 disabled.
    assert_eq!(decode_semaphored(0x0010_0002), release(true, false));
    assert_eq!(decode_semaphored(0x0110_0002), release(false, false));
    // ACQUIRE_SWITCH does not change a release.
    assert_eq!(decode_semaphored(0x0100_1002), release(false, true));
}

#[test]
fn semaphored_reduction_fields() {
    // Divergence from Eden, which logs operation 0x10 as invalid.
    let reduce = |op, format, wfi| {
        Ok(HostSemaphore::Reduce {
            reduction: Reduction { op, format },
            size: PayloadSize::Four,
            wfi,
        })
    };
    // REDUCTION 30:27, FORMAT 31 (0 = signed).
    assert_eq!(
        decode_semaphored(0x0000_0010),
        reduce(ReductionOp::Min, ReductionFormat::Signed, true)
    );
    assert_eq!(
        decode_semaphored(0x8000_0010),
        reduce(ReductionOp::Min, ReductionFormat::Unsigned, true)
    );
    assert_eq!(
        decode_semaphored(0x2810_0010),
        reduce(ReductionOp::Add, ReductionFormat::Signed, false)
    );
    assert_eq!(
        decode_semaphored(0xb800_0010),
        reduce(ReductionOp::Dec, ReductionFormat::Unsigned, true)
    );
    // RELEASE_SIZE is ignored for a reduction.
    assert_eq!(
        decode_semaphored(0x0100_0010),
        decode_semaphored(0x0000_0010)
    );
}

#[test]
fn semaphored_host_reduction_table() {
    let expected = [
        (HOST_REDUCTION_MIN, ReductionOp::Min),
        (HOST_REDUCTION_MAX, ReductionOp::Max),
        (HOST_REDUCTION_XOR, ReductionOp::Xor),
        (HOST_REDUCTION_AND, ReductionOp::And),
        (HOST_REDUCTION_OR, ReductionOp::Or),
        (HOST_REDUCTION_ADD, ReductionOp::Add),
        (HOST_REDUCTION_INC, ReductionOp::Inc),
        (HOST_REDUCTION_DEC, ReductionOp::Dec),
    ];
    // The header's order: MIN MAX XOR AND OR ADD INC DEC = 0..7.
    for (value, (constant, op)) in expected.into_iter().enumerate() {
        assert_eq!(u32::try_from(value).unwrap(), constant);
        let word = (constant << 27) | 0x10;
        let Ok(HostSemaphore::Reduce { reduction, .. }) = decode_semaphored(word) else {
            unreachable!("{word:#x} is a reduction");
        };
        assert_eq!(reduction.op, op);
    }
}

#[test]
fn semaphored_undefined_reductions_are_errors() {
    for value in 8..=15u32 {
        let word = (value << 27) | 0x10;
        assert_eq!(
            decode_semaphored(word),
            Err(DecodeError::UndefinedReduction {
                word: SEMAPHORED,
                value
            })
        );
    }
    assert_eq!(
        decode_semaphored(0x4000_0010).map_err(|e| e.to_string()),
        Err("SEMAPHORED reduction 8 is not defined".to_owned())
    );
}

#[test]
fn semaphored_operation_is_five_bits() {
    // Every value of 4:0 that is not one of the five one-hot operations is an error,
    // including 0x11, which Eden's 4-bit mask would read as an acquire.
    let defined = [
        SEMAPHORED_OPERATION_ACQUIRE,
        SEMAPHORED_OPERATION_RELEASE,
        SEMAPHORED_OPERATION_ACQ_GEQ,
        SEMAPHORED_OPERATION_ACQ_AND,
        SEMAPHORED_OPERATION_REDUCTION,
    ];
    assert_eq!(defined, [0x1, 0x2, 0x4, 0x8, 0x10]);
    for value in 0..=SEMAPHORED_OPERATION.max() {
        let result = decode_semaphored(value);
        if defined.contains(&value) {
            assert!(result.is_ok(), "{value:#x}");
        } else {
            assert_eq!(
                result,
                Err(DecodeError::UndefinedOperation {
                    word: SEMAPHORED,
                    value
                })
            );
        }
    }
    assert_eq!(
        decode_semaphored(0x11).map_err(|e| e.to_string()),
        Err("SEMAPHORED operation 0x11 is not defined".to_owned())
    );
    // Bits above the field do not reach the operation.
    assert_eq!(
        decode_semaphored(0x0000_0020).map_err(|e| e.to_string()),
        Err("SEMAPHORED operation 0x0 is not defined".to_owned())
    );
}

#[test]
fn semaphored_encoder_matches_the_hand_laid_words() {
    for word in [
        0x0000_0001,
        0x0000_1004,
        0x0000_0008,
        0x0000_0002,
        0x0110_0002,
        0x0010_0002,
        0x0100_0002,
        0x2810_0010,
        0xb800_0010,
        0x0000_0010,
    ] {
        let op = decode_semaphored(word).unwrap();
        assert_eq!(encode_semaphored(&op), Ok(word), "{op:?}");
    }
}

#[test]
fn semaphored_cannot_express_wide_or_sem_execute_only_operations() {
    let not = |what| Err(EncodeError::NotInSemaphored { what });
    let acquire = |op, size| HostSemaphore::Acquire {
        op,
        size,
        switch: false,
    };
    assert_eq!(
        encode_semaphored(&acquire(
            AcquireOp::CircularGreaterOrEqual,
            PayloadSize::Four
        )),
        not("a circular greater-or-equal acquire")
    );
    assert_eq!(
        encode_semaphored(&acquire(AcquireOp::Nor, PayloadSize::Four)),
        not("a nor acquire")
    );
    assert_eq!(
        encode_semaphored(&acquire(AcquireOp::Equal, PayloadSize::Eight)),
        not("a 64-bit payload")
    );
    assert_eq!(
        encode_semaphored(&HostSemaphore::Release {
            form: ReleaseForm {
                size: PayloadSize::Eight,
                timestamp: false
            },
            wfi: true,
        }),
        not("a 64-bit payload")
    );
    assert_eq!(
        encode_semaphored(&HostSemaphore::Reduce {
            reduction: red(ReductionOp::Add),
            size: PayloadSize::Eight,
            wfi: true,
        })
        .map_err(|e| e.to_string()),
        Err("SEMAPHORED cannot express a 64-bit payload".to_owned())
    );
}

// ---------------------------------------------------------------------------------------------
// SEM_EXECUTE
// ---------------------------------------------------------------------------------------------

#[test]
fn sem_execute_operation_table() {
    // OPERATION 2:0: 0 acquire, 1 release, 2 strict GEQ, 3 circular GEQ, 4 and, 5 nor,
    // 6 reduction; 7 is undefined (Eden treats it as an acquire already met).
    let acquire = |op| {
        Ok(HostSemaphore::Acquire {
            op,
            size: PayloadSize::Four,
            switch: false,
        })
    };
    assert_eq!(decode_sem_execute(0), acquire(AcquireOp::Equal));
    assert_eq!(decode_sem_execute(2), acquire(AcquireOp::GreaterOrEqual));
    assert_eq!(
        decode_sem_execute(3),
        acquire(AcquireOp::CircularGreaterOrEqual)
    );
    assert_eq!(decode_sem_execute(4), acquire(AcquireOp::And));
    assert_eq!(decode_sem_execute(5), acquire(AcquireOp::Nor));
    assert!(matches!(
        decode_sem_execute(1),
        Ok(HostSemaphore::Release { .. })
    ));
    assert!(matches!(
        decode_sem_execute(6),
        Ok(HostSemaphore::Reduce { .. })
    ));
    assert_eq!(
        decode_sem_execute(7),
        Err(DecodeError::UndefinedOperation {
            word: SEM_EXECUTE,
            value: 7
        })
    );
    assert_eq!(
        decode_sem_execute(0xffff_ffff).map_err(|e| e.to_string()),
        Err("SEM_EXECUTE operation 0x7 is not defined".to_owned())
    );
}

#[test]
fn sem_execute_acquire_fields() {
    // ACQUIRE_SWITCH_TSG bit 12, PAYLOAD_SIZE bit 24.
    assert_eq!(
        decode_sem_execute(0x0100_1003),
        Ok(HostSemaphore::Acquire {
            op: AcquireOp::CircularGreaterOrEqual,
            size: PayloadSize::Eight,
            switch: true,
        })
    );
    // Release and reduction fields do not change an acquire.
    assert_eq!(
        decode_sem_execute(0xfa10_0000),
        Ok(HostSemaphore::Acquire {
            op: AcquireOp::Equal,
            size: PayloadSize::Four,
            switch: false,
        })
    );
}

#[test]
fn sem_execute_release_fields() {
    // Divergence from Eden, which writes 4 bytes for every SEM_EXECUTE release.
    let release = |size, timestamp, wfi| {
        Ok(HostSemaphore::Release {
            form: ReleaseForm { size, timestamp },
            wfi,
        })
    };
    // WFI bit 20 (1 = enabled), PAYLOAD_SIZE bit 24, RELEASE_TIMESTAMP bit 25.
    assert_eq!(
        decode_sem_execute(0x0000_0001),
        release(PayloadSize::Four, false, false)
    );
    assert_eq!(
        decode_sem_execute(0x0010_0001),
        release(PayloadSize::Four, false, true)
    );
    assert_eq!(
        decode_sem_execute(0x0100_0001),
        release(PayloadSize::Eight, false, false)
    );
    assert_eq!(
        decode_sem_execute(0x0200_0001),
        release(PayloadSize::Four, true, false)
    );
    assert_eq!(
        decode_sem_execute(0x0310_0001),
        release(PayloadSize::Eight, true, true)
    );
    // ACQUIRE_SWITCH_TSG does not change a release.
    assert_eq!(
        decode_sem_execute(0x0000_1001),
        release(PayloadSize::Four, false, false)
    );
}

#[test]
fn sem_execute_reduction_fields() {
    // REDUCTION 30:27 (IADD = 5), REDUCTION_FORMAT 31 (1 = unsigned), PAYLOAD_SIZE 24.
    assert_eq!(
        decode_sem_execute(0xa910_0006),
        Ok(HostSemaphore::Reduce {
            reduction: Reduction {
                op: ReductionOp::Add,
                format: ReductionFormat::Unsigned,
            },
            size: PayloadSize::Eight,
            wfi: true,
        })
    );
    assert_eq!(
        decode_sem_execute(0x3000_0006),
        Ok(HostSemaphore::Reduce {
            reduction: Reduction {
                op: ReductionOp::Inc,
                format: ReductionFormat::Signed,
            },
            size: PayloadSize::Four,
            wfi: false,
        })
    );
    assert_eq!(
        decode_sem_execute(0x7800_0006),
        Err(DecodeError::UndefinedReduction {
            word: SEM_EXECUTE,
            value: 15
        })
    );
    // Timestamp is ignored for a reduction.
    assert_eq!(
        decode_sem_execute(0x3200_0006),
        decode_sem_execute(0x3000_0006)
    );
}

#[test]
fn sem_execute_encoder_matches_the_hand_laid_words() {
    for word in [
        0x0100_1003,
        0x0000_0000,
        0x0000_0005,
        0x0310_0001,
        0x0200_0001,
        0xa910_0006,
        0x3000_0006,
    ] {
        let op = decode_sem_execute(word).unwrap();
        assert_eq!(encode_sem_execute(&op), word, "{op:?}");
    }
}

#[test]
fn semaphored_and_sem_execute_agree_on_every_shared_operation() {
    for word in 0..=SEMAPHORED_OPERATION.max() {
        for extra in [0, 0x0110_1000, 0xf800_0000, 0x7810_0000] {
            if let Ok(op) = decode_semaphored(word | extra) {
                let via_sem_execute = decode_sem_execute(encode_sem_execute(&op));
                assert_eq!(via_sem_execute, Ok(op), "{:#x}", word | extra);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// MME_DMA_REDUCTION and QMD releases
// ---------------------------------------------------------------------------------------------

#[test]
fn red_encoding_table() {
    // The header's order: ADD MIN MAX INC DEC AND OR XOR = 0..7, for MME and QMD alike.
    let expected = [
        (RED_ADD, ReductionOp::Add),
        (RED_MIN, ReductionOp::Min),
        (RED_MAX, ReductionOp::Max),
        (RED_INC, ReductionOp::Inc),
        (RED_DEC, ReductionOp::Dec),
        (RED_AND, ReductionOp::And),
        (RED_OR, ReductionOp::Or),
        (RED_XOR, ReductionOp::Xor),
    ];
    for (value, (constant, op)) in expected.into_iter().enumerate() {
        assert_eq!(u32::try_from(value).unwrap(), constant);
        assert_eq!(
            decode_mme_dma_reduction(constant).unwrap().reduction,
            red(op)
        );
        assert_eq!(
            decode_qmd_release_reduction((1 << 26) | (constant << 20)),
            Ok(Some(red(op)))
        );
    }
}

#[test]
fn mme_dma_reduction_fields() {
    // OP 2:0 (RED_MAX = 2), FORMAT 5:4 (1 = signed), SIZE 8 (1 = eight bytes).
    assert_eq!(
        decode_mme_dma_reduction(0x0000_0112),
        Ok(MmeDmaReduction {
            reduction: signed(ReductionOp::Max),
            size: PayloadSize::Eight,
        })
    );
    assert_eq!(
        decode_mme_dma_reduction(0x0000_0007),
        Ok(MmeDmaReduction {
            reduction: red(ReductionOp::Xor),
            size: PayloadSize::Four,
        })
    );
    // Bits outside the three fields are ignored.
    assert_eq!(
        decode_mme_dma_reduction(0xffff_fe08),
        decode_mme_dma_reduction(0x0000_0000)
    );
}

#[test]
fn mme_dma_reduction_undefined_formats_are_errors() {
    for value in 2..=3u32 {
        assert_eq!(
            decode_mme_dma_reduction(value << 4),
            Err(DecodeError::UndefinedFormat {
                word: MME_DMA_REDUCTION,
                value
            })
        );
    }
    assert_eq!(
        decode_mme_dma_reduction(0x30).map_err(|e| e.to_string()),
        Err("MME_DMA_REDUCTION reduction format 3 is not defined".to_owned())
    );
}

#[test]
fn qmd_release_reduction_fields() {
    // OP 22:20 (RED_DEC = 4), FORMAT 25:24 (1 = signed), ENABLE 26.
    assert_eq!(
        decode_qmd_release_reduction(0x0540_0000),
        Ok(Some(signed(ReductionOp::Dec)))
    );
    // Enable clear: a plain release, whatever the other bits say, even an undefined format.
    assert_eq!(decode_qmd_release_reduction(0x0370_0000), Ok(None));
    assert_eq!(decode_qmd_release_reduction(0), Ok(None));
    // The slot's own enable (bit 23) and membar (bit 19) are not this decoder's.
    assert_eq!(
        decode_qmd_release_reduction(0x0488_0000),
        Ok(Some(red(ReductionOp::Add)))
    );
    assert_eq!(
        decode_qmd_release_reduction(0x0700_0000).map_err(|e| e.to_string()),
        Err("QMD release reduction format 3 is not defined".to_owned())
    );
    assert_eq!(
        decode_qmd_release_reduction(0x0600_0000),
        Err(DecodeError::UndefinedFormat {
            word: QMD_RELEASE,
            value: 2
        })
    );
}

#[test]
fn red_encoders_match_the_hand_laid_words() {
    for word in [0x0000_0112, 0x0000_0007, 0x0000_0000, 0x0000_0013] {
        let reduction = decode_mme_dma_reduction(word).unwrap();
        assert_eq!(encode_mme_dma_reduction(&reduction), word);
    }
    for word in [0x0540_0000, 0x0400_0000, 0x0570_0000, 0x0410_0000] {
        let reduction = decode_qmd_release_reduction(word).unwrap();
        assert_eq!(encode_qmd_release_reduction(reduction), word);
    }
    assert_eq!(encode_qmd_release_reduction(None), 0);
}

#[test]
fn the_red_decoder_rejects_a_value_past_three_bits() {
    // Unreachable from the 3-bit fields; pinned so the fallback stays an error, not a panic.
    assert_eq!(
        red_reduction(QMD_RELEASE, 8, RED_FORMAT_UNSIGNED),
        Err(DecodeError::UndefinedReduction {
            word: QMD_RELEASE,
            value: 8
        })
    );
}

#[test]
fn host_and_red_encodings_invert_the_format_bit() {
    let unsigned = HostSemaphore::Reduce {
        reduction: red(ReductionOp::Min),
        size: PayloadSize::Four,
        wfi: true,
    };
    // Host: format 1 is unsigned. RED: format 0 is unsigned.
    assert_eq!(encode_semaphored(&unsigned), Ok(0x8000_0010));
    assert_eq!(
        encode_mme_dma_reduction(&MmeDmaReduction {
            reduction: red(ReductionOp::Min),
            size: PayloadSize::Four,
        }),
        0x0000_0001
    );
}

// ---------------------------------------------------------------------------------------------
// Release forms
// ---------------------------------------------------------------------------------------------

#[test]
fn release_form_sizes() {
    let form = |size, timestamp| ReleaseForm { size, timestamp };
    assert_eq!(form(PayloadSize::Four, false).bytes(), 4);
    assert_eq!(form(PayloadSize::Eight, false).bytes(), 8);
    assert_eq!(form(PayloadSize::Four, true).bytes(), 16);
    assert_eq!(form(PayloadSize::Eight, true).bytes(), 16);
    assert_eq!(PayloadSize::Four.bytes(), 4);
    assert_eq!(PayloadSize::Eight.bytes(), 8);
    // The timestamp follows a 64-bit payload slot.
    assert_eq!(RELEASE_TIMESTAMP_OFFSET, PayloadSize::Eight.bytes());
}

// ---------------------------------------------------------------------------------------------
// Generated
// ---------------------------------------------------------------------------------------------

fn any_reduction() -> impl Strategy<Value = Reduction> {
    (0..ALL_OPS.len(), 0..ALL_FORMATS.len()).prop_map(|(op, format)| Reduction {
        op: ALL_OPS[op],
        format: ALL_FORMATS[format],
    })
}

fn any_size() -> impl Strategy<Value = PayloadSize> {
    prop_oneof![Just(PayloadSize::Four), Just(PayloadSize::Eight)]
}

fn any_acquire_op() -> impl Strategy<Value = AcquireOp> {
    prop_oneof![
        Just(AcquireOp::Equal),
        Just(AcquireOp::GreaterOrEqual),
        Just(AcquireOp::CircularGreaterOrEqual),
        Just(AcquireOp::And),
        Just(AcquireOp::Nor),
    ]
}

fn any_host_semaphore() -> impl Strategy<Value = HostSemaphore> {
    prop_oneof![
        (any_acquire_op(), any_size(), any::<bool>())
            .prop_map(|(op, size, switch)| HostSemaphore::Acquire { op, size, switch }),
        (any_size(), any::<bool>(), any::<bool>()).prop_map(|(size, timestamp, wfi)| {
            HostSemaphore::Release {
                form: ReleaseForm { size, timestamp },
                wfi,
            }
        }),
        (any_reduction(), any_size(), any::<bool>()).prop_map(|(reduction, size, wfi)| {
            HostSemaphore::Reduce {
                reduction,
                size,
                wfi,
            }
        }),
    ]
}

proptest! {
    #![proptest_config(config())]

    /// Any word decodes or errors in every decoder, never panics; a decoded value re-encodes to
    /// a word that decodes the same and sets no bit the original did not.
    #[test]
    fn any_word_decodes_or_errors(word in any::<u32>()) {
        if let Ok(op) = decode_semaphored(word) {
            let canonical = encode_semaphored(&op).unwrap();
            prop_assert_eq!(decode_semaphored(canonical), Ok(op));
            prop_assert_eq!(canonical & !word, 0);
        }
        if let Ok(op) = decode_sem_execute(word) {
            let canonical = encode_sem_execute(&op);
            prop_assert_eq!(decode_sem_execute(canonical), Ok(op));
            prop_assert_eq!(canonical & !word, 0);
        }
        if let Ok(reduction) = decode_mme_dma_reduction(word) {
            let canonical = encode_mme_dma_reduction(&reduction);
            prop_assert_eq!(decode_mme_dma_reduction(canonical), Ok(reduction));
            prop_assert_eq!(canonical & !word, 0);
        }
        if let Ok(reduction) = decode_qmd_release_reduction(word) {
            let canonical = encode_qmd_release_reduction(reduction);
            prop_assert_eq!(decode_qmd_release_reduction(canonical), Ok(reduction));
            prop_assert_eq!(canonical & !word, 0);
        }
    }

    /// Build-then-parse through SEM_EXECUTE, which expresses every operation: every field
    /// survives, so two swapped reads would fail.
    #[test]
    fn generated_sem_execute_round_trips(op in any_host_semaphore()) {
        prop_assert_eq!(decode_sem_execute(encode_sem_execute(&op)), Ok(op));
    }

    /// Build-then-parse through SEMAPHORED, for the operations it expresses.
    #[test]
    fn generated_semaphored_round_trips(op in any_host_semaphore()) {
        if let Ok(word) = encode_semaphored(&op) {
            prop_assert_eq!(decode_semaphored(word), Ok(op));
        }
    }

    /// Build-then-parse for the RED encodings.
    #[test]
    fn generated_red_encodings_round_trip(reduction in any_reduction(), size in any_size()) {
        let mme = MmeDmaReduction { reduction, size };
        prop_assert_eq!(decode_mme_dma_reduction(encode_mme_dma_reduction(&mme)), Ok(mme));
        let qmd = encode_qmd_release_reduction(Some(reduction));
        prop_assert_eq!(decode_qmd_release_reduction(qmd), Ok(Some(reduction)));
    }

    /// The 32-bit and 64-bit reductions agree on values that fit 32 bits, for the operations
    /// whose result cannot carry past bit 31; and signed 64-bit min and max agree with
    /// sign-extended 32-bit ones.
    #[test]
    fn widths_agree(reduction in any_reduction(), old in any::<u32>(), payload in any::<u32>()) {
        let wide = |v: u32| match reduction.format {
            ReductionFormat::Unsigned => u64::from(v),
            ReductionFormat::Signed => i64::from(v.cast_signed()).cast_unsigned(),
        };
        let narrow = reduction.apply_u32(old, payload);
        let full = reduction.apply_u64(wide(old), wide(payload));
        match reduction.op {
            ReductionOp::Add => {}
            ReductionOp::Inc | ReductionOp::Dec if reduction.format == ReductionFormat::Signed => {}
            _ => prop_assert_eq!(full, wide(narrow)),
        }
    }

    /// Generated acquires: exactly one of And and Nor holds, and circular GEQ agrees with
    /// strict GEQ when both values are below 2^31.
    #[test]
    fn acquire_relations(value in any::<u32>(), payload in any::<u32>()) {
        prop_assert_ne!(AcquireOp::And.is_met(value, payload), AcquireOp::Nor.is_met(value, payload));
        if value <= i32::MAX.cast_unsigned() && payload <= i32::MAX.cast_unsigned() {
            prop_assert_eq!(
                AcquireOp::CircularGreaterOrEqual.is_met(value, payload),
                AcquireOp::GreaterOrEqual.is_met(value, payload)
            );
        }
        prop_assert_eq!(AcquireOp::Equal.is_met(value, payload), value == payload);
    }
}

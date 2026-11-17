//! `Break`: the reason word decoded as Eden splits it, and the buffer read when Eden reads it.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed};
use starquake_memory::FlatMemory;

use super::*;

/// Above 4 GiB, so an address truncated to 32 bits would miss it.
const BASE: u64 = 0x80_0000_0000;
/// The word the tests' buffers hold: every byte different, so a byte-order mistake shows.
const CODE: u32 = 0x1234_5678;

/// Fixed seed so a run is reproducible; failures are not persisted to disk.
const PROPTEST_SEED: u64 = 0x4252_4541_4B5F_5345;

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 512,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPTEST_SEED),
        failure_persistence: None,
        ..Config::default()
    }
}

/// A page of memory at [`BASE`] whose first four bytes hold [`CODE`].
fn memory() -> FlatMemory {
    let memory = FlatMemory::new(BASE, 0x1000);
    memory.write(BASE, &CODE.to_le_bytes()).unwrap();
    memory
}

fn code_of(memory: &FlatMemory, raw_reason: u32, address: u64, size: u64) -> Option<u32> {
    break_execution(memory, raw_reason, address, size).code
}

#[test]
fn the_named_reasons_have_eden_s_numbers_and_names() {
    // `svc_types.h` @ cf654052, lines 171-178.
    let expected = [
        (BreakReason::Panic, 0, "panic"),
        (BreakReason::Assert, 1, "assert"),
        (BreakReason::User, 2, "user"),
        (BreakReason::PreLoadDll, 3, "pre_load_dll"),
        (BreakReason::PostLoadDll, 4, "post_load_dll"),
        (BreakReason::PreUnloadDll, 5, "pre_unload_dll"),
        (BreakReason::PostUnloadDll, 6, "post_unload_dll"),
        (BreakReason::CppException, 7, "cpp_exception"),
    ];
    assert_eq!(BreakReason::NAMED.len(), expected.len());
    for ((reason, number, name), &named) in expected.into_iter().zip(BreakReason::NAMED) {
        assert_eq!(named, reason);
        assert_eq!(reason.raw(), number);
        assert_eq!(BreakReason::from_raw(number), reason);
        assert_eq!(reason.name(), Some(name));
    }
    // The first number past them, and the flag on its own, are not named.
    let past = BreakReason::CppException.raw() + 1;
    assert_eq!(BreakReason::from_raw(past), BreakReason::Other(past));
    assert_eq!(BreakReason::Other(past).name(), None);
    assert_eq!(
        BreakReason::from_raw(NOTIFICATION_ONLY_FLAG),
        BreakReason::Other(NOTIFICATION_ONLY_FLAG)
    );
}

#[test]
fn hand_built_reason_words_decode_as_eden_splits_them() {
    // `reason & ~NotificationOnlyFlag` and `True(reason & NotificationOnlyFlag)`.
    let cases = [
        (0x0000_0000, BreakReason::Panic, false),
        (0x0000_0001, BreakReason::Assert, false),
        (0x8000_0000, BreakReason::Panic, true),
        (0x8000_0003, BreakReason::PreLoadDll, true),
        (0x8000_0007, BreakReason::CppException, true),
        (0x0000_0008, BreakReason::Other(8), false),
        (0x7fff_ffff, BreakReason::Other(0x7fff_ffff), false),
        (0xffff_ffff, BreakReason::Other(0x7fff_ffff), true),
    ];
    for (raw, reason, notification_only) in cases {
        let call = Break::decode(raw, 0x11, 0x22);
        assert_eq!(
            call,
            Break {
                reason,
                notification_only,
                address: 0x11,
                size: 0x22,
                code: None,
            },
            "{raw:#x}"
        );
    }
}

proptest! {
    #![proptest_config(config())]

    /// Decoding then encoding gives the word back, and each part comes from its own bits.
    #[test]
    fn a_reason_word_survives_decode_and_encode(raw in any::<u32>(), address in any::<u64>(), size in any::<u64>()) {
        let call = Break::decode(raw, address, size);
        prop_assert_eq!(call.raw_reason(), raw);
        prop_assert_eq!(call.reason.raw(), raw & !NOTIFICATION_ONLY_FLAG);
        prop_assert_eq!(call.notification_only, raw & NOTIFICATION_ONLY_FLAG != 0);
        prop_assert_eq!((call.address, call.size), (address, size));
        prop_assert_eq!(Break::decode(call.raw_reason(), address, size), call);
    }
}

#[test]
fn a_break_that_is_not_a_notification_reads_its_four_byte_buffer_whatever_its_reason() {
    let memory = memory();
    // Every named reason, and one Eden does not name: `if (!notification_only)` reads it.
    let reasons = BreakReason::NAMED
        .iter()
        .copied()
        .chain([BreakReason::Other(0x99)]);
    for reason in reasons {
        let call = break_execution(&memory, reason.raw(), BASE, RESULT_ARGUMENT_SIZE);
        assert_eq!(call.code, Some(CODE), "{reason:?}");
        assert_eq!((call.reason, call.notification_only), (reason, false));
    }
}

#[test]
fn a_notification_reads_its_buffer_only_for_the_reasons_eden_s_switch_reads_it_for() {
    let memory = memory();
    let read = [
        BreakReason::Panic,
        BreakReason::Assert,
        BreakReason::User,
        BreakReason::Other(0x99),
    ];
    let not_read = [
        BreakReason::PreLoadDll,
        BreakReason::PostLoadDll,
        BreakReason::PreUnloadDll,
        BreakReason::PostUnloadDll,
        BreakReason::CppException,
    ];
    assert_eq!(read.len() - 1 + not_read.len(), BreakReason::NAMED.len());
    for reason in read {
        let raw = reason.raw() | NOTIFICATION_ONLY_FLAG;
        assert_eq!(
            code_of(&memory, raw, BASE, RESULT_ARGUMENT_SIZE),
            Some(CODE),
            "{reason:?}"
        );
    }
    for reason in not_read {
        let raw = reason.raw() | NOTIFICATION_ONLY_FLAG;
        assert_eq!(
            code_of(&memory, raw, BASE, RESULT_ARGUMENT_SIZE),
            None,
            "{reason:?}"
        );
    }
}

#[test]
fn only_a_four_byte_buffer_is_read() {
    // Divergence: Eden hex-dumps a buffer of any other size.
    let memory = memory();
    let panic = BreakReason::Panic.raw();
    for size in [
        0,
        1,
        RESULT_ARGUMENT_SIZE - 1,
        RESULT_ARGUMENT_SIZE + 1,
        8,
        u64::MAX,
    ] {
        assert_eq!(code_of(&memory, panic, BASE, size), None, "size {size}");
    }
    assert_eq!(
        code_of(&memory, panic, BASE, RESULT_ARGUMENT_SIZE),
        Some(CODE)
    );
}

#[test]
fn a_buffer_at_address_zero_is_not_read() {
    // `if (sz == 0 || addr == 0 ...) return;`. Memory at zero holds a word, so a read would
    // find it.
    let memory = FlatMemory::new(0, 0x10);
    memory.write(0, &CODE.to_le_bytes()).unwrap();
    memory.write(4, &CODE.to_le_bytes()).unwrap();
    let panic = BreakReason::Panic.raw();
    assert_eq!(code_of(&memory, panic, 0, RESULT_ARGUMENT_SIZE), None);
    assert_eq!(code_of(&memory, panic, 4, RESULT_ARGUMENT_SIZE), Some(CODE));
}

#[test]
fn an_unreadable_buffer_gives_no_code_and_a_readable_one_at_the_edge_does() {
    // Divergence: Eden's `Read32` reads zero from memory it cannot read.
    let memory = FlatMemory::new(BASE, 0x10);
    let last = BASE + 0x10 - RESULT_ARGUMENT_SIZE;
    memory.write(last, &CODE.to_le_bytes()).unwrap();
    let panic = BreakReason::Panic.raw();
    // The last four bytes are readable; one byte later the word's last byte is not, and one
    // byte before the memory its first is not.
    assert_eq!(
        code_of(&memory, panic, last, RESULT_ARGUMENT_SIZE),
        Some(CODE)
    );
    assert_eq!(
        code_of(&memory, panic, last + 1, RESULT_ARGUMENT_SIZE),
        None
    );
    assert_eq!(
        code_of(&memory, panic, BASE - 1, RESULT_ARGUMENT_SIZE),
        None
    );
}

#[test]
fn a_buffer_that_wraps_the_address_space_is_not_read() {
    // Divergence: Eden does not check. The memory covers the top eight bytes of the address
    // space, so only the extent check can refuse a word whose last byte would wrap.
    let base = u64::MAX - 7;
    let memory = FlatMemory::new(base, 8);
    memory.write(base, &[0xaa; 8]).unwrap();
    let panic = BreakReason::Panic.raw();
    // The last word of the address space: its last byte is u64::MAX and its exclusive end is
    // exactly 2^64, which does not wrap, so it is read.
    let top_word = u64::MAX - (RESULT_ARGUMENT_SIZE - 1);
    assert_eq!(
        code_of(&memory, panic, top_word, RESULT_ARGUMENT_SIZE),
        Some(0xaaaa_aaaa)
    );
    // One higher, the last byte would be at 2^64: wraps by exactly one.
    assert_eq!(
        code_of(&memory, panic, top_word + 1, RESULT_ARGUMENT_SIZE),
        None
    );
    assert_eq!(
        code_of(&memory, panic, u64::MAX, RESULT_ARGUMENT_SIZE),
        None
    );
    // One lower is read too.
    assert_eq!(
        code_of(&memory, panic, top_word - 1, RESULT_ARGUMENT_SIZE),
        Some(0xaaaa_aaaa)
    );
}

//! Tests for GP entry and method header decoding and the push-buffer encoder.
//!
//! The hand-laid words below are written out from `clc76f.h`'s bit ranges, deliberately not
//! built from this module's constants, so a wrong constant cannot agree with itself.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use super::*;

fn config() -> Config {
    Config {
        cases: 2048,
        rng_seed: RngSeed::Fixed(0x6766_6966),
        failure_persistence: None,
        ..Config::default()
    }
}

// ---------------------------------------------------------------------------------------------
// GP entries
// ---------------------------------------------------------------------------------------------

#[test]
fn every_field_constant_is_a_non_empty_field() {
    for field in [
        GP_ENTRY0_FETCH,
        GP_ENTRY0_GET,
        GP_ENTRY1_GET_HI,
        GP_ENTRY1_LEVEL,
        GP_ENTRY1_LENGTH,
        GP_ENTRY1_SYNC,
        METHOD_ADDRESS,
        METHOD_SUBCHANNEL,
        METHOD_COUNT,
        IMMD_DATA,
        SEC_OP,
        TERT_OP,
        METHOD_ADDRESS_OLD,
        METHOD_COUNT_OLD,
        SUBDEVICE_MASK,
        SUBDEVICE_OPCODE,
    ] {
        assert_ne!(field.width(), 0, "{field:?}");
    }
}

#[test]
fn a_zero_entry_is_main_unconditional_proceed_and_empty() {
    assert_eq!(GpEntry::from_words([0, 0]), GpEntry::default());
    assert_eq!(
        GpEntry::default(),
        GpEntry {
            address: 0,
            length: 0,
            level: Level::Main,
            fetch: Fetch::Unconditional,
            sync: Sync::Proceed,
        }
    );
}

#[test]
fn entry0_bits_31_to_2_are_address_bits_31_to_2() {
    let entry = GpEntry::from_words([0x89ab_cdec, 0]);
    assert_eq!(entry.address, 0x89ab_cdec);
    assert_eq!(entry.fetch, Fetch::Unconditional);
}

#[test]
fn entry1_bits_7_to_0_are_address_bits_39_to_32() {
    let entry = GpEntry::from_words([0, 0x0000_00a5]);
    assert_eq!(entry.address, 0xa5_0000_0000);
    assert_eq!(entry.length, 0);
}

#[test]
fn entry0_low_bits_are_fetch_and_ignored_not_address() {
    // Divergence from Eden: its `CommandListHeader::addr` takes bits 39:0, bits 1:0 included.
    let fetch = GpEntry::from_words([0x0000_0001, 0]);
    assert_eq!(fetch.fetch, Fetch::Conditional);
    assert_eq!(fetch.address, 0);
    let bit1 = GpEntry::from_words([0x0000_0002, 0]);
    assert_eq!(bit1, GpEntry::default());
}

#[test]
fn entry1_bit_9_is_the_level() {
    let entry = GpEntry::from_words([0, 0x0000_0200]);
    assert_eq!(entry.level, Level::Subroutine);
    assert_eq!(
        GpEntry {
            level: Level::Main,
            ..entry
        },
        GpEntry::default()
    );
}

#[test]
fn entry1_bit_8_is_ignored() {
    // Eden's `allow_flush` (raw bit 40); not a field in `clc76f.h`.
    assert_eq!(GpEntry::from_words([0, 0x0000_0100]), GpEntry::default());
}

#[test]
fn entry1_bits_30_to_10_are_the_length() {
    assert_eq!(GpEntry::from_words([0, 0x0000_0400]).length, 1);
    assert_eq!(GpEntry::from_words([0, 0x7fff_fc00]).length, 0x1f_ffff);
    assert_eq!(
        GpEntry::from_words([0, 0x7fff_fc00]).length,
        GP_ENTRY1_LENGTH.max()
    );
}

#[test]
fn entry1_bit_31_is_sync() {
    let entry = GpEntry::from_words([0, 0x8000_0000]);
    assert_eq!(entry.sync, Sync::Wait);
    assert_eq!(
        GpEntry {
            sync: Sync::Proceed,
            ..entry
        },
        GpEntry::default()
    );
}

#[test]
fn every_entry_field_at_once_with_distinct_values() {
    // Address 0xab_1234_5678, fetch conditional, length 0x2345, subroutine, sync wait.
    let words = [0x1234_5679, 0x808d_16ab];
    let entry = GpEntry::from_words(words);
    assert_eq!(
        entry,
        GpEntry {
            address: 0xab_1234_5678,
            length: 0x2345,
            level: Level::Subroutine,
            fetch: Fetch::Conditional,
            sync: Sync::Wait,
        }
    );
    assert_eq!(entry.to_words(), Ok(words));
}

#[test]
fn raw_entries_are_the_eight_bytes_little_endian() {
    let bytes = [0x79, 0x56, 0x34, 0x12, 0xab, 0x16, 0x8d, 0x80];
    let raw = u64::from_le_bytes(bytes);
    assert_eq!(
        GpEntry::from_raw(raw),
        GpEntry::from_words([0x1234_5679, 0x808d_16ab])
    );
    assert_eq!(GpEntry::from_raw(raw).to_raw(), Ok(raw));
}

#[test]
fn raw_entries_agree_with_edens_command_list_header() {
    // `CommandListHeader`: addr bits 39:0, size bits 62:42, sync bit 63.
    assert_eq!(GpEntry::from_raw(1 << 42).length, 1);
    assert_eq!(GpEntry::from_raw(0x1f_ffff << 42).length, 0x1f_ffff);
    assert_eq!(GpEntry::from_raw(1 << 63).sync, Sync::Wait);
    assert_eq!(GpEntry::from_raw(0xff_ffff_fffc).address, 0xff_ffff_fffc);
}

#[test]
fn byte_len_is_four_bytes_per_word_without_overflow() {
    let entry = GpEntry {
        length: GP_ENTRY1_LENGTH.max(),
        ..GpEntry::default()
    };
    assert_eq!(entry.byte_len(), u64::from(GP_ENTRY1_LENGTH.max()) * 4);
    assert_eq!(GpEntry::default().byte_len(), 0);
}

#[test]
fn encoding_rejects_a_misaligned_address() {
    for address in [1, 2, 3, 0x1000_0002] {
        let entry = GpEntry {
            address,
            ..GpEntry::default()
        };
        assert_eq!(
            entry.to_words(),
            Err(EncodeError::MisalignedAddress { address })
        );
    }
    let entry = GpEntry {
        address: 2,
        ..GpEntry::default()
    };
    assert_eq!(
        entry.to_raw().map_err(|e| e.to_string()),
        Err("GP entry address 0x2 is not 4-byte aligned".to_owned())
    );
}

#[test]
fn encoding_takes_the_last_address_below_bit_40_and_rejects_bit_40() {
    let bits = GP_ENTRY_GET_HI_SHIFT + GP_ENTRY1_GET_HI.width();
    let limit = 1u64 << bits;
    let last = GpEntry {
        address: limit - 4,
        ..GpEntry::default()
    };
    assert_eq!(last.to_words(), Ok([0xffff_fffc, 0x0000_00ff]));
    let over = GpEntry {
        address: limit,
        ..GpEntry::default()
    };
    assert_eq!(
        over.to_words(),
        Err(EncodeError::FieldOverflow {
            field: "GP entry address",
            value: limit,
            bits,
        })
    );
    assert_eq!(
        over.to_raw().map_err(|e| e.to_string()),
        Err("GP entry address 0x10000000000 does not fit in 40 bits".to_owned())
    );
    let top = GpEntry {
        address: u64::MAX - 3,
        ..GpEntry::default()
    };
    assert!(matches!(
        top.to_words(),
        Err(EncodeError::FieldOverflow { .. })
    ));
}

#[test]
fn encoding_takes_the_largest_length_and_rejects_one_more() {
    let max = GpEntry {
        length: GP_ENTRY1_LENGTH.max(),
        ..GpEntry::default()
    };
    assert_eq!(max.to_words(), Ok([0, 0x7fff_fc00]));
    let over = GpEntry {
        length: GP_ENTRY1_LENGTH.max() + 1,
        ..GpEntry::default()
    };
    assert_eq!(
        over.to_words(),
        Err(EncodeError::FieldOverflow {
            field: "GP entry length",
            value: u64::from(GP_ENTRY1_LENGTH.max() + 1),
            bits: GP_ENTRY1_LENGTH.width(),
        })
    );
}

// ---------------------------------------------------------------------------------------------
// Method headers: hand-laid words
// ---------------------------------------------------------------------------------------------

#[test]
fn incrementing_header_fields() {
    // Opcode 1 (31:29), count 5 (28:16), subchannel 3 (15:13), method 0x123 (11:0).
    assert_eq!(
        decode_header(0x2005_6123),
        Ok(MethodHeader::Methods {
            mode: MethodMode::Incrementing,
            subchannel: 3,
            method: 0x123,
            count: 5,
        })
    );
}

#[test]
fn non_incrementing_header_fields() {
    // Opcode 3, count 0x1fff, subchannel 0, method 0xfff.
    assert_eq!(
        decode_header(0x7fff_0fff),
        Ok(MethodHeader::Methods {
            mode: MethodMode::NonIncrementing,
            subchannel: 0,
            method: 0xfff,
            count: 0x1fff,
        })
    );
}

#[test]
fn increment_once_header_fields() {
    // Opcode 5, count 2, subchannel 7, method 0x40.
    assert_eq!(
        decode_header(0xa002_e040),
        Ok(MethodHeader::Methods {
            mode: MethodMode::IncrementOnce,
            subchannel: 7,
            method: 0x40,
            count: 2,
        })
    );
}

#[test]
fn immediate_header_fields() {
    // Opcode 4, data 0x1abc (28:16), subchannel 7, method 0xfff.
    assert_eq!(
        decode_header(0x9abc_efff),
        Ok(MethodHeader::Immediate {
            subchannel: 7,
            method: 0xfff,
            data: 0x1abc,
        })
    );
}

#[test]
fn bit_12_is_not_part_of_the_method() {
    // Divergence from Eden, whose `CommandHeader::method` is 13 bits and would read 0x1123.
    let expected = MethodHeader::Methods {
        mode: MethodMode::Incrementing,
        subchannel: 0,
        method: 0x123,
        count: 1,
    };
    assert_eq!(decode_header(0x2001_1123), Ok(expected));
    assert_eq!(decode_header(0x2001_0123), Ok(expected));
    assert_eq!(
        decode_header(0x8001_1123),
        Ok(MethodHeader::Immediate {
            subchannel: 0,
            method: 0x123,
            data: 1,
        })
    );
}

#[test]
fn end_segment_ignores_the_rest_of_the_word() {
    assert_eq!(decode_header(0xe000_0000), Ok(MethodHeader::EndSegment));
    assert_eq!(decode_header(0xffff_ffff), Ok(MethodHeader::EndSegment));
}

#[test]
fn opcode_6_is_reserved() {
    assert_eq!(
        decode_header(0xc000_0000),
        Err(DecodeError::ReservedOpcode { word: 0xc000_0000 })
    );
    assert_eq!(
        decode_header(0xdfff_ffff).map_err(|e| e.to_string()),
        Err("method header 0xdfffffff uses the reserved opcode 6".to_owned())
    );
}

#[test]
fn old_format_incrementing_header_fields() {
    // Opcode 0, tertiary 0 (17:16), count 0x7ff (28:18), subchannel 5, method 0x7ff (12:2).
    assert_eq!(
        decode_header(0x1ffc_bffc),
        Ok(MethodHeader::Legacy {
            mode: MethodMode::Incrementing,
            subchannel: 5,
            method: 0x7ff,
            count: 0x7ff,
        })
    );
}

#[test]
fn old_format_non_incrementing_header_fields() {
    // Opcode 2, tertiary 0, count 1, subchannel 1, method 2.
    assert_eq!(
        decode_header(0x4004_2008),
        Ok(MethodHeader::Legacy {
            mode: MethodMode::NonIncrementing,
            subchannel: 1,
            method: 2,
            count: 1,
        })
    );
}

#[test]
fn the_zero_word_is_an_empty_old_format_method() {
    // `NVC76F_DMA_NOP` is 0x00000000.
    assert_eq!(
        decode_header(0),
        Ok(MethodHeader::Legacy {
            mode: MethodMode::Incrementing,
            subchannel: 0,
            method: 0,
            count: 0,
        })
    );
}

#[test]
fn old_format_low_two_bits_are_ignored() {
    assert_eq!(decode_header(0x0004_0007), decode_header(0x0004_0004));
}

#[test]
fn subdevice_mask_operations() {
    assert_eq!(
        decode_header(0x0001_0ab0),
        Ok(MethodHeader::SubdeviceMask(SubdeviceMask::Set {
            mask: 0xab
        }))
    );
    assert_eq!(
        decode_header(0x0002_fff0),
        Ok(MethodHeader::SubdeviceMask(SubdeviceMask::Store {
            mask: 0xfff
        }))
    );
    assert_eq!(
        decode_header(0x0003_0000),
        Ok(MethodHeader::SubdeviceMask(SubdeviceMask::Use))
    );
    // USE has no value field: bits 15:0 are ignored.
    assert_eq!(
        decode_header(0x0003_ffff),
        Ok(MethodHeader::SubdeviceMask(SubdeviceMask::Use))
    );
    // SET's value is 15:4: bits 3:0 are ignored.
    assert_eq!(decode_header(0x0001_0abf), decode_header(0x0001_0ab0));
}

#[test]
fn a_subdevice_tertiary_with_count_bits_set_is_undefined() {
    // Opcode 0, tertiary 1, but bits 31:16 are 0x0005, not 1, 2 or 3.
    assert_eq!(
        decode_header(0x0005_0000),
        Err(DecodeError::UndefinedSubdeviceOp { word: 0x0005_0000 })
    );
    assert_eq!(
        decode_header(0x0007_0000).map_err(|e| e.to_string()),
        Err("method header 0x00070000 is not a defined subdevice-mask operation".to_owned())
    );
}

#[test]
fn opcode_2_with_a_non_zero_tertiary_is_undefined() {
    for word in [0x4001_0000, 0x4002_0000, 0x4003_0000] {
        assert_eq!(
            decode_header(word),
            Err(DecodeError::UndefinedGroup2Op { word })
        );
    }
    assert_eq!(
        decode_header(0x4001_0000).map_err(|e| e.to_string()),
        Err("method header 0x40010000 has an undefined tertiary opcode for opcode 2".to_owned())
    );
}

/// The kind of header each opcode and tertiary opcode decodes to, with nothing else set.
#[test]
fn opcode_and_tertiary_table() {
    #[derive(Debug, PartialEq)]
    enum Kind {
        Methods(MethodMode),
        Immediate,
        Legacy(MethodMode),
        Subdevice,
        End,
        Reserved,
        UndefinedSubdevice,
        UndefinedGroup2,
    }
    let kind = |word| match decode_header(word) {
        Ok(MethodHeader::Methods { mode, .. }) => Kind::Methods(mode),
        Ok(MethodHeader::Immediate { .. }) => Kind::Immediate,
        Ok(MethodHeader::Legacy { mode, .. }) => Kind::Legacy(mode),
        Ok(MethodHeader::SubdeviceMask(_)) => Kind::Subdevice,
        Ok(MethodHeader::EndSegment) => Kind::End,
        Err(DecodeError::ReservedOpcode { .. }) => Kind::Reserved,
        Err(DecodeError::UndefinedSubdeviceOp { .. }) => Kind::UndefinedSubdevice,
        Err(DecodeError::UndefinedGroup2Op { .. }) => Kind::UndefinedGroup2,
    };
    for op in 0..=SEC_OP.max() {
        for tert in 0..=TERT_OP.max() {
            let word = SEC_OP.insert(0, op).and_then(|w| TERT_OP.insert(w, tert));
            let word = word.unwrap();
            let expected = match (op, tert) {
                (SEC_OP_GRP0_USE_TERT, TERT_OP_GRP0_INC_METHOD) => {
                    Kind::Legacy(MethodMode::Incrementing)
                }
                (SEC_OP_GRP0_USE_TERT, _) => Kind::Subdevice,
                (SEC_OP_INC_METHOD, _) => Kind::Methods(MethodMode::Incrementing),
                (SEC_OP_GRP2_USE_TERT, TERT_OP_GRP2_NON_INC_METHOD) => {
                    Kind::Legacy(MethodMode::NonIncrementing)
                }
                (SEC_OP_GRP2_USE_TERT, _) => Kind::UndefinedGroup2,
                (SEC_OP_NON_INC_METHOD, _) => Kind::Methods(MethodMode::NonIncrementing),
                (SEC_OP_IMMD_DATA_METHOD, _) => Kind::Immediate,
                (SEC_OP_ONE_INC, _) => Kind::Methods(MethodMode::IncrementOnce),
                (SEC_OP_RESERVED6, _) => Kind::Reserved,
                (SEC_OP_END_PB_SEGMENT, _) => Kind::End,
                _ => unreachable!("a 3-bit opcode"),
            };
            assert_eq!(kind(word), expected, "opcode {op} tertiary {tert}");
        }
    }
    // The subdevice operation each tertiary names, and one undefined 16-bit opcode.
    assert_eq!(
        decode_header(SUBDEVICE_OPCODE.insert(0, SUBDEVICE_OPCODE_USE).unwrap() | (1 << 18)),
        Err(DecodeError::UndefinedSubdeviceOp { word: 0x0007_0000 })
    );
}

// ---------------------------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------------------------

#[test]
fn encoder_matches_the_hand_laid_words() {
    let cases = [
        (
            MethodHeader::Methods {
                mode: MethodMode::Incrementing,
                subchannel: 3,
                method: 0x123,
                count: 5,
            },
            0x2005_6123,
        ),
        (
            MethodHeader::Methods {
                mode: MethodMode::NonIncrementing,
                subchannel: 0,
                method: 0xfff,
                count: 0x1fff,
            },
            0x7fff_0fff,
        ),
        (
            MethodHeader::Methods {
                mode: MethodMode::IncrementOnce,
                subchannel: 7,
                method: 0x40,
                count: 2,
            },
            0xa002_e040,
        ),
        (
            MethodHeader::Immediate {
                subchannel: 7,
                method: 0xfff,
                data: 0x1abc,
            },
            0x9abc_efff,
        ),
        (
            MethodHeader::Legacy {
                mode: MethodMode::Incrementing,
                subchannel: 5,
                method: 0x7ff,
                count: 0x7ff,
            },
            0x1ffc_bffc,
        ),
        (
            MethodHeader::Legacy {
                mode: MethodMode::NonIncrementing,
                subchannel: 1,
                method: 2,
                count: 1,
            },
            0x4004_2008,
        ),
        (
            MethodHeader::SubdeviceMask(SubdeviceMask::Set { mask: 0xab }),
            0x0001_0ab0,
        ),
        (
            MethodHeader::SubdeviceMask(SubdeviceMask::Store { mask: 0xfff }),
            0x0002_fff0,
        ),
        (MethodHeader::SubdeviceMask(SubdeviceMask::Use), 0x0003_0000),
        (MethodHeader::EndSegment, 0xe000_0000),
    ];
    for (header, word) in cases {
        assert_eq!(encode_header(&header), Ok(word), "{header:?}");
        assert_eq!(decode_header(word), Ok(header));
    }
}

#[test]
fn encoder_matches_mesa_nv_push() {
    // NVC0_FIFO_PKHDR_SQ(subc 3, mthd 0x48c, size 5): Mesa passes byte address 0x48c.
    let mut push = PushBuffer::new();
    push.methods(MethodMode::Incrementing, 3, 0x48c >> 2, &[1, 2, 3, 4, 5])
        .unwrap();
    assert_eq!(push.words(), [0x2005_6123, 1, 2, 3, 4, 5]);
    // NVC0_FIFO_PKHDR_IL(subc 1, mthd 0x100, data 0x1fff).
    let mut push = PushBuffer::new();
    push.immediate(1, 0x100 >> 2, 0x1fff).unwrap();
    assert_eq!(push.words(), [0x9fff_2040]);
    // NVC0_FIFO_PKHDR_1I and _0I with one data word each.
    let mut push = PushBuffer::new();
    push.methods(MethodMode::IncrementOnce, 2, 0x10, &[7])
        .unwrap()
        .methods(MethodMode::NonIncrementing, 2, 0x10, &[8])
        .unwrap();
    assert_eq!(push.words(), [0xa001_4010, 7, 0x6001_4010, 8]);
}

#[test]
fn push_buffer_streams_decode_back() {
    let mut push = PushBuffer::new();
    push.methods(MethodMode::Incrementing, 0, 0x45, &[0xaaaa, 0xbbbb])
        .unwrap()
        .immediate(1, 0x80, 3)
        .unwrap()
        .header(&MethodHeader::Methods {
            mode: MethodMode::NonIncrementing,
            subchannel: 2,
            method: 0x6d,
            count: 3,
        })
        .unwrap()
        .data(&[1, 2])
        .header(&MethodHeader::EndSegment)
        .unwrap();
    let words = push.clone().into_words();
    assert_eq!(words, push.words());
    assert_eq!(words.len(), 8);
    assert_eq!(
        decode_header(words[0]),
        Ok(MethodHeader::Methods {
            mode: MethodMode::Incrementing,
            subchannel: 0,
            method: 0x45,
            count: 2,
        })
    );
    assert_eq!(words[1..3], [0xaaaa, 0xbbbb]);
    assert_eq!(
        decode_header(words[3]),
        Ok(MethodHeader::Immediate {
            subchannel: 1,
            method: 0x80,
            data: 3,
        })
    );
    assert_eq!(
        decode_header(words[4]),
        Ok(MethodHeader::Methods {
            mode: MethodMode::NonIncrementing,
            subchannel: 2,
            method: 0x6d,
            count: 3,
        })
    );
    assert_eq!(words[5..7], [1, 2]);
    assert_eq!(decode_header(words[7]), Ok(MethodHeader::EndSegment));
}

#[test]
fn push_buffer_takes_the_largest_count_and_rejects_one_more() {
    let max = usize::try_from(METHOD_COUNT.max()).unwrap();
    let data = vec![0x5a5a_5a5a; max + 1];
    let mut push = PushBuffer::new();
    push.methods(MethodMode::NonIncrementing, 0, 0x10, &data[..max])
        .unwrap();
    assert_eq!(push.words().len(), max + 1);
    assert_eq!(
        decode_header(push.words()[0]),
        Ok(MethodHeader::Methods {
            mode: MethodMode::NonIncrementing,
            subchannel: 0,
            method: 0x10,
            count: u16::try_from(max).unwrap(),
        })
    );

    let mut over = PushBuffer::new();
    assert_eq!(
        over.methods(MethodMode::NonIncrementing, 0, 0x10, &data)
            .map(|_| ()),
        Err(EncodeError::FieldOverflow {
            field: "method count",
            value: u64::try_from(max + 1).unwrap(),
            bits: METHOD_COUNT.width(),
        })
    );
    assert!(over.words().is_empty(), "a failed call appends nothing");
    // A count too large even for u16.
    let huge = vec![0; usize::from(u16::MAX) + 1];
    assert_eq!(
        over.methods(MethodMode::Incrementing, 0, 0, &huge)
            .map(|_| ())
            .map_err(|e| e.to_string()),
        Err("method count 0x10000 does not fit in 13 bits".to_owned())
    );
    assert_eq!(over.words(), []);
}

#[test]
fn encoder_rejects_each_field_one_past_its_limit() {
    let over = |field: Field| u16::try_from(field.max() + 1).unwrap();
    let err = |field: &'static str, f: Field| EncodeError::FieldOverflow {
        field,
        value: u64::from(f.max() + 1),
        bits: f.width(),
    };
    let methods = |subchannel, method, count| MethodHeader::Methods {
        mode: MethodMode::Incrementing,
        subchannel,
        method,
        count,
    };
    let max_sub = u8::try_from(METHOD_SUBCHANNEL.max()).unwrap();
    let max_method = u16::try_from(METHOD_ADDRESS.max()).unwrap();
    let max_count = u16::try_from(METHOD_COUNT.max()).unwrap();
    assert!(encode_header(&methods(max_sub, max_method, max_count)).is_ok());
    assert_eq!(
        encode_header(&methods(max_sub + 1, 0, 0)),
        Err(err("subchannel", METHOD_SUBCHANNEL))
    );
    assert_eq!(
        encode_header(&methods(0, over(METHOD_ADDRESS), 0)),
        Err(err("method", METHOD_ADDRESS))
    );
    assert_eq!(
        encode_header(&methods(0, 0, over(METHOD_COUNT))),
        Err(err("method count", METHOD_COUNT))
    );

    let immediate = |data| MethodHeader::Immediate {
        subchannel: 0,
        method: 0,
        data,
    };
    assert!(encode_header(&immediate(u16::try_from(IMMD_DATA.max()).unwrap())).is_ok());
    assert_eq!(
        encode_header(&immediate(over(IMMD_DATA))),
        Err(err("immediate data", IMMD_DATA))
    );
    let mut push = PushBuffer::new();
    assert_eq!(
        push.immediate(0, 0, over(IMMD_DATA)).map(|_| ()),
        Err(err("immediate data", IMMD_DATA))
    );
    assert_eq!(push.words(), []);

    let legacy = |method, count| MethodHeader::Legacy {
        mode: MethodMode::NonIncrementing,
        subchannel: 0,
        method,
        count,
    };
    let max_old_method = u16::try_from(METHOD_ADDRESS_OLD.max()).unwrap();
    let max_old_count = u16::try_from(METHOD_COUNT_OLD.max()).unwrap();
    assert!(encode_header(&legacy(max_old_method, max_old_count)).is_ok());
    assert_eq!(
        encode_header(&legacy(over(METHOD_ADDRESS_OLD), 0)),
        Err(err("old-format method", METHOD_ADDRESS_OLD))
    );
    assert_eq!(
        encode_header(&legacy(0, over(METHOD_COUNT_OLD))),
        Err(err("old-format method count", METHOD_COUNT_OLD))
    );

    let max_mask = u16::try_from(SUBDEVICE_MASK.max()).unwrap();
    let ops: [fn(u16) -> SubdeviceMask; 2] = [
        |mask| SubdeviceMask::Set { mask },
        |mask| SubdeviceMask::Store { mask },
    ];
    for op in ops {
        assert!(encode_header(&MethodHeader::SubdeviceMask(op(max_mask))).is_ok());
        assert_eq!(
            encode_header(&MethodHeader::SubdeviceMask(op(over(SUBDEVICE_MASK)))),
            Err(err("subdevice mask", SUBDEVICE_MASK))
        );
    }
}

#[test]
fn old_format_has_no_increment_once() {
    let header = MethodHeader::Legacy {
        mode: MethodMode::IncrementOnce,
        subchannel: 0,
        method: 0,
        count: 0,
    };
    assert_eq!(
        encode_header(&header),
        Err(EncodeError::LegacyIncrementOnce)
    );
    assert_eq!(
        EncodeError::LegacyIncrementOnce.to_string(),
        "an old-format method header has no increment-once mode"
    );
}

// ---------------------------------------------------------------------------------------------
// Generated
// ---------------------------------------------------------------------------------------------

fn field_value(field: Field) -> impl Strategy<Value = u32> {
    0..=field.max()
}

fn any_mode() -> impl Strategy<Value = MethodMode> {
    prop_oneof![
        Just(MethodMode::Incrementing),
        Just(MethodMode::NonIncrementing),
        Just(MethodMode::IncrementOnce),
    ]
}

fn to_u8(v: u32) -> u8 {
    u8::try_from(v).unwrap()
}

fn to_u16(v: u32) -> u16 {
    u16::try_from(v).unwrap()
}

fn any_header() -> impl Strategy<Value = MethodHeader> {
    prop_oneof![
        (
            any_mode(),
            field_value(METHOD_SUBCHANNEL),
            field_value(METHOD_ADDRESS),
            field_value(METHOD_COUNT)
        )
            .prop_map(|(mode, s, m, c)| MethodHeader::Methods {
                mode,
                subchannel: to_u8(s),
                method: to_u16(m),
                count: to_u16(c),
            }),
        (
            field_value(METHOD_SUBCHANNEL),
            field_value(METHOD_ADDRESS),
            field_value(IMMD_DATA)
        )
            .prop_map(|(s, m, d)| MethodHeader::Immediate {
                subchannel: to_u8(s),
                method: to_u16(m),
                data: to_u16(d),
            }),
        (
            prop_oneof![
                Just(MethodMode::Incrementing),
                Just(MethodMode::NonIncrementing)
            ],
            field_value(METHOD_SUBCHANNEL),
            field_value(METHOD_ADDRESS_OLD),
            field_value(METHOD_COUNT_OLD)
        )
            .prop_map(|(mode, s, m, c)| MethodHeader::Legacy {
                mode,
                subchannel: to_u8(s),
                method: to_u16(m),
                count: to_u16(c),
            }),
        field_value(SUBDEVICE_MASK)
            .prop_map(|m| MethodHeader::SubdeviceMask(SubdeviceMask::Set { mask: to_u16(m) })),
        field_value(SUBDEVICE_MASK)
            .prop_map(|m| MethodHeader::SubdeviceMask(SubdeviceMask::Store { mask: to_u16(m) })),
        Just(MethodHeader::SubdeviceMask(SubdeviceMask::Use)),
        Just(MethodHeader::EndSegment),
    ]
}

fn any_entry() -> impl Strategy<Value = GpEntry> {
    (
        field_value(GP_ENTRY0_GET),
        field_value(GP_ENTRY1_GET_HI),
        field_value(GP_ENTRY1_LENGTH),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(get, get_hi, length, sub, cond, wait)| GpEntry {
            address: (u64::from(get_hi) << GP_ENTRY_GET_HI_SHIFT)
                | (u64::from(get) << GP_ENTRY0_GET.low()),
            length,
            level: if sub { Level::Subroutine } else { Level::Main },
            fetch: if cond {
                Fetch::Conditional
            } else {
                Fetch::Unconditional
            },
            sync: if wait { Sync::Wait } else { Sync::Proceed },
        })
}

proptest! {
    #![proptest_config(config())]

    /// Any word decodes or errors, never panics; a decoded header re-encodes to a word that
    /// decodes to the same header.
    #[test]
    fn any_word_decodes_or_errors(word in any::<u32>()) {
        if let Ok(header) = decode_header(word) {
            let canonical = encode_header(&header).unwrap();
            prop_assert_eq!(decode_header(canonical), Ok(header));
            // The canonical word sets no bit the original did not.
            prop_assert_eq!(canonical & !word, 0);
        }
    }

    /// Build-then-parse: every field survives, so two swapped reads would fail.
    #[test]
    fn generated_headers_round_trip(header in any_header()) {
        let word = encode_header(&header).unwrap();
        prop_assert_eq!(decode_header(word), Ok(header));
    }

    /// Any 64-bit value is an entry; it re-encodes to a value that decodes the same.
    #[test]
    fn any_raw_entry_decodes(raw in any::<u64>()) {
        let entry = GpEntry::from_raw(raw);
        let canonical = entry.to_raw().unwrap();
        prop_assert_eq!(GpEntry::from_raw(canonical), entry);
        prop_assert_eq!(canonical & !raw, 0);
    }

    /// Build-then-parse for entries.
    #[test]
    fn generated_entries_round_trip(entry in any_entry()) {
        let words = entry.to_words().unwrap();
        prop_assert_eq!(GpEntry::from_words(words), entry);
        prop_assert_eq!(GpEntry::from_raw(entry.to_raw().unwrap()), entry);
    }
}

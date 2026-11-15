use super::*;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

/// The file name the fixtures use: class prefix `NVC797_`.
const FILE: &str = "clc797.h";

/// A licence comment in the ` * ` style (`cla140.h`, `clc797tex.h`).
const NOTICE: &str = "/*\n\
     * Copyright (c) 2020, Test Holder. All rights reserved.\n\
     *\n\
     * Permission is hereby granted, free of charge, to any person obtaining a copy.\n\
     */\n";

/// The notice above, as the reader cleans it.
fn notice() -> Vec<String> {
    vec![
        "Copyright (c) 2020, Test Holder. All rights reserved.".to_owned(),
        String::new(),
        "Permission is hereby granted, free of charge, to any person obtaining a copy.".to_owned(),
    ]
}

/// The line number of line `n` (from 1) of a body placed after [`NOTICE`].
fn body_line(n: usize) -> usize {
    NOTICE.lines().count() + n
}

fn read(body: &str) -> Result<Header, ParseError> {
    parse(FILE, &format!("{NOTICE}{body}"), None)
}

fn read_ok(body: &str) -> Header {
    read(body).unwrap()
}

fn header() -> Header {
    Header {
        prefix: "NVC797_".to_owned(),
        notice: notice(),
        class: None,
        methods: Vec::new(),
        fields: Vec::new(),
        constants: Vec::new(),
    }
}

fn value(name: &str, value: u32) -> Value {
    Value {
        name: name.to_owned(),
        value,
    }
}

fn word(name: &str, high: u32, low: u32, values: Vec<Value>) -> Field {
    Field {
        name: name.to_owned(),
        bits: Bits::Word { high, low },
        values,
    }
}

fn method(name: &str, byte_address: u32, strides: Vec<u32>, fields: Vec<Field>) -> Method {
    Method {
        name: name.to_owned(),
        byte_address,
        strides,
        fields,
    }
}

/// The error the reader gives for `body`, as text.
fn error(body: &str) -> String {
    read(body).unwrap_err().to_string()
}

// --- Accepting cases, one per format feature, built by hand. ---

#[test]
fn class_prefix_is_nv_and_the_four_hex_digits_of_the_file_name() {
    assert_eq!(class_prefix("clc797.h").as_deref(), Some("NVC797_"));
    assert_eq!(class_prefix("clc797tex.h").as_deref(), Some("NVC797_"));
    assert_eq!(class_prefix("clc7c0qmd.h").as_deref(), Some("NVC7C0_"));
    assert_eq!(class_prefix("cl902d.h").as_deref(), Some("NV902D_"));
    assert_eq!(class_prefix("cla140.h").as_deref(), Some("NVA140_"));
    assert_eq!(class_prefix("clzz00.h"), None);
    assert_eq!(class_prefix("c797.h"), None);
    assert_eq!(class_prefix("cl79.h"), None);
}

#[test]
fn the_star_box_notice_style_is_cleaned_too() {
    // `clc797.h` and `clc76f.h` frame the notice in a box of stars with unprefixed lines.
    let text = "/*******************************************************************************\n\
        \x20   Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.\n\
        \n\
        \x20   Permission is hereby granted, free of charge, to any person obtaining a\n\
        \n\
        *******************************************************************************/\n\
        #define NVC797_NOP 0x0100\n\
        #define NVC797_NOP_V 31:0\n";
    let parsed = parse(FILE, text, None).unwrap();
    assert_eq!(
        parsed.notice,
        [
            "Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.",
            "",
            "Permission is hereby granted, free of charge, to any person obtaining a",
        ]
    );
}

#[test]
fn directives_guards_typedefs_and_c_code_are_skipped_and_the_class_id_is_read() {
    let parsed = read_ok(
        "#ifndef _cl_test_h_\n\
         #define _cl_test_h_\n\
         #ifdef __cplusplus\n\
         extern \"C\" {\n\
         #endif\n\
         #include \"nvtypes.h\"\n\
         #define  AMPERE_B    (0x0000C797)\n\
         #define NVC797_TYPEDEF AMPERE_BChannel\n\
         typedef volatile struct { NvU32 Put; } Control;\n\
         #endif\n",
    );
    assert_eq!(
        parsed,
        Header {
            class: Some(ClassId {
                name: "AMPERE_B".to_owned(),
                id: 0xC797
            }),
            ..header()
        }
    );
}

#[test]
fn a_method_its_fields_and_their_values_are_grouped() {
    let parsed = read_ok(
        "#define NVC797_SET_OBJECT 0x0000\n\
         #define NVC797_SET_OBJECT_CLASS_ID 15:0\n\
         #define NVC797_SET_OBJECT_ENGINE_ID 20:16\n\
         #define NVC797_SET_OBJECT_ENGINE_ID_SW 0x0000001f\n\
         \n\
         #define NVC797_NOTIFY (0x0000010c) // a trailing comment\n\
         #define NVC797_NOTIFY_TYPE 31:0 /* a block comment */\n\
         #define NVC797_NOTIFY_TYPE_WRITE_ONLY 0\n\
         #define NVC797_NOTIFY_TYPE_WRITE_THEN_AWAKEN (0x00000001)\n",
    );
    assert_eq!(
        parsed.methods,
        [
            method(
                "SET_OBJECT",
                0,
                vec![],
                vec![
                    word("CLASS_ID", 15, 0, vec![]),
                    word("ENGINE_ID", 20, 16, vec![value("SW", 0x1f)]),
                ]
            ),
            method(
                "NOTIFY",
                0x10c,
                vec![],
                vec![word(
                    "TYPE",
                    31,
                    0,
                    vec![value("WRITE_ONLY", 0), value("WRITE_THEN_AWAKEN", 1)]
                )]
            ),
        ]
    );
    assert!(parsed.fields.is_empty() && parsed.constants.is_empty());
}

#[test]
fn array_methods_with_one_or_two_indices_keep_their_strides_in_parameter_order() {
    let parsed = read_ok(
        "#define NVC797_SET_SCALE_X(j) (0x0a00+(j)*32)\n\
         #define NVC797_SET_SCALE_X_V 31:0\n\
         #define NVC797_LOAD_DATA(i) (0x0320+(i)*4)\n\
         #define NVC797_LOAD_DATA_V 31:0\n\
         #define NVC797_SET_LAYOUT_SELECT(i,j) (0x2800+(i)*128+(j)*4)\n\
         #define NVC797_SET_LAYOUT_SELECT_ATTRIBUTE_NUMBER00 7:0\n",
    );
    assert_eq!(
        parsed.methods,
        [
            method("SET_SCALE_X", 0xa00, vec![32], vec![word("V", 31, 0, vec![])]),
            method("LOAD_DATA", 0x320, vec![4], vec![word("V", 31, 0, vec![])]),
            method(
                "SET_LAYOUT_SELECT",
                0x2800,
                vec![128, 4],
                vec![word("ATTRIBUTE_NUMBER00", 7, 0, vec![])]
            ),
        ]
    );
}

#[test]
fn a_number_that_no_field_follows_is_a_constant() {
    let parsed = read_ok(
        "#define NVC797_NUMBER_OF_SUBCHANNELS (8)\n\
         #define NVC797_SET_OBJECT (0x00000000)\n\
         #define NVC797_SET_OBJECT_NVCLASS 15:0\n\
         #define NVC797_GP_ENTRY__SIZE 8\n\
         #define NVC797_GP_ENTRY0_FETCH 0:0\n\
         #define NVC797_DMA_NOP (0x00000000)\n",
    );
    assert_eq!(
        parsed.constants,
        [
            value("NUMBER_OF_SUBCHANNELS", 8),
            value("GP_ENTRY__SIZE", 8),
            value("DMA_NOP", 0)
        ]
    );
    assert_eq!(parsed.methods.len(), 1);
    assert_eq!(parsed.fields, [word("GP_ENTRY0_FETCH", 0, 0, vec![])]);
}

#[test]
fn a_held_back_number_before_a_multi_word_field_is_a_constant() {
    // Only a single-word range makes a method of the number before it.
    let parsed = read_ok(
        "#define NVC797_QMD 0x0100\n\
         #define NVC797_QMD_GROUP MW(5:0)\n",
    );
    assert_eq!(parsed.constants, [value("QMD", 0x100)]);
    assert!(parsed.methods.is_empty());
    assert_eq!(
        parsed.fields,
        [Field {
            name: "QMD_GROUP".to_owned(),
            bits: Bits::Multi { high: 5, low: 0 },
            values: vec![],
        }]
    );
}

#[test]
fn fields_of_no_method_take_values_and_end_the_method() {
    let parsed = read_ok(
        "#define NVC797_WFI 0x0078\n\
         #define NVC797_WFI_SCOPE 0:0\n\
         #define NVC797_GP_ENTRY1_LEVEL 9:9\n\
         #define NVC797_GP_ENTRY1_LEVEL_MAIN 0x00000000\n\
         #define NVC797_GP_ENTRY1_LEVEL_SUBROUTINE 0x00000001\n\
         #define NVC797_WFI_SCOPE_ALL 0x00000001\n",
    );
    // The last define extends WFI_SCOPE's name, but the free field ended WFI: a constant.
    assert_eq!(
        parsed.methods,
        [method("WFI", 0x78, vec![], vec![word("SCOPE", 0, 0, vec![])])]
    );
    assert_eq!(
        parsed.fields,
        [word(
            "GP_ENTRY1_LEVEL",
            9,
            9,
            vec![value("MAIN", 0), value("SUBROUTINE", 1)]
        )]
    );
    assert_eq!(parsed.constants, [value("WFI_SCOPE_ALL", 1)]);
}

#[test]
fn multi_word_fields_and_arrays_of_them_are_read() {
    let parsed = read_ok(
        "#define NVC797_TEXHEAD_BL_COMPONENTS MW(6:0)\n\
         #define NVC797_TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8 0x00000008\n\
         #define NVC797_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(i) MW((1055+(i)*64):(1024+(i)*64))\n\
         #define NVC797_QMDV03_00_CONSTANT_BUFFER_VALID(i) MW((640+(i)*1):(640+(i)*1))\n\
         #define NVC797_QMDV03_00_CONSTANT_BUFFER_VALID_TRUE 0x00000001\n\
         #define NVC797_TEXSAMP0_ADDRESS_U 2:0\n",
    );
    assert_eq!(
        parsed.fields,
        [
            Field {
                name: "TEXHEAD_BL_COMPONENTS".to_owned(),
                bits: Bits::Multi { high: 6, low: 0 },
                values: vec![value("SIZES_A8B8G8R8", 8)],
            },
            Field {
                name: "QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER".to_owned(),
                bits: Bits::MultiArray {
                    high: 1055,
                    low: 1024,
                    stride: 64
                },
                values: vec![],
            },
            Field {
                name: "QMDV03_00_CONSTANT_BUFFER_VALID".to_owned(),
                bits: Bits::MultiArray {
                    high: 640,
                    low: 640,
                    stride: 1
                },
                values: vec![value("TRUE", 1)],
            },
            word("TEXSAMP0_ADDRESS_U", 2, 0, vec![]),
        ]
    );
}

#[test]
fn a_stray_define_inside_a_method_becomes_a_constant_and_the_method_goes_on() {
    // The shape of `clc76f.h`'s MEM_OP_D: a MEM_OP_B value among MEM_OP_D's values. Mesa's
    // parser loses everything after it; this reader keeps it (a listed divergence).
    let parsed = read_ok(
        "#define NVC797_MEM_OP_D (0x00000034)\n\
         #define NVC797_MEM_OP_D_OPERATION 31:27\n\
         #define NVC797_MEM_OP_D_OPERATION_MEMBAR 0x00000005\n\
         #define NVC797_MEM_OP_B_OPERATION_L2_INVALIDATE_CLEAN_LINES 0x0000000e\n\
         #define NVC797_MEM_OP_D_OPERATION_L2_CLEAN_COMPTAGS 0x0000000f\n\
         #define NVC797_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE 1:0\n\
         #define NVC797_MEM_OP_D_ACCESS_COUNTER_CLR_TYPE_MIMC 0x00000000\n",
    );
    assert_eq!(
        parsed.methods,
        [method(
            "MEM_OP_D",
            0x34,
            vec![],
            vec![
                word(
                    "OPERATION",
                    31,
                    27,
                    vec![value("MEMBAR", 5), value("L2_CLEAN_COMPTAGS", 0xf)]
                ),
                word("ACCESS_COUNTER_CLR_TYPE", 1, 0, vec![value("MIMC", 0)]),
            ]
        )]
    );
    assert_eq!(
        parsed.constants,
        [value("MEM_OP_B_OPERATION_L2_INVALIDATE_CLEAN_LINES", 0xe)]
    );
}

#[test]
fn a_value_goes_to_the_longest_field_of_the_method_it_extends() {
    // After PDB_ADDR_LO, a PDB_APERTURE value goes back to PDB_APERTURE, not to PDB.
    let parsed = read_ok(
        "#define NVC797_MEM_OP_C 0x0030\n\
         #define NVC797_MEM_OP_C_PDB 0:0\n\
         #define NVC797_MEM_OP_C_PDB_APERTURE 11:10\n\
         #define NVC797_MEM_OP_C_PDB_ADDR_LO 31:12\n\
         #define NVC797_MEM_OP_C_PDB_APERTURE_SYS_MEM 0x00000002\n\
         #define NVC797_MEM_OP_C_PDB_ONE 0x00000000\n",
    );
    assert_eq!(
        parsed.methods[0].fields,
        [
            word("PDB", 0, 0, vec![value("ONE", 0)]),
            word("PDB_APERTURE", 11, 10, vec![value("SYS_MEM", 2)]),
            word("PDB_ADDR_LO", 31, 12, vec![]),
        ]
    );
}

#[test]
fn every_value_is_kept_even_two_names_for_one_number() {
    // `clc76f.h`, WFI_SCOPE; Mesa skips CURRENT_SCG_TYPE (a listed divergence).
    let parsed = read_ok(
        "#define NVC797_WFI 0x0078\n\
         #define NVC797_WFI_SCOPE 0:0\n\
         #define NVC797_WFI_SCOPE_CURRENT_SCG_TYPE 0x00000000\n\
         #define NVC797_WFI_SCOPE_CURRENT_VEID 0x00000000\n\
         #define NVC797_WFI_SCOPE_ALL 0x00000001\n",
    );
    assert_eq!(
        parsed.methods[0].fields[0].values,
        [
            value("CURRENT_SCG_TYPE", 0),
            value("CURRENT_VEID", 0),
            value("ALL", 1)
        ]
    );
}

#[test]
fn a_repeated_identical_define_is_read_once() {
    let parsed = read_ok(
        "#define NVC797_DMA_IMMD_DATA 28:16\n\
         #define NVC797_DMA_IMMD_ADDRESS 11:0\n\
         #define NVC797_DMA_IMMD_DATA 28:16\n",
    );
    assert_eq!(
        parsed.fields,
        [
            word("DMA_IMMD_DATA", 28, 16, vec![]),
            word("DMA_IMMD_ADDRESS", 11, 0, vec![])
        ]
    );
}

#[test]
fn lower_case_letters_after_the_first_are_accepted() {
    let parsed = read_ok(
        "#define NVC797_SET_PATTERN_SELECT 0x08e8\n\
         #define NVC797_SET_PATTERN_SELECT_V 1:0\n\
         #define NVC797_SET_PATTERN_SELECT_V_MONOCHROME_8x8 0x00000000\n",
    );
    assert_eq!(
        parsed.methods[0].fields[0].values,
        [value("MONOCHROME_8x8", 0)]
    );
}

#[test]
fn only_keeps_the_defines_with_the_given_prefix() {
    let text = format!(
        "{NOTICE}#define NVC797_QMDV02_03_OUTER_PUT MW(30:0)\n\
         #define NVC797_QMDV03_00_OUTER_PUT MW(30:0)\n\
         #define NVC797_QMDV03_00_IS_QUEUE MW(136:136)\n\
         #define NVC797_QMDV03_00_IS_QUEUE_TRUE 0x00000001\n\
         #define NVC797_QMDV02_03_IS_QUEUE_TRUE 0x00000001\n"
    );
    let parsed = parse(FILE, &text, Some("QMDV03_00_")).unwrap();
    assert_eq!(
        parsed.fields,
        [
            Field {
                name: "QMDV03_00_OUTER_PUT".to_owned(),
                bits: Bits::Multi { high: 30, low: 0 },
                values: vec![],
            },
            Field {
                name: "QMDV03_00_IS_QUEUE".to_owned(),
                bits: Bits::Multi {
                    high: 136,
                    low: 136
                },
                values: vec![value("TRUE", 1)],
            },
        ]
    );
    // Without the filter, the 02_03 defines are read too.
    assert_eq!(parse(FILE, &text, None).unwrap().fields.len(), 3);
}

#[test]
fn a_comment_spanning_lines_hides_the_defines_inside_it() {
    let parsed = read_ok(
        "/* #define NVC797_HIDDEN 0x0004\n\
         #define NVC797_HIDDEN_V 31:0 */ #define NVC797_AFTER 0x0008\n\
         #define NVC797_SHOWN 0x000c\n\
         #define NVC797_SHOWN_V 31:0\n",
    );
    // As in C, a comment counts as a space: what follows `*/` starts the line, so it is a
    // directive.
    assert_eq!(
        parsed.methods,
        [method("SHOWN", 0xc, vec![], vec![word("V", 31, 0, vec![])])]
    );
    assert_eq!(parsed.constants, [value("AFTER", 8)]);
}

#[test]
fn bit_ranges_at_their_limits_are_accepted() {
    let parsed = read_ok(
        "#define NVC797_WIDE 31:0\n\
         #define NVC797_MW_WIDE MW(63:32)\n\
         #define NVC797_MW_SPAN MW(47:16)\n",
    );
    assert_eq!(parsed.fields[0].bits, Bits::Word { high: 31, low: 0 });
    assert_eq!(parsed.fields[1].bits, Bits::Multi { high: 63, low: 32 });
    assert_eq!(parsed.fields[2].bits, Bits::Multi { high: 47, low: 16 });
}

#[test]
fn numbers_at_the_32_bit_limit_are_accepted() {
    let parsed = read_ok(
        "#define NVC797_ALL_ONES 0xffffffff\n\
         #define NVC797_ALL_ONES_DECIMAL 4294967295\n",
    );
    assert_eq!(
        parsed.constants,
        [
            value("ALL_ONES", u32::MAX),
            value("ALL_ONES_DECIMAL", u32::MAX)
        ]
    );
}

// --- Errors: each names the file and the line, in plain words. ---

#[test]
fn a_file_name_without_a_class_number_is_refused() {
    assert_eq!(
        parse("nvtypes.h", NOTICE, None).unwrap_err().to_string(),
        "nvtypes.h:1: the file name does not start with cl and four hex digits"
    );
}

#[test]
fn a_file_without_the_mit_grant_is_refused() {
    let message = "clc797.h:1: the file does not start with a licence notice granting the MIT \
                   permissions";
    assert_eq!(
        parse(FILE, "#define NVC797_NOP 0x0100\n", None)
            .unwrap_err()
            .to_string(),
        message
    );
    assert_eq!(
        parse(FILE, "/* Copyright (c) Someone. */\n", None)
            .unwrap_err()
            .to_string(),
        message
    );
}

#[test]
fn an_unclosed_comment_names_the_line_it_opens_on() {
    assert_eq!(
        error("#define NVC797_NOP 0x0100\n/* never closed\n#define NVC797_NOP_V 31:0\n"),
        format!(
            "clc797.h:{}: the comment opened here is never closed",
            body_line(2)
        )
    );
}

#[test]
fn an_unsupported_directive_is_refused() {
    assert_eq!(
        error("#if 0\n"),
        format!("clc797.h:{}: unsupported directive #if", body_line(1))
    );
    assert_eq!(
        error("#define NVC797_A 0x0\n#undef NVC797_A\n"),
        format!("clc797.h:{}: unsupported directive #undef", body_line(2))
    );
}

#[test]
fn a_define_without_a_name_is_refused() {
    assert_eq!(
        error("#define\n"),
        format!("clc797.h:{}: #define without a name", body_line(1))
    );
}

#[test]
fn a_name_the_reader_does_not_understand_is_refused() {
    for name in ["set_object", "_SET_OBJECT", "SET_OBJECT(k)", "SET_OBJECT(i", "SET-OBJECT"] {
        assert_eq!(
            error(&format!("#define NVC797_{name} 0x0\n")),
            format!(
                "clc797.h:{}: {name} is not a name this reader understands",
                body_line(1)
            )
        );
    }
}

#[test]
fn a_prefixed_define_without_a_value_is_refused() {
    assert_eq!(
        error("#define NVC797_EMPTY\n"),
        format!("clc797.h:{}: NVC797_EMPTY has no value", body_line(1))
    );
}

#[test]
fn a_define_with_two_values_is_refused() {
    assert_eq!(
        error("#define NVC797_NOP 0x0100 0x0104\n"),
        format!(
            "clc797.h:{}: NVC797_NOP has more than one value: 0x0100 0x0104",
            body_line(1)
        )
    );
}

#[test]
fn a_value_of_an_unknown_form_is_refused() {
    for text in ["1+2", "0x10:0", "MW(6-0)", "MW(abc:0)", "(0x10+(i)*4", "0x0g", "(0x10+(k)*4)"]
    {
        assert_eq!(
            error(&format!("#define NVC797_ODD {text}\n")),
            format!(
                "clc797.h:{}: {text} is not a number, a bit range or an array address",
                body_line(1)
            ),
            "{text}"
        );
    }
}

#[test]
fn an_unprefixed_define_with_a_word_value_is_refused() {
    assert_eq!(
        error("#define OTHER_THING SomeType\n"),
        format!(
            "clc797.h:{}: OTHER_THING is outside the class prefix and is not a class id",
            body_line(1)
        )
    );
}

#[test]
fn a_second_class_id_is_refused() {
    assert_eq!(
        error("#define AMPERE_B 0xC797\n#define AMPERE_C 0xC997\n"),
        format!(
            "clc797.h:{}: AMPERE_C is a second class id (the first is AMPERE_B)",
            body_line(2)
        )
    );
}

#[test]
fn a_number_past_32_bits_is_refused() {
    for text in ["0x100000000", "4294967296"] {
        assert_eq!(
            error(&format!("#define NVC797_BIG {text}\n")),
            format!("clc797.h:{}: {text} does not fit in 32 bits", body_line(1))
        );
    }
    // A class id too.
    assert_eq!(
        error("#define AMPERE (0x100000000)\n"),
        format!("clc797.h:{}: 0x100000000 does not fit in 32 bits", body_line(1))
    );
    assert_eq!(
        error("#define NVC797_BIG_BIT 4294967296:0\n"),
        format!("clc797.h:{}: 4294967296 does not fit in 32 bits", body_line(1))
    );
    assert_eq!(
        error("#define NVC797_BIG_ARRAY(i) (0x100000000+(i)*4)\n"),
        format!("clc797.h:{}: 0x100000000 does not fit in 32 bits", body_line(1))
    );
    assert_eq!(
        error("#define NVC797_BIG_STRIDE(i) (0x10+(i)*4294967296)\n"),
        format!("clc797.h:{}: 4294967296 does not fit in 32 bits", body_line(1))
    );
}

#[test]
fn bit_ranges_past_their_limits_are_refused() {
    assert_eq!(
        error("#define NVC797_BACKWARDS 0:1\n"),
        format!("clc797.h:{}: bit range 0:1 is inverted", body_line(1))
    );
    assert_eq!(
        error("#define NVC797_MW_BACKWARDS MW(32:33)\n"),
        format!("clc797.h:{}: bit range 32:33 is inverted", body_line(1))
    );
    assert_eq!(
        error("#define NVC797_PAST 32:0\n"),
        format!(
            "clc797.h:{}: bit range 32:0 goes past bit 31 of a word",
            body_line(1)
        )
    );
    assert_eq!(
        error("#define NVC797_MW_WIDE MW(64:32)\n"),
        format!(
            "clc797.h:{}: bit range 64:32 is wider than 32 bits",
            body_line(1)
        )
    );
    assert_eq!(
        error("#define NVC797_MW_ARRAY_WIDE(i) MW((32+(i)*64):(0+(i)*64))\n"),
        format!(
            "clc797.h:{}: bit range 32:0 is wider than 32 bits",
            body_line(1)
        )
    );
}

#[test]
fn a_method_address_or_stride_that_is_not_a_whole_word_is_refused() {
    // A plain method is only known to be one when its first field arrives, but the error names
    // the method's own line.
    assert_eq!(
        error("#define NVC797_ODD 0x0102\n#define NVC797_ODD_V 31:0\n"),
        format!(
            "clc797.h:{}: ODD: byte address or stride 0x102 is not a multiple of 4",
            body_line(1)
        )
    );
    assert_eq!(
        error("#define NVC797_ODD(i) (0x0102+(i)*4)\n"),
        format!(
            "clc797.h:{}: ODD: byte address or stride 0x102 is not a multiple of 4",
            body_line(1)
        )
    );
    assert_eq!(
        error("#define NVC797_ODD(i,j) (0x0100+(i)*4+(j)*2)\n"),
        format!(
            "clc797.h:{}: ODD: byte address or stride 0x2 is not a multiple of 4",
            body_line(1)
        )
    );
    // A number no field follows is a constant, and need not be aligned.
    assert_eq!(read_ok("#define NVC797_ODD 0x0102\n").constants, [value("ODD", 0x102)]);
}

#[test]
fn indices_that_do_not_match_the_name_are_refused() {
    for (name, text) in [
        ("A(i)", "(0x0100+(j)*4)"),
        ("A(i,j)", "(0x0100+(j)*4+(i)*4)"),
        ("A(i,j)", "(0x0100+(i)*4)"),
        ("A", "(0x0100+(i)*4)"),
        ("A(i)", "0x0100"),
        ("A(i)", "31:0"),
        ("A(i)", "MW(31:0)"),
        ("A", "MW((31+(i)*32):(0+(i)*32))"),
        ("A(j)", "MW((31+(i)*32):(0+(i)*32))"),
        ("A(i)", "MW((31+(i)*32):(0+(j)*32))"),
        ("A(i)", "MW((31+(i)*32+(i)*4):(0+(i)*32))"),
        ("A(i)", "MW(31:(0+(i)*32))"),
    ] {
        assert_eq!(
            error(&format!("#define NVC797_{name} {text}\n")),
            format!(
                "clc797.h:{}: {name}: the indices of {text} do not match the name's parameters",
                body_line(1)
            ),
            "{name} {text}"
        );
    }
}

#[test]
fn a_multi_word_array_whose_ends_step_differently_is_refused() {
    assert_eq!(
        error("#define NVC797_A(i) MW((31+(i)*64):(0+(i)*32))\n"),
        format!(
            "clc797.h:{}: A(i): the high and low bits of the array step by different amounts",
            body_line(1)
        )
    );
}

#[test]
fn a_name_defined_again_differently_is_refused() {
    let message = |name: &str| {
        format!(
            "clc797.h:{}: {name} is defined again with another value or in another case \
             (first at line {})",
            body_line(2),
            body_line(1)
        )
    };
    assert_eq!(
        error("#define NVC797_DMA_DATA 28:16\n#define NVC797_DMA_DATA 28:18\n"),
        message("DMA_DATA")
    );
    assert_eq!(
        error("#define NVC797_MODE_8x8 0x0\n#define NVC797_MODE_8X8 0x0\n"),
        message("MODE_8X8")
    );
    // An array and a plain define of one name clash too.
    assert_eq!(
        error("#define NVC797_A(i) (0x0100+(i)*4)\n#define NVC797_A 0x0200\n"),
        message("A")
    );
}

#[test]
fn an_array_method_without_fields_is_refused() {
    assert_eq!(
        error("#define NVC797_A(i) (0x0100+(i)*4)\n#define NVC797_B 0x0200\n#define NVC797_B_V 31:0\n"),
        format!("clc797.h:{}: array method A has no fields", body_line(1))
    );
}

// --- Round trip: a generated header written as C text reads back as the same header. ---

/// A field of no method, as the generator below builds it.
#[derive(Debug, Clone)]
enum FreeShape {
    Word,
    Multi,
    Array,
}

/// Writes `header` as class header text, the way NVIDIA lays the headers out. Numbers are
/// written in hex or decimal by `decimal` (indexed by a running count).
fn render(header: &Header, decimal: &[bool], j_index: &[bool]) -> String {
    let mut text = String::from(NOTICE);
    let mut count = 0usize;
    let mut number = |value: u32| {
        count += 1;
        if decimal.get(count % decimal.len().max(1)).copied().unwrap_or(false) {
            format!("({value})")
        } else {
            format!("0x{value:08x}")
        }
    };
    text.push_str("#ifndef _cl_round_trip_h_\n#define _cl_round_trip_h_\n");
    if let Some(class) = &header.class {
        text.push_str(&format!("#define {} {}\n", class.name, number(class.id)));
    }
    let p = &header.prefix;
    for (index, method) in header.methods.iter().enumerate() {
        text.push('\n');
        match method.strides.as_slice() {
            [] => text.push_str(&format!(
                "#define {p}{} {}\n",
                method.name,
                number(method.byte_address)
            )),
            [stride] => {
                let var = if j_index.get(index).copied().unwrap_or(false) { 'j' } else { 'i' };
                text.push_str(&format!(
                    "#define {p}{}({var}) (0x{:04x}+({var})*{stride})\n",
                    method.name, method.byte_address
                ));
            }
            [first, second] => text.push_str(&format!(
                "#define {p}{}(i,j) (0x{:04x}+(i)*{first}+(j)*{second})\n",
                method.name, method.byte_address
            )),
            _ => unreachable!("at most two strides are generated"),
        }
        for field in &method.fields {
            let Bits::Word { high, low } = field.bits else {
                unreachable!("method fields are single-word")
            };
            text.push_str(&format!("#define {p}{}_{} {high}:{low}\n", method.name, field.name));
            for value in &field.values {
                text.push_str(&format!(
                    "#define {p}{}_{}_{} {}\n",
                    method.name,
                    field.name,
                    value.name,
                    number(value.value)
                ));
            }
        }
    }
    text.push('\n');
    for constant in &header.constants {
        text.push_str(&format!("#define {p}{} {}\n", constant.name, number(constant.value)));
    }
    for field in &header.fields {
        let (suffix, bits) = match field.bits {
            Bits::Word { high, low } => (String::new(), format!("{high}:{low}")),
            Bits::Multi { high, low } => (String::new(), format!("MW({high}:{low})")),
            Bits::MultiArray { high, low, stride } => (
                "(i)".to_owned(),
                format!("MW(({high}+(i)*{stride}):({low}+(i)*{stride}))"),
            ),
        };
        text.push_str(&format!("#define {p}{}{suffix} {bits}\n", field.name));
        for value in &field.values {
            text.push_str(&format!(
                "#define {p}{}_{} {}\n",
                field.name,
                value.name,
                number(value.value)
            ));
        }
    }
    text.push_str("#endif\n");
    text
}

fn config() -> Config {
    Config {
        cases: 256,
        rng_seed: RngSeed::Fixed(0x636c_6173),
        ..Config::default()
    }
}

/// A single-word range: `low <= high <= 31`.
fn word_range() -> impl Strategy<Value = (u32, u32)> {
    (0u32..32).prop_flat_map(|low| (low..32, Just(low)))
}

/// A multi-word range: `low <= high`, at most 32 bits wide.
fn multi_range() -> impl Strategy<Value = (u32, u32)> {
    (0u32..4096).prop_flat_map(|low| (low..low + 32, Just(low)))
}

fn values(owner: usize) -> impl Strategy<Value = Vec<Value>> {
    prop::collection::vec(any::<u32>(), 0..4).prop_map(move |numbers| {
        numbers
            .into_iter()
            .enumerate()
            .map(|(index, value)| Value {
                name: format!("V{owner}X{index}"),
                value,
            })
            .collect()
    })
}

fn method_fields(method: usize) -> impl Strategy<Value = Vec<Field>> {
    prop::collection::vec((word_range(), values(method)), 1..4).prop_map(|fields| {
        fields
            .into_iter()
            .enumerate()
            .map(|(index, ((high, low), values))| Field {
                name: format!("F{index}"),
                bits: Bits::Word { high, low },
                values,
            })
            .collect()
    })
}

fn methods() -> impl Strategy<Value = Vec<Method>> {
    prop::collection::vec(
        (
            0u32..0x4000,
            prop::collection::vec(1u32..0x400, 0..3),
        ),
        0..6,
    )
    .prop_flat_map(|shapes| {
        shapes
            .into_iter()
            .enumerate()
            .map(|(index, (word, strides))| {
                method_fields(index).prop_map(move |fields| Method {
                    name: format!("M{index}"),
                    byte_address: word * 4,
                    strides: strides.iter().map(|stride| stride * 4).collect(),
                    fields,
                })
            })
            .collect::<Vec<_>>()
    })
}

fn free_fields() -> impl Strategy<Value = Vec<Field>> {
    prop::collection::vec(
        (
            prop_oneof![
                Just(FreeShape::Word),
                Just(FreeShape::Multi),
                Just(FreeShape::Array)
            ],
            word_range(),
            multi_range(),
            1u32..256,
        ),
        0..4,
    )
    .prop_flat_map(|shapes| {
        shapes
            .into_iter()
            .enumerate()
            .map(|(index, (shape, word, multi, stride))| {
                values(100 + index).prop_map(move |values| Field {
                    name: format!("S{index}"),
                    bits: match shape {
                        FreeShape::Word => Bits::Word {
                            high: word.0,
                            low: word.1,
                        },
                        FreeShape::Multi => Bits::Multi {
                            high: multi.0,
                            low: multi.1,
                        },
                        FreeShape::Array => Bits::MultiArray {
                            high: multi.0,
                            low: multi.1,
                            stride,
                        },
                    },
                    values,
                })
            })
            .collect::<Vec<_>>()
    })
}

fn headers() -> impl Strategy<Value = Header> {
    (
        prop::option::of(any::<u32>()),
        methods(),
        free_fields(),
        prop::collection::vec(any::<u32>(), 0..3),
    )
        .prop_map(|(class, methods, fields, constants)| Header {
            prefix: "NVC797_".to_owned(),
            notice: notice(),
            class: class.map(|id| ClassId {
                name: "ROUND_TRIP_CLASS".to_owned(),
                id,
            }),
            methods,
            fields,
            constants: constants
                .into_iter()
                .enumerate()
                .map(|(index, value)| Value {
                    name: format!("K{index}"),
                    value,
                })
                .collect(),
        })
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn a_written_header_reads_back_the_same(
        header in headers(),
        decimal in prop::collection::vec(any::<bool>(), 1..8),
        j_index in prop::collection::vec(any::<bool>(), 6),
    ) {
        let text = render(&header, &decimal, &j_index);
        prop_assert_eq!(parse(FILE, &text, None), Ok(header));
    }
}

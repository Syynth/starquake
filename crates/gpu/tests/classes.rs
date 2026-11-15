//! The generated class tables (`starquake_gpu::classes`) and their hand-written types.
//!
//! The golden entries were checked by hand against the header text: each cites the header (in
//! `mesa-nouveau/src/nouveau/headers/nvidia/classes/` of the reference folder, at the commit
//! pinned in `docs/references.md`) and the line. The literals in them are the header's, on
//! purpose: they are what the generator is checked against.

use std::collections::BTreeSet;

use starquake_gpu::classes::{
    Bits, ClassHeader, ClassId, HEADERS, Method, MwArray, MwField, NamedField, Value, cl902d,
    cla140, clc76f, clc797, clc797tex, clc7b5, clc7c0, clc7c0qmd,
};
use starquake_gpu::field::Field;
use starquake_gpu::{fifo, sema};

fn value(name: &'static str, value: u32) -> Value {
    Value { name, value }
}

fn field_named(fields: &'static [NamedField], name: &str) -> NamedField {
    *fields.iter().find(|field| field.name == name).unwrap()
}

// --- Golden entries, per class. ---

#[test]
fn c76f_host_methods_match_the_header() {
    // clc76f.h:42 AMPERE_CHANNEL_GPFIFO_B (0x0000C76F)
    assert_eq!(clc76f::CLASS, 0xC76F);
    assert_eq!(
        clc76f::HEADER.class,
        Some(ClassId {
            name: "AMPERE_CHANNEL_GPFIFO_B",
            id: 0xC76F
        })
    );
    // clc76f.h:67-70: SET_OBJECT (0x00000000), NVCLASS 15:0, ENGINE 20:16, ENGINE_SW 0x1f.
    assert_eq!(clc76f::SET_OBJECT.byte_address, 0);
    assert_eq!(clc76f::SET_OBJECT_NVCLASS, Field::new(15, 0));
    assert_eq!(clc76f::SET_OBJECT_ENGINE, Field::new(20, 16));
    assert_eq!(clc76f::SET_OBJECT_ENGINE_SW, 0x1f);
    assert_eq!(
        clc76f::SET_OBJECT.fields,
        [
            NamedField {
                name: "NVCLASS",
                bits: Bits::Word(Field::new(15, 0)),
                values: &[],
            },
            NamedField {
                name: "ENGINE",
                bits: Bits::Word(Field::new(20, 16)),
                values: &[Value {
                    name: "SW",
                    value: 0x1f
                }],
            },
        ]
    );
    // clc76f.h:81-87: SEMAPHORED (0x0000001C), OPERATION 4:0 with five values.
    assert_eq!(clc76f::SEMAPHORED.byte_address, 0x1C);
    assert_eq!(clc76f::SEMAPHORED.method(), 0x7);
    assert_eq!(
        field_named(clc76f::SEMAPHORED.fields, "OPERATION").values,
        [
            value("ACQUIRE", 1),
            value("RELEASE", 2),
            value("ACQ_GEQ", 4),
            value("ACQ_AND", 8),
            value("REDUCTION", 0x10)
        ]
    );
    // clc76f.h:259-265: CLEAR_FAULTED (0x00000084), HANDLE 30:0, TYPE 31:31.
    assert_eq!(clc76f::CLEAR_FAULTED.byte_address, 0x84);
    assert_eq!(clc76f::CLEAR_FAULTED_HANDLE, Field::new(30, 0));
    assert_eq!(clc76f::CLEAR_FAULTED_TYPE_ENG_FAULTED, 1);
}

#[test]
fn c76f_keeps_both_names_for_one_wfi_scope_value() {
    // clc76f.h:250-254: WFI (0x00000078), SCOPE 0:0, CURRENT_SCG_TYPE 0, CURRENT_VEID 0, ALL 1.
    // Mesa's parser drops CURRENT_SCG_TYPE; the generator keeps it (a listed divergence).
    assert_eq!(clc76f::WFI.byte_address, 0x78);
    assert_eq!(
        field_named(clc76f::WFI.fields, "SCOPE").values,
        [
            value("CURRENT_SCG_TYPE", 0),
            value("CURRENT_VEID", 0),
            value("ALL", 1)
        ]
    );
}

#[test]
fn c76f_mem_op_d_keeps_the_values_and_fields_after_a_stray_define() {
    // clc76f.h:180-205: among MEM_OP_D's OPERATION values sits MEM_OP_B_OPERATION_..._CLEAN_LINES
    // (line 190), which belongs to no field. The generator makes it a constant and keeps the
    // values and fields after it; Mesa's parser loses them (a listed divergence).
    let operation = field_named(clc76f::MEM_OP_D.fields, "OPERATION");
    assert_eq!(operation.bits, Bits::Word(Field::new(31, 27)));
    let names: Vec<&str> = operation.values.iter().map(|value| value.name).collect();
    assert_eq!(
        names,
        [
            "MEMBAR",
            "MMU_TLB_INVALIDATE",
            "MMU_TLB_INVALIDATE_TARGETED",
            "MMU_OPERATION",
            "L2_PEERMEM_INVALIDATE",
            "L2_SYSMEM_INVALIDATE",
            "L2_CLEAN_COMPTAGS",
            "L2_FLUSH_DIRTY",
            "L2_WAIT_FOR_SYS_PENDING_READS",
            "ACCESS_COUNTER_CLR",
        ]
    );
    assert_eq!(clc76f::MEM_OP_D_OPERATION_L2_CLEAN_COMPTAGS, 0xf);
    assert_eq!(clc76f::MEM_OP_D_ACCESS_COUNTER_CLR_TYPE, Field::new(1, 0));
    assert_eq!(clc76f::MEM_OP_D_MMU_OPERATION_TYPE, Field::new(23, 20));
    assert_eq!(clc76f::MEM_OP_B_OPERATION_L2_INVALIDATE_CLEAN_LINES, 0xe);
    assert_eq!(
        clc76f::HEADER.constants,
        [
            value("NUMBER_OF_SUBCHANNELS", 8),
            value("MEM_OP_B_OPERATION_L2_INVALIDATE_CLEAN_LINES", 0xe),
            value("GP_ENTRY__SIZE", 8),
            value("DMA_NOP", 0),
        ]
    );
}

#[test]
fn c76f_gpfifo_entry_and_method_header_fields_match_the_header() {
    // clc76f.h:269-287 (GPFIFO entry) and 290-311 (method header).
    assert_eq!(clc76f::GP_ENTRY0_GET, Field::new(31, 2));
    assert_eq!(clc76f::GP_ENTRY1_LENGTH, Field::new(30, 10));
    assert_eq!(clc76f::GP_ENTRY1_SYNC_WAIT, 1);
    assert_eq!(clc76f::GP_ENTRY1_OPCODE_PB_CRC, 3);
    assert_eq!(clc76f::DMA_METHOD_COUNT, Field::new(28, 16));
    assert_eq!(clc76f::DMA_SEC_OP_END_PB_SEGMENT, 7);
    // DMA_IMMD_DATA is defined twice (lines 302 and 338) with the same range: read once.
    let immd: Vec<_> = clc76f::HEADER
        .fields
        .iter()
        .filter(|field| field.name == "DMA_IMMD_DATA")
        .collect();
    assert_eq!(immd.len(), 1);
}

#[test]
fn the_hand_typed_fifo_and_semaphore_fields_agree_with_the_generated_ones() {
    // `fifo` and `sema` (slices G1 and G2) typed these from clc76f.h by hand.
    let pairs = [
        (fifo::GP_ENTRY0_FETCH, clc76f::GP_ENTRY0_FETCH),
        (fifo::GP_ENTRY0_GET, clc76f::GP_ENTRY0_GET),
        (fifo::GP_ENTRY1_GET_HI, clc76f::GP_ENTRY1_GET_HI),
        (fifo::GP_ENTRY1_LEVEL, clc76f::GP_ENTRY1_LEVEL),
        (fifo::GP_ENTRY1_LENGTH, clc76f::GP_ENTRY1_LENGTH),
        (fifo::GP_ENTRY1_SYNC, clc76f::GP_ENTRY1_SYNC),
        (fifo::METHOD_ADDRESS, clc76f::DMA_METHOD_ADDRESS),
        (fifo::METHOD_SUBCHANNEL, clc76f::DMA_METHOD_SUBCHANNEL),
        (fifo::METHOD_COUNT, clc76f::DMA_METHOD_COUNT),
        (fifo::IMMD_DATA, clc76f::DMA_IMMD_DATA),
        (fifo::SEC_OP, clc76f::DMA_SEC_OP),
        (fifo::TERT_OP, clc76f::DMA_TERT_OP),
        (fifo::METHOD_ADDRESS_OLD, clc76f::DMA_METHOD_ADDRESS_OLD),
        (fifo::METHOD_COUNT_OLD, clc76f::DMA_METHOD_COUNT_OLD),
        (fifo::SUBDEVICE_MASK, clc76f::DMA_SUBDEVICE_MASK),
        (fifo::SUBDEVICE_OPCODE, clc76f::DMA_SET_SUBDEVICE_MASK_OPCODE),
        (sema::SEMAPHORED_OPERATION, clc76f::SEMAPHORED_OPERATION),
        (sema::SEMAPHORED_ACQUIRE_SWITCH, clc76f::SEMAPHORED_ACQUIRE_SWITCH),
        (sema::SEMAPHORED_RELEASE_WFI, clc76f::SEMAPHORED_RELEASE_WFI),
        (sema::SEMAPHORED_RELEASE_SIZE, clc76f::SEMAPHORED_RELEASE_SIZE),
        (sema::SEMAPHORED_REDUCTION, clc76f::SEMAPHORED_REDUCTION),
        (sema::SEMAPHORED_FORMAT, clc76f::SEMAPHORED_FORMAT),
        (sema::SEM_EXECUTE_OPERATION, clc76f::SEM_EXECUTE_OPERATION),
        (
            sema::SEM_EXECUTE_ACQUIRE_SWITCH_TSG,
            clc76f::SEM_EXECUTE_ACQUIRE_SWITCH_TSG,
        ),
        (sema::SEM_EXECUTE_RELEASE_WFI, clc76f::SEM_EXECUTE_RELEASE_WFI),
        (sema::SEM_EXECUTE_PAYLOAD_SIZE, clc76f::SEM_EXECUTE_PAYLOAD_SIZE),
        (
            sema::SEM_EXECUTE_RELEASE_TIMESTAMP,
            clc76f::SEM_EXECUTE_RELEASE_TIMESTAMP,
        ),
        (sema::SEM_EXECUTE_REDUCTION, clc76f::SEM_EXECUTE_REDUCTION),
        (
            sema::SEM_EXECUTE_REDUCTION_FORMAT,
            clc76f::SEM_EXECUTE_REDUCTION_FORMAT,
        ),
    ];
    for (index, (typed, generated)) in pairs.iter().enumerate() {
        assert_eq!(typed, generated, "pair {index}");
    }
    let values = [
        (fifo::SEC_OP_INC_METHOD, clc76f::DMA_SEC_OP_INC_METHOD),
        (fifo::SEC_OP_NON_INC_METHOD, clc76f::DMA_SEC_OP_NON_INC_METHOD),
        (fifo::SEC_OP_IMMD_DATA_METHOD, clc76f::DMA_SEC_OP_IMMD_DATA_METHOD),
        (fifo::SEC_OP_ONE_INC, clc76f::DMA_SEC_OP_ONE_INC),
        (fifo::SEC_OP_END_PB_SEGMENT, clc76f::DMA_SEC_OP_END_PB_SEGMENT),
        (
            fifo::SUBDEVICE_OPCODE_SET,
            clc76f::DMA_SET_SUBDEVICE_MASK_OPCODE_VALUE,
        ),
        (
            fifo::SUBDEVICE_OPCODE_STORE,
            clc76f::DMA_STORE_SUBDEVICE_MASK_OPCODE_VALUE,
        ),
        (
            fifo::SUBDEVICE_OPCODE_USE,
            clc76f::DMA_USE_SUBDEVICE_MASK_OPCODE_VALUE,
        ),
        (sema::SEMAPHORED_OPERATION_ACQUIRE, clc76f::SEMAPHORED_OPERATION_ACQUIRE),
        (sema::SEMAPHORED_OPERATION_RELEASE, clc76f::SEMAPHORED_OPERATION_RELEASE),
        (sema::SEMAPHORED_OPERATION_ACQ_GEQ, clc76f::SEMAPHORED_OPERATION_ACQ_GEQ),
        (sema::SEMAPHORED_OPERATION_ACQ_AND, clc76f::SEMAPHORED_OPERATION_ACQ_AND),
        (
            sema::SEMAPHORED_OPERATION_REDUCTION,
            clc76f::SEMAPHORED_OPERATION_REDUCTION,
        ),
        (sema::SEMAPHORED_RELEASE_WFI_DIS, clc76f::SEMAPHORED_RELEASE_WFI_DIS),
        (
            sema::SEM_EXECUTE_OPERATION_ACQUIRE,
            clc76f::SEM_EXECUTE_OPERATION_ACQUIRE,
        ),
        (
            sema::SEM_EXECUTE_OPERATION_RELEASE,
            clc76f::SEM_EXECUTE_OPERATION_RELEASE,
        ),
        (
            sema::SEM_EXECUTE_OPERATION_ACQ_STRICT_GEQ,
            clc76f::SEM_EXECUTE_OPERATION_ACQ_STRICT_GEQ,
        ),
        (
            sema::SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ,
            clc76f::SEM_EXECUTE_OPERATION_ACQ_CIRC_GEQ,
        ),
        (
            sema::SEM_EXECUTE_OPERATION_ACQ_AND,
            clc76f::SEM_EXECUTE_OPERATION_ACQ_AND,
        ),
        (
            sema::SEM_EXECUTE_OPERATION_ACQ_NOR,
            clc76f::SEM_EXECUTE_OPERATION_ACQ_NOR,
        ),
        (
            sema::SEM_EXECUTE_OPERATION_REDUCTION,
            clc76f::SEM_EXECUTE_OPERATION_REDUCTION,
        ),
        (sema::SEM_EXECUTE_RELEASE_WFI_EN, clc76f::SEM_EXECUTE_RELEASE_WFI_EN),
        (
            sema::SEM_EXECUTE_RELEASE_TIMESTAMP_EN,
            clc76f::SEM_EXECUTE_RELEASE_TIMESTAMP_EN,
        ),
        (sema::HOST_REDUCTION_MIN, clc76f::SEMAPHORED_REDUCTION_MIN),
        (sema::HOST_REDUCTION_MAX, clc76f::SEMAPHORED_REDUCTION_MAX),
        (sema::HOST_REDUCTION_XOR, clc76f::SEMAPHORED_REDUCTION_XOR),
        (sema::HOST_REDUCTION_AND, clc76f::SEMAPHORED_REDUCTION_AND),
        (sema::HOST_REDUCTION_OR, clc76f::SEMAPHORED_REDUCTION_OR),
        (sema::HOST_REDUCTION_ADD, clc76f::SEMAPHORED_REDUCTION_ADD),
        (sema::HOST_REDUCTION_INC, clc76f::SEMAPHORED_REDUCTION_INC),
    ];
    for (index, (typed, generated)) in values.iter().enumerate() {
        assert_eq!(typed, generated, "value {index}");
    }
}

#[test]
fn c797_methods_and_arrays_match_the_header() {
    // clc797.h:32 AMPERE_B 0xC797
    assert_eq!(clc797::CLASS, 0xC797);
    // clc797.h:34-36 SET_OBJECT 0x0000, CLASS_ID 15:0, ENGINE_ID 20:16.
    assert_eq!(clc797::SET_OBJECT.byte_address, 0);
    assert_eq!(clc797::SET_OBJECT_CLASS_ID, Field::new(15, 0));
    assert_eq!(clc797::SET_OBJECT_ENGINE_ID, Field::new(20, 16));
    // clc797.h:38 NO_OPERATION 0x0100: method 0x40.
    assert_eq!(clc797::NO_OPERATION.byte_address, 0x0100);
    assert_eq!(clc797::NO_OPERATION.method(), 0x40);
    // clc797.h:67-72 SET_MME_SHADOW_RAM_CONTROL 0x0124, MODE 1:0 with four values.
    assert_eq!(clc797::SET_MME_SHADOW_RAM_CONTROL.byte_address, 0x0124);
    assert_eq!(
        field_named(clc797::SET_MME_SHADOW_RAM_CONTROL.fields, "MODE"),
        NamedField {
            name: "MODE",
            bits: Bits::Word(Field::new(1, 0)),
            values: &[
                Value {
                    name: "METHOD_TRACK",
                    value: 0
                },
                Value {
                    name: "METHOD_TRACK_WITH_FILTER",
                    value: 1
                },
                Value {
                    name: "METHOD_PASSTHROUGH",
                    value: 2
                },
                Value {
                    name: "METHOD_REPLAY",
                    value: 3
                },
            ],
        }
    );
    // clc797.h:1120-1121 SET_VIEWPORT_SCALE_X(j) (0x0a00+(j)*32), V 31:0.
    assert_eq!(clc797::SET_VIEWPORT_SCALE_X.byte_address, 0x0a00);
    assert_eq!(clc797::SET_VIEWPORT_SCALE_X.strides, [32]);
    assert_eq!(clc797::SET_VIEWPORT_SCALE_X.byte_address_at(&[3]), Some(0x0a60));
    assert_eq!(clc797::SET_VIEWPORT_SCALE_X_V, Field::new(31, 0));
    // clc797.h:4535-4539 SET_STREAM_OUT_LAYOUT_SELECT(i,j) (0x2800+(i)*128+(j)*4).
    let select = clc797::SET_STREAM_OUT_LAYOUT_SELECT;
    assert_eq!((select.byte_address, select.strides), (0x2800, &[128, 4][..]));
    assert_eq!(select.byte_address_at(&[1, 2]), Some(0x2800 + 128 + 8));
    assert_eq!(select.method_at(&[1, 2]), Some((0x2800 + 128 + 8) / 4));
    assert_eq!(
        clc797::SET_STREAM_OUT_LAYOUT_SELECT_ATTRIBUTE_NUMBER03,
        Field::new(31, 24)
    );
    // clc797.h:630-633: value names with lower-case letters keep them in the table; the
    // constant names are upper-cased.
    assert_eq!(clc797::ASSIGN_ZCULL_SUBREGIONS_ALGORITHM_STATIC, 0);
    assert_eq!(clc797::ASSIGN_ZCULL_SUBREGIONS_ALGORITHM_ADAPTIVE, 1);
    let algorithm = field_named(clc797::ASSIGN_ZCULL_SUBREGIONS.fields, "ALGORITHM");
    assert_eq!(
        algorithm.values,
        [value("Static", 0), value("Adaptive", 1)]
    );
}

#[test]
fn c7c0_compute_methods_match_the_header() {
    // clc7c0.h:32 AMPERE_COMPUTE_B 0xC7C0
    assert_eq!(clc7c0::CLASS, 0xC7C0);
    // clc7c0.h:441-442 SEND_PCAS_A 0x02b4, QMD_ADDRESS_SHIFTED8 31:0.
    assert_eq!(clc7c0::SEND_PCAS_A.byte_address, 0x02b4);
    assert_eq!(clc7c0::SEND_PCAS_A_QMD_ADDRESS_SHIFTED8, Field::new(31, 0));
    // clc7c0.h:497 LOAD_INLINE_QMD_DATA(i) (0x0320+(i)*4).
    assert_eq!(clc7c0::LOAD_INLINE_QMD_DATA.byte_address, 0x0320);
    assert_eq!(clc7c0::LOAD_INLINE_QMD_DATA.strides, [4]);
    // clc7c0.h:842 CALL_MME_MACRO(j) (0x3800+(j)*8).
    assert_eq!(clc7c0::CALL_MME_MACRO.byte_address, 0x3800);
    assert_eq!(clc7c0::CALL_MME_MACRO.strides, [8]);
}

#[test]
fn c7b5_copy_methods_match_the_header() {
    // clc7b5.h:33 AMPERE_DMA_COPY_B (0x0000C7B5)
    assert_eq!(clc7b5::CLASS, 0xC7B5);
    // clc7b5.h:39-42 SET_MONITORED_FENCE_TYPE (0x0000021C), TYPE 0:0, two values.
    assert_eq!(clc7b5::SET_MONITORED_FENCE_TYPE.byte_address, 0x021C);
    assert_eq!(
        field_named(clc7b5::SET_MONITORED_FENCE_TYPE.fields, "TYPE").values,
        [value("MONITORED_FENCE", 0), value("MONITORED_FENCE_EXT", 1)]
    );
    // clc7b5.h:47-48 SET_SEMAPHORE_A (0x00000240), UPPER 16:0.
    assert_eq!(clc7b5::SET_SEMAPHORE_A.byte_address, 0x0240);
    assert_eq!(clc7b5::SET_SEMAPHORE_A_UPPER, Field::new(16, 0));
    // clc7b5.h:84-88 LAUNCH_DMA (0x00000300), DATA_TRANSFER_TYPE 1:0.
    assert_eq!(clc7b5::LAUNCH_DMA.byte_address, 0x0300);
    assert_eq!(clc7b5::LAUNCH_DMA_DATA_TRANSFER_TYPE, Field::new(1, 0));
    assert_eq!(clc7b5::LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED, 2);
}

#[test]
fn a140_inline_to_memory_methods_match_the_header() {
    // cla140.h:34 KEPLER_INLINE_TO_MEMORY_B 0xA140
    assert_eq!(cla140::CLASS, 0xA140);
    // cla140.h:137-140 LAUNCH_DMA 0x01b0, DST_MEMORY_LAYOUT 0:0, BLOCKLINEAR 0, PITCH 1.
    assert_eq!(cla140::LAUNCH_DMA.byte_address, 0x01b0);
    assert_eq!(cla140::LAUNCH_DMA_DST_MEMORY_LAYOUT, Field::new(0, 0));
    assert_eq!(cla140::LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH, 1);
    // cla140.h:170 LOAD_INLINE_DATA 0x01b4.
    assert_eq!(cla140::LOAD_INLINE_DATA.byte_address, 0x01b4);
    // cla140.h:310 SET_MME_SHADOW_SCRATCH(i) (0x3400+(i)*4).
    assert_eq!(cla140::SET_MME_SHADOW_SCRATCH.byte_address, 0x3400);
    assert_eq!(cla140::SET_MME_SHADOW_SCRATCH.strides, [4]);
}

#[test]
fn twod_methods_match_the_header() {
    // cl902d.h:26 FERMI_TWOD_A 0x902D
    assert_eq!(cl902d::CLASS, 0x902D);
    // cl902d.h:585-590 SET_PATTERN_SELECT 0x02b4, V 1:0 with four values.
    assert_eq!(cl902d::SET_PATTERN_SELECT.byte_address, 0x02b4);
    assert_eq!(
        field_named(cl902d::SET_PATTERN_SELECT.fields, "V").values,
        [
            value("MONOCHROME_8x8", 0),
            value("MONOCHROME_64x1", 1),
            value("MONOCHROME_1x64", 2),
            value("COLOR", 3)
        ]
    );
    assert_eq!(cl902d::SET_PATTERN_SELECT_V_MONOCHROME_1X64, 2);
    // cl902d.h:803 RENDER_SOLID_PRIM_POINT_SET_X(j) (0x0600+(j)*8).
    assert_eq!(cl902d::RENDER_SOLID_PRIM_POINT_SET_X.byte_address, 0x0600);
    assert_eq!(cl902d::RENDER_SOLID_PRIM_POINT_SET_X.strides, [8]);
}

#[test]
fn qmd_03_00_fields_match_the_header_and_only_that_version_is_kept() {
    // clc7c0qmd.h:508 DEPENDENCE_COUNTER MW(157:142).
    assert_eq!(clc7c0qmd::QMDV03_00_DEPENDENCE_COUNTER, MwField::new(157, 142));
    // clc7c0qmd.h:557-559 API_VISIBLE_CALL_LIMIT MW(378:378), _32 0, NO_CHECK 1.
    assert_eq!(clc7c0qmd::QMDV03_00_API_VISIBLE_CALL_LIMIT__32, 0);
    assert_eq!(clc7c0qmd::QMDV03_00_API_VISIBLE_CALL_LIMIT_NO_CHECK, 1);
    // clc7c0qmd.h:567 CTA_RASTER_WIDTH MW(415:384); 706 PROGRAM_ADDRESS_LOWER MW(1567:1536).
    assert_eq!(clc7c0qmd::QMDV03_00_CTA_RASTER_WIDTH, MwField::new(415, 384));
    assert_eq!(
        clc7c0qmd::QMDV03_00_PROGRAM_ADDRESS_LOWER,
        MwField::new(1567, 1536)
    );
    // clc7c0qmd.h:608-610 CONSTANT_BUFFER_VALID(i) MW((640+(i)*1):(640+(i)*1)).
    assert_eq!(
        clc7c0qmd::QMDV03_00_CONSTANT_BUFFER_VALID,
        MwArray::new(640, 640, 1)
    );
    assert_eq!(clc7c0qmd::QMDV03_00_CONSTANT_BUFFER_VALID_TRUE, 1);
    // clc7c0qmd.h:697 CONSTANT_BUFFER_ADDR_LOWER(i) MW((1055+(i)*64):(1024+(i)*64)).
    let lower = clc7c0qmd::QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER;
    assert_eq!(lower, MwArray::new(1055, 1024, 64));
    assert_eq!(lower.element(2), Some(MwField::new(1055 + 128, 1024 + 128)));
    // The header also holds QMD 02_03 and 02_04; only 03_00 is generated.
    assert!(clc7c0qmd::HEADER.class.is_none());
    assert!(clc7c0qmd::HEADER.methods.is_empty());
    assert!(
        clc7c0qmd::HEADER
            .fields
            .iter()
            .all(|field| field.name.starts_with("QMDV03_00_"))
    );
}

#[test]
fn texture_header_and_sampler_fields_match_the_header() {
    // clc797tex.h:33 and 41: TEXHEAD_BL_COMPONENTS MW(6:0), SIZES_A8B8G8R8 0x8.
    assert_eq!(clc797tex::TEXHEAD_BL_COMPONENTS, MwField::new(6, 0));
    assert_eq!(clc797tex::TEXHEAD_BL_COMPONENTS_SIZES_A8B8G8R8, 8);
    // clc797tex.h:180 ADDRESS_BITS31TO9 MW(63:41); 250 WIDTH_MINUS_ONE MW(144:128).
    assert_eq!(clc797tex::TEXHEAD_BL_ADDRESS_BITS31TO9, MwField::new(63, 41));
    assert_eq!(clc797tex::TEXHEAD_BL_WIDTH_MINUS_ONE, MwField::new(144, 128));
    // clc797tex.h:1001 TEXHEAD_PITCH_PITCH_BITS20TO5 MW(111:96).
    assert_eq!(clc797tex::TEXHEAD_PITCH_PITCH_BITS20TO5, MwField::new(111, 96));
    // clc797tex.h:1361-1369: the sampler's fields are per word, numbered in the name.
    assert_eq!(clc797tex::TEXSAMP0_ADDRESS_U, Field::new(2, 0));
    assert_eq!(clc797tex::TEXSAMP0_ADDRESS_U_WRAP, 0);
    assert_eq!(clc797tex::TEXSAMP0_ADDRESS_U_MIRROR_ONCE_CLAMP_OGL, 7);
    assert!(clc797tex::HEADER.class.is_none());
}

// --- Properties of every generated table. ---

fn all_fields(header: &ClassHeader) -> impl Iterator<Item = &NamedField> {
    header
        .methods
        .iter()
        .flat_map(|method| method.fields)
        .chain(header.fields)
}

/// The width of a field, from its bit range.
fn width(bits: Bits) -> u32 {
    match bits {
        Bits::Word(field) => field.width(),
        Bits::Multi(field) => field.width().unwrap(),
        Bits::MultiArray(array) => array.element(0).unwrap().width().unwrap(),
    }
}

#[test]
fn the_header_list_is_every_module_in_name_order() {
    let names: Vec<&str> = HEADERS.iter().map(|header| header.header).collect();
    assert_eq!(
        names,
        [
            "cl902d.h",
            "cla140.h",
            "clc76f.h",
            "clc797.h",
            "clc797tex.h",
            "clc7b5.h",
            "clc7c0.h",
            "clc7c0qmd.h"
        ]
    );
}

#[test]
fn every_method_is_word_aligned_and_every_field_is_valid() {
    for header in HEADERS {
        for method in header.methods {
            assert_eq!(method.byte_address % 4, 0, "{}", method.name);
            assert!(method.strides.iter().all(|stride| stride % 4 == 0 && *stride > 0));
            assert!(method.strides.len() <= 2);
            assert!(!method.fields.is_empty(), "{}", method.name);
            assert!(
                method
                    .fields
                    .iter()
                    .all(|field| matches!(field.bits, Bits::Word(_)))
            );
        }
        for field in all_fields(header) {
            let width = width(field.bits);
            assert!((1..=32).contains(&width), "{} {}", header.header, field.name);
        }
    }
}

#[test]
fn every_named_value_fits_its_field() {
    for header in HEADERS {
        for field in all_fields(header) {
            let max = u32::MAX >> (32 - width(field.bits));
            for value in field.values {
                assert!(
                    value.value <= max,
                    "{}: {}_{} = {:#x} does not fit {} bits",
                    header.header,
                    field.name,
                    value.name,
                    value.value,
                    width(field.bits)
                );
            }
        }
    }
}

#[test]
fn names_are_unique_within_each_table() {
    for header in HEADERS {
        let mut methods = BTreeSet::new();
        for method in header.methods {
            assert!(methods.insert(method.name), "{}", method.name);
            let mut fields = BTreeSet::new();
            for field in method.fields {
                assert!(fields.insert(field.name), "{}_{}", method.name, field.name);
            }
        }
        let mut free = BTreeSet::new();
        for field in header.fields {
            assert!(free.insert(field.name), "{}", field.name);
        }
        for field in all_fields(header) {
            let mut values = BTreeSet::new();
            for value in field.values {
                assert!(values.insert(value.name), "{}_{}", field.name, value.name);
            }
        }
    }
}

#[test]
fn the_tables_hold_every_kind_of_item_where_the_headers_have_one() {
    let count = |header: &ClassHeader| {
        (
            header.methods.len(),
            header.fields.len(),
            header.constants.len(),
            header.methods.iter().filter(|m| !m.strides.is_empty()).count(),
        )
    };
    // Methods, fields of no method, constants, array methods; counted from the generated
    // tables, which hold one entry per define of each header.
    let c76f = count(&clc76f::HEADER);
    assert!(c76f.0 > 0 && c76f.1 > 0 && c76f.2 == 4 && c76f.3 == 0);
    let c797 = count(&clc797::HEADER);
    assert!(c797.0 > 0 && c797.1 == 0 && c797.2 == 0 && c797.3 > 0);
    let tex = count(&clc797tex::HEADER);
    assert!(tex.0 == 0 && tex.1 > 0 && tex.2 == 0);
    assert!(
        clc7c0qmd::HEADER
            .fields
            .iter()
            .any(|field| matches!(field.bits, Bits::MultiArray(_)))
    );
}

// --- The hand-written types. ---

const PLAIN: Method = Method {
    name: "PLAIN",
    byte_address: 0x0124,
    strides: &[],
    fields: &[],
};

const TWO: Method = Method {
    name: "TWO",
    byte_address: 0x2800,
    strides: &[128, 4],
    fields: &[],
};

#[test]
fn a_method_number_is_the_byte_address_over_four() {
    assert_eq!(PLAIN.method(), 0x49);
    assert_eq!(PLAIN.byte_address_at(&[]), Some(0x0124));
    assert_eq!(PLAIN.method_at(&[]), Some(0x49));
}

#[test]
fn an_element_address_needs_one_index_per_stride() {
    assert_eq!(TWO.byte_address_at(&[0, 0]), Some(0x2800));
    assert_eq!(TWO.byte_address_at(&[2, 3]), Some(0x2800 + 256 + 12));
    assert_eq!(TWO.byte_address_at(&[1]), None);
    assert_eq!(TWO.byte_address_at(&[1, 1, 1]), None);
    assert_eq!(PLAIN.byte_address_at(&[0]), None);
    assert_eq!(TWO.method_at(&[1]), None);
}

#[test]
fn an_element_address_past_32_bits_is_none_and_exactly_at_the_limit_is_kept() {
    let one = Method {
        name: "ONE",
        byte_address: 4,
        strides: &[4],
        fields: &[],
    };
    // 4 + 4 * i = u32::MAX - 3 is the last aligned address.
    let last = (u32::MAX - 3 - 4) / 4;
    assert_eq!(one.byte_address_at(&[last]), Some(u32::MAX - 3));
    assert_eq!(one.byte_address_at(&[last + 1]), None);
    // The product alone overflowing is caught too.
    assert_eq!(one.byte_address_at(&[u32::MAX]), None);
    // The sum of two in-range terms overflowing.
    let big = Method {
        name: "BIG",
        byte_address: 0,
        strides: &[0x8000_0000, 0x8000_0000],
        fields: &[],
    };
    assert_eq!(big.byte_address_at(&[1, 0]), Some(0x8000_0000));
    assert_eq!(big.byte_address_at(&[1, 1]), None);
}

#[test]
fn a_multi_word_field_reads_within_one_word() {
    // Word 1, bits 5..=9 (MW(41:37)).
    let field = MwField::new(41, 37);
    let words = [0xffff_ffff, 0b10110 << 5, 0xffff_ffff];
    assert_eq!(field.get(&words), Some(0b10110));
    assert_eq!(field.width(), Some(5));
    assert_eq!((field.high(), field.low()), (41, 37));
}

#[test]
fn a_multi_word_field_spanning_two_words_joins_them() {
    // MW(39:24): bits 24..=31 of word 0 (low byte of the value) and 0..=7 of word 1.
    let field = MwField::new(39, 24);
    let words = [0xab00_0000, 0x0000_00cd];
    assert_eq!(field.get(&words), Some(0xcdab));
    let mut written = [0x00ff_ffff, 0xffff_ff00];
    field.insert(&mut written, 0x1234).unwrap();
    assert_eq!(written, [0x34ff_ffff, 0xffff_ff12]);
    assert_eq!(field.get(&written), Some(0x1234));
}

#[test]
fn whole_words_at_the_32_bit_width_limit_work() {
    let first = MwField::new(31, 0);
    let second = MwField::new(63, 32);
    let mut words = [0u32; 2];
    first.insert(&mut words, 0xdead_beef).unwrap();
    second.insert(&mut words, 0x0123_4567).unwrap();
    assert_eq!(words, [0xdead_beef, 0x0123_4567]);
    assert_eq!(first.get(&words), Some(0xdead_beef));
    assert_eq!(second.get(&words), Some(0x0123_4567));
    assert_eq!(first.width(), Some(32));
    // One bit wider is invalid.
    let wide = MwField::new(32, 0);
    assert_eq!(wide.width(), None);
    assert_eq!(wide.get(&words), None);
    assert_eq!(wide.insert(&mut words, 0), None);
}

#[test]
fn an_inverted_field_is_invalid() {
    let inverted = MwField::new(3, 4);
    let mut words = [0u32; 1];
    assert_eq!(inverted.width(), None);
    assert_eq!(inverted.get(&words), None);
    assert_eq!(inverted.insert(&mut words, 0), None);
}

#[test]
fn a_field_ending_past_the_words_is_none_and_one_ending_at_the_last_word_is_read() {
    let field = MwField::new(95, 64);
    assert_eq!(field.get(&[0, 0, 7]), Some(7));
    assert_eq!(field.get(&[0, 0]), None);
    let mut short = [1u32, 2];
    assert_eq!(field.insert(&mut short, 1), None);
    assert_eq!(short, [1, 2]);
    // A field spanning into a missing second word.
    let spanning = MwField::new(39, 24);
    assert_eq!(spanning.get(&[0]), None);
    let mut one = [5u32];
    assert_eq!(spanning.insert(&mut one, 1), None);
    assert_eq!(one, [5]);
}

#[test]
fn inserting_a_value_too_wide_changes_nothing_and_the_widest_value_fits() {
    let field = MwField::new(38, 37);
    let mut words = [0x1111_1111u32, 0x2222_2222];
    assert_eq!(field.insert(&mut words, 4), None);
    assert_eq!(words, [0x1111_1111, 0x2222_2222]);
    field.insert(&mut words, 3).unwrap();
    assert_eq!(field.get(&words), Some(3));
}

#[test]
fn inserting_leaves_the_neighbouring_fields_alone() {
    // Neighbours that end just below and start just above the field.
    let below = MwField::new(36, 33);
    let field = MwField::new(40, 37);
    let above = MwField::new(44, 41);
    let mut words = [0u32; 2];
    below.insert(&mut words, 0xa).unwrap();
    above.insert(&mut words, 0x5).unwrap();
    field.insert(&mut words, 0xf).unwrap();
    assert_eq!(below.get(&words), Some(0xa));
    assert_eq!(above.get(&words), Some(0x5));
    assert_eq!(field.get(&words), Some(0xf));
    field.insert(&mut words, 0).unwrap();
    assert_eq!(below.get(&words), Some(0xa));
    assert_eq!(above.get(&words), Some(0x5));
    assert_eq!(field.get(&words), Some(0));
}

#[test]
fn array_elements_step_by_the_stride_until_the_positions_overflow() {
    let array = MwArray::new(1055, 1024, 64);
    assert_eq!(array.stride(), 64);
    assert_eq!(array.element(0), Some(MwField::new(1055, 1024)));
    assert_eq!(array.element(1), Some(MwField::new(1119, 1088)));
    // The last element whose high bit fits in 32 bits, and the first that does not.
    let last = (u32::MAX - 1055) / 64;
    assert!(array.element(last).is_some());
    assert_eq!(array.element(last + 1), None);
    assert_eq!(array.element(u32::MAX), None);
    // The stride times the index overflowing is caught even where the wrapped product would
    // land back on a small position.
    let wide = MwArray::new(0, 0, 0x8000_0000);
    assert_eq!(wide.element(1), Some(MwField::new(0x8000_0000, 0x8000_0000)));
    assert_eq!(wide.element(2), None);
}

#[test]
fn a_qmd_built_through_the_tables_reads_back() {
    // A QMD 03_00 is 64 words (its highest bit, MW(2047:...), is in word 63).
    let mut qmd = [0u32; 64];
    let lower = clc7c0qmd::QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER
        .element(1)
        .unwrap();
    let valid = clc7c0qmd::QMDV03_00_CONSTANT_BUFFER_VALID.element(1).unwrap();
    lower.insert(&mut qmd, 0x8000_1000).unwrap();
    valid
        .insert(&mut qmd, clc7c0qmd::QMDV03_00_CONSTANT_BUFFER_VALID_TRUE)
        .unwrap();
    clc7c0qmd::QMDV03_00_CTA_RASTER_WIDTH
        .insert(&mut qmd, 640)
        .unwrap();
    // Element 1 of the address array is word (1024 + 64) / 32 = 34.
    assert_eq!(qmd[34], 0x8000_1000);
    // Valid bit 641 is bit 1 of word 20; CTA_RASTER_WIDTH is word 12.
    assert_eq!(qmd[20], 1 << 1);
    assert_eq!(qmd[12], 640);
    assert_eq!(lower.get(&qmd), Some(0x8000_1000));
    assert_eq!(valid.get(&qmd), Some(1));
}

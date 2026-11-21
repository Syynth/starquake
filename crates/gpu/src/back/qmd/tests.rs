//! Tests for QMD decoding, the PCAS scheduler and the inline QMD.
//!
//! QMDs are built with [`encode`], which writes every field through the same generated
//! `clc7c0qmd` constants the decoder reads. Where a wrong constant could agree with itself, a
//! hand-laid test writes the words out from Eden-NX2's literal word indices and masks instead
//! (`golden_hand_laid_qmd_decodes_every_field`, `constants_match_the_references_written_out`).

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use starquake_memory::{FlatMemory, PAGE_SIZE, PageRange};

use super::*;
use crate::classes::clc7c0::{
    SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INCREMENT_PUT, SET_QMD_VIRTUALIZATION_CONTROL,
};
use crate::memory::TestGpuMemory;
use crate::sema::DecodeError;

/// Where the rig's GPU memory starts: a 40-bit address, so the upper words are non-zero.
const GPU_BASE: u64 = 0x12_3400_0000;
/// The CPU address it is mapped onto.
const CPU_BASE: u64 = 0x8000_0000;
/// Mapped pages: room for 320 QMDs, enough for the longest chain plus one.
const PAGES: u64 = 80;

type Memory = TestGpuMemory<FlatMemory>;

fn memory() -> Memory {
    let size = usize::try_from(PAGES * PAGE_SIZE).unwrap();
    let mem = TestGpuMemory::new(FlatMemory::new(CPU_BASE, size));
    mem.map(
        PageRange::new(GPU_BASE, PAGES * PAGE_SIZE).unwrap(),
        CPU_BASE,
    )
    .unwrap();
    mem
}

/// The end (exclusive) of the rig's mapped GPU memory.
const GPU_END: u64 = GPU_BASE + PAGES * PAGE_SIZE;

/// The address of the rig's QMD number `n`.
fn qmd_at(n: u64) -> u64 {
    GPU_BASE + n * QMD_BYTES as u64
}

/// A release target inside the rig.
fn target(n: u64) -> u64 {
    GPU_END - PAGE_SIZE + n * 4
}

fn set(words: &mut [u32; QMD_WORDS], field: MwField, value: u32) {
    field.insert(words, value).unwrap();
}

fn set_element(words: &mut [u32; QMD_WORDS], array: MwArray, index: u32, value: u32) {
    array.element(index).unwrap().insert(words, value).unwrap();
}

/// One release slot as the test encoder writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotSpec {
    address: u64,
    payload: u32,
    /// `RELEASE*_MEMBAR_TYPE` (bit 19 of the slot's second word).
    membar: u32,
    /// `RELEASE*_REDUCTION_ENABLE`, `_OP` and `_FORMAT`.
    reduction: Option<(u32, u32)>,
    /// `_REDUCTION_FORMAT` when the reduction is not enabled.
    stray_format: u32,
    structure_size: u32,
    payload64b: u32,
    trap: u32,
}

impl SlotSpec {
    /// A plain one-word release of `payload` to `address`.
    pub(crate) fn write(address: u64, payload: u32) -> Self {
        Self {
            address,
            payload,
            membar: 0,
            reduction: None,
            stray_format: 0,
            structure_size: q::QMDV03_00_RELEASE0_STRUCTURE_SIZE_SEMAPHORE_ONE_WORD,
            payload64b: 0,
            trap: 0,
        }
    }
}

/// Everything the test encoder writes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Spec {
    major: u32,
    minor: u32,
    queue: bool,
    require_scheduling_pcas: bool,
    counter: u16,
    pub(crate) dispatch: Dispatch,
    slots: [Option<SlotSpec>; RELEASE_SLOTS],
    dependent: Option<Dependent>,
    second_dependent: bool,
}

impl Spec {
    /// A runnable QMD: version 3.0, a 1x1x1 grid of 1x1x1 blocks, program `program`, nothing
    /// else.
    pub(crate) fn new(program: u64) -> Self {
        Self {
            major: QMD_MAJOR_VERSION,
            minor: QMD_MINOR_VERSION,
            queue: false,
            require_scheduling_pcas: true,
            counter: 0,
            dispatch: Dispatch {
                qmd: 0,
                program,
                grid: [1, 1, 1],
                block: [1, 1, 1],
                shared_memory: 0,
                local_memory_low: 0,
                local_memory_high: 0,
                constant_buffers: [ConstantBuffer::default(); CONSTANT_BUFFERS],
            },
            slots: [None; RELEASE_SLOTS],
            dependent: None,
            second_dependent: false,
        }
    }

    fn counter(mut self, counter: u16) -> Self {
        self.counter = counter;
        self
    }

    pub(crate) fn slot(mut self, slot: usize, spec: SlotSpec) -> Self {
        self.slots[slot] = Some(spec);
        self
    }

    fn dependent(mut self, address: u64, action: u32) -> Self {
        self.dependent = Some(Dependent { address, action });
        self
    }
}

fn lo(x: u64) -> u32 {
    u32::try_from(x & 0xffff_ffff).unwrap()
}

fn hi(x: u64) -> u32 {
    u32::try_from(x >> 32).unwrap()
}

/// The test encoder: every field through the decoder's own constants.
pub(crate) fn encode(spec: &Spec) -> [u32; QMD_WORDS] {
    let mut w = [0u32; QMD_WORDS];
    set(&mut w, q::QMDV03_00_QMD_MAJOR_VERSION, spec.major);
    set(&mut w, q::QMDV03_00_QMD_VERSION, spec.minor);
    set(&mut w, q::QMDV03_00_IS_QUEUE, u32::from(spec.queue));
    set(
        &mut w,
        q::QMDV03_00_REQUIRE_SCHEDULING_PCAS,
        u32::from(spec.require_scheduling_pcas),
    );
    set(
        &mut w,
        q::QMDV03_00_DEPENDENCE_COUNTER,
        u32::from(spec.counter),
    );
    encode_dispatch(&mut w, &spec.dispatch);
    for (slot, fields) in spec.slots.iter().zip(&SLOTS) {
        if let Some(slot) = slot {
            encode_slot(&mut w, fields, slot);
        }
    }
    if let Some(dep) = spec.dependent {
        set(&mut w, q::QMDV03_00_DEPENDENT_QMD0_ENABLE, 1);
        set(
            &mut w,
            q::QMDV03_00_DEPENDENT_QMD0_POINTER,
            u32::try_from(dep.address >> QMD_ADDRESS_SHIFT).unwrap(),
        );
        set(&mut w, q::QMDV03_00_DEPENDENT_QMD0_ACTION, dep.action);
    }
    set(
        &mut w,
        q::QMDV03_00_DEPENDENT_QMD1_ENABLE,
        u32::from(spec.second_dependent),
    );
    w
}

fn encode_dispatch(w: &mut [u32; QMD_WORDS], d: &Dispatch) {
    set(w, q::QMDV03_00_PROGRAM_ADDRESS_LOWER, lo(d.program));
    set(w, q::QMDV03_00_PROGRAM_ADDRESS_UPPER, hi(d.program));
    set(w, q::QMDV03_00_CTA_RASTER_WIDTH, d.grid[0]);
    set(w, q::QMDV03_00_CTA_RASTER_HEIGHT, d.grid[1]);
    set(w, q::QMDV03_00_CTA_RASTER_DEPTH, d.grid[2]);
    set(w, q::QMDV03_00_CTA_THREAD_DIMENSION0, d.block[0]);
    set(w, q::QMDV03_00_CTA_THREAD_DIMENSION1, d.block[1]);
    set(w, q::QMDV03_00_CTA_THREAD_DIMENSION2, d.block[2]);
    set(w, q::QMDV03_00_SHARED_MEMORY_SIZE, d.shared_memory);
    set(
        w,
        q::QMDV03_00_SHADER_LOCAL_MEMORY_LOW_SIZE,
        d.local_memory_low,
    );
    set(
        w,
        q::QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE,
        d.local_memory_high,
    );
    for (i, cb) in (0u32..).zip(&d.constant_buffers) {
        set_element(
            w,
            q::QMDV03_00_CONSTANT_BUFFER_VALID,
            i,
            u32::from(cb.valid),
        );
        set_element(
            w,
            q::QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER,
            i,
            lo(cb.address),
        );
        set_element(
            w,
            q::QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER,
            i,
            hi(cb.address),
        );
        set_element(
            w,
            q::QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4,
            i,
            cb.size >> CONSTANT_BUFFER_SIZE_SHIFT,
        );
    }
}

fn encode_slot(w: &mut [u32; QMD_WORDS], fields: &ReleaseSlot, s: &SlotSpec) {
    set(w, fields.enable, q::QMDV03_00_RELEASE0_ENABLE_TRUE);
    set(w, fields.address_lower, lo(s.address));
    set(w, fields.address_upper, hi(s.address));
    set(w, fields.payload_lower, s.payload);
    set(w, fields.membar_type, s.membar);
    set(w, fields.structure_size, s.structure_size);
    set(w, fields.payload64b, s.payload64b);
    set(w, fields.trap_type, s.trap);
    set(w, fields.reduction_format, s.stray_format);
    if let Some((op, format)) = s.reduction {
        set(
            w,
            fields.reduction_enable,
            q::QMDV03_00_RELEASE0_REDUCTION_ENABLE_TRUE,
        );
        set(w, fields.reduction_op, op);
        set(w, fields.reduction_format, format);
    }
}

fn to_bytes(words: &[u32; QMD_WORDS]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Writes a QMD into the rig.
fn put(mem: &Memory, address: u64, spec: &Spec) {
    mem.write(address, &to_bytes(&encode(spec))).unwrap();
}

fn read_u32(mem: &Memory, address: u64) -> u32 {
    let mut b = [0u8; 4];
    mem.read(address, &mut b).unwrap();
    u32::from_le_bytes(b)
}

/// No events, typed.
const NOTHING: [Event; 0] = [];

/// What a sink saw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Dispatch(Dispatch),
    Release(Release),
    Completed(u64, Outcome),
}

/// A sink that records everything and rejects the programs listed.
#[derive(Debug, Default)]
struct Recorder {
    events: Vec<Event>,
    reject: Vec<u64>,
}

impl QmdSink for Recorder {
    fn dispatch(&mut self, dispatch: &Dispatch) -> Program {
        self.events.push(Event::Dispatch(*dispatch));
        if self.reject.contains(&dispatch.program) {
            Program::Rejected
        } else {
            Program::Accepted
        }
    }

    fn release(&mut self, release: Release) {
        self.events.push(Event::Release(release));
    }

    fn completed(&mut self, qmd: u64, outcome: Outcome) {
        self.events.push(Event::Completed(qmd, outcome));
    }
}

impl Recorder {
    fn completions(&self) -> Vec<(u64, Outcome)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Completed(a, o) => Some((*a, *o)),
                _ => None,
            })
            .collect()
    }

    fn releases(&self) -> Vec<Release> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Release(r) => Some(*r),
                _ => None,
            })
            .collect()
    }
}

fn write_release(address: u64, payload: u32) -> Release {
    Release {
        address,
        payload: u64::from(payload),
        form: ReleaseForm {
            size: PayloadSize::Four,
            timestamp: false,
        },
        op: ReleaseOp::Write,
    }
}

fn dec(count: u16) -> PcasAction {
    PcasAction::DecrementDependence { count }
}

fn inc(count: u16) -> PcasAction {
    PcasAction::IncrementDependence { count }
}

/// The expected dispatch of `spec` at `address`.
fn dispatch_of(spec: &Spec, address: u64) -> Dispatch {
    Dispatch {
        qmd: address,
        ..spec.dispatch
    }
}

// ---------------------------------------------------------------------------------------------
// Constants and layout.

#[test]
fn constants_match_the_references_written_out() {
    // native_qmd.h: Descriptor = std::array<Word, 64>; line 35's 4096; line 30's 256.
    assert_eq!(QMD_WORDS, 64);
    assert_eq!(QMD_BYTES, 256);
    assert_eq!(MAX_QMDS, 4096);
    assert_eq!(MAX_CHAIN, 256);
    // kepler_compute.h: NumConstBuffers = 8.
    assert_eq!(CONSTANT_BUFFERS, 8);
    assert_eq!(CONSTANT_BUFFER_SIZE_SHIFT, 4);
    assert_eq!(QMD_ADDRESS_SHIFT, 8);
    assert_eq!((QMD_MAJOR_VERSION << 4) | QMD_MINOR_VERSION, 0x30);
    // kepler_compute.cpp's method numbers, and clc7c0.h's byte addresses / 4.
    assert_eq!(LAST_INLINE_QMD_METHOD, 0x107);
    assert_eq!(LOAD_INLINE_QMD_DATA.method_at(&[63]), Some(0x107));
    assert_eq!(LOAD_INLINE_QMD_DATA.method(), 0x320 / 4);
    assert_eq!(LOAD_INLINE_QMD_DATA.method(), 0xc8);
    assert_eq!(LOAD_INLINE_QMD_DATA.method_at(&[4]), Some(0xcc));
    assert_eq!(SEND_PCAS_A.method(), 0xad);
    assert_eq!(SEND_SIGNALING_PCAS2_B.method(), 0xb0);
    assert_eq!(SET_QMD_VIRTUALIZATION_CONTROL.method(), 0xa7);
    assert_eq!(SET_INLINE_QMD_ADDRESS_A.method(), 0xc6);
    assert_eq!(SET_INLINE_QMD_ADDRESS_B.method(), 0xc7);
    assert_eq!(SET_QMD_VIRTUALIZATION_CONTROL_SEND_PCAS_ENABLE.low(), 24);
    // clc7c0.h's PCAS action values.
    assert_eq!(SEND_SIGNALING_PCAS2_B_PCAS_ACTION_NOP, 0);
    assert_eq!(SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE, 1);
    assert_eq!(SEND_SIGNALING_PCAS2_B_PCAS_ACTION_SCHEDULE, 2);
    assert_eq!(
        SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE,
        3
    );
    assert_eq!(SEND_SIGNALING_PCAS2_B_PCAS_ACTION_DECREMENT_DEPENDENCE, 7);
    assert_eq!(SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INCREMENT_DEPENDENCE, 12);
    // native_qmd.h line 118: ReleaseSlots{24, 28, 52}; flags in the next word, payload after.
    let words = |f: fn(&ReleaseSlot) -> MwField| SLOTS.map(|s| f(&s).low() / 32);
    assert_eq!(words(|s| s.address_lower), [24, 28, 52]);
    assert_eq!(words(|s| s.address_upper), [25, 29, 53]);
    assert_eq!(words(|s| s.enable), [25, 29, 53]);
    assert_eq!(words(|s| s.payload_lower), [26, 30, 54]);
    // Bits within the flags word: enable 23 (`1U << 23`), reduction enable 26 (`1U << 26`),
    // bit 19 (`1U << 19`), address upper 7:0 (`flags & 0xff`), structure size 31:30.
    for slot in SLOTS {
        assert_eq!(slot.enable.low() % 32, 23);
        assert_eq!(slot.reduction_enable.low() % 32, 26);
        assert_eq!(slot.membar_type.low() % 32, 19);
        assert_eq!(
            (
                slot.address_upper.low() % 32,
                slot.address_upper.high() % 32
            ),
            (0, 7)
        );
        assert_eq!(slot.structure_size.low() % 32, 30);
        assert_eq!(
            slot.reduction_op.low() % 32,
            sema::QMD_RELEASE_REDUCTION_OP.low()
        );
        assert_eq!(
            slot.reduction_format.low() % 32,
            sema::QMD_RELEASE_REDUCTION_FORMAT.low()
        );
    }
    assert_eq!(q::QMDV03_00_RELEASE0_STRUCTURE_SIZE_SEMAPHORE_ONE_WORD, 1);
    assert_eq!(q::QMDV03_00_RELEASE0_MEMBAR_TYPE_FE_SYSMEMBAR, 1);
    // The dependent QMD: pointer word 15, enable word 16 bit 0, action 3:1, QMD1 enable bit 5.
    assert_eq!(q::QMDV03_00_DEPENDENT_QMD0_POINTER.low(), 15 * 32);
    assert_eq!(q::QMDV03_00_DEPENDENT_QMD0_ENABLE.low(), 16 * 32);
    assert_eq!(q::QMDV03_00_DEPENDENT_QMD0_ACTION.low(), 16 * 32 + 1);
    assert_eq!(q::QMDV03_00_DEPENDENT_QMD1_ENABLE.low(), 16 * 32 + 5);
    assert_eq!(q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_SCHEDULE, 1);
    assert_eq!(
        q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_INVALIDATE_COPY_SCHEDULE,
        3
    );
    assert_eq!(
        q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_DECREMENT_DEPENDENCE,
        4
    );
    // Word 4: queue bit 8 (`qmd[4] & 0x100`), counter 29:14, REQUIRE_SCHEDULING_PCAS bit 12.
    assert_eq!(q::QMDV03_00_IS_QUEUE.low(), 4 * 32 + 8);
    assert_eq!(
        (
            q::QMDV03_00_DEPENDENCE_COUNTER.low(),
            q::QMDV03_00_DEPENDENCE_COUNTER.high()
        ),
        (4 * 32 + 14, 4 * 32 + 29)
    );
    assert_eq!(q::QMDV03_00_REQUIRE_SCHEDULING_PCAS.low(), 4 * 32 + 12);
    // `NVC7C0_QMDV03_00_REQUIRE_SCHEDULING_PCAS` is `MW(140:140)`, `_FALSE` 0 and `_TRUE` 1
    // (`clc7c0qmd.h` lines 504 to 506); `NVC7C0_LOAD_INLINE_QMD_DATA(i)` is `(0x0320+(i)*4)`
    // (`clc7c0.h`), so word 4 is method 0xCC, the `native_registers[0xcc]` of
    // `kepler_compute.cpp`.
    assert_eq!(
        (
            q::QMDV03_00_REQUIRE_SCHEDULING_PCAS.low(),
            q::QMDV03_00_REQUIRE_SCHEDULING_PCAS.high()
        ),
        (140, 140)
    );
    assert_eq!(q::QMDV03_00_REQUIRE_SCHEDULING_PCAS_FALSE, 0);
    assert_eq!(q::QMDV03_00_REQUIRE_SCHEDULING_PCAS_TRUE, 1);
    assert_eq!(LOAD_INLINE_QMD_DATA.method_at(&[4]), Some(0xcc));
}

#[test]
fn the_layout_leaves_room_for_eight_constant_buffers_in_64_words() {
    let first = q::QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER.element(0).unwrap();
    let room = q::QMDV03_00_PROGRAM_ADDRESS_LOWER.low() - first.low();
    assert_eq!(
        room / q::QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER.stride(),
        u32::try_from(CONSTANT_BUFFERS).unwrap()
    );
    assert_eq!(
        q::QMDV03_00_DEBUG_ID_LOWER.high() + 1,
        u32::try_from(QMD_WORDS).unwrap() * 32
    );
}

#[test]
fn every_field_read_lies_inside_the_qmd() {
    let mut fields = vec![
        q::QMDV03_00_QMD_MAJOR_VERSION,
        q::QMDV03_00_QMD_VERSION,
        q::QMDV03_00_IS_QUEUE,
        q::QMDV03_00_REQUIRE_SCHEDULING_PCAS,
        q::QMDV03_00_DEPENDENCE_COUNTER,
        q::QMDV03_00_PROGRAM_ADDRESS_LOWER,
        q::QMDV03_00_PROGRAM_ADDRESS_UPPER,
        q::QMDV03_00_CTA_RASTER_WIDTH,
        q::QMDV03_00_CTA_RASTER_HEIGHT,
        q::QMDV03_00_CTA_RASTER_DEPTH,
        q::QMDV03_00_CTA_THREAD_DIMENSION0,
        q::QMDV03_00_CTA_THREAD_DIMENSION1,
        q::QMDV03_00_CTA_THREAD_DIMENSION2,
        q::QMDV03_00_SHARED_MEMORY_SIZE,
        q::QMDV03_00_SHADER_LOCAL_MEMORY_LOW_SIZE,
        q::QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE,
        q::QMDV03_00_DEPENDENT_QMD0_ENABLE,
        q::QMDV03_00_DEPENDENT_QMD0_POINTER,
        q::QMDV03_00_DEPENDENT_QMD0_ACTION,
        q::QMDV03_00_DEPENDENT_QMD1_ENABLE,
    ];
    for slot in SLOTS {
        fields.extend([
            slot.address_lower,
            slot.address_upper,
            slot.membar_type,
            slot.reduction_op,
            slot.enable,
            slot.reduction_format,
            slot.reduction_enable,
            slot.trap_type,
            slot.payload64b,
            slot.structure_size,
            slot.payload_lower,
        ]);
    }
    for i in 0..u32::try_from(CONSTANT_BUFFERS).unwrap() {
        for array in [
            q::QMDV03_00_CONSTANT_BUFFER_VALID,
            q::QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER,
            q::QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER,
            q::QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4,
        ] {
            fields.push(array.element(i).unwrap());
        }
    }
    let words = [u32::MAX; QMD_WORDS];
    for field in fields {
        assert!(field.width().is_some(), "{field:?}");
        assert!(field.get(&words).is_some(), "{field:?}");
    }
}

/// The golden test: words laid out by hand from Eden-NX2's literal word indices and masks
/// (`DispatchNativeCompute`, `Signal`'s release and dependent code), with a distinct value in
/// every field and stray bits set around the fields wherever the header has other fields or
/// nothing, so a field read from the wrong word, at the wrong shift or with the wrong width
/// fails.
#[test]
fn golden_hand_laid_qmd_decodes_every_field() {
    let mut w = [0u32; QMD_WORDS];
    // Word 4: counter 0x2345 in 29:14; bits 31:30 (QMD_RESERVED04C, SELF_COPY) and 7:0 stray.
    w[4] = (0x2345 << 14) | 0xc000_00ff;
    // Grid: width all 32 bits, height and depth the low 16 bits of words 13 and 14.
    w[12] = 0x8765_4321;
    w[13] = 0xdead_0405;
    w[14] = 0xbeef_0607;
    // Dependent: pointer word 15; word 16 enable bit 0, action 3:1 = 4, stray bits above 5.
    w[15] = 0x0123_4567;
    w[16] = 1 | (4 << 1) | 0xffff_ffc0;
    // Shared memory 17:0 of word 17; stray SM config bits above.
    w[17] = 0x3_1234 | 0xfffc_0000;
    // Word 18: version 0x30 in 7:0, OCCUPANCY 15:8 stray, block x 31:16.
    w[18] = 0x30 | 0xab00 | (0x0809 << 16);
    // Word 19: block y 15:0, block z 31:16.
    w[19] = 0x0a0b | (0x0c0d << 16);
    // Word 20: constant buffer valid mask 7:0, REGISTER_COUNT_V and others stray above.
    w[20] = 0b1010_0101 | 0xffff_ff00;
    // Word 23: local memory low 23:0, BARRIER_COUNT stray above.
    w[23] = 0x12_3456 | 0xff00_0000;
    // Release slots: (address low, flags | address high, payload).
    w[24] = 0x1111_1110;
    w[25] = (1 << 23) | (1 << 30) | 0x21;
    w[26] = 0x3333_3333;
    w[28] = 0x4444_4440;
    w[29] = (1 << 23) | (1 << 30) | (1 << 26) | 0x22; // reduction RED_ADD
    w[30] = 0x5555_5555;
    w[52] = 0x6666_6660;
    w[53] = (1 << 23) | (1 << 30) | (1 << 19) | 0x23; // bit 19, no reduction
    w[54] = 0x7777_7777;
    // Constant buffers: low at 32 + 2i, high at 33 + 2i (17-bit upper, size 31:19, stray
    // PREFETCH_POST and INVALIDATE bits 17 and 18).
    for i in 0..8u32 {
        let i_usize = usize::try_from(i).unwrap();
        w[32 + 2 * i_usize] = 0x1000_0000 + i * 0x100;
        w[33 + 2 * i_usize] = (0x1_0010 + i) | (3 << 17) | ((0x100 + i) << 19);
    }
    // Program: low word 48, upper 16:0 of word 49, stray above.
    w[48] = 0xcafe_f00d;
    w[49] = 0x1_0bcd | 0xfffe_0000;
    // Local memory high 23:0 of word 50, stray above.
    w[50] = 0x65_4321 | 0xff00_0000;

    let address = qmd_at(3);
    let qmd = Qmd::decode(address, &w).unwrap();
    let mut constant_buffers = [ConstantBuffer::default(); CONSTANT_BUFFERS];
    for (i, cb) in (0u32..).zip(constant_buffers.iter_mut()) {
        *cb = ConstantBuffer {
            valid: (0b1010_0101 >> i) & 1 == 1,
            address: (u64::from(0x1_0010 + i) << 32) | u64::from(0x1000_0000 + i * 0x100),
            size: (0x100 + i) << 4,
        };
    }
    assert_eq!(
        qmd,
        Qmd {
            dependence_counter: 0x2345,
            dispatch: Dispatch {
                qmd: address,
                program: 0x1_0bcd_cafe_f00d,
                grid: [0x8765_4321, 0x0405, 0x0607],
                block: [0x0809, 0x0a0b, 0x0c0d],
                shared_memory: 0x3_1234,
                local_memory_low: 0x12_3456,
                local_memory_high: 0x65_4321,
                constant_buffers,
            },
            releases: [
                Some(write_release(0x21_1111_1110, 0x3333_3333)),
                Some(Release {
                    op: ReleaseOp::Reduce(Reduction {
                        op: ReductionOp::Add,
                        format: ReductionFormat::Unsigned
                    }),
                    ..write_release(0x22_4444_4440, 0x5555_5555)
                }),
                Some(Release {
                    op: ReleaseOp::Reduce(MAX_RELEASE),
                    ..write_release(0x23_6666_6660, 0x7777_7777)
                }),
            ],
            dependent: Some(Dependent {
                address: 0x01_2345_6700,
                action: 4
            }),
            // The stray bits of word 16 start at bit 6, above DEPENDENT_QMD1_ENABLE.
            second_dependent: false,
        }
    );
    // Bit 5 of word 16 (`qmd[16] & (1U << 5)`) is the second dependent.
    w[16] |= 1 << 5;
    assert!(Qmd::decode(address, &w).unwrap().second_dependent);
}

fn small_dispatch() -> impl Strategy<Value = Dispatch> {
    let cb = (any::<bool>(), 0u64..(1 << 49), 0u32..(1 << 13)).prop_map(|(valid, address, s)| {
        ConstantBuffer {
            valid,
            address,
            size: s << CONSTANT_BUFFER_SIZE_SHIFT,
        }
    });
    (
        any::<u64>(),
        0u64..(1 << 49),
        (any::<u32>(), 0u32..=0xffff, 0u32..=0xffff),
        (0u32..=0xffff, 0u32..=0xffff, 0u32..=0xffff),
        (0u32..(1 << 18), 0u32..(1 << 24), 0u32..(1 << 24)),
        prop::array::uniform8(cb),
    )
        .prop_map(|(qmd, program, grid, block, mem, cbs)| Dispatch {
            qmd,
            program,
            grid: [grid.0, grid.1, grid.2],
            block: [block.0, block.1, block.2],
            shared_memory: mem.0,
            local_memory_low: mem.1,
            local_memory_high: mem.2,
            constant_buffers: cbs,
        })
}

fn slot_spec() -> impl Strategy<Value = Option<SlotSpec>> {
    prop::option::of(
        (
            0u64..(1 << 40),
            any::<u32>(),
            0u32..=1,
            prop::option::of((0u32..8, 0u32..2)),
        )
            .prop_map(|(address, payload, membar, reduction)| SlotSpec {
                membar,
                reduction,
                ..SlotSpec::write(address, payload)
            }),
    )
}

proptest! {
    #![proptest_config(Config {
        cases: 1024,
        rng_seed: RngSeed::Fixed(0x716d_6430),
        failure_persistence: None,
        ..Config::default()
    })]

    /// Build-then-decode: every field the encoder writes comes back, so two reads that were
    /// swapped (or read from the wrong slot or element) fail.
    #[test]
    fn encode_then_decode_round_trips(
        dispatch in small_dispatch(),
        counter in any::<u16>(),
        slots in [slot_spec(), slot_spec(), slot_spec()],
        dependent in prop::option::of((0u64..(1 << 32), 0u32..8)),
        second in any::<bool>(),
    ) {
        let mut spec = Spec::new(0).counter(counter);
        spec.dispatch = dispatch;
        spec.slots = slots;
        spec.dependent = dependent.map(|(pointer, action)| Dependent {
            address: pointer << QMD_ADDRESS_SHIFT,
            action,
        });
        spec.second_dependent = second;
        let qmd = Qmd::decode(dispatch.qmd, &encode(&spec)).unwrap();
        prop_assert_eq!(qmd.dependence_counter, counter);
        prop_assert_eq!(qmd.dispatch, dispatch);
        prop_assert_eq!(qmd.dependent, spec.dependent);
        prop_assert_eq!(qmd.second_dependent, second);
        for (release, slot) in qmd.releases.iter().zip(&slots) {
            let expected = slot.map(|s| Release {
                op: match s.reduction {
                    Some((op, format)) => ReleaseOp::Reduce(
                        sema::red_reduction(QMD_RELEASE, op, format).unwrap(),
                    ),
                    None if s.membar == 1 => ReleaseOp::Reduce(MAX_RELEASE),
                    None => ReleaseOp::Write,
                },
                ..write_release(s.address, s.payload)
            });
            prop_assert_eq!(*release, expected);
        }
    }

    /// Any 64 words decode to a QMD or an error, never a panic.
    #[test]
    fn any_words_decode_or_fail(words in prop::array::uniform32(any::<u32>()), more in prop::array::uniform32(any::<u32>())) {
        let mut all = [0u32; QMD_WORDS];
        all[..32].copy_from_slice(&words);
        all[32..].copy_from_slice(&more);
        let _ = Qmd::decode(0, &all);
    }
}

// ---------------------------------------------------------------------------------------------
// Header checks.

#[test]
fn version_and_queue_are_checked_in_order() {
    let address = qmd_at(1);
    let decode = |spec: Spec| Qmd::decode(address, &encode(&spec)).map(|_| ());
    assert_eq!(decode(Spec::new(0)), Ok(()));
    for (major, minor) in [(2, 0), (4, 0), (3, 1), (0, 3)] {
        let spec = Spec {
            major,
            minor,
            ..Spec::new(0)
        };
        assert_eq!(
            decode(spec),
            Err(QmdError::Version {
                address,
                major,
                minor
            })
        );
    }
    let queue = Spec {
        queue: true,
        ..Spec::new(0)
    };
    assert_eq!(decode(queue), Err(QmdError::Queue { address }));
    // Both faults: the version is reported.
    let both = Spec { major: 2, ..queue };
    assert_eq!(
        decode(both),
        Err(QmdError::Version {
            address,
            major: 2,
            minor: 0
        })
    );
}

#[test]
fn release_refusals_each_and_in_order() {
    let address = qmd_at(1);
    let good = SlotSpec::write(target(0), 7);
    let refusal =
        |slot: SlotSpec| Qmd::decode(address, &encode(&Spec::new(0).slot(1, slot))).map(|_| ());
    let err = |refusal| {
        Err(QmdError::Release {
            address,
            slot: 1,
            refusal,
        })
    };
    assert_eq!(refusal(good), Ok(()));
    for size in [0, 2, 3] {
        let slot = SlotSpec {
            structure_size: size,
            ..good
        };
        assert_eq!(refusal(slot), err(ReleaseRefusal::StructureSize(size)));
    }
    let wide = SlotSpec {
        payload64b: 1,
        ..good
    };
    assert_eq!(refusal(wide), err(ReleaseRefusal::Payload64));
    for trap in [1, 2, 3] {
        let slot = SlotSpec { trap, ..good };
        assert_eq!(refusal(slot), err(ReleaseRefusal::Trap(trap)));
    }
    for format in [2, 3] {
        let slot = SlotSpec {
            reduction: Some((sema::RED_ADD, format)),
            ..good
        };
        assert_eq!(
            refusal(slot),
            err(ReleaseRefusal::Reduction(DecodeError::UndefinedFormat {
                word: QMD_RELEASE,
                value: format
            }))
        );
    }
    // Two faults at once: the earlier check wins.
    let size_and_wide = SlotSpec {
        structure_size: 0,
        payload64b: 1,
        ..good
    };
    assert_eq!(
        refusal(size_and_wide),
        err(ReleaseRefusal::StructureSize(0))
    );
    let wide_and_trap = SlotSpec {
        payload64b: 1,
        trap: 1,
        ..good
    };
    assert_eq!(refusal(wide_and_trap), err(ReleaseRefusal::Payload64));
    let trap_and_format = SlotSpec {
        trap: 2,
        reduction: Some((sema::RED_ADD, 3)),
        ..good
    };
    assert_eq!(refusal(trap_and_format), err(ReleaseRefusal::Trap(2)));
    // A disabled slot is not looked at.
    let mut words = encode(&Spec::new(0).slot(1, wide));
    set(&mut words, SLOTS[1].enable, 0);
    assert_eq!(Qmd::decode(address, &words).unwrap().releases, [None; 3]);
}

#[test]
fn release_slots_are_checked_in_slot_order() {
    let address = qmd_at(1);
    let bad = |trap| SlotSpec {
        trap,
        ..SlotSpec::write(target(0), 1)
    };
    let spec = Spec::new(0).slot(0, bad(1)).slot(2, bad(2));
    assert_eq!(
        Qmd::decode(address, &encode(&spec)),
        Err(QmdError::Release {
            address,
            slot: 0,
            refusal: ReleaseRefusal::Trap(1)
        })
    );
    let spec = Spec::new(0).slot(1, bad(3)).slot(2, bad(2));
    assert_eq!(
        Qmd::decode(address, &encode(&spec)),
        Err(QmdError::Release {
            address,
            slot: 1,
            refusal: ReleaseRefusal::Trap(3)
        })
    );
}

// ---------------------------------------------------------------------------------------------
// Scheduling.

#[test]
fn a_fresh_table_is_empty() {
    let table = QmdTable::new();
    assert_eq!(table.len(), 0);
    assert!(table.is_empty());
    assert_eq!(table.status(qmd_at(0)), None);
    assert_eq!(
        QmdStatus::default(),
        QmdStatus {
            loaded: false,
            scheduled: false,
            completed: false,
            dependencies: 0,
            deferred: 0
        }
    );
}

#[test]
fn a_run_emits_dispatch_releases_in_slot_order_then_completion_then_the_dependent() {
    let mem = memory();
    let (a, b) = (qmd_at(1), qmd_at(2));
    let spec_a = Spec::new(0xa000)
        .slot(0, SlotSpec::write(target(0), 10))
        .slot(1, SlotSpec::write(target(1), 11))
        .slot(2, SlotSpec::write(target(2), 12))
        .dependent(b, q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_SCHEDULE);
    let spec_b = Spec::new(0xb000).slot(2, SlotSpec::write(target(3), 13));
    put(&mem, a, &spec_a);
    put(&mem, b, &spec_b);
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(
        sink.events,
        [
            Event::Dispatch(dispatch_of(&spec_a, a)),
            Event::Release(write_release(target(0), 10)),
            Event::Release(write_release(target(1), 11)),
            Event::Release(write_release(target(2), 12)),
            Event::Completed(a, Outcome::Dispatched),
            Event::Dispatch(dispatch_of(&spec_b, b)),
            Event::Release(write_release(target(3), 13)),
            Event::Completed(b, Outcome::Dispatched),
        ]
    );
    for qmd in [a, b] {
        assert_eq!(
            table.status(qmd),
            Some(QmdStatus {
                loaded: true,
                scheduled: true,
                completed: true,
                dependencies: 0,
                deferred: 0
            })
        );
    }
}

#[test]
fn a_qmd_waits_until_its_counter_reaches_zero() {
    let mem = memory();
    let a = qmd_at(1);
    put(&mem, a, &Spec::new(0xa000).counter(2));
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(sink.events, NOTHING);
    assert_eq!(table.status(a).unwrap().dependencies, 2);
    table.signal(a, dec(1), &mem, &mut sink).unwrap();
    assert_eq!(sink.events, NOTHING, "one above zero does not run");
    assert_eq!(table.status(a).unwrap().dependencies, 1);
    table.signal(a, dec(1), &mem, &mut sink).unwrap();
    assert_eq!(sink.completions(), [(a, Outcome::Dispatched)]);
}

#[test]
fn a_decrement_by_more_than_one_and_an_increment_move_the_counter_by_their_count() {
    let mem = memory();
    let a = qmd_at(1);
    put(&mem, a, &Spec::new(0xa000).counter(3));
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    table.signal(a, inc(5), &mem, &mut sink).unwrap();
    assert_eq!(table.status(a).unwrap().dependencies, 8);
    table.signal(a, dec(7), &mem, &mut sink).unwrap();
    assert_eq!(table.status(a).unwrap().dependencies, 1);
    assert_eq!(sink.events, NOTHING);
    table.signal(a, dec(1), &mem, &mut sink).unwrap();
    assert_eq!(sink.completions(), [(a, Outcome::Dispatched)]);
}

/// Row 84: one channel decrements a QMD before another schedules it. The table is shared, so
/// the decrement is deferred and applied when the schedule loads the QMD, which then runs.
#[test]
fn a_decrement_from_another_channel_before_the_schedule_is_kept() {
    let mem = memory();
    let a = qmd_at(5);
    put(&mem, a, &Spec::new(0xa000).counter(1));
    let pcas_a = usize::try_from(SEND_PCAS_A.method()).unwrap();
    let pcas2_b = usize::try_from(SEND_SIGNALING_PCAS2_B.method()).unwrap();
    let address_word = u32::try_from(a >> QMD_ADDRESS_SHIFT).unwrap();
    // Two channels, each with its own compute register file.
    let mut channel_a = vec![0u32; REGISTERS_NEEDED];
    let mut channel_b = vec![0u32; REGISTERS_NEEDED];
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();

    channel_b[pcas_a] = address_word;
    channel_b[pcas2_b] = dec(1).encode().unwrap();
    table
        .compute_method(SEND_SIGNALING_PCAS2_B.method(), &channel_b, &mem, &mut sink)
        .unwrap();
    assert_eq!(sink.events, NOTHING);
    assert_eq!(
        table.status(a),
        Some(QmdStatus {
            deferred: -1,
            ..QmdStatus::default()
        })
    );

    channel_a[pcas_a] = address_word;
    channel_a[pcas2_b] = PcasAction::Schedule.encode().unwrap();
    table
        .compute_method(SEND_SIGNALING_PCAS2_B.method(), &channel_a, &mem, &mut sink)
        .unwrap();
    assert_eq!(sink.completions(), [(a, Outcome::Dispatched)]);
    assert_eq!(table.status(a).unwrap().deferred, 0);
}

/// The unchanged outcome of the case above: with no decrement first, the schedule waits.
#[test]
fn without_the_early_decrement_the_schedule_waits() {
    let mem = memory();
    let a = qmd_at(5);
    put(&mem, a, &Spec::new(0xa000).counter(1));
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(sink.events, NOTHING);
    assert_eq!(table.status(a).unwrap().dependencies, 1);
}

#[test]
fn schedule_loads_only_a_qmd_not_loaded_but_invalidate_copy_schedule_always_reloads() {
    let mem = memory();
    let a = qmd_at(1);
    put(&mem, a, &Spec::new(0xa000).counter(1));
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    // The counter in memory changes; a second schedule does not read it.
    put(&mem, a, &Spec::new(0xa000).counter(5));
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(table.status(a).unwrap().dependencies, 1);
    table
        .signal(a, PcasAction::InvalidateCopySchedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(table.status(a).unwrap().dependencies, 5);
    assert_eq!(sink.events, NOTHING);
}

#[test]
fn a_completed_qmd_runs_again_only_when_reloaded() {
    let mem = memory();
    let a = qmd_at(1);
    put(&mem, a, &Spec::new(0xa000));
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(sink.completions(), [(a, Outcome::Dispatched)]);
    table
        .signal(a, PcasAction::InvalidateCopySchedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(
        sink.completions(),
        [(a, Outcome::Dispatched), (a, Outcome::Dispatched)]
    );
}

#[test]
fn changes_to_a_completed_qmd_are_deferred_to_its_next_load() {
    let mem = memory();
    let a = qmd_at(1);
    put(&mem, a, &Spec::new(0xa000));
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    table.signal(a, inc(3), &mem, &mut sink).unwrap();
    table.signal(a, dec(1), &mem, &mut sink).unwrap();
    let status = table.status(a).unwrap();
    assert_eq!((status.dependencies, status.deferred), (0, 2));
    put(&mem, a, &Spec::new(0xa000).counter(1));
    table
        .signal(a, PcasAction::InvalidateCopySchedule, &mem, &mut sink)
        .unwrap();
    let status = table.status(a).unwrap();
    assert_eq!((status.dependencies, status.deferred), (3, 0));
    assert!(!status.completed);
    table.signal(a, dec(3), &mem, &mut sink).unwrap();
    assert_eq!(sink.completions().len(), 2);
}

#[test]
fn a_qmd_is_read_again_when_it_runs_so_a_patched_payload_is_released() {
    let mem = memory();
    let a = qmd_at(1);
    let spec = Spec::new(0xa000)
        .counter(1)
        .slot(2, SlotSpec::write(target(0), 0x1111));
    put(&mem, a, &spec);
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    // NVN2 patches the payload while the QMD waits.
    let patched = spec.slot(2, SlotSpec::write(target(0), 0x2222));
    put(&mem, a, &patched);
    table.signal(a, dec(1), &mem, &mut sink).unwrap();
    assert_eq!(sink.releases(), [write_release(target(0), 0x2222)]);
}

#[test]
fn a_counter_below_zero_when_ready_is_an_error_and_zero_runs() {
    let mem = memory();
    let a = qmd_at(1);
    put(&mem, a, &Spec::new(0xa000).counter(1));
    let mut sink = Recorder::default();

    let mut table = QmdTable::new();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    table.signal(a, dec(1), &mem, &mut sink).unwrap();
    assert_eq!(sink.completions(), [(a, Outcome::Dispatched)]);

    let mut sink = Recorder::default();
    let mut table = QmdTable::new();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(
        table.signal(a, dec(2), &mem, &mut sink),
        Err(QmdError::DependenceUnderflow {
            address: a,
            counter: -1
        })
    );
    assert_eq!(sink.events, NOTHING);
    assert_eq!(table.status(a).unwrap().dependencies, -1);
}

#[test]
fn a_deferred_counter_below_zero_is_not_an_error_until_it_loads() {
    let mem = memory();
    let a = qmd_at(1);
    put(&mem, a, &Spec::new(0xa000).counter(1));
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table.signal(a, dec(3), &mem, &mut sink).unwrap();
    assert_eq!(table.status(a).unwrap().deferred, -3);
    assert_eq!(
        table.signal(a, PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::DependenceUnderflow {
            address: a,
            counter: -2
        })
    );
}

#[test]
fn counter_overflow_is_an_error_at_each_edge() {
    let mem = memory();
    let a = qmd_at(1);
    let mut sink = Recorder::default();
    let overflow = Err(QmdError::DependenceOverflow { address: a });
    let seeded = |status: QmdStatus| {
        let mut table = QmdTable::new();
        table.states.insert(a, status);
        table
    };

    // The deferred change, while not loaded.
    let mut table = seeded(QmdStatus {
        deferred: i32::MIN + 1,
        ..QmdStatus::default()
    });
    assert_eq!(table.signal(a, dec(1), &mem, &mut sink), Ok(()));
    assert_eq!(table.status(a).unwrap().deferred, i32::MIN);
    assert_eq!(table.signal(a, dec(1), &mem, &mut sink), overflow);
    assert_eq!(table.status(a).unwrap().deferred, i32::MIN, "unchanged");

    // The counter of a loaded QMD.
    let loaded = QmdStatus {
        loaded: true,
        dependencies: i32::MAX - 1,
        ..QmdStatus::default()
    };
    let mut table = seeded(loaded);
    assert_eq!(table.signal(a, inc(1), &mem, &mut sink), Ok(()));
    assert_eq!(table.signal(a, inc(1), &mem, &mut sink), overflow);
    assert_eq!(table.status(a).unwrap().dependencies, i32::MAX);

    // The counter plus the deferred change at load.
    put(&mem, a, &Spec::new(0xa000).counter(1));
    let mut table = seeded(QmdStatus {
        deferred: i32::MAX - 1,
        ..QmdStatus::default()
    });
    assert_eq!(
        table.signal(a, PcasAction::Schedule, &mem, &mut sink),
        Ok(())
    );
    assert_eq!(table.status(a).unwrap().dependencies, i32::MAX);
    let mut table = seeded(QmdStatus {
        deferred: i32::MAX,
        ..QmdStatus::default()
    });
    assert_eq!(
        table.signal(a, PcasAction::Schedule, &mem, &mut sink),
        overflow
    );
    assert_eq!(
        table.status(a),
        Some(QmdStatus {
            deferred: i32::MAX,
            ..QmdStatus::default()
        })
    );
    assert_eq!(sink.events, NOTHING);
}

// ---------------------------------------------------------------------------------------------
// Outcomes.

/// Row 61: a QMD whose program cannot run still completes and releases, and its dependent
/// still applies.
#[test]
fn a_rejected_program_still_completes_and_releases() {
    let mem = memory();
    let (a, b) = (qmd_at(1), qmd_at(2));
    let spec_a = Spec::new(0xbad0)
        .slot(0, SlotSpec::write(target(0), 1))
        .dependent(b, q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_SCHEDULE);
    put(&mem, a, &spec_a);
    put(&mem, b, &Spec::new(0xb000));
    let mut table = QmdTable::new();
    let mut sink = Recorder {
        reject: vec![0xbad0],
        ..Recorder::default()
    };
    // A listed divergence: Eden-NX2's `Signal` reports "native shader dispatch required" here.
    assert_eq!(
        table.signal(a, PcasAction::Schedule, &mem, &mut sink),
        Ok(())
    );
    assert_eq!(
        sink.events[..3],
        [
            Event::Dispatch(dispatch_of(&spec_a, a)),
            Event::Release(write_release(target(0), 1)),
            Event::Completed(a, Outcome::Rejected),
        ]
    );
    assert_eq!(
        sink.completions(),
        [(a, Outcome::Rejected), (b, Outcome::Dispatched)]
    );
    assert!(table.status(a).unwrap().completed);
}

#[test]
fn an_empty_grid_or_block_completes_and_releases_without_reaching_the_sink() {
    for (dim, empty) in [
        (0, true),
        (1, true),
        (2, true),
        (0, false),
        (1, false),
        (2, false),
    ] {
        let mem = memory();
        let a = qmd_at(1);
        let mut spec = Spec::new(0xa000).slot(1, SlotSpec::write(target(0), 9));
        spec.dispatch.grid = [3, 4, 5];
        spec.dispatch.block = [6, 7, 8];
        if empty {
            spec.dispatch.grid[dim] = 0;
        } else {
            spec.dispatch.block[dim] = 0;
        }
        put(&mem, a, &spec);
        let mut table = QmdTable::new();
        let mut sink = Recorder::default();
        // An empty block is a listed divergence: Eden-NX2's `Signal` reports it as an error.
        assert_eq!(
            table.signal(a, PcasAction::Schedule, &mem, &mut sink),
            Ok(()),
            "dim {dim}, grid {empty}"
        );
        let outcome = if empty {
            Outcome::EmptyGrid
        } else {
            Outcome::EmptyBlock
        };
        assert_eq!(
            sink.events,
            [
                Event::Release(write_release(target(0), 9)),
                Event::Completed(a, outcome)
            ],
            "dim {dim}, grid {empty}"
        );
    }
}

/// The grid is checked before the block (`native_qmd.h` line 89 tests the grid, and
/// `DispatchNativeCompute` refuses an empty block only after that): with both empty, the run is
/// an empty grid, and the sink sees no dispatch.
#[test]
fn a_zero_grid_and_a_zero_block_together_are_an_empty_grid() {
    for dim in 0..3 {
        let mem = memory();
        let a = qmd_at(1);
        let mut spec = Spec::new(0xa000).slot(0, SlotSpec::write(target(0), 5));
        spec.dispatch.grid = [3, 4, 5];
        spec.dispatch.block = [6, 7, 8];
        spec.dispatch.grid[dim] = 0;
        spec.dispatch.block[2 - dim] = 0;
        put(&mem, a, &spec);
        let mut table = QmdTable::new();
        let mut sink = Recorder::default();
        assert_eq!(
            table.signal(a, PcasAction::Schedule, &mem, &mut sink),
            Ok(()),
            "dim {dim}"
        );
        assert_eq!(
            sink.events,
            [
                Event::Release(write_release(target(0), 5)),
                Event::Completed(a, Outcome::EmptyGrid)
            ],
            "dim {dim}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Dependent QMDs.

#[test]
fn each_dependent_action_maps_to_its_pcas_action() {
    let qmd = qmd_at(0);
    let dep = |action| Dependent {
        address: qmd_at(1),
        action,
    };
    assert_eq!(
        dep(q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_SCHEDULE).pcas_action(qmd),
        Ok(PcasAction::Schedule)
    );
    assert_eq!(
        dep(q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_INVALIDATE_COPY_SCHEDULE).pcas_action(qmd),
        Ok(PcasAction::InvalidateCopySchedule)
    );
    assert_eq!(
        dep(q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_DECREMENT_DEPENDENCE).pcas_action(qmd),
        Ok(dec(1))
    );
    for action in [0, 2, 5, 6, 7] {
        assert_eq!(
            dep(action).pcas_action(qmd),
            Err(QmdError::DependentAction {
                address: qmd,
                action
            })
        );
    }
}

#[test]
fn a_dependent_decrement_releases_a_waiting_qmd() {
    let mem = memory();
    let (a, b) = (qmd_at(1), qmd_at(2));
    put(&mem, b, &Spec::new(0xb000).counter(1));
    put(
        &mem,
        a,
        &Spec::new(0xa000).dependent(
            b,
            q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_DECREMENT_DEPENDENCE,
        ),
    );
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .signal(b, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(sink.events, NOTHING);
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(
        sink.completions(),
        [(a, Outcome::Dispatched), (b, Outcome::Dispatched)]
    );
}

#[test]
fn a_dependent_invalidate_copy_schedule_reloads_the_dependent() {
    let mem = memory();
    let (a, b) = (qmd_at(1), qmd_at(2));
    put(&mem, b, &Spec::new(0xb000));
    put(
        &mem,
        a,
        &Spec::new(0xa000).dependent(
            b,
            q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_INVALIDATE_COPY_SCHEDULE,
        ),
    );
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    // B has run once; a plain schedule would not run it again, a reload does.
    table
        .signal(b, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(
        sink.completions(),
        [
            (b, Outcome::Dispatched),
            (a, Outcome::Dispatched),
            (b, Outcome::Dispatched)
        ]
    );
}

#[test]
fn dependent_faults_are_reported_after_the_qmd_completed_and_released() {
    let mem = memory();
    let (a, b) = (qmd_at(1), qmd_at(2));
    put(&mem, b, &Spec::new(0xb000));
    let base = Spec::new(0xa000).slot(0, SlotSpec::write(target(0), 4));
    let ran = [
        Event::Dispatch(dispatch_of(&base, a)),
        Event::Release(write_release(target(0), 4)),
        Event::Completed(a, Outcome::Dispatched),
    ];
    let cases = [
        // An unsupported action.
        (
            base.dependent(b, 0),
            Err(QmdError::DependentAction {
                address: a,
                action: 0,
            }),
        ),
        // A second dependent alone.
        (
            Spec {
                second_dependent: true,
                ..base
            },
            Err(QmdError::SecondDependent { address: a }),
        ),
        // Both: the action is reported first.
        (
            Spec {
                second_dependent: true,
                ..base.dependent(b, 2)
            },
            Err(QmdError::DependentAction {
                address: a,
                action: 2,
            }),
        ),
        // A good dependent with a second one: the dependent is not applied.
        (
            Spec {
                second_dependent: true,
                ..base.dependent(b, q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_SCHEDULE)
            },
            Err(QmdError::SecondDependent { address: a }),
        ),
    ];
    for (spec, expected) in cases {
        put(&mem, a, &spec);
        let mut table = QmdTable::new();
        let mut sink = Recorder::default();
        assert_eq!(
            table.signal(a, PcasAction::Schedule, &mem, &mut sink),
            expected
        );
        assert_eq!(sink.events, ran, "{spec:?}");
        assert!(table.status(a).unwrap().completed);
        assert_eq!(table.status(b), None, "the dependent was not applied");
    }
}

// ---------------------------------------------------------------------------------------------
// Limits.

/// A chain of `len` QMDs from QMD 0, each scheduling the next.
fn chain(mem: &Memory, len: u64) {
    for n in 0..len {
        let mut spec = Spec::new(0x1000 + n);
        if n + 1 < len {
            spec = spec.dependent(
                qmd_at(n + 1),
                q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_SCHEDULE,
            );
        }
        put(mem, qmd_at(n), &spec);
    }
}

#[test]
fn a_chain_of_exactly_the_limit_runs() {
    let mem = memory();
    chain(&mem, MAX_CHAIN as u64);
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .signal(qmd_at(0), PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(sink.completions().len(), MAX_CHAIN);
    assert_eq!(table.len(), MAX_CHAIN);
}

#[test]
fn a_chain_one_past_the_limit_is_refused_after_the_limit_ran() {
    let mem = memory();
    let len = MAX_CHAIN as u64 + 1;
    chain(&mem, len);
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    assert_eq!(
        table.signal(qmd_at(0), PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::ChainTooLong {
            address: qmd_at(len - 1)
        })
    );
    assert_eq!(sink.completions().len(), MAX_CHAIN);
    assert_eq!(table.status(qmd_at(len - 1)), None);
}

/// A QMD that reloads itself as its own dependent runs once per request until the chain limit.
#[test]
fn a_self_reloading_qmd_stops_at_the_chain_limit() {
    let mem = memory();
    let a = qmd_at(0);
    put(
        &mem,
        a,
        &Spec::new(0xa000).dependent(
            a,
            q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_INVALIDATE_COPY_SCHEDULE,
        ),
    );
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    assert_eq!(
        table.signal(a, PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::ChainTooLong { address: a })
    );
    assert_eq!(sink.completions().len(), MAX_CHAIN);
}

/// A chain at the limit whose last QMD also enables a second dependent: the chain limit is
/// checked first.
#[test]
fn the_chain_limit_is_reported_before_a_second_dependent() {
    let mem = memory();
    let a = qmd_at(0);
    put(
        &mem,
        a,
        &Spec {
            second_dependent: true,
            ..Spec::new(0xa000).dependent(
                a,
                q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_INVALIDATE_COPY_SCHEDULE,
            )
        },
    );
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    // The first run already has a second dependent: that error comes before any chain limit.
    assert_eq!(
        table.signal(a, PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::SecondDependent { address: a })
    );
    // At the limit, with an unsupported second dependent present on the last run only.
    chain(&mem, MAX_CHAIN as u64 + 1);
    let last = qmd_at(MAX_CHAIN as u64 - 1);
    put(
        &mem,
        last,
        &Spec {
            second_dependent: true,
            ..Spec::new(0x2000).dependent(
                qmd_at(MAX_CHAIN as u64),
                q::QMDV03_00_DEPENDENT_QMD0_ACTION_QMD_SCHEDULE,
            )
        },
    );
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    assert_eq!(
        table.signal(qmd_at(0), PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::ChainTooLong {
            address: qmd_at(MAX_CHAIN as u64)
        })
    );
}

/// The last QMD a chain allows has an unsupported dependent action: the action is reported,
/// not the chain limit (`native_qmd.h` line 107 comes before line 108).
#[test]
fn an_unsupported_dependent_action_is_reported_before_the_chain_limit() {
    let mem = memory();
    chain(&mem, MAX_CHAIN as u64);
    let last = qmd_at(MAX_CHAIN as u64 - 1);
    put(
        &mem,
        last,
        &Spec::new(0x2000).dependent(qmd_at(MAX_CHAIN as u64), 0),
    );
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    assert_eq!(
        table.signal(qmd_at(0), PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::DependentAction {
            address: last,
            action: 0
        })
    );
    assert_eq!(sink.completions().len(), MAX_CHAIN);
}

#[test]
fn the_table_holds_exactly_the_limit() {
    let mem = memory();
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    // Deferred decrements add entries without reading memory.
    let at = |n: u64| n << QMD_ADDRESS_SHIFT;
    for n in 0..MAX_QMDS as u64 - 1 {
        table.signal(at(n), dec(1), &mem, &mut sink).unwrap();
    }
    assert_eq!(table.len(), MAX_QMDS - 1);
    // One below the limit: a new QMD is added.
    let last = at(MAX_QMDS as u64 - 1);
    assert_eq!(table.signal(last, dec(1), &mem, &mut sink), Ok(()));
    assert_eq!(table.len(), MAX_QMDS);
    // At the limit: a new one is refused before anything else (here, before its read), and a
    // known one is still served.
    let new = at(MAX_QMDS as u64);
    assert_eq!(
        table.signal(new, PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::TableFull { address: new })
    );
    assert_eq!(
        table.signal(new, dec(1), &mem, &mut sink),
        Err(QmdError::TableFull { address: new })
    );
    assert_eq!(table.status(new), None);
    assert_eq!(table.signal(last, dec(1), &mem, &mut sink), Ok(()));
    assert_eq!(table.status(last).unwrap().deferred, -2);
    assert_eq!(table.len(), MAX_QMDS);
    assert_eq!(sink.events, NOTHING);
}

// ---------------------------------------------------------------------------------------------
// Reading QMDs.

#[test]
fn an_unreadable_qmd_is_an_error_naming_the_first_unmapped_byte() {
    let mem = memory();
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    let outside = GPU_END + PAGE_SIZE;
    assert_eq!(
        table.signal(outside, PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::Read {
            address: outside,
            fault: GpuFault::Unmapped { va: outside }
        })
    );
    // A QMD whose last half lies past the mapping.
    let straddling = GPU_END - 128;
    assert_eq!(
        table.signal(straddling, PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::Read {
            address: straddling,
            fault: GpuFault::Unmapped { va: GPU_END }
        })
    );
    // One that ends exactly at the end of the mapping is read.
    let last = GPU_END - QMD_BYTES as u64;
    put(&mem, last, &Spec::new(0xa000));
    assert_eq!(
        table.signal(last, PcasAction::Schedule, &mem, &mut sink),
        Ok(())
    );
    // Past the end of the address space.
    let top = u64::MAX - 127;
    assert_eq!(
        table.signal(top, PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::Read {
            address: top,
            fault: GpuFault::OutOfRange {
                va: top,
                len: QMD_BYTES as u64
            }
        })
    );
}

#[test]
fn a_qmd_that_goes_bad_while_waiting_fails_when_it_would_run() {
    let mem = memory();
    let a = qmd_at(1);
    put(&mem, a, &Spec::new(0xa000).counter(1));
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    put(
        &mem,
        a,
        &Spec {
            queue: true,
            ..Spec::new(0xa000)
        },
    );
    assert_eq!(
        table.signal(a, dec(1), &mem, &mut sink),
        Err(QmdError::Queue { address: a })
    );
    let bad_release = Spec::new(0xa000).slot(
        2,
        SlotSpec {
            trap: 1,
            ..SlotSpec::write(target(0), 1)
        },
    );
    put(&mem, a, &bad_release);
    // It is still ready (loaded, scheduled, counter zero): an increment and a decrement rerun it.
    table.signal(a, inc(1), &mem, &mut sink).unwrap();
    assert_eq!(
        table.signal(a, dec(1), &mem, &mut sink),
        Err(QmdError::Release {
            address: a,
            slot: 2,
            refusal: ReleaseRefusal::Trap(1)
        })
    );
    assert_eq!(sink.events, NOTHING, "nothing was made visible");
    assert!(!table.status(a).unwrap().completed);
}

/// A request that fails after its entry is written keeps its own change, as in Eden-NX2,
/// which updates `states[address]` in place before it reads the QMD again to run it. Here a QMD
/// new to the table loads (its version and queue flag pass), is scheduled with its counter at
/// zero, and is refused at run time by a release slot: it keeps a new entry, loaded and
/// scheduled but not completed, and nothing reaches the sink.
#[test]
fn a_failed_run_keeps_the_requests_own_change() {
    let mem = memory();
    let a = qmd_at(1);
    let bad_release = Spec::new(0xa000).slot(
        0,
        SlotSpec {
            trap: 1,
            ..SlotSpec::write(target(0), 1)
        },
    );
    put(&mem, a, &bad_release);
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    assert_eq!(
        table.signal(a, PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::Release {
            address: a,
            slot: 0,
            refusal: ReleaseRefusal::Trap(1)
        })
    );
    assert_eq!(table.len(), 1);
    assert_eq!(
        table.status(a),
        Some(QmdStatus {
            loaded: true,
            scheduled: true,
            completed: false,
            dependencies: 0,
            deferred: 0,
        })
    );
    assert_eq!(sink.events, NOTHING);
}

// ---------------------------------------------------------------------------------------------
// PCAS words.

#[test]
fn every_pcas_action_value_decodes_as_the_header_names_it() {
    for action in 0..=SEND_SIGNALING_PCAS2_B_PCAS_ACTION.max() {
        // Count field 2 (a count of 3), and SELECT bits set, which are ignored.
        let word = action | (2 << 14) | (0x3f << 8);
        let expected = match action {
            0 | 1 => Ok(None),
            2 => Ok(Some(PcasAction::Schedule)),
            3 => Ok(Some(PcasAction::InvalidateCopySchedule)),
            7 => Ok(Some(dec(3))),
            12 => Ok(Some(inc(3))),
            _ => Err(QmdError::PcasAction { action }),
        };
        assert_eq!(PcasAction::decode(word), expected, "action {action}");
    }
}

#[test]
fn the_count_field_runs_from_one_to_1024() {
    assert_eq!(PcasAction::decode(7), Ok(Some(dec(1))));
    assert_eq!(PcasAction::decode(0x7 | (0x3ff << 14)), Ok(Some(dec(1024))));
    // Bit 24 is outside the field.
    assert_eq!(PcasAction::decode(0xc | (1 << 24)), Ok(Some(inc(1))));
    assert_eq!(dec(1).encode(), Some(7));
    assert_eq!(dec(1024).encode(), Some(0x7 | (0x3ff << 14)));
    assert_eq!(inc(2).encode(), Some(0xc | (1 << 14)));
    assert_eq!(dec(1025).encode(), None);
    assert_eq!(inc(0).encode(), None);
    assert_eq!(PcasAction::Schedule.encode(), Some(2));
    assert_eq!(PcasAction::InvalidateCopySchedule.encode(), Some(3));
    for action in [
        PcasAction::Schedule,
        PcasAction::InvalidateCopySchedule,
        dec(1),
        dec(513),
        inc(1024),
    ] {
        assert_eq!(
            PcasAction::decode(action.encode().unwrap()),
            Ok(Some(action))
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Compute methods.

/// A register file with an inline QMD destined for `address`, `control` in
/// `SET_QMD_VIRTUALIZATION_CONTROL` and the words of `spec`.
fn inline_registers(address: u64, control: u32, spec: &Spec) -> Vec<u32> {
    let mut regs = vec![0u32; REGISTERS_NEEDED];
    let index = |method: u32| usize::try_from(method).unwrap();
    let shifted = address >> QMD_ADDRESS_SHIFT;
    regs[index(SET_INLINE_QMD_ADDRESS_A.method())] = u32::try_from(shifted >> 32).unwrap();
    regs[index(SET_INLINE_QMD_ADDRESS_B.method())] = lo(shifted);
    regs[index(SET_QMD_VIRTUALIZATION_CONTROL.method())] = control;
    let first = index(LOAD_INLINE_QMD_DATA.method());
    regs[first..first + QMD_WORDS].copy_from_slice(&encode(spec));
    regs
}

const SEND_PCAS: u32 = 1 << 24;

#[test]
fn the_last_inline_word_writes_the_qmd_and_schedules_it_when_send_pcas_is_enabled() {
    let mem = memory();
    let a = qmd_at(9);
    let spec = Spec::new(0xa000).slot(0, SlotSpec::write(target(0), 0x55));
    let regs = inline_registers(a, SEND_PCAS, &spec);
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .compute_method(LAST_INLINE_QMD_METHOD, &regs, &mem, &mut sink)
        .unwrap();
    let mut written = vec![0u8; QMD_BYTES];
    mem.read(a, &mut written).unwrap();
    assert_eq!(written, to_bytes(&encode(&spec)));
    assert_eq!(
        sink.events,
        [
            Event::Dispatch(dispatch_of(&spec, a)),
            Event::Release(write_release(target(0), 0x55)),
            Event::Completed(a, Outcome::Dispatched),
        ]
    );
}

#[test]
fn an_inline_qmd_requiring_a_pcas_is_written_but_not_scheduled_without_send_pcas() {
    let mem = memory();
    let a = qmd_at(9);
    let spec = Spec::new(0xa000);
    // Every control bit but SEND_PCAS_ENABLE.
    let regs = inline_registers(a, !SEND_PCAS, &spec);
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    table
        .compute_method(LAST_INLINE_QMD_METHOD, &regs, &mem, &mut sink)
        .unwrap();
    assert_eq!(
        read_u32(&mem, a + 18 * 4),
        encode(&spec)[18],
        "the QMD was written"
    );
    assert_eq!(sink.events, NOTHING);
    assert_eq!(table.len(), 0);
}

#[test]
fn an_inline_qmd_is_scheduled_when_send_pcas_is_set_or_require_scheduling_pcas_is_clear() {
    // `kepler_compute.cpp`'s `(control & (1U << 24)) || !(native_registers[0xcc] & (1U << 12))`,
    // where method 0xCC bit 12 is the inline QMD's `REQUIRE_SCHEDULING_PCAS` (QMD bit 140): both
    // values of each bit. Every other control bit is set, so only SEND_PCAS_ENABLE counts.
    for require_scheduling_pcas in [false, true] {
        for send_pcas in [false, true] {
            let mem = memory();
            let a = qmd_at(9);
            let spec = Spec {
                require_scheduling_pcas,
                ..Spec::new(0xa000)
            };
            let control = if send_pcas { u32::MAX } else { !SEND_PCAS };
            let regs = inline_registers(a, control, &spec);
            let mut table = QmdTable::new();
            let mut sink = Recorder::default();
            table
                .compute_method(LAST_INLINE_QMD_METHOD, &regs, &mem, &mut sink)
                .unwrap();
            let case = format!("send_pcas {send_pcas}, require {require_scheduling_pcas}");
            assert_eq!(read_u32(&mem, a + 18 * 4), encode(&spec)[18], "{case}");
            if send_pcas || !require_scheduling_pcas {
                assert_eq!(sink.completions(), [(a, Outcome::Dispatched)], "{case}");
                assert_eq!(table.len(), 1, "{case}");
            } else {
                assert_eq!(sink.events, NOTHING, "{case}");
                assert_eq!(table.len(), 0, "{case}");
            }
        }
    }
}

#[test]
fn an_inline_qmd_is_scheduled_with_a_reload() {
    // A QMD that already ran at the inline address runs again: the inline QMD is signalled
    // with INVALIDATE_COPY_SCHEDULE, as Eden-NX2's `ProcessNativeMethod(system, 0xb0, 3, ..)`.
    let mem = memory();
    let a = qmd_at(9);
    let spec = Spec::new(0xa000);
    let regs = inline_registers(a, SEND_PCAS, &spec);
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    for _ in 0..2 {
        table
            .compute_method(LAST_INLINE_QMD_METHOD, &regs, &mem, &mut sink)
            .unwrap();
    }
    assert_eq!(sink.completions().len(), 2);
}

#[test]
fn divergence_inline_address_overflow_is_refused() {
    let mem = memory();
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    let mut regs = inline_registers(0, SEND_PCAS, &Spec::new(0));
    let a = usize::try_from(SET_INLINE_QMD_ADDRESS_A.method()).unwrap();
    let b = usize::try_from(SET_INLINE_QMD_ADDRESS_B.method()).unwrap();
    // The largest upper word that fits: the address is computed, and is unmapped.
    regs[a] = 0x00ff_ffff;
    regs[b] = 0x0000_0001;
    let fits = 0xffff_ff00_0000_0100;
    assert_eq!(
        table.compute_method(LAST_INLINE_QMD_METHOD, &regs, &mem, &mut sink),
        Err(QmdError::InlineWrite {
            address: fits,
            fault: GpuFault::Unmapped { va: fits }
        })
    );
    // One more bit does not fit: Eden-NX2 would drop it.
    regs[a] = 0x0100_0000;
    assert_eq!(
        table.compute_method(LAST_INLINE_QMD_METHOD, &regs, &mem, &mut sink),
        Err(QmdError::InlineAddress { upper: 0x0100_0000 })
    );
    assert_eq!(table.len(), 0);
}

#[test]
fn an_unwritable_inline_qmd_is_an_error_and_is_not_scheduled() {
    let mem = memory();
    let outside = GPU_END;
    let regs = inline_registers(outside, SEND_PCAS, &Spec::new(0xa000));
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    assert_eq!(
        table.compute_method(LAST_INLINE_QMD_METHOD, &regs, &mem, &mut sink),
        Err(QmdError::InlineWrite {
            address: outside,
            fault: GpuFault::Unmapped { va: outside }
        })
    );
    assert_eq!(table.len(), 0);
    assert_eq!(sink.events, NOTHING);
}

#[test]
fn the_register_file_must_reach_the_last_inline_word() {
    let mem = memory();
    let a = qmd_at(9);
    let full = inline_registers(a, SEND_PCAS, &Spec::new(0xa000));
    let short = &full[..REGISTERS_NEEDED - 1];
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    for method in [LAST_INLINE_QMD_METHOD, SEND_SIGNALING_PCAS2_B.method()] {
        assert_eq!(
            table.compute_method(method, short, &mem, &mut sink),
            Err(QmdError::Registers {
                len: REGISTERS_NEEDED - 1,
                needed: REGISTERS_NEEDED
            })
        );
    }
    // Other methods do not read the file at all.
    assert_eq!(
        table.compute_method(SEND_PCAS_A.method(), &[], &mem, &mut sink),
        Ok(())
    );
    assert_eq!(
        table.compute_method(LAST_INLINE_QMD_METHOD, &full, &mem, &mut sink),
        Ok(())
    );
    assert_eq!(sink.completions(), [(a, Outcome::Dispatched)]);
}

#[test]
fn methods_other_than_the_two_triggers_do_nothing() {
    let mem = memory();
    let a = qmd_at(9);
    let mut regs = inline_registers(a, SEND_PCAS, &Spec::new(0xa000));
    let pcas_a = usize::try_from(SEND_PCAS_A.method()).unwrap();
    regs[pcas_a] = u32::try_from(a >> QMD_ADDRESS_SHIFT).unwrap();
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    let others = [
        SEND_PCAS_A.method(),
        SEND_PCAS_A.method() + 1, // SEND_PCAS_B
        SEND_PCAS_A.method() + 2, // SEND_SIGNALING_PCAS_B
        SET_QMD_VIRTUALIZATION_CONTROL.method(),
        LOAD_INLINE_QMD_DATA.method(),
        LAST_INLINE_QMD_METHOD - 1,
        LAST_INLINE_QMD_METHOD + 1,
    ];
    for method in others {
        assert_eq!(table.compute_method(method, &regs, &mem, &mut sink), Ok(()));
    }
    assert_eq!(sink.events, NOTHING);
    assert_eq!(table.len(), 0);
    let mut untouched = vec![0u8; QMD_BYTES];
    mem.read(a, &mut untouched).unwrap();
    assert!(untouched.iter().all(|&b| b == 0), "nothing was written");
}

#[test]
fn pcas2_b_signals_the_qmd_named_by_send_pcas_a() {
    let mem = memory();
    let a = qmd_at(7);
    put(&mem, a, &Spec::new(0xa000));
    let mut regs = vec![0u32; REGISTERS_NEEDED];
    let pcas_a = usize::try_from(SEND_PCAS_A.method()).unwrap();
    let pcas2_b = usize::try_from(SEND_SIGNALING_PCAS2_B.method()).unwrap();
    regs[pcas_a] = u32::try_from(a >> QMD_ADDRESS_SHIFT).unwrap();
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    // NOP and INVALIDATE do nothing, and add no entry.
    for word in [0, 1] {
        regs[pcas2_b] = word;
        table
            .compute_method(SEND_SIGNALING_PCAS2_B.method(), &regs, &mem, &mut sink)
            .unwrap();
    }
    assert_eq!(table.len(), 0);
    regs[pcas2_b] = 6; // INCREMENT_PUT
    assert_eq!(
        table.compute_method(SEND_SIGNALING_PCAS2_B.method(), &regs, &mem, &mut sink),
        Err(QmdError::PcasAction { action: 6 })
    );
    assert_eq!(table.len(), 0);
    regs[pcas2_b] = PcasAction::Schedule.encode().unwrap();
    table
        .compute_method(SEND_SIGNALING_PCAS2_B.method(), &regs, &mem, &mut sink)
        .unwrap();
    assert_eq!(sink.completions(), [(a, Outcome::Dispatched)]);
}

// ---------------------------------------------------------------------------------------------
// Divergences.

#[test]
fn divergence_bit_19_release_is_a_max_release() {
    let address = qmd_at(1);
    let release_op = |membar, reduction| {
        let slot = SlotSpec {
            membar,
            reduction,
            ..SlotSpec::write(target(0), 3)
        };
        Qmd::decode(address, &encode(&Spec::new(0).slot(0, slot)))
            .unwrap()
            .releases[0]
            .unwrap()
            .op
    };
    assert_eq!(release_op(0, None), ReleaseOp::Write);
    assert_eq!(release_op(1, None), ReleaseOp::Reduce(MAX_RELEASE));
    // With reduction enabled, the slot's own reduction wins.
    let add = Reduction {
        op: ReductionOp::Add,
        format: ReductionFormat::Unsigned,
    };
    assert_eq!(
        release_op(1, Some((sema::RED_ADD, sema::RED_FORMAT_UNSIGNED))),
        ReleaseOp::Reduce(add)
    );
    assert_eq!(
        MAX_RELEASE,
        Reduction {
            op: ReductionOp::Max,
            format: ReductionFormat::Unsigned
        }
    );
    // Through the scheduler too, in every slot.
    let mem = memory();
    let a = qmd_at(1);
    let max = |n| SlotSpec {
        membar: 1,
        ..SlotSpec::write(target(n), 7)
    };
    put(
        &mem,
        a,
        &Spec::new(0xa000)
            .slot(0, max(0))
            .slot(1, max(1))
            .slot(2, max(2)),
    );
    let mut sink = Recorder::default();
    QmdTable::new()
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    let ops: Vec<_> = sink.releases().iter().map(|r| r.op).collect();
    assert_eq!(ops, [ReleaseOp::Reduce(MAX_RELEASE); 3]);
}

#[test]
fn divergence_every_header_reduction_is_released() {
    let address = qmd_at(1);
    for op in 0..8 {
        for format in [sema::RED_FORMAT_UNSIGNED, sema::RED_FORMAT_SIGNED] {
            let slot = SlotSpec {
                reduction: Some((op, format)),
                ..SlotSpec::write(target(0), 3)
            };
            let qmd = Qmd::decode(address, &encode(&Spec::new(0).slot(2, slot))).unwrap();
            assert_eq!(
                qmd.releases[2].unwrap().op,
                ReleaseOp::Reduce(sema::red_reduction(QMD_RELEASE, op, format).unwrap())
            );
        }
    }
    // Spot checks against the header's names, so a swapped table cannot agree with itself.
    let op_of = |op| {
        let slot = SlotSpec {
            reduction: Some((op, sema::RED_FORMAT_SIGNED)),
            ..SlotSpec::write(target(0), 3)
        };
        Qmd::decode(address, &encode(&Spec::new(0).slot(0, slot)))
            .unwrap()
            .releases[0]
            .unwrap()
            .op
    };
    assert_eq!(
        op_of(q::QMDV03_00_RELEASE0_REDUCTION_OP_RED_MAX),
        ReleaseOp::Reduce(Reduction {
            op: ReductionOp::Max,
            format: ReductionFormat::Signed
        })
    );
    assert_eq!(
        op_of(q::QMDV03_00_RELEASE0_REDUCTION_OP_RED_XOR),
        ReleaseOp::Reduce(Reduction {
            op: ReductionOp::Xor,
            format: ReductionFormat::Signed
        })
    );
}

#[test]
fn divergence_format_bits_without_reduction_are_ignored() {
    let address = qmd_at(1);
    for stray_format in [1, 2, 3] {
        let slot = SlotSpec {
            stray_format,
            ..SlotSpec::write(target(0), 3)
        };
        let qmd = Qmd::decode(address, &encode(&Spec::new(0).slot(0, slot))).unwrap();
        assert_eq!(qmd.releases[0], Some(write_release(target(0), 3)));
    }
}

#[test]
fn divergence_constant_buffer_upper_bits_are_kept() {
    let mem = memory();
    let a = qmd_at(1);
    let mut spec = Spec::new(0xa000);
    // Upper bits 16:8 set: Eden-NX2 rejects the program (`high & 0x1ff00`).
    spec.dispatch.constant_buffers[3] = ConstantBuffer {
        valid: true,
        address: 0x1_ff12_3456_7800,
        size: 0x100,
    };
    put(&mem, a, &spec);
    let mut sink = Recorder::default();
    QmdTable::new()
        .signal(a, PcasAction::Schedule, &mem, &mut sink)
        .unwrap();
    assert_eq!(sink.events[0], Event::Dispatch(dispatch_of(&spec, a)));
    assert_eq!(sink.completions(), [(a, Outcome::Dispatched)]);
}

#[test]
fn divergence_a_failed_request_adds_no_entry() {
    let mem = memory();
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    let unmapped = GPU_END;
    assert!(matches!(
        table.signal(unmapped, PcasAction::Schedule, &mem, &mut sink),
        Err(QmdError::Read { .. })
    ));
    let a = qmd_at(1);
    put(
        &mem,
        a,
        &Spec {
            major: 2,
            ..Spec::new(0)
        },
    );
    assert!(matches!(
        table.signal(a, PcasAction::InvalidateCopySchedule, &mem, &mut sink),
        Err(QmdError::Version { .. })
    ));
    assert_eq!(table.len(), 0);
}

/// `compute_method` decodes the `SEND_SIGNALING_PCAS2_B` action before it signals, so an
/// unsupported action on a QMD not in a full table is `PcasAction`; Eden-NX2 checks the table
/// limit (`native_qmd.h` line 35) before the action (line 66) and reports "QMD cache limit".
/// A supported action on the same QMD reaches the table limit.
#[test]
fn divergence_an_unsupported_action_is_reported_before_the_table_limit() {
    let mem = memory();
    let mut table = QmdTable::new();
    let mut sink = Recorder::default();
    let at = |n: u64| n << QMD_ADDRESS_SHIFT;
    for n in 0..MAX_QMDS as u64 {
        table.signal(at(n), dec(1), &mem, &mut sink).unwrap();
    }
    assert_eq!(table.len(), MAX_QMDS);
    let new = at(MAX_QMDS as u64);
    let mut regs = vec![0u32; REGISTERS_NEEDED];
    let pcas_a = usize::try_from(SEND_PCAS_A.method()).unwrap();
    let pcas2_b = usize::try_from(SEND_SIGNALING_PCAS2_B.method()).unwrap();
    regs[pcas_a] = u32::try_from(new >> QMD_ADDRESS_SHIFT).unwrap();
    regs[pcas2_b] = SEND_SIGNALING_PCAS2_B_PCAS_ACTION
        .insert(0, SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INCREMENT_PUT)
        .unwrap();
    assert_eq!(
        table.compute_method(SEND_SIGNALING_PCAS2_B.method(), &regs, &mem, &mut sink),
        Err(QmdError::PcasAction {
            action: SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INCREMENT_PUT
        })
    );
    regs[pcas2_b] = dec(1).encode().unwrap();
    assert_eq!(
        table.compute_method(SEND_SIGNALING_PCAS2_B.method(), &regs, &mem, &mut sink),
        Err(QmdError::TableFull { address: new })
    );
    assert_eq!(table.len(), MAX_QMDS);
    assert_eq!(table.status(new), None);
    assert_eq!(sink.events, NOTHING);
}

#[test]
fn divergence_release_addresses_are_not_checked_here() {
    let mem = memory();
    let a = qmd_at(1);
    let unmapped = GPU_END + 0x40;
    put(
        &mem,
        a,
        &Spec::new(0xa000).slot(1, SlotSpec::write(unmapped, 5)),
    );
    let mut sink = Recorder::default();
    assert_eq!(
        QmdTable::new().signal(a, PcasAction::Schedule, &mem, &mut sink),
        Ok(())
    );
    assert_eq!(sink.releases(), [write_release(unmapped, 5)]);
}

// ---------------------------------------------------------------------------------------------
// Error text.

#[test]
fn errors_read_as_plain_words() {
    let fault = GpuFault::Unmapped { va: 0x1000 };
    let cases: [(QmdError, &str); 14] = [
        (
            QmdError::TableFull { address: 0x100 },
            "the QMD table is full (4096 QMDs), so QMD 0x100 cannot be added",
        ),
        (
            QmdError::ChainTooLong { address: 0x200 },
            "dependent QMD 0x200 would make one signal's chain longer than 256 QMDs",
        ),
        (
            QmdError::Read {
                address: 0x300,
                fault,
            },
            "QMD 0x300 cannot be read: GPU address 0x1000 is not mapped",
        ),
        (
            QmdError::Version {
                address: 0x400,
                major: 2,
                minor: 4,
            },
            "QMD 0x400 is version 2.4, not 3.0",
        ),
        (
            QmdError::Queue { address: 0x500 },
            "QMD 0x500 is a queue, which is not supported",
        ),
        (
            QmdError::PcasAction { action: 6 },
            "PCAS action 6 is not supported",
        ),
        (
            QmdError::DependenceUnderflow {
                address: 0x600,
                counter: -1,
            },
            "QMD 0x600 is ready to run with its dependence counter at -1",
        ),
        (
            QmdError::DependenceOverflow { address: 0x700 },
            "QMD 0x700: the change to its dependence counter does not fit",
        ),
        (
            QmdError::Release {
                address: 0x800,
                slot: 0,
                refusal: ReleaseRefusal::StructureSize(0),
            },
            "QMD 0x800: release slot 0: its structure size 0 is not one word",
        ),
        (
            QmdError::DependentAction {
                address: 0x900,
                action: 0,
            },
            "QMD 0x900: dependent QMD action 0 is not supported",
        ),
        (
            QmdError::SecondDependent { address: 0xa00 },
            "QMD 0xa00 enables a second dependent QMD, which is not supported",
        ),
        (
            QmdError::InlineAddress { upper: 0x0100_0000 },
            "the inline QMD address with upper word 0x1000000 does not fit in 64 bits",
        ),
        (
            QmdError::InlineWrite {
                address: 0xb00,
                fault,
            },
            "the inline QMD cannot be written at 0xb00: GPU address 0x1000 is not mapped",
        ),
        (
            QmdError::Registers {
                len: 0,
                needed: 264,
            },
            "the compute register file has 0 registers, but the QMD methods need 264",
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
    let refusals = [
        (ReleaseRefusal::Payload64, "it has a 64-bit payload"),
        (ReleaseRefusal::Trap(2), "its trap type 2 is not none"),
        (
            ReleaseRefusal::Reduction(DecodeError::UndefinedFormat {
                word: QMD_RELEASE,
                value: 3,
            }),
            "QMD release reduction format 3 is not defined",
        ),
    ];
    for (refusal, text) in refusals {
        assert_eq!(refusal.to_string(), text);
    }
    assert_eq!(REGISTERS_NEEDED, 0x108);
}

/// The local-memory size is the low field plus the high field (every reference's
/// `ComputeEnvironment`: `qmd.local_pos_alloc + qmd.local_crs_alloc`), decoded from a laid-out
/// QMD with a distinct value in each, and with either field alone, so a size that took only
/// one field, or either field twice, fails; and with two values that share bits (0x130 and
/// 0x110, whose sum carries), so an OR of the fields in place of their sum fails.
#[test]
fn the_local_memory_size_is_the_sum_of_both_fields() {
    let mut w = encode(&Spec::new(0x1_0000));
    for (low, high, size) in [
        (0x0001_2000, 0x0000_0300, 0x0001_2300),
        (0x0000_0130, 0x0000_0110, 0x0000_0240),
        (0x0000_0040, 0, 0x0000_0040),
        (0, 0x0000_0500, 0x0000_0500),
        (0, 0, 0),
    ] {
        set(&mut w, q::QMDV03_00_SHADER_LOCAL_MEMORY_LOW_SIZE, low);
        set(&mut w, q::QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE, high);
        let dispatch = Qmd::decode(qmd_at(1), &w).unwrap().dispatch;
        assert_eq!(dispatch.local_memory_size(), size, "{low:#x} + {high:#x}");
    }
}

/// Divergence: bits 20 to 23 of each local-memory field are kept, where Eden-NX2's 20-bit
/// `BitField`s drop them (`launch_description.local_pos_alloc.Assign(qmd[23] & 0xffffff)`
/// into `BitField<0, 20, u32>`): Eden-NX2 would size these as 0x1 and 0x20 bytes. Each field's
/// widest value sums without wrapping, and a hand-built wider sum saturates.
#[test]
fn divergence_local_memory_keeps_bits_20_to_23() {
    let mut w = encode(&Spec::new(0x1_0000));
    set(&mut w, q::QMDV03_00_SHADER_LOCAL_MEMORY_LOW_SIZE, 0x10_0001);
    set(
        &mut w,
        q::QMDV03_00_SHADER_LOCAL_MEMORY_HIGH_SIZE,
        0x80_0020,
    );
    let dispatch = Qmd::decode(qmd_at(1), &w).unwrap().dispatch;
    assert_eq!(dispatch.local_memory_size(), 0x90_0021);
    let widest = Dispatch {
        local_memory_low: 0xff_ffff,
        local_memory_high: 0xff_ffff,
        ..dispatch
    };
    assert_eq!(widest.local_memory_size(), 0x1ff_fffe);
    let wider = Dispatch {
        local_memory_low: u32::MAX,
        local_memory_high: 1,
        ..dispatch
    };
    assert_eq!(wider.local_memory_size(), u32::MAX);
}

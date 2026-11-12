//! Buffers by index: the check of a request against the declared buffers, which descriptor an
//! index means, and the bounds on reading and writing.

use starquake_ipc::{BufferDescriptor, ReceiveDescriptor, RequestBuilder, StaticDescriptor};
use starquake_kernel::result::{KernelError, ResultCode};

use super::super::result::{INVALID_CMIF_REQUEST, NOT_SUPPORTED};
use super::{Answer, Rig, STALE, UPPER_LIMIT, only, plain, request, toy};

/// Where tests map the text `upper` reads.
const IN: u64 = 0x0010_0000;
/// A second in range, for requests that carry two in descriptors.
const IN_OTHER: u64 = 0x0018_0000;
/// Where tests map the room `upper` writes to.
const OUT: u64 = 0x0020_0000;
/// A second out range.
const OUT_OTHER: u64 = 0x0028_0000;
/// The size of each mapped out range.
const OUT_ROOM: usize = 32;

const TEXT: &[u8] = b"hello";
const OTHER_TEXT: &[u8] = b"other";

fn rig() -> Rig {
    let mut rig = Rig::strict();
    rig.kernel.map(IN, TEXT.to_vec());
    rig.kernel.map(IN_OTHER, OTHER_TEXT.to_vec());
    rig.kernel.map(OUT, vec![STALE; OUT_ROOM]);
    rig.kernel.map(OUT_OTHER, vec![STALE; OUT_ROOM]);
    rig
}

fn a(address: u64, size: usize) -> BufferDescriptor {
    BufferDescriptor {
        address,
        size: u64::try_from(size).unwrap(),
        flags: 0,
    }
}

fn x(address: u64, size: usize) -> StaticDescriptor {
    StaticDescriptor {
        counter: 0,
        address,
        size: u16::try_from(size).unwrap(),
    }
}

fn c(address: u64, size: usize) -> ReceiveDescriptor {
    ReceiveDescriptor {
        address,
        size: u16::try_from(size).unwrap(),
    }
}

fn upper() -> RequestBuilder {
    request(toy::UPPER)
}

/// `upper` sent as a client sends its two auto-select buffers through map-alias descriptors:
/// `text` as the A descriptor and `room` as the B descriptor, each with an empty pointer
/// descriptor beside it (libnx `cmifRequestInAutoBuffer`, `cmifRequestOutAutoBuffer`).
fn upper_mapped(text: BufferDescriptor, room: BufferDescriptor) -> RequestBuilder {
    upper()
        .send_buffer(text)
        .static_descriptor(x(0, 0))
        .receive_buffer(room)
        .receive_descriptor(c(0, 0))
}

fn send(rig: &mut Rig, build: &RequestBuilder) -> Answer {
    plain(&rig.send(build))
}

/// The reply `upper` gives after writing `len` bytes.
fn wrote(len: usize) -> Answer {
    Answer {
        data: u32::try_from(len).unwrap().to_le_bytes().to_vec(),
        ..only(ResultCode::SUCCESS)
    }
}

/// What a mapped out range holds after `text` was written to its start.
fn holding(text: &[u8]) -> Vec<u8> {
    let mut bytes = text.to_vec();
    bytes.resize(OUT_ROOM, STALE);
    bytes
}

/// What a mapped out range holds after `upper` wrote `text` to its start.
fn filled(text: &[u8]) -> Vec<u8> {
    holding(&text.to_ascii_uppercase())
}

#[test]
fn an_auto_select_buffer_is_its_a_or_b_descriptor_when_that_is_not_empty() {
    let mut rig = rig();
    let build = upper_mapped(a(IN, TEXT.len()), a(OUT, OUT_ROOM));
    assert_eq!(send(&mut rig, &build), wrote(TEXT.len()));
    assert_eq!(rig.kernel.memory(OUT, OUT_ROOM).unwrap(), filled(TEXT));
}

#[test]
fn an_auto_select_buffer_is_its_x_or_c_descriptor_when_the_a_or_b_is_empty() {
    let mut rig = rig();
    let build = upper()
        .send_buffer(a(0, 0))
        .static_descriptor(x(IN, TEXT.len()))
        .receive_buffer(a(0, 0))
        .receive_descriptor(c(OUT, OUT_ROOM));
    assert_eq!(send(&mut rig, &build), wrote(TEXT.len()));
    assert_eq!(rig.kernel.memory(OUT, OUT_ROOM).unwrap(), filled(TEXT));
}

/// Eden's rule: a non-empty A (or B) descriptor wins over its X (or C) descriptor, and an empty
/// one gives way to it.
#[test]
fn a_non_empty_a_or_b_wins_and_an_empty_one_gives_way() {
    let mut rig = rig();
    let both = upper()
        .send_buffer(a(IN, TEXT.len()))
        .static_descriptor(x(IN_OTHER, OTHER_TEXT.len()))
        .receive_buffer(a(OUT, OUT_ROOM))
        .receive_descriptor(c(OUT_OTHER, OUT_ROOM));
    assert_eq!(send(&mut rig, &both), wrote(TEXT.len()));
    assert_eq!(rig.kernel.memory(OUT, OUT_ROOM).unwrap(), filled(TEXT));
    assert_eq!(rig.kernel.memory(OUT_OTHER, OUT_ROOM).unwrap(), filled(b""));

    let mut rig = self::rig();
    let empty_first = upper()
        .send_buffer(a(IN, 0))
        .static_descriptor(x(IN_OTHER, OTHER_TEXT.len()))
        .receive_buffer(a(OUT, 0))
        .receive_descriptor(c(OUT_OTHER, OUT_ROOM));
    assert_eq!(send(&mut rig, &empty_first), wrote(OTHER_TEXT.len()));
    assert_eq!(rig.kernel.memory(OUT, OUT_ROOM).unwrap(), filled(b""));
    assert_eq!(
        rig.kernel.memory(OUT_OTHER, OUT_ROOM).unwrap(),
        filled(OTHER_TEXT)
    );
}

/// Both descriptors of an auto-select buffer empty: an empty buffer, as Eden reads it.
#[test]
fn an_auto_select_buffer_with_both_descriptors_empty_is_empty() {
    let mut rig = rig();
    let nothing_to_read = upper_mapped(a(IN, 0), a(OUT, OUT_ROOM));
    assert_eq!(send(&mut rig, &nothing_to_read), wrote(0));
    assert_eq!(rig.kernel.memory(OUT, OUT_ROOM).unwrap(), filled(b""));

    // No room to write, and nothing to write into it.
    let no_room = upper_mapped(a(IN, 0), a(OUT, 0));
    assert_eq!(send(&mut rig, &no_room), wrote(0));
}

/// Divergence from Eden, which logs a failed assertion and reads or writes nothing: a request
/// whose descriptors are not the declared buffers is refused, and the method never runs.
#[test]
fn a_request_that_fails_the_check_never_reaches_the_method() {
    let mut rig = rig();
    // Each lacks one of the four descriptors two auto-select buffers take.
    let no_x = upper()
        .send_buffer(a(IN, TEXT.len()))
        .receive_buffer(a(OUT, OUT_ROOM))
        .receive_descriptor(c(0, 0));
    let no_a = upper()
        .static_descriptor(x(IN, TEXT.len()))
        .receive_buffer(a(OUT, OUT_ROOM))
        .receive_descriptor(c(0, 0));
    let no_b = upper()
        .send_buffer(a(IN, TEXT.len()))
        .static_descriptor(x(0, 0))
        .receive_descriptor(c(OUT, OUT_ROOM));
    let no_c = upper()
        .send_buffer(a(IN, TEXT.len()))
        .static_descriptor(x(0, 0))
        .receive_buffer(a(OUT, OUT_ROOM));
    // Everything, and a W descriptor besides.
    let exchange = upper_mapped(a(IN, TEXT.len()), a(OUT, OUT_ROOM))
        .exchange_buffer(a(IN_OTHER, OTHER_TEXT.len()));
    for build in [no_x, no_a, no_b, no_c, exchange] {
        assert_eq!(send(&mut rig, &build), only(INVALID_CMIF_REQUEST));
    }
    assert_eq!(rig.kernel.memory(OUT, OUT_ROOM).unwrap(), filled(b""));
}

/// Divergence from Eden, which reads in buffer 1 from A descriptor 1: each descriptor list is
/// numbered on its own, as a client fills it.
#[test]
fn each_list_is_numbered_on_its_own() {
    let mut rig = rig();
    // In buffer 0 is pointer (X 0), in buffer 1 map-alias (A 0); out buffer 0 is map-alias (B 0),
    // out buffer 1 pointer (C 0).
    let build = request(toy::COPY_MIXED)
        .static_descriptor(x(IN, TEXT.len()))
        .send_buffer(a(IN_OTHER, OTHER_TEXT.len()))
        .receive_buffer(a(OUT, OUT_ROOM))
        .receive_descriptor(c(OUT_OTHER, OUT_ROOM));
    assert_eq!(send(&mut rig, &build), only(ResultCode::SUCCESS));
    assert_eq!(
        rig.kernel.memory(OUT, OUT_ROOM).unwrap(),
        holding(OTHER_TEXT)
    );
    assert_eq!(
        rig.kernel.memory(OUT_OTHER, OUT_ROOM).unwrap(),
        holding(TEXT)
    );
}

/// starquake's own: a method that asks for a buffer its row does not declare is a mistake in the
/// service, answered `NOT_SUPPORTED`.
#[test]
fn a_buffer_the_row_does_not_declare_is_not_supported() {
    let mut rig = rig();
    let build = request(toy::PEEK_UNDECLARED).send_buffer(a(IN, TEXT.len()));
    assert_eq!(send(&mut rig, &build), only(NOT_SUPPORTED));
}

#[test]
fn an_in_buffer_is_read_only_up_to_the_limit_the_method_states() {
    let mut rig = Rig::strict();
    rig.kernel.map(IN, vec![b'a'; UPPER_LIMIT + 1]);
    rig.kernel.map(OUT, vec![STALE; OUT_ROOM]);
    // Exactly at the limit.
    let at_limit = upper_mapped(a(IN, UPPER_LIMIT), a(OUT, OUT_ROOM));
    assert_eq!(send(&mut rig, &at_limit), wrote(UPPER_LIMIT));
    // One byte more is refused before anything is read or written.
    let mut rig = Rig::strict();
    rig.kernel.map(IN, vec![b'a'; UPPER_LIMIT + 1]);
    rig.kernel.map(OUT, vec![STALE; OUT_ROOM]);
    let over = upper_mapped(a(IN, UPPER_LIMIT + 1), a(OUT, OUT_ROOM));
    assert_eq!(send(&mut rig, &over), only(INVALID_CMIF_REQUEST));
    assert_eq!(rig.kernel.memory(OUT, OUT_ROOM).unwrap(), filled(b""));
}

/// Divergence from Eden, which cuts the write down to the buffer: a write past the end of an out
/// buffer is an error and writes nothing.
#[test]
fn a_write_past_the_end_of_an_out_buffer_writes_nothing() {
    let mut rig = rig();
    // Exactly enough room.
    let exact = upper_mapped(a(IN, TEXT.len()), a(OUT, TEXT.len()));
    assert_eq!(send(&mut rig, &exact), wrote(TEXT.len()));
    assert_eq!(rig.kernel.memory(OUT, OUT_ROOM).unwrap(), filled(TEXT));
    // One byte short, into the other range.
    let short = upper_mapped(a(IN, TEXT.len()), a(OUT_OTHER, TEXT.len() - 1));
    assert_eq!(send(&mut rig, &short), only(INVALID_CMIF_REQUEST));
    assert_eq!(rig.kernel.memory(OUT_OTHER, OUT_ROOM).unwrap(), filled(b""));
}

#[test]
fn a_buffer_must_be_mapped_for_its_whole_extent() {
    let fault = only(KernelError::InvalidCurrentMemory.code());
    // The in buffer claims one byte more than is mapped.
    let mut rig = rig();
    let in_past_end = upper_mapped(a(IN, TEXT.len() + 1), a(OUT, OUT_ROOM));
    assert_eq!(send(&mut rig, &in_past_end), fault);
    // The out buffer starts one byte before the end of its range, and five bytes go in.
    let last = OUT + u64::try_from(OUT_ROOM).unwrap() - 1;
    let out_past_end = upper_mapped(a(IN, TEXT.len()), a(last, OUT_ROOM));
    assert_eq!(send(&mut rig, &out_past_end), fault);
    assert_eq!(rig.kernel.memory(OUT, OUT_ROOM).unwrap(), filled(b""));
    // Nothing mapped at all.
    let unmapped = upper_mapped(a(0x7000_0000, TEXT.len()), a(OUT, OUT_ROOM));
    assert_eq!(send(&mut rig, &unmapped), fault);
}

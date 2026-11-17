//! Test helpers the TIPC servers' tests share: messages laid out by hand, word by word, and
//! whole reply buffers to compare with.

use starquake_ipc::{BUFFER_SIZE, Message, SessionKind, TIPC_COMMAND_OFFSET};
use starquake_kernel::result::ResultCode;

/// What fills a message buffer before a request is laid into it.
pub(crate) const STALE: u8 = 0xA5;

/// The TIPC header's "a special header follows" bit (PR `DecodeBootstrap`: `packet[1] !=
/// 0x8000000c`, a special header and twelve data words).
pub(crate) const SPECIAL: u32 = 0x8000_0000;

/// Little-endian words as bytes.
pub(crate) fn le(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A whole message buffer that starts with `words` and is zero after them.
pub(crate) fn buffer_of(words: &[u32]) -> Vec<u8> {
    let mut buffer = le(words);
    buffer.resize(BUFFER_SIZE, 0);
    buffer
}

/// A TIPC message's type word for command `id`.
pub(crate) fn tipc_type(id: u32) -> u32 {
    u32::from(TIPC_COMMAND_OFFSET) + id
}

/// The low and high words of a `u64`.
pub(crate) fn halves(value: u64) -> [u32; 2] {
    let bytes = value.to_le_bytes();
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    [word(0), word(4)]
}

/// A TIPC reply to command `id` that holds a result and nothing else.
pub(crate) fn result_only(id: u32, result: ResultCode) -> Vec<u8> {
    buffer_of(&[tipc_type(id), 1, result.raw()])
}

/// A successful TIPC reply to `id` with these copied handles and data words: the header (raw
/// size: the result word and the data), the special header when there are handles, the
/// handles, the result, the data.
pub(crate) fn success(id: u32, copies: &[u32], data: &[u32]) -> Vec<u8> {
    let raw = 1 + u32::try_from(data.len()).unwrap();
    let mut words = vec![tipc_type(id)];
    if copies.is_empty() {
        words.push(raw);
    } else {
        words.push(SPECIAL | raw);
        words.push(u32::try_from(copies.len()).unwrap() << 1);
        words.extend(copies);
    }
    words.push(ResultCode::SUCCESS.raw());
    words.extend(data);
    buffer_of(&words)
}

/// The handles a reply copied.
pub(crate) fn copied(buffer: &[u8]) -> Vec<u32> {
    Message::parse_reply(buffer, SessionKind::Session)
        .unwrap()
        .copy_handles
}

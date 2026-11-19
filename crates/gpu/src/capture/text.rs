//! The text `starquake gpu-replay --dump writes` prints: one line per write a replayed thread
//! made, in plain words.

use std::fmt::Write as _;

use super::format::ContentHash;
use super::replay::ReplayedWrite;

/// The most bytes of a write shown as hex; a longer write shows its length and hash.
pub const WRITE_BYTES_SHOWN: usize = 64;

/// One line per write, in order: `<thread> space <n> wrote <len> bytes at <address>: <hex>`, the
/// hex replaced by `hash <hash>` past [`WRITE_BYTES_SHOWN`] bytes, and `(it faulted: <fault>)`
/// after a write whose outcome was a fault.
pub fn writes_text(writes: &[ReplayedWrite]) -> String {
    let mut text = String::new();
    for write in writes {
        let _ = write!(
            text,
            "{} space {} wrote {} bytes at {:#x}: ",
            write.thread.name(),
            write.space.0,
            write.data.len(),
            write.va
        );
        if write.data.len() > WRITE_BYTES_SHOWN {
            let _ = write!(text, "hash {}", ContentHash::of(&write.data));
        } else {
            let hex: Vec<String> = write.data.iter().map(|byte| format!("{byte:02x}")).collect();
            text.push_str(&hex.join(" "));
        }
        if let Err(fault) = write.result {
            let _ = write!(text, " (it faulted: {fault})");
        }
        text.push('\n');
    }
    text
}

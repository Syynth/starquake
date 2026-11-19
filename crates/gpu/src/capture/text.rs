//! The text `starquake gpu-replay --dump writes` prints: one line per write a replayed thread
//! made, in plain words.

use std::fmt::Write as _;

use super::format::Blobs;
use super::replay::ReplayedWrite;

/// The most bytes of a write shown as hex; a longer write shows its length and hash.
pub const WRITE_BYTES_SHOWN: usize = 64;

/// One line per write, in order: `<thread> space <n> wrote <len> bytes at <address>: <hex>`, the
/// hex looked up in `blobs` (the replay's [`super::ReplayOutput::blobs`]) and replaced by
/// `hash <hash>` past [`WRITE_BYTES_SHOWN`] bytes or for a blob `blobs` does not hold, and
/// `(it faulted: <fault>)` after a write whose outcome was a fault.
pub fn writes_text(writes: &[ReplayedWrite], blobs: &Blobs) -> String {
    let mut text = String::new();
    for write in writes {
        let _ = write!(
            text,
            "{} space {} wrote {} bytes at {:#x}: ",
            write.thread.name(),
            write.space.0,
            write.len,
            write.va
        );
        match blobs.get(write.data) {
            Some(bytes) if bytes.len() <= WRITE_BYTES_SHOWN => {
                let hex: Vec<String> = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
                text.push_str(&hex.join(" "));
            }
            _ => {
                let _ = write!(text, "hash {}", write.data);
            }
        }
        if let Err(fault) = write.result {
            let _ = write!(text, " (it faulted: {fault})");
        }
        text.push('\n');
    }
    text
}

//! The recording backend through its public API: the CPU model with hand-computed pixels,
//! every command, refused lists, resource lifetimes, fences and the snapshot. No GPU.
//!
//! A test crate's root resolves modules next to itself, so the files in `recording/` are named
//! by path.

#[path = "recording/commands.rs"]
mod commands;
#[path = "recording/contents.rs"]
mod contents;
#[path = "recording/fixtures.rs"]
mod fixtures;
#[path = "recording/lifecycle.rs"]
mod lifecycle;
#[path = "recording/replay.rs"]
mod replay;

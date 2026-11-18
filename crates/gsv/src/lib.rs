//! The Switch 2 graphics server (`gsv`) and the sessions it hands out.
//!
//! syncpt (shared page, armed waits, slot events), the GPU virtual-machine sessions
//! (create/map/unmap/sparse regions, GPU VA == CPU VA), vi2 display and present pacing, and the
//! video codec sessions. This is the largest new protocol surface in Switch 2 support.
//!
//! * [`protocol`]: the root's command table, with each fact's mark and source.
//! * [`root`]: the root's server: the bootstrap, and every other root command as a logged stub.
//! * [`completion`]: the completion queue host threads report through.
//! * [`shared`]: the state handlers and the timer thread share, the lock order, the drain, the
//!   shared syncpoint page gsv 204 binds ([`shared::page`]), and the vsync clock
//!   ([`shared::vsync`]).
//! * [`note`]: the notes gsv records for replay.
//! * [`nvsrv`]: the NVIDIA driver session (the nvdrv protocol), `/dev/nvmap`, and the GPU devices
//!   (ctrl-gpu characteristics, as-gpu address-space views, tsg-gpu timeslice groups).
//! * [`syncpt`]: the syncpt session: slot events, armed waits and counter queries.
//! * [`gpu`]: the gpu session (497) and its worker: GPU memory, mappings and sparse regions.
//! * [`native`]: native memory and native images (root 111, 112, 114, 314, 315, 423, 429, 430,
//!   435, 1510) and their size.
//! * [`video`]: the video codec sessions (root 902, 906, 1002) with a null decoder, the video
//!   image pools' layout (the pools are [`native::pools`]: root 911, 912, 961, 991, 426–428),
//!   and the camera session (root 602).

pub mod completion;
pub mod gpu;
pub mod native;
pub mod note;
pub mod nvsrv;
pub mod protocol;
pub mod root;
pub mod shared;
pub mod syncpt;
pub mod video;

#[cfg(test)]
mod testing;

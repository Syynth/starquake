//! The Switch 2 graphics server (`gsv`) and the sessions it hands out.
//!
//! syncpt (shared page, armed waits, slot events), the GPU virtual-machine sessions
//! (create/map/unmap/sparse regions, GPU VA == CPU VA), vi2 display and present pacing, and the
//! video codec sessions. This is the largest new protocol surface in Switch 2 support.
//!
//! * [`protocol`]: the root's command table, with each fact's mark and source.
//! * [`root`]: the root's server: the bootstrap, and every other root command as a logged stub.
//! * [`completion`]: the completion queue host threads report through.
//! * [`shared`]: the state handlers and the timer thread share, the lock order, and the drain.
//! * [`note`]: the notes gsv records for replay.

pub mod completion;
pub mod note;
pub mod protocol;
pub mod root;
pub mod shared;

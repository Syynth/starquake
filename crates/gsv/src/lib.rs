//! The Switch 2 graphics server (`gsv`) and the sessions it hands out.
//!
//! syncpt (shared page, armed waits, slot events), the GPU virtual-machine sessions
//! (create/map/unmap/sparse regions, GPU VA == CPU VA), vi2 display and present pacing, and the
//! video codec sessions. This is the largest new protocol surface in Switch 2 support.

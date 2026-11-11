//! The event stream (`docs/specs/observability.md`).
//!
//! One structured event per crossing of a boundary we own, recorded at a single point per
//! boundary; an always-on flight recorder; and the injectable environment through which the
//! kernel and services read the clock, random numbers and host I/O, so their runs can be replayed.

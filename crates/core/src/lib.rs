//! Wires the emulator together.
//!
//! Owns the system object (kernel, services, graphics server, GPU, render backend), timing, and
//! configuration. Frontends talk to this crate only.

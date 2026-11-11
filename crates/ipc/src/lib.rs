//! CMIF and TIPC message framing.
//!
//! Parses and builds the request/response words in a thread's IPC buffer: headers, handle
//! descriptors, buffer descriptors and raw data. TIPC replies carry no CMIF result header; Eden
//! got this wrong (Eden-NX2 notes, row 22), so both forms need property tests.

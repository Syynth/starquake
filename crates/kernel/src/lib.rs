//! HLE Horizon kernel.
//!
//! Guest threads and their scheduling onto vCPUs, SVC dispatch, handles and kernel objects,
//! synchronisation (`WaitForAddress`/`SignalToAddress` are hot: ~1,400 per frame), memory
//! regions, and `svcGetInfo` including the Switch 2 ids 29-33.

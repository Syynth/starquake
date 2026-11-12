//! Loads Switch 2 titles into a guest process.
//!
//! - NSO/NRO executables, NPDM metadata (both the Switch 1 and the Switch 2 ACID layouts),
//!   ExeFS and RomFS.
//! - ZBIC ([`zbic`]): the zstd variant that compresses a Switch 2 NSO's segments.
//! - NUP containers ([`nup`]): decrypted dumps that hold a base program, its updates and
//!   add-ons in one file, read without loading the file whole.
//! - The load-time code patcher: rewrites `mrs xN, cntpct_el0` (and, where a backend needs it,
//!   `svc` and PAuth returns) into branches to backend-provided stubs. Native backends need this
//!   because counter reads exit to the host at ~760 ns each (see `tools/hvprobe/README.md`).

pub mod npdm;
pub mod nro;
pub mod nso;
pub mod nup;
pub mod patch;
pub mod pfs0;
pub mod romfs;
pub mod zbic;

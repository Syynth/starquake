//! HLE system services.
//!
//! `sm`, `fsp-srv`, `am`/`appletOE`, `hid`, `aud:u`, `pl:u` and the rest. Unknown commands go
//! through an auto-stub that logs and answers success, so boot can progress while the protocol
//! is being worked out.
//!
//! * [`framework`]: what every service is written against, and what runs one.
//! * [`sm`]: the service manager, `sm:`.
//! * [`set`]: system settings, `set:sys` (the firmware version).
//! * [`fsp`]: the file-system service, `fsp-srv` (a first subset).

pub mod framework;
pub mod fsp;
pub mod set;
pub mod sm;

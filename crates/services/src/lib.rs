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
//! * [`apm`]: performance management, `apm` (as far as a libnx start-up asks).
//! * [`am`]: the applet manager's `appletOE` (as far as a libnx start-up asks).
//! * [`hid`]: input, `hid` (an applet resource and its shared memory, as far as a libnx start-up
//!   asks).
//! * [`time`]: time, `time:u` (as far as a libnx start-up asks).
//! * [`lm`]: the log manager, `lm` (a logger whose messages go to the run's guest log).
//! * [`ectx`]: the error-context service as an application reaches it, `ectx:aw`.
//! * [`aoc`]: the add-on content manager, `aoc:u` (no add-on content installed).

pub mod am;
pub mod aoc;
pub mod apm;
pub mod ectx;
mod events;
pub mod framework;
pub mod fsp;
pub mod hid;
pub mod lm;
pub mod set;
mod shared_image;
pub mod sm;
pub mod time;

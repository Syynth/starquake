//! `Break` (0x26): the program stops itself, or only tells a debugger something.
//!
//! Ported from Eden `eden/src/core/hle/kernel/svc/svc_exception.cpp` @ cf654052
//! (GPL-3.0-or-later), `Break`; the reasons are `BreakReason` in
//! `eden/src/core/hle/kernel/svc_types.h` @ cf654052, lines 170-181. PR #4588 changes neither
//! (`eden-pr4588/src/core/hle/kernel/svc/svc_exception.cpp` @ e5fb3b97 differs only in a
//! formatting call), so a Switch 2 program breaks the same way.
//!
//! What Eden does with the reason word:
//!
//! - It splits it into the reason (`reason & ~NotificationOnlyFlag`) and the flag
//!   (`True(reason & BreakReason::NotificationOnlyFlag)`, [`NOTIFICATION_ONLY_FLAG`]).
//! - For `Panic`, `Assert`, `User` and any reason it does not name, and then again for every
//!   break that is not a notification, it reads the argument buffer unless `sz == 0`,
//!   `addr == 0`, or it has read it already (`handle_debug_buffer`). A buffer of
//!   `sizeof(u32)` bytes ([`RESULT_ARGUMENT_SIZE`]) is read as one word, "typically an error
//!   code"; any other size is hex-dumped to the log.
//! - It returns to the guest in every case unless a debugger is attached.
//!
//! [`break_execution`] decodes the call and reads the buffer as Eden does; the dispatcher decides
//! what happens next (`dispatch.rs`, the `Break` arm): a notification-only break returns to the
//! guest, as in Eden, and any other break ends the run as a guest crash, with this call's
//! [`Break`] in the crash report.
//!
//! # Divergences from Eden
//!
//! - **A break that is not a notification ends the run.** Eden logs it and returns to the guest
//!   when no debugger is attached. Horizon exits the process (fact only, from Atmosphère's
//!   `atmosphere/libraries/libmesosphere/source/svc/kern_svc_exception.cpp` @ 4e3f370, `Break`:
//!   with no debugger to notify and no JIT debugging, `GetCurrentProcess().Exit(-1ll)`). Here
//!   the run ends as a crash, which keeps the registers for the report. Test (in
//!   `dispatch/tests/info_and_break.rs`): `a_break_that_is_not_a_notification_ends_the_dispatch_in_broke`.
//! - **Only a four-byte buffer is read.** Eden hex-dumps a buffer of any other size, reading
//!   `sz` bytes, as many as the guest says. Here nothing else is read, so a guest cannot make
//!   the host allocate any amount. Test: `only_a_four_byte_buffer_is_read`.
//! - **A buffer whose extent wraps past the top of the address space gives no code**
//!   (`address + 4` overflows), without a read. Eden does not check. Test:
//!   `a_buffer_that_wraps_the_address_space_is_not_read`.
//! - **An unreadable four-byte buffer gives no code.** Eden's `Read32` logs the fault and reads
//!   zero (`eden/src/core/memory.cpp`). Here [`Break::code`] is `None`, so a crash report does
//!   not show a code the guest never wrote. Test:
//!   `an_unreadable_buffer_gives_no_code_and_a_readable_one_at_the_edge_does`.

use starquake_memory::GuestMemory;

/// `BreakReason::NotificationOnlyFlag` (`svc_types.h` @ cf654052, line 180): the break only
/// tells a debugger something, and the program goes on.
pub const NOTIFICATION_ONLY_FLAG: u32 = 0x8000_0000;

/// The buffer size Eden reads as one word, an error code (`if (sz == sizeof(u32))` in
/// `svc_exception.cpp`).
pub const RESULT_ARGUMENT_SIZE: u64 = 4;

/// Defines [`BreakReason`] from `Name = number, "name";` rows, so a reason, its number and its
/// name are written once.
macro_rules! break_reasons {
    ($($(#[$doc:meta])* $name:ident = $number:literal, $text:literal;)*) => {
        /// Why the program broke: `Svc::BreakReason` without its [`NOTIFICATION_ONLY_FLAG`].
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BreakReason {
            $($(#[$doc])* $name,)*
            /// A reason Eden does not name, by its number (the flag clear).
            Other(u32),
        }

        impl BreakReason {
            /// Every reason Eden names, in number order.
            pub const NAMED: &'static [Self] = &[$(Self::$name,)*];

            /// The reason with number `raw`, which must not have [`NOTIFICATION_ONLY_FLAG`] set
            /// (a number with it set is [`BreakReason::Other`]).
            #[must_use]
            pub const fn from_raw(raw: u32) -> Self {
                match raw {
                    $($number => Self::$name,)*
                    other => Self::Other(other),
                }
            }

            /// The reason's number.
            #[must_use]
            pub const fn raw(self) -> u32 {
                match self {
                    $(Self::$name => $number,)*
                    Self::Other(raw) => raw,
                }
            }

            /// The reason in words (`panic`, `pre_load_dll`), or `None` for a reason Eden does
            /// not name.
            #[must_use]
            pub const fn name(self) -> Option<&'static str> {
                match self {
                    $(Self::$name => Some($text),)*
                    Self::Other(_) => None,
                }
            }
        }
    };
}

break_reasons! {
    /// The program panicked.
    Panic = 0, "panic";
    /// An assertion failed.
    Assert = 1, "assert";
    /// The program broke on purpose.
    User = 2, "user";
    /// A module is about to be loaded.
    PreLoadDll = 3, "pre_load_dll";
    /// A module was loaded.
    PostLoadDll = 4, "post_load_dll";
    /// A module is about to be unloaded.
    PreUnloadDll = 5, "pre_unload_dll";
    /// A module was unloaded.
    PostUnloadDll = 6, "post_unload_dll";
    /// An uncaught C++ exception.
    CppException = 7, "cpp_exception";
}

impl BreakReason {
    /// Whether Eden reads the buffer of a notification-only break with this reason: its
    /// `switch` calls `handle_debug_buffer` for `Panic`, `Assert`, `User` and its `default`, and
    /// not for the four module notices or `CppException`.
    const fn reads_buffer(self) -> bool {
        matches!(
            self,
            Self::Panic | Self::Assert | Self::User | Self::Other(_)
        )
    }
}

/// One `Break` call, decoded, with the code its buffer held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Break {
    /// Why it broke.
    pub reason: BreakReason,
    /// Whether [`NOTIFICATION_ONLY_FLAG`] was set: the program goes on.
    pub notification_only: bool,
    /// The buffer's address (`arg`, `x1`).
    pub address: u64,
    /// The buffer's size (`size`, `x2`).
    pub size: u64,
    /// The word in the buffer, when Eden reads it as an error code (see the module docs) and
    /// it was readable; `None` otherwise.
    pub code: Option<u32>,
}

impl Break {
    /// The call `Break(raw_reason, address, size)`, its buffer not read ([`Break::code`] is
    /// `None`).
    #[must_use]
    pub const fn decode(raw_reason: u32, address: u64, size: u64) -> Self {
        Self {
            reason: BreakReason::from_raw(raw_reason & !NOTIFICATION_ONLY_FLAG),
            notification_only: raw_reason & NOTIFICATION_ONLY_FLAG != 0,
            address,
            size,
            code: None,
        }
    }

    /// The reason word the guest passed: the reason, with [`NOTIFICATION_ONLY_FLAG`] when the
    /// break is only a notification. [`Break::decode`] gives this break back from it.
    #[must_use]
    pub const fn raw_reason(&self) -> u32 {
        let flag = if self.notification_only {
            NOTIFICATION_ONLY_FLAG
        } else {
            0
        };
        self.reason.raw() | flag
    }

    /// Whether Eden reads this break's buffer as a code: the reason or the break asks for it
    /// (`handle_debug_buffer` is called), the buffer has an address (`addr == 0` returns), and
    /// it is [`RESULT_ARGUMENT_SIZE`] bytes (which also covers Eden's `sz == 0`).
    const fn reads_code(&self) -> bool {
        (self.reason.reads_buffer() || !self.notification_only)
            && self.address != 0
            && self.size == RESULT_ARGUMENT_SIZE
    }
}

/// Handles `Break(raw_reason, address, size)` from a program whose memory is `memory`: decodes
/// it and reads the buffer's code as Eden does (see the module docs). Logs the break; what
/// happens to the program is the dispatcher's to say.
pub fn break_execution(
    memory: &dyn GuestMemory,
    raw_reason: u32,
    address: u64,
    size: u64,
) -> Break {
    let mut call = Break::decode(raw_reason, address, size);
    // The whole extent must fit the address space before anything is read.
    if call.reads_code() && address.checked_add(RESULT_ARGUMENT_SIZE).is_some() {
        let mut word = [0; size_of::<u32>()];
        call.code = memory
            .read(address, &mut word)
            .ok()
            .map(|()| u32::from_le_bytes(word));
    }
    let reason = call.reason.name().unwrap_or("unknown");
    if call.notification_only {
        tracing::info!(
            reason,
            raw_reason,
            address,
            size,
            code = call.code,
            "the program sent a break notification"
        );
    } else {
        tracing::error!(
            reason,
            raw_reason,
            address,
            size,
            code = call.code,
            "the program called Break"
        );
    }
    call
}

#[cfg(test)]
mod tests;

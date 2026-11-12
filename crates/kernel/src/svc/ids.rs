//! Every Horizon system-call number, with its name and its status in starquake.
//!
//! The numbers and names are Eden's `SvcId` enum (`eden/src/core/hle/kernel/svc.h` @ cf654052,
//! GPL-3.0-or-later) plus the Switch 2 call `SetGpuMemoryAttribute = 0x80` from
//! `eden-pr4588/src/core/hle/kernel/svc.h` @ e5fb3b97 (scoping doc, "Process, memory, kernel":
//! "New SVC: `svc 0x80`"; Eden-NX2 notes rows 27 and 47). The tests check them against a copy of
//! that enum (`reference/svc_ids.txt`).
//!
//! Statuses follow the kernel spec ("What is built now, what is deferred, and what is not
//! planned", rule 6):
//! - [`Status::Implemented`]: the dispatcher runs a real implementation.
//! - [`Status::NotPlanned`]: the hardware-facing families the spec names, and the calls this table
//!   counts in them: I/O pools and regions (`CreateIoPool` 0x39, `CreateIoRegion` 0x3a,
//!   `MapIoRegion` 0x46, `UnmapIoRegion` 0x47, `QueryIoMapping` 0x55), interrupt events
//!   (`CreateInterruptEvent` 0x53), device address spaces (0x56-0x5a, 0x5c), the secure monitor
//!   (`CallSecureMonitor` 0x7f), and two calls that reach physical hardware directly
//!   (`ReadWriteRegister` 0x4e, `QueryPhysicalAddress` 0x54). Counting `QueryIoMapping`,
//!   `ReadWriteRegister` and `QueryPhysicalAddress` as hardware-facing is this table's reading of
//!   the spec, which names only the families.
//! - [`Status::Deferred`]: every other call. That covers both calls the spec builds now whose port
//!   slice hasn't landed yet and the spec's "deferred, designed for" column.
//!
//! A call that isn't [`Status::Implemented`], and a number that isn't listed at all, returns
//! [`KernelError::NotImplemented`](crate::result::KernelError::NotImplemented) from the
//! dispatcher.

/// Where a system call stands in starquake (kernel spec, rule 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    /// The dispatcher runs a real implementation.
    Implemented,
    /// Not implemented yet, and designed for: returns "not implemented" and is logged.
    Deferred,
    /// A hardware-facing call starquake doesn't plan to emulate: returns "not implemented" and is
    /// logged.
    NotPlanned,
}

/// Defines [`SvcId`] from `Name = number, Status;` rows, so the number, the name and the status
/// of a call are written once.
macro_rules! svc_ids {
    ($($name:ident = $number:literal, $status:ident;)*) => {
        /// A Horizon system-call number (the immediate of `svc #imm`).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(u16)]
        pub enum SvcId {
            $(
                #[doc = concat!("`svc ", stringify!($number), "`: ", stringify!($name), ".")]
                $name = $number,
            )*
        }

        impl SvcId {
            /// Every listed call, in number order.
            pub const ALL: &'static [Self] = &[$(Self::$name,)*];

            /// The call for an `svc` immediate, or `None` if Horizon has no call with that number.
            #[must_use]
            pub const fn from_number(number: u16) -> Option<Self> {
                match number {
                    $($number => Some(Self::$name),)*
                    _ => None,
                }
            }

            /// The `svc` immediate.
            #[must_use]
            pub const fn number(self) -> u16 {
                self as u16
            }

            /// The call's name, as Eden spells it.
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$name => stringify!($name),)*
                }
            }

            /// Where the call stands in starquake.
            #[must_use]
            pub const fn status(self) -> Status {
                match self {
                    $(Self::$name => Status::$status,)*
                }
            }
        }
    };
}

svc_ids! {
    SetHeapSize = 0x01, Deferred;
    SetMemoryPermission = 0x02, Deferred;
    SetMemoryAttribute = 0x03, Deferred;
    MapMemory = 0x04, Deferred;
    UnmapMemory = 0x05, Deferred;
    QueryMemory = 0x06, Deferred;
    ExitProcess = 0x07, Implemented;
    CreateThread = 0x08, Deferred;
    StartThread = 0x09, Deferred;
    ExitThread = 0x0a, Deferred;
    SleepThread = 0x0b, Deferred;
    GetThreadPriority = 0x0c, Deferred;
    SetThreadPriority = 0x0d, Deferred;
    GetThreadCoreMask = 0x0e, Deferred;
    SetThreadCoreMask = 0x0f, Deferred;
    GetCurrentProcessorNumber = 0x10, Implemented;
    SignalEvent = 0x11, Deferred;
    ClearEvent = 0x12, Deferred;
    MapSharedMemory = 0x13, Deferred;
    UnmapSharedMemory = 0x14, Deferred;
    CreateTransferMemory = 0x15, Deferred;
    CloseHandle = 0x16, Deferred;
    ResetSignal = 0x17, Deferred;
    WaitSynchronization = 0x18, Deferred;
    CancelSynchronization = 0x19, Deferred;
    ArbitrateLock = 0x1a, Deferred;
    ArbitrateUnlock = 0x1b, Deferred;
    WaitProcessWideKeyAtomic = 0x1c, Deferred;
    SignalProcessWideKey = 0x1d, Deferred;
    GetSystemTick = 0x1e, Implemented;
    ConnectToNamedPort = 0x1f, Deferred;
    SendSyncRequestLight = 0x20, Deferred;
    SendSyncRequest = 0x21, Deferred;
    SendSyncRequestWithUserBuffer = 0x22, Deferred;
    SendAsyncRequestWithUserBuffer = 0x23, Deferred;
    GetProcessId = 0x24, Deferred;
    GetThreadId = 0x25, Deferred;
    Break = 0x26, Deferred;
    OutputDebugString = 0x27, Implemented;
    ReturnFromException = 0x28, Deferred;
    GetInfo = 0x29, Deferred;
    FlushEntireDataCache = 0x2a, Deferred;
    FlushDataCache = 0x2b, Deferred;
    MapPhysicalMemory = 0x2c, Deferred;
    UnmapPhysicalMemory = 0x2d, Deferred;
    GetDebugFutureThreadInfo = 0x2e, Deferred;
    GetLastThreadInfo = 0x2f, Deferred;
    GetResourceLimitLimitValue = 0x30, Deferred;
    GetResourceLimitCurrentValue = 0x31, Deferred;
    SetThreadActivity = 0x32, Deferred;
    GetThreadContext3 = 0x33, Deferred;
    WaitForAddress = 0x34, Deferred;
    SignalToAddress = 0x35, Deferred;
    SynchronizePreemptionState = 0x36, Deferred;
    GetResourceLimitPeakValue = 0x37, Deferred;
    CreateIoPool = 0x39, NotPlanned;
    CreateIoRegion = 0x3a, NotPlanned;
    KernelDebug = 0x3c, Deferred;
    ChangeKernelTraceState = 0x3d, Deferred;
    CreateSession = 0x40, Deferred;
    AcceptSession = 0x41, Deferred;
    ReplyAndReceiveLight = 0x42, Deferred;
    ReplyAndReceive = 0x43, Deferred;
    ReplyAndReceiveWithUserBuffer = 0x44, Deferred;
    CreateEvent = 0x45, Deferred;
    MapIoRegion = 0x46, NotPlanned;
    UnmapIoRegion = 0x47, NotPlanned;
    MapPhysicalMemoryUnsafe = 0x48, Deferred;
    UnmapPhysicalMemoryUnsafe = 0x49, Deferred;
    SetUnsafeLimit = 0x4a, Deferred;
    CreateCodeMemory = 0x4b, Deferred;
    ControlCodeMemory = 0x4c, Deferred;
    SleepSystem = 0x4d, Deferred;
    ReadWriteRegister = 0x4e, NotPlanned;
    SetProcessActivity = 0x4f, Deferred;
    CreateSharedMemory = 0x50, Deferred;
    MapTransferMemory = 0x51, Deferred;
    UnmapTransferMemory = 0x52, Deferred;
    CreateInterruptEvent = 0x53, NotPlanned;
    QueryPhysicalAddress = 0x54, NotPlanned;
    QueryIoMapping = 0x55, NotPlanned;
    CreateDeviceAddressSpace = 0x56, NotPlanned;
    AttachDeviceAddressSpace = 0x57, NotPlanned;
    DetachDeviceAddressSpace = 0x58, NotPlanned;
    MapDeviceAddressSpaceByForce = 0x59, NotPlanned;
    MapDeviceAddressSpaceAligned = 0x5a, NotPlanned;
    UnmapDeviceAddressSpace = 0x5c, NotPlanned;
    InvalidateProcessDataCache = 0x5d, Deferred;
    StoreProcessDataCache = 0x5e, Deferred;
    FlushProcessDataCache = 0x5f, Deferred;
    DebugActiveProcess = 0x60, Deferred;
    BreakDebugProcess = 0x61, Deferred;
    TerminateDebugProcess = 0x62, Deferred;
    GetDebugEvent = 0x63, Deferred;
    ContinueDebugEvent = 0x64, Deferred;
    GetProcessList = 0x65, Deferred;
    GetThreadList = 0x66, Deferred;
    GetDebugThreadContext = 0x67, Deferred;
    SetDebugThreadContext = 0x68, Deferred;
    QueryDebugProcessMemory = 0x69, Deferred;
    ReadDebugProcessMemory = 0x6a, Deferred;
    WriteDebugProcessMemory = 0x6b, Deferred;
    SetHardwareBreakPoint = 0x6c, Deferred;
    GetDebugThreadParam = 0x6d, Deferred;
    GetSystemInfo = 0x6f, Deferred;
    CreatePort = 0x70, Deferred;
    ManageNamedPort = 0x71, Deferred;
    ConnectToPort = 0x72, Deferred;
    SetProcessMemoryPermission = 0x73, Deferred;
    MapProcessMemory = 0x74, Deferred;
    UnmapProcessMemory = 0x75, Deferred;
    QueryProcessMemory = 0x76, Deferred;
    MapProcessCodeMemory = 0x77, Deferred;
    UnmapProcessCodeMemory = 0x78, Deferred;
    CreateProcess = 0x79, Deferred;
    StartProcess = 0x7a, Deferred;
    TerminateProcess = 0x7b, Deferred;
    GetProcessInfo = 0x7c, Deferred;
    CreateResourceLimit = 0x7d, Deferred;
    SetResourceLimitLimitValue = 0x7e, Deferred;
    CallSecureMonitor = 0x7f, NotPlanned;
    SetGpuMemoryAttribute = 0x80, Deferred;
    MapInsecureMemory = 0x90, Deferred;
    UnmapInsecureMemory = 0x91, Deferred;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eden's `SvcId` enum, copied line for line (see the file's header).
    const REFERENCE: &str = include_str!("reference/svc_ids.txt");

    fn reference() -> Vec<(&'static str, u16)> {
        REFERENCE
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| {
                let (name, number) = l.split_once(" = ").unwrap();
                let number = number.trim().trim_start_matches("0x");
                (name.trim(), u16::from_str_radix(number, 16).unwrap())
            })
            .collect()
    }

    #[test]
    fn every_id_matches_the_reference_enum_by_name_and_number() {
        let ours: Vec<_> = SvcId::ALL.iter().map(|s| (s.name(), s.number())).collect();
        let reference = reference();
        for (i, (name, number)) in reference.iter().enumerate() {
            assert_eq!(ours.get(i), Some(&(*name, *number)), "entry {i}");
        }
        assert_eq!(ours.len(), reference.len());
    }

    #[test]
    fn from_number_finds_every_listed_call_and_nothing_else() {
        for &id in SvcId::ALL {
            assert_eq!(SvcId::from_number(id.number()), Some(id), "{}", id.name());
        }
        let listed = (0..=u16::MAX)
            .filter(|&n| SvcId::from_number(n).is_some())
            .count();
        assert_eq!(listed, SvcId::ALL.len());
        // Gaps in Horizon's numbering stay unknown.
        for (name, number) in reference() {
            if let Some(next) = number.checked_add(1)
                && reference().iter().all(|(_, n)| *n != next)
            {
                assert_eq!(SvcId::from_number(next), None, "after {name}");
            }
        }
    }

    #[test]
    fn ids_are_in_number_order() {
        for pair in SvcId::ALL.windows(2) {
            assert!(pair[0].number() < pair[1].number(), "{pair:?}");
        }
    }

    /// The hardware-facing families of the kernel spec, recognised by name independently of the
    /// table, so a status typed into the wrong row shows up here.
    fn hardware_facing(name: &str) -> bool {
        [
            "IoPool",
            "IoRegion",
            "IoMapping",
            "InterruptEvent",
            "DeviceAddressSpace",
            "SecureMonitor",
        ]
        .iter()
        .any(|family| name.contains(family))
            || matches!(name, "ReadWriteRegister" | "QueryPhysicalAddress")
    }

    #[test]
    fn every_status_follows_the_spec() {
        for &id in SvcId::ALL {
            let expected = match id {
                SvcId::ExitProcess
                | SvcId::GetCurrentProcessorNumber
                | SvcId::GetSystemTick
                | SvcId::OutputDebugString => Status::Implemented,
                _ if hardware_facing(id.name()) => Status::NotPlanned,
                _ => Status::Deferred,
            };
            assert_eq!(id.status(), expected, "{} ({:#x})", id.name(), id.number());
        }
    }

    #[test]
    fn server_side_ipc_is_deferred_not_dropped() {
        // Kernel spec, "Deferred, and designed for": server-side IPC.
        for id in [
            SvcId::CreateSession,
            SvcId::AcceptSession,
            SvcId::ReplyAndReceive,
            SvcId::ReplyAndReceiveLight,
            SvcId::ReplyAndReceiveWithUserBuffer,
        ] {
            assert_eq!(id.status(), Status::Deferred, "{}", id.name());
        }
    }
}

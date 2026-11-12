//! Horizon kernel result codes.
//!
//! A Horizon result is a 32-bit word: the module that produced it in bits 0-8 and a description in
//! bits 9-21; zero is success. Every kernel error has module [`KERNEL_MODULE`]. This module is
//! the kernel's one encoding of them: every other module names a result as a [`KernelError`] and
//! gets its word from [`KernelError::code`].
//!
//! # What a system call returns
//!
//! - [`SvcResult`] when the only way it fails is with a result the guest sees.
//! - [`CallResult`] when it works through the tables. Its error, [`SvcError`], is either a
//!   result for the guest or [`SvcError::Internal`]: the kernel's own bookkeeping failed. Each
//!   table's error type converts into it in one place (`From<HandleError>`, `From<SchedError>`,
//!   `From<ObjectError>`), so a system call propagates a table's failure with `?` and never
//!   decides for itself what a broken table looks like to a guest.
//!
//! The dispatcher turns either into the word in `x0`, and is the one place that decides what
//! happens on an internal failure (see its module docs).
//!
//! Ported from Eden (GPL-3.0-or-later), paths relative to the reference folder
//! (`docs/references.md`):
//! - `eden/src/core/hle/result.h` @ cf654052: the bit layout (`Module = BitField<0, 9>`,
//!   `Description = BitField<9, 13>`), success as raw 0, and `ErrorModule::Kernel = 1`.
//! - `eden/src/core/hle/kernel/svc_results.h` @ cf654052: the kernel's descriptions. The tests
//!   check [`KernelError`] against a copy of that list (`svc/reference/svc_results.txt`).

/// The error module of every Horizon kernel result (`ErrorModule::Kernel` in Eden's `result.h`).
pub const KERNEL_MODULE: u32 = 1;

/// Width of the module field, bits 0 up (`BitField<0, 9>` in Eden's `result.h`).
pub const MODULE_BITS: u32 = 9;

/// Width of the description field, starting at bit [`MODULE_BITS`] (`BitField<9, 13>` in Eden's
/// `result.h`).
pub const DESCRIPTION_BITS: u32 = 13;

const MODULE_MASK: u32 = (1 << MODULE_BITS) - 1;
const DESCRIPTION_MASK: u32 = (1 << DESCRIPTION_BITS) - 1;

/// A raw Horizon result word, as the guest sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResultCode(u32);

impl ResultCode {
    /// Success: the all-zero word (`ResultSuccess(0)` in Eden's `result.h`).
    pub const SUCCESS: Self = Self(0);

    /// Wraps a raw result word. Bits above the description are kept as given.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Builds a result from its module and description, or `None` if either is too wide for its
    /// field ([`MODULE_BITS`], [`DESCRIPTION_BITS`]).
    #[must_use]
    pub const fn new(module: u32, description: u32) -> Option<Self> {
        if module > MODULE_MASK || description > DESCRIPTION_MASK {
            return None;
        }
        Some(Self(module | (description << MODULE_BITS)))
    }

    /// The raw word.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// The module field (bits 0-8).
    #[must_use]
    pub const fn module(self) -> u32 {
        self.0 & MODULE_MASK
    }

    /// The description field (bits 9-21).
    #[must_use]
    pub const fn description(self) -> u32 {
        (self.0 >> MODULE_BITS) & DESCRIPTION_MASK
    }

    /// Whether this is [`ResultCode::SUCCESS`].
    #[must_use]
    pub const fn is_success(self) -> bool {
        self.0 == Self::SUCCESS.0
    }
}

/// Defines [`KernelError`] from `(variant, description, message)` rows, so the numbers and the
/// list used by the tests come from one place.
macro_rules! kernel_errors {
    ($($name:ident = $description:literal, $message:literal;)*) => {
        /// A Horizon kernel error: a description in module [`KERNEL_MODULE`]. The variants and
        /// their numbers are Eden's `svc_results.h`, one for one.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
        pub enum KernelError {
            $(
                #[doc = concat!("Description ", stringify!($description), ": ", $message, ".")]
                #[error($message)]
                $name,
            )*
        }

        impl KernelError {
            /// Every kernel error, in the order of Eden's `svc_results.h`.
            pub const ALL: &'static [Self] = &[$(Self::$name,)*];

            /// The description number.
            #[must_use]
            pub const fn description(self) -> u32 {
                match self {
                    $(Self::$name => $description,)*
                }
            }

            /// Eden's name for the error, without its `Result` prefix.
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$name => stringify!($name),)*
                }
            }
        }
    };
}

kernel_errors! {
    OutOfSessions = 7, "out of sessions";
    InvalidArgument = 14, "invalid argument";
    NotImplemented = 33, "not implemented";
    NoSynchronizationObject = 57, "no synchronization object";
    TerminationRequested = 59, "termination requested";
    InvalidSize = 101, "invalid size";
    InvalidAddress = 102, "invalid address";
    OutOfResource = 103, "out of resource";
    OutOfMemory = 104, "out of memory";
    OutOfHandles = 105, "out of handles";
    InvalidCurrentMemory = 106, "invalid current memory";
    InvalidNewMemoryPermission = 108, "invalid new memory permission";
    InvalidMemoryRegion = 110, "invalid memory region";
    InvalidPriority = 112, "invalid priority";
    InvalidCoreId = 113, "invalid core id";
    InvalidHandle = 114, "invalid handle";
    InvalidPointer = 115, "invalid pointer";
    InvalidCombination = 116, "invalid combination";
    TimedOut = 117, "timed out";
    Cancelled = 118, "cancelled";
    OutOfRange = 119, "out of range";
    InvalidEnumValue = 120, "invalid enum value";
    NotFound = 121, "not found";
    Busy = 122, "busy";
    SessionClosed = 123, "session closed";
    InvalidState = 125, "invalid state";
    ReservedUsed = 126, "reserved used";
    PortClosed = 131, "port closed";
    LimitReached = 132, "limit reached";
    ReceiveListBroken = 258, "receive list broken";
    OutOfAddressSpace = 259, "out of address space";
    MessageTooLarge = 260, "message too large";
    InvalidId = 519, "invalid id";
}

impl KernelError {
    /// The result word the guest sees for this error.
    #[must_use]
    pub const fn code(self) -> ResultCode {
        // Every description in the table fits in DESCRIPTION_BITS (checked by a test), so the
        // shift cannot carry into higher bits.
        ResultCode(KERNEL_MODULE | (self.description() << MODULE_BITS))
    }

    /// The kernel error a result word stands for, or `None` if it is success, from another module,
    /// or a description this table doesn't list.
    #[must_use]
    pub fn from_code(code: ResultCode) -> Option<Self> {
        if code.module() != KERNEL_MODULE {
            return None;
        }
        Self::ALL
            .iter()
            .copied()
            .find(|e| e.code().raw() == code.raw())
    }
}

/// What a system call returns: a value, or the kernel error the guest sees in `x0`.
pub type SvcResult<T> = Result<T, KernelError>;

/// Why a system call that works through the kernel's tables produced no value.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SvcError {
    /// A Horizon result: the guest sees its code in `x0`.
    #[error(transparent)]
    Kernel(#[from] KernelError),
    /// The kernel's own bookkeeping is inconsistent: a table operation failed in a way no guest
    /// input explains (an id the kernel stored is stale, a handle names an object with no
    /// references). Horizon has no result for this; it asserts. The text is the table's own
    /// error message.
    #[error("the kernel's bookkeeping is inconsistent: {0}")]
    Internal(String),
}

impl SvcError {
    /// An internal failure, described by the error a table operation returned.
    pub fn internal(error: &dyn core::error::Error) -> Self {
        Self::Internal(error.to_string())
    }
}

/// What a system call that works through the tables returns: a value, or an [`SvcError`].
pub type CallResult<T> = Result<T, SvcError>;

/// The result word for a system call's outcome: [`ResultCode::SUCCESS`] or the error's code.
#[must_use]
pub fn result_code<T>(result: &SvcResult<T>) -> ResultCode {
    match result {
        Ok(_) => ResultCode::SUCCESS,
        Err(e) => e.code(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eden's list, copied from `svc_results.h` (see the file's header).
    const REFERENCE: &str = include_str!("svc/reference/svc_results.txt");

    fn reference() -> Vec<(&'static str, u32)> {
        REFERENCE
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| {
                let (name, number) = l.split_once(" = ").unwrap();
                (name.trim(), number.trim().parse().unwrap())
            })
            .collect()
    }

    #[test]
    fn every_kernel_error_matches_edens_list_in_order() {
        let ours: Vec<_> = KernelError::ALL
            .iter()
            .map(|e| (e.name(), e.description()))
            .collect();
        assert_eq!(ours, reference());
    }

    #[test]
    fn every_kernel_error_round_trips_through_its_code() {
        for &e in KernelError::ALL {
            let code = e.code();
            assert_eq!(code.module(), KERNEL_MODULE, "{e:?}");
            assert_eq!(code.description(), e.description(), "{e:?}");
            assert_eq!(ResultCode::new(KERNEL_MODULE, e.description()), Some(code));
            assert_eq!(KernelError::from_code(code), Some(e), "{e:?}");
            assert!(!code.is_success());
        }
    }

    #[test]
    fn descriptions_fit_their_field() {
        for &e in KernelError::ALL {
            assert!(e.description() < (1 << DESCRIPTION_BITS), "{e:?}");
        }
    }

    #[test]
    fn module_and_description_occupy_their_own_bits() {
        // Distinct, all-ones values in each field: a swapped or mis-shifted field changes the word.
        let module = (1 << MODULE_BITS) - 1;
        let description = (1 << DESCRIPTION_BITS) - 1;
        let code = ResultCode::new(module, description).unwrap();
        assert_eq!(code.raw(), (1 << (MODULE_BITS + DESCRIPTION_BITS)) - 1);
        assert_eq!(code.module(), module);
        assert_eq!(code.description(), description);

        let code = ResultCode::new(KERNEL_MODULE, 0).unwrap();
        assert_eq!(code.raw(), KERNEL_MODULE);
        let code = ResultCode::new(0, 1).unwrap();
        assert_eq!(code.raw(), 1 << MODULE_BITS);
        assert_eq!((code.module(), code.description()), (0, 1));
    }

    #[test]
    fn fields_wider_than_their_bits_are_rejected_at_the_limit() {
        assert!(ResultCode::new((1 << MODULE_BITS) - 1, 0).is_some());
        assert_eq!(ResultCode::new(1 << MODULE_BITS, 0), None);
        assert!(ResultCode::new(0, (1 << DESCRIPTION_BITS) - 1).is_some());
        assert_eq!(ResultCode::new(0, 1 << DESCRIPTION_BITS), None);
    }

    #[test]
    fn not_implemented_is_the_word_horizon_returns() {
        // svc_results.h: ResultNotImplemented{ErrorModule::Kernel, 33}.
        let (_, description) = reference()
            .into_iter()
            .find(|(n, _)| *n == "NotImplemented")
            .unwrap();
        assert_eq!(
            KernelError::NotImplemented.code().raw(),
            KERNEL_MODULE | (description << MODULE_BITS)
        );
    }

    #[test]
    fn from_code_rejects_success_other_modules_and_unknown_descriptions() {
        assert_eq!(KernelError::from_code(ResultCode::SUCCESS), None);
        let other_module =
            ResultCode::new(KERNEL_MODULE + 1, KernelError::InvalidHandle.description()).unwrap();
        assert_eq!(KernelError::from_code(other_module), None);
        let unlisted = (0..1 << DESCRIPTION_BITS)
            .find(|d| KernelError::ALL.iter().all(|e| e.description() != *d) && *d != 0)
            .unwrap();
        assert_eq!(
            KernelError::from_code(ResultCode::new(KERNEL_MODULE, unlisted).unwrap()),
            None
        );
    }

    #[test]
    fn result_code_maps_ok_to_success_and_err_to_its_code() {
        assert_eq!(result_code(&SvcResult::Ok(5u64)), ResultCode::SUCCESS);
        assert!(ResultCode::SUCCESS.is_success());
        assert_eq!(
            result_code::<()>(&Err(KernelError::InvalidCurrentMemory)),
            KernelError::InvalidCurrentMemory.code()
        );
    }

    #[test]
    fn a_kernel_error_becomes_the_guests_result_and_keeps_its_message() {
        fn call() -> CallResult<u32> {
            Err(KernelError::InvalidHandle)?
        }
        let error = SvcError::from(KernelError::InvalidHandle);
        assert_eq!(error, SvcError::Kernel(KernelError::InvalidHandle));
        assert_eq!(error.to_string(), KernelError::InvalidHandle.to_string());
        // `?` converts in a function that returns a `CallResult`.
        assert_eq!(call(), Err(SvcError::Kernel(KernelError::InvalidHandle)));
    }

    #[test]
    fn an_internal_failure_carries_the_tables_message() {
        let error = SvcError::internal(&KernelError::InvalidState);
        assert_eq!(
            error,
            SvcError::Internal(KernelError::InvalidState.to_string())
        );
        assert_eq!(
            error.to_string(),
            "the kernel's bookkeeping is inconsistent: invalid state"
        );
    }

    #[test]
    fn error_messages_are_stable() {
        assert_eq!(KernelError::NotImplemented.to_string(), "not implemented");
        assert_eq!(
            KernelError::InvalidCurrentMemory.to_string(),
            "invalid current memory"
        );
    }
}

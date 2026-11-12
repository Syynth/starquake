//! `RenderError`: why the backend refused a description, a command or a call.
//!
//! Every description is derived from guest state, so a bad one is an error the GPU crate logs,
//! counts and skips (docs/specs/render-backend.md, "Errors and untrusted input"). Messages are
//! plain words; the tests assert them exactly.

use crate::format::PixelFormat;
use crate::shader::ShaderStage;
use thiserror::Error;

/// Why the backend refused something.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum RenderError {
    /// A number is outside the range the device (and, where it is smaller, the guest GPU)
    /// allows.
    #[error("{what} {value} is outside {min}..={max}")]
    OutOfRange {
        /// What the number is, in words ("texture width").
        what: &'static str,
        /// The value given.
        value: u64,
        /// The smallest value allowed.
        min: u64,
        /// The largest value allowed.
        max: u64,
    },
    /// An index (a mip level, a bind group, a location) is not below the count it indexes.
    #[error("{what} {index} is not below {count}")]
    NotBelow {
        /// What the index is, in words ("mip level").
        what: &'static str,
        /// The index given.
        index: u64,
        /// How many there are.
        count: u64,
    },
    /// A byte count does not match what a copy or upload needs.
    #[error("{what} is {actual} bytes, but {expected} are needed")]
    SizeMismatch {
        /// What was measured, in words ("staging slice").
        what: &'static str,
        /// The bytes needed.
        expected: u64,
        /// The bytes given.
        actual: u64,
    },
    /// Arithmetic on a description's numbers does not fit in 64 bits.
    #[error("{what} overflows")]
    Overflow {
        /// What was being computed, in words ("buffer range end").
        what: &'static str,
    },
    /// A range or region ends past the resource it names.
    #[error("{what} ends at {end}, past the end at {limit}")]
    PastEnd {
        /// What the range is, in words ("buffer range").
        what: &'static str,
        /// One past the last element the range covers.
        end: u64,
        /// One past the last element the resource has.
        limit: u64,
    },
    /// The device cannot use a format the way a description asks.
    #[error("{format} cannot be used for {usage} on this device")]
    FormatUnsupported {
        /// The format.
        format: PixelFormat,
        /// The use, in words ("render targets").
        usage: &'static str,
    },
    /// The device does not support a sample count.
    #[error("sample count {0} is not supported by this device")]
    SampleCount(u32),
    /// The description needs an optional device feature the device lacks.
    #[error("the device lacks {0}")]
    MissingFeature(&'static str),
    /// The device cannot run a shader stage (geometry and tessellation under wgpu).
    #[error("the device cannot run {} shaders", .0.name())]
    StageUnsupported(ShaderStage),
    /// A resource is used in a way its usage flags do not allow.
    #[error("the {resource} lacks the {usage} usage")]
    MissingUsage {
        /// Which resource, in words ("copy source texture").
        resource: &'static str,
        /// The usage it needs, in words ("copy source").
        usage: &'static str,
    },
    /// A description breaks a structural rule; the text says which.
    #[error("{0}")]
    Invalid(&'static str),
    /// A flag set has bits that name no flag.
    #[error(transparent)]
    UnknownFlags(#[from] UnknownFlags),
    /// The host device was lost; the one error that ends the run.
    #[error("the host device was lost")]
    DeviceLost,
}

/// A flag set's bits include some that name no flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("{name} has unknown bits {bits:#x}")]
pub struct UnknownFlags {
    /// The flag set's type name.
    pub name: &'static str,
    /// The bits that name no flag.
    pub bits: u32,
}

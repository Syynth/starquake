//! Which operations the texture pass rewrites, and into what: Eden's `IndexedInstruction`,
//! `IsBindless` and `IsStorageImageOpcode` (`eden/src/shader_recompiler/ir_opt/texture_pass.cpp`
//! @ cf654052; the same three functions in `eden-pr4588` @ e5fb3b97), extended with the
//! unbound `Image*` forms the texture translator emits (slice 12f; see the parent module's
//! "Handle forms").

use crate::ir::Opcode;

/// How a texture operation's first argument names its handle, before the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Form {
    /// `BoundImage*`: an immediate byte offset in the program's texture descriptor buffer
    /// (Eden's bound form).
    Bound,
    /// `BindlessImage*`: a value to trace to the constant buffer it was loaded from (Eden's
    /// bindless form).
    Bindless,
    /// `Image*`, as the texture translator emits it: an immediate offset is a bound handle and
    /// a value a bindless one (`crate::translate::texture`, "Handles").
    Unbound,
}

/// One row of the table: the bound and bindless forms of an operation and its unbound form,
/// which the pass leaves (Eden's `IndexedInstruction`).
const ROWS: [(Opcode, Opcode, Opcode); 23] = [
    (
        Opcode::BoundImageSampleImplicitLod,
        Opcode::BindlessImageSampleImplicitLod,
        Opcode::ImageSampleImplicitLod,
    ),
    (
        Opcode::BoundImageSampleExplicitLod,
        Opcode::BindlessImageSampleExplicitLod,
        Opcode::ImageSampleExplicitLod,
    ),
    (
        Opcode::BoundImageSampleDrefImplicitLod,
        Opcode::BindlessImageSampleDrefImplicitLod,
        Opcode::ImageSampleDrefImplicitLod,
    ),
    (
        Opcode::BoundImageSampleDrefExplicitLod,
        Opcode::BindlessImageSampleDrefExplicitLod,
        Opcode::ImageSampleDrefExplicitLod,
    ),
    (
        Opcode::BoundImageGather,
        Opcode::BindlessImageGather,
        Opcode::ImageGather,
    ),
    (
        Opcode::BoundImageGatherDref,
        Opcode::BindlessImageGatherDref,
        Opcode::ImageGatherDref,
    ),
    (
        Opcode::BoundImageFetch,
        Opcode::BindlessImageFetch,
        Opcode::ImageFetch,
    ),
    (
        Opcode::BoundImageQueryDimensions,
        Opcode::BindlessImageQueryDimensions,
        Opcode::ImageQueryDimensions,
    ),
    (
        Opcode::BoundImageQueryLod,
        Opcode::BindlessImageQueryLod,
        Opcode::ImageQueryLod,
    ),
    (
        Opcode::BoundImageGradient,
        Opcode::BindlessImageGradient,
        Opcode::ImageGradient,
    ),
    (
        Opcode::BoundImageRead,
        Opcode::BindlessImageRead,
        Opcode::ImageRead,
    ),
    (
        Opcode::BoundImageWrite,
        Opcode::BindlessImageWrite,
        Opcode::ImageWrite,
    ),
    (
        Opcode::BoundImageAtomicIAdd32,
        Opcode::BindlessImageAtomicIAdd32,
        Opcode::ImageAtomicIAdd32,
    ),
    (
        Opcode::BoundImageAtomicSMin32,
        Opcode::BindlessImageAtomicSMin32,
        Opcode::ImageAtomicSMin32,
    ),
    (
        Opcode::BoundImageAtomicUMin32,
        Opcode::BindlessImageAtomicUMin32,
        Opcode::ImageAtomicUMin32,
    ),
    (
        Opcode::BoundImageAtomicSMax32,
        Opcode::BindlessImageAtomicSMax32,
        Opcode::ImageAtomicSMax32,
    ),
    (
        Opcode::BoundImageAtomicUMax32,
        Opcode::BindlessImageAtomicUMax32,
        Opcode::ImageAtomicUMax32,
    ),
    (
        Opcode::BoundImageAtomicInc32,
        Opcode::BindlessImageAtomicInc32,
        Opcode::ImageAtomicInc32,
    ),
    (
        Opcode::BoundImageAtomicDec32,
        Opcode::BindlessImageAtomicDec32,
        Opcode::ImageAtomicDec32,
    ),
    (
        Opcode::BoundImageAtomicAnd32,
        Opcode::BindlessImageAtomicAnd32,
        Opcode::ImageAtomicAnd32,
    ),
    (
        Opcode::BoundImageAtomicOr32,
        Opcode::BindlessImageAtomicOr32,
        Opcode::ImageAtomicOr32,
    ),
    (
        Opcode::BoundImageAtomicXor32,
        Opcode::BindlessImageAtomicXor32,
        Opcode::ImageAtomicXor32,
    ),
    (
        Opcode::BoundImageAtomicExchange32,
        Opcode::BindlessImageAtomicExchange32,
        Opcode::ImageAtomicExchange32,
    ),
];

/// The form of `opcode` and the unbound operation the pass turns it into, or `None` for an
/// operation the pass does not touch (Eden's `IsTextureInstruction` is false).
pub(super) fn classify(opcode: Opcode) -> Option<(Form, Opcode)> {
    ROWS.iter().find_map(|&(bound, bindless, unbound)| {
        if opcode == bound {
            Some((Form::Bound, unbound))
        } else if opcode == bindless {
            Some((Form::Bindless, unbound))
        } else if opcode == unbound {
            Some((Form::Unbound, unbound))
        } else {
            None
        }
    })
}

/// Whether `opcode`, an unbound operation, reads or writes a storage image (Eden's
/// `IsStorageImageOpcode`): `ImageRead`, `ImageWrite` and the image atomics.
pub(super) const fn is_storage_image(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::ImageRead
            | Opcode::ImageWrite
            | Opcode::ImageAtomicIAdd32
            | Opcode::ImageAtomicSMin32
            | Opcode::ImageAtomicUMin32
            | Opcode::ImageAtomicSMax32
            | Opcode::ImageAtomicUMax32
            | Opcode::ImageAtomicInc32
            | Opcode::ImageAtomicDec32
            | Opcode::ImageAtomicAnd32
            | Opcode::ImageAtomicOr32
            | Opcode::ImageAtomicXor32
            | Opcode::ImageAtomicExchange32
    )
}

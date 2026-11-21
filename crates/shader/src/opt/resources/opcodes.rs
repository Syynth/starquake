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

/// A `match` on `$opcode` over rows of the bound and bindless forms of an operation and its
/// unbound form, which the pass leaves (Eden's `IndexedInstruction`): one arm per opcode, so
/// classifying an instruction is one jump, not a scan of the rows.
macro_rules! classify_rows {
    ($opcode:expr; $(($bound:ident, $bindless:ident, $unbound:ident),)+) => {
        match $opcode {
            $(
                Opcode::$bound => Some((Form::Bound, Opcode::$unbound)),
                Opcode::$bindless => Some((Form::Bindless, Opcode::$unbound)),
                Opcode::$unbound => Some((Form::Unbound, Opcode::$unbound)),
            )+
            _ => None,
        }
    };
}

/// The form of `opcode` and the unbound operation the pass turns it into, or `None` for an
/// operation the pass does not touch (Eden's `IsTextureInstruction` is false). The rows are
/// pinned against Eden's table by `tests::forms::every_row_matches_edens_table`.
pub(super) const fn classify(opcode: Opcode) -> Option<(Form, Opcode)> {
    classify_rows!(opcode;
        (BoundImageSampleImplicitLod, BindlessImageSampleImplicitLod, ImageSampleImplicitLod),
        (BoundImageSampleExplicitLod, BindlessImageSampleExplicitLod, ImageSampleExplicitLod),
        (BoundImageSampleDrefImplicitLod, BindlessImageSampleDrefImplicitLod, ImageSampleDrefImplicitLod),
        (BoundImageSampleDrefExplicitLod, BindlessImageSampleDrefExplicitLod, ImageSampleDrefExplicitLod),
        (BoundImageGather, BindlessImageGather, ImageGather),
        (BoundImageGatherDref, BindlessImageGatherDref, ImageGatherDref),
        (BoundImageFetch, BindlessImageFetch, ImageFetch),
        (BoundImageQueryDimensions, BindlessImageQueryDimensions, ImageQueryDimensions),
        (BoundImageQueryLod, BindlessImageQueryLod, ImageQueryLod),
        (BoundImageGradient, BindlessImageGradient, ImageGradient),
        (BoundImageRead, BindlessImageRead, ImageRead),
        (BoundImageWrite, BindlessImageWrite, ImageWrite),
        (BoundImageAtomicIAdd32, BindlessImageAtomicIAdd32, ImageAtomicIAdd32),
        (BoundImageAtomicSMin32, BindlessImageAtomicSMin32, ImageAtomicSMin32),
        (BoundImageAtomicUMin32, BindlessImageAtomicUMin32, ImageAtomicUMin32),
        (BoundImageAtomicSMax32, BindlessImageAtomicSMax32, ImageAtomicSMax32),
        (BoundImageAtomicUMax32, BindlessImageAtomicUMax32, ImageAtomicUMax32),
        (BoundImageAtomicInc32, BindlessImageAtomicInc32, ImageAtomicInc32),
        (BoundImageAtomicDec32, BindlessImageAtomicDec32, ImageAtomicDec32),
        (BoundImageAtomicAnd32, BindlessImageAtomicAnd32, ImageAtomicAnd32),
        (BoundImageAtomicOr32, BindlessImageAtomicOr32, ImageAtomicOr32),
        (BoundImageAtomicXor32, BindlessImageAtomicXor32, ImageAtomicXor32),
        (BoundImageAtomicExchange32, BindlessImageAtomicExchange32, ImageAtomicExchange32),
    )
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

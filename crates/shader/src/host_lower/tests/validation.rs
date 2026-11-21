//! The lowered modules validate with naga under the reduced capabilities: after the passes,
//! every mapped opcode's program lowers for a target without 16- or 64-bit floats, needs
//! neither, and its module validates with capabilities that grant neither; the 64-bit integer
//! conversions are the lowering's own refusals, which the passes do not change.

use naga::valid::{Capabilities, Validator};

use super::super::{HostPass, lower_for_host};
use super::{first, runnable, single};
use crate::host::{Feature, HostCaps, HostCode};
use crate::lower::{LowerError, LowerInputs, capabilities, lower_with, validation_flags};
use crate::translate::GuestStage;

/// A target without 16- or 64-bit floats and without subgroups, but with 64-bit integers (the
/// `BitCast*` conversions of the f64 pass take a 64-bit integer).
const TARGET: HostCaps = HostCaps {
    f16: false,
    f64: false,
    i64: true,
    subgroup: false,
    subgroup_size_32: false,
};

/// The renamed operations the lowering refuses itself: it has no 64-bit integer conversions
/// from or to floats (`crate::lower`, "Operations").
fn refused_by_the_lowering(opcode: crate::ir::Opcode) -> Option<crate::ir::Opcode> {
    use crate::ir::Opcode as O;
    Some(match opcode {
        O::ConvertS64F64 | O::ConvertS64F16 => O::ConvertS64F32,
        O::ConvertU64F64 | O::ConvertU64F16 => O::ConvertU64F32,
        O::ConvertF64S64 | O::ConvertF16S64 => O::ConvertF32S64,
        O::ConvertF64U64 | O::ConvertF16U64 => O::ConvertF32U64,
        _ => return None,
    })
}

#[test]
fn lowered_modules_validate_without_f16_or_f64() {
    let inputs = LowerInputs {
        caps: TARGET,
        ..LowerInputs::DEFAULT
    };
    for pass in HostPass::ORDER {
        for opcode in runnable(pass) {
            let mut function = single(opcode).function;
            let applied = lower_for_host(&mut function, TARGET).expect("lowers for the host");
            assert!(applied.ran(pass), "{opcode}");
            let result = lower_with(GuestStage::Compute, &function, &inputs);
            if let Some(renamed) = refused_by_the_lowering(opcode) {
                assert_eq!(
                    result.err(),
                    Some(LowerError::Operation {
                        at: first(&function, renamed),
                        opcode: renamed
                    }),
                    "{opcode}"
                );
                continue;
            }
            let shader = result
                .map_err(|error| format!("{opcode}: {error}\n{function}"))
                .expect("lowers");
            assert!(
                !shader.requirements.f16 && !shader.requirements.f64,
                "{opcode}"
            );
            let caps = capabilities(shader.requirements);
            assert!(
                !caps.intersects(
                    Capabilities::SHADER_FLOAT16
                        | Capabilities::SHADER_FLOAT16_IN_FLOAT32
                        | Capabilities::FLOAT64
                ),
                "{opcode}"
            );
            let HostCode::Naga { module, .. } = &shader.code;
            Validator::new(validation_flags(), caps)
                .validate(module)
                .map_err(|error| format!("{opcode}: {error:?}"))
                .expect("validates");
        }
    }
}

/// Without the passes, the same programs do not lower for that target: a 16-bit one needs
/// 16-bit floats, and a 64-bit one's operations are refused by name, so the passes are what
/// makes them lower.
#[test]
fn without_the_passes_the_programs_do_not_lower() {
    let inputs = LowerInputs {
        caps: TARGET,
        ..LowerInputs::DEFAULT
    };
    let half = single(crate::ir::Opcode::FPAdd16).function;
    assert_eq!(
        lower_with(GuestStage::Compute, &half, &inputs).err(),
        Some(LowerError::Missing(Feature::Float16))
    );
    let double = single(crate::ir::Opcode::FPAdd64).function;
    assert_eq!(
        lower_with(GuestStage::Compute, &double, &inputs).err(),
        Some(LowerError::Operation {
            at: first(&double, crate::ir::Opcode::PackDouble2x32),
            opcode: crate::ir::Opcode::PackDouble2x32
        })
    );
}

//! Compute entry points, the inputs' bounds, the host target's features and the order of the
//! checks.

use naga::{Barrier, Statement};

use super::*;
use crate::host::Feature;

/// A compute program that reads both ids and synchronises its workgroup three ways.
const COMPUTE: &str = "block0:
    %0: u32x3 = WorkgroupId
    %1: u32x3 = LocalInvocationId
    %2: u32 = CompositeExtractU32x3 %0, u32 2
    %3: u32 = CompositeExtractU32x3 %1, u32 0
    %4: u32 = IAdd32 %2, %3
    %5 = SetRegister r0, %4
    %6 = Barrier
    %7 = WorkgroupMemoryBarrier
    %8 = DeviceMemoryBarrier
    return
";

fn inputs(workgroup_size: [u32; 3]) -> LowerInputs {
    LowerInputs {
        workgroup_size,
        ..LowerInputs::DEFAULT
    }
}

/// The compute entry point: the stage, the workgroup size in the module and the shader, the two
/// ids as built-in arguments (not interface slots), the barriers in order, no result; and wgpu
/// builds a compute pipeline from it (locally on a Mac).
#[test]
fn a_compute_entry_point() {
    let size = [8, 4, 2];
    let shader = lowered(GuestStage::Compute, &ir(COMPUTE), &inputs(size));
    assert_eq!(shader.stage, HostStage::Compute);
    assert_eq!(shader.workgroup_size, Some(size));
    assert_eq!((shader.inputs.len(), shader.outputs.len()), (0, 0));
    assert_eq!(shader.requirements, Requirements::default());
    let entry = &naga_module(&shader).entry_points[0];
    assert_eq!(entry.stage, ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, size);
    assert!(entry.function.result.is_none());
    let bindings: Vec<Option<Binding>> = entry
        .function
        .arguments
        .iter()
        .map(|arg| arg.binding.clone())
        .collect();
    assert_eq!(
        bindings,
        [
            Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
            Some(Binding::BuiltIn(BuiltIn::LocalInvocationId)),
        ]
    );
    let barriers: Vec<&Statement> = entry
        .function
        .body
        .iter()
        .filter(|s| {
            matches!(
                s,
                Statement::ControlBarrier(_) | Statement::MemoryBarrier(_)
            )
        })
        .collect();
    assert!(matches!(
        barriers.as_slice(),
        [
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Statement::MemoryBarrier(Barrier::WORK_GROUP),
            Statement::MemoryBarrier(Barrier::STORAGE),
        ]
    ));
    insta::assert_snapshot!("compute_entry_point", msl(&shader));
    let Some(gpu) = TestDevice::open("a_compute_entry_point", wgpu::Features::empty()) else {
        return;
    };
    let module = gpu
        .module(naga_module(&shader).clone())
        .expect("wgpu takes the module");
    assert_eq!(gpu.compute_pipeline(&module, &shader.entry), Ok(()));
}

/// The ids and barriers only compute programs have are refused in the other stages, with the
/// opcode and stage named.
#[test]
fn compute_operations_are_refused_in_other_stages() {
    let function = ir(COMPUTE);
    for stage in [GuestStage::Vertex, GuestStage::Pixel] {
        assert_eq!(
            lower(stage, &function).err(),
            Some(LowerError::NotInStage {
                at: value_at(&function, 0),
                opcode: Opcode::WorkgroupId,
                stage,
            })
        );
    }
    for (opcode, name) in [
        (Opcode::LocalInvocationId, "LocalInvocationId"),
        (Opcode::Barrier, "Barrier"),
        (Opcode::WorkgroupMemoryBarrier, "WorkgroupMemoryBarrier"),
        (Opcode::DeviceMemoryBarrier, "DeviceMemoryBarrier"),
    ] {
        let result = if opcode == Opcode::LocalInvocationId {
            "%0: u32x3 = "
        } else {
            "%0 = "
        };
        let function = ir(&format!("block0:\n    {result}{name}\n    return\n"));
        assert!(lower(GuestStage::Compute, &function).is_ok());
        let refused = lower(GuestStage::Pixel, &function).expect_err("not compute");
        assert_eq!(
            refused.to_string(),
            format!("%0: a pixel program cannot use {name}")
        );
    }
}

/// The workgroup size: every axis at least 1 and at most [`MAX_WORKGROUP_INVOCATIONS`]
/// invocations in all, each edge on both sides, a product that overflows `u32`, and the size
/// ignored outside compute.
#[test]
fn the_workgroup_size_at_its_edges() {
    let function = ir("block0:\n    return\n");
    let lower_size = |size| lower_with(GuestStage::Compute, &function, &inputs(size));
    let max = MAX_WORKGROUP_INVOCATIONS;
    for size in [[1, 1, 1], [max, 1, 1], [1, 1, max], [max / 2, 2, 1]] {
        assert_eq!(
            lower_size(size).map(|shader| shader.workgroup_size),
            Ok(Some(size)),
            "{size:?}"
        );
    }
    for size in [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [max + 1, 1, 1],
        [1, max + 1, 1],
        [max / 2 + 1, 2, 1],
        [0x1_0000, 0x1_0000, 0x1_0000],
        [u32::MAX, u32::MAX, 0],
    ] {
        assert_eq!(
            lower_size(size).err(),
            Some(LowerError::WorkgroupSize { size }),
            "{size:?}"
        );
    }
    assert_eq!(
        LowerError::WorkgroupSize { size: [0, 2, 3] }.to_string(),
        format!("a workgroup of 0 by 2 by 3 is not between 1 and {max} invocations")
    );
    let pixel = lower_with(GuestStage::Pixel, &function, &inputs([0, 0, 0]))
        .expect("the size is a compute program's");
    assert_eq!(pixel.workgroup_size, None);
}

/// A function that needs 16-bit floats, 64-bit integers or subgroups, as its requirements say.
fn needing(f16: bool, i64: bool, subgroup: bool) -> Function {
    let mut text = String::from("block0:\n");
    if f16 {
        text.push_str("    %0: f16 = ConvertF16F32 f32 1.0\n");
    }
    if i64 {
        text.push_str("    %1: u64 = ConvertU64U32 u32 1\n");
    }
    if subgroup {
        text.push_str("    %2: u32 = LaneId\n");
    }
    text.push_str("    return\n");
    ir(&text)
}

/// Every combination of requirements against every combination of the target's features:
/// lowered exactly when the target has each one needed, else refused with the first missing
/// one in [`Feature`] order; the requirements recorded are the ones the function needs.
#[test]
fn a_feature_the_target_lacks_is_refused() {
    let bit = |set: u8, index: u8| set & (1 << index) != 0;
    for needed in 0_u8..8 {
        let (f16, i64, subgroup) = (bit(needed, 0), bit(needed, 1), bit(needed, 2));
        let function = needing(f16, i64, subgroup);
        for offered in 0_u8..8 {
            let caps = HostCaps {
                f16: bit(offered, 0),
                i64: bit(offered, 1),
                subgroup: bit(offered, 2),
            };
            let model = if f16 && !caps.f16 {
                Some(Feature::Float16)
            } else if i64 && !caps.i64 {
                Some(Feature::Int64)
            } else if subgroup && !caps.subgroup {
                Some(Feature::Subgroup)
            } else {
                None
            };
            let result = lower_with(
                GuestStage::Compute,
                &function,
                &LowerInputs {
                    caps,
                    ..LowerInputs::DEFAULT
                },
            );
            if let Some(feature) = model {
                assert_eq!(result.err(), Some(LowerError::Missing(feature)), "{caps:?}");
            } else {
                let shader = result.expect("the target has every feature needed");
                assert_eq!(
                    shader.requirements,
                    Requirements {
                        subgroup,
                        f16,
                        i64,
                        ..Requirements::default()
                    },
                    "{caps:?}"
                );
            }
        }
    }
    let words = [
        (Feature::Float16, "16-bit floats"),
        (Feature::Int64, "64-bit integers"),
        (Feature::Subgroup, "subgroup operations"),
    ];
    for (feature, name) in words {
        assert_eq!(
            LowerError::Missing(feature).to_string(),
            format!("the shader needs {name}, which the host target lacks")
        );
    }
}

/// The documented order of the checks, two faults at a time: the inputs before the
/// control-flow form, the form before the instructions, the instructions before the
/// requirements.
#[test]
fn the_checks_run_in_their_order() {
    let none = LowerInputs {
        workgroup_size: [0, 1, 1],
        caps: HostCaps::NONE,
        ..LowerInputs::DEFAULT
    };
    let graph = ir("block0:\n    jump block1\nblock1:\n    return\n");
    // The workgroup size before the control-flow form.
    assert_eq!(
        lower_with(GuestStage::Compute, &graph, &none).err(),
        Some(LowerError::WorkgroupSize { size: [0, 1, 1] })
    );
    // The local-memory size before the control-flow form, and after the workgroup size.
    let big = LowerInputs {
        local_memory_bytes: MAX_LOCAL_MEMORY_BYTES + 1,
        shared_memory_bytes: MAX_SHARED_MEMORY_BYTES + 1,
        ..none
    };
    assert_eq!(
        lower_with(GuestStage::Compute, &graph, &big).err(),
        Some(LowerError::WorkgroupSize { size: [0, 1, 1] })
    );
    let one = LowerInputs {
        workgroup_size: [1, 1, 1],
        ..big
    };
    assert_eq!(
        lower_with(GuestStage::Compute, &graph, &one).err(),
        Some(LowerError::LocalMemory {
            bytes: MAX_LOCAL_MEMORY_BYTES + 1
        })
    );
    let local_fits = LowerInputs {
        local_memory_bytes: 0,
        ..one
    };
    assert_eq!(
        lower_with(GuestStage::Compute, &graph, &local_fits).err(),
        Some(LowerError::SharedMemory {
            bytes: MAX_SHARED_MEMORY_BYTES + 1
        })
    );
    // The control-flow form before an unlowered operation.
    let unlowered_graph =
        ir("block0:\n    %0: f64 = FPAbs64 f64 1.0\n    jump block1\nblock1:\n    return\n");
    let fits = LowerInputs {
        workgroup_size: [1, 1, 1],
        caps: HostCaps::NONE,
        ..LowerInputs::DEFAULT
    };
    assert_eq!(
        lower_with(GuestStage::Compute, &unlowered_graph, &fits).err(),
        Some(LowerError::ControlFlow)
    );
    // A control-flow graph of more than one block is refused even when its entry block returns
    // or is unreachable: only a graph of that one block lowers.
    for end in ["return", "unreachable"] {
        let graph = ir(&format!("block0:\n    {end}\nblock1:\n    return\n"));
        assert_eq!(
            lower_with(GuestStage::Compute, &graph, &fits).err(),
            Some(LowerError::ControlFlow),
            "{end}"
        );
        let alone = ir(&format!("block0:\n    {end}\n"));
        assert!(
            lower_with(GuestStage::Compute, &alone, &fits).is_ok(),
            "{end} alone"
        );
    }
    // An unlowered operation before a missing feature.
    let function = ir("block0:
    %0: u32 = LaneId
    %1: f64 = FPAbs64 f64 1.0
    return
");
    assert_eq!(
        lower_with(GuestStage::Compute, &function, &fits).err(),
        Some(LowerError::Operation {
            at: value_at(&function, 1),
            opcode: Opcode::FPAbs64
        })
    );
    // A stage fault (collected first) before an unlowered operation listed ahead of it.
    let function = ir("block0:
    %0: f64 = FPAbs64 f64 1.0
    %1 = Barrier
    return
");
    assert_eq!(
        lower_with(GuestStage::Pixel, &function, &fits).err(),
        Some(LowerError::NotInStage {
            at: value_at(&function, 1),
            opcode: Opcode::Barrier,
            stage: GuestStage::Pixel,
        })
    );
}

/// Every new error reads as plain words.
#[test]
fn compute_errors_read_as_words() {
    let function = ir("block0:\n    %0: u32 = IAdd32 u32 1, u32 2\n    return\n");
    let cases = [
        (
            LowerError::LocalMemory {
                bytes: MAX_LOCAL_MEMORY_BYTES + 1,
            },
            "524289 bytes of local memory are more than the 524288 the lowering allows".to_owned(),
        ),
        (
            LowerError::SharedMemory {
                bytes: MAX_SHARED_MEMORY_BYTES + 1,
            },
            "101377 bytes of shared memory are more than the 101376 the lowering allows".to_owned(),
        ),
        (
            LowerError::Phi(value_at(&function, 0)),
            "%0: phis are not lowered".to_owned(),
        ),
    ];
    for (error, text) in cases {
        assert_eq!(error.to_string(), text);
    }
}

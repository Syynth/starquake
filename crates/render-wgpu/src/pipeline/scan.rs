//! What the backend reads from a shader's naga module beside its `ShaderInterface`: whether the
//! entry point exists for the shader's stage, whether the module uses subgroup operations, and,
//! per bound global, its binding-array size and storage-texture access.
//!
//! The interface table says what kind of resource sits at each group and binding; the module is
//! where the shader crate wrote the rest (render-backend, "Shader emission": "group and binding
//! numbers already assigned in the module by the shader crate"). Two layout facts are only
//! there:
//!
//! - **A binding array's size.** wgpu needs an explicit layout entry with a `count` for a
//!   binding array: a layout derived from the shader fails with "mapping of `ResourceBinding` ...
//!   is missing" (`docs/specs/shader-pipeline.md`, the E1 table). The size is the module's
//!   `TypeInner::BindingArray` size; an array whose size is not a constant (a runtime-sized or
//!   override-sized array) is refused, since a layout needs a number.
//! - **A storage texture's access.** wgpu-core accepts a write-only shader global with a
//!   read-write layout entry, but a read-only global needs a read-only entry
//!   (`check_binding_use` in wgpu-core 30.0.2 `src/validation.rs`: "A write-only shader (but not
//!   a read-only shader) is compatible with a read-write binding"); a write-only entry needs no
//!   read-write format support, so the global's declared access picks the entry
//!   (`crate::pipeline::layout`).
//!
//! **Subgroups.** A module uses subgroups when a function or entry point holds a
//! `SubgroupBallot`, `SubgroupGather` or `SubgroupCollectiveOperation` statement, at any depth
//! of nested blocks, or when an entry point argument, an entry point result or a struct member
//! is bound to one of the built-ins `SubgroupSize`, `SubgroupInvocationId`, `NumSubgroups` or
//! `SubgroupId`. Those are the operations and built-ins whose results depend on how many
//! invocations a subgroup has, which the guest's 32-lane warps fix at 32 (shader-pipeline, "What
//! naga 30.0.1 can express": subgroup operations need "a subgroup size of 32"). A subgroup
//! barrier (`Barrier::SUB_GROUP`) does not count: its meaning does not depend on the size.
//! Helper functions are scanned whether an entry point calls them or not, which can only
//! over-count. The walk keeps its own stack of blocks, so a deeply nested module costs heap, not
//! call stack, and visits each block once.

use std::collections::BTreeMap;
use std::num::NonZeroU32;
use wgpu::naga;

/// How many resources a global binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArrayCount {
    /// One: not a binding array.
    Single,
    /// A binding array of this constant size.
    Sized(NonZeroU32),
    /// A binding array whose size is not a constant.
    Unsized,
}

/// What the module says about one bound global.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BindingFacts {
    /// Its binding-array size.
    pub(crate) count: ArrayCount,
    /// For a storage texture (or an array of them), whether the shader only stores to it.
    pub(crate) write_only: bool,
}

/// What the backend reads from a module (module docs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ModuleFacts {
    /// Whether the module uses subgroup operations or built-ins.
    pub(crate) subgroups: bool,
    /// The bound globals, by group and binding.
    pub(crate) bindings: BTreeMap<(u32, u32), BindingFacts>,
}

/// The naga stage a shader stage runs as, or `None` for a stage naga has no entry point for
/// (geometry and tessellation, shader-pipeline's gap table).
pub(crate) const fn naga_stage(stage: starquake_render::ShaderStage) -> Option<naga::ShaderStage> {
    use starquake_render::ShaderStage as S;
    match stage {
        S::Vertex => Some(naga::ShaderStage::Vertex),
        S::Fragment => Some(naga::ShaderStage::Fragment),
        S::Compute => Some(naga::ShaderStage::Compute),
        S::TessControl | S::TessEval | S::Geometry => None,
    }
}

/// The index of the entry point named `entry` for `stage`, if the module has one.
pub(crate) fn entry_point(
    module: &naga::Module,
    entry: &str,
    stage: naga::ShaderStage,
) -> Option<usize> {
    module
        .entry_points
        .iter()
        .position(|ep| ep.name == entry && ep.stage == stage)
}

/// Whether `builtin` is one whose value depends on the subgroup size (module docs).
const fn is_subgroup_builtin(builtin: naga::BuiltIn) -> bool {
    matches!(
        builtin,
        naga::BuiltIn::SubgroupSize
            | naga::BuiltIn::SubgroupInvocationId
            | naga::BuiltIn::NumSubgroups
            | naga::BuiltIn::SubgroupId
    )
}

fn binds_subgroup(binding: Option<&naga::Binding>) -> bool {
    matches!(binding, Some(naga::Binding::BuiltIn(b)) if is_subgroup_builtin(*b))
}

/// Whether any of `blocks`, or a block nested in one, holds a subgroup operation.
fn blocks_use_subgroups<'a>(blocks: impl IntoIterator<Item = &'a naga::Block>) -> bool {
    let mut stack: Vec<&naga::Block> = blocks.into_iter().collect();
    while let Some(block) = stack.pop() {
        for statement in block {
            match statement {
                naga::Statement::SubgroupBallot { .. }
                | naga::Statement::SubgroupGather { .. }
                | naga::Statement::SubgroupCollectiveOperation { .. } => return true,
                naga::Statement::Block(inner) => stack.push(inner),
                naga::Statement::If { accept, reject, .. } => {
                    stack.push(accept);
                    stack.push(reject);
                }
                naga::Statement::Switch { cases, .. } => {
                    stack.extend(cases.iter().map(|case| &case.body));
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    stack.push(body);
                    stack.push(continuing);
                }
                _ => {}
            }
        }
    }
    false
}

/// Whether the module uses subgroups (module docs).
fn uses_subgroups(module: &naga::Module) -> bool {
    let entry_bindings = module.entry_points.iter().any(|ep| {
        ep.function
            .arguments
            .iter()
            .any(|a| binds_subgroup(a.binding.as_ref()))
            || ep
                .function
                .result
                .as_ref()
                .is_some_and(|r| binds_subgroup(r.binding.as_ref()))
    });
    let member_bindings = module.types.iter().any(|(_, ty)| match &ty.inner {
        naga::TypeInner::Struct { members, .. } => {
            members.iter().any(|m| binds_subgroup(m.binding.as_ref()))
        }
        _ => false,
    });
    let bodies = module
        .functions
        .iter()
        .map(|(_, f)| &f.body)
        .chain(module.entry_points.iter().map(|ep| &ep.function.body));
    entry_bindings || member_bindings || blocks_use_subgroups(bodies)
}

/// The facts for one global's type.
fn binding_facts(module: &naga::Module, ty: naga::Handle<naga::Type>) -> BindingFacts {
    let inner = |h| module.types.get_handle(h).ok().map(|t| &t.inner);
    let (count, element) = match inner(ty) {
        Some(naga::TypeInner::BindingArray { base, size }) => (
            match size {
                naga::ArraySize::Constant(n) => ArrayCount::Sized(*n),
                naga::ArraySize::Pending(_) | naga::ArraySize::Dynamic => ArrayCount::Unsized,
            },
            inner(*base),
        ),
        other => (ArrayCount::Single, other),
    };
    let write_only = matches!(
        element,
        Some(naga::TypeInner::Image {
            class: naga::ImageClass::Storage { access, .. },
            ..
        }) if *access == naga::StorageAccess::STORE
    );
    BindingFacts { count, write_only }
}

/// Reads the facts from `module` (module docs).
pub(crate) fn scan(module: &naga::Module) -> ModuleFacts {
    let bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            let binding = global.binding.as_ref()?;
            Some((
                (binding.group, binding.binding),
                binding_facts(module, global.ty),
            ))
        })
        .collect();
    ModuleFacts {
        subgroups: uses_subgroups(module),
        bindings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starquake_render::ShaderStage;

    fn parse(source: &str) -> naga::Module {
        naga::front::wgsl::parse_str(source).expect("the test module parses")
    }

    #[test]
    fn plain_modules_use_no_subgroups() {
        let m = parse(
            "@group(0) @binding(0) var<uniform> u: vec4<f32>;
             fn helper(x: f32) -> f32 { if x > 0.0 { return x; } return -x; }
             @fragment fn main() -> @location(0) vec4<f32> {
                 var v = u;
                 loop { if v.x > 1.0 { break; } v.x += helper(v.y); }
                 switch i32(v.z) { case 1: { v.w = 0.0; } default: {} }
                 return v;
             }",
        );
        assert_eq!(
            scan(&m),
            ModuleFacts {
                subgroups: false,
                bindings: BTreeMap::from([(
                    (0, 0),
                    BindingFacts {
                        count: ArrayCount::Single,
                        write_only: false
                    }
                )]),
            }
        );
    }

    /// One accepting case per way a module can use subgroups (module docs), each alone.
    #[test]
    fn every_subgroup_form_is_found() {
        let cases = [
            (
                "a ballot in an entry point",
                "
                 @compute @workgroup_size(32) fn main() { let b = subgroupBallot(true); }",
            ),
            (
                "a collective operation nested in a loop and an if",
                "
                 @compute @workgroup_size(32) fn main() {
                     var i = 0u;
                     loop { if i > 3u { break; } if i == 1u { let s = subgroupAdd(i); } i += 1u; }
                 }",
            ),
            (
                "a gather in a switch case",
                "
                 @compute @workgroup_size(32) fn main(@builtin(local_invocation_index) l: u32) {
                     switch l { case 0u: { let g = subgroupShuffle(l, 1u); } default: {} }
                 }",
            ),
            (
                "an operation in a helper function",
                "
                 fn helper(x: u32) -> u32 { return subgroupMax(x); }
                 @compute @workgroup_size(32) fn main() { }",
            ),
            (
                "the subgroup invocation id as an argument",
                "
                 @compute @workgroup_size(32)
                 fn main(@builtin(subgroup_invocation_id) lane: u32) { }",
            ),
            (
                "the subgroup size as an argument",
                "
                 @compute @workgroup_size(32) fn main(@builtin(subgroup_size) n: u32) { }",
            ),
            (
                "the number of subgroups as an argument",
                "
                 @compute @workgroup_size(32) fn main(@builtin(num_subgroups) n: u32) { }",
            ),
            (
                "the subgroup id as an argument",
                "
                 @compute @workgroup_size(32) fn main(@builtin(subgroup_id) n: u32) { }",
            ),
            (
                "a subgroup built-in as a struct member",
                "
                 struct In { @builtin(subgroup_invocation_id) lane: u32 }
                 @compute @workgroup_size(32) fn main(i: In) { }",
            ),
        ];
        for (what, source) in cases {
            assert!(scan(&parse(source)).subgroups, "{what}");
        }
    }

    #[test]
    fn a_subgroup_barrier_alone_is_not_a_size_dependence() {
        let m = parse(
            "
             @compute @workgroup_size(32) fn main() { subgroupBarrier(); }",
        );
        assert!(!scan(&m).subgroups);
    }

    /// A module with `statement` at the bottom of `depth` nested blocks, built by hand: the WGSL
    /// front end limits its own nesting.
    fn nested(
        depth: usize,
        statement: impl FnOnce(&mut naga::Function) -> naga::Statement,
    ) -> naga::Module {
        let mut m = parse("@compute @workgroup_size(32) fn main() { }");
        let function = &mut m.entry_points[0].function;
        let mut inner = naga::Block::new();
        inner.push(statement(function), naga::Span::UNDEFINED);
        for _ in 0..depth {
            let mut outer = naga::Block::new();
            outer.push(naga::Statement::Block(inner), naga::Span::UNDEFINED);
            inner = outer;
        }
        function.body = inner;
        m
    }

    #[test]
    fn deep_nesting_is_walked_without_recursion() {
        let depth = 100_000;
        let ballot = nested(depth, |f| naga::Statement::SubgroupBallot {
            result: f.expressions.append(
                naga::Expression::SubgroupBallotResult,
                naga::Span::UNDEFINED,
            ),
            predicate: None,
        });
        assert!(scan(&ballot).subgroups, "a ballot at depth {depth}");
        let barrier = nested(depth, |_| {
            naga::Statement::ControlBarrier(naga::Barrier::SUB_GROUP)
        });
        assert!(!scan(&barrier).subgroups, "a barrier at depth {depth}");
        // naga's own `Drop` of the nested blocks recurses; leak them rather than overflow the
        // test thread's stack while freeing them.
        std::mem::forget(ballot);
        std::mem::forget(barrier);
    }

    #[test]
    fn binding_arrays_and_storage_access_are_read() {
        let m = parse(
            "enable wgpu_binding_array;
             @group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 4>;
             @group(1) @binding(3) var w: texture_storage_2d<rgba8unorm, write>;
             @group(1) @binding(4) var r: texture_storage_2d<rgba8unorm, read>;
             @group(1) @binding(5) var rw: texture_storage_2d<rgba8unorm, read_write>;
             @group(2) @binding(0) var ws: binding_array<texture_storage_2d<r32float, write>, 2>;
             @compute @workgroup_size(1) fn main() {
                 let d = textureDimensions(textures[0]);
                 textureStore(w, vec2(0), vec4(0.0));
                 let a = textureLoad(r, vec2(0));
                 textureStore(rw, vec2(0), textureLoad(rw, vec2(0)));
                 textureStore(ws[1], vec2(0), vec4(0.0));
             }",
        );
        let f = scan(&m).bindings;
        let sized = |n| ArrayCount::Sized(NonZeroU32::new(n).expect("non-zero"));
        let expect = [
            ((0, 0), sized(4), false),
            ((1, 3), ArrayCount::Single, true),
            ((1, 4), ArrayCount::Single, false),
            ((1, 5), ArrayCount::Single, false),
            ((2, 0), sized(2), true),
        ];
        assert_eq!(f.len(), expect.len());
        for (slot, count, write_only) in expect {
            assert_eq!(f[&slot], BindingFacts { count, write_only }, "{slot:?}");
        }
    }

    #[test]
    fn a_binding_array_without_a_constant_size_is_unsized() {
        let m = parse(
            "enable wgpu_binding_array;
             @group(0) @binding(0) var<storage> bufs: binding_array<array<u32>>;
             @compute @workgroup_size(1) fn main() { let x = arrayLength(&bufs[0]); }",
        );
        assert_eq!(scan(&m).bindings[&(0, 0)].count, ArrayCount::Unsized);
    }

    #[test]
    fn entry_points_are_matched_by_name_and_stage() {
        let m = parse(
            "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4(0.0); }
             @fragment fn frag() -> @location(0) vec4<f32> { return vec4(0.0); }",
        );
        let vertex = naga::ShaderStage::Vertex;
        let fragment = naga::ShaderStage::Fragment;
        assert_eq!(entry_point(&m, "main", vertex), Some(0));
        assert_eq!(entry_point(&m, "frag", fragment), Some(1));
        assert_eq!(entry_point(&m, "main", fragment), None, "wrong stage");
        assert_eq!(entry_point(&m, "mai", vertex), None, "wrong name");
    }

    #[test]
    fn stages_map_to_naga_where_naga_has_them() {
        let cases = [
            (ShaderStage::Vertex, Some(naga::ShaderStage::Vertex)),
            (ShaderStage::Fragment, Some(naga::ShaderStage::Fragment)),
            (ShaderStage::Compute, Some(naga::ShaderStage::Compute)),
            (ShaderStage::TessControl, None),
            (ShaderStage::TessEval, None),
            (ShaderStage::Geometry, None),
        ];
        for (stage, naga) in cases {
            assert_eq!(naga_stage(stage), naga, "{stage:?}");
        }
    }
}

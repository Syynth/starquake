//! The fixed-function state the host lacks, done in the shader before each return (slice 17;
//! spec, "Pipeline-state specialisation": the render backend's `EmulationFlags`, mirrored in
//! [`crate::variant::EmulationKey`]; render-backend, "Fixed-function state the host lacks").
//!
//! Eden runs its emulation in `EmitEpilogue`
//! (`eden/src/shader_recompiler/backend/spirv/emit_spirv_special.cpp` @ cf654052), at each
//! `Epilogue` operation, which its structurizer puts before every return; starquake's
//! structurizer does the same ([`crate::structure`]). A program of one block is not structured
//! (the lowering takes it as that block and a return, [`super::lower_with`]'s "What it takes"),
//! so it has no `Epilogue`. The lowering therefore does the steps below, in this order, before
//! each return it writes for the tree's `return` nodes and before the entry point's closing
//! return, and lowers `Epilogue` itself as a no-op; a structured function's returns are exactly
//! where its `Epilogue`s are, so the two placements agree there. An `unreachable` node, which
//! the lowering also writes as a return, does no step: Eden writes `OpUnreachable` there and
//! runs no epilogue.
//!
//! 1. **The depth-range rewrite** (vertex programs, [`EmulationKey::depth_range_rewrite`]):
//!    the written position's z becomes `(z + w) * 0.5`, moving the guest's -1..1 depth range
//!    to the host's 0..1, as Eden's `ConvertDepthMode`. Eden does it where
//!    `runtime_info.convert_depth_mode && !profile.support_native_ndc`; here the render backend
//!    sets the flag exactly when the host needs it, so the flag alone decides.
//! 2. **The flipped viewport** (vertex programs, [`EmulationKey::flipped_viewport`]): the
//!    written position `(x, y, z, w)` becomes
//!    `(x, y * PositionYScale, z * PositionZScale + w * PositionZOffset, w)`, reading the three
//!    driver constants ([`crate::driver_constants`], "The flipped viewport"; owner's ruling 28
//!    of 2026-11-19), after the rewrite of step 1 where both flags are set. No reference does
//!    this: Eden hands a flipped viewport to Vulkan as it is.
//! 3. **The alpha test** (pixel programs, [`EmulationKey::alpha_test`]): when colour target 0
//!    is written, its alpha is compared with the driver constant `AlphaReference` by the key's
//!    function, and the invocation is discarded (naga's `Kill`) when the comparison fails, as
//!    Eden's `AlphaTest` (`OpKill`). `Always` adds nothing and `Never` always discards, reading
//!    no constant, as Eden's `ComparisonFunction` gives a constant condition for both. Eden's
//!    comparisons are the ordered ones (`OpFOrdLessThan` and the rest, `OpFOrdNotEqual` for
//!    not-equal), so a NaN alpha or reference fails every function but `Always`; the comparison
//!    goes through the NaN-safe helper of `ops` (spec, "The backend through naga": every float
//!    comparison does).
//!
//! The other mirrored flags change nothing here yet: [`EmulationKey::typed_outputs`] and
//! [`EmulationKey::attribute_fetch`] need interface types the host tables do not have yet, and
//! [`EmulationKey::primitive_restart_rewrite`] is not lowered (the rest of slice 17, #376).
//! A flag of another stage's (an alpha test on a vertex program, a flip on a pixel program) is
//! ignored, as Eden's `EmitEpilogue` tests the stage first.
//!
//! # Where this differs from Eden
//!
//! | What | Here | Eden @ cf654052 | Pinned by |
//! |---|---|---|---|
//! | The alpha reference | Read from the driver constants each draw (decision 16) | Compiled in from `RuntimeInfo::alpha_test_reference` | `tests::emulation::the_alpha_test_reads_the_reference` |
//! | The flipped viewport | Done in the vertex shader | Not done: Vulkan takes the viewport | `tests::emulation::the_flipped_viewport_reads_three_constants` |
//! | Where the steps run | Before each `return` the lowering writes for a `return` node or at the end of the entry point | At each `Epilogue`, which Eden's structurizer puts before each return | `tests::emulation::each_return_does_the_steps`, `tests::emulation::a_return_inside_a_loop_does_the_steps`, `tests::emulation::a_program_of_one_block_does_the_steps` |

use naga::{BinaryOperator, Block, Expression, Handle, Literal, Statement};

use super::builder::push;
use super::{LowerError, Lowering, OutputMember};
use crate::driver_constants::DriverConstant;
use crate::ir::Node;
use crate::variant::{CompareFunction, EmulationKey};

/// The component of a `vec4` that is y, z or w (alpha).
const Y: u32 = 1;
const Z: u32 = 2;
const W: u32 = 3;

/// What the epilogue does for a stage, decided before the module is built, so the driver
/// constants are declared exactly when a step reads them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Epilogue {
    /// Step 1.
    pub(super) depth_rewrite: bool,
    /// Step 2.
    pub(super) flip: bool,
    /// Step 3: the comparison, when it is neither `Always` nor absent.
    pub(super) alpha_test: Option<CompareFunction>,
}

impl Epilogue {
    /// The steps a function of `vertex` or `pixel` stage with `key` takes; `writes_target_0` is
    /// whether a pixel program writes colour target 0.
    pub(super) fn plan(
        vertex: bool,
        pixel: bool,
        writes_target_0: bool,
        key: EmulationKey,
    ) -> Self {
        Self {
            depth_rewrite: vertex && key.depth_range_rewrite,
            flip: vertex && key.flipped_viewport,
            alpha_test: key.alpha_test.filter(|&function| {
                pixel && writes_target_0 && function != CompareFunction::Always
            }),
        }
    }

    /// Whether a step reads the driver constants.
    pub(super) fn reads_driver_constants(self) -> bool {
        self.flip
            || self
                .alpha_test
                .is_some_and(|function| function != CompareFunction::Never)
    }
}

/// Whether the entry point lowered from `nodes` writes a return that does the steps: a
/// `return` node anywhere in the tree, or a closing return (the tree does not end in `return`
/// or `unreachable`). The recursion is as deep as the tree, which the verifier bounds
/// ([`crate::ir::MAX_NESTING`]).
pub(super) fn returns(nodes: &[Node]) -> bool {
    fn any_return(nodes: &[Node]) -> bool {
        nodes.iter().any(|node| match node {
            Node::Return => true,
            Node::If { body, .. } | Node::Loop { body, .. } => any_return(body),
            Node::Block(_) | Node::Break { .. } | Node::Unreachable => false,
        })
    }
    !matches!(nodes.last(), Some(Node::Return | Node::Unreachable)) || any_return(nodes)
}

/// The ordered comparison `function` makes, as `ops`' float comparisons take it: the operator
/// for the four relational functions and the two equalities; `None` for `Never` and `Always`.
const fn comparison(function: CompareFunction) -> Option<BinaryOperator> {
    Some(match function {
        CompareFunction::Less => BinaryOperator::Less,
        CompareFunction::Equal => BinaryOperator::Equal,
        CompareFunction::LessThanEqual => BinaryOperator::LessEqual,
        CompareFunction::Greater => BinaryOperator::Greater,
        CompareFunction::NotEqual => BinaryOperator::NotEqual,
        CompareFunction::GreaterThanEqual => BinaryOperator::GreaterEqual,
        CompareFunction::Never | CompareFunction::Always => return None,
    })
}

impl Lowering {
    /// Does the steps before a return (module docs).
    pub(super) fn epilogue(&mut self, body: &mut Block) -> Result<(), LowerError> {
        let plan = self.epilogue;
        if plan.depth_rewrite || plan.flip {
            let position = self.position;
            let z = self.output_component(body, position, Z)?;
            let w = self.output_component(body, position, W)?;
            if plan.depth_rewrite {
                let z_value = self.expr(body, Expression::Load { pointer: z });
                let w_value = self.expr(body, Expression::Load { pointer: w });
                let sum = self.binary(body, BinaryOperator::Add, z_value, w_value);
                let half = self.lit(body, Literal::F32(0.5));
                let depth = self.binary(body, BinaryOperator::Multiply, sum, half);
                push(
                    body,
                    Statement::Store {
                        pointer: z,
                        value: depth,
                    },
                );
            }
            if plan.flip {
                let y = self.output_component(body, position, Y)?;
                let y_value = self.expr(body, Expression::Load { pointer: y });
                let y_scale = self.driver_constant(body, DriverConstant::PositionYScale)?;
                let flipped = self.binary(body, BinaryOperator::Multiply, y_value, y_scale);
                push(
                    body,
                    Statement::Store {
                        pointer: y,
                        value: flipped,
                    },
                );
                let z_value = self.expr(body, Expression::Load { pointer: z });
                let w_value = self.expr(body, Expression::Load { pointer: w });
                let z_scale = self.driver_constant(body, DriverConstant::PositionZScale)?;
                let z_offset = self.driver_constant(body, DriverConstant::PositionZOffset)?;
                let scaled = self.binary(body, BinaryOperator::Multiply, z_value, z_scale);
                let offset = self.binary(body, BinaryOperator::Multiply, w_value, z_offset);
                let depth = self.binary(body, BinaryOperator::Add, scaled, offset);
                push(
                    body,
                    Statement::Store {
                        pointer: z,
                        value: depth,
                    },
                );
            }
        }
        if let Some(function) = plan.alpha_test {
            let passed = match comparison(function) {
                Some(op) => {
                    let target = self.target_members.get(&0).copied();
                    let alpha_pointer = self.output_component(body, target, W)?;
                    let alpha = self.expr(
                        body,
                        Expression::Load {
                            pointer: alpha_pointer,
                        },
                    );
                    let reference = self.driver_constant(body, DriverConstant::AlphaReference)?;
                    self.float_compare(body, op, false, alpha, reference)
                }
                None => self.lit(body, Literal::Bool(false)),
            };
            push(
                body,
                Statement::If {
                    condition: passed,
                    accept: Block::new(),
                    reject: Block::from_vec(vec![Statement::Kill]),
                },
            );
        }
        Ok(())
    }

    /// A pointer to component `component` of output member `member`.
    fn output_component(
        &mut self,
        body: &mut Block,
        member: Option<OutputMember>,
        component: u32,
    ) -> Result<Handle<Expression>, LowerError> {
        let (Some(output), Some(member)) = (self.output, member) else {
            return Err(LowerError::Internal(
                "the epilogue's output member is not declared".into(),
            ));
        };
        let base = self.expr(body, Expression::LocalVariable(output));
        let member = self.expr(
            body,
            Expression::AccessIndex {
                base,
                index: member.index,
            },
        );
        Ok(self.expr(
            body,
            Expression::AccessIndex {
                base: member,
                index: component,
            },
        ))
    }
}

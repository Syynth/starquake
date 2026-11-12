//! A small helper for building naga IR by hand.
//!
//! naga needs every expression that is not "pre-emitted" (literals, constants, arguments,
//! variables: `naga::Expression::needs_pre_emit`) to be covered by a `Statement::Emit` in the
//! block that uses it, before its first use. [`FunctionBuilder::expr`] appends the expression and
//! emits it in one step, so callers cannot forget the `Emit`. Statement results (atomics, subgroup
//! operations, calls) are appended with [`FunctionBuilder::result`], which never emits, because
//! the statement that produces them defines them.
//!
//! This module and its parent are the only code that builds naga IR (spec, "The backend through
//! naga"), so a naga upgrade touches one place.

use naga::{
    Block, Expression, Function, Handle, LocalVariable, Module, Span, Statement, Type, TypeInner,
};

/// Builds one naga function.
#[derive(Debug, Default)]
pub(crate) struct FunctionBuilder {
    /// The function being built; its `body` is filled by [`FunctionBuilder::finish`].
    pub(crate) function: Function,
}

impl FunctionBuilder {
    /// A builder for a function with the given name.
    pub(crate) fn new(name: &str) -> Self {
        Self {
            function: Function {
                name: Some(name.to_owned()),
                ..Function::default()
            },
        }
    }

    /// Appends `expression`, and an `Emit` for it to `block` when naga needs one.
    pub(crate) fn expr(&mut self, block: &mut Block, expression: Expression) -> Handle<Expression> {
        let pre_emitted = expression.needs_pre_emit();
        let handle = self
            .function
            .expressions
            .append(expression, Span::UNDEFINED);
        if !pre_emitted {
            block.push(
                Statement::Emit(naga::Range::new_from_bounds(handle, handle)),
                Span::UNDEFINED,
            );
        }
        handle
    }

    #[cfg(test)]
    /// Appends a statement's result expression, which the statement itself defines.
    pub(crate) fn result(&mut self, expression: Expression) -> Handle<Expression> {
        self.function
            .expressions
            .append(expression, Span::UNDEFINED)
    }

    /// Declares a zero-initialised local variable.
    pub(crate) fn local(&mut self, name: &str, ty: Handle<Type>) -> Handle<LocalVariable> {
        self.function.local_variables.append(
            LocalVariable {
                name: Some(name.to_owned()),
                ty,
                init: None,
            },
            Span::UNDEFINED,
        )
    }

    /// Sets the function's body and returns the function.
    pub(crate) fn finish(mut self, body: Block) -> Function {
        self.function.body = body;
        self.function
    }
}

/// Pushes `statement` to `block` with no source span.
pub(crate) fn push(block: &mut Block, statement: Statement) {
    block.push(statement, Span::UNDEFINED);
}

/// Interns an unnamed type in `module`.
pub(crate) fn ty(module: &mut Module, inner: TypeInner) -> Handle<Type> {
    module
        .types
        .insert(Type { name: None, inner }, Span::UNDEFINED)
}

/// Interns a named type (structs need names for some writers) in `module`.
pub(crate) fn named_ty(module: &mut Module, name: &str, inner: TypeInner) -> Handle<Type> {
    module.types.insert(
        Type {
            name: Some(name.to_owned()),
            inner,
        },
        Span::UNDEFINED,
    )
}

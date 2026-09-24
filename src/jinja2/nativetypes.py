import typing as t
from ast import literal_eval
from ast import parse
from itertools import chain
from itertools import islice
from types import GeneratorType

from markupsafe import escape
from markupsafe import Markup

from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
from .compiler import has_safe_repr
from .environment import Environment
from .environment import Template
from .runtime import Undefined


def native_concat(values: t.Iterable[t.Any], *, strict: bool = False) -> t.Any | None:
    """Return a native Python type from the list of compiled nodes. If
    the result is a single node, its value is returned. Otherwise, the
    nodes are concatenated as strings. If the result can be parsed with
    :func:`ast.literal_eval`, the parsed value is returned. Otherwise,
    the string is returned.

    A single node that is not a string is returned unchanged, preserving
    its type. A single :class:`~jinja2.runtime.Undefined` node is
    converted with ``str()`` so that the environment's undefined policy
    is applied rather than silently returning the object.

    When concatenating multiple nodes, if any node is
    :class:`markupsafe.Markup`, the joined string keeps the ``Markup``
    type if it is not parsed as a literal.

    Parsing uses :func:`ast.literal_eval`, which only creates simple
    Python types and never executes code. Nested numbers, booleans,
    ``None``, Unicode strings, and dicts with duplicate keys are handled
    safely.

    :param values: Iterable of outputs to concatenate.
    :param strict: If ``True``, raise a :exc:`ValueError` if the
        concatenated nodes can't be parsed as a Python literal, instead
        of falling back to returning a string.

    .. versionchanged:: 3.2
        A single ``Undefined`` node applies the undefined policy instead
        of being returned silently. ``Markup`` is preserved when
        concatenating. Added the ``strict`` parameter.
    """
    head = list(islice(values, 2))

    if not head:
        return None

    if len(head) == 1:
        raw = head[0]

        if not isinstance(raw, str):
            if isinstance(raw, Undefined):
                # Don't return an undefined object silently, apply the
                # environment's policy. ``Undefined`` becomes an empty
                # string, ``DebugUndefined`` becomes a message, and
                # ``StrictUndefined`` raises ``UndefinedError``.
                raw = str(raw)
            else:
                return raw
    else:
        if isinstance(values, GeneratorType):
            values = chain(head, values)

        strings = []
        has_markup = False

        for value in values:
            if isinstance(value, Markup):
                has_markup = True

            strings.append(str(value))

        raw = "".join(strings)

        if has_markup:
            # If any node was Markup, the concatenation should remain
            # Markup if it's not parsed as a literal below.
            raw = Markup(raw)

    try:
        return literal_eval(
            # In Python 3.10+ ast.literal_eval removes leading spaces/tabs
            # from the given string. For backwards compatibility we need to
            # parse the string ourselves without removing leading spaces/tabs.
            parse(raw, mode="eval")
        )
    except (ValueError, SyntaxError, MemoryError, RecursionError) as e:
        if strict:
            raise ValueError(
                f"Could not convert the rendered value to a Python literal: {raw!r}"
            ) from e

        return raw


class NativeCodeGenerator(CodeGenerator):
    """A code generator which renders Python types by not adding
    ``str()`` around output nodes.
    """

    @staticmethod
    def _default_finalize(value: t.Any) -> t.Any:
        return value

    def _output_const_repr(self, group: t.Iterable[t.Any]) -> str:
        return repr("".join([str(v) for v in group]))

    def _output_child_to_const(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> t.Any:
        const = node.as_const(frame.eval_ctx)

        if frame.eval_ctx.autoescape:
            const = escape(const)

        if not has_safe_repr(const):
            raise nodes.Impossible()

        if isinstance(node, nodes.TemplateData):
            return const

        return finalize.const(const)  # type: ignore

    def _output_child_pre(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> None:
        if frame.eval_ctx.volatile:
            self.write("(escape if context.eval_ctx.autoescape else identity)(")
        elif frame.eval_ctx.autoescape:
            self.write("escape(")

        if finalize.src is not None:
            self.write(finalize.src)

    def _output_child_post(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> None:
        if frame.eval_ctx.autoescape or frame.eval_ctx.volatile:
            self.write(")")

        if finalize.src is not None:
            self.write(")")


class NativeEnvironment(Environment):
    """An environment that renders templates to native Python types."""

    code_generator_class = NativeCodeGenerator
    concat = staticmethod(native_concat)  # type: ignore


class NativeTemplate(Template):
    environment_class = NativeEnvironment

    def render(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Render the template to produce a native Python type. If the
        result is a single node, its value is returned. Otherwise, the
        nodes are concatenated as strings. If the result can be parsed
        with :func:`ast.literal_eval`, the parsed value is returned.
        Otherwise, the string is returned.
        """
        ctx = self.new_context(dict(*args, **kwargs))

        try:
            return self.environment.concat(  # type: ignore
                self.root_render_func(ctx)
            )
        except Exception:
            return self.environment.handle_exception()

    async def render_async(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if not self.environment.is_async:
            raise RuntimeError(
                "The environment was not created with async mode enabled."
            )

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            return self.environment.concat(  # type: ignore
                [n async for n in self.root_render_func(ctx)]  # type: ignore
            )
        except Exception:
            return self.environment.handle_exception()


NativeEnvironment.template_class = NativeTemplate

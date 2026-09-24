import typing as t
from ast import literal_eval
from ast import parse
from functools import partial
from itertools import chain
from itertools import islice

from markupsafe import escape
from markupsafe import Markup

from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
from .compiler import has_safe_repr
from .environment import Environment
from .environment import Template


def native_concat(values: t.Iterable[t.Any], *, strict: bool = False) -> t.Any:
    """Return a native Python type from the list of compiled nodes,
    following a consistent set of rules:

    -   No nodes returns ``None``.
    -   A single node that is not a plain string is returned unchanged.
        This preserves lists, dicts, numbers, :class:`Undefined`, and
        arbitrary user objects.
    -   A single :class:`~markupsafe.Markup` node is returned unchanged.
        Markup is already-escaped text, never a Python literal.
    -   A single plain string is parsed as a Python literal if possible,
        otherwise it is returned unchanged.
    -   Multiple nodes are concatenated as strings. If any node is
        :class:`~markupsafe.Markup`, the joined string is returned as
        ``Markup`` without attempting to parse it. Otherwise, if the
        joined string can be parsed as a Python literal, the parsed
        value is returned, otherwise the joined string is returned.

    Parsing only uses :func:`ast.literal_eval`, never :func:`eval`, so
    nested numbers, booleans, ``None``, Unicode, and duplicate dict keys
    are handled without executing code. Concatenating multiple nodes
    calls ``str`` on each node, which means an :class:`Undefined` node
    still raises :class:`~jinja2.exceptions.UndefinedError` rather than
    being silently converted.

    If ``strict`` is enabled, a string that cannot be parsed as a Python
    literal raises a :exc:`ValueError` instead of falling back to
    returning the string.

    :param values: Iterable of outputs to concatenate.
    :param strict: Raise a :exc:`ValueError` instead of returning a
        string when the rendered output cannot be converted to a native
        Python type.

    .. versionchanged:: 3.2
        ``Markup`` nodes are preserved instead of being parsed as
        Python literals, and the ``strict`` parameter was added.
    """
    head = list(islice(values, 2))

    if not head:
        return None

    if len(head) == 1:
        raw = head[0]

        if isinstance(raw, Markup):
            return raw

        if not isinstance(raw, str):
            return raw
    else:
        if iter(values) is values:
            # ``islice`` consumed the head from an iterator, put it back.
            values = chain(head, values)

        parts = []
        has_markup = False

        for value in values:
            if isinstance(value, Markup):
                has_markup = True

            parts.append(str(value))

        raw = "".join(parts)

        if has_markup:
            return Markup(raw)

    try:
        return literal_eval(
            # In Python 3.10+ ast.literal_eval removes leading spaces/tabs
            # from the given string. For backwards compatibility we need to
            # parse the string ourselves without removing leading spaces/tabs.
            parse(raw, mode="eval")
        )
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
        if strict:
            raise ValueError(
                "Could not convert the rendered template to a native Python"
                f" type: {raw!r}"
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
        joined = "".join([str(v) for v in group])

        if any(isinstance(v, Markup) for v in group):
            # Escaped constant data stays ``Markup`` when yielded.
            return f"Markup({joined!r})"

        return repr(joined)

    def _output_child_to_const(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> t.Any:
        const = node.as_const(frame.eval_ctx)

        if not has_safe_repr(const):
            raise nodes.Impossible()

        if frame.eval_ctx.autoescape:
            const = escape(const)

        if isinstance(node, nodes.TemplateData):
            return const

        return finalize.const(const)  # type: ignore

    def _output_child_pre(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> None:
        if frame.eval_ctx.volatile:
            self.write("(escape if context.eval_ctx.autoescape else (lambda v: v))(")
        elif frame.eval_ctx.autoescape:
            self.write("escape(")

        if finalize.src is not None:
            self.write(finalize.src)

    def _output_child_post(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> None:
        if frame.eval_ctx.volatile or frame.eval_ctx.autoescape:
            self.write(")")

        if finalize.src is not None:
            self.write(")")


class NativeEnvironment(Environment):
    """An environment that renders templates to native Python types."""

    code_generator_class = NativeCodeGenerator
    concat = staticmethod(native_concat)  # type: ignore

    def __init__(
        self, *args: t.Any, native_strict: bool = False, **kwargs: t.Any
    ) -> None:
        """Create the environment. If ``native_strict`` is enabled,
        rendering raises a :exc:`ValueError` instead of falling back to
        a string when the output cannot be converted to a native Python
        type.

        .. versionadded:: 3.2
            The ``native_strict`` parameter.
        """
        super().__init__(*args, **kwargs)

        if native_strict:
            self.concat = partial(native_concat, strict=True)  # type: ignore


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
            return self.environment.concat(self.root_render_func(ctx))  # type: ignore
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
                [n async for n in self.root_render_func(ctx)]
            )
        except Exception:
            return self.environment.handle_exception()


NativeEnvironment.template_class = NativeTemplate

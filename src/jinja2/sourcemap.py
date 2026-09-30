"""Source mapping support for :meth:`Template.render_with_map`.

Templates compiled in source map mode yield :class:`_MapEvent` objects
instead of plain strings.  Each event carries the text it produced, the
position in the template source that produced it, and a snapshot of the
macro / include call stack.  The events are assembled into a
:class:`RenderMap` by :func:`build_render_map`.

Nothing in this module is used by the regular ``render`` / ``generate``
code paths, so normal rendering does not pay for any of this.
"""

import typing as t
from contextvars import ContextVar

from markupsafe import escape
from markupsafe import Markup

from .utils import pass_eval_context


class SourceSpan(t.NamedTuple):
    """A position range in a template's source code.

    Line numbers are 1-based and exact.  Columns are 1-based and cover
    the whole source line (or lines, for multi-line template data) the
    producing node is located on, since Jinja's parser only tracks line
    numbers internally.
    """

    #: The load name of the template, or ``None`` if it has none.
    template: str | None
    #: First line of the source range, 1-based.
    line: int
    #: First column of the source range, 1-based.
    col: int
    #: Last line of the source range, 1-based.
    end_line: int
    #: End column of the source range, 1-based and exclusive.
    end_col: int


class CallFrame(t.NamedTuple):
    """One frame of the macro / include call stack."""

    #: Either ``"macro"`` or ``"include"``.
    kind: str
    #: The macro name or the name of the included template.
    name: str | None
    #: The template in which the call or include was written.
    template: str | None
    #: Line of the call or include in that template, 1-based.
    line: int
    #: Column of the call or include in that template, 1-based.
    col: int


class MapEntry(t.NamedTuple):
    """Maps a range of output characters to the source that produced
    them.  Ranges are Python character offsets into the final rendered
    string, ``start`` inclusive and ``end`` exclusive.
    """

    start: int
    end: int
    #: The load name of the template that produced the text.
    template: str | None
    #: First line of the source range, 1-based.
    line: int
    #: First column of the source range, 1-based.
    col: int
    #: Last line of the source range, 1-based.
    end_line: int
    #: End column of the source range, 1-based and exclusive.
    end_col: int
    #: The macro / include call stack active when the text was
    #: produced, outermost call first.
    stack: tuple[CallFrame, ...]


class RenderMap(t.NamedTuple):
    """The result of :meth:`Template.render_with_map`.  Can be unpacked
    as ``text, spans``.
    """

    #: The rendered string, identical to what ``render`` returns.
    text: str
    #: Source map entries ordered by output character range.  The
    #: entries cover every character of :attr:`text` exactly once.
    spans: list[MapEntry]

    def span_at(self, index: int) -> MapEntry:
        """Return the map entry covering the output character at
        ``index``.  Raises :exc:`IndexError` if there is none.
        """
        for span in self.spans:
            if span.start <= index < span.end:
                return span

        raise IndexError(index)


class _MapEvent(t.NamedTuple):
    """A single piece of mapped output, yielded by generators compiled
    in source map mode.
    """

    text: str
    span: SourceSpan
    stack: tuple[CallFrame, ...]


#: The macro / include call stack of the currently executing mapped
#: render.  A context variable is used so the stack follows the
#: execution across included templates and macro closures without
#: touching the template context objects.
_stack_var: ContextVar[tuple[CallFrame, ...]] = ContextVar(
    "jinja2_source_map_stack", default=()
)


class MappedString(str):
    """A string that carries the source map events that produced it.

    Used for values that cross generator boundaries in mapped mode,
    such as macro return values, so that the inner source information
    survives until the value is written to the output.
    """

    __slots__ = ("events",)

    events: tuple[_MapEvent, ...]

    def __new__(cls, value: str, events: t.Iterable[_MapEvent] = ()) -> "MappedString":
        obj = str.__new__(cls, value)
        obj.events = tuple(events)
        return obj

    def as_markup(self) -> "MappedMarkup":
        """Return a markup-safe version with the same events."""
        return MappedMarkup(str(self), self.events)

    def escaped(self) -> "MappedMarkup":
        """Return an escaped version, escaping each event's text
        individually so the source mapping is preserved.
        """
        return MappedMarkup(
            str(escape(str(self))),
            tuple(
                event._replace(text=str(escape(event.text))) for event in self.events
            ),
        )


class MappedMarkup(MappedString, Markup):
    """A :class:`MappedString` that is also markup-safe."""

    __slots__ = ()


def _map_push(frame: CallFrame) -> t.Any:
    """Push a call frame onto the current map stack.  Returns a token
    for :func:`_map_pop`.  Called from generated template code.
    """
    return _stack_var.set(_stack_var.get() + (frame,))


def _map_pop(token: t.Any) -> None:
    """Pop a call frame pushed by :func:`_map_push`."""
    _stack_var.reset(token)


def _map_event(text: str, span: SourceSpan) -> _MapEvent:
    """Create a map event for ``text`` with the current call stack.
    Called from generated template code.
    """
    return _MapEvent(text, span, _stack_var.get())


def _map_wrap(value: t.Any, span: SourceSpan) -> tuple[_MapEvent, ...]:
    """Convert an evaluated output expression into map events.  If the
    value carries its own events (for example a macro return value),
    those are used so the output points at the actual producing
    location.  Called from generated template code.
    """
    if isinstance(value, MappedString):
        return value.events

    return (_MapEvent(value, span, _stack_var.get()),)


def _map_str(value: t.Any) -> t.Any:
    """Mapped replacement for ``str`` around output expressions."""
    if isinstance(value, MappedString):
        return value

    return str(value)


def _map_escape(value: t.Any) -> t.Any:
    """Mapped replacement for :func:`markupsafe.escape` around output
    expressions.  Escaping is character based, so escaping each event
    individually preserves the source mapping.
    """
    if isinstance(value, MappedString):
        if isinstance(value, Markup):
            return value

        return value.escaped()

    return escape(value)


def _map_flatten(buffer: t.Iterable[_MapEvent], markup: bool = False) -> MappedString:
    """Join the events collected in a buffer (macro bodies, set blocks,
    filter blocks) into a single value that keeps the source mapping.
    Called from generated template code.
    """
    events = tuple(buffer)
    text = "".join(event.text for event in events)

    if markup:
        return MappedMarkup(text, events)

    return MappedString(text, events)


def concat_events(
    events: t.Iterable[t.Any], concat: t.Callable[[t.Iterable[t.Any]], str]
) -> str:
    """Concatenate the output of a block or loop generator.  In mapped
    mode the generator yields events, which are flattened into a
    :class:`MappedString`; otherwise this behaves like ``concat``.
    """
    items = list(events)

    if items and isinstance(items[0], _MapEvent):
        return _map_flatten(items)

    return concat(items)


def _map_callee(obj: t.Any, pos: tuple[str | None, int, int]) -> t.Any:
    """Wrap macros so that a call frame is pushed while they execute.
    Anything else is returned unchanged.  Called from generated
    template code around the callee of output expressions.
    """
    from .runtime import Macro

    if isinstance(obj, Macro):
        return _MappedMacroProxy(obj, pos)

    return obj


class _MappedMacroProxy:
    """Pushes a ``"macro"`` call frame while the wrapped macro runs."""

    __slots__ = ("_macro", "_pos")

    def __init__(self, macro: t.Any, pos: tuple[str | None, int, int]) -> None:
        self._macro = macro
        self._pos = pos

    # Mirror the pass_eval_context marker of Macro.__call__ so that
    # Context.call injects the eval context like for a plain macro.
    @pass_eval_context
    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        frame = CallFrame(
            "macro", self._macro.name, self._pos[0], self._pos[1], self._pos[2]
        )
        token = _map_push(frame)

        try:
            return self._macro(*args, **kwargs)
        finally:
            _map_pop(token)


def build_render_map(events: t.Iterable[_MapEvent]) -> RenderMap:
    """Assemble the rendered string and the source map from the events
    a mapped template produces.

    Each event becomes one map entry, so repeated output (for example
    from loops) produces multiple entries even when the source
    position is the same.  Callers may merge adjacent entries with
    identical source and call stack; the ``stack`` field keeps
    entries from different call stacks distinguishable.
    """
    text_parts: list[str] = []
    spans: list[MapEntry] = []
    pos = 0

    for event in events:
        text_parts.append(event.text)
        size = len(event.text)

        if size == 0:
            continue

        end = pos + size
        span = event.span
        spans.append(
            MapEntry(
                pos,
                end,
                span.template,
                span.line,
                span.col,
                span.end_line,
                span.end_col,
                event.stack,
            )
        )
        pos = end

    return RenderMap("".join(text_parts), spans)

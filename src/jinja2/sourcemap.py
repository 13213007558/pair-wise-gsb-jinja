"""Runtime support for render output source maps."""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from markupsafe import escape as markup_escape
from markupsafe import Markup
from markupsafe import soft_str

LocationTuple = tuple[str | None, int, int, int, int, int, int]
FrameTuple = tuple[str, str, LocationTuple]
StackTuple = tuple[FrameTuple, ...]
SegmentTuple = tuple[int, int, str | None, int, int, int, int, int, int, StackTuple]


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """A zero-based range in one template source."""

    template: str | None
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class MapFrame:
    """A macro, include, block, or block.super invocation."""

    type: str
    name: str
    location: SourceLocation


@dataclass(frozen=True, slots=True)
class SourceMapSegment:
    """An output interval mapped to one template source range."""

    start: int
    end: int
    template: str | None
    source_start_line: int
    source_start_column: int
    source_end_line: int
    source_end_column: int
    source_start_offset: int
    source_end_offset: int
    call_stack: tuple[MapFrame, ...]


class MappedText(str):
    """A rendered string carrying relative source map segments."""

    segments: tuple[SegmentTuple, ...]

    @t.no_type_check
    def __new__(cls, value: object, segments: tuple[SegmentTuple, ...]) -> MappedText:
        rv = super().__new__(cls, value)
        rv.segments = segments
        return rv


class MappedMarkup(Markup, MappedText):
    @t.no_type_check
    def __new__(cls, value: object, segments: tuple[SegmentTuple, ...]) -> MappedMarkup:
        rv = str.__new__(cls, soft_str(value))
        rv.segments = segments
        return rv


def source_location(
    name: str | None, source: str, start: int | None, end: int | None
) -> LocationTuple:
    """Convert source offsets to a public source location tuple."""
    if start is None:
        start = 0

    if end is None:
        end = len(source)

    start = max(start, 0)
    end = max(end, start)

    def line_column(offset: int) -> tuple[int, int]:
        line_start = source.rfind("\n", 0, offset) + 1
        return source.count("\n", 0, offset), offset - line_start

    start_line, start_column = line_column(start)
    end_line, end_column = line_column(end)
    return (
        name,
        start_line,
        start_column,
        end_line,
        end_column,
        start,
        end,
    )


def map_frame(frame_type: str, name: str, location: LocationTuple) -> FrameTuple:
    return (frame_type, name, location)


def push_frame(
    segments: tuple[SegmentTuple, ...], frame: FrameTuple
) -> tuple[SegmentTuple, ...]:
    return tuple(segment[:9] + (segment[9] + (frame,),) for segment in segments)


def mapped_text(
    value: object,
    location: LocationTuple,
    stack: StackTuple = (),
    safe: bool = False,
) -> str:
    """Create a mapped fragment for literal or dynamic text."""
    value = soft_str(value)
    segment: SegmentTuple = (0, len(value), *location, stack)
    cls = MappedMarkup if safe else MappedText
    return t.cast(str, cls(value, (segment,)))


def mapped_concat(parts: t.Iterable[object]) -> str:
    """Concatenate rendered parts, preserving their mapped segments."""
    text_parts: list[str] = []
    segments: list[SegmentTuple] = []
    offset = 0
    safe = False

    for part in parts:
        value = soft_str(part)
        text_parts.append(value)

        if isinstance(part, MappedText):
            safe = safe or isinstance(part, Markup)

            for segment in part.segments:
                segments.append(
                    (
                        offset + segment[0],
                        offset + segment[1],
                        *segment[2:],
                    )
                )

        offset += len(value)

    text = "".join(text_parts)

    if not segments:
        return Markup(text) if safe else text

    cls = MappedMarkup if safe else MappedText
    return t.cast(str, cls(text, tuple(segments)))


mapped_markup_join = mapped_concat
mapped_str_join = mapped_concat


def mapped_output(
    value: object,
    location: LocationTuple,
    stack: StackTuple = (),
    safe: bool = False,
) -> str:
    """Wrap one expression result after finalization."""
    value = soft_str(value)

    if isinstance(value, MappedText):
        return t.cast(str, value)

    return mapped_text(value, location, stack, safe)


def mapped_escape(
    value: object, location: LocationTuple, stack: StackTuple = ()
) -> str:
    """Escape one expression and map every escaped character."""
    mapped_value = value if isinstance(value, MappedText) else None
    value = soft_str(value)

    if isinstance(value, Markup):
        if isinstance(value, MappedText):
            return t.cast(str, value)

        markup_segment: SegmentTuple = (0, len(value), *location, stack)
        return t.cast(str, MappedMarkup(value, (markup_segment,)))

    text = soft_str(value)
    escaped = str(markup_escape(text))

    if mapped_value is not None:
        segments: list[SegmentTuple] = []
        output_offset = 0

        for (
            start,
            end,
            template_name,
            start_line,
            start_column,
            end_line,
            end_column,
            _source_start,
            _source_end,
            stack,
        ) in mapped_value.segments:
            for source_index in range(start, end):
                char = text[source_index]
                replacement = str(markup_escape(char))
                new_segment: SegmentTuple = (
                    output_offset,
                    output_offset + len(replacement),
                    template_name,
                    start_line,
                    start_column,
                    end_line,
                    end_column,
                    source_index,
                    source_index + 1,
                    stack,
                )
                segments.append(new_segment)
                output_offset += len(replacement)

        return t.cast(str, MappedMarkup(escaped, tuple(segments)))

    fallback_segment: SegmentTuple = (0, len(escaped), *location, stack)
    return t.cast(str, MappedMarkup(escaped, (fallback_segment,)))


def mapped_include_events(
    events: t.Iterable[str],
    template_name: str | None,
    location: LocationTuple,
) -> t.Iterator[str]:
    frame = map_frame("include", template_name or "", location)

    for event in events:
        if isinstance(event, MappedText):
            yield type(event)(soft_str(event), push_frame(event.segments, frame))
        elif event:
            yield mapped_text(event, location, (frame,))


def mapped_block_events(
    events: t.Iterable[str], name: str, location: LocationTuple
) -> t.Iterator[str]:
    frame = map_frame("block", name, location)

    for event in events:
        if isinstance(event, MappedText):
            yield type(event)(soft_str(event), push_frame(event.segments, frame))
        elif event:
            yield mapped_text(event, location, (frame,))


def mapped_macro_result(
    result: object,
    macro_name: str,
    definition: LocationTuple,
    invocation: LocationTuple | None,
) -> str:
    """Add macro definition and invocation frames to a macro result."""
    frame = map_frame("macro", macro_name, invocation or definition)
    result = result if isinstance(result, MappedText) else soft_str(result)

    if isinstance(result, MappedText):
        definition_frame = map_frame("macro-definition", macro_name, definition)
        segments: tuple[SegmentTuple, ...] = tuple(
            segment[:9] + ((definition_frame, *segment[9], frame),)
            for segment in result.segments
        )
        return t.cast(str, type(result)(result, segments))

    return result


def mapped_call(
    context: t.Any,
    obj: t.Any,
    source_call: LocationTuple,
    *args: t.Any,
    **kwargs: t.Any,
) -> t.Any:
    """Call a macro or callable while temporarily passing its call site."""
    context.eval_ctx._jinja_source_call = source_call
    try:
        if context.environment.sandboxed:
            return context.environment.call(context, obj, *args, **kwargs)

        return context.call(obj, *args, **kwargs)
    finally:
        delattr(context.eval_ctx, "_jinja_source_call")


def public_segments(value: object) -> list[SourceMapSegment]:
    """Convert internal tuple segments into public dataclasses."""
    segments = getattr(value, "segments", ())
    result: list[SourceMapSegment] = []

    for start, end, *location_and_stack in segments:
        location = tuple(location_and_stack[:7])
        stack = tuple(location_and_stack[7])
        result.append(
            SourceMapSegment(
                start,
                end,
                location[0],
                location[1],
                location[2],
                location[3],
                location[4],
                location[5],
                location[6],
                tuple(
                    MapFrame(
                        frame[0],
                        frame[1],
                        SourceLocation(*frame[2]),
                    )
                    for frame in stack
                ),
            )
        )

    return result

from __future__ import annotations

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.sourcemap import MapFrame
from jinja2.sourcemap import SourceMapSegment

TEMPLATES = {
    "base.html": (
        "Subject: {{ title }}\n" "{% block body %}Base {{ title }}\n{% endblock %}"
    ),
    "macros.html": "{% macro item(value) %}<{{ value }}>{% endmacro %}",
    "mail.html": (
        '{% extends "base.html" %}'
        '{% import "macros.html" as m %}'
        "{% block body %}你好 {{ name }}\n"
        "{% for value in values %}{{ m.item(value) }}\n{% endfor %}"
        '{{ super() }}{% include "part.html" %}{% endblock %}'
    ),
    "part.html": "Part {{ tag }}\n",
}


def make_environment(*, finalize=None) -> Environment:
    return Environment(
        loader=DictLoader(TEMPLATES),
        autoescape=True,
        finalize=finalize,
    )


def assert_complete(output: str, segments: list[SourceMapSegment]) -> None:
    covered = [False] * len(output)

    for segment in segments:
        assert segment.start < segment.end
        for index in range(segment.start, segment.end):
            assert not covered[index]
            covered[index] = True

    assert all(covered)


def frames(segment: SourceMapSegment, frame_type: str) -> list[MapFrame]:
    return [frame for frame in segment.call_stack if frame.type == frame_type]


def test_render_with_map_covers_output_and_matches_render() -> None:
    env = make_environment()
    template = env.get_template("mail.html")
    context = {
        "title": "<标题>",
        "name": "<名>",
        "values": ["a&b", "中"],
        "tag": "<p>",
    }

    rendered = template.render(context)
    mapped, segments = template.render_with_map(context)

    assert mapped == rendered
    assert_complete(mapped, segments)


def test_in_memory_template_can_render_with_map() -> None:
    template = Environment(autoescape=True).from_string("你好 {{ value }}\n")
    rendered = template.render(value="<中>")
    mapped, segments = template.render_with_map(value="<中>")

    assert mapped == rendered
    assert_complete(mapped, segments)
    assert mapped == "你好 &lt;中&gt;"


@pytest.mark.parametrize(
    ("source", "context"),
    [
        (
            "a\r\nb {{ x }}\r\n{% if true %}中\n{% endif %}",
            {"x": "<"},
        ),
        (
            "{%- if true -%} x {%- endif -%}\n{{ a ~ b }}",
            {"a": "a&", "b": "b<"},
        ),
    ],
)
def test_newline_whitespace_and_concat_maps(
    source: str, context: dict[str, str]
) -> None:
    env = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(source)
    rendered = template.render(context)
    mapped, segments = template.render_with_map(context)

    assert mapped == rendered
    assert_complete(mapped, segments)


def test_loop_macro_include_and_super_point_to_real_templates() -> None:
    env = make_environment()
    template = env.get_template("mail.html")
    context = {
        "title": "<标题>",
        "name": "<名>",
        "values": ["a&b", "中"],
        "tag": "<p>",
    }

    mapped, segments = template.render_with_map(context)

    macro_ranges = [
        (segment.start, segment.end)
        for segment in segments
        if segment.template == "macros.html"
    ]
    assert len(macro_ranges) == 6
    assert macro_ranges[0] != macro_ranges[3]

    for segment in segments:
        if segment.template == "macros.html":
            stack = [frame.type for frame in segment.call_stack]
            assert "macro-definition" in stack
            assert "macro" in stack
            assert frames(segment, "macro")[0].location.template == "mail.html"

        if segment.template == "part.html":
            include_frame = frames(segment, "include")[0]
            assert include_frame.location.template == "mail.html"
            assert include_frame.name == "part.html"

    super_segments = [segment for segment in segments if frames(segment, "block.super")]
    assert super_segments
    assert all(segment.template == "base.html" for segment in super_segments)
    assert all(
        frames(segment, "block.super")[0].location.template == "mail.html"
        for segment in super_segments
    )

    assert "你好" in mapped
    assert "&amp;" in mapped


def test_finalize_result_is_attributed_to_expression() -> None:
    env = make_environment(finalize=lambda value: str(value).upper())
    template = env.get_template("mail.html")
    context = {
        "title": "abc",
        "name": "def",
        "values": [],
        "tag": "ghi",
    }

    rendered = template.render(context)
    mapped, segments = template.render_with_map(context)

    assert mapped == rendered
    assert "ABC" in mapped
    assert_complete(mapped, segments)

    title_segments = [
        segment
        for segment in segments
        if segment.template == "base.html"
        and mapped[segment.start : segment.end] == "ABC"
    ]
    assert title_segments


@pytest.mark.parametrize("template_name", ["base.html", "part.html", "macros.html"])
def test_source_positions_are_column_accurate(template_name: str) -> None:
    env = make_environment()
    template = env.get_template(template_name)
    source = TEMPLATES[template_name]

    if template_name == "macros.html":
        rendered, segments = template.render_with_map(value="中")
    else:
        rendered, segments = template.render_with_map(title="中", tag="中")

    assert rendered == template.render(title="中", tag="中", value="中")
    assert_complete(rendered, segments)

    for segment in segments:
        source_text = source[segment.source_start_offset : segment.source_end_offset]
        source_line = source.splitlines()[segment.source_start_line]
        assert segment.source_start_column == (
            segment.source_start_offset
            - source.rfind("\n", 0, segment.source_start_offset)
            - 1
        )
        if segment.source_start_line == segment.source_end_line:
            assert (
                source_line[segment.source_start_column : segment.source_end_column]
                == source_text
            )
        else:
            assert source_text.startswith("\n")

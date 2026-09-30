import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.sourcemap import CallFrame
from jinja2.sourcemap import RenderMap

MACROS = (
    "{% macro greeting(name) %}<b>{{ name }}</b>，你好！{% endmacro %}\n"
    "{% macro signature() %}此致 -- {{ 'ACME & Co' }}{% endmacro %}\n"
    "{% macro full(name) %}{{ greeting(name) }}|{{ signature() }}{% endmacro %}\n"
    "页脚由 macros.html 提供"
)

LAYOUT = (
    '{% import "macros.html" as m %}<html>\n'
    "<body>\n"
    "{% block content %}默认内容{% endblock %}\n"
    "{{ m.signature() }}\n"
    '{% include "macros.html" %}\n'
    "</body>\n"
    "</html>"
)

MAIL = (
    '{% extends "layout.html" %}\n'
    '{% import "macros.html" as m %}\n'
    "{% block content %}尊敬的 {{ name }}：\n"
    "{{ m.greeting(name) }}\n"
    "{% set full = m.full('李<四>') %}{{ full }}\n"
    "{% for item in items %}[{{ item }}]{% endfor %}\n"
    "{{ name|upper }}\n"
    "{{ super() }}{% endblock %}"
)

CTX = {"name": 'Zhang <san> & "友"', "items": ["甲", "<乙>", "丙"]}


@pytest.fixture
def mail_env():
    return Environment(
        loader=DictLoader(
            {"macros.html": MACROS, "layout.html": LAYOUT, "mail.html": MAIL}
        ),
        autoescape=True,
    )


@pytest.fixture
def mail_render(mail_env):
    template = mail_env.get_template("mail.html")
    return template, template.render_with_map(**CTX)


def assert_full_coverage(render_map):
    """Every output character is covered by exactly one span."""
    pos = 0

    for span in render_map.spans:
        assert span.start == pos
        assert span.end > span.start
        pos = span.end

    assert pos == len(render_map.text)
    assert (
        "".join(render_map.text[s.start : s.end] for s in render_map.spans)
        == render_map.text
    )


def spans_with_text(render_map, text):
    return [s for s in render_map.spans if render_map.text[s.start : s.end] == text]


def spans_containing(render_map, text):
    return [s for s in render_map.spans if text in render_map.text[s.start : s.end]]


class TestRenderWithMap:
    def test_text_matches_render(self, mail_render):
        template, rm = mail_render
        assert rm.text == template.render(**CTX)

    def test_full_coverage(self, mail_render):
        _, rm = mail_render
        assert_full_coverage(rm)

    def test_unpacking_and_type(self, mail_render):
        _, rm = mail_render
        assert isinstance(rm, RenderMap)
        text, spans = rm
        assert text == rm.text
        assert spans == rm.spans

    def test_span_at(self, mail_render):
        _, rm = mail_render

        for span in rm.spans:
            assert rm.span_at(span.start) is span

        with pytest.raises(IndexError):
            rm.span_at(len(rm.text) + 1)

    def test_template_data_points_at_owning_template(self, mail_render):
        _, rm = mail_render
        (span,) = spans_with_text(rm, "尊敬的 ")
        assert span.template == "mail.html"
        assert span.line == 3
        assert span.stack == ()

    def test_escaped_output_indices(self, mail_render):
        _, rm = mail_render
        # The escaped name is longer than the input, spans must use
        # the indices of the final, escaped output.
        escaped = "Zhang &lt;san&gt; &amp; &#34;友&#34;"
        found = spans_with_text(rm, escaped)
        assert len(found) >= 2

        for span in found:
            assert span.end - span.start == len(escaped)

    def test_chinese_char_indices(self, mail_render):
        _, rm = mail_render
        (span,) = spans_with_text(rm, "甲")
        # Chinese characters count as single Python characters.
        assert span.end - span.start == 1
        assert span.template == "mail.html"

    def test_imported_macro_points_at_macro_definition(self, mail_render):
        _, rm = mail_render
        # "<b>" is produced once by the direct greeting() call and
        # once through full(); pick the direct call.
        (span,) = (s for s in spans_with_text(rm, "<b>") if len(s.stack) == 1)
        assert span.template == "macros.html"
        assert span.line == 1
        assert span.stack == (CallFrame("macro", "greeting", "mail.html", 4, 1),)

    def test_nested_macro_call_stack(self, mail_render):
        _, rm = mail_render
        (span,) = spans_with_text(rm, "李&lt;四&gt;")
        assert span.template == "macros.html"
        assert span.line == 1
        assert span.stack == (
            CallFrame("macro", "full", "mail.html", 5, 1),
            CallFrame("macro", "greeting", "macros.html", 3, 1),
        )

    def test_same_source_different_stacks_not_confused(self, mail_render):
        _, rm = mail_render
        # "此致 -- " is produced by the signature macro twice: called
        # directly from the layout and indirectly through full().
        found = spans_with_text(rm, "此致 -- ")
        assert len(found) == 2
        stacks = {tuple(s.stack) for s in found}
        assert stacks == {
            (CallFrame("macro", "signature", "layout.html", 4, 1),),
            (
                CallFrame("macro", "full", "mail.html", 5, 1),
                CallFrame("macro", "signature", "macros.html", 3, 1),
            ),
        }

        for span in found:
            assert span.template == "macros.html"
            assert span.line == 2

    def test_block_super_points_at_parent_template(self, mail_render):
        _, rm = mail_render
        (span,) = spans_with_text(rm, "默认内容")
        assert span.template == "layout.html"
        assert span.line == 3
        assert span.stack == ()

    def test_include_carries_include_frame(self, mail_render):
        _, rm = mail_render
        found = spans_containing(rm, "页脚由 macros.html 提供")
        assert len(found) == 1
        span = found[0]
        assert span.template == "macros.html"
        assert span.stack == (CallFrame("include", "macros.html", "layout.html", 5, 1),)

    def test_loop_produces_multiple_spans(self, mail_render):
        _, rm = mail_render
        items = [s for s in rm.spans if s.template == "mail.html" and s.line == 6]
        texts = [rm.text[s.start : s.end] for s in items]
        # Three iterations, each producing "[", item, "]", plus the
        # trailing newline of the line.
        assert texts == ["[", "甲", "]", "[", "&lt;乙&gt;", "]", "[", "丙", "]", "\n"]

    def test_filter_result_attributed_to_expression(self, mail_render):
        _, rm = mail_render
        # upper() returns a new string, so the result is attributed to
        # the expression in the calling template, not to the filter.
        (span,) = spans_with_text(rm, "ZHANG &lt;SAN&gt; &amp; &#34;友&#34;")
        assert span.template == "mail.html"
        assert span.line == 7
        assert span.stack == ()

    def test_set_keeps_producing_location(self, mail_render):
        _, rm = mail_render
        # {{ full }} outputs a value produced earlier by m.full(); the
        # map still points at the macro in macros.html, not at the
        # {{ full }} expression in mail.html.
        (span,) = spans_with_text(rm, "李&lt;四&gt;")
        assert span.template == "macros.html"
        assert span.stack[0] == CallFrame("macro", "full", "mail.html", 5, 1)

    def test_normal_render_has_no_mapping_overhead(self, mail_env):
        template = mail_env.get_template("mail.html")
        result = template.render(**CTX)
        assert isinstance(result, str)
        assert all(isinstance(piece, str) for piece in template.generate(**CTX))
        # No mapped template was compiled for regular rendering.
        assert mail_env._mapped_cache == {}

    def test_mapped_templates_are_cached(self, mail_env):
        template = mail_env.get_template("mail.html")
        first = template.render_with_map(**CTX)
        cached = dict(mail_env._mapped_cache)
        second = template.render_with_map(**CTX)
        assert first == second
        assert mail_env._mapped_cache == cached


class TestSmallTemplates:
    def test_from_string(self):
        env = Environment()
        template = env.from_string("Hello {{ name }}!")
        rm = template.render_with_map(name="World")
        assert rm.text == "Hello World!"
        assert_full_coverage(rm)
        assert all(span.template is None for span in rm.spans)

    def test_empty_template(self):
        env = Environment()
        rm = env.from_string("").render_with_map()
        assert rm.text == ""
        assert rm.spans == []

    def test_zero_length_output_pieces(self):
        env = Environment()
        template = env.from_string("a{{ '' }}b")
        rm = template.render_with_map()
        assert rm.text == "ab"
        assert_full_coverage(rm)

    def test_loop_spans(self):
        env = Environment()
        template = env.from_string("{% for x in xs %}({{ x }}){% endfor %}")
        rm = template.render_with_map(xs=[1, 2, 3])
        assert rm.text == "(1)(2)(3)"
        assert_full_coverage(rm)
        values = spans_with_text(rm, "1") + spans_with_text(rm, "2")
        assert len(values) == 2

    def test_finalize_result_attributed_to_expression(self):
        env = Environment(finalize=lambda value: f"[{value}]")
        template = env.from_string("{{ value }}")
        rm = template.render_with_map(value="x")
        assert rm.text == template.render(value="x") == "[x]"
        assert_full_coverage(rm)
        (span,) = rm.spans
        assert span.template is None
        assert span.stack == ()

    def test_filter_block(self):
        env = Environment()
        template = env.from_string("{% filter upper %}ab{{ x }}{% endfilter %}")
        rm = template.render_with_map(x="cd")
        assert rm.text == template.render(x="cd") == "ABCD"
        assert_full_coverage(rm)

    def test_call_block(self):
        env = Environment(
            loader=DictLoader(
                {"m.html": "{% macro wrap() %}<div>{{ caller() }}</div>{% endmacro %}"}
            ),
            autoescape=True,
        )
        template = env.from_string(
            "{% import 'm.html' as m %}{% call m.wrap() %}Hi {{ name }}{% endcall %}"
        )
        rm = template.render_with_map(name="<x>")
        assert rm.text == template.render(name="<x>") == "<div>Hi &lt;x&gt;</div>"
        assert_full_coverage(rm)
        (span,) = spans_with_text(rm, "<div>")
        assert span.template == "m.html"
        assert span.stack[0].kind == "macro"
        assert span.stack[0].name == "wrap"

    def test_recursive_loop(self):
        env = Environment()
        template = env.from_string(
            "{% for item in tree recursive %}{{ item.v }}{{ loop(item.c) }}{% endfor %}"
        )
        ctx = {"tree": [{"v": "a", "c": [{"v": "b", "c": []}]}]}
        rm = template.render_with_map(**ctx)
        assert rm.text == template.render(**ctx) == "ab"
        assert_full_coverage(rm)

    def test_include_select_and_ignore_missing(self):
        env = Environment(loader=DictLoader({"b.html": "B{{ 1 + 1 }}"}))
        template = env.from_string(
            "{% include ['missing.html', 'b.html'] %}"
            "{% include 'nope.html' ignore missing %}"
        )
        rm = template.render_with_map()
        assert rm.text == template.render() == "B2"
        assert_full_coverage(rm)
        (span,) = spans_with_text(rm, "B")
        assert span.template == "b.html"
        assert span.stack == (CallFrame("include", "b.html", None, 1, 1),)

    def test_async_not_supported(self):
        env = Environment(enable_async=True)
        template = env.from_string("hello")

        with pytest.raises(RuntimeError):
            template.render_with_map()

    def test_macro_frame_has_call_site(self):
        env = Environment(
            loader=DictLoader({"m.html": "{% macro hi() %}hi{% endmacro %}"})
        )
        template = env.from_string("{% import 'm.html' as m %}\n{{ m.hi() }}")
        rm = template.render_with_map()
        (span,) = spans_with_text(rm, "hi")
        assert span.template == "m.html"
        (frame,) = span.stack
        assert frame.kind == "macro"
        assert frame.name == "hi"
        assert frame.template is None
        assert frame.line == 2

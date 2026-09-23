import asyncio
import threading

import pytest
from markupsafe import Markup

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.exceptions import TemplateRuntimeError
from jinja2.exceptions import TemplateSyntaxError
from jinja2.runtime import MacroStream

MACRO_DEF = (
    "{% macro rows(data) %}"
    "{% for row in data %}<tr><td>{{ row }}</td></tr>\n{% endfor %}"
    "{% endmacro %}"
)
STREAMING_DEF = MACRO_DEF.replace("{% macro", "{% streaming macro")
TABLE_TEMPLATE = "<table>\n{{ rows(data) }}</table>"

ROWS = ["<alpha>", "&beta"]
ROW_1_ESCAPED = "<tr><td>&lt;alpha&gt;</td></tr>\n"
ROW_2_ESCAPED = "<tr><td>&amp;beta&gt;</td></tr>\n"


def render_normal(env, **kwargs):
    return env.from_string(MACRO_DEF + TABLE_TEMPLATE).render(**kwargs)


class TestStreamingGenerate:
    def test_first_row_arrives_while_source_blocked(self):
        """The caller reads the complete first row while the data
        iterator is still blocked on an event.  Nothing consumes the
        source in the background.
        """
        env = Environment(autoescape=True)
        t = env.from_string(STREAMING_DEF + TABLE_TEMPLATE)
        release = threading.Event()

        def data():
            yield ROWS[0]
            # If the implementation buffered the whole macro body, the
            # caller could not release the event and this would time
            # out instead of blocking forever.
            if not release.wait(5):
                raise RuntimeError("source was consumed without being released")
            yield ROWS[1]

        chunks = []
        saw_first_row_early = False

        for chunk in t.generate(data=data()):
            chunks.append(chunk)
            text = "".join(chunks)

            if not release.is_set() and text.endswith(ROW_1_ESCAPED):
                # The complete first row is readable even though the
                # source has not produced the second row yet.
                assert ROW_2_ESCAPED not in text
                saw_first_row_early = True
                release.set()

        assert saw_first_row_early
        assert "".join(chunks) == render_normal(env, data=ROWS)

    def test_first_row_arrives_while_source_blocked_async(self):
        """Same as the sync test, but through generate_async."""
        env = Environment(autoescape=True, enable_async=True)
        t = env.from_string(STREAMING_DEF + TABLE_TEMPLATE)

        async def main():
            release = asyncio.Event()
            first_row_seen = asyncio.Event()

            async def data():
                yield ROWS[0]
                # Times out instead of blocking forever if the macro
                # body is buffered and the caller can never release.
                await asyncio.wait_for(release.wait(), 5)
                yield ROWS[1]

            async def setter():
                await first_row_seen.wait()
                release.set()

            setter_task = asyncio.ensure_future(setter())
            chunks = []
            saw_first_row_early = False

            try:
                async for chunk in t.generate_async(data=data()):
                    chunks.append(chunk)
                    text = "".join(chunks)

                    if not release.is_set() and text.endswith(ROW_1_ESCAPED):
                        assert ROW_2_ESCAPED not in text
                        saw_first_row_early = True
                        first_row_seen.set()
            finally:
                setter_task.cancel()

            return "".join(chunks), saw_first_row_early

        text, saw_first_row_early = asyncio.run(main())
        assert saw_first_row_early
        assert text == render_normal(env, data=ROWS)

    def test_autoescape_identical_to_normal_macro(self):
        """Concatenated streamed chunks are character identical to the
        buffered output of an equivalent normal macro, with and
        without autoescape.
        """
        data = ["<a>", "&b", "'c\"", "d>"]

        for autoescape in (True, False):
            env = Environment(autoescape=autoescape)
            expected = render_normal(env, data=data)
            t = env.from_string(STREAMING_DEF + TABLE_TEMPLATE)
            assert "".join(t.generate(data=data)) == expected
            assert t.render(data=data) == expected

    def test_generate_async(self, run_async_fn):
        env = Environment(enable_async=True)
        t = env.from_string(STREAMING_DEF + TABLE_TEMPLATE)

        async def main():
            return [chunk async for chunk in t.generate_async(data=ROWS)]

        chunks = run_async_fn(main)
        assert "".join(chunks) == render_normal(env, data=ROWS)

    def test_render_async(self, run_async_fn):
        env = Environment(enable_async=True, autoescape=True)
        t = env.from_string(STREAMING_DEF + TABLE_TEMPLATE)

        async def main():
            return await t.render_async(data=ROWS)

        assert run_async_fn(main) == render_normal(env, data=ROWS)

    def test_template_stream(self, env):
        t = env.from_string(STREAMING_DEF + TABLE_TEMPLATE)
        stream = t.stream(data=ROWS)
        assert "".join(stream) == render_normal(env, data=ROWS)


class TestStreamingMacroBehavior:
    def test_nested_streaming_macro(self, env):
        t = env.from_string(
            "{% streaming macro inner(x) %}<i>{{ x }}</i>{% endmacro %}"
            "{% streaming macro outer(data) %}"
            "{% for x in data %}[{{ inner(x) }}]{% endfor %}"
            "{% endmacro %}"
            "{{ outer(data) }}"
        )
        assert t.render(data=[1, 2]) == "[<i>1</i>][<i>2</i>]"

    def test_normal_macro_inside_streaming_macro(self, env):
        t = env.from_string(
            "{% macro plain(x) %}<b>{{ x }}</b>{% endmacro %}"
            "{% streaming macro sm(data) %}"
            "{% for x in data %}{{ plain(x) }}{% endfor %}"
            "{% endmacro %}"
            "{{ sm(data) }}"
        )
        assert t.render(data=[1, 2]) == "<b>1</b><b>2</b>"

    def test_imported_streaming_macro(self):
        env = Environment(
            loader=DictLoader({"macros.html": STREAMING_DEF}),
            autoescape=True,
        )
        t = env.from_string("{% from 'macros.html' import rows %}" + TABLE_TEMPLATE)
        assert t.render(data=ROWS) == render_normal(env, data=ROWS)

    def test_imported_streaming_macro_as_module(self):
        env = Environment(
            loader=DictLoader({"macros.html": STREAMING_DEF}),
            autoescape=True,
        )
        t = env.from_string(
            "{% import 'macros.html' as m %}<table>\n{{ m.rows(data) }}</table>"
        )
        assert t.render(data=ROWS) == render_normal(env, data=ROWS)

    def test_arguments_and_defaults(self, env):
        t = env.from_string(
            "{% streaming macro m(a, b=2) %}"
            "{{ a }}|{{ b }}|{{ varargs|join(',') }};"
            "{% endmacro %}"
            "{{ m(1) }}{{ m(1, 3, 4, 5) }}"
        )
        assert t.render() == "1|2|;1|3|4,5;"

    def test_call_block(self, env):
        t = env.from_string(
            "{% streaming macro wrap(tag) %}"
            "<{{ tag }}>{{ caller() }}</{{ tag }}>"
            "{% endmacro %}"
            "{% call wrap('div') %}<p>x</p>{% endcall %}"
        )
        assert t.render() == "<div><p>x</p></div>"

    def test_empty_streaming_macro(self, env):
        t = env.from_string("{% streaming macro m() %}{% endmacro %}<{{ m() }}>")
        assert t.render() == "<>"

    def test_macro_stream_api(self, env):
        t = env.from_string(
            "{% streaming macro m(data) %}"
            "{% for x in data %}[{{ x }}]{% endfor %}"
            "{% endmacro %}"
        )
        stream = t.module.m([1, 2])
        assert isinstance(stream, MacroStream)
        assert list(stream.iter_chunks()) == ["[", "1", "]", "[", "2", "]"]

    def test_normal_macro_return_type_unchanged(self):
        t = Environment().from_string("{% macro m() %}<x>{% endmacro %}")
        rv = t.module.m()
        assert isinstance(rv, str)
        assert not isinstance(rv, Markup)
        assert rv == "<x>"

        t = Environment(autoescape=True).from_string("{% macro m() %}<x>{% endmacro %}")
        rv = t.module.m()
        assert isinstance(rv, Markup)
        assert rv == "<x>"


class TestCollect:
    def test_collect_in_assign(self, env):
        t = env.from_string(
            "{% streaming macro m(x) %}[{{ x }}]{% endmacro %}"
            "{% set s = m('a')|collect %}{{ s|upper }}"
        )
        assert t.render() == "[A]"

    def test_collect_in_expressions(self, env):
        t = env.from_string(
            "{% streaming macro m(x) %}[{{ x }}]{% endmacro %}"
            "{{ m('b')|collect ~ '!' }}|{{ [m('c')|collect]|join }}"
        )
        assert t.render() == "[b]!|[c]"

    def test_collect_passthrough(self, env):
        t = env.from_string("{{ 'a'|collect }}|{{ [1]|collect }}")
        assert t.render() == "a|[1]"

    def test_collect_returns_markup_with_autoescape(self):
        t = Environment(autoescape=True).from_string(
            "{% streaming macro m() %}<x>{% endmacro %}"
        )
        rv = t.module.m().collect()
        assert isinstance(rv, Markup)
        assert rv == "<x>"

        t = Environment().from_string("{% streaming macro m() %}<x>{% endmacro %}")
        rv = t.module.m().collect()
        assert isinstance(rv, str)
        assert not isinstance(rv, Markup)
        assert rv == "<x>"

    def test_collect_async(self, run_async_fn):
        env = Environment(enable_async=True, autoescape=True)
        t = env.from_string(
            "{% streaming macro m(x) %}[{{ x }}]{% endmacro %}"
            "{% set s = m('a')|collect %}{{ s }}"
        )

        async def main():
            return await t.render_async()

        assert run_async_fn(main) == "[a]"


class TestImplicitBufferingRejected:
    def test_assign_rejected(self, env):
        t = env.from_string(
            "{% streaming macro m() %}x{% endmacro %}{% set s = m() %}{{ s }}"
        )
        with pytest.raises(TemplateRuntimeError, match="collect"):
            t.render()

    def test_filter_rejected(self, env):
        t = env.from_string("{% streaming macro m() %}x{% endmacro %}{{ m()|upper }}")
        with pytest.raises(TemplateRuntimeError, match="streaming macro"):
            t.render()

    def test_concat_rejected(self, env):
        t = env.from_string("{% streaming macro m() %}x{% endmacro %}{{ 'a' ~ m() }}")
        with pytest.raises(TemplateRuntimeError, match="streaming macro"):
            t.render()

    def test_join_rejected(self, env):
        t = env.from_string("{% streaming macro m() %}x{% endmacro %}{{ m()|join }}")
        with pytest.raises(TemplateRuntimeError, match="streaming macro"):
            t.render()

    def test_iteration_rejected(self, env):
        t = env.from_string(
            "{% streaming macro m() %}x{% endmacro %}{% for x in m() %}{% endfor %}"
        )
        with pytest.raises(TemplateRuntimeError, match="streaming macro"):
            t.render()

    def test_condition_rejected(self, env):
        t = env.from_string(
            "{% streaming macro m() %}x{% endmacro %}{% if m() %}{% endif %}"
        )
        with pytest.raises(TemplateRuntimeError, match="streaming macro"):
            t.render()

    def test_buffered_output_rejected(self, env):
        """A streaming macro call inside a normal macro, filter block,
        or set block would have to be buffered implicitly.
        """
        t = env.from_string(
            "{% streaming macro m() %}x{% endmacro %}"
            "{% macro normal() %}{{ m() }}{% endmacro %}"
            "{{ normal() }}"
        )
        with pytest.raises(TemplateRuntimeError, match="streaming macro"):
            t.render()

    def test_set_block_rejected(self, env):
        t = env.from_string(
            "{% streaming macro m() %}x{% endmacro %}"
            "{% set s %}{{ m() }}{% endset %}"
        )
        with pytest.raises(TemplateRuntimeError, match="streaming macro"):
            t.render()

    def test_rejected_async(self, run_async_fn):
        env = Environment(enable_async=True)
        t = env.from_string("{% streaming macro m() %}x{% endmacro %}{{ m()|upper }}")

        async def main():
            return await t.render_async()

        with pytest.raises(TemplateRuntimeError, match="streaming macro"):
            run_async_fn(main)


class TestStreamingSyntax:
    def test_streaming_requires_macro(self, env):
        with pytest.raises(TemplateSyntaxError, match="expected 'macro'"):
            env.from_string("{% streaming foo %}{% endmacro %}")

    def test_streaming_macro_requires_endmacro(self, env):
        with pytest.raises(TemplateSyntaxError):
            env.from_string("{% streaming macro m() %}")

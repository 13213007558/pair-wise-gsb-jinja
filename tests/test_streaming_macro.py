import asyncio
import threading
from contextlib import aclosing

import pytest
import sniffio
import trio
from markupsafe import Markup

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import TemplateRuntimeError
from jinja2.exceptions import FilterArgumentError
from jinja2.runtime import AsyncMacroStream
from jinja2.runtime import MacroStream

ROWS_MACRO = (
    "{% macro rows(items) streaming %}"
    "{% for item in items %}<tr>{{ item }}</tr>{% endfor %}"
    "{% endmacro %}"
)
ROWS_MACRO_PLAIN = ROWS_MACRO.replace(" streaming %}", " %}")
TABLE = "<table>{{ rows(data) }}</table>"


def render_normal(env, data):
    """The full text produced by an equivalent regular macro."""
    return env.from_string(ROWS_MACRO_PLAIN + TABLE).render(data=data)


class BlockingRows:
    """An iterator that produces the first row, then blocks on an event
    before producing the rest.  ``requested`` records that the consumer
    asked for data beyond the first row.
    """

    def __init__(self, rows, release, requested):
        self._rows = rows
        self._release = release
        self._requested = requested

    def __iter__(self):
        yield self._rows[0]
        self._requested.set()
        assert self._release.wait(10), "test did not release the source"
        yield from self._rows[1:]


@pytest.fixture(params=[True, False], ids=["autoescape", "no-autoescape"])
def any_env(request):
    return Environment(autoescape=request.param)


def test_generate_streams_first_row_before_source_blocks(any_env):
    release = threading.Event()
    requested = threading.Event()
    data = BlockingRows(["r<1", "r2", "r3"], release, requested)
    t = any_env.from_string(ROWS_MACRO + TABLE)
    stream = t.generate(data=data)

    chunks = []
    for chunk in stream:
        chunks.append(chunk)
        if "".join(map(str, chunks)).endswith("</tr>"):
            break

    # The complete first row was delivered although the source iterator
    # is still blocked, and nothing asked it for more data.
    assert (
        "".join(map(str, chunks))
        == "<table><tr>" + any_env.from_string("{{ v }}").render(v="r<1") + "</tr>"
    )
    assert not requested.is_set()

    release.set()
    rest = "".join(map(str, stream))
    assert "".join(map(str, chunks)) + rest == render_normal(
        any_env, ["r<1", "r2", "r3"]
    )


def test_generate_matches_normal_macro(any_env):
    data = ["a<b", "&\\", "c>d"]
    t = any_env.from_string(ROWS_MACRO + TABLE)
    chunks = list(t.generate(data=data))
    assert len(chunks) > 1
    assert "".join(map(str, chunks)) == render_normal(any_env, data)
    assert t.render(data=data) == render_normal(any_env, data)


def test_nested_streaming_macros(any_env):
    t = any_env.from_string(
        "{% macro cell(x) streaming %}<td>{{ x }}</td>{% endmacro %}"
        "{% macro row(items) streaming %}"
        "{% for i in items %}{{ cell(i) }}{% endfor %}"
        "{% endmacro %}"
        "{{ row(data) }}"
    )
    assert t.render(data=["a<", "b"]) == any_env.from_string(
        "{% macro cell(x) %}<td>{{ x }}</td>{% endmacro %}"
        "{% macro row(items) %}"
        "{% for i in items %}{{ cell(i) }}{% endfor %}"
        "{% endmacro %}"
        "{{ row(data) }}"
    ).render(data=["a<", "b"])


def test_imported_streaming_macro(any_env):
    any_env.loader = DictLoader(
        {
            "forms": "{% macro input(name) streaming %}"
            "<input name='{{ name }}'>{% endmacro %}"
        }
    )
    t = any_env.from_string("{% import 'forms' as f %}{{ f.input('a<b') }}")
    expected = "<input name='a&lt;b'>" if any_env.autoescape else "<input name='a<b'>"
    assert t.render() == expected
    assert "".join(map(str, t.generate())) == expected


def test_streaming_macro_arguments(any_env):
    t = any_env.from_string(
        "{% macro m(a, b='B') streaming %}"
        "{{ a }}|{{ b }}|{{ varargs|join(',') }}|{{ kwargs['k'] }}"
        "{% endmacro %}"
        "{{ m('x', 'y', 1, 2, k='v') }}"
    )
    assert t.render() == "x|y|1,2|v"
    t = any_env.from_string(
        "{% macro m(a, b='B') streaming %}{{ a }}{{ b }}{% endmacro %}{{ m('x') }}"
    )
    assert t.render() == "xB"


def test_normal_macro_return_type_unchanged(any_env):
    mod = any_env.from_string("{% macro m() %}x{% endmacro %}").module
    rv = mod.m()
    assert isinstance(rv, Markup if any_env.autoescape else str)
    assert not isinstance(rv, MacroStream)


def test_streaming_macro_module_call(any_env):
    mod = any_env.from_string(
        "{% macro m(v) streaming %}[{{ v }}]{% endmacro %}"
    ).module
    stream = mod.m("<")
    assert isinstance(stream, MacroStream)
    rv = stream.collect()
    if any_env.autoescape:
        assert isinstance(rv, Markup)
        assert rv == "[&lt;]"
    else:
        assert rv == "[<]"


def test_collect_filter(any_env):
    t = any_env.from_string(ROWS_MACRO + "{% set t = rows(data)|collect %}[{{ t }}]")
    data = ["a<", "b"]
    inner = render_normal(any_env, data)[len("<table>") : -len("</table>")]
    assert t.render(data=data) == f"[{inner}]"


def test_collect_in_expressions(any_env):
    t = any_env.from_string(
        "{% macro m() streaming %}ab{% endmacro %}"
        "{{ m()|collect ~ '!' }} {{ (m()|collect)|upper }}"
    )
    assert t.render() == "ab! AB"


def test_collect_rejects_other_values(any_env):
    with pytest.raises(FilterArgumentError):
        any_env.from_string("{{ 'x'|collect }}").render()


@pytest.mark.parametrize(
    "template",
    [
        # concatenation
        "{{ 'y' ~ m() }}",
        # string conversion in a filter
        "{{ m()|upper }}",
        # join filter
        "{{ m()|join(',') }}",
        # set block
        "{% set v %}{{ m() }}{% endset %}{{ v }}",
        # filter block
        "{% filter upper %}{{ m() }}{% endfilter %}",
        # inside a regular macro
        "{% macro outer() %}{{ m() }}{% endmacro %}{{ outer() }}",
        # string format
        "{{ '%s'|format(m()) }}",
    ],
)
def test_implicit_buffering_is_rejected(any_env, template):
    src = "{% macro m() streaming %}x{% endmacro %}" + template
    with pytest.raises(TemplateRuntimeError, match="streaming macro"):
        any_env.from_string(src).render()


def test_implicit_str_is_rejected(any_env):
    mod = any_env.from_string("{% macro m() streaming %}x{% endmacro %}").module
    with pytest.raises(TemplateRuntimeError, match="streaming macro"):
        str(mod.m())
    with pytest.raises(TemplateRuntimeError, match="streaming macro"):
        list(mod.m())


def test_streaming_macro_in_loop(any_env):
    t = any_env.from_string(
        "{% macro m(v) streaming %}({{ v }}){% endmacro %}"
        "{% for i in data %}{{ m(i) }}{% endfor %}"
    )
    assert t.render(data=[1, 2]) == "(1)(2)"


def test_empty_streaming_macro(any_env):
    t = any_env.from_string("{% macro m() streaming %}{% endmacro %}{{ m() }}x")
    assert t.render() == "x"


class AsyncBlockingRows:
    def __init__(self, rows, release, requested):
        self._rows = rows
        self._release = release
        self._requested = requested

    def __aiter__(self):
        return _AsyncBlockingRowsIter(self._rows, self._release, self._requested)


class _AsyncBlockingRowsIter:
    def __init__(self, rows, release, requested):
        self._rows = rows
        self._release = release
        self._requested = requested
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._rows):
            raise StopAsyncIteration

        if self._idx == 1:
            self._requested.set()
            await wait_event(self._release)

        row = self._rows[self._idx]
        self._idx += 1
        return row


async def wait_event(event):
    if sniffio.current_async_library() == "trio":
        with trio.move_on_after(10):
            await event.wait()
    else:
        await asyncio.wait_for(event.wait(), 10)


def make_event():
    if sniffio.current_async_library() == "trio":
        return trio.Event()
    return asyncio.Event()


def test_generate_async_streams_first_row_before_source_blocks(run_async_fn):
    async def main():
        env = Environment(autoescape=True, enable_async=True)
        release = make_event()
        requested = make_event()
        data = AsyncBlockingRows(["r<1", "r2", "r3"], release, requested)
        t = env.from_string(ROWS_MACRO + TABLE)

        chunks = []
        async with aclosing(t.generate_async(data=data)) as stream:
            async for chunk in stream:
                chunks.append(chunk)
                if "".join(map(str, chunks)).endswith("</tr>"):
                    break

        assert "".join(map(str, chunks)) == "<table><tr>r&lt;1</tr>"
        assert not requested.is_set()

        release.set()
        rest = []
        async with aclosing(t.generate_async(data=data)) as stream:
            async for chunk in stream:
                rest.append(chunk)
        return "".join(map(str, chunks)), "".join(map(str, rest))

    first, rest = run_async_fn(main)
    assert first == "<table><tr>r&lt;1</tr>"
    assert rest == "<table><tr>r&lt;1</tr><tr>r2</tr><tr>r3</tr></table>"


def test_generate_async_matches_normal_macro(run_async_fn):
    async def main():
        env = Environment(autoescape=True, enable_async=True)
        data = ["a<b", "&\\", "c>d"]
        t = env.from_string(ROWS_MACRO + TABLE)
        chunks = [c async for c in t.generate_async(data=data)]
        normal = env.from_string(ROWS_MACRO_PLAIN + TABLE)
        return chunks, await normal.render_async(data=data)

    chunks, expected = run_async_fn(main)
    assert len(chunks) > 1
    assert "".join(map(str, chunks)) == expected


def test_async_nested_and_imported(run_async_fn):
    async def main():
        env = Environment(
            autoescape=True,
            enable_async=True,
            loader=DictLoader(
                {"forms": "{% macro input(n) streaming %}<i>{{ n }}</i>{% endmacro %}"}
            ),
        )
        t = env.from_string(
            "{% import 'forms' as f %}"
            "{% macro wrap(v) streaming %}<b>{{ f.input(v) }}</b>{% endmacro %}"
            "{{ wrap('x<') }}"
        )
        return await t.render_async()

    assert run_async_fn(main) == "<b><i>x&lt;</i></b>"


def test_async_collect(run_async_fn):
    async def main():
        env = Environment(autoescape=True, enable_async=True)
        t = env.from_string(
            "{% macro m(v) streaming %}[{{ v }}]{% endmacro %}{{ m('<')|collect }}"
        )
        captured = []
        t2 = env.from_string(
            "{% macro m() streaming %}x{% endmacro %}{{ capture(m()) }}"
        )

        def capture(value):
            captured.append(value)
            return ""

        await t2.render_async(capture=capture)
        stream = captured[0]
        assert isinstance(stream, AsyncMacroStream)
        return await t.render_async(), await stream.collect()

    rv, collected = run_async_fn(main)
    assert rv == "[&lt;]"
    assert collected == "x"


def test_async_implicit_buffering_is_rejected(run_async_fn):
    async def main():
        env = Environment(enable_async=True)
        t = env.from_string("{% macro m() streaming %}x{% endmacro %}{{ 'y' ~ m() }}")
        await t.render_async()

    with pytest.raises(TemplateRuntimeError, match="streaming macro"):
        run_async_fn(main)

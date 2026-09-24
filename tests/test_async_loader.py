import asyncio

import pytest

from jinja2 import BaseLoader
from jinja2 import ChoiceLoader
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import PrefixLoader
from jinja2.bccache import FileSystemBytecodeCache
from jinja2.exceptions import TemplateNotFound


class RecordingAsyncLoader(BaseLoader):
    """An async-only loader that records how its source coroutines end.

    ``calls`` lists every requested template, ``finished`` lists templates
    whose coroutine returned a source, and ``aborted`` lists templates
    whose coroutine was cancelled or closed before completing.
    """

    def __init__(self, mapping, gate=None, settle=False):
        self.mapping = mapping
        self.gate = gate
        self.settle = settle
        self.calls = []
        self.finished = []
        self.aborted = []

    async def get_source_async(self, environment, template):
        self.calls.append(template)

        try:
            if self.gate is not None:
                await self.gate.wait()

            if self.settle:
                await asyncio.sleep(0)

            if template not in self.mapping:
                raise TemplateNotFound(template)

            source = self.mapping[template]
        except (asyncio.CancelledError, GeneratorExit):
            self.aborted.append(template)
            raise
        else:
            self.finished.append(template)
            return source, None, lambda: True


@pytest.fixture
def async_mapping():
    return {
        "macros": "{% macro val() %}M{% endmacro %}",
        "frag": "<F>",
        "base": "[{% block b %}D{% endblock %}]",
        "child": (
            "{% extends 'base' %}{% block b %}{% include 'frag' %}"
            "{% import 'macros' as m %}{{ m.val() }}{% endblock %}"
        ),
    }


def test_async_loader_used_for_include_import_extends(async_mapping):
    loader = RecordingAsyncLoader(async_mapping)
    env = Environment(loader=loader, enable_async=True)

    async def main():
        t = await env.get_template_async("child")
        return await t.render_async()

    assert asyncio.run(main()) == "[<F>M]"
    # extends, include, and import all went through the async loader.
    assert loader.calls == ["child", "base", "frag", "macros"]
    assert loader.finished == ["child", "base", "frag", "macros"]
    assert loader.aborted == []


def test_nested_includes_and_cache():
    loader = RecordingAsyncLoader(
        {
            "a": "A{% include 'b' %}",
            "b": "B{% include 'c' %}",
            "c": "C",
        }
    )
    env = Environment(loader=loader, enable_async=True)

    async def main():
        t = await env.get_template_async("a")
        first = await t.render_async()
        second = await t.render_async()
        return first, second

    assert asyncio.run(main()) == ("ABC", "ABC")
    # Each template was read once, the cache served repeated renders.
    assert loader.calls == ["a", "b", "c"]


def test_sync_loader_still_works_in_async_env():
    env = Environment(
        loader=DictLoader({"frag": "<F>", "main": "{% include 'frag' %}"}),
        enable_async=True,
    )

    async def main():
        t = await env.get_template_async("main")
        return await t.render_async()

    assert asyncio.run(main()) == "<F>"


def test_sync_render_rejects_async_only_loader(async_mapping):
    env = Environment(loader=RecordingAsyncLoader(async_mapping))

    with pytest.raises(RuntimeError, match="only implements async loading"):
        env.get_template("child")


def test_sync_render_rejects_async_only_loader_include(async_mapping):
    env = Environment(loader=RecordingAsyncLoader(async_mapping))
    t = env.from_string("{% include 'frag' %}")

    with pytest.raises(RuntimeError, match="only implements async loading"):
        t.render()


def test_concurrent_renders_do_not_share_coroutines():
    gate = asyncio.Event()
    loader = RecordingAsyncLoader({"main": "{{ 40 + 2 }}"}, gate=gate)
    env = Environment(loader=loader, enable_async=True)

    async def main():
        async def render_one():
            t = await env.get_template_async("main")
            return await t.render_async()

        tasks = [asyncio.create_task(render_one()) for _ in range(5)]
        # Let every task reach the gated source read.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert loader.calls == ["main"] * 5
        gate.set()
        return await asyncio.gather(*tasks)

    assert asyncio.run(main()) == ["42"] * 5
    # Each concurrent load drove its own coroutine to completion.
    assert loader.finished == ["main"] * 5
    assert loader.aborted == []


def test_concurrent_renders_populate_cache():
    loader = RecordingAsyncLoader({"main": "{{ 40 + 2 }}"})
    env = Environment(loader=loader, enable_async=True)

    async def main():
        async def render_one():
            t = await env.get_template_async("main")
            return await t.render_async()

        first = await asyncio.gather(*(render_one() for _ in range(3)))
        calls_after_first = len(loader.calls)
        second = await asyncio.gather(*(render_one() for _ in range(3)))
        return first, second, calls_after_first

    first, second, calls_after_first = asyncio.run(main())
    assert first == ["42"] * 3
    assert second == ["42"] * 3
    assert calls_after_first >= 1
    # The cached template served the second round without new reads.
    assert len(loader.calls) == calls_after_first


def test_cancel_during_source_read():
    gate = asyncio.Event()
    loader = RecordingAsyncLoader({"main": "never"}, gate=gate)
    env = Environment(loader=loader, enable_async=True)

    async def main():
        task = asyncio.create_task(env.get_template_async("main"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert loader.calls == ["main"]
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    # The unfinished coroutine was closed and nothing was cached.
    assert loader.aborted == ["main"]
    assert loader.finished == []
    assert len(env.cache) == 0


class CancelOnceCompileEnvironment(Environment):
    """Requests task cancellation from inside the first compile."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancel_compile = True

    def compile(self, source, name=None, filename=None, raw=False, defer_init=False):
        if self.cancel_compile:
            self.cancel_compile = False
            asyncio.current_task().cancel()

        return super().compile(source, name, filename, raw, defer_init)


def test_cancel_requested_during_compile():
    loader = RecordingAsyncLoader(
        {"main": "{% include 'inner' %}", "inner": "I"},
        settle=True,
    )
    env = CancelOnceCompileEnvironment(loader=loader, enable_async=True)

    async def main():
        # The cancellation requested during compile is delivered at the
        # next await point, the source read for the included template.
        with pytest.raises(asyncio.CancelledError):
            t = await env.get_template_async("main")
            await t.render_async()

        assert loader.aborted == ["inner"]
        # The cancelled include was not cached, the completed parent was.
        assert "inner" not in [key[1] for key in env.cache]
        assert "main" in [key[1] for key in env.cache]

        # No hidden retry happened, an explicit retry renders cleanly.
        t = await env.get_template_async("main")
        assert await t.render_async() == "I"

    asyncio.run(main())
    assert loader.calls == ["main", "inner", "inner"]


def test_cancel_during_render_callback():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def wait_fn():
        entered.set()
        await release.wait()
        return "late"

    loader = RecordingAsyncLoader({"main": "{{ wait_fn() }}|done"})
    env = Environment(loader=loader, enable_async=True)

    async def main():
        t = await env.get_template_async("main")
        task = asyncio.create_task(t.render_async(wait_fn=wait_fn))
        await entered.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    # The load completed before the render callback was cancelled.
    assert loader.finished == ["main"]
    assert loader.aborted == []


def test_cancel_during_generate_keeps_partial_output():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hook():
        entered.set()
        await release.wait()
        return "x"

    loader = RecordingAsyncLoader({"main": "1{{ hook() }}2"})
    env = Environment(loader=loader, enable_async=True)

    async def main():
        t = await env.get_template_async("main")
        collected = []

        async def consume():
            async for chunk in t.generate_async(hook=hook):
                collected.append(chunk)

        task = asyncio.create_task(consume())
        await entered.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        return collected

    # The already yielded part is kept, the generator ends without retry.
    assert asyncio.run(main()) == ["1"]


def test_include_ignore_missing_async():
    loader = RecordingAsyncLoader({"main": "{% include 'missing' ignore missing %}ok"})
    env = Environment(loader=loader, enable_async=True)

    async def main():
        t = await env.get_template_async("main")
        return await t.render_async()

    assert asyncio.run(main()) == "ok"
    assert "missing" in loader.calls


def test_select_template_async():
    loader = RecordingAsyncLoader(
        {"main": "{% include ['missing', 'frag'] %}", "frag": "F"}
    )
    env = Environment(loader=loader, enable_async=True)

    async def main():
        t = await env.get_template_async("main")
        return await t.render_async()

    assert asyncio.run(main()) == "F"
    assert loader.calls == ["main", "missing", "frag"]


def test_prefix_and_choice_loader_async():
    async_loader = RecordingAsyncLoader({"frag": "F"})
    env = Environment(
        loader=PrefixLoader({"a": ChoiceLoader([DictLoader({}), async_loader])}),
        enable_async=True,
    )

    async def main():
        t = await env.get_template_async("a/frag")
        return await t.render_async()

    assert asyncio.run(main()) == "F"
    assert async_loader.calls == ["frag"]


class CountingCompileEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compiles = 0

    def compile(self, source, name=None, filename=None, raw=False, defer_init=False):
        self.compiles += 1
        return super().compile(source, name, filename, raw, defer_init)


def test_bytecode_cache_with_async_loader(tmp_path):
    bcc = FileSystemBytecodeCache(str(tmp_path))
    loader = RecordingAsyncLoader({"main": "{{ 6 * 7 }}"})
    env1 = CountingCompileEnvironment(
        loader=loader, enable_async=True, bytecode_cache=bcc
    )
    env2 = CountingCompileEnvironment(
        loader=loader, enable_async=True, bytecode_cache=bcc
    )

    async def main():
        t1 = await env1.get_template_async("main")
        first = await t1.render_async()
        t2 = await env2.get_template_async("main")
        second = await t2.render_async()
        return first, second

    assert asyncio.run(main()) == ("42", "42")
    # The source was read once per environment, but compiled only once.
    assert loader.calls == ["main", "main"]
    assert env1.compiles == 1
    assert env2.compiles == 0


def test_auto_reload_uptodate_from_async_loader():
    mapping = {"main": "one"}

    class ReloadLoader(BaseLoader):
        async def get_source_async(self, environment, template):
            source = mapping[template]
            return source, None, lambda: source == mapping[template]

    env = Environment(loader=ReloadLoader(), enable_async=True)

    async def main():
        t = await env.get_template_async("main")
        assert await t.render_async() == "one"
        mapping["main"] = "two"
        t = await env.get_template_async("main")
        return await t.render_async()

    assert asyncio.run(main()) == "two"

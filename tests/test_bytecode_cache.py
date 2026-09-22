import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.bccache import Bucket
from jinja2.bccache import FileSystemBytecodeCache
from jinja2.bccache import MemcachedBytecodeCache
from jinja2.exceptions import TemplateNotFound


@pytest.fixture
def env(package_loader, tmp_path):
    bytecode_cache = FileSystemBytecodeCache(str(tmp_path))
    return Environment(loader=package_loader, bytecode_cache=bytecode_cache)


class TestByteCodeCache:
    def test_simple(self, env):
        tmpl = env.get_template("test.html")
        assert tmpl.render().strip() == "BAR"
        pytest.raises(TemplateNotFound, env.get_template, "missing.html")

    def test_macro_keyword_only_args_roundtrip(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        loader = DictLoader(
            {
                "macros.html": (
                    "{% macro badge(text, *, tone='info', compact) %}"
                    "[{{ text }}|{{ tone }}|{{ compact }}]{% endmacro %}"
                ),
                "page.html": (
                    "{% import 'macros.html' as m %}"
                    "{% macro plain(text, *, tone='info') %}"
                    "<{{ tone }}>{{ text }}</{{ tone }}>{% endmacro %}"
                    "{% macro outer(fn, *, text) %}{{ fn(text=text) }}"
                    "{% endmacro %}"
                    "{% macro inner(*, text, tone='info') %}"
                    "{{ tone }}:{{ text }}{% endmacro %}"
                    "{{ plain('a', tone='warn') }}"
                    "{{ m.badge('b', compact=true) }}"
                    "{{ outer(inner, text='c') }}"
                ),
                "broken.html": (
                    "{% macro badge(*, compact) %}{{ compact }}{% endmacro %}"
                    "{{ badge() }}"
                ),
            }
        )

        env1 = Environment(loader=loader, bytecode_cache=cache)
        tmpl = env1.get_template("page.html")
        assert tmpl.render() == "<warn>a</warn>[b|info|True]info:c"
        assert tmpl.module.plain.kwonlyarguments == ("tone",)
        assert list(tmp_path.iterdir()), "expected bytecode cache files"

        # A fresh environment must load the compiled bytecode from the
        # cache and bind keyword-only arguments the same way.
        env2 = Environment(loader=loader, bytecode_cache=cache)
        tmpl = env2.get_template("page.html")
        assert tmpl.render() == "<warn>a</warn>[b|info|True]info:c"
        assert tmpl.module.plain.arguments == ("text",)
        assert tmpl.module.plain.kwonlyarguments == ("tone",)
        imported = env2.get_template("macros.html")
        assert imported.module.badge.kwonlyarguments == ("tone", "compact")

        with pytest.raises(TypeError, match="missing required keyword-only"):
            env2.get_template("broken.html").render()


class MockMemcached:
    class Error(Exception):
        pass

    key = None
    value = None
    timeout = None

    def get(self, key):
        return self.value

    def set(self, key, value, timeout=None):
        self.key = key
        self.value = value
        self.timeout = timeout

    def get_side_effect(self, key):
        raise self.Error()

    def set_side_effect(self, *args):
        raise self.Error()


class TestMemcachedBytecodeCache:
    def test_dump_load(self):
        memcached = MockMemcached()
        m = MemcachedBytecodeCache(memcached)

        b = Bucket(None, "key", "")
        b.code = "code"
        m.dump_bytecode(b)
        assert memcached.key == "jinja2/bytecode/key"

        b = Bucket(None, "key", "")
        m.load_bytecode(b)
        assert b.code == "code"

    def test_exception(self):
        memcached = MockMemcached()
        memcached.get = memcached.get_side_effect
        memcached.set = memcached.set_side_effect
        m = MemcachedBytecodeCache(memcached)
        b = Bucket(None, "key", "")
        b.code = "code"

        m.dump_bytecode(b)
        m.load_bytecode(b)

        m.ignore_memcache_errors = False

        with pytest.raises(MockMemcached.Error):
            m.dump_bytecode(b)

        with pytest.raises(MockMemcached.Error):
            m.load_bytecode(b)

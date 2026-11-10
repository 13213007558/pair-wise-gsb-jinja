import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.bccache import Bucket
from jinja2.bccache import FileSystemBytecodeCache
from jinja2.bccache import MemcachedBytecodeCache
from jinja2.exceptions import SecurityError
from jinja2.exceptions import TemplateNotFound
from jinja2.exceptions import TemplateRuntimeError
from jinja2.sandbox import SandboxedEnvironment
from jinja2.sandbox import unsafe


@pytest.fixture
def env(package_loader, tmp_path):
    bytecode_cache = FileSystemBytecodeCache(str(tmp_path))
    return Environment(loader=package_loader, bytecode_cache=bytecode_cache)


class CountingBytecodeCache(FileSystemBytecodeCache):
    """Records how many times a cache miss forced a recompile."""

    def __init__(self, directory):
        super().__init__(str(directory))
        self.compiles = 0

    def load_bytecode(self, bucket):
        super().load_bytecode(bucket)
        if bucket.code is None:
            self.compiles += 1


class PrivateStuff:
    @unsafe
    def secret(self):
        return "leaked"

    def safe(self):
        return "ok"


class TestByteCodeCache:
    def test_simple(self, env):
        tmpl = env.get_template("test.html")
        assert tmpl.render().strip() == "BAR"
        pytest.raises(TemplateNotFound, env.get_template, "missing.html")


class TestCacheSecurityBoundary:
    """The bytecode cache must distinguish compile settings that change the
    generated security checks instead of keying only on template name and
    source.
    """

    source = "{{ obj.secret() }}"

    def _envs(self, tmp_path):
        bcc = CountingBytecodeCache(tmp_path)
        loader = DictLoader({"tpl.html": self.source})
        normal = Environment(loader=loader, bytecode_cache=bcc)
        sandbox = SandboxedEnvironment(loader=loader, bytecode_cache=bcc)
        return bcc, normal, sandbox

    @pytest.mark.parametrize("order", ["normal-first", "sandbox-first"])
    def test_sandbox_never_reuses_unguarded_bytecode(self, tmp_path, order):
        bcc, normal, sandbox = self._envs(tmp_path)
        first, second = (
            (normal, sandbox) if order == "normal-first" else (sandbox, normal)
        )

        first.get_template("tpl.html")
        second.get_template("tpl.html")

        # The harmless call is allowed in a normal environment but rejected by
        # the sandbox, regardless of which environment warmed the cache.
        assert (
            normal.get_template("tpl.html").render(obj=PrivateStuff())
            == "leaked"
        )
        with pytest.raises(SecurityError):
            sandbox.get_template("tpl.html").render(obj=PrivateStuff())

        # Each distinct configuration owns its own entry.
        assert bcc.compiles == 2

    def test_same_configuration_reuses_cached_bytecode(self, tmp_path):
        bcc, normal, _ = self._envs(tmp_path)
        normal.get_template("tpl.html")
        another = Environment(
            loader=DictLoader({"tpl.html": self.source}), bytecode_cache=bcc
        )
        another.get_template("tpl.html")
        assert bcc.compiles == 1

    def test_intercepted_operators_get_distinct_entries(self, tmp_path):
        bcc = CountingBytecodeCache(tmp_path)
        loader = DictLoader({"sum.html": "{{ 1 + 2 }}"})

        def boom(left, right):
            raise TemplateRuntimeError("operator guard executed")

        guarded = SandboxedEnvironment(loader=loader, bytecode_cache=bcc)
        guarded.intercepted_binops = frozenset(["+"])
        guarded.binop_table["+"] = boom

        plain = SandboxedEnvironment(loader=loader, bytecode_cache=bcc)

        # Warm the cache with the intercepted variant, whose guard must run.
        with pytest.raises(TemplateRuntimeError):
            guarded.get_template("sum.html").render()

        # The non-intercepting environment must not inherit that bytecode and
        # renders with the native operator instead.
        assert plain.get_template("sum.html").render() == "3"
        assert bcc.compiles == 2

        # A third environment with the same interception setting reuses the
        # guarded entry, so its guard executes without recompiling.
        guarded_too = SandboxedEnvironment(loader=loader, bytecode_cache=bcc)
        guarded_too.intercepted_binops = frozenset(["+"])
        guarded_too.binop_table["+"] = boom
        with pytest.raises(TemplateRuntimeError):
            guarded_too.get_template("sum.html").render()
        assert bcc.compiles == 2

    def test_async_and_sync_get_distinct_entries(self, tmp_path):
        bcc = CountingBytecodeCache(tmp_path)
        loader = DictLoader({"loop.html": "{% for i in [1, 2] %}{{ i }}{% endfor %}"})
        sync_env = Environment(loader=loader, bytecode_cache=bcc)
        async_env = Environment(
            loader=loader, bytecode_cache=bcc, enable_async=True
        )

        assert sync_env.get_template("loop.html").render() == "12"

        import asyncio

        async def render_async():
            tmpl = async_env.get_template("loop.html")
            return await tmpl.render_async()

        assert asyncio.run(render_async()) == "12"
        assert bcc.compiles == 2

    def test_namespace_is_stable_and_address_free(self):
        first = SandboxedEnvironment()
        first.intercepted_binops = frozenset(["+", "-"])
        first.intercepted_unops = frozenset(["-"])

        second = SandboxedEnvironment()
        # A different set construction order must produce the same namespace.
        second.intercepted_binops = frozenset(["-", "+"])
        second.intercepted_unops = frozenset(["-"])

        namespace = first.get_cache_namespace()
        assert namespace == second.get_cache_namespace()
        assert "0x" not in namespace
        assert "object at" not in namespace

        plain = SandboxedEnvironment()
        assert plain.get_cache_namespace() != namespace

    def test_explicit_cache_namespace_separates_custom_compilers(self, tmp_path):
        bcc = CountingBytecodeCache(tmp_path)
        loader = DictLoader({"tpl.html": self.source})
        first = Environment(
            loader=loader, bytecode_cache=bcc, cache_namespace="custom-v1"
        )
        second = Environment(
            loader=loader, bytecode_cache=bcc, cache_namespace="custom-v2"
        )
        first.get_template("tpl.html")
        second.get_template("tpl.html")
        assert bcc.compiles == 2

        # A process restart with the same explicit name hits the same entry.
        third = Environment(
            loader=loader, bytecode_cache=bcc, cache_namespace="custom-v1"
        )
        third.get_template("tpl.html")
        assert bcc.compiles == 2

    def test_old_format_entries_are_rejected(self, tmp_path):
        import marshal
        import pickle
        import sys

        bcc = FileSystemBytecodeCache(str(tmp_path))
        loader = DictLoader({"tpl.html": self.source})
        env = Environment(loader=loader, bytecode_cache=bcc)
        bucket = bcc.get_bucket(env, "tpl.html", None, self.source)

        old_magic = b"j2" + pickle.dumps(5, 2) + pickle.dumps(
            (sys.version_info[0] << 24) | sys.version_info[1], 2
        )
        filename = bcc._get_cache_filename(bucket)
        with open(filename, "wb") as f:
            f.write(old_magic)
            pickle.dump(bucket.checksum, f, 2)
            marshal.dump(compile("x = 1", "<template>", "exec"), f)

        stale = bcc.get_bucket(env, "tpl.html", None, self.source)
        assert stale.code is None


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

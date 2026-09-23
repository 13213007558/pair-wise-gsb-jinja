import os
import pickle
import sys
import threading

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import StrictUndefined
from jinja2.bccache import bc_magic
from jinja2.bccache import Bucket
from jinja2.bccache import FileSystemBytecodeCache
from jinja2.bccache import get_environment_key
from jinja2.bccache import MemcachedBytecodeCache
from jinja2.exceptions import TemplateNotFound
from jinja2.ext import Extension
from jinja2.loaders import ModuleLoader


@pytest.fixture
def env(package_loader, tmp_path):
    bytecode_cache = FileSystemBytecodeCache(str(tmp_path))
    return Environment(loader=package_loader, bytecode_cache=bytecode_cache)


class TestByteCodeCache:
    def test_simple(self, env):
        tmpl = env.get_template("test.html")
        assert tmpl.render().strip() == "BAR"
        pytest.raises(TemplateNotFound, env.get_template, "missing.html")


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


@pytest.fixture
def compile_counter(monkeypatch):
    """Counts how often templates were compiled, so tests can tell
    cache hits from cache misses."""
    calls = []
    original = Environment.compile

    def counting_compile(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Environment, "compile", counting_compile)
    return calls


def make_env(cache, source="{{ value }}", **kwargs):
    """An environment with an in-memory loader sharing ``cache``."""
    return Environment(
        loader=DictLoader({"index.html": source}), bytecode_cache=cache, **kwargs
    )


class TestEnvironmentBoundCache:
    def test_same_policy_hits(self, tmp_path, compile_counter):
        cache = FileSystemBytecodeCache(str(tmp_path))
        env1 = make_env(cache)
        assert env1.get_template("index.html").render(value="42") == "42"
        assert len(compile_counter) == 1

        # a second environment with identical options reuses the bytecode
        env2 = make_env(cache)
        assert env2.get_template("index.html").render(value="42") == "42"
        assert len(compile_counter) == 1
        assert len(os.listdir(tmp_path)) == 1

    def test_autoescape_change_misses(self, tmp_path, compile_counter):
        cache = FileSystemBytecodeCache(str(tmp_path))
        env_plain = make_env(cache, autoescape=False)
        assert env_plain.get_template("index.html").render(value="<b>") == "<b>"

        env_escape = make_env(cache, autoescape=True)
        assert env_escape.get_template("index.html").render(value="<b>") == "&lt;b&gt;"
        assert len(compile_counter) == 2
        assert len(os.listdir(tmp_path)) == 2

        # alternating back reuses the original entry instead of
        # recompiling or returning the other environment's bytecode
        env_plain2 = make_env(cache, autoescape=False)
        assert env_plain2.get_template("index.html").render(value="<b>") == "<b>"
        assert len(compile_counter) == 2
        assert len(os.listdir(tmp_path)) == 2

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"optimized": False},
            {"undefined": StrictUndefined},
            {"trim_blocks": True},
            {"lstrip_blocks": True},
            {"keep_trailing_newline": True},
            {"enable_async": True},
        ],
    )
    def test_option_change_misses(self, tmp_path, compile_counter, kwargs):
        cache = FileSystemBytecodeCache(str(tmp_path))
        make_env(cache).get_template("index.html")
        assert len(compile_counter) == 1

        make_env(cache, **kwargs).get_template("index.html")
        assert len(compile_counter) == 2
        assert len(os.listdir(tmp_path)) == 2

    def test_extension_version_misses(self, tmp_path, compile_counter):
        class VersionedExtension(Extension):
            version = "1"

        cache = FileSystemBytecodeCache(str(tmp_path))
        make_env(cache, extensions=[VersionedExtension]).get_template("index.html")
        assert len(compile_counter) == 1

        VersionedExtension.version = "2"

        try:
            make_env(cache, extensions=[VersionedExtension]).get_template("index.html")
            assert len(compile_counter) == 2
        finally:
            VersionedExtension.version = "1"

        # back to the original version, the original entry is reused
        make_env(cache, extensions=[VersionedExtension]).get_template("index.html")
        assert len(compile_counter) == 2

    def test_source_checksum_misses(self, tmp_path, compile_counter):
        cache = FileSystemBytecodeCache(str(tmp_path))
        loader = DictLoader({"index.html": "{{ 40 + 2 }}"})
        env1 = Environment(loader=loader, bytecode_cache=cache)
        assert env1.get_template("index.html").render() == "42"

        loader.mapping["index.html"] = "{{ 41 + 2 }}"
        env2 = Environment(loader=loader, bytecode_cache=cache)
        assert env2.get_template("index.html").render() == "43"
        assert len(compile_counter) == 2

    def test_inheritance_and_include_share_rules(self, tmp_path, compile_counter):
        cache = FileSystemBytecodeCache(str(tmp_path))
        templates = {
            "base.html": "[{% block body %}{% endblock %}]",
            "child.html": "{% extends 'base.html' %}{% block body %}{{ value }}{% endblock %}",
            "inc.html": "I{{ value }}",
            "main.html": "{% include 'inc.html' %}",
        }
        env1 = Environment(
            loader=DictLoader(templates), bytecode_cache=cache, autoescape=False
        )
        assert env1.get_template("child.html").render(value="x") == "[x]"
        assert env1.get_template("main.html").render(value="y") == "Iy"
        compiled = len(compile_counter)

        # same policy: inherited and included templates are all reused
        env2 = Environment(
            loader=DictLoader(templates), bytecode_cache=cache, autoescape=False
        )
        assert env2.get_template("child.html").render(value="x") == "[x]"
        assert env2.get_template("main.html").render(value="y") == "Iy"
        assert len(compile_counter) == compiled

        # changed policy: the parent template is recompiled as well
        env3 = Environment(
            loader=DictLoader(templates), bytecode_cache=cache, autoescape=True
        )
        assert env3.get_template("child.html").render(value="x") == "[x]"
        assert len(compile_counter) > compiled

    def test_environment_key_is_stable(self):
        assert get_environment_key(Environment()) == get_environment_key(Environment())
        assert get_environment_key(Environment(autoescape=True)) != get_environment_key(
            Environment()
        )

    def test_key_contains_no_paths_or_secrets(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        env = make_env(cache, autoescape=lambda name: True)
        env.get_template("index.html")
        (name,) = os.listdir(tmp_path)
        assert str(tmp_path) not in name
        assert "<locals>" not in name


class TestCorruptCache:
    def test_truncated_cache_is_miss(self, tmp_path, compile_counter):
        cache = FileSystemBytecodeCache(str(tmp_path))
        env1 = make_env(cache)
        assert env1.get_template("index.html").render(value="a") == "a"

        # truncate the file, destroying the marshal data
        (filename,) = os.listdir(tmp_path)
        path = tmp_path / filename
        data = path.read_bytes()
        path.write_bytes(data[: len(data) - 5])

        env2 = make_env(cache)
        assert env2.get_template("index.html").render(value="a") == "a"
        assert len(compile_counter) == 2

    def test_garbage_cache_is_miss(self, tmp_path, compile_counter):
        cache = FileSystemBytecodeCache(str(tmp_path))
        env1 = make_env(cache)
        env1.get_template("index.html")

        (filename,) = os.listdir(tmp_path)
        (tmp_path / filename).write_bytes(b"this is not a cache file")

        env2 = make_env(cache)
        assert env2.get_template("index.html").render(value="a") == "a"
        assert len(compile_counter) == 2

    def test_cache_load_never_unpickles(self):
        executed = []

        class Evil:
            def __reduce__(self):
                return (executed.append, (True,))

        # valid new-style magic, but the checksum region is a pickle
        # that would run code if the loader used pickle
        bucket = Bucket(None, "key", "checksum")
        bucket.bytecode_from_string(bc_magic + pickle.dumps(Evil()))
        assert bucket.code is None
        assert executed == []

    def test_old_format_is_miss_without_unpickling(self):
        executed = []

        class Evil:
            def __reduce__(self):
                return (executed.append, (True,))

        # a file in the previous (version 5) format, with a hostile
        # pickle where the checksum used to be
        old_magic = (
            b"j2"
            + pickle.dumps(5, 2)
            + pickle.dumps((sys.version_info[0] << 24) | sys.version_info[1], 2)
        )
        bucket = Bucket(None, "key", "checksum")
        bucket.bytecode_from_string(old_magic + pickle.dumps(Evil()))
        assert bucket.code is None
        assert executed == []

    def test_concurrent_writes_leave_valid_files(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        code = compile("pass", "<template>", "exec")

        def write(i):
            bucket = Bucket(None, f"key-{i % 4}", "checksum")
            bucket.code = code
            cache.dump_bytecode(bucket)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(64)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        filenames = os.listdir(tmp_path)
        assert len(filenames) == 4

        for filename in filenames:
            assert not filename.endswith(".tmp")
            bucket = Bucket(None, "key", "checksum")
            bucket.bytecode_from_string((tmp_path / filename).read_bytes())
            assert bucket.code is not None

    def test_clear_only_clears_own_namespace(self, tmp_path):
        cache_a = FileSystemBytecodeCache(str(tmp_path), pattern="a_%s.cache")
        cache_b = FileSystemBytecodeCache(str(tmp_path), pattern="b_%s.cache")

        for cache in (cache_a, cache_b):
            bucket = Bucket(None, "key", "checksum")
            bucket.code = compile("pass", "<template>", "exec")
            cache.dump_bytecode(bucket)

        assert len(os.listdir(tmp_path)) == 2
        cache_a.clear()
        (remaining,) = os.listdir(tmp_path)
        assert remaining.startswith("b_")


class TestMemcachedEnvironmentKey:
    def test_key_includes_environment(self):
        memcached = MockMemcached()
        cache = MemcachedBytecodeCache(memcached)
        env1 = make_env(cache)
        env1.get_template("index.html")
        key1 = memcached.key

        env2 = make_env(cache, autoescape=True)
        env2.get_template("index.html")
        assert memcached.key != key1


class TestModuleLoaderEnvironmentKey:
    def test_module_loader_rejects_stale_environment(self, tmp_path):
        source_loader = DictLoader({"index.html": "{{ value }}"})
        compile_env = Environment(loader=source_loader, autoescape=True)
        compile_env.compile_templates(str(tmp_path), zip=None)

        module_loader = ModuleLoader(str(tmp_path))
        ok_env = Environment(loader=module_loader, autoescape=True)
        assert ok_env.get_template("index.html").render(value="<") == "&lt;"

        # compiled with autoescape on, so it must not be used with it off
        stale_env = Environment(loader=module_loader, autoescape=False)
        pytest.raises(TemplateNotFound, stale_env.get_template, "index.html")

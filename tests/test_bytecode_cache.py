import os
import pickle
import sys
import threading

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.bccache import bc_magic
from jinja2.bccache import Bucket
from jinja2.bccache import FileSystemBytecodeCache
from jinja2.bccache import MemcachedBytecodeCache
from jinja2.exceptions import TemplateNotFound
from jinja2.ext import Extension
from jinja2.loaders import ModuleLoader
from jinja2.runtime import DebugUndefined


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


class CountingFileSystemBytecodeCache(FileSystemBytecodeCache):
    """A dump only happens after a cache miss, so counting dumps
    tracks how often a template had to be recompiled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dump_count = 0

    def dump_bytecode(self, bucket):
        self.dump_count += 1
        super().dump_bytecode(bucket)


def make_env(cache, source, **kwargs):
    kwargs.setdefault("cache_size", 0)
    return Environment(
        loader=DictLoader({"index": source}),
        bytecode_cache=cache,
        **kwargs,
    )


class VersionedExtension(Extension):
    version = 1


class TestEnvironmentBoundCache:
    def test_autoescape_does_not_share_bytecode(self, tmp_path):
        cache = CountingFileSystemBytecodeCache(str(tmp_path))
        env_plain = make_env(cache, "{{ value }}", autoescape=False)
        env_escape = make_env(cache, "{{ value }}", autoescape=True)

        assert env_plain.get_template("index").render(value="<b>") == "<b>"
        assert cache.dump_count == 1

        # the escaped environment must not reuse the plain bytecode
        assert env_escape.get_template("index").render(value="<b>") == "&lt;b&gt;"
        assert cache.dump_count == 2

        # alternating between both environments now hits the cache
        for _ in range(2):
            assert env_plain.get_template("index").render(value="<i>") == "<i>"
            assert env_escape.get_template("index").render(value="<i>") == "&lt;i&gt;"

        assert cache.dump_count == 2

    def test_optimized_option_invalidates(self, tmp_path):
        cache = CountingFileSystemBytecodeCache(str(tmp_path))
        env_opt = make_env(cache, "{{ 1 + 1 }}", optimized=True)
        env_no_opt = make_env(cache, "{{ 1 + 1 }}", optimized=False)

        assert env_opt.get_template("index").render() == "2"
        assert cache.dump_count == 1
        assert env_no_opt.get_template("index").render() == "2"
        assert cache.dump_count == 2
        assert env_opt.get_template("index").render() == "2"
        assert cache.dump_count == 2

    def test_undefined_option_invalidates(self, tmp_path):
        cache = CountingFileSystemBytecodeCache(str(tmp_path))
        env_default = make_env(cache, "{{ missing }}")
        env_debug = make_env(cache, "{{ missing }}", undefined=DebugUndefined)

        assert env_default.get_template("index").render() == ""
        assert env_debug.get_template("index").render() == "{{ missing }}"
        assert cache.dump_count == 2

    def test_extension_set_invalidates(self, tmp_path):
        cache = CountingFileSystemBytecodeCache(str(tmp_path))
        env_plain = make_env(cache, "{{ 42 }}")
        env_ext = make_env(cache, "{{ 42 }}", extensions=[VersionedExtension])

        assert env_plain.get_template("index").render() == "42"
        assert env_ext.get_template("index").render() == "42"
        assert cache.dump_count == 2

    def test_extension_version_invalidates(self, tmp_path):
        cache = CountingFileSystemBytecodeCache(str(tmp_path))
        env_v1 = make_env(cache, "{{ 42 }}", extensions=[VersionedExtension])
        assert env_v1.get_template("index").render() == "42"
        assert cache.dump_count == 1

        VersionedExtension.version = 2

        try:
            env_v2 = make_env(cache, "{{ 42 }}", extensions=[VersionedExtension])
            assert env_v2.get_template("index").render() == "42"
            assert cache.dump_count == 2
            # unchanged version hits the cache again
            assert env_v2.get_template("index").render() == "42"
            assert cache.dump_count == 2
        finally:
            VersionedExtension.version = 1

    def test_source_checksum_invalidates(self, tmp_path):
        cache = CountingFileSystemBytecodeCache(str(tmp_path))
        loader = DictLoader({"index": "old"})
        env = Environment(loader=loader, bytecode_cache=cache, cache_size=0)

        assert env.get_template("index").render() == "old"
        assert cache.dump_count == 1

        loader.mapping["index"] = "new"
        assert env.get_template("index").render() == "new"
        assert cache.dump_count == 2

        loader.mapping["index"] = "old"
        assert env.get_template("index").render() == "old"
        assert cache.dump_count == 3

    def test_extends_and_include_share_invalidation(self, tmp_path):
        cache = CountingFileSystemBytecodeCache(str(tmp_path))
        templates = {
            "base": "{{ value }}",
            "child": "{% extends 'base' %}",
            "inc": "{% include 'base' %}",
        }
        env_plain = Environment(
            loader=DictLoader(templates),
            bytecode_cache=cache,
            cache_size=0,
            autoescape=False,
        )
        env_escape = Environment(
            loader=DictLoader(templates),
            bytecode_cache=cache,
            cache_size=0,
            autoescape=True,
        )

        assert env_plain.get_template("child").render(value="<") == "<"
        assert env_plain.get_template("inc").render(value="<") == "<"
        dumps = cache.dump_count

        assert env_escape.get_template("child").render(value="<") == "&lt;"
        assert env_escape.get_template("inc").render(value="<") == "&lt;"
        assert cache.dump_count > dumps

    def test_cache_key_contains_no_paths_or_secrets(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        env = make_env(cache, "x")
        secret_path = str(tmp_path / "secret-template.html")
        bucket = cache.get_bucket(env, "index", secret_path, "x")

        assert "secret" not in bucket.key
        assert tmp_path.name not in bucket.key
        assert all(c in "0123456789abcdef_" for c in bucket.key)

    def test_bucket_key_binds_environment(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        env_plain = make_env(cache, "x", autoescape=False)
        env_escape = make_env(cache, "x", autoescape=True)
        env_plain_2 = make_env(cache, "x", autoescape=False)

        bucket_plain = cache.get_bucket(env_plain, "index", None, "x")
        bucket_escape = cache.get_bucket(env_escape, "index", None, "x")
        bucket_plain_2 = cache.get_bucket(env_plain_2, "index", None, "x")

        assert bucket_plain.key != bucket_escape.key
        assert bucket_plain.checksum != bucket_escape.checksum
        assert bucket_plain.key == bucket_plain_2.key
        assert bucket_plain.checksum == bucket_plain_2.checksum


class TestCorruptCacheHandling:
    def test_garbage_cache_file_is_miss(self, tmp_path):
        cache = CountingFileSystemBytecodeCache(str(tmp_path))
        env = make_env(cache, "working")
        assert env.get_template("index").render() == "working"
        assert cache.dump_count == 1

        (path,) = (p for p in tmp_path.iterdir() if p.is_file())
        path.write_bytes(b"garbage, not a valid cache file")

        assert env.get_template("index").render() == "working"
        assert cache.dump_count == 2
        # the cache was rewritten and hits again
        assert env.get_template("index").render() == "working"
        assert cache.dump_count == 2

    def test_truncated_marshal_is_miss(self, tmp_path):
        cache = CountingFileSystemBytecodeCache(str(tmp_path))
        env = make_env(cache, "still works")
        assert env.get_template("index").render() == "still works"

        (path,) = (p for p in tmp_path.iterdir() if p.is_file())
        data = path.read_bytes()
        # keep the valid header and checksum, truncate the marshal payload
        path.write_bytes(data[: len(bc_magic) + 2 + 40 + 2])

        assert env.get_template("index").render() == "still works"
        assert cache.dump_count == 2

    def test_cache_file_is_never_unpickled(self, tmp_path):
        cache = CountingFileSystemBytecodeCache(str(tmp_path))
        env = make_env(cache, "safe")
        assert env.get_template("index").render() == "safe"

        (path,) = (p for p in tmp_path.iterdir() if p.is_file())
        marker = tmp_path / "unpickled"

        class Evil:
            def __reduce__(self):
                return (os.system, (f"touch {marker}",))

        # a valid header followed by a malicious pickle payload must be
        # treated as a cache miss, the payload must not be executed
        path.write_bytes(bc_magic + pickle.dumps(Evil()))

        assert env.get_template("index").render() == "safe"
        assert not marker.exists()

    def test_old_pickle_format_is_miss(self, tmp_path):
        cache = CountingFileSystemBytecodeCache(str(tmp_path))
        env = make_env(cache, "v6")
        assert env.get_template("index").render() == "v6"

        (path,) = (p for p in tmp_path.iterdir() if p.is_file())
        # the pre-v6 format stored the checksum as a pickle, such files
        # must be rejected by the magic header without unpickling
        old_magic = (
            b"j2"
            + pickle.dumps(5, 2)
            + pickle.dumps((sys.version_info[0] << 24) | sys.version_info[1], 2)
        )
        path.write_bytes(old_magic + pickle.dumps("checksum"))

        assert env.get_template("index").render() == "v6"
        assert cache.dump_count == 2

    def test_concurrent_writes_leave_valid_cache(self, tmp_path):
        cache = FileSystemBytecodeCache(str(tmp_path))
        env = make_env(cache, "{{ 40 + 2 }}")
        code = env.compile("{{ 40 + 2 }}", "index", None)

        def write_bucket():
            bucket = cache.get_bucket(env, "index", None, "{{ 40 + 2 }}")
            bucket.code = code
            cache.set_bucket(bucket)

        threads = [threading.Thread(target=write_bucket) for _ in range(16)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # the cache must contain one complete entry, never a truncated one
        bucket = cache.get_bucket(env, "index", None, "{{ 40 + 2 }}")
        assert bucket.code is not None


class TestClearNamespace:
    def test_clear_only_clears_own_pattern(self, tmp_path):
        cache_a = CountingFileSystemBytecodeCache(str(tmp_path), "__a_%s.cache")
        cache_b = CountingFileSystemBytecodeCache(str(tmp_path), "__b_%s.cache")
        env_a = make_env(cache_a, "a")
        env_b = make_env(cache_b, "b")

        env_a.get_template("index")
        env_b.get_template("index")
        assert cache_a.dump_count == 1
        assert cache_b.dump_count == 1

        cache_a.clear()

        # the other namespace still hits its cache
        env_b.get_template("index")
        assert cache_b.dump_count == 1
        # the cleared namespace has to recompile
        env_a.get_template("index")
        assert cache_a.dump_count == 2


class TestModuleLoaderChecksum:
    def test_module_loader_validates_environment(self, tmp_path):
        target = str(tmp_path / "modules")
        src_env = Environment(
            loader=DictLoader({"index": "{{ value }}"}), autoescape=False
        )
        src_env.compile_templates(target, zip=None)

        ok_env = Environment(loader=ModuleLoader(target), autoescape=False)
        assert ok_env.get_template("index").render(value="<") == "<"

        other_env = Environment(loader=ModuleLoader(target), autoescape=True)

        with pytest.raises(TemplateNotFound):
            other_env.get_template("index")

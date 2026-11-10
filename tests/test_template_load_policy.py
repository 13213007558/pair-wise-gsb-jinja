import asyncio
import threading

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import get_render_identity
from jinja2 import render_identity
from jinja2 import TemplateAccessDenied

TEMPLATES = {
    "public/page": (
        "PAGE[{% include 'public/footer' %}][{% include 'internal/secret' %}]"
    ),
    "public/footer": "FOOTER",
    "public/only-footer": "ONLY[{% include 'public/footer' %}]",
    "public/dynamic": "DYN[{% include name %}]",
    "public/candidates": "CAND[{% include ['internal/secret', 'public/footer'] %}]",
    "public/ignore-missing": "IGN[{% include 'internal/secret' ignore missing %}]",
    "public/ignore-really-missing": "IGN[{% include 'public/nope' ignore missing %}]",
    "public/child": "{% extends 'internal/base' %}{% block body %}CHILD{% endblock %}",
    "public/importer": "{% import 'internal/macros' as m %}{{ m.hello() }}",
    "internal/base": "BASE[{% block body %}{% endblock %}]",
    "internal/secret": "INTERNAL-SECRET",
    "internal/macros": "{% macro hello() %}MACRO-SECRET{% endmacro %}",
}


def domain_policy(identity, parent, name, purpose):
    """Two permission domains: 'admin' may load everything, any other
    identity may only load templates outside the 'internal/' domain.
    """
    if identity == "admin":
        return True

    return name is None or not name.startswith("internal/")


@pytest.fixture
def env():
    return Environment(loader=DictLoader(TEMPLATES), template_load_policy=domain_policy)


def test_denied_after_cache_warm_and_no_partial_secret_output(env):
    # Warm the cache of the whole dependency graph with a high-privilege
    # identity. The low-privilege render below hits the same cached
    # template objects and must still be authorized.
    with render_identity("admin"):
        assert (
            env.get_template("public/page").render() == "PAGE[FOOTER][INTERNAL-SECRET]"
        )

    chunks = []

    with pytest.raises(TemplateAccessDenied) as exc_info:
        with render_identity("user"):
            for chunk in env.get_template("public/page").generate():
                chunks.append(chunk)

    # The denial happened when loading the restricted template, before
    # it executed and before any of its content was output.
    assert "INTERNAL-SECRET" not in "".join(chunks)
    assert exc_info.value.name == "internal/secret"
    assert exc_info.value.parent == "public/page"
    assert exc_info.value.purpose == "include"


def test_denied_direct_load_of_cached_template(env):
    with render_identity("admin"):
        assert env.get_template("internal/secret").render() == "INTERNAL-SECRET"

    with render_identity("user"):
        with pytest.raises(TemplateAccessDenied):
            env.get_template("internal/secret")


def test_shared_public_fragment_reusable_from_cache(env):
    with render_identity("admin"):
        cached = env.get_template("public/footer")
        env.get_template("public/page").render()

    with render_identity("user"):
        # The cached object is reused, the policy allows the public domain.
        assert env.get_template("public/footer") is cached
        assert env.get_template("public/only-footer").render() == "ONLY[FOOTER]"


def test_denied_candidate_does_not_fall_back(env):
    with render_identity("admin"):
        assert env.get_template("public/candidates").render() == "CAND[INTERNAL-SECRET]"

    # A denied candidate is not "missing": the allowed fallback must not
    # be tried.
    with render_identity("user"):
        with pytest.raises(TemplateAccessDenied):
            env.get_template("public/candidates").render()


def test_missing_candidate_still_falls_back(env):
    with render_identity("user"):
        tmpl = env.select_template(["public/nope", "public/footer"])

    assert tmpl.render() == "FOOTER"


def test_ignore_missing_does_not_swallow_denial(env):
    with render_identity("user"):
        with pytest.raises(TemplateAccessDenied):
            env.get_template("public/ignore-missing").render()

        # A genuinely missing template is still ignored.
        assert env.get_template("public/ignore-really-missing").render() == "IGN[]"


def test_dynamic_name_denied(env):
    with render_identity("user"):
        with pytest.raises(TemplateAccessDenied):
            env.get_template("public/dynamic").render(name="internal/secret")

        assert (
            env.get_template("public/dynamic").render(name="public/footer")
            == "DYN[FOOTER]"
        )


def test_dynamic_template_object_denied(env):
    with render_identity("admin"):
        secret = env.get_template("internal/secret")

    with render_identity("user"):
        with pytest.raises(TemplateAccessDenied):
            env.get_template("public/dynamic").render(name=secret)


def test_extends_denied(env):
    with render_identity("user"):
        with pytest.raises(TemplateAccessDenied) as exc_info:
            env.get_template("public/child").render()

    assert exc_info.value.purpose == "extends"

    with render_identity("admin"):
        assert env.get_template("public/child").render() == "BASE[CHILD]"


def test_import_denied(env):
    with render_identity("user"):
        with pytest.raises(TemplateAccessDenied) as exc_info:
            env.get_template("public/importer").render()

    assert exc_info.value.purpose == "import"

    with render_identity("admin"):
        assert env.get_template("public/importer").render() == "MACRO-SECRET"


def test_existing_template_object_still_authorized(env):
    with render_identity("admin"):
        tmpl = env.get_template("internal/secret")

    with render_identity("user"):
        with pytest.raises(TemplateAccessDenied):
            env.get_template(tmpl)

        with pytest.raises(TemplateAccessDenied):
            env.select_template([tmpl])

    with render_identity("admin"):
        assert env.get_template(tmpl) is tmpl


def test_policy_receives_identity_parent_name_purpose():
    calls = []

    def policy(identity, parent, name, purpose):
        calls.append((identity, parent, name, purpose))
        return True

    env = Environment(loader=DictLoader(TEMPLATES), template_load_policy=policy)

    with render_identity("admin"):
        env.get_template("public/page").render()

    assert ("admin", None, "public/page", "load") in calls
    assert ("admin", "public/page", "public/footer", "include") in calls
    assert ("admin", "public/page", "internal/secret", "include") in calls


def test_no_policy_allows_everything():
    env = Environment(loader=DictLoader(TEMPLATES))
    assert env.get_template("public/page").render() == "PAGE[FOOTER][INTERNAL-SECRET]"


def test_identity_scoped_and_reset():
    assert get_render_identity() is None

    with render_identity("admin"):
        assert get_render_identity() == "admin"

        with render_identity("user"):
            assert get_render_identity() == "user"

        assert get_render_identity() == "admin"

    assert get_render_identity() is None


def test_identity_isolated_between_threads():
    seen = []

    def check():
        seen.append(get_render_identity())

    with render_identity("admin"):
        thread = threading.Thread(target=check)
        thread.start()
        thread.join()
        assert get_render_identity() == "admin"

    assert seen == [None]


def test_async_render_respects_identity(run_async_fn):
    env = Environment(
        loader=DictLoader(TEMPLATES),
        template_load_policy=domain_policy,
        enable_async=True,
    )

    async def render(identity):
        with render_identity(identity):
            return await env.get_template("public/page").render_async()

    assert run_async_fn(render, "admin") == "PAGE[FOOTER][INTERNAL-SECRET]"

    with pytest.raises(TemplateAccessDenied):
        run_async_fn(render, "user")


def test_async_import_denied(run_async_fn):
    env = Environment(
        loader=DictLoader(TEMPLATES),
        template_load_policy=domain_policy,
        enable_async=True,
    )

    async def render():
        with render_identity("user"):
            return await env.get_template("public/importer").render_async()

    with pytest.raises(TemplateAccessDenied) as exc_info:
        run_async_fn(render)

    assert exc_info.value.purpose == "import"


def test_async_concurrent_renders_keep_own_identity():
    env = Environment(
        loader=DictLoader(TEMPLATES),
        template_load_policy=domain_policy,
        enable_async=True,
    )

    async def main():
        gate = asyncio.Event()

        async def render_as(identity):
            with render_identity(identity):
                gate.set()
                await asyncio.sleep(0.01)
                return await env.get_template("public/only-footer").render_async()

        async def render_denied():
            with render_identity("user"):
                await gate.wait()
                return await env.get_template("public/page").render_async()

        return await asyncio.gather(
            render_as("admin"), render_denied(), return_exceptions=True
        )

    ok, denied = asyncio.run(main())
    assert ok == "ONLY[FOOTER]"
    assert isinstance(denied, TemplateAccessDenied)

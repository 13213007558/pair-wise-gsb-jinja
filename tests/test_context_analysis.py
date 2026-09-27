import pytest

from jinja2 import Environment
from jinja2.context_analysis import analyze_context_requirements
from jinja2.context_analysis import DynamicAccess
from jinja2.context_analysis import UnresolvedTemplate
from jinja2.context_analysis import VariableAccess


@pytest.fixture
def env():
    return Environment()


def analyze(env, source, templates=None, **kwargs):
    return analyze_context_requirements(env.parse(source), templates, **kwargs)


def test_required_dotted_path(env):
    req = analyze(env, "{{ user.id }} {{ user.address.city }}")
    assert req.required_paths == {"user.id", "user.address.city"}
    assert req.protected == []
    assert req.dynamic == []


def test_required_paths_have_linenos(env):
    req = analyze(env, "{{ user.id }}\n{{ user.name }}")
    assert req.required == [
        VariableAccess("user.id", 1),
        VariableAccess("user.name", 2),
    ]


def test_local_names_are_not_required(env):
    req = analyze(
        env,
        "{% set user = get_user() %}{{ user.id }}"
        "{% for item in items %}{{ item.name }} {{ loop.index }}{% endfor %}"
        "{% with title = 'x' %}{{ title }}{% endwith %}",
    )
    assert req.required_paths == {"get_user", "items"}


def test_macro_params_shadow_context(env):
    req = analyze(
        env,
        "{% macro render(user, prefix='id') %}{{ user.id }} {{ prefix }}{% endmacro %}"
        "{% call(row) dump() %}{{ row.id }}{% endcall %}",
    )
    assert req.required_paths == {"dump"}


def test_import_defines_names(env):
    req = analyze(
        env,
        "{% import 'forms.html' as forms %}{{ forms.input('x') }}"
        "{% from 'macros.html' import field as f %}{{ f() }}",
    )
    assert req.required == []
    # Imports without "with context" are self-contained.
    assert req.unresolved_templates == []


def test_is_defined_protects_branch(env):
    req = analyze(
        env,
        "{% if user.nickname is defined %}{{ user.nickname }}{% endif %}"
        "{{ user.id }}",
    )
    assert req.required_paths == {"user.id"}
    assert req.protected_paths == {"user.nickname"}


def test_default_filter_protects(env):
    req = analyze(env, "{{ user.nickname|default('anon') }} {{ user.id }}")
    assert req.required_paths == {"user.id"}
    assert req.protected_paths == {"user.nickname"}


def test_short_circuit_and_or(env):
    req = analyze(
        env,
        "{{ user is defined and user.name }}"
        "{% if a is undefined or b is undefined %}none"
        "{% else %}{{ a }} {{ b }}{% endif %}",
    )
    assert req.required == []
    assert req.protected_paths == {"user", "user.name", "a", "b"}


def test_if_else_branches_tracked_separately(env):
    req = analyze(
        env,
        "{% if user is defined %}{{ user.id }}{% else %}{{ guest_name }}{% endif %}",
    )
    assert req.required_paths == {"guest_name"}
    assert req.protected_paths == {"user", "user.id"}


def test_elif_branches(env):
    req = analyze(
        env,
        "{% if a is defined %}{{ a }}"
        "{% elif b is defined %}{{ b }}"
        "{% else %}{{ c }}{% endif %}",
    )
    assert req.protected_paths == {"a", "b"}
    assert req.required_paths == {"c"}


def test_dynamic_getitem_reported_separately(env):
    req = analyze(env, "{{ user[key] }}")
    assert req.required_paths == {"user", "key"}
    assert req.dynamic == [DynamicAccess("user", 1)]


def test_conditional_expression(env):
    req = analyze(env, "{{ user.name if user is defined else 'anon' }}")
    assert req.required == []
    assert req.protected_paths == {"user", "user.name"}


def test_include_followed_when_ast_provided(env):
    frag = env.parse("{{ shared.value }}")
    req = analyze(env, "{% include 'frag.html' %}", {"frag.html": frag})
    assert req.required_paths == {"shared.value"}
    assert req.unresolved_templates == []


def test_static_ref_without_ast_is_unresolved(env):
    req = analyze(env, "{% extends 'base.html' %}")
    assert req.unresolved_templates == [UnresolvedTemplate("base.html", 1, "extends")]


def test_dynamic_template_name_is_unresolved(env):
    req = analyze(env, "{% include some_name %}{% include ['a.html', dyn] %}")
    assert req.required_paths == {"some_name", "dyn"}
    assert req.unresolved_templates == [
        UnresolvedTemplate(None, 1, "include"),
        UnresolvedTemplate("a.html", 1, "include"),
        UnresolvedTemplate(None, 1, "include"),
    ]


def test_cyclic_includes_do_not_recurse_forever(env):
    a = env.parse("{% include 'b.html' %}{{ av }}")
    b = env.parse("{% include 'a.html' %}{{ bv }}")
    req = analyze_context_requirements(a, {"a.html": a, "b.html": b})
    assert req.required_paths == {"av", "bv"}


def test_full_scenario(env):
    source = (
        "{% for item in items %}{{ item.name }}{% endfor %}\n"
        "{% macro show(user) %}{{ user.id }}{% endmacro %}\n"
        "{% if user.nickname is defined %}{{ user.nickname }}{% endif %}\n"
        "{{ user.id }}\n"
    )
    req = analyze(env, source)
    assert req.required_paths == {"items", "user.id"}
    assert req.protected_paths == {"user.nickname"}
    assert req.dynamic == []
    nick = req.protected[0]
    assert nick == VariableAccess("user.nickname", 3)


def test_ast_and_environment_unchanged(env):
    env.globals["helper"] = lambda: None
    source = (
        "{% for item in items %}{{ item }}{% endfor %}"
        "{% if user.nickname is defined %}{{ user.nickname }}{% endif %}"
        "{{ user.id }}{{ user[key] }}"
    )
    ast = env.parse(source)
    dump_before = ast.dump()
    globals_before = dict(env.globals)

    analyze_context_requirements(ast, globals=env.globals)

    assert ast.dump() == dump_before
    assert env.globals == globals_before

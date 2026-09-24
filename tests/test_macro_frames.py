"""Tests for macro and call-block frame isolation.

Every macro or call block invocation must establish an independent frame:
its own parameters, local variables, caller, and loop context. When a
(recursive) call returns, the parent frame must be restored unchanged.
"""

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import UndefinedError


class Node:
    def __init__(self, name, perm, children=()):
        self.name = name
        self.perm = perm
        self.children = children


def make_tree():
    return Node("a", "ra", [Node("b", "rb", [Node("c", "rc")]), Node("d", "rd")])


RECURSIVE_TREE = """\
{%- macro tree(node) -%}
{{ node.name }}:{{ caller(node.perm) }}
{%- for child in node.children -%}
{% call(p) tree(child) -%}[{{ p }}|{{ record(node.name, loop.index) }}]{%- endcall -%}
{%- endfor -%}
{%- endmacro -%}
{% call(p) tree(root) %}[{{ p }}]{% endcall %}
"""


class TestRecursiveMacroFrames:
    def test_three_level_recursion(self, env):
        """Each recursive level sees its own caller and loop context."""
        records = []

        def record(name, index):
            records.append((name, index))
            return f"{name}#{index}"

        t = env.from_string(RECURSIVE_TREE)
        out = t.render(root=make_tree(), record=record)
        assert out == "a:[ra]b:[rb|a#1]c:[rc|b#1]d:[rd|a#2]"
        # the caller body of each level closes over the loop of the frame
        # where the call block was written, not any other frame
        assert records == [("a", 1), ("b", 1), ("a", 2)]

    def test_recursion_restores_parent_frame(self, env):
        """After a recursive call returns, the parent's loop continues."""
        t = env.from_string(
            "{% macro tree(node) %}"
            "{% for child in node.children %}"
            "{{ tree(child) }}({{ child.name }}@{{ loop.index }})"
            "{% endfor %}{{ node.name }}"
            "{% endmacro %}"
            "{{ tree(root) }}"
        )
        out = t.render(root=make_tree())
        # innermost first: c, then b resumes and finishes, then d, then a
        assert out == "c(c@1)b(b@1)d(d@2)a"

    def test_nested_call_blocks(self, env):
        t = env.from_string(
            "{% macro outer() %}[{{ caller() }}]{% endmacro %}"
            "{% macro inner() %}<{{ caller() }}>{% endmacro %}"
            "{% call outer() %}{% call inner() %}X{% endcall %}{% endcall %}"
        )
        assert t.render() == "[<X>]"

    def test_nested_call_same_name_params(self, env):
        """An inner caller parameter shadows the outer one only inside the
        inner call block; the outer frame is restored afterwards."""
        t = env.from_string(
            "{% macro show(v) %}{{ caller(v) }}{% endmacro %}"
            "{% call(x) show('a') %}"
            "{% call(x) show('b') %}{{ x }}{% endcall %}{{ x }}"
            "{% endcall %}"
        )
        assert t.render() == "ba"

    def test_same_name_variables(self, env):
        """Macro parameters shadow globals; macro locals don't leak out."""
        t = env.from_string(
            "{% set x = 'global' %}"
            "{% macro m(x) %}{% set y = x + '-inner' %}{{ x }}:{{ y }}{% endmacro %}"
            "{{ m('param') }}|{{ x }}|{{ y is defined }}"
        )
        assert t.render() == "param:param-inner|global|False"

    def test_exception_unwinds_stack(self, env):
        """An exception deep in recursion unwinds all frames; the same
        environment renders fine afterwards."""
        records = []

        def record(name):
            records.append(name)
            return ""

        t = env.from_string(
            "{% macro tree(node) %}"
            "{{ record(node.name) }}"
            "{% if node.name == 'c' %}{{ 1 / 0 }}{% endif %}"
            "{% for child in node.children %}{{ tree(child) }}{% endfor %}"
            "{% endmacro %}"
            "{{ tree(root) }}"
        )

        with pytest.raises(ZeroDivisionError):
            t.render(root=make_tree(), record=record)

        # frames up to the failure ran, nothing after
        assert records == ["a", "b", "c"]
        # the environment and template are not corrupted by the unwind
        ok = env.from_string("{% macro m(v) %}{{ v }}{% endmacro %}{{ m(1) }}")
        assert ok.render() == "1"
        assert records == ["a", "b", "c"]

    def test_non_recursive_macro_unchanged(self, env):
        t = env.from_string(
            "{% macro m(a, b=2) %}{{ a + b }}{% endmacro %}"
            "{{ m(1) }}|{{ m(1, 5) }}|{{ m(a=3) }}"
        )
        assert t.render() == "3|6|5"


class TestCallerIsolation:
    def test_each_call_gets_own_caller(self, env):
        """Two calls to the same macro never share a caller object."""
        seen = []

        def record(value):
            seen.append(value)
            return ""

        t = env.from_string(
            "{% macro m() %}{{ record(caller) }}{{ caller() }}{% endmacro %}"
            "{% call m() %}A{% endcall %}"
            "{% call m() %}B{% endcall %}"
        )
        assert t.render(record=record) == "AB"
        assert len(seen) == 2
        assert len({id(x) for x in seen}) == 2

    def test_caller_not_leaked_into_kwargs(self, env):
        """A macro that catches **kwargs but does not declare a caller
        must not receive the caller through kwargs."""
        t = env.from_string(
            "{% macro m() %}{{ kwargs }}{% endmacro %}"
            "{% call m() %}body{% endcall %}"
        )

        with pytest.raises(TypeError, match="caller"):
            t.render()

    def test_recursive_kwargs_forward_sees_no_parent_caller(self, env):
        """Forwarding **kwargs down a recursion must not hand the parent's
        caller (and the parent frame it closes over) to child calls."""
        t = env.from_string(
            "{% macro tree(node) %}"
            "{{ node.name }}:{{ caller(node.perm) }}"
            "{% for child in node.children %}{{ tree(child, **kwargs) }}{% endfor %}"
            "{% endmacro %}"
            "{% call(p) tree(root) %}[{{ p }}]{% endcall %}"
        )

        # the recursive call gets a fresh "No caller defined" undefined,
        # not the caller of the parent frame
        with pytest.raises(UndefinedError, match="No caller defined"):
            t.render(root=make_tree())

    def test_caller_body_sees_definition_frame(self, env):
        """The caller body is scoped where the call block was written."""
        t = env.from_string(
            "{% macro m() %}{{ caller() }}{% endmacro %}"
            "{% for i in [1, 2, 3] %}{% call m() %}{{ i }}{% endcall %}{% endfor %}"
        )
        assert t.render() == "123"


class TestDefaultsAndParameters:
    def test_defaults_evaluated_per_call(self, env):
        """Default expressions are evaluated at call time, not cached."""
        calls = []

        def stamp():
            calls.append(1)
            return len(calls)

        t = env.from_string(
            "{% macro m(v=stamp()) %}{{ v }}{% endmacro %}{{ m() }},{{ m() }}"
        )
        assert t.render(stamp=stamp) == "1,2"

    def test_mutable_default_not_shared(self, env):
        """Two calls to the same macro don't share a mutable default."""
        t = env.from_string(
            "{% macro m(acc=[]) %}{% set _ = acc.append(1) %}{{ acc | length }}"
            "{% endmacro %}{{ m() }},{{ m() }}"
        )
        assert t.render() == "1,1"

    def test_macro_as_parameter(self, env):
        """A macro passed as an argument keeps its own valid frame."""
        t = env.from_string(
            "{% macro inner(v) %}inner({{ v }}){% endmacro %}"
            "{% macro apply(fn, v) %}{{ fn(v) }}{% endmacro %}"
            "{{ apply(inner, 'x') }}"
        )
        assert t.render() == "inner(x)"


class TestScopingSemantics:
    def test_imported_macro_globals_only(self):
        """Imported macros see environment globals, not the importing
        template's local variables."""
        env = Environment(
            loader=DictLoader(
                {
                    "macros.html": "{% macro show() %}"
                    "{{ g }}:{{ v | default('none') }}{% endmacro %}"
                }
            )
        )
        env.globals["g"] = "G"
        t = env.from_string(
            "{% set v = 'local' %}{% import 'macros.html' as f %}{{ f.show() }}"
        )
        assert t.render() == "G:none"

    def test_imported_macro_with_context(self):
        env = Environment(
            loader=DictLoader(
                {"macros.html": "{% macro show() %}{{ v }}{% endmacro %}"}
            )
        )
        t = env.from_string(
            "{% set v = 'local' %}"
            "{% from 'macros.html' import show with context %}{{ show() }}"
        )
        assert t.render() == "local"

    def test_namespace_persists_across_macro_calls(self, env):
        t = env.from_string(
            "{% set ns = namespace(x=0) %}"
            "{% macro bump() %}{% set ns.x = ns.x + 1 %}{% endmacro %}"
            "{{ bump() }}{{ bump() }}{{ ns.x }}"
        )
        assert t.render() == "2"

    def test_with_scope_does_not_leak(self, env):
        t = env.from_string(
            "{% macro m() %}{{ x is defined }}{% endmacro %}"
            "{% with x = 1 %}{{ x }}{% endwith %}{{ x is defined }}{{ m() }}"
        )
        assert t.render() == "1FalseFalse"


class TestAutoescape:
    @pytest.fixture
    def env_ae(self):
        return Environment(autoescape=True)

    def test_macro_output_escaped_per_call_site(self, env_ae):
        t = env_ae.from_string(
            "{% macro m(v) %}{{ v }}{% endmacro %}{{ m('<b>') }}"
        )
        assert t.render() == "&lt;b&gt;"

    def test_caller_body_escaped(self, env_ae):
        t = env_ae.from_string(
            "{% macro m() %}{{ caller() }}{% endmacro %}"
            "{% call m() %}{{ unsafe }}{% endcall %}"
        )
        assert t.render(unsafe="<i>") == "&lt;i&gt;"

    def test_recursion_autoescape(self, env_ae):
        records = []

        def record(name, index):
            records.append((name, index))
            return f"{name}#{index}"

        t = env_ae.from_string(RECURSIVE_TREE)
        out = t.render(root=make_tree(), record=record)
        assert out == "a:[ra]b:[rb|a#1]c:[rc|b#1]d:[rd|a#2]"
        assert records == [("a", 1), ("b", 1), ("a", 2)]


class TestAsyncMacroFrames:
    @pytest.fixture
    def async_env(self):
        return Environment(enable_async=True)

    def test_async_recursion(self, async_env, run_async_fn):
        records = []

        def record(name, index):
            records.append((name, index))
            return f"{name}#{index}"

        t = async_env.from_string(RECURSIVE_TREE)

        async def go():
            return await t.render_async(root=make_tree(), record=record)

        out = run_async_fn(go)
        assert out == "a:[ra]b:[rb|a#1]c:[rc|b#1]d:[rd|a#2]"
        assert records == [("a", 1), ("b", 1), ("a", 2)]

    def test_async_nested_call(self, async_env, run_async_fn):
        t = async_env.from_string(
            "{% macro outer() %}[{{ caller() }}]{% endmacro %}"
            "{% macro inner() %}<{{ caller() }}>{% endmacro %}"
            "{% call outer() %}{% call inner() %}X{% endcall %}{% endcall %}"
        )

        async def go():
            return await t.render_async()

        assert run_async_fn(go) == "[<X>]"

    def test_async_exception_unwinds(self, async_env, run_async_fn):
        t = async_env.from_string(
            "{% macro tree(node) %}"
            "{{ node.name }}"
            "{% if node.name == 'c' %}{{ 1 / 0 }}{% endif %}"
            "{% for child in node.children %}{{ tree(child) }}{% endfor %}"
            "{% endmacro %}"
            "{{ tree(root) }}"
        )

        async def go():
            return await t.render_async(root=make_tree())

        with pytest.raises(ZeroDivisionError):
            run_async_fn(go)

        ok = async_env.from_string(
            "{% macro m(v) %}{{ v }}{% endmacro %}{{ m(1) }}"
        )

        async def go_ok():
            return await ok.render_async()

        assert run_async_fn(go_ok) == "1"

    def test_async_caller_not_leaked_into_kwargs(self, async_env, run_async_fn):
        t = async_env.from_string(
            "{% macro m() %}{{ kwargs }}{% endmacro %}"
            "{% call m() %}body{% endcall %}"
        )

        async def go():
            return await t.render_async()

        with pytest.raises(TypeError, match="caller"):
            run_async_fn(go)

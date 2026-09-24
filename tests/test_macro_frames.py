import pytest

from jinja2 import DictLoader
from jinja2 import Environment


class DirNode:
    """A node of a directory tree with a permission label."""

    def __init__(self, name, perm, children=()):
        self.name = name
        self.perm = perm
        self.children = list(children)


@pytest.fixture
def tree():
    """A three-level directory tree."""
    return DirNode(
        "root",
        "rwx",
        [
            DirNode("a", "rw-", [DirNode("a1", "r--")]),
            DirNode("b", "r-x"),
        ],
    )


class TestMacroCallFrames:
    def test_recursive_directory_tree_with_caller(self, env, tree):
        """A recursive macro renders a directory tree; the caller renders
        each node with the permission label of the current node.  Every
        level of the recursion must see its own caller and arguments.
        """
        tmpl = env.from_string(
            """\
{% macro render_dir(node) -%}
{{ caller(node.perm) }}{{ node.name }};
{%- for child in node.children -%}
{{ render_dir(child, caller=caller) }}
{%- endfor -%}
{%- endmacro -%}
{% call(p) render_dir(tree) %}[{{ p }}]{% endcall %}
"""
        )
        out = tmpl.render(tree=tree)
        assert out == "[rwx]root;[rw-]a;[r--]a1;[r-x]b;"

    def test_recursive_nested_call_blocks(self, env, tree):
        """A call block nested inside the recursive macro gives every
        level its own caller that renders that level's permission.
        """
        tmpl = env.from_string(
            """\
{% macro render_dir(node) -%}
{{ caller(node.perm) }}{{ node.name }};
{%- for child in node.children -%}
{% call(p) render_dir(child) %}[{{ p }}]
{%- endcall %}{%- endfor -%}
{%- endmacro -%}
{% call(p) render_dir(tree) %}[{{ p }}]{% endcall %}
"""
        )
        out = tmpl.render(tree=tree)
        assert out == "[rwx]root;[rw-]a;[r--]a1;[r-x]b;"

    def test_recursive_loop_context_isolated(self, env, tree):
        """Each recursion level gets its own loop context; returning from
        a recursive call restores the parent frame's loop index.
        """
        tmpl = env.from_string(
            """\
{% macro render_dir(nodes, depth=0) -%}
{% for node in nodes -%}
{{ node.name }}:d{{ depth }}:i{{ loop.index }};
{{- render_dir(node.children, depth + 1) }}
{%- endfor -%}
{%- endmacro -%}
{{ render_dir([tree]) }}
"""
        )
        out = tmpl.render(tree=tree)
        assert out == "root:d0:i1;a:d1:i1;a1:d2:i1;b:d1:i2;"

    def test_nested_call_blocks(self, env, tree):
        """Nested call blocks each bind their own caller arguments."""
        tmpl = env.from_string(
            """\
{% macro show(node) -%}
{{ caller(node, node.perm) }}
{%- endmacro -%}
{% for child in tree.children -%}
{% call(n, p) show(child) %}[{{ p }}]{{ n.name }}@{{ loop.index }};
{%- endcall -%}
{% endfor %}
"""
        )
        out = tmpl.render(tree=tree)
        assert out == "[rw-]a@1;[r-x]b@2;"

    def test_same_name_variables_restored(self, env):
        """Same-name variables in recursive frames are independent and the
        parent frame's value is restored when the recursive call returns.
        """
        tmpl = env.from_string(
            """\
{% macro rec(depth) -%}
{% set x = depth * 10 -%}
x={{ x }};
{%- if depth < 3 %}{{ rec(depth + 1) }}{% endif -%}
after:{{ x }};
{%- endmacro -%}
{{ rec(1) }}
"""
        )
        out = tmpl.render()
        assert out == "x=10;x=20;x=30;after:30;after:20;after:10;"

    def test_deferred_caller_keeps_defining_frame(self, env, tree):
        """A caller object created inside a loop and invoked after the
        loop finished must still see the values of its defining
        iteration, not the cleared loop frame.
        """
        tmpl = env.from_string(
            """\
{% macro capture() %}{% set ns.saved = caller %}{% endmacro -%}
{% set ns = namespace(saved=none) -%}
{% for child in tree.children -%}
{% if child.perm == 'rw-' -%}
{% call capture() %}[{{ child.perm }}]{{ child.name }}@{{ loop.index }}{% endcall %}
{%- endif -%}
{% endfor -%}
{{ ns.saved() }}
"""
        )
        out = tmpl.render(tree=tree)
        assert out == "[rw-]a@1"

    def test_deferred_caller_keeps_loop_snapshot(self, env):
        """The loop variable captured by a deferred caller reflects the
        defining iteration, including index and length.
        """
        tmpl = env.from_string(
            """\
{% macro capture() %}{% set ns.saved = caller %}{% endmacro -%}
{% set ns = namespace(saved=none) -%}
{% for i in [1, 2, 3] -%}
{% if i == 2 -%}
{% call capture() %}{{ loop.index }}/{{ loop.length }}{% endcall %}
{%- endif -%}
{% endfor -%}
{{ ns.saved() }}
"""
        )
        out = tmpl.render()
        assert out == "2/3"

    def test_macro_passed_as_parameter_no_dead_frame(self, env):
        """A macro defined inside a loop and passed as a parameter must
        not capture the loop frame that is already gone when it is
        finally called.
        """
        tmpl = env.from_string(
            """\
{% macro capture(m) %}{% set ns.saved = m %}{% endmacro -%}
{% set ns = namespace(saved=none) -%}
{% for i in [1, 2, 3] -%}
{% macro label() %}[perm@{{ loop.index }}]{% endmacro -%}
{% if i == 1 %}{{ capture(label) }}{% endif -%}
{% endfor -%}
{{ ns.saved() }}
"""
        )
        out = tmpl.render()
        assert out == "[perm@1]"

    def test_same_macro_twice_no_shared_mutable_default(self, env):
        """Calling the same macro twice must not share mutable default
        values between the calls.
        """
        tmpl = env.from_string(
            """\
{% macro m(x=[]) %}{{ x.append(1) or '' }}{{ x|length }}{% endmacro -%}
{{ m() }}|{{ m() }}
"""
        )
        assert tmpl.render() == "1|1"

    def test_same_macro_twice_independent_callers(self, env):
        """Two calls of the same macro get independent caller objects."""
        tmpl = env.from_string(
            """\
{% macro m() %}{{ caller() }}{% endmacro -%}
{% call m() %}first{% endcall %}|{% call m() %}second{% endcall %}
"""
        )
        assert tmpl.render() == "first|second"

    def test_exception_during_recursion_cleans_up(self, env, tree):
        """An exception that unwinds a recursive macro must not leave
        stale state behind; the template renders correctly afterwards.
        """
        tmpl = env.from_string(
            """\
{% macro render_dir(node) -%}
{% if node.perm == 'r--' %}{{ boom() }}{% endif -%}
{{ node.name }};
{%- for child in node.children -%}
{{ render_dir(child) }}
{%- endfor -%}
{%- endmacro -%}
{{ render_dir(tree) }}
"""
        )

        def boom():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            tmpl.render(tree=tree, boom=boom)

        out = tmpl.render(tree=tree, boom=lambda: "")
        assert out == "root;a;a1;b;"

    def test_macro_defaults_evaluated_at_call_time(self, env):
        """Default expressions are evaluated when the macro is called,
        not when it is defined.
        """
        tmpl = env.from_string(
            """\
{% macro m(x=items|length) %}{{ x }}{% endmacro -%}
{% set items = [1, 2] %}{{ m() }}{% set items = [1, 2, 3] %}{{ m() }}
"""
        )
        assert tmpl.render() == "23"

    def test_imported_macro_globals_read_only(self):
        """Imported macros see environment globals but not the importing
        template's local variables.
        """
        env = Environment(
            loader=DictLoader(
                {
                    "lib.html": "{% macro show() %}{{ g }}{% endmacro %}",
                    "main.html": (
                        "{% import 'lib.html' as lib %}"
                        "{% set g = 'local' %}{{ lib.show() }}"
                    ),
                }
            )
        )
        env.globals["g"] = "global"
        out = env.get_template("main.html").render()
        assert out == "global"

    def test_namespace_semantics_kept(self, env):
        """Namespace assignments inside a macro remain visible after the
        call, as documented.
        """
        tmpl = env.from_string(
            """\
{% set ns = namespace(count=0) -%}
{% macro bump() %}{% set ns.count = ns.count + 1 %}{% endmacro -%}
{{ bump() }}{{ bump() }}{{ ns.count }}
"""
        )
        assert tmpl.render() == "2"

    def test_with_scope_semantics_kept(self, env):
        """Assignments inside a with block do not leak to the outer scope."""
        tmpl = env.from_string(
            """\
{% set x = 1 -%}
{% with x = 2 %}{{ x }}{% endwith %}{{ x }}
"""
        )
        assert tmpl.render() == "21"

    def test_non_recursive_macro_unaffected(self, env):
        """Plain non-recursive macros keep working as before."""
        tmpl = env.from_string(
            """\
{% macro greet(name, punct='!') %}Hello {{ name }}{{ punct }}{% endmacro -%}
{{ greet('World') }} {{ greet('Jinja', punct='?') }}
"""
        )
        assert tmpl.render() == "Hello World! Hello Jinja?"

    def test_autoescape_per_call_site(self):
        """Whether a macro's result is escaped depends on the autoescape
        setting at the call site.
        """
        env = Environment(autoescape=True)
        tmpl = env.from_string(
            """\
{% macro m() %}<b>{% endmacro -%}
{% autoescape false %}{{ m() }}{% endautoescape %}|{{ m() }}
"""
        )
        assert tmpl.render() == "<b>|<b>"

    def test_loop_local_macro_recursion(self, env):
        """A macro defined inside a loop can still recurse through late
        binding of its own name.
        """
        tmpl = env.from_string(
            """\
{% for n in [[1, 2, 3], [4, 5]] -%}
{% macro x(l) %}{{ l.pop() }}{% if l %}{{ x(l) }}{% endif %}{% endmacro -%}
{{ x(n) }};
{%- endfor %}
"""
        )
        assert tmpl.render() == "321;54;"


class TestAsyncMacroFrames:
    def test_async_recursive_directory_tree(self, run_async_fn, tree):
        """Async macros get independent frames for recursive calls too."""
        env = Environment(enable_async=True)
        tmpl = env.from_string(
            """\
{% macro render_dir(node) -%}
{{ caller(node.perm) }}{{ node.name }};
{%- for child in node.children -%}
{{ render_dir(child, caller=caller) }}
{%- endfor -%}
{%- endmacro -%}
{% call(p) render_dir(tree) %}[{{ p }}]{% endcall %}
"""
        )

        async def func():
            return await tmpl.render_async(tree=tree)

        out = run_async_fn(func)
        assert out == "[rwx]root;[rw-]a;[r--]a1;[r-x]b;"

    def test_async_deferred_caller_keeps_defining_frame(self, run_async_fn):
        """The loop frame snapshot also applies to async templates."""
        env = Environment(enable_async=True)
        tmpl = env.from_string(
            """\
{% macro capture() %}{% set ns.saved = caller %}{% endmacro -%}
{% set ns = namespace(saved=none) -%}
{% for i in [1, 2, 3] -%}
{% if i == 2 %}{% call capture() %}[{{ i }}@{{ loop.index }}]{% endcall %}{% endif -%}
{% endfor -%}
{{ ns.saved() }}
"""
        )

        async def func():
            return await tmpl.render_async()

        out = run_async_fn(func)
        assert out == "[2@2]"

import math

import pytest
from markupsafe import Markup

from jinja2.exceptions import UndefinedError
from jinja2.nativetypes import native_concat
from jinja2.nativetypes import NativeEnvironment
from jinja2.nativetypes import NativeTemplate
from jinja2.runtime import StrictUndefined
from jinja2.runtime import Undefined


@pytest.fixture
def env():
    return NativeEnvironment()


@pytest.fixture
def async_native_env():
    return NativeEnvironment(enable_async=True)


def test_is_defined_native_return(env):
    t = env.from_string("{{ missing is defined }}")
    assert not t.render()


def test_undefined_native_return(env):
    t = env.from_string("{{ missing }}")
    assert isinstance(t.render(), Undefined)


def test_adding_undefined_native_return(env):
    t = env.from_string("{{ 3 + missing }}")

    with pytest.raises(UndefinedError):
        t.render()


def test_cast_int(env):
    t = env.from_string("{{ value|int }}")
    result = t.render(value="3")
    assert isinstance(result, int)
    assert result == 3


def test_list_add(env):
    t = env.from_string("{{ a + b }}")
    result = t.render(a=["a", "b"], b=["c", "d"])
    assert isinstance(result, list)
    assert result == ["a", "b", "c", "d"]


def test_multi_expression_add(env):
    t = env.from_string("{{ a }} + {{ b }}")
    result = t.render(a=["a", "b"], b=["c", "d"])
    assert not isinstance(result, list)
    assert result == "['a', 'b'] + ['c', 'd']"


def test_loops(env):
    t = env.from_string("{% for x in value %}{{ x }}{% endfor %}")
    result = t.render(value=["a", "b", "c", "d"])
    assert isinstance(result, str)
    assert result == "abcd"


def test_loops_with_ints(env):
    t = env.from_string("{% for x in value %}{{ x }}{% endfor %}")
    result = t.render(value=[1, 2, 3, 4])
    assert isinstance(result, int)
    assert result == 1234


def test_loop_look_alike(env):
    t = env.from_string("{% for x in value %}{{ x }}{% endfor %}")
    result = t.render(value=[1])
    assert isinstance(result, int)
    assert result == 1


@pytest.mark.parametrize(
    ("source", "expect"),
    (
        ("{{ value }}", True),
        ("{{ value }}", False),
        ("{{ 1 == 1 }}", True),
        ("{{ 2 + 2 == 5 }}", False),
        ("{{ None is none }}", True),
        ("{{ '' == None }}", False),
    ),
)
def test_booleans(env, source, expect):
    t = env.from_string(source)
    result = t.render(value=expect)
    assert isinstance(result, bool)
    assert result is expect


def test_variable_dunder(env):
    t = env.from_string("{{ x.__class__ }}")
    result = t.render(x=True)
    assert isinstance(result, type)


def test_constant_dunder(env):
    t = env.from_string("{{ true.__class__ }}")
    result = t.render()
    assert isinstance(result, type)


def test_constant_dunder_to_string(env):
    t = env.from_string("{{ true.__class__|string }}")
    result = t.render()
    assert not isinstance(result, type)
    assert result in {"<type 'bool'>", "<class 'bool'>"}


def test_string_literal_var(env):
    t = env.from_string("[{{ 'all' }}]")
    result = t.render()
    assert isinstance(result, str)
    assert result == "[all]"


def test_string_top_level(env):
    t = env.from_string("'Jinja'")
    result = t.render()
    assert result == "Jinja"


def test_string_concatenation(async_native_env, run_async_fn):
    async def async_render():
        t = async_native_env.from_string(
            "{%- macro x(y) -%}{{ y }}{%- endmacro -%}{{- x('not') }} {{ x('bad') -}}"
        )
        result = await t.render_async()
        assert isinstance(result, str)
        assert result == "not bad"

    run_async_fn(async_render)


def test_tuple_of_variable_strings(env):
    t = env.from_string("'{{ a }}', 'data', '{{ b }}', b'{{ c }}'")
    result = t.render(a=1, b=2, c="bytes")
    assert isinstance(result, tuple)
    assert result == ("1", "data", "2", b"bytes")


def test_concat_strings_with_quotes(env):
    t = env.from_string("--host='{{ host }}' --user \"{{ user }}\"")
    result = t.render(host="localhost", user="Jinja")
    assert result == "--host='localhost' --user \"Jinja\""


def test_no_intermediate_eval(env):
    t = env.from_string("0.000{{ a }}")
    result = t.render(a=7)
    assert isinstance(result, float)
    # If intermediate eval happened, 0.000 would render 0.0, then 7
    # would be appended, resulting in 0.07.
    assert math.isclose(result, 0.0007)


def test_spontaneous_env():
    t = NativeTemplate("{{ true }}")
    assert isinstance(t.environment, NativeEnvironment)


def test_leading_spaces(env):
    t = env.from_string(" {{ True }}")
    result = t.render()
    assert result == " True"


def test_macro(env):
    t = env.from_string("{%- macro x() -%}{{- [1,2] -}}{%- endmacro -%}{{- x()[1] -}}")
    result = t.render()
    assert result == 2
    assert isinstance(result, int)


def test_block(env):
    t = env.from_string(
        "{% block b %}{% for i in range(1) %}{{ loop.index }}{% endfor %}"
        "{% endblock %}{{ self.b() }}"
    )
    result = t.render()
    assert result == 11
    assert isinstance(result, int)


def test_nested_dict_list_roundtrip(env):
    value = {
        "list": [1, 2.5, -3],
        "tuple": (True, False),
        "none": None,
        "unicode": "héllo ünïcode ☃",
        "nested": {"a": [{"b": (1,)}]},
    }
    t = env.from_string("{{ value }}")
    result = t.render(value=value)
    assert result == value
    assert result is value
    assert isinstance(result["list"], list)
    assert isinstance(result["tuple"], tuple)
    assert result["none"] is None
    assert result["unicode"] == "héllo ünïcode ☃"


def test_nested_mixed_markup_and_undefined(env):
    missing = env.undefined(name="missing")
    value = {"markup": Markup("<b>"), "items": [missing, (1, None)]}
    t = env.from_string("{{ value }}")
    result = t.render(value=value)
    assert result is value
    assert isinstance(result["markup"], Markup)
    assert isinstance(result["items"][0], Undefined)


class LiteralLookalike:
    """An object whose repr looks like a Python literal."""

    def __repr__(self):
        return "[1, 2, 3]"

    __str__ = __repr__


class MaliciousRepr:
    def __repr__(self):
        return "__import__('os').getcwd()"

    __str__ = __repr__


def test_custom_object_single_node(env):
    obj = LiteralLookalike()
    assert env.from_string("{{ obj }}").render(obj=obj) is obj
    result = env.from_string("{{ [obj] }}").render(obj=obj)
    assert isinstance(result, list)
    assert result[0] is obj


def test_custom_object_multi_node_is_string(env):
    result = env.from_string("{{ obj }}!").render(obj=LiteralLookalike())
    assert isinstance(result, str)
    assert result == "[1, 2, 3]!"


def test_custom_object_repr_not_evaluated(env):
    result = env.from_string("{{ obj }}").render(obj=MaliciousRepr())
    assert isinstance(result, MaliciousRepr)
    result = env.from_string("{{ obj }} x").render(obj=MaliciousRepr())
    assert isinstance(result, str)
    assert result == "__import__('os').getcwd() x"


@pytest.mark.parametrize(
    "source",
    (
        "__import__('os').getcwd()",
        "os.system('echo pwned')",
        "f'{1 + 1}'",
        "().__class__.__bases__[0].__subclasses__()",
        "[1, 2",
        "{'a':",
        "(" * 1000,
        "[" * 1000,
    ),
)
def test_malicious_and_unclosed_literals_stay_strings(env, source):
    result = env.from_string("{{ value }}").render(value=source)
    assert isinstance(result, str)
    assert result == source


def test_malicious_literal_not_executed(env, tmp_path):
    marker = tmp_path / "pwned"
    source = f"__import__('pathlib').Path({str(marker)!r}).write_text('x')"
    result = env.from_string("{{ value }}").render(value=source)
    assert result == source
    assert not marker.exists()


def test_duplicate_keys_no_eval(env):
    t = env.from_string('{"a": {{ 1 }}, "a": 2}')
    result = t.render()
    assert isinstance(result, dict)
    assert result == {"a": 2}


def test_adjacent_expressions(env):
    t = env.from_string("{{ a }}{{ b }}")
    result = t.render(a=1, b=2)
    assert isinstance(result, int)
    assert result == 12
    result = t.render(a="x", b="y")
    assert isinstance(result, str)
    assert result == "xy"
    result = t.render(a=[1], b=[2])
    assert isinstance(result, str)
    assert result == "[1][2]"


def test_template_whitespace(env):
    # Leading whitespace is never a literal, the string is preserved.
    assert env.from_string(" {{ 1 }}").render() == " 1"
    assert env.from_string("\t{{ 1 }}").render() == "\t1"
    # Trailing whitespace doesn't prevent parsing.
    assert env.from_string("{{ 1 }} ").render() == 1
    # Whitespace between nodes joins as text.
    assert env.from_string("{{ 1 }} {{ 2 }}").render() == "1 2"


def test_markup_single_node_preserved(env):
    # Markup is escaped text, never parsed as a Python literal.
    result = env.from_string("{{ value }}").render(value=Markup("[1, 2]"))
    assert isinstance(result, Markup)
    assert result == "[1, 2]"


def test_markup_multi_node_preserved(env):
    result = env.from_string("{{ a }} {{ b }}").render(
        a=Markup("<b>"), b=Markup("[1, 2]")
    )
    assert isinstance(result, Markup)
    assert result == "<b> [1, 2]"


def test_autoescape_markup(env):
    env = NativeEnvironment(autoescape=True)
    result = env.from_string("{{ value }}").render(value="<b>")
    assert isinstance(result, Markup)
    assert result == "&lt;b&gt;"
    result = env.from_string("{{ value }}!").render(value="<b>")
    assert isinstance(result, Markup)
    assert result == "&lt;b&gt;!"
    # Literal template data is safe and stays Markup.
    result = env.from_string("<b>{{ value }}</b>").render(value="<i>")
    assert isinstance(result, Markup)
    assert result == "<b>&lt;i&gt;</b>"
    # Constant escaped data stays Markup too.
    result = env.from_string("{{ '<b>' }}").render()
    assert isinstance(result, Markup)
    assert result == "&lt;b&gt;"


def test_strict_undefined_errors_not_swallowed():
    env = NativeEnvironment(undefined=StrictUndefined)
    # A single node returns the undefined object, which still enforces
    # the error policy when it's used.
    result = env.from_string("{{ missing }}").render()
    assert isinstance(result, StrictUndefined)
    with pytest.raises(UndefinedError):
        str(result)
    # Concatenating nodes calls str on each one, which raises.
    with pytest.raises(UndefinedError):
        env.from_string("{{ missing }} text").render()
    with pytest.raises(UndefinedError):
        env.from_string("{% for x in [1, 2] %}{{ missing }}{% endfor %}").render()


def test_undefined_default_single_node(env):
    result = env.from_string("{{ missing }}").render()
    assert isinstance(result, Undefined)
    # The default undefined prints as an empty string when concatenated.
    assert env.from_string("{{ missing }}!").render() == "!"


def test_strict_mode():
    env = NativeEnvironment(native_strict=True)
    assert env.from_string("{{ 1 + 2 }}").render() == 3
    assert env.from_string("{{ [1, 2] }}").render() == [1, 2]
    with pytest.raises(ValueError, match="native Python type"):
        env.from_string("hello").render()
    with pytest.raises(ValueError, match="native Python type"):
        env.from_string("{{ a }} + {{ b }}").render(a=1, b=2)


def test_native_concat_direct():
    assert native_concat([]) is None
    assert native_concat([42]) == 42
    assert native_concat([Markup("[1]")]) == Markup("[1]")
    assert native_concat(iter([Markup("a"), " ", "b"])) == Markup("a b")
    assert native_concat(iter(["1", "2"])) == 12
    assert native_concat(["a b"]) == "a b"
    with pytest.raises(ValueError, match="native Python type"):
        native_concat(["a b"], strict=True)


def test_finalize_boundary():
    env = NativeEnvironment(
        finalize=lambda v: v.upper() if isinstance(v, str) else v
    )
    result = env.from_string("{{ 'abc' }}").render()
    assert isinstance(result, str)
    assert result == "ABC"
    assert env.from_string("{{ 1 }}").render() == 1


def test_macro_returns_native(env):
    t = env.from_string(
        "{%- macro m() -%}{{ {'a': [1, 2]} }}{%- endmacro -%}{{ m() }}"
    )
    result = t.render()
    assert isinstance(result, dict)
    assert result == {"a": [1, 2]}


def test_for_loop_tuple_trailing_comma(env):
    t = env.from_string("{% for i in range(3) %}{{ i }},{% endfor %}")
    result = t.render()
    assert isinstance(result, tuple)
    assert result == (0, 1, 2)


def test_async_nested_roundtrip(async_native_env, run_async_fn):
    async def async_render():
        value = {"a": [1, (2,)], "b": None, "c": "ü"}
        t = async_native_env.from_string("{{ value }}")
        result = await t.render_async(value=value)
        assert result == value
        assert isinstance(result, dict)

    run_async_fn(async_render)


def test_async_macro_and_loop(async_native_env, run_async_fn):
    async def async_render():
        t = async_native_env.from_string(
            "{%- macro m(x) -%}{{ x }}{%- endmacro -%}"
            "{% for i in [m(1), m(2)] %}{{ i }}{% endfor %}"
        )
        result = await t.render_async()
        assert isinstance(result, int)
        assert result == 12

    run_async_fn(async_render)


def test_async_markup_preserved(async_native_env, run_async_fn):
    async def async_render():
        async_native_env.autoescape = True
        t = async_native_env.from_string("{{ value }}")
        result = await t.render_async(value="<b>")
        assert isinstance(result, Markup)
        assert result == "&lt;b&gt;"

    run_async_fn(async_render)


def test_async_undefined_error_not_swallowed(run_async_fn):
    env = NativeEnvironment(undefined=StrictUndefined, enable_async=True)

    async def async_render():
        with pytest.raises(UndefinedError):
            await env.from_string("{{ missing }}!").render_async()

    run_async_fn(async_render)


def test_async_strict_mode(run_async_fn):
    env = NativeEnvironment(enable_async=True, native_strict=True)

    async def async_render():
        assert await env.from_string("{{ 2 + 2 }}").render_async() == 4
        with pytest.raises(ValueError, match="native Python type"):
            await env.from_string("plain text").render_async()

    run_async_fn(async_render)

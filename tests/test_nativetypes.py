import math
from functools import partial

import pytest
from markupsafe import Markup

from jinja2.exceptions import UndefinedError
from jinja2.nativetypes import NativeEnvironment
from jinja2.nativetypes import NativeTemplate
from jinja2.nativetypes import native_concat
from jinja2.runtime import DebugUndefined
from jinja2.runtime import StrictUndefined


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
    # The default undefined policy renders an empty string, the
    # object is not returned silently.
    assert t.render() == ""


def test_strict_undefined_native_return():
    env = NativeEnvironment(undefined=StrictUndefined)
    t = env.from_string("{{ missing }}")

    with pytest.raises(UndefinedError):
        t.render()


def test_debug_undefined_native_return():
    env = NativeEnvironment(undefined=DebugUndefined)
    t = env.from_string("{{ missing }}")
    assert t.render() == "{{ missing }}"


def test_undefined_in_concat(env):
    t = env.from_string("x{{ missing }}")
    assert t.render() == "x"


def test_strict_undefined_in_concat():
    env = NativeEnvironment(undefined=StrictUndefined)
    t = env.from_string("x{{ missing }}")

    with pytest.raises(UndefinedError):
        t.render()


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
        "list": [1, 2.5, (3, 4)],
        "dict": {"a": None, "b": True, "c": False},
        "unicode": "h\xe9llo \u4f60\u597d \U0001f600",
    }
    result = env.from_string("{{ x }}").render(x=value)
    assert result == value
    assert isinstance(result, dict)
    assert isinstance(result["list"], list)
    assert isinstance(result["list"][2], tuple)


def test_nested_dict_list_from_string(env):
    t = env.from_string("{{ x|string }}")
    result = t.render(x={"a": [1, (2, 3)], "b": None, "c": "h\xe9llo"})
    assert result == {"a": [1, (2, 3)], "b": None, "c": "h\xe9llo"}


def test_nested_dict_list_roundtrip_async(async_native_env, run_async_fn):
    async def async_render():
        value = {"a": [1, (2, 3)], "b": None, "c": "h\xe9llo"}
        t = async_native_env.from_string("{{ x }}")
        result = await t.render_async(x=value)
        assert result == value
        assert isinstance(result, dict)

    run_async_fn(async_render)


def test_duplicate_keys(env):
    # Later keys win, the same as Python, without executing anything.
    result = env.from_string("{{ x }}").render(x="{'a': 1, 'a': 2}")
    assert result == {"a": 2}


@pytest.mark.parametrize(
    "value",
    [
        "__import__('os').system('echo pwned')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd').read()",
        "lambda: 1",
        "f'{1 + 1}'",
        "(x for x in range(3))",
    ],
)
def test_malicious_literal_not_evaluated(env, value):
    # Values that aren't simple literals are never evaluated, they are
    # returned as strings.
    result = env.from_string("{{ x }}").render(x=value)
    assert isinstance(result, str)
    assert result == value


def test_unclosed_literal(env):
    result = env.from_string("{{ x }}").render(x="[1, 2")
    assert isinstance(result, str)
    assert result == "[1, 2"


def test_deeply_nested_literal_safe(env):
    # A pathological literal must not crash the parser, it's returned
    # as a string instead.
    value = "[" * 1000
    result = env.from_string("{{ x }}").render(x=value)
    assert result == value


def test_custom_object_single_node(env):
    class Foo:
        def __repr__(self):
            return "Foo()"

    foo = Foo()
    result = env.from_string("{{ x }}").render(x=foo)
    assert result is foo


def test_custom_object_in_concat(env):
    class Foo:
        def __str__(self):
            return "Foo()"

    result = env.from_string("{{ x }} + {{ y }}").render(x=Foo(), y=2)
    assert isinstance(result, str)
    assert result == "Foo() + 2"


def test_adjacent_expressions(env):
    # Adjacent expressions are joined before parsing, so two numbers
    # become one number.
    result = env.from_string("{{ a }}{{ b }}").render(a=1, b=2)
    assert result == 12


def test_adjacent_expressions_with_space(env):
    result = env.from_string("{{ a }} {{ b }}").render(a=1, b=2)
    assert isinstance(result, str)
    assert result == "1 2"


def test_markup_single_node(env):
    result = env.from_string("{{ x }}").render(x=Markup("<b>1</b>"))
    assert isinstance(result, Markup)
    assert result == "<b>1</b>"


def test_markup_concat_preserved(env):
    result = env.from_string("{{ x }}!").render(x=Markup("<b>"))
    assert isinstance(result, Markup)
    assert result == "<b>!"


def test_markup_literal_parsed(env):
    # A Markup string that looks like a literal is still parsed.
    result = env.from_string("{{ x }}").render(x=Markup("[1, 2]"))
    assert result == [1, 2]


def test_autoescape_escapes():
    env = NativeEnvironment(autoescape=True)
    result = env.from_string("{{ x }}").render(x="<b>")
    assert isinstance(result, Markup)
    assert result == "&lt;b&gt;"


def test_autoescape_concat():
    env = NativeEnvironment(autoescape=True)
    result = env.from_string("{{ x }}!").render(x="<b>")
    assert isinstance(result, Markup)
    assert result == "&lt;b&gt;!"


def test_autoescape_safe_not_double_escaped():
    env = NativeEnvironment(autoescape=True)
    result = env.from_string("{{ x }}").render(x=Markup("<b>"))
    assert isinstance(result, Markup)
    assert result == "<b>"


def test_autoescape_async(run_async_fn):
    env = NativeEnvironment(autoescape=True, enable_async=True)

    async def async_render():
        result = await env.from_string("{{ x }}").render_async(x="<b>")
        assert isinstance(result, Markup)
        assert result == "&lt;b&gt;"

    run_async_fn(async_render)


def test_finalize():
    env = NativeEnvironment(finalize=lambda v: v * 2 if isinstance(v, int) else v)
    result = env.from_string("{{ x }}").render(x=21)
    assert result == 42


def test_macro_return_value(env):
    # Without autoescape a macro returns a plain string.
    t = env.from_string("{%- macro m() -%}hi{%- endmacro -%}{{ m() }}")
    result = t.render()
    assert isinstance(result, str)
    assert result == "hi"


def test_macro_returns_markup_autoescape():
    env = NativeEnvironment(autoescape=True)
    t = env.from_string("{%- macro m() -%}hi{%- endmacro -%}{{ m() }}")
    result = t.render()
    assert isinstance(result, Markup)
    assert result == "hi"


def test_loop_async(async_native_env, run_async_fn):
    async def async_render():
        t = async_native_env.from_string("{% for x in value %}{{ x }}{% endfor %}")
        result = await t.render_async(value=[1, 2, 3])
        assert result == 123

    run_async_fn(async_render)


def test_strict_undefined_async(run_async_fn):
    env = NativeEnvironment(undefined=StrictUndefined, enable_async=True)

    async def async_render():
        t = env.from_string("{{ missing }}")

        with pytest.raises(UndefinedError):
            await t.render_async()

    run_async_fn(async_render)


def test_whitespace_around_expression(env):
    # Leading whitespace prevents parsing, trailing does not.
    assert env.from_string(" {{ x }}").render(x=1) == " 1"
    assert env.from_string("{{ x }} ").render(x=1) == 1
    assert env.from_string(" ").render() == " "


def test_empty_template(env):
    assert env.from_string("").render() is None


def test_native_concat_strict():
    with pytest.raises(ValueError):
        native_concat(iter(["not a literal"]), strict=True)

    assert native_concat(iter(["[1, 2]"]), strict=True) == [1, 2]
    # A single non-string node is returned as-is even in strict mode.
    foo = object()
    assert native_concat(iter([foo]), strict=True) is foo


def test_strict_environment():
    class StrictNativeEnvironment(NativeEnvironment):
        concat = staticmethod(partial(native_concat, strict=True))

    env = StrictNativeEnvironment()
    assert env.from_string("{{ x }}").render(x="[1, 2]") == [1, 2]
    assert env.from_string("{{ x }}").render(x=3) == 3

    with pytest.raises(ValueError):
        env.from_string("{{ x }}").render(x="hello")


def test_strict_environment_async(run_async_fn):
    class StrictNativeEnvironment(NativeEnvironment):
        concat = staticmethod(partial(native_concat, strict=True))

    env = StrictNativeEnvironment(enable_async=True)

    async def async_render():
        t = env.from_string("{{ x }}")
        assert await t.render_async(x="{'a': 1}") == {"a": 1}

        with pytest.raises(ValueError):
            await t.render_async(x="hello")

    run_async_fn(async_render)

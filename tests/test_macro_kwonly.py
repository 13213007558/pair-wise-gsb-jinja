import pytest
from markupsafe import Markup

from jinja2 import Environment
from jinja2.exceptions import TemplateSyntaxError


@pytest.fixture
def macro_env(filesystem_loader):
    return Environment(loader=filesystem_loader)


def test_from_template(macro_env):
    t = macro_env.from_string(
        '{% from "kwonly_macros.html" import price %}'
        "{{ price(3, currency='USD') }}|{{ price(3, currency='EUR', digits=0) }}"
    )
    assert t.render() == "USD3.00|EUR3"


def test_from_python_macro_object(macro_env):
    price = macro_env.get_template("kwonly_macros.html").module.price
    assert price.arguments == ("value",)
    assert price.kwonlyarguments == ("currency", "digits")
    assert "*" not in price.arguments
    assert price(3, currency="USD") == "USD3.00"
    assert price(3, currency="USD", digits=4) == "USD3.0000"


def test_from_async_template(filesystem_loader, run_async_fn):
    env = Environment(loader=filesystem_loader, enable_async=True)
    t = env.from_string(
        '{% from "kwonly_macros.html" import price %}'
        "{{ price(2.5, currency='JPY', digits=1) }}"
    )

    async def render():
        return await t.render_async()

    assert run_async_fn(render) == "JPY2.5"


def test_default_evaluated_like_old_macros():
    calls = []

    def default_digits():
        calls.append(1)
        return 2

    env = Environment()
    env.globals["default_digits"] = default_digits
    t = env.from_string(
        "{% macro m(value, *, digits=default_digits()) %}"
        "{{ value }}:{{ digits }}"
        "{% endmacro %}"
        "{{ m(1) }}|{{ m(2, digits=9) }}|{{ m(3) }}"
    )
    assert t.render() == "1:2|2:9|3:2"
    # once per call that omitted the argument, never when it was given
    assert len(calls) == 2


def test_autoescape_matches_old_macros(filesystem_loader):
    env = Environment(loader=filesystem_loader, autoescape=True)
    t = env.from_string(
        '{% from "kwonly_macros.html" import badge %}'
        "{% macro old(text) %}<b>{{ text }}</b>{% endmacro %}"
        "{{ badge('<x>', label='new') }}|{{ old('<x>') }}"
    )
    assert t.render() == "[new] &lt;x&gt;|<b>&lt;x&gt;</b>"
    rv = env.get_template("kwonly_macros.html").module.badge("<x>", label="new")
    assert isinstance(rv, Markup)
    assert rv == "[new] &lt;x&gt;"


def test_positional_past_separator_fails(macro_env):
    t = macro_env.from_string(
        '{% from "kwonly_macros.html" import price %}{{ price(1, "USD") }}'
    )
    with pytest.raises(TypeError) as excinfo:
        t.render()
    assert str(excinfo.value) == (
        "macro 'price' parameter 'currency' is keyword-only"
        " and cannot be passed as a positional argument"
    )


def test_missing_required_keyword_fails(macro_env):
    t = macro_env.from_string(
        '{% from "kwonly_macros.html" import price %}{{ price(1) }}'
    )
    with pytest.raises(TypeError) as excinfo:
        t.render()
    assert (
        str(excinfo.value) == "macro 'price' requires keyword-only argument 'currency'"
    )


def test_duplicate_assignment_fails(macro_env):
    t = macro_env.from_string(
        '{% from "kwonly_macros.html" import price %}'
        "{{ price(1, currency='EUR', value=2) }}"
    )
    with pytest.raises(TypeError) as excinfo:
        t.render()
    assert (
        str(excinfo.value) == "macro 'price' got multiple values for argument 'value'"
    )


def test_unknown_keyword_fails(macro_env):
    t = macro_env.from_string(
        '{% from "kwonly_macros.html" import price %}'
        "{{ price(1, currency='EUR', bogus=1) }}"
    )
    with pytest.raises(TypeError) as excinfo:
        t.render()
    assert str(excinfo.value) == "macro 'price' takes no keyword argument 'bogus'"


def test_duplicate_separator_parse_error():
    with pytest.raises(TemplateSyntaxError) as excinfo:
        Environment().from_string(
            "{% macro m(a,\n  *,\n  b,\n  *,\n  c) %}{% endmacro %}"
        )
    assert excinfo.value.lineno == 4
    assert "duplicate keyword-only argument separator '*'" in excinfo.value.message


def test_non_default_after_default_parse_error():
    with pytest.raises(TemplateSyntaxError) as excinfo:
        Environment().from_string("{% macro m(a=1,\n  b) %}{% endmacro %}")
    assert excinfo.value.lineno == 2
    assert "non-default argument follows default argument" in excinfo.value.message


def test_kwonly_defaults_any_order(macro_env):
    t = macro_env.from_string(
        "{% macro m(*, a=1, b) %}{{ a }}-{{ b }}{% endmacro %}{{ m(b=2) }}"
    )
    assert t.render() == "1-2"


def test_caller_convention_preserved(macro_env):
    t = macro_env.from_string(
        '{% from "kwonly_macros.html" import panel %}'
        '{% call panel(title="hi") %}body{% endcall %}'
    )
    assert t.render() == "hi: body"


def test_varargs_kwargs_convention_preserved(macro_env):
    t = macro_env.from_string(
        '{% from "kwonly_macros.html" import join_all %}'
        "{{ join_all(1, 2, 3, sep='-', extra=4) }}"
    )
    assert t.render() == "1-2-3|extra"


def test_call_block_kwonly_signature():
    env = Environment()
    t = env.from_string(
        "{% macro dump(sep) %}{{ caller('a', 'b', sep=sep) }}{% endmacro %}"
        "{% call(x, y, *, sep) dump('-') %}{{ x }}{{ sep }}{{ y }}{% endcall %}"
    )
    assert t.render() == "a-b"

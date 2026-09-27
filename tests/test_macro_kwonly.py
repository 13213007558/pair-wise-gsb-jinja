import pytest
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.exceptions import TemplateAssertionError
from jinja2.exceptions import TemplateSyntaxError
from markupsafe import Markup


@pytest.fixture
def lib_env():
    return Environment(
        loader=DictLoader(
            {
                "macros.html": (
                    "{% macro price(value, *, currency, digits=2) %}"
                    "{{ currency }}{{ '%.*f' % (digits, value) }}"
                    "{% endmacro %}"
                    "{% macro wrap(tag='div', *, class='box') %}"
                    '<{{ tag }} class="{{ class }}">{{ caller() }}</{{ tag }}>'
                    "{% endmacro %}"
                )
            }
        ),
        autoescape=True,
    )


class TestKeywordOnlyCalls:
    def test_call_from_template(self, lib_env):
        tmpl = lib_env.from_string(
            '{% import "macros.html" as m %}'
            "{{ m.price(3.14159, currency='$') }}"
            "|{{ m.price(2.5, currency='EUR ', digits=1) }}"
        )
        assert tmpl.render() == "$3.14|EUR 2.5"

    def test_from_import(self, lib_env):
        tmpl = lib_env.from_string(
            '{% from "macros.html" import price %}'
            "{{ price(1.005, currency='$', digits=3) }}"
        )
        assert tmpl.render() == "$1.005"

    def test_macro_object_from_python(self, lib_env):
        price = lib_env.get_template("macros.html").module.price
        assert price(9.9, currency="\u00a3") == "\u00a39.90"
        assert price(9.9, currency="$", digits=3) == "$9.900"

    def test_async_template(self, lib_env, run_async_fn):
        lib_env.is_async = True
        tmpl = lib_env.from_string(
            "{% import \"macros.html\" as m %}{{ m.price(3.14159, currency='$') }}"
        )

        async def func():
            return await tmpl.render_async()

        assert run_async_fn(func) == "$3.14"

    def test_async_macro_object(self, lib_env, run_async_fn):
        lib_env.is_async = True
        tmpl = lib_env.get_template("macros.html")

        async def func():
            module = await tmpl.make_module_async()
            return await module.price(2.5, currency="EUR ", digits=1)

        assert run_async_fn(func) == "EUR 2.5"

    def test_autoescape_result(self, lib_env):
        price = lib_env.get_template("macros.html").module.price
        rv = price(1, currency="<")
        assert isinstance(rv, Markup)
        assert rv == "&lt;1.00"


class TestKeywordOnlyBinding:
    def test_positional_crossing_separator_fails(self, lib_env):
        tmpl = lib_env.from_string(
            '{% import "macros.html" as m %}{{ m.price(1, "$") }}'
        )
        with pytest.raises(TypeError) as excinfo:
            tmpl.render()
        assert "'currency'" in str(excinfo.value)
        assert "keyword-only" in str(excinfo.value)

    def test_missing_required_keyword_fails(self, lib_env):
        tmpl = lib_env.from_string('{% import "macros.html" as m %}{{ m.price(1) }}')
        with pytest.raises(TypeError) as excinfo:
            tmpl.render()
        assert "'currency'" in str(excinfo.value)
        assert "missing required keyword-only" in str(excinfo.value)

    def test_duplicate_assignment_fails(self, lib_env):
        tmpl = lib_env.from_string(
            '{% import "macros.html" as m %}{{ m.price(1, value=2, currency="$") }}'
        )
        with pytest.raises(TypeError) as excinfo:
            tmpl.render()
        assert "multiple values for argument 'value'" in str(excinfo.value)

    def test_unknown_keyword_fails(self, lib_env):
        tmpl = lib_env.from_string(
            '{% import "macros.html" as m %}{{ m.price(1, currency="$", x=1) }}'
        )
        with pytest.raises(TypeError) as excinfo:
            tmpl.render()
        assert "takes no keyword argument 'x'" in str(excinfo.value)

    def test_too_many_positionals_fails(self, lib_env):
        tmpl = lib_env.from_string(
            '{% import "macros.html" as m %}{{ m.price(1, 2, 3, currency="$") }}'
        )
        with pytest.raises(TypeError) as excinfo:
            tmpl.render()
        assert "positional" in str(excinfo.value)

    def test_errors_before_body_from_python(self, lib_env):
        price = lib_env.get_template("macros.html").module.price
        with pytest.raises(TypeError, match="keyword-only"):
            price(1, 2)
        with pytest.raises(TypeError, match="missing required keyword-only"):
            price(1)
        with pytest.raises(TypeError, match="multiple values"):
            price(1, value=2, currency="$")


class TestKeywordOnlySyntax:
    @pytest.mark.parametrize(
        "source",
        [
            "{% macro m(a, *, b, *, c) %}{% endmacro %}",
            "{% macro m(*, a, *, b) %}{% endmacro %}",
        ],
    )
    def test_duplicate_separator_fails(self, env, source):
        with pytest.raises(TemplateSyntaxError) as excinfo:
            env.from_string(source)
        assert "only contain one '*'" in str(excinfo.value)
        assert excinfo.value.lineno == 1

    @pytest.mark.parametrize(
        "source",
        [
            "{% macro m(a, *) %}{% endmacro %}",
            "{% macro m(*) %}{% endmacro %}",
            "{% macro m(a, *,) %}{% endmacro %}",
        ],
    )
    def test_dangling_separator_fails(self, env, source):
        with pytest.raises(TemplateSyntaxError) as excinfo:
            env.from_string(source)
        assert "keyword-only parameter" in str(excinfo.value)
        assert excinfo.value.lineno == 1

    def test_separator_lineno(self, env):
        with pytest.raises(TemplateSyntaxError) as excinfo:
            env.from_string("\n\n{% macro m(*, a, *, b) %}{% endmacro %}")
        assert excinfo.value.lineno == 3

    def test_positional_default_rule_still_checked(self, env):
        with pytest.raises(TemplateSyntaxError) as excinfo:
            env.from_string("{% macro m(a=1, b) %}{% endmacro %}")
        assert "non-default argument follows default argument" in str(excinfo.value)

    def test_keyword_only_defaults_free_order(self, env):
        tmpl = env.from_string(
            "{% macro m(*, a=1, b, c=3) %}{{ a }}{{ b }}{{ c }}{% endmacro %}"
            "{{ m(b=2) }}"
        )
        assert tmpl.render() == "123"

    def test_separator_not_a_parameter(self, lib_env):
        price = lib_env.get_template("macros.html").module.price
        assert price.arguments == ("value",)
        assert price.kwonlyargs == ("currency", "digits")
        assert "*" not in price.arguments
        assert "*" not in price.kwonlyargs


class TestKeywordOnlyConventions:
    def test_call_block_supplies_caller(self, lib_env):
        tmpl = lib_env.from_string(
            '{% import "macros.html" as m %}'
            '{% call m.wrap("b", class="y") %}body{% endcall %}'
        )
        assert tmpl.render() == '<b class="y">body</b>'

    def test_call_block_keyword_only_signature(self, env):
        tmpl = env.from_string(
            "{% macro m() %}[{{ caller(1, y=2) }}]{% endmacro %}"
            "{% call(x, *, y) m() %}{{ x }}-{{ y }}{% endcall %}"
        )
        assert tmpl.render() == "[1-2]"

    def test_caller_not_an_extra_argument(self, env):
        # the caller passed by a call block must not be reported as an
        # unexpected user argument.
        tmpl = env.from_string(
            "{% macro m(a) %}{{ a }}:{{ kwargs }}{% endmacro %}"
            "{% call m(1) %}body{% endcall %}"
        )
        assert tmpl.render().startswith("1:{'caller': <Macro ")

    def test_varargs_and_kwargs_kept(self, env):
        tmpl = env.from_string(
            "{% macro m(a, *, b=1) %}{{ a }}|{{ b }}|{{ varargs }}|{{ kwargs }}"
            "{% endmacro %}"
            "{{ m(1, 2, 3, b=4, c=5) }}"
        )
        assert tmpl.render() == "1|4|(2, 3)|{'c': 5}"

    def test_duplicate_with_catch_kwargs_fails(self, env):
        tmpl = env.from_string(
            "{% macro m(a) %}{{ kwargs }}{% endmacro %}{{ m(1, a=2) }}"
        )
        with pytest.raises(TypeError) as excinfo:
            tmpl.render()
        assert "multiple values for argument 'a'" in str(excinfo.value)

    def test_caller_keyword_only_with_default(self, env):
        tmpl = env.from_string(
            "{% macro m(*, caller=None) %}[{{ caller() }}]{% endmacro %}"
            "{% call m() %}x{% endcall %}"
        )
        assert tmpl.render() == "[x]"

    def test_caller_keyword_only_required_fails(self, env):
        with pytest.raises(TemplateAssertionError):
            env.from_string("{% macro m(*, caller) %}{{ caller() }}{% endmacro %}")


class TestKeywordOnlyDefaults:
    def test_default_evaluated_once_per_call(self, env):
        calls = []
        env.globals["side"] = lambda v: (calls.append(v), v)[1]
        tmpl = env.from_string(
            "{% macro m(a, *, b=side(10), c=side(20)) %}{{ a }}{{ b }}{{ c }}"
            "{% endmacro %}{{ m(1) }}{{ m(2, c=99) }}"
        )
        assert tmpl.render() == "1102021099"
        assert calls == [10, 20, 10]

    def test_old_macros_unchanged(self, env):
        tmpl = env.from_string(
            "{% macro m(a, b=1) %}{{ a }}{{ b }}{% endmacro %}{{ m(2) }}"
        )
        assert tmpl.render() == "21"
        assert tmpl.module.m.kwonlyargs == ()

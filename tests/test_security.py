import pytest
from markupsafe import escape

from jinja2 import ChainableUndefined
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.exceptions import SecurityError
from jinja2.exceptions import TemplateRuntimeError
from jinja2.exceptions import TemplateSyntaxError
from jinja2.nodes import EvalContext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.sandbox import SandboxedEnvironment
from jinja2.sandbox import unsafe


class PrivateStuff:
    def bar(self):
        return 23

    @unsafe
    def foo(self):
        return 42

    def __repr__(self):
        return "PrivateStuff"


class PublicStuff:
    def bar(self):
        return 23

    def _foo(self):
        return 42

    def __repr__(self):
        return "PublicStuff"


class TestSandbox:
    def test_unsafe(self, env):
        env = SandboxedEnvironment()
        pytest.raises(
            SecurityError, env.from_string("{{ foo.foo() }}").render, foo=PrivateStuff()
        )
        assert env.from_string("{{ foo.bar() }}").render(foo=PrivateStuff()) == "23"

        pytest.raises(
            SecurityError, env.from_string("{{ foo._foo() }}").render, foo=PublicStuff()
        )
        assert env.from_string("{{ foo.bar() }}").render(foo=PublicStuff()) == "23"
        assert env.from_string("{{ foo.__class__ }}").render(foo=42) == ""
        assert env.from_string("{{ foo.func_code }}").render(foo=lambda: None) == ""
        # security error comes from __class__ already.
        pytest.raises(
            SecurityError,
            env.from_string("{{ foo.__class__.__subclasses__() }}").render,
            foo=42,
        )

    def test_immutable_environment(self, env):
        env = ImmutableSandboxedEnvironment()
        pytest.raises(SecurityError, env.from_string("{{ [].append(23) }}").render)
        pytest.raises(SecurityError, env.from_string("{{ [].clear() }}").render)
        pytest.raises(SecurityError, env.from_string("{{ [1].pop() }}").render)
        pytest.raises(SecurityError, env.from_string("{{ {1:2}.clear() }}").render)

    def test_restricted(self, env):
        env = SandboxedEnvironment()
        pytest.raises(
            TemplateSyntaxError,
            env.from_string,
            "{% for item.attribute in seq %}...{% endfor %}",
        )
        pytest.raises(
            TemplateSyntaxError,
            env.from_string,
            "{% for foo, bar.baz in seq %}...{% endfor %}",
        )

    def test_template_data(self, env):
        env = Environment(autoescape=True)
        t = env.from_string(
            "{% macro say_hello(name) %}"
            "<p>Hello {{ name }}!</p>{% endmacro %}"
            '{{ say_hello("<blink>foo</blink>") }}'
        )
        escaped_out = "<p>Hello &lt;blink&gt;foo&lt;/blink&gt;!</p>"
        assert t.render() == escaped_out
        assert str(t.module) == escaped_out
        assert escape(t.module) == escaped_out
        assert t.module.say_hello("<blink>foo</blink>") == escaped_out
        assert (
            escape(t.module.say_hello(EvalContext(env), "<blink>foo</blink>"))
            == escaped_out
        )
        assert escape(t.module.say_hello("<blink>foo</blink>")) == escaped_out

    def test_attr_filter(self, env):
        env = SandboxedEnvironment()
        tmpl = env.from_string('{{ cls|attr("__subclasses__")() }}')
        pytest.raises(SecurityError, tmpl.render, cls=int)

    def test_binary_operator_intercepting(self, env):
        def disable_op(left, right):
            raise TemplateRuntimeError("that operator so does not work")

        for expr, ctx, rv in ("1 + 2", {}, "3"), ("a + 2", {"a": 2}, "4"):
            env = SandboxedEnvironment()
            env.binop_table["+"] = disable_op
            t = env.from_string(f"{{{{ {expr} }}}}")
            assert t.render(ctx) == rv
            env.intercepted_binops = frozenset(["+"])
            t = env.from_string(f"{{{{ {expr} }}}}")
            with pytest.raises(TemplateRuntimeError):
                t.render(ctx)

    def test_unary_operator_intercepting(self, env):
        def disable_op(arg):
            raise TemplateRuntimeError("that operator so does not work")

        for expr, ctx, rv in ("-1", {}, "-1"), ("-a", {"a": 2}, "-2"):
            env = SandboxedEnvironment()
            env.unop_table["-"] = disable_op
            t = env.from_string(f"{{{{ {expr} }}}}")
            assert t.render(ctx) == rv
            env.intercepted_unops = frozenset(["-"])
            t = env.from_string(f"{{{{ {expr} }}}}")
            with pytest.raises(TemplateRuntimeError):
                t.render(ctx)


class TestStringFormat:
    def test_basic_format_safety(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ "a{0.__class__}b".format(42) }}')
        assert t.render() == "ab"

    def test_basic_format_all_okay(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ "a{0.foo}b".format({"foo": 42}) }}')
        assert t.render() == "a42b"

    def test_safe_format_safety(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ ("a{0.__class__}b{1}"|safe).format(42, "<foo>") }}')
        assert t.render() == "ab&lt;foo&gt;"

    def test_safe_format_all_okay(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ ("a{0.foo}b{1}"|safe).format({"foo": 42}, "<foo>") }}')
        assert t.render() == "a42b&lt;foo&gt;"

    def test_empty_braces_format(self):
        env = SandboxedEnvironment()
        t1 = env.from_string('{{ ("a{}b{}").format("foo", "42")}}')
        t2 = env.from_string('{{ ("a{}b{}"|safe).format(42, "<foo>") }}')
        assert t1.render() == "afoob42"
        assert t2.render() == "a42b&lt;foo&gt;"


class TestStringFormatMap:
    def test_basic_format_safety(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ "a{x.__class__}b".format_map({"x":42}) }}')
        assert t.render() == "ab"

    def test_basic_format_all_okay(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ "a{x.foo}b".format_map({"x":{"foo": 42}}) }}')
        assert t.render() == "a42b"

    def test_safe_format_all_okay(self):
        env = SandboxedEnvironment()
        t = env.from_string(
            '{{ ("a{x.foo}b{y}"|safe).format_map({"x":{"foo": 42}, "y":"<foo>"}) }}'
        )
        assert t.render() == "a42b&lt;foo&gt;"

    def test_indirect_call(self):
        def run(value, arg):
            return value.run(arg)

        env = SandboxedEnvironment()
        env.filters["run"] = run
        t = env.from_string(
            """{% set
                ns = namespace(run="{0.__call__.__builtins__[__import__]}".format)
            %}
            {{ ns | run(not_here) }}
            """
        )

        with pytest.raises(SecurityError):
            t.render()

    def test_attr_filter(self) -> None:
        env = SandboxedEnvironment()
        t = env.from_string(
            """{{ "{0.__call__.__builtins__[__import__]}"
                  | attr("format")(not_here) }}"""
        )

        with pytest.raises(SecurityError):
            t.render()


class PropertyRecorder:
    """Records the order in which its properties are resolved."""

    def __init__(self):
        self.calls = []

    @property
    def public_prop(self):
        self.calls.append("public_prop")
        return "public-value"

    @property
    def _private_prop(self):
        self.calls.append("_private_prop")
        return "private-value"


class CallableDescriptor:
    """A descriptor that records when it is resolved and returns a
    callable when accessed."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        obj.calls.append(self.name)
        return lambda: f"{self.name}-value"


class DescriptorRecorder:
    callback = CallableDescriptor("callback")
    _callback = CallableDescriptor("_callback")

    def __init__(self):
        self.calls = []


class CollisionDict(dict):
    """A dict that also has properties with the same names as keys."""

    @property
    def public(self):
        raise AttributeError("descriptor failed")

    @property
    def _private(self):
        raise AttributeError("descriptor failed")


class TestSandboxAttributePolicy:
    """The sandbox applies one attribute policy to every lookup path:
    template expressions, the attr filter, item access, and filters that
    take attribute names such as map and selectattr.
    """

    @pytest.fixture
    def env(self):
        return SandboxedEnvironment()

    @pytest.mark.parametrize(
        ("template", "expect"),
        [
            ("{{ o.public_prop }}", "public-value"),
            ("{{ o|attr('public_prop') }}", "public-value"),
            ("{{ o['public_prop'] }}", "public-value"),
            ("{{ [o]|map(attribute='public_prop')|first }}", "public-value"),
            (
                "{{ [o]|selectattr('public_prop', 'eq', 'public-value')|list|length }}",
                "1",
            ),
        ],
    )
    def test_safe_attribute_allowed(self, env, template, expect):
        obj = PropertyRecorder()
        assert env.from_string(template).render(o=obj) == expect
        assert obj.calls == ["public_prop"]

    @pytest.mark.parametrize(
        "template",
        [
            "{{ o._private_prop }}",
            "{{ o|attr('_private_prop') }}",
            "{{ o['_private_prop'] }}",
            "{{ [o]|map(attribute='_private_prop')|list }}",
            "{{ [o]|selectattr('_private_prop')|list }}",
        ],
    )
    def test_private_attribute_denied_without_side_effect(self, env, template):
        obj = PropertyRecorder()
        out = env.from_string(template).render(o=obj)
        assert "private-value" not in out
        # The rejected descriptor was never executed.
        assert obj.calls == []

    @pytest.mark.parametrize(
        "template",
        [
            "{{ o._private_prop + 1 }}",
            "{{ (o|attr('_private_prop'))() }}",
            "{{ [o]|map(attribute='_private_prop')|sum }}",
        ],
    )
    def test_private_attribute_use_raises_security_error(self, env, template):
        with pytest.raises(SecurityError):
            env.from_string(template).render(o=PropertyRecorder())

    def test_missing_attribute_is_undefined(self, env):
        obj = PropertyRecorder()
        assert env.from_string("{{ o.missing is undefined }}").render(o=obj) == "True"
        assert (
            env.from_string("{{ o|attr('missing') is undefined }}").render(o=obj)
            == "True"
        )
        assert obj.calls == []

    def test_undefined_has_existing_type(self):
        # Missing and rejected attributes produce the environment's
        # configured Undefined type, not a Python AttributeError.
        env = SandboxedEnvironment(undefined=ChainableUndefined)
        obj = PropertyRecorder()
        assert type(env.getattr(obj, "missing")) is ChainableUndefined
        assert type(env.getattr(obj, "_private_prop")) is ChainableUndefined
        assert type(env.getitem(obj, "missing")) is ChainableUndefined

    def test_private_name_not_leaked_through_item_fallback(self, env):
        obj = CollisionDict(_private="dict-private", public="dict-public")
        # The private descriptor is rejected by the policy, the dict item
        # with the same name is not leaked as a fallback.
        assert env.from_string("{{ o._private }}").render(o=obj) == ""
        assert env.from_string("{{ o|attr('_private') }}").render(o=obj) == ""
        # Item access itself is still allowed.
        assert env.from_string("{{ o['_private'] }}").render(o=obj) == "dict-private"
        # A descriptor that raises AttributeError for a safe name falls
        # back to the item, like the base environment.
        assert env.from_string("{{ o.public }}").render(o=obj) == "dict-public"

    def test_callable_descriptor(self, env):
        obj = DescriptorRecorder()
        assert env.from_string("{{ o.callback() }}").render(o=obj) == "callback-value"
        assert obj.calls == ["callback"]

    def test_private_callable_descriptor_not_resolved(self, env):
        obj = DescriptorRecorder()
        assert env.from_string("{{ o._callback }}").render(o=obj) == ""
        assert env.from_string("{{ o|attr('_callback') }}").render(o=obj) == ""
        assert obj.calls == []

    def test_unsafe_callable_access_allowed_call_denied(self, env):
        # Attribute access and calling are checked separately: an unsafe
        # callable can be referenced, but not called.
        assert (
            env.from_string("{{ o.foo is defined }}").render(o=PrivateStuff())
            == "True"
        )
        with pytest.raises(SecurityError):
            env.from_string("{{ o.foo() }}").render(o=PrivateStuff())

    def test_policy_checked_before_descriptor(self):
        events = []

        class Tracked:
            @property
            def value(self):
                events.append("descriptor")
                return "v"

            @property
            def _hidden(self):
                events.append("descriptor")
                return "h"

        class TrackedEnv(SandboxedEnvironment):
            def is_safe_attribute(self, obj, attr, value):
                events.append(f"policy:{attr}")
                return super().is_safe_attribute(obj, attr, value)

        env = TrackedEnv()
        assert env.from_string("{{ o.value }}").render(o=Tracked()) == "v"
        # The policy ran before the descriptor was executed.
        assert events[0] == "policy:value"
        assert events.index("policy:value") < events.index("descriptor")

        events.clear()
        assert env.from_string("{{ o._hidden }}").render(o=Tracked()) == ""
        # The rejected descriptor was never executed.
        assert events == ["policy:_hidden"]

    def test_policy_sees_stable_object_and_name(self):
        seen = []

        class RecordingEnv(SandboxedEnvironment):
            def is_safe_attribute(self, obj, attr, value):
                seen.append((obj, attr))
                return super().is_safe_attribute(obj, attr, value)

        env = RecordingEnv()
        obj = PropertyRecorder()
        env.from_string("{{ o.public_prop }}").render(o=obj)
        env.from_string("{{ o|attr('public_prop') }}").render(o=obj)
        env.from_string("{{ [o]|map(attribute='public_prop')|list }}").render(o=obj)
        assert seen
        assert all(o is obj and a == "public_prop" for o, a in seen)

    def test_policy_not_cached_between_renders(self):
        class FlipEnv(SandboxedEnvironment):
            allow = True

            def is_safe_attribute(self, obj, attr, value):
                if attr == "value":
                    return self.allow
                return super().is_safe_attribute(obj, attr, value)

        class Obj:
            @property
            def value(self):
                return "v"

        env = FlipEnv()
        t = env.from_string("{{ o.value }}")
        assert t.render(o=Obj()) == "v"
        env.allow = False
        assert t.render(o=Obj()) == ""
        env.allow = True
        assert t.render(o=Obj()) == "v"

    def test_chained_attr_filter(self, env):
        class Outer:
            def __init__(self, inner):
                self.inner = inner

        inner = PropertyRecorder()
        outer = Outer(inner)
        t = "{{ o|attr('inner')|attr('public_prop') }}"
        assert env.from_string(t).render(o=outer) == "public-value"
        t = "{{ o|attr('inner')|attr('_private_prop') }}"
        assert env.from_string(t).render(o=outer) == ""
        assert inner.calls == ["public_prop"]

    def test_macro_and_inheritance_do_not_bypass(self):
        env = SandboxedEnvironment(
            loader=DictLoader(
                {
                    "base": "{% block content %}{{ o._private_prop }}{% endblock %}",
                    "child": (
                        "{% extends 'base' %}"
                        "{% block content %}{{ o|attr('_private_prop') }}{% endblock %}"
                    ),
                    "macro": (
                        "{% macro show(o) %}{{ o._private_prop }}{% endmacro %}"
                        "{{ show(o) }}"
                    ),
                }
            )
        )
        obj = PropertyRecorder()
        assert env.get_template("base").render(o=obj) == ""
        assert env.get_template("child").render(o=obj) == ""
        assert env.get_template("macro").render(o=obj) == ""
        assert obj.calls == []

    def test_async_filter_uses_policy(self, run_async_fn):
        env = SandboxedEnvironment(enable_async=True)
        obj = PropertyRecorder()
        t = env.from_string("{{ [o]|map(attribute='_private_prop')|list }}")

        async def run():
            return await t.render_async(o=obj)

        assert run_async_fn(run) == "[Undefined]"
        assert obj.calls == []

    def test_plain_environment_behavior_unchanged(self):
        env = Environment()
        obj = PropertyRecorder()
        # Outside the sandbox, private attributes are accessible and
        # descriptors are executed.
        assert (
            env.from_string("{{ o._private_prop }}").render(o=obj) == "private-value"
        )
        assert (
            env.from_string("{{ o|attr('_private_prop') }}").render(o=obj)
            == "private-value"
        )
        assert obj.calls == ["_private_prop", "_private_prop"]
        # A missing attribute is undefined, not an AttributeError.
        assert (
            env.from_string("{{ o|attr('missing') is undefined }}").render(o=obj)
            == "True"
        )

    def test_plain_environment_descriptor_error_falls_back_to_item(self):
        env = Environment()
        obj = CollisionDict(_private="dict-private", public="dict-public")
        assert env.from_string("{{ o._private }}").render(o=obj) == "dict-private"
        assert env.from_string("{{ o.public }}").render(o=obj) == "dict-public"

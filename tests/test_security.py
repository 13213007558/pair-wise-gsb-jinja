import pytest
from markupsafe import escape

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import Undefined
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


class RecordingObject:
    """Records descriptor executions in ``log`` so tests can assert the
    order and count of attribute accesses."""

    def __init__(self, log):
        self.log = log

    @property
    def allowed(self):
        self.log.append(("exec", "allowed"))
        return "A"

    @property
    def _denied(self):
        self.log.append(("exec", "_denied"))
        return "D"

    @property
    def raises_attr_error(self):
        raise AttributeError("raises_attr_error")

    @property
    def raises_value_error(self):
        raise ValueError("raises_value_error")


class CallableDescriptor:
    def __init__(self):
        self.get_count = 0

    def __get__(self, obj, objtype=None):
        self.get_count += 1
        return self

    def __call__(self):
        return "called"


class DescriptorHolder:
    allowed_desc = CallableDescriptor()
    _denied_desc = CallableDescriptor()


class AttrAndItem:
    """Has an attribute and a dict-style item with the same name."""

    def __init__(self):
        self.same = "attr-value"

    def __getitem__(self, key):
        if key in {"same", "only_item"}:
            return f"item-{key}"
        raise KeyError(key)


class RecordingPolicyEnv(SandboxedEnvironment):
    """Records the object and attribute name seen by the policy."""

    def __init__(self, log, denied=(), **kwargs):
        super().__init__(**kwargs)
        self._log = log
        self._denied = set(denied)

    def is_safe_attribute(self, obj, attr, value):
        self._log.append(("policy", attr))
        if attr in self._denied:
            return False
        return super().is_safe_attribute(obj, attr, value)


class TestUnifiedAttributePolicy:
    def test_allowed_attribute_policy_before_execution(self):
        log = []
        env = RecordingPolicyEnv(log)
        obj = RecordingObject(log)
        assert env.from_string("{{ obj.allowed }}").render(obj=obj) == "A"
        assert log == [("policy", "allowed"), ("exec", "allowed")]

    def test_denied_attribute_not_executed(self):
        log = []
        env = RecordingPolicyEnv(log)
        obj = RecordingObject(log)
        assert env.from_string("{{ obj._denied }}").render(obj=obj) == ""
        assert log == [("policy", "_denied")]

    def test_denied_attribute_undefined_raises_security_error(self):
        env = SandboxedEnvironment()
        obj = RecordingObject([])
        out = env.getattr(obj, "_denied")
        assert isinstance(out, Undefined)
        pytest.raises(
            SecurityError, env.from_string("{{ obj._denied() }}").render, obj=obj
        )

    def test_no_cached_policy_decision_between_renders(self):
        log = []
        env = RecordingPolicyEnv(log)
        obj = RecordingObject(log)
        t = env.from_string("{{ obj.allowed }}")
        assert t.render(obj=obj) == "A"
        assert t.render(obj=obj) == "A"
        execs = [entry for entry in log if entry[0] == "exec"]
        assert execs == [("exec", "allowed"), ("exec", "allowed")]

    def test_custom_denied_descriptor_not_executed(self):
        log = []
        env = RecordingPolicyEnv(log, denied={"allowed"})
        obj = RecordingObject(log)
        assert env.from_string("{{ obj.allowed }}").render(obj=obj) == ""
        assert ("exec", "allowed") not in log

    @pytest.mark.parametrize(
        "template",
        [
            "{{ obj|attr('_denied') }}",
            "{{ obj._denied }}",
            "{{ obj['_denied'] }}",
            "{{ [obj]|map(attribute='_denied')|list }}",
            "{{ [obj]|selectattr('_denied')|list }}",
        ],
    )
    def test_all_paths_deny_private_attribute(self, template):
        log = []
        env = SandboxedEnvironment()
        obj = RecordingObject(log)
        out = env.from_string(template).render(obj=obj)
        assert out in ("", "[Undefined]", "[]")
        assert log == []

    def test_attr_filter_allow_deny_undefined(self):
        env = SandboxedEnvironment()
        obj = RecordingObject([])
        t = env.from_string(
            "{{ obj|attr('allowed') is defined }}"
            "{{ obj|attr('_denied') is defined }}"
            "{{ obj|attr('missing') is defined }}"
        )
        assert t.render(obj=obj) == "TrueFalseFalse"

    def test_attr_filter_missing_returns_undefined(self):
        env = SandboxedEnvironment()
        obj = RecordingObject([])
        out = env.from_string("{{ obj|attr('missing') }}").render(obj=obj)
        assert out == ""
        assert isinstance(env.getattr(obj, "missing"), Undefined)

    def test_chained_attr_filter_checked_each_step(self):
        log = []
        env = RecordingPolicyEnv(log)
        obj = RecordingObject(log)
        obj.sub = obj
        t = env.from_string("{{ obj|attr('sub')|attr('allowed') }}")
        assert t.render(obj=obj) == "A"
        assert ("policy", "sub") in log
        assert ("policy", "allowed") in log
        t = env.from_string("{{ obj|attr('sub')|attr('_denied') }}")
        assert t.render(obj=obj) == ""
        assert ("exec", "_denied") not in log

    def test_item_and_attribute_same_name(self):
        for env in Environment(), SandboxedEnvironment():
            obj = AttrAndItem()
            assert env.from_string("{{ obj['same'] }}").render(obj=obj) == "item-same"
            assert env.from_string("{{ obj.same }}").render(obj=obj) == "attr-value"
            assert (
                env.from_string("{{ obj|attr('same') }}").render(obj=obj)
                == "attr-value"
            )
            assert (
                env.from_string("{{ [obj]|map(attribute='same')|first }}").render(
                    obj=obj
                )
                == "item-same"
            )
            assert (
                env.from_string("{{ obj.only_item }}").render(obj=obj)
                == "item-only_item"
            )

    def test_callable_descriptor(self):
        env = SandboxedEnvironment()
        holder = DescriptorHolder()
        assert env.from_string("{{ holder.allowed_desc() }}").render(holder=holder) == (
            "called"
        )
        assert DescriptorHolder.__dict__["allowed_desc"].get_count == 1
        assert env.from_string("{{ holder._denied_desc }}").render(holder=holder) == ""
        assert DescriptorHolder.__dict__["_denied_desc"].get_count == 0

    def test_descriptor_exceptions(self):
        env = SandboxedEnvironment()
        obj = RecordingObject([])
        pytest.raises(
            ValueError,
            env.from_string("{{ obj.raises_value_error }}").render,
            obj=obj,
        )
        assert env.from_string("{{ obj.raises_attr_error }}").render(obj=obj) == ""

    def test_macro_and_inheritance_do_not_bypass(self):
        log = []
        env = RecordingPolicyEnv(
            log,
            loader=DictLoader(
                {
                    "base": "{% block content %}{% endblock %}",
                    "child": (
                        "{% extends 'base' %}{% block content %}"
                        "{% macro show(x) %}"
                        "{{ x.allowed }}{{ x._denied }}"
                        "{% endmacro %}"
                        "{{ show(obj) }}"
                        "{% endblock %}"
                    ),
                }
            ),
        )
        obj = RecordingObject(log)
        assert env.get_template("child").render(obj=obj) == "A"
        assert ("exec", "allowed") in log
        assert ("exec", "_denied") not in log

    def test_async_filters_do_not_bypass(self, run_async_fn):
        log = []
        env = RecordingPolicyEnv(log, enable_async=True)
        obj = RecordingObject(log)

        async def test():
            t = env.from_string(
                "{{ [obj]|map(attribute='allowed')|list }}"
                "{{ [obj]|map(attribute='_denied')|list }}"
                "{{ [obj]|selectattr('_denied')|list }}"
            )
            return await t.render_async(obj=obj)

        assert run_async_fn(test) == "['A'][Undefined][]"
        assert ("exec", "allowed") in log
        assert ("exec", "_denied") not in log

    def test_policy_sees_stable_object_and_name(self):
        log = []
        env = RecordingPolicyEnv(log)
        obj = RecordingObject(log)
        env.from_string("{{ obj.allowed }}").render(obj=obj)
        env.from_string("{{ obj|attr('allowed') }}").render(obj=obj)
        env.from_string("{{ [obj]|map(attribute='allowed')|list }}").render(obj=obj)
        policy_entries = [entry for entry in log if entry[0] == "policy"]
        assert policy_entries == [("policy", "allowed")] * 3

    def test_plain_environment_behavior_unchanged(self):
        log = []
        env = Environment()
        obj = RecordingObject(log)
        assert env.from_string("{{ obj.allowed }}").render(obj=obj) == "A"
        assert env.from_string("{{ obj._denied }}").render(obj=obj) == "D"
        assert ("exec", "allowed") in log
        assert ("exec", "_denied") in log
        assert env.from_string("{{ obj|attr('missing') }}").render(obj=obj) == ""
        assert isinstance(env.getattr(obj, "missing"), Undefined)

    def test_attr_filter(self) -> None:
        env = SandboxedEnvironment()
        t = env.from_string(
            """{{ "{0.__call__.__builtins__[__import__]}"
                  | attr("format")(not_here) }}"""
        )

        with pytest.raises(SecurityError):
            t.render()

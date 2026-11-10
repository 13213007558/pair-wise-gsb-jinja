"""Regression tests proving that no template-level access path can reach an
attribute or callable without going through the sandbox guards.

Every way a template can obtain a value is exercised against the same probe
object:

* dot access            ``obj.attr``
* subscript access      ``obj["attr"]``
* the ``attr`` filter   ``obj|attr("attr")``
* ``map(attribute=...)``
* ``selectattr(...)``

The probe object simultaneously exposes a normal attribute, an attribute the
custom guard rejects, a dynamic (``__getattr__``) attribute in both allowed and
rejected variants, an ``unsafe`` marked method and plain mapping items.
Counters on a counting environment record exactly when ``is_safe_attribute``,
``is_safe_callable`` and ``wrap_str_format`` run, so the tests assert both that
the guards run and that rejected values never leak (and rejected methods never
execute). Mapping keys follow their own rule and must not be conflated with
object attribute checks.
"""

import typing as t

import pytest

from jinja2.async_utils import auto_aiter
from jinja2.exceptions import SecurityError
from jinja2.sandbox import SandboxedEnvironment
from jinja2.sandbox import unsafe

#: Attributes the custom guard is configured to reject, regardless of how the
#: template tries to reach them.
REJECTED_ATTRIBUTES = frozenset({"secret", "dynamic_secret"})

#: Distinct markers so a leaked value would immediately fail the assertions.
SAFE_MARKER = "SAFE-ATTRIBUTE-VALUE"
SECRET_MARKER = "SECRET-ATTRIBUTE-VALUE"
DYNAMIC_SAFE_MARKER = "DYNAMIC-SAFE-VALUE"
DYNAMIC_SECRET_MARKER = "DYNAMIC-SECRET-VALUE"
DANGEROUS_MARKER = "DANGEROUS-METHOD-RESULT"
SAFE_METHOD_MARKER = "SAFE-METHOD-RESULT"
MAP_SECRET_MARKER = "MAPPING-SECRET-VALUE"


class ProbeObject:
    """Object with every kind of member the sandbox must distinguish."""

    safe_attr = SAFE_MARKER
    secret = SECRET_MARKER

    def __init__(self) -> None:
        self.dangerous_calls = 0
        self.safe_method_calls = 0
        self.dynamic_lookups: list[str] = []

    def safe_method(self) -> str:
        self.safe_method_calls += 1
        return SAFE_METHOD_MARKER

    @unsafe
    def dangerous(self) -> str:
        # If a guard is bypassed this counter increments and the marker leaks.
        self.dangerous_calls += 1
        return DANGEROUS_MARKER

    def __getattr__(self, name: str) -> str:
        # Only reached for names not found through normal attribute lookup.
        self.dynamic_lookups.append(name)

        if name == "dynamic_safe":
            return DYNAMIC_SAFE_MARKER

        if name == "dynamic_secret":
            return DYNAMIC_SECRET_MARKER

        raise AttributeError(name)


class CountingSandbox(SandboxedEnvironment):
    """Sandbox that records every guard decision it makes."""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.attribute_guard_calls: list[tuple[str, str]] = []
        self.callable_guard_calls = 0
        self.format_wrap_calls = 0

    def is_safe_attribute(self, obj: t.Any, attr: str, value: t.Any) -> bool:
        self.attribute_guard_calls.append((type(obj).__name__, attr))
        return attr not in REJECTED_ATTRIBUTES and super().is_safe_attribute(
            obj, attr, value
        )

    def is_safe_callable(self, obj: t.Any) -> bool:
        self.callable_guard_calls += 1
        return super().is_safe_callable(obj)

    def wrap_str_format(self, value: t.Any) -> t.Any:
        self.format_wrap_calls += 1
        return super().wrap_str_format(value)


@pytest.fixture
def probe() -> ProbeObject:
    return ProbeObject()


@pytest.fixture
def mapping() -> dict[str, str]:
    # The same key name as a rejected object attribute must stay accessible
    # through mapping rules.
    return {"secret": MAP_SECRET_MARKER, "safe_key": SAFE_MARKER}


def render(template: str, context: dict[str, t.Any], *, enable_async: bool = False):
    env = CountingSandbox(enable_async=enable_async)
    rendered = env.from_string(template)

    if enable_async:
        import asyncio

        return env, asyncio.run(rendered.render_async(**context))

    return env, rendered.render(**context)


def render_async(runner: t.Any, template: str, context: dict[str, t.Any]):
    env = CountingSandbox(enable_async=True)
    rendered = env.from_string(template)

    async def run() -> str:
        return await rendered.render_async(**context)

    return env, runner(run)


# Real template syntax used for each entry point. Each case accesses both an
# allowed and a rejected attribute so a single render proves both directions.
ACCESS_CASES = [
    pytest.param(
        ("dot", "{{ obj.safe_attr }}|{{ obj.secret }}"),
        id="dot-access",
    ),
    pytest.param(
        ("subscript", '{{ obj["safe_attr"] }}|{{ obj["secret"] }}'),
        id="subscript-access",
    ),
    pytest.param(
        ("attr-filter", '{{ obj|attr("safe_attr") }}|{{ obj|attr("secret") }}'),
        id="attr-filter",
    ),
    pytest.param(
        (
            "map-attribute",
            "{{ [obj]|map(attribute='safe_attr')|join('~') }}"
            "|{{ [obj]|map(attribute='secret')|join('~') }}",
        ),
        id="map-attribute",
    ),
    pytest.param(
        (
            "selectattr",
            "{{ [obj]|selectattr('safe_attr')|list|length }}"
            "|{{ [obj]|selectattr('secret')|list|length }}",
        ),
        id="selectattr",
    ),
]


class TestAttributeGuardCoverage:
    @pytest.mark.parametrize("case", ACCESS_CASES)
    def test_every_entry_runs_attribute_guard(self, probe, case):
        label, template = case
        env, output = render(template, {"obj": probe})

        # The allowed member is visible; the rejected one renders as empty.
        assert SECRET_MARKER not in output
        assert ("ProbeObject", "safe_attr") in env.attribute_guard_calls
        assert ("ProbeObject", "secret") in env.attribute_guard_calls

        if label == "selectattr":
            # Allowed (truthy) attribute keeps the item, rejected drops it.
            assert output == "1|0"
        elif label == "map-attribute":
            # The rejected member is materialized as an Undefined, whose
            # string form is empty, rather than exposing the sensitive value.
            assert output == SAFE_MARKER + "|"
        else:
            assert output == SAFE_MARKER + "|"

    @pytest.mark.parametrize("case", ACCESS_CASES)
    def test_every_entry_runs_attribute_guard_async(self, run_async_fn, probe, case):
        label, template = case
        env, output = render_async(run_async_fn, template, {"obj": probe})

        assert SECRET_MARKER not in output
        assert ("ProbeObject", "safe_attr") in env.attribute_guard_calls
        assert ("ProbeObject", "secret") in env.attribute_guard_calls

        if label == "selectattr":
            assert output == "1|0"
        else:
            assert output == SAFE_MARKER + "|"

    @pytest.mark.parametrize(
        "template",
        [
            "{{ obj.dynamic_safe }}|{{ obj.dynamic_secret }}",
            '{{ obj|attr("dynamic_safe") }}|{{ obj|attr("dynamic_secret") }}',
            "{{ [obj]|map(attribute='dynamic_safe')|join('~') }}"
            "|{{ [obj]|map(attribute='dynamic_secret')|join('~') }}",
        ],
    )
    def test_dynamic_attributes_are_guarded(self, probe, template):
        env, output = render(template, {"obj": probe})

        assert output == DYNAMIC_SAFE_MARKER + "|"
        assert DYNAMIC_SECRET_MARKER not in output
        assert ("ProbeObject", "dynamic_safe") in env.attribute_guard_calls
        assert ("ProbeObject", "dynamic_secret") in env.attribute_guard_calls
        # The dynamic lookup itself happened, but the guard hid the result.
        assert "dynamic_secret" in probe.dynamic_lookups

    def test_dynamic_attributes_are_guarded_async(self, run_async_fn, probe):
        template = '{{ obj|attr("dynamic_safe") }}|{{ obj|attr("dynamic_secret") }}'
        env, output = render_async(run_async_fn, template, {"obj": probe})

        assert output == DYNAMIC_SAFE_MARKER + "|"
        assert DYNAMIC_SECRET_MARKER not in output
        assert ("ProbeObject", "dynamic_secret") in env.attribute_guard_calls

    @pytest.mark.parametrize(
        "template",
        [
            "{{ obj.secret }}",
            '{{ obj["secret"] }}',
            '{{ obj|attr("secret") }}',
            "{{ [obj]|map(attribute='secret')|first }}",
        ],
    )
    def test_rejected_access_returns_undefined_not_value(self, probe, template):
        env = CountingSandbox()
        result = env.from_string(template).render(obj=probe)

        assert result == ""
        assert SECRET_MARKER not in result
        assert ("ProbeObject", "secret") in env.attribute_guard_calls

    def test_rejected_access_undefined_carries_security_error(self, probe):
        env = CountingSandbox()

        # Plain rendering stays inert: the Undefined stringifies to empty.
        assert env.from_string("{{ obj.secret }}").render(obj=probe) == ""

        # Forcing the Undefined through an operation raises the SecurityError
        # it carries, without ever producing the sensitive value.
        with pytest.raises(SecurityError):
            env.from_string("{{ obj.secret + '' }}").render(obj=probe)

    def test_async_iterable_uses_async_filter_path(self, run_async_fn, probe):
        # An async iterator forces do_map/do_selectattr onto their async branch.
        env, output = render_async(
            run_async_fn,
            "{{ objs|map(attribute='secret')|list }}",
            {"objs": auto_aiter([probe])},
        )

        assert SECRET_MARKER not in output
        assert output == "[Undefined]"
        assert env.attribute_guard_calls == [("ProbeObject", "secret")]

        env, count = render_async(
            run_async_fn,
            "{{ objs|selectattr('secret')|list|length }}",
            {"objs": auto_aiter([ProbeObject()])},
        )
        assert count == "0"
        assert env.attribute_guard_calls == [("ProbeObject", "secret")]


class TestMappingKeysAreSeparateRule:
    def test_mapping_key_does_not_invoke_attribute_guard(self, mapping):
        env, output = render(
            "{{ data.secret }}|{{ data['secret'] }}|{{ data.safe_key }}",
            {"data": mapping},
        )

        assert output == "|".join([MAP_SECRET_MARKER, MAP_SECRET_MARKER, SAFE_MARKER])
        # Mapping item access must never be treated as an object attribute.
        assert env.attribute_guard_calls == []

    def test_mapping_key_does_not_invoke_attribute_guard_async(
        self, run_async_fn, mapping
    ):
        env, output = render_async(
            run_async_fn,
            "{{ data.secret }}|{{ data['secret'] }}",
            {"data": mapping},
        )

        assert MAP_SECRET_MARKER in output
        assert env.attribute_guard_calls == []

    def test_object_attribute_and_mapping_key_differ(self, probe, mapping):
        env, output = render(
            "{{ obj.secret }}|{{ data.secret }}",
            {"obj": probe, "data": mapping},
        )

        # Object attribute is blocked by is_safe_attribute; the same-named
        # mapping key remains readable.
        assert output == "|" + MAP_SECRET_MARKER
        assert env.attribute_guard_calls == [("ProbeObject", "secret")]


class TestCallableGuardCoverage:
    @pytest.mark.parametrize(
        "template",
        [
            "{{ obj.dangerous() }}",
            '{{ (obj|attr("dangerous"))() }}',
            '{{ obj["dangerous"]() }}',
            "{% for method in [obj]|map(attribute='dangerous') %}"
            "{{ method() }}{% endfor %}",
        ],
    )
    def test_unsafe_callable_rejected_at_every_entry(self, probe, template):
        env = CountingSandbox()
        with pytest.raises(SecurityError):
            env.from_string(template).render(obj=probe)

        # The rejected method must never actually execute, before or after
        # the error is raised.
        assert probe.dangerous_calls == 0
        assert DANGEROUS_MARKER not in str(env.attribute_guard_calls)
        assert env.callable_guard_calls >= 1

    def test_unsafe_callable_rejected_at_every_entry_async(self, run_async_fn, probe):
        env = CountingSandbox(enable_async=True)
        template = "{{ obj.dangerous() }}"
        rendered = env.from_string(template)

        async def run():
            await rendered.render_async(obj=probe)

        with pytest.raises(SecurityError):
            run_async_fn(run)

        assert probe.dangerous_calls == 0
        assert env.callable_guard_calls == 1

    def test_rejected_attribute_call_fails_without_invoking(self, probe):
        # A rejected plain attribute rendered as Undefined; calling it raises
        # the SecurityError carried by that Undefined, without leaking the
        # underlying value.
        env = CountingSandbox()
        with pytest.raises(SecurityError):
            env.from_string("{{ obj.secret() }}").render(obj=probe)

        assert SECRET_MARKER not in env.from_string("{{ obj.secret }}").render(
            obj=probe
        )

    @pytest.mark.parametrize(
        "template",
        [
            "{{ obj.safe_method() }}",
            '{{ (obj|attr("safe_method"))() }}',
            '{{ obj["safe_method"]() }}',
        ],
    )
    def test_safe_callable_still_works(self, probe, template):
        env, output = render(template, {"obj": probe})

        assert output == SAFE_METHOD_MARKER
        assert probe.safe_method_calls == 1
        assert env.callable_guard_calls == 1
        assert ("ProbeObject", "safe_method") in env.attribute_guard_calls

    def test_selectattr_does_not_invoke_unsafe_method(self, probe):
        # The bound method passes the attribute guard (its name is not
        # rejected) and is truthy, so the item is selected; the decisive
        # property is that merely selecting it never executes the method.
        env, output = render(
            "{{ [obj]|selectattr('dangerous')|list|length }}", {"obj": probe}
        )

        assert output == "1"
        assert probe.dangerous_calls == 0
        assert DANGEROUS_MARKER not in output
        assert ("ProbeObject", "dangerous") in env.attribute_guard_calls


# Every entry point that can hand back a bound str.format/format_map must hand
# back the sandboxed wrapper instead of the raw builtin method.
FORMAT_ACCESS_TEMPLATES = [
    "{{ text.format(42) }}",
    '{{ text["format"](42) }}',
    '{{ text|attr("format")(42) }}',
    "{% for bound in [text]|map(attribute='format') %}" "{{ bound(42) }}{% endfor %}",
]

FORMAT_MAP_ACCESS_TEMPLATES = [
    "{{ text.format_map(data) }}",
    '{{ text["format_map"](data) }}',
    '{{ text|attr("format_map")(data) }}',
    "{% for bound in [text]|map(attribute='format_map') %}"
    "{{ bound(data) }}{% endfor %}",
]


class TestStringFormatWrapping:
    @pytest.mark.parametrize("template", FORMAT_ACCESS_TEMPLATES)
    def test_format_bound_method_is_wrapped(self, template):
        env, output = render(template, {"text": "v={0}"})

        assert output == "v=42"
        assert env.format_wrap_calls == 1
        # The wrapper itself still passes through the callable guard.
        assert env.callable_guard_calls == 1

    @pytest.mark.parametrize("template", FORMAT_MAP_ACCESS_TEMPLATES)
    def test_format_map_bound_method_is_wrapped(self, template):
        env, output = render(template, {"text": "{x}", "data": {"x": 42}})

        assert output == "42"
        assert env.format_wrap_calls == 1
        assert env.callable_guard_calls == 1

    @pytest.mark.parametrize(
        "template",
        [
            '{{ "{0.__class__}".format(42) }}',
            '{{ "{0.__class__}"|attr("format")(42) }}',
            '{{ ("{0.__class__}"|attr("format"))(42) }}',
            "{% for bound in ['{0.__class__}']|map(attribute='format') %}"
            "{{ bound(42) }}{% endfor %}",
        ],
    )
    def test_wrapped_format_still_sandboxes_fields(self, template):
        env, output = render(template, {})

        assert output == ""
        assert "<class" not in output
        assert env.format_wrap_calls >= 1

    @pytest.mark.parametrize(
        "template",
        [
            '{{ "{x.__class__}".format_map(data) }}',
            '{{ "{x.__class__}"|attr("format_map")(data) }}',
        ],
    )
    def test_wrapped_format_map_still_sandboxes_fields(self, template):
        env, output = render(template, {"data": {"x": 42}})

        assert output == ""
        assert "<class" not in output

    def test_wrapped_format_works_through_selectattr(self):
        # selectattr retrieves the bound format method for its truthiness test;
        # rendering it afterwards must still use the wrapped variant.
        env, output = render(
            "{% for item in [text]|selectattr('format') %}"
            "{{ item.format(42) }}{% endfor %}",
            {"text": "v={0}"},
        )

        assert output == "v=42"
        assert env.format_wrap_calls == 2

    @pytest.mark.parametrize("template", FORMAT_ACCESS_TEMPLATES)
    def test_format_wrapping_async(self, run_async_fn, template):
        env, output = render_async(run_async_fn, template, {"text": "v={0}"})

        assert output == "v=42"
        assert env.format_wrap_calls == 1
        assert env.callable_guard_calls == 1

    def test_markup_format_uses_escape_formatter(self):
        from markupsafe import Markup

        env, output = render(
            "{{ text.format(value) }}",
            {"text": Markup("v={0}"), "value": "<b>"},
        )

        assert output == "v=&lt;b&gt;"
        assert env.format_wrap_calls == 1


class TestPositiveCases:
    """Allowed functionality must remain fully usable."""

    def test_plain_mapping_access(self, mapping):
        env, output = render(
            "{% for key in mapping|sort %}{{ key }}={{ mapping[key] }};{% endfor %}",
            {"mapping": mapping},
        )

        assert MAP_SECRET_MARKER in output
        assert SAFE_MARKER in output
        assert env.attribute_guard_calls == []

    def test_allowed_attributes_and_methods(self, probe):
        env, output = render(
            "{{ obj.safe_attr }}-{{ obj.safe_method() }}", {"obj": probe}
        )

        assert output == f"{SAFE_MARKER}-{SAFE_METHOD_MARKER}"
        assert probe.safe_method_calls == 1

    def test_allowed_dynamic_attribute(self, probe):
        env, output = render("{{ obj.dynamic_safe }}", {"obj": probe})

        assert output == DYNAMIC_SAFE_MARKER
        assert ("ProbeObject", "dynamic_safe") in env.attribute_guard_calls

    def test_safe_formatting_not_disabled(self):
        # Pass the format string as a variable so constant folding does not
        # invoke the guards during compilation; the count then reflects the
        # single runtime access precisely.
        env, output = render("{{ text.format(42, name='n') }}", {"text": "{name}: {0}"})

        assert output == "n: 42"
        assert env.format_wrap_calls == 1

        env, output = render(
            "{{ text.format_map(data) }}",
            {"text": "{x.y}", "data": {"x": {"y": "deep"}}},
        )
        assert output == "deep"
        assert env.format_wrap_calls == 1

    def test_map_and_selectattr_positive_path(self):
        items = [ProbeObject(), ProbeObject()]
        env, output = render(
            "{{ items|map(attribute='safe_attr')|join(',') }}",
            {"items": items},
        )

        assert output == ",".join([SAFE_MARKER, SAFE_MARKER])

        env, output = render(
            "{{ items|selectattr('safe_attr')|list|length }}", {"items": items}
        )
        assert output == "2"

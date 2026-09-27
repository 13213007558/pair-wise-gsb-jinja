"""Extension API for adding custom tags and behavior."""

import pprint
import re
import typing as t
from collections.abc import Mapping

from markupsafe import Markup

from . import defaults
from . import nodes
from .environment import Environment
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .runtime import concat  # type: ignore
from .runtime import Context
from .runtime import Undefined
from .utils import import_string
from .utils import pass_context

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .lexer import Token
    from .lexer import TokenStream
    from .parser import Parser

    class _TranslationsBasic(te.Protocol):
        def gettext(self, message: str) -> str: ...

        def ngettext(self, singular: str, plural: str, n: int) -> str:
            pass

    class _TranslationsContext(_TranslationsBasic):
        def pgettext(self, context: str, message: str) -> str: ...

        def npgettext(
            self, context: str, singular: str, plural: str, n: int
        ) -> str: ...

    _SupportedTranslations = _TranslationsBasic | _TranslationsContext


# I18N functions available in Jinja templates. If the I18N library
# provides ugettext, it will be assigned to gettext.
DOMAIN_GETTEXT_FUNCTIONS: tuple[str, ...] = (
    "dgettext",
    "dngettext",
    "dpgettext",
    "dnpgettext",
)

GETTEXT_FUNCTIONS: tuple[str, ...] = (
    "_",
    "gettext",
    "ngettext",
    "pgettext",
    "npgettext",
    *DOMAIN_GETTEXT_FUNCTIONS,
)
_ws_re = re.compile(r"\s*\n\s*")

_CatalogCallables = dict[str, t.Callable[..., t.Any] | None]
_Catalogs = dict[tuple[str | None, str], _CatalogCallables]
_missing = object()


class _MappingTranslations:
    def __init__(self, messages: Mapping[t.Any, str]) -> None:
        self.messages = messages

    def gettext(self, message: str) -> str:
        return self.messages.get(message, message)

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        message = singular if n == 1 else plural
        return self.messages.get(message, message)

    def pgettext(self, context: str, message: str) -> str:
        return self.messages.get(
            (context, message),
            self.messages.get(f"{context}\x04{message}", message),
        )

    def npgettext(self, context: str, singular: str, plural: str, n: int) -> str:
        message = singular if n == 1 else plural
        return self.messages.get(
            (context, message),
            self.messages.get(f"{context}\x04{message}", message),
        )


class Extension:
    """Extensions can be used to add extra functionality to the Jinja template
    system at the parser level.  Custom extensions are bound to an environment
    but may not store environment specific data on `self`.  The reason for
    this is that an extension can be bound to another environment (for
    overlays) by creating a copy and reassigning the `environment` attribute.

    As extensions are created by the environment they cannot accept any
    arguments for configuration.  One may want to work around that by using
    a factory function, but that is not possible as extensions are identified
    by their import name.  The correct way to configure the extension is
    storing the configuration values on the environment.  Because this way the
    environment ends up acting as central configuration storage the
    attributes may clash which is why extensions have to ensure that the names
    they choose for configuration are not too generic.  ``prefix`` for example
    is a terrible name, ``fragment_cache_prefix`` on the other hand is a good
    name as includes the name of the extension (fragment cache).
    """

    identifier: t.ClassVar[str]

    def __init_subclass__(cls) -> None:
        cls.identifier = f"{cls.__module__}.{cls.__name__}"

    #: if this extension parses this is the list of tags it's listening to.
    tags: set[str] = set()

    #: the priority of that extension.  This is especially useful for
    #: extensions that preprocess values.  A lower value means higher
    #: priority.
    #:
    #: .. versionadded:: 2.4
    priority = 100

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def bind(self, environment: Environment) -> "te.Self":
        """Create a copy of this extension bound to another environment."""
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.environment = environment
        return rv

    def preprocess(
        self, source: str, name: str | None, filename: str | None = None
    ) -> str:
        """This method is called before the actual lexing and can be used to
        preprocess the source.  The `filename` is optional.  The return value
        must be the preprocessed source.
        """
        return source

    def filter_stream(
        self, stream: "TokenStream"
    ) -> t.Union["TokenStream", t.Iterable["Token"]]:
        """It's passed a :class:`~jinja2.lexer.TokenStream` that can be used
        to filter tokens returned.  This method has to return an iterable of
        :class:`~jinja2.lexer.Token`\\s, but it doesn't have to return a
        :class:`~jinja2.lexer.TokenStream`.
        """
        return stream

    def parse(self, parser: "Parser") -> nodes.Node | list[nodes.Node]:
        """If any of the :attr:`tags` matched this method is called with the
        parser as first argument.  The token the parser stream is pointing at
        is the name token that matched.  This method has to return one or a
        list of multiple nodes.
        """
        raise NotImplementedError()

    def attr(self, name: str, lineno: int | None = None) -> nodes.ExtensionAttribute:
        """Return an attribute node for the current extension.  This is useful
        to pass constants on extensions to generated template code.

        ::

            self.attr('_my_attribute', lineno=lineno)
        """
        return nodes.ExtensionAttribute(self.identifier, name, lineno=lineno)

    def call_method(
        self,
        name: str,
        args: list[nodes.Expr] | None = None,
        kwargs: list[nodes.Keyword] | None = None,
        dyn_args: nodes.Expr | None = None,
        dyn_kwargs: nodes.Expr | None = None,
        lineno: int | None = None,
    ) -> nodes.Call:
        """Call a method of the extension.  This is a shortcut for
        :meth:`attr` + :class:`jinja2.nodes.Call`.
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = []
        return nodes.Call(
            self.attr(name, lineno=lineno),
            args,
            kwargs,
            dyn_args,
            dyn_kwargs,
            lineno=lineno,
        )


@pass_context
def _gettext_alias(
    __context: Context, *args: t.Any, **kwargs: t.Any
) -> t.Any | Undefined:
    return __context.call(__context.resolve("gettext"), *args, **kwargs)


def _make_new_gettext(func: t.Callable[[str], str]) -> t.Callable[..., str]:
    @pass_context
    def gettext(__context: Context, __string: str, **variables: t.Any) -> str:
        rv = __context.call(func, __string)
        if __context.eval_ctx.autoescape:
            rv = Markup(rv)
        # Always treat as a format string, even if there are no
        # variables. This makes translation strings more consistent
        # and predictable. This requires escaping
        return rv % variables  # type: ignore

    return gettext


def _make_new_ngettext(func: t.Callable[[str, str, int], str]) -> t.Callable[..., str]:
    @pass_context
    def ngettext(
        __context: Context,
        __singular: str,
        __plural: str,
        __num: int,
        **variables: t.Any,
    ) -> str:
        variables.setdefault("num", __num)
        rv = __context.call(func, __singular, __plural, __num)
        if __context.eval_ctx.autoescape:
            rv = Markup(rv)
        # Always treat as a format string, see gettext comment above.
        return rv % variables  # type: ignore

    return ngettext


def _make_new_pgettext(func: t.Callable[[str, str], str]) -> t.Callable[..., str]:
    @pass_context
    def pgettext(
        __context: Context, __string_ctx: str, __string: str, **variables: t.Any
    ) -> str:
        variables.setdefault("context", __string_ctx)
        rv = __context.call(func, __string_ctx, __string)

        if __context.eval_ctx.autoescape:
            rv = Markup(rv)

        # Always treat as a format string, see gettext comment above.
        return rv % variables  # type: ignore

    return pgettext


def _make_new_npgettext(
    func: t.Callable[[str, str, str, int], str],
) -> t.Callable[..., str]:
    @pass_context
    def npgettext(
        __context: Context,
        __string_ctx: str,
        __singular: str,
        __plural: str,
        __num: int,
        **variables: t.Any,
    ) -> str:
        variables.setdefault("context", __string_ctx)
        variables.setdefault("num", __num)
        rv = __context.call(func, __string_ctx, __singular, __plural, __num)

        if __context.eval_ctx.autoescape:
            rv = Markup(rv)

        # Always treat as a format string, see gettext comment above.
        return rv % variables  # type: ignore

    return npgettext


def _normalize_locale(locale: t.Any) -> str | None:
    if locale is None:
        return None

    return str(locale)


def _get_default_domain(environment: Environment) -> str:
    return str(getattr(environment, "i18n_default_domain", "messages"))


def _get_context_locale(context: Context) -> str | None:
    for key in (
        "_i18n_locale",
        "i18n_locale",
        "_locale",
        "locale",
        "LANGUAGE",
        "language",
    ):
        value = context.get(key, _missing)

        if value is not _missing and value is not None:
            return str(value)

    return _normalize_locale(getattr(context.environment, "i18n_default_locale", None))


def _get_domain_chain(environment: Environment, domain: str) -> t.Iterator[str]:
    fallbacks = t.cast(
        dict[str, tuple[str, ...]], getattr(environment, "i18n_domain_fallbacks", {})
    )
    seen: set[str] = set()

    def visit(current: str) -> t.Iterator[str]:
        if current in seen:
            return

        seen.add(current)
        yield current

        for fallback in fallbacks.get(current, ()):
            yield from visit(fallback)

    yield from visit(domain)


def _get_source_message(func_name: str, args: tuple[t.Any, ...]) -> str:
    if func_name == "gettext":
        return t.cast(str, args[0])

    if func_name == "ngettext":
        return t.cast(str, args[0] if args[2] == 1 else args[1])

    if func_name == "pgettext":
        return t.cast(str, args[1])

    return t.cast(str, args[1] if args[3] == 1 else args[2])


def _format_missing_translation(
    context: Context,
    func_name: str,
    args: tuple[t.Any, ...],
    variables: dict[str, t.Any],
) -> str:
    if variables and not context.environment.newstyle_gettext:  # type: ignore
        raise TypeError(f"{func_name}() got unexpected keyword arguments")

    rv = _get_source_message(func_name, args)

    if not context.environment.newstyle_gettext:  # type: ignore
        return rv

    variables = variables.copy()

    if func_name in {"pgettext", "npgettext"}:
        variables.setdefault("context", args[0])

    if func_name in {"ngettext", "npgettext"}:
        variables.setdefault("num", args[-1])

    value: str | Markup = rv

    if context.eval_ctx.autoescape:
        value = Markup(value)

    return value % variables


def _translate(
    context: Context,
    func_name: str,
    domain: str | None,
    args: tuple[t.Any, ...],
    variables: dict[str, t.Any],
) -> t.Any:
    environment = context.environment
    catalogs = t.cast(_Catalogs, getattr(environment, "i18n_catalogs", {}))
    locale = _get_context_locale(context)
    domain = _get_default_domain(environment) if domain is None else str(domain)

    locales = (locale,) if locale is None else (locale, None)

    for current_locale in locales:
        for current_domain in _get_domain_chain(environment, domain):
            catalog = catalogs.get((current_locale, current_domain))

            if catalog is None:
                continue

            func = catalog.get(func_name)

            if func is None:
                continue

            try:
                rv = context.call(func, *args, **variables)
            except KeyError:
                continue

            missing = _format_missing_translation(context, func_name, args, variables)

            if rv is not None and rv != missing:
                return rv

    return _format_missing_translation(context, func_name, args, variables)


def _make_registry_gettext(func_name: str, has_domain: bool) -> t.Callable[..., t.Any]:
    @pass_context
    def registry_gettext(__context: Context, *args: t.Any, **variables: t.Any) -> t.Any:
        domain = None

        if has_domain:
            domain = t.cast(str, args[0])
            args = args[1:]

        return _translate(__context, func_name, domain, args, variables)

    return registry_gettext


_registry_gettext = _make_registry_gettext("gettext", False)
_registry_ngettext = _make_registry_gettext("ngettext", False)
_registry_pgettext = _make_registry_gettext("pgettext", False)
_registry_npgettext = _make_registry_gettext("npgettext", False)
_registry_dgettext = _make_registry_gettext("gettext", True)
_registry_dngettext = _make_registry_gettext("ngettext", True)
_registry_dpgettext = _make_registry_gettext("pgettext", True)
_registry_dnpgettext = _make_registry_gettext("npgettext", True)


class InternationalizationExtension(Extension):
    """This extension adds gettext support to Jinja."""

    tags = {"trans"}

    # TODO: the i18n extension is currently reevaluating values in a few
    # situations.  Take this example:
    #   {% trans count=something() %}{{ count }} foo{% pluralize
    #     %}{{ count }} fooss{% endtrans %}
    # something is called twice here.  One time for the gettext value and
    # the other time for the n-parameter of the ngettext function.

    def __init__(self, environment: Environment) -> None:
        super().__init__(environment)
        environment.globals["_"] = _gettext_alias
        environment.extend(
            install_gettext_translations=self._install,
            install_null_translations=self._install_null,
            install_gettext_callables=self._install_callables,
            uninstall_gettext_translations=self._uninstall,
            extract_translations=self._extract,
            configure_gettext_domains=self._configure_domains,
            newstyle_gettext=False,
            i18n_catalogs={},
            i18n_default_domain="messages",
            i18n_default_domain_configured=False,
            i18n_default_locale=None,
            i18n_domain_fallbacks={},
        )

    def _install(
        self,
        translations: "_SupportedTranslations | Mapping[t.Any, str]",
        newstyle: bool | None = None,
        *,
        domain: str | None = None,
        locale: t.Any = None,
        default_domain: str | None = None,
        default_locale: t.Any = None,
        fallback_domains: t.Any = None,
        fallbacks: t.Any = None,
        default: bool = False,
    ) -> None:
        translation_obj: _SupportedTranslations

        if isinstance(translations, Mapping):
            translation_obj = _MappingTranslations(translations)
        else:
            translation_obj = translations

        # ugettext and ungettext are preferred in case the I18N library
        # is providing compatibility with older Python versions.
        gettext = getattr(translation_obj, "ugettext", None)
        if gettext is None:
            gettext = translation_obj.gettext
        ngettext = getattr(translation_obj, "ungettext", None)
        if ngettext is None:
            ngettext = translation_obj.ngettext

        pgettext = getattr(translation_obj, "pgettext", None)
        npgettext = getattr(translation_obj, "npgettext", None)
        self._install_callables(
            gettext,
            ngettext,
            newstyle=newstyle,
            pgettext=pgettext,
            npgettext=npgettext,
            domain=domain,
            locale=locale,
            default_domain=default_domain,
            default_locale=default_locale,
            fallback_domains=fallback_domains,
            fallbacks=fallbacks,
            default=default,
        )

    def _install_null(
        self,
        newstyle: bool | None = None,
        *,
        domain: str | None = None,
        locale: t.Any = None,
        default_domain: str | None = None,
        default_locale: t.Any = None,
        fallback_domains: t.Any = None,
        fallbacks: t.Any = None,
        default: bool = False,
    ) -> None:
        import gettext

        translations = gettext.NullTranslations()
        self._install_callables(
            gettext=translations.gettext,
            ngettext=translations.ngettext,
            newstyle=newstyle,
            pgettext=translations.pgettext,
            npgettext=translations.npgettext,
            domain=domain,
            locale=locale,
            default_domain=default_domain,
            default_locale=default_locale,
            fallback_domains=fallback_domains,
            fallbacks=fallbacks,
            default=default,
        )

    @staticmethod
    def _fallback_tuple(value: t.Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)

        return tuple(str(item) for item in value)

    def _add_domain_fallbacks(self, domain: str, fallbacks: t.Any) -> None:
        registry = t.cast(
            dict[str, tuple[str, ...]],
            self.environment.i18n_domain_fallbacks,  # type: ignore
        )
        registry[str(domain)] = self._fallback_tuple(fallbacks)

    def _configure_domains(
        self,
        default_domain: str | None = None,
        fallbacks: t.Any = None,
        default_locale: t.Any = None,
        domain_fallbacks: t.Any = None,
        fallback_domains: t.Any = None,
    ) -> None:
        if default_domain is not None:
            self.environment.i18n_default_domain = str(default_domain)  # type: ignore
            self.environment.i18n_default_domain_configured = True  # type: ignore

        if default_locale is not None:
            self.environment.i18n_default_locale = _normalize_locale(default_locale)  # type: ignore

        for config in (fallbacks, domain_fallbacks, fallback_domains):
            if config is None:
                continue

            if isinstance(config, Mapping):
                for domain, fallback_chain in config.items():
                    self._add_domain_fallbacks(str(domain), fallback_chain)
            else:
                self._add_domain_fallbacks(
                    _get_default_domain(self.environment), config
                )

    def _prepare_callables(
        self,
        gettext: t.Callable[[str], str],
        ngettext: t.Callable[[str, str, int], str],
        newstyle: bool | None,
        pgettext: t.Callable[[str, str], str] | None,
        npgettext: t.Callable[[str, str, str, int], str] | None,
    ) -> _CatalogCallables:
        if newstyle is not None:
            self.environment.newstyle_gettext = newstyle  # type: ignore

        prepared_gettext: t.Callable[..., t.Any] = gettext
        prepared_ngettext: t.Callable[..., t.Any] = ngettext
        prepared_pgettext = pgettext
        prepared_npgettext = npgettext

        if self.environment.newstyle_gettext:  # type: ignore
            prepared_gettext = _make_new_gettext(gettext)
            prepared_ngettext = _make_new_ngettext(ngettext)

            if pgettext is not None:
                prepared_pgettext = _make_new_pgettext(pgettext)

            if npgettext is not None:
                prepared_npgettext = _make_new_npgettext(npgettext)

        return {
            "gettext": prepared_gettext,
            "ngettext": prepared_ngettext,
            "pgettext": prepared_pgettext,
            "npgettext": prepared_npgettext,
        }

    def _install_registry_callables(self) -> None:
        self.environment.globals.update(
            gettext=_registry_gettext,
            ngettext=_registry_ngettext,
            pgettext=_registry_pgettext,
            npgettext=_registry_npgettext,
            dgettext=_registry_dgettext,
            dngettext=_registry_dngettext,
            dpgettext=_registry_dpgettext,
            dnpgettext=_registry_dnpgettext,
        )

    def _install_callables(
        self,
        gettext: t.Callable[[str], str],
        ngettext: t.Callable[[str, str, int], str],
        newstyle: bool | None = None,
        pgettext: t.Callable[[str, str], str] | None = None,
        npgettext: t.Callable[[str, str, str, int], str] | None = None,
        *,
        domain: str | None = None,
        locale: t.Any = None,
        default_domain: str | None = None,
        default_locale: t.Any = None,
        fallback_domains: t.Any = None,
        fallbacks: t.Any = None,
        default: bool = False,
    ) -> None:
        catalog = self._prepare_callables(
            gettext, ngettext, newstyle, pgettext, npgettext
        )
        domain_aware = (
            any(
                value is not None
                for value in (
                    domain,
                    locale,
                    default_domain,
                    default_locale,
                    fallback_domains,
                    fallbacks,
                )
            )
            or default
        )

        if not domain_aware:
            catalogs = t.cast(_Catalogs, self.environment.i18n_catalogs)  # type: ignore
            catalogs.clear()

            for key in DOMAIN_GETTEXT_FUNCTIONS:
                self.environment.globals.pop(key, None)

            self.environment.globals.update(catalog)
            return

        mapping_fallbacks = []
        local_fallbacks = []

        for config in (fallback_domains, fallbacks):
            if config is None:
                continue

            if isinstance(config, Mapping):
                mapping_fallbacks.append(config)
            else:
                local_fallbacks.append(config)

        for config in mapping_fallbacks:
            self._configure_domains(fallbacks=config)

        self._configure_domains(default_locale=default_locale)

        if default_domain is not None:
            self._configure_domains(default_domain=default_domain)

        if domain is not None:
            selected_domain = str(domain)
        else:
            selected_domain = _get_default_domain(self.environment)

        if default or (
            domain is not None and not self.environment.i18n_default_domain_configured  # type: ignore
        ):
            self._configure_domains(default_domain=selected_domain)

        for config in local_fallbacks:
            self._add_domain_fallbacks(selected_domain, config)

        catalogs = t.cast(_Catalogs, self.environment.i18n_catalogs)  # type: ignore
        catalogs[(_normalize_locale(locale), selected_domain)] = catalog
        self._install_registry_callables()

    def _uninstall(self, translations: "_SupportedTranslations | None" = None) -> None:
        catalogs = t.cast(_Catalogs, self.environment.i18n_catalogs)  # type: ignore
        catalogs.clear()

        for key in GETTEXT_FUNCTIONS[1:]:
            self.environment.globals.pop(key, None)

    def _extract(
        self,
        source: str | nodes.Template,
        gettext_functions: t.Sequence[str] = GETTEXT_FUNCTIONS,
    ) -> t.Iterator[tuple[int, str, str | None | tuple[str | None, ...]]]:
        if isinstance(source, str):
            source = self.environment.parse(source)
        return extract_from_ast(source, gettext_functions)

    def parse(self, parser: "Parser") -> nodes.Node | list[nodes.Node]:
        """Parse a translatable tag."""
        lineno = next(parser.stream).lineno

        context = None
        domain = None
        context_token = parser.stream.next_if("string")

        if context_token is not None:
            context = context_token.value

        # find all the variables referenced.  Additionally a variable can be
        # defined in the body of the trans block too, but this is checked at
        # a later state.
        plural_expr: nodes.Expr | None = None
        plural_expr_assignment: nodes.Assign | None = None
        num_called_num = False
        variables: dict[str, nodes.Expr] = {}
        trimmed = None
        while parser.stream.current.type != "block_end":
            if variables:
                parser.stream.expect("comma")
            else:
                parser.stream.skip_if("comma")

            # skip colon for python compatibility
            if parser.stream.skip_if("colon"):
                break

            if parser.stream.current.type == "string":
                if context is not None:
                    parser.fail(
                        "translatable context defined twice.",
                        parser.stream.current.lineno,
                        exc=TemplateAssertionError,
                    )

                context = next(parser.stream).value
                continue

            token = parser.stream.expect("name")

            if token.value == "domain" and parser.stream.current.type == "assign":
                if domain is not None:
                    parser.fail(
                        "translation domain defined twice.",
                        token.lineno,
                        exc=TemplateAssertionError,
                    )

                next(parser.stream)
                domain = parser.stream.expect("string").value
                continue

            if token.value == "context" and parser.stream.current.type == "assign":
                if context is not None:
                    parser.fail(
                        "translatable context defined twice.",
                        token.lineno,
                        exc=TemplateAssertionError,
                    )

                next(parser.stream)
                context = parser.stream.expect("string").value
                continue

            if token.value in variables:
                parser.fail(
                    f"translatable variable {token.value!r} defined twice.",
                    token.lineno,
                    exc=TemplateAssertionError,
                )

            # expressions
            if parser.stream.current.type == "assign":
                next(parser.stream)
                variables[token.value] = var = parser.parse_expression()
            elif trimmed is None and token.value in ("trimmed", "notrimmed"):
                trimmed = token.value == "trimmed"
                continue
            else:
                variables[token.value] = var = nodes.Name(token.value, "load")

            if plural_expr is None:
                if isinstance(var, nodes.Call):
                    plural_expr = nodes.Name("_trans", "load")
                    variables[token.value] = plural_expr
                    plural_expr_assignment = nodes.Assign(
                        nodes.Name("_trans", "store"), var
                    )
                else:
                    plural_expr = var
                num_called_num = token.value == "num"

        parser.stream.expect("block_end")

        plural = None
        have_plural = False
        referenced = set()

        # now parse until endtrans or pluralize
        singular_names, singular = self._parse_block(parser, True)
        if singular_names:
            referenced.update(singular_names)
            if plural_expr is None:
                plural_expr = nodes.Name(singular_names[0], "load")
                num_called_num = singular_names[0] == "num"

        # if we have a pluralize block, we parse that too
        if parser.stream.current.test("name:pluralize"):
            have_plural = True
            next(parser.stream)
            if parser.stream.current.type != "block_end":
                token = parser.stream.expect("name")
                if token.value not in variables:
                    parser.fail(
                        f"unknown variable {token.value!r} for pluralization",
                        token.lineno,
                        exc=TemplateAssertionError,
                    )
                plural_expr = variables[token.value]
                num_called_num = token.value == "num"
            parser.stream.expect("block_end")
            plural_names, plural = self._parse_block(parser, False)
            next(parser.stream)
            referenced.update(plural_names)
        else:
            next(parser.stream)

        # register free names as simple name expressions
        for name in referenced:
            if name not in variables:
                variables[name] = nodes.Name(name, "load")

        if not have_plural:
            plural_expr = None
        elif plural_expr is None:
            parser.fail("pluralize without variables", lineno)

        if trimmed is None:
            trimmed = self.environment.policies["ext.i18n.trimmed"]
        if trimmed:
            singular = self._trim_whitespace(singular)
            if plural:
                plural = self._trim_whitespace(plural)

        node = self._make_node(
            singular,
            plural,
            context,
            domain,
            variables,
            plural_expr,
            bool(referenced),
            num_called_num and have_plural,
        )
        node.set_lineno(lineno)
        if plural_expr_assignment is not None:
            return [plural_expr_assignment, node]
        else:
            return node

    def _trim_whitespace(self, string: str, _ws_re: t.Pattern[str] = _ws_re) -> str:
        return _ws_re.sub(" ", string.strip())

    def _parse_block(
        self, parser: "Parser", allow_pluralize: bool
    ) -> tuple[list[str], str]:
        """Parse until the next block tag with a given name."""
        referenced = []
        buf = []

        while True:
            if parser.stream.current.type == "data":
                buf.append(parser.stream.current.value.replace("%", "%%"))
                next(parser.stream)
            elif parser.stream.current.type == "variable_begin":
                next(parser.stream)
                name = parser.stream.expect("name").value
                referenced.append(name)
                buf.append(f"%({name})s")
                parser.stream.expect("variable_end")
            elif parser.stream.current.type == "block_begin":
                next(parser.stream)
                block_name = (
                    parser.stream.current.value
                    if parser.stream.current.type == "name"
                    else None
                )
                if block_name == "endtrans":
                    break
                elif block_name == "pluralize":
                    if allow_pluralize:
                        break
                    parser.fail(
                        "a translatable section can have only one pluralize section"
                    )
                elif block_name == "trans":
                    parser.fail(
                        "trans blocks can't be nested; did you mean `endtrans`?"
                    )
                parser.fail(
                    f"control structures in translatable sections are not allowed; "
                    f"saw `{block_name}`"
                )
            elif parser.stream.eos:
                parser.fail("unclosed translation block")
            else:
                raise RuntimeError("internal parser error")

        return referenced, concat(buf)

    def _make_node(
        self,
        singular: str,
        plural: str | None,
        context: str | None,
        domain: str | None,
        variables: dict[str, nodes.Expr],
        plural_expr: nodes.Expr | None,
        vars_referenced: bool,
        num_called_num: bool,
    ) -> nodes.Output:
        """Generates a useful node from the data provided."""
        newstyle = self.environment.newstyle_gettext  # type: ignore
        node: nodes.Expr

        # no variables referenced?  no need to escape for old style
        # gettext invocations only if there are vars.
        if not vars_referenced and not newstyle:
            singular = singular.replace("%%", "%")
            if plural:
                plural = plural.replace("%%", "%")

        func_name = "gettext"
        func_args: list[nodes.Expr] = [nodes.Const(singular)]

        if context is not None:
            func_args.insert(0, nodes.Const(context))
            func_name = f"p{func_name}"

        if plural_expr is not None:
            func_name = f"n{func_name}"
            func_args.extend((nodes.Const(plural), plural_expr))

        if domain is not None:
            func_args.insert(0, nodes.Const(domain))
            func_name = f"d{func_name}"

        node = nodes.Call(nodes.Name(func_name, "load"), func_args, [], None, None)

        # in case newstyle gettext is used, the method is powerful
        # enough to handle the variable expansion and autoescape
        # handling itself
        if newstyle:
            for key, value in variables.items():
                # the function adds that later anyways in case num was
                # called num, so just skip it.
                if num_called_num and key == "num":
                    continue
                node.kwargs.append(nodes.Keyword(key, value))

        # otherwise do that here
        else:
            # mark the return value as safe if we are in an
            # environment with autoescaping turned on
            node = nodes.MarkSafeIfAutoescape(node)
            if variables:
                node = nodes.Mod(
                    node,
                    nodes.Dict(
                        [
                            nodes.Pair(nodes.Const(key), value)
                            for key, value in variables.items()
                        ]
                    ),
                )
        return nodes.Output([node])


class ExprStmtExtension(Extension):
    """Adds a `do` tag to Jinja that works like the print statement just
    that it doesn't print the return value.
    """

    tags = {"do"}

    def parse(self, parser: "Parser") -> nodes.ExprStmt:
        node = nodes.ExprStmt(lineno=next(parser.stream).lineno)
        node.node = parser.parse_tuple()
        return node


class LoopControlExtension(Extension):
    """Adds break and continue to the template engine."""

    tags = {"break", "continue"}

    def parse(self, parser: "Parser") -> nodes.Break | nodes.Continue:
        token = next(parser.stream)
        if token.value == "break":
            return nodes.Break(lineno=token.lineno)
        return nodes.Continue(lineno=token.lineno)


class DebugExtension(Extension):
    """A ``{% debug %}`` tag that dumps the available variables,
    filters, and tests.

    .. code-block:: html+jinja

        <pre>{% debug %}</pre>

    .. code-block:: text

        {'context': {'cycler': <class 'jinja2.utils.Cycler'>,
                     ...,
                     'namespace': <class 'jinja2.utils.Namespace'>},
         'filters': ['abs', 'attr', 'batch', 'capitalize', 'center', 'count', 'd',
                     ..., 'urlencode', 'urlize', 'wordcount', 'wordwrap', 'xmlattr'],
         'tests': ['!=', '<', '<=', '==', '>', '>=', 'callable', 'defined',
                   ..., 'odd', 'sameas', 'sequence', 'string', 'undefined', 'upper']}

    .. versionadded:: 2.11.0
    """

    tags = {"debug"}

    def parse(self, parser: "Parser") -> nodes.Output:
        lineno = parser.stream.expect("name:debug").lineno
        context = nodes.ContextReference()
        result = self.call_method("_render", [context], lineno=lineno)
        return nodes.Output([result], lineno=lineno)

    def _render(self, context: Context) -> str:
        result = {
            "context": context.get_all(),
            "filters": sorted(self.environment.filters.keys()),
            "tests": sorted(self.environment.tests.keys()),
        }

        # Set the depth since the intent is to show the top few names.
        return pprint.pformat(result, depth=3, compact=True)


def extract_from_ast(
    ast: nodes.Template,
    gettext_functions: t.Sequence[str] = GETTEXT_FUNCTIONS,
    babel_style: bool = True,
) -> t.Iterator[tuple[int, str, str | None | tuple[str | None, ...]]]:
    """Extract localizable strings from the given template node.  Per
    default this function returns matches in babel style that means non string
    parameters as well as keyword arguments are returned as `None`.  This
    allows Babel to figure out what you really meant if you are using
    gettext functions that allow keyword arguments for placeholder expansion.
    If you don't want that behavior set the `babel_style` parameter to `False`
    which causes only strings to be returned and parameters are always stored
    in tuples.  As a consequence invalid gettext calls (calls without a single
    string parameter or string parameters after non-string parameters) are
    skipped.

    This example explains the behavior:

    >>> from jinja2 import Environment
    >>> env = Environment()
    >>> node = env.parse('{{ (_("foo"), _(), ngettext("foo", "bar", 42)) }}')
    >>> list(extract_from_ast(node))
    [(1, '_', 'foo'), (1, '_', ()), (1, 'ngettext', ('foo', 'bar', None))]
    >>> list(extract_from_ast(node, babel_style=False))
    [(1, '_', ('foo',)), (1, 'ngettext', ('foo', 'bar'))]

    For every string found this function yields a ``(lineno, function,
    message)`` tuple, where:

    * ``lineno`` is the number of the line on which the string was found,
    * ``function`` is the name of the ``gettext`` function used (if the
      string was extracted from embedded Python code), and
    *   ``message`` is the string, or a tuple of strings for functions
         with multiple string arguments.

    This extraction function operates on the AST and is because of that unable
    to extract any comments.  For comment support you have to use the babel
    extraction interface or extract comments yourself.
    """
    out: str | None | tuple[str | None, ...]

    for node in ast.find_all(nodes.Call):
        if (
            not isinstance(node.node, nodes.Name)
            or node.node.name not in gettext_functions
        ):
            continue

        strings: list[str | None] = []

        for arg in node.args:
            if isinstance(arg, nodes.Const) and isinstance(arg.value, str):
                strings.append(arg.value)
            else:
                strings.append(None)

        for _ in node.kwargs:
            strings.append(None)
        if node.dyn_args is not None:
            strings.append(None)
        if node.dyn_kwargs is not None:
            strings.append(None)

        if not babel_style:
            out = tuple(x for x in strings if x is not None)

            if not out:
                continue
        else:
            if len(strings) == 1:
                out = strings[0]
            else:
                out = tuple(strings)

        yield node.lineno, node.node.name, out


class _CommentFinder:
    """Helper class to find comments in a token stream.  Can only
    find comments for gettext calls forwards.  Once the comment
    from line 4 is found, a comment for line 1 will not return a
    usable value.
    """

    def __init__(
        self, tokens: t.Sequence[tuple[int, str, str]], comment_tags: t.Sequence[str]
    ) -> None:
        self.tokens = tokens
        self.comment_tags = comment_tags
        self.offset = 0
        self.last_lineno = 0

    def find_backwards(self, offset: int) -> list[str]:
        try:
            for _, token_type, token_value in reversed(
                self.tokens[self.offset : offset]
            ):
                if token_type in ("comment", "linecomment"):
                    try:
                        prefix, comment = token_value.split(None, 1)
                    except ValueError:
                        continue
                    if prefix in self.comment_tags:
                        return [comment.rstrip()]
            return []
        finally:
            self.offset = offset

    def find_comments(self, lineno: int) -> list[str]:
        if not self.comment_tags or self.last_lineno > lineno:
            return []
        for idx, (token_lineno, _, _) in enumerate(self.tokens[self.offset :]):
            if token_lineno > lineno:
                return self.find_backwards(self.offset + idx)
        return self.find_backwards(len(self.tokens))


def babel_extract(
    fileobj: t.BinaryIO,
    keywords: t.Sequence[str],
    comment_tags: t.Sequence[str],
    options: dict[str, t.Any],
) -> t.Iterator[tuple[int, str, str | None | tuple[str | None, ...], list[str]]]:
    """Babel extraction method for Jinja templates.

    .. versionchanged:: 2.3
       Basic support for translation comments was added.  If `comment_tags`
       is now set to a list of keywords for extraction, the extractor will
       try to find the best preceding comment that begins with one of the
       keywords.  For best results, make sure to not have more than one
       gettext call in one line of code and the matching comment in the
       same line or the line before.

    .. versionchanged:: 2.5.1
       The `newstyle_gettext` flag can be set to `True` to enable newstyle
       gettext calls.

    .. versionchanged:: 2.7
       A `silent` option can now be provided.  If set to `False` template
       syntax errors are propagated instead of being ignored.

    :param fileobj: the file-like object the messages should be extracted from
    :param keywords: a list of keywords (i.e. function names) that should be
                     recognized as translation functions
    :param comment_tags: a list of translator tags to search for and include
                         in the results.
    :param options: a dictionary of additional options (optional)
    :return: an iterator over ``(lineno, funcname, message, comments)`` tuples.
             (comments will be empty currently)
    """
    extensions: dict[type[Extension], None] = {}

    for extension_name in options.get("extensions", "").split(","):
        extension_name = extension_name.strip()

        if not extension_name:
            continue

        extensions[import_string(extension_name)] = None

    if InternationalizationExtension not in extensions:
        extensions[InternationalizationExtension] = None

    def getbool(options: t.Mapping[str, str], key: str, default: bool = False) -> bool:
        return options.get(key, str(default)).lower() in {"1", "on", "yes", "true"}

    silent = getbool(options, "silent", True)
    environment = Environment(
        options.get("block_start_string", defaults.BLOCK_START_STRING),
        options.get("block_end_string", defaults.BLOCK_END_STRING),
        options.get("variable_start_string", defaults.VARIABLE_START_STRING),
        options.get("variable_end_string", defaults.VARIABLE_END_STRING),
        options.get("comment_start_string", defaults.COMMENT_START_STRING),
        options.get("comment_end_string", defaults.COMMENT_END_STRING),
        options.get("line_statement_prefix") or defaults.LINE_STATEMENT_PREFIX,
        options.get("line_comment_prefix") or defaults.LINE_COMMENT_PREFIX,
        getbool(options, "trim_blocks", defaults.TRIM_BLOCKS),
        getbool(options, "lstrip_blocks", defaults.LSTRIP_BLOCKS),
        defaults.NEWLINE_SEQUENCE,
        getbool(options, "keep_trailing_newline", defaults.KEEP_TRAILING_NEWLINE),
        tuple(extensions),
        cache_size=0,
        auto_reload=False,
    )

    if getbool(options, "trimmed"):
        environment.policies["ext.i18n.trimmed"] = True
    if getbool(options, "newstyle_gettext"):
        environment.newstyle_gettext = True  # type: ignore

    source = fileobj.read().decode(options.get("encoding", "utf-8"))
    try:
        node = environment.parse(source)
        tokens = list(environment.lex(environment.preprocess(source)))
    except TemplateSyntaxError:
        if not silent:
            raise
        # skip templates with syntax errors
        return

    finder = _CommentFinder(tokens, comment_tags)
    extract_keywords = tuple(dict.fromkeys((*keywords, *DOMAIN_GETTEXT_FUNCTIONS)))

    for lineno, func, message in extract_from_ast(node, extract_keywords):
        yield lineno, func, message, finder.find_comments(lineno)


#: nicer import names
i18n = InternationalizationExtension
do = ExprStmtExtension
loopcontrols = LoopControlExtension
debug = DebugExtension

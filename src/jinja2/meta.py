"""Functions that expose information about templates that might be
interesting for introspection.
"""

import typing as t

from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame

if t.TYPE_CHECKING:
    from .environment import Environment


class DynamicAccess(t.NamedTuple):
    """A context access that cannot be resolved statically, for example
    ``user[key]`` where the subscript is not a constant.

    .. versionadded:: 3.2
    """

    #: The statically known prefix of the path, such as ``("user",)``.
    parts: tuple[str, ...]
    #: The line number of the access in the template source.
    lineno: int
    #: Why the access could not be resolved statically.
    reason: str


class UnresolvedTemplate(t.NamedTuple):
    """A template dependency that could not be analyzed, either because
    the template name is dynamic or because the caller did not provide
    the referenced template's AST.

    .. versionadded:: 3.2
    """

    #: The referenced template name, or `None` if it is dynamic.
    name: str | None
    #: The line number of the reference in the template source.
    lineno: int
    #: The kind of reference: ``"extends"``, ``"include"``, ``"import"``,
    #: or ``"from_import"``.
    kind: str


class ContextRequirements(t.NamedTuple):
    """The result of :func:`analyze_context_requirements`.

    .. versionadded:: 3.2
    """

    #: Paths that are definitely looked up from the context, no matter
    #: which branch the rendering takes. Maps each path to the line
    #: number of its first access.
    required: dict[tuple[str, ...], int]
    #: Paths that are only looked up when a condition holds, or that are
    #: protected by an ``is defined`` test or the ``default`` filter.
    #: Maps each path to the line number of its first access. Paths that
    #: are also required are excluded.
    guarded: dict[tuple[str, ...], int]
    #: Accesses that could not be resolved to a static path.
    dynamic: list[DynamicAccess]
    #: Referenced templates that could not be analyzed.
    unresolved_templates: list[UnresolvedTemplate]


class TrackingCodeGenerator(CodeGenerator):
    """We abuse the code generator for introspection."""

    def __init__(self, environment: "Environment") -> None:
        super().__init__(environment, "<introspection>", "<introspection>")
        self.undeclared_identifiers: set[str] = set()

    def write(self, x: str) -> None:
        """Don't write."""

    def enter_frame(self, frame: Frame) -> None:
        """Remember all undeclared identifiers."""
        super().enter_frame(frame)

        for _, (action, param) in frame.symbols.loads.items():
            if action == "resolve" and param not in self.environment.globals:
                self.undeclared_identifiers.add(param)


def find_undeclared_variables(ast: nodes.Template) -> set[str]:
    """Returns a set of all variables in the AST that will be looked up from
    the context at runtime.  Because at compile time it's not known which
    variables will be used depending on the path the execution takes at
    runtime, all variables are returned.

    >>> from jinja2 import Environment, meta
    >>> env = Environment()
    >>> ast = env.parse('{% set foo = 42 %}{{ bar + foo }}')
    >>> meta.find_undeclared_variables(ast) == {'bar'}
    True

    .. admonition:: Implementation

       Internally the code generator is used for finding undeclared variables.
       This is good to know because the code generator might raise a
       :exc:`TemplateAssertionError` during compilation and as a matter of
       fact this function can currently raise that exception as well.
    """
    codegen = TrackingCodeGenerator(ast.environment)  # type: ignore
    codegen.visit(ast)
    return codegen.undeclared_identifiers


_ref_types = (nodes.Extends, nodes.FromImport, nodes.Import, nodes.Include)
_RefType = nodes.Extends | nodes.FromImport | nodes.Import | nodes.Include


def find_referenced_templates(ast: nodes.Template) -> t.Iterator[str | None]:
    """Finds all the referenced templates from the AST.  This will return an
    iterator over all the hardcoded template extensions, inclusions and
    imports.  If dynamic inheritance or inclusion is used, `None` will be
    yielded.

    >>> from jinja2 import Environment, meta
    >>> env = Environment()
    >>> ast = env.parse('{% extends "layout.html" %}{% include helper %}')
    >>> list(meta.find_referenced_templates(ast))
    ['layout.html', None]

    This function is useful for dependency tracking.  For example if you want
    to rebuild parts of the website after a layout template has changed.
    """
    template_name: t.Any

    for node in ast.find_all(_ref_types):
        template: nodes.Expr = node.template  # type: ignore

        if not isinstance(template, nodes.Const):
            # a tuple with some non consts in there
            if isinstance(template, (nodes.Tuple, nodes.List)):
                for template_name in template.items:
                    # something const, only yield the strings and ignore
                    # non-string consts that really just make no sense
                    if isinstance(template_name, nodes.Const):
                        if isinstance(template_name.value, str):
                            yield template_name.value
                    # something dynamic in there
                    else:
                        yield None
            # something dynamic we don't know about here
            else:
                yield None
            continue
        # constant is a basestring, direct template name
        if isinstance(template.value, str):
            yield template.value
        # a tuple or list (latter *should* not happen) made of consts,
        # yield the consts that are strings.  We could warn here for
        # non string values
        elif isinstance(node, nodes.Include) and isinstance(
            template.value, (tuple, list)
        ):
            for template_name in template.value:
                if isinstance(template_name, str):
                    yield template_name
        # something else we don't care about, we could warn here
        else:
            yield None


_LOCAL_NAMES = {"loop", "self", "super", "caller", "varargs", "kwargs"}


def _target_names(node: nodes.Node) -> t.Iterator[str]:
    """Yield the names bound by an assignment or loop target."""
    if isinstance(node, nodes.Name):
        yield node.name
    elif isinstance(node, (nodes.Tuple, nodes.List)):
        for item in node.items:
            yield from _target_names(item)


class _ContextAnalyzer:
    """Walks a template AST and records how it accesses the render
    context, without rendering anything.
    """

    def __init__(
        self,
        environment: "Environment",
        templates: t.Mapping[str, nodes.Template],
    ) -> None:
        self.environment = environment
        self.templates = templates
        self.required: dict[tuple[str, ...], int] = {}
        self.guarded: dict[tuple[str, ...], int] = {}
        self.dynamic: list[DynamicAccess] = []
        self.unresolved_templates: list[UnresolvedTemplate] = []
        self._active_templates: set[str] = set()

    def analyze(self, ast: nodes.Template) -> None:
        self._visit_all(ast.body, set(), False)

        # A path that is required anywhere is not merely guarded, and
        # neither is any of its prefixes.
        for path in list(self.guarded):
            if any(
                len(required) >= len(path) and required[: len(path)] == path
                for required in self.required
            ):
                del self.guarded[path]

    def _is_local(self, name: str, scope: set[str]) -> bool:
        return name in scope or name in _LOCAL_NAMES or name in self.environment.globals

    def _record(self, parts: tuple[str, ...], lineno: int, guarded: bool) -> None:
        storage = self.guarded if guarded else self.required

        if parts not in storage:
            storage[parts] = lineno

    def _resolve_path(
        self, node: nodes.Expr, scope: set[str]
    ) -> tuple[str, ...] | None:
        """If ``node`` is a static chain of attribute or string item
        accesses rooted at a context name, return the full path. Return
        `None` for anything rooted at a local name or a computed value.
        """
        parts: list[str] = []

        while True:
            if isinstance(node, nodes.Getattr):
                parts.append(node.attr)
                node = node.node
            elif (
                isinstance(node, nodes.Getitem)
                and isinstance(node.arg, nodes.Const)
                and isinstance(node.arg.value, str)
            ):
                parts.append(node.arg.value)
                node = node.node
            elif isinstance(node, nodes.Name) and node.ctx == "load":
                if self._is_local(node.name, scope):
                    return None

                parts.append(node.name)
                return tuple(reversed(parts))
            else:
                return None

    def _visit_all(
        self, body: list[nodes.Node], scope: set[str], guarded: bool
    ) -> None:
        for node in body:
            self._visit(node, scope, guarded)

    def _visit(self, node: nodes.Node, scope: set[str], guarded: bool) -> None:
        if isinstance(node, nodes.Name):
            if node.ctx == "load" and not self._is_local(node.name, scope):
                self._record((node.name,), node.lineno, guarded)

            return

        if isinstance(node, nodes.NSRef):
            if not self._is_local(node.name, scope):
                self._record((node.name,), node.lineno, guarded)

            return

        if isinstance(node, nodes.Getattr):
            path = self._resolve_path(node, scope)

            if path is not None:
                self._record(path, node.lineno, guarded)
            else:
                self._visit(node.node, scope, guarded)

            return

        if isinstance(node, nodes.Getitem):
            if isinstance(node.arg, nodes.Const) and isinstance(node.arg.value, str):
                path = self._resolve_path(node, scope)

                if path is not None:
                    self._record(path, node.lineno, guarded)
                else:
                    self._visit(node.node, scope, guarded)

                return

            prefix = self._resolve_path(node.node, scope)

            if prefix is not None:
                self._record(prefix, node.lineno, guarded)
                self.dynamic.append(
                    DynamicAccess(prefix, node.lineno, "dynamic subscript")
                )
            else:
                self._visit(node.node, scope, guarded)

            self._visit(node.arg, scope, guarded)
            return

        if isinstance(node, (nodes.And, nodes.Or)):
            self._visit(node.left, scope, guarded)
            # The right side only evaluates if the left side short-circuits.
            self._visit(node.right, scope, True)
            return

        if isinstance(node, nodes.If):
            self._visit(node.test, scope, guarded)
            self._visit_all(node.body, set(scope), True)

            for elif_ in node.elif_:
                # An elif test only runs if all previous tests were false.
                self._visit(elif_, scope, True)

            self._visit_all(node.else_, set(scope), True)
            return

        if isinstance(node, nodes.CondExpr):
            self._visit(node.test, scope, guarded)
            self._visit(node.expr1, scope, True)

            if node.expr2 is not None:
                self._visit(node.expr2, scope, True)

            return

        if isinstance(node, nodes.Test):
            # ``is defined`` and ``is undefined`` protect the tested value.
            self._visit(
                node.node,
                scope,
                guarded or node.name in {"defined", "undefined"},
            )
            self._visit_test_filter_args(node, scope, guarded)
            return

        if isinstance(node, nodes.Filter):
            if node.node is not None:
                # The ``default`` filter protects the value it is applied to.
                self._visit(
                    node.node,
                    scope,
                    guarded or node.name in {"default", "d"},
                )

            self._visit_test_filter_args(node, scope, guarded)
            return

        if isinstance(node, nodes.For):
            self._visit(node.iter, scope, guarded)
            body_scope = scope | set(_target_names(node.target)) | {"loop"}

            if node.test is not None:
                self._visit(node.test, body_scope, guarded)

            self._visit_all(node.body, body_scope, guarded)
            self._visit_all(node.else_, set(scope), guarded)
            return

        if isinstance(node, nodes.With):
            inner_scope = set(scope)

            for target, value in zip(node.targets, node.values, strict=False):
                self._visit(value, inner_scope, guarded)
                inner_scope.update(_target_names(target))

            self._visit_all(node.body, inner_scope, guarded)
            return

        if isinstance(node, nodes.Macro):
            scope.add(node.name)

            for default in node.defaults:
                self._visit(default, scope, guarded)

            inner_scope = scope | {arg.name for arg in node.args}
            self._visit_all(node.body, inner_scope, guarded)
            return

        if isinstance(node, nodes.CallBlock):
            self._visit(node.call, scope, guarded)

            for default in node.defaults:
                self._visit(default, scope, guarded)

            inner_scope = scope | {arg.name for arg in node.args}
            self._visit_all(node.body, inner_scope, guarded)
            return

        if isinstance(node, nodes.Assign):
            self._visit(node.node, scope, guarded)
            self._bind_target(node.target, scope, guarded)
            return

        if isinstance(node, nodes.AssignBlock):
            if node.filter is not None:
                self._visit(node.filter, scope, guarded)

            self._visit_all(node.body, set(scope), guarded)
            self._bind_target(node.target, scope, guarded)
            return

        if isinstance(node, nodes.Block):
            self._visit_all(node.body, scope | {"self", "super"}, guarded)
            return

        if isinstance(node, nodes.Extends):
            self._visit_template_ref(node.template, "extends", scope, guarded, True)
            return

        if isinstance(node, nodes.Include):
            self._visit_template_ref(node.template, "include", scope, guarded, True)
            return

        if isinstance(node, nodes.Import):
            scope.add(node.target)
            self._visit_template_ref(
                node.template, "import", scope, guarded, node.with_context
            )
            return

        if isinstance(node, nodes.FromImport):
            for name in node.names:
                if isinstance(name, tuple):
                    scope.add(name[1])
                else:
                    scope.add(name)

            self._visit_template_ref(
                node.template, "from_import", scope, guarded, node.with_context
            )
            return

        # Everything else (output, calls, expressions, literals, ...) is
        # analyzed generically by visiting its children in the same mode.
        for child in node.iter_child_nodes():
            self._visit(child, scope, guarded)

    def _visit_test_filter_args(
        self,
        node: nodes.Test | nodes.Filter,
        scope: set[str],
        guarded: bool,
    ) -> None:
        for arg in node.args:
            self._visit(arg, scope, guarded)

        for kwarg in node.kwargs:
            self._visit(kwarg, scope, guarded)

        if node.dyn_args is not None:
            self._visit(node.dyn_args, scope, guarded)

        if node.dyn_kwargs is not None:
            self._visit(node.dyn_kwargs, scope, guarded)

    def _bind_target(self, target: nodes.Expr, scope: set[str], guarded: bool) -> None:
        if isinstance(target, nodes.NSRef):
            if not self._is_local(target.name, scope):
                self._record((target.name,), target.lineno, guarded)
        else:
            scope.update(_target_names(target))

    def _visit_template_ref(
        self,
        template: nodes.Expr,
        kind: str,
        scope: set[str],
        guarded: bool,
        follow: bool,
    ) -> None:
        names: list[str | None]

        if isinstance(template, nodes.Const):
            if isinstance(template.value, str):
                names = [template.value]
            elif isinstance(template.value, (tuple, list)):
                names = [
                    item if isinstance(item, str) else None for item in template.value
                ]
            else:
                names = [None]
        elif isinstance(template, (nodes.Tuple, nodes.List)):
            names = []

            for item in template.items:
                if isinstance(item, nodes.Const) and isinstance(item.value, str):
                    names.append(item.value)
                else:
                    names.append(None)
                    self._visit(item, scope, guarded)
        else:
            names = [None]
            self._visit(template, scope, guarded)

        for name in names:
            if name is None:
                if follow or kind in {"extends", "include"}:
                    self.unresolved_templates.append(
                        UnresolvedTemplate(None, template.lineno, kind)
                    )
            elif follow:
                if name in self.templates:
                    if name not in self._active_templates:
                        self._active_templates.add(name)
                        self._visit_all(self.templates[name].body, set(), guarded)
                        self._active_templates.discard(name)
                else:
                    self.unresolved_templates.append(
                        UnresolvedTemplate(name, template.lineno, kind)
                    )


def analyze_context_requirements(
    ast: nodes.Template,
    templates: t.Mapping[str, nodes.Template] | None = None,
) -> ContextRequirements:
    """Statically analyze which values a template looks up from the
    render context, without rendering the template.

    Unlike :func:`find_undeclared_variables`, which only returns the
    top-level names that will be resolved from the context, this traces
    full attribute and item access paths such as ``user.address.city``,
    and classifies each path by the control flow that guards it:

    -   ``required`` paths are looked up no matter which branch the
        rendering takes.
    -   ``guarded`` paths are only looked up when a condition holds, or
        are protected by an ``is defined`` test or the ``default``
        filter. Accesses in the branches of an ``if`` statement, on the
        right side of a short-circuited ``and`` or ``or``, and in the
        branches of a conditional expression are guarded.
    -   ``dynamic`` lists accesses that cannot be resolved to a static
        path, such as ``user[key]``, together with the known prefix.
    -   ``unresolved_templates`` lists ``extends``, ``include``,
        ``import``, and ``from ... import`` references whose template
        name is dynamic or whose AST was not provided.

    Names bound by ``set``, ``for``, ``with``, macro parameters, and
    imports are tracked, so accesses through them are not reported as
    context lookups. Names found in :attr:`Environment.globals` are not
    reported either.

    The analysis never renders anything: no filter, test, loader, or
    user function is called, and neither the AST nor the environment is
    modified. Statically named ``extends`` and ``include`` references,
    as well as imports that use ``with context``, are analyzed
    recursively if the caller provides their ASTs in ``templates``, a
    mapping of template name to parsed AST.

    >>> from jinja2 import Environment, meta
    >>> env = Environment()
    >>> ast = env.parse(
    ...     "{{ user.id }}{% if user.nick is defined %}{{ user.nick }}{% endif %}"
    ... )
    >>> reqs = meta.analyze_context_requirements(ast)
    >>> sorted(reqs.required)
    [('user', 'id')]
    >>> sorted(reqs.guarded)
    [('user', 'nick')]

    .. versionadded:: 3.2
    """
    analyzer = _ContextAnalyzer(
        ast.environment,  # type: ignore
        {} if templates is None else templates,
    )
    analyzer.analyze(ast)
    return ContextRequirements(
        required=analyzer.required,
        guarded=analyzer.guarded,
        dynamic=analyzer.dynamic,
        unresolved_templates=analyzer.unresolved_templates,
    )

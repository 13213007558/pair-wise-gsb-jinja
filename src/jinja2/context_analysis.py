"""Static analysis of the context values a template needs to render.

This module provides :func:`analyze_context_requirements`, a purely
static analysis that reports which context paths a template will read
(`user.id`), which reads are protected by a conditional check such as
`is defined` or the `default` filter, and which accesses cannot be
determined statically (`user[key]`).

The analysis never renders anything. It does not execute filters, tests,
loader callbacks, or user functions, and it does not mutate the AST or
the environment. Templates referenced by `include` or `extends` are
only analyzed when the caller explicitly provides their parsed ASTs.
"""

from __future__ import annotations

import dataclasses
import typing as t

from . import nodes
from .defaults import DEFAULT_NAMESPACE


@dataclasses.dataclass(frozen=True)
class VariableAccess:
    """A read of a context value at a known dotted path."""

    path: str
    lineno: int


@dataclasses.dataclass(frozen=True)
class DynamicAccess:
    """A read whose exact path cannot be determined statically, such as
    `user[key]`. `path` is the known static prefix, for example
    `"user"`.
    """

    path: str
    lineno: int


@dataclasses.dataclass(frozen=True)
class UnresolvedTemplate:
    """A template dependency that could not be analyzed, either because
    the template name is dynamic (`name` is `None`) or because the
    caller did not provide the referenced template's AST.
    """

    name: str | None
    lineno: int
    kind: str  # "include", "extends", or "import"


class ContextRequirements:
    """The result of :func:`analyze_context_requirements`.

    .. attribute:: required
        Accesses that are unconditionally read from the context. These
        values must be present for the template to render as intended.

    .. attribute:: protected
        Accesses guarded by a conditional check (`is defined`, an
        `if` branch, short-circuit `and`/`or`) or the `default`
        filter. They are only read when the guard allows it.

    .. attribute:: dynamic
        Accesses that cannot be resolved to a concrete path statically.

    .. attribute:: unresolved_templates
        Referenced templates that could not be analyzed.
    """

    def __init__(self) -> None:
        self.required: list[VariableAccess] = []
        self.protected: list[VariableAccess] = []
        self.dynamic: list[DynamicAccess] = []
        self.unresolved_templates: list[UnresolvedTemplate] = []

    @property
    def required_paths(self) -> set[str]:
        return {access.path for access in self.required}

    @property
    def protected_paths(self) -> set[str]:
        return {access.path for access in self.protected}

    def __repr__(self) -> str:
        return (
            f"ContextRequirements(required={self.required!r},"
            f" protected={self.protected!r}, dynamic={self.dynamic!r},"
            f" unresolved_templates={self.unresolved_templates!r})"
        )


def analyze_context_requirements(
    ast: nodes.Template,
    templates: t.Mapping[str, nodes.Template] | None = None,
    *,
    globals: t.Iterable[str] | None = None,
) -> ContextRequirements:
    """Statically analyze which context values a template needs.

    :param ast: The parsed template, as returned by
        :meth:`jinja2.Environment.parse`.
    :param templates: Parsed ASTs of referenced templates, keyed by
        template name. Static `include` and `extends` references are
        only followed when the target is present here. Dynamic template
        names and missing ASTs are reported as unresolved dependencies.
    :param globals: Additional top-level names that are known to be
        defined, such as `environment.globals`. Jinja's default
        globals (`range`, `namespace`, ...) are always known.

    The analysis is conservative: anything it reports as required is
    definitely read, and accesses it cannot prove are reported
    separately instead of being silently dropped.
    """
    analyzer = _Analyzer(templates if templates is not None else {}, globals)
    analyzer.visit_body(ast.body, frozenset())
    return analyzer.result


class _Analyzer:
    def __init__(
        self,
        templates: t.Mapping[str, nodes.Template],
        globals: t.Iterable[str] | None,
    ) -> None:
        self.templates = templates
        self.result = ContextRequirements()
        self.scopes: list[set[str]] = [set(DEFAULT_NAMESPACE)]
        if globals is not None:
            self.scopes[0].update(globals)
        self._seen_templates: set[str] = set()

    # -- scope handling --------------------------------------------------

    def _is_local(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self.scopes))

    def _define(self, target: nodes.Node) -> None:
        if isinstance(target, nodes.Name):
            self.scopes[-1].add(target.name)
        elif isinstance(target, (nodes.Tuple, nodes.List)):
            for item in target.items:
                self._define(item)

    # -- path resolution -------------------------------------------------

    def _resolve(self, node: nodes.Expr) -> tuple[str, str | None]:
        """Resolve an expression to a static context path, a local
        variable, or something not statically trackable.
        """
        if isinstance(node, nodes.Name):
            if self._is_local(node.name):
                return "local", None
            return "path", node.name

        if isinstance(node, nodes.NSRef):
            return "local", None

        if isinstance(node, nodes.Getattr):
            kind, path = self._resolve(node.node)
            if kind == "path":
                return "path", f"{path}.{node.attr}"
            return kind, None

        if isinstance(node, nodes.Getitem) and isinstance(node.arg, nodes.Const):
            kind, path = self._resolve(node.node)
            if kind == "path":
                value = node.arg.value
                if isinstance(value, str) and value.isidentifier():
                    return "path", f"{path}.{value}"
                return "path", f"{path}[{value!r}]"
            return kind, None

        return "other", None

    def _record(
        self, node: nodes.Expr, guards: frozenset[str], protected: bool
    ) -> None:
        kind, path = self._resolve(node)

        if kind == "path":
            assert path is not None
            access = VariableAccess(path=path, lineno=node.lineno)
            if protected or self._is_covered(path, guards):
                self.result.protected.append(access)
            else:
                self.result.required.append(access)
        elif kind == "other":
            # e.g. an attribute access on a call result; the inner
            # expression may still read from the context.
            if isinstance(node, (nodes.Getattr, nodes.Getitem)):
                self.visit_expr(node.node, guards, protected)

    @staticmethod
    def _is_covered(path: str, guards: frozenset[str]) -> bool:
        for guard in guards:
            if (
                path == guard
                or path.startswith(guard + ".")
                or guard.startswith(path + ".")
            ):
                return True
        return False

    # -- guard extraction ------------------------------------------------

    def _guarded_path(self, node: nodes.Expr) -> str | None:
        kind, path = self._resolve(node)
        return path if kind == "path" else None

    def _true_guards(self, node: nodes.Expr) -> frozenset[str]:
        """Paths guaranteed to be defined when `node` is truthy."""
        if isinstance(node, nodes.Test):
            if node.name == "defined":
                path = self._guarded_path(node.node)
                return frozenset({path}) if path is not None else frozenset()
            return frozenset()
        if isinstance(node, nodes.Not):
            return self._false_guards(node.node)
        if isinstance(node, nodes.And):
            return self._true_guards(node.left) | self._true_guards(node.right)
        return frozenset()

    def _false_guards(self, node: nodes.Expr) -> frozenset[str]:
        """Paths guaranteed to be defined when `node` is falsy."""
        if isinstance(node, nodes.Test):
            if node.name == "undefined":
                path = self._guarded_path(node.node)
                return frozenset({path}) if path is not None else frozenset()
            return frozenset()
        if isinstance(node, nodes.Not):
            return self._true_guards(node.node)
        if isinstance(node, nodes.Or):
            return self._false_guards(node.left) | self._false_guards(node.right)
        return frozenset()

    # -- expressions -----------------------------------------------------

    def visit_expr(
        self, node: nodes.Expr, guards: frozenset[str], protected: bool
    ) -> None:
        if isinstance(node, (nodes.Name, nodes.Getattr)):
            self._record(node, guards, protected)
            return

        if isinstance(node, nodes.Getitem):
            if isinstance(node.arg, nodes.Const):
                self._record(node, guards, protected)
            else:
                # The index expression is always evaluated.
                self.visit_expr(node.arg, guards, False)
                kind, path = self._resolve(node.node)
                if kind == "path":
                    self._record(node.node, guards, protected)
                    assert path is not None
                    self.result.dynamic.append(
                        DynamicAccess(path=path, lineno=node.lineno)
                    )
                elif kind == "other":
                    self.visit_expr(node.node, guards, protected)
            return

        if isinstance(node, nodes.And):
            self.visit_expr(node.left, guards, protected)
            self.visit_expr(
                node.right, guards | self._true_guards(node.left), protected
            )
            return

        if isinstance(node, nodes.Or):
            self.visit_expr(node.left, guards, protected)
            self.visit_expr(
                node.right, guards | self._false_guards(node.left), protected
            )
            return

        if isinstance(node, nodes.Test):
            # The operand of an (un)defined check is protected by the
            # check itself.
            self.visit_expr(
                node.node, guards, protected or node.name in ("defined", "undefined")
            )
            self._visit_call_args(node, guards)
            return

        if isinstance(node, nodes.Filter):
            if node.node is not None:
                self.visit_expr(
                    node.node,
                    guards,
                    protected or node.name in ("default", "d"),
                )
            self._visit_call_args(node, guards)
            return

        if isinstance(node, nodes.CondExpr):
            self.visit_expr(node.test, guards, False)
            self.visit_expr(
                node.expr1, guards | self._true_guards(node.test), protected
            )
            if node.expr2 is not None:
                self.visit_expr(
                    node.expr2, guards | self._false_guards(node.test), protected
                )
            return

        if isinstance(node, nodes.Call):
            self.visit_expr(node.node, guards, protected)
            self._visit_call_args(node, guards)
            return

        if isinstance(node, nodes.NSRef):
            return

        # Generic expressions (arithmetic, comparisons, literals, ...):
        # recurse into children.
        self._visit_children(node, guards, protected)

    def _visit_children(
        self, node: nodes.Node, guards: frozenset[str], protected: bool
    ) -> None:
        for child in node.iter_child_nodes():
            if isinstance(child, nodes.Expr):
                self.visit_expr(child, guards, protected)
            elif isinstance(child, nodes.Node):
                self._visit_children(child, guards, protected)

    def _visit_call_args(
        self, node: nodes.Call | nodes.Filter | nodes.Test, guards: frozenset[str]
    ) -> None:
        # Arguments are evaluated unconditionally when the call, filter,
        # or test runs, so they are not protected by an outer `default`.
        for arg in node.args:
            self.visit_expr(arg, guards, False)
        for kwarg in node.kwargs:
            self.visit_expr(kwarg.value, guards, False)
        if node.dyn_args is not None:
            self.visit_expr(node.dyn_args, guards, False)
        if node.dyn_kwargs is not None:
            self.visit_expr(node.dyn_kwargs, guards, False)

    # -- statements ------------------------------------------------------

    def visit_body(self, body: list[nodes.Node], guards: frozenset[str]) -> None:
        for node in body:
            self.visit_stmt(node, guards)

    def visit_stmt(self, node: nodes.Node, guards: frozenset[str]) -> None:
        if isinstance(node, nodes.Output):
            for expr in node.nodes:
                self.visit_expr(expr, guards, False)
            return

        if isinstance(node, nodes.If):
            test = t.cast(nodes.Expr, node.test)
            self.visit_expr(test, guards, False)
            self.visit_body(node.body, guards | self._true_guards(test))
            false_guards = guards | self._false_guards(test)
            for elif_node in node.elif_:
                elif_test = t.cast(nodes.Expr, elif_node.test)
                self.visit_expr(elif_test, false_guards, False)
                self.visit_body(
                    elif_node.body, false_guards | self._true_guards(elif_test)
                )
                false_guards = false_guards | self._false_guards(elif_test)
            self.visit_body(node.else_, false_guards)
            return

        if isinstance(node, nodes.For):
            self.visit_expr(t.cast(nodes.Expr, node.iter), guards, False)
            self.scopes.append({"loop"})
            self._define(node.target)
            if node.test is not None:
                self.visit_expr(t.cast(nodes.Expr, node.test), guards, False)
            self.visit_body(node.body, guards)
            self.scopes.pop()
            # The else block runs when the loop is empty; the loop
            # target is not defined there.
            self.visit_body(node.else_, guards)
            return

        if isinstance(node, nodes.Macro):
            self.scopes[-1].add(node.name)
            for default in node.defaults:
                self.visit_expr(default, guards, False)
            self.scopes.append(
                {arg.name for arg in node.args} | {"caller", "varargs", "kwargs"}
            )
            self.visit_body(node.body, guards)
            self.scopes.pop()
            return

        if isinstance(node, nodes.CallBlock):
            self.visit_expr(node.call, guards, False)
            for default in node.defaults:
                self.visit_expr(default, guards, False)
            self.scopes.append({arg.name for arg in node.args})
            self.visit_body(node.body, guards)
            self.scopes.pop()
            return

        if isinstance(node, nodes.FilterBlock):
            self._visit_call_args(node.filter, guards)
            self.visit_body(node.body, guards)
            return

        if isinstance(node, nodes.With):
            for value in node.values:
                self.visit_expr(value, guards, False)
            self.scopes.append(set())
            for target in node.targets:
                self._define(target)
            self.visit_body(node.body, guards)
            self.scopes.pop()
            return

        if isinstance(node, nodes.Assign):
            self.visit_expr(t.cast(nodes.Expr, node.node), guards, False)
            self._define(node.target)
            return

        if isinstance(node, nodes.AssignBlock):
            if node.filter is not None:
                self._visit_call_args(node.filter, guards)
            self.visit_body(node.body, guards)
            self._define(node.target)
            return

        if isinstance(node, nodes.Include):
            self._visit_template_ref(node, "include", guards, node.with_context)
            return

        if isinstance(node, nodes.Extends):
            self._visit_template_ref(node, "extends", guards, True)
            return

        if isinstance(node, (nodes.Import, nodes.FromImport)):
            if isinstance(node, nodes.Import):
                self.scopes[-1].add(node.target)
            else:
                for name in node.names:
                    if isinstance(name, str):
                        self.scopes[-1].add(name)
                    else:
                        self.scopes[-1].add(name[1])
            # Imported macros are self-contained unless "with context"
            # is used.
            self._visit_template_ref(node, "import", guards, node.with_context)
            return

        if isinstance(node, (nodes.Block, nodes.Template)):
            self.visit_body(node.body, guards)
            return

        # Generic statements (do, expression statements from
        # extensions, ...): recurse into children.
        for child in node.iter_child_nodes():
            if isinstance(child, nodes.Expr):
                self.visit_expr(child, guards, False)
            elif isinstance(child, nodes.Node):
                self.visit_stmt(child, guards)

    def _template_names(
        self, node: nodes.Expr, guards: frozenset[str]
    ) -> list[str | None]:
        if isinstance(node, nodes.Const) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, (nodes.Tuple, nodes.List)):
            names: list[str | None] = []
            for item in node.items:
                if isinstance(item, nodes.Const) and isinstance(item.value, str):
                    names.append(item.value)
                else:
                    names.append(None)
                    self.visit_expr(item, guards, False)
            return names
        self.visit_expr(node, guards, False)
        return [None]

    def _visit_template_ref(
        self,
        node: nodes.Include | nodes.Extends | nodes.Import | nodes.FromImport,
        kind: str,
        guards: frozenset[str],
        needs_context: bool,
    ) -> None:
        for name in self._template_names(node.template, guards):
            if name is not None and name in self.templates:
                if name in self._seen_templates:
                    continue
                self._seen_templates.add(name)
                self.visit_body(self.templates[name].body, guards)
            elif needs_context or name is None:
                self.result.unresolved_templates.append(
                    UnresolvedTemplate(name=name, lineno=node.lineno, kind=kind)
                )

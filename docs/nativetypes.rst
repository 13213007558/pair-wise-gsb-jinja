.. module:: jinja2.nativetypes

.. _nativetypes:

Native Python Types
===================

The default :class:`~jinja2.Environment` renders templates to strings. With
:class:`NativeEnvironment`, rendering a template produces a native Python type.
This is useful if you are using Jinja outside the context of creating text
files. For example, your code may have an intermediate step where users may use
templates to define values that will then be passed to a traditional string
environment.

Examples
--------

Adding two values results in an integer, not a string with a number:

>>> env = NativeEnvironment()
>>> t = env.from_string('{{ x + y }}')
>>> result = t.render(x=4, y=2)
>>> print(result)
6
>>> print(type(result))
int

Rendering list syntax produces a list:

>>> t = env.from_string('[{% for item in data %}{{ item + 1 }},{% endfor %}]')
>>> result = t.render(data=range(5))
>>> print(result)
[1, 2, 3, 4, 5]
>>> print(type(result))
list

Rendering something that doesn't look like a Python literal produces a string:

>>> t = env.from_string('{{ x }} * {{ y }}')
>>> result = t.render(x=4, y=2)
>>> print(result)
4 * 2
>>> print(type(result))
str

Rendering a Python object produces that object as long as it is the only node:

>>> class Foo:
...     def __init__(self, value):
...         self.value = value
...
>>> result = env.from_string('{{ x }}').render(x=Foo(15))
>>> print(type(result).__name__)
Foo
>>> print(result.value)
15

Conversion Rules
----------------

Rendered output is converted to a native type following a consistent set of
rules, applied the same way for sync and async rendering, macros, loops,
and blocks:

-   A single node that is not a plain string is returned unchanged. This
    preserves lists, dicts, numbers, ``Undefined``, and arbitrary objects,
    including nested structures.
-   A single :class:`~markupsafe.Markup` node is returned unchanged. Markup
    is already-escaped text, never a Python literal.
-   A single plain string is parsed as a Python literal if possible,
    otherwise it is returned unchanged.
-   Multiple nodes are concatenated as strings. If any node is
    :class:`~markupsafe.Markup`, the joined string is returned as ``Markup``.
    Otherwise, if the joined string parses as a Python literal, the parsed
    value is returned, otherwise the joined string is returned.

Parsing only uses :func:`ast.literal_eval`, never :func:`eval`, so values
that look like code are not executed. Concatenating multiple nodes calls
``str`` on each node, so an undefined value with a strict ``undefined``
policy still raises :class:`~jinja2.exceptions.UndefinedError`.

If falling back to a string is not acceptable, enable ``native_strict`` to
raise a :exc:`ValueError` instead:

>>> env = NativeEnvironment(native_strict=True)
>>> env.from_string('{{ x + y }}').render(x=4, y=2)
6
>>> env.from_string('hello').render()
Traceback (most recent call last):
  ...
ValueError: Could not convert the rendered template to a native Python type: 'hello'

.. versionchanged:: 3.2
    ``Markup`` is preserved instead of being parsed as a Python literal,
    and the ``native_strict`` parameter was added.

Sandboxed Native Environment
----------------------------

You can combine :class:`.SandboxedEnvironment` and :class:`NativeEnvironment` to
get both behaviors.

.. code-block:: python

    class SandboxedNativeEnvironment(SandboxedEnvironment, NativeEnvironment):
        pass

API
---

.. autoclass:: NativeEnvironment([options])

.. autoclass:: NativeTemplate([options])
    :members: render

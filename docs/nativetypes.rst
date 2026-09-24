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

Concatenation Rules
-------------------+

If rendering produces a single node, its value is returned unchanged,
preserving its type. If it produces multiple nodes, they are concatenated as
strings, then parsed with :func:`ast.literal_eval`. If parsing succeeds, the
parsed value is returned, otherwise the string is returned. Parsing never
executes code, values that aren't simple literals such as
``__import__('os')`` are returned as strings.

A single :class:`~jinja2.runtime.Undefined` node follows the environment's
undefined policy rather than being returned silently. For example,
``StrictUndefined`` raises an error, while the default ``Undefined`` renders
an empty string.

If any concatenated node is :class:`markupsafe.Markup`, the result remains
``Markup`` if it's not parsed as a literal. With ``autoescape`` enabled,
escaped values are ``Markup`` and keep that type through concatenation.

To fail instead of falling back to a string when the result can't be parsed
as a literal, use the ``strict`` parameter to
:func:`~jinja2.nativetypes.native_concat`:

.. code-block:: python

    from functools import partial
    from jinja2.nativetypes import NativeEnvironment, native_concat

    class StrictNativeEnvironment(NativeEnvironment):
        concat = staticmethod(partial(native_concat, strict=True))

.. versionchanged:: 3.2
    A single ``Undefined`` node applies the undefined policy. ``Markup`` is
    preserved when concatenating. ``autoescape`` escapes values. Added the
    ``strict`` parameter to ``native_concat``.

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

"""What a package needs to import its public names when they are first asked for (PEP 562).

``process_improve`` and ``process_improve.experiments`` import nothing of their own when they are imported
(#1235). Each then answers ``__getattr__`` and ``__dir__`` with the three functions here, so that the package
still looks, to its users, as it did when it imported everything itself:

- a name of the package is imported under one lock, so several threads may ask the package at once;
- a module of the package is its attribute without an import of its own (``experiments.models.lm``);
- ``dir(package)`` lists the lazy names and the public modules, for a reader and for tab completion.
"""

from __future__ import annotations

import importlib
import pkgutil
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

#: Held while a package imports one of its names. A package that imports everything itself does so under its
#: own import lock: every other thread waits for the whole of it. Asked for one by one, two names would be
#: imported by two threads at once, by different ways into statsmodels, whose modules import each other in a
#: circle; Python then hands one thread a module the other has only half made. One lock puts the threads back
#: in a queue. It is re-entrant, because importing one name can ask the package for another.
#:
#: shortcut: the lock covers what is asked of the two packages (``experiments.lm``), not an ``import`` statement
#: that runs in another thread at the same moment (``import process_improve.multivariate``), which Python
#: imports under that module's own lock; the two can still meet in a third package. A program with threads
#: imports what they will use before it starts them, as with any package. Upgrade if that proves too much to
#: ask: the import system would have to be given the lock, by a finder on ``sys.meta_path``.
#:
#: No module of the library may ask these packages for a name while it is itself being imported: it would wait
#: here holding its own import lock, which the thread ahead of it may need. ``tests/test_lazy_imports.py`` holds
#: the library to that.
_IMPORTING = threading.RLock()


def load(module: str) -> ModuleType:
    """Import ``module``, one thread at a time."""
    with _IMPORTING:
        return importlib.import_module(module)


def submodule(package: str, name: str) -> ModuleType:
    """Return the module ``name`` of ``package``, imported if it was not; ``AttributeError`` if it cannot be had.

    A name that cannot be a module is looked for no further: a dunder is what tools probe a module for, and a
    name with a dot in it would import a parent that is not there. A module that is there and cannot be
    imported, for want of an optional dependency say, is no attribute either, and the error says why:
    ``hasattr``, ``help`` and ``inspect.getmembers`` ask for every name ``dir`` lists and expect nothing else.
    ``from package import name`` then imports the module itself, and raises what the module raises.
    """
    full = f"{package}.{name}"
    message = f"module {package!r} has no attribute {name!r}"
    if name.isidentifier() and not name.startswith("__"):
        try:
            return load(full)
        except ImportError as error:
            if not (isinstance(error, ModuleNotFoundError) and error.name == full):  # there, and not importable
                raise AttributeError(f"{message}: {full} could not be imported ({error})") from error
    raise AttributeError(message)


def listing(package: ModuleType, lazy: Iterable[str]) -> list[str]:
    """Return what ``dir(package)`` gives: what the package holds, its lazy names and its public modules."""
    modules = (found.name for found in pkgutil.iter_modules(package.__path__) if not found.name.startswith("_"))
    return sorted({*vars(package), *lazy, *modules})

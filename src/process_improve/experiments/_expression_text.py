# (c) Kevin Dunn, 2010-2026. MIT License.

"""How constraint expressions name factors, shared by the parser of inequalities and the parser of rules.

Both :mod:`~process_improve.experiments.designs_constrained` and
:mod:`~process_improve.experiments._rules` read these, so they live in a module that
imports neither.
"""

from __future__ import annotations

import unicodedata

#: Longest constraint expression accepted, which also bounds the parse depth.
MAX_EXPRESSION_LENGTH = 500


def nfkc(name: str) -> str:
    """Return ``name`` in Unicode NFKC form: how Python's parser spells it in an expression (#808).

    The parser rewrites every identifier into this form, so a factor named with a
    decomposed accent (``e`` and a combining acute) or a full-width letter (``\uff34``) reaches
    a parsed limit as the composed accent or the plain ``T``.
    """
    return unicodedata.normalize("NFKC", name)

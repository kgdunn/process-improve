"""(c) Kevin Dunn, 2010-2026. MIT License.

Host-side injection channel for the fake-data simulator tools.

``simulate_process`` and ``reveal_simulator`` need two pieces of state
that the LLM must never be able to supply itself:

- ``simulator_state`` - the hidden response-surface model the host
  persists per conversation.
- ``confirmed`` - whether the user has cleared the double-confirmation
  gate guarding a model reveal.

Earlier these travelled as keyword arguments deliberately kept out of
the public JSON schema, on the assumption the host would inject them
server-side. But the dispatch path forwards the tool ``input`` verbatim
as ``**kwargs``, so a prompt-injected agent could re-introduce them as
ordinary kwargs and bypass the reveal gate (SEC-15). The schema filter
in :func:`process_improve.tool_spec.execute_tool_call` now drops such
keys, and this module removes them from the kwarg surface entirely:
they live in :class:`contextvars.ContextVar` slots that only the host
populates, so they cannot be reached through a tool call even if the
filter regresses.

Host usage
----------

The host wraps each dispatch in :func:`simulator_host_context`::

    from process_improve.simulation.context import simulator_host_context
    from process_improve.tool_spec import execute_tool_call

    with simulator_host_context(simulator_state=state, confirmed=user_ok):
        result = execute_tool_call(tool_name, tool_input)

Outside such a block the context is empty, so ``simulate_process``
reports ``simulator_state_missing`` and ``reveal_simulator`` returns
``confirmation_needed`` - the safe defaults.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# Hidden model state injected by the host for the current dispatch. ``None``
# means "no state available" (the default outside a host context).
_SIMULATOR_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "process_improve_simulator_state", default=None
)

# Whether the host has cleared the reveal double-confirmation gate for the
# current dispatch. Defaults to ``False`` so an un-wrapped call never reveals.
_REVEAL_CONFIRMED: ContextVar[bool] = ContextVar(
    "process_improve_reveal_confirmed", default=False
)


def get_injected_simulator_state() -> dict[str, Any] | None:
    """Return the host-injected ``simulator_state`` for the current dispatch.

    Returns ``None`` when called outside a :func:`simulator_host_context`
    block, which the simulator tools surface as ``simulator_state_missing``.
    """
    return _SIMULATOR_STATE.get()


def get_reveal_confirmed() -> bool:
    """Return whether the host has confirmed a reveal for the current dispatch.

    Defaults to ``False`` outside a :func:`simulator_host_context` block so an
    un-gated call can never expose the hidden model.
    """
    return _REVEAL_CONFIRMED.get()


@contextlib.contextmanager
def simulator_host_context(
    *,
    simulator_state: dict[str, Any] | None = None,
    confirmed: bool = False,
) -> Iterator[None]:
    """Inject simulator state / reveal confirmation for the enclosed dispatch.

    Parameters
    ----------
    simulator_state:
        The hidden model the host persists for this conversation, or ``None``
        to leave it unavailable.
    confirmed:
        ``True`` only after the user has cleared the reveal double-confirmation
        gate. Coerced to ``bool``.

    Notes
    -----
    The values are stored in :class:`contextvars.ContextVar` slots and reset
    on exit, so concurrent hosts (e.g. async tasks) do not see each other's
    state. ``ContextVar`` values do not cross process boundaries, so a host
    that dispatches through the subprocess-isolated
    :func:`process_improve.tool_safety.safe_execute_tool_call` will not leak
    state into the worker; that untrusted-transport path is expected to report
    ``simulator_state_missing`` rather than carry persisted state.
    """
    state_token = _SIMULATOR_STATE.set(simulator_state)
    confirmed_token = _REVEAL_CONFIRMED.set(bool(confirmed))
    try:
        yield
    finally:
        _SIMULATOR_STATE.reset(state_token)
        _REVEAL_CONFIRMED.reset(confirmed_token)

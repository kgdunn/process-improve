"""Stopping Python: a search the reader does not want to wait for is ended, and Python starts again.

A worker cannot be interrupted, so the page ends it and starts another; every call that was waiting is answered
as stopped, and the page's next call is answered by the new worker. Stop is offered only once a call has taken a
while, so a reader never sees it for the calls Python answers at once.
"""

from __future__ import annotations

from tests.web.e2e._page import choose, generate, status


def test_a_call_that_takes_a_while_can_be_stopped_and_python_answers_again(blank_page):
    """Stop is offered once Python has taken a while over a call; it ends the call, and the next one is answered."""
    page = blank_page
    assert page.is_hidden("#stop"), "nothing is waiting"
    choose(page, design_type="sobol")  # a design the comparison has not made already, so Python is asked for it
    page.evaluate("window.__pi.hold('api_make_design')")
    page.click("#generate")
    page.wait_for_selector("#stop:not([hidden])", timeout=10_000)
    page.click("#stop")
    page.wait_for_function("document.querySelector('#stop').hidden")
    assert status(page)[0] == "Stopped, as you asked: nothing was changed.", "what the call it stopped is answered with"
    page.evaluate(
        "window.__pi.release('api_make_design')"
    )  # the reply of the worker that was ended: the page hears none
    generate(page, design_type="full_factorial")
    assert "11 runs" in page.text_content("#design-summary"), "Python answers the page's next call"
    assert page.is_hidden("#stop")


def test_stop_is_not_offered_for_a_call_python_answers_quickly(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial")
    assert page.is_hidden("#stop")

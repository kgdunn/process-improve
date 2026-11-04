"""The xdist controller rebuilds the warnings its workers report one at a time (#1137)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from xdist import workermanage

from tests import conftest

THREADS = 4


def test_the_controller_rebuilds_one_warning_at_a_time(monkeypatch):
    # Each worker's reports arrive on a thread of their own. Rebuilding a warning imports the module
    # that defines it, and two threads importing two unimported packages at once can deadlock.
    active = 0
    most_at_once = 0
    count = threading.Lock()

    def rebuild(data):
        nonlocal active, most_at_once
        with count:
            active += 1
            most_at_once = max(most_at_once, active)
        time.sleep(0.02)  # long enough for every other thread to arrive while this one is inside
        with count:
            active -= 1
        return data

    monkeypatch.setattr(workermanage, "unserialize_warning_message", rebuild)
    conftest.pytest_configure(SimpleNamespace())  # the controller: a config without workerinput

    barrier = threading.Barrier(THREADS)
    rebuilt = []

    def report(data):
        barrier.wait()
        rebuilt.append(workermanage.unserialize_warning_message(data))

    threads = [threading.Thread(target=report, args=(n,)) for n in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(rebuilt) == list(range(THREADS))
    assert most_at_once == 1


def test_a_worker_leaves_xdist_as_it_is(monkeypatch):
    for name in conftest.ONE_THREAD:
        monkeypatch.setenv(name, "2")  # set, so the worker branch changes nothing outside this test
    original = workermanage.unserialize_warning_message

    conftest.pytest_configure(SimpleNamespace(workerinput={}))

    assert workermanage.unserialize_warning_message is original

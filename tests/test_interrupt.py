"""Tests for KeyboardInterrupt (Ctrl-C) handling in long C++ computations.

The SignalGuard mechanism installs a C-level SIGINT handler that flips
``oineus::g_stop_flag``; workers poll the flag at coarse checkpoints in
hot loops (decomposition, VR/Freudenthal filtration construction, Hera
auctions) and exit cleanly. The binding's guard destructor reacquires
the GIL and raises KeyboardInterrupt.
"""

import os
import signal
import threading
import time

import numpy as np
import pytest

import oineus as oin


pytestmark = pytest.mark.skipif(
    os.name == 'nt',
    reason='SignalGuard uses POSIX sigaction; Windows path not implemented',
)


def _small_vr_filtration(seed=0):
    rng = np.random.default_rng(seed)
    points = rng.random((40, 3)).astype(np.float64)
    return oin.vr_filtration(points, max_dim=2, max_diameter=0.6, n_threads=1)


def test_normal_completion_unaffected():
    """A non-interrupted call must return the expected result."""
    fil = _small_vr_filtration()
    dcmp = oin.Decomposition(fil, dualize=False)
    params = oin.ReductionParams()
    dcmp.reduce(params)
    dgm0 = dcmp.diagram(fil).in_dimension(0)
    assert dgm0.shape[0] >= 1


def test_signal_handler_restored():
    """The user-installed SIGINT handler survives an Oineus call."""

    def my_handler(signum, frame):
        pass

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, my_handler)
    try:
        _small_vr_filtration()
        assert signal.getsignal(signal.SIGINT) is my_handler
    finally:
        signal.signal(signal.SIGINT, previous)


def _send_sigint_to_process():
    # os.kill on a PID delivers via kill(2): process-targeted, so the
    # kernel picks any thread whose signal mask does not block SIGINT.
    # signal.raise_signal would go to the calling (Timer) thread, where
    # Python masks signals by default.
    os.kill(os.getpid(), signal.SIGINT)


def _arm_sigint(delay_s):
    return threading.Timer(delay_s, _send_sigint_to_process)


def test_vr_filtration_keyboard_interrupt():
    """Construction of a moderately large VR filtration responds to Ctrl-C."""
    rng = np.random.default_rng(1)
    # 250 points in 3D, max_dim 3 with generous diameter yields millions
    # of candidate simplices; construction takes seconds uninterrupted.
    points = rng.random((250, 3)).astype(np.float64)
    timer = _arm_sigint(0.1)
    timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            oin.vr_filtration(points, max_dim=3, max_diameter=1.0, n_threads=1)
        elapsed = time.monotonic() - start
        assert elapsed < 3.0, f'KeyboardInterrupt arrived too late: {elapsed:.2f}s'
    finally:
        timer.cancel()
        timer.join(timeout=2.0)


def test_freudenthal_filtration_keyboard_interrupt():
    """Freudenthal filtration construction on a big grid is interruptible."""
    # 150^3 = 3.4M vertices; construction is several hundred ms on a
    # modern laptop.
    data = np.random.default_rng(2).random((150, 150, 150)).astype(np.float64)
    timer = _arm_sigint(0.1)
    timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            oin.freudenthal_filtration(
                data, negate=False, wrap=False, max_dim=3, n_threads=1,
            )
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f'KeyboardInterrupt arrived too late: {elapsed:.2f}s'
    finally:
        timer.cancel()
        timer.join(timeout=2.0)

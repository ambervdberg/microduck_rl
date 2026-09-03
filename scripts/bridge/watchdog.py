"""Brain liveness timer shared by the sim runners.

Counts sim time since the last brain request and fires once when the brain
has gone quiet, so the runner can fall back to the trained stand-still.
"""

from bridge.state import BRAIN_TIMEOUT_S, BridgeState


class BrainWatchdog:
    """Fires once after timeout_s of sim time without a brain request."""

    def __init__(self, state: BridgeState, control_dt: float, timeout_s: float = BRAIN_TIMEOUT_S):
        self._state = state
        self._control_dt = float(control_dt)
        self._timeout_s = float(timeout_s)
        self._quiet_seconds = 0.0
        self._last_request_count = state.request_count()
        self._fired = False

    def tick(self) -> bool:
        """One control step. True on the step the quiet time reaches the timeout, then False."""
        if self._brain_spoke():
            return False

        self._quiet_seconds += self._control_dt

        if self._fired or self._quiet_seconds < self._timeout_s:
            return False

        self._fired = True

        return True

    def _brain_spoke(self) -> bool:
        """True when a new request arrived. Rearms the timer for the next silence."""
        count = self._state.request_count()

        if count == self._last_request_count:
            return False

        self._last_request_count = count
        self._quiet_seconds = 0.0
        self._fired = False

        return True

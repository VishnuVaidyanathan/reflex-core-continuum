"""
rcc.core.still
==============
Phase 15 — The Still Layer

Provides low-energy homeostasis where all layers rest in synchrony.
"The Still Layer is not shutdown; it's conscious quiet."

Core principles (spec Vol. 8)
--------------------------------
Non-reactivity       : no new impulses processed; only residual tone observed
Temporal Transparency: time passes without data loss; context preserved
Resonant Reset       : emotional charge neutralised while rhythm retained

State equation
--------------
    E(t+1) = E(t) × (1 − λ)
    λ ≈ 0.45 … 0.55 (decay constant)

Activation conditions
---------------------
    Echo intensity < 0.10  (STILL_THRESHOLD)
    Turn > ACTIVATION_MIN_TURN  (at least 3 turns in)
    Not already in Still

Settling time
-------------
    Within 3 seconds of activation:
    • Echo drops to < 0.1
    • Ripple stops oscillating
    • Virtue remains anchored (score frozen at activation)
"""
from __future__ import annotations
import time

from rcc.types import EchoState, StillState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAMBDA:                  float = 0.50   # decay constant (mid-range of spec 0.45–0.55)
ACTIVATION_THRESHOLD:    float = 0.10   # E < this → eligible
ACTIVATION_MIN_TURN:     int   = 3      # must have at least 3 turns
SETTLING_TIME_S:         float = 3.0   # spec: within 3 s all layers settle
RESUME_SEED_INTENSITY:   float = 0.10  # echo seeded at 0.1 on resume (spec)


# ---------------------------------------------------------------------------
# StillLayer  (Phase 15 entry point)
# ---------------------------------------------------------------------------

class StillLayer:
    """
    Phase 15 — Still Layer

    The Still Layer monitors the Echo Layer on each turn and activates
    when the conversation goes quiet (low echo, sufficient turns).

    Once active it suppresses the pipeline from processing new impulses
    (the Continuum orchestrator checks ``is_active`` before dispatching
    to Reflex) and instead holds the last virtue score as an anchor.

    ``resume()`` is called by the user or automatically when new input
    arrives while still is active.

    Spec quote: "Silence is not absence; it's perfect synchronisation."
    """

    def __init__(self) -> None:
        self._state = StillState(active=False)
        self._activation_monotonic: float = 0.0

    # --- public API -------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._state.active

    @property
    def state(self) -> StillState:
        return self._state

    def check_activation(
        self,
        echo: EchoState,
        turn: int,
        virtue_score: float = 0.0,
    ) -> bool:
        """
        Determine whether Still Layer should activate.

        Conditions (all must hold):
        1. Not already in Still.
        2. At least ACTIVATION_MIN_TURN turns have elapsed.
        3. Echo intensity is below ACTIVATION_THRESHOLD.
        """
        if self._state.active:
            return False
        if turn < ACTIVATION_MIN_TURN:
            return False
        if echo.intensity >= ACTIVATION_THRESHOLD:
            return False

        # All conditions met → activate
        self._activate(echo, turn, virtue_score)
        return True

    def decay_echo(self, echo: EchoState) -> EchoState:
        """
        Apply Still-layer decay to the echo state while active.
        E(t+1) = E(t) × (1 − λ)
        Valence fades more slowly (rhythm retained, charge neutralised).
        """
        if not self._state.active:
            return echo

        new_intensity = max(0.0, echo.intensity * (1.0 - LAMBDA))
        new_valence   = echo.valence * (1.0 - LAMBDA * 0.4)   # valence fades 40 % of λ

        return EchoState(
            intensity=new_intensity,
            valence=max(-1.0, min(1.0, new_valence)),
            decay_lambda=LAMBDA,
            residue_age_s=echo.residue_age_s + 1.0,
            still_eligible=True,
        )

    def resume(self, seed_valence: float = 0.0) -> EchoState:
        """
        Exit Still Layer and seed the Echo with a minimal intensity.
        Returns the seed EchoState for the Echo layer to absorb.
        Spec: echo = 0.1 on resume.
        """
        elapsed = time.monotonic() - self._activation_monotonic
        self._state = StillState(
            active=False,
            activation_turn=self._state.activation_turn,
            activation_time_s=elapsed,
            echo_at_activation=self._state.echo_at_activation,
            virtue_anchor=self._state.virtue_anchor,
        )
        # Seed echo just above threshold
        return EchoState(
            intensity=RESUME_SEED_INTENSITY,
            valence=seed_valence,
            decay_lambda=LAMBDA,
            residue_age_s=0.0,
            still_eligible=False,
        )

    def time_in_still(self) -> float:
        """Seconds elapsed since Still activation (0 if not active)."""
        if not self._state.active:
            return 0.0
        return time.monotonic() - self._activation_monotonic

    def is_settled(self) -> bool:
        """
        True once the settling time has passed (Echo < 0.01 expected).
        Spec: "Within 3 seconds of still activation, Echo drops to < 0.1,
        Ripple stops oscillation, Virtue remains anchored."
        """
        return self._state.active and self.time_in_still() >= SETTLING_TIME_S

    # --- private helpers --------------------------------------------------

    def _activate(self, echo: EchoState, turn: int, virtue_score: float) -> None:
        self._activation_monotonic = time.monotonic()
        self._state = StillState(
            active=True,
            activation_turn=turn,
            activation_time_s=0.0,
            echo_at_activation=echo.intensity,
            virtue_anchor=virtue_score,
        )

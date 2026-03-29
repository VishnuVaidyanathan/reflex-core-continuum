"""
rcc.core.echo
=============
Phase 2 + Phase 5 — Echo Layer

Holds short-term emotional residue (2–3 s) with natural, asymmetric decay.

State equation (per tick)
-------------------------
    E(t+1) = E(t) * (1 - λ)

where λ is asymmetric:
    • Positive valence (joy, gratitude …)  → λ = 0.40  (lasts longer)
    • Negative valence (anger, sadness …)  → λ = 0.55  (fades faster)
    • Neutral                              → λ = 0.48

Human calibration insight (from spec Vol. 1):
    "Anger fades quickly; joy lasts longer → asymmetric decay constants."
    "Physical tiredness → sarcasm; mental tiredness → calm →
     Ripple tunes tone dynamically."
"""
from __future__ import annotations
import math

from rcc.types import EchoState, EmotionClass, ReflexSignal


# ---------------------------------------------------------------------------
# Decay constants
# ---------------------------------------------------------------------------

LAMBDA_POSITIVE: float = 0.40   # slower decay for positive valence
LAMBDA_NEGATIVE: float = 0.55   # faster decay for negative valence
LAMBDA_NEUTRAL:  float = 0.48

STILL_THRESHOLD: float  = 0.10  # E < 0.10 → eligible for Still Layer activation
RESIDUE_DURATION_S: float = 2.5 # spec: "holds 2–3 s emotional residue"


def _select_lambda(valence: float) -> float:
    """Pick decay constant based on emotional valence."""
    if valence > 0.05:
        return LAMBDA_POSITIVE
    if valence < -0.05:
        return LAMBDA_NEGATIVE
    return LAMBDA_NEUTRAL


# ---------------------------------------------------------------------------
# EchoLayer
# ---------------------------------------------------------------------------

class EchoLayer:
    """
    Phase 2 + Phase 5 — Echo Layer

    After each ``update()``, the Echo Layer carries the last emotional impression
    forward in time. On every ``decay()`` tick it moves toward silence via the
    state equation ``E(t+1) = E(t) * (1 - λ)``.

    The layer also models the *dual fatigue* insight from the spec:
    •  Physical tiredness  → slightly elevated arousal (sarcasm potential)
    •  Mental tiredness    → suppressed arousal (calm potential)
    Both are encoded as a ``fatigue_mode`` flag consumed by the Ripple layer.
    """

    def __init__(self) -> None:
        self._state = EchoState(
            intensity=0.0,
            valence=0.0,
            decay_lambda=LAMBDA_NEUTRAL,
            residue_age_s=0.0,
            still_eligible=False,
        )
        self._fatigue_mode: str = "none"   # "physical" | "mental" | "none"
        self._prev_intensity: float = 0.0

    # --- public API -------------------------------------------------------

    @property
    def state(self) -> EchoState:
        return self._state

    @property
    def previous_intensity(self) -> float:
        return self._prev_intensity

    @property
    def fatigue_mode(self) -> str:
        return self._fatigue_mode

    def update(self, signal: ReflexSignal) -> EchoState:
        """
        Absorb a new ReflexSignal.

        The echo *rises* to meet the incoming intensity if it is stronger,
        and *blends* if the incoming signal is weaker (residue + new signal).
        This models the bell-curve rise-and-fall visible in Vol. 1 Table.
        """
        self._prev_intensity = self._state.intensity
        iv = signal.impulse.emotional_vector
        lam = _select_lambda(iv.valence)

        # Rising edge: new signal pushes echo up (or blends if weaker)
        if signal.intensity >= self._state.intensity:
            new_intensity = signal.intensity
        else:
            # Blend: residue + new signal contribution
            new_intensity = max(self._state.intensity * 0.6 + signal.intensity * 0.4,
                                signal.intensity)

        new_intensity = min(1.0, new_intensity)
        new_valence   = (self._state.valence * 0.3 + iv.valence * 0.7)
        new_valence   = max(-1.0, min(1.0, new_valence))

        self._fatigue_mode = self._classify_fatigue(signal)

        self._state = EchoState(
            intensity=new_intensity,
            valence=new_valence,
            decay_lambda=lam,
            residue_age_s=0.0,
            still_eligible=new_intensity < STILL_THRESHOLD,
        )
        return self._state

    def decay(self, dt_seconds: float = 1.0) -> EchoState:
        """
        Apply time-based decay: ``E(t+1) = E(t) * (1 - λ) ^ dt_factor``.

        ``dt_seconds`` represents elapsed real time since last decay tick.
        The spec target is that Echo should drop to near-zero within
        RESIDUE_DURATION_S seconds at natural λ.
        """
        # Scale λ to dt so that at dt=RESIDUE_DURATION_S the echo has mostly cleared
        effective_lambda = 1.0 - math.pow(1.0 - self._state.decay_lambda, dt_seconds)
        new_intensity = max(0.0, self._state.intensity * (1.0 - effective_lambda))
        new_valence   = self._state.valence * (1.0 - effective_lambda * 0.5)  # valence fades slower

        self._state = EchoState(
            intensity=new_intensity,
            valence=max(-1.0, min(1.0, new_valence)),
            decay_lambda=self._state.decay_lambda,
            residue_age_s=self._state.residue_age_s + dt_seconds,
            still_eligible=new_intensity < STILL_THRESHOLD,
        )
        return self._state

    def force_clear(self) -> None:
        """Hard reset (called by Still Layer activation)."""
        self._state = EchoState(
            intensity=0.0,
            valence=0.0,
            decay_lambda=LAMBDA_NEUTRAL,
            residue_age_s=0.0,
            still_eligible=True,
        )
        self._prev_intensity = 0.0
        self._fatigue_mode = "none"

    def seed(self, intensity: float = 0.10, valence: float = 0.0) -> EchoState:
        """Seed the layer after a Still resume (spec: echo = 0.1 on resume)."""
        self._prev_intensity = 0.0
        self._state = EchoState(
            intensity=intensity,
            valence=valence,
            decay_lambda=_select_lambda(valence),
            residue_age_s=0.0,
            still_eligible=False,
        )
        return self._state

    # --- private helpers --------------------------------------------------

    @staticmethod
    def _classify_fatigue(signal: ReflexSignal) -> str:
        """
        Spec insight:
            Physical tiredness → arousal slightly above baseline → sarcasm mode
            Mental tiredness   → arousal suppressed           → calm mode
        Use arousal vs intensity mismatch as the discriminator.
        """
        tm = signal.impulse.tone_markers
        if tm.fatigue < 0.3:
            return "none"
        arousal = signal.impulse.emotional_vector.arousal
        if arousal > 0.4:
            return "physical"   # tired but wired
        return "mental"         # tired and quiet

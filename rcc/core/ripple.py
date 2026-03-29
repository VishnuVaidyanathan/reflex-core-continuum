"""
rcc.core.ripple
===============
Phase 2 + Phase 5 — Ripple Layer

Tracks the *direction* of emotional change — warmer, calmer, sharper, etc.

Human calibration insights from spec Vol. 1:
    1. "Mirroring comforts briefly, contrast revives connection → Ripple shifts
       tone after 3–4 turns."
    2. "Humor can hide pain; return to seriousness later → Layered coping pattern →
       Ripple suspends serious tone beneath humour."
    3. Oscillation pattern from Vol. 1 table: humor peak → reflection → calm → stillness.

The Ripple layer consumes two consecutive EchoStates and produces a RippleState
that tells every downstream layer *where the emotion is heading*, not just where it is.
"""
from __future__ import annotations

from rcc.types import EchoState, RippleState, RippleTrend


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

WARMING_THRESHOLD:  float = 0.30    # direction > this → "warming"
RISING_THRESHOLD:   float = 0.05    # direction > this → "rising"
FADING_THRESHOLD:   float = -0.05   # direction < this → "fading"
COOLING_THRESHOLD:  float = -0.30   # direction < this → "cooling"

CONTRAST_TURN_THRESHOLD: int = 4    # shift to contrast after N turns on same trend
HUMOR_MASK_VALENCE:  float = 0.55   # valence above this in FUNNY context = humor masking pain


class RippleLayer:
    """
    Phase 2 + Phase 5 — Ripple Layer

    ``compute()`` is the main entry point; it consumes the current and
    previous EchoState and returns a fresh RippleState.

    Internal state is kept to track multi-turn patterns (contrast trigger,
    humor mask detection).
    """

    def __init__(self) -> None:
        self._turns_on_trend: int = 0
        self._last_trend: RippleTrend = RippleTrend.NEUTRAL
        self._momentum: float = 0.0      # second-order rate of change
        self._humor_mask_counter: int = 0

    # --- public API -------------------------------------------------------

    def compute(
        self,
        current: EchoState,
        previous: EchoState,
        turn: int = 0,
        fatigue_mode: str = "none",
    ) -> RippleState:
        """
        Derive the Ripple state from the delta between current and previous echo.

        Parameters
        ----------
        current     : EchoState from this turn
        previous    : EchoState from the prior turn
        turn        : global turn counter (used for contrast detection)
        fatigue_mode: 'physical' | 'mental' | 'none' (from EchoLayer)
        """
        direction = current.intensity - previous.intensity
        magnitude = abs(direction)

        # Second-order momentum — acceleration of emotional change
        prev_direction = previous.intensity - (previous.intensity * (1 + previous.decay_lambda))
        self._momentum = direction - prev_direction

        trend = self._classify_trend(direction, current.valence, fatigue_mode)

        # Multi-turn contrast logic: "Ripple shifts tone after 3-4 turns"
        if trend == self._last_trend:
            self._turns_on_trend += 1
        else:
            self._turns_on_trend = 1
            self._last_trend = trend

        # Humor mask detection: "Humor can hide pain; Ripple suspends serious tone beneath it"
        humor_mask = self._detect_humor_mask(current, direction)

        return RippleState(
            direction=direction,
            magnitude=magnitude,
            trend=trend,
            momentum=self._momentum,
            humor_mask_active=humor_mask,
            turns_on_trend=self._turns_on_trend,
        )

    @property
    def should_contrast(self) -> bool:
        """
        True when the layer has mirrored the user's tone for long enough
        and now recommends switching to contrast to re-engage.
        Spec: "Mirroring comforts briefly, contrast revives connection."
        """
        return self._turns_on_trend >= CONTRAST_TURN_THRESHOLD

    def reset(self) -> None:
        """Called by the Still Layer on activation."""
        self._turns_on_trend = 0
        self._last_trend = RippleTrend.NEUTRAL
        self._momentum = 0.0
        self._humor_mask_counter = 0

    # --- private helpers --------------------------------------------------

    def _classify_trend(
        self,
        direction: float,
        valence: float,
        fatigue_mode: str,
    ) -> RippleTrend:
        """
        Classify direction into a named trend.
        Fatigue modes modulate thresholds per spec calibration notes.
        """
        # Physical fatigue can create ironic warming (sarcasm feels warm on surface)
        if fatigue_mode == "physical" and direction < 0:
            direction *= 0.5   # dampen cooling signal
        # Mental fatigue suppresses activation
        if fatigue_mode == "mental":
            direction *= 0.7

        if direction > WARMING_THRESHOLD:
            return RippleTrend.WARMING
        if direction > RISING_THRESHOLD:
            return RippleTrend.RISING
        if direction < COOLING_THRESHOLD:
            return RippleTrend.COOLING
        if direction < FADING_THRESHOLD:
            return RippleTrend.FADING
        return RippleTrend.NEUTRAL

    @staticmethod
    def _detect_humor_mask(echo: EchoState, direction: float) -> bool:
        """
        Spec insight: "Humor can hide pain; return to seriousness later."
        Detect when high positive valence (humor) co-exists with a
        downward direction — humor is masking an underlying negative state.
        """
        return (
            echo.valence > HUMOR_MASK_VALENCE
            and direction < -0.05
        )

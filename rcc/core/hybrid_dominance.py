"""
rcc.core.hybrid_dominance
=========================
Phase 4 — Hybrid Dominance Shift

Enables smooth, natural transfer of control from Reflex dominance to
Adaptive dominance as conversations mature.

Transition curve (from spec Vol. 2)
-------------------------------------
    Dominance(t) = 1 / (1 + e^(-k * (t - t0)))

    t0 ≈ 20   (threshold turns)
    k  ≈ 0.2  (transition steepness)

Behavioural flow (spec Vol. 2 table)
--------------------------------------
    Early  (1–10 turns)   reflex=0.85  adaptive=0.15  quick emotional reflex
    Mid    (10–25 turns)  reflex=0.55  adaptive=0.45  adaptive learning emerges
    Late   (25+ turns)    reflex=0.25  adaptive=0.75  stable contextual understanding

Ripple assistance: if conversation drifts cold, Ripple re-amplifies Reflex
warmth temporarily (emergency warm-up path).
"""
from __future__ import annotations
import math

from rcc.types import (
    AdaptiveOutput, BlendedSignal, HybridWeights, ReflexSignal, RippleTrend,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

T0: float = 20.0   # sigmoid midpoint (turns)
K:  float = 0.2    # transition steepness

EMERGENCY_REFLEX_BOOST: float = 0.15  # added to reflex_weight when Ripple is COOLING


# ---------------------------------------------------------------------------
# HybridDominanceController
# ---------------------------------------------------------------------------

class HybridDominanceController:
    """
    Phase 4 — Hybrid Dominance Controller

    Computes blend weights at each turn and produces a ``BlendedSignal``
    that combines the instantaneous Reflex signal with the context-aware
    Adaptive output.

    The Ripple layer can trigger a temporary emergency boost to Reflex
    weight when the conversation drifts cold (spec: "Ripple Layer monitors
    emotional direction; if conversation drifts cold, it temporarily
    re-amplifies Reflex warmth to re-engage the user.").
    """

    def __init__(self) -> None:
        self._last_weights: HybridWeights = HybridWeights(
            reflex_weight=0.85,
            adaptive_weight=0.15,
            turn=0,
        )

    # --- public API -------------------------------------------------------

    def compute_weights(self, turn: int, ripple_trend: RippleTrend = RippleTrend.NEUTRAL) -> HybridWeights:
        """
        Compute sigmoid-derived weights for this turn.

        If the Ripple trend is COOLING, a temporary emergency boost is
        added to the reflex weight and subtracted from adaptive weight,
        then re-normalised to sum to 1.
        """
        adaptive = self._sigmoid(turn)
        reflex   = 1.0 - adaptive

        if ripple_trend == RippleTrend.COOLING:
            reflex   = min(1.0, reflex + EMERGENCY_REFLEX_BOOST)
            adaptive = 1.0 - reflex

        self._last_weights = HybridWeights(
            reflex_weight=round(reflex, 4),
            adaptive_weight=round(adaptive, 4),
            turn=turn,
        )
        return self._last_weights

    def blend(
        self,
        reflex_out: ReflexSignal,
        adaptive_out: AdaptiveOutput,
        turn: int,
        ripple_trend: RippleTrend = RippleTrend.NEUTRAL,
    ) -> BlendedSignal:
        """
        Produce a BlendedSignal that merges Reflex and Adaptive according
        to the current dominance weights.

        Intensity  : weighted sum
        Latency    : weighted harmonic (fast reflex, slower adaptive)
        Valence    : weighted sum (Adaptive does not change valence, only moderates it)
        """
        weights = self.compute_weights(turn, ripple_trend)
        wr, wa  = weights.reflex_weight, weights.adaptive_weight

        blended_intensity = wr * reflex_out.intensity + wa * adaptive_out.adjusted_intensity
        # Harmonic blend of latencies: reflex is fast, adaptive is slower
        blended_latency = (
            wr * reflex_out.latency_ms + wa * adaptive_out.adjusted_latency_ms
        )
        blended_valence  = reflex_out.impulse.emotional_vector.valence

        return BlendedSignal(
            intensity=min(1.0, blended_intensity),
            latency_ms=blended_latency,
            valence=max(-1.0, min(1.0, blended_valence)),
            weights=weights,
            context_score=adaptive_out.context_score,
        )

    @property
    def weights(self) -> HybridWeights:
        return self._last_weights

    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(t: float, t0: float = T0, k: float = K) -> float:
        """Dominance(t) = 1 / (1 + e^(-k*(t-t0)))"""
        exponent = -k * (t - t0)
        # Clamp exponent to avoid overflow
        exponent = max(-500.0, min(500.0, exponent))
        return 1.0 / (1.0 + math.exp(exponent))

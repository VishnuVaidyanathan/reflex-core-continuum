"""
rcc.core.voice
==============
Phase 6 — Emergent Voice Formation

Develops a consistent speaking style by fusing Echo/Ripple data with
language output control.

Pipeline (spec Vol. 3)
-----------------------
    Input → Emotional State Vector (E) → Tone Regulator (TR)
                                       → Voice Composer (VC) → Output

Key processes
-------------
Tone Regulator   — aligns emotional amplitude with context; prevents
                   over- or under-reaction.
Voice Composer   — shapes rhythm, pause, and warmth; ensures responses
                   "sound like" the same entity.
Harmony Index H  — quantifies resonance between Reflex timing and
                   Adaptive content.

Harmony metrics (spec Vol. 3 table)
-------------------------------------
    Turn  1 → H ≈ 0.70  (soft response start)
    Turn  5 → H ≈ 0.85  (natural blend emerges)
    Turn 10 → H ≈ 0.88  (stable voice tone)

After ~10 turns the system reaches H ≈ 0.88 — perceived as a steady,
human-like personality.
"""
from __future__ import annotations
import math

from rcc.types import (
    BlendedSignal, EchoState, RippleState, ToneProfile, VoiceOutput,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARMONY_TARGET:    float = 0.88     # spec: H ≈ 0.88 at steady state
HARMONY_TURN_REF:  int   = 10       # turns to reach target
OPTIMAL_LATENCY:   float = 460.0    # spec: 460 ms harmonic sweet-spot
WARMTH_DECAY:      float = 0.85     # EMA coefficient for warmth smoothing
RHYTHM_DECAY:      float = 0.90     # EMA for rhythm ms


# ---------------------------------------------------------------------------
# ToneRegulator
# ---------------------------------------------------------------------------

class ToneRegulator:
    """
    Aligns emotional amplitude with context so the system neither
    over-reacts (amplified reflex) nor under-reacts (flat affect).

    Outputs a ``ToneProfile`` that the VoiceComposer uses to shape timing
    and warmth.

    Spec guarantee: "prevents over- or under-reaction."
    """

    def __init__(self) -> None:
        self._smoothed_amplitude: float = 0.5
        self._smoothed_warmth:    float = 0.5

    def regulate(
        self,
        echo: EchoState,
        ripple: RippleState,
        blended: BlendedSignal,
    ) -> ToneProfile:
        """
        Produce a regulated ToneProfile.

        Amplitude  — EMA of blended intensity, dampened when in humor-mask
                     state (surface warmth hiding deeper signal).
        Warmth     — derived from echo valence + ripple direction.
        Pace       — high ripple magnitude → brisker; Still-eligible → slower.
        Pause hint — larger when echo is decaying (reflective moment needed).
        """
        raw_amplitude = blended.intensity
        if ripple.humor_mask_active:
            raw_amplitude *= 0.8   # don't let masked humor overdrive tone

        self._smoothed_amplitude = (
            WARMTH_DECAY * self._smoothed_amplitude + (1 - WARMTH_DECAY) * raw_amplitude
        )

        # Warmth from valence + positive ripple direction
        raw_warmth = (echo.valence + 1.0) / 2.0   # map [-1,+1] → [0,1]
        if ripple.direction > 0:
            raw_warmth = min(1.0, raw_warmth + ripple.magnitude * 0.2)
        elif ripple.direction < 0:
            raw_warmth = max(0.0, raw_warmth + ripple.direction * 0.2)

        self._smoothed_warmth = (
            WARMTH_DECAY * self._smoothed_warmth + (1 - WARMTH_DECAY) * raw_warmth
        )

        # Pace: magnitude drives quickness; still-eligible state → very slow
        pace = min(1.0, 0.5 + ripple.magnitude * 0.5)
        if echo.still_eligible:
            pace = max(0.0, pace - 0.4)

        # Pause hint: high when echo is fading (moment of reflection)
        residue_factor = 1.0 - min(1.0, echo.residue_age_s / 3.0)
        pause_hint = (1.0 - self._smoothed_amplitude) * residue_factor * 300.0

        return ToneProfile(
            amplitude=round(self._smoothed_amplitude, 4),
            warmth=round(self._smoothed_warmth, 4),
            pace=round(pace, 4),
            pause_hint_ms=round(pause_hint, 1),
        )


# ---------------------------------------------------------------------------
# VoiceComposer
# ---------------------------------------------------------------------------

class VoiceComposer:
    """
    Phase 6 — Voice Composer

    Shapes rhythm, pause, and warmth into a coherent ``VoiceOutput``.
    Tracks the Harmony Index H across turns: H rises toward 0.88 as
    timing and content converge.

    Spec table reproduced:
        Turn 1:  reflex_timing 300 ms,  adaptive_context 0.74  → H 0.70
        Turn 5:  reflex_timing 410 ms,  adaptive_context 0.89  → H 0.85
        Turn 10: reflex_timing 460 ms,  adaptive_context 0.91  → H 0.88
    """

    def __init__(self) -> None:
        self._harmony: float = 0.50   # starts low, converges to 0.88
        self._smoothed_rhythm: float = OPTIMAL_LATENCY
        self._regulator = ToneRegulator()

    # --- public API -------------------------------------------------------

    def compose(
        self,
        blended: BlendedSignal,
        echo: EchoState,
        ripple: RippleState,
        turn: int,
    ) -> VoiceOutput:
        """
        Produce a VoiceOutput for this turn.

        1. Regulate tone amplitude through ToneRegulator.
        2. Compute Harmony Index from latency + context score.
        3. Smooth rhythm (EMA) so timing feels consistent.
        4. Return full VoiceOutput.
        """
        tone = self._regulator.regulate(echo, ripple, blended)
        self._harmony = self.compute_harmony(blended.latency_ms, blended.context_score, turn)
        self._smoothed_rhythm = (
            RHYTHM_DECAY * self._smoothed_rhythm + (1 - RHYTHM_DECAY) * blended.latency_ms
        )

        # Add pause hint on top of base rhythm
        final_rhythm_ms = self._smoothed_rhythm + tone.pause_hint_ms

        return VoiceOutput(
            tone_profile=tone,
            harmony_index=round(self._harmony, 4),
            rhythm_ms=round(final_rhythm_ms, 1),
            warmth=tone.warmth,
        )

    def compute_harmony(
        self,
        reflex_timing_ms: float,
        adaptive_context_score: float,
        turn: int = 0,
    ) -> float:
        """
        H = resonance(reflex_timing, adaptive_content)

        Timing score: how close reflex_timing is to the OPTIMAL_LATENCY.
        Content score: directly from adaptive context alignment.
        Converges toward HARMONY_TARGET as turn increases.

        H = 0.5 * timing_score + 0.5 * content_score
        then blended with running H via turn-weighted EMA.
        """
        timing_score = 1.0 - abs(reflex_timing_ms - OPTIMAL_LATENCY) / OPTIMAL_LATENCY
        timing_score = max(0.0, min(1.0, timing_score))

        raw_h = 0.50 * timing_score + 0.50 * adaptive_context_score

        # Learning rate: approaches HARMONY_TARGET faster with more turns
        lr = min(0.30, 0.05 + turn * 0.02)
        h = self._harmony * (1 - lr) + raw_h * lr

        # Clamp; never exceeds HARMONY_TARGET (spec: H ≈ 0.88)
        return min(HARMONY_TARGET, max(0.0, h))

    @property
    def harmony(self) -> float:
        return self._harmony

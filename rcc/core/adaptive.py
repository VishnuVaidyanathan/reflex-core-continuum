"""
rcc.core.adaptive
=================
Phase 3 — Adaptive Layer Integration

Blends the instinctive Reflex responses with adaptive reasoning without
losing spontaneity.

Key processes (per spec Vol. 2)
--------------------------------
Feedback Alignment  — compares Reflex output against user satisfaction to
                      learn desirable tone ranges.
Latency Balance     — adjusts delay dynamically; too fast = impulsive,
                      too slow = mechanical. Optimal ≈ 460 ms.
Tone Normalization  — re-centres emotional amplitude so Reflex does not
                      drift into over-reaction.

After ~20 turns the Adaptive Layer learns user rhythm and begins to
predict tone requirements before Reflex fires.
"""
from __future__ import annotations
import math
from collections import deque

from rcc.types import (
    AdaptiveOutput, ConversationContext, EmotionClass, ReflexSignal,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEARNING_THRESHOLD: int   = 20    # turns before full learning activates
OPTIMAL_LATENCY_MS: float = 460.0 # spec Vol. 2 "Latency (Ms) 460 — feels human-natural"
LATENCY_TOO_FAST:   float = 240.0 # spec: pure reflex
LATENCY_TOO_SLOW:   float = 820.0 # spec: pure adaptive
FEEDBACK_WINDOW:    int   = 10    # rolling window for feedback alignment
TONE_DECAY:         float = 0.92  # exponential smoothing coefficient for tone range


# ---------------------------------------------------------------------------
# Sub-components
# ---------------------------------------------------------------------------

class FeedbackAligner:
    """
    Tracks whether Reflex outputs are being "accepted" by the user
    (positive emotional escalation / continued engagement) and returns
    a running alignment score in [0, 1].

    A score near 1.0 means Reflex is on-target; near 0 means it is
    consistently mis-calibrated and the Adaptive layer should intervene more.
    """

    def __init__(self, window: int = FEEDBACK_WINDOW) -> None:
        self._history: deque[float] = deque(maxlen=window)

    def record(self, accepted: bool) -> None:
        self._history.append(1.0 if accepted else 0.0)

    def score(self) -> float:
        if not self._history:
            return 0.75   # default optimism before data
        return sum(self._history) / len(self._history)

    def infer_from_ripple(self, ripple_direction: float) -> None:
        """
        In the absence of explicit feedback, a positive ripple direction
        is treated as implicit acceptance; negative as rejection.
        """
        self.record(ripple_direction >= 0.0)


class LatencyBalancer:
    """
    Adjusts response latency dynamically.
    • Too fast (< ~240 ms) → feels impulsive, reduce to OPTIMAL
    • Too slow (> ~820 ms) → feels mechanical, speed up to OPTIMAL
    • Target convergence toward 460 ms with learned user rhythm.
    """

    def __init__(self) -> None:
        self._current_ms: float = OPTIMAL_LATENCY_MS
        self._rhythm_mult: float = 1.0   # user rhythm multiplier

    def balance(self, reflex_latency_ms: float, feedback_score: float) -> float:
        """
        Pull reflex latency toward OPTIMAL weighted by feedback quality.
        Users who respond quickly (short messages fast) favour lower latency;
        users who write slowly favour higher latency.
        """
        pull = (OPTIMAL_LATENCY_MS - reflex_latency_ms) * (1.0 - feedback_score) * 0.3
        balanced = reflex_latency_ms + pull
        balanced *= self._rhythm_mult
        self._current_ms = balanced
        return max(LATENCY_TOO_FAST * 0.9, min(LATENCY_TOO_SLOW * 1.1, balanced))

    def learn_rhythm(self, context: ConversationContext) -> None:
        """
        Infer user rhythm from message pace history.
        Fast pace → tighten latency; slow pace → relax.
        """
        if len(context.message_lengths) < 5:
            return
        recent = context.message_lengths[-5:]
        avg = sum(recent) / len(recent)
        # Pace multiplier: short messages → faster rhythm
        if avg < 20:
            self._rhythm_mult = 0.85
        elif avg > 120:
            self._rhythm_mult = 1.15
        else:
            self._rhythm_mult = 1.0
        context.learned_rhythm = self._rhythm_mult

    @property
    def current_ms(self) -> float:
        return self._current_ms


class ToneNormalizer:
    """
    Prevents Reflex from drifting into over-reaction by maintaining a
    rolling estimate of the "natural" tone range for this user and
    re-centering intensity when it exceeds 2 standard deviations above
    the recent mean.
    """

    def __init__(self, window: int = 20) -> None:
        self._history: deque[float] = deque(maxlen=window)
        self._tone_floor: float = -1.0
        self._tone_ceil:  float =  1.0

    def record(self, intensity: float) -> None:
        self._history.append(intensity)
        if len(self._history) >= 5:
            mean = sum(self._history) / len(self._history)
            var  = sum((x - mean) ** 2 for x in self._history) / len(self._history)
            std  = math.sqrt(var)
            self._tone_floor = max(-1.0, mean - 2 * std)
            self._tone_ceil  = min( 1.0, mean + 2 * std)

    def normalize(self, intensity: float, valence: float) -> tuple[float, float]:
        """Return (normalized_intensity, tone_range) clamped to learned range."""
        n_intensity = max(0.0, min(1.0, intensity))
        # If intensity is in the extreme tail, dampen it slightly
        if n_intensity > self._tone_ceil and self._tone_ceil < 1.0:
            n_intensity = self._tone_ceil + (n_intensity - self._tone_ceil) * 0.5
        return n_intensity, (self._tone_floor, self._tone_ceil)


# ---------------------------------------------------------------------------
# AdaptiveLayer  (Phase 3 entry point)
# ---------------------------------------------------------------------------

class AdaptiveLayer:
    """
    Phase 3 — Adaptive Layer Integration

    Receives a ReflexSignal and the current ConversationContext and returns
    an AdaptiveOutput that downstream layers (Hybrid Dominance, Voice) use.

    Spec guarantee: "After ~20 turns, the Adaptive Layer learns user rhythm
    and begins to predict tone requirements before Reflex fires."
    """

    def __init__(self) -> None:
        self.feedback_aligner  = FeedbackAligner()
        self.latency_balancer  = LatencyBalancer()
        self.tone_normalizer   = ToneNormalizer()

    def process(
        self,
        signal: ReflexSignal,
        context: ConversationContext,
        ripple_direction: float = 0.0,
    ) -> AdaptiveOutput:
        """
        Main processing method.

        1. Record implicit feedback from ripple direction.
        2. Normalize tone amplitude.
        3. Balance latency.
        4. Compute context_score (how well reflex matched expected tone).
        5. Apply learned rhythm once LEARNING_THRESHOLD is reached.
        """
        learning_active = context.turns >= LEARNING_THRESHOLD

        # Implicit feedback from ripple direction
        self.feedback_aligner.infer_from_ripple(ripple_direction)
        feedback_score = self.feedback_aligner.score()

        # Learn rhythm after threshold
        if learning_active:
            self.latency_balancer.learn_rhythm(context)

        # Normalize tone
        self.tone_normalizer.record(signal.intensity)
        adj_intensity, tone_range = self.tone_normalizer.normalize(
            signal.intensity,
            signal.impulse.emotional_vector.valence,
        )

        # Balance latency
        adj_latency = self.latency_balancer.balance(signal.latency_ms, feedback_score)

        # Context score: alignment between detected emotion and conversation history
        context_score = self._compute_context_score(signal, context, feedback_score)

        return AdaptiveOutput(
            adjusted_intensity=adj_intensity,
            adjusted_latency_ms=adj_latency,
            context_score=context_score,
            learning_active=learning_active,
            tone_range=tone_range,
        )

    def record_explicit_feedback(self, accepted: bool) -> None:
        """Called by higher layers when user satisfaction can be assessed."""
        self.feedback_aligner.record(accepted)

    # ------------------------------------------------------------------

    def _compute_context_score(
        self,
        signal: ReflexSignal,
        context: ConversationContext,
        feedback_score: float,
    ) -> float:
        """
        Context score measures how well the reflex response type matches
        the current conversation trajectory.

        Heuristics:
        - High feedback score → high context alignment
        - Emotion consistency with history → bonus
        - Pattern mismatch (e.g. humor during grief streak) → penalty
        """
        score = feedback_score

        if len(context.emotion_history) >= 3:
            recent_emotions = context.emotion_history[-3:]
            current_ec = signal.impulse.emotional_vector.emotion_class
            # Consistency bonus
            if all(e == current_ec for e in recent_emotions):
                score = min(1.0, score + 0.10)
            # Abrupt switch penalty (e.g. FUNNY after 3× SAD)
            if (current_ec == EmotionClass.FUNNY
                    and all(e == EmotionClass.SAD for e in recent_emotions)):
                score = max(0.0, score - 0.15)

        # Clamp and smooth
        return max(0.0, min(1.0, score))

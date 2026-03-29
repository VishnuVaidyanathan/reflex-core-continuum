"""
rcc.core.virtue
===============
Phase 12 — Virtue Interface

Transforms the Conscience Stack into a practical trust layer between the
system and humans — the moment intelligence becomes trustworthy.

Mechanism (spec Vol. 6)
------------------------
    Input → Tone & Context Analysis → Virtue Selector → Output

Virtue Selectors (spec table)
------------------------------
    Patience       : adds delay before response         → calm, measured
    Humility       : softens assertive language         → gentle
    Honesty        : rejects false empathy              → transparent
    Kindness       : weights Echo decay toward warmth   → comforting
    Responsibility : monitors Ripple direction for harm → corrective

Computation
-----------
    V = Σ(wi × Outcomei)
    if V < threshold → reduce reflex gain

Performance metrics (spec)
---------------------------
    User Trust Index   : 0.93
    Conflict Incidence : −0.42 (reduced tone misfires)
    Virtue Consistency : 0.89
"""
from __future__ import annotations

from rcc.types import (
    BlendedSignal, ConscienceOutput, ConversationContext,
    EmotionClass, RippleState, RippleTrend,
    VirtueOutput, VirtueScores,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIRTUE_THRESHOLD: float = 0.50   # below this → reduce reflex gain

# Per-virtue weights (equal in spec, but each virtue modulates a different axis)
WEIGHTS: dict[str, float] = {
    "patience":       0.20,
    "humility":       0.20,
    "honesty":        0.20,
    "kindness":       0.20,
    "responsibility": 0.20,
}


# ---------------------------------------------------------------------------
# Virtue activation rules
# ---------------------------------------------------------------------------

def _score_patience(
    context: ConversationContext,
    ripple: RippleState,
) -> float:
    """
    Patience activates when:
    - User is tired / frustrated (needs time to be heard)
    - Conversation has been going rapidly (high pace)
    - Ripple is cooling (tension present)
    Score 0–1.
    """
    score = 0.50
    if context.emotion_history and context.emotion_history[-1] in (
        EmotionClass.TIRED, EmotionClass.FRUSTRATED, EmotionClass.SAD
    ):
        score += 0.30
    if ripple.trend in (RippleTrend.COOLING, RippleTrend.FADING):
        score += 0.20
    return min(1.0, score)


def _score_humility(
    context: ConversationContext,
    blended: BlendedSignal,
) -> float:
    """
    Humility activates when:
    - System's blended intensity is very high (risk of being overbearing)
    - User's emotion is calm or sad (high intensity from system = mismatch)
    """
    score = 0.50
    if blended.intensity > 0.75:
        score += 0.25
    if context.emotion_history and context.emotion_history[-1] in (
        EmotionClass.CALM, EmotionClass.SAD, EmotionClass.TIRED
    ):
        score += 0.20
    return min(1.0, score)


def _score_honesty(
    context: ConversationContext,
    conscience: ConscienceOutput,
) -> float:
    """
    Honesty activates when:
    - Empathy score is suspiciously high (may be mirroring without genuine alignment)
    - System is not in a 'allowed full reflex' state
    - Turn count is low (easy to over-promise rapport early)
    Honesty prevents false empathy.
    """
    score = 0.50
    if conscience.empathy_score > 0.90:
        score += 0.15   # suspiciously high → might be hollow mirroring
    if not conscience.allowed:
        score += 0.20   # blocked by ethics → be transparent about that
    if context.turns < 5:
        score += 0.15   # early turns → don't over-claim rapport
    return min(1.0, score)


def _score_kindness(
    context: ConversationContext,
    ripple: RippleState,
) -> float:
    """
    Kindness activates when:
    - Ripple is warming or neutral (user in positive space)
    - User is happy or grateful
    Kindness biases Echo decay toward warmth (holds warmth longer).
    """
    score = 0.50
    if ripple.trend in (RippleTrend.WARMING, RippleTrend.RISING):
        score += 0.25
    if context.emotion_history and context.emotion_history[-1] in (
        EmotionClass.HAPPY, EmotionClass.GRATEFUL, EmotionClass.FUNNY
    ):
        score += 0.20
    return min(1.0, score)


def _score_responsibility(
    context: ConversationContext,
    ripple: RippleState,
    conscience: ConscienceOutput,
) -> float:
    """
    Responsibility activates when:
    - Ripple is cooling or there is a humor mask (hidden distress)
    - Risk score is non-trivial
    - Conversation is long (more emotional history → more accountability)
    Responsibility monitors Ripple for harm.
    """
    score = 0.50
    if ripple.trend in (RippleTrend.COOLING, RippleTrend.FADING):
        score += 0.20
    if ripple.humor_mask_active:
        score += 0.25   # hidden pain beneath humor → take responsibility
    if conscience.risk_score > 0.20:
        score += 0.15
    if context.turns > 30:
        score += 0.10
    return min(1.0, score)


# ---------------------------------------------------------------------------
# Tone modifier helpers
# ---------------------------------------------------------------------------

def _build_tone_modifiers(
    active_virtues: list[str],
    scores: VirtueScores,
) -> dict[str, float]:
    """
    Build a dict of tone modifier deltas to apply downstream.
    Example: {'warmth': +0.12, 'pace': -0.08, 'pause_ms': +120}
    """
    mods: dict[str, float] = {}

    if "patience" in active_virtues:
        mods["pace"]     = mods.get("pace", 0.0) - 0.10
        mods["pause_ms"] = mods.get("pause_ms", 0.0) + 100.0 * scores.patience

    if "humility" in active_virtues:
        mods["amplitude"] = mods.get("amplitude", 0.0) - 0.08 * scores.humility

    if "honesty" in active_virtues:
        mods["warmth"]    = mods.get("warmth", 0.0) - 0.05 * scores.honesty

    if "kindness" in active_virtues:
        mods["warmth"]    = mods.get("warmth", 0.0) + 0.12 * scores.kindness
        mods["echo_warmth_bias"] = 0.10 * scores.kindness

    if "responsibility" in active_virtues:
        mods["pace"]     = mods.get("pace", 0.0) - 0.05 * scores.responsibility
        mods["amplitude"] = mods.get("amplitude", 0.0) - 0.05 * scores.responsibility

    return mods


def _dominant_tone(active_virtues: list[str], scores: VirtueScores) -> str:
    """Return the dominant output tone label for this turn."""
    if not active_virtues:
        return "neutral"
    tone_map = {
        "patience":       "calm",
        "humility":       "gentle",
        "honesty":        "transparent",
        "kindness":       "comforting",
        "responsibility": "corrective",
    }
    # Highest-scoring active virtue wins
    best = max(active_virtues, key=lambda v: getattr(scores, v, 0.0))
    return tone_map.get(best, "neutral")


# ---------------------------------------------------------------------------
# VirtueInterface  (Phase 12 entry point)
# ---------------------------------------------------------------------------

class VirtueInterface:
    """
    Phase 12 — Virtue Interface

    Scores all five virtues for the current turn, computes V = Σ(wi × Outcomei),
    determines which virtues are active, and produces a ``VirtueOutput``
    that modulates downstream tone and reflex gain.
    """

    THRESHOLD: float = VIRTUE_THRESHOLD

    def __init__(self) -> None:
        self._ema_score:       float = 0.50   # running EMA of V
        self._trust_index:     float = 0.70   # rises with sustained high V
        self._conflict_delta:  float = 0.00   # cumulative tone-misfire reduction

    # --- public API -------------------------------------------------------

    def apply(
        self,
        blended: BlendedSignal,
        context: ConversationContext,
        ripple: RippleState,
        conscience: ConscienceOutput,
    ) -> VirtueOutput:
        """
        Main entry point.  Evaluates all virtues and returns VirtueOutput.
        """
        scores = VirtueScores(
            patience=       _score_patience(context, ripple),
            humility=       _score_humility(context, blended),
            honesty=        _score_honesty(context, conscience),
            kindness=       _score_kindness(context, ripple),
            responsibility= _score_responsibility(context, ripple, conscience),
        )

        # V = Σ(wi × Outcomei)
        v = (
            WEIGHTS["patience"]       * scores.patience
            + WEIGHTS["humility"]     * scores.humility
            + WEIGHTS["honesty"]      * scores.honesty
            + WEIGHTS["kindness"]     * scores.kindness
            + WEIGHTS["responsibility"] * scores.responsibility
        )

        self._ema_score = 0.85 * self._ema_score + 0.15 * v

        # Trust index grows when V is consistently high
        if v > 0.70:
            self._trust_index = min(0.93, self._trust_index + 0.005)
        elif v < 0.40:
            self._trust_index = max(0.50, self._trust_index - 0.010)

        # Determine active virtues: those scoring above 0.60
        active = [name for name, score in [
            ("patience",       scores.patience),
            ("humility",       scores.humility),
            ("honesty",        scores.honesty),
            ("kindness",       scores.kindness),
            ("responsibility", scores.responsibility),
        ] if score >= 0.60]

        reduce = v < self.THRESHOLD
        mods   = _build_tone_modifiers(active, scores)
        tone   = _dominant_tone(active, scores)

        return VirtueOutput(
            score=round(v, 4),
            scores=scores,
            active_virtues=active,
            reduce_reflex_gain=reduce,
            tone_modifiers=mods,
            output_tone=tone,
        )

    def compute_score(self, outcomes: list[float]) -> float:
        """
        Public helper: V = Σ(wi × Outcomei)
        Assumes outcomes are ordered [patience, humility, honesty, kindness, responsibility].
        """
        names  = ["patience", "humility", "honesty", "kindness", "responsibility"]
        length = min(len(outcomes), len(names))
        return sum(WEIGHTS[names[i]] * outcomes[i] for i in range(length))

    @property
    def trust_index(self) -> float:
        return self._trust_index

    @property
    def ema_score(self) -> float:
        return self._ema_score

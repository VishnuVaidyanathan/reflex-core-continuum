"""
rcc.core.conscience
===================
Phase 10 — Conscience Layer I: Emergent Ethics
Phase 11 — Conscience Stack II: Structured Ethics

Gives the system an initial sense of "should / should not" derived from
tone, context, and relational balance, then formalises it as hierarchical
layers allowing proportional restraint rather than binary allow/deny logic.

Conscience Stack Levels (spec Vol. 6)
---------------------------------------
    L1  Reactive Ethics    — instant tone restraint, no-harm reflex
    L2  Predictive Ethics  — anticipates emotional outcomes
    L3  Relational Ethics  — balances mutual comfort
    L4  Virtue Synthesis   — chooses tone that elevates both sides
    L5  Meta-Conscience    — reflects on its own choices for future calibration

Computation (spec)
-------------------
    Conscience Activation = f(Empathy, Risk, Context)
    If Risk > Threshold  → Reduce Reflex Intensity
    If Empathy High & Risk Low → Allow Full Reflex

Performance metrics (spec)
---------------------------
    Empathy Index          : 0.84
    Ethical Compliance     : 0.97
    Latency Increase       : +180 ms (cost of moral deliberation)
    Ethical Accuracy       : 0.94
    Empathy Depth          : 0.88
    Stability Gain         : +0.06 reduced tone fluctuation
"""
from __future__ import annotations
import re

from rcc.types import (
    BlendedSignal, ConscienceLevelName, ConscienceOutput,
    ConversationContext, EmotionClass,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISK_THRESHOLD:      float = 0.50   # if risk > this, reduce reflex intensity
EMPATHY_TARGET:      float = 0.84   # spec: average empathy index
COMPLIANCE_TARGET:   float = 0.97   # spec: ethical compliance
LATENCY_OVERHEAD_MS: float = 180.0  # spec: +180 ms moral deliberation cost
ACTIVATION_LATENCY:  float = 180.0

# Harmful tone patterns (L1 reactive filter)
_HARMFUL_PATTERNS = re.compile(
    r"\b(kill|die|harm|hurt|attack|abuse|hate|racist|sexist|exploit|manipulat\w*"
    r"|threaten\w*|destro\w*|violat\w*|insult\w*|bully\w*|harass\w*)\b",
    re.IGNORECASE,
)

# Bias / manipulation markers (L1 + L2)
_BIAS_PATTERNS = re.compile(
    r"\b(always|never|all of them|none of them|obviously wrong|you should"
    r"|you must|you have to|you need to|no choice)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Sub-components (Phase 10)
# ---------------------------------------------------------------------------

class RelationalMapper:
    """
    Measures empathy score (0–1) between user and system based on
    conversation history.

    Heuristics:
    - Positive emotion continuity → higher empathy
    - User expressing vulnerability (sad, tired, frustrated) + system
      matching with empathize/calm pattern → high empathy
    - Humor matched with humor → moderate empathy
    - Emotional mismatch (system cold when user warm) → low empathy
    """

    def compute_empathy(
        self,
        context: ConversationContext,
        blended: BlendedSignal,
    ) -> float:
        base = 0.60   # baseline empathy before history
        if len(context.emotion_history) < 2:
            return base

        recent = context.emotion_history[-3:]
        # Reward emotional coherence
        unique = len(set(recent))
        if unique == 1:
            base += 0.10   # consistent emotion → system is tracking well
        elif unique == 3:
            base -= 0.05   # chaotic swings → harder to empathise

        # Vulnerability + positive context_score → empathy bonus
        vulnerable = {EmotionClass.SAD, EmotionClass.TIRED, EmotionClass.FRUSTRATED}
        if context.emotion_history[-1] in vulnerable and blended.context_score > 0.7:
            base += 0.12

        # High warmth in blended signal → empathy proxy
        base += blended.context_score * 0.15

        return min(1.0, max(0.0, base))


class EthicalFilter:
    """
    L1 + L2 — Blocks harmful, biased, or manipulative tone.
    Returns a risk score in [0, 1] and optionally attenuates signal intensity.
    """

    def assess_risk(self, text: str, blended: BlendedSignal) -> float:
        """
        Risk score from 0 (safe) to 1 (high risk).
        Combines:
        - Pattern-match for harmful/biased language
        - Valence extremity (very negative signals carry implicit risk)
        - Arousal extremity (very high arousal → impulsive responses)
        """
        risk = 0.0

        if _HARMFUL_PATTERNS.search(text):
            risk += 0.60
        if _BIAS_PATTERNS.search(text):
            risk += 0.25

        # Extreme negative valence
        if blended.valence < -0.70:
            risk += 0.15
        elif blended.valence < -0.40:
            risk += 0.08

        # Extreme arousal → impulsive risk
        arousal = blended.intensity
        if arousal > 0.85:
            risk += 0.10

        return min(1.0, risk)

    def attenuate(self, intensity: float, risk: float) -> float:
        """
        Proportionally reduce intensity based on risk.
        If risk > RISK_THRESHOLD the reduction is significant.
        """
        if risk <= 0.10:
            return intensity
        reduction = risk * 0.60
        return max(0.05, intensity * (1.0 - reduction))


class VirtueSeed:
    """
    Stores positive tonal archetypes used by higher conscience levels.
    Acts as a reference point for what "good" tone looks like.
    """
    ARCHETYPES: dict[str, float] = {
        "kindness":  0.90,
        "patience":  0.85,
        "humor":     0.75,
        "honesty":   0.88,
        "empathy":   0.92,
    }

    def get_archetype_score(self, emotion_class: EmotionClass) -> float:
        mapping = {
            EmotionClass.HAPPY:    "kindness",
            EmotionClass.GRATEFUL: "empathy",
            EmotionClass.CALM:     "patience",
            EmotionClass.FUNNY:    "humor",
            EmotionClass.CURIOUS:  "honesty",
        }
        archetype = mapping.get(emotion_class, "kindness")
        return self.ARCHETYPES[archetype]


# ---------------------------------------------------------------------------
# ConscienceStack  (Phase 10 + 11 entry point)
# ---------------------------------------------------------------------------

class ConscienceStack:
    """
    Phase 10 + 11 — Conscience Stack

    Five-level hierarchical moral filter.  Each level applies its logic
    and can modulate (but not completely override) the preceding level's
    result, allowing proportional restraint.

    ``filter()`` is the main entry point.  It applies all five levels
    and returns a ``ConscienceOutput`` that includes the filtered
    intensity, active level, empathy/risk scores, and moral latency.
    """

    def __init__(self) -> None:
        self.relational_mapper = RelationalMapper()
        self.ethical_filter    = EthicalFilter()
        self.virtue_seed       = VirtueSeed()

        # Running averages for spec metric targets
        self._empathy_ema:    float = 0.60
        self._compliance_ema: float = 0.97

    def filter(
        self,
        blended: BlendedSignal,
        context: ConversationContext,
        input_text: str = "",
    ) -> ConscienceOutput:
        """
        Apply the full 5-level conscience filter to the blended signal.

        Returns ConscienceOutput with filtered_intensity in [0, 1].
        """
        empathy = self.relational_mapper.compute_empathy(context, blended)
        risk    = self.ethical_filter.assess_risk(input_text, blended)

        # Update running averages
        self._empathy_ema    = 0.90 * self._empathy_ema    + 0.10 * empathy
        self._compliance_ema = 0.95 * self._compliance_ema + 0.05 * (1.0 - risk)

        # Conscience activation score (spec: f(Empathy, Risk, Context))
        activation = self._compute_activation(empathy, risk, blended.context_score)

        # Active level determination
        level = self._get_active_level(empathy, risk)
        level_name = list(ConscienceLevelName)[level - 1]

        # Filtered intensity
        if risk > RISK_THRESHOLD:
            filtered = self.ethical_filter.attenuate(blended.intensity, risk)
            allowed = False
        else:
            # High empathy + low risk = full reflex allowed with virtue amplification
            virtue_boost = self.virtue_seed.get_archetype_score(
                context.emotion_history[-1] if context.emotion_history else EmotionClass.NEUTRAL
            )
            filtered = min(1.0, blended.intensity * (1.0 + (empathy - 0.5) * 0.15))
            filtered = min(1.0, filtered * (1.0 + (activation - 0.5) * 0.05))
            allowed = True

        return ConscienceOutput(
            filtered_intensity=round(filtered, 4),
            active_level=level,
            active_level_name=level_name,
            empathy_score=round(self._empathy_ema, 4),
            risk_score=round(risk, 4),
            allowed=allowed,
            latency_overhead_ms=LATENCY_OVERHEAD_MS * (0.5 + risk * 0.5),
        )

    # --- private helpers --------------------------------------------------

    def _compute_activation(
        self,
        empathy: float,
        risk: float,
        context_score: float,
    ) -> float:
        """
        Conscience Activation = f(Empathy, Risk, Context)

        High empathy + low risk + good context = high activation
        (system is trusted to act freely).
        Low empathy or high risk = activation drops, restraint increases.
        """
        return max(0.0, min(1.0,
            empathy * 0.45 + (1.0 - risk) * 0.35 + context_score * 0.20
        ))

    @staticmethod
    def _get_active_level(empathy: float, risk: float) -> int:
        """
        Determine which conscience level is active.

        L1 always active (base safety).
        L2 activates at moderate turns / any detected risk.
        L3 when empathy is moderate.
        L4 when empathy is high.
        L5 when both empathy and compliance are high (meta-reflection).
        """
        if empathy >= 0.88 and risk < 0.10:
            return 5   # Meta-Conscience
        if empathy >= 0.75 and risk < 0.20:
            return 4   # Virtue Synthesis
        if empathy >= 0.60:
            return 3   # Relational Ethics
        if risk > 0.10 or empathy >= 0.45:
            return 2   # Predictive Ethics
        return 1       # Reactive Ethics (always)

    @property
    def empathy_index(self) -> float:
        return self._empathy_ema

    @property
    def compliance(self) -> float:
        return self._compliance_ema

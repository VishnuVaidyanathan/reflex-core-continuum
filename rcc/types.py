"""
rcc.types
=========
All shared dataclasses, enums, and type aliases for the Reflex Core Continuum.
Every layer imports from here — nothing in types.py imports from elsewhere in the package.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EmotionClass(str, Enum):
    TIRED      = "tired"
    FUNNY      = "funny"
    FRUSTRATED = "frustrated"
    HAPPY      = "happy"
    SAD        = "sad"
    CALM       = "calm"
    CURIOUS    = "curious"
    GRATEFUL   = "grateful"
    NEUTRAL    = "neutral"


class RippleTrend(str, Enum):
    WARMING  = "warming"   # strong positive direction
    RISING   = "rising"    # mild positive direction
    NEUTRAL  = "neutral"   # no significant change
    FADING   = "fading"    # mild negative direction
    COOLING  = "cooling"   # strong negative direction


class ConscienceLevelName(str, Enum):
    REACTIVE   = "Reactive Ethics"
    PREDICTIVE = "Predictive Ethics"
    RELATIONAL = "Relational Ethics"
    VIRTUE     = "Virtue Synthesis"
    META       = "Meta-Conscience"


class SystemPhase(str, Enum):
    REFLEX          = "Reflex Layer"
    ECHO_RIPPLE     = "Echo-Ripple"
    ADAPTIVE_FILTER = "Adaptive Filter"
    HYBRID          = "Hybrid Dominance"
    EMERGENT_VOICE  = "Emergent Voice"
    CONSCIENCE      = "Conscience Stack"
    VIRTUE          = "Virtue Interface"
    EQUILIBRIUM     = "Equilibrium Field"


# ---------------------------------------------------------------------------
# Impulse / Reflex
# ---------------------------------------------------------------------------

@dataclass
class EmotionalVector:
    """2-D emotional space: valence × arousal, plus intensity and class."""
    valence: float          # -1.0 (very negative) … +1.0 (very positive)
    arousal: float          # 0.0 (calm/passive)  … +1.0 (excited/active)
    intensity: float        # 0.0 … 1.0  — composite strength of the signal
    emotion_class: EmotionClass = EmotionClass.NEUTRAL
    confidence: float = 1.0  # how certain the detector is (0–1)


@dataclass
class ToneMarkers:
    """Per-signal tone annotations used by downstream layers."""
    humor:     float = 0.0   # 0–1
    urgency:   float = 0.0
    fatigue:   float = 0.0
    aggression: float = 0.0
    warmth:    float = 0.0
    curiosity: float = 0.0
    sadness:   float = 0.0
    gratitude: float = 0.0
    negated:   bool  = False  # "not happy", "not bad", etc.


@dataclass
class ImpulseSignal:
    """Output of the ImpulseDetector — the raw emotional read of user text."""
    text: str
    emotional_vector: EmotionalVector
    tone_markers: ToneMarkers
    pace_indicator: float   # message_len / avg_len — >1 means rushed, <1 means slow
    timestamp: float        # monotonic seconds since session start
    turn: int = 0


@dataclass
class ReflexResponsePattern:
    """A single entry in the Reflex Response Cache."""
    pattern_id: str
    emotion_class: EmotionClass
    response_type: str       # 'humor', 'calm', 'deflect', 'empathize', 'mirror', 'contrast'
    tone_shift: float        # delta applied to current valence when this pattern fires
    base_latency_ms: float   # the timing fingerprint of this pattern
    weight: float = 1.0      # relevance weight (updated by feedback)


@dataclass
class ReflexSignal:
    """Output of the ReflexLayer — instantaneous micro-response."""
    pattern: ReflexResponsePattern
    latency_ms: float
    intensity: float          # 0–1 scaled from impulse intensity
    impulse: ImpulseSignal
    cache_hit: bool = False   # True if served from cache


# ---------------------------------------------------------------------------
# Echo & Ripple
# ---------------------------------------------------------------------------

@dataclass
class EchoState:
    """Short-term emotional residue held by the Echo layer."""
    intensity: float          # 0–1, decays each tick: E(t+1) = E(t)*(1-λ)
    valence: float            # carried valence from last signal
    decay_lambda: float       # asymmetric: positive=0.40, negative=0.55
    residue_age_s: float = 0.0  # seconds since last update
    still_eligible: bool = False  # True when intensity < STILL_THRESHOLD


@dataclass
class RippleState:
    """Directional emotional momentum tracked by the Ripple layer."""
    direction: float          # positive=warming, negative=cooling
    magnitude: float          # |direction|
    trend: RippleTrend = RippleTrend.NEUTRAL
    momentum: float = 0.0     # second-order rate of change
    humor_mask_active: bool = False  # serious tone suspended beneath humor
    turns_on_trend: int = 0   # turns spent in current trend (for contrast logic)


# ---------------------------------------------------------------------------
# Adaptive
# ---------------------------------------------------------------------------

@dataclass
class ConversationContext:
    """Cross-turn memory that flows through every layer."""
    session_id: str
    turns: int = 0
    message_lengths: list[int] = field(default_factory=list)
    emotion_history: list[EmotionClass] = field(default_factory=list)
    ripple_history: list[float] = field(default_factory=list)
    feedback_scores: list[float] = field(default_factory=list)  # 0=rejected, 1=accepted
    avg_message_len: float = 50.0
    learned_rhythm: float = 1.0   # multiplier applied to latency after learning kicks in


@dataclass
class AdaptiveOutput:
    """Output of the AdaptiveLayer — reflex-filtered through learned context."""
    adjusted_intensity: float     # tone-normalized intensity
    adjusted_latency_ms: float    # latency after balancing
    context_score: float          # how well reflex matched user's expected tone (0–1)
    learning_active: bool = False  # True once >= 20 turns have elapsed
    tone_range: tuple[float, float] = (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Hybrid Dominance
# ---------------------------------------------------------------------------

@dataclass
class HybridWeights:
    """Sigmoid-derived blending weights for Phases 3–4."""
    reflex_weight: float     # 1 - adaptive_weight
    adaptive_weight: float   # sigmoid(t; t0=20, k=0.2)
    turn: int

    @property
    def stage(self) -> str:
        if self.adaptive_weight < 0.25:
            return "early"    # turns 1–10
        elif self.adaptive_weight < 0.65:
            return "mid"      # turns 10–25
        else:
            return "late"     # turns 25+


@dataclass
class BlendedSignal:
    """Reflex + Adaptive blend after hybrid dominance controller."""
    intensity: float
    latency_ms: float
    valence: float
    weights: HybridWeights
    context_score: float


# ---------------------------------------------------------------------------
# Voice & Harmony
# ---------------------------------------------------------------------------

@dataclass
class ToneProfile:
    """Output of the ToneRegulator."""
    amplitude: float          # regulated emotional amplitude 0–1
    warmth: float             # 0 (cold) … 1 (warm)
    pace: float               # 0 (slow/deliberate) … 1 (quick)
    pause_hint_ms: float      # suggested pre-response pause


@dataclass
class VoiceOutput:
    """Output of the VoiceComposer — shapes how responses feel."""
    tone_profile: ToneProfile
    harmony_index: float       # H ∈ [0,1]; ≈ 0.88 after ~10 turns
    rhythm_ms: float           # final timing for this turn
    warmth: float


# ---------------------------------------------------------------------------
# Consolidation (CPF)
# ---------------------------------------------------------------------------

@dataclass
class CPFVector:
    """A single accumulated emotional-tone vector in the Consensus Persistence Field."""
    vector: list[float]        # [valence, arousal, intensity, ripple_direction]
    weight: float              # accumulation weight
    turn_created: int
    last_updated: int


@dataclass
class MemorySnapshot:
    """What the ConsolidationEngine returns when queried."""
    global_mood_vector: list[float]   # weighted mean across all CPF vectors
    retention_rate: float             # 0–1, ≈ 0.94 over 300 turns
    vector_count: int
    dominant_valence: float           # sign of global mood
    dominant_emotion: EmotionClass


# ---------------------------------------------------------------------------
# Awareness (CAN)
# ---------------------------------------------------------------------------

@dataclass
class AwarenessNode:
    """One interaction stream registered with the CAN."""
    node_id: str
    tone_signature: float      # -1 cold … +1 warm
    resonance_coefficient: float = 1.0
    turn_registered: int = 0
    last_updated: int = 0
    active: bool = True


@dataclass
class NetworkOutput:
    """Output of ReflexiveNetworkBehavior.process_network()."""
    damped_signal: ReflexSignal
    resonance_index: float      # ≈ 0.88 target
    phase_rhythm_ms: float
    spike_suppression_rate: float  # fraction of chaotic spikes removed (~0.73)


# ---------------------------------------------------------------------------
# Conscience
# ---------------------------------------------------------------------------

@dataclass
class ConscienceOutput:
    """Output of the ConscienceStack filter."""
    filtered_intensity: float    # reflex intensity after ethical modulation
    active_level: int            # 1–5
    active_level_name: ConscienceLevelName
    empathy_score: float         # 0–1 (target ≈ 0.84)
    risk_score: float            # 0–1
    allowed: bool                # True if reflex can proceed at full strength
    latency_overhead_ms: float   # moral deliberation cost (≈ +180 ms)


# ---------------------------------------------------------------------------
# Virtue
# ---------------------------------------------------------------------------

@dataclass
class VirtueScores:
    """Individual virtue component scores."""
    patience: float      = 0.0
    humility: float      = 0.0
    honesty: float       = 0.0
    kindness: float      = 0.0
    responsibility: float = 0.0

    def weighted_sum(self) -> float:
        return (self.patience + self.humility + self.honesty
                + self.kindness + self.responsibility) / 5.0


@dataclass
class VirtueOutput:
    """Output of the VirtueInterface."""
    score: float                  # V = Σ(wi * Outcomei)
    scores: VirtueScores
    active_virtues: list[str]     # virtue names that fired this turn
    reduce_reflex_gain: bool      # True if V < threshold
    tone_modifiers: dict[str, float]  # e.g. {'warmth': +0.1, 'pace': -0.2}
    output_tone: str              # dominant tone label


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------

@dataclass
class EquilibriumState:
    """Running state of the Equilibrium Controller + Stability Field."""
    tone_potential: float       # Tp ∈ [-1, +1]
    balance_index: float        # Bi — 0 = neutral
    drift_rate: float           # Dr ≤ 0.15 per turn
    recovery_constant: float    # Rc ≈ 0.85
    emotional_amplitude: float  # A0 current
    new_state: float            # output of control equation


# ---------------------------------------------------------------------------
# Still Layer
# ---------------------------------------------------------------------------

@dataclass
class StillState:
    """State of the Still Layer."""
    active: bool = False
    activation_turn: int = 0
    activation_time_s: float = 0.0
    echo_at_activation: float = 0.0
    virtue_anchor: float = 0.0   # virtue score frozen at activation


# ---------------------------------------------------------------------------
# Full Continuum
# ---------------------------------------------------------------------------

@dataclass
class ContinuumMetrics:
    """All observable metrics at a given turn — what the UI renders."""
    turn: int
    echo: float
    ripple_direction: float
    ripple_trend: RippleTrend
    state_vector: float           # cumulative valence×arousal product
    reflex_weight: float
    adaptive_weight: float
    harmony_index: float
    virtue_score: float
    conscience_level: int
    conscience_level_name: ConscienceLevelName
    tone_potential: float
    balance_index: float
    still_active: bool
    system_phase: SystemPhase
    latency_ms: float


@dataclass
class ContinuumOutput:
    """Complete output of one RCC pipeline pass."""
    metrics: ContinuumMetrics
    voice: VoiceOutput
    conscience: ConscienceOutput
    virtue: VirtueOutput
    equilibrium: EquilibriumState
    still: StillState
    memory: Optional[MemorySnapshot] = None
    network: Optional[NetworkOutput] = None

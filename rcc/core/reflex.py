"""
rcc.core.reflex
===============
Phase 1 — Reflex Layer

Provides instantaneous micro-responses before logic engages.

Pipeline:
    text → ImpulseDetector → ReflexResponseCache → ReflexGenerator → ReflexSignal

Key spec parameters
-------------------
Latency window : 15–400 ms  (mimics human micro-timing)
Cache size     : 300–600 patterns
"""
from __future__ import annotations
import math
import re
import time
from typing import Optional

from rcc.types import (
    EmotionClass, EmotionalVector, ToneMarkers,
    ImpulseSignal, ReflexResponsePattern, ReflexSignal,
)


# ---------------------------------------------------------------------------
# Emotion knowledge base — (valence, arousal, base_intensity)
# ---------------------------------------------------------------------------

_EMOTION_COORDS: dict[EmotionClass, tuple[float, float]] = {
    EmotionClass.TIRED:      (-0.30, 0.10),
    EmotionClass.FUNNY:      ( 0.80, 0.70),
    EmotionClass.FRUSTRATED: (-0.70, 0.80),
    EmotionClass.HAPPY:      ( 0.80, 0.60),
    EmotionClass.SAD:        (-0.60, 0.20),
    EmotionClass.CALM:       ( 0.20, 0.10),
    EmotionClass.CURIOUS:    ( 0.30, 0.50),
    EmotionClass.GRATEFUL:   ( 0.60, 0.40),
    EmotionClass.NEUTRAL:    ( 0.00, 0.30),
}

# Weighted keyword lists; each entry is (word_pattern, score)
_EMOTION_KEYWORDS: dict[EmotionClass, list[tuple[str, float]]] = {
    EmotionClass.TIRED: [
        (r"\btired\b", 1.0), (r"\bexhausted\b", 1.2), (r"\bsleepy\b", 0.9),
        (r"\bfatigue\b", 1.0), (r"\bworn out\b", 1.1), (r"\bdrained\b", 1.0),
        (r"\bno energy\b", 1.0), (r"\bdead tired\b", 1.3), (r"\bgroggy\b", 0.8),
    ],
    EmotionClass.FUNNY: [
        (r"\bfunny\b", 1.0), (r"\bjoke\b", 1.0), (r"\blaugh\b", 1.0),
        (r"\bhaha\b", 1.1), (r"\blol\b", 0.9), (r"\bhumor\b", 1.0),
        (r"\bsilly\b", 0.8), (r"\bhilarious\b", 1.3), (r"\bamusing\b", 0.9),
        (r"\bwit\b", 0.8), (r"\bcomic\b", 0.9), (r"\bchuckle\b", 0.9),
    ],
    EmotionClass.FRUSTRATED: [
        (r"\bfrustr\w*\b", 1.0), (r"\bangry\b", 1.2), (r"\banger\b", 1.1),
        (r"\bannoyed\b", 0.9), (r"\bupset\b", 0.9), (r"\bmad\b", 1.0),
        (r"\birrit\w*\b", 0.9), (r"\bugh\b", 0.8), (r"\bfurious\b", 1.4),
        (r"\brage\b", 1.4), (r"\bfed up\b", 1.1), (r"\bdriving me\b", 0.8),
    ],
    EmotionClass.HAPPY: [
        (r"\bhappy\b", 1.0), (r"\bjoy\b", 1.0), (r"\bsmile\b", 0.9),
        (r"\bgreat\b", 0.8), (r"\bwonderful\b", 1.1), (r"\blove\b", 1.0),
        (r"\bexcited\b", 1.1), (r"\belated\b", 1.2), (r"\bthrill\w*\b", 1.1),
        (r"\boverjoyed\b", 1.4), (r"\bbeam\w*\b", 0.9), (r"\bcheer\w*\b", 0.9),
    ],
    EmotionClass.SAD: [
        (r"\bsad\b", 1.0), (r"\bdown\b", 0.7), (r"\bdepress\w*\b", 1.2),
        (r"\bunhappy\b", 1.0), (r"\blonely\b", 1.1), (r"\bcry\w*\b", 1.1),
        (r"\bgrief\b", 1.3), (r"\bmiss\b", 0.7), (r"\bhurt\b", 0.9),
        (r"\bheartbreak\w*\b", 1.4), (r"\bsorrow\b", 1.2), (r"\bblue\b", 0.6),
    ],
    EmotionClass.CALM: [
        (r"\bcalm\b", 1.0), (r"\bpeace\w*\b", 1.0), (r"\brelax\w*\b", 1.0),
        (r"\bquiet\b", 0.8), (r"\bstill\b", 0.7), (r"\bserene\b", 1.1),
        (r"\bbreathe\b", 0.9), (r"\btranquil\b", 1.1), (r"\bmellow\b", 0.8),
        (r"\bmeditat\w*\b", 1.0), (r"\bgentle\b", 0.8),
    ],
    EmotionClass.CURIOUS: [
        (r"\bwhat\b", 0.5), (r"\bhow\b", 0.5), (r"\bwhy\b", 0.6),
        (r"\btell me\b", 0.8), (r"\bcurious\b", 1.0), (r"\bwonder\w*\b", 0.9),
        (r"\bexplain\b", 0.8), (r"\binterest\w*\b", 0.7), (r"\blearn\b", 0.7),
        (r"\bfascinate\w*\b", 1.0), (r"\bdiscover\b", 0.9),
    ],
    EmotionClass.GRATEFUL: [
        (r"\bthank\w*\b", 1.0), (r"\bgrateful\b", 1.1), (r"\bappreciat\w*\b", 1.0),
        (r"\bglad\b", 0.8), (r"\bblessed\b", 1.1), (r"\bthankful\b", 1.1),
        (r"\bgratitude\b", 1.2), (r"\bowe you\b", 0.9),
    ],
}

# Negation window: if a negation word precedes a keyword within N tokens, flip valence
_NEGATION_PATTERNS = re.compile(
    r"\b(not|no|never|barely|hardly|don't|doesn't|didn't|won't|isn't|aren't|wasn't)\b"
)
_INTENSITY_MODIFIERS: dict[str, float] = {
    "very": 1.3, "extremely": 1.5, "really": 1.2, "so": 1.1,
    "absolutely": 1.4, "incredibly": 1.4, "pretty": 0.9, "kind of": 0.7,
    "sort of": 0.7, "a bit": 0.6, "slightly": 0.5, "little": 0.5,
}


# ---------------------------------------------------------------------------
# ImpulseDetector
# ---------------------------------------------------------------------------

class ImpulseDetector:
    """
    Phase 1 — captures tone, pace, and emotional intensity from raw text.

    Output is an ``ImpulseSignal`` carrying an ``EmotionalVector`` (valence,
    arousal, intensity) plus per-axis ``ToneMarkers``.

    Detection approach
    ------------------
    1. Apply intensity modifiers (very/extremely/slightly …)
    2. Score each emotion via weighted keyword matching
    3. Check for negation within a ±5-token window of each match
    4. Blend the top two scores to produce a continuous emotional vector
    5. Compute pace from message length vs rolling average
    """

    _NEGATION_WINDOW = 5   # tokens

    def __init__(self) -> None:
        self._avg_len: float = 50.0   # running average updated per call

    def detect(self, text: str, turn: int = 0, timestamp: float = 0.0) -> ImpulseSignal:
        tokens = text.lower().split()
        modifier_mult = self._extract_modifier(tokens)
        scores = self._score_emotions(text.lower(), tokens)
        negated = bool(_NEGATION_PATTERNS.search(text.lower()))
        vec = self._build_vector(scores, modifier_mult, negated)
        markers = self._build_tone_markers(scores, negated)
        pace = self._compute_pace(text)
        return ImpulseSignal(
            text=text,
            emotional_vector=vec,
            tone_markers=markers,
            pace_indicator=pace,
            timestamp=timestamp or time.monotonic(),
            turn=turn,
        )

    # ------------------------------------------------------------------
    def _extract_modifier(self, tokens: list[str]) -> float:
        """Scan for intensity modifiers; return the strongest found."""
        mult = 1.0
        text_joined = " ".join(tokens)
        for phrase, m in _INTENSITY_MODIFIERS.items():
            if phrase in text_joined:
                mult = max(mult, m)
        return mult

    def _score_emotions(self, text_lower: str, tokens: list[str]) -> dict[EmotionClass, float]:
        """Return raw score for each emotion class."""
        scores: dict[EmotionClass, float] = {e: 0.0 for e in EmotionClass}
        for emotion, patterns in _EMOTION_KEYWORDS.items():
            for pattern, weight in patterns:
                for m in re.finditer(pattern, text_lower):
                    neg = self._has_negation(tokens, m.start(), text_lower)
                    scores[emotion] += weight * (-0.5 if neg else 1.0)
        return scores

    def _has_negation(self, tokens: list[str], char_offset: int, text: str) -> bool:
        """Check if a negation token appears within _NEGATION_WINDOW tokens before offset."""
        prefix = text[:char_offset].split()
        window = prefix[-self._NEGATION_WINDOW:]
        return any(_NEGATION_PATTERNS.match(t) for t in window)

    def _build_vector(
        self,
        scores: dict[EmotionClass, float],
        modifier: float,
        negated: bool,
    ) -> EmotionalVector:
        total = sum(max(s, 0.0) for s in scores.values())
        if total < 0.01:
            return EmotionalVector(
                valence=0.0, arousal=0.3, intensity=0.15,
                emotion_class=EmotionClass.NEUTRAL, confidence=0.5,
            )
        # Weighted blend of all emotions above threshold
        valence = 0.0
        arousal = 0.0
        for emotion, score in scores.items():
            if score > 0:
                frac = score / total
                v, a = _EMOTION_COORDS[emotion]
                valence += frac * v
                arousal += frac * a
        # Clamp
        valence = max(-1.0, min(1.0, valence * modifier))
        arousal = max(0.0, min(1.0, arousal * modifier))
        if negated:
            valence *= -0.6   # partial flip; negation attenuates rather than fully inverses
        # Dominant emotion
        best = max(scores, key=lambda e: scores[e])
        confidence = scores[best] / total if total > 0 else 0.5
        intensity = min(1.0, (total / 3.0) * modifier)
        return EmotionalVector(
            valence=valence,
            arousal=arousal,
            intensity=intensity,
            emotion_class=best if scores[best] > 0.1 else EmotionClass.NEUTRAL,
            confidence=min(1.0, confidence),
        )

    def _build_tone_markers(
        self,
        scores: dict[EmotionClass, float],
        negated: bool,
    ) -> ToneMarkers:
        def s(e: EmotionClass) -> float:
            return max(0.0, min(1.0, scores.get(e, 0.0) / 2.0))
        return ToneMarkers(
            humor=s(EmotionClass.FUNNY),
            urgency=s(EmotionClass.FRUSTRATED) * 0.7 + s(EmotionClass.CURIOUS) * 0.3,
            fatigue=s(EmotionClass.TIRED),
            aggression=s(EmotionClass.FRUSTRATED),
            warmth=s(EmotionClass.HAPPY) * 0.5 + s(EmotionClass.GRATEFUL) * 0.5,
            curiosity=s(EmotionClass.CURIOUS),
            sadness=s(EmotionClass.SAD),
            gratitude=s(EmotionClass.GRATEFUL),
            negated=negated,
        )

    def _compute_pace(self, text: str) -> float:
        """pace_indicator = len(text) / rolling_avg_len.  >1 rushed, <1 slow."""
        l = len(text)
        if l > 0:
            self._avg_len = 0.85 * self._avg_len + 0.15 * l
        return l / max(1.0, self._avg_len)


# ---------------------------------------------------------------------------
# ReflexResponseCache
# ---------------------------------------------------------------------------

# Seed patterns: (emotion_class, response_type, tone_shift, base_latency_ms, weight)
_SEED_PATTERNS: list[tuple[EmotionClass, str, float, float, float]] = [
    # TIRED
    (EmotionClass.TIRED, "empathize",  -0.10,  80.0, 1.0),
    (EmotionClass.TIRED, "calm",       -0.05,  90.0, 1.0),
    (EmotionClass.TIRED, "mirror",     -0.08, 100.0, 0.9),
    # FUNNY
    (EmotionClass.FUNNY, "humor",      +0.40,  60.0, 1.2),
    (EmotionClass.FUNNY, "humor",      +0.45,  55.0, 1.1),
    (EmotionClass.FUNNY, "mirror",     +0.35,  65.0, 1.0),
    (EmotionClass.FUNNY, "contrast",   +0.20,  80.0, 0.8),
    # FRUSTRATED
    (EmotionClass.FRUSTRATED, "empathize", -0.20,  70.0, 1.2),
    (EmotionClass.FRUSTRATED, "deflect",   -0.10,  85.0, 0.9),
    (EmotionClass.FRUSTRATED, "calm",      -0.05, 100.0, 1.0),
    # HAPPY
    (EmotionClass.HAPPY, "mirror",     +0.35,  55.0, 1.2),
    (EmotionClass.HAPPY, "humor",      +0.30,  60.0, 1.0),
    (EmotionClass.HAPPY, "empathize",  +0.25,  70.0, 0.9),
    # SAD
    (EmotionClass.SAD, "empathize",    -0.15,  90.0, 1.2),
    (EmotionClass.SAD, "calm",         -0.05, 110.0, 1.1),
    (EmotionClass.SAD, "mirror",       -0.10,  95.0, 1.0),
    # CALM
    (EmotionClass.CALM, "calm",        +0.05, 120.0, 1.0),
    (EmotionClass.CALM, "mirror",      +0.05, 130.0, 0.9),
    # CURIOUS
    (EmotionClass.CURIOUS, "deflect",  +0.10,  75.0, 1.0),
    (EmotionClass.CURIOUS, "mirror",   +0.15,  70.0, 1.1),
    (EmotionClass.CURIOUS, "humor",    +0.20,  65.0, 0.9),
    # GRATEFUL
    (EmotionClass.GRATEFUL, "mirror",  +0.30,  65.0, 1.1),
    (EmotionClass.GRATEFUL, "empathize",+0.25, 75.0, 1.0),
    (EmotionClass.GRATEFUL, "calm",    +0.20,  80.0, 0.9),
    # NEUTRAL
    (EmotionClass.NEUTRAL, "deflect",   0.00,  75.0, 1.0),
    (EmotionClass.NEUTRAL, "curious",   0.05,  70.0, 0.9),
    (EmotionClass.NEUTRAL, "mirror",    0.00,  80.0, 0.8),
]


class ReflexResponseCache:
    """
    Stores up to ``max_size`` instinctive response patterns.

    On a cache *hit*, the matching pattern is returned and its weight is
    slightly reinforced.  On a *miss*, ``None`` is returned and the
    ReflexGenerator takes over.

    Feedback calls (``reinforce`` / ``penalise``) let the AdaptiveLayer
    nudge weights over time.
    """

    def __init__(self, max_size: int = 512) -> None:
        self._max = max_size
        self._patterns: list[ReflexResponsePattern] = []
        self._next_id = 0
        self._seed()

    # --- public ----------------------------------------------------------

    def lookup(self, impulse: ImpulseSignal) -> Optional[ReflexResponsePattern]:
        """Return highest-weight matching pattern, or None."""
        ec = impulse.emotional_vector.emotion_class
        candidates = [p for p in self._patterns if p.emotion_class == ec]
        if not candidates:
            return None
        best = max(candidates, key=lambda p: p.weight)
        best.weight = min(2.0, best.weight * 1.02)   # mild reinforcement on hit
        return best

    def store(self, pattern: ReflexResponsePattern) -> None:
        if len(self._patterns) >= self._max:
            # Evict lowest-weight entry
            self._patterns.sort(key=lambda p: p.weight)
            self._patterns.pop(0)
        self._patterns.append(pattern)

    def reinforce(self, pattern_id: str, delta: float = 0.05) -> None:
        for p in self._patterns:
            if p.pattern_id == pattern_id:
                p.weight = min(2.0, p.weight + delta)

    def penalise(self, pattern_id: str, delta: float = 0.05) -> None:
        for p in self._patterns:
            if p.pattern_id == pattern_id:
                p.weight = max(0.1, p.weight - delta)

    @property
    def size(self) -> int:
        return len(self._patterns)

    # --- private ---------------------------------------------------------

    def _seed(self) -> None:
        for ec, rtype, tshift, latency, weight in _SEED_PATTERNS:
            # Each seed expands to ~12 variants via slight latency/weight jitter
            for i in range(12):
                jitter_l = latency * (1 + (i - 6) * 0.04)
                jitter_w = weight * (1 + (i - 6) * 0.01)
                self._patterns.append(ReflexResponsePattern(
                    pattern_id=f"seed_{self._next_id:04d}",
                    emotion_class=ec,
                    response_type=rtype,
                    tone_shift=tshift,
                    base_latency_ms=jitter_l,
                    weight=jitter_w,
                ))
                self._next_id += 1
                if len(self._patterns) >= self._max:
                    return


# ---------------------------------------------------------------------------
# ReflexGenerator
# ---------------------------------------------------------------------------

class ReflexGenerator:
    """
    Synthesises a new ReflexResponsePattern when the cache returns None.
    The generated pattern is based on emotional coordinates and pace.
    """

    _RESPONSE_TYPE_MAP: dict[EmotionClass, list[str]] = {
        EmotionClass.TIRED:      ["empathize", "calm"],
        EmotionClass.FUNNY:      ["humor", "humor", "mirror"],
        EmotionClass.FRUSTRATED: ["empathize", "calm", "deflect"],
        EmotionClass.HAPPY:      ["mirror", "humor", "empathize"],
        EmotionClass.SAD:        ["empathize", "calm"],
        EmotionClass.CALM:       ["calm", "mirror"],
        EmotionClass.CURIOUS:    ["deflect", "mirror", "humor"],
        EmotionClass.GRATEFUL:   ["mirror", "empathize"],
        EmotionClass.NEUTRAL:    ["deflect", "curious"],
    }

    _next_id: int = 10_000

    def generate(self, impulse: ImpulseSignal) -> ReflexResponsePattern:
        ec = impulse.emotional_vector.emotion_class
        types = self._RESPONSE_TYPE_MAP.get(ec, ["deflect"])
        rtype = types[self.__class__._next_id % len(types)]
        v, a = _EMOTION_COORDS[ec]
        tone_shift = v * impulse.emotional_vector.intensity * 0.5
        # Latency inversely proportional to arousal (higher arousal → faster reflex)
        latency = 15 + (400 - 15) * (1.0 - impulse.emotional_vector.arousal)
        pid = f"gen_{self.__class__._next_id:06d}"
        self.__class__._next_id += 1
        return ReflexResponsePattern(
            pattern_id=pid,
            emotion_class=ec,
            response_type=rtype,
            tone_shift=tone_shift,
            base_latency_ms=latency,
            weight=1.0,
        )


# ---------------------------------------------------------------------------
# ReflexLayer  (Phase 1 entry point)
# ---------------------------------------------------------------------------

class ReflexLayer:
    """
    Phase 1 — Reflex Layer

    Processes raw text into a ``ReflexSignal`` in a single pass:
        text → ImpulseDetector → Cache lookup / Generator → ReflexSignal

    Latency window : 15–400 ms
    Cache hit path : ~15–120 ms (fast, instinctive)
    Miss (generate): ~100–400 ms (slightly slower, still pre-conscious)
    """

    LATENCY_MIN_MS: float = 15.0
    LATENCY_MAX_MS: float = 400.0

    def __init__(self) -> None:
        self.impulse_detector = ImpulseDetector()
        self.cache = ReflexResponseCache(max_size=512)
        self.generator = ReflexGenerator()

    def process(self, text: str, turn: int = 0) -> ReflexSignal:
        impulse = self.impulse_detector.detect(text, turn=turn)
        pattern = self.cache.lookup(impulse)
        cache_hit = pattern is not None
        if pattern is None:
            pattern = self.generator.generate(impulse)
            self.cache.store(pattern)
        latency = self._compute_latency(impulse.emotional_vector.intensity, cache_hit)
        intensity = self._compute_intensity(impulse)
        return ReflexSignal(
            pattern=pattern,
            latency_ms=latency,
            intensity=intensity,
            impulse=impulse,
            cache_hit=cache_hit,
        )

    # ------------------------------------------------------------------

    def _compute_latency(self, intensity: float, cache_hit: bool) -> float:
        """
        Latency is inversely scaled with intensity (stronger signal = faster reflex),
        then clamped to [LATENCY_MIN, LATENCY_MAX].
        Cache hits are 20% faster (pre-cached micro-timing).
        """
        base = self.LATENCY_MAX_MS - (self.LATENCY_MAX_MS - self.LATENCY_MIN_MS) * intensity
        return max(self.LATENCY_MIN_MS, base * (0.80 if cache_hit else 1.0))

    def _compute_intensity(self, impulse: ImpulseSignal) -> float:
        """
        Composite intensity: emotional intensity × pace modifier.
        High pace (rushed messages) amplifies intensity by up to 20%.
        """
        pace_factor = min(1.2, max(0.8, impulse.pace_indicator))
        return min(1.0, impulse.emotional_vector.intensity * pace_factor)

    def feedback(self, signal: ReflexSignal, accepted: bool) -> None:
        """Allow downstream layers to feed results back into the cache."""
        if accepted:
            self.cache.reinforce(signal.pattern.pattern_id)
        else:
            self.cache.penalise(signal.pattern.pattern_id)

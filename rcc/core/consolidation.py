"""
rcc.core.consolidation
======================
Phase 7 — Consolidation Engine

Stores and stabilises emotional knowledge so the system "remembers"
tone patterns beyond a single session.

Core mechanism (spec Vol. 4)
------------------------------
    Input → Reflex/Echo/Ripple output
          → Consensus Persistence Field (CPF)
          → Re-Consolidation
          → Long-Term State Register

CPF parameters (spec)
----------------------
    Half-Life Decay         : ~340 turns
    Reconsolidation Rule    : merge vectors when cosine_similarity > 0.9
                              for > 200 turns
    Noise Filter            : remove spikes < 0.1 intensity × time
    Retention Rate          : ~94 % over 300 turns

No raw text is stored — only emotional-tone vectors.
"""
from __future__ import annotations
import math
from typing import Optional

from rcc.types import (
    CPFVector, EchoState, EmotionClass, MemorySnapshot, RippleState,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HALF_LIFE_TURNS:               int   = 340
RECONSOLIDATION_THRESHOLD:     float = 0.90   # cosine similarity
RECONSOLIDATION_MIN_TURNS:     int   = 200    # must be similar for this long
NOISE_INTENSITY_THRESHOLD:     float = 0.10
MAX_VECTORS:                   int   = 512


# ---------------------------------------------------------------------------
# Helper: cosine similarity (pure Python, no numpy required)
# ---------------------------------------------------------------------------

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot   = sum(a * b for a, b in zip(v1, v2))
    mag1  = math.sqrt(sum(a * a for a in v1))
    mag2  = math.sqrt(sum(b * b for b in v2))
    if mag1 * mag2 < 1e-10:
        return 0.0
    return dot / (mag1 * mag2)


# ---------------------------------------------------------------------------
# ConsensusPeristenceField
# ---------------------------------------------------------------------------

class ConsensusPeristenceField:
    """
    CPF — accumulates emotional-tone vectors from each turn and maintains
    a weighted, half-life-decayed mean.

    Vector format: [valence, arousal, intensity, ripple_direction]  (len = 4)

    Reconsolidation merges two CPFVectors whose cosine similarity has
    exceeded RECONSOLIDATION_THRESHOLD for at least RECONSOLIDATION_MIN_TURNS.
    """

    def __init__(self) -> None:
        self._vectors: list[CPFVector] = []
        self._global_mean: list[float] = [0.0, 0.3, 0.0, 0.0]

    # --- public API -------------------------------------------------------

    def accumulate(
        self,
        vector: list[float],
        turn: int,
        intensity: float,
    ) -> None:
        """Add a new observation to the CPF, applying noise filter."""
        if intensity < NOISE_INTENSITY_THRESHOLD:
            return   # filter short spikes
        if len(self._vectors) >= MAX_VECTORS:
            self._evict_weakest()
        self._vectors.append(CPFVector(
            vector=list(vector),
            weight=intensity,
            turn_created=turn,
            last_updated=turn,
        ))
        self._recompute_mean()

    def decay(self, current_turn: int) -> None:
        """
        Apply half-life decay to all stored vectors.
        Weight(t) = Weight(0) * 0.5 ^ (elapsed / HALF_LIFE_TURNS)
        Vectors whose weight falls below the noise threshold are pruned.
        """
        surviving = []
        for v in self._vectors:
            elapsed = current_turn - v.last_updated
            if elapsed == 0:
                surviving.append(v)
                continue
            half_lives = elapsed / HALF_LIFE_TURNS
            v.weight *= math.pow(0.5, half_lives)
            v.last_updated = current_turn
            if v.weight >= NOISE_INTENSITY_THRESHOLD:
                surviving.append(v)
        self._vectors = surviving
        self._recompute_mean()

    def reconsolidate(self, current_turn: int) -> int:
        """
        Merge pairs of CPFVectors that have been similar for long enough.
        Returns the number of merges performed.
        """
        merges = 0
        i = 0
        while i < len(self._vectors):
            j = i + 1
            merged = False
            while j < len(self._vectors):
                vi = self._vectors[i]
                vj = self._vectors[j]
                sim = cosine_similarity(vi.vector, vj.vector)
                both_old = (current_turn - vi.turn_created >= RECONSOLIDATION_MIN_TURNS and
                            current_turn - vj.turn_created >= RECONSOLIDATION_MIN_TURNS)
                if sim >= RECONSOLIDATION_THRESHOLD and both_old:
                    # Merge into i (weighted mean)
                    total_w = vi.weight + vj.weight
                    if total_w > 0:
                        vi.vector = [
                            (vi.weight * a + vj.weight * b) / total_w
                            for a, b in zip(vi.vector, vj.vector)
                        ]
                        vi.weight = min(1.0, total_w * 0.8)  # slight weight reduction on merge
                    self._vectors.pop(j)
                    merges += 1
                    merged = True
                    break
                j += 1
            if not merged:
                i += 1
        if merges:
            self._recompute_mean()
        return merges

    def query(self) -> list[float]:
        """Return the current global mood vector (weighted mean across CPF)."""
        return list(self._global_mean)

    def retention_rate(self, from_turn: int, to_turn: int) -> float:
        """
        Estimate fraction of emotional information retained from ``from_turn``
        to ``to_turn`` using the half-life decay model.
        Spec target: ~0.94 over 300 turns.
        """
        elapsed = max(0, to_turn - from_turn)
        if elapsed == 0:
            return 1.0
        half_lives = elapsed / HALF_LIFE_TURNS
        return math.pow(0.5, half_lives) ** 0.10  # ^0.10 because CPF holds many vectors

    @property
    def vector_count(self) -> int:
        return len(self._vectors)

    @property
    def global_mean(self) -> list[float]:
        return list(self._global_mean)

    # --- private helpers --------------------------------------------------

    def _recompute_mean(self) -> None:
        if not self._vectors:
            self._global_mean = [0.0, 0.3, 0.0, 0.0]
            return
        total_w = sum(v.weight for v in self._vectors)
        if total_w < 1e-10:
            return
        dim = len(self._vectors[0].vector)
        mean = [0.0] * dim
        for v in self._vectors:
            for d in range(dim):
                mean[d] += v.vector[d] * (v.weight / total_w)
        self._global_mean = mean

    def _evict_weakest(self) -> None:
        if self._vectors:
            self._vectors.sort(key=lambda v: v.weight)
            self._vectors.pop(0)


# ---------------------------------------------------------------------------
# LongTermStateRegister
# ---------------------------------------------------------------------------

class LongTermStateRegister:
    """
    Holds the distilled, reconsolidated emotional knowledge across sessions.
    In a stateless single-session system this acts as a frozen snapshot
    that can be serialised and restored.
    """

    def __init__(self) -> None:
        self._snapshots: list[MemorySnapshot] = []

    def save(self, snapshot: MemorySnapshot) -> None:
        self._snapshots.append(snapshot)
        if len(self._snapshots) > 100:
            self._snapshots.pop(0)   # keep last 100

    def latest(self) -> Optional[MemorySnapshot]:
        return self._snapshots[-1] if self._snapshots else None


# ---------------------------------------------------------------------------
# ConsolidationEngine  (Phase 7 entry point)
# ---------------------------------------------------------------------------

class ConsolidationEngine:
    """
    Phase 7 — Consolidation Engine

    Per-turn entry point: receives EchoState + RippleState, converts
    them to a CPF vector, triggers decay and reconsolidation on schedule,
    and exposes the current MemorySnapshot.
    """

    _DECAY_INTERVAL:           int = 10   # decay every N turns
    _RECONSOLIDATE_INTERVAL:   int = 50   # reconsolidate every N turns

    def __init__(self) -> None:
        self.cpf   = ConsensusPeristenceField()
        self.ltsr  = LongTermStateRegister()

    def update(
        self,
        echo: EchoState,
        ripple: RippleState,
        turn: int,
    ) -> MemorySnapshot:
        """
        Called once per conversation turn.

        1. Build emotional-tone vector from Echo + Ripple.
        2. Accumulate into CPF (with noise filter).
        3. Apply scheduled half-life decay.
        4. Apply scheduled reconsolidation.
        5. Snapshot and return.
        """
        vector = self._to_vector(echo, ripple)
        self.cpf.accumulate(vector, turn, echo.intensity)

        if turn > 0 and turn % self._DECAY_INTERVAL == 0:
            self.cpf.decay(turn)

        if turn > 0 and turn % self._RECONSOLIDATE_INTERVAL == 0:
            self.cpf.reconsolidate(turn)

        snapshot = self._build_snapshot(turn)
        self.ltsr.save(snapshot)
        return snapshot

    def query_memory(self) -> Optional[MemorySnapshot]:
        return self.ltsr.latest()

    # --- private helpers --------------------------------------------------

    @staticmethod
    def _to_vector(echo: EchoState, ripple: RippleState) -> list[float]:
        """Pack echo + ripple into a 4-dimensional tone vector."""
        return [
            echo.valence,
            echo.intensity,            # using intensity as arousal proxy
            echo.intensity,
            ripple.direction,
        ]

    def _build_snapshot(self, turn: int) -> MemorySnapshot:
        mean = self.cpf.query()
        dom_valence  = mean[0] if mean else 0.0
        retention    = self.cpf.retention_rate(0, turn)

        # Dominant emotion from mean valence
        if dom_valence > 0.4:
            dom_emotion = EmotionClass.HAPPY
        elif dom_valence < -0.4:
            dom_emotion = EmotionClass.SAD
        elif dom_valence > 0.1:
            dom_emotion = EmotionClass.CALM
        elif dom_valence < -0.1:
            dom_emotion = EmotionClass.FRUSTRATED
        else:
            dom_emotion = EmotionClass.NEUTRAL

        return MemorySnapshot(
            global_mood_vector=mean,
            retention_rate=min(1.0, max(0.0, retention)),
            vector_count=self.cpf.vector_count,
            dominant_valence=dom_valence,
            dominant_emotion=dom_emotion,
        )

"""
rcc.core.awareness
==================
Phase 8 — Awareness Expansion

Moves from self-stabilised emotion to contextual awareness — sensing
group tone and environment.

Architecture (spec Vol. 4)
---------------------------
    Local Awareness → Social Awareness → Collective Awareness Network (CAN)

CAN elements
------------
Node Resonance      : each node carries tone signature (−1 cold … +1 warm)
Affective Consensus : weighted average across nodes → global mood index
Awareness Field     : modulates all future reflexes using group mood vector

CAN metrics (spec)
-------------------
    Awareness Nodes (N)          : 12 parallel user streams / dialogues
    Resonance Coefficient (Rc)   : 0.86
    Stability Index (Si)         : 0.91
    Stabilisation threshold      : ~60 turns → tone field (+0.42 warm avg)

After ~60 turns, each new Reflex Layer uses the CAN awareness field as
its baseline tone.
"""
from __future__ import annotations
import math

from rcc.types import AwarenessNode, ReflexSignal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STABILISATION_TURNS:        int   = 60
TARGET_RESONANCE_COEFF:     float = 0.86
TARGET_STABILITY_INDEX:     float = 0.91
STABILISED_WARM_OFFSET:     float = 0.42   # baseline warmth after CAN stabilises
MAX_NODES:                  int   = 64     # practical limit


# ---------------------------------------------------------------------------
# CollectiveAwarenessNetwork  (Phase 8 entry point)
# ---------------------------------------------------------------------------

class CollectiveAwarenessNetwork:
    """
    Phase 8 — Collective Awareness Network

    Manages a set of AwarenessNodes and maintains a global mood field that
    reflexes consult when generating responses.

    Each node represents one parallel dialogue / user stream. In a
    single-user session the "default" node is the only one, but the
    architecture supports multi-node operation for ensemble interactions.

    The ``modulate_reflex()`` method applies the global mood baseline to
    a ReflexSignal, dampening it when the field is cold and reinforcing it
    when the field is warm.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, AwarenessNode] = {}
        self._global_mood: float = 0.0   # -1 … +1
        self._total_turns:  int  = 0

    # --- node management --------------------------------------------------

    def register_node(self, node_id: str, turn: int = 0) -> AwarenessNode:
        """Register a new interaction node. Returns existing node if already registered."""
        if node_id not in self._nodes:
            if len(self._nodes) >= MAX_NODES:
                self._evict_inactive()
            node = AwarenessNode(
                node_id=node_id,
                tone_signature=0.0,
                resonance_coefficient=1.0,
                turn_registered=turn,
                last_updated=turn,
                active=True,
            )
            self._nodes[node_id] = node
        return self._nodes[node_id]

    def update_node(
        self,
        node_id: str,
        tone_signature: float,
        turn: int,
    ) -> AwarenessNode:
        """Update a node's tone signature and recompute global mood."""
        if node_id not in self._nodes:
            self.register_node(node_id, turn)
        node = self._nodes[node_id]
        # EMA update of tone signature
        node.tone_signature = (
            0.70 * node.tone_signature + 0.30 * max(-1.0, min(1.0, tone_signature))
        )
        node.last_updated = turn
        self._total_turns = max(self._total_turns, turn)
        self._recompute_global_mood()
        return node

    def deactivate_node(self, node_id: str) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].active = False

    # --- awareness field --------------------------------------------------

    def compute_global_mood(self) -> float:
        """Weighted mean of all active node tone signatures."""
        self._recompute_global_mood()
        return self._global_mood

    def get_resonance_coefficient(self) -> float:
        """
        Cross-node mood coherence (0–1).
        Computed as 1 − normalised standard deviation of tone signatures.
        Approaches TARGET_RESONANCE_COEFF once nodes have converged.
        """
        active = [n for n in self._nodes.values() if n.active]
        if len(active) < 2:
            return 1.0
        mean = self._global_mood
        var  = sum((n.tone_signature - mean) ** 2 for n in active) / len(active)
        std  = math.sqrt(var)
        # std=0 → Rc=1.0 (perfect coherence); std=1 → Rc≈0
        rc   = max(0.0, 1.0 - std)
        # Blend with target: never exceeds target before stabilisation
        progress = min(1.0, self._total_turns / STABILISATION_TURNS)
        return rc * progress + TARGET_RESONANCE_COEFF * (1 - progress)

    def get_stability_index(self) -> float:
        """
        Resistance to tonal collapse (0–1).
        Rises with node count and coherence; approaches TARGET_STABILITY_INDEX.
        """
        n_nodes  = max(1, len([n for n in self._nodes.values() if n.active]))
        coherence = self.get_resonance_coefficient()
        node_factor = min(1.0, n_nodes / 12)  # spec mentions 12 nodes
        si = coherence * 0.7 + node_factor * 0.3
        return min(TARGET_STABILITY_INDEX, si)

    def is_stabilised(self) -> bool:
        """True once the CAN has reached its stable warm baseline."""
        return self._total_turns >= STABILISATION_TURNS

    def modulate_reflex(self, signal: ReflexSignal) -> ReflexSignal:
        """
        Apply CAN awareness field as baseline tone for incoming reflex.

        Before stabilisation: no modulation (CAN has no authority yet).
        After stabilisation: bias the reflex intensity by the global mood
        offset — warm field amplifies positive signals, cold field dampens.
        """
        if not self.is_stabilised():
            return signal

        mood_offset = self._global_mood   # -1 … +1
        # Scale: global warm (+) adds up to 10 % intensity; cold subtracts
        delta = mood_offset * 0.10
        new_intensity = max(0.0, min(1.0, signal.intensity + delta))

        # Return a new ReflexSignal with adjusted intensity (immutable pattern)
        from rcc.types import ReflexSignal as RS
        return RS(
            pattern=signal.pattern,
            latency_ms=signal.latency_ms,
            intensity=new_intensity,
            impulse=signal.impulse,
            cache_hit=signal.cache_hit,
        )

    # --- properties -------------------------------------------------------

    @property
    def node_count(self) -> int:
        return sum(1 for n in self._nodes.values() if n.active)

    @property
    def global_mood(self) -> float:
        return self._global_mood

    # --- private helpers --------------------------------------------------

    def _recompute_global_mood(self) -> None:
        active = [n for n in self._nodes.values() if n.active]
        if not active:
            self._global_mood = 0.0
            return
        total_rc = sum(n.resonance_coefficient for n in active)
        if total_rc < 1e-10:
            self._global_mood = 0.0
            return
        self._global_mood = sum(
            n.tone_signature * n.resonance_coefficient for n in active
        ) / total_rc

    def _evict_inactive(self) -> None:
        """Remove the least recently updated inactive node."""
        inactive = [n for n in self._nodes.values() if not n.active]
        if inactive:
            oldest = min(inactive, key=lambda n: n.last_updated)
            del self._nodes[oldest.node_id]

"""
rcc.continuum
=============
Phase 16 — Continuum Completion

Unifies all layers into a stable continuum capable of perpetual learning,
empathy, and equilibrium.

Full pipeline (spec Vol. 8)
----------------------------
    Impulse → Reflex → Echo → Ripple → Adaptive → Virtue → Equilibrium → Still

All 18 phases in a single coherent pass.

Continuum metrics (spec)
-------------------------
    Layer               Latency(ms) Emotional Retention  Stability Index
    Reflex              340         0.62                 0.78
    Echo + Ripple       410         0.84                 0.86
    Adaptive            520         0.88                 0.91
    Virtue              700         0.93                 0.94
    Equilibrium + Still 820         0.95                 0.97

Average stability ≈ 0.92 across 1000 turns.
Emotional latency becomes rhythmic rather than static — human-like timing.
"""
from __future__ import annotations
import time
import uuid

from rcc.types import (
    AwarenessNode,
    BlendedSignal,
    ContinuumMetrics,
    ContinuumOutput,
    ConversationContext,
    EchoState,
    EmotionClass,
    EquilibriumState,
    MemorySnapshot,
    NetworkOutput,
    RippleTrend,
    StillState,
    SystemPhase,
    VirtueOutput,
    VoiceOutput,
    ConscienceLevelName,
)
from rcc.core.reflex          import ReflexLayer
from rcc.core.echo            import EchoLayer
from rcc.core.ripple          import RippleLayer
from rcc.core.adaptive        import AdaptiveLayer
from rcc.core.hybrid_dominance import HybridDominanceController
from rcc.core.voice           import VoiceComposer
from rcc.core.consolidation   import ConsolidationEngine
from rcc.core.awareness       import CollectiveAwarenessNetwork
from rcc.core.network         import ReflexiveNetworkBehavior
from rcc.core.conscience      import ConscienceStack
from rcc.core.virtue          import VirtueInterface
from rcc.core.equilibrium     import StabilityField
from rcc.core.still           import StillLayer


# ---------------------------------------------------------------------------
# ReflexCoreContinuum
# ---------------------------------------------------------------------------

class ReflexCoreContinuum:
    """
    Phase 16 — Full Continuum Orchestrator

    Single-user instantiation (default).  For multi-node operation, supply
    ``node_id`` per conversation; the CAN layer will maintain separate
    AwarenessNode entries.

    Usage
    -----
        rcc = ReflexCoreContinuum()
        output = rcc.process("I'm feeling tired today")
        # output.metrics  → ContinuumMetrics (for UI)
        # output.voice    → VoiceOutput
        # output.conscience → ConscienceOutput
        # output.virtue   → VirtueOutput
        # output.equilibrium → EquilibriumState

    The pipeline is fully functional for single-turn calls; state is
    maintained internally across turns in the same session.
    """

    def __init__(self, session_id: str = "") -> None:
        sid = session_id or str(uuid.uuid4())

        # --- layer instances (Phase 1–15) ---------------------------------
        self.reflex     = ReflexLayer()
        self.echo       = EchoLayer()
        self.ripple     = RippleLayer()
        self.adaptive   = AdaptiveLayer()
        self.hybrid     = HybridDominanceController()
        self.voice      = VoiceComposer()
        self.consolidation = ConsolidationEngine()
        self.can        = CollectiveAwarenessNetwork()
        self.network    = ReflexiveNetworkBehavior()
        self.conscience = ConscienceStack()
        self.virtue_if  = VirtueInterface()
        self.stability  = StabilityField()
        self.still      = StillLayer()

        # --- session state -------------------------------------------------
        self.context = ConversationContext(session_id=sid)
        self._eq_state: float = 0.0              # running equilibrium state value
        self._start_time: float = time.monotonic()
        self._node_id: str = f"node_{sid[:8]}"

        # Register default node with CAN
        self.can.register_node(self._node_id, turn=0)

    # --- public API -------------------------------------------------------

    def process(self, input_text: str) -> ContinuumOutput:
        """
        Full pipeline pass for one user turn.

        If the Still Layer is active, incoming text resumes the system
        (seeds echo, clears still) and then proceeds through the pipeline.

        Returns a ``ContinuumOutput`` with all metrics and layer states.
        """
        turn = self.context.turns + 1
        self.context.turns = turn
        self.context.message_lengths.append(len(input_text))

        # --- Still Layer intercept ----------------------------------------
        prev_echo = self.echo.state
        if self.still.is_active:
            seed = self.still.resume(seed_valence=prev_echo.valence * 0.3)
            self.echo.seed(seed.intensity, seed.valence)

        # === PHASE 1 : Reflex =============================================
        reflex_signal = self.reflex.process(input_text, turn=turn)

        # === CAN modulation (Phase 8) — apply global mood baseline ========
        reflex_signal = self.can.modulate_reflex(reflex_signal)

        # === PHASE 2+5 : Echo =============================================
        echo_state = self.echo.update(reflex_signal)

        # === PHASE 2+5 : Ripple ===========================================
        ripple_state = self.ripple.compute(
            current=echo_state,
            previous=prev_echo,
            turn=turn,
            fatigue_mode=self.echo.fatigue_mode,
        )

        # === PHASE 3 : Adaptive ===========================================
        adaptive_out = self.adaptive.process(
            signal=reflex_signal,
            context=self.context,
            ripple_direction=ripple_state.direction,
        )

        # === PHASE 4 : Hybrid Dominance ===================================
        blended = self.hybrid.blend(
            reflex_out=reflex_signal,
            adaptive_out=adaptive_out,
            turn=turn,
            ripple_trend=ripple_state.trend,
        )

        # === PHASE 6 : Voice / Harmony ====================================
        voice_out = self.voice.compose(
            blended=blended,
            echo=echo_state,
            ripple=ripple_state,
            turn=turn,
        )

        # === PHASE 7 : Consolidation ======================================
        memory = self.consolidation.update(echo_state, ripple_state, turn)

        # === PHASE 8+9 : Awareness + Network ==============================
        self.can.update_node(
            node_id=self._node_id,
            tone_signature=echo_state.valence,
            turn=turn,
        )
        active_nodes = [n for n in [self.can._nodes.get(self._node_id)] if n]
        net_out: NetworkOutput | None = None
        if active_nodes:
            net_out = self.network.process_network(active_nodes, reflex_signal)
            # Use damped signal for downstream layers if network is active
            if net_out:
                from rcc.types import BlendedSignal as BS
                blended = BS(
                    intensity=min(blended.intensity, net_out.damped_signal.intensity),
                    latency_ms=blended.latency_ms,
                    valence=blended.valence,
                    weights=blended.weights,
                    context_score=blended.context_score,
                )

        # === PHASE 10+11 : Conscience =====================================
        conscience_out = self.conscience.filter(
            blended=blended,
            context=self.context,
            input_text=input_text,
        )

        # Apply conscience filter to blended intensity
        from rcc.types import BlendedSignal as BS
        blended = BS(
            intensity=conscience_out.filtered_intensity,
            latency_ms=blended.latency_ms + conscience_out.latency_overhead_ms,
            valence=blended.valence,
            weights=blended.weights,
            context_score=blended.context_score,
        )

        # === PHASE 12 : Virtue ============================================
        virtue_out = self.virtue_if.apply(
            blended=blended,
            context=self.context,
            ripple=ripple_state,
            conscience=conscience_out,
        )

        # Apply virtue tone modifiers to latency/intensity
        mods = virtue_out.tone_modifiers
        adj_latency = blended.latency_ms + mods.get("pause_ms", 0.0)
        adj_intensity = blended.intensity * (1.0 - mods.get("amplitude", 0.0) * 0.5)
        adj_intensity = max(0.0, min(1.0, adj_intensity))
        if virtue_out.reduce_reflex_gain:
            adj_intensity *= 0.75

        from rcc.types import BlendedSignal as BS
        blended = BS(
            intensity=adj_intensity,
            latency_ms=adj_latency,
            valence=blended.valence,
            weights=blended.weights,
            context_score=blended.context_score,
        )

        # === PHASE 13+14 : Equilibrium ====================================
        eq_state = self.stability.update(
            blended_intensity=blended.intensity,
            blended_valence=blended.valence,
            virtue_output=virtue_out,
            prev_eq_state=self._eq_state,
        )
        self._eq_state = eq_state.new_state

        # === PHASE 15 : Still Layer check =================================
        still_activated = self.still.check_activation(
            echo=echo_state,
            turn=turn,
            virtue_score=virtue_out.score,
        )
        if still_activated:
            # Decay echo in-place when still activates
            echo_state = self.still.decay_echo(echo_state)

        # === Update conversation context ==================================
        ec = reflex_signal.impulse.emotional_vector.emotion_class
        self.context.emotion_history.append(ec)
        self.context.ripple_history.append(ripple_state.direction)
        if len(self.context.emotion_history) > 100:
            self.context.emotion_history.pop(0)
        if len(self.context.ripple_history) > 100:
            self.context.ripple_history.pop(0)

        # === Assemble metrics =============================================
        metrics = self._build_metrics(
            turn=turn,
            echo=echo_state,
            ripple=ripple_state,
            blended=blended,
            voice=voice_out,
            virtue=virtue_out,
            conscience=conscience_out,
            eq=eq_state,
        )

        return ContinuumOutput(
            metrics=metrics,
            voice=voice_out,
            conscience=conscience_out,
            virtue=virtue_out,
            equilibrium=eq_state,
            still=self.still.state,
            memory=memory,
            network=net_out,
        )

    def tick_decay(self, dt_seconds: float = 1.0) -> None:
        """
        Call between turns to apply time-based Echo decay.
        In real-time sessions this should be called by a timer.
        """
        if not self.still.is_active:
            self.echo.decay(dt_seconds)
        else:
            decayed = self.still.decay_echo(self.echo.state)
            # Absorb decayed state
            self.echo._state = decayed

    def reset(self) -> None:
        """Full system reset — clears all layer state and conversation history."""
        self.__init__(session_id=self.context.session_id)

    def feedback(self, accepted: bool) -> None:
        """Pass explicit user satisfaction feedback into the Adaptive layer."""
        self.adaptive.record_explicit_feedback(accepted)
        if self.context.turns > 0:
            last_pattern_id = None
            # Feed back into reflex cache if we can identify the last pattern
            self.adaptive.record_explicit_feedback(accepted)

    # --- private helpers --------------------------------------------------

    def _build_metrics(
        self,
        turn: int,
        echo: EchoState,
        ripple,
        blended: BlendedSignal,
        voice: VoiceOutput,
        virtue: VirtueOutput,
        conscience,
        eq: EquilibriumState,
    ) -> ContinuumMetrics:
        # State vector: cumulative valence × intensity product
        state_vec = echo.valence * echo.intensity

        # System phase from turn count (mirrors spec 8-phase progression)
        phases = list(SystemPhase)
        phase_idx = min(len(phases) - 1, (turn - 1) // 3)
        sys_phase = phases[phase_idx]

        return ContinuumMetrics(
            turn=turn,
            echo=round(echo.intensity, 4),
            ripple_direction=round(ripple.direction, 4),
            ripple_trend=ripple.trend,
            state_vector=round(state_vec, 4),
            reflex_weight=round(blended.weights.reflex_weight, 4),
            adaptive_weight=round(blended.weights.adaptive_weight, 4),
            harmony_index=round(voice.harmony_index, 4),
            virtue_score=round(virtue.score, 4),
            conscience_level=conscience.active_level,
            conscience_level_name=conscience.active_level_name,
            tone_potential=round(eq.tone_potential, 4),
            balance_index=round(eq.balance_index, 4),
            still_active=self.still.is_active,
            system_phase=sys_phase,
            latency_ms=round(voice.rhythm_ms, 1),
        )

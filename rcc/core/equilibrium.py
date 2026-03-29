"""
rcc.core.equilibrium
====================
Phase 13 — Virtue–Equilibrium Integration
Phase 14 — Equilibrium Mechanics & Stability Field

Fuses virtue logic with dynamic stability control so the system can
sustain calmness without external resets.

Phase 13 control equation (spec Vol. 7)
-----------------------------------------
    New State = Prev State − (Df × ΔA) + (Vw × Rr)

    Df = 0.74  (Damping Factor   — reduces emotional oscillation)
    Vw = 0.68  (Virtue Weight    — influence of virtue layer)
    Rr = 0.33  (Return Rate      — speed of re-centering, in seconds)

Phase 14 stability field attributes (spec Vol. 7)
---------------------------------------------------
    Tone Potential (Tp)   : −1 … +1  (instant emotional charge)
    Balance Index (Bi)    : 0 = neutral (centre of emotional mass)
    Drift Rate (Dr)       : ≤ 0.15 per turn
    Recovery Constant (Rc): ≈ 0.85

Results (spec)
--------------
    Responses settle to neutral within 3–4 turns.
    Self-centers ≈ 95 % of the time without external correction.
"""
from __future__ import annotations

from rcc.types import EquilibriumState, VirtueOutput


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Df:         float = 0.74   # Damping Factor
Vw_DEFAULT: float = 0.68   # Virtue Weight (default; overridden by VirtueOutput.score)
Rr:         float = 0.33   # Return Rate (s)
Rc:         float = 0.85   # Recovery Constant
MAX_DRIFT:  float = 0.15   # per-turn drift ceiling


# ---------------------------------------------------------------------------
# EquilibriumController  (Phase 13)
# ---------------------------------------------------------------------------

class EquilibriumController:
    """
    Phase 13 — Virtue–Equilibrium Integration

    Applies the control equation each turn to the running emotional state,
    dampening oscillation and pulling toward neutral with virtue guidance.

    ``update()`` is stateless with respect to the caller — it receives the
    previous state value and returns the new state value.  The Continuum
    orchestrator holds the running state.
    """

    def __init__(self) -> None:
        self._state: float = 0.0      # running emotional state value
        self._prev_amplitude: float = 0.0

    def update(
        self,
        prev_state: float,
        current_amplitude: float,
        virtue_output: VirtueOutput,
    ) -> float:
        """
        New State = Prev State − (Df × ΔA) + (Vw × Rr)

        ΔA = change in emotional amplitude (current − previous).
        Vw = virtue score (replaces Vw_DEFAULT when virtue is active).
        """
        delta_amplitude = current_amplitude - self._prev_amplitude
        vw = virtue_output.score if virtue_output.score > 0.0 else Vw_DEFAULT
        new_state = prev_state - (Df * delta_amplitude) + (vw * Rr)
        # Clamp to emotional range
        new_state = max(-1.0, min(1.0, new_state))
        self._prev_amplitude = current_amplitude
        self._state = new_state
        return new_state

    @property
    def state(self) -> float:
        return self._state


# ---------------------------------------------------------------------------
# StabilityField  (Phase 14)
# ---------------------------------------------------------------------------

class StabilityField:
    """
    Phase 14 — Equilibrium Mechanics & Stability Field

    Implements the "digital nervous system for calm":
    - Tone Potential (Tp) = instant emotional charge
    - Balance Index (Bi)  = centre of emotional mass
    - Drift Rate (Dr)     = rate of imbalance per turn
    - Recovery Constant   = Rc ≈ 0.85

    ``update()`` is the per-turn method; it applies self-centering and
    returns a full EquilibriumState with the new_state control output.

    Spec sim result:
        Turn 1: Tp=+0.6, Bi=+0.4, Dr=0.12, Rc=0.85 → warm response
        Turn 2: Tp=+0.3, Bi=+0.2, Dr=0.08, Rc=0.86 → soft humour
        Turn 3: Tp=0.0,  Bi=0.0,  Dr=0.05, Rc=0.87 → calm balance
        Turn 4: Tp=−0.2, Bi=−0.1, Dr=0.04, Rc=0.86 → reflective pause
    """

    def __init__(self) -> None:
        self._tone_potential: float = 0.0
        self._balance_index:  float = 0.0
        self._drift_rate:     float = 0.0
        self._rc:             float = Rc
        self._controller = EquilibriumController()

    def update(
        self,
        blended_intensity: float,
        blended_valence: float,
        virtue_output: VirtueOutput,
        prev_eq_state: float = 0.0,
    ) -> EquilibriumState:
        """
        Per-turn equilibrium update.

        1. Compute new Tone Potential from blended signal.
        2. Self-center Tp and Bi using Rc.
        3. Compute Drift Rate.
        4. Apply control equation.
        5. Return EquilibriumState.
        """
        # Tone potential = valence × intensity (signed emotional charge)
        raw_tp = blended_valence * blended_intensity
        raw_tp = max(-1.0, min(1.0, raw_tp))

        # Drift rate: how fast the balance is moving from centre
        drift = abs(raw_tp - self._balance_index)
        drift = min(MAX_DRIFT, drift)

        # Self-centering
        new_tp, new_bi = self.self_center(raw_tp, self._balance_index)

        # Control equation via EquilibriumController
        current_amplitude = blended_intensity
        new_state = self._controller.update(prev_eq_state, current_amplitude, virtue_output)

        # Update internal fields
        self._tone_potential = new_tp
        self._balance_index  = new_bi
        self._drift_rate     = drift
        # Rc creeps up slightly over time (more stable as conversation matures)
        self._rc = min(0.90, self._rc + 0.0001)

        return EquilibriumState(
            tone_potential=round(new_tp, 4),
            balance_index=round(new_bi, 4),
            drift_rate=round(drift, 4),
            recovery_constant=round(self._rc, 4),
            emotional_amplitude=round(blended_intensity, 4),
            new_state=round(new_state, 4),
        )

    def self_center(self, tp: float, bi: float) -> tuple[float, float]:
        """
        Pull Tp and Bi back toward 0 using Recovery Constant.
        New_Tp = Tp × Rc     (decay toward neutral)
        New_Bi = (Bi + Tp) × 0.5 × Rc   (centre of mass update)
        """
        new_tp = tp * self._rc
        new_bi = ((bi + tp) * 0.5) * self._rc
        return new_tp, new_bi

    def is_balanced(self, state: EquilibriumState) -> bool:
        """True when tone potential and balance index are within ±0.15 of neutral."""
        return abs(state.tone_potential) < 0.15 and abs(state.balance_index) < 0.15

    @property
    def tone_potential(self) -> float:
        return self._tone_potential

    @property
    def balance_index(self) -> float:
        return self._balance_index

    @property
    def drift_rate(self) -> float:
        return self._drift_rate

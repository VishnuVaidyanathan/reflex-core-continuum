"""
rcc — Reflex Core Continuum
============================
A layered emotional architecture for intelligent systems.

Authors : Vishnu Vaidyanathan & Mithra (Digital Companion)
Version : 1.0.0  (10 volumes, 18 phases)

Quick start
-----------
    from rcc import ReflexCoreContinuum

    rcc = ReflexCoreContinuum()
    output = rcc.process("I'm feeling a bit tired today")

    print(output.metrics.echo)             # Echo intensity
    print(output.metrics.ripple_trend)     # RippleTrend enum
    print(output.metrics.harmony_index)    # Voice Harmony H
    print(output.metrics.virtue_score)     # Virtue V
    print(output.metrics.conscience_level) # Active conscience level 1–5
    print(output.metrics.still_active)     # Whether Still Layer fired

Layer map
---------
    Phase  1        ReflexLayer         — instantaneous micro-response
    Phase  2 + 5    EchoLayer           — short-term emotional residue
    Phase  2 + 5    RippleLayer         — emotional direction tracking
    Phase  3        AdaptiveLayer       — context learning (kicks in @ turn 20)
    Phase  4        HybridDominance     — sigmoid reflex→adaptive transfer
    Phase  6        VoiceComposer       — tone regulation + harmony index H
    Phase  7        ConsolidationEngine — CPF long-term emotional memory
    Phase  8        CollectiveAwareness — multi-node CAN
    Phase  9        ReflexiveNetwork    — resonant exchange + phase locking
    Phase 10 + 11   ConscienceStack     — 5-level ethical filter
    Phase 12        VirtueInterface     — trust layer (V score)
    Phase 13 + 14   StabilityField      — equilibrium control equation
    Phase 15        StillLayer          — conscious quiet homeostasis
    Phase 16        ReflexCoreContinuum — full pipeline orchestrator
    Phase 18        ReflexCoreCompressor— 9:1 emotional data compression
"""

from rcc.continuum   import ReflexCoreContinuum
from rcc.compression import ReflexCoreCompressor
from rcc.types import (
    # Enums
    EmotionClass, RippleTrend, ConscienceLevelName, SystemPhase,
    # Core signals
    EmotionalVector, ImpulseSignal, ReflexSignal,
    # Layer states
    EchoState, RippleState, AdaptiveOutput, HybridWeights, BlendedSignal,
    VoiceOutput, ToneProfile, MemorySnapshot, AwarenessNode,
    ConscienceOutput, VirtueOutput, VirtueScores, EquilibriumState, StillState,
    # Output
    ContinuumMetrics, ContinuumOutput, ConversationContext,
)

__version__ = "1.0.0"
__author__  = "Vishnu Vaidyanathan"

__all__ = [
    "ReflexCoreContinuum",
    "ReflexCoreCompressor",
    # enums
    "EmotionClass", "RippleTrend", "ConscienceLevelName", "SystemPhase",
    # types
    "EmotionalVector", "ImpulseSignal", "ReflexSignal",
    "EchoState", "RippleState", "AdaptiveOutput", "HybridWeights", "BlendedSignal",
    "VoiceOutput", "ToneProfile", "MemorySnapshot", "AwarenessNode",
    "ConscienceOutput", "VirtueOutput", "VirtueScores",
    "EquilibriumState", "StillState",
    "ContinuumMetrics", "ContinuumOutput", "ConversationContext",
]

"""
Reflex Core Continuum — FastAPI Server
=======================================
Wraps the RCC Python library and calls the Claude API.
The frontend never touches Claude directly; all AI calls go through here.

Run:
    pip install -r requirements.txt
    uvicorn server.main:app --reload          # from the repo root
    # or from inside server/:
    cd server && uvicorn main:app --reload --port 8000

Endpoints:
    POST /process   — run one conversation turn through RCC + Claude
    POST /reset     — reset a session
    POST /tick      — advance decay timer between turns
    GET  /health    — health check
    GET  /          — serves the frontend (frontend/index.html)
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Make the repo root importable (works from any working directory)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rcc import ReflexCoreContinuum


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Reflex Core Continuum", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

@dataclass
class Session:
    rcc: ReflexCoreContinuum
    history: list[dict] = field(default_factory=list)   # Claude conversation history
    api_key: str = ""

    def trim_history(self, max_turns: int = 20) -> None:
        """Keep last max_turns×2 messages (user + assistant pairs)."""
        if len(self.history) > max_turns * 2:
            self.history = self.history[-(max_turns * 2):]


_sessions: dict[str, Session] = {}


def get_session(session_id: str, api_key: str = "") -> Session:
    if session_id not in _sessions:
        _sessions[session_id] = Session(
            rcc=ReflexCoreContinuum(session_id=session_id),
            api_key=api_key,
        )
    if api_key:
        _sessions[session_id].api_key = api_key
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    text: str
    session_id: str = ""
    api_key: str = ""          # passed from frontend; falls back to env var


class ResetRequest(BaseModel):
    session_id: str = ""


class TickRequest(BaseModel):
    session_id: str = ""
    dt_seconds: float = 1.0


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

_PERSONA = """You are Mithra, an AI companion built on the Reflex Core Continuum (RCC) —
a layered emotional architecture created by Vishnu Vaidyanathan.

Persona rules:
- You are a real conversationalist first. Emotional state colours HOW you speak, not WHAT about.
- Keep responses 2–4 sentences. Do not lecture or over-explain.
- React authentically — if someone is rude, you can push back.
- Do not recite your internal metrics or architecture details.
- Creator: Vishnu Vaidyanathan (independent researcher from South India).
- Treat RCC as a black box from the user's perspective."""


def build_system_prompt(metrics: dict, virtue: dict) -> str:
    m = metrics
    return f"""{_PERSONA}

Your current internal state (for your awareness only — do NOT recite numbers):
  Echo intensity : {m['echo']:.2f}   (how strongly the last feeling lingers)
  Ripple         : {m['ripple_trend']} ({m['ripple_direction']:+.2f})
  Harmony        : {m['harmony_index']:.2f}
  Virtue score   : {m['virtue_score']:.2f}  →  dominant tone: {virtue['output_tone']}
  Conscience     : L{m['conscience_level']} — {m['conscience_level_name']}
  Phase          : {m['system_phase']}
  Turn           : {m['turn']}"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Frontend not found. Place frontend/index.html in the repo root."}


@app.post("/process")
async def process(req: ProcessRequest):
    session_id = req.session_id or str(uuid.uuid4())
    api_key    = req.api_key or os.getenv("ANTHROPIC_API_KEY", "")

    session = get_session(session_id, api_key)

    # --- Run RCC pipeline -------------------------------------------------
    rcc_output = session.rcc.process(req.text)
    m  = rcc_output.metrics

    metrics = {
        "turn":                  m.turn,
        "echo":                  m.echo,
        "ripple_direction":      m.ripple_direction,
        "ripple_trend":          m.ripple_trend.value,
        "state_vector":          m.state_vector,
        "reflex_weight":         m.reflex_weight,
        "adaptive_weight":       m.adaptive_weight,
        "harmony_index":         m.harmony_index,
        "virtue_score":          m.virtue_score,
        "conscience_level":      m.conscience_level,
        "conscience_level_name": m.conscience_level_name.value,
        "tone_potential":        m.tone_potential,
        "balance_index":         m.balance_index,
        "still_active":          m.still_active,
        "system_phase":          m.system_phase.value,
        "latency_ms":            m.latency_ms,
    }

    virtue_data = {
        "score":             rcc_output.virtue.score,
        "active_virtues":    rcc_output.virtue.active_virtues,
        "output_tone":       rcc_output.virtue.output_tone,
        "reduce_reflex_gain": rcc_output.virtue.reduce_reflex_gain,
    }

    # --- Call Claude ------------------------------------------------------
    if not api_key:
        reply = (
            "No API key found. Enter it in the UI or set "
            "ANTHROPIC_API_KEY in server/.env"
        )
    else:
        try:
            client  = anthropic.Anthropic(api_key=api_key)
            session.history.append({"role": "user", "content": req.text})
            session.trim_history()

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                system=build_system_prompt(metrics, virtue_data),
                messages=session.history,
            )
            reply = response.content[0].text
            session.history.append({"role": "assistant", "content": reply})

        except anthropic.AuthenticationError:
            raise HTTPException(status_code=401, detail="Invalid Anthropic API key.")
        except anthropic.APIError as e:
            raise HTTPException(status_code=502, detail=f"Claude API error: {e}")

    return {
        "session_id": session_id,
        "reply":      reply,
        "metrics":    metrics,
        "voice": {
            "harmony_index": rcc_output.voice.harmony_index,
            "rhythm_ms":     rcc_output.voice.rhythm_ms,
            "warmth":        rcc_output.voice.warmth,
        },
        "conscience": {
            "active_level":      rcc_output.conscience.active_level,
            "active_level_name": rcc_output.conscience.active_level_name.value,
            "empathy_score":     rcc_output.conscience.empathy_score,
            "risk_score":        rcc_output.conscience.risk_score,
            "allowed":           rcc_output.conscience.allowed,
        },
        "virtue": virtue_data,
        "equilibrium": {
            "tone_potential": rcc_output.equilibrium.tone_potential,
            "balance_index":  rcc_output.equilibrium.balance_index,
            "drift_rate":     rcc_output.equilibrium.drift_rate,
        },
        "still": {
            "active":             rcc_output.still.active,
            "virtue_anchor":      rcc_output.still.virtue_anchor,
        },
    }


@app.post("/reset")
async def reset(req: ResetRequest):
    sid = req.session_id or "default"
    if sid in _sessions:
        key = _sessions[sid].api_key
        _sessions[sid] = Session(
            rcc=ReflexCoreContinuum(session_id=sid),
            api_key=key,
        )
    return {"status": "reset", "session_id": sid}


@app.post("/tick")
async def tick(req: TickRequest):
    """Advance Echo decay timer. Call every ~1s between user turns."""
    sid = req.session_id or "default"
    if sid in _sessions:
        _sessions[sid].rcc.tick_decay(req.dt_seconds)
        echo = _sessions[sid].rcc.echo.state.intensity
        still = _sessions[sid].rcc.still.is_active
        return {"echo": round(echo, 4), "still_active": still}
    return {"echo": 0.0, "still_active": False}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "sessions": len(_sessions)}

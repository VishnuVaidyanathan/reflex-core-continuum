Reflex Core Continuum (RCC)
A coded cognitive architecture for AI companions — where emotional continuity, conscience escalation, and behavioral emergence are computed layer by layer, not prompted.
What is RCC?
Most AI systems are stateless. Every message, they read you like a syllabus.
RCC gives an AI companion genuine session continuity through six computed layers that run silently underneath every response:
Layer
What it does
Echo
Asymmetric emotional residue — positive fades slower than negative
Ripple
Tracks reasoning quality direction across turns
Virtue
Intellectual honesty score — erodes on confabulation, builds on accuracy
Conscience Stack
5-level escalation — at L3+, silence over retaliation
Still Layer
Activates when Echo < 0.10 — suppresses generation, prefers honesty
Meta-Conscience
Pre-response reflection — pattern-matched vs genuinely supported
The result: not emotional states, but emotional coloring. The architecture knows it runs on Claude — it doesn't confuse itself about its own nature.
Live Demo — Try Mithra
https://reflex-core-continuum-production.up.railway.app
No API key needed. Try emoting her emotional intelligence 😄
Architecture
rcc/
├── echo.py              # E(t+1) = E(t)×(1−λ)^dt, λ_pos=0.40, λ_neg=0.55
├── ripple.py            # Directional delta tracking, humor-mask detection
├── conscience.py        # 5-level escalation with ethical filtering
├── still.py             # Activates when E < 0.10 after turn 3
├── virtue.py            # V = Σ(wᵢ × Outcomeᵢ) across 5 virtues
├── hybrid_dominance.py  # Sigmoid 1/(1+e^(−k(t−t₀))), t₀=20, k=0.2
├── equilibrium.py       # NewState = PrevState − (0.74×ΔA) + (0.68×Rr)
└── consolidation.py     # Cosine-similarity memory reconsolidation, half-life 340 turns
16 modules · ~2,800 lines · Zero external dependencies beyond FastAPI and Anthropic SDK
Run Locally
git clone https://github.com/VishnuVaidyanathan/reflex-core-continuum
cd reflex-core-continuum
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
uvicorn main:app --reload
Research
RCC was A/B tested against a strong baseline across 100 tasks (5 metacognitive categories):
Category
Baseline
RCC
Epistemic Humility
0.850
1.000
Strategic Withdrawal
0.850
1.000
Self Error Detection
0.950
0.950
Overall
0.884
0.933

Author
Vishnu Vaidyanathan — Independent Researcher
GitHub: @VishnuVaidyanathan

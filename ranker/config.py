"""
All tunable knobs. Score formula:
  core  = w_role*role + w_skill*skill + w_sem*semantic + w_company*company + w_edu*edu
  fit   = core * experience_mult * location_mult * consulting_penalty
  final = fit * behavioral_modifier * honeypot_multiplier
"""

# --- core additive weights (sum to 1.0) ---
W_ROLE = 0.34       # title + career trajectory: the decisive anti-stuffer signal
W_SKILL = 0.30      # retrieval/ranking skills, trust-weighted
W_SEMANTIC = 0.20   # TF-IDF intent match: catches plain-language fits
W_COMPANY = 0.10    # product vs services
W_EDUCATION = 0.06  # minor

# --- experience band (JD: 5-9 ideal, 6-8 sweet spot, "range not requirement") ---
EXP_PEAK_LOW = 6.0
EXP_PEAK_HIGH = 8.0
EXP_BAND_LOW = 5.0
EXP_BAND_HIGH = 9.0
EXP_MULT_FLOOR = 0.45   # multiplier for very junior/very senior, never zero

# --- consulting-career penalty (JD: services-only career is a disqualifier) ---
CONSULTING_PENALTY = 0.60   # applied when the ENTIRE career is consulting firms

# --- behavioral modifier envelope (signals doc: "multiplier on top") ---
BEHAVIORAL_FLOOR = 0.60     # 6-months-inactive, 5% response "not actually available"
BEHAVIORAL_CEIL = 1.15      # active, responsive, open-to-work, low notice, in demand

# --- honeypot guard ---
HONEYPOT_MULTIPLIER = 0.02  # forces impossible profiles to the bottom

# --- CV/speech dominance penalty (JD: CV/speech without NLP/IR not wanted) ---
CV_DOMINANCE_PENALTY = 0.70

# --- skill scoring shape ---
SKILL_SATURATION_K = 3.0    # ~3 strong retrieval skills ≈ near-max; more is marginal

# --- output ---
TOP_N = 100
SCORE_DECIMALS = 6

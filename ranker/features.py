"""
Per-candidate feature extraction. Each function returns a [0,1] score
(or multiplier) plus the facts that produced it, for downstream reasoning.
"""
import math
from . import taxonomies as tax
from . import config as cfg


# ---------------------------------------------------------------------------
# Role fit: title bucket, lifted by a genuinely ML/AI career trajectory.
# ---------------------------------------------------------------------------
def role_fit(candidate):
    p = candidate.get("profile", {})
    title = p.get("current_title", "")
    base = tax.title_bucket_score(title)

    history = candidate.get("career_history", []) or []
    # Fraction of career (weighted by tenure) spent in ML/AI/IR roles.
    ml_months = 0.0
    total_months = 0.0
    ml_roles = 0
    for h in history:
        dur = (h.get("duration_months", 0) or 0)
        total_months += dur
        if tax.is_ml_title(h.get("title", "")):
            ml_months += dur
            ml_roles += 1
    ml_fraction = (ml_months / total_months) if total_months > 0 else 0.0

    # Trajectory: sustained ML/AI career lifts generic titles, punishes stuffers.
    trajectory = min(1.0, 0.45 * ml_roles + 0.9 * ml_fraction)

    # Title-dominant blend; trajectory can lift a weak title or confirm a strong one.
    score = max(base, 0.55 * base + 0.55 * trajectory)
    score = min(1.0, score)

    facts = {
        "title": title,
        "ml_roles_in_history": ml_roles,
        "ml_career_fraction": round(ml_fraction, 2),
    }
    return score, facts


# ---------------------------------------------------------------------------
# Skill score: trust-weighted retrieval/ranking skills, with CV/speech-dominance
# penalty and a small authenticity bonus for "deep" skills stuffers rarely have.
# ---------------------------------------------------------------------------
def _skill_trust(skill):
    prof = tax.PROFICIENCY_WEIGHT.get(skill.get("proficiency", "intermediate"), 0.7)
    dur = skill.get("duration_months", 0) or 0
    # Saturating duration factor: 0 months -> 0 (kills honeypot expert/0mo too),
    # ~2yrs -> ~1.0, long tenure -> up to 1.5.
    dur_factor = min(dur / 24.0, 1.5)
    end = skill.get("endorsements", 0) or 0
    end_factor = 1.0 + min(end / 30.0, 0.5)
    return prof * dur_factor * end_factor


def skill_score(candidate):
    skills = candidate.get("skills", []) or []
    retrieval_mass = 0.0      # CORE_RETRIEVAL + NLP, the bullseye
    support_mass = 0.0        # LLM + ML foundation + data eng
    cv_mass = 0.0             # CV/speech relevance proxy (by trust, even at 0 weight)
    deep_hits = 0
    named = []

    for s in skills:
        name = s.get("name", "")
        rel = tax.SKILL_RELEVANCE.get(name)
        trust = _skill_trust(s)
        if name in tax.CORE_RETRIEVAL or name in tax.NLP_FOUNDATION:
            retrieval_mass += (rel if rel is not None else 0.0) * trust
            if name in tax.CORE_RETRIEVAL and s.get("proficiency") in ("advanced", "expert"):
                named.append((name, s.get("duration_months", 0) or 0))
        elif rel is not None and rel > 0:
            support_mass += rel * trust
        elif name in tax.CV_SPEECH_SKILLS:
            cv_mass += trust
        if name in tax.DEEP_RETRIEVAL:
            deep_hits += 1

    # Combine: retrieval mass dominates, support contributes with diminishing weight.
    raw = retrieval_mass + 0.45 * support_mass
    # Soft saturation so ~3 strong retrieval skills approach the ceiling.
    score = 1.0 - math.exp(-raw / cfg.SKILL_SATURATION_K)

    # Authenticity nudge for deep skills stuffers seldom list.
    score = min(1.0, score + 0.03 * min(deep_hits, 3))

    # CV/speech dominance: if the candidate's ML identity is mostly CV/speech and
    # retrieval signal is weak, this is the "CV specialist without NLP/IR" the JD
    # does not want.
    cv_dominant = cv_mass > (retrieval_mass + 0.45 * support_mass) and retrieval_mass < 1.0
    if cv_dominant:
        score *= cfg.CV_DOMINANCE_PENALTY

    # Keep the strongest few named retrieval skills (by duration) for reasoning.
    named.sort(key=lambda x: -x[1])
    facts = {
        "top_retrieval_skills": named[:4],
        "deep_skill_hits": deep_hits,
        "cv_dominant": cv_dominant,
    }
    return score, facts


# ---------------------------------------------------------------------------
# Experience band multiplier.
# ---------------------------------------------------------------------------
def experience_mult(candidate):
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    if cfg.EXP_PEAK_LOW <= yoe <= cfg.EXP_PEAK_HIGH:
        m = 1.0
    elif cfg.EXP_BAND_LOW <= yoe < cfg.EXP_PEAK_LOW:
        m = 0.92 + 0.08 * (yoe - cfg.EXP_BAND_LOW) / (cfg.EXP_PEAK_LOW - cfg.EXP_BAND_LOW)
    elif cfg.EXP_PEAK_HIGH < yoe <= cfg.EXP_BAND_HIGH:
        m = 0.92 + 0.08 * (cfg.EXP_BAND_HIGH - yoe) / (cfg.EXP_BAND_HIGH - cfg.EXP_PEAK_HIGH)
    elif yoe < cfg.EXP_BAND_LOW:
        # 5yr -> 0.92 down to floor at ~1yr
        frac = max(0.0, (yoe - 1.0) / (cfg.EXP_BAND_LOW - 1.0))
        m = cfg.EXP_MULT_FLOOR + (0.92 - cfg.EXP_MULT_FLOOR) * frac
    else:
        # >9yr -> taper toward floor by ~16yr (still "in scope, bar gets higher")
        frac = max(0.0, 1.0 - (yoe - cfg.EXP_BAND_HIGH) / 7.0)
        m = cfg.EXP_MULT_FLOOR + (0.92 - cfg.EXP_MULT_FLOOR) * frac
    return max(cfg.EXP_MULT_FLOOR, min(1.0, m)), {"years_of_experience": yoe}


# ---------------------------------------------------------------------------
# Company fit (current + best-in-career) and consulting-career penalty.
# ---------------------------------------------------------------------------
def company_fit(candidate):
    p = candidate.get("profile", {})
    cur = tax.company_score(p.get("current_company", ""), p.get("current_industry", ""))
    best = cur
    consulting_roles = 0
    total_roles = 0
    for h in candidate.get("career_history", []) or []:
        total_roles += 1
        best = max(best, tax.company_score(h.get("company", ""), h.get("industry", "")))
        if tax.is_consulting(h.get("company", "")):
            consulting_roles += 1
    # Reward the best product company they've worked at, not just the current one.
    score = 0.6 * cur + 0.4 * best

    consulting_pen = 1.0
    if total_roles > 0 and consulting_roles == total_roles:
        consulting_pen = cfg.CONSULTING_PENALTY  # entire career in services

    facts = {
        "current_company": p.get("current_company", ""),
        "all_consulting": (total_roles > 0 and consulting_roles == total_roles),
    }
    return score, consulting_pen, facts


# ---------------------------------------------------------------------------
# Education (minor).
# ---------------------------------------------------------------------------
def education_fit(candidate):
    edus = candidate.get("education", []) or []
    if not edus:
        return 0.55, {}
    best = max(tax.EDU_TIER_SCORE.get(e.get("tier", "unknown"), 0.6) for e in edus)
    return best, {}


# ---------------------------------------------------------------------------
# Behavioral modifier: availability (can we actually hire them) + market demand.
# Implements the JD's explicit instruction to down-weight perfect-on-paper
# candidates who are inactive / unresponsive.
# ---------------------------------------------------------------------------
def _days_since(date_str, today):
    from datetime import date
    try:
        y, m, d = map(int, date_str.split("-"))
        return (today - date(y, m, d)).days
    except Exception:
        return 365


def behavioral_modifier(candidate, today):
    s = candidate.get("redrob_signals", {})

    # Availability sub-signals (each 0..1) ---------------------------------
    resp = s.get("recruiter_response_rate", 0.0) or 0.0          # 0..1
    open_flag = 1.0 if s.get("open_to_work_flag") else 0.0
    days = _days_since(s.get("last_active_date", ""), today)
    recency = max(0.0, 1.0 - days / 180.0)                       # 0 at 6 months stale
    notice = s.get("notice_period_days", 90) or 90
    notice_fit = max(0.0, 1.0 - notice / 180.0)                  # JD prefers <30d

    availability = (0.40 * resp + 0.20 * open_flag +
                    0.25 * recency + 0.15 * notice_fit)

    # Market-demand sub-signals (others want them) -------------------------
    saved = s.get("saved_by_recruiters_30d", 0) or 0
    search = s.get("search_appearance_30d", 0) or 0
    demand = 0.5 * min(saved / 15.0, 1.0) + 0.5 * min(search / 200.0, 1.0)

    # Blend, then map onto the [floor, ceil] envelope.
    quality = 0.78 * availability + 0.22 * demand              # 0..1
    mod = cfg.BEHAVIORAL_FLOOR + (cfg.BEHAVIORAL_CEIL - cfg.BEHAVIORAL_FLOOR) * quality

    facts = {
        "recruiter_response_rate": round(resp, 2),
        "open_to_work": bool(s.get("open_to_work_flag")),
        "days_since_active": days,
        "notice_period_days": notice,
        "saved_by_recruiters_30d": saved,
    }
    return mod, facts

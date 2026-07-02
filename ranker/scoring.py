"""
Combines features, semantic score, and honeypot guard into one final score.
Returns a full component breakdown alongside the score for reasoning/audit.
"""
from . import features as F
from . import taxonomies as tax
from . import config as cfg
from .honeypot import honeypot_reasons


def score_candidate(candidate, semantic_score, today):
    p = candidate.get("profile", {})

    role, role_facts = F.role_fit(candidate)
    skill, skill_facts = F.skill_score(candidate)
    comp, consulting_pen, comp_facts = F.company_fit(candidate)
    edu, _ = F.education_fit(candidate)
    exp_mult, exp_facts = F.experience_mult(candidate)
    beh_mod, beh_facts = F.behavioral_modifier(candidate, today)

    loc_mult = tax.location_multiplier(
        p.get("location", ""), p.get("country", ""),
        bool(candidate.get("redrob_signals", {}).get("willing_to_relocate")),
    )

    core = (cfg.W_ROLE * role +
            cfg.W_SKILL * skill +
            cfg.W_SEMANTIC * semantic_score +
            cfg.W_COMPANY * comp +
            cfg.W_EDUCATION * edu)

    fit = core * exp_mult * loc_mult * consulting_pen
    pre = fit * beh_mod

    hp_reasons = honeypot_reasons(candidate, today)
    hp_mult = cfg.HONEYPOT_MULTIPLIER if hp_reasons else 1.0
    final = pre * hp_mult

    breakdown = {
        "role": round(role, 3), "skill": round(skill, 3),
        "semantic": round(float(semantic_score), 3), "company": round(comp, 3),
        "education": round(edu, 3),
        "core": round(core, 3),
        "experience_mult": round(exp_mult, 3),
        "location_mult": round(loc_mult, 3),
        "consulting_penalty": round(consulting_pen, 3),
        "behavioral_modifier": round(beh_mod, 3),
        "honeypot": bool(hp_reasons), "honeypot_reasons": hp_reasons,
        "final": final,
        "facts": {**role_facts, **skill_facts, **comp_facts, **exp_facts, **beh_facts,
                  "location": p.get("location", ""), "country": p.get("country", "")},
    }
    return final, breakdown

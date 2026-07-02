"""
Builds a 1-2 sentence reasoning string per candidate from their breakdown
facts. No hallucination -- every claim comes straight from the profile data.
"""

JD_HOOK = {
    "retrieval": "matches the JD's core need (production retrieval/ranking)",
    "recsys": "built recommendation/search systems, the JD's mandate",
    "adjacent": "adjacent engineering the JD says can convert with the right career",
}


def _exp_phrase(yoe):
    return f"{yoe:.1f} yrs"


def _skill_phrase(top_skills):
    """top_skills: list[(name, duration_months)] already sorted, advanced/expert."""
    if not top_skills:
        return ""
    bits = []
    for name, dur in top_skills[:3]:
        if dur and dur >= 12:
            bits.append(f"{name} ({dur//12}+ yrs)")
        else:
            bits.append(name)
    return ", ".join(bits)


def build_reasoning(breakdown, rank):
    f = breakdown["facts"]
    title = f.get("title", "candidate")
    yoe = f.get("years_of_experience", 0) or 0
    company = f.get("current_company", "")
    top_skills = f.get("top_retrieval_skills", []) or []
    ml_roles = f.get("ml_roles_in_history", 0)

    positives = []
    concerns = []

    # --- lead with role/skill fit -----------------------------------------
    skills_txt = _skill_phrase(top_skills)
    if breakdown["role"] >= 0.8 and skills_txt:
        positives.append(f"{title} ({_exp_phrase(yoe)}) with hands-on {skills_txt}")
    elif breakdown["role"] >= 0.8:
        positives.append(f"{title} ({_exp_phrase(yoe)}) with a consistent ML/AI career")
    elif ml_roles >= 1 and skills_txt:
        positives.append(
            f"{title} ({_exp_phrase(yoe)}) but career shows {ml_roles} ML/AI role(s) and {skills_txt}")
    elif skills_txt:
        positives.append(f"{title} ({_exp_phrase(yoe)}); retrieval skills: {skills_txt}")
    else:
        positives.append(f"{title} with {_exp_phrase(yoe)} experience")

    # --- company / product signal -----------------------------------------
    if breakdown["company"] >= 0.85 and company:
        positives.append(f"product/AI company experience ({company})")

    # --- semantic / depth -------------------------------------------------
    if breakdown["semantic"] >= 0.6:
        positives.append("profile text describes building search/ranking systems")
    if f.get("deep_skill_hits", 0) >= 2:
        positives.append("uses deep retrieval tooling (vector DBs / ranking), not just buzzwords")

    # --- availability positives -------------------------------------------
    resp = f.get("recruiter_response_rate", 0)
    if f.get("open_to_work") and resp >= 0.5:
        positives.append(f"open to work and responsive (response rate {resp:.0%})")

    # --- concerns -----------------------------------------------------------
    if breakdown["location_mult"] < 0.6:
        concerns.append(f"based in {f.get('location') or f.get('country')} (no visa sponsorship)")
    elif 0.6 <= breakdown["location_mult"] < 0.92:
        concerns.append("outside Pune/Noida (relocation needed)")

    if yoe < 5:
        concerns.append(f"slightly below the 5-9 band at {_exp_phrase(yoe)}")
    elif yoe > 11:
        concerns.append(f"senior at {_exp_phrase(yoe)} for a founding IC role")

    if f.get("days_since_active", 0) > 120:
        concerns.append(f"last active ~{f['days_since_active']//30} months ago")
    if resp < 0.25:
        concerns.append(f"low recruiter response rate ({resp:.0%})")
    if (f.get("notice_period_days") or 0) >= 120:
        concerns.append(f"long notice period ({f['notice_period_days']}d)")
    if f.get("all_consulting"):
        concerns.append("career entirely at services firms")
    if f.get("cv_dominant"):
        concerns.append("skills lean CV/speech rather than NLP/IR")
    if breakdown["skill"] < 0.45 and breakdown["role"] >= 0.8:
        concerns.append("retrieval-skill depth is thin")

    # --- assemble, tone tracking rank -------------------------------------
    lead = positives[0]
    extra = "; ".join(positives[1:3])
    sentence = lead + ("; " + extra if extra else "") + "."
    if concerns:
        sentence += " Concern: " + "; ".join(concerns[:2]) + "."
    elif rank > 60:
        sentence += " Solid but not a top-tier match on retrieval depth."

    # Capitalise first letter, keep it tight.
    sentence = sentence[0].upper() + sentence[1:]
    return sentence

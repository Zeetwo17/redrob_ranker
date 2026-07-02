"""
honeypot.py
===========
Detects "subtly impossible" profiles. The spec seeds ~80 honeypots that are
forced to relevance tier 0 in the ground truth; ranking them in the top 100 is a
disqualifier (>10% honeypot rate). They are also a litmus test: a system that
ranks them highly is doing keyword embedding, not reading profiles.

We do NOT special-case them away after the fact — we detect logical
contradictions and apply a hard score penalty, so the guard is explainable and
generalises. Three structural checks (validated on the released pool to fire on
~65 candidates with zero collisions with genuine profiles):

  1. expert_zero_duration : "expert" proficiency in a skill used 0 months.
  2. role_exceeds_calendar : a role's stated duration_months is longer than the
     calendar time between its start_date and end_date (or today).
  3. experience_exceeds_history : years_of_experience far exceeds the sum of all
     career tenures (claiming experience no role accounts for — e.g. "8 yrs at a
     3-yr-old company").

Returns a reason list; empty means clean.
"""
from datetime import date

_TODAY = date(2026, 6, 8)  # dataset "now"; passed explicitly so logic is deterministic


def _parse(d):
    try:
        y, m, dd = map(int, d.split("-"))
        return date(y, m, dd)
    except Exception:
        return None


def _months_between(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month)


def honeypot_reasons(candidate, today=_TODAY):
    reasons = []
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0) or 0
    skills = candidate.get("skills", []) or []
    history = candidate.get("career_history", []) or []

    # 1. expert proficiency, zero months of use
    for s in skills:
        if s.get("proficiency") == "expert" and (s.get("duration_months", 1) or 0) == 0:
            reasons.append("expert_skill_with_zero_months")
            break

    # 2. a role claims more months than the calendar allows
    for h in history:
        start = _parse(h.get("start_date", "") or "")
        end = _parse(h["end_date"]) if h.get("end_date") else today
        dur = h.get("duration_months", 0) or 0
        if start and end:
            calendar = _months_between(start, end)
            if calendar < -1:
                reasons.append("role_dates_reversed")
                break
            if dur - calendar > 4:  # >4 months of slack tolerated for rounding
                reasons.append("role_duration_exceeds_calendar")
                break

    # 3. stated experience far exceeds summed career tenure
    career_years = sum((h.get("duration_months", 0) or 0) for h in history) / 12.0
    if yoe - career_years > 4.5:
        reasons.append("experience_exceeds_career_history")

    return reasons


def is_honeypot(candidate, today=_TODAY) -> bool:
    return len(honeypot_reasons(candidate, today)) > 0

"""
Turn-label verification for RP scenes.

Detects "user-turn poaching" (a model writing the other speaker's turns inside
its own messages) and turn-label leakage (role/turn markers left inside turn
content, which break downstream parsing).
"""

import re
from typing import Dict, List

from rp_pipeline.data.schemas import Scene

# Bracketed labels: [USER - Turn 3], [Turn 4 - ASSISTANT], [user], etc.
_BRACKET_LABEL = re.compile(
    r"\[\s*(USER|ASSISTANT|Turn\s*\d+\s*[-–—]\s*(USER|ASSISTANT)|"
    r"(USER|ASSISTANT)\s*[-–—]\s*Turn\s*\d+)\s*\]",
    re.I,
)
# Bold/markdown headers: **Turn 3 — JUBAL**, ** Turn 5 - MARLOW **
_MD_LABEL = re.compile(
    r"\*\*\s*Turn\s*\d+\s*[-–—:]\s*(?P<name>[A-Z][A-Za-z.' ]{1,40})\s*\*\*"
)
# Bare "Turn 3 — NAME" headers on their own line (no bold, no brackets)
_BARE_LABEL = re.compile(
    r"^\s*Turn\s*\d+\s*[-–—:]\s*(?P<name>[A-Z][A-Za-z.' ]{1,40})\s*$", re.M
)


def _label_role(label_text: str, user_name: str, assistant_name: str) -> str:
    """Classify a turn-label fragment as 'user', 'assistant', or 'unknown'."""
    t = label_text.lower()
    if "user" in t:
        return "user"
    if "assistant" in t:
        return "assistant"
    names = {
        user_name.strip().lower(),
        assistant_name.strip().lower(),
    }
    for name in names:
        if name and name in t:
            return "user" if name == user_name.strip().lower() else "assistant"
    return "unknown"


def check_turn_labels(scene: Scene) -> Dict[str, List[str]]:
    """
    Scan every turn's content for turn-label leakage and cross-role poaching.

    Returns {"poaching": [descriptions...], "label_leak": [descriptions...]}.
    - poaching: a turn contains a label for the OPPOSITE role (the model wrote
      the other speaker's turn) — data-invalid, must be rejected.
    - label_leak: a turn contains a label for its OWN role or an unresolvable
      marker — parse hazard, should be cleaned or rejected.
    """
    user_name = (scene.metadata or {}).get("user_name", "") or "User"
    assistant_name = (scene.metadata or {}).get("assistant_name", "") or "Assistant"

    poaching: List[str] = []
    label_leak: List[str] = []

    for i, turn in enumerate(scene.turns):
        content = turn.content or ""
        own = "assistant" if turn.role.lower().startswith("assist") else "user"
        other = "user" if own == "assistant" else "assistant"

        found: List[str] = []

        for m in _BRACKET_LABEL.finditer(content):
            found.append(("bracket", m.group(0)))
        for m in _MD_LABEL.finditer(content):
            found.append(("md", m.group(0)))
        for m in _BARE_LABEL.finditer(content):
            found.append(("bare", m.group(0).strip()))

        for kind, fragment in found:
            role = _label_role(fragment, user_name, assistant_name)
            desc = f"turn {i} ({own}): {kind} label {fragment!r}"
            if role == other:
                poaching.append(desc)
            else:
                # own-role label or unresolvable name — parse hazard either way
                label_leak.append(desc)

    return {"poaching": poaching, "label_leak": label_leak}


def scene_has_poaching(scene: Scene) -> bool:
    """True if any assistant turn writes the user's side of the scene."""
    return bool(check_turn_labels(scene)["poaching"])


def check_turn_variety(scene: Scene, min_turns: int = 4, uniform_cv: float = 0.12) -> Dict:
    """
    Measure assistant turn-length variability. Scenes where every assistant
    turn is near-identical in size (coefficient of variation below
    `uniform_cv`) read as machine-made; fixing them in rewrite means touching
    every turn, so they are better rejected at draft time.

    Returns {"cv": float, "n_assistant": int, "uniform": bool}.
    """
    lens = [t.word_count for t in scene.turns if t.role.upper().startswith("ASSIST")]
    if len(lens) < min_turns:
        return {"cv": None, "n_assistant": len(lens), "uniform": False}
    mean = sum(lens) / len(lens)
    if mean <= 0:
        return {"cv": None, "n_assistant": len(lens), "uniform": False}
    cv = (sum((x - mean) ** 2 for x in lens) / len(lens)) ** 0.5 / mean
    return {"cv": round(cv, 3), "n_assistant": len(lens), "uniform": cv < uniform_cv}

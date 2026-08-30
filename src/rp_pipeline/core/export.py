"""
Dataset export for RP Pipeline.
Validates, deduplicates, formats and splits final scenes into training-ready files.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rp_pipeline.data.cards import CardDatabase


VALID_ROLES = {"user", "assistant"}


def load_records(path: Path) -> List[Dict[str, Any]]:
    """
    Load scene records from a JSONL file or a directory of JSONL files.

    Each line must be a JSON object with at least `messages`.
    """
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    records: List[Dict[str, Any]] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_source_file"] = f.name
                records.append(rec)
    return records


def load_judgments(judge_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load judge verdicts keyed by scene_id from a judge output directory.

    Judge writes one `<scene_id>.judgment.jsonl` per scene (possibly
    multi-line); later lines override earlier ones.
    """
    if judge_dir is None or not Path(judge_dir).is_dir():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for f in sorted(Path(judge_dir).glob("*.judgment.jsonl")):
        scene_id = f.name[: -len(".judgment.jsonl")]
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                out[scene_id] = data.get("judgments", {})
    return out


def validate_record(
    record: Dict[str, Any], quality: Dict[str, Any]
) -> Tuple[bool, str]:
    """Apply hard quality gates. Returns (ok, reason)."""
    messages = record.get("messages") or []
    if not messages:
        return False, "empty messages"
    for m in messages:
        if m.get("role") not in VALID_ROLES:
            return False, f"invalid role: {m.get('role')!r}"
        if not str(m.get("content", "")).strip():
            return False, "empty message content"

    meta = record.get("metadata", {})
    turn_count = int(meta.get("turn_count", len(messages)))
    token_count = int(meta.get("total_token_count", 0))

    if turn_count < int(quality.get("min_turn_count", 0)):
        return False, f"turn_count {turn_count} < min"
    if turn_count > int(quality.get("max_turn_count", 10**9)):
        return False, f"turn_count {turn_count} > max"
    if token_count < int(quality.get("min_token_count", 0)):
        return False, f"token_count {token_count} < min"
    if token_count > int(quality.get("max_token_count", 10**9)):
        return False, f"token_count {token_count} > max"
    return True, ""


def _normalized_words(text: str) -> set:
    return frozenset(text.lower().split())


def _message_fingerprint(messages: List[Dict[str, str]]) -> str:
    canonical = json.dumps(
        [[m.get("role"), m.get("content", "").strip()] for m in messages],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedup_records(
    records: List[Dict[str, Any]], similarity_threshold: float
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Drop exact duplicates (md5 of messages) and near-duplicates
    (Jaccard over word sets of all message content, vs already-kept items).

    Deterministic: input order decides which copy survives.
    Returns (kept_records, dropped_count).
    """
    kept: List[Dict[str, Any]] = []
    seen_fingerprints: set = set()
    kept_wordsets: List[set] = []
    dropped = 0

    for rec in records:
        messages = rec["messages"]
        fp = _message_fingerprint(messages)
        if fp in seen_fingerprints:
            dropped += 1
            continue

        words = _normalized_words(
            " ".join(str(m.get("content", "")) for m in messages)
        )
        is_near_dup = any(
            _jaccard(words, ws) >= similarity_threshold for ws in kept_wordsets
        )
        if is_near_dup:
            dropped += 1
            continue

        kept.append(rec)
        seen_fingerprints.add(fp)
        kept_wordsets.append(words)

    return kept, dropped


def build_system_prompt(
    card: Optional[Any], record: Dict[str, Any], header: str
) -> str:
    """
    Build the system prompt from the full card profile plus record metadata.
    Falls back to metadata-only if no card is found.
    """
    meta = record.get("metadata", {})
    lines: List[str] = [header, ""]

    if card is not None:
        lines.append(f"## Character: {card.assistant_name}")
        lines.append(card.assistant_character)
        if getattr(card, "assistant_appearance", None):
            lines.append(f"\n## {card.assistant_name}'s appearance")
            lines.append(card.assistant_appearance)

        lines.append(f"\n## Human character: {card.user_name}")
        lines.append(card.user_character)
        if getattr(card, "user_appearance", None):
            lines.append(f"\n## {card.user_name}'s appearance")
            lines.append(card.user_appearance)

        lines.append(f"\n## Scenario")
        lines.append(card.scenario)

        genre = card.genre
        tone = card.tone
    else:
        # Metadata-only fallback
        for key, title in [
            ("assistant_name", "Character"),
            ("user_name", "Human character"),
        ]:
            if meta.get(key):
                lines.append(f"## {title}: {meta[key]}")
        genre = meta.get("genre")
        tone = meta.get("tone")

    tags = [t for t in [genre, tone] if t]
    if tags:
        lines.append(f"\n## Style tags")
        lines.append(", ".join(tags))

    return "\n".join(lines).strip()


def _to_sharegpt(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    role_map = {"system": "system", "user": "human", "assistant": "gpt"}
    return [{"from": role_map[m["role"]], "value": m["content"]} for m in messages]


def stratified_split_by_card(
    records: List[Dict[str, Any]], val_frac: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split whole card_id groups into train/val so variants of the same
    card never straddle the boundary.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(rec.get("card_id") or rec.get("id", ""), []).append(rec)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    val_target = int(round(len(records) * val_frac))
    val, train = [], []
    val_count = 0
    for key in group_keys:
        if val_count < val_target:
            val.extend(groups[key])
            val_count += len(groups[key])
        else:
            train.extend(groups[key])
    return train, val


def export_scenes(
    input_dir: Path,
    output_dir: Path,
    *,
    cards_dir: Optional[Path] = None,
    judge_dir: Optional[Path] = None,
    fmt: str = "messages",
    system_header: str = "You are entering a roleplay scene. Stay fully in character.",
    val_frac: float = 0.02,
    seed: int = 42,
    dedup: bool = True,
    similarity_threshold: float = 0.85,
    quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the export stage. Returns a summary dict (also written as manifest).
    """
    quality = quality or {}

    records = load_records(input_dir)
    stats: Dict[str, Any] = {
        "input_records": len(records),
        "dropped_validation": 0,
        "dropped_duplicates": 0,
        "exported": 0,
        "train": 0,
        "val": 0,
        "validation_reasons": {},
    }

    # 1. Validate
    valid = []
    for rec in records:
        ok, reason = validate_record(rec, quality)
        if ok:
            valid.append(rec)
        else:
            stats["dropped_validation"] += 1
            stats["validation_reasons"][reason] = (
                stats["validation_reasons"].get(reason, 0) + 1
            )

    # 2. Dedup
    if dedup:
        valid, dropped = dedup_records(valid, similarity_threshold)
        stats["dropped_duplicates"] = dropped

    # Deterministic order
    valid.sort(key=lambda r: (r.get("card_id") or "", r.get("id") or ""))

    # 3. Card lookup for full-profile system prompts
    card_db = None
    if cards_dir is not None and Path(cards_dir).exists():
        try:
            card_db = CardDatabase(str(cards_dir))
        except Exception:
            card_db = None

    # 4. Judgments for the metadata sidecar
    judgments = load_judgments(judge_dir)

    # 5. Format
    def to_train_record(rec: Dict[str, Any]) -> Dict[str, Any]:
        meta = rec.get("metadata", {})
        card = None
        if card_db is not None:
            card = card_db.get_by_name(
                meta.get("assistant_name") or rec.get("card_id") or "",
                meta.get("user_name") or "",
            )
        system = build_system_prompt(card, rec, system_header)
        if fmt == "messages":
            return {
                "id": rec.get("id"),
                "messages": [{"role": "system", "content": system}]
                + rec["messages"],
            }
        if fmt == "sharegpt":
            return {
                "id": rec.get("id"),
                "conversations": _to_sharegpt(
                    [{"role": "system", "content": system}] + rec["messages"]
                ),
            }
        if fmt == "text":
            return {"id": rec.get("id"), "text": rec.get("conversation", "")}
        raise ValueError(f"unknown export format: {fmt}")

    # 6. Split + write
    train, val = stratified_split_by_card(valid, val_frac, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    train_records = [to_train_record(r) for r in train]
    val_records = [to_train_record(r) for r in val]

    manifest: Dict[str, Any] = {
        "format": fmt,
        "seed": seed,
        "val_frac": val_frac,
        "dedup": dedup,
        "similarity_threshold": similarity_threshold,
        "quality": quality,
        "stats": stats,
    }

    if train_records:
        manifest["train"] = {
            "file": "train.jsonl",
            "count": len(train_records),
            "sha256": write_jsonl(output_dir / "train.jsonl", train_records),
        }
    if val_records:
        manifest["val"] = {
            "file": "val.jsonl",
            "count": len(val_records),
            "sha256": write_jsonl(output_dir / "val.jsonl", val_records),
        }

    # 7. Metadata sidecar (provenance, kept out of training files)
    meta_path = output_dir / "export_metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as fh:
        for rec in valid:
            sid = rec.get("id", "")
            fh.write(
                json.dumps(
                    {
                        "id": sid,
                        "card_id": rec.get("card_id"),
                        "split": "val"
                        if any(v.get("id") == sid for v in val_records)
                        else "train",
                        "source_file": rec.get("_source_file"),
                        "metadata": rec.get("metadata", {}),
                        "judgments": judgments.get(sid, {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    manifest["metadata_sidecar"] = {
        "file": "export_metadata.jsonl",
        "count": len(valid),
        "sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
    }

    stats["exported"] = len(valid)
    stats["train"] = len(train_records)
    stats["val"] = len(val_records)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest

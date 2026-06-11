from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

VALID_UNITS = [
    "m/s^2", "m/s", "mm", "cm", "km", "kg", "ms", "Hz",
    "m", "g", "s", "N", "J", "W", "V", "A",
]
UNIT_PATTERN = "(?:" + "|".join(re.escape(u) for u in sorted(VALID_UNITS, key=len, reverse=True)) + ")"
NUM_PATTERN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"

FULLWIDTH_TRANS = str.maketrans({
    "−": "-", "–": "-", "—": "-",
    "，": ",", "。": ".", "：": ":", "；": ";",
    "（": "(", "）": ")", "［": "[", "］": "]",
})

MINI_EVAL_TYPE_PRIORITY = [
    "single_integer",
    "decimal",
    "unit_decimal",
    "interval",
    "inequality",
    "equation",
    "fraction",
    "symbolic",
    "boolean",
    "short_text",
    "other",
]


def clean(text: Any) -> str:
    return str(text or "").strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_latex_units(text: str) -> str:
    out = clean(text).translate(FULLWIDTH_TRANS)
    out = re.sub(r"\\(?:mathrm|text)\s*\{\s*~?\s*([^{}]+?)\s*\}", lambda m: " " + m.group(1).strip(), out)
    out = out.replace("\\,", " ").replace("\\;", " ").replace("~", " ")
    out = out.replace("\\cdot", "*").replace("\\times", "*")
    out = out.replace("\\pi", "pi").replace("π", "pi")
    out = re.sub(r"\\sqrt\s*\{\s*([^{}]+?)\s*\}", r"sqrt(\1)", out)
    out = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"\1/\2", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def strip_answer_wrappers(raw: Any) -> str:
    text = clean(raw)
    text = re.sub(r"(?is)^\s*(?:answer|solution|final answer|final solution|the answer|the solution|our answer)\s*(?:is|=|:|：)?\s*", "", text).strip()
    boxed = re.fullmatch(r"\\boxed\{(.+)\}", text, flags=re.S)
    if boxed:
        text = boxed.group(1).strip()
    for left, right in [("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")]:
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left):-len(right)].strip()
    text = text.strip().strip("` '")
    if len(text) > 1 and text.endswith((".", ";", "。")) and not re.search(r"\d\.\d$", text):
        text = text[:-1].strip()
    return normalize_latex_units(text)


def is_interval_answer(compact: str) -> bool:
    return bool(re.fullmatch(r"[\(\[]\s*[^,]+\s*,\s*[^,]+\s*[\)\]]", compact))


def infer_answer_type(answer: Any) -> str:
    ans = strip_answer_wrappers(answer)
    compact = re.sub(r"\s+", "", ans)
    lower = compact.lower()
    if not compact:
        return "other"
    if lower in {"true", "false", "yes", "no"}:
        return "boolean"
    if is_interval_answer(compact):
        return "interval"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:<=|>=|<|>)" + NUM_PATTERN, compact):
        return "inequality"
    if "=" in compact:
        return "equation"
    if re.fullmatch(r"[+-]?\d+/[1-9]\d*", compact):
        return "fraction"
    if re.fullmatch(rf"{NUM_PATTERN}{UNIT_PATTERN}", compact):
        return "unit_decimal"
    if re.fullmatch(r"[-+]?\d+", compact):
        return "single_integer"
    if re.fullmatch(r"[-+]?(?:\d+\.\d+|\.\d+)", compact):
        return "decimal"
    if re.search(r"\d", compact) and re.search(r"[A-Za-z]", compact):
        if re.search(UNIT_PATTERN, compact):
            return "unit_decimal"
        return "symbolic"
    if any(x in compact for x in ["sqrt", "^", "pi", "*", "\\frac"]):
        return "symbolic"
    return "short_text"


def sample_id(row: Dict[str, Any], fallback: int) -> str:
    return clean(row.get("id") or row.get("sample_id") or f"row_{fallback:06d}")


def select_val_mini(rows: List[Dict[str, Any]], size: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    size = max(0, int(size))
    if size <= 0 or not rows:
        return [], {
            "strategy": "disabled" if size <= 0 else "empty_val",
            "requested_size": size,
            "selected_size": 0,
            "selected": [],
            "type_counts": {},
        }

    rng = random.Random(seed)
    buckets: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    type_counts: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        t = clean(row.get("answer_type")) or infer_answer_type(row.get("answer", row.get("solution", "")))
        row = dict(row)
        row["answer_type"] = t
        buckets.setdefault(t, []).append((idx, row))
        type_counts[t] = type_counts.get(t, 0) + 1

    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: List[Tuple[int, Dict[str, Any]]] = []
    selected_ids: set[str] = set()

    def add_candidate(pair: Tuple[int, Dict[str, Any]]) -> bool:
        idx, row = pair
        sid = sample_id(row, idx)
        if sid in selected_ids:
            return False
        selected.append(pair)
        selected_ids.add(sid)
        return True

    for t in MINI_EVAL_TYPE_PRIORITY:
        if len(selected) >= size:
            break
        bucket = buckets.get(t, [])
        if bucket:
            add_candidate(bucket[0])

    remaining: List[Tuple[int, Dict[str, Any]]] = []
    for bucket in buckets.values():
        remaining.extend(bucket)
    rng.shuffle(remaining)
    for pair in remaining:
        if len(selected) >= size:
            break
        add_candidate(pair)

    selected = sorted(selected, key=lambda x: x[0])
    mini_rows = [row for _, row in selected]
    manifest_rows = []
    selected_type_counts: Dict[str, int] = {}
    for idx, row in selected:
        answer_type = clean(row.get("answer_type")) or infer_answer_type(row.get("answer", row.get("solution", "")))
        selected_type_counts[answer_type] = selected_type_counts.get(answer_type, 0) + 1
        manifest_rows.append({
            "source_val_index": idx,
            "id": sample_id(row, idx),
            "answer_type": answer_type,
            "answer": row.get("answer", row.get("solution", "")),
            "source": row.get("source", ""),
            "question_preview": clean(row.get("question"))[:160],
        })

    return mini_rows, {
        "strategy": "stratified_answer_type",
        "requested_size": size,
        "selected_size": len(mini_rows),
        "type_priority": MINI_EVAL_TYPE_PRIORITY,
        "type_counts": type_counts,
        "selected_type_counts": selected_type_counts,
        "selected": manifest_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create S1K_RL/val-mini.jsonl from an existing S1K_RL/val.jsonl without rebuilding the dataset.")
    parser.add_argument("--data-dir", type=str, default="rl_qra_pipeline/data/S1K_RL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mini-eval-size", type=int, default=8)
    parser.add_argument("--keep-validation", action="store_true", help="Do not delete redundant validation*.jsonl files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    val_path = data_dir / "val.jsonl"
    if not val_path.exists():
        fallback = data_dir / "validation.jsonl"
        if fallback.exists():
            val_path = fallback
        else:
            raise FileNotFoundError(f"Neither {data_dir / 'val.jsonl'} nor {fallback} exists.")

    val_rows = read_jsonl(val_path)
    mini_rows, manifest = select_val_mini(val_rows, size=args.mini_eval_size, seed=args.seed + 404)
    manifest.update({
        "data_dir": str(data_dir),
        "source_val_file": str(val_path),
        "seed": args.seed,
    })

    write_jsonl(data_dir / "val-mini.jsonl", mini_rows)
    write_json(data_dir / "val_mini_manifest.json", manifest)

    removed: List[str] = []
    if not args.keep_validation:
        for name in ["validation.jsonl", "validation-mini.jsonl", "validation_mini_manifest.json"]:
            path = data_dir / name
            if path.exists() and path != val_path:
                path.unlink()
                removed.append(str(path))
    manifest["removed_redundant_files"] = removed
    write_json(data_dir / "val_mini_manifest.json", manifest)

    print(json.dumps({
        "data_dir": str(data_dir),
        "source_val_file": str(val_path),
        "val_count": len(val_rows),
        "val_mini_count": len(mini_rows),
        "selected_type_counts": manifest.get("selected_type_counts", {}),
        "removed_redundant_files": removed,
        "output": str(data_dir / "val-mini.jsonl"),
        "manifest": str(data_dir / "val_mini_manifest.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

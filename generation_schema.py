from pathlib import Path


def segment_value(sample, name, default=""):
    segment = sample.get("segments", {}).get(name)
    if not segment:
        return default
    return str(segment.get("text", default))


def assistant_completion(sample):
    parts = []
    seen_assistant = False
    order = sample.get("segment_order", [])
    segments = sample.get("segments", {})
    for name in order:
        segment = segments.get(name)
        if segment is None:
            continue
        if seen_assistant:
            parts.append(str(segment.get("text", "")))
        if name == "assistant_label":
            seen_assistant = True
    return "".join(parts)


def completion_from_full_text(text):
    raw = str(text or "")
    marker = "Assistant:"
    idx = raw.find(marker)
    if idx >= 0:
        return raw[idx + len(marker):].strip()
    return raw.strip()


def split_sections(completion, default_reasoning=""):
    text = str(completion or "").strip()
    reasoning = str(default_reasoning or "")
    answer = text
    lower = text.lower()
    r_idx = lower.find("reasoning:")
    a_idx = lower.find("answer:")
    if r_idx >= 0 and a_idx >= 0 and r_idx < a_idx:
        reasoning = text[r_idx + len("reasoning:") : a_idx].strip()
        answer = text[a_idx + len("answer:") :].strip()
    elif a_idx >= 0:
        answer = text[a_idx + len("answer:") :].strip()
    return {"reasoning": reasoning.strip(), "answer": answer.strip()}


def gt_sections(sample):
    return {
        "question": segment_value(sample, "user").strip(),
        "reasoning": str(sample.get("reasoning") or segment_value(sample, "reasoning")).strip(),
        "answer": str(sample.get("answer") or segment_value(sample, "answer")).strip(),
    }


def make_generation_record(sample, mode, split, data_path, model_generations):
    return {
        "id": sample.get("id", ""),
        "mode": mode,
        "split": split,
        "data_path": str(Path(data_path).as_posix()),
        "GT": gt_sections(sample),
        "model_generations": model_generations,
    }


def completion_block(sections):
    reasoning = str(sections.get("reasoning", "")).strip()
    answer = str(sections.get("answer", "")).strip()
    return f"Reasoning:\n{reasoning}\n\nAnswer:\n{answer}".strip()


def write_generation_markdown(path, title, records):
    lines = [f"# {title}", ""]
    for item in records:
        lines.append(f"## {item['id']}")
        lines.append("")
        lines.append("### Question")
        lines.append("```text")
        lines.append(item["GT"]["question"])
        lines.append("```")
        lines.append("")
        lines.append("### GT")
        lines.append("```text")
        lines.append(completion_block(item["GT"]))
        lines.append("```")
        lines.append("")
        for model_name, sections in item["model_generations"].items():
            lines.append(f"### {model_name}")
            lines.append("```text")
            lines.append(completion_block(sections))
            lines.append("```")
            lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

import argparse
import json
import shutil
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent


def copy_qar_data(source, target, clean_target=True):
    source = (REPO_DIR / source).resolve() if not Path(source).is_absolute() else Path(source)
    target = (REPO_DIR / target).resolve() if not Path(target).is_absolute() else Path(target)
    if clean_target and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        src = source / name
        dst = target / name
        if not src.exists():
            raise FileNotFoundError(f"Missing source data: {src}")
        shutil.copy2(src, dst)
        copied.append(dst)

    # Copy useful reports/manifest when present; these are not required for
    # training but make it clear which QAR version the RL run used.
    optional_names = ["manifest.json"]
    optional_names += [p.name for p in source.glob("*_load_report.json")]
    for name in sorted(set(optional_names)):
        src = source / name
        if src.exists():
            shutil.copy2(src, target / name)
            copied.append(target / name)

    info = {
        "source": str(source),
        "target": str(target),
        "copied": [str(p) for p in copied],
        "note": "RL data is a snapshot of the regenerated QAR JSONL. Re-run this after every QAR data-format change.",
    }
    (target / "copy_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    copied.append(target / "copy_info.json")
    return copied


def main():
    parser = argparse.ArgumentParser(description="Copy regenerated QAR data into the SFT-RL pipeline.")
    parser.add_argument("--source-data", default="sft_answer_pipeline/data/QAR")
    parser.add_argument("--target-data", default="sft_rl_pipeline/data/QAR")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete the old target directory before copying.")
    args = parser.parse_args()
    copied = copy_qar_data(Path(args.source_data), Path(args.target_data), clean_target=not args.no_clean)
    for path in copied:
        print(f"copied {path.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()

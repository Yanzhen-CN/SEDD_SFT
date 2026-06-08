import argparse
import shutil
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent


def copy_qar_data(source, target):
    source = REPO_DIR / source
    target = REPO_DIR / target
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        src = source / name
        dst = target / name
        if not src.exists():
            raise FileNotFoundError(f"Missing source data: {src}")
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main():
    parser = argparse.ArgumentParser(description="Copy QAR data into the SFT-RL pipeline.")
    parser.add_argument("--source-data", default="sft_answer_pipeline/data/QAR")
    parser.add_argument("--target-data", default="sft_rl_pipeline/data/QAR")
    args = parser.parse_args()

    copied = copy_qar_data(Path(args.source_data), Path(args.target_data))
    for path in copied:
        print(f"copied {path.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()

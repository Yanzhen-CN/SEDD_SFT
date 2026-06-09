"""Deprecated one-command runner for answer SFT.

The current experiments are intentionally run step by step:
prepare data -> train -> test -> generate -> visualize.  Keeping this file as
a clear stop sign avoids accidentally rebuilding data and starting training
with stale assumptions.
"""

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    print(
        "\n".join(
            [
                "run_answer_sft.py is deprecated for the current anchored-SFT workflow.",
                "",
                "Use the explicit server workflow instead:",
                f"  python {SCRIPT_DIR / 'prepare_answer_data.py'} --config {SCRIPT_DIR / 'answer_config.yaml'}",
                f"  python {SCRIPT_DIR / 'train_answer.py'} --config {SCRIPT_DIR / 'answer_config.yaml'}",
                f"  python {SCRIPT_DIR / 'test_answer_eval.py'} --config {SCRIPT_DIR / 'answer_config.yaml'}",
                f"  python {SCRIPT_DIR / 'generate_answer_examples.py'} --config {SCRIPT_DIR / 'answer_config.yaml'} --dataset QAR",
                f"  python {SCRIPT_DIR / 'visual_answer.py'}",
                "",
                "RL still has a batch launcher: bash sft_rl_pipeline/launch_rl_six.sh",
            ]
        )
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()

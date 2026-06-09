"""Deprecated entry for the first SFT pipeline.

The active experiments now live in:
  - sft_answer_pipeline
  - sft_reasoning_pipeline
  - sft_rl_pipeline

Use SERVER_WORKFLOW.md for the current server commands.
"""


def main():
    print(
        "run_sft.py belongs to the old block-style pipeline and is deprecated.\n"
        "Use SERVER_WORKFLOW.md and the explicit pipeline commands instead."
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()

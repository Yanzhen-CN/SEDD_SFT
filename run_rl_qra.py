"""Root entry point for the guided RL-QRA pipeline.

Run from the repository root, for example:

    python run_rl_qra.py --start QRA --run-name smoke

This simply delegates to rl_qra_pipeline.run_rl_qra so that both of the
following commands are supported:

    python run_rl_qra.py ...
    python rl_qra_pipeline/run_rl_qra.py ...
"""

from rl_qra_pipeline.run_rl_qra import main


if __name__ == "__main__":
    main()

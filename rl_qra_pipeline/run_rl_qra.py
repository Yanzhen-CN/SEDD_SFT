if __package__:
    from .train_rl_qra import main
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_rl_qra import main

if __name__ == "__main__":
    main()

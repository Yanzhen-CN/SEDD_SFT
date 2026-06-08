import argparse
import json
from pathlib import Path

import torch

from model import SEDD
from model.ema import ExponentialMovingAverage


def main():
    parser = argparse.ArgumentParser(description="Save a Hugging Face SEDD pretrained model as a local .pth checkpoint.")
    parser.add_argument("--model", default="louaaron/sedd-medium")
    parser.add_argument("--output", default="pretrained.pth")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--ema", type=float, default=0.9999)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = SEDD.from_pretrained(args.model)
    model.config.model.length = args.length
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema)
    state = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "step": 0,
        "source": args.model,
        "length": args.length,
    }
    torch.save(state, output)

    meta_path = output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": str(output),
                "source": args.model,
                "length": args.length,
                "ema": args.ema,
                "note": "Local copy of the pretrained starting point for comparable SFT/RL experiments.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {output}")
    print(f"saved {meta_path}")


if __name__ == "__main__":
    main()

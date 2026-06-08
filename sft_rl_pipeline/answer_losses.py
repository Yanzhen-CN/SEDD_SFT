import importlib.util
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
SOURCE = REPO_DIR / "sft_answer_pipeline" / "answer_losses.py"
spec = importlib.util.spec_from_file_location("_sft_answer_losses", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

get_answer_loss_fn = module.get_answer_loss_fn
evaluate_answer_loss = module.evaluate_answer_loss

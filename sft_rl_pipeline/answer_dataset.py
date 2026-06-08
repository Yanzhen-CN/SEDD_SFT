import importlib.util
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
SOURCE = REPO_DIR / "sft_answer_pipeline" / "answer_dataset.py"
spec = importlib.util.spec_from_file_location("_sft_answer_dataset", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

AnswerSegmentDataset = module.AnswerSegmentDataset
collate_answer_batch = module.collate_answer_batch
make_answer_loader = module.make_answer_loader
ordered_segments = module.ordered_segments
sample_text = module.sample_text

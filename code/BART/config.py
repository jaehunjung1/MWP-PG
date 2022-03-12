import os
from typing import Final

from transformers import BartForConditionalGeneration

from dataset import MWPDataset

CONFIGS: Final[dict] = {
    "mawps": {
        "train_filename": [f"data/mawps/proc/{i}.train.proc.json" for i in range(5)],
        "dev_filename": [f"data/mawps/proc/{i}.dev.proc.json" for i in range(5)],
        "dataset_class": MWPDataset,
        "model_class": BartForConditionalGeneration,
    },
    "asdiv-a": {
        "train_filename": [f"data/asdiv-a/proc/{i}.train.proc.json" for i in range(5)],
        "dev_filename": [f"data/asdiv-a/proc/{i}.dev.proc.json" for i in range(5)],
        "dataset_class": MWPDataset,
        "model_class": BartForConditionalGeneration,
    }
}


"""Tokenizer vocabulary generator"""

from PDDiffusion.datasets.WikimediaCommons import local_wikimedia

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataclasses import field
from argparse_dataclass import dataclass

import os.path, sys, json

@dataclass
class TokenizerTrainingOptions:
    output_dir: str = field(default='pd-diffusion-clip', metadata={"args": ["output_dir"], "help": "Where to store the tokenizer vocabulary"})

def label_extractor(dataset_gen):
    for item in dataset_gen:
        yield item["text"]

config = TokenizerTrainingOptions.parse_args(sys.argv[1:])

model = BPE(unk_token="[UNK]")
tokenizer = Tokenizer(model)
trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[START]", "[END]"])

tokenizer.pre_tokenizer = Whitespace()

tokenizer.train_from_iterator(label_extractor(local_wikimedia(load_images=False)), trainer=trainer)

if not os.path.exists("output"):
    os.makedirs("output")

outpath = os.path.join("output", config.output_dir)
if not os.path.exists(outpath):
    os.makedirs(outpath)

tokenizer.save(os.path.join(outpath, "tokenizer.json"))
model.save(outpath)
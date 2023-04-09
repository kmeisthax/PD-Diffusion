"""Command to update old CLIP models' tokenizers to CLIPTokenizerFast."""

from dataclasses import field
from argparse_dataclass import dataclass

from transformers import CLIPTokenizerFast

import os.path, sys

@dataclass
class UpdateTokenizerOptions:
    model_dir: str = field(metadata={"args": ["model_dir"], "help": "The model to update the tokenizer of"})

config = UpdateTokenizerOptions.parse_args(sys.argv[1:])

outpath = os.path.join("output", config.model_dir)

tokenizer = CLIPTokenizerFast.from_pretrained(outpath, from_slow=True)
tokenizer.save_pretrained(outpath)
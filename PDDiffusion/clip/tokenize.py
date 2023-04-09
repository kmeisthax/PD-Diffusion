"""Tokenizer vocabulary generator"""

from PDDiffusion.datasets.WikimediaCommons import local_wikimedia_base
from PDDiffusion.datasets.augment import all_labels_in_item
from PDDiffusion.datasets.load import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from transformers import CLIPModel, CLIPTokenizer, CLIPFeatureExtractor, CLIPProcessor

from dataclasses import field
from argparse_dataclass import dataclass

import os.path, sys, json

@dataclass
class TokenizerTrainingOptions:
    output_dir: str = field(default='pd-diffusion-clip', metadata={"args": ["output_dir"], "help": "Where to store the tokenizer vocabulary"})
    dataset_name: str = field(default="", metadata={"args": ["--dataset_name"], "help": "Dataset name to train on"})
    vision_image_size: int = field(default=224, metadata={"args": ["--vision_image_size"], "help": "The size of each image you intend to train on"})
    text_max_position_embeddings: int = field(default=77, metadata={"args": ["--text_max_position_embeddings"], "help": "Maximum length of labels supported by the text model"})

def label_extractor(dataset_gen):
    for item in dataset_gen:
        yield all_labels_in_item(item)

config = TokenizerTrainingOptions.parse_args(sys.argv[1:])

model = BPE(unk_token="[UNK]")
tokenizer = Tokenizer(model)
trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[START]", "[END]"])

tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(label_extractor(load_dataset(config.dataset_name)), trainer=trainer)

if not os.path.exists("output"):
    os.makedirs("output")

outpath = os.path.join("output", config.output_dir)
if not os.path.exists(outpath):
    os.makedirs(outpath)

feature_extractor = CLIPFeatureExtractor(
    size=config.vision_image_size,
    crop_size=config.vision_image_size,
)

tokenizer.save(os.path.join(outpath, "tokenizer.json"))
model.save(outpath)

clip_tokenizer = CLIPTokenizer(
    vocab_file = os.path.join(outpath, "vocab.json"),
    merges_file = os.path.join(outpath, "merges.txt"),
    tokenizer_file = os.path.join(outpath, "tokenizer.json"),
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[START]",
    eos_token="[END]",
    model_max_length = config.text_max_position_embeddings
)

processor = CLIPProcessor(feature_extractor, clip_tokenizer)
processor.save_pretrained(outpath)
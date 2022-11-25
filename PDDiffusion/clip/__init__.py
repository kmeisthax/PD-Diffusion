from PDDiffusion.datasets.WikimediaCommons import local_wikimedia

from transformers import CLIPModel, CLIPTokenizer, CLIPFeatureExtractor, CLIPProcessor
from datasets import Dataset
import os.path, json

from torchvision import transforms
from datasets import Dataset

import torch

def load_model_and_progress(config, processor):
    model = CLIPModel(config.as_clip_config(processor.tokenizer))

    modeldir = os.path.join("output", config.output_dir)
    progressfile = os.path.join(modeldir, "progress.json")

    if os.path.exists(progressfile):
        model = CLIPModel.from_pretrained(modeldir)
        with open(progressfile, 'r') as progressfile_json:
            progress = json.load(progressfile_json)

            return (model, progress)
    else:
        return (model, {"last_epoch": -1})

def load_processor(config):
    """Load a tokenizer & feature extractor combo that was previously trained and saved in the CLIP model directory."""
    return CLIPProcessor(
        #TODO: Actually calculate dataset image means/stds
        feature_extractor=CLIPFeatureExtractor(
            size=config.vision_image_size,
            crop_size=config.vision_image_size,
        ),
        tokenizer = CLIPTokenizer(
            vocab_file = os.path.join("output", config.output_dir, "vocab.json"),
            merges_file = os.path.join("output", config.output_dir, "merges.txt"),
            tokenizer_file = os.path.join("output", config.output_dir, "tokenizer.json"),
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[START]",
            eos_token="[END]",
            model_max_length = config.text_max_position_embeddings
        )
    )

def convert_and_augment_pipeline(image_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean, std)
        ]
    )

def transformer(image_size, processor):
    preprocess = convert_and_augment_pipeline(image_size, processor.feature_extractor.image_mean, processor.feature_extractor.image_std)

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        texts = processor.tokenizer(examples["text"], padding='max_length', truncation=True)
        
        return {
            "image": images, 
            "input_ids": texts.input_ids,
            "attention_mask": texts.attention_mask
        }
    
    return transform

def load_dataset_with_processor(image_size, processor):
    dataset = Dataset.from_generator(local_wikimedia)

    dataset.set_transform(transformer(image_size, processor), columns=["image", "text"], output_all_columns=True)

    return dataset
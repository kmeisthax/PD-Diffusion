from PDDiffusion.datasets.augment import augment_labels
from PDDiffusion.datasets.load import load_dataset
from PDDiffusion.image_loader import append_base_path_fn

from transformers import CLIPModel, CLIPTokenizer, CLIPFeatureExtractor, CLIPProcessor
import os.path, json

from torchvision import transforms

import torch, datasets, os.path

def load_model_and_progress(model_name, new_model_config=None):
    """Load a model from disk. Also returns its progress file if present.
    
    If the model does not exist and a `new_model_config` is provided, then we
    will return a new model and progress object.
    
    If the model has no progress file, then the progress object returned will
    be None."""

    #TODO: We don't actually SAVE the progress file so that should go away.
    #CLIPTrainer handles checkpointing.

    modeldir = os.path.join("output", model_name)
    progressfile = os.path.join(modeldir, "progress.json")

    if os.path.exists(modeldir):
        model = CLIPModel.from_pretrained(modeldir)
        if os.path.exists(progressfile):
            with open(progressfile, 'r') as progressfile_json:
                progress = json.load(progressfile_json)

                return (model, progress)
        else:
            return (model, None)
    elif new_model_config is not None:
        model = CLIPModel(new_model_config)

        return (model, {"last_epoch": -1})
    else:
        raise Exception(f"Model {model_name} not found on disk and no new model config was provided")

def load_processor(model_name):
    """Load a tokenizer & feature extractor combo that was previously trained."""
    return CLIPProcessor.from_pretrained(os.path.join("output", model_name))

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
        texts = processor.tokenizer(augment_labels(examples), padding='max_length', truncation=True)
        
        return {
            "image": images,
            "input_ids": texts.input_ids,
            "attention_mask": texts.attention_mask
        }
    
    return transform

def load_dataset_with_processor(dataset_name, image_size, processor):
    """Load a dataset with an image column.
    
    This works similarly to the generic loader in image_loader.py, but we
    require an additional transform step with an image normalizer for CLIP
    things to work."""
    path = os.path.join("output", dataset_name)

    dataset = load_dataset(dataset_name).map(
        append_base_path_fn(path),
        input_columns="image"
    ).cast_column("image", datasets.Image())

    print(f"Dataset loaded with {dataset.num_rows} rows")

    dataset.set_transform(transformer(image_size, processor), output_all_columns=True)

    return dataset
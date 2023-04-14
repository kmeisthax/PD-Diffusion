"""Command to add CLIP vectors to a dataset in advance.

Precalculated CLIP vectors can be used to train other models that need them
without keeping both CLIP and the other model in memory at once. Multiple
vectors can be calculated with different inputs (e.g. image augments, different
caption arrangements) to ensure better data coverage.

This command copies the whole dataset when adding the CLIP vectors. The
original dataset will be unmodified."""

from PDDiffusion.datasets.augment import augment_labels
from PDDiffusion.datasets.load import load_dataset
from PDDiffusion.clip import load_processor, load_model_and_progress, convert_and_augment_pipeline

from dataclasses import field
from argparse_dataclass import dataclass
from accelerate import find_executable_batch_size, Accelerator
from tqdm.auto import tqdm
from PIL import Image

import sys, os.path, torch, math

@dataclass
class PrecalculateOptions:
    source_dir: str = field(metadata={"args": ["source_dir"], "help": "The dataset to add CLIP vectors to"})
    output_dir: str = field(metadata={"args": ["output_dir"], "help": "Where to store the modified dataset"})

    clip_model: str = field(metadata={"args": ["--clip_model"], "help": "The CLIP model to calculate condition vectors with"})
    batch_size: int = field(default = 128, metadata={"args": ["--batch_size"], "help": "How many images to calculate CLIP vectors for at once"})

    mixed_precision: str = field(default = 'no', metadata={"args": ["--mixed_precision"], "choices": ["no", "fp16", "bf16"], "help": "What mixed-precision mode to use, if any."})

    num_augments: int = field(default = 4, metadata={"args": ["--num_augments"], "help": "How many CLIP vectors to generate per example. One vector will be calculated from image data, the rest will be text."})

config = PrecalculateOptions.parse_args(sys.argv[1:])

processor = load_processor(config.clip_model)
(clip_model, progress) = load_model_and_progress(config.clip_model)
dataset = load_dataset(config.source_dir)
augment_pipeline = convert_and_augment_pipeline(
    clip_model.vision_model.config.image_size,
    processor.feature_extractor.image_mean,
    processor.feature_extractor.image_std
)

# We copy all the config data out so that the classify step doesn't print a
# Pickle error about circular data.
mixed_precision = config.mixed_precision
source_dir = config.source_dir
output_dir = config.output_dir
batch_size = config.batch_size
num_augments = config.num_augments

def clip_classify(data_items):
    """Computes CLIP vectors for all data items in the model.
    
    Intended to be used as a datasets map function, called with all the data in
    one go. Batches calculations internally.
    
    Inserts a new column into the output called "condition". This is a Torch
    tensor of dimension (d, 2a, c):
    
        - *d* refers to the number of Data items in the dataset
        - *a* refers to the number of Augments calculated per dataset item
            (one image vector + a-1 text vectors generated from random
            combinations of text)
        - *c* refers to the length of CLIP's output vector"""

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        logging_dir=os.path.join("output", output_dir, "logs")
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("clip_precalculate")
    
    model = accelerator.prepare(clip_model)
    
    @find_executable_batch_size(starting_batch_size=batch_size)
    def inner_batch(batch_size):
        progress_bar = tqdm(total=math.ceil(len(data_items["image"]) / batch_size), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"CLIP Image Processing")

        conditions = []

        if str(model.device).startswith("cuda:"):
            progress_bar.set_postfix({"mem": torch.cuda.memory_allocated(), "max": torch.cuda.max_memory_allocated()})

        for i in range(0, len(data_items["image"]), batch_size):
            if str(model.device).startswith("cuda:"):
                progress_bar.set_postfix({"mem": torch.cuda.memory_allocated(), "max": torch.cuda.max_memory_allocated()})
            
            clip_images = [augment_pipeline(Image.open(os.path.join("output", source_dir, image)).convert("RGB")) for image in data_items["image"][i:i+batch_size]]
            encoded_images = processor(images=clip_images, return_tensors="pt")["pixel_values"]
            encoded_text = processor(text=augment_labels(data_items, i, i+batch_size, num_augments - 1), padding='max_length', truncation=True, max_length=model.text_model.config.max_position_embeddings)

            encoded_text_ids = torch.tensor(encoded_text["input_ids"])
            encoded_text_mask = torch.tensor(encoded_text["attention_mask"])

            if model.device != "cpu":
                encoded_images = encoded_images.to(model.device)
                encoded_text_ids = encoded_text_ids.to(model.device)
                encoded_text_mask = encoded_text_mask.to(model.device)
            
            condition = model(input_ids=encoded_text_ids, attention_mask=encoded_text_mask, pixel_values=encoded_images)
            image_embeds = condition.image_embeds.reshape(
                condition.image_embeds.shape[0],
                1,
                *list(condition.image_embeds.shape[1:])
            )
            text_embeds = condition.text_embeds.reshape(
                condition.text_embeds.shape[0] // (num_augments - 1),
                (num_augments - 1),
                *list(condition.text_embeds.shape[1:])
            )

            if mixed_precision == "fp16":
                image_embeds = image_embeds.type(torch.float16)
                text_embeds = text_embeds.type(torch.float16)
            elif mixed_precision == "bf16":
                image_embeds = image_embeds.type(torch.bfloat16)
                text_embeds = text_embeds.type(torch.bfloat16)
            else:
                image_embeds = image_embeds.type(torch.float32)
                text_embeds = text_embeds.type(torch.float32)
            
            for (image_row, text_row) in zip(image_embeds, text_embeds):
                conditions.append(torch.concat((image_row, text_row)))
            
            progress_bar.update(1)
        
        if str(model.device).startswith("cuda:"):
            #Fully unload CLIP off the GPU since we won't need it anymore.
            torch.cuda.empty_cache()
            progress_bar.set_postfix({"mem": torch.cuda.memory_allocated(), "max": torch.cuda.max_memory_allocated()})
        
        return {
            "image": data_items["image"],
            "conditions": conditions,
        }
    
    return inner_batch()

with torch.no_grad():
    dataset = dataset.map(clip_classify, input_columns=None, batched=True, batch_size=None)

dataset.save_to_disk(
    os.path.join("output", config.output_dir),
    max_shard_size="25MB"
)
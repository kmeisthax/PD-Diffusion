import dataclasses

from PDDiffusion.datasets.WikimediaCommons import local_wikimedia
from datasets import Dataset
from diffusers import UNet2DModel
from torchvision import transforms
import os.path, json

@dataclasses.dataclass
class SampleConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500

    #Generate an image every n epochs.
    #Image generation is processor-intensive, so this number should be relatively high.
    #Images will also be generated on the last epoch in the current run, no matter what.
    save_image_epochs = 999

    #Save the model every n epochs.
    #This is cheap, so I recommend doing it every epoch.
    #Training will resume from a saved model.
    save_model_epochs = 1
    mixed_precision = 'no'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'pd-diffusion'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

def preprocessor(config):
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def transformer(config):
    preprocess = preprocessor(config)

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}
    
    return transform

def load_dataset(config):
    dataset = Dataset.from_generator(local_wikimedia)

    dataset.set_transform(transformer(config))

    return dataset

def load_model_and_progress(config):
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
        down_block_types=( 
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D", 
            "DownBlock2D", 
            "DownBlock2D", 
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ), 
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D", 
            "UpBlock2D"  
        ),
    )

    progress = {"last_epoch": 0}

    if os.path.exists(os.path.join(config.output_dir, "unet")):
        model = UNet2DModel.from_pretrained(os.path.join(config.output_dir, "unet"))

        with open(os.path.join(config.output_dir, "progress.json"), "r") as progress_file:
            progress = json.load(progress_file)
    
    return (model, progress)
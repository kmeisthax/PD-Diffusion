from PDDiffusion.datasets.WikimediaCommons import local_wikimedia
from PDDiffusion.unet.test import load_pretrained_unet

from diffusers import UNet2DModel
import os.path, json, torch
from dataclasses import field
from argparse_dataclass import dataclass
from PIL import Image

@dataclass
class TrainingOptions:
    image_size: int = field(default = 128, metadata={"args": ["--image_size"], "help": "The image resolution to train the model at."})
    train_batch_size: int = field(default = 16, metadata={"args": ["--train_batch_size"], "help": "How many images to train per step. Will be reduced automatically if this exceeds the memory size of your GPU."})
    eval_batch_size: int = field(default = 16, metadata={"args": ["--eval_batch_size"], "help": "How many output image samples to generate on a save-image epoch"})
    num_epochs: int = field(default = 50, metadata={"args": ["--num_epochs"], "help": "How many epochs to train for. You can only generate sample images or checkpoint the model at an epoch boundary."})
    gradient_accumulation_steps: int = field(default=1, metadata={"args": ["--gradient_accumulation_steps"]})
    learning_rate: float = field(default=1e-4, metadata={"args": ["--learning_rate"]})
    lr_warmup_steps: int = field(default=500, metadata={"args": ["--lr_warmup_steps"]})

    #Generate an image every n epochs.
    #Image generation is processor-intensive, so this number should be relatively high.
    #Images will also be generated on the last epoch in the current run, no matter what.
    save_image_epochs: int = field(default=999, metadata={"args": ["--save_image_epochs"], "help": "How many epochs to wait in between sample image generation."})

    #Save the model every n epochs.
    #This is cheap, so I recommend doing it every epoch.
    #Training will resume from a saved model.
    save_model_epochs: int = field(default=1, metadata={"args": ["--save_model_epochs"], "help": "How many epochs to wait in between saving model checkpoints."})
    mixed_precision: str = field(default = 'no', metadata={"args": ["--mixed_precision"], "choices": ["no", "fp16", "bf16"], "help": "What mixed-precision mode to use, if any."})
    output_dir: str = field(default='pd-diffusion', metadata={"args": ["output_dir"], "help": "The directory name to save the model to. Will also be used as the name of the model for Huggingface upload, if selected."})

    push_to_hub: bool = field(default=False, metadata={"args": ["--push_to_hub"], "help": "Automatically upload trained model weights to Huggingface's model hub"})
    seed: int = field(default=0, metadata={"args": ["--seed"], "help": "The seed to use when generating sample images."})

    ddpm_train_timesteps: int = field(default=1000, metadata={"args": ["--ddpm_train_timesteps"]})
    ddpm_beta_schedule: str = field(default="linear", metadata={"args": ["--ddpm_beta_schedule"]})

    adam_beta1:float = field(default=0.95, metadata={"args": ["--adam_beta1"]})
    adam_beta2:float = field(default=0.999, metadata={"args": ["--adam_beta2"]})
    adam_weight_decay:float = field(default=1e-6, metadata={"args": ["--adam_weight_decay"]})
    adam_epsilon:float = field(default=1e-08, metadata={"args": ["--adam_epsilon"]})

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

    progress = {"last_epoch": -1}

    if os.path.exists(os.path.join(config.output_dir, "unet")):
        model = load_pretrained_unet(config.output_dir)

        with open(os.path.join(config.output_dir, "progress.json"), "r") as progress_file:
            progress = json.load(progress_file)
    
    return (model, progress)

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    """Generate a grid of sample images using a partially-trained pipeline.

    The training seed will be stable across multiple epochs so that human
    observers can visually inspect training progress over time.
    
    Images will be stored in the model's samples directory with the given epoch's number."""
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    
    return image_grid

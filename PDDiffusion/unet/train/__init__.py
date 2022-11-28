from PDDiffusion.datasets.WikimediaCommons import local_wikimedia
from PDDiffusion.image_loader import load_dataset, convert_and_augment_pipeline, transformer
from PDDiffusion.unet.test import load_pretrained_unet
from PDDiffusion.unet.pipeline import DDPMConditionalPipeline

from diffusers import UNet2DModel, UNet2DConditionModel, DDPMPipeline
from transformers import CLIPModel, CLIPTextModel, CLIPProcessor, CLIPFeatureExtractor, CLIPTokenizer
from datasets import Dataset
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

    #Conditional training options
    conditioned_on: str = field(default=None, metadata={"args": ["--conditioned_on"], "help": "Train a conditional model using this CLIP model's space as guidance."})
    evaluation_prompt: str = field(default="a guinea pig", metadata={"args": ["--evaluation_prompt"], "help": "Sample prompt for evaluating a conditional U-Net"})

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

def load_condition_model_and_processor(config):
    """Load the CLIP model as specified in our config.

    Returns both the CLIP Processor and Model, in that order. If the model
    calls for unconditional generation, returns nothing."""
    if config.conditioned_on is None:
        return (None, None)
    
    location = os.path.join("output", config.conditioned_on)
    return (
        CLIPProcessor(
            CLIPFeatureExtractor.from_pretrained(location),
            CLIPTokenizer.from_pretrained(location)
        ),
        CLIPModel.from_pretrained(location)
    )

def load_model_and_progress(config, conditional_model=None):
    """Load a (potentially partially-trained) model from disk.
    
    If the configuration calls for a conditional U-Net (e.g. for use with CLIP)
    then the conditional_model to train against should be provided here. We
    will produce a model whose encoder_hidden_states is compatible with your
    provided conditional model."""
    common_model_settings = {
        "sample_size": config.image_size,  # the target image resolution
        "in_channels": 3,  # the number of input channels, 3 for RGB images
        "out_channels": 3,  # the number of output channels
        "layers_per_block": 2,  # how many ResNet layers to use per UNet block
        "block_out_channels": (128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    }

    if config.conditioned_on is not None:
        if conditional_model is None:
            raise Exception("Cannot do conditional training without model to train against")

        model = UNet2DConditionModel(
            cross_attention_dim=conditional_model.config.projection_dim,

            #Conditional U-Nets need cross-attention block types.
            #This is roughly patterned against the Stable Diffusion ones,
            #though Stable Diffusion works in a latent space and we're not (yet)
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D", 
                "DownBlock2D",
                "DownBlock2D",
            ), 
            up_block_types= (
                "UpBlock2D",
                "UpBlock2D",
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D"
            ),
            **common_model_settings
        )
    else:
        model = UNet2DModel(
            **common_model_settings,
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ), 
            up_block_types= (
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"  
            ),
        )
    
    progress = {"last_epoch": -1}

    if os.path.exists(os.path.join("output", config.output_dir, "unet")):
        model = load_pretrained_unet(os.path.join("output", config.output_dir), config.conditioned_on is not None)

        with open(os.path.join("output", config.output_dir, "progress.json"), "r") as progress_file:
            progress = json.load(progress_file)
    
    return (model, progress)

def load_dataset_with_condition(config, cond_processor=None, cond_model=None):
    """Load all our datasets.
    
    If a conditional model and processor are provided, condition-space vectors
    for that model will be calculated for every image in the training set. They
    will be stored in the "condition" column for your dataset."""

    if cond_model is None or cond_processor is None:
        return load_dataset(config.image_size)
    
    dataset = Dataset.from_generator(local_wikimedia)
    
    preprocess = convert_and_augment_pipeline(config.image_size)
    clip_preprocess = convert_and_augment_pipeline(cond_model.vision_model.config.image_size)

    def clip_classify(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        clip_images = [clip_preprocess(image.convert("RGB")) for image in examples["image"]]
        encoded_images = cond_processor(images=clip_images, return_tensors="pt")
        if cond_model.device != "cpu":
            encoded_images.to(cond_model.device)
        
        condition = cond_model.visual_projection(cond_model.vision_model(**encoded_images)[1])

        #Don't ask why but it does a normalization step in CLIPModel.
        #I can't do that here because CLIPModel will not allow you to use it
        #without both text and image input, which is stupid
        condition = condition / condition.norm(p=2, dim=-1, keepdim=True)

        return {
            "image": images,
            "condition": torch.unsqueeze(condition, 0)
        }

    dataset.set_transform(clip_classify, columns=["image"], output_all_columns=True)

    return dataset

def create_model_pipeline(config, accelerator, model, noise_scheduler, cond_model=None, cond_processor=None):
    """Create a model pipeline for evaluation or saving.
    
    cond_model and cond_processor should be the conditional model and
    preprocessor used to calcluate guidance during training. If you aren't
    training a conditional model, these may be omitted."""

    if config.conditioned_on is not None:
        #We can't actually USE the text model from cond_model lol
        return DDPMConditionalPipeline(
            unet=accelerator.unwrap_model(model),
            text_encoder=CLIPTextModel.from_pretrained(os.path.join("output", config.conditioned_on)).to(model.device),
            tokenizer=cond_processor.tokenizer,
            scheduler=noise_scheduler
        )
    else:
        return DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

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
    
    Images will be stored in the model's samples directory with the given epoch's number.

    If the config has conditional training enabled, we assume the pipeline is
    also conditional and provide it with a prompt."""
    pipeline_params = {
        "batch_size": config.eval_batch_size,
        "generator": torch.manual_seed(config.seed)
    }
    if config.conditioned_on is not None:
        pipeline_params["prompt"] = config.evaluation_prompt

    images = pipeline(**pipeline_params).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join("output", config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    
    return image_grid

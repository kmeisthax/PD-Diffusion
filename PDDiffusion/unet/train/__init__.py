from PDDiffusion.datasets.WikimediaCommons import local_wikimedia_base
from PDDiffusion.datasets.augment import augment_labels
from PDDiffusion.image_loader import load_dataset, convert_and_augment_pipeline, transformer
from PDDiffusion.unet.test import load_pretrained_unet
from PDDiffusion.unet.pipeline import DDPMConditionalPipeline

from accelerate import find_executable_batch_size
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMPipeline
from transformers import CLIPModel, CLIPTextModel, CLIPProcessor, CLIPFeatureExtractor, CLIPTokenizer
from datasets import Dataset, Features, Value, Image as DatasetsImage
from tqdm.auto import tqdm
import os.path, json, torch, math
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
    gradient_checkpointing: bool = field(default=False, metadata={"args": ["--gradient_checkpointing"]})
    learning_rate: float = field(default=1e-4, metadata={"args": ["--learning_rate"]})
    lr_warmup_steps: int = field(default=500, metadata={"args": ["--lr_warmup_steps"]})

    #Data load strategy
    pin_data_in_memory: bool = field(default=False, metadata={"args": ["--pin_data_in_memory"], "help": "Force dataset to remain in CPU memory"})
    data_load_workers: int = field(default=0, metadata={"args": ["--data_load_workers"], "help": "Number of workers to load data with"})
    image_limit: float = field(default=None, metadata={"args": ["--image_limit"], "help": "Number of images per batch to train"})

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

def load_model_and_progress(config, conditional_model_config=None):
    """Load a (potentially partially-trained) model from disk.
    
    If the configuration calls for a conditional U-Net (e.g. for use with CLIP)
    then the configuration of the conditional_model to train against should be
    provided here. We will produce a model whose encoder_hidden_states is
    compatible with your provided conditional model."""
    common_model_settings = {
        "sample_size": config.image_size,  # the target image resolution
        "in_channels": 3,  # the number of input channels, 3 for RGB images
        "out_channels": 3,  # the number of output channels
        "layers_per_block": 2,  # how many ResNet layers to use per UNet block
        "block_out_channels": (128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    }

    if config.conditioned_on is not None:
        if conditional_model_config is None:
            raise Exception("Cannot do conditional training without model config to train against")

        model = UNet2DConditionModel(
            cross_attention_dim=conditional_model_config.projection_dim,

            #Conditional U-Nets need cross-attention block types.
            #This is roughly patterned against the Stable Diffusion ones,
            #though Stable Diffusion works in a latent space and we're not (yet)
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
                "CrossAttnDownBlock2D",  # a ResNet downsampling block with cross attention
                "DownBlock2D",
            ), 
            up_block_types= (
                "UpBlock2D",  # a regular ResNet upsampling block
                "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"  
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
    
    if isinstance(model, UNet2DConditionModel):
        model.set_use_memory_efficient_attention_xformers(True)

    return (model, progress)

def load_dataset_with_condition(config, accelerator):
    """Load all our datasets.

    If the config calls for conditional training, we will load the specified
    conditional model and insert condition vectors into the dataset. This is an
    all-at-once process so that the conditional model does not remain loaded in
    accelerator memory during the training process.
    
    All models used in this process will be loaded onto the given accerlator
    and then unloaded when done.
    
    Returns the conditional model configuration if applicable."""

    if config.conditioned_on is None:
        return (load_dataset(config.image_size), None)
    
    dataset = Dataset.from_generator(local_wikimedia_base)

    transform = transformer(config.image_size)

    with torch.no_grad():
        (cond_processor, cond_model) = load_condition_model_and_processor(config)
        clip_preprocess = convert_and_augment_pipeline(cond_model.vision_model.config.image_size)

        #Extract the config outside of the batched/no-grad area
        cond_config = cond_model.config

        def clip_classify(data_items, num_augments = 4):
            """Computes CLIP vectors for all data items in the model.
            
            Intended to be used as a datasets map function, called with all the
            data in one go. Batches calculations internally.
            
            Inserts a new column into the output called "condition". This is a
            Torch tensor of dimension (d, 2a, c):
            
             - *d* refers to the number of Data items in the dataset
             - *a* refers to the number of Augments calculated per dataset item
                   (we generate both an image and text side vector per augment)
             - *c* refers to the length of CLIP's output vector
            
            By default, we prepare four image augments and four text augments.
            These are calcluated by shuffling the text parameters around with
            augment_labels, and the images around with clip_preprocess, which
            applies flips to the image."""
            nonlocal cond_model

            cond_model = accelerator.prepare(cond_model)

            clip_batch_size = config.train_batch_size * 16

            @find_executable_batch_size(starting_batch_size=clip_batch_size)
            def inner_batch(batch_size):
                progress_bar = tqdm(total=math.ceil(len(data_items["image"]) / batch_size), disable=not accelerator.is_local_main_process)
                progress_bar.set_description(f"CLIP Image Processing")

                conditions = []

                if str(cond_model.device).startswith("cuda:"):
                    progress_bar.set_postfix({"mem": torch.cuda.memory_allocated(), "max": torch.cuda.max_memory_allocated()})

                for i in range(0, len(data_items["image"]), batch_size):
                    if str(cond_model.device).startswith("cuda:"):
                        progress_bar.set_postfix({"mem": torch.cuda.memory_allocated(), "max": torch.cuda.max_memory_allocated()})
                    
                    clip_images = [clip_preprocess(image.convert("RGB")) for _ in range(0, num_augments) for image in data_items["image"][i:i+batch_size]]
                    encoded_images = cond_processor(images=clip_images, return_tensors="pt")["pixel_values"]
                    encoded_text = cond_processor(text=augment_labels(data_items, i, i+batch_size, num_augments), padding='max_length', truncation=True, max_length=cond_model.text_model.config.max_position_embeddings)

                    encoded_text_ids = torch.tensor(encoded_text["input_ids"])
                    encoded_text_mask = torch.tensor(encoded_text["attention_mask"])

                    if cond_model.device != "cpu":
                        encoded_images = encoded_images.to(cond_model.device)
                        encoded_text_ids = encoded_text_ids.to(cond_model.device)
                        encoded_text_mask = encoded_text_mask.to(cond_model.device)
                    
                    condition = cond_model(input_ids=encoded_text_ids, attention_mask=encoded_text_mask, pixel_values=encoded_images)
                    image_embeds = condition.image_embeds.reshape(condition.image_embeds.shape[0] // num_augments, num_augments, *list(condition.image_embeds.shape[1:]))
                    text_embeds = condition.text_embeds.reshape(condition.image_embeds.shape[0] // num_augments, num_augments, *list(condition.text_embeds.shape[1:]))

                    if config.mixed_precision == "fp16":
                        image_embeds = image_embeds.type(torch.float16)
                        text_embeds = text_embeds.type(torch.float16)
                    elif config.mixed_precision == "bf16":
                        image_embeds = image_embeds.type(torch.bfloat16)
                        text_embeds = text_embeds.type(torch.bfloat16)
                    else:
                        image_embeds = image_embeds.type(torch.float32)
                        text_embeds = text_embeds.type(torch.float32)
                    
                    for (image_row, text_row) in zip(image_embeds, text_embeds):
                        conditions.append(torch.concat((image_row, text_row)))
                    
                    progress_bar.update(1)
                
                if str(cond_model.device).startswith("cuda:"):
                    #Fully unload CLIP off the GPU since we won't need it anymore.
                    torch.cuda.empty_cache()
                    progress_bar.set_postfix({"mem": torch.cuda.memory_allocated(), "max": torch.cuda.max_memory_allocated()})
                
                return {
                    "image": data_items["image"],
                    "conditions": conditions,
                }
            
            return inner_batch()
        
        dataset = dataset.map(clip_classify,
            input_columns=None,
            batched=True,
            batch_size=None)
        dataset.set_transform(transform, columns=["image"], output_all_columns=True)

        return (dataset, cond_config)

def create_model_pipeline(config, accelerator, model, noise_scheduler):
    """Create a model pipeline for evaluation or saving."""

    if config.conditioned_on is not None:
        (cond_processor, cond_model) = load_condition_model_and_processor(config)

        #Shut up warnings about loading both model heads into the text model
        CLIPTextModel._keys_to_ignore_on_load_unexpected = ['vision_model\..*', 'visual_projection\..*', 'text_projection\..*', 'logit_scale']

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

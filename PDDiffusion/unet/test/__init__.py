import os, torch, json

from PIL import Image
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMPipeline, DDPMScheduler

def load_pretrained_unet(output_dir, is_conditional=False):
    """Load pretrained UNet weights from a trained model package."""
    if is_conditional:
        return UNet2DConditionModel.from_pretrained(os.path.join(output_dir, "unet"))
    else:
        return UNet2DModel.from_pretrained(os.path.join(output_dir, "unet"))

def load_pretrained_pipeline(output_dir, accelerator=None):
    """Load a pretrained unconditional image generation pipeline from disk.
    
    If an accelerator is provided, the pipeline will execute the given model on
    that accelerator."""
    with open(os.path.join(output_dir, "scheduler", "scheduler_config.json"), "r") as config_file:
        config = json.load(config_file)

        noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_train_timesteps"], beta_schedule=config["beta_schedule"])
        model = load_pretrained_unet(output_dir)
        if accelerator is not None:
            model = accelerator.prepare(model)

        return DDPMPipeline(unet=model, scheduler=noise_scheduler)

import os, torch, json

from PIL import Image
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMPipeline
from PDDiffusion.unet.pipeline import DDPMConditionalPipeline

def load_pretrained_unet(output_dir, is_conditional=False):
    """Load pretrained UNet weights from a trained model package."""
    if is_conditional:
        return UNet2DConditionModel.from_pretrained(os.path.join(output_dir, "unet"))
    else:
        return UNet2DModel.from_pretrained(os.path.join(output_dir, "unet"))

def load_pretrained_pipeline_into_accelerator(output_dir, accelerator=None):
    """Load a pretrained unconditional image generation pipeline from disk.
    
    Returns both the pipeline itself as well as a bool that flags if it supports
    text prompts or not.

    The provided accelerator will have all the pipeline's models loaded into it."""
    with open(os.path.join(output_dir, "model_index.json"), "r") as index_file:
        index = json.load(index_file)

        if index["_class_name"] == "DDPMPipeline":
            pipeline = DDPMPipeline.from_pretrained(output_dir)

            if accelerator is not None:
                pipeline.unet = accelerator.prepare(pipeline.unet)
            
            return (pipeline, False)
        elif index["_class_name"] == "DDPMConditionalPipeline":
            pipeline = DDPMConditionalPipeline.from_pretrained(output_dir)

            if accelerator is not None:
                (pipeline.unet, pipeline.text_encoder) = accelerator.prepare(pipeline.unet, pipeline.text_encoder)
            
            return (pipeline, True)
        else:
            raise Exception("Unsupported pipeline")

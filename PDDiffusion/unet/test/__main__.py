from PDDiffusion.unet.test import load_pretrained_pipeline_into_accelerator
from dataclasses import field
from argparse_dataclass import dataclass
from accelerate import Accelerator
from PIL import Image
import sys, torch, os.path, math

@dataclass
class GenerationOptions:
    output: str = field(metadata={"args": ["output"]})
    model_dir: str = field(default='pd-diffusion', metadata={"args": ["--model_dir"]})
    text_prompt: str = field(default=None, metadata={"args": ["--text_prompt"], "help": "The text prompt to use for conditional models."})

    num_images: int = field(default=1, metadata={"args": ["--num_images"], "help": "Number of images to generate."})
    seed: int = None

options = GenerationOptions.parse_args(sys.argv[1:])

accelerator = Accelerator()

#TODO: What if the user has an absolute model path?
(pipeline, is_conditional) = load_pretrained_pipeline_into_accelerator(os.path.join("output", options.model_dir), accelerator=accelerator)

seed = None
if options.seed is not None:
    seed = torch.manual_seed(options.seed)

pipeline_args = {
    "generator": seed
}

if is_conditional:
    if options.text_prompt is None:
        raise Exception("Attempting to use conditional pipeline without a text prompt.")
    
    pipeline_args["prompt"] = options.text_prompt
elif not is_conditional and options.text_prompt is not None:
    raise Exception("Attempting to use unconditional pipeline with a text prompt.")

pipeline_args["batch_size"] = options.num_images
square_size = math.ceil(math.sqrt(options.num_images))

images = pipeline(**pipeline_args).images

target_image = Image.new('RGBA', size=(images[0].size[0] * square_size, images[0].size[1] * square_size))
for i, image in enumerate(images):
    target_image.paste(image.convert("RGBA"), box=(i%square_size * image.size[0], i//square_size * image.size[1]))

target_image.save(options.output)
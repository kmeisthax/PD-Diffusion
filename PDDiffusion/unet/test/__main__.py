from PDDiffusion.unet.test import load_pretrained_pipeline_into_accelerator
from dataclasses import field
from argparse_dataclass import dataclass
from accelerate import Accelerator
import sys, torch, os.path

@dataclass
class GenerationOptions:
    output: str = field(metadata={"args": ["output"]})
    model_dir: str = field(default='pd-diffusion', metadata={"args": ["--model_dir"]})
    text_prompt: str = field(default=None, metadata={"args": ["--text_prompt"], "help": "The text prompt to use for conditional models."})
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

image = pipeline(**pipeline_args).images[0]

image.save(options.output)
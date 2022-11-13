from PDDiffusion.unet.test import load_pretrained_pipeline
from dataclasses import field
from argparse_dataclass import dataclass
from accelerate import Accelerator
import sys, torch, os.path

@dataclass
class GenerationOptions:
    output: str = field(metadata={"args": ["output"]})
    model_dir: str = field(default='pd-diffusion', metadata={"args": ["--model_dir"]})
    seed: int = None

options = GenerationOptions.parse_args(sys.argv[1:])

accelerator = Accelerator()

#TODO: What if the user has an absolute model path?
pipeline = load_pretrained_pipeline(os.path.join("output", options.model_dir), accelerator)

seed = None
if options.seed is not None:
    seed = torch.manual_seed(options.seed)

image = pipeline(generator=seed).images[0]

image.save(options.output)
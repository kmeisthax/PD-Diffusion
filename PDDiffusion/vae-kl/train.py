"""Train loop for the Variational Autoencoder with KL Loss (AutoencoderKL) model"""

from PDDiffusion.datasets.WikimediaCommons import local_wikimedia_base
from PDDiffusion.image_loader import load_dataset
from diffusers import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator, find_executable_batch_size
from dataclasses import field
from argparse_dataclass import dataclass
from tqdm.auto import tqdm
import sys, os.path, json, torch, math

@dataclass
class TrainingOptions:
    output_dir: str = field(default='pd-diffusion-vae', metadata={"args": ["output_dir"], "help": "Where to store the trained model"})

    #Training options
    num_epochs: int = field(default = 50, metadata={"args": ["--num_epochs"], "help": "How many epochs to train for"})
    train_batch_size: int = field(default = 16, metadata={"args": ["--train_batch_size"], "help": "How many images to train per step. Will be reduced automatically if this exceeds the memory size of your GPU."})
    mixed_precision: str = field(default = 'no', metadata={"args": ["--mixed_precision"], "choices": ["no", "fp16", "bf16"], "help": "What mixed-precision mode to use, if any."})
    gradient_accumulation_steps: int = field(default=1, metadata={"args": ["--gradient_accumulation_steps"]})
    learning_rate: float = field(default=1e-4, metadata={"args": ["--learning_rate"]})
    lr_warmup_steps: int = field(default=500, metadata={"args": ["--lr_warmup_steps"]})

    #Data load strategy
    pin_data_in_memory: bool = field(default=False, metadata={"args": ["--pin_data_in_memory"], "help": "Force dataset to remain in CPU memory"})
    data_load_workers: int = field(default=0, metadata={"args": ["--data_load_workers"], "help": "Number of workers to load data with"})
    image_limit: float = field(default=None, metadata={"args": ["--image_limit"], "help": "Number of images per batch to train"})

    #VAE & Latent space configuration
    block_out_channels: str = field(default="128,256,512,512", metadata={"args": ["--block_out_channels"], "help": "The number of channels for each coder block. Commas add multiple blocks."})
    layers_per_block: int = field(default=2, metadata={"args": ["--layers_per_block"], "help": "Number of hidden layers per coder block."})
    act_fn: str = field(default="silu", metadata={"args": ["--act_fn"], "help": "Activation function to use"})
    latent_channels: int = field(default=4, metadata={"args": ["--latent_channels"], "help": "Number of channels in the latent space"})

    #AdamW optimizer params
    adam_beta1:float = field(default=0.95, metadata={"args": ["--adam_beta1"]})
    adam_beta2:float = field(default=0.999, metadata={"args": ["--adam_beta2"]})
    adam_weight_decay:float = field(default=1e-6, metadata={"args": ["--adam_weight_decay"]})
    adam_epsilon:float = field(default=1e-08, metadata={"args": ["--adam_epsilon"]})

    def as_vae_kwargs(self):
        channels = tuple([int(channel.strip()) for channel in self.block_out_channels.split(",")])

        return {
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": tuple(["DownEncoderBlock2D"] * len(channels)),
            "up_block_types": tuple(["UpDecoderBlock2D"] * len(channels)),
            "block_out_channels": channels,
            "layers_per_block": self.layers_per_block,
            "act_fn": self.act_fn,
            "latent_channels": self.latent_channels
        }
    
    def wanted_image_size(self):
        channels = [int(channel.strip()) for channel in self.block_out_channels.split(",")]

        return max(channels)

config = TrainingOptions.parse_args(sys.argv[1:])
model_dir = os.path.join("output", config.output_dir)

if not os.path.exists("output"):
    os.makedirs("output")

accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps, 
    log_with="tensorboard",
    logging_dir = os.path.join(model_dir, "logs")
)

if accelerator.is_main_process:
    accelerator.init_trackers("train_vae")

dataset = load_dataset(config.wanted_image_size())

@find_executable_batch_size(starting_batch_size = config.train_batch_size)
def train(batch_size):
    accelerator.free_memory()

    train_dataloader = torch.utils.data.DataLoader(dataset, 
        pin_memory=config.pin_data_in_memory,
        num_workers = config.data_load_workers,
        batch_size=batch_size,
        shuffle=True)

    if os.path.exists(os.path.join(model_dir, "progress.json")):
        model = AutoencoderKL.from_pretrained(model_dir)

        with open(os.path.join(model_dir, "progress.json"), 'r') as progress_file:
            progress = json.load(progress_file)
    else:
        model = AutoencoderKL(**config.as_vae_kwargs())
        progress = {"last_epoch": -1}
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon
    )
    
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
    
    last_step = -1
    if progress["last_epoch"] != -1:
        print("Restarting after epoch {}".format(progress["last_epoch"]))
        last_step = len(train_dataloader) * progress["last_epoch"]
        
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
        last_epoch=last_step #Since we step per batch, not per epoch
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
        
    global_step = 0

    #Calculate how much batch data to toss in order to meet our image limit
    num_batch_to_skip = len(train_dataloader)
    if config.image_limit is not None:
        num_batch_to_skip = min(num_batch_to_skip, math.ceil(config.image_limit / batch_size))
    
    for epoch in range(progress["last_epoch"] + 1, config.num_epochs):
        progress_bar = tqdm(total=num_batch_to_skip, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for batch_number, batch in enumerate(train_dataloader):
            if batch_number > num_batch_to_skip:
                break

            with accelerator.accumulate(model):
                sampled_and_recompressed = model(batch["image"]).sample
                loss = torch.nn.functional.mse_loss(sampled_and_recompressed, batch["image"])
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if str(model.device).startswith("cuda:"):
                logs["mem"] = torch.cuda.memory_allocated()
                logs["max"] = torch.cuda.max_memory_allocated()
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        if accelerator.is_main_process:
            model.save_pretrained(model_dir)

            with open(os.path.join(model_dir, "progress.json"), "w") as progress_file:
                progress["last_epoch"] = epoch
                json.dump(progress, progress_file)

train()
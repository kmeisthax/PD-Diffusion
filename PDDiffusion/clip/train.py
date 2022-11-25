"""CLIP model trainer

Tokenizer must be pretrained first, see tokenize.py"""

from PDDiffusion.clip import load_model_and_progress, load_processor, load_dataset_with_processor

from dataclasses import field
from argparse_dataclass import dataclass
from transformers import CLIPConfig, PreTrainedTokenizerBase, Trainer, TrainingArguments
from accelerate import Accelerator, find_executable_batch_size
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

import sys, os.path, json, torch

@dataclass
class CLIPTrainingOptions:
    output_dir: str = field(default='pd-diffusion-clip', metadata={"args": ["output_dir"], "help": "Where to store the trained model"})

    #Training loop parameters
    train_batch_size: int = field(default = 16, metadata={"args": ["--train_batch_size"], "help": "How many images to train per step. Will be reduced automatically if this exceeds the memory size of your GPU."})
    num_epochs: int = field(default = 50, metadata={"args": ["--num_epochs"], "help": "How many epochs to train for. You can only checkpoint the model at an epoch boundary."})
    save_model_epochs: int = field(default=1, metadata={"args": ["--save_model_epochs"], "help": "How many epochs to wait in between saving model checkpoints."})
    mixed_precision: str = field(default = 'no', metadata={"args": ["--mixed_precision"], "choices": ["no", "fp16", "bf16"], "help": "What mixed-precision mode to use, if any."})

    #CLIP parameters
    projection_dim: int = field(default = 512, metadata={"args": ["--projection_dim"], "help": "Dimensionality of the text and vision projection layers."})
    logit_scale: int = field(default = 2.6592, metadata={"args": ["--logit_scale"]})

    #Text-specific parameters
    text_hidden_size: int = field(default=512, metadata={"args": ["--text_hidden_size"], "help": "Dimensionality of the encoder layers and pooler layer in text model"})
    text_intermediate_size: int = field(default=2048, metadata={"args": ["--text_intermediate_size"], "help": "Dimensionality of the feed-forward layer in text model"})
    text_num_hidden_layers: int = field(default=12, metadata={"args": ["--text_num_hidden_layers"], "help": "Number of hidden layers in text model"})
    text_num_attention_heads: int = field(default=8, metadata={"args": ["--text_num_attention_heads"], "help": "Number of attention heads in text model"})
    text_max_position_embeddings: int = field(default=77, metadata={"args": ["--text_max_position_embeddings"], "help": "Maximum length of labels supported by the text model"})
    text_hidden_act: str = field(default="quick_gelu", metadata={"args": ["--text_hidden_act"], "help": "Activation function used for text model encoder and pooler layers", "choices": ["gelu", "relu", "selu", "gelu_new", "quick_gelu"]})
    text_layer_norm_eps: float = field(default=1e-5, metadata={"args": ["--text_layer_norm_eps"], "help": "The epsilon used by the layer normalization layers in the text model"})
    text_attention_dropout: float = field(default=0.0, metadata={"args": ["--text_attention_dropout"], "help": "The dropout ratio for attention probabilities in the text model"})
    text_dropout: float = field(default=0.0, metadata={"args": ["--text_dropout"], "help": "The dropout ratio for all fully-connected layers in the text model"})
    text_initializer_range: float = field(default=0.02, metadata={"args": ["--text_initializer_range"], "help": "The standard deviation of the truncated normal initializer for initializing all weight matrices in the text model"})
    text_initializer_factor: float = field(default=1, metadata={"args": ["--text_initializer_factor"], "help": "A factor for intializing all weight matricies in the text model, should be left at 1"})

    #Vision-specific parameters
    vision_hidden_size: int = field(default=768, metadata={"args": ["--vision_hidden_size"], "help": "Dimensionality of the encoder layers and pooler layer in vision model"})
    vision_intermediate_size: int = field(default=3072, metadata={"args": ["--vision_intermediate_size"], "help": "Dimensionality of the feed-forward layer in vision model"})
    vision_num_hidden_layers: int = field(default=12, metadata={"args": ["--vision_num_hidden_layers"], "help": "Number of hidden layers in vision model"})
    vision_num_attention_heads: int = field(default=12, metadata={"args": ["--vision_num_attention_heads"], "help": "Number of attention heads in vision model"})
    vision_image_size: int = field(default=224, metadata={"args": ["--vision_image_size"], "help": "The size of each image"})
    vision_patch_size: int = field(default=32, metadata={"args": ["--vision_patch_size"], "help": "The size of each patch"})
    vision_hidden_act: str = field(default="quick_gelu", metadata={"args": ["--vision_hidden_act"], "help": "Activation function used for vision model encoder and pooler layers", "choices": ["gelu", "relu", "selu", "gelu_new", "quick_gelu"]})
    vision_attention_dropout: float = field(default=0.0, metadata={"args": ["--vision_attention_dropout"], "help": "The dropout ratio for attention probabilities in the vision model"})
    vision_dropout: float = field(default=0.0, metadata={"args": ["--vision_dropout"], "help": "The dropout ratio for all fully-connected layers in the vision model"})
    vision_initializer_range: float = field(default=0.02, metadata={"args": ["--vision_initializer_range"], "help": "The standard deviation of the truncated normal initializer for initializing all weight matrices in the vision model"})
    vision_initializer_factor: float = field(default=1, metadata={"args": ["--vision_initializer_factor"], "help": "A factor for intializing all weight matricies in the vision model, should be left at 1"})

    #AdamW optimizer & learning schedule parameters
    learning_rate: float = field(default=1e-4, metadata={"args": ["--learning_rate"]})
    lr_warmup_steps: int = field(default=500, metadata={"args": ["--lr_warmup_steps"]})
    adam_beta1:float = field(default=0.95, metadata={"args": ["--adam_beta1"]})
    adam_beta2:float = field(default=0.999, metadata={"args": ["--adam_beta2"]})
    adam_weight_decay:float = field(default=1e-6, metadata={"args": ["--adam_weight_decay"]})
    adam_epsilon:float = field(default=1e-08, metadata={"args": ["--adam_epsilon"]})

    def as_clip_config(self, tokenizer: PreTrainedTokenizerBase):
        """Convert command-line parameters into CLIP config"""
        return CLIPConfig(
            projection_dim=self.projection_dim,
            logit_scale_init_value=self.logit_scale,
            text_config_dict={
                "vocab_size": len(tokenizer.get_vocab()),
                "hidden_size": self.text_hidden_size,
                "intermediate_size": self.text_intermediate_size,
                "num_hidden_layers": self.text_num_hidden_layers,
                "num_attention_heads": self.text_num_attention_heads,
                "max_position_embeddings": self.text_max_position_embeddings,
                "hidden_act": self.text_hidden_act,
                "layer_norm_eps": self.text_layer_norm_eps,
                "attention_dropout": self.text_attention_dropout,
                "dropout": self.text_dropout,
                "initializer_range": self.text_initializer_range,
                "initializer_factor": self.text_initializer_factor,
            },
            vision_config_dict={
                "hidden_size": self.vision_hidden_size,
                "intermediate_size": self.vision_intermediate_size,
                "num_hidden_layers": self.vision_num_hidden_layers,
                "num_attention_heads": self.vision_num_attention_heads,
                "image_size": self.vision_image_size,
                "patch_size": self.vision_patch_size,
                "hidden_act": self.vision_hidden_act,
                "attention_dropout": self.vision_attention_dropout,
                "dropout": self.vision_dropout,
                "initializer_range": self.vision_initializer_range,
                "initializer_factor": self.vision_initializer_factor,
            }
        )

config = CLIPTrainingOptions.parse_args(sys.argv[1:])
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    log_with="tensorboard",
    logging_dir=os.path.join("output", config.output_dir, "logs")
)

if accelerator.is_main_process:
    accelerator.init_trackers("train_example")

processor = load_processor(config)
dataset = load_dataset_with_processor(config.vision_image_size, processor)
(model, progress) = load_model_and_progress(config, processor)

def collate_fn(examples):
    pixel_values = torch.stack([example["image"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=os.path.join("output", config.output_dir),
        do_train=True, 

        # ABSOLUTELY NOT. If you enable this, Trainer will DELETE your entire
        # dataset, and then throw an inscrutible error about invalid keys. Why?
        # Because we had the AUDACITY to use Dataset transforms to add image
        # columns to the dataset!
        remove_unused_columns=False),
    train_dataset=dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer
)
result = trainer.train()

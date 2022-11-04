import dataclasses

@dataclasses.dataclass
class SampleConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500

    #Generate an image every n epochs.
    #Image generation is processor-intensive, so this number should be relatively high.
    #Images will also be generated on the last epoch in the current run, no matter what.
    save_image_epochs = 999

    #Save the model every n epochs.
    #This is cheap, so I recommend doing it every epoch.
    #Training will resume from a saved model.
    save_model_epochs = 1
    mixed_precision = 'no'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'pd-diffusion'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

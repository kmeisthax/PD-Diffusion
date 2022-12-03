from PDDiffusion.unet.train import TrainingOptions, load_model_and_progress, evaluate, load_condition_model_and_processor, load_dataset_with_condition, create_model_pipeline

import os.path, torch, json, sys

from accelerate import Accelerator, find_executable_batch_size
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.hub_utils import init_git_repo, push_to_hub
from tqdm.auto import tqdm
import torch.nn.functional as F
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

config = TrainingOptions.parse_args(sys.argv[1:])
config.dataset_name = "pd-diffusion-wikimedia"

if not os.path.exists("output"):
    os.makedirs("output")

def train_loop(config):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        logging_dir=os.path.join("output", config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo = init_git_repo(config, at_init=True)
        accelerator.init_trackers("train_example")
    
    (dataset, cond_model_config) = load_dataset_with_condition(config, accelerator)

    @find_executable_batch_size(starting_batch_size=config.train_batch_size)
    def inner_training_loop(batch_size):
        nonlocal accelerator
        accelerator.free_memory()

        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        (model, progress) = load_model_and_progress(config, conditional_model_config=cond_model_config)

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

        noise_scheduler = DDPMScheduler(num_train_timesteps=config.ddpm_train_timesteps, beta_schedule=config.ddpm_beta_schedule)
        
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the 
        # objects in the same order you gave them to the prepare method.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        
        global_step = 0

        # Now you train the model
        for epoch in range(progress["last_epoch"] + 1, config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch['image']
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                with accelerator.accumulate(model):
                    parameters = {
                        "return_dict": False
                    }
                    if config.conditioned_on is not None:
                        # The stack operation unflips the dataset CLIP vectors.
                        # For some reason, batching turns 16 512-wide CLIP vectors
                        # into one 512x16 (ish) vector.
                        # The unsqueeze adds a dimension because Cross Attention
                        # blocks support multiple condition inputs per batch.
                        parameters["encoder_hidden_states"] = torch.stack(batch["condition"], 1).unsqueeze(1)

                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps, **parameters)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                with torch.no_grad():
                    pipeline = create_model_pipeline(config, accelerator, model, noise_scheduler)

                    if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                        evaluate(config, epoch, pipeline)

                    if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                        if config.push_to_hub:
                            push_to_hub(config, pipeline, repo, commit_message=f"Epoch {epoch}", blocking=True)
                        else:
                            pipeline.save_pretrained(os.path.join("output", config.output_dir))

                            with open(os.path.join("output", config.output_dir, "progress.json"), "w") as progress_file:
                                progress["last_epoch"] = epoch
                                json.dump(progress, progress_file)
    
    inner_training_loop()

train_loop(config)
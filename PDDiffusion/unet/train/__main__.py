from PDDiffusion.unet.train import TrainingOptions, load_model_and_progress, evaluate, load_condition_model_and_processor, load_dataset_with_condition, create_model_pipeline

import os.path, torch, json, sys, math

from accelerate import Accelerator, find_executable_batch_size
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import torch.nn.functional as F
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

config = TrainingOptions.parse_args(sys.argv[1:])

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
        accelerator.init_trackers("train_example")
    
    (dataset, cond_model_config) = load_dataset_with_condition(config, accelerator)

    @find_executable_batch_size(starting_batch_size=config.train_batch_size)
    def inner_training_loop(batch_size):
        nonlocal accelerator
        accelerator.free_memory()

        train_dataloader = torch.utils.data.DataLoader(dataset,
            pin_memory=config.pin_data_in_memory,
            num_workers = config.data_load_workers,
            batch_size=batch_size,
            shuffle=True)

        (model, progress) = load_model_and_progress(config, conditional_model_config=cond_model_config)

        if config.gradient_checkpointing:
            model.enable_gradient_checkpointing()

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

        #Calculate how much batch data to toss in order to meet our image limit
        num_batch_to_skip = len(train_dataloader)
        if config.image_limit is not None:
            num_batch_to_skip = min(num_batch_to_skip, math.ceil(config.image_limit / batch_size))

        # Now you train the model
        for epoch in range(progress["last_epoch"] + 1, config.num_epochs):
            progress_bar = tqdm(total=num_batch_to_skip, disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for batch_number, batch in enumerate(train_dataloader):
                if batch_number > num_batch_to_skip:
                    break

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
                        # For some reason, batching turns a list of 16 8x512
                        # CLIP tensors into a list of 8 lists of 512 16-wide
                        # vectors. This requires some transpositions in order
                        # to work as intended.
                        conditions = torch.stack([torch.stack(augment) for augment in batch["conditions"]]).transpose(2, 0).transpose(1, 2)

                        # We need to shuffle our augments around. We also have
                        # to provide an extra dimension for the multiple heads
                        # of our cross-attention block. So let's just do both
                        # at the same time.
                        index = (torch.rand(conditions.shape[0], model.config.attention_head_dim) * conditions.shape[1]) \
                            .type(torch.int64).unsqueeze(2) \
                            .repeat(1, 1, conditions.shape[2]) \
                            .to(conditions.device)
                        conditions = torch.gather(conditions, 1, index)

                        if config.mixed_precision == "no":
                            #bonus points: fp32 gets magically turned into fp64
                            conditions = conditions.type(torch.float32)
                        
                        parameters["encoder_hidden_states"] = conditions

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
                if str(model.device).startswith("cuda:"):
                    logs["mem"] = torch.cuda.memory_allocated()
                    logs["max"] = torch.cuda.max_memory_allocated()
                
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                with torch.no_grad():
                    pipeline = create_model_pipeline(config, accelerator, model, noise_scheduler)

                    if config.save_model_epochs <= config.num_epochs and ((epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1):
                        pipeline.save_pretrained(os.path.join("output", config.output_dir))

                        with open(os.path.join("output", config.output_dir, "progress.json"), "w") as progress_file:
                            progress["last_epoch"] = epoch
                            json.dump(progress, progress_file)

                    if config.save_image_epochs <= config.num_epochs and ((epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1):
                        evaluate(config, epoch, pipeline)
    
    inner_training_loop()

train_loop(config)
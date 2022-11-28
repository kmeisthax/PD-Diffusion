# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for PDDiffusion. All the conditional pipelines in Diffusers expect
# to have a latent space representation.

from typing import Optional, Tuple, Union

import torch

from diffusers.configuration_utils import FrozenDict
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import deprecate

class DDPMConditionalPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DConditionModel`]): U-Net architecture to denoise the encoded image.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. DDPMConditionalPipeline uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, text_encoder, tokenizer, scheduler):
        super().__init__()
        self.register_modules(unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler)
    
    def _encode_prompt(self, prompt:str, device, batch_size:int=1):
        """Encode a text2img prompt.
        
        Args:
            prompt (`str`)
                prompt to be encoded
            device: (`torch.device`)
                device to encode on
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
        """
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(device), attention_mask=attention_mask)[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, batch_size, 1)
        text_embeddings = text_embeddings.view(bs_embed * batch_size, seq_len, -1)

        return text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            prompt (`str`):
                The text prompt to generate.
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        message = (
            "Please make sure to instantiate your scheduler with `prediction_type` instead. E.g. `scheduler ="
            " DDPMScheduler.from_pretrained(<model_id>, prediction_type='epsilon')`."
        )
        predict_epsilon = deprecate("predict_epsilon", "0.10.0", message, take_from=kwargs)

        if predict_epsilon is not None:
            new_config = dict(self.scheduler.config)
            new_config["prediction_type"] = "epsilon" if predict_epsilon else "sample"
            self.scheduler._internal_dict = FrozenDict(new_config)

        if generator is not None and generator.device.type != self.unet.device.type and self.unet.device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.unet.device}`, so the `generator` will be ignored. "
                f'Please use `torch.Generator(device="{self.unet.device}")` instead.'
            )
            deprecate(
                "generator.device == 'cpu'",
                "0.11.0",
                message,
            )
            generator = None
        
        text_embeddings = self._encode_prompt(prompt, self.unet.device, batch_size)

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.sample_size, int):
            image_shape = (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
        else:
            image_shape = (batch_size, self.unet.in_channels, *self.unet.sample_size)

        if self.unet.device.type == "mps":
            # randn does not work reproducibly on mps
            image = torch.randn(image_shape, generator=generator)
            image = image.to(self.unet.device)
        else:
            image = torch.randn(image_shape, generator=generator, device=self.unet.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t, encoder_hidden_states=text_embeddings).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler
from tqdm.auto import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_unclip_img2img import StableUnCLIPImageNormalizer
import torch.nn as nn
from typing import Optional
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_timestep_embedding

class StableDiffusionUnclip(nn.Module):
    def __init__(self, device='cuda', model_id = "stabilityai/stable-diffusion-2-1-unclip"):
        super().__init__()
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True)
        self.feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")
        # self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, subfolder="image_encoder")
        self.image_noising_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="image_noising_scheduler")
        self.image_normalizer = StableUnCLIPImageNormalizer.from_pretrained(model_id, subfolder="image_normalizer")
        self.scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", use_safetensors=True
        )

        self.vae.to(self.device)
        self.image_encoder.to(self.device)
        self.unet.to(self.device)
        # self.text_encoder.to(self.device)

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def encode_image(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        image = image.to(self.device)
        image_embedding = self.image_encoder(image).image_embeds
        image_embedding = self.noise_image_embeddings(image_embedding, noise_level=5)
        negative_prompt_embeds = torch.zeros_like(image_embedding)
        image_embedding = torch.cat([negative_prompt_embeds, image_embedding])
        return image_embedding
    
    def forward(self, image, prompt, height=512, width=512, num_inference_steps=25, guidance_scale=7.5, seed=0):
        image_embeds = self.encode_image(image)
        # batch_size = len(prompt)
        # if len(prompt) > self.tokenizer.model_max_length:
        #     print("Warning: The prompt length is larger than the tokenizer's max length. It will be truncated.")
        # text_input = self.tokenizer(
        #     prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        # )
        # with torch.no_grad():
        #     text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        # max_length = text_input.input_ids.shape[-1]
        # uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        # uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (1, self.unet.config.in_channels, height // 8, width // 8),
            device=self.device,
        )

        text_embeddings = torch.zeros((2,77,1024)).to(self.device)
        latents = latents * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            empty_text = torch.zeros_like(image_embeds)
            empty_text = empty_text.unsqueeze(1)  # Now has shape (batch_size, 1, hidden_dim)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,class_labels=image_embeds).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def noise_image_embeddings(
        self,
        image_embeds: torch.Tensor,
        noise_level: int,
        noise: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Add noise to the image embeddings. The amount of noise is controlled by a `noise_level` input. A higher
        `noise_level` increases the variance in the final un-noised images.

        The noise is applied in two ways:
        1. A noise schedule is applied directly to the embeddings.
        2. A vector of sinusoidal time embeddings are appended to the output.

        In both cases, the amount of noise is controlled by the same `noise_level`.

        The embeddings are normalized before the noise is applied and un-normalized after the noise is applied.
        """
        if noise is None:
            noise = randn_tensor(
                image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype
            )

        noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)

        self.image_normalizer.to(image_embeds.device)
        image_embeds = self.image_normalizer.scale(image_embeds)

        image_embeds = self.image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)

        image_embeds = self.image_normalizer.unscale(image_embeds)

        noise_level = get_timestep_embedding(
            timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
        )

        # `get_timestep_embeddings` does not contain any weights and will always return f32 tensors,
        # but we might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        noise_level = noise_level.to(image_embeds.dtype)

        image_embeds = torch.cat((image_embeds, noise_level), 1)

        return image_embeds

if __name__ == "__main__":
    model = StableDiffusionUnclip()
    image = Image.open("test2.png")
    with torch.no_grad():
        output = model(image=image, prompt=["an asian women"])
    output  = (output.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    output = Image.fromarray(output)
    output.show()
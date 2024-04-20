from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm.auto import tqdm
class StableDiffusion:
    def __init__(self, device='cuda',freeze_textEncoder = False):
        self.device = device
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae", use_safetensors=True)
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="text_encoder", use_safetensors=True
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="unet", use_safetensors=True
        )
        self.scheduler = PNDMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler")
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        if freeze_textEncoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        self.freeze_textEncoder = freeze_textEncoder

    def forward(self, prompt, height=512, width=512, num_inference_steps=15, guidance_scale=7.5, seed=0):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        batch_size = len(prompt)
        if len(prompt) > self.tokenizer.model_max_length:
            print("Warning: The prompt length is larger than the tokenizer's max length. It will be truncated.")
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        if self.freeze_textEncoder:
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        else:
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
        )

        latents = latents * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        return image

if __name__ == "__main__":
    sd = StableDiffusion()
    prompt = ["A complete picture of a organge adidas T-shirt with a white background."]
    image = sd.forward(prompt)
    image.save("generated_image.png")
        

# vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae", use_safetensors=True)
# tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")
# text_encoder = CLIPTextModel.from_pretrained(
#     "stabilityai/stable-diffusion-2-1", subfolder="text_encoder", use_safetensors=True
# )
# unet = UNet2DConditionModel.from_pretrained(
#     "stabilityai/stable-diffusion-2-1", subfolder="unet", use_safetensors=True
# )

# scheduler = PNDMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler")

# torch_device = "cuda"
# vae.to(torch_device)
# text_encoder.to(torch_device)
# unet.to(torch_device)

# prompt = ["A complete picture of a organge adidas T-shirt with a white background."]
# height = 512  # default height of Stable Diffusion
# width = 512  # default width of Stable Diffusion
# num_inference_steps = 25  # Number of denoising steps
# guidance_scale = 7.5  # Scale for classifier-free guidance
# generator = torch.Generator(device=torch_device).manual_seed(0)  # Seed generator to create the initial latent noise
# batch_size = len(prompt)

# text_input = tokenizer(
#     prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
# )

# with torch.no_grad():
#     text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

# max_length = text_input.input_ids.shape[-1]
# uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
# uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

# text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# latents = torch.randn(
#     (batch_size, unet.config.in_channels, height // 8, width // 8),
#     generator=generator,
#     device=torch_device,
# )

# latents = latents * scheduler.init_noise_sigma

# scheduler.set_timesteps(num_inference_steps)

# for t in tqdm(scheduler.timesteps):
#     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
#     latent_model_input = torch.cat([latents] * 2)

#     latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#     # predict the noise residual
#     with torch.no_grad():
#         noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#     # perform guidance
#     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

#     # compute the previous noisy sample x_t -> x_t-1
#     latents = scheduler.step(noise_pred, t, latents).prev_sample

# # scale and decode the image latents with vae
# latents = 1 / 0.18215 * latents
# with torch.no_grad():
#     image = vae.decode(latents).sample

# image = (image / 2 + 0.5).clamp(0, 1).squeeze()
# image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
# image = Image.fromarray(image)
# image.save("generated_image.png")
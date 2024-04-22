from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm.auto import tqdm
import torch.nn as nn
class StableDiffusion(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae", use_safetensors=True)
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="text_encoder", use_safetensors=True
        )
        
        #self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        #self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')

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

    def forward(self, prompt, height=512, width=512, num_inference_steps=25, guidance_scale=7.5, seed=0, tokens=None):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        batch_size = len(prompt)
        if len(prompt) > self.tokenizer.model_max_length:
            print("Warning: The prompt length is larger than the tokenizer's max length. It will be truncated.")
        if tokens is not None:
            text_embeddings = tokens
            max_length = tokens.shape[1]
        else:
            text_input = self.tokenizer(
                prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
            )
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
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        return image

if __name__ == "__main__":
    sd = StableDiffusion()
    prompt = ["A front view of of a organge T-shirt with a pure white background."]
    sd.requires_grad_(False) # Note: Running out of memory if require grad

    output = sd.forward(prompt)
    # loss = nn.MSELoss()
    # target = torch.zeros_like(output)
    # loss = loss(output, target)
    # loss.backward()
    output  = (output.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    output = Image.fromarray(output)
    output.show()

    # # Check gradients
    # for name, param in sd.named_parameters():
    #     if param.requires_grad:
    #         if param.grad is not None:
    #             print(f"Gradient for parameter {name}:")

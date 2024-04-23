from load_data import get_sample
from PIL import Image
from stablediffusion import StableDiffusion
import torch
from vision_encoder import vision_encoder
from llm import ClipCaptionModel, generate_text_with_gumbel_softmax
import os
from transformers import GPT2Tokenizer, AdamW, CLIPProcessor, CLIPModel
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
from mapping import get_gpt2_logits, map_prompt_to_clip, recover_text_from_one_hot
from custom_clip import CustomCLIPTextModel 
from torchvision import transforms
import random
import gc

def set_seed(seed):
    """Set all seeds to make computations deterministic where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Seed all GPUs for multi-GPU setups
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Enable deterministic behavior in PyTorch (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

our_clip = CustomCLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="text_encoder")
sd = StableDiffusion()
encoder = vision_encoder()
fclip = FashionCLIP('fashion-clip')
CPU = torch.device('cpu')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

clip_vit = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_vit.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

for param in clip_vit.parameters():
    param.requires_grad = False

for param in sd.parameters():
    param.requires_grad = False

prefix_length = 10
current_directory = os.getcwd()
save_path = os.path.join(current_directory, "saved_models")
os.makedirs(save_path, exist_ok=True)

model = ClipCaptionModel(prefix_length)
model_path = os.path.join(save_path, 'fashion.pt')
model.load_state_dict(torch.load(model_path, map_location=CPU)) 
model = model.eval() 
model = model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

for name, parameter in model.named_parameters():
    param.requires_grad = True  # Ensure all parameters are trainable



# Set up the optimizer with only those parameters that need to be updated
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
checkpoint_dir = './model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
# Example loop for processing and training
num_training_steps = 100000
for step in range(num_training_steps):
    pil_image, row = get_sample()

    # Encode images and prepare embeddings
    image_embeddings = fclip.encode_images([pil_image], batch_size=1)
    image_embeddings /= np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
    image_embeddings = torch.tensor(image_embeddings, device=device, dtype=torch.float32)

    # Ensure gradients are computed for embeddings part
    image_embeddings.requires_grad_(True)

    prefix_embed = model.clip_project(image_embeddings).reshape(1, prefix_length, -1)
    set_seed(42)
    out, soft_tokens_list = generate_text_with_gumbel_softmax(model, tokenizer, embed=prefix_embed, temperature=0.1)
    
    print(f"Prompt: {out}")

    combined_softmax_outputs = torch.cat(soft_tokens_list, dim=0)
    tokens, attention_mask, input_ids, out_prompt = map_prompt_to_clip(combined_softmax_outputs)
    token_embeddings = our_clip(inputs_embeds=tokens.unsqueeze(0), attention_mask=attention_mask, input_ids = input_ids.unsqueeze(0))[0]
    sd.requires_grad_(False)
    output = sd(token_embeddings.to(device))
    #show_image(output)
    loss = encoder.calc_loss_vector(output, pil_image)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Step {step} Loss: {loss.item()}")
    if step % 100 == 0 and step != 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'loss': loss.item()
        }, checkpoint_path)
        print(f"Checkpoint saved at step {step} to {checkpoint_path}")
    
    del pil_image, row, image_embeddings, prefix_embed, out, combined_softmax_outputs, tokens, attention_mask, input_ids, token_embeddings, output, loss
    torch.cuda.empty_cache()  # Clear CUDA cache
    gc.collect()
    
    print(f"Step {step}: Gradients for LLM (model)")

from load_data import get_sample, sample_entry
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
import torchvision.transforms as T

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

our_clip = CustomCLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
sd = StableDiffusion()
sd.load_lora_weights("full-lora/", "lora2")
encoder = vision_encoder()
fclip = FashionCLIP('fashion-clip')
CPU = torch.device('cpu')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


for param in sd.parameters():
    param.requires_grad = False

for param in our_clip.parameters():
    param.requires_grad = False

prefix_length = 10
current_directory = os.getcwd()
save_path = os.path.join(current_directory, "saved_models")
os.makedirs(save_path, exist_ok=True)

model = ClipCaptionModel(prefix_length)
model_path = os.path.join(save_path, 'coco_prefix-078.pt')
pretrained_state_dict = torch.load(model_path, map_location='cpu')
model_keys = model.state_dict().keys()
filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_keys}
model.load_state_dict(filtered_state_dict, strict=False)
model = model.eval() 

model = model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def show_image(output):
    output_image = (output.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    output_image = Image.fromarray(output_image)
    output_image.show()

for name, parameter in model.named_parameters():
    parameter.requires_grad = True  # Ensure all parameters are trainable

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# Example loop for processing and training
num_training_steps = 10000

checkpoint_dir = './model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

for step in range(num_training_steps):
    try:
        pil_image, row = get_sample()
        pil_image.show()
        image_embeddings = fclip.encode_images([pil_image], batch_size=1)
        image_embeddings /= np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        image_embeddings = torch.tensor(image_embeddings, device=device, dtype=torch.float32)

        prefix_embed = model.clip_project(image_embeddings).reshape(1, prefix_length, -1)
    
        out, soft_tokens_list = generate_text_with_gumbel_softmax(model, tokenizer, embed=prefix_embed, temperature=0.1)
        print(out)
        
        combined_softmax_outputs = torch.cat(soft_tokens_list, dim=0)
        tokens, attention_mask, input_ids, out_prompt = map_prompt_to_clip(combined_softmax_outputs)
        token_embeddings = our_clip(inputs_embeds=tokens.unsqueeze(0), attention_mask=attention_mask, input_ids=input_ids.unsqueeze(0))[0]
        output = sd(token_embeddings.to(device))
        show_image(output)
        loss = encoder.calc_loss_vector(output, pil_image)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'loss': loss.item()
            }, checkpoint_path)
            print(f"Checkpoint saved at step {step} to {checkpoint_path}")
        
        print(f"Step {step}: Gradients for LLM (model)")

    finally:
        # Delete variables if they exist
        vars_to_delete = ['image_embeddings', 'prefix_embed', 'out', 'soft_tokens_list', 
                          'combined_softmax_outputs', 'tokens', 'attention_mask', 
                          'input_ids', 'token_embeddings', 'output', 'loss']
        for var in vars_to_delete:
            globals_var = globals().get(var)
            locals_var = locals().get(var)
            if globals_var is not None:
                del globals()[var]
            elif locals_var is not None:
                del locals()[var]

        torch.cuda.empty_cache()  # Clear CUDA cache
        gc.collect()


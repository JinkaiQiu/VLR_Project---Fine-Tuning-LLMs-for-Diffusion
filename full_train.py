import clip
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
model_path = os.path.join(save_path, 'coco_weights.pt')
pretrained_state_dict = torch.load(model_path, map_location='cpu')
model_keys = model.state_dict().keys()
filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_keys}
model.load_state_dict(filtered_state_dict, strict=False)
model = model.eval() 

model = model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

for name, parameter in model.named_parameters():
    parameter.requires_grad = True  # Ensure all parameters are trainable

clip_model_type = "ViT-B/32"
clip_model, preprocess = clip.load(clip_model_type, device="cuda", jit=False)
# Set up the optimizer with only those parameters that need to be updated
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
checkpoint_dir = './model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

def preprocess_image_tensor(img_tensor, pil=True):
    # Assuming img_tensor is a [3, 512, 512] tensor
    transform = T.Compose([
        T.Resize((224, 224)),        # Resize to the input size expected by CLIP
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP's normalization parameters
                    std=[0.26862954, 0.26130258, 0.27577711])
    ])
    if not pil:
        transform = T.Compose([
            T.Resize((224, 224)), 
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP's normalization parameters
                        std=[0.26862954, 0.26130258, 0.27577711])
        ])
    return transform(img_tensor)
# Example loop for processing and training
loss_function = torch.nn.MSELoss()
num_training_steps = 100000
for step in range(num_training_steps):    
    image, selected_caption, selected_embedding = sample_entry()
    
    process_image = preprocess(image).unsqueeze(0).to(device)
    prefix = clip_model.encode_image(process_image).float()
    embedding = prefix.to("cuda:0")
    embedding.requires_grad_(True)
    embedding = embedding.unsqueeze(0)
    prefix_embed = model.clip_project(embedding).reshape(1, prefix_length, -1)
    #set_seed(42)
    out, soft_tokens_list = generate_text_with_gumbel_softmax(model, tokenizer, embed=prefix_embed, temperature=0.1)
    print(out)
    combined_softmax_outputs = torch.cat(soft_tokens_list, dim=0)
    tokens, attention_mask, input_ids, out_prompt = map_prompt_to_clip(combined_softmax_outputs)
    token_embeddings = our_clip(inputs_embeds=tokens.unsqueeze(0), attention_mask=attention_mask, input_ids = input_ids.unsqueeze(0))[0]
    sd.requires_grad_(False)
    output = sd(token_embeddings.to(device))
    
    out1 = clip_model.encode_image(preprocess_image_tensor(output, pil=False).unsqueeze(0))
    out2 = clip_model.encode_image(preprocess_image_tensor(image, pil=True).unsqueeze(0).to("cuda:0"))
    loss = loss_function(out1,out2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Step {step} Loss: {loss.item()}")
    
    if step % 100 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'loss': loss.item()
        }, checkpoint_path)
        print(f"Checkpoint saved at step {step} to {checkpoint_path}")
    
    del image, prefix, embedding, prefix_embed, out, combined_softmax_outputs, tokens, attention_mask, input_ids, token_embeddings, output, loss
    torch.cuda.empty_cache()  # Clear CUDA cache
    gc.collect()
    
    print(f"Step {step}: Gradients for LLM (model)")

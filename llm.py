import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image
from IPython.display import Image 
from fashion_clip.fashion_clip import FashionCLIP
from load_data import get_sample
import torch.nn.functional as F
import random

fclip = FashionCLIP('fashion-clip')

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
    



N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device
#@title Model

class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))
            
set_seed(42)

# Define the Gumbel-Softmax application
def gumbel_softmax(logits, temperature=1):
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)  # Sample from Gumbel(0, 1)
    y_soft = logits + gumbels
    return F.softmax(y_soft / temperature, dim=-1)

# Function to generate text and return soft tokens
def generate_text_with_gumbel_softmax(model, tokenizer, embed, temperature=1.0, entry_length=67, device='cuda'):
    model.eval()
    generated = embed  # Start from the image-derived prefix embeddings
    soft_tokens_list = []
    
    for i in range(entry_length):
        outputs = model.gpt(inputs_embeds=generated)
        logits = outputs.logits[:, -1, :]  # Get logits for the next token
        soft_tokens = gumbel_softmax(logits, temperature=temperature)  # Apply Gumbel-Softmax
        soft_tokens_list.append(soft_tokens)
        next_token_embeds = torch.matmul(soft_tokens, model.gpt.transformer.wte.weight)  # Convert soft tokens to embeddings
        generated = torch.cat((generated, next_token_embeds.unsqueeze(1)), dim=1)  # Append to generated sequence

    # Convert soft token logits to token IDs for text generation
    final_tokens = torch.cat(soft_tokens_list, dim=0).argmax(dim=-1)
    generated_text = tokenizer.decode(final_tokens.cpu().numpy())

    return generated_text, soft_tokens_list

# Main setup and execution
if __name__ == "__main__":
    current_directory = os.getcwd()
    save_path = os.path.join(current_directory, "saved_models")
    os.makedirs(save_path, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    prefix_length = 10
    
    model = ClipCaptionModel(prefix_length)
    model_path = os.path.join(save_path, 'fashion.pt')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
    model = model.eval() 
    model = model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    fclip = FashionCLIP('fashion-clip')
    pil_image, row = get_sample(0)
    
    print(row['detail_desc'])
    
    image_embeddings = fclip.encode_images([pil_image], batch_size=1)
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
    image_embeddings = torch.tensor(image_embeddings).to(device)
        
    prefix_embed = model.clip_project(image_embeddings).reshape(1, prefix_length, -1)
    
    # Generate text and get soft tokens
    generated_text, soft_tokens = generate_text_with_gumbel_softmax(model, tokenizer, prefix_embed, temperature=0.5, device=device)
    print("Generated Text:", generated_text)
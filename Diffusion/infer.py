from diffusers import StableDiffusionPipeline
import torch
import torch
import numpy as np
import random
from safetensors import safe_open
from safetensors.torch import save_file


seed = 11  # choose any seed you want

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you're using CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",torch_dtype=torch.float16)
# pipe.load_lora_weights(model_path)
pipe.to("cuda")

# prompt =  "Dark Gray T-shirt with a pocket on the front and a round neck"
prompt = "solid dark green sweat shirt with long sleeves and a pocket on the front"
# prompt = "A tan kitchen with redwood floor"
image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
image.save("1_output_baseline.png")
def make_callback(switch_step, loras):
    def switch_callback(pipeline, step_index, timestep, callback_kwargs):
        callback_outputs = {}
        if step_index > 0 and step_index % switch_step == 0:
            for cur_lora_index, lora in enumerate(loras):
                if lora in pipeline.get_active_adapters():
                    next_lora_index = (cur_lora_index + 1) % len(loras)
                    pipeline.set_adapters(loras[next_lora_index])
                    break
        return callback_outputs
    return switch_callback

lora1_pth = "cat-lora/"
lora2_pth = "full-lora/"

# def create_interpolated_lora(lora1_path, lora2_path, interpolation_weight):

#     lora_tensors1 = {}
#     with safe_open(lora1_pth, framework="pt", device=0) as f:
#         for k in f.keys():
#             lora_tensors1[k] = f.get_tensor(k)
    
#     lora_tensor2 = {}
#     with safe_open(lora2_pth, framework="pt", device=0) as f:
#         for k in f.keys():
#             lora_tensor2[k] = f.get_tensor(k)

#     interpolated_weights = {}
#     for key in lora_tensors1.keys():
#         interpolated_weights[key] = lora_tensors1[key] * interpolation_weight + lora_tensor2[key] * (1 - interpolation_weight)
    
#     save_file(interpolated_weights, "interp.safetensors")

#     return

# dcreated_lora = create_interpolated_lora(lora1_pth, lora2_pth, 0.5)

pipe.load_lora_weights(lora1_pth,adapter_name="lora1")
pipe.load_lora_weights(lora2_pth,adapter_name="lora2")

curLoras = ["lora1", "lora2"]
pipe.set_adapters([curLoras[0]])
switch_callback = make_callback(switch_step=10, loras=curLoras)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you're using CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5, callback_on_step_end=switch_callback).images[0]
image.save("1_mixed_lora.png")
pipe.unload_lora_weights()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you're using CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_path = lora1_pth
pipe.load_lora_weights(model_path)
image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
image.save("1_category.png")
pipe.unload_lora_weights()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you're using CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_path = lora2_pth
pipe.load_lora_weights(model_path)
image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
image.save("1_full.png")
pipe.unload_lora_weights()

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you're using CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_path = "interp-lora/"
pipe.load_lora_weights(model_path)
image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
image.save("1_interp.png")
pipe.unload_lora_weights()
import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
import pandas as pd

fclip = FashionCLIP('fashion-clip')


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    out_path = f"./data/coco/oscar_split_{clip_model_type}_train.pkl"
    #clip_model_name = clip_model_type.replace('/', '_')
    #clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    #with open('./data/coco/annotations/train_caption.json', 'r') as f:
    #    data = json.load(f)
    subset = pd.read_csv("subset_data.csv")
    print("%0d captions loaded from json " % len(subset))
    all_embeddings = []
    all_captions = []
    for i, row in subset.iterrows():
        if i > 5:
            break
        img_path = f"./Images/0{str(row['article_id'])[:2]}/0{str(row['article_id'])}.jpg"
        pil_image = Image.open(img_path)
        image_embeddings = fclip.encode_images([pil_image], batch_size=1)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        prefix = torch.tensor(image_embeddings).to(device)
        all_embeddings.append(prefix)
        all_captions.append(row['detail_desc'])
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
    #for i in tqdm(range(len(data))):
    #    d = data[i]
    #    img_id = d["image_id"]
    #    filename = f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
    #    if not os.path.isfile(filename):
    #        filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
    #    pil_image = io.imread(filename)
    #    image_embeddings = fclip.encode_images([pil_image], batch_size=1)
    #    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
    #    prefix = torch.tensor(image_embeddings).to(device)
    #    #image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    #    #with torch.no_grad():
    #    #    prefix = clip_model.encode_image(image).cpu()
    #    d["clip_embedding"] = i
    #    all_embeddings.append(prefix)
    #    all_captions.append(d)
    #    if (i + 1) % 10000 == 0:
    #        with open(out_path, 'wb') as f:
    #            pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="fashion", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'fashion'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))

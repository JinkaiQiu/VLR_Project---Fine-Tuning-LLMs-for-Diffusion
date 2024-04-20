import sys
import os
#sys.path.append("fashion-clip/")
from fashion_clip.fashion_clip import FashionCLIP
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression

import torch.nn as nn

class vision_encoder(nn.Module):
    def __init__(self):
        super(vision_encoder, self).__init__()
        self.encoder = FashionCLIP('fashion-clip')
        self.batch_size = 32
        self.loss_function = nn.MSELoss()

    def encode_images(self, images):
        return self.encoder.encode_images(images, batch_size = self.batch_size)
    
    def calc_loss(self, orig_images, images):
        orig_embedding = self.encode_images(orig_images)
        embedding = self.encode_images(images)
        return self.loss_function(orig_embedding, embedding)

if __name__ == "__main__":
    vision_encoder = vision_encoder()
    print("Model loaded")
    # Load data
    data_root = os.path.join(os.path.dirname(os.getcwd()), 'data_for_fashion_clip/')

    data = pd.read_csv(data_root + 'articles.csv')
        # drop items that have the same description
    subset = data
    print(len(subset))
    # subset = data.drop_duplicates("detail_desc").copy()

    # # remove items of unkown category
    # subset = subset[subset["product_group_name"].isin(["Unknown"])]

    # # FashionCLIP has a limit of 77 tokens, let's play it safe and drop things with more than 40 tokens
    # subset = subset[subset["detail_desc"].apply(lambda x : 4 < len(str(x).split()) < 40)]

    # # We also drop products types that do not occur very frequently in this subset of data
    most_frequent_product_types = [k for k, v in dict(Counter(subset["product_type_name"].tolist())).items() if v > 5]
    subset = subset[subset["product_type_name"].isin(most_frequent_product_types)]

    images_path = ["data_for_fashion_clip/" + str(k) + ".jpg" for k in subset["article_id"].tolist()]

    # for image in images_path:
    #     print(image)
    
    print(len(images_path))
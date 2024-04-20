import pandas as pd
import os
import skimage.io as io
import PIL.Image

data_path = 'data/subset_data.csv'
data = pd.read_csv(data_path)

def get_sample(i = -1, limit = None):
    if i == -1:
        if limit:
            data_small = data.head(limit)
            row = data_small.sample(n=1)
        else:
            row = data.sample(n=1)
        row = row.iloc[0]
    else:
        row = data.iloc[i]
    # row = data.iloc[4]
    article_id = row['article_id']
    img_path = f"./data/images/0{str(article_id)[:2]}/0{article_id}.jpg"
    if not os.path.exists(img_path):
        row = data.iloc[0]
        article_id = row['article_id']
        img_path = f"./data/images/0{str(article_id)[:2]}/0{article_id}.jpg"
    image = io.imread(img_path)
    pil_image = PIL.Image.fromarray(image)
    return pil_image, row
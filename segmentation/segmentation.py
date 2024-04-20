import sys
import numpy as np
import torch
from PIL import Image
import cv2

sys.path.append("GroundingDINO/") # Location of GroundingDNIO/

from segment_anything import sam_model_registry, SamPredictor
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sam_model_type = "vit_l"
sam_checkpoint = 'sam_vit_l_0b3195.pth'
#sam_model_type = "vit_b"
#sam_checkpoint = 'sam_vit_b_01ec64.pth'

gdino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
gdino_checkpoint = "groundingdino_swint_ogc.pth"

class segmentation_model():
    def __init__(self):

        # Load segment anything
        print("Loading SAM...")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam = SamPredictor(sam)

        # Load grounded DINO
        print("Loading GDINO...")
        args = SLConfig.fromfile(gdino_config)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(gdino_checkpoint) #, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        model = model.to(device)
        model.eval()
        self.gdino = model

        self.box_threshold = 0.2
        print("Done init!")


    def forward(self, caption, image):

        # GDINO
        print("Detecting boxes...")
        transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_gdino, _ = transform(image, None)  # 3, h, w
        image_gdino = image_gdino.to(device)
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        with torch.no_grad():
            outputs = self.gdino(image_gdino[None], captions=[caption])

        # Filder output
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]

        # Resizing magic
        size = image.shape
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes_filt, image_gdino.shape[:2]).to(device)

        # SAM
        print("Generating masks...")
        self.sam.set_image(image)
        masks, _, _ = self.sam.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )

        return masks

if __name__ == "__main__":

    segmentation_model = segmentation_model()

    image_path = "examples/image0.png"
    text_prompt = "clothing" #orange sport short sleeve"
    mask_path = "examples/newmask%d.png"

    #image = Image.open(image_path).convert("RGB")  # load image
    image = cv2.imread(image_path)
    masks = segmentation_model.forward(text_prompt, image)

    idx = 0
    for mask in masks:
        mask = (mask.to(torch.uint8) * 255).cpu().numpy()
        cv2.imwrite(mask_path % idx, mask[0,:,:])
        idx += 1


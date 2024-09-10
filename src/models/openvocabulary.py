import os
import copy
import json
import cv2
import torch

from collections import OrderedDict
from PIL import Image
from torchvision.transforms import (
    InterpolationMode, Compose, Resize, ToTensor, Normalize
)

from src.utils.visualize_utils import *

import sys
sys.path.append("src/external/OpenVocabulary")

import models
from utils import setup, get_similarity_map, get_segmentation_map


class OpenVocabulary(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        class Arguments:
            def __init__(self, root_repo, ov_thold=0):
                self.config_file = f"{root_repo}/vpt/configs/prompt/cub.yaml"
                self.train_type = ""
                self.output_path = "output"
                self.data_path = None
                self.checkpoint_path = "weights/openvocab/sketch_seg_best_miou.pth"
                self.threshold = ov_thold
           
        self.ov_thold = 0
        self.device = "cuda"
        self.root_repo = "src/external/OpenVocabulary"
        args = Arguments(self.root_repo, ov_thold=self.ov_thold)
        cfg = setup(args)
        
        Ours, preprocess = models.load(
            "CS-ViT-B/16", device=self.device, cfg=cfg, zero_shot=False)
        state_dict = torch.load(args.checkpoint_path)
    
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v 
        Ours.load_state_dict(new_state_dict)     
        Ours.eval().cuda()
        
        preprocess = Compose([
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC), 
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073), 
                (0.26862954, 0.26130258, 0.27577711))
        ])
    
        with open(f"{self.root_repo}/cgatnet_to_fscoco_mappings.json", "r") as f:
            rev_fscoco_map = json.load(f)
            
        self.model = Ours
        self.cfg = cfg
        self.preprocessor = preprocess
        self.labels_info = rev_fscoco_map


    def forward(self, scene_strokes, labels, 
                data_labels_info, W, H, thickness, to_vis=False):

        mdl_label_names = []
        for idx in labels:
            cgatnet_label = data_labels_info["idx_to_label"][idx]
            if cgatnet_label in self.labels_info:
                mdl_label_names.append(self.labels_info[cgatnet_label])
            else:
                print(f"{cgatnet_label} not in mapping, taking as it is.")
                mdl_label_names.append(cgatnet_label)
                self.labels_info[cgatnet_label] = cgatnet_label # adds to dict

        if not to_vis:
            scene_visuals, _ = draw_sketch(
                scene_strokes,
                canvas_size=[W, H],
                margin=0,
                white_bg=True,
                color_fg=False,
                shift=False,
                scale_to=-1,
                is_absolute=True,
                thickness=thickness
            )
        
        else:
            scene_visuals, _ = draw_sketch(
                scene_strokes,
                canvas_size=[800, 800],
                margin=50,
                white_bg=True,
                color_fg=False,
                shift=True,
                scale_to=800,
                is_absolute=True,
                thickness=thickness
            )
            W, H = 800, 800
    
        scene_visuals = scene_visuals.astype(np.uint8)
        new_scene_visuals = np.full((max(H, W), max(H, W), 3), 255, dtype=np.uint8)
        new_scene_visuals[:H, :W, :] = scene_visuals
        scene_visuals = new_scene_visuals
        binary_sketch = np.where(scene_visuals > 0, 255, scene_visuals)
    
        pil_img = Image.fromarray(scene_visuals).convert("RGB")
        sketch_tensor = self.preprocessor(pil_img).unsqueeze(0).to(self.device)
    
        with torch.no_grad():
            text_features = models.encode_text_with_prompt_ensemble(
                self.model, mdl_label_names, self.device, no_module=True)
            redundant_features = models.encode_text_with_prompt_ensemble(
                self.model, [""], self.device, no_module=True) 
        
            sketch_features = self.model.encode_image(
                sketch_tensor,
                layers=12,
                text_features=text_features-redundant_features,
                mode="test")
            sketch_features /= sketch_features.norm(dim=1, keepdim=True)
        
        similarity = sketch_features @ (text_features - redundant_features).t()
        patches_similarity = similarity[0, self.cfg.MODEL.PROMPT.NUM_TOKENS + 1:, :]
        pixel_similarity = get_similarity_map(
            patches_similarity.unsqueeze(0), (max(W, H), max(W, H)))
        pixel_similarity[pixel_similarity < self.ov_thold] = 0
        pixel_similarity_array = pixel_similarity.cpu().numpy().transpose(2, 0, 1)
        
        pred_mtx = get_segmentation_map(
            pixel_similarity_array,
            binary_sketch,
            labels)[:H, :W]
        
        return pred_mtx
    
    
    
import math
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from src.utils.sketch_utils import *
from src.utils.visualize_utils import *

class ClassifierPreprocessor:
    
    def __init__(self, 
                 margin_size: int=5, 
                 out_sketch_size: int=299, 
                 color_images: bool=False):
        
        self.task = "classification"
        self.margin_size = margin_size
        self.out_sketch_size = out_sketch_size
        self.color_images = color_images
        
        self.sketch_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ]
        )
    
    def __call__(self, 
                 stroke3: torch.Tensor, 
                 divisions: torch.LongTensor, 
                 img_sizes: torch.Tensor):
        """
        * stroke3 -> B x S x 3 with padding
        * divisions -> B x (max_obj_count + 1) with padding 
        * img_sizes -> B x 2 (scene image sizes)
        """
        no_batch_dim = len(stroke3.shape) == 2
        if no_batch_dim:
            stroke3 = stroke3.unsqueeze(0)
            divisions = divisions.unsqueeze(0)
            img_sizes = img_sizes.unsqueeze(0)
        
        max_obj_cnt = divisions.shape[-1] - 1
        batch_visuals = []
        for b in range(stroke3.shape[0]):
            w, h = img_sizes[b, 0], img_sizes[b, 1]
            stroke_starts = divisions[b, :].long()
            scene = stroke3[b, ...]

            pad_start = torch.where(stroke_starts < 0)[0]
            if len(pad_start) > 0:
                pad_start = pad_start[0]
            else:
                pad_start = stroke_starts.shape[0]

            # if no valid class is in a scene
            if pad_start == 0:
                sketch_visuals = torch.zeros(
                    max_obj_cnt, 3, self.out_sketch_size, self.out_sketch_size)
            else:
                stroke_starts = stroke_starts[:pad_start]
                point_starts = [0] + (torch.where(scene[..., -1] == 1)[0] + 1).tolist()

                sketch_visuals, boxes = [], []
                for str_start in range(1, len(stroke_starts)):
                    
                    start_str = stroke_starts[str_start - 1]
                    end_str = stroke_starts[str_start]
                    start, end = point_starts[start_str], point_starts[end_str]
                    sketch = scene[start:end, ...].numpy()
                    boxes.append(get_absolute_bounds(sketch))

                    sketch_vis = self.draw_sketch(sketch)
                    sketch_visuals.append(sketch_vis)   
            
                sketch_visuals = self.pad_items(sketch_visuals, max_obj_cnt)
            
            batch_visuals.append(sketch_visuals.tolist())
        
        batch_visuals = torch.Tensor(batch_visuals)
        
        if no_batch_dim:
            batch_visuals = batch_visuals.squeeze(0)

        return batch_visuals

    
    def pad_items(self, sketch_images: np.ndarray, max_obj_cnt: int): 
        
        diff_size = max(0, max_obj_cnt - len(sketch_images))
        sketch_images = torch.stack(sketch_images, dim=0)
        
        if diff_size > 0:
            # padding for sketch_images
            empty_images = torch.zeros(
                diff_size, 3, self.out_sketch_size, self.out_sketch_size)
            sketch_images = torch.cat([sketch_images, empty_images], dim=0)
        
        elif diff_size < 0: # and self.split == "train":            
            sketch_images = sketch_images[:max_obj_cnt]

        return sketch_images
    
    
    def draw_sketch(self, sketch, save_path=None):
        
        sketch_divisions = [0] + (np.where(sketch[..., -1] == 1)[0] + 1).tolist()
        
        sketch_img, _ = draw_sketch(
            np.asarray(sketch).astype(float),
            sketch_divisions,
            margin=self.margin_size,
            scale_to=self.out_sketch_size - (2 * self.margin_size),
            is_absolute=True,
            color_fg=self.color_images,
            white_bg=True,
            shift=True,
            canvas_size=self.out_sketch_size,
            save_path=save_path)
    
        sketch_img = Image.fromarray(cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB))
        sketch_img = self.sketch_transforms(sketch_img)

        return sketch_img
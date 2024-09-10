import os
import json
import torch
import torchvision.transforms as T
import random
import numpy as np
import math

from PIL import Image
from torch.utils.data import Dataset

from src.utils.sketch_utils import *
from src.utils.visualize_utils import *


class BaseDataset(Dataset):
    
    def __init__(self, split, cfg, save_dir: str=None, preprocessor=None):
        super().__init__()
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.dataset_name = cfg[split]["dataset_name"]
        self.max_point_cnt = cfg[split]["max_point_cnt"]
        self.labels_info = self.read_labels_info(save_dir)
        self.preprocessor = preprocessor
  
    def __getitem__(self, idx):
        raise NotImplementedError
        
    
    def read_labels_info(self, save_dir: str=None):
        if self.split == "train":
            # in train, labels are generated from scratch
            labels_info = None
        else:
            # in validation and test, already existing labels info is read
            assert os.path.exists(os.path.join(save_dir, "labels_info.json"))
            with open(os.path.join(save_dir, "labels_info.json"), "r") as f:
                labels_info = json.load(f)
                
            self.num_categories = len(labels_info["idx_to_label"])

        return labels_info
    
    
    def save_labels_info(self, save_dir: str=None):
        assert self.split == "train"
        if save_dir is not None:
            with open(os.path.join(save_dir, "labels_info.json"), "w") as f:
                json.dump(self.labels_info, f)
                

    def read_and_scale_sketch(self, npy_path, bbox=None):
        
        sketch = read_npy(npy_path)

        # shift sketch to top-left
        abs_sketch = relative_to_absolute(sketch)
        xmin, ymin, xmax, ymax = get_absolute_bounds(abs_sketch)
        s_w = xmax - xmin
        s_h = ymax - ymin
        
        # shift sketch to bbox top-left
        abs_sketch[:, 0] -= xmin
        abs_sketch[:, 1] -= ymin
        
        if bbox is not None:
            # get obj bbox information
            o_xmin, o_ymin, o_xmax, o_ymax = bbox
            o_h = (o_ymax - o_ymin)
            o_w = (o_xmax - o_xmin)
            
            # re-scale sketch without losing orig object ratio
            r_h = o_h / max(1, s_h)
            r_w = o_w / max(1, s_w)
            
            # scale sketch coords according to bbox w & h
            abs_sketch[:, 0] *= r_w
            abs_sketch[:, 1] *= r_h
            
            # shift sketch coords according to bbox start
            abs_sketch[:, 0] += o_xmin
            abs_sketch[:, 1] += o_ymin
        
        return abs_sketch.astype(int).astype(float)
    
    
    def read_and_scale_sketch_alternative(self, npy_path, bbox=None):
        
        sketch = read_npy(npy_path)

        # shift sketch to top-left
        abs_sketch = relative_to_absolute(sketch)
        xmin, ymin, xmax, ymax = get_absolute_bounds(abs_sketch)
        s_w = xmax - xmin
        s_h = ymax - ymin
        
        # shift sketch to bbox top-left
        abs_sketch[:, 0] -= xmin
        abs_sketch[:, 1] -= ymin
        
        s_cx = (xmax - xmin) // 2
        s_cy = (ymax - ymin) // 2
        
        if bbox is not None:
            # get obj bbox information
            o_xmin, o_ymin, o_xmax, o_ymax = bbox
            o_h = (o_ymax - o_ymin)
            o_w = (o_xmax - o_xmin)
            o_cx = (o_xmax + o_xmin) // 2
            o_cy = (o_ymax + o_ymin) // 2
            
            sft_x = o_cx - s_cx
            sft_y = o_cy - s_cy
            
            if s_h / max(1, s_w) > o_h / max(1, o_w):
                r = o_h / max(1, s_h)
            else:
                r = o_w / max(1, s_w)
                
            # scale sketch coords according to bbox w & h
            abs_sketch[:, 0] *= r
            abs_sketch[:, 1] *= r
            
            # shift sketch coords according to bbox start
            abs_sketch[:, 0] += s_cx
            abs_sketch[:, 1] += s_cy
        
        return abs_sketch.astype(int).astype(float)
    
    
    def pad_vector_sketch(self, sketch: np.ndarray):
        sk_len = sketch.shape[0]
        
        if self.max_point_cnt > 0:
            if sk_len < self.max_point_cnt:
                diff = self.max_point_cnt - sk_len
                pad_pnts = np.full((diff, 3), -1)
                sketch = np.concatenate([sketch, pad_pnts], axis=0)
            else:
                sketch = sketch[:self.max_point_cnt, :]
                sketch[-1, -1] = 1
        
        return sketch
    
    def run_preprocessor(self, return_dict: dict):
        
        if self.preprocessor is not None:
            if self.preprocessor.task == "classification":
                return_dict["obj_visuals"] = self.preprocessor(return_dict["vectors"])
            else:
                raise ValueError
        
        return return_dict
        
        
        
class BaseSceneDataset(BaseDataset):
    
    def __init__(self, split, cfg, save_dir=None, preprocessor=None):
        super().__init__(split, cfg, save_dir=save_dir, preprocessor=preprocessor)
        self.max_obj_cnt = cfg[split]["max_obj_cnt"]
        
        self.mapping_dict, labels_set = self.read_mapping_dict(
            cfg[split]["mapping_file"])
        
        if self.labels_info is None:
            assert split == "train"
            # initialized in the BaseDataset
            self.labels_info = self.generate_labels_info(labels_set)   
            self.save_labels_info(save_dir)
            
        if "extra_filter_file" in cfg[split]:
            extra_filter_file = cfg[split]["extra_filter_file"]
        else:
            extra_filter_file = None
            
        self.extra_filter_classes = self.read_extra_filter_classes(extra_filter_file)
    
    
    def read_extra_filter_classes(self, extra_filter_file: str=None):
        if extra_filter_file is None:
            filter_classes = []
            for k in self.mapping_dict:
                if self.mapping_dict[k] is not None:
                    filter_classes.extend(self.mapping_dict[k])
            return set(filter_classes)
        
        with open(extra_filter_file, "r") as f:
            lines = f.readlines()
        classes = set([cls_name.replace("\n", "").strip() for cls_name in lines if len(cls_name) > 2])
        
        return classes
    
    
    def read_mapping_dict(self, mapping_file):
        if self.dataset_name == "coco":
            key = "coco_to_sketch"
        elif self.dataset_name == "vg":
            key = "vg_to_sketch"
        elif self.dataset_name == "cbsc":
            key = "cbsc_to_qd"
        elif self.dataset_name == "friss":
            key = "friss_to_qd"
        elif self.dataset_name == "fscoco":
            key = "fscoco_to_sketch"
            
        mapping_dict = json.load(open(mapping_file, 'r'))[key]
        labels_set = set()
        for class_name in mapping_dict:
            if mapping_dict[class_name] is not None:
                if type(mapping_dict[class_name]) == str:
                    mapping_dict[class_name] = [mapping_dict[class_name]]
                for mapped_cls in mapping_dict[class_name]:
                    labels_set.add(mapped_cls)
        
        return mapping_dict, labels_set
    
    
    def generate_labels_info(self, labels_set):
        labels_info = {"idx_to_label": {}, "label_to_idx": {}}
        for i, val in enumerate(sorted(list(labels_set))):
            labels_info["idx_to_label"][i] = val
            labels_info["label_to_idx"][val] = i            
        
        self.num_categories = len(labels_info["idx_to_label"])      
        return labels_info
    

    def pad_items(self, sketch_vectors: np.ndarray, gt_labels: list, divisions: list): 
        
        sketch_vectors = self.pad_vector_sketch(np.asarray(sketch_vectors))
        diff_size = max(0, self.max_obj_cnt - len(gt_labels))

        sketch_vectors = torch.Tensor(sketch_vectors)
        gt_labels = torch.LongTensor(gt_labels)
        divisions = torch.LongTensor(divisions)
        
        # generation of padding mask
        labels_length = gt_labels.shape[0]
        
        if self.max_obj_cnt > 0:
            if diff_size > 0:
                # padding for gt_labels
                gt_labels = torch.cat([gt_labels, torch.full((diff_size,), -1, dtype=int)], dim=0)
                # padding for divisions
                divisions = torch.cat([divisions, torch.full((diff_size,), -1, dtype=int)], dim=0)
                
            elif diff_size < 0:       
                # padding for gt_labels
                gt_labels = gt_labels[:self.max_obj_cnt]
                # padding for divisions
                divisions = divisions[:self.max_obj_cnt+1]
        
            padding_mask = torch.ones([self.max_obj_cnt, 1], dtype=int)
            if labels_length < self.max_obj_cnt:
                padding_mask[labels_length:, 0] = 0
        else:
            padding_mask = torch.ones([labels_length, 1], dtype=int)

        if sketch_vectors[-1, -1] != 1:
            sketch_vectors[-1, -1] = 1
        
        return sketch_vectors, gt_labels, divisions, padding_mask

    
    def filter_objects_from_scene(self, gt_classes, scene_strokes, object_divisions):
        
        stroke_start_points = [0] + (np.where(scene_strokes[..., -1] == 1)[0] + 1).tolist()
        abs_scene = relative_to_absolute(scene_strokes)
        
        sketch_vectors, gt_labels, gt_divisions = [], [], [0]
        for idx, cls_name in enumerate(gt_classes):
            start_id = object_divisions[idx]
            end_id = object_divisions[idx+1]
            stroke_cnt = end_id - start_id
            
            start_point = stroke_start_points[start_id]
            end_point = stroke_start_points[end_id]

            if (cls_name in self.mapping_dict and 
                self.mapping_dict[cls_name] is not None):
                if self.split != "train":
                    mapped_cls = self.mapping_dict[cls_name][0]
                else:
                    mapped_cls = random.choice(self.mapping_dict[cls_name])
            else:
                mapped_cls = None
                    
            # a subset of mapping_dict keys
            if (mapped_cls is not None and mapped_cls in self.extra_filter_classes): 
                obj_strokes = abs_scene[start_point:end_point].tolist()
                sketch_vectors.extend(obj_strokes)
                gt_labels.append(self.labels_info["label_to_idx"][mapped_cls]) 
                gt_divisions.append(gt_divisions[-1] + stroke_cnt)
            
            start_point = end_point
        
        return sketch_vectors, gt_labels, gt_divisions
    
    
    def run_preprocessor(self, return_dict: dict):
        
        if self.preprocessor is not None:
            if self.preprocessor.task == "classification":
                sketch_visuals, attns = self.preprocessor(
                    return_dict["vectors"], 
                    return_dict["divisions"], 
                    return_dict["img_size"])
                return_dict.update({
                    "obj_visuals": sketch_visuals,
                    "attns": attns
                })
            elif self.preprocessor.task == "segmentation":
                (sketch_visuals, 
                 boxes, 
                 stroke_areas, 
                 stroke_area_inds, 
                 new_sizes) = self.preprocessor(
                    return_dict["vectors"], 
                    return_dict["divisions"], 
                    return_dict["img_size"])
                return_dict.update({
                    "scene_visuals": sketch_visuals,
                    "boxes": boxes,
                    "stroke_areas": stroke_areas,
                    "stroke_area_inds": stroke_area_inds,
                    "segmentation_sizes": new_sizes
                })
            
            else:
                raise ValueError
        
        return return_dict
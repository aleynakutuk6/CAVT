import argparse
import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.datasets import CBSCDataset
from src.data.preprocessors import CAVTPreprocessor, ClassifierPreprocessor
from src.metrics import AllOrNothing, SegmentationMetrics, SequenceIoU
from src.utils.cfg_utils import parse_configs
from src.utils.visualize_utils import *

# ---------------------------------------------------------------------------------
# PARSE ARGUMENTS
# ---------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-d", "--dataset-name", type=str, required=True)
parser.add_argument("-c", "--common-class-type", type=str, required=True)
parser.add_argument("-foq", "--filter-out-qd", action="store_true")
parser.add_argument("-ov", "--ov-style-preds", action="store_true")
parser.add_argument("-lg", "--result-log-file", type=str, default=None)
parser.add_argument("-vd", "--visualize-root-dir", type=str, default=None)
parser.add_argument("-vil", "--visualize-instance-level", action="store_true")
args = parser.parse_args()

model_type = args.model
dataset_name = args.dataset_name
common_class_type = args.common_class_type
filter_out_qd = args.filter_out_qd
ov_style_preds = args.ov_style_preds
result_log_file = args.result_log_file
visualize_root_dir = args.visualize_root_dir
visualize_instance_level = args.visualize_instance_level

if visualize_root_dir is not None:
    visualize_dir = os.path.join(
        visualize_root_dir,
        f"{dataset_name}_{common_class_type}",
        model_type)
    os.system(f"mkdir -p {visualize_dir}")
    
    cmap_p = f"datasets/colors/{dataset_name}_colors.json"
    with open(cmap_p, "r") as f:
        class_color_mappings = json.load(f)
else:
    visualize_dir = None
    class_color_mappings = None

# ---------------------------------------------------------------------------------
# LOAD THE RELATED MODEL
# ---------------------------------------------------------------------------------

if "cavt" in model_type:
    from src.models.cavt import CAVT
    if "no-t" in model_type:
        w_color, color_fg = "gray", False
    else:
        w_color, color_fg = "color", True
    
    w_cls = "with_cls" if "no-ca" in model_type else "no_cls"
    simple_eval = "no-pp" in model_type

    model = CAVT(
        f"weights/CAVT/mmdet_yolox_cfg_{w_cls}.py",
        f"weights/CAVT/{w_cls}_{w_color}_ep600.pth",
        "cuda",
        preprocessor=CAVTPreprocessor(color_fg=color_fg, thickness=2),
        simple_eval=simple_eval)
    model = model.cuda().eval()

elif "inception" in model_type:
    from src.models.inception_v3 import InceptionV3
    model = InceptionV3(preprocessor=ClassifierPreprocessor())
    d = torch.load(
        f"weights/InceptionV3/best_model_{dataset_name}.pth", 
        map_location="cpu")
    model.load_state_dict(d["model"])
    model = model.eval().cuda()

elif "skformer" in model_type:
    from src.models.sketchformer import SketchFormer
    model = SketchFormer()

elif model_type == "openvocab":
    from src.models.openvocabulary import OpenVocabulary
    model = OpenVocabulary()
    model = model.eval().cuda()
    
elif model_type == "ldp":
    from src.models.ldp import LDP
    if "sky" in common_class_type:
        data_typ = "sky"
    elif "sketchy" in common_class_type:
        data_typ = "sketchy" 
    model = LDP(data_typ)

elif model_type != "gt":
    raise ValueError

# ---------------------------------------------------------------------------------
# LOAD THE DATA
# ---------------------------------------------------------------------------------

partition, data_dir = "test", f"datasets/data/{dataset_name}"
data_cfg = {
    partition: {
        "dataset_name": dataset_name,
        "data_dir": f"{data_dir}/{partition}",
        "max_obj_cnt": -1,
        "max_point_cnt": -1,
        "mapping_file": f"{data_dir}/{dataset_name}_to_qd_mappings.json",
        "extra_filter_file": f"datasets/class_lists/{common_class_type}.txt"
    }
}

dataset = CBSCDataset(partition, data_cfg, save_dir="datasets/data")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

segm_out_file = f"{dataset_name}_{common_class_type}_cavtpreds.json"
if "skformer" in model_type or "inception" in model_type:
    with open(segm_out_file, "r") as f:
        segm_outs = json.load(f)
else:
    segm_outs = {}
    
# ---------------------------------------------------------------------------------
# SET VALID INDICES
# ---------------------------------------------------------------------------------

valid_idxs, valid_names = set(), set()
for data in tqdm(dataloader, desc="Calculating valid class set"):
    gt_labels = data["labels"][0, :].numpy().tolist()
    for gt_label in gt_labels:
        if gt_label >= 0:
            valid_idxs.add(gt_label)
            valid_names.add(dataset.labels_info["idx_to_label"][str(gt_label)])

num_valid_cls = len(valid_idxs) + 1
valid_idx_dict = {"label_to_idx": {}, "idx_to_label": {}} 
for v_idx, valid_idx in enumerate(valid_idxs):
    valid_name = dataset.labels_info["idx_to_label"][str(valid_idx)]
    valid_idx_dict["label_to_idx"][valid_name] = v_idx + 1
    valid_idx_dict["idx_to_label"][v_idx + 1] = valid_name
    
def to_valid_labels(cls):
    name = dataset.labels_info["idx_to_label"][str(cls)]
    return valid_idx_dict["label_to_idx"][name]

# ---------------------------------------------------------------------------------
# DECLARE EVALUATION METRICS
# ---------------------------------------------------------------------------------

if "cavt" in model_type:        
    metric_fns = {
        "AoN": AllOrNothing(filter_out_qd=filter_out_qd),
        "sIoU": SequenceIoU(filter_out_qd=filter_out_qd)
    }
elif model_type != "gt":
    metric_fns = {
        "segm": SegmentationMetrics(num_valid_cls, ignore_bg=True)
    }
else:
    metric_fns = {}
    
# ---------------------------------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------------------------------

if visualize_dir is not None and dataset_name == "cbsc":
    image_ids_filter_set = {4, 41, 63, 76, 81, 98, 
                            168, 173, 185, 202, 217, 184}
elif visualize_dir is not None and dataset_name == "friss":
    image_ids_filter_set = {45, 70, 242, 284, 417, 420, 
                            480, 433, 434, 441, 484, 469, 492}
else:
    image_ids_filter_set = set()


def draw_instance_level(scene_strokes, divisions, label_names, save_path):
    draw_sketch(
        scene_strokes,
        divisions,
        class_ids=None,
        canvas_size=[800, 800],
        margin=50,
        is_absolute=True,
        white_bg=True,
        color_fg=True,
        shift=True,
        scale_to=800,
        save_path=save_path,
        class_names=label_names,
        thickness=4)
    
    
def get_class_mtx(
    strokes, divisions, label_ids, 
    vis_mode: bool=False, thickness=2):
    
    if vis_mode:
        return draw_sketch(        
            scene_strokes,
            divisions,
            label_ids,
            canvas_size=[800, 800], 
            margin=50,
            white_bg=False,
            color_fg=True,
            shift=True,
            scale_to=800,
            is_absolute=True, 
            thickness=4)[0][:,:,0]
    else:
        return draw_sketch(        
            scene_strokes,
            divisions,
            label_ids,
            canvas_size=[W, H], 
            margin=0,
            white_bg=False,
            color_fg=True,
            shift=False,
            scale_to=-1,
            is_absolute=True, 
            thickness=thickness)[0][:,:,0]
    

for data in tqdm(dataloader):
    # if an empty scene is returned due to class filtering in dataset
    if data["vectors"] is None or data["vectors"][0, 0, -1] < 0: continue
    assert data["vectors"][0, -1, -1] == 1
    
    img_id = data["image_id"].squeeze(0).data.item()
    if len(image_ids_filter_set) > 0 and img_id not in image_ids_filter_set:
        continue
    
    W, H = data["img_size"].squeeze(0).numpy().tolist()
    scene_strokes = data["vectors"].squeeze(0).numpy().astype(float)
    divisions = data["divisions"].squeeze(0).numpy().astype(int)
    orig_labels = data["labels"].squeeze(0).numpy().astype(int)

    str_starts = [0] + (np.where(scene_strokes[:, -1] == 1)[0] + 1).tolist()
    str_to_pnt_maps = {}
    for str_idx, str_start in enumerate(str_starts):
        str_to_pnt_maps[str_idx] = str_start
    
    labels = [to_valid_labels(cls) for cls in orig_labels]
    pnt_divisions = [str_to_pnt_maps[div] for div in divisions]
    gt_mtx = get_class_mtx(
        scene_strokes, pnt_divisions, labels,
        vis_mode=visualize_dir is not None,
        thickness=4 if "ldp" in model_type or "openvocab" in model_type else 2)

    if model_type == "gt":
        pred_mtx = gt_mtx
        if visualize_instance_level:
            label_names = [
                valid_idx_dict["idx_to_label"][idx] for idx in labels
            ]
            draw_instance_level(
                scene_strokes, 
                pnt_divisions, 
                label_names, 
                os.path.join(visualize_dir, f"{img_id}_instlevel.jpg"))
        
    elif "cavt" in model_type:
        with torch.no_grad():
            scores, boxes, ranges = model(
                data["vectors"],
                data["divisions"],
                data["img_size"],
                return_pnt_level_ranges=False)
        ranges = ranges.squeeze(0).cpu().numpy().astype(int)
            
        if visualize_dir is None:
            for k in metric_fns:
                metric_fns[k].add(ranges, divisions, orig_labels)
        
        segm_outs[img_id] = ranges.tolist()
    
    
    elif "inception" in model_type or "skformer" in model_type:
        pred_divisions = segm_outs[str(img_id)]
        pred_pnt_divisions = [str_to_pnt_maps[div] for div in pred_divisions]
        
        if "inception" in model_type:
        
            with torch.no_grad():
                out_probs = model(
                    data["vectors"], 
                    torch.Tensor(pred_divisions).unsqueeze(0), 
                    data["img_size"]).squeeze(0)

            pred_classes = out_probs.argsort(dim=-1, descending=True).long()
            pred_labels = []
            for s in range(pred_classes.shape[0]):
                for obj in range(pred_classes.shape[-1]):
                    pred_result = pred_classes[s, obj].item()
                    
                    if ov_style_preds:
                        if pred_result not in orig_labels:
                            continue
                    else:
                        if pred_result not in valid_idxs:
                            continue
    
                    mapped_result = to_valid_labels(pred_result)
                    pred_labels.append(mapped_result)
                    break

        elif "skformer" in model_type:
    
            pred_names = model(scene_strokes, pred_divisions)
            pred_labels = []
            for pred_name_list in pred_names:
                for pred_name in pred_name_list:
                    if pred_name not in dataset.labels_info["label_to_idx"]:
                        continue
                    pred_idx = int(dataset.labels_info["label_to_idx"][pred_name])
                    if ov_style_preds:
                        if int(pred_idx) not in orig_labels: continue
                    else:
                        if pred_idx not in valid_idxs: continue
                   
                    mapped_idx = to_valid_labels(pred_idx)
                    pred_labels.append(mapped_idx)  
                    break

        pred_mtx = get_class_mtx(
            scene_strokes, pred_pnt_divisions, pred_labels,
            vis_mode=visualize_dir is not None, thickness=2)
    
        
        if visualize_dir is None:
            for k in metric_fns:
                metric_fns[k].add(pred_mtx, gt_mtx)
        elif visualize_instance_level:
            label_names = [
                valid_idx_dict["idx_to_label"][idx] for idx in pred_labels
            ]
            if ov_style_preds:
                save_name = os.path.join(visualize_dir, f"{img_id}_instlevel_ovfilter.jpg")
            else:
                save_name = os.path.join(visualize_dir, f"{img_id}_instlevel.jpg")
            draw_instance_level(
                scene_strokes, 
                pred_pnt_divisions, 
                label_names, 
                save_name)

    elif "ldp" in model_type or "openvocab" in model_type:
        
        pred_mtx = model(
            copy.deepcopy(scene_strokes), 
            copy.deepcopy(labels), 
            valid_idx_dict, W, H, thickness=4, 
            to_vis=visualize_dir is not None)
        
        if visualize_dir is None:
            for k in metric_fns:
                metric_fns[k].add(pred_mtx, gt_mtx)
        
                
    if visualize_dir is not None and not visualize_instance_level:
        
        # flat_pred = set(pred_mtx.flatten().tolist())
        # flat_pred = [
        #     valid_idx_dict["idx_to_label"][val] for val in flat_pred if val > 0
        # ]
        # print(img_id, flat_pred)

        if ov_style_preds:
            save_path = os.path.join(visualize_dir, f"{img_id}_ovfilter.jpg")
        else:
            save_path = os.path.join(visualize_dir, f"{img_id}.jpg")
            
        visualize_from_mtx(
            pred_mtx, 
            valid_idx_dict,
            class_color_mappings[str(img_id)], 
            save_path)

# ---------------------------------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------------------------------

os.system("rm -rf CUB")

if visualize_dir is None:

    log_txt = "\n" + "#"*60 + "\n"
    for k in metric_fns:
        
        if k == "segm":
            ova_acc, mean_acc, mean_iou, fw_iou = metric_fns[k].calculate()
            log_txt += f"--> OVA-acc  : {ova_acc}\n"
            log_txt += f"--> Mean-acc : {mean_acc}\n"
            log_txt += f"--> Mean-IoU : {mean_iou}\n"
            log_txt += f"--> FW-IoU   : {fw_iou}\n"
            
        elif k == "AoN":
            aon_score = metric_fns[k].calculate()
            log_txt += f"--> AoN-SCORE  : {aon_score}\n"
        
        elif k == "sIoU":
            siou_score = metric_fns[k].calculate()
            log_txt += f"--> SIoU-SCORE : {siou_score}\n"
    
    
    log_txt += f"--> Dataset  : {dataset_name} ({common_class_type})\n"
    log_txt += f"--> Model    : {model_type}\n"
    log_txt += f"--> F-Out QD : {filter_out_qd}\n"
    log_txt += f"--> OV Style : {ov_style_preds}\n"
    log_txt += "#"*60 + "\n\n"
    
    print(log_txt)
    if result_log_file is not None:
        with open(result_log_file, "a") as f:
            f.write(log_txt)
    
    if "cavt" in model_type:
        with open(segm_out_file, "w") as f:
            json.dump(segm_outs, f)
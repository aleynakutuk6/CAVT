import cv2
import os
import numpy as np

ROOT = "visual_results"
datasets = [file for file in os.listdir(ROOT) if "." not in file and "pycache" not in file]
models = [file for file in os.listdir(os.path.join(ROOT, datasets[0])) if "." not in file and "_" not in file]

for dataset in datasets:
    overall = {}
    
    for model in models:
        overall[model] = {"cls": [], "cls_ov": [], "inst": [], "inst_ov": []}
        files = [file for file in os.listdir(os.path.join(ROOT, dataset, model)) if ".jpg" in file]
        files = sorted(files)
        
        for file in files:
            if "_instlevel_ovfilter" in file:
                overall[model]["inst_ov"].append(os.path.join(ROOT, dataset, model, file))
                
            elif "_ovfilter" in file:
                overall[model]["cls_ov"].append(os.path.join(ROOT, dataset, model, file))
            
            elif "_instlevel" in file:
                overall[model]["inst"].append(os.path.join(ROOT, dataset, model, file))
                
            else:
                overall[model]["cls"].append(os.path.join(ROOT, dataset, model, file))

    # Class Level Visualizations
    
    # order = [
    #     ["gt", "cls"], None, 
    #     ["ldp", "cls"], ["skformer", "cls"], ["inception", "cls"], None, 
    #     ["openvocab", "cls"], ["skformer", "cls_ov"], ["inception", "cls_ov"], None,
    #     ["skformer", "inst"], ["inception", "inst"]
    # ]
    
    orders = {
        "paper": [
            ["gt", "cls"], None,
            ["ldp", "cls"], ["inception", "cls"], None,
            ["openvocab", "cls"], ["inception", "cls_ov"], None, None,
            ["inception", "inst"]
        ],
        "suppl_cls": [
            ["gt", "cls"], None,
            ["ldp", "cls"], ["inception", "cls"], ["skformer", "cls"], None,
            ["openvocab", "cls"], ["inception", "cls_ov"],["skformer", "cls_ov"],
        ],
        "suppl_inst": [
            ["gt", "inst"], None,
            ["skformer", "inst"], ["inception", "inst"]
        ]
    }
    
    for k_o in orders:
        order = orders[k_o]
    
        S = len(overall[order[0][0]][order[0][1]])
        L = 804
        D = 20
        
        num_breaks = 0
        for o in order:
            if o is None:
                num_breaks += 1
        
        num_imgs = len(order) - num_breaks
                
        W = (L * num_imgs) + D * (num_imgs - num_breaks - 1) + 2 * D * num_breaks + D
        H = S * L + (S - 1) * D + D
        total_canvas = np.full([H, W, 3], 255)
        
        P = D // 2
        for o in order:
            if o is None:
                total_canvas[:, P - 4: P + 4, :] = 40
                P += D
                
            else:
                model, subset = o
                for i, img_p in enumerate(overall[model][subset]):
                    img_orig = cv2.imread(img_p, cv2.IMREAD_COLOR)
                    ih, iw, ic = img_orig.shape
                    img = np.full((ih+4, iw+4, ic), 40)
                    img[2:-2, 2:-2, :] = img_orig
                    
                    h = D // 2 + i * (L + D)
                    total_canvas[h:h+L, P:P+L, :] = img
                
                P += L + D
                
        cv2.imwrite(f"visual_results/{k_o}_vis_for_{dataset}.jpg", total_canvas)

        
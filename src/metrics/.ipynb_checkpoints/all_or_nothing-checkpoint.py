import torch
import numpy as np

from tqdm import tqdm


class AllOrNothing:
    
    def __init__(self, filter_out_qd: bool=False):
        self.pred = []
        self.gt = []
        self.gt_class = []
        self.filter_out_qd = filter_out_qd
        self.non_qd_classes = {
            7, 17, 23, 26, 34, 35, 39, 44, 46, 47, 51, 53, 56, 66, 67, 69, 81, 85, 86,
            92, 93, 99, 103, 104, 107, 109, 114, 118, 119, 121, 125, 130, 137, 138, 143, 
            150, 156, 157, 158, 159, 163, 168, 170, 175, 176, 178, 179, 180, 186, 187, 
            188, 191, 193, 194, 198, 205, 215, 219, 223, 232, 244, 250, 253, 258, 261, 
            263, 265, 268, 269, 270, 272, 277, 280, 286, 291, 292, 293, 299, 300, 307, 
            309, 311, 319, 320, 325, 330, 332, 335, 336, 339, 340, 341, 342, 343, 345, 
            346, 350, 351, 356, 357, 358, 360, 362, 365, 369, 370, 376, 378, 385, 387, 
            390, 396, 397, 399, 400, 402, 406, 408, 411, 413, 417, 420, 421, 424, 430, 
            435, 438, 443, 444, 449, 450, 452, 455, 456, 459, 463, 469, 470, 472, 477, 
            479, 482, 483 
        }

    def add(self, pred: np.ndarray, gt: np.ndarray, gt_class: np.ndarray=None):
        if len(gt.shape) == 1:
            pred = pred[np.newaxis, :]
            gt = gt[np.newaxis, :]
            gt_class = gt_class[np.newaxis, :]
        
        for b in range(gt.shape[0]):
            self.pred.append(pred[b, :].astype(int).tolist())
            self.gt.append(gt[b, :].astype(int).tolist())
            if gt_class is not None:
                self.gt_class.append(gt_class[b, :].astype(int).tolist())
    
    def calculate(self):
        total_score, total_num_cnt = 0, 0
        for b in range(len(self.gt)):
            corrs, num_cnt = 0, 0
            for i in range(1, len(self.gt[b])):
                if (self.filter_out_qd and 
                    self.gt_class[b][i-1] not in self.non_qd_classes):
                    continue
                else:
                    num_cnt += 1
                
                for j in range(1, len(self.pred[b])):
                    if (self.gt[b][i-1] == self.pred[b][j-1] and 
                        self.gt[b][i] == self.pred[b][j]):
                        # both indices are the same
                        corrs += 1
                        break
                    elif self.gt[b][i] < self.pred[b][j]:
                        # no need to check the remaining predictions
                        break
            
            total_score += corrs / max(1, num_cnt)
            if num_cnt > 0:
                total_num_cnt += 1

        total_score = total_score / max(1, total_num_cnt)
        if total_num_cnt < 1:
            print("No class is found to calculate the AoN.")
        self.pred, self.gt, self.gt_class = [], [], []
        return total_score
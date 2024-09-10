import torch
import torchvision.models as models

from src.data.preprocessors import ClassifierPreprocessor

class InceptionV3(models.Inception3):
    
    def __init__(self, 
                 preprocessor: ClassifierPreprocessor=None, 
                 n_classes: int=488):
        
        super().__init__(
            num_classes=n_classes, 
            aux_logits=False, 
            init_weights=True)
        
        if preprocessor is None:
            self.preprocessor = ClassifierPreprocessor()
        else:
            self.preprocessor = preprocessor
        
    def forward(self, 
                stroke3: torch.Tensor, 
                divisions: torch.LongTensor, 
                img_sizes: torch.Tensor):
        
        visuals = self.preprocessor(stroke3, divisions, img_sizes)
        visuals = visuals.cuda()
        b, s, c, h, w = visuals.shape
        outputs = super().forward(visuals.view(-1, c, h, w))
        outputs = outputs.view(b, s, outputs.shape[-1])
        return outputs
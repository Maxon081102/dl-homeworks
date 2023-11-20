import torch

from torch import nn

from functools import partial
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

class PeopleArtModel(nn.Module):
    def __init__(self, device="cpu"):
        super(PeopleArtModel, self).__init__()
        weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        self.transform = weights.transforms()
        self.model = retinanet_resnet50_fpn_v2(weights=weights)
        
        num_anchors = self.model.head.classification_head.num_anchors
        self.model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=1,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        

    def forward(self, x, annotation):
        return self.model(x, annotation)
    
    def get_data_transform(self):
        return self.transform
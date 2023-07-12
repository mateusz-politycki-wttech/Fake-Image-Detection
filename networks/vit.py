import torch
from PIL import Image
import open_clip


class ViTModel(torch.nn.Module):
    def __init__(self):
        super(ViTModel, self).__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L/14',
                                                                     pretrained='laion400m_e32')
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.classifier = torch.nn.Linear(768, 1)
        
    def forward(self, x):
        x = self.model.encode_image(x)
        x = self.classifier(x)
        # return torch.sigmoid(x)
        return x

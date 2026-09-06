import torch
from torchvision import models

model = models.resnet18(pretrained=True)

class CustomResnet(torch.nn.Module):
    def __init__(self, original_model):
        super(CustomResnet, self).__init__()
        self.stage1 = torch.nn.Sequential(*list(original_model.children())[:4])  
        self.stage2 = original_model.layer1  
        self.stage3 = original_model.layer2  
        self.stage4 = original_model.layer3  
        self.stage5 = original_model.layer4  

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x5

Resnet18 = CustomResnet(model)

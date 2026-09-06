import torch
from torchvision import models

# 加载预训练的ResNet50模型
model = models.resnet50(pretrained=True)

# 自定义一个新的模块来获取5个阶段的输出
class CustomResnet(torch.nn.Module):
    def __init__(self, original_model):
        super(CustomResnet, self).__init__()
        self.stage1 = torch.nn.Sequential(*list(original_model.children())[:4])  # 64通道
        self.stage2 = original_model.layer1  # 256通道
        self.stage3 = original_model.layer2  # 512通道
        self.stage4 = original_model.layer3  # 1024通道
        self.stage5 = original_model.layer4  # 2048通道

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return [x2, x3, x4, x5]



Resnet50 = CustomResnet(model)
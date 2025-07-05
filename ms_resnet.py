import torch
import torch.nn as nn
import torchvision.models as models

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MultiScaleResBlock(nn.Module):
    """Multi-scale residual blocks"""
    def __init__(self, in_channels):
        super(MultiScaleResBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)  # The output channel is changed to 512
        self.conv5x5 = nn.Conv2d(in_channels, 512, kernel_size=5, padding=2)  # The output channel is changed to 512
        self.conv7x7 = nn.Conv2d(in_channels, 1024, kernel_size=7, padding=3)  # The output channel is changed to 1024
        self.attention = SEBlock(2048)  # 3×512 + 1024 = 2048
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        f3 = self.conv3x3(x)
        f5 = self.conv5x5(x)
        f7 = self.conv7x7(x)
        fused = torch.cat([f3, f5, f7], dim=1)  # The number of spliced channels is 2048
        fused = self.attention(fused)
        fused = self.dropout(fused)
        return x + fused  # Residual connections

class MSResNet(nn.Module):
    """MS-ResNet model"""
    def __init__(self, num_classes=19):  # WHU-RS19 has 19 categories
        super(MSResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.multi_scale_block = MultiScaleResBlock(2048)  # The number of ResNet-50 last layer output channels
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.multi_scale_block(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
import torch
import torch.nn as nn 
from ..utils import load_pretrained_weights


__all__ = [
    "ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152" 
]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, expansion, is_Bottleneck, stride):
        super(ResidualBlock, self).__init__()
        self.expansion = expansion
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.is_Bottleneck = is_Bottleneck

        # is_Bottleneck = True for all ResNet 50+
        if self.is_Bottleneck:
            # bottleneck
            # 1x1
            self.conv1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.middle_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.bn1 = nn.BatchNorm2d(self.middle_channels)

            # 3x3
            self.conv2 = nn.Conv2d(
                in_channels=self.middle_channels,
                out_channels=self.middle_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(self.middle_channels)

            # 1x1
            self.conv3 = nn.Conv2d(
                in_channels=self.middle_channels,
                out_channels=self.middle_channels * self.expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
            self.bn3 = nn.BatchNorm2d(self.middle_channels * self.expansion)

        else:
            # basicblock
            # 3x3
            self.conv1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.middle_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            )
            self.bn1 = nn.BatchNorm2d(self.middle_channels)

            # 3x3
            self.conv2 = nn.Conv2d(
                in_channels=self.middle_channels,
                out_channels=self.middle_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(self.middle_channels)

        self.relu = nn.ReLU(inplace=True)

        # if dim(x) == dim(F) => Identity function
        if self.in_channels == self.middle_channels * self.expansion:
            self.identity = True
        else:
            self.identity = False
            downsample_layer = []
            downsample_layer.append(nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.middle_channels * self.expansion,
                                              kernel_size=1, stride=stride, padding=0, bias=False))
            downsample_layer.append(nn.BatchNorm2d(self.middle_channels * self.expansion))
            self.downsample = nn.Sequential(*downsample_layer)

    def forward(self,x):
        in_x = x

        if self.is_Bottleneck:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
        else:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))

        if self.identity:
            x += in_x
        else:
            x += self.downsample(in_x)

        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, resnet_variant, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.channels_list = resnet_variant[0]
        self.repeatition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_Bottleneck = resnet_variant[3]

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, 
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(64 , self.channels_list[0], 
                                        self.repeatition_list[0], self.expansion, 
                                        self.is_Bottleneck, stride=1)
        self.layer2 = self._make_layers(self.channels_list[0] * self.expansion, self.channels_list[1],
                                         self.repeatition_list[1], self.expansion, 
                                         self.is_Bottleneck, stride=2)
        self.layer3 = self._make_layers(self.channels_list[1] * self.expansion, self.channels_list[2], 
                                        self.repeatition_list[2], self.expansion, 
                                        self.is_Bottleneck, stride=2)
        self.layer4 = self._make_layers(self.channels_list[2] * self.expansion, self.channels_list[3], 
                                        self.repeatition_list[3], self.expansion, 
                                        self.is_Bottleneck, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels_list[3] * self.expansion, num_classes)

    def forward(self,x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)
            return x
    
    def _make_layers(self, in_channels, middle_channels, num_repeat, expansion, is_Bottleneck, stride):
        layers = [] 
        layers.append(ResidualBlock(in_channels, middle_channels, 
                                    expansion, is_Bottleneck, stride=stride))
        for num in range(1, num_repeat):
            layers.append(ResidualBlock(middle_channels * expansion, middle_channels, 
                                        expansion, is_Bottleneck, stride=1))

        return nn.Sequential(*layers)



def resnet18(in_channels=3, num_classes=1000, pretrained=False):
    model = ResNet(([64, 128, 256, 512], [2, 2, 2, 2], 1, False), in_channels, num_classes)
    if pretrained:
        url = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

def resnet34(in_channels=3, num_classes=1000, pretrained=False):
    model = ResNet(([64, 128, 256, 512], [3, 4, 6, 3], 1, False), in_channels, num_classes)
    if pretrained:
        url = 'https://download.pytorch.org/models/resnet34-b627a593.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

def resnet50(in_channels=3, num_classes=1000, pretrained=False):
    model = ResNet(([64, 128, 256, 512], [3, 4, 6, 3], 4, True), in_channels, num_classes)
    if pretrained:
        url = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model 

def resnet101(in_channels=3, num_classes=1000, pretrained=False):
    model = ResNet(([64, 128, 256, 512], [3, 4, 23, 3], 4, True), in_channels, num_classes)
    if pretrained:
        url = 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

def resnet152(in_channels=3, num_classes=1000, pretrained=False):
    model = ResNet(([64, 128, 256, 512], [3, 8, 36, 3], 4, True), in_channels, num_classes)
    if pretrained:
        url = 'https://download.pytorch.org/models/resnet152-394f9c45.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

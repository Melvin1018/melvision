import torch
import torch.nn as nn
from ..utils import load_pretrained_weights


__all__ = [
    "VGG", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", 
    "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"
]

class VGG(nn.Module):
    def __init__(self, vgg_parameters, in_channels, num_classes, use_batch_norm=False):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_parameters, in_channels, use_batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
    def _make_layers(self, vgg_parameters, in_channels, use_batch_norm):
        layers = []
        filter_list, repeat_list = vgg_parameters  
        for filters, repeats in zip(filter_list, repeat_list):  
            for _ in range(repeats):  
                conv2d = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
                layers.append(conv2d)
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(filters))
                layers.append(nn.ReLU(inplace=True))
                in_channels = filters  
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))             
        return nn.Sequential(*layers)



def vgg11(in_channels=3, num_classes=1000, pretrained=False):
    model = VGG(([64, 128, 256, 512, 512], [1, 1, 2, 2, 2]), in_channels, num_classes, False)
    if pretrained:
        url = 'https://download.pytorch.org/models/vgg11-8a719046.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

def vgg11_bn(in_channels=3, num_classes=1000, pretrained=False):
    model = VGG(([64, 128, 256, 512, 512], [1, 1, 2, 2, 2]), in_channels, num_classes, True)
    if pretrained:
        url = 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

def vgg13(in_channels=3, num_classes=1000, pretrained=False):
    model = VGG(([64, 128, 256, 512, 512], [2, 2, 2, 2, 2]), in_channels, num_classes, False)
    if pretrained:
        url = 'https://download.pytorch.org/models/vgg13-19584684.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

def vgg13_bn(in_channels=3, num_classes=1000, pretrained=False):
    model = VGG(([64, 128, 256, 512, 512], [2, 2, 2, 2, 2]), in_channels, num_classes, True)
    if pretrained:
        url = 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

def vgg16(in_channels=3, num_classes=1000, pretrained=False):
    model = VGG(([64, 128, 256, 512, 512], [2, 2, 3, 3, 3]), in_channels, num_classes, False)
    if pretrained:
        url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

def vgg16_bn(in_channels=3, num_classes=1000, pretrained=False):
    model = VGG(([64, 128, 256, 512, 512], [2, 2, 3, 3, 3]), in_channels, num_classes, True)
    if pretrained:
        url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

def vgg19(in_channels=3, num_classes=1000, pretrained=False):
    model = VGG(([64, 128, 256, 512, 512], [2, 2, 4, 4, 4]), in_channels, num_classes, False)
    if pretrained:
        url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

def vgg19_bn(in_channels=3, num_classes=1000, pretrained=False):
    model = VGG(([64, 128, 256, 512, 512], [2, 2, 4, 4, 4]), in_channels, num_classes, True)
    if pretrained:
        url = 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
        pretrained_file, pretrained_state_dict = load_pretrained_weights(url)
        model.load_state_dict(pretrained_state_dict)
        print(f"Pre-trained weights {pretrained_file} loaded.")
    return model

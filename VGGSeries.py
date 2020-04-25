import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def conv3x3_br(in_channels, out_channels, norm_layer):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
    if norm_layer:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __int__(self, blocks, norm_layer, num_class= 1000):
        super(VGG, self).__int__()

        self.stage1 = self.make_layer(inplanes=3, planes=64, block_num=blocks[0], norm_layer=norm_layer)
        self.stage2 = self.make_layer(inplanes=64, planes=128, block_num=blocks[1], norm_layer=norm_layer)
        self.stage3 = self.make_layer(inplanes=128, planes=256, block_num=blocks[2], norm_layer=norm_layer)
        self.stage4 = self.make_layer(inplanes=256, planes=512, block_num=blocks[3], norm_layer=norm_layer)
        self.stage5 = self.make_layer(inplanes=512, planes=512, block_num=blocks[4], norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifer = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m.weight, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m.weight, nn.ReLU):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, inplanes, planes, block_num, norm_layer):
        layers = []
        layers.append(conv3x3_br(inplanes, planes, norm_layer=norm_layer))

        for i in range(1, block_num):
            layers.append(conv3x3_br(inplanes, planes, norm_layer=norm_layer))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def forwar(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.avgpool(out)
        out = torch.flatten(out)
        out = self.classifer(out)

        return out

def _vgg(arch, blocks, pretrained, progress, norn_layer, **kawargs):
    model=VGG(blocks, norn_layer)
    if pretrained:
        model_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(model_dict)
    return model

def vgg11_bn(pretrained=False, progress=True, **kawrgs):
    return _vgg("vgg11_bn", [1,1,2,2,2], pretrained, progress, norn_layer=True)

def vgg11(pretrained=False, progress=True, **kawrgs):
    return _vgg("vgg11_bn", [1,1,2,2,2], pretrained, progress, norn_layer=False)

def vgg13_bn(pretrained=False, progress=True, **kawrgs):
    return _vgg("vgg11_bn", [2,2,2,2,2], pretrained, progress, norn_layer=True)

def vgg13(pretrained=False, progress=True, **kawrgs):
    return _vgg("vgg11_bn", [2,2,2,2,2], pretrained, progress, norn_layer=False)

def vgg16_bn(pretrained=False, progress=True, **kawrgs):
    return _vgg("vgg11_bn", [2,2,3,3,3], pretrained, progress, norn_layer=True)

def vgg16(pretrained=False, progress=True, **kawrgs):
    return _vgg("vgg11_bn", [2,2,3,3,3], pretrained, progress, norn_layer=False)

def vgg19_bn(pretrained=False, progress=True, **kawrgs):
    return _vgg("vgg11_bn", [2,2,4,4,4], pretrained, progress, norn_layer=True)

def vgg19(pretrained=False, progress=True, **kawrgs):
    return _vgg("vgg11_bn", [2,2,4,4,4], pretrained, progress, norn_layer=False)
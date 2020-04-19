import torch
import  torch.nn as nn
from torch.hub import load_state_dict_from_url
"""
"""
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

#define 3x3 convolution function
def conv3x3(in_planes, planes, stride = 1, padding = 1):
    """
    @param in_planes: 卷积处理接收的featureMap的层数
    @param planes: 卷积处理后输出的featureMap层数
    @param stride: 卷积处理的步长
    @param padding: 输入的featureMap图像边缘扩充的像素数
    @return:卷积得到的featureMap
    同时需要清楚3X3的卷积处理卷积核（kernelsize）的大小为3
    """
    # 此处不加入bias是由于BN层的存在,由于BN层会把加入的偏置项归一化掉，所以即使加上也不会起作用，而且不用bias还可以节约内存的开销
    return nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False)

def conv1x1(in_planes, planes, stride = 1):
    """
    :param in_planes: 卷积处理接收的featureMap的层数
    :param planes: 卷积处理后输出的featureMap层数
    :param stride: 卷积处理的步长
    :return: 卷积得到的featureMap
    此处需要注意的是1x1的卷积操作不会使处理后的图像尺寸发生
    变化，所以不需要padding操作
    """
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    首先作者是想通过训练得到H(x)，但是想通过训练直接得到H(x)并不容易
    所以此处作者先对F(x)进行训练，然后在F(x)基础上加上残差 x得到H(x)所以
    就有了H(x) = F(x) + x,也就有了文中的F(x)=H(x)-x的操作了。这里首先再
    说一下BasicBlock的形状，此处一共需要进行两次卷积操作，分别是两个3x3的
    卷积核进行处理，这里个人感觉是借用了VGG大量使用3x3卷积操作的思想。
    第一次操作
        首先：3x3的卷积操作                conv3x3

        其次：做一个batchnormalization     nn.BatchNorm2d

        最后：进行relu的激活操作            nn.ReLU
    第二次操作
        首先：3x3的卷积操作                conv3x3

        其次：做一个batchnormalization     nn.BatchNorm2d.

        再次：对上文中提到的x进行升维操作    downsample

        再次：对结果进行F(x)+x的操作        out += identity
    """
    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_layer=None):
        expansion = 1  #由于此处每一个block内部没有维度变化所以令expansion = 1
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        sel.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride= stride)
        self.bn2 = norm_layer(planes)
        self.downsample=downsample
        self.sride=stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    此处与basic block的不同之处在于bock的内部结构是一个bottleneck卷积处理顺序
    每一个block中有三次卷积，首先是1x1的卷积核进处理，其次是3x3的卷积和处理，
    最后是1x1的卷积核进行处理的，另外最后一层1x1的卷积操作吧原来的channel个数
    增加到原来的4倍，比如之前一层channel个数是64，经过最后一层1x1卷积之后会变成
    256个channel
    第一次操作
        首先：1x1的卷积操作                conv1x1

        其次：做一个batchnormalization     nn.BatchNorm2d

        最后：进行relu的激活操作            nn.ReLU
    第二次操作
        首先：3x3的卷积操作                conv3x3

        其次：做一个batchnormalization     nn.BatchNorm2d.

        再次：做一个batchnormalization     nn.BatchNorm2d

    第三次操作
        首先：1x1的卷积操作                conv1x1

        其次：做一个batchnormalization     nn.BatchNorm2d.

        再次：对上文中提到的x进行升维操作    downsample

        再次：对结果进行F(x)+x的操作        out += identity
    """
    expansion = 4 #此处是由于Bottleneck中一个block中FeatureMap的维度变化比例是1:1:4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion,kernel_size=1, stride=stride)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet类的作用：
        1.是构建ResNet的基本框架
            1).根据论文的可知，RenNet首先是把输入的图像做一个7x7的卷积操作，
        得到一个有64个cahnnel组成的featureMap。
            2).其次是对得到的Feature Map做最大池化，得到64 channel，且每个channel
        图像尺寸为56x56的feature。
            3).最后再把得到64channel尺寸为56x56的featureMap进行4个stage的处理，
        其中每个stage中有不同的block个数，而这些block的个数决定了网络的层数。
            4).而不同的网络层数又会使用不同的block类型。block的类型一共有两种：
                第一种是basicblock，主要用于低于50层的网络
                第二种是bolttleblock，主要用于50层及以上的网络

        2.会实现每一层的网络结构
          这里用到的是函数封装，把构建网络的方法封装成make_layer函数
    """
    def __init__(self, block, layers, num_class = 1000, norm_layer = None):
        """
        :param block: 表示使用的block类型，这里有两种：
            1.BasicBlock
            2.BottlNeck
        :param layers: 此处传入的是列表，里面一共有四个参数分别表示四个stage， 每个数的数值表示stage中block的个数
        :param num_class: 表示最终分类的个数，这里默认的是1000个类别
        :param norm_layer: 是否做了归一化操作，弄人为否
        """
        super(ResNet, self).__init__()
        if norm_layer is not None: #判断是否做了归一化处理，若没有，下面进行归一化操作
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64

        #此处输入图像的的大小是224x224经过卷积处理之后图像大小变成112x112
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#做一次最大池化擦操作，最后输出图像大小为56x56

        self.layers1 = _make_layer(block, 64, layers[0])
        self.layers2 = _make_layer(block, 128, layers[1], stride=2)
        self.layers4 = _make_layer(block, 256, layers[2], stride=2)
        self.layers4 = _make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #此处对1全局平均池话得到的是一个1x1的核
        self.fc = nn.Linear(512*block.expansion, num_class) #由于上一步做了全局平均池化，此处输入的数据量只是卷积层的个数就行了

        for m in self.weights:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weights, model="fan_out", nonlinearity="relu")
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weights, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1):

        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                norm_layer(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, downsample, norm_layer))
        return nn.Sequential(*layers)

    def forwars(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.layers4(out)

        out = self.avgpool(out)
        out = self.fc(out)



def _resnet(arcgh, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers)
    if pretrained:
        static_url = load_state_dict_from_url(model_urls(arch),
                                              progress = progress)
        model.load_state_dict((static_url)
    return model

def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)





# https://github.com/hustzxd/dynamic-pruning/blob/main/train_baseline.py
import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ["cifar10_resnet20", "cifar10_resnet56"]

model_urls = {
    "cifar10_resnet56": "https://github.com/rhhc/zxd_releases/releases/download/Re/cifar10-resnet56-zxd-93.31-fewfw3.pth",
    "cifar10_resnet20": "https://github.com/rhhc/zxd_releases/releases/download/Re/cifar10-resnet20-eff-91.70-06a927.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CifarResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def cifar10_resnet20(pretrained=False, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = CifarResNet(BasicBlock, [3, 3, 3], 10)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["cifar10_resnet20"], map_location="cpu"))
    return model


def cifar10_resnet56(pretrained=False, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = CifarResNet(BasicBlock, [9, 9, 9], 10)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["cifar10_resnet56"], map_location="cpu"))
    return model

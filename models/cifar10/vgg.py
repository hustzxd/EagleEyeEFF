"""VGG11/13/16/19 in Pytorch."""
import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ["VGG", "cifar10_vggsmall", "cifar10_vgg16_bn", "cifar10_vgg19_bn"]

model_urls = {
    "vgg7": "https://github.com/rhhc/zxd_releases/releases/download/Re/cifar10-vggsmall-zxd-93.4-8943fa3.pth",
}


def cifar10_vggsmall(pretrained=False, **kwargs):
    """VGG small model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG("VGG7")
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg7"], map_location="cpu"))
    return model


cfg = {
    "VGG7": [128, 128, "M", 256, 256, "M", 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512 * 16, 10)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


defaultcfg = {
    11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    16: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
    ],
}


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CifarVGG(nn.Module):
    def __init__(self, depth=16, num_classes=10):
        super(CifarVGG, self).__init__()
        cfg = defaultcfg[depth]
        self.feature = self.make_layers(cfg)
        self.num_classes = num_classes
        self.classifier = nn.Linear(cfg[-1], num_classes)
        self._initialize_weights()

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [ConvBNReLU(in_channels, v, kernel_size=3, padding=1, bias=False)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def cifar10_vgg16_bn(pretrained=False, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = CifarVGG(16, 10)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg7"], map_location="cpu"))
    return model


def cifar10_vgg19_bn(pretrained=False, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = CifarVGG(19, 10)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg7"], map_location="cpu"))
    return model

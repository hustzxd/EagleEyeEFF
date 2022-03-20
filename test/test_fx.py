import time

import torch
import torch.nn as nn
import torchvision.models as models

import distiller

######################################################################
# For this tutorial, we are going to create a model consisting of convolutions
# and batch norms. Note that this model has some tricky components - some of
# the conv/batch norm patterns are hidden within Sequentials and one of the
# BatchNorms is wrapped in another Module.


class WrappedBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = nn.BatchNorm2d(1)

    def forward(self, x):
        return self.mod(x)


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.conv3 = nn.Conv2d(1, 1, 1)
        self.nested = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, 1),
        )
        self.wrapped = WrappedBatchNorm()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        y = self.conv3(x)
        x = self.conv2(x)
        x = self.nested(x)
        x = self.wrapped(x)
        x = torch.nn.functional.relu(x)
        return x


def test_bn_fusion():
    model = M()
    model.eval()
    fused_model = distiller.apputils.fuse_conv_bn(model)
    print(fused_model.code)
    inp = torch.randn(5, 1, 1, 1)
    # effp.print_graph(model)
    # effp.print_graph(fused_model)
    torch.testing.assert_allclose(fused_model(inp), model(inp))

    ######################################################################
    # Benchmarking our Fusion on ResNet18
    # ----------
    # We can test our fusion pass on a larger model like ResNet18 and see how much
    # this pass improves inference performance.

    rn18 = models.resnet18()
    rn18.eval()

    inp = torch.randn(10, 3, 224, 224)
    output = rn18(inp)

    fused_rn18 = distiller.apputils.fuse_conv_bn(rn18)

    torch.testing.assert_allclose(fused_rn18(inp), rn18(inp))

    # effp.print_graph(rn18)
    # effp.print_graph(fused_rn18)

    print("Unfused time: ", benchmark(rn18, inp))
    print("Fused time: ", benchmark(fused_rn18, inp))
    ######################################################################
    # As we previously saw, the output of our FX transformation is
    # (Torchscriptable) PyTorch code, we can easily `jit.script` the output to try
    # and increase our performance even more. In this way, our FX model
    # transformation composes with Torchscript with no issues.
    jit_rn18 = torch.jit.script(fused_rn18)
    print("jit time: ", benchmark(jit_rn18, inp))

    ############
    # Conclusion
    # ----------
    # As we can see, using FX we can easily write static graph transformations on
    # PyTorch code.
    #
    # Since FX is still in beta, we would be happy to hear any
    # feedback you have about using it. Please feel free to use the
    # PyTorch Forums (https://discuss.pytorch.org/) and the issue tracker
    # (https://github.com/pytorch/pytorch/issues) to provide any feedback
    # you might have.


def benchmark(
    model,
    inp,
    iters=20,
):
    for _ in range(10):
        model(inp)
    begin = time.time()
    for _ in range(iters):
        model(inp)
    return str(time.time() - begin)


def test_prune():
    import models.cifar10 as cifar10_extra_models

    model = cifar10_extra_models.__dict__["cifar10_vggsmall"]()
    model.eval()
    prune_model = distiller.prune(model)
    inp = torch.randn(10, 3, 32, 32)
    print("unpruned time: ", benchmark(model, inp))
    print("pruned time: ", benchmark(prune_model, inp))


if __name__ == "__main__":
    test_bn_fusion()
    test_prune()

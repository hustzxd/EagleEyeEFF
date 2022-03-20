import numpy as np
import torch


def model_device(model):
    """Determine the device the model is allocated on."""
    # Source: https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
    if next(model.parameters()).is_cuda:
        return "cuda"
    return "cpu"


def volume(tensor):
    """return the volume of a pytorch tensor"""
    if isinstance(tensor, torch.FloatTensor) or isinstance(tensor, torch.cuda.FloatTensor):
        return np.prod(tensor.shape)
    if isinstance(tensor, tuple) or isinstance(tensor, list):
        return np.prod(tensor)
    raise ValueError


def size_to_str(torch_size):
    """Convert a pytorch Size object to a string"""
    assert isinstance(torch_size, torch.Size) or isinstance(torch_size, tuple) or isinstance(torch_size, list)
    return "(" + (", ").join(["%d" % v for v in torch_size]) + ")"


def set_bn_eval(m):
    """[summary]
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
    https://github.com/pytorch/pytorch/issues/16149
        requires_grad does not change the train/eval mode,
        but will avoid calculating the gradients for the affine parameters (weight and bias).
        bn.train() and bn.eval() will change the usage of the running stats (running_mean and running_var).
    For detailed computation of Batch Normalization, please refer to the source code here.
    https://github.com/pytorch/pytorch/blob/83c054de481d4f65a8a73a903edd6beaac18e8bc/torch/csrc/jit/passes/graph_fuser.cpp#L232
    The input is normalized by the calculated mean and variance first.
    Then the transformation of w*x+b is applied on it by adding the operations to the computational graph.
    """
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()
    return


def set_bn_train(m):  # update running mean and running var
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.train()
    return


def reset_bn(m):  # reset running mean and running var
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.num_batches_tracked.data.fill_(0)
        m.running_mean.data.fill_(0)
        m.running_var.data.fill_(1)
    return


def set_bn_grad_false(m):
    """freeze \gamma and \beta in BatchNorm
    model.apply(set_bn_grad_false)
    optimizer = SGD(model.parameters())
    """
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        if m.affine:
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)


def set_param_grad_false(model):
    for name, param in model.named_parameters():  # same to set bn val? No
        if param.requires_grad:
            param.requires_grad_(False)
            print("frozen weights. shape:{}".format(param.shape))


def set_param_grad_true(model):
    for name, param in model.named_parameters():  # same to set bn val? No
        if param.requires_grad:
            param.requires_grad_(True)
            print("unfrozen weights. shape:{}".format(param.shape))

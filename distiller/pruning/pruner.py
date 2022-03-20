import copy
from enum import Enum
from typing import Any, Dict, Iterable, List, Tuple, Type

import ipdb
import torch
import torch.nn as nn

import distiller
import models._modules as my_nn

__all__ = ["prune_channel", "RankType"]


class RankType(Enum):
    L1Norm = 1
    Random = 2


def _rank_filter(weight: torch.Tensor, left_filter_num, rank_type=RankType.L1Norm):
    if rank_type == RankType.L1Norm:
        num_filters = weight.size(0)
        # First, reshape the weights tensor such that each channel (kernel) in the original
        # tensor, is now a row in the 2D tensor.
        view_2d = weight.view(num_filters, -1)
        filter_mags = torch.norm(view_2d, p=1, dim=1)
        bottomk, indices = torch.topk(filter_mags, left_filter_num, largest=True, sorted=True)
        left_filter_idx = torch.sort(indices)[0]
        return left_filter_idx

    elif rank_type == RankType.Random:
        num_filters = weight.size(0)
        rand = torch.rand(num_filters)
        bottomk, indices = torch.topk(rand, left_filter_num, largest=True, sorted=True)
        left_filter_idx = torch.sort(indices)[0]
        return left_filter_idx
    else:
        raise NotImplementedError


def prune_channel(
    model: torch.nn.Module,
    layer_prune_name: List,
    per_layer_s: List,
    rank_type: RankType,
    follower_layer_map=None,
) -> torch.nn.Module:
    fx_graph = distiller.FXGraph(model)
    # Step. 1 pruning and remember the leader layer names
    layer_to_prune_name = copy.deepcopy(layer_prune_name)
    for k in follower_layer_map:
        layer_to_prune_name.append(k)
    new_name_to_module_map = {}  # new !!
    for node in fx_graph.nodes:
        # If our current node isn't calling a Module then we can ignore it.
        if node.op != "call_module":
            continue
        if issubclass(fx_graph.get_node_module_type(node), nn.Conv2d):
            cur_conv = node
            pre_conv = fx_graph.get_pre_conv(cur_conv)
            cur_conv_module = fx_graph.get_node_module(cur_conv)
            cur_conv_module_name = fx_graph.get_module_name(cur_conv_module)
            m = cur_conv_module
            if pre_conv is None:
                if cur_conv_module_name in layer_to_prune_name:
                    # first conv && pruning is required
                    idx = layer_to_prune_name.index(cur_conv_module_name)
                    prune_ratio = per_layer_s[idx]
                    left_out_channel = m.out_channels - int(prune_ratio * m.out_channels)
                    left_filter_idx = _rank_filter(m.weight, left_out_channel, rank_type)
                    has_bias = m.bias is not None
                    if has_bias:
                        raise NotImplementedError
                    new_conv_module = my_nn.Conv2dCP(
                        m.in_channels,
                        left_out_channel,
                        m.kernel_size,
                        m.stride,
                        m.padding,
                        m.dilation,
                        groups=m.groups,
                        bias=has_bias,
                    )
                    new_conv_module.weight = torch.nn.Parameter(m.weight[left_filter_idx, :, :, :])
                    new_conv_module._tmp_filter_idx = left_filter_idx
                    new_conv_module._legacy_filter_num = m.out_channels
                    new_conv_module.to(distiller.model_device(m))
                    new_name_to_module_map[cur_conv_module_name] = new_conv_module
                    fx_graph.replace_node_module(cur_conv, new_conv_module)
                else:  # first conv and no pruning is required
                    m._tmp_filter_idx = _rank_filter(m.weight, m.out_channels, rank_type)
                    m._legacy_filter_num = m.out_channels
            else:  # not fisrt layer
                pre_conv_module = fx_graph.get_node_module(pre_conv)
                if cur_conv_module_name in layer_to_prune_name:
                    # inner conv && pruning is required
                    idx = layer_to_prune_name.index(cur_conv_module_name)
                    left_in_channel = pre_conv_module.out_channels
                    left_in_idx = pre_conv_module._tmp_filter_idx
                    if cur_conv_module_name in follower_layer_map:  # follower layer inherits from leader layer
                        leader_module = new_name_to_module_map[follower_layer_map[cur_conv_module_name]]
                        left_out_channel = leader_module.out_channels
                        left_filter_idx = leader_module._tmp_filter_idx  # inherited from the leader
                    else:
                        prune_ratio = per_layer_s[idx]
                        left_out_channel = m.out_channels - int(prune_ratio * m.out_channels)
                        left_filter_idx = _rank_filter(m.weight, left_out_channel, rank_type)
                    has_bias = m.bias is not None
                    if has_bias:
                        raise NotImplementedError
                    new_conv_module = my_nn.Conv2dCP(
                        left_in_channel,
                        left_out_channel,
                        m.kernel_size,
                        m.stride,
                        m.padding,
                        m.dilation,
                        groups=m.groups,
                        bias=has_bias,
                    )
                    new_conv_module.weight = torch.nn.Parameter(
                        m.weight[left_filter_idx, :, :, :][:, left_in_idx, :, :]
                    )
                    new_conv_module._tmp_filter_idx = left_filter_idx
                    new_conv_module._legacy_filter_num = m.out_channels
                    new_conv_module.to(distiller.model_device(m))
                    new_name_to_module_map[cur_conv_module_name] = new_conv_module
                    fx_graph.replace_node_module(cur_conv, new_conv_module)

                else:  # inner conv but pruning is not required
                    if m.groups == m.in_channels == m.in_channels:
                        # inner layer && depth-wise conv2d && no pruning
                        new_conv_module = copy.deepcopy(m)
                        new_conv_module.in_channels = pre_conv_module.out_channels
                        new_conv_module.out_channels = pre_conv_module.out_channels
                        new_conv_module.groups = pre_conv_module.out_channels
                        new_conv_module.weight = torch.nn.Parameter(
                            cur_conv_module.weight[
                                pre_conv_module._tmp_filter_idx,
                                :,
                                :,
                                :,
                            ]
                        )
                        new_conv_module._tmp_filter_idx = pre_conv_module._tmp_filter_idx
                        new_conv_module._legacy_filter_num = m.in_channels  # depth-wise conv
                        new_conv_module.to(distiller.model_device(m))
                        fx_graph.replace_node_module(cur_conv, new_conv_module)
                    else:  # inner layer && norm conv2d && no pruning
                        new_conv_module = copy.deepcopy(m)
                        new_conv_module.in_channels = pre_conv_module.out_channels
                        new_conv_module.weight = torch.nn.Parameter(
                            cur_conv_module.weight[
                                :,
                                pre_conv_module._tmp_filter_idx,
                                :,
                                :,
                            ]
                        )
                        new_conv_module._tmp_filter_idx = _rank_filter(m.weight, m.out_channels, rank_type)
                        new_conv_module._legacy_filter_num = m.out_channels  # depth-wise conv
                        new_conv_module.to(distiller.model_device(m))
                        fx_graph.replace_node_module(cur_conv, new_conv_module)

        if issubclass(fx_graph.get_node_module_type(node), nn.BatchNorm2d):
            cur_bn = node
            pre_conv = fx_graph.get_pre_conv(cur_bn)
            if pre_conv is not None:  # prune bn according to pre conv
                pre_conv_module = fx_graph.get_node_module(pre_conv)
                m = fx_graph.get_node_module(cur_bn)
                new_bn_module = copy.deepcopy(m)
                if m.affine:
                    new_bn_module.weight = torch.nn.Parameter(m.weight[pre_conv_module._tmp_filter_idx])
                    new_bn_module.bias = torch.nn.Parameter(m.bias[pre_conv_module._tmp_filter_idx])
                new_bn_module.register_buffer("running_mean", m.running_mean[pre_conv_module._tmp_filter_idx])
                new_bn_module.register_buffer("running_var", m.running_var[pre_conv_module._tmp_filter_idx])
                new_bn_module.to(distiller.model_device(pre_conv_module))
                fx_graph.replace_node_module(cur_bn, new_bn_module)
        if issubclass(fx_graph.get_node_module_type(node), nn.Linear):
            cur_fc = node
            # pre_fc = TODO if there are multiple fc layers.
            pre_conv = fx_graph.get_pre_conv(cur_bn)
            if pre_conv is not None:  # prune bn according to pre conv
                pre_conv_module = fx_graph.get_node_module(pre_conv)
                cur_fc_module = fx_graph.get_node_module(cur_fc)
                has_bias = cur_fc_module.bias is not None
                new_fc_module = copy.deepcopy(cur_fc_module)
                h_w_multipy = new_fc_module.in_features // pre_conv_module._legacy_filter_num
                new_fc_module.in_features = pre_conv_module.out_channels * h_w_multipy
                new_fc_module.weight = torch.nn.Parameter(
                    cur_fc_module.weight.view(cur_fc_module.out_features, pre_conv_module._legacy_filter_num, -1)[
                        :, pre_conv_module._tmp_filter_idx, :
                    ].view(cur_fc_module.out_features, -1)
                )
                new_fc_module.to(distiller.model_device(cur_fc_module))
                fx_graph.replace_node_module(cur_fc, new_fc_module)

    fx_graph.lint_recompile()
    return fx_graph.fx_model

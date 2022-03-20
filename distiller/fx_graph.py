import copy
from typing import Any, Dict, Iterable, Tuple, Type

import torch
import torch.fx as fx
import torch.nn as nn

import distiller


class FXGraph:
    # Test on Pytorch 1.10.0
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = copy.deepcopy(model)  # Use new model
        self.fx_model: fx.GraphModule = fx.symbolic_trace(self.model)
        self.nodes = self.fx_model.graph.nodes

    @property
    def modules(self):
        return dict(self.fx_model.named_modules())

    def print_graph(self):
        self.fx_model.graph.print_tabular()

    def lint_recompile(self):
        self.fx_model.graph.lint()
        self.fx_model.recompile()
        return self.fx_model

    def get_next_nodes(self, node):
        output_nodes = []
        for n in node.users:
            output_nodes.append(n)
        return output_nodes

    def get_pre_nodes(self, node):  # TODO: add need two inputs
        # if node.op == 'call_function':
        #     ipdb.set_trace()
        #     return node.args[0]
        return node.args

    def get_node_module(self, node):
        return self.modules[node.target]

    def get_node_module_type(self, node):
        return type(self.get_node_module(node))

    def get_module_name(self, module):
        return distiller.model_find_module_name(self.model, module)

    def get_module(self, name):  # get old module
        return distiller.model_find_module(self.model, name)

    def get_pre_conv(self, node):
        pre_nodes = self.get_pre_nodes(node)
        if len(pre_nodes) == 0:
            return None
        for p_n in pre_nodes:
            if p_n.op == "call_module" and issubclass(self.get_node_module_type(p_n), nn.Conv2d):
                return p_n
            else:
                return self.get_pre_conv(p_n)

    def get_pre_fc(self, node):
        print("To be tested!!!!!!!!!")
        pre_nodes = self.get_pre_nodes(node)
        if len(pre_nodes) == 0:
            return None
        for p_n in pre_nodes:
            if p_n.op == "call_module" and issubclass(self.get_node_module_type(p_n), nn.Linear):
                return p_n
            else:
                return self.get_pre_fc(p_n)

    def _parent_name(target: str) -> Tuple[str, str]:
        """
        Splits a qualname into parent path and last atom.
        For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
        """
        *parent, name = target.rsplit(".", 1)
        return parent[0] if parent else "", name

    def replace_node_module(self, node: fx.Node, new_module: torch.nn.Module):
        assert isinstance(node.target, str)
        parent_name, name = FXGraph._parent_name(node.target)
        setattr(self.modules[parent_name], name, new_module)

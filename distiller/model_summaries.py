#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Model statistics summaries.

    - weights sparsities
    - optimizer state
    - model details
"""
from functools import partial

import pandas as pd
import torch
import torch.optim
from tabulate import tabulate

import distiller

__all__ = [
    "performance_summary",
    "model_performance_summary",
    "model_performance_tbl_summary",
]

# Performance data collection  code follows from here down
def conv_visitor(self, input, output, df, model, memo):
    assert isinstance(self, torch.nn.Conv2d)
    if self in memo:
        return

    weights_vol = distiller.volume(self.weight)
    # weights_vol = self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1]

    # Multiply-accumulate operations: MACs = volume(OFM) * (#IFM * K^2) / #Groups
    # Bias is ignored
    macs = distiller.volume(output) * (self.in_channels / self.groups * self.kernel_size[0] * self.kernel_size[1])
    attrs = "k=" + "(" + (", ").join(["%d" % v for v in self.kernel_size]) + ")"
    mod_name = distiller.model_find_module_name(model, self)
    module_visitor(self, input, output, df, model, weights_vol, macs, attrs)


def fc_visitor(self, input, output, df, model, memo):
    assert isinstance(self, torch.nn.Linear)
    if self in memo:
        return

    # Multiply-accumulate operations: MACs = #IFM * #OFM
    # Bias is ignored
    weights_vol = macs = self.in_features * self.out_features
    module_visitor(self, input, output, df, model, weights_vol, macs)


def module_visitor(self, input, output, df, model, weights_vol, macs, attrs=None):
    in_features_shape = input[0].size()
    out_features_shape = output.size()

    mod_name = distiller.model_find_module_name(model, self)
    df.loc[len(df.index)] = [
        mod_name,
        self.__class__.__name__,
        attrs if attrs is not None else "",
        distiller.size_to_str(in_features_shape),
        distiller.volume(input[0]),
        distiller.size_to_str(out_features_shape),
        distiller.volume(output),
        int(weights_vol),
        int(macs),
    ]


def model_performance_summary(model, dummy_input, batch_size=1):
    """Collect performance data"""

    def install_perf_collector(m):
        if isinstance(m, torch.nn.Conv2d):
            hook_handles.append(m.register_forward_hook(partial(conv_visitor, df=df, model=model, memo=memo)))
        elif isinstance(m, torch.nn.Linear):
            hook_handles.append(m.register_forward_hook(partial(fc_visitor, df=df, model=model, memo=memo)))

    df = pd.DataFrame(
        columns=[
            "Name",
            "Type",
            "Attrs",
            "IFM",
            "IFM volume",
            "OFM",
            "OFM volume",
            "Weights volume",
            "MACs",
        ]
    )

    hook_handles = []
    memo = []

    # model = distiller.make_non_parallel_copy(model)
    model.apply(install_perf_collector)
    # Now run the forward path and collect the data
    dummy_input = dummy_input.to(distiller.model_device(model))
    model(dummy_input)
    # Unregister from the forward hooks
    for handle in hook_handles:
        handle.remove()

    return df


def model_performance_tbl_summary(model, dummy_input, batch_size):
    df = model_performance_summary(model, dummy_input, batch_size)
    t = tabulate(df, headers="keys", tablefmt="psql", floatfmt=".5f")
    return t


def performance_summary(model, dummy_input, opt=None, prefix=""):
    try:
        df = distiller.model_performance_summary(model.module, dummy_input)
    except AttributeError:
        df = distiller.model_performance_summary(model, dummy_input)
    new_entry = {
        "Name": ["Total"],
        "MACs": [df["MACs"].sum()],
    }
    MAC_total = df["MACs"].sum()
    param_total = df["Weights volume"].sum()
    return MAC_total, param_total, df

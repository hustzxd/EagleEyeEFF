import json
from collections import OrderedDict


def load_pre_state_dict(model, original_state_dict, key_map=None):
    if not isinstance(key_map, OrderedDict):
        with open("models/weight_keys_map/{}".format(key_map)) as rf:
            key_map = json.load(rf)
    for k, v in key_map.items():
        if "num_batches_tracked" in k:
            continue
        else:
            print("{} <== {}".format(k, v))
            model.state_dict()[k].copy_(original_state_dict[v])

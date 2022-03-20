import argparse
import os

import google.protobuf as pb
import google.protobuf.text_format

from proto import efficient_pytorch_pb2 as eppb

parser = argparse.ArgumentParser(description="EfficientPyTorch Begin")
parser.add_argument("hp", type=str, help="File path to save hyperparameter configuration")
args = parser.parse_args()
assert os.path.exists(args.hp)
hp = eppb.HyperParam()
with open(args.hp, "r") as rf:
    pb.text_format.Merge(rf.read(), hp)
command = "python {} --hp {}".format(hp.main_file, args.hp)
os.system(command)

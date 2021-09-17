import argparse
import os
import sys

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

parser = argparse.ArgumentParser(
    description="Training script",
    fromfile_prefix_chars="@",
    conflict_handler="resolve",
)
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument("--root", default="./", type=str, help="path to root")
parser.add_argument("--train_path", "--train-path", default="dataset/train.csv", type=str, help="path to training data")
parser.add_argument("--test_path", "--test-path", default="dataset/pretest.csv", type=str, help="path to testing data")
parser.add_argument("--trainpred_path", "--trainpred-path", default="pred-train/", type=str, help="path to training data")
parser.add_argument("--testpred_path", "--testpred-path", default="pred-test/", type=str, help="path to testing data")
parser.add_argument("--checkpoint_path", "--checkpoint-path", default="checkpoint/", type=str, help="path to testing data")

parser.add_argument("--mode", default="train", type=str, help="path to testing data")

parser.add_argument("--seed", default=0, type=int, help="number of fold")

parser.add_argument("--fold", default=5, type=int, help="number of fold")
parser.add_argument("--pretrainer", default=False, help="if set, will use pretrainer for tabnet", action="store_true")

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = "@" + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()
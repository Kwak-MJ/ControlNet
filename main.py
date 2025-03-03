import argparse
from Generation import *

parser = argparse.ArgumentParser(description="ControlNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--type", default="electric", type=str, dest="type")

parser.add_argument("--seed", default=10, type=int, dest="seed")

parser.add_argument("--data_dir", default="None", type=str, dest="data_dir")

parser.add_argument("--img_path", default="None", type=str, dest="img_path")

parser.add_argument("--result_dir", default="./result",
                    type=str, dest="result_dir")


args = parser.parse_args()

if __name__ == "__main__":
    generation(args)
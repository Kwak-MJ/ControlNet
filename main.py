import argparse
from Generation import *

parser = argparse.ArgumentParser(description="ControlNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="canny",
                    choices=["canny", "pose"], type=str, dest="mode")

parser.add_argument("--prompt", default="masterpiece, best quality, ultra-detailed, illustration, a person with a yellow lightning and spark", type=str, dest="prompt")

parser.add_argument("--num_steps", default=30, type=int, dest="num_steps")

parser.add_argument("--seed", default=10, type=int, dest="seed")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")

parser.add_argument("--result_dir", default="./result",
                    type=str, dest="result_dir")

args = parser.parse_args()

if __name__ == "__main__":
    generation(args)
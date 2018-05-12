import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import lib
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, dest="input_dir", help="Input dir or file path")
    parser.add_argument("-o", required=True, dest="output_file", help="Output json file")
    args = parser.parse_args()

    codes = lib.load_repository(args.input_dir)
    index = lib.index_maker.make_index(codes)

    with open(args.output_file, "w") as f:
        json.dump(index, f)
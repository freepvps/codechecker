import model
import os
import argparse
import json


def dir_search(path):
    yield path
    if os.path.isdir(path):
        sub = os.listdir(path)
        for x in sub:
            for r in dir_search(os.path.join(path, x)):
                yield r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, dest="input_dir", help="Input dir or file path")
    parser.add_argument("-o", required=True, dest="output_file", help="Output json file")
    args = parser.parse_args()

    codes = []
    for path in dir_search(args.input_dir):
        print(path)
        if path.lower().endswith("java") and os.path.isfile(path):
            with open(path) as f:
                codes.append(f.read())
    index = model.make_index(codes)

    with open(args.output_file, "w") as f:
        json.dump(index, f)
from pathlib import Path
from seutil import IOUtils
import ijson
import random
import sys


def split_project(method_file: Path, output_dir: Path):
    proj_list = set()
    with open(method_file, "r") as f:
        objects = ijson.items(f, "item")
        for o in objects:
            proj_list.add(o["prj_name"])
    num_proj = len(proj_list)
    proj_list = list(proj_list)
    print(f"Number of total proj is {num_proj}")
    random.shuffle(proj_list)
    train_index = round(num_proj*0.8)
    valid_index = train_index + round(num_proj*0.1)
    training_projs = proj_list[: train_index]
    valid_projs = proj_list[train_index: valid_index]
    test_projs = proj_list[valid_index:]
    train_method_data = []
    valid_method_data = []
    test_method_data = []
    with open(method_file, "r") as f:
        objects = ijson.items(f, "item")
        for o in objects:
            if o["prj_name"] in training_projs:
                train_method_data.append(o)
            elif o["prj_name"] in valid_projs:
                valid_method_data.append(o)
            elif o["prj_name"] in test_projs:
                test_method_data.append(o)
    print(f"Number of train method is {len(train_method_data)}")
    print(f"Number of val method is {len(valid_method_data)}")
    print(f"Number of test method is {len(test_method_data)}")
    IOUtils.dump(output_dir/"train-method-data.json", train_method_data)
    IOUtils.dump(output_dir/"val-method-data.json", valid_method_data)
    IOUtils.dump(output_dir/"test-method-data.json", test_method_data)

if __name__ == "__main__":
    src_method_file = sys.argv[1]
    output_dir = sys.argv[2]
    split_project(Path(src_method_file), Path(output_dir))

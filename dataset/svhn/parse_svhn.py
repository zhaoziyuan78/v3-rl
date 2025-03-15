import os
import argparse

import svhnl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/svhn")
    args = parser.parse_args()
    data_dir = args.data_dir

    print("Doing the job... Laggy...")

    test_mat_path = os.path.join(data_dir, "test/digitStruct.mat")
    svhnl.ann_to_json(
        file_path=test_mat_path,
        save_path=os.path.join(data_dir, "svhn_test_ann.json"),
        bbox_type="normalize",
    )
    train_mat_path = os.path.join(data_dir, "train/digitStruct.mat")
    svhnl.ann_to_json(
        file_path=train_mat_path,
        save_path=os.path.join(data_dir, "svhn_train_ann.json"),
        bbox_type="normalize",
    )
    extra_mat_path = os.path.join(data_dir, "extra/digitStruct.mat")
    svhnl.ann_to_json(
        file_path=extra_mat_path,
        save_path=os.path.join(data_dir, "svhn_extra_ann.json"),
        bbox_type="normalize",
    )

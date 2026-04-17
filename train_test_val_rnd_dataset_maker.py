import os
import random
import argparse


def move_dataset_part(files_names, base_dir, target_dir):
    for file in files_names:
        txt_name = os.path.splitext(file)[0] + ".txt"
        if os.path.isfile(os.path.join(base_dir, txt_name)):
            os.rename(
                os.path.join(base_dir, txt_name),
                os.path.join(target_dir, txt_name),
            )
            os.rename(
                os.path.join(base_dir, file),
                os.path.join(target_dir, file),
            )


parser = argparse.ArgumentParser(description="Split dataset to train, test, val")
parser.add_argument( "--dir", "-d", type=str, help="Dataset directory" )
parser.add_argument( "--out", "-o", type=str, help="Output directory" )
args = parser.parse_args()


test_val_part = 0.3

imgs_list = sorted(
    [
        d
        for d in os.listdir(args.dir)
        if d.endswith(".jpg") or d.endswith(".jpeg") or d.endswith(".png")
    ]
)

train_out_dir = os.path.join(args.out, "train")
test_out_dir = os.path.join(args.out, "test")
val_out_dir = os.path.join(args.out, "val")
if not os.path.exists(train_out_dir):
    os.mkdir(train_out_dir)
if not os.path.exists(test_out_dir):
    os.mkdir(test_out_dir)
if not os.path.exists(val_out_dir):
    os.mkdir(val_out_dir)


move_part_test_val = random.sample(imgs_list, int(len(imgs_list) * test_val_part))
move_part_test = set(random.sample(move_part_test_val, int(len(move_part_test_val) * 0.5)))
move_part_val = list(set(move_part_test_val).difference(move_part_test))
move_part_train = list(set(imgs_list).difference(set(move_part_test_val)))

print("train %s / test %s / val %s  | total %s"%(len(move_part_train), len(move_part_test), len(move_part_val), len(imgs_list)))
move_dataset_part(move_part_train, args.dir, train_out_dir)
move_dataset_part(move_part_val, args.dir, val_out_dir)
move_dataset_part(move_part_test, args.dir, test_out_dir)
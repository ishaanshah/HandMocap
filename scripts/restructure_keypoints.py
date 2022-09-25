import json
import os
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--root_dir", required=True, type=str)

args = parser.parse_args()
root_dir = args.root_dir

with open(os.path.join(root_dir, "keypoint_id.json")) as f:
    keypoint_id = json.load(f)

keypoint_dir = os.path.join(root_dir, "keypoints_2d")
for idx, (camera_name, frame_id) in tqdm(enumerate(keypoint_id)):
    with open(os.path.join(keypoint_dir, f"{idx}_keypoints.json")) as f:
        data = json.load(f)
    
    subdir_name = os.path.join(keypoint_dir, camera_name)
    if not os.path.exists(subdir_name):
        os.makedirs(subdir_name)

    with open(os.path.join(subdir_name, f"{frame_id}.json"), "w") as f:
        json.dump(data, f)

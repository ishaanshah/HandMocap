import json
import os
import jax.numpy as jnp
from glob import glob
from argparse import ArgumentParser
from src.skeleton import KinematicChain
from src.utils.plot import plot_keypoints_3d
from src.utils.misc import natural_sort

parser = ArgumentParser()
parser.add_argument("--rest_pose", required=True)
parser.add_argument("--keypoints_dir", required=True)
args = parser.parse_args()

with open(args.rest_pose, "r") as f:
    bones = json.load(f)
    
chain = KinematicChain(bones["bones"], bones["root"])
joints = chain.forward()

keypoints_path = natural_sort(glob("*.json", root_dir=args.keypoints_dir))
keypoints = []
for keypoint_path in keypoints_path:
    with open(os.path.join(args.keypoints_dir, keypoint_path)) as f:
        keypoints.append(json.load(f))

keypoints = jnp.asarray(keypoints)
chain.update_bone_lengths(keypoints)
plot_keypoints_3d([joints, chain.forward()], "rest_pose", (-2, 2))
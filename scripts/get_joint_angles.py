import json
import os
import jax.numpy as jnp
import shutil
import logging
import trimesh
import numpy as np
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser
from src.skeleton import KinematicChain
from src.utils.plot import plot_keypoints_3d
from src.utils.misc import natural_sort

parser = ArgumentParser()
parser.add_argument("--rest_pose", required=True)
parser.add_argument("--root_dir", required=True)
parser.add_argument("--point_cloud", action="store_true")
args = parser.parse_args()

with open(args.rest_pose, "r") as f:
    bones = json.load(f)
    
chain = KinematicChain(bones["bones"], bones["root"])
angles = np.zeros((len(bones["bones"]), 3))

keypoints_dir = os.path.join(os.path.join(args.root_dir, "keypoints_3d"))
keypoints_path = natural_sort(glob("*.json", root_dir=keypoints_dir))
keypoints = []
for keypoint_path in keypoints_path:
    with open(os.path.join(keypoints_dir, keypoint_path)) as f:
        keypoints.append(json.load(f))

assert len(keypoints) != 0, "Atleast one frame needed"

keypoints = jnp.asarray(keypoints)
chain.update_bone_lengths(keypoints)

output_dir = os.path.join(args.root_dir, "joint_angles")
logging.warning(f"Deleting files at {output_dir}")
try:
    shutil.rmtree(output_dir)
except:
    pass
os.makedirs(output_dir)

if args.point_cloud:
    point_cloud_dir = os.path.join(args.root_dir, "point_cloud_ik")
    try:
        shutil.rmtree(point_cloud_dir)
    except FileNotFoundError:
        pass

    os.makedirs(point_cloud_dir)

angles = np.zeros((21, 3))

for frame in tqdm(range(keypoints.shape[0])):
    if jnp.isclose(keypoints[frame,0,3], 0):
        logging.warning(f"Root bone not detected, skipping")
        continue

    # Zero center the root bone
    keypoints_z = keypoints[frame,:,:3] - keypoints[frame,0,:3] + jnp.asarray([0, 0, chain.bones["bone_0"]["len"]])
    target = jnp.vstack([jnp.zeros(3), keypoints_z])
    to_use = jnp.hstack([True, ~jnp.isclose(keypoints[frame,:,3], 0)])
    params = chain.IK(target, max_iter=100, mse_threshold=1e-6, to_use=to_use)
    ik_keyp, heads, tails = chain.forward(params)
    with open(os.path.join(output_dir, keypoints_path[frame]), "w") as f:
        json.dump(params.tolist(), f)
    
    if args.point_cloud:
        pcd = trimesh.PointCloud(ik_keyp)
        _ = pcd.export(os.path.join(args.root_dir, "point_cloud_ik", f"{frame:08d}.ply"))

    # chain.plot_skeleton(params, target)
import numpy as np
import joblib
import json
import os
from src.triangulate import triangulate_joints, simple_processor, ransac_processor
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--keypoints_dir", type=str, required=True)
parser.add_argument("--camera_params", type=str, required=True)
parser.add_argument("--handedness", choices=["right", "left"], required=True)
parser.add_argument("--min_joints", type=int, default=10)
parser.add_argument("--conf_thresh_start", type=float, default=0.75)
parser.add_argument("--method", choices=["ransac", "simple"], default="ransac")
parser.add_argument("--output_dir", type=str)

args = parser.parse_args()

# Create projection matrices
params = joblib.load(args.camera_params)
proj_mats = {}
for cam_name, cam_param in params[str(0)].items():
    K =  np.identity(3)
    K[0,0] = cam_param['K'][0]
    K[1,1] = cam_param['K'][1]
    K[0,2] = cam_param['K'][2]
    K[1,2] = cam_param['K'][3]
    proj_mats[cam_name] = K @ cam_param['extrinsics_opencv']

# Get frames
frames = list(sorted([frame.split(".")[0] for frame in os.listdir(os.path.join(args.keypoints_dir, cam_name)) if "json" in frame]))

processor = ransac_processor if args.method == "ransac" else simple_processor

if not args.output_dir:
    print("WARNING: Output directory not provided, not writing predicted keypoints")

mean_residuals = []
all_keypoints = []
for frame in frames:
    # Get 2D keypoints
    keypoints = {}
    for cam_name in proj_mats.keys():
        with open(os.path.join(args.keypoints_dir, cam_name, f"{frame}.json"), "r") as f:
            data = json.load(f)
            keypoints[cam_name] = np.array(data['people'][0][f'hand_{args.handedness}_keypoints_2d']).reshape(-1, 3)
    
    num_joints = keypoints[cam_name].shape[0]
    
    keypoints3d, residuals = triangulate_joints(keypoints, proj_mats, processor=processor, residual_threshold=10, min_samples=5)
    all_keypoints.append(keypoints3d.tolist())
    mean_residuals.append(residuals.mean())
    print(f"INFO: Mean reprojection error for frame {frame}: {residuals.mean()}")

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

        with open(os.path.join(args.output_dir, f"{frame}.json"), "w") as f:
            json.dump(np.hstack((keypoints3d, residuals[:,None])).tolist(), f)

print(f"INFO: Mean reprojection error over all frames: {sum(mean_residuals) / len(mean_residuals)}")

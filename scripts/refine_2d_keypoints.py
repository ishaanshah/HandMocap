import numpy as np
import joblib
import cv2
import json
import os
from argparse import ArgumentParser
from src.utils import project

parser = ArgumentParser()
parser.add_argument("--keypoints_dir", type=str, required=True)
parser.add_argument("--camera_params", type=str, required=True)
parser.add_argument("--image_dir", type=str, required=True)
parser.add_argument("--frame", type=str)
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
if not args.frame:
    print("INFO: Processing all frames")
    frames = list(sorted([frame.split(".")[0] for frame in os.listdir(os.path.join(args.image_dir, cam_name)) if "png" in frame]))
else:
    print(f"INFO: Processing frame {args.frame}")
    frames = [ args.frame ]

if not args.output_dir:
    print("WARNING: Output directory not provided, not writing refined keypoints")

mean_residuals = []
all_keypoints = []
for frame in frames:
    # Get 3D keypoints
    with open(os.path.join(args.keypoints_dir, f"{frame}.json"), "r") as f:
        keypoints = np.asarray(json.load(f))
    
    num_joints = keypoints.shape[0]

    cam_names = list(sorted(proj_mats.keys()))

    for cam_name in cam_names:
        img = cv2.imread(os.path.join(args.image_dir, cam_name, f"{frame}.png"), cv2.IMREAD_UNCHANGED)
        keypoints2d = project(keypoints[:,:3], np.asarray([proj_mats[cam_name]]))
        for keypoint2d in keypoints2d[0]:
            cv2.circle(img, (int(keypoint2d[0]), int(keypoint2d[1])), radius=3, color=(255, 0, 0), thickness=-1)
        cv2.imshow("Refine keypoints", img)
        cv2.waitKey(0)
    
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

        with open(os.path.join(args.output_dir, f"{frame}.json"), "w") as f:
            json.dump(np.hstack((keypoints3d, residuals[:,None])).tolist(), f)

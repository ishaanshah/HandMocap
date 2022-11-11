# From Python
# It requires OpenCV installed for Python
import cv2
import argparse
import glob
import os
import json
import numpy as np
from numpy.lib import recfunctions as rfn
from tqdm import tqdm
from openpose import pyopenpose as op
from src.detect_hand import detect_hand
from src.utils.misc import natural_sort
from src.utils.params import read_params, get_intr

debug = False 

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True)
parser.add_argument("--model_dir", required=True)
parser.add_argument("--glob_pattern", required=False, default="**/*.jpg")
parser.add_argument("--filter", required=False, default="")
parser.add_argument("--handedness", required=False, choices=["left", "right"], default="right")
parser.add_argument("--dump_pred_image", action="store_true")
parser.add_argument("--cam_params", required=False)
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = args[0].model_dir
params["hand"] = True
params["hand_detector"] = 2
params["body"] = 0
params["write_json"] = os.path.join(args[0].root_dir, "keypoints_2d")

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Read image and find rectangle locations
image_dir = os.path.join(args[0].root_dir, "image")
images_to_process = natural_sort(glob.glob(args[0].glob_pattern, root_dir=image_dir))

# Store the order of keypoints
keypoint_id = []
if args[0].dump_pred_image:
    os.makedirs(os.path.join(args[0].root_dir, "pred_images"), exist_ok=True)

# Get intrinsics if given
params = None
undist_params = []
if args[0].cam_params:
    params = read_params(args[0].cam_params)

keypoints = {}
images_to_process = list(filter(lambda x: args[0].filter in x, images_to_process))
cam_idx = 0
for image_path in tqdm(images_to_process):
    camera_name = os.path.dirname(image_path)
    img = cv2.imread(os.path.join(image_dir, image_path))

    # Undistort if camera parameters are given
    if params is not None and params[cam_idx]['cam_name'] == camera_name:
        intr, dist = get_intr(params[cam_idx])
        h, w = img.shape[:2]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(intr, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(img, intr, dist, None, newcameramtx)
        img = dst.copy()
        undist_params.append((newcameramtx[0,0], newcameramtx[1,1], newcameramtx[0,2], newcameramtx[1,2]))
        cam_idx += 1

    bbox = detect_hand(img)

    # TODO: Incorparate hand information, for now only right hand
    bbox = [ op.Rectangle(0, 0, 0, 0), op.Rectangle(*bbox) ]
    hand_rectangles = [
        bbox if args[0].handedness == "right" else list(reversed(bbox))
    ]

    # Create new datum
    datum = op.Datum()
    datum.cvInputData = img 
    datum.handRectangles = hand_rectangles

    # Process and display image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    if debug:
        print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
        cv2.imshow("OpenPose Prediction", datum.cvOutputData)
        ch = cv2.waitKey(0)
        if ch == 27:
            break

    if args[0].dump_pred_image:
        cv2.imwrite(os.path.join(args[0].root_dir, "pred_images", os.path.basename(image_path)), datum.cvOutputData)

    keypoints[image_path.split("/")[0]] = [datum.handKeypoints[0].tolist(), datum.handKeypoints[1].tolist()]

    # Specific to our dataset
    frame_id = os.path.basename(image_path).split(".")[0]
    keypoint_id.append((camera_name, frame_id))

if undist_params:
    undist_params = np.asarray(undist_params)
    params = rfn.append_fields(params, ("fx_undist", "fy_undist", "cx_undist", "cy_undist"), (undist_params[:,0], undist_params[:,1], undist_params[:,2], undist_params[:,3]), dtypes=(float, float, float, float))
    np.savetxt(args[0].cam_params, params, fmt="%s", header=" ".join(params.dtype.fields))

with open(os.path.join(args[0].root_dir, "keypoint_id.json"), "w") as f:
    json.dump(keypoint_id, f)

with open(os.path.join(args[0].root_dir, "keypoints_2d", "all.json"), "w") as f:
    json.dump(keypoints, f)

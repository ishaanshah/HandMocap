# From Python
# It requires OpenCV installed for Python
import cv2
import argparse
import glob
import os
import json
from tqdm import tqdm
from openpose import pyopenpose as op
from src.detect_hand import detect_hand

debug = False

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True)
parser.add_argument("--model_dir", required=True)
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
images_to_process = glob.glob("**/*.png", root_dir=image_dir)

# Store the order of keypoints
keypoint_id = []
for image_path in tqdm(images_to_process):
    img = cv2.imread(os.path.join(image_dir, image_path))
    bbox = detect_hand(img)

    # TODO: Incorparate hand information, for now only right hand
    hand_rectangles = [
        [
            op.Rectangle(0, 0, 0, 0),
            op.Rectangle(*bbox)
        ]
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
        cv2.waitKey(0)

    # Specific to our dataset
    camera_name = os.path.dirname(image_path)
    frame_id = os.path.basename(image_path).split(".")[0]
    keypoint_id.append((camera_name, frame_id))

with open(os.path.join(args[0].root_dir, "keypoint_id.json"), "w") as f:
    json.dump(keypoint_id, f)

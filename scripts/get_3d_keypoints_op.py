import argparse
import cv2
import json
import mediapipe as mp
import numpy as np
import os
import plotly.graph_objects as go
import shutil
import xml.etree.cElementTree as ET
import src.utils.params as param_utils
import tempfile
import trimesh
from openpose import pyopenpose as op
from tqdm import tqdm
from src.utils.misc import get_files

mp_hands = mp.solutions.hands

# Flag
parser = argparse.ArgumentParser()
parser.add_argument("--root", "-r", required=True, type=str)
parser.add_argument("--models_dir", "-m", required=True, type=str)
parser.add_argument("--handedness", choices=["left", "right"], required=True, type=str)
parser.add_argument("--point_cloud", action="store_true")
subparser = parser.add_subparsers(dest="mode", required=True)
single_parser = subparser.add_parser("single")
single_parser.add_argument("--filter", default="", type=str)
multi_parser = subparser.add_parser("multi")
multi_parser.add_argument("--frequency", default=1, type=int)

args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1]) - 1:
        next_item = args[1][i + 1]
    else:
        next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace("-", "")
        if key not in params:
            params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace("-", "")
        if key not in params:
            params[key] = next_item

params["model_folder"] = args[0].models_dir
params["hand"] = True
params["hand_detector"] = 2
params["body"] = 0
params["3d"] = True
params["number_people_max"] = 1
params["frame_undistort"] = 1
tmpdir = tempfile.mkdtemp()
params["camera_parameter_path"] = os.path.join(tmpdir, "param/") # TODO: Fix the extra /
params["image_dir"] = os.path.join(tmpdir, "output")

# Starting OpenPose
opWrapper = op.WrapperPython(op.ThreadManagerMode.Asynchronous)
opWrapper.configure(params)
opWrapper.start()

# By how much to pad hand detection bounding box
padding = 0.3

bboxes = []
data = []
handRectangles = []
image_base = os.path.join(args[0].root, "image")
if args[0].mode == "single":
    all_files = [get_files("**/*.jpg", image_base, args[0].filter)]
else:
    frame_ids = [frame.split("_")[-1] for frame in get_files("camera_000/*.jpg", image_base, "")]
    all_files = []
    for frame_id in tqdm(frame_ids[::args[0].frequency]):
        all_files.append(get_files(f"**/*{frame_id}", image_base, ""))

if args[0].point_cloud:
    point_cloud_dir = os.path.join(args[0].root, "point_cloud")
    try:
        shutil.rmtree(point_cloud_dir)
    except FileNotFoundError:
        pass
    os.makedirs(point_cloud_dir, exist_ok=True)

keypoints_dir = os.path.join(args[0].root, "keypoints_3d")
try:
    shutil.rmtree(keypoints_dir)
except FileNotFoundError:
    pass
os.makedirs(keypoints_dir, exist_ok=True)

# To plot
fig_frames = []
frame_num = 0
for valid_files in tqdm(all_files):
    assert len(valid_files) <= 54, "Total number of images per frame shouldn't exceed 54"

    # Use mediapipe to get bounding boxes
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        for file in tqdm(valid_files, leave=False):
            file_path = os.path.join(os.path.join(args[0].root, "image"), file)
            bbox = []
            image = cv2.flip(cv2.imread(file_path), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            image_height, image_width, _ = image.shape
            if not results.multi_handedness:
                tqdm.write(f"-- WARNING: Returning entire image: {file}")
                img_sqr = [0, 0, image_width, image_height]
                if image_width < image_height:
                    img_sqr[1] = image_width/4
                    img_sqr[3] = image_width
                else:
                    img_sqr[0] = image_height/4
                    img_sqr[2] = image_height
                bboxes.append(img_sqr)
                handRectangles += [
                    op.Rectangle(0, 0, 0, 0),
                    op.Rectangle(*bboxes[-1]),
                ]
                continue

            handedness = results.multi_handedness[0].classification[0].label
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            landmarks = results.multi_hand_landmarks[0].landmark
            for landmark in landmarks:
                bbox.append([landmark.x * image_width, landmark.y * image_height])

            # Calculate bounding box
            bbox = np.array(bbox)
            bbox_min = bbox.min(0)
            bbox_max = bbox.max(0)
            bbox_size = bbox_max - bbox_min

            # Pad hand bounding box
            bbox_min -= bbox_size * padding
            bbox_max += bbox_size * padding
            bbox_size = bbox_max - bbox_min

            # Convert bbox to square of length equal
            # to longer edge
            diff = bbox_size[0] - bbox_size[1]
            if diff > 0:
                bbox_min[1] -= diff / 2
                bbox_max[1] += diff / 2
                bbox_size[1] = bbox_size[0]
            else:
                bbox_min[0] -= -diff / 2
                bbox_max[0] += -diff / 2
                bbox_size[0] = bbox_size[1]

            # Flip
            tmp = bbox_min[0]
            bbox_min[0] = image_width - bbox_max[0]
            bbox_max[0] = image_width - tmp
            image = cv2.flip(image, 1)

            bboxes.append([*bbox_min, *bbox_size])

            handRectangles += [
                op.Rectangle(*bboxes[-1]) if args[0].handedness == "left" else op.Rectangle(0, 0, 0, 0),
                op.Rectangle(*bboxes[-1]) if args[0].handedness == "right" else op.Rectangle(0, 0, 0, 0),
            ]

    try:
        shutil.rmtree(params["image_dir"])
    except FileNotFoundError:
        pass
    os.makedirs(params["image_dir"])
    try:
        shutil.rmtree(params["camera_parameter_path"])
    except FileNotFoundError:
        pass
    os.makedirs(params["camera_parameter_path"])

    print(f"-- INFO: Found bounding boxes for {len(valid_files)} images")

    params_path = os.path.join(args[0].root, "params.txt")
    params_act = param_utils.read_params(params_path)

    for i in range(params_act.shape[0]):
        extr = param_utils.get_extr(params_act[i]) 
        intr, dist = param_utils.get_intr(params_act[i])

        root = ET.Element("opencv_storage")
        ext_matrix = ET.SubElement(root, "CameraMatrix", type_id="opencv-matrix")
        ET.SubElement(ext_matrix, "rows").text = str(3)
        ET.SubElement(ext_matrix, "cols").text = str(4)
        ET.SubElement(ext_matrix, "dt").text = "d"
        ET.SubElement(ext_matrix, "data").text = " ".join(
            map(str, extr.flatten().tolist())
        )

        int_matrix = ET.SubElement(root, "Intrinsics", type_id="opencv-matrix")
        ET.SubElement(int_matrix, "rows").text = str(3)
        ET.SubElement(int_matrix, "cols").text = str(3)
        ET.SubElement(int_matrix, "dt").text = "d"
        ET.SubElement(int_matrix, "data").text = " ".join(map(str, intr.flatten().tolist()))

        dist_matrix = ET.SubElement(root, "Distortion", type_id="opencv-matrix")
        ET.SubElement(dist_matrix, "rows").text = str(4)
        ET.SubElement(dist_matrix, "cols").text = str(1)
        ET.SubElement(dist_matrix, "dt").text = "d"
        ET.SubElement(dist_matrix, "data").text = " ".join(
            map(str, dist.flatten().tolist())
        )

        tree = ET.ElementTree(root)
        text = ET.tostring(root, xml_declaration=True).decode("utf-8")
        text = str.replace(text, "'", '"')
        with open(os.path.join(params["camera_parameter_path"], f"{i}.xml"), "w") as f:
            f.write(text)

        image_path = os.path.join(args[0].root, "image", valid_files[i])
        shutil.copyfile(image_path, os.path.join(params["image_dir"], f"{i}.png"))

    # Run openpose 3D hand detection module
    datums = op.VectorDatum()
    result = opWrapper.detectHandKeypoints3D(datums, handRectangles)

    if result:
        try:
            coords = datums[0].handKeypoints3D[0 if args[0].handedness == "left" else 1][0]
            with open("./results.json", "w") as f:
                json.dump(coords.tolist(), f)
        except Exception:
            tqdm.write("-- ERROR: Pose estimation failed")
            continue
    else:
        tqdm.write("-- ERROR: Pose estimation failed")
        continue

    fig_frames.append(
        go.Frame(
            data = go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers"
            )
        )
    )

    if args[0].point_cloud:
        pcd = trimesh.PointCloud(coords[:,:3])
        _ = pcd.export(os.path.join(point_cloud_dir, f"{frame_num:08d}.ply"))

    with open(os.path.join(keypoints_dir, f"{frame_num:08d}.json"), "w") as f:
        json.dump(coords.tolist(), f)

    frame_num += 1

fig = go.Figure(
    data = fig_frames[0].data,
    layout=go.Layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    frames = fig_frames
)
fig.write_html("keypoints.html", auto_open=False)
tqdm.write("-- INFO: Saved 3D visualization in 'keypoints.html'")
shutil.rmtree(tmpdir)

import cv2
import numpy as np
import json
import pickle
import joblib
import os
import glob
from skimage.measure import ransac
from src.triangulate import triangulate_joints, simple_processor, ransac_processor
from src.utils import plot_keypoints_3d, natural_sort, plot_keypoints_2d

# Parameters
params = "/ssd_scratch/cvit/ishaan/STIC_real/hand_pose_02/params/"
frame = "00000"
root_dir = "/ssd_scratch/cvit/ishaan/STIC_real/hand_pose_02/"
handedness = "right"

# Read intinsics
with open(os.path.join(params, "intrinsics.txt")) as f:
    intr = np.eye(3)
    for i in range(3):
        intr[i] = [float(k) for k in f.readline().split(',')]

# Read distortion coefficients
with open(os.path.join(params, "distortion.txt")) as f:
    dist = np.asarray([float(k) for k in f.readlines()])
    
# Read extrinsics
with open(os.path.join(params, "tvecs.txt")) as f:
    tvecs = []
    for line in f.readlines():
        tvecs.append(np.asarray([float(k) for k in line.split(',')]))
tvecs = np.asarray(tvecs)

with open(os.path.join(params, "rvecs.txt")) as f:
    rvecs = []
    for line in f.readlines():
        rvecs.append(np.asarray([float(k) for k in line.split(',')]))
rvecs = np.asarray(rvecs)
print(len(rvecs))

assert tvecs.shape == rvecs.shape

extr = []
for i in range(tvecs.shape[0]):
    r, _ = cv2.Rodrigues(rvecs[i])  # Also outputs Jacobian
    # t = -r.T @ tvecs[i]
    t = -tvecs[i]
    extr.append(np.hstack((r.T, t[:,None])))

extr = np.asarray(extr)

# Calculate projection matrices
proj_mats = []
for i in range(extr.shape[0]):
    proj_mats.append(intr @ extr[i])

proj_mats = np.asarray(proj_mats)

keypoints = []
keypoint_files = natural_sort(glob.glob("*.json", root_dir=os.path.join(root_dir, "keypoints_2d")))

image_dir = os.path.join(root_dir, "image")
images = list(natural_sort(glob.glob("**/*0000000.jpg", root_dir=image_dir)))

i = 0
for keypoint_file in keypoint_files:
    with open(f"{root_dir}/keypoints_2d/{keypoint_file}", "r") as f:
        data = json.load(f)
        kypts = np.array(data['people'][0][f'hand_{handedness}_keypoints_2d']).reshape(-1, 3)

        # Undistort keypoints
        undist_kypts = cv2.undistortPoints(kypts[:,:2].astype(np.float32), intr, dist).reshape(-1, 2)
        undist_kypts = np.hstack((undist_kypts, np.ones((len(undist_kypts), 1))))
        undist_kypts = undist_kypts @ intr.T
        keypoints.append(kypts)
        img = cv2.imread(os.path.join(image_dir, images[i]))
        for keypoint in kypts:
            if keypoint[2] < 0.75:
                continue
            cv2.circle(img, (int(keypoint[0]), int(keypoint[1])), 5, (0, 0, 255), -1)
        # cv2.imshow("debug", img)
        # k = cv2.waitKey(0)
        # if k == 27:
        #     exit()
        i += 1

keypoints = np.asarray(keypoints)
num_joints = keypoints.shape[1]
keypoints3d, residuals = triangulate_joints(keypoints, proj_mats[:-1], processor=simple_processor, conf_thresh_start=0.5, min_cams=10)
# keypoints3d, residuals = triangulate_joints(keypoints, proj_mats[:-1], processor=ransac_processor, conf_thresh_start=0.01, min_cams=10, residual_threshold=100, min_samples=5)
print(f"Error: {residuals.mean()}")

plot_keypoints_3d([keypoints3d - keypoints3d.mean(axis=0)], "real_data")
image_dir = os.path.join(root_dir, "image")
i = 0
for image in sorted(glob.glob("**/*0000000.jpg", root_dir=image_dir)):
    print(image)
    img = cv2.imread(os.path.join(image_dir, image))
    res = plot_keypoints_2d(keypoints3d, img, proj_mats[i])
    cv2.imshow("reprojection", res)
    k = cv2.waitKey(0)
    i += 1
    if k == 27:
        break

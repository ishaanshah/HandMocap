import numpy as np
import matplotlib.pyplot as plt
import json
from src.interpolate import interpolate_nans

frame_count = 20
joint_count = 21

joints = np.zeros((frame_count, joint_count, 4))

for i in range(frame_count):
    with open(f"./hands_maximo/keypoints/keypoints_{i}.json") as f:
        keypoints = np.array(json.load(f))
        joints[i] = keypoints
        joints[i,keypoints[:,3] == 0] = np.nan

joints[:,:,3] = 1
joints_lerp = joints.copy()
for i in range(joint_count):
    joints_lerp[:,i,0] = interpolate_nans(joints[:,i,0])
    joints_lerp[:,i,1] = interpolate_nans(joints[:,i,1])
    joints_lerp[:,i,2] = interpolate_nans(joints[:,i,2])

for i in range(frame_count):
    with open(f"./hands_maximo/keypoints_lerp/keypoints_{i}.json", "w") as f:
        json.dump(joints_lerp[i].tolist(), f)

plt.plot(np.arange(frame_count), joints_lerp[:,5,0])
plt.show()

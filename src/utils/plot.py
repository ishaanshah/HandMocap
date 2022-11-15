import numpy as np
import plotly.graph_objects as go
import cv2
from typing import List, Tuple

def plot_keypoints_3d(joints: List[np.ndarray], name: str, bounds: Tuple[int,int]=(-4,4)) -> None:
    fig= go.Figure(
        [
            go.Scatter3d(
                x=joint[:,0],
                y=joint[:,1],
                z=joint[:,2],
                mode='markers'
            ) for joint in joints
        ]
    )
    fig.update_layout(scene_aspectmode='cube')
    fig.update_layout(scene=dict(
        xaxis = dict(range=bounds),
        yaxis = dict(range=bounds),
        zaxis = dict(range=bounds)
    ))
    print(f"INFO: Writting keypoints to {name}.html")
    fig.write_html(f"{name}.html")

def plot_keypoints_2d(joints: np.ndarray, image: np.ndarray, proj_mat: np.ndarray) -> np.ndarray:
    keypoints_2d = project(joints, np.asarray([proj_mat]))
    res = image.copy()
    for keypoint in keypoints_2d[0]:
        cv2.circle(res, (int(keypoint[0]), int(keypoint[1])), 5, (0, 0, 255), -1)

    return res

def project(keypoints3d: np.ndarray, P: np.ndarray):
    """
    Project keypoints to 2D using

    Inputs -
        keypoints3d (N, 3): 3D keypoints
        P (V,3,4): Projection matrices
    Outputs -
        keypoints2d (V, N, 2): Projected 2D keypoints
    """
    hom = np.hstack((keypoints3d, np.ones((keypoints3d.shape[0], 1))))
    projected = np.matmul(P, hom.T).transpose(0, 2, 1) # (V, N, 2)
    projected = (projected / projected[:,:,-1:])[:,:,:-1]
    return projected
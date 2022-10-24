import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple

def plot_keypoints(joints: List[np.array], name: str, bounds: Tuple[int,int]=(-4,4)) -> None:
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


def reprojection_error(keypoints3d: np.ndarray, keypoints2d: np.ndarray, P: np.ndarray, confidence: np.ndarray=None) -> float:
    """
    Calculate average reprojection error across all cameras.
    The error is weighted with confidence score if provided with one.
    
    Inputs -
        keypoints3d (N,3): Predicted 3D keypoints
        keypoints2d (V,N,2): 2D keypoints
        P (V,3,4): Projection matrices for each camera
        confidence (V, N): Confidence scores for 2D keypoints
    Outputs -
        error (N): Average reprojection error
    """
    projected = np.matmul(P, keypoints3d.T).transpose(0, 2, 1) # (V, N, 2)
    projected = (projected / projected[:,:,-1:])[:,:,:-1]
    if confidence is None:
        confidence = np.ones(P.shape[0], keypoints3d.shape[0])
    error = np.linalg.norm(np.abs(projected - keypoints2d), axis=-1) # (V, N)
    error = (confidence * error) / np.sum(confidence)
    return error.sum(axis=0)

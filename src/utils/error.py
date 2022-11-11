import numpy as np

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

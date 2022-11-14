import jax.numpy as jnp
import jax
from typing import List
from jaxlie import SO3

class KinematicChain:
    def __init__(self, bones, root):
        self.bones = bones
        self.root = root
        self.rest_angles = [[0., 0., 0., 0.] for _ in range(len(bones))]
        for _, bone in bones.items():
            bone_idx = bone["idx"]
            self.rest_angles[bone_idx] = bone["quat"]
        self.rest_angles = jnp.array(self.rest_angles)
    
    def forward(self, angles=None):
        if angles is None:
            # angles = jnp.array([[1., 0., 0., 0.] for i in range(len(bones))])
            angles = self.rest_angles
            
        joints = jnp.zeros((len(self.bones)+1, 3), dtype=float)
        stk = [(self.root, jnp.array([0, 0, 0]), jnp.array([0, 0, 1]))]
        while len(stk):
            bone_name, loc, vec = stk.pop()
            bone_idx = self.bones[bone_name]["idx"]
            r = SO3(wxyz=angles[bone_idx])
            # r_rest = SO3(wxyz=self.rest_angles[bone_idx])
            # dir_vec = r_rest.apply(vec)
            dir_vec = r.apply(vec)
            dir_vec = dir_vec / jnp.linalg.norm(dir_vec)
            joints = joints.at[bone_idx+1].set(dir_vec*self.bones[bone_name]["len"] + loc)
            for child in self.bones[bone_name]["child"]:
                stk.append((child, joints[bone_idx+1], dir_vec))
        
        return joints

    def update_bone_lengths(self, keypoints: jnp.ndarray):
        for _, bone in self.bones.items():
            curr_id = bone["idx"]
            for child in bone["child"]:
                child_id = self.bones[child]["idx"]
                bone_vecs = keypoints[:,curr_id]-keypoints[:,child_id] 
                to_use = ~jnp.logical_or(jnp.isclose(keypoints[:,curr_id,3], 0), jnp.isclose(keypoints[:,child_id,3], 0))
                bone_lens = jnp.linalg.norm(bone_vecs[:,:3], axis=1)[to_use]
                self.bones[child]["len"] = bone_lens.mean().item()
        
    def IK(self, target, max_iter, mse_threshold, to_use=None, init=None):
        num_bones = len(self.bones)
        num_joints = num_bones+1
        if init is None:
            init = self.rest_angles

        u = 1e-3
        v = 1.5
        last_update = 0
        last_mse = 0
        params = init
        if to_use is None:
            to_use = jnp.ones(num_joints, dtype=bool)
            
        for i in range(max_iter):
            residual = (self.forward(params)[to_use] - target[to_use]).reshape(-1, 1)
            j = jax.jacrev(self.forward)(params)[to_use].reshape(-1, num_bones, 4).reshape(-1, 4*num_bones)

            mse = jnp.mean(jnp.square(residual))
            
            if abs(mse - last_mse) < mse_threshold:
                return params
            
            jtj = jnp.matmul(j.T, j)
            jtj = jtj + u * jnp.eye(jtj.shape[0])
            
            update = last_mse - mse
            delta = jnp.matmul(
                jnp.matmul(jnp.linalg.inv(jtj), j.T), residual
            ).ravel()
            params -= delta.reshape(num_bones, 4)

            if update > last_update and update > 0:
                u /= v
            else:
                u *= v

            last_update = update
            last_mse = mse

            print(f"Iteration {i}: {mse}")

        return params

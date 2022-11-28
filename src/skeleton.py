import jax.numpy as jnp
import jax
import polyscope as ps
import numpy as np
import jaxopt
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from jaxlie import SO3

class KinematicChain:
    def __init__(self, bones, root):
        self.bones = bones
        self.root = root
        self.rest_angles = [[0., 0., 0., 0.] for _ in range(len(bones))]
        self.parents = [-1 for _ in range(len(bones))]
        # Temporary, rename if stuff works
        self.rest_matrixs = [np.eye(4) for _ in range(len(bones))]
        for _, bone in bones.items():
            bone_idx = bone["idx"]
            self.rest_angles[bone_idx] = bone["quat"]
            self.rest_matrixs[bone_idx] = bone["rest_matrix"]
            self.parents[bone_idx] = bone["parent_idx"]
        self.rest_angles = jnp.array(self.rest_angles)
        # self.rest_joints, self.rest_matrices = self.get_rest_pose()
        self.rest_matrixs = jnp.asarray(self.rest_matrixs)

        self.dof = np.zeros((len(self.bones), 3), dtype=bool) # num_joint, 3 (x, y, z)
        
        # Wrist
        self.dof[0,:] = True # all

        # Palm
        self.dof[1:3,0] = True # x
        self.dof[1:3,2] = True # z
        self.dof[5:18:4,0] = True # x
        self.dof[5:18:4,2] = True # z
        
        # First finger joints
        self.dof[6:19:4,0] = True # x
        self.dof[6:19:4,2] = True # z

        # Thumb
        # Special case for thumb because it's roll isn't oriented correctly
        self.dof[3:5,0] = True  # x
        self.dof[3:5,2] = True  # z

        # Index finger
        self.dof[7:9,2] = True  # z

        # Middle finger
        self.dof[11:13,2] = True  # z

        # Ring finger
        self.dof[15:17,2] = True  # z
        
        # Pinky finger
        self.dof[19:21,2] = True  # z

        self.dof = jnp.asarray(self.dof)

    def plot_skeleton(self, angles=None, target=None):
        """ Debug function """
        ps.init()

        _, heads, tails, = self.forward(angles)
        pt_cloud = ps.register_point_cloud("Skeleton", heads)
        ps.register_point_cloud("heads", heads)

        pt_cloud.add_vector_quantity("bones", tails - heads, color=(0,0,0), enabled=True, vectortype='ambient', radius=0.004)

        if not target is None:
            ps.register_point_cloud("target", target)
        
        ps.show()
        
    def forward(self, params):
        tails = jnp.zeros((len(self.bones), 3), dtype=float)
        heads = tails.copy()
        stk = [self.root]
        # global_translation = params[0]
        matrix = jnp.asarray([np.eye(4) for _ in range(len(self.bones))])
        # angles = params[1:]
        angles = params
        constrained_angles = jnp.zeros_like(angles)
        constrained_angles = constrained_angles.at[self.dof].set(angles[self.dof])
        while len(stk):
            bone_name = stk.pop()
            bone_idx = self.bones[bone_name]["idx"]
            parent_idx = self.bones[bone_name]["parent_idx"]

            r = self.getRotationFromEuler(constrained_angles[bone_idx]).as_matrix()

            # Convert to homegenous
            r = jnp.vstack([r, jnp.array([0, 0, 0])])
            r = jnp.hstack([r, jnp.array([[0], [0], [0], [1]])])

            parent_local = jnp.linalg.inv(self.rest_matrixs[parent_idx])
            if parent_idx == -1:
                # Root bone
                parent_local = jnp.eye(4)

            current_local = self.rest_matrixs[bone_idx]
            local = jnp.einsum('ij,jk->ik', parent_local, current_local)
            pose = jnp.einsum('ij,jk->ik', local, r)
            matrix = matrix.at[bone_idx].set(jnp.einsum('ij,jk->ik', matrix[parent_idx], pose))

            for child in self.bones[bone_name]["child"]:
                stk.append(child)
            
        for bone_name, bone_data in self.bones.items():
            bone_idx = bone_data["idx"]
            rest_pose_inverse = jnp.linalg.inv(self.rest_matrixs[bone_idx])

            head_rest_joint = np.hstack([bone_data["head"], [1]])
            head_keypoint = rest_pose_inverse @ head_rest_joint
            head_keypoint = matrix[bone_idx] @ head_keypoint
            heads = heads.at[bone_data["idx"]].set(head_keypoint[:3])

            tail_rest_joint = np.hstack([bone_data["tail"], [1]])
            tail_keypoint = rest_pose_inverse @ tail_rest_joint
            tail_keypoint = matrix[bone_idx] @ tail_keypoint
            tails = tails.at[bone_data["idx"]].set(tail_keypoint[:3])

        stk = [(self.root, heads[self.bones[self.root]["idx"]])]
        scaled_tails = jnp.zeros((len(self.bones), 3), dtype=float)
        scaled_heads = scaled_tails.copy()
        while len(stk):
            bone_name, head = stk.pop()
            bone_idx = self.bones[bone_name]["idx"];

            dir_vec = tails[bone_idx] - heads[bone_idx]
            dir_vec = dir_vec / jnp.linalg.norm(dir_vec)
            bone_len = self.bones[bone_name]["len"]
            tail = dir_vec*bone_len + head

            scaled_heads = scaled_heads.at[bone_idx].set(head)
            scaled_tails = scaled_tails.at[bone_idx].set(tail)

            for child in self.bones[bone_name]["child"]:
                stk.append((child, tail))
        
        # scaled_heads += global_translation
        # scaled_tails += global_translation

        keypoints = jnp.r_[scaled_heads[:1], scaled_tails]
        return keypoints, scaled_heads, scaled_tails


    def update_bone_lengths(self, keypoints: jnp.ndarray):
        for _, bone in self.bones.items():
            curr_id = bone["idx"]
            for child in bone["child"]:
                child_id = self.bones[child]["idx"]
                bone_vecs = keypoints[:,curr_id]-keypoints[:,child_id] 
                to_use = ~jnp.logical_or(jnp.isclose(keypoints[:,curr_id,3], 0), jnp.isclose(keypoints[:,child_id,3], 0))
                if not jnp.count_nonzero(to_use):
                    raise ValueError(f"No frame has length of bone {child}")
                bone_lens = jnp.linalg.norm(bone_vecs[:,:3], axis=1)[to_use]
                self.bones[child]["len"] = bone_lens.mean().item()

        # stk = [(self.root, jnp.asarray(self.bones[self.root]["head"]))]
        # while len(stk):
        #     bone_name, head = stk.pop()

        #     dir_vec = jnp.array(self.bones[bone_name]["tail"]) - jnp.array(self.bones[bone_name]["head"])
        #     dir_vec = dir_vec / jnp.linalg.norm(dir_vec)
        #     bone_len = self.bones[bone_name]["len"]
        #     tail = dir_vec*bone_len + head

        #     self.bones[bone_name]["head"] = head
        #     self.bones[bone_name]["tail"] = tail

        #     for child in self.bones[bone_name]["child"]:
        #         stk.append((child, tail))

    def IK_jaxopt(self, target):
        pass
        
    def IK(self, target, max_iter, mse_threshold, to_use=None, init=None):
        num_bones = len(self.bones)
        num_joints = num_bones+1
        if init is None:
            init = jnp.zeros((len(self.bones), 3))

        u = 1e-3
        v = 1.5
        last_update = 0
        last_mse = 0
        params = init
        if to_use is None:
            to_use = jnp.ones(num_joints, dtype=bool)
            

        pbar = tqdm(range(max_iter))
        for i in pbar:
            keypoints, _, __ = self.forward(params)
            residual = (keypoints[to_use] - target[to_use]).reshape(-1, 1)
            j = jax.jacrev(self.forward)(params)[0][to_use].reshape(-1, num_bones, 3).reshape(-1, 3*(num_bones))
            # j = j[:,self.dof.flatten()]

            mse = jnp.mean(jnp.square(residual))
            
            if abs(mse - last_mse) < mse_threshold:
                return params
            
            jtj = jnp.matmul(j.T, j)
            jtj = jtj + u * jnp.eye(jtj.shape[0])
            
            update = last_mse - mse
            delta = jnp.matmul(
                jnp.matmul(jnp.linalg.inv(jtj), j.T), residual
            ).ravel()
            # params = params.at[self.dof].set(params[self.dof] - delta)
            params = params - delta.reshape(-1, 3)

            if update > last_update and update > 0:
                u /= v
            else:
                u *= v

            last_update = update
            last_mse = mse

            pbar.set_description(f"Iteration {i}: {mse}")

        return params

    def solveIK(self):
        pass

    @staticmethod
    def getRotationFromEuler(angles):
        flipAngles = jnp.flip(angles)
        r = SO3.from_z_radians(flipAngles[0]) @ \
            SO3.from_y_radians(flipAngles[1]) @ \
            SO3.from_x_radians(flipAngles[2])

        return r
import jax.numpy as jnp
import jax
import polyscope as ps
import numpy as np
import trimesh
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
        self.rest_joints, self.rest_matrices = self.get_rest_pose()
        self.rest_matrixs = jnp.asarray(self.rest_matrixs)

        self.dof = np.zeros((len(self.bones), 3), dtype=bool) # num_joint, 3 (x, y, z)
        
        # Wrist
        self.dof[0,:] = True # all

        # Palm
        self.dof[1:3,0:2] = True
        self.dof[5:18:4,0:2] = True

        # Thumb
        # Special case for thumb because it's roll isn't oriented correctly
        self.dof[3:5,0:2] = True  # x, y

        # Index finger
        self.dof[6,0:2] = True  # x, y
        self.dof[7:9,1] = True  # x

        # Middle finger
        self.dof[10,0:2] = True  # x, y
        self.dof[11:13,1] = True  # x

        # Ring finger
        self.dof[14,0:2] = True  # x, y
        self.dof[15:17,1] = True  # x
        
        # Pinky finger
        self.dof[18,0:2] = True  # x, y
        self.dof[19:21,1] = True  # x

        # self.dof[:,:] = True
        self.dof = jnp.asarray(self.dof)

    def plot_skeleton(self, angles=None, target=None):
        """ Debug function """
        ps.init()

        if angles is None:
            joints = self.rest_joints
            pt_cloud = ps.register_point_cloud("Skeleton", joints)
            pt_cloud.add_vector_quantity("x", self.rest_matrices[:,0], color=(1, 0, 0), enabled=True)
            pt_cloud.add_vector_quantity("y", self.rest_matrices[:,1], color=(0, 1, 0), enabled=True)
            pt_cloud.add_vector_quantity("z", self.rest_matrices[:,2], color=(0, 0, 1), enabled=True)
        else:
            heads, tails, rest_matrices = self.forward2(angles)
            pt_cloud = ps.register_point_cloud("Skeleton", heads)
            ps.register_point_cloud("heads", heads)
            pcd = trimesh.PointCloud(heads)
            _ = pcd.export("points.ply")
            exit()
            # pt_cloud.add_vector_quantity("x", rest_matrices[:,0,:3], color=(1, 0, 0), enabled=True)
            # pt_cloud.add_vector_quantity("y", rest_matrices[:,1,:3], color=(0, 1, 0), enabled=True)
            # pt_cloud.add_vector_quantity("z", rest_matrices[:,2,:3], color=(0, 0, 1), enabled=True)
        
        # bone_vecs = jnp.zeros((len(self.bones)+1, 3))
        # for _, bone in self.bones.items():
        #     bone_idx = bone["idx"]
        #     if bone_idx:
        #         bone_vecs = bone_vecs.at[bone_idx-1].set(tails[bone_idx-1] - heads[bone_idx-1])
        #     # for child in bone["child"]:
        #     #     child_idx = self.bones[child]["idx"]
        #     #     bone_vecs = bone_vecs.at[child_idx+1].set(-(joints[child_idx+1] - joints[bone_idx+1]))
        pt_cloud.add_vector_quantity("bones", tails - heads, color=(0,0,0), enabled=True, vectortype='ambient', radius=0.004)

        if not target is None:
            ps.register_point_cloud("target", target)
        
        ps.show()
        

    def get_rest_pose(self):
        rest_joints = jnp.zeros((len(self.bones)+1, 3), dtype=float)
        rest_matrix = jnp.zeros((len(self.bones)+1, 3, 3), dtype=float)
        rest_matrix = rest_matrix.at[0].set(jnp.identity(3))
        stk = [(self.root, jnp.array([0, 0, 0]), [jnp.array([1, 0, 0]), jnp.array([0, 1, 0]), jnp.array([0, 0, 1])])]
        while len(stk):
            bone_name, loc, [x, y, z]= stk.pop()
            bone_idx = self.bones[bone_name]["idx"]
            r = self.getRotationFromEuler(self.rest_angles[bone_idx])

            # Build orthonormal basis for joint w.r.t to parent
            x_dir_vec = r @ x
            x_dir_vec = x_dir_vec / jnp.linalg.norm(x_dir_vec)
            y_dir_vec = r @ y
            y_dir_vec = y_dir_vec / jnp.linalg.norm(y_dir_vec)
            z_dir_vec = r @ z
            z_dir_vec = z_dir_vec / jnp.linalg.norm(z_dir_vec)
            rest_matrix = rest_matrix.at[bone_idx+1].set(jnp.vstack([x_dir_vec, y_dir_vec, z_dir_vec]))

            # Calculate joint location
            rest_joints = rest_joints.at[bone_idx+1].set(z_dir_vec*self.bones[bone_name]["len"] + loc)

            for child in self.bones[bone_name]["child"]:
                stk.append((child, rest_joints[bone_idx+1], [x_dir_vec, y_dir_vec, z_dir_vec]))
        
        return rest_joints, rest_matrix
    
    def forward(self, angles):
        joints = jnp.zeros((len(self.bones)+1, 3), dtype=float)
        stk = [(self.root, jnp.array([0, 0, 0]), [jnp.array([1, 0, 0]), jnp.array([0, 1, 0]), jnp.array([0, 0, 1])])]
        rest_matrix = jnp.zeros((len(self.bones)+1, 3, 3), dtype=float)
        while len(stk):
            bone_name, loc, [x, y, z]= stk.pop()
            bone_idx = self.bones[bone_name]["idx"]
            
            # Rotation with respect to parent
            r = self.getRotationFromEuler(self.rest_angles[bone_idx] + angles[bone_idx]).as_matrix()

            # Build orthonormal basis for joint w.r.t to parent
            x_dir_vec = r @ x
            x_dir_vec = x_dir_vec / jnp.linalg.norm(x_dir_vec)
            y_dir_vec = r @ y
            y_dir_vec = y_dir_vec / jnp.linalg.norm(y_dir_vec)
            z_dir_vec = r @ z
            z_dir_vec = z_dir_vec / jnp.linalg.norm(z_dir_vec)

            # Apply rest matrix
            # dir_vec = SO3.from_matrix(self.rest_matrices[bone_idx+1]).apply(z_dir_vec)
            # dir_vec = dir_vec / jnp.linalg.norm(dir_vec)
            dir_vec = z_dir_vec

            rest_matrix = rest_matrix.at[bone_idx+1].set(jnp.vstack([x_dir_vec, y_dir_vec, z_dir_vec]))

            # Calculate joint location
            joints = joints.at[bone_idx+1].set(dir_vec*self.bones[bone_name]["len"] + loc)

            for child in self.bones[bone_name]["child"]:
                stk.append((child, joints[bone_idx+1], [x_dir_vec, y_dir_vec, z_dir_vec]))
        
        return joints, rest_matrix
    
    def forward2(self, angles):
        tails = jnp.zeros((len(self.bones)+1, 3), dtype=float)
        heads = tails.copy()
        stk = [(self.root, jnp.array([0, 0, 0]))]
        matrix = jnp.asarray([np.eye(4) for _ in range(len(self.bones))])
        while len(stk):
            bone_name, loc = stk.pop()
            bone_idx = self.bones[bone_name]["idx"]
            parent_idx = self.bones[bone_name]["parent_idx"]

            r = self.getRotationFromEuler(angles[bone_idx]).as_matrix()

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
                stk.append((child, tails[bone_idx+1]))
            
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

        return heads, tails, matrix


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
            residual = (self.forward(params)[to_use] - target[to_use]).reshape(-1, 1)
            j = jax.jacrev(self.forward)(params)[to_use].reshape(-1, num_bones, 3).reshape(-1, 3*num_bones)
            j = j[:,self.dof.flatten()]

            mse = jnp.mean(jnp.square(residual))
            # print(params)
            
            if abs(mse - last_mse) < mse_threshold:
                return params
            
            jtj = jnp.matmul(j.T, j)
            jtj = jtj + u * jnp.eye(jtj.shape[0])
            
            update = last_mse - mse
            delta = jnp.matmul(
                jnp.matmul(jnp.linalg.inv(jtj), j.T), residual
            ).ravel()
            params = params.at[self.dof].set(params[self.dof] - delta)
            print(delta)
            print(params)
            print(self.dof)

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
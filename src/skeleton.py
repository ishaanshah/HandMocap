import jax.numpy as jnp
import jax
import polyscope as ps
from tqdm import tqdm
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
        self.rest_joints, self.rest_matrices = self.get_rest_pose()

    def plot_skeleton(self, angles=None):
        """ Debug function """
        ps.init()

        if angles is None:
            joints = self.rest_joints
            pt_cloud = ps.register_point_cloud("Skeleton", joints)
            pt_cloud.add_vector_quantity("x", self.rest_matrices[:,0], color=(1, 0, 0), enabled=False)
            pt_cloud.add_vector_quantity("y", self.rest_matrices[:,1], color=(0, 1, 0), enabled=False)
            pt_cloud.add_vector_quantity("z", self.rest_matrices[:,2], color=(0, 0, 1), enabled=False)
        else:
            joints = self.forward(angles)
            pt_cloud = ps.register_point_cloud("Skeleton", joints)
        
        bone_vecs = jnp.zeros((len(self.bones)+1, 3))
        for _, bone in self.bones.items():
            bone_idx = bone["idx"]
            for child in bone["child"]:
                child_idx = self.bones[child]["idx"]
                bone_vecs = bone_vecs.at[child_idx+1].set(-(joints[child_idx+1] - joints[bone_idx+1]))
        pt_cloud.add_vector_quantity("bones", bone_vecs, color=(0,0,0), enabled=True, vectortype='ambient')
        
        ps.show()
        

    def get_rest_pose(self):
        rest_joints = jnp.zeros((len(self.bones)+1, 3), dtype=float)
        rest_matrix = jnp.zeros((len(self.bones)+1, 3, 3), dtype=float)
        stk = [(self.root, jnp.array([0, 0, 0]), [jnp.array([1, 0, 0]), jnp.array([0, 1, 0]), jnp.array([0, 0, 1])])]
        rest_matrix = rest_matrix.at[0].set(jnp.identity(3))
        while len(stk):
            bone_name, loc, [x, y, z]= stk.pop()
            bone_idx = self.bones[bone_name]["idx"]
            r = SO3(wxyz=self.rest_angles[bone_idx])

            # Build orthonormal basis for joint w.r.t to parent
            x_dir_vec = r.apply(x)
            x_dir_vec = x_dir_vec / jnp.linalg.norm(x_dir_vec)
            y_dir_vec = r.apply(y)
            y_dir_vec = y_dir_vec / jnp.linalg.norm(y_dir_vec)
            z_dir_vec = r.apply(z)
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
        while len(stk):
            bone_name, loc, [x, y, z]= stk.pop()
            bone_idx = self.bones[bone_name]["idx"]
            
            # Rotation with respect to parent
            r = SO3.from_rpy_radians(*angles[bone_idx])

            # Build orthonormal basis for joint w.r.t to parent
            x_dir_vec = r.apply(x)
            x_dir_vec = x_dir_vec / jnp.linalg.norm(x_dir_vec)
            y_dir_vec = r.apply(y)
            y_dir_vec = y_dir_vec / jnp.linalg.norm(y_dir_vec)
            z_dir_vec = r.apply(z)
            z_dir_vec = z_dir_vec / jnp.linalg.norm(z_dir_vec)

            # Apply rest matrix
            dir_vec = SO3.from_matrix(self.rest_matrices[bone_idx+1]).apply(z_dir_vec)
            dir_vec = dir_vec / jnp.linalg.norm(dir_vec)

            # Calculate joint location
            joints = joints.at[bone_idx+1].set(dir_vec*self.bones[bone_name]["len"] + loc)

            for child in self.bones[bone_name]["child"]:
                stk.append((child, joints[bone_idx+1], [x_dir_vec, y_dir_vec, z_dir_vec]))
        
        return joints

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
            init = self.rest_angles

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

            pbar.set_description(f"Iteration {i}: {mse}")

        return params
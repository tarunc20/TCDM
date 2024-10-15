import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import os 
from tqdm import tqdm

import genesis as gs
import numpy as np
import copy, collections
from dm_env import specs
from dm_control.rl import control
from dm_env._environment import StepType, TimeStep

from tcdm.envs.control import Environment, ReferenceMotionTask
from tcdm.envs.reference import HandObjectReferenceMotion
from tcdm.envs.wrappers import GymWrapper 
from tcdm.rl.models.policies import ActorCriticPolicy
from tcdm.rl.trainers.eval import EvalCallback
from tcdm.rl.trainers.util import make_policy_kwargs

from stable_baselines3 import PPO

from dataset import DexYCBVideoDataset, TwoHandDexYCBVideoDataset
from hand_robot_viewer import RobotHandDatasetSAPIENViewer, TwoRobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer, TwoHandDatasetSAPIENViewer
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import RobotName, HandType, get_default_config_path, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
import sapien

np.int = int 
np.bool = bool
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_

def rotation_matrix_to_quaternion(matrix):
    """
    Convert a 4x4 rotation matrix to a quaternion.
    
    Parameters:
    matrix (np.ndarray): 4x4 rotation matrix
    
    Returns:
    np.ndarray: Quaternion [w, x, y, z]
    """
    # Ensure the matrix is 4x4
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4")
    
    # Extract the rotation part (3x3) from the matrix
    rotation_matrix = matrix[:3, :3]

    # Compute the trace of the matrix
    trace = np.trace(rotation_matrix)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    else:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])

# physics class that only does rendering ?? or physics attribute 
def get_hand_poses(sampled_data):
    """
    - load mano hand -> for each step get hand geometry then get 
    - figure out joint naming scheme -> make sure correct
    - return list of hand poses 
    """
    # load viewer
    viewer = RobotHandDatasetSAPIENViewer([RobotName.shadow], HandType.right, headless=True)
    viewer.load_object_hand(sampled_data)
    right_hand_pose = sampled_data["hand_pose"]
    object_pose = sampled_data["object_pose"]
    robot_qposes = []
    object_qposes = []
    pose_offsets = []
    # set robot pose to 0
    y_offset = 0.8# * len(viewer.robots) / 2
    # for i in range(len(viewer.robots) + 1):
    #     pose = sapien.Pose([0, -y_offset * i, 0])
    #     pose_offsets.append(pose)
    #     if i >= 1:
    #         viewer.robots[i - 1].set_pose(pose)
    # pose offsets
    rh_poses = []
    dists = []
    for i in tqdm(range(len(right_hand_pose))):
        right_hand_pose_frame = right_hand_pose[i]
        object_pose_frame = object_pose[i]
        vertex, joint = viewer._compute_hand_geometry(right_hand_pose_frame, use_camera_frame=True)
        viewer._update_hand(vertex)
        # update robot qpos
        robot, retargeting, retarget2sapien = viewer.robots[0], viewer.retargetings[0], viewer.retarget2sapien[0]
        indices = retargeting.optimizer.target_link_human_indices
        ref_value = joint[indices, :]
        qpos = retargeting.retarget(ref_value)[retarget2sapien]
        viewer.robots[0].set_qpos(qpos)
        # setting object pose 
        pos_quat = object_pose_frame[0]
        qt = rotation_matrix_to_quaternion(pos_quat)
        # compute pose offsets
        num_copy = 2
        # Quaternion convention: wxyz
        #pose = viewer.camera_pose * sapien.Pose(pos_quat[:3, 3], qt)
        pose = sapien.Pose(pos_quat[:3, 3], qt)
        #viewer.objects[1].set_pose(pose_offsets[1] * pose)
        viewer.objects[1].set_pose(pose)
        viewer.objects[0].set_pose(pose)
        object_qposes.append(np.concatenate((viewer.objects[1].pose.p, viewer.objects[1].pose.q)))
        rh_poses.append(qpos)
    return object_qposes, rh_poses


# only need access through gym environment - otherwise just return env
class GenesisPhysics:
    
    def __init__(self, scene):
        self.scene = scene 
        self.camera = scene.add_camera(res=(1024, 1024), GUI=False)
    
    def render(self, camera_id, height=None, width=None):
        return self.camera.render()[camera_id]

class GenesisTask:

    def __init__(self):
        pass

    @property 
    def step_info(self):
        return {}

class GenesisEnvironment:
    
    def __init__(self, scene, physics, task, hand_poses, object_poses):
        # TODO: add parsing genesis domain name/other stuff
        self.scene = scene 
        scene.build()
        self.physics = physics 
        self.task = task 
        self.default_camera_id = 0
        self.hand_poses, self.object_poses = hand_poses, object_poses

    def observation_spec(self):
        return specs.Array((self.scene.rigid_solver._n_dofs,), np.float32)

    def action_spec(self):
        return specs.BoundedArray((30,), np.float32, -1, 1)
        
    def reset(self):
        self.step_count = 0
        self.scene.rigid_solver.entities[1].set_dofs_position(self.hand_poses[0])
        self.scene.step()
        state = self.scene.rigid_solver.dofs_state.to_numpy()['pos'].flatten()
        return TimeStep(
            observation=state, 
            reward=0, 
            discount=1, 
            step_type=StepType.FIRST,
        )
    
    def get_observation():
        return self.scene.rigid_solver.dofs_state.to_numpy()['pos'].flatten()

    def _compute_reward(self):
        # eturn self.scene.entities[1].get_dofs_position().cpu().numpy()[0]
        return np.exp(-np.linalg.norm(
          self.hand_poses[self.step_count] - self.scene.entities[1].get_dofs_position().cpu().numpy() 
        ))
    
    def step(self, action):
        old_dofs_position = self.scene.entities[1].get_dofs_position().cpu().numpy()
        new_dofs_position = old_dofs_position + action
        for _ in range(33):
            self.scene.entities[1].control_dofs_position(new_dofs_position)
            self.scene.step()
        reward = self._compute_reward()
        self.step_count += 1
        return TimeStep(
            observation=self.scene.rigid_solver.dofs_state.to_numpy()['pos'].flatten(),
            reward=reward,
            discount=1,
            step_type=StepType.MID if self.step_count < 100 else StepType.LAST 
        )  
       
TRIPLET = "(brush, brush, bowl)"
TOOL_NAME = "brush"
TARGET_NAME = "bowl"
SEQUENCE_NAME = "20230927_027"
DATASET_ROOT = "/home/tarunc/Desktop/research/taco_videos"
OBJECT_MODEL_ROOT = "/home/tarunc/Desktop/research/taco_videos/Object_Models/object_models_released"

def main():
    gs.init(seed=0, precision='32', logging_level='warning')
    mat_objs = gs.materials.Rigid()
    object_pose_dir = os.path.join(DATASET_ROOT, "Object_Poses", TRIPLET, SEQUENCE_NAME)
    tool_name, target_name = None, None
    for file_name in os.listdir(object_pose_dir):
        if file_name.startswith("tool_"):
            tool_name = file_name.split(".")[0].split("_")[-1]
        elif file_name.startswith("target_"):
            target_name = file_name.split(".")[0].split("_")[-1]
    assert (not tool_name is None) and (not target_name is None)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps_local=50,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -0.5, 0.2),
            upper_bound=(1.0, 0.5, 0.5),
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(960, 960),
            camera_pos=(2.5, -0.15, 2.82),
            camera_lookat=(0.5, 0.5, 0.5),
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, 0),
        ),
        show_viewer=False,
    )
    
    tool_entity = scene.add_entity(
        material=mat_objs,
        morph=gs.morphs.Mesh(
            file=os.path.join(OBJECT_MODEL_ROOT, tool_name + "_cm.obj"),
            scale=0.01
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 1.0, 1.0, 1.0),
        )
    )

    # load robot hand 
    shadow_right = "/home/tarunc/Desktop/research/dex-retargeting/assets/robots/hands/shadow_hand/shadow_hand_right.urdf"
    right_hand = scene.add_entity(
        morph=gs.morphs.URDF(
            scale=1.0,
            file=shadow_right,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        )
    )
    # generate dataset 
    data_root = "/home/tarunc/Desktop/research/taco_videos"
    RetargetingConfig.set_default_urdf_dir("/home/tarunc/Desktop/research/dex-retargeting/assets/robots/hands")
    dataset = DexYCBVideoDataset(data_root, hand_type="right")
    sampled_data = dataset[9]
    object_poses, hand_poses = get_hand_poses(sampled_data)
    task = GenesisTask()
    physics = GenesisPhysics(scene)
    env = GenesisEnvironment(scene, physics, task, hand_poses, object_poses)
    env = GymWrapper(env)
    o = env.reset()
    print("Running")
    imgs = []
    for i in tqdm(range(len(hand_poses))):
        action = hand_poses[i] - o[6:]
        o, r, d, _ = env.step(action)
        imgs.append(env.render())
        print(f"Diff: {np.linalg.norm(hand_poses[i] - o[6:])}")
    gs.tools.animate(imgs, os.path.join("", f"all_steps.mp4"))
    # imgs = []
    # for i in range(1, len(hand_poses)):
    #     o,r,d,i = env.step(hand_poses[i] - hand_poses[i-1])
    #     imgs.append(env.render())
    #     print(f"reward: {r}")
    # gs.tools.animate(imgs, os.path.join("", f"all_steps.mp4"))
    model = PPO(
        ActorCriticPolicy, 
        env, verbose=1, 
        tensorboard_log=f"logs/", 
        n_steps=100, # should be 209
        gamma=1.,
        gae_lambda=1.,
        learning_rate=1e-4,
        ent_coef=1.,
        vf_coef=1.,
        clip_range=1.,
        batch_size=64,
        n_epochs=5,
        # policy_kwargs=dict(
        #     net_arch = dict(
        #         pi=[256, 128],
        #         vf=[256, 128]
        #     )
        # )
    )
    eval_callback = EvalCallback(100, env)
    model.learn(
        total_timesteps=10000,
        callback=[eval_callback],
        reset_num_timesteps=False,
    )

if __name__ == "__main__":
    main()
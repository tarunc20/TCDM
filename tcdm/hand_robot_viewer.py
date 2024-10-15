import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2
import os
from tqdm import trange
import sapien
import transforms3d.quaternions
from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import RobotName, HandType, get_default_config_path, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from tcdm.hand_viewer import HandDatasetSAPIENViewer, TwoHandDatasetSAPIENViewer

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

ROBOT2MANO = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)
ROBOT2MANO_POSE = sapien.Pose(q=transforms3d.quaternions.mat2quat(ROBOT2MANO))


def prepare_position_retargeting(joint_pos: np.array, link_hand_indices: np.ndarray):
    link_pos = joint_pos[link_hand_indices]
    return link_pos


def prepare_vector_retargeting(joint_pos: np.array, link_hand_indices_pairs: np.ndarray):
    joint_pos = joint_pos @ ROBOT2MANO
    origin_link_pos = joint_pos[link_hand_indices_pairs[0]]
    task_link_pos = joint_pos[link_hand_indices_pairs[1]]
    return task_link_pos - origin_link_pos


class RobotHandDatasetSAPIENViewer(HandDatasetSAPIENViewer):
    def __init__(self, robot_names: List[RobotName], hand_type: HandType, headless=False, use_ray_tracing=False):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []

        # Load optimizer and filter
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        for robot_name in robot_names:
            config_path = get_default_config_path(robot_name, RetargetingType.position, hand_type)

            # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            # Build robot
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                urdf_path = str(urdf_path).replace(".urdf", "_glb.urdf")
            robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
            urdf_name = urdf_path.split("/")[-1]
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_name}"
            robot_urdf.write_xml_file(temp_path)

            robot = loader.load(temp_path)
            self.robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array([retargeting.joint_names.index(n) for n in sapien_joint_names]).astype(int)
            self.retarget2sapien.append(retarget2sapien)

    def load_object_hand(self, data: Dict):
        super().load_object_hand(data)
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]

        # Load the same YCB objects for n times, n is the number of robots
        # So that for each robot, there will be an identical set of objects
        for _ in range(len(self.robots)):
            for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
                self._load_ycb_object(ycb_id, ycb_mesh_file)

    def render_dexycb_data(self, data: Dict, fps=5, y_offset=0.8):
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2
        # self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([0.5, global_y_offset + 0.75, 1.7]))
            self.camera.set_local_pose(local_pose)

        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        num_frame = hand_pose.shape[0]
        num_copy = len(self.robots) + 1
        num_ycb_objects = len(data["ycb_ids"])
        pose_offsets = []

        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1].set_pose(pose)

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        for i in range(0, num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame, use_camera_frame=True)
            if vertex is not None:
                start_frame = i
                break

        if self.headless:
            robot_names = [robot.name for robot in self.robot_names]
            robot_names = "_".join(robot_names)
            video_path = Path(__file__).parent.resolve() / f"data/{robot_names}_video.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (self.camera.get_width(), self.camera.get_height())
            )
        # Loop rendering
        step_per_frame = int(60 / fps)
        robot_poses = []
        object_poses = []
        all_obj_poses = [[] for _ in range(len(self.objects))] 
        for i in trange(start_frame, num_frame):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)

            # Update poses for YCB objects
            for k in range(num_ycb_objects):
                pos_quat = object_pose_frame[k]
                qt = rotation_matrix_to_quaternion(pos_quat)

                # Quaternion convention: wxyz
                i = 0
                #pose = self.camera_pose * sapien.Pose(pos_quat[:3, 3], qt)
                self.objects[k].set_pose(pose)
                for copy_ind in range(num_copy):
                    self.objects[k + copy_ind * num_ycb_objects].set_pose(pose_offsets[copy_ind] * pose)
                    all_obj_poses[i].append((pose_offsets[copy_ind] * pose).p)
                    i += 1

            # Update pose for human hand
            self._update_hand(vertex)

            # Update poses for robot hands
            for robot, retargeting, retarget2sapien in zip(self.robots, self.retargetings, self.retarget2sapien):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)
                np.set_printoptions(suppress=True)
                robot_poses.append(qpos)
            #print(f"Pose differences: {self.objects[1].get_pose().p - self.robots[0].links[-3].get_pose().p}")
            #print(f"Pose differences: {self.robots[0].links[-3].get_pose().p}")
            # print(f"Qpos: {qpos[:3]}")
            # print(f"Obj pose: {self.objects[0].get_pose().p}")
            self.scene.update_render()
            if self.headless:
                #old_pose = self.camera.get_local_pose()
                #self.camera.set_local_pose(sapien.Pose([2.5, 0.3, 3.3], [0.9447637 , 0.32630064, 0.02912487, 0.01005909]))
                self.camera.take_picture()
                rgb = self.camera.get_picture("Color")[..., :3]
                self.camera.set_local_pose(old_pose)
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                writer.write(rgb[..., ::-1])
            else:
                for k in range(start_frame):
                    self.viewer.render()
        #breakpoint()
        if self.headless:
            writer.release()
        else:
            self.viewer.paused = True
            self.viewer.render()
        return robot_poses, object_poses


class TwoRobotHandDatasetSAPIENViewer(TwoHandDatasetSAPIENViewer):
    def __init__(self, robot_names: List[RobotName], headless=False, use_ray_tracing=False):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[List[sapien.Articulation]] = []
        self.robot_file_names: List[List[str]] = []
        self.retargetings: List[List[SeqRetargeting]] = []
        self.retarget2sapien: List[List[np.ndarray]] = []

        # Load optimizer and filter
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        for robot_name in robot_names:
            left_config_path = get_default_config_path(robot_name, RetargetingType.position, HandType.left)
            right_config_path = get_default_config_path(robot_name, RetargetingType.position, HandType.right)

            # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
            override = dict(add_dummy_free_joint=True)
            left_config = RetargetingConfig.load_from_file(left_config_path, override=override)
            right_config = RetargetingConfig.load_from_file(right_config_path, override=override)
            left_retargeting = left_config.build()
            right_retargeting = right_config.build()
            left_robot_file_name = Path(left_config.urdf_path).stem
            right_robot_file_name = Path(right_config.urdf_path).stem
            self.robot_file_names.append([left_robot_file_name, right_robot_file_name])
            self.retargetings.append([left_retargeting, right_retargeting])

            # Build robot
            left_urdf_path = Path(left_config.urdf_path)
            right_urdf_path = Path(right_config.urdf_path)
            if "glb" not in left_urdf_path.stem:
                left_urdf_path = left_urdf_path.with_stem(left_urdf_path.stem + "_glb")
            if "glb" not in right_urdf_path.stem:
                right_urdf_path = right_urdf_path.with_stem(right_urdf_path.stem + "_glb")
            left_robot_urdf = urdf.URDF.load(str(left_urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
            right_robot_urdf = urdf.URDF.load(str(right_urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
            left_urdf_name = left_urdf_path.name
            right_urdf_name = right_urdf_path.name
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            left_temp_path = f"{temp_dir}/{left_urdf_name}"
            left_robot_urdf.write_xml_file(left_temp_path)
            right_temp_path = f"{temp_dir}/{right_urdf_name}"
            right_robot_urdf.write_xml_file(right_temp_path)

            left_robot = loader.load(left_temp_path)
            right_robot = loader.load(right_temp_path)
            self.robots.append([left_robot, right_robot])
            left_sapien_joint_names = [joint.name for joint in left_robot.get_active_joints()]
            right_sapien_joint_names = [joint.name for joint in right_robot.get_active_joints()]
            left_retarget2sapien = np.array([left_retargeting.joint_names.index(n) for n in left_sapien_joint_names]).astype(int)
            right_retarget2sapien = np.array([right_retargeting.joint_names.index(n) for n in right_sapien_joint_names]).astype(int)
            self.retarget2sapien.append((left_retarget2sapien, right_retarget2sapien))

    def load_object_hand(self, data: Dict):
        super().load_object_hand(data)
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]

        # Load the same YCB objects for n times, n is the number of robots
        # So that for each robot, there will be an identical set of objects
        for _ in range(len(self.robots)):
            for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
                self._load_ycb_object(ycb_id, ycb_mesh_file)

    def render_dexycb_data(self, data: Dict, fps=5, y_offset=0.8):
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2
        #self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([0.5, global_y_offset + 0.75, 1.7]))
            self.camera.set_local_pose(local_pose)

        left_hand_pose = data["left_hand_pose"]
        right_hand_pose = data["right_hand_pose"]
        object_pose = data["object_pose"]
        num_frame = left_hand_pose.shape[0]
        num_copy = len(self.robots) + 1
        num_ycb_objects = len(data["ycb_ids"])
        pose_offsets = []

        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1][0].set_pose(pose)
                self.robots[i - 1][1].set_pose(pose)

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        for i in range(0, num_frame):
            init_left_hand_pose_frame = left_hand_pose[i]
            init_right_hand_pose_frame = right_hand_pose[i]
            left_vertex, joint = self._compute_hand_geometry(init_left_hand_pose_frame, right=False)
            right_vertex, joint = self._compute_hand_geometry(init_right_hand_pose_frame, right=True)
            if left_vertex is not None and right_vertex is not None:
                start_frame = i
                break

        if self.headless:
            robot_names = [robot.name for robot in self.robot_names]
            robot_names = "_".join(robot_names)
            video_path = Path(__file__).parent.resolve() / f"data/{robot_names}_video.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (self.camera.get_width(), self.camera.get_height())
            )

        # Loop rendering
        step_per_frame = int(60 / fps)
        all_obj_poses = []
        for _ in range(len(objects)):
            all_obj_poses.append([])
        for i in trange(start_frame, num_frame):
            object_pose_frame = object_pose[i]
            right_hand_pose_frame = right_hand_pose[i]
            left_hand_pose_frame = left_hand_pose[i]

            left_vertex, left_joint = self._compute_hand_geometry(left_hand_pose_frame, right=False)
            right_vertex, right_joint = self._compute_hand_geometry(right_hand_pose_frame, right=True)
            

            # Update poses for YCB objects
            i = 0
            for k in range(num_ycb_objects):
                pos_quat = object_pose_frame[k]
                qt = rotation_matrix_to_quaternion(pos_quat)
                # Quaternion convention: xyzw -> wxyz
                pose = self.camera_pose * sapien.Pose(pos_quat[:3, 3], qt)
                self.objects[k].set_pose(pose)
                for copy_ind in range(num_copy):
                    self.objects[k + copy_ind * num_ycb_objects].set_pose(pose_offsets[copy_ind] * pose)
                    all_obj_poses[i].append((pose_offsets[copy_ind] * pose).p)
                    i += 1

            # Update pose for human hand
            self._update_hand(left_vertex, right_vertex)

            # Update poses for robot hands
            for robot, retargeting, retarget2sapien in zip(self.robots, self.retargetings, self.retarget2sapien):
                #print(f"lens: {len(robot), len(retargeting), len(retarget2sapien)}")
                for i in range(2):
                    indices = retargeting[i].optimizer.target_link_human_indices
                    joint = left_joint if i == 0 else right_joint
                    ref_value = joint[indices, :]
                    qpos = retargeting[i].retarget(ref_value)[retarget2sapien[i]]
                    robot[i].set_qpos(qpos)

            self.scene.update_render()
            if self.headless:
                self.camera.take_picture()
                rgb = self.camera.get_picture("Color")[..., :3]
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                writer.write(rgb[..., ::-1])
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()
        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        else:
            writer.release()

# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]
# Modified by Yuzhe Qin to use the sequential information inside the dataset

"""DexYCB dataset."""

from pathlib import Path

import numpy as np
import yaml
import os

_SUBJECTS = [
    "20200709-subject-01",
    "20200813-subject-02",
    "20200820-subject-03",
    "20200903-subject-04",
    "20200908-subject-05",
    "20200918-subject-06",
    "20200928-subject-07",
    "20201002-subject-08",
    "20201015-subject-09",
    "20201022-subject-10",
]

_TACO_SUBJECTS = [
    '(stir, spoon, bowl)',
    '(brush, brush, plate)',
    '(brush, brush, bowl)',
    '(cut, spatula, pan)',
    '(empty, cup, bowl)',
    '(pour, in, some, bowl, kettle)',
    '(put, in, spatula, pan)',
    '(dust, brush, box)',
    '(measure, ruler, bowl)',
    '(pour, in, some, teapot, cup)'
]

YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

_MANO_JOINTS = [
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]

_MANO_JOINT_CONNECT = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
]

_SERIALS = [
    "836212060125",
    "839512060362",
    "840412060917",
    "841412060263",
    "932122060857",
    "932122060861",
    "932122061900",
    "932122062010",
]

_BOP_EVAL_SUBSAMPLING_FACTOR = 4


class DexYCBVideoDataset:
    def __init__(self, data_dir, hand_type="right", filter_objects=[]):
        self._data_dir = Path(data_dir)
        self._calib_dir = self._data_dir / "calibration"
        self._model_dir = self._data_dir / "models"
        self._hand_dir = self._data_dir / "Hand_Poses"

        # Filter
        self.use_filter = len(filter_objects) > 0
        inverse_ycb_class = {"_".join(value.split("_")[1:]): key for key, value in YCB_CLASSES.items()}  # TODO: remoev this
        ycb_object_names = list(inverse_ycb_class.keys()) # TODO: remoev this
        filter_ids = []
        for obj in filter_objects:
            if obj not in ycb_object_names:
                print(f"Filter object name {obj} is not a valid YCB name")
            else:
                filter_ids.append(inverse_ycb_class[obj])

        # Camera and mano
        self._intrinsics, self._extrinsics = self._load_camera_parameters()
        self._mano_side = hand_type
        self._mano_parameters = self._load_mano()

        # Capture data
        self._subject_dirs = [sub for sub in self._hand_dir.iterdir() if sub.stem in _TACO_SUBJECTS]
        self._capture_meta = {}
        self._capture_pose = {}
        self._capture_filter = {}
        self._captures = []
        for subject_dir in self._subject_dirs:
            for capture_dir in subject_dir.iterdir():
                if hand_type == 'right':
                    meta_file = capture_dir / "right_hand.pkl"
                else:
                    meta_file = capture_dir / "left_hand.pkl"
                hand_pose = np.load(meta_file, allow_pickle=True)
                # print(capture_dir)
                # change hand to object in the capture dir string
                
                object_pose_path = str(capture_dir).replace('Hand_Poses', 'Object_Poses')
                # object_pose_path = capture_dir
                sequence_key = object_pose_path.split('/')[-2] + '-' + object_pose_path.split('/')[-1]
                target = None
                tool = None
                target_object_pose = None
                tool_object_pose = None
                for file in os.listdir(object_pose_path):
                    if file.split('_')[0] == 'target':
                        target_object_pose = np.load(os.path.join(object_pose_path, file))
                        target = int(file.split('_')[1].split('.')[0])
                    else:
                        tool_object_pose = np.load(os.path.join(object_pose_path, file))
                        tool = int(file.split('_')[1].split('.')[0])
                
                hand_pose_list = np.array([np.concatenate((v['hand_pose'], v['hand_trans'])) for v in hand_pose.values()])

                pose = {
                    'hand_pose': hand_pose_list,
                    'target_object_pose': target_object_pose,
                    'tool_object_pose': tool_object_pose,
                }
                meta = {
                    'target': target,
                    'tool': tool,
                    'sequence_key': sequence_key,
                }
                self._capture_meta[sequence_key] = meta
                self._capture_pose[sequence_key] = pose
                self._captures.append(sequence_key)

    def __len__(self):
        return len(self._captures)

    def __getitem__(self, item):
        # item = 7
        # print(item, self.__len__())
        # if item > self.__len__():
        #     raise ValueError(f"Index {item} out of range")

        capture_name = self._captures[item]
        meta = self._capture_meta[capture_name]
        pose = self._capture_pose[capture_name]
        hand_pose = np.expand_dims(pose["hand_pose"], axis=1)
        # print(hand_pose)
        # print(hand_pose.shape)
        # print(capture_name)
        # exit()
        object_pose = np.expand_dims(pose["tool_object_pose"], axis=1)
        ycb_ids = [meta["tool"]]
        
        # Load extrinsic and mano parameters
        extrinsic_mat = self._extrinsics
        # extrinsic_mat = np.array(self._extrinsics[extrinsic_name]["extrinsics"]["apriltag"]).reshape([3, 4])
        # extrinsic_mat = np.concatenate([extrinsic_mat, np.array([[0, 0, 0, 1]])], axis=0)
        # mano_name = meta["mano_calib"][0]
        mano_parameters = self._mano_parameters

        if self.use_filter:
            capture_filter = np.array(self._capture_filter[capture_name])
            frame_indices, _ = self._filter_object_motion_frame(capture_filter, object_pose)
            ycb_ids = [ycb_ids[valid_id] for valid_id in self._capture_filter[capture_name]]
            hand_pose = hand_pose[frame_indices]
            object_pose = object_pose[frame_indices][:, capture_filter, :]
        
        object_mesh_files = [str(self._data_dir) + f"/Object_Models/object_models_released/{meta['tool']:03}_cm.obj"]
        

        ycb_data = dict(
            hand_pose=hand_pose,
            object_pose=object_pose,
            extrinsics=extrinsic_mat,
            ycb_ids=ycb_ids,
            hand_shape=mano_parameters,
            object_mesh_file=object_mesh_files,
            capture_name=capture_name,
        )
        return ycb_data

    def _filter_object_motion_frame(self, capture_filter, object_pose, frame_margin=40):
        frames = np.arange(0)
        for filter_id in capture_filter:
            filter_object_pose = object_pose[:, filter_id, :]
            object_move_list = []
            for frame in range(filter_object_pose.shape[0] - 2):
                object_move_list.append(self.is_object_move(filter_object_pose[frame:, :]))
            if True not in object_move_list:
                continue
            first_frame = object_move_list.index(True)
            last_frame = len(object_move_list) - object_move_list[::-1].index(True) - 1
            start = max(0, first_frame - frame_margin)
            end = min(filter_object_pose.shape[0], last_frame + frame_margin)
            frames = np.arange(start, end)
            break
        return frames, filter_id

    @staticmethod
    def is_object_move(single_object_pose: np.ndarray):
        single_object_trans = single_object_pose[:, 4:]
        future_frame = min(single_object_trans.shape[0] - 1, 5)
        current_move = np.linalg.norm(single_object_trans[1] - single_object_trans[0]) > 2e-2
        future_move = np.linalg.norm(single_object_trans[future_frame] - single_object_trans[0]) > 5e-2
        return current_move or future_move

    def _object_mesh_file(self, object_id):
        obj_file = self._data_dir / "models" / YCB_CLASSES[object_id] / "textured_simple.obj"
        return str(obj_file.resolve())

    def _load_camera_parameters(self):
        intrinsics = np.float32([
                        [9533.359863759411, 0.0, 2231.699969508665],
                        [0.0, 9593.722282299485, 1699.3865932992662],
                        [0.0,0.0,1.0]
                    ]) / 4.0
        extrinsics = np.float32([
                        [-9.861640182402281463e-01, -6.260122508879048531e-02, 1.534979340110796397e-01, -2.296398434650619436e-01],
                        [-1.649820266894134468e-01, 2.803133276113436434e-01, -9.456243277501432676e-01, 1.459474531833096833e+00],
                        [1.616972472681078854e-02, -9.578650870455813759e-01, -2.867629945118802537e-01, 1.091449991261020935e+00],
                        [0, 0, 0, 1],
                    ])
        extrinsics = np.linalg.inv(extrinsics)

        return intrinsics, extrinsics

    def _load_mano(self):
        # mano_parameters = {}
        # for cali_dir in self._calib_dir.iterdir():
        #     if not cali_dir.stem.startswith("mano"):
        #         continue

        #     mano_file = cali_dir / "mano.yml"
        #     with mano_file.open(mode="r") as f:
        #         shape_parameters = yaml.load(f, Loader=yaml.FullLoader)
        #     mano_name = "_".join(cali_dir.stem.split("_")[1:])
        #     mano_parameters[mano_name] = np.array(shape_parameters["betas"])
        
        mano_parameters = np.array([1.1689, -2.7789, -1.7785,  2.7776,  3.1024, -2.6009,  4.9058,  5.0095, -1.2958,  1.0639])

        return mano_parameters

class TwoHandDexYCBVideoDataset:
    def __init__(self, data_dir, filter_objects=[]):
        self._data_dir = Path(data_dir)
        self._calib_dir = self._data_dir / "calibration"
        self._model_dir = self._data_dir / "models"
        self._hand_dir = self._data_dir / "Hand_Poses"

        # Filter
        self.use_filter = len(filter_objects) > 0
        inverse_ycb_class = {"_".join(value.split("_")[1:]): key for key, value in YCB_CLASSES.items()}  # TODO: remoev this
        ycb_object_names = list(inverse_ycb_class.keys()) # TODO: remoev this
        filter_ids = []
        for obj in filter_objects:
            if obj not in ycb_object_names:
                print(f"Filter object name {obj} is not a valid YCB name")
            else:
                filter_ids.append(inverse_ycb_class[obj])

        # Camera and mano
        self._intrinsics, self._extrinsics = self._load_camera_parameters()
        #self._mano_side = hand_type
        self._left_mano_parameters, self._right_mano_parameters = self._load_mano()

        # Capture data
        self._subject_dirs = [sub for sub in self._hand_dir.iterdir() if sub.stem in _TACO_SUBJECTS]
        self._capture_meta = {}
        self._capture_pose = {}
        self._capture_filter = {}
        self._captures = []
        for subject_dir in self._subject_dirs:
            for capture_dir in subject_dir.iterdir():
                right_meta_file = capture_dir / "right_hand.pkl"
                left_meta_file = capture_dir / "left_hand.pkl"
                left_hand_pose = np.load(left_meta_file, allow_pickle=True)
                right_hand_pose = np.load(right_meta_file, allow_pickle=True)
                # print(capture_dir)
                # change hand to object in the capture dir string
                
                object_pose_path = str(capture_dir).replace('Hand_Poses', 'Object_Poses')
                # object_pose_path = capture_dir
                sequence_key = object_pose_path.split('/')[-2] + '-' + object_pose_path.split('/')[-1]
                target = None
                tool = None
                target_object_pose = None
                tool_object_pose = None
                for file in os.listdir(object_pose_path):
                    if file.split('_')[0] == 'target':
                        target_object_pose = np.load(os.path.join(object_pose_path, file))
                        target = int(file.split('_')[1].split('.')[0])
                    else:
                        tool_object_pose = np.load(os.path.join(object_pose_path, file))
                        tool = int(file.split('_')[1].split('.')[0])
                
                left_hand_pose_list = np.array([np.concatenate((v['hand_pose'], v['hand_trans'])) for v in left_hand_pose.values()])
                right_hand_pose_list = np.array([np.concatenate((v['hand_pose'], v['hand_trans'])) for v in right_hand_pose.values()])
                pose = {
                    'left_hand_pose': left_hand_pose_list,
                    'right_hand_pose': right_hand_pose_list,
                    'target_object_pose': target_object_pose,
                    'tool_object_pose': tool_object_pose,
                }
                meta = {
                    'target': target,
                    'tool': tool,
                    'sequence_key': sequence_key,
                }
                self._capture_meta[sequence_key] = meta
                self._capture_pose[sequence_key] = pose
                self._captures.append(sequence_key)

    def __len__(self):
        return len(self._captures)

    def __getitem__(self, item):
        # item = 7
        # print(item, self.__len__())
        # if item > self.__len__():
        #     raise ValueError(f"Index {item} out of range")

        capture_name = self._captures[item]
        meta = self._capture_meta[capture_name]
        pose = self._capture_pose[capture_name]
        left_hand_pose = np.expand_dims(pose["left_hand_pose"], axis=1)
        right_hand_pose = np.expand_dims(pose["right_hand_pose"], axis=1)
        # print(hand_pose)
        # print(hand_pose.shape)
        # print(capture_name)
        # exit()
        object_pose = np.concatenate((
            np.expand_dims(pose["tool_object_pose"], axis=1),
            np.expand_dims(pose["target_object_pose"], axis=1)
        ), axis=1)
        ycb_ids = [meta["tool"]]
        
        # Load extrinsic and mano parameters
        extrinsic_mat = self._extrinsics
        # extrinsic_mat = np.array(self._extrinsics[extrinsic_name]["extrinsics"]["apriltag"]).reshape([3, 4])
        # extrinsic_mat = np.concatenate([extrinsic_mat, np.array([[0, 0, 0, 1]])], axis=0)
        # mano_name = meta["mano_calib"][0]
        left_mano_parameters, right_mano_parameters = self._left_mano_parameters, self._right_mano_parameters

        if self.use_filter:
            capture_filter = np.array(self._capture_filter[capture_name])
            frame_indices, _ = self._filter_object_motion_frame(capture_filter, object_pose)
            ycb_ids = [ycb_ids[valid_id] for valid_id in self._capture_filter[capture_name]]
            hand_pose = hand_pose[frame_indices]
            object_pose = object_pose[frame_indices][:, capture_filter, :]
        
        object_mesh_files = [
            str(self._data_dir) + f"/Object_Models/object_models_released/{meta['tool']:03}_cm.obj",
            str(self._data_dir) + f"/Object_Models/object_models_released/{meta['target']:03}_cm.obj"
        ]
        

        ycb_data = dict(
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            object_pose=object_pose,
            extrinsics=extrinsic_mat,
            ycb_ids=[i + 1 for i in range(len(object_mesh_files))], #ycb_ids,
            left_hand_shape=left_mano_parameters,
            right_hand_shape=right_mano_parameters,
            object_mesh_file=object_mesh_files,
            capture_name=capture_name,
        )
        return ycb_data

    def _filter_object_motion_frame(self, capture_filter, object_pose, frame_margin=40):
        frames = np.arange(0)
        for filter_id in capture_filter:
            filter_object_pose = object_pose[:, filter_id, :]
            object_move_list = []
            for frame in range(filter_object_pose.shape[0] - 2):
                object_move_list.append(self.is_object_move(filter_object_pose[frame:, :]))
            if True not in object_move_list:
                continue
            first_frame = object_move_list.index(True)
            last_frame = len(object_move_list) - object_move_list[::-1].index(True) - 1
            start = max(0, first_frame - frame_margin)
            end = min(filter_object_pose.shape[0], last_frame + frame_margin)
            frames = np.arange(start, end)
            break
        return frames, filter_id

    @staticmethod
    def is_object_move(single_object_pose: np.ndarray):
        single_object_trans = single_object_pose[:, 4:]
        future_frame = min(single_object_trans.shape[0] - 1, 5)
        current_move = np.linalg.norm(single_object_trans[1] - single_object_trans[0]) > 2e-2
        future_move = np.linalg.norm(single_object_trans[future_frame] - single_object_trans[0]) > 5e-2
        return current_move or future_move

    def _object_mesh_file(self, object_id):
        obj_file = self._data_dir / "models" / YCB_CLASSES[object_id] / "textured_simple.obj"
        return str(obj_file.resolve())

    def _load_camera_parameters(self):
        intrinsics = np.float32([
                        [9533.359863759411, 0.0, 2231.699969508665],
                        [0.0, 9593.722282299485, 1699.3865932992662],
                        [0.0,0.0,1.0]
                    ]) / 4.0
        extrinsics = np.float32([
                        [-9.861640182402281463e-01, -6.260122508879048531e-02, 1.534979340110796397e-01, -2.296398434650619436e-01],
                        [-1.649820266894134468e-01, 2.803133276113436434e-01, -9.456243277501432676e-01, 1.459474531833096833e+00],
                        [1.616972472681078854e-02, -9.578650870455813759e-01, -2.867629945118802537e-01, 1.091449991261020935e+00],
                        [0, 0, 0, 1],
                    ])
        extrinsics = np.linalg.inv(extrinsics)

        return intrinsics, extrinsics

    def _load_mano(self): 
        left_mano_parameters = np.array([0.8413, -2.9855, -0.9436, -2.0170, -0.3040,  5.3951,  5.9580, -1.7685, 0.4152,  5.0897])
        right_mano_parameters = np.array([1.1689, -2.7789, -1.7785,  2.7776,  3.1024, -2.6009,  4.9058,  5.0095, -1.2958,  1.0639])

        return left_mano_parameters, right_mano_parameters


def main(dexycb_dir: str):
    from collections import Counter

    dataset = DexYCBVideoDataset(dexycb_dir)
    print(len(dataset))

    ycb_names = []
    for i, data in enumerate(dataset):
        ycb_ids = data["ycb_ids"][0]
        ycb_names.append(YCB_CLASSES[ycb_ids])

    counter = Counter(ycb_names)
    print(counter)

    sample = dataset[0]
    print(sample.keys())


if __name__ == "__main__":
    import tyro

    tyro.cli(main)

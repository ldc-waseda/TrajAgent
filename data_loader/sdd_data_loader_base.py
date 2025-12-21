import os
import math
import pickle
import numpy as np
# import pandas as pd
import torch
import networkx as nx
from util import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

DELIMITER_t = '\t'
DELIMITER_space = ' '
#     1   Track ID. All rows with the same ID belong to the same path.
#     2   xmin. The top left x-coordinate of the bounding box.
#     3   ymin. The top left y-coordinate of the bounding box.
#     4   xmax. The bottom right x-coordinate of the bounding box.
#     5   ymax. The bottom right y-coordinate of the bounding box.
#     6   frame. The frame that this annotation represents.
#     7   lost. If 1, the annotation is outside of the view screen.
#     8   occluded. If 1, the annotation is occluded.
#     9   generated. If 1, the annotation was automatically interpolated.
#     10  label. The label for this annotation, enclosed in quotation marks.
class SDDLoader():
    def __init__(self, datasets_path):
        self.datasets_path = datasets_path
        self.load_data()
        self.preprocess_data()
        self.sort_by_frame()
        # self.generate_input_data()
        # self.generate_norm_input_data()
        # self.surrounding_infor()
        # self.kf = self.calculate_kinematics_features(self.target_obs)
        # self.generate_kbs_feature()
        # print(self.kinematics_features.shape)
    def load_data(self):
        print("Loading SDD data from", self.datasets_path)
        data_file = open(self.datasets_path, "r")
        total_lines = sum(1 for line in data_file)  # 计算总行数
        data_file.seek(0)  # 重置文件指针到起始位置
        traj_data = []
        for line in tqdm(data_file, total=total_lines, desc=f"Loading {self.datasets_path}"):
            # 处理每一行
            line = line.strip().split(DELIMITER_space) # original data type ID xmin ymin xmax ymax frame loss occluded generated label
            infor = line
            id = infor[0]
            xmin = int(infor[1])
            ymin = int(infor[2])
            xmax = int(infor[3])
            ymax = int(infor[4])

            pos_x = xmin + (xmax - xmin) / 2
            pos_y = ymin + (ymax - ymin) / 2

            frame = infor[5]
            style = infor[-1]
            if infor[6] == 0 or infor[7] == 0:
                continue
            # line_data = [id, frame, pos_x, pos_y, style]

            # 跳过 "Biker" 行
            if style != '"Pedestrian"':
                # print('skip as', style)
                continue
            else:
                # print('record as', style)
                traj_data.append([frame, id, pos_x, pos_y]) #same index as ETH datasets
        self.processed_data = np.asarray(traj_data, dtype=int)

    def preprocess_data(self):
        # self.h_data, self.xy_tag = generate_dataset_tag(self.datasets_path)
        # original_data_pos = self.processed_data[:,[2,3]]
        # pixel_data = trajectory2pixel(original_data_pos, self.h_data)
        # self.processed_data = np.concatenate((self.processed_data[:,[0,1]], pixel_data), axis=1).astype(int)
        # print(self.processed_data.shape) # Nx6 #  self.processed_data  [frame, id, x_original, y_original, x_pixel, y_pixel] 
        pass
    def sort_by_frame(self):
        # get the target obs which frame over 20
        interval = 30
        num_samples = 20
        short_term_target_obs = []
        long_term_target_obs = []
        X = []
        y = []
        traj_data = self.processed_data
        ids = np.unique(traj_data[:, 1])
        for id_ in ids:
            tmp = traj_data[traj_data[:, 1] == id_, :]
            # 如果数据足够帧：(20-1)*12 + 1 = 229
            if tmp.shape[0] >= (num_samples - 1) * interval + 1:
                # 按 12 帧间隔采样，再取前 20 个
                sampled = tmp[::interval][:num_samples]
                start_xy = sampled[0, [2, 3]]
                end_xy   = sampled[8, [2, 3]]
                if np.linalg.norm(end_xy - start_xy) < 20:
                    continue
                short_term_target_obs.append(sampled)

        self.target_obs = np.asarray(short_term_target_obs)
        # print(self.target_obs)
    def calculate_kinematics_features(self, feat):
        if feat.ndim != 3 or 0 in feat.shape:
            return None  # 或者 return [] / return torch.empty(0) / 按你的下游需求
        N, T, C =feat.shape
        frame_ = feat[..., 0]
        # frame_min = frame_.min().item()  
        # frame_norm = (frame_ - frame_min) / 20.0  
        id_    = feat[..., 1]
        # ========= 0) pos =========
        x_ = feat[..., 2]  # [N, 20]
        y_ = feat[..., 3]  # [N, 20]
        x_rel = x_ - x_[:, 0:1]
        y_rel = y_ - y_[:, 0:1]
        x_min = x_.min(axis=1, keepdims=True)
        x_max = x_.max(axis=1, keepdims=True)
        x_norm = (x_ - x_min) / (x_max - x_min + 1e-8)  # 加上1e-8防止除0
        # 对 y_rel 进行 min-max 归一化
        y_min = y_.min(axis=1, keepdims=True)
        y_max = y_.max(axis=1, keepdims=True)
        y_norm = (y_ - y_min) / (y_max - y_min + 1e-8)
        # ========= 1) heading =========
        baseline_angle = np.arctan2(y_rel[:, 1], x_rel[:, 1])  # shape: [N,]
        angles = np.arctan2(y_rel, x_rel)  # shape: [N, T]
        relative_angles = angles - baseline_angle[:, np.newaxis]
        heading_rel = (relative_angles + np.pi) % (2 * np.pi) - np.pi
        sin_heading = np.sin(heading_rel)
        cos_heading = np.cos(heading_rel)

        
        # ========= 2) 速度 (vx, vy) =========
        vx_ = np.zeros((N, T))
        vy_ = np.zeros((N, T))
        vx_[:, 1:] = x_[:, 1:] - x_[:, :-1]
        vy_[:, 1:] = y_[:, 1:] - y_[:, :-1]
        vx_[:, 0] = vx_[:, 1]
        vy_[:, 0] = vy_[:, 1]
        # 对 vx_ 进行 min-max 归一化
        vx_min = vx_.min(axis=1, keepdims=True)
        vx_max = vx_.max(axis=1, keepdims=True)
        vx_norm = (vx_ - vx_min) / (vx_max - vx_min + 1e-8)

        # 对 vy_ 进行 min-max 归一化
        vy_min = vy_.min(axis=1, keepdims=True)
        vy_max = vy_.max(axis=1, keepdims=True)
        vy_norm = (vy_ - vy_min) / (vy_max - vy_min + 1e-8)
        speed_   = np.sqrt(vx_**2 + vy_**2)            # [N,20]
        # ========= 3) 加速度 (ax, ay) =========
        ax_ = np.zeros((N, T))
        ay_ = np.zeros((N, T))
        ax_[:, 2:] = vx_[:, 2:] - vx_[:, 1:-1]
        ay_[:, 2:] = vy_[:, 2:] - vy_[:, 1:-1]
        ax_[:, 0] = ax_[:, 2]
        ax_[:, 1] = ax_[:, 2]
        ay_[:, 0] = ay_[:, 2]
        ay_[:, 1] = ay_[:, 2]

        # 对 ax_ 进行标准化
        ax_mean = ax_.mean(axis=1, keepdims=True)
        ax_std = ax_.std(axis=1, keepdims=True)
        ax_norm = (ax_ - ax_mean) / (ax_std + 1e-8)

        # 对 ay_ 进行标准化
        ay_mean = ay_.mean(axis=1, keepdims=True)
        ay_std = ay_.std(axis=1, keepdims=True)
        ay_norm = (ay_ - ay_mean) / (ay_std + 1e-8)
        # ========= 4) jerk (加加速度) =========
        # jerk_x[t] = ax[t] - ax[t-1], jerk_y[t] = ay[t] - ay[t-1]
        jerk_x_ = np.zeros((N, T))
        jerk_y_ = np.zeros((N, T))
        jerk_x_[:, 3:] = ax_[:, 3:] - ax_[:, 2:-1]
        jerk_y_[:, 3:] = ay_[:, 3:] - ay_[:, 2:-1]
        # jerk magnitude
        jerk_ = np.sqrt(jerk_x_**2 + jerk_y_**2)   # [N,20]

        # ========= 5) 曲率 curvature =========
        # 2D离散近似: curvature[t] = 
        #   | vx[t]*ay[t] - vy[t]*ax[t] | / (speed[t]^3 + eps)
        # speed[t]^3 可能为0 => 加个小epsilon防止NaN
        eps = 1e-8
        numerator = np.abs(vx_ * ay_ - vy_ * ax_)
        denom = np.power(speed_, 3) + eps   # speed^3
        curvature_ = numerator / denom      # [N,20]

        arr_list = [
                frame_, id_, x_, y_, x_rel, y_rel, x_norm, y_norm,
                sin_heading, cos_heading, speed_,
                vx_, vy_, vx_norm, vy_norm,
                ax_, ay_, ax_norm, ay_norm,
                curvature_, jerk_x_, jerk_y_, jerk_, 
                    ]

        return np.stack(arr_list, axis=-1)
        
    
    def generate_kbs_feature(self):
        traj = self.target_obs[..., [2,3]]
        traj_proj, final_y, direction, angle = project_trajectory_and_compute_offset_batch(traj, threshold_deg=20)
        direction = direction.reshape(-1, 1, 3)        
        direction = np.repeat(direction, repeats=20, axis=1)
        self.kinematics_features = np.concatenate([self.kinematics_features, direction], axis=-1)

    def generate_input_data(self):
        print(self.target_obs.shape)
        self.agent_train_data = self.target_obs[:, 0:8, :]
        self.agent_gt_data = self.target_obs[:, 8:, :]

    def generate_norm_input_data(self):
        total_train_norm_data = []
        total_gt_norm_data = []
        for train, gt in zip(self.agent_train_data, self.agent_gt_data):
            train_infor = train[:, [0, 1]]
            gt_infor = gt[:, [0, 1]]
            norm_train = train[:, 2:] - train[0, 2:]
            norm_gt = gt[:, 2:] - train[0, 2:]
            norm_train_ = np.concatenate((train_infor, norm_train), axis=1)
            norm_gt_ = np.concatenate((gt_infor, norm_gt), axis=1)
            total_train_norm_data.append(norm_train_)
            total_gt_norm_data.append(norm_gt_)
        self.agent_train_norm_data = np.asarray(total_train_norm_data)
        self.agent_gt_norm_data = np.asarray(total_gt_norm_data)
        
    def surrounding_infor(self):
        # get the target obs surrounding frame data size: target_obs_num x 20 x sur_obs_num x 6
        total_surrounding_frame = []
        # tqdm 进度条，遍历 target_obs
        for data in tqdm(self.target_obs, desc="Processing surrounding observations"):
            surrounding_frame = []  # 存储当前目标观测的邻居帧
            target_obs_id = data[0, 1]  # 获取目标观测的 ID
            target_obs_frames = data[:, 0]  # 获取目标观测的帧
            # 遍历目标观测帧
            for frame in target_obs_frames:
                # 从 processed_data 中找到对应帧
                target_bool = self.processed_data[frame == self.processed_data[:, 0]]

                # 排除与当前 target_obs_id 相同的记录（即当前目标本身）
                target_bool_ = target_bool[target_obs_id != target_bool[:, 1]]

                # 保存邻居信息
                surrounding_frame.append(target_bool_)

            # 将每个目标观测的邻居帧数据加入总列表
            total_surrounding_frame.append(surrounding_frame)
        self.surrounding_frames = np.asarray(total_surrounding_frame, dtype=object)
    
    def generate_surrounding_obs_norm_data(self, max_obstacles=5, fill_value=0):
        surrounding_train_data = self.surrounding_frames[:, 0:8]
        total_norm_sur_data = []
        for agent_idx, (agent_target_data, sur_data) in enumerate(
                tqdm(zip(self.agent_train_data, surrounding_train_data),
                     desc="Processing agents surrounding obs norm data",
                     total=len(self.agent_train_data))):
            frame_norm_sur_data = []
            # print("current agent_target_data ", agent_target_data.shape)
            # print("current sur_data ", sur_data.shape)
            # 内层循环：遍历每个frame
            for frame_idx, (frame_target_data, frame_sur_data) in enumerate(zip(agent_target_data, sur_data)):
                # 计算每个障碍物到agent的距离
                # 假设 frame_target_data 和 frame_sur_data 的格式为 [id, ... , pos_x, pos_y, ...]
                agent_pos = frame_target_data[2:].astype(int)  # agent的 (x, y) 位置
                obstacles_pos = frame_sur_data[:, 2:].astype(int)  # 障碍物的 (x, y) 位置
                # 计算欧几里得距离
                distances = np.linalg.norm(obstacles_pos - agent_pos, axis=1)
                # 获取排序索引，选择距离最近的max_obstacles个
                sorted_indices = np.argsort(distances)
                top_indices = sorted_indices[:max_obstacles]
                # 选择最近的障碍物
                top_obstacles = frame_sur_data[top_indices, :]
                # 如果障碍物数量少于max_obstacles，进行填充
                num_obstacles = top_obstacles.shape[0]
                if num_obstacles < max_obstacles:
                    # 创建填充数组，形状为 (max_obstacles - num_obstacles, 6)
                    padding = np.full((max_obstacles - num_obstacles, frame_sur_data.shape[1]), fill_value)
                    top_obstacles = np.vstack((top_obstacles, padding))
                # 规范化轨迹数据
                norm_frame_sur_traj_data = top_obstacles[:, [2, 3]].astype(int) - agent_pos.astype(int)
                # 拼接 ID 和规范化的轨迹数据
                norm_frame_sur_infor_data = np.concatenate(
                    (top_obstacles[:, [0, 1]].astype(int), norm_frame_sur_traj_data),
                    axis=1
                )
                frame_norm_sur_data.append(norm_frame_sur_infor_data)
            # 将当前agent的所有frames的障碍物信息添加到总列表
            total_norm_sur_data.append(frame_norm_sur_data)

            # 将列表转换为NumPy数组
        total_norm_sur_data = np.asarray(total_norm_sur_data)
        # 检查生成的数组形状
        # print(f"Shape of total_norm_sur_data: {total_norm_sur_data.shape}")  # 例如 (num_agents, 8, 4, 4)

        # 划分训练数据和 GT 数据
        # 假设 total_norm_sur_data 的形状为 (num_agents, 20, 4, 4)
        # 其中前8帧作为训练数据，后12帧作为 GT 数据
        self.sur_train_norm_data = total_norm_sur_data[:, 0:8, :, :]  # 形状: (num_agents, 8, 4, 4)
        self.sur_gt_norm_data = total_norm_sur_data[:, 8:20, :, :]   # 形状: (num_agents, 12, 4, 4)

class TrainingLoader(Dataset):
    def __init__(self, input_data,):
        self.input_data =  torch.tensor(input_data, dtype=torch.float32) # Nx20x2
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return {
            'input': self.input_data[idx], # Bxseq_lenxfeature_num Bx20x10
        }


def load_sdd_paths(sdd_root, scene_list):
    """
    遍历 SDD 数据集目录，返回两个 dict：
      - scenario_txt_path[scene]   = [所有 segment 下的 ann.txt 路径]
      - scenario_video_path[scene] = [所有 segment 下的 video.mov 路径]
    """
    scenario_txt_path = {}
    scenario_video_path = {}

    for scene in scene_list:
        ann_scene_dir = os.path.join(sdd_root, 'annotations', scene)
        vid_scene_dir = os.path.join(sdd_root, 'videos', scene)
        txt_paths = []
        video_paths = []

        if not os.path.isdir(ann_scene_dir) or not os.path.isdir(vid_scene_dir):
            # 如果某个子目录不存在，就跳过
            continue

        # annotations/scene 下，每个子文件夹都是一个 segment
        for segment in sorted(os.listdir(ann_scene_dir)):
            ann_seg_dir = os.path.join(ann_scene_dir, segment)
            vid_seg_dir = os.path.join(vid_scene_dir, segment)

            txt_file = os.path.join(ann_seg_dir, 'annotations.txt')
            vid_file = os.path.join(vid_seg_dir, 'video.mov')
            # 确保两个文件都存在才加入
            if os.path.isfile(txt_file) and os.path.isfile(vid_file):
                txt_paths.append(txt_file)
                video_paths.append(vid_file)
        
        scenario_txt_path[scene] = txt_paths
        scenario_video_path[scene] = video_paths
    return scenario_txt_path, scenario_video_path

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    SDD_PATH = os.path.join(project_dir, 'sdd')
    SDD_FILE = [
        # 'bookstore',
        # 'coupa',
        # 'deathCircle',
        # 'gates',
        # 'hyang',
        # 'little',
        # 'nexus',
        'quad',
        ]
    scenario_txt_path, scenario_video_path = load_sdd_paths(SDD_PATH, SDD_FILE)
    for scenario in SDD_FILE:
        current_sec_txt = scenario_txt_path.get(scenario, [])
        current_sec_mov = scenario_video_path.get(scenario, [])
        sec_snippet_id = 0
        for txt_path, mov_path in zip(current_sec_txt, current_sec_mov):
            # print(mov_path)
            data_loader = SDDLoader(txt_path)  # 加载单个场景

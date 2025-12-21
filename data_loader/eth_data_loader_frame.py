import os
import math
import json
import pickle
import numpy as np
# import pandas as pd
import torch
import networkx as nx
from .util import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

DELIMITER_t = '\t'
DELIMITER_space = ' '

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

class ETHLoader():
    def __init__(self, datasets_path, video_path):
        self.datasets_path = datasets_path
        self.video_path = video_path
        self.load_data()
        self.preprocess_data()
        self.sort_by_frame()
        self.generate_input_data()
        # self.generate_norm_input_data()
        # self.surrounding_infor()
        # self.calculate_kinematics_features()
        # self.generate_kbs_feature()
        # print(self.kinematics_features.shape)
        # arr_list = [
                # frame_, id_, 
                # x_, y_, x_rel, y_rel, heading_,
                # vx_, vy_, speed_, 
                # ax_, ay_, curvature_,
                # jerk_x_, jerk_y_, jerk_, 
                # project_x, project_y, text_heading,
                #     ]
    def load_data(self):
        print("Loading ETH&UCY data from", self.datasets_path)
        print("Loading ETH&UCY video from", self.video_path)
        data_file = open(self.datasets_path, "r")
        traj_data = []
        for line in data_file:
            line = line.strip().split(DELIMITER_t)
            line_data = [round(float(i), 2) for i in line]
            traj_data.append(line_data)
        self.processed_data = np.asarray(traj_data)
        self.video_data = cv2.VideoCapture(self.video_path)
        if not self.video_data.isOpened():
            print(f"无法打开视频文件: {self.video_path}")
            return None
    def preprocess_data(self):
        self.h_data, self.xy_tag = generate_dataset_tag(self.datasets_path)
        original_data_pos = self.processed_data[:,[2,3]]
        self.pixel_data = trajectory2pixel(original_data_pos, self.h_data)
        if self.xy_tag==0:
            self.processed_data = np.concatenate((self.processed_data[:,[0,1]], self.pixel_data[:,[1,0]]), axis=1).astype(int)
        else:
            self.processed_data = np.concatenate((self.processed_data[:,[0,1]], self.pixel_data), axis=1).astype(int)
        # print(self.processed_data.shape) # Nx6 #  self.processed_data  [frame, id, x_original, y_original, x_pixel, y_pixel] 
        
    def generate_each_frame_image(self, save_dir, frame_interval: int = 5, radius: int = 4, ):
        os.makedirs(save_dir, exist_ok=True)

        traj_data = self.processed_data
        # 假定列为 [frame, id, x_pixel, y_pixel, ...]
        FRAME_COL = 0
        ID_COL = 1
        X_COL = 2
        Y_COL = 3

        # 所有出现过的帧号（按升序）
        all_frames = np.unique(traj_data[:, FRAME_COL].astype(int))
        all_frames = np.sort(all_frames)

        # 视频总帧数（仅用于检查）
        total_frames = int(self.video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Info] Video total frames: {total_frames}")
        print(f"[Info] Unique frames in txt: {len(all_frames)}")

        # 间隔采样：每隔 frame_interval 个“标注帧”取一张
        sampled_frames = all_frames[::frame_interval]
        print(f"[Info] Sampled frames: {len(sampled_frames)} (interval={frame_interval})")

        for f in tqdm(sampled_frames, desc="Dumping annotated frames"):
            frame_idx = int(f)
            if frame_idx < 0 or frame_idx >= total_frames:
                # 有些数据集会有 offset，这里先简单跳过超界的
                print(f"[Warn] frame {frame_idx} 越界，total={total_frames}，跳过")
                continue

            # 定位到对应帧并读取
            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = self.video_data.read()
            if not success or frame is None:
                print(f"[Warn] 无法读取 frame {frame_idx}，跳过")
                continue

            H_img, W_img = frame.shape[:2]

            # 当前帧的所有行人
            rows = traj_data[traj_data[:, FRAME_COL] == f]

            for row in rows:
                pid = int(row[ID_COL])
                px = int(round(row[X_COL]))
                py = int(round(row[Y_COL]))

                if not (0 <= px < W_img and 0 <= py < H_img):
                    continue

                # 画一个圆点（绿色）标在行人位置
                cv2.circle(frame, (px, py), radius, (0, 255, 0), thickness=-1)
                # 在旁边写上 id（可选）
                cv2.putText(frame, str(pid), (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # 保存图像
            save_name = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(save_name, frame)

        print(f"[Done] 共导出 {len(sampled_frames)} 张标注帧到 {save_dir}")

    def generate_each_frame_with_his_image(self, save_dir, frame_interval: int = 5, radius: int = 4):
        os.makedirs(save_dir, exist_ok=True)

        traj_data = self.processed_data
        # 假定列为 [frame, id, x_pixel, y_pixel, ...]
        FRAME_COL = 0
        ID_COL = 1
        X_COL = 2
        Y_COL = 3

        # 所有出现过的帧号（按升序）
        all_frames = np.unique(traj_data[:, FRAME_COL].astype(int))
        all_frames = np.sort(all_frames)

        # 视频总帧数（仅用于检查）
        total_frames = int(self.video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Info] Video total frames: {total_frames}")
        print(f"[Info] Unique frames in txt: {len(all_frames)}")

        # 间隔采样: 每隔 frame_interval 个“标注帧”取一张
        sampled_frames = all_frames[::frame_interval]
        print(f"[Info] Sampled frames: {len(sampled_frames)} (interval={frame_interval})")

        # 颜色表 (BGR)
        color_palette = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 255),
            (255, 255, 0),
            (255, 0, 255),
            (128, 255, 0),
            (128, 0, 255),
            (0, 128, 255),
            (255, 128, 0),
        ]
        id2color = {}

        for f in tqdm(sampled_frames, desc="Dumping annotated frames"):
            frame_idx = int(f)
            if frame_idx < 0 or frame_idx >= total_frames:
                print(f"[Warn] frame {frame_idx} 越界, total={total_frames}, 跳过")
                continue

            # 定位到对应帧并读取
            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = self.video_data.read()
            if not success or frame is None:
                print(f"[Warn] 无法读取 frame {frame_idx}, 跳过")
                continue

            H_img, W_img = frame.shape[:2]

            # 当前帧的所有行人
            rows_current = traj_data[traj_data[:, FRAME_COL] == f]

            # 按行人 id 排序
            if rows_current.shape[0] > 0:
                order = np.argsort(rows_current[:, ID_COL].astype(int))
                rows_current = rows_current[order]

            # 当前帧 JSON 里要写的列表
            json_agents = []

            for row in rows_current:
                pid = int(row[ID_COL])

                # 为每个行人分配固定颜色
                if pid not in id2color:
                    color = color_palette[len(id2color) % len(color_palette)]
                    id2color[pid] = color
                color = id2color[pid]

                # 该行人到当前帧为止的历史轨迹 (frame <= f)
                history_rows = traj_data[
                    (traj_data[:, ID_COL] == pid) &
                    (traj_data[:, FRAME_COL] <= f)
                ]
                history_rows = history_rows[np.argsort(history_rows[:, FRAME_COL])]

                if history_rows.shape[0] >= 2:
                    hist_xy = history_rows[:, [X_COL, Y_COL]].astype(int)
                    hist_xy[:, 0] = np.clip(hist_xy[:, 0], 0, W_img - 1)
                    hist_xy[:, 1] = np.clip(hist_xy[:, 1], 0, H_img - 1)
                    pts = hist_xy.reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

                # 当前帧的位置
                px = int(round(row[X_COL]))
                py = int(round(row[Y_COL]))
                if not (0 <= px < W_img and 0 <= py < H_img):
                    continue

                # 当前点与 id
                cv2.circle(frame, (px, py), radius, color, thickness=-1)
                cv2.putText(
                    frame,
                    str(pid),
                    (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                json_agents.append({
                    "id": pid,
                    "x": px,
                    "y": py,
                    "frame": int(row[FRAME_COL]),
                })

            # 保存图像
            img_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(img_path, frame)

            # 保存 JSON（agents 已按 id 排序）
            json_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.json")
            frame_record = {
                "frame_id": frame_idx,
                "agents": json_agents,
            }
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(frame_record, jf, ensure_ascii=False, indent=2)

        print(f"[Done] 共导出 {len(sampled_frames)} 张标注帧到 {save_dir}")


    def generate_sliding_window_data(self, save_dir, window_size: int = 20, step: int = 10, radius: int = 4):
        """
        以滑动窗口的方式提取和保存轨迹数据。

        对于每个窗口 (t, t + window_size):
        1. 保存起始帧 t 的图像，并标注当时所有行人。
        2. 保存一个 JSON 文件，包含该窗口内所有行人的完整轨迹片段。
        """
        os.makedirs(save_dir, exist_ok=True)

        traj_data = self.processed_data
        FRAME_COL, ID_COL, X_COL, Y_COL = 0, 1, 2, 3

        all_frames = np.unique(traj_data[:, FRAME_COL].astype(int))
        min_frame, max_frame = all_frames.min(), all_frames.max()
        
        total_frames = int(self.video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Info] Video total frames: {total_frames}")
        print(f"[Info] Frames with annotations: {min_frame} to {max_frame}")

        window_starts = range(min_frame, max_frame - window_size + 1, step)

        for start_frame in tqdm(window_starts, desc="Processing sliding windows"):
            end_frame = start_frame + window_size

            # 1. 创建当前窗口的保存目录
            window_save_dir = os.path.join(save_dir, f"window_{start_frame:06d}_{end_frame:06d}")
            os.makedirs(window_save_dir, exist_ok=True)

            # 2. 保存起始帧图像
            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            success, frame_image = self.video_data.read()
            if not success or frame_image is None:
                print(f"[Warn] 无法读取 frame {start_frame}，跳过此窗口")
                continue
            
            H_img, W_img = frame_image.shape[:2]
            
            # 标注起始帧上的行人
            rows_at_start = traj_data[traj_data[:, FRAME_COL] == start_frame]
            for row in rows_at_start:
                px, py = int(round(row[X_COL])), int(round(row[Y_COL]))
                if 0 <= px < W_img and 0 <= py < H_img:
                    cv2.circle(frame_image, (px, py), radius, (0, 255, 0), -1)

            img_path = os.path.join(window_save_dir, f"frame_{start_frame:06d}.jpg")
            cv2.imwrite(img_path, frame_image)

            # 3. 提取窗口内的所有轨迹数据
            window_mask = (traj_data[:, FRAME_COL] >= start_frame) & (traj_data[:, FRAME_COL] < end_frame)
            window_data = traj_data[window_mask]

            pids_in_window = np.unique(window_data[:, ID_COL]).astype(int)
            pids_in_window = np.sort(pids_in_window)

            # 4. 构建并保存 JSON 文件
            json_agents = []
            for pid in pids_in_window:
                agent_traj_rows = window_data[window_data[:, ID_COL] == pid]
                # 按帧排序
                agent_traj_rows = agent_traj_rows[np.argsort(agent_traj_rows[:, FRAME_COL])]
                
                trajectory = []
                for row in agent_traj_rows:
                    trajectory.append({
                        "frame": int(row[FRAME_COL]),
                        "x": int(round(row[X_COL])),
                        "y": int(round(row[Y_COL])),
                    })
                
                json_agents.append({
                    "id": int(pid),
                    "trajectory": trajectory,
                })

            json_path = os.path.join(window_save_dir, "trajectories.json")
            window_record = {
                "window_start_frame": start_frame,
                "window_end_frame": end_frame,
                "agents": json_agents,
            }
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(window_record, jf, ensure_ascii=False, indent=2)

        print(f"[Done] 共处理 {len(window_starts)} 个滑动窗口，数据保存在 {save_dir}")

    def yield_sliding_windows(self, window_size: int = 20, step: int = 10):
        """
        以生成器的方式逐一产出滑动窗口数据，用于内存处理。
        """
        traj_data = self.processed_data
        FRAME_COL, ID_COL, X_COL, Y_COL = 0, 1, 2, 3

        all_frames = np.unique(traj_data[:, FRAME_COL].astype(int))
        if len(all_frames) == 0:
            return

        min_frame, max_frame = all_frames.min(), all_frames.max()
        window_starts = range(min_frame, max_frame - window_size + 1, step)
        scenario_name = os.path.basename(self.datasets_path).split('.')[0]

        for start_frame in window_starts:
            end_frame = start_frame + window_size

            # 1. 读取起始帧图像
            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            success, frame_image = self.video_data.read()
            if not success or frame_image is None:
                continue

            # 2. 提取窗口内的轨迹数据
            window_mask = (traj_data[:, FRAME_COL] >= start_frame) & (traj_data[:, FRAME_COL] < end_frame)
            window_data = traj_data[window_mask]

            pids_in_window = np.unique(window_data[:, ID_COL]).astype(int)
            pids_in_window = np.sort(pids_in_window)

            # 3. 构建JSON对象
            json_agents = []
            for pid in pids_in_window:
                agent_traj_rows = window_data[window_data[:, ID_COL] == pid]
                agent_traj_rows = agent_traj_rows[np.argsort(agent_traj_rows[:, FRAME_COL])]
                
                trajectory = [
                    {"frame": int(r[FRAME_COL]), "x": int(round(r[X_COL])), "y": int(round(r[Y_COL]))}
                    for r in agent_traj_rows
                ]
                
                json_agents.append({"id": int(pid), "trajectory": trajectory})

            window_record = {
                "window_start_frame": start_frame,
                "window_end_frame": end_frame,
                "agents": json_agents,
            }
            
            yield {
                "image": frame_image,
                "trajectory_data": window_record,
                "window_info": (start_frame, end_frame),
                "scenario": scenario_name,
            }

    def sort_by_frame(self):
        # get the target obs which frame over 20
        target_obs = []
        X = []
        y = []
        traj_data = self.processed_data
        ids = np.unique(traj_data[:, 1])
        # print(ids.shape)
        for id in tqdm(ids, desc="Processing frames"):
            tmp = traj_data[id == traj_data[:, 1], :]
            if tmp.shape[0] >= 20:
                distance = np.linalg.norm(tmp[0, [2,3]] - tmp[19, [2,3]])
                if distance >= 5:
                    target_obs.append(tmp[0:20, :])
        self.target_obs = np.asarray(target_obs)
            

    def calculate_kinematics_features(self):
        N, T, C = self.target_obs.shape
        frame_ = self.target_obs[..., 0]
        # frame_min = frame_.min().item()  
        # frame_norm = (frame_ - frame_min) / 20.0  
        id_    = self.target_obs[..., 1]
        # ========= 0) pos =========
        x_ = self.target_obs[..., 2]  # [N, 20]
        y_ = self.target_obs[..., 3]  # [N, 20]
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

        self.kinematics_features = np.stack(arr_list, axis=-1)
    
    def generate_kbs_feature(self):
        traj = self.target_obs[..., [2,3]]
        traj_proj, final_y, direction, angle = project_trajectory_and_compute_offset_batch(traj, threshold_deg=20)
        direction = direction.reshape(-1, 1, 3)        
        direction = np.repeat(direction, repeats=20, axis=1)
        self.kinematics_features = np.concatenate([self.kinematics_features, direction], axis=-1)

    def generate_input_data(self):
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

if __name__ == "__main__":
    ETH_FILE = [
    'ETH',
    'HOTEL',
    'ZARA01',
    'ZARA02',
    'STUDENT'
    ]

    for scenario in ETH_FILE:
        scenario_txt_path, scenario_video_path = collect_scenario_files(r"./ETH_datasets/", scenario)
        if len(scenario_txt_path) == 1:
            scenario_txt_path = scenario_txt_path[0]
            scenario_video_path = scenario_video_path[0]

        data_loader = ETHLoader(scenario_txt_path, scenario_video_path)  # 加载单个场景
        # data_loader.generate_each_frame_image(save_dir=f'./frame_data/{scenario}', frame_interval=5, radius=4)
        # data_loader.generate_each_frame_with_his_image(save_dir=f'./frame_data_with_history/{scenario}', frame_interval=5, radius=4)
        # data_loader.generate_sliding_window_data(save_dir=f'./processed_sliding_window_data/{scenario}', window_size=200, step=200)
        output_npz_path = os.path.join(PROJECT_ROOT, "processed_data", f"{scenario}.npz")
        data_loader.generate_sliding_window_npz(save_path=output_npz_path, window_size=20, step=10)

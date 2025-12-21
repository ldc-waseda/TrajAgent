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
from typing import Any, Dict, List, Optional, Tuple
DELIMITER_t = '\t'
DELIMITER_space = ' '
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)  
DATA_ROOT = os.path.join(PROJECT_ROOT, "ETH_datasets")

class ETHLoader():
    def __init__(self, datasets_path, video_path):
        self.datasets_path = datasets_path
        self.video_path = video_path
        self.load_data()
        self.preprocess_data()
        self.sort_by_frame()
        # self.generate_input_data()
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

        # 颜色表 (BGR)
        color_palette = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
            (255, 255, 0), (255, 0, 255), (128, 255, 0), (128, 0, 255),
            (0, 128, 255), (255, 128, 0),
        ]
        id2color = {}

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
            
            # 提取窗口内的所有轨迹数据
            window_mask = (traj_data[:, FRAME_COL] >= start_frame) & (traj_data[:, FRAME_COL] < end_frame)
            window_data = traj_data[window_mask]

            pids_in_window = np.unique(window_data[:, ID_COL]).astype(int)
            pids_in_window = np.sort(pids_in_window)

            # 在起始帧图像上绘制窗口内的所有轨迹
            for pid in pids_in_window:
                # 为每个行人分配固定颜色
                if pid not in id2color:
                    color = color_palette[len(id2color) % len(color_palette)]
                    id2color[pid] = color
                color = id2color[pid]

                # 获取该行人在窗口内的轨迹
                agent_traj_rows = window_data[window_data[:, ID_COL] == pid]
                agent_traj_rows = agent_traj_rows[np.argsort(agent_traj_rows[:, FRAME_COL])]

                if agent_traj_rows.shape[0] >= 2:
                    # 绘制轨迹线
                    traj_xy = agent_traj_rows[:, [X_COL, Y_COL]].astype(int)
                    traj_xy[:, 0] = np.clip(traj_xy[:, 0], 0, W_img - 1)
                    traj_xy[:, 1] = np.clip(traj_xy[:, 1], 0, H_img - 1)
                    pts = traj_xy.reshape(-1, 1, 2)
                    cv2.polylines(frame_image, [pts], isClosed=False, color=color, thickness=2)

                    # 在轨迹起点标注ID
                    start_pos = (traj_xy[0, 0], traj_xy[0, 1])
                    cv2.circle(frame_image, start_pos, radius, color, -1)
                    cv2.putText(
                        frame_image, str(pid), (start_pos[0] + 5, start_pos[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
                    )

            img_path = os.path.join(window_save_dir, f"frame_{start_frame:06d}.jpg")
            cv2.imwrite(img_path, frame_image)

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

    def generate_sliding_window_npz(self, save_path, window_size: int = 20, step: int = 10):
        """
        以滑动窗口的方式提取数据，并将所有窗口的数据保存到一个 npz 文件中。
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        traj_data = self.processed_data
        FRAME_COL, ID_COL, X_COL, Y_COL = 0, 1, 2, 3

        all_frames = np.unique(traj_data[:, FRAME_COL].astype(int))
        min_frame, max_frame = all_frames.min(), all_frames.max()

        window_starts = range(min_frame, max_frame - window_size + 1, step)

        images_list = []
        trajectories_list = []
        window_info_list = []

        for start_frame in tqdm(window_starts, desc="Processing sliding windows for NPZ"):
            end_frame = start_frame + window_size

            # 1. 读取起始帧图像
            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            success, frame_image = self.video_data.read()
            if not success or frame_image is None:
                print(f"[Warn] 无法读取 frame {start_frame}，跳过此窗口")
                continue
            
            images_list.append(frame_image)
            window_info_list.append([start_frame, end_frame])

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

            window_record = {
                "window_start_frame": start_frame,
                "window_end_frame": end_frame,
                "agents": json_agents,
            }
            trajectories_list.append(json.dumps(window_record))

        # 4. 保存到 NPZ 文件
        scenario_name = os.path.basename(self.datasets_path).split('.')[0]
        np.savez(
            save_path,
            images=np.array(images_list, dtype=object),
            trajectories=np.array(trajectories_list),
            window_info=np.array(window_info_list),
            scenario=scenario_name
        )

        print(f"[Done] 共处理 {len(images_list)} 个窗口，数据已保存到 {save_path}")

    def generate_data4api(
        self,
        window_size: int = 20,
        step: int = 10,
        radius: int = 4,
        draw_overlay: bool = True,
    ):
        """
        以生成器的方式逐一产出滑动窗口数据，用于内存处理。

        Yields:
            A dictionary for each window containing the image and trajectory data.
        """
        traj_data = self.processed_data
        FRAME_COL, ID_COL, X_COL, Y_COL = 0, 1, 2, 3

        all_frames = np.unique(traj_data[:, FRAME_COL].astype(int))
        if all_frames.size == 0:
            return

        min_frame, max_frame = int(all_frames.min()), int(all_frames.max())
        scenario_name = os.path.basename(self.datasets_path).split('.')[0]
        
        window_starts = range(min_frame, max_frame - window_size + 1, step)

        color_palette = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
            (255, 255, 0), (255, 0, 255), (128, 255, 0), (128, 0, 255),
            (0, 128, 255), (255, 128, 0),
        ]
        id2color: Dict[int, Tuple[int, int, int]] = {}

        for start_frame in tqdm(window_starts, desc=f"Streaming windows for {scenario_name}"):
            end_frame = start_frame + window_size

            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            success, frame_image = self.video_data.read()
            if not success or frame_image is None:
                continue

            H_img, W_img = frame_image.shape[:2]
            window_mask = (traj_data[:, FRAME_COL] >= start_frame) & (traj_data[:, FRAME_COL] < end_frame)
            window_data = traj_data[window_mask]
            pids_in_window = np.unique(window_data[:, ID_COL]).astype(int)

            if draw_overlay:
                for pid in pids_in_window:
                    if pid not in id2color:
                        id2color[pid] = color_palette[len(id2color) % len(color_palette)]
                    color = id2color[pid]
                    
                    agent_traj_rows = window_data[window_data[:, ID_COL] == pid]
                    if agent_traj_rows.shape[0] >= 2:
                        pts = agent_traj_rows[:, [X_COL, Y_COL]].astype(int).reshape(-1, 1, 2)
                        cv2.polylines(frame_image, [pts], isClosed=False, color=color, thickness=2)
                        start_pos = (pts[0][0][0], pts[0][0][1])
                        cv2.circle(frame_image, start_pos, radius, color, -1)
                        cv2.putText(frame_image, str(pid), (start_pos[0] + 5, start_pos[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            json_agents = []
            for pid in sorted(pids_in_window):
                agent_traj_rows = window_data[window_data[:, ID_COL] == pid]
                trajectory = [{"frame": int(r[FRAME_COL]), "x": int(round(r[X_COL])), "y": int(round(r[Y_COL]))}
                              for r in agent_traj_rows]
                json_agents.append({"id": int(pid), "trajectory": trajectory})

            window_record = {
                "window_start_frame": int(start_frame),
                "window_end_frame": int(end_frame),
                "agents": json_agents,
            }

            yield {
                "scenario": scenario_name,
                "window_info": (int(start_frame), int(end_frame)),
                "image": frame_image,
                "trajectory_data": window_record,
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
            
    def load_sliding_window_data(
                                        self,
                                        window_size: int = 20,
                                        step: int = 10,
                                        radius: int = 4,
                                        draw_overlay: bool = True,
                                        ) -> List[Dict[str, Any]]:
        """
        以滑动窗口的方式提取轨迹数据，并直接返回所有窗口的数据（不落盘）。

        Returns:
            windows: List[dict], 每个元素包含：
                {
                "window_info": (start_frame, end_frame),
                "image": frame_image (np.ndarray, BGR),
                "trajectory_data": window_record (dict),
                }
        """
        traj_data = self.processed_data
        FRAME_COL, ID_COL, X_COL, Y_COL = 0, 1, 2, 3

        all_frames = np.unique(traj_data[:, FRAME_COL].astype(int))
        if all_frames.size == 0:
            return []

        min_frame, max_frame = int(all_frames.min()), int(all_frames.max())

        total_frames = int(self.video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Info] Video total frames: {total_frames}")
        print(f"[Info] Frames with annotations: {min_frame} to {max_frame}")

        window_starts = list(range(min_frame, max_frame - window_size + 1, step))

        # 颜色表 (BGR)
        color_palette = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
            (255, 255, 0), (255, 0, 255), (128, 255, 0), (128, 0, 255),
            (0, 128, 255), (255, 128, 0),
        ]
        id2color: Dict[int, Tuple[int, int, int]] = {}

        windows: List[Dict[str, Any]] = []

        for start_frame in tqdm(window_starts, desc="Processing sliding windows"):
            end_frame = start_frame + window_size

            # 读取起始帧图像
            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            success, frame_image = self.video_data.read()
            if (not success) or frame_image is None:
                print(f"[Warning] Failed to read frame {start_frame}, skipping window.")
                continue

            H_img, W_img = frame_image.shape[:2]

            # 提取窗口内的所有轨迹数据
            window_mask = (traj_data[:, FRAME_COL] >= start_frame) & (traj_data[:, FRAME_COL] < end_frame)
            window_data = traj_data[window_mask]
            pids_in_window = np.unique(window_data[:, ID_COL]).astype(int)

            # (Optional) 在图像上绘制轨迹
            if draw_overlay:
                for pid in pids_in_window:
                    if pid not in id2color:
                        id2color[pid] = color_palette[len(id2color) % len(color_palette)]
                    
                    agent_traj = window_data[window_data[:, ID_COL] == pid]
                    agent_traj = agent_traj[agent_traj[:, FRAME_COL].argsort()] # Sort by frame
                    
                    points = agent_traj[:, [X_COL, Y_COL]].astype(np.int32)
                    for i in range(len(points) - 1):
                        cv2.line(frame_image, tuple(points[i]), tuple(points[i+1]), id2color[pid], 2)
                    
                    # Mark current position
                    if agent_traj.shape[0] > 0:
                        curr_pos = agent_traj[0, [X_COL, Y_COL]].astype(np.int32)
                        cv2.circle(frame_image, tuple(curr_pos), radius + 2, id2color[pid], -1)

            # 构建轨迹数据字典
            json_agents = {}
            for pid in sorted(pids_in_window):
                agent_traj = window_data[window_data[:, ID_COL] == pid]
                # Ensure trajectory is sorted by frame
                agent_traj = agent_traj[agent_traj[:, FRAME_COL].argsort()]
                json_agents[str(pid)] = agent_traj[:, [X_COL, Y_COL]].tolist()

            window_record = {
                "window_start_frame": int(start_frame),
                "window_end_frame": int(end_frame),
                "agents": json_agents,
            }

            windows.append({
                "window_info": (int(start_frame), int(end_frame)),
                "image": frame_image,
                "trajectory_data": window_record,
            })

        print(f"[Done] 共处理 {len(windows)} 个滑动窗口（原计划 {len(window_starts)}），已返回到内存")
        return windows
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
        scenario_txt_path, scenario_video_path = collect_scenario_files(DATA_ROOT, scenario)
        if len(scenario_txt_path) == 1:
            scenario_txt_path = scenario_txt_path[0]
            scenario_video_path = scenario_video_path[0]

        data_loader = ETHLoader(scenario_txt_path, scenario_video_path)  # 加载单个场景
        # data_loader.generate_each_frame_image(save_dir=f'./frame_data/{scenario}', frame_interval=5, radius=4)
        # data_loader.generate_each_frame_with_his_image(save_dir=f'./frame_data_with_history/{scenario}', frame_interval=5, radius=4)
        # data_loader.generate_sliding_window_data(save_dir=f'./processed_sliding_window_data/{scenario}', window_size=200, step=200)
        # output_npz_path = os.path.join(PROJECT_ROOT, "processed_data", f"{scenario}.npz")
        # data_loader.generate_sliding_window_npz(save_path=output_npz_path, window_size=200, step=200)

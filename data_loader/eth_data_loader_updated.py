import os
import math
import json
import pickle
import numpy as np
# import pandas as pd
import torch
import shutil
import networkx as nx
from .util import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from pathlib import Path

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

    def generate_window_records(
        self,
        save_dir: str,
        scenario_name: str = "eth",
        window_size: int = 200,
        stride: int | None = None,
        radius: int = 4,
        mask_src_dir: str | None = None,
        mask_keyword: str = "mask",  # 保留参数但不再使用（兼容你原来的调用）
            ) -> None:

   

        os.makedirs(save_dir, exist_ok=True)

        if stride is None:
            stride = window_size

        traj_data = self.processed_data
        FRAME_COL, ID_COL, X_COL, Y_COL = 0, 1, 2, 3

        if traj_data is None or len(traj_data) == 0:
            raise ValueError("self.processed_data is empty")

        # =========================
        # 固定拷贝的 mask 源文件（只做一次检查）
        # =========================
        mask_source_path = Path(mask_src_dir).resolve() / scenario_name / f"{scenario_name}.jpg"

        if not mask_source_path.exists():
            raise FileNotFoundError(f"mask source image not found: {mask_source_path}")

        total_frames = int(self.video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Info] Video total frames: {total_frames}")

        all_frames = traj_data[:, FRAME_COL].astype(int)
        min_f = int(all_frames.min())
        max_f = int(all_frames.max())
        print(f"[Info] Labeled frame range: [{min_f}, {max_f}]")

        # 颜色表 (BGR) + 行人固定颜色映射（跨窗口保持一致）
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
        id2color: dict[int, tuple[int, int, int]] = {}

        window_starts = list(range(min_f, max_f + 1, stride))
        for start_f in tqdm(window_starts, desc="Dumping windows"):
            end_f = start_f + window_size - 1

            in_win = (traj_data[:, FRAME_COL] >= start_f) & (traj_data[:, FRAME_COL] <= end_f)
            win_rows = traj_data[in_win]
            if win_rows.shape[0] == 0:
                continue

            # 窗口当前帧：窗口内最后一个标注帧
            now_f = int(win_rows[:, FRAME_COL].max())
            if now_f < 0 or now_f >= total_frames:
                print(f"[Warn] now_f {now_f} out of video range [0, {total_frames-1}], skip window")
                continue

            # 读取 now_f 对应的视频帧
            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, now_f)
            success, frame = self.video_data.read()
            if not success or frame is None:
                print(f"[Warn] failed to read video frame {now_f}, skip window")
                continue

            H_img, W_img = frame.shape[:2]

            # 窗口内按 (frame, id) 排序，便于稳定写 JSON
            order = np.lexsort((win_rows[:, ID_COL].astype(int), win_rows[:, FRAME_COL].astype(int)))
            win_rows_sorted = win_rows[order]

            # 1) JSON：聚合窗口内所有行人轨迹（按 id 排序）
            agent_map: dict[int, list[dict]] = {}
            for row in win_rows_sorted:
                pid = int(row[ID_COL])
                f = int(row[FRAME_COL])
                x = int(round(row[X_COL]))
                y = int(round(row[Y_COL]))
                agent_map.setdefault(pid, []).append({"frame": f, "x": x, "y": y})

            agents_list = [{"id": pid, "trajectory": agent_map[pid]} for pid in sorted(agent_map.keys())]

            window_record = {
                "window_start": int(start_f),
                "window_end": int(end_f),
                "now_frame": int(now_f),
                "agents": agents_list,
            }

            # 2) 画图：只标注 now_f 这一帧出现的行人位置 + id
            now_rows = win_rows_sorted[win_rows_sorted[:, FRAME_COL].astype(int) == now_f]
            if now_rows.shape[0] > 0:
                now_rows = now_rows[np.argsort(now_rows[:, ID_COL].astype(int))]

            for row in now_rows:
                pid = int(row[ID_COL])
                x = int(round(row[X_COL]))
                y = int(round(row[Y_COL]))
                if not (0 <= x < W_img and 0 <= y < H_img):
                    continue

                if pid not in id2color:
                    id2color[pid] = color_palette[len(id2color) % len(color_palette)]
                color = id2color[pid]

                cv2.circle(frame, (x, y), radius, color, thickness=-1)
                cv2.putText(
                    frame,
                    str(pid),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # 3) 保存：每个窗口单独子文件夹
            case_name = f"case_{start_f:06d}"
            case_dir = Path(save_dir) / case_name
            case_dir.mkdir(parents=True, exist_ok=True)

            # 3.1 保存窗口 JSON（文件名与 case_name 保持一致）
            json_path = case_dir / f"{case_name}.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(window_record, jf, ensure_ascii=False, indent=2)

            # 3.2 保存带行人标记的原图：固定命名为 case_scenario.jpg
            scenario_img_path = case_dir / "case_scenario.jpg"
            cv2.imwrite(str(scenario_img_path), frame)

            # 3.3 拷贝固定 mask：每个窗口都复制同一张 A/scenario.jpg 到 case_mask.jpg
            #     若 mask_src_dir=None，则跳过（不报错）
            if mask_source_path is not None:
                dst_mask_path = case_dir / "case_mask.jpg"
                shutil.copy2(str(mask_source_path), str(dst_mask_path))

        print(f"[Done] Window records dumped to {save_dir}")



   

   


   

if __name__ == "__main__":
    ETH_FILE = [
    'ETH',
    'HOTEL',
    'ZARA01',
    'ZARA02',
    'STUDENT'
    ]

    out_root = Path("processed_data")  # 或者你想放的绝对路径
    out_root.mkdir(parents=True, exist_ok=True)

    for scenario in ETH_FILE:
        scenario_txt_path, scenario_video_path = collect_scenario_files(DATA_ROOT, scenario)
        if len(scenario_txt_path) == 1:
            scenario_txt_path = scenario_txt_path[0]
            scenario_video_path = scenario_video_path[0]

        data_loader = ETHLoader(scenario_txt_path, scenario_video_path)

        scenario_out_dir = out_root / str(scenario)   # scenario 名字做文件夹
        scenario_out_dir.mkdir(parents=True, exist_ok=True)

        data_loader.generate_window_records(
            save_dir=str(scenario_out_dir),
            scenario_name=scenario,
            mask_src_dir='ETH_datasets/',
            window_size=200,
            stride=200,
            radius=4,
        )


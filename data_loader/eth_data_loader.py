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
            

    def generate_window_records(
                                    self,
                                    save_dir: str,
                                    window_size: int = 200,
                                    stride: int | None = None,
                                    radius: int = 4,
                                ) -> None:

        os.makedirs(save_dir, exist_ok=True)

        if stride is None:
            stride = window_size

        traj_data = self.processed_data
        FRAME_COL, ID_COL, X_COL, Y_COL = 0, 1, 2, 3

        if traj_data is None or len(traj_data) == 0:
            raise ValueError("self.processed_data is empty")

        total_frames = int(self.video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Info] Video total frames: {total_frames}")

        all_frames = traj_data[:, FRAME_COL].astype(int)
        min_f = int(all_frames.min())
        max_f = int(all_frames.max())
        print(f"[Info] Labeled frame range: [{min_f}, {max_f}]")

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

            in_window = (traj_data[:, FRAME_COL] >= start_f) & (traj_data[:, FRAME_COL] <= end_f)
            win_rows = traj_data[in_window]
            if win_rows.shape[0] == 0:
                continue

            # 窗口当前帧：窗口内最后一个标注帧
            now_f = int(win_rows[:, FRAME_COL].max())
            if now_f < 0 or now_f >= total_frames:
                print(f"[Warn] now_f {now_f} out of video range [0, {total_frames-1}], skip window")
                continue

            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, 0)
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

            # 2) 图片：只标注 now_f 这一帧出现的行人位置 + id
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

            # 2.5) GT 图：在 now_f 的原始帧上，画出窗口内所有行人的完整轨迹（polyline）
            gt_frame = frame.copy()  # 用已标注 now_f 的 frame 当底图也可以；不想带点/ID就改成原始帧备份
            for pid in sorted(agent_map.keys()):
                traj = agent_map[pid]
                if len(traj) < 2:
                    continue

                if pid not in id2color:
                    id2color[pid] = color_palette[len(id2color) % len(color_palette)]
                color = id2color[pid]

                pts = np.array([[t["x"], t["y"]] for t in traj], dtype=np.int32)
                # 防止越界
                pts[:, 0] = np.clip(pts[:, 0], 0, W_img - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, H_img - 1)
                pts = pts.reshape(-1, 1, 2)

                cv2.polylines(gt_frame, [pts], isClosed=False, color=color, thickness=2)

            # 3) 保存：每个窗口单独子文件夹
            case_name = f"case_{start_f:06d}"
            case_dir = os.path.join(save_dir, case_name)
            os.makedirs(case_dir, exist_ok=True)

            img_path = os.path.join(case_dir, "case_scenario.jpg")
            json_path = os.path.join(case_dir, f"{case_name}.json")
            gt_path = os.path.join(case_dir, "case_gt.jpg")

            cv2.imwrite(img_path, frame)
            cv2.imwrite(gt_path, gt_frame)
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(window_record, jf, ensure_ascii=False, indent=2)

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
            window_size=200,
            stride=200,
            radius=4,
        )


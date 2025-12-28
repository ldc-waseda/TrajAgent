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

def color_for_id_rgb3(pid: int) -> tuple[int, int, int]:
    """
    Stable 3-color cycle (BGR):
      pid%3==0 -> Green
      pid%3==1 -> Red
      pid%3==2 -> Blue
    """
    colors = [
        (0, 255, 0),   # Green (BGR)
        (0, 0, 255),   # Red
        (255, 0, 0),   # Blue
    ]
    return colors[int(pid) % 3]


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
            scenario_name: str = "eth",
            window_size: int = 200,
            stride: int | None = None,
            radius: int = 4,
            mask_src_dir: str | None = None,
            mask_keyword: str = "mask",  # 保留参数但不再使用（兼容你原来的调用）
            ) -> None:
        """
        Window export with filtering + start/end markers.

        Changes:
        - Filter: only keep agents whose trajectory length within this window > 5 frames (i.e., >= 6 points).
        - Visualization: for each kept agent, draw:
            - polyline across window
            - start point marker (filled circle)
            - end point marker (filled square)
            - optional id text near end point
        - The saved JSON also only contains these kept agents.
        - NEW: also save the raw frame-0 image as case_base_map.jpg in each case folder.
    """

        os.makedirs(save_dir, exist_ok=True)

        if stride is None:
            stride = window_size

        traj_data = self.processed_data
        FRAME_COL, ID_COL, X_COL, Y_COL = 0, 1, 2, 3

        if traj_data is None or len(traj_data) == 0:
            raise ValueError("self.processed_data is empty")

        # =========================
        # Fixed mask source (optional)
        # =========================
        mask_source_path = None
        if mask_src_dir:
            mask_source_path = (Path(mask_src_dir).resolve() / scenario_name / f"{scenario_name}.jpg")
            if not mask_source_path.exists():
                raise FileNotFoundError(f"mask source image not found: {mask_source_path}")

        total_frames = int(self.video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Info] Video total frames: {total_frames}")

        all_frames = traj_data[:, FRAME_COL].astype(int)
        min_f = int(all_frames.min())
        max_f = int(all_frames.max())
        print(f"[Info] Labeled frame range: [{min_f}, {max_f}]")

        window_starts = list(range(min_f, max_f + 1, stride))
        for start_f in tqdm(window_starts, desc="Dumping windows"):
            end_f = start_f + window_size - 1

            in_win = (traj_data[:, FRAME_COL] >= start_f) & (traj_data[:, FRAME_COL] <= end_f)
            win_rows = traj_data[in_win]
            if win_rows.shape[0] == 0:
                continue

            # Window "current" frame = last labeled frame in window
            now_f = int(win_rows[:, FRAME_COL].max())
            if now_f < 0 or now_f >= total_frames:
                print(f"[Warn] now_f {now_f} out of video range [0, {total_frames-1}], skip window")
                continue

            # ====== NEW: read raw frame 0 (base map) ======
            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok0, base_map = self.video_data.read() # (480 640 related2 height x weidth)
            print(base_map.shape)
            if not ok0 or base_map is None:
                print(f"[Warn] failed to read video frame 0, skip window")
                continue

            # ====== read now_f frame for drawing ======
            self.video_data.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.video_data.read()
            if not success or frame is None:
                print(f"[Warn] failed to read video frame {now_f}, skip window")
                continue

            H_img, W_img = frame.shape[:2]

            # Sort by (frame, id) for stable aggregation
            order = np.lexsort((win_rows[:, ID_COL].astype(int), win_rows[:, FRAME_COL].astype(int)))
            win_rows_sorted = win_rows[order]

            # 1) Aggregate trajectories within window
            agent_map: dict[int, list[dict]] = {}
            for row in win_rows_sorted:
                pid = int(row[ID_COL])
                f = int(row[FRAME_COL])
                x = int(round(row[X_COL]))
                y = int(round(row[Y_COL]))
                agent_map.setdefault(pid, []).append({"frame": f, "x": x, "y": y})

            # =========================
            # Filter: keep only agents with >5 frames in this window
            # =========================
            kept_agent_ids = [pid for pid, traj in agent_map.items() if len(traj) > 5]
            if len(kept_agent_ids) == 0:
                continue

            # JSON only for kept agents (sorted by id)
            agents_list = [{"id": pid, "trajectory": agent_map[pid]} for pid in sorted(kept_agent_ids)]
            window_record = {
                "window_start": int(start_f),
                "window_end": int(end_f),
                "now_frame": int(now_f),
                "agents": agents_list,
            }

            # =========================
            # 2) Visualization: draw full window trajectories for kept agents
            #    + mark start/end points
            # =========================
            for pid in sorted(kept_agent_ids):
                traj = agent_map[pid]
                if len(traj) == 0:
                    continue

                color = color_for_id_rgb3(pid)

                pts = np.array([[t["x"], t["y"]] for t in traj], dtype=np.int32)
                pts[:, 0] = np.clip(pts[:, 0], 0, W_img - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, H_img - 1)

                # Polyline
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=2)

                # Start/End markers
                sx, sy = int(pts[0, 0]), int(pts[0, 1])
                ex, ey = int(pts[-1, 0]), int(pts[-1, 1])

                cv2.circle(frame, (sx, sy), radius, color, thickness=-1)  # start
                r = radius + 1
                cv2.rectangle(frame, (ex - r, ey - r), (ex + r, ey + r), color, thickness=-1)  # end (square)

                # Put id near the end point
                cv2.putText(
                    frame,
                    str(pid),
                    (ex + 5, ey - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # =========================
            # 3) Save per-window folder
            # =========================
            case_name = f"case_{start_f:06d}"
            case_dir = Path(save_dir) / case_name
            case_dir.mkdir(parents=True, exist_ok=True)

            # 3.1 JSON
            json_path = case_dir / f"{case_name}.json"
            json_path.write_text(json.dumps(window_record, ensure_ascii=False, indent=2), encoding="utf-8")

            # 3.2 annotated scenario image
            scenario_img_path = case_dir / "case_scenario.jpg"
            cv2.imwrite(str(scenario_img_path), frame)

            # 3.3 NEW: save base map (raw frame0)
            base_map_path = case_dir / "case_base_map.jpg"
            cv2.imwrite(str(base_map_path), base_map)

            # 3.4 copy fixed mask (optional)
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


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import cv2
import numpy as np
TrajDict = Dict[int, List[List[Tuple[int, int]]]]  # agent_id -> [traj -> [(x,y), ...]]
Point = Tuple[int, int]
Traj = List[Point]
GptTrajDict = Dict[int, List[Traj]]          # agent_id -> [candidate_traj1, candidate_traj2, ...]
OrigTrajDict = Dict[int, Traj]               # agent_id -> original_traj

def load_gpt_output_candidates(
    gpt_json_path: Union[str, Path],
) -> Dict[int, List[List[Tuple[int, int]]]]:
    """
    输入文件（gpt_json_path）必须是 GPT 输出的 JSON，
    其中：
    - agents: list[object]，每个 object 表示一个行人
    - agent_id: int，行人 ID
    - candidates: list[object]，候选轨迹集合
    - points: list[[x,y]]，每条轨迹的点序列（像素坐标），每个点是长度为2的数组

    返回：
    - dict[agent_id] -> list[trajectory]
    - trajectory: list[(x,y)]，其中 (x,y) 为 int

    Args:
        gpt_json_path: GPT 输出 JSON 文件路径（str 或 Path）。

    Returns:
        Dict[int, List[List[Tuple[int, int]]]]:
            {
              1: [[(x,y), (x,y), ...], [...], ...],
              2: [[(x,y), ...], ...],
              ...
            }
    """
    path = Path(gpt_json_path)
    if not path.exists():
        raise FileNotFoundError(f"GPT output json not found: {path}")

    try:
        data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, dict) or "agents" not in data:
        raise ValueError(
            "Invalid GPT output format: root must be an object with key 'agents'. "
            f"Got type={type(data)} keys={list(data.keys()) if isinstance(data, dict) else None}"
        )

    agents = data["agents"]
    if not isinstance(agents, list):
        raise ValueError("Invalid GPT output format: 'agents' must be a list.")

    out: Dict[int, List[List[Tuple[int, int]]]] = {}

    for ai, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ValueError(f"Invalid agent entry at agents[{ai}]: must be an object.")
        if "agent_id" not in agent:
            raise ValueError(f"Invalid agent entry at agents[{ai}]: missing 'agent_id'.")
        if "candidates" not in agent:
            raise ValueError(f"Invalid agent entry at agents[{ai}]: missing 'candidates'.")

        agent_id = agent["agent_id"]
        if not isinstance(agent_id, int):
            raise ValueError(f"Invalid agent_id at agents[{ai}]: expected int, got {type(agent_id)}")

        candidates = agent["candidates"]
        if not isinstance(candidates, list):
            raise ValueError(f"Invalid candidates for agent_id={agent_id}: expected list, got {type(candidates)}")

        trajs: List[List[Tuple[int, int]]] = []
        for ci, cand in enumerate(candidates):
            if not isinstance(cand, dict) or "points" not in cand:
                raise ValueError(f"Invalid candidate at agents[{ai}].candidates[{ci}]: expected object with 'points'.")

            points = cand["points"]
            if not isinstance(points, list):
                raise ValueError(f"Invalid points at agent_id={agent_id}, candidate[{ci}]: expected list.")

            traj: List[Tuple[int, int]] = []
            for pi, pt in enumerate(points):
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    raise ValueError(
                        f"Invalid point at agent_id={agent_id}, candidate[{ci}], points[{pi}]: "
                        f"expected [x,y], got {pt}"
                    )
                x, y = pt
                try:
                    traj.append((int(x), int(y)))
                except Exception as e:
                    raise ValueError(
                        f"Invalid coordinate at agent_id={agent_id}, candidate[{ci}], points[{pi}]: {pt}"
                    ) from e

            trajs.append(traj)

        out[agent_id] = trajs

    return out


def filter_trajs_by_walkable_mask(
    trajs_by_agent: TrajDict,
    mask_path: Union[str, Path],
    *,
    walkable_is_white: bool = True,
    threshold: int = 128,
) -> Tuple[TrajDict, Dict[str, Any]]:
    """
    输入:
      - trajs_by_agent: load_gpt_output_candidates(...) 的返回字典
          {
            agent_id: [
              [(x,y), (x,y), ...],   # candidate 0
              [(x,y), (x,y), ...],   # candidate 1
              ...
            ],
            ...
          }

      - mask_path: 二值 mask 图片路径（黑白图，0/255 或接近）
        默认约定:
          walkable_is_white=True  -> 白色(>=threshold) 表示“可行走区域”
          walkable_is_white=False -> 黑色(<threshold)  表示“可行走区域”

    校验逻辑:
      - 对每条 trajectory 的每个点 (x,y)：
          1) 必须在图像范围内 (0<=x<W, 0<=y<H)
          2) 必须落在“可行走区域”上，否则判为 invalid
      - 只要出现一个点违规，这条轨迹就会被过滤掉

    返回:
      - filtered_trajs_by_agent: 同结构字典，但只包含通过 mask 校验的轨迹
      - report: 汇报统计信息（总数/保留数/剔除原因/每个 agent 的保留比例等）
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"mask image not found: {mask_path}")

    # 读灰度 mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"failed to read mask image: {mask_path}")

    H, W = mask.shape[:2]

    # 生成 bool 可行走区域
    # walkable_is_white=True:  mask>=threshold -> True
    # walkable_is_white=False: mask<threshold  -> True
    if walkable_is_white:
        walkable = mask >= threshold
    else:
        walkable = mask < threshold

    filtered: TrajDict = {}
    per_agent_stats: Dict[int, Dict[str, Any]] = {}

    total_traj = 0
    kept_traj = 0
    removed_oob = 0          # out of bounds
    removed_nonwalk = 0      # hits non-walkable

    # 记录部分样例违规点（方便你 debug）
    examples: List[Dict[str, Any]] = []

    for agent_id, traj_list in trajs_by_agent.items():
        agent_total = len(traj_list)
        agent_kept = 0
        agent_removed = 0

        kept_list: List[List[Tuple[int, int]]] = []

        for cand_idx, traj in enumerate(traj_list):
            total_traj += 1

            invalid_reason = None
            invalid_at = None  # (point_index, x, y)

            for pi, (x, y) in enumerate(traj):
                # 1) 边界检查
                if x < 0 or x >= W or y < 0 or y >= H:
                    invalid_reason = "out_of_bounds"
                    invalid_at = (pi, x, y)
                    break

                # 2) mask 可行走检查
                if not bool(walkable[y, x]):
                    invalid_reason = "non_walkable"
                    invalid_at = (pi, x, y)
                    break

            if invalid_reason is None:
                kept_list.append(traj)
                kept_traj += 1
                agent_kept += 1
            else:
                agent_removed += 1
                if invalid_reason == "out_of_bounds":
                    removed_oob += 1
                else:
                    removed_nonwalk += 1

                # 只收集少量例子，避免 report 太大
                if len(examples) < 20 and invalid_at is not None:
                    pi, x, y = invalid_at
                    examples.append(
                        {
                            "agent_id": agent_id,
                            "candidate_index": cand_idx,
                            "reason": invalid_reason,
                            "first_invalid_point_index": pi,
                            "x": x,
                            "y": y,
                        }
                    )

        filtered[agent_id] = kept_list
        per_agent_stats[agent_id] = {
            "total_candidates": agent_total,
            "kept_candidates": agent_kept,
            "removed_candidates": agent_removed,
            "keep_ratio": (agent_kept / agent_total) if agent_total > 0 else 0.0,
        }

    report: Dict[str, Any] = {
        "mask_path": str(mask_path),
        "mask_size": {"H": H, "W": W},
        "walkable_is_white": walkable_is_white,
        "threshold": threshold,
        "total_candidates": total_traj,
        "kept_candidates": kept_traj,
        "removed_candidates": total_traj - kept_traj,
        "removed_out_of_bounds": removed_oob,
        "removed_non_walkable": removed_nonwalk,
        "overall_keep_ratio": (kept_traj / total_traj) if total_traj > 0 else 0.0,
        "per_agent": per_agent_stats,
        "examples_first_invalid": examples,  # 用于快速定位问题
    }

    return filtered, report


def visualize_trajs_on_scenario(
    scenario_img: Union[str, Path, np.ndarray],
    original_trajs: Optional[OrigTrajDict],
    gpt_trajs_by_agent: GptTrajDict,
    out_path: Optional[Union[str, Path]] = None,
    *,
    draw_points: bool = False,
    draw_agent_id: bool = True,
    origin_thickness: int = 3,
    cand_thickness: int = 2,
    point_radius: int = 2,
) -> np.ndarray:
    """
    可视化函数

    输入:
      - scenario_img:
          1) 图片路径(str/Path)，例如 case_dir/"case_scenario.jpg"
          2) 或者 이미加载好的 BGR 图像(np.ndarray, shape=(H,W,3))

      - original_trajs (可选):
          dict[int, list[(x,y)]]
          例如:
            {
              1: [(x,y), (x,y), ...],
              2: [(x,y), ...],
            }
          用于画原始轨迹, 你不想画可以传 None

      - gpt_trajs_by_agent:
          dict[int, list[list[(x,y)]]]
          例如 load_gpt_output_candidates(...) 的输出:
            {
              1: [[(x,y)...], [(x,y)...], ...],
              2: [[(x,y)...], ...],
            }

      - out_path:
          若提供, 则保存可视化结果到该路径

    输出:
      - 返回绘制后的 BGR 图像 np.ndarray
    """
    # 1) load image
    if isinstance(scenario_img, (str, Path)):
        img = cv2.imread(str(scenario_img), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"failed to read scenario image: {scenario_img}")
    else:
        if not isinstance(scenario_img, np.ndarray) or scenario_img.ndim != 3:
            raise ValueError("scenario_img must be a path or a BGR image ndarray")
        img = scenario_img.copy()

    H, W = img.shape[:2]

    def _clip_xy(x: int, y: int) -> Point:
        return (int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1)))

    def _color_for_id(agent_id: int) -> Tuple[int, int, int]:
        # 稳定颜色, BGR
        palette = [
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
            (0, 200, 100),
            (200, 100, 0),
            (100, 200, 0),
            (0, 100, 200),
            (200, 0, 100),
            (100, 0, 200),
        ]
        return palette[agent_id % len(palette)]

    def _dim_color(c: Tuple[int, int, int], factor: float = 0.6) -> Tuple[int, int, int]:
        b, g, r = c
        return (int(b * factor), int(g * factor), int(r * factor))

    def _draw_polyline(points: Traj, color: Tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2:
            return
        pts = np.array([_clip_xy(x, y) for (x, y) in points], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)

    def _draw_points(points: Traj, color: Tuple[int, int, int]) -> None:
        for (x, y) in points:
            x2, y2 = _clip_xy(x, y)
            cv2.circle(img, (x2, y2), point_radius, color, thickness=-1)

    def _put_agent_id_at(points: Traj, agent_id: int, color: Tuple[int, int, int]) -> None:
        if not points:
            return
        x, y = _clip_xy(points[0][0], points[0][1])
        cv2.putText(
            img,
            str(agent_id),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # 2) draw GPT candidates first (thin, dim color)
    for agent_id, cand_list in gpt_trajs_by_agent.items():
        base = _color_for_id(agent_id)
        cand_color = _dim_color(base, 0.55)

        for traj in cand_list:
            _draw_polyline(traj, cand_color, cand_thickness)
            if draw_points:
                _draw_points(traj, cand_color)

        if draw_agent_id and cand_list:
            _put_agent_id_at(cand_list[0], agent_id, cand_color)

    # 3) draw original on top (thicker, bright color)
    if original_trajs is not None:
        for agent_id, traj in original_trajs.items():
            base = _color_for_id(agent_id)
            _draw_polyline(traj, base, origin_thickness)
            if draw_points:
                _draw_points(traj, base)

            if draw_agent_id:
                _put_agent_id_at(traj, agent_id, base)

    # 4) save if needed
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), img)
        if not ok:
            raise ValueError(f"failed to write output image: {out_path}")

    return img


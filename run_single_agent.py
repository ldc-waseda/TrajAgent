"""
TrajAgent 单次调用框架 (Responses API, multimodal input)

核心流程:
1 读取 system.md / user.md (纯文本)
2 读取 scene_json (结构化输入)
3 根据关键词在目录中按顺序选取图片 (保证第1张mask, 第2张xx, 第3张BB)
4 用 build_input_items_separated 构造 Responses API input
5 调用 client.responses.create 并打印输出

依赖:
  pip install -U openai

环境变量:
  OPENAI_API_KEY: 必填
  OPENAI_BASE_URL: 可选
  OPENAI_MODEL: 可选, 默认 gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence
from agent.agent_util import *
from openai import AsyncOpenAI


# =========================
# 0) 配置区
# =========================

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "1"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8000"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# 你说 keywords 写死是有目的的：保持不改
KEYWORDS = ["mask", "scenario"]



# =========================
# 1) 文件读取与输入构造
# =========================

def read_text_file(path: str) -> str:
    """读取纯文本文件，返回去掉首尾空白后的文本。"""
    return Path(path).read_text(encoding="utf-8").strip()


def read_json_file(path: str | Path) -> Dict[str, Any]:
    """读取 JSON 文件并返回 dict。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def image_path_to_data_url(path: str) -> str:
    """
    读取本地图片并转成 base64 data URL，供 Responses API 的 input_image 使用。
    """
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def build_input_items_separated(
    user_md_text: str,
    scene_json_obj: Dict[str, Any],
    image_paths: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Build Responses API 'input' with separated blocks:
      1) user instructions (from user.md) - pure text
      2) scene JSON - pure text
      3) images - input_image list (ordered)
    """
    input_items: List[Dict[str, Any]] = []

    # 1) user prompt text
    input_items.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_md_text}],
        }
    )

    # 2) scene json text
    scene_json_text = json.dumps(scene_json_obj, ensure_ascii=False, indent=2, default=str)
    input_items.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "SCENE_JSON:\n" + scene_json_text}],
        }
    )

    # 3) images (ordered)
    image_parts: List[Dict[str, Any]] = [{"type": "input_text", "text": "SCENE_IMAGES:"}]
    for img_path in image_paths:
        image_parts.append({"type": "input_image", "image_url": image_path_to_data_url(img_path)})

    input_items.append({"role": "user", "content": image_parts})
    return input_items


# =========================
# 2) 关键词检索图片（保证顺序）
# =========================

def find_one_image_by_keyword(images_dir: str, keyword: str) -> str:
    """
    在 images_dir 下找一个文件名包含 keyword 的图片（不区分大小写）。
    若匹配多张，取按文件名排序后的第一张（稳定可复现）。
    """
    d = Path(images_dir)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"images_dir not found or not a directory: {images_dir}")

    kw = keyword.lower().strip()
    if not kw:
        raise ValueError("keyword is empty")

    candidates: List[Path] = []
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and kw in p.name.lower():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No image matched keyword='{keyword}' in dir: {images_dir}")

    candidates.sort(key=lambda x: x.name.lower())
    return str(candidates[0])


def build_ordered_image_paths_by_keywords(images_dir: str, keywords: Sequence[str]) -> List[str]:
    """
    根据关键词列表生成有序 image_paths：
      keywords = ["mask","xx","BB"]
      -> [mask图, xx图, BB图]
    """
    return [find_one_image_by_keyword(images_dir, kw) for kw in keywords]


# =========================
# 3) main
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sys", dest="sys_prompt_path", required=True, help="path to system prompt (.md/.txt)")
    p.add_argument("--user", dest="user_prompt_path", required=True, help="path to user prompt (.md/.txt)")
    p.add_argument("--case_dir", dest="case_dir", required=True, help="path to one case directory")

    p.add_argument("--model", dest="model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", dest="temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max_tokens", dest="max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    return p.parse_args()


async def main_single_call_async(args: argparse.Namespace) -> None:
    # 0) case_dir and scene.json
    case_dir = Path(args.case_dir).resolve()
    scene_json_path = case_dir / f"{case_dir.name}.json"  # e.g., case_000780/case_000780.json


    # 1) images: ordered by fixed keywords (search in case_dir)
    image_paths = build_ordered_image_paths_by_keywords(str(case_dir), KEYWORDS)

    # 2) read prompts
    system_text = read_text_file(args.sys_prompt_path)
    user_text = read_text_file(args.user_prompt_path)

    # 3) read scene json (from case_dir/scene.json)
    scene_json_obj = read_json_file(scene_json_path)

    # 4) build separated input items
    input_items = build_input_items_separated(
        user_md_text=user_text,
        scene_json_obj=scene_json_obj,
        image_paths=image_paths,
    )

    # 5) call Responses API
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    resp = await client.responses.create(
        model=args.model,
        instructions=system_text,
        input=input_items,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning={"effort": "minimal"},
        text={"format": {"type": "json_object"}},
    )
    print("error:", resp.error)
    print("incomplete_details:", resp.incomplete_details)
    print("output item types:", [getattr(x, "type", None) for x in resp.output])

    out_text = resp.output_text or ""

    # # 8) Save to file (case_dir/output.json)
    out_path = Path(args.case_dir).resolve() / "gpt_output.json"
    out_path.write_text(out_text, encoding="utf-8")

    # ====== 9) Load GPT output (util) ======
    gpt_trajs = load_gpt_output_candidates(out_path)

    # ====== 10) Filter by mask (util) ======
    mask_path = case_dir / "case_mask.jpg"
    if not mask_path.exists():
        raise FileNotFoundError(f"case_mask.jpg not found: {mask_path}")

    filtered_trajs, report = filter_trajs_by_walkable_mask(
        trajs_by_agent=gpt_trajs,
        mask_path=mask_path,
        walkable_is_white=True,  # 若你mask黑色可走白色不可走，则改成 False
        threshold=128,
    )

    # ====== 11) Save filtered + report (不做额外结构转换) ======
    # 注意：dict[int,...] dump 时 key 会变成字符串，这是 JSON 规范行为，读回时再转回 int 就行
    filtered_out_path = case_dir / "gpt_output_filtered_dict.json"
    filtered_out_path.write_text(
        json.dumps(filtered_trajs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = case_dir / "gpt_output_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ====== 12) Visualize (util) ======
    scenario_img_path = case_dir / "case_scenario.jpg"
    if not scenario_img_path.exists():
        raise FileNotFoundError(f"case_scenario.jpg not found: {scenario_img_path}")

    # 原始轨迹：如果你没有就传 None；如果你有 path.json 里能抽，就自己写一个 extractor
    # 这里先最简：不画原始轨迹
    vis_out_path = case_dir / "vis_overlay.jpg"
    _ = visualize_trajs_on_scenario(
        scenario_img=scenario_img_path,
        original_trajs=None,
        gpt_trajs_by_agent=gpt_trajs,
        out_path=vis_out_path,
        draw_points=False,
        draw_agent_id=True,
    )

    print(f"[Saved] raw={out_path}")
    print(f"[Saved] filtered_dict={filtered_out_path}")
    print(f"[Saved] report={report_path}")
    print(f"[Saved] vis={vis_out_path}")
    print(f"[Filter] kept {report.get('kept_candidates')}/{report.get('total_candidates')}, "
        f"ratio={report.get('overall_keep_ratio')}")

def main() -> None:
    args = parse_args()
    asyncio.run(main_single_call_async(args))


if __name__ == "__main__":
    main()

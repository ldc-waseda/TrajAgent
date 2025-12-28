"""
TrajAgent Step2 批量调用框架 (Responses API, multimodal input)

支持:
- --case_dir 传单个 case 文件夹 或 根目录(包含多个 case_* 子目录)
- 并发执行 (asyncio + semaphore)
- 可选跳过已存在的 gpt_output_step2.json

核心流程(每个 case):
1 读取 system.md / user.md (纯文本)
2 读取 gpt_output_filtered_dict.json (结构化输入)
3 按 KEYWORDS 在 case_dir 内顺序选图 (用于 SCENE_IMAGES)
4 构造 separated input items
5 调用 client.responses.create 得到 step2 json
6 保存 gpt_output_step2.json
7 解析 worldlines 并可视化 worldline_001.jpg, worldline_002.jpg, ...

依赖:
  pip install -U openai
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from openai import AsyncOpenAI
from agent.agent_util import *  # visualize_trajs_on_scenario 等


# =========================
# 0) 配置区
# =========================

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "1"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# 你说 keywords 写死是有目的的：保持不改
KEYWORDS = ["select"]  # 按顺序找图片


# =========================
# 1) 文件读取与输入构造
# =========================

def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def read_json_file(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def image_path_to_data_url(path: str) -> str:
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
    return [find_one_image_by_keyword(images_dir, kw) for kw in keywords]


# =========================
# 3) 批量入口
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sys", dest="sys_prompt_path", required=True, help="path to system prompt (.md/.txt)")
    p.add_argument("--user", dest="user_prompt_path", required=True, help="path to user prompt (.md/.txt)")
    p.add_argument(
        "--case_dir",
        dest="case_dir",
        required=True,
        help="path to ONE case folder OR a ROOT folder containing many case_* subfolders",
    )

    p.add_argument("--model", dest="model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", dest="temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max_tokens", dest="max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)

    p.add_argument("--max_concurrency", type=int, default=int(os.getenv("MAX_CONCURRENCY", "2")))
    p.add_argument("--case_glob", type=str, default="case_*")
    p.add_argument("--skip_existing", action="store_true", help="skip case if gpt_output_step2.json exists")

    return p.parse_args()


def discover_case_dirs(root_or_case: Path, case_glob: str = "case_*") -> List[Path]:
    """
    单 case 判定:
      - 存在 gpt_output_filtered_dict.json (Step2 输入) 则认为是 case
    根目录判定:
      - 枚举其子目录 case_* 作为 case
    """
    p = root_or_case.resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"case_dir not found or not a directory: {p}")

    # 单 case 判定
    step2_input = p / "gpt_output_filtered_dict.json"
    if step2_input.exists():
        return [p]

    # 否则按 glob 找子 case
    case_dirs = [x for x in sorted(p.glob(case_glob)) if x.is_dir()]
    return case_dirs


async def process_one_case(
    case_dir: Path,
    *,
    client: AsyncOpenAI,
    system_text: str,
    user_text: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> Tuple[bool, str]:
    async with semaphore:
        try:
            case_dir = case_dir.resolve()

            out_path = case_dir / "gpt_output_step2.json"
            if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                return True, f"[Skip] {case_dir.name} already has gpt_output_step2.json"

            # Step2 输入 json 固定
            scene_json_path = case_dir / "gpt_output_filtered_dict.json"
            if not scene_json_path.exists():
                return False, f"[Fail] {case_dir.name} missing gpt_output_filtered_dict.json"

            # images: ordered by fixed keywords (search in case_dir)
            image_paths = build_ordered_image_paths_by_keywords(str(case_dir), KEYWORDS)

            # read scene json
            scene_json_obj = read_json_file(scene_json_path)

            # build separated input items
            input_items = build_input_items_separated(
                user_md_text=user_text,
                scene_json_obj=scene_json_obj,
                image_paths=image_paths,
            )

            # call Responses API
            resp = await client.responses.create(
                model=args.model,
                instructions=system_text,
                input=input_items,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                reasoning={"effort": "minimal"},
                text={"format": {"type": "json_object"}},
            )

            out_text = resp.output_text or ""
            out_path.write_text(out_text, encoding="utf-8")

            if not out_text.strip():
                return False, f"[Fail] {case_dir.name} resp.output_text is empty"

            try:
                step2_obj = json.loads(out_text)
            except json.JSONDecodeError as e:
                return False, f"[Fail] {case_dir.name} step2 output is not valid JSON: {repr(e)}"

            worldlines = step2_obj.get("worldlines", [])
            if not isinstance(worldlines, list) or len(worldlines) == 0:
                return False, f"[Fail] {case_dir.name} no worldlines found (expect key 'worldlines')"

            # background image for visualization
            scenario_img_path = case_dir / "case_base_map.jpg"
            if scenario_img_path.exists():
                bg_img = scenario_img_path
            else:
                if len(image_paths) == 0:
                    return False, f"[Fail] {case_dir.name} no images found for visualization fallback"
                bg_img = Path(image_paths[0])

            # visualize each worldline
            saved = 0
            for k, wl in enumerate(worldlines, start=1):
                agents = wl.get("agents", [])
                if not isinstance(agents, list) or len(agents) == 0:
                    continue

                gpt_trajs_by_agent = {}
                for a in agents:
                    aid = a.get("agent_id", None)
                    pts = a.get("points", None)
                    if aid is None or pts is None:
                        continue
                    try:
                        aid_int = int(aid)
                    except Exception:
                        continue

                    traj = []
                    for pxy in pts:
                        if not (isinstance(pxy, (list, tuple)) and len(pxy) == 2):
                            continue
                        x, y = int(round(pxy[0])), int(round(pxy[1]))
                        traj.append((x, y))

                    if len(traj) >= 2:
                        gpt_trajs_by_agent[aid_int] = [traj]

                if len(gpt_trajs_by_agent) == 0:
                    continue

                out_img_path = case_dir / f"worldline_{k:03d}.jpg"
                visualize_trajs_on_scenario(
                    scenario_img=bg_img,
                    original_trajs=None,
                    gpt_trajs_by_agent=gpt_trajs_by_agent,
                    out_path=out_img_path,
                    draw_points=False,
                    draw_agent_id=True,
                    origin_thickness=3,
                    cand_thickness=2,
                    point_radius=2,
                )
                saved += 1

            return True, f"[OK] {case_dir.name} worldlines={len(worldlines)} vis_saved={saved}"

        except Exception as e:
            return False, f"[Fail] {case_dir.name} exception: {repr(e)}"


async def main_batch_async(args: argparse.Namespace) -> None:
    root = Path(args.case_dir).resolve()
    case_dirs = discover_case_dirs(root, args.case_glob)
    if not case_dirs:
        raise RuntimeError(f"No case dirs found under: {root} (glob={args.case_glob})")

    # prompts only once
    system_text = read_text_file(args.sys_prompt_path)
    user_text = read_text_file(args.user_prompt_path)

    # shared client
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    sem = asyncio.Semaphore(max(1, int(args.max_concurrency)))

    tasks = [
        process_one_case(
            d,
            client=client,
            system_text=system_text,
            user_text=user_text,
            args=args,
            semaphore=sem,
        )
        for d in case_dirs
    ]

    ok_cnt = 0
    fail_cnt = 0
    for fut in asyncio.as_completed(tasks):
        ok, msg = await fut
        print(msg)
        if ok:
            ok_cnt += 1
        else:
            fail_cnt += 1

    print(f"[Done] total={len(case_dirs)} ok={ok_cnt} fail={fail_cnt}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_batch_async(args))


if __name__ == "__main__":
    main()

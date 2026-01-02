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
import numpy as np
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence
from agent.agent_util import *
from openai import AsyncOpenAI
from PIL import Image


# =========================
# 0) 配置区
# =========================

# DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "1"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8000"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# 你说 keywords 写死是有目的的：保持不改
KEYWORDS = ["mask", "scenario",]  # 按顺序找图片



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

def load_walkable_mask_bool(mask_path, *, walkable_is_white: bool = True, threshold: int = 128) -> np.ndarray:
    """
    读取 walkable mask，返回 bool 数组 mask_bool[H,W]
    True 表示可走，False 表示不可走

    walkable_is_white=True: 像素值 >= threshold 视为可走（白色可走）
    walkable_is_white=False: 像素值 <= threshold 视为可走（黑色可走）
    """

    img = Image.open(mask_path).convert("L")      # 灰度
    arr = np.array(img, dtype=np.uint8)           # HxW
    if walkable_is_white:
        return arr >= threshold
    return arr <= threshold


def build_tools_check_walkable_pixel() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "check_walkable_pixel",
            "description": "Batch check walkability for a list of pixel points on the mask. Input: points=[[x,y],...].",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "description": "List of [x,y] integer pixel coordinates to check.",
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "integer"},
                        },
                    }
                },
                "required": ["points"],
                "additionalProperties": False,
            },
        }
    ]


def _as_dict(item: Any) -> Dict[str, Any]:
    """
    openai sdk 输出 item 可能是 pydantic 对象。
    统一转成 dict，方便访问 type/name/arguments/call_id。
    """
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    return dict(item.__dict__)

async def run_responses_with_tool_loop(
    *,
    client: AsyncOpenAI,
    model: str,
    instructions: str,
    input_items: list[dict],
    tools: list[dict],
    tool_handler,  # callable(name:str, args:dict) -> dict
    temperature: float,
    max_output_tokens: int,
    max_rounds: int = 32,
) -> str:
    """
    关键循环：
    - 发起 responses.create
    - 如果模型返回 function_call，就执行本地 tool_handler
    - 立刻把 function_call_output 追加回 input_items
    - 再次 responses.create，直到不再出现 function_call
    """
    last_text = ""
    for _ in range(max_rounds):
        resp = await client.responses.create(
            model=model,
            instructions=instructions,
            input=input_items,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning={"effort": "none"},
            text={"format": {"type": "json_object"}},
        )

        last_text = resp.output_text or ""

        # 把模型输出 items 追加回上下文，保持状态
        output_items = getattr(resp, "output", None) or []
        for it in output_items:
            input_items.append(_as_dict(it))

        # 找工具调用
        tool_calls = []
        for it in output_items:
            d = _as_dict(it)
            if d.get("type") == "function_call":
                tool_calls.append(d)

        # 没有工具调用：模型已经给出最终答案
        if not tool_calls:
            return last_text

        # 有工具调用：逐个执行，并把 function_call_output 回填
        for call in tool_calls:
            name = call.get("name")
            call_id = call.get("call_id")
            arguments = call.get("arguments") or "{}"

            try:
                args = json.loads(arguments)
            except Exception:
                args = {}

            result_dict = tool_handler(name, args)

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result_dict, ensure_ascii=False),
                }
            )

    raise RuntimeError(f"Tool loop exceeded max_rounds={max_rounds}")
# =========================
# 3) main
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

    # 并发控制，建议 1-4 之间
    p.add_argument("--max_concurrency", type=int, default=int(os.getenv("MAX_CONCURRENCY", "2")))
    # 批量时只扫 case_* 目录
    p.add_argument("--case_glob", type=str, default="case_*")
    # 已经有输出就跳过
    p.add_argument("--skip_existing", action="store_true", help="skip case if gpt_output.json exists")

    return p.parse_args()


# ====== 新增: 自动发现要跑的 case 目录 ======
def discover_case_dirs(root_or_case: Path, case_glob: str = "case_*") -> List[Path]:
    """
    如果传入的是单个 case 文件夹: 里面存在 {case_name}.json 就认为是 case
    如果传入的是根目录: 就枚举其子目录 case_* 作为 case
    """
    p = root_or_case.resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"case_dir not found or not a directory: {p}")

    # 单 case 判定: case_dir/case_dir.name.json 存在
    single_json = p / f"{p.name}.json"
    if single_json.exists():
        return [p]

    # 否则按 glob 找子 case
    case_dirs = [x for x in sorted(p.glob(case_glob)) if x.is_dir()]
    return case_dirs

# ====== 新增: 单个 case 的处理逻辑封装成函数 ======
async def process_one_case(
    case_dir: Path,
    *,
    client: AsyncOpenAI,
    system_text: str,
    user_text: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> tuple[bool, str]:
    """
    返回 (ok, msg)
    ok=True 表示成功
    """
    async with semaphore:
        try:
            case_dir = case_dir.resolve()

            # 可选跳过
            out_path = case_dir / "gpt_output.json"
            if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                return True, f"[Skip] {case_dir.name} already has gpt_output.json"

            # 0) scene json 固定为 case_xxx.json
            scene_json_path = case_dir / f"{case_dir.name}.json"
            if not scene_json_path.exists():
                return False, f"[Fail] {case_dir.name} missing scene json: {scene_json_path.name}"

            # 1) images: ordered by fixed keywords (search in case_dir)
            image_paths = build_ordered_image_paths_by_keywords(str(case_dir), KEYWORDS)

            # 2) read scene json
            scene_json_obj = read_json_file(scene_json_path)

            # 3) build separated input items
            input_items = build_input_items_separated(
                user_md_text=user_text,
                scene_json_obj=scene_json_obj,
                image_paths=image_paths,
            )

            # 4) tool: 单点可走检查
            mask_path = case_dir / "case_mask.jpg"
            if not mask_path.exists():
                return False, f"[Fail] {case_dir.name} missing case_mask.jpg"

            mask_bool = load_walkable_mask_bool(
                mask_path,
                walkable_is_white=True,  # 和你后面 filter_trajs_by_walkable_mask 一致
                threshold=128,
            )
            H, W = mask_bool.shape

            tools = build_tools_check_walkable_pixel()

            def tool_handler(name: str, args: dict) -> dict:
                """
                Tool does one thing: check whether points are walkable on the mask.
                Supports single-point check and batch check.
                Return is kept simple and model-friendly.
                """
                if name != "check_walkable_pixel":
                    return {"ok": False, "reason": "unknown_tool", "detail": {"name": name}}

                def _to_int(v):
                    # model may pass float/str
                    return int(float(v))

                # Accept either a batch list or a single (x,y)
                pts = args.get("points", None)
                if pts is None:
                    # single point mode
                    try:
                        x = _to_int(args.get("x"))
                        y = _to_int(args.get("y"))
                        pts = [[x, y]]
                    except Exception:
                        return {"ok": False, "reason": "bad_args", "detail": {"args": args}}
                else:
                    # batch mode
                    if not isinstance(pts, (list, tuple)):
                        return {"ok": False, "reason": "bad_args", "detail": {"args": args, "hint": "points must be a list"}}

                results = []
                all_ok = True

                for i, p in enumerate(pts):
                    try:
                        x = _to_int(p[0])
                        y = _to_int(p[1])
                    except Exception:
                        all_ok = False
                        results.append({
                            "ok": False,
                            "reason": "bad_point",
                            "detail": {"index": i, "point": p}
                        })
                        continue

                    # bounds check
                    if x < 0 or x >= W or y < 0 or y >= H:
                        all_ok = False
                        results.append({
                            "ok": False,
                            "reason": "out_of_bounds",
                            "detail": {"index": i, "x": x, "y": y, "W": W, "H": H}
                        })
                        continue

                    # walkability check
                    ok = bool(mask_bool[y, x])
                    if not ok:
                        all_ok = False

                    results.append({
                        "ok": ok,
                        "reason": "ok" if ok else "not_walkable",
                        "detail": {"index": i, "x": x, "y": y}
                    })

                return {
                    "ok": all_ok,
                    "reason": "ok" if all_ok else "some_not_walkable",
                    "detail": {
                        "W": W,
                        "H": H,
                        "results": results
                    }
                }
                        # 可选：不改你 system.md 文件，直接在 instructions 末尾补一句工具提示
            tool_hint = (
                "\n\nTool usage rule: You MUST call check_walkable_pixel in batch mode once "
                "using points=[[x,y], ...] that includes ALL starts and ends for ALL agents. "
                "Do NOT call it one-by-one."
            )
            instructions = system_text + tool_hint


            # 5) call Responses API with tool loop
            out_text = await run_responses_with_tool_loop(
                client=client,
                model=args.model,
                instructions=instructions,
                input_items=input_items,
                tools=tools,
                tool_handler=tool_handler,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                max_rounds=32,
            )

            out_path = case_dir / "gpt_output.json"
            out_path.write_text(out_text, encoding="utf-8")


            # 5) Load GPT output (util)
            gpt_trajs = load_gpt_output_candidates(out_path)

            # # 6) Filter by mask (util)
            # mask_path = case_dir / "case_mask.jpg"
            # if not mask_path.exists():
            #     return False, f"[Fail] {case_dir.name} missing case_mask.jpg"

            # filtered_trajs, report = filter_trajs_by_walkable_mask(
            #     trajs_by_agent=gpt_trajs,
            #     mask_path=mask_path,
            #     walkable_is_white=True,
            #     threshold=128,
            # )

            # # 7) Save filtered + report
            # filtered_out_path = case_dir / "gpt_output_filtered_dict.json"
            # filtered_out_path.write_text(
            #     json.dumps(filtered_trajs, ensure_ascii=False, indent=2),
            #     encoding="utf-8",
            # )

            # report_path = case_dir / "gpt_output_report.json"
            # report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

            # 8) Visualize (util)
            scenario_img_path = case_dir / "case_mask.jpg"
            if not scenario_img_path.exists():
                return False, f"[Fail] {case_dir.name} missing case_mask.jpg"

            vis_raw_path = case_dir / "vis_gpt_trajs.jpg"
            _ = visualize_trajs_on_scenario(
                scenario_img=scenario_img_path,
                original_trajs=None,
                gpt_trajs_by_agent=gpt_trajs,
                out_path=vis_raw_path,
                draw_points=False,
                draw_agent_id=True,
            )

            # vis_sel_path = case_dir / "vis_select_trajs.jpg"
            # _ = visualize_trajs_on_scenario(
            #     scenario_img=scenario_img_path,
            #     original_trajs=None,
            #     gpt_trajs_by_agent=filtered_trajs,
            #     out_path=vis_sel_path,
            #     draw_points=False,
            #     draw_agent_id=True,
            # )

            # kept = report.get("kept_candidates")
            # total = report.get("total_candidates")
            # ratio = report.get("overall_keep_ratio")
            # return True, f"[OK] {case_dir.name} kept={kept}/{total} ratio={ratio}"
            return True, f"[OK] {case_dir.name} wrote gpt_output.json"
        except Exception as e:
            return False, f"[Fail] {case_dir.name} exception: {repr(e)}"


# ====== 修改: main 改成批量入口 ======
async def main_batch_async(args: argparse.Namespace) -> None:
    root = Path(args.case_dir).resolve()
    case_dirs = discover_case_dirs(root, args.case_glob)
    if not case_dirs:
        raise RuntimeError(f"No case dirs found under: {root} (glob={args.case_glob})")

    # 读 prompts 只做一次
    system_text = read_text_file(args.sys_prompt_path)
    user_text = read_text_file(args.user_prompt_path)

    # client 复用
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
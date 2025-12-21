import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import cv2

# Corrected imports relative to the project root
from agent.utils.debug import IS_DEBUG_ENV, ensure_cache_dirs, save_debug_image, save_debug_text
from agent.utils.vllm_power import is_sleeping, wake_up
from agent.utils.openai_wrapper import create_wrapped_client
from data_loader.eth_data_loader import ETHLoader, collect_scenario_files
from agent.utils.prompt_template import render_template
from agent.utils.image_io import image_to_data_url
from agent.utils.traj_draw import draw_trajectory_on_image

# --- Pydantic Models (previously in base.py) ---
class Alternative(BaseModel):
    description: str = Field(description="The description of the trajectory.")
    points: List[List[int]]

class TrajectoryAlternatives(BaseModel):
    alternatives: List[Alternative]

# --- Configuration ---
# Path to the root of the ETH/UCY datasets
DATASET_ROOT = os.environ.get("DATASET_PATH", "./ETH_datasets/")

# LLM Configuration
API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE") # <-- IMPORTANT: Set your API key
MODEL = os.environ.get("VL_MODEL", "gpt-5-mini")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "10"))
AUTO_AWAKE = os.environ.get("AUTO_AWAKE", "1") in ("1", "true", "TRUE", "True")

# Strategy and Debugging
STRATEGY_NAME = os.environ.get("STRATEGY_NAME", "case")
DEBUG_LIMIT = int(os.environ.get("DEBUG_LIMIT", "5"))

# Scenarios to process
SCENARIOS = [
    'ETH',
    'HOTEL',
    'ZARA01',
    'ZARA02',
    'STUDENT'
]
# To process only specific scenarios, set this list e.g., ["ETH", "HOTEL"]
SELECTED_SCENARIOS = None

# --- Logic moved from CaseGenerator and BaseTrajectoryGenerator ---

def _convert_trajectory_to_int(points: np.ndarray) -> List[List[int]]:
    """Convert trajectory points to integers."""
    return [[int(point[0]), int(point[1])] for point in points]

def prepare_template_data(agents_data: Dict[str, List[List[int]]]) -> Dict[str, Any]:
    """Prepares template data for the 'case' strategy."""
    # This function now directly handles the logic from CaseGenerator.prepare_template_data
    # It receives the "agents" dictionary directly.
    
    if not agents_data:
        raise ValueError("agents_data dictionary is empty.")

    # For this example, let's assume we focus on the first agent in the window
    first_agent_id = next(iter(agents_data))
    first_agent_traj = np.array(agents_data[first_agent_id])
    
    traj_int = _convert_trajectory_to_int(first_agent_traj)
    start_point = traj_int[0]
    
    return {
        'traj': traj_int,
        'anno_text': f"Trajectory for agent {first_agent_id}", # Example annotation
        'start_point': start_point,
        # You can add all agents' data if the prompt is designed for it
        'all_agents_data': {pid: traj for pid, traj in agents_data.items()}
    }

def create_full_trajectory(original_traj_points: List[List[int]], predicted_points: List[List[float]]) -> np.ndarray:
    """Creates a full 20-point trajectory from original and predicted points."""
    # This function replaces CaseGenerator.create_full_trajectory
    start_point = np.array(original_traj_points[0:1], dtype=np.float32)
    pred_array = np.array(predicted_points, dtype=np.float32)
    full_traj = np.vstack([start_point, pred_array])
    return full_traj[:20] # Ensure exactly 20 points

def get_image_dimensions(img_any: Any) -> Tuple[int, int]:
    """Get image dimensions (width, height)."""
    height, width = img_any.shape[:2]
    return width, height

def load_scenario_mask(scenario: str) -> Optional[np.ndarray]:
    """Loads the scenario mask image."""
    mask_dir = Path("./seg_shrink")
    mask_path = mask_dir / f"{scenario}_masked.jpg"
    if not mask_path.exists():
        mask_path = mask_dir / f"{scenario.lower()}_masked.jpg"
    if mask_path.exists():
        return cv2.imread(str(mask_path))
    raise FileNotFoundError(f"Mask not found for scenario: {scenario}")

async def run_generation_for_window(
    client: AsyncOpenAI,
    idx: int,
    window_pack: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
    """Prepares and runs trajectory generation for a single data window."""
    
    # Unpack the data for the window
    image = window_pack["image"]
    trajectory_data = window_pack["trajectory_data"]
    window_info = window_pack["window_info"]
    scenario = window_pack["scenario"]
    start_frame, end_frame = window_info
    
    traj_id = f"window_{start_frame:06d}_{end_frame:06d}"

    # --- All generation logic is now in this function ---
    
    # 1. Prepare Template Data
    if STRATEGY_NAME == "case":
        # Pass the nested 'agents' dictionary to the prepare function
        template_data = prepare_template_data(trajectory_data['agents'])
        system_template = "gen_full_traj_system_v1"
        user_template = "gen_full_traj_user_v1"
        expected_pred_count = 19
    else:
        raise ValueError(f"Unknown strategy: {STRATEGY_NAME}")

    width, height = get_image_dimensions(image)
    template_data['image_size'] = f"{width}x{height}"

    # 2. Render Prompts
    system_prompt = render_template(system_template, template_data)
    user_prompt = render_template(user_template, template_data)
    
    # 3. Prepare API Message
    data_url = image_to_data_url(image)
    user_content = [
        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        {"type": "text", "text": user_prompt},
    ]
    
    try:
        scenario_mask = load_scenario_mask(scenario)
        mask_data_url = image_to_data_url(scenario_mask)
        user_content.insert(0, {"type": "image_url", "image_url": {"url": mask_data_url, "detail": "high"}})
        save_debug_image(f'{scenario}_{traj_id}_mask', scenario_mask)
    except FileNotFoundError as e:
        print(f"[Warning] {e}, continuing without mask.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # 4. Call API
    try:
        async with semaphore:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
                response_format={"type": "json_object", "schema": TrajectoryAlternatives.model_json_schema()}
            )
    except Exception as e:
        print(f"[Error] API call failed for idx={idx}, id={traj_id}: {e}")
        return idx, None

    # 5. Parse and Process Response
    parsed_json = json.loads(resp.choices[0].message.content or "{}")
    
    save_debug_text(f'{scenario}_{traj_id}_system_prompt.md', system_prompt)
    save_debug_text(f'{scenario}_{traj_id}_user_prompt.md', user_prompt)
    save_debug_image(f'{scenario}_{traj_id}_img', image)
    save_debug_text(f'{scenario}_{traj_id}_traj.json', json.dumps(parsed_json, ensure_ascii=False, indent=2))

    valid_trajectories = []
    if "alternatives" in parsed_json:
        # This logic focuses on the first agent, matching prepare_template_data
        agents_data = trajectory_data['agents']
        if not agents_data:
            return idx, None # No agents in window to begin with

        first_agent_id = next(iter(agents_data))
        original_traj_points = agents_data[first_agent_id]

        for alt_idx, alternative in enumerate(parsed_json["alternatives"]):
            points = alternative.get("points", [])
            if len(points) != expected_pred_count:
                continue # Validation failed

            full_traj = create_full_trajectory(original_traj_points, points)
            
            # Create debug image
            img_with_traj = draw_trajectory_on_image(image, full_traj)
            save_debug_image(f'{scenario}_{traj_id}_alt{alt_idx}', img_with_traj)

            valid_trajectories.append({
                "traj_id": f"{traj_id}_alt{alt_idx}",
                "trajectory": full_traj,
                "annotation": alternative.get("description", ""),
                "filename": f"{scenario}_{traj_id}.jpg",
                "scenario": scenario,
            })

    return idx, valid_trajectories


async def main_async():
    """Main asynchronous entry point."""
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("[Error] Please set your OpenAI API key in the script or via the OPENAI_API_KEY environment variable.")
        return

    if AUTO_AWAKE and "api.openai.com" not in API_BASE:
        try:
            if await is_sleeping(API_BASE, api_key=API_KEY) in (True, None):
                print("[Info] Waking up model...")
                await wake_up(API_BASE, api_key=API_KEY)
        except Exception as e:
            print(f"[Warn] Failed to check or wake up model: {e}")

    # --- Data Loading ---
    scenarios_to_process = SELECTED_SCENARIOS or SCENARIOS
    print(f"[Data] Starting data loading for scenarios: {scenarios_to_process}")

    all_windows_data = []
    for scenario in scenarios_to_process:
        try:
            txt_path, video_path = collect_scenario_files(DATASET_ROOT, scenario)
            if not txt_path:
                print(f"[Warn] No data found for scenario '{scenario}', skipping.")
                continue
            
            data_loader = ETHLoader(txt_path[0], video_path[0])
            
            # Load all window data for the scenario into memory
            scenario_windows = data_loader.load_sliding_window_data(window_size=200, step=200)
            
            # Add scenario name to each pack
            for window_pack in scenario_windows:
                window_pack['scenario'] = scenario
            
            all_windows_data.extend(scenario_windows)

        except Exception as e:
            print(f"[Error] Failed during data processing for {scenario}: {e}")

    if not all_windows_data:
        print("[Error] No data windows were loaded. Exiting.")
        return

    print(f"[Data] Total windows to process: {len(all_windows_data)}")

    if IS_DEBUG_ENV:
        all_windows_data = all_windows_data[:DEBUG_LIMIT]
        print(f"[Debug] Limited to {len(all_windows_data)} windows.")

    # --- API Client and Task Execution ---
    ensure_cache_dirs()

    client = create_wrapped_client(
        api_key=API_KEY,
        base_url=API_BASE,
        model=MODEL,
        strategy=STRATEGY_NAME
    )
    
    print(f"[Strategy] Using trajectory generation strategy: '{STRATEGY_NAME}'")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [
        run_generation_for_window(client, i, window_pack, semaphore)
        for i, window_pack in enumerate(all_windows_data)
    ]

    results = []
    for fut in asyncio.as_completed(tasks):
        results.append(await fut)

    results.sort(key=lambda x: x[0])

    # --- Result Processing ---
    all_generated_trajectories = [traj for _, gen_trajs in results if gen_trajs for traj in gen_trajs]

    print(f"\n[Done] Generated {len(all_generated_trajectories)} new trajectories from {len(all_windows_data)} windows.")
    
    # Note: Saving generated data is disabled. You can add logic here to save
    # `all_generated_trajectories` to a file if needed.
    print("[Info] Saving of generated data is not implemented in this script.")

def main():
    """Synchronous entry point."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()


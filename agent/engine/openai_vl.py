import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

from agent.utils.debug import IS_DEBUG_ENV, ensure_cache_dirs
from agent.utils.data_io import load_npz, select_scenarios
from agent.utils.vllm_power import is_sleeping, wake_up
from agent.utils.openai_wrapper import create_wrapped_client
from agent.engine.generation import *

NPZ_PATH = os.environ.get("NPZ_PATH", "all_data.npz")
AUGMENTED_NPZ_PATH = os.environ.get("AUGMENTED_NPZ_PATH", "augmented_data.npz")
VL_MODEL = os.environ.get("VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")


API_BASE = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:18100/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "100"))

AUTO_AWAKE = os.environ.get("AUTO_AWAKE", "1") in ("1", "true", "TRUE", "True")

STRATEGY_NAME = os.environ.get("STRATEGY_NAME", "pred_traj_v1")

# Stepwise-specific parameters (only used when STRATEGY_NAME == "stepwise_traj_v1")
STEPWISE_NUM_STEPS = int(os.environ.get("STEPWISE_NUM_STEPS", "20"))
STEPWISE_REFINEMENT_ITERATIONS = int(os.environ.get("STEPWISE_REFINEMENT_ITERATIONS", "0"))

DEBUG_N = int(os.environ.get("DEBUG_N", "1"))
DEBUG_LIMIT = int(os.environ.get("DEBUG_LIMIT", "5"))


# API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
# VL_MODEL = "gemini-2.5-flash"

# VL_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"

# API_BASE = "http://192.168.4.193:18100/v1"
# VL_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# VL_MODEL = "Qwen/Qwen3-VL-30B-A3B-Thinking"
# VL_MODEL = "Qwen/Qwen3-VL-32B-Thinking"

API_BASE = "https://api.openai.com/v1"
VL_MODEL = "gpt-5-mini"
# VL_MODEL = "gpt-5"
TEMPERATURE = 1 # static for gpt5
MAX_NEW_TOKENS = 4096 # for gpt5, to control the output length

# STRATEGY_NAME = "stepwise_traj_v1"
# STRATEGY_NAME = "full_traj_v2"
# STRATEGY_NAME = "full_traj_v2_nomask"
# STRATEGY_NAME = "full_traj_v2_maskonly"
# STRATEGY_NAME = "full_traj_v2_noimage"
STRATEGY_NAME = "full_traj_v3"
# STRATEGY_NAME = "full_traj_v3_nomask"
# STRATEGY_NAME = "full_traj_v3_noimage"

# NPZ_PATH = "sdd_data.npz"
# AUGMENTED_NPZ_PATH = "sdd_syn_data_noback.npz"

NPZ_PATH = "all_data.npz"
# AUGMENTED_NPZ_PATH = "eth_syn_data_noback.npz"
# AUGMENTED_NPZ_PATH = "eth_syn_data_noback_nomask.npz"
# AUGMENTED_NPZ_PATH = "eth_syn_data_noback_noimage.npz"
AUGMENTED_NPZ_PATH = "temp.npz"

SELECTED_SCENARIOS = None
# SELECTED_SCENARIOS = ["bookstore_0", "bookstore_1", "bookstore_2"]


def get_trajectory_generator(strategy_name: str, model: str, temperature: float = 0.2, max_completion_tokens: int = 1024, **kwargs):
    """
    Factory function to create trajectory generators based on strategy name.
    
    Args:
        strategy_name: Name of the strategy ("pred_traj_v1", "full_traj_v1", "full_traj_v2", "stepwise_traj_v1")
        model: Model name to use
        temperature: Temperature for generation
        max_completion_tokens: Maximum tokens for generation
        **kwargs: Additional parameters for specific generators (e.g., refinement_iterations, num_steps)
    """
    if strategy_name == "case":
        return CaseGenerator(model, temperature, max_completion_tokens)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

async def run_generation(
    client: AsyncOpenAI,
    idx: int,
    traj_id: Any,
    traj: np.ndarray,
    anno_text: str,
    img_any: Any,
    filename: str,
    scenario: str,
    semaphore: asyncio.Semaphore,
    generator=None,
) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
    """Run trajectory generation using the specified generator strategy."""
    if generator is None:
        generator = get_trajectory_generator(STRATEGY_NAME, VL_MODEL, TEMPERATURE, MAX_NEW_TOKENS)
    
    try:
        return await generator.run_generation(
            client, idx, traj_id, traj, anno_text, img_any, filename, scenario, semaphore
        )
    except RuntimeError as e:
        error_msg = str(e)
        # Handle wrapper-specific errors (queued/cache miss)
        if error_msg.startswith("REQUEST_QUEUED:") or error_msg.startswith("CACHE_MISS:"):
            # This is expected in batch_write/cache_first modes
            return idx, None
        # Re-raise other RuntimeErrors
        raise

async def main_async():
    if AUTO_AWAKE and API_BASE and API_BASE.startswith("http") and "api.openai.com" not in API_BASE:
        try:
            # Prefer dev sleep mode endpoints if available
            sleeping = await is_sleeping(API_BASE, api_key=API_KEY)
            # If sleeping or unknown, attempt to wake up weights first to reduce peak mem
            if sleeping is None or sleeping:
                ok = await wake_up(API_BASE, tags=["weights"], api_key=API_KEY)
                if ok:
                    # Ensure fully awake after weights are restored
                    await wake_up(API_BASE, api_key=API_KEY)
        except (OSError, ValueError) as e:
            print(f"[Awake] skipped or failed: {e}")

    pack = load_npz(NPZ_PATH)
    
    if SELECTED_SCENARIOS is not None:
        pack = select_scenarios(SELECTED_SCENARIOS, pack)
    
    # Load original data - note the actual structure from NPZ file
    anno_feats = [str(x) for x in pack["annotations"]]
    scenarios_imgs = pack["imgs"]
    traj_ids = pack["traj_ids"]
    trajs = pack["trajs"]  # Shape: (N, 20, 2) float32
    traj_lengths = pack["traj_lengths"]  # Shape: (N,) int32
    filenames = pack["filenames"]
    scenarios = pack["scenarios"]
    
    N = len(anno_feats)
    
    # Sort all data by scenario (primary) and traj_id (secondary)
    sort_keys = [(scenarios[i], traj_ids[i], i) for i in range(N)]
    sort_keys.sort(key=lambda x: (x[0], x[1]))  # Sort by (scenario, traj_id)
    sorted_indices = [x[2] for x in sort_keys]
    
    # Apply sorted indices to all data arrays
    anno_feats = [anno_feats[i] for i in sorted_indices]
    scenarios_imgs = scenarios_imgs[sorted_indices]
    traj_ids = traj_ids[sorted_indices]
    trajs = trajs[sorted_indices]
    traj_lengths = traj_lengths[sorted_indices]
    filenames = filenames[sorted_indices]
    scenarios = scenarios[sorted_indices]
    
    print(f"[Data] Sorted {N} samples by scenario and traj_id")
    
    # In debug mode, select first DEBUG_N samples per scenario
    selected_indices = list(range(N))
    if IS_DEBUG_ENV:
        # Group indices by scenario
        scenario_indices = {}
        for i in range(N):
            scenario = scenarios[i]
            if scenario not in scenario_indices:
                scenario_indices[scenario] = []
            scenario_indices[scenario].append(i)
        
        # Select first DEBUG_N from each scenario
        selected_indices = []
        for scenario, indices in sorted(scenario_indices.items()):
            selected = indices[:DEBUG_N]
            selected_indices.extend(selected)
            print(f"[Debug] Scenario '{scenario}': selected {len(selected)} of {len(indices)} samples")
        
        selected_indices.sort()  # Keep original order
        print(f"[Debug] Total samples selected: {len(selected_indices)} from {len(scenario_indices)} scenarios")
        
        # limit the number of samples to DEBUG_LIMIT
        selected_indices = selected_indices[:DEBUG_LIMIT]
    
    print(f"[Data] samples = {len(selected_indices)}")
    print(f"[Data] trajs shape = {trajs.shape}, dtype = {trajs.dtype}")
    print(f"[Data] img example shape = {scenarios_imgs[0].shape}, dtype = {scenarios_imgs[0].dtype}")

    ensure_cache_dirs()

    # Use wrapped client for cache and batch support
    client = create_wrapped_client(
        api_key=API_KEY or "",
        base_url=API_BASE,
        model=VL_MODEL,
        strategy=STRATEGY_NAME
    )
    
    # Create the trajectory generator based on strategy
    # Pass stepwise-specific parameters if using stepwise strategy
    if STRATEGY_NAME == "stepwise_traj_v1":
        generator = get_trajectory_generator(
            STRATEGY_NAME, VL_MODEL, TEMPERATURE, MAX_NEW_TOKENS,
            num_steps=STEPWISE_NUM_STEPS,
            refinement_iterations=STEPWISE_REFINEMENT_ITERATIONS
        )
        print(f"[Strategy] Using trajectory generation strategy: {STRATEGY_NAME}")
        print(f"[Strategy]   - num_steps: {STEPWISE_NUM_STEPS}")
        print(f"[Strategy]   - refinement_iterations: {STEPWISE_REFINEMENT_ITERATIONS}")
    else:
        generator = get_trajectory_generator(STRATEGY_NAME, VL_MODEL, TEMPERATURE, MAX_NEW_TOKENS)
        print(f"[Strategy] Using trajectory generation strategy: {STRATEGY_NAME}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = []
    for i in selected_indices:
        tasks.append(
            run_generation(
                client=client,
                idx=i,
                traj_id=traj_ids[i],
                traj=trajs[i],
                anno_text=anno_feats[i],
                img_any=scenarios_imgs[i],
                filename=filenames[i],
                scenario=scenarios[i],
                semaphore=semaphore,
                generator=generator,
            )
        )

    results: List[Tuple[int, Optional[List[Dict[str, Any]]]]] = []
    for fut in asyncio.as_completed(tasks):
        results.append(await fut)

    results.sort(key=lambda x: x[0])

    # Collect all generated trajectories
    all_generated_trajectories = []
    for _, generated_trajs in results:
        if generated_trajs is not None:
            all_generated_trajectories.extend(generated_trajs)

    print(f"[Generated] {len(all_generated_trajectories)} new trajectories from {len(selected_indices)} original samples")
    
    # Check if we're in batch mode
    mode = os.environ.get("MODE", "realtime")
    if mode in ("batch_write", "cache_first"):
        from agent.utils.openai_wrapper import BatchQueueManager
        # Get task name from client
        task_name = client.queue_mgr.task_name if hasattr(client, 'queue_mgr') else "default"
        task_info = client.queue_mgr.get_task_info() if hasattr(client, 'queue_mgr') else None
        
        print(f"\n[Mode] Running in '{mode}' mode")
        print(f"[Task] Task name: {task_name}")
        if task_info:
            print(f"[Task] Model: {task_info.get('model', 'N/A')}")
            print(f"[Task] Strategy: {task_info.get('strategy', 'N/A')}")
            print(f"[Queue] {task_info.get('total_requests', 0)} requests queued")
        if all_generated_trajectories:
            print(f"[Cache] {len(all_generated_trajectories)} trajectories from cache")
        print(f"\nNext steps:")
        print(f"  1. Run: python scripts/batch_prepare.py {task_name}")
        print(f"  2. Run: python scripts/batch_upload.py {task_name}")
        print("  3. Wait for batches to complete")
        print("  4. Run: python scripts/batch_download.py")
        print("  5. Run: python scripts/batch_to_cache.py")
        print("  6. Re-run this script with MODE=cache_first to use results")
        if not all_generated_trajectories:
            return  # Skip NPZ generation if no trajectories

    # Prepare data for saving
    if all_generated_trajectories:
        # Combine original and generated data
        all_traj_ids = list(traj_ids) + [traj["traj_id"] for traj in all_generated_trajectories]
        all_annotations = list(anno_feats) + [traj["annotation"] for traj in all_generated_trajectories]
        all_filenames = list(filenames) + [traj["filename"] for traj in all_generated_trajectories]
        all_scenarios = list(scenarios) + [traj["scenario"] for traj in all_generated_trajectories]
        
        # Handle trajectory data - ensure consistent shape (N, 20, 2)
        original_trajs_list = []
        for i in range(len(trajs)):
            original_trajs_list.append(trajs[i])  # Already (20, 2) shape
        
        generated_trajs_list = []
        generated_lengths_list = []
        for traj in all_generated_trajectories:
            traj_data = traj["trajectory"]  # Should be (20, 2)
            if traj_data.shape != (20, 2):
                print(f"[Warning] Generated trajectory shape {traj_data.shape} != (20, 2), padding/truncating")
                # Pad or truncate to (20, 2)
                if len(traj_data) < 20:
                    # Pad with last point
                    last_point = traj_data[-1:] if len(traj_data) > 0 else np.array([[0, 0]])
                    padding = np.repeat(last_point, 20 - len(traj_data), axis=0)
                    traj_data = np.vstack([traj_data, padding])
                else:
                    traj_data = traj_data[:20]
            generated_trajs_list.append(traj_data.astype(np.float32))
            generated_lengths_list.append(traj["traj_length"])
        
        all_trajs_list = original_trajs_list + generated_trajs_list
        all_traj_lengths = list(traj_lengths) + generated_lengths_list
        
        # Handle image data
        all_imgs_list = []
        for i in range(len(scenarios_imgs)):
            all_imgs_list.append(scenarios_imgs[i])
        for traj in all_generated_trajectories:
            all_imgs_list.append(traj["img"])

        # Convert to numpy arrays with correct dtypes
        all_trajs = np.array(all_trajs_list, dtype=np.float32)  # Shape: (N_total, 20, 2)
        all_traj_lengths = np.array(all_traj_lengths, dtype=np.int32)
        all_annotations = np.array(all_annotations, dtype=object)
        all_filenames = np.array(all_filenames, dtype=object)
        all_scenarios = np.array(all_scenarios, dtype=object)
        all_traj_ids = np.array(all_traj_ids, dtype=object)
        all_imgs = np.array(all_imgs_list, dtype=object)

        print("[NPZ Info] Final shapes:")
        print(f"  trajs: {all_trajs.shape} {all_trajs.dtype}")
        print(f"  traj_lengths: {all_traj_lengths.shape} {all_traj_lengths.dtype}")
        print(f"  annotations: {all_annotations.shape} {all_annotations.dtype}")
        print(f"  filenames: {all_filenames.shape} {all_filenames.dtype}")
        print(f"  scenarios: {all_scenarios.shape} {all_scenarios.dtype}")
        print(f"  traj_ids: {all_traj_ids.shape} {all_traj_ids.dtype}")
        print(f"  overlay_imgs: {all_imgs.shape} {all_imgs.dtype}")

        # Save only the generated data
        augmented_npz_path = AUGMENTED_NPZ_PATH
        generated_trajs = np.array(generated_trajs_list, dtype=np.float32)
        generated_traj_lengths = np.array(generated_lengths_list, dtype=np.int32)
        generated_annotations = np.array([traj["annotation"] for traj in all_generated_trajectories], dtype=object)
        generated_filenames = np.array([traj["filename"] for traj in all_generated_trajectories], dtype=object)
        generated_scenarios = np.array([traj["scenario"] for traj in all_generated_trajectories], dtype=object)
        generated_traj_ids = np.array([traj["traj_id"] for traj in all_generated_trajectories], dtype=object)
        generated_imgs = np.array([traj["img"] for traj in all_generated_trajectories], dtype=object)
        
        np.savez_compressed(
            augmented_npz_path,
            trajs=generated_trajs,
            traj_lengths=generated_traj_lengths,
            annotations=generated_annotations,
            filenames=generated_filenames,
            scenarios=generated_scenarios,
            traj_ids=generated_traj_ids,
            overlay_imgs=generated_imgs
        )
        print(f"[Saved] Augmented data with {len(all_generated_trajectories)} trajectories to {augmented_npz_path}")
    else:
        print("[Warning] No valid trajectories were generated")

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

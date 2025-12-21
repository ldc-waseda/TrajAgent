import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent.utils.debug import IS_DEBUG_ENV, ensure_cache_dirs
from agent.utils.data_io import load_npz, select_scenarios
from agent.utils.openai_wrapper import create_wrapped_client

# Configuration from environment
NPZ_PATH = os.environ.get("NPZ_PATH", "all_data.npz")
AUGMENTED_NPZ_PATH = os.environ.get("AUGMENTED_NPZ_PATH", "gpt5_multiturn_augmented.npz")
VL_MODEL = os.environ.get("VL_MODEL", "gpt-5-mini")
API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "16"))

# Multi-turn specific
MULTI_TURNS = int(os.environ.get("MULTI_TURNS", "3"))
TURN_PROMPT_TEMPLATE = os.environ.get(
    "TURN_PROMPT_TEMPLATE",
    "Given the annotation and current assistant reply, provide a refined summary and (optionally) the predicted trajectory as a JSON object under key `trajectory` when applicable.\nAnnotation: {annotation}\nAssistant: {assistant_reply}\n"
)

STRATEGY_NAME = os.environ.get("STRATEGY_NAME", "multi_turn_gpt5mini")
SELECTED_SCENARIOS = None


async def multi_turn_generation(
    client: Any,
    idx: int,
    traj_id: Any,
    annotation: str,
    img_any: Any,
    filename: str,
    scenario: str,
    semaphore: asyncio.Semaphore,
    num_turns: int = MULTI_TURNS,
    temperature: float = TEMPERATURE,
    max_completion_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Perform a multi-turn chat with the model and return final text + full dialog.

    Returns (idx, result_dict) where result_dict is None if request was queued / cache miss.
    """
    messages: List[Dict[str, Any]] = []

    # System message to set assistant behavior
    messages.append({
        "role": "system",
        "content": (
            "You are an expert visual-language reasoning assistant. "
            "When asked, provide concise, structured answers. If asked to output a trajectory, "
            "format it as a JSON array under key `trajectory` with shape (20,2) or variable length."
        ),
    })

    # Initial user prompt includes the annotation and an instruction for the first turn
    initial_user = (
        f"Annotation: {annotation}\n" f"Filename: {filename}\n" f"Scenario: {scenario}\n"
        "Please summarize the important cues from the annotation and (optionally) propose a predicted trajectory in JSON format under key `trajectory`."
    )
    messages.append({"role": "user", "content": initial_user})

    dialog: List[Dict[str, str]] = []

    try:
        for turn in range(num_turns):
            # Use semaphore to limit parallel calls at this script level
            if semaphore is not None:
                async with semaphore:
                    resp = await client.chat.completions.parse(
                        messages=messages,
                        model=VL_MODEL,
                        temperature=temperature,
                        max_completion_tokens=max_completion_tokens,
                    )
            else:
                resp = await client.chat.completions.parse(
                    messages=messages,
                    model=VL_MODEL,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                )

            # Extract content
            content = ""
            try:
                content = resp.choices[0].message.content
            except Exception:
                # Fallback: try attribute access
                content = getattr(resp, "content", "")

            # Save dialog entry
            dialog.append({"role": "assistant", "content": content})

            # Prepare next user turn: ask for refinement using template
            # For last turn we don't add another user message
            if turn < num_turns - 1:
                assistant_reply = content
                user_msg = TURN_PROMPT_TEMPLATE.format(annotation=annotation, assistant_reply=assistant_reply)
                messages.append({"role": "assistant", "content": assistant_reply})
                messages.append({"role": "user", "content": user_msg})

        # After all turns, return final assistant content and whole dialog
        result = {
            "idx": idx,
            "traj_id": traj_id,
            "filename": filename,
            "scenario": scenario,
            "final_text": content,
            "dialog": dialog,
            "img": img_any,
        }
        return idx, result

    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("REQUEST_QUEUED:") or msg.startswith("CACHE_MISS:"):
            return idx, None
        raise


async def main_async():
    pack = load_npz(NPZ_PATH)

    if SELECTED_SCENARIOS is not None:
        pack = select_scenarios(SELECTED_SCENARIOS, pack)

    annotations = [str(x) for x in pack["annotations"]]
    scenarios_imgs = pack["imgs"]
    traj_ids = pack["traj_ids"]
    filenames = pack["filenames"]
    scenarios = pack["scenarios"]

    N = len(annotations)

    # Simple selection / debug same as other scripts
    selected_indices = list(range(N))
    if IS_DEBUG_ENV:
        selected_indices = selected_indices[: min(5, len(selected_indices))]

    print(f"[MultiTurn] Samples = {len(selected_indices)}, model={VL_MODEL}, turns={MULTI_TURNS}")

    ensure_cache_dirs()

    client = create_wrapped_client(
        api_key=API_KEY or "",
        base_url=API_BASE,
        model=VL_MODEL,
        strategy=STRATEGY_NAME,
        max_concurrency=MAX_CONCURRENCY,
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = []
    for i in selected_indices:
        tasks.append(
            multi_turn_generation(
                client,
                i,
                traj_ids[i],
                annotations[i],
                scenarios_imgs[i],
                filenames[i],
                scenarios[i],
                semaphore,
                num_turns=MULTI_TURNS,
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_NEW_TOKENS,
            )
        )

    results = []
    for fut in asyncio.as_completed(tasks):
        results.append(await fut)

    results.sort(key=lambda x: x[0])

    generated = [r for _, r in results if r is not None]

    print(f"[MultiTurn] Completed {len(generated)} dialogs")

    if generated:
        # Save generated final_texts and dialogs to augmented npz
        idxs = np.array([g["idx"] for g in generated], dtype=object)
        traj_ids_out = np.array([g["traj_id"] for g in generated], dtype=object)
        final_texts = np.array([g["final_text"] for g in generated], dtype=object)
        dialogs = np.array([g["dialog"] for g in generated], dtype=object)
        filenames_out = np.array([g["filename"] for g in generated], dtype=object)
        scenarios_out = np.array([g["scenario"] for g in generated], dtype=object)
        imgs_out = np.array([g["img"] for g in generated], dtype=object)

        np.savez_compressed(
            AUGMENTED_NPZ_PATH,
            idxs=idxs,
            traj_ids=traj_ids_out,
            final_texts=final_texts,
            dialogs=dialogs,
            filenames=filenames_out,
            scenarios=scenarios_out,
            overlay_imgs=imgs_out,
        )
        print(f"[Saved] Multi-turn results to {AUGMENTED_NPZ_PATH}")
    else:
        print("[Warning] No dialogs were generated")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

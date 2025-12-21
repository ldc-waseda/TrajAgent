import os
import asyncio
import argparse
import base64
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from openai import AsyncOpenAI
from pathlib import Path
from agent.utils.openai_wrapper import create_wrapped_client


# --- Constants ---
# IMPORTANT: Replace with your actual OpenAI API key
API_KEY =  os.environ["OPENAI_API_KEY"] 


# --- Helper Functions ---

def image_to_base64_url(image_path: Path) -> str:
    """Reads an image and converts it to a base64 data URL."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None or not mime_type.startswith("image"):
        raise ValueError(f"Unsupported file type: {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"


# --- Main Logic ---

async def run_single_case(
    client: AsyncOpenAI,
    prompt_text: str,
    image_paths: List[Path],
    model: str,
    temperature: float,
    max_new_tokens: int,
) -> Optional[str]:
    """
    Runs a single inference case with multiple images and a text prompt.

    Args:
        client: The AsyncOpenAI client.
        prompt_text: The text prompt.
        image_paths: A list of paths to image files.
        model: The model name to use.
        temperature: The generation temperature.
        max_new_tokens: The maximum number of tokens to generate.

    Returns:
        The content of the assistant's response, or None on failure.
    """
    print("[Request] Preparing API request...")
    
    # 1. Construct the message content with text and images
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    
    for img_path in image_paths:
        try:
            base64_url = image_to_base64_url(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": base64_url},
            })
            print(f"[Request] Added image: {img_path.name}")
        except (FileNotFoundError, ValueError) as e:
            print(f"[Error] Skipping image {img_path}: {e}")
            continue

    if len(content) == 1: # Only text, no valid images
        print("[Error] No valid images were provided or loaded. Aborting.")
        return None

    # 2. Create the full message payload
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # 3. Make the API call
    try:
        print(f"[Request] Sending request to model '{model}'...")
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_new_tokens,
        )
        
        assistant_response = response.choices[0].message.content
        print("[Response] Received response successfully.")
        return assistant_response

    except Exception as e:
        print(f"[Error] API call failed: {e}")
        return None


async def main_async(args: argparse.Namespace):
    """Main asynchronous entry point."""
    
    # 1. Read prompt from file
    try:
        prompt_text = args.prompt_file.read_text(encoding="utf-8")
        print(f"[Input] Loaded prompt from: {args.prompt_file.name}")
    except Exception as e:
        print(f"[Error] Failed to read prompt file: {e}")
        return

    # 2. Create OpenAI client
    # We use the wrapper for potential future caching, but mode is 'realtime'
    client = create_wrapped_client(
        api_key=API_KEY,  # Use the hardcoded API key
        base_url=args.api_base,
        model=args.model,
        strategy="run_case_single", # A unique name for this strategy
        mode="realtime", # Force realtime mode
    )

    # 3. Run the generation
    result = await run_single_case(
        client=client,
        prompt_text=prompt_text,
        image_paths=args.images,
        model=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # 4. Print and save the result
    if result:
        print("\n--- Assistant Response ---")
        print(result)
        print("--------------------------\n")

        if args.output_file:
            try:
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                args.output_file.write_text(result, encoding="utf-8")
                print(f"[Output] Saved response to: {args.output_file}")
            except Exception as e:
                print(f"[Error] Failed to save output file: {e}")
    else:
        print("[Info] No response was generated.")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run a single inference case with a prompt and multiple images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Inputs
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="Path to the markdown file containing the prompt.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        nargs="+",
        help="Paths to one or more image files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional path to save the assistant's response.",
    )

    # API and Model Parameters
    parser.add_argument("--model", type=str, default=os.environ.get("VL_MODEL", "gpt-4o"), help="Model name to use.")
    parser.add_argument("--api-base", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="API base URL.")
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "1")), help="Generation temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "4096")), help="Maximum number of new tokens to generate.")

    args = parser.parse_args()
    
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("[Error] Please replace 'YOUR_API_KEY_HERE' with your actual OpenAI API key in the script.")
        return

    asyncio.run(main_async(args))


if __name__ == "__main__":
    import mimetypes
    main()

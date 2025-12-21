import json
from typing import Any, Dict, Optional


def extract_text_from_chat_choice(choice: Any) -> str:
    try:
        return choice["message"]["content"]
    except Exception:
        pass
    try:
        return choice.message.content
    except Exception:
        return ""


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        try:
            first = s.find("\n")
            last = s.rfind("```")
            if first != -1 and last != -1:
                s = s[first + 1:last]
        except Exception:
            pass
    if s.lower().startswith("json"):
        s = s[4:].strip()
    try:
        parsed: Any = json.loads(s)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None

import os
from typing import Any, Dict, Optional


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def load_template_file(template_name: str) -> Optional[str]:
    path = resolve_default_template_path(template_name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_template(template_name: str, variables: Dict[str, Any]) -> str:
    template = load_template_file(template_name)
    return template.format_map(_SafeDict(variables))


def resolve_default_template_path(template_name: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.normpath(os.path.join(base_dir, "prompts", f"{template_name}.md"))
    return path

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "prompts" / "base_prompt.md"
FRAGMENTS = ROOT / "prompts" / "fragments" / "rag_fragments.yaml"
CONDITIONS = ("v2", "v3", "v3_flat")


def build_prompt(version: str) -> str:
    if version not in CONDITIONS:
        raise ValueError(f"unknown condition {version!r}; expected one of {CONDITIONS}")
    text = BASE.read_text(encoding="utf-8")
    fragments = yaml.safe_load(FRAGMENTS.read_text(encoding="utf-8"))[version]
    for key, value in fragments.items():
        text = text.replace("{{%s}}" % key, value or "")
    if "{{RAG_" in text:
        raise RuntimeError(f"unfilled placeholder remains in the {version} prompt")
    return text

#!/usr/bin/env python3
"""Dump the three assembled prompts and diff v2 against v3.

Run before any generation batch: the diff is the complete, auditable statement
of how the conditions differ. If it shows anything beyond the RAG fragments,
the controlled contrast is broken.
"""
import difflib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_builder import CONDITIONS, build_prompt

out = Path("rendered_prompts"); out.mkdir(exist_ok=True)
p = {v: build_prompt(v) for v in CONDITIONS}
for v, text in p.items():
    (out / f"prompt_{v}.md").write_text(text, encoding="utf-8")
    print(f"prompt_{v}.md  {len(text)} chars")

diff = list(difflib.unified_diff(p["v2"].splitlines(), p["v3"].splitlines(),
                                 "v2", "v3", lineterm="", n=1))
(out / "diff_v2_v3.txt").write_text("\n".join(diff), encoding="utf-8")
print(f"\ndiff_v2_v3.txt  {sum(1 for d in diff if d.startswith(('+','-')) and not d.startswith(('+++','---')))} changed lines")

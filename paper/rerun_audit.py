"""Re-run only the claim audit of a saved case-study run (same prompt and paragraph).

    python paper/rerun_audit.py paper/outputs/<run_dir>
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xascribe import report  # noqa: E402

run = Path(sys.argv[1])
os.environ.setdefault("XASCRIBE_LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free,qwen/qwen3.8-27b:free")
os.environ.setdefault("XASCRIBE_LLM_RETRIES", "4")
prompt = (run / "prompt.txt").read_text(encoding="utf-8")
paragraph = (run / "paragraph.txt").read_text(encoding="utf-8")
claims, model, raw = report.audit(paragraph, prompt)
(run / "audit_v2.json").write_text(json.dumps(claims, indent=2, ensure_ascii=False), encoding="utf-8")
(run / "audit_v2_raw.txt").write_text(raw, encoding="utf-8")
info = json.loads((run / "run.json").read_text(encoding="utf-8"))
info.update(auditor_v2=model, audit_v2_summary=report.summarise_audit(claims))
(run / "run_v2.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
print(model, report.summarise_audit(claims))
for c in claims:
    print("-", c.get("type"), "|", c.get("verdict"), "|", str(c.get("claim"))[:120])

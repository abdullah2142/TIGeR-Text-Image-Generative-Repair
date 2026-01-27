# scripts/vlm_client.py
import base64
import json
import os
import re
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llava")

SYSTEM_RULES = """
You are a data repair assistant for an e-commerce dataset.

STRICT RULES:
- Output MUST be valid JSON only. No markdown, no extra text.
- Choose ONLY ONE field to edit.
- field must be one of editable_fields.
- old_value must match current attributes[field] (case-insensitive).
- confidence must be between 0 and 1.
- If unsure, set confidence < 0.5.

Output JSON format:
{
  "field": "color",
  "old_value": "red",
  "new_value": "blue",
  "confidence": 0.82,
  "evidence": "short visual + diagnostic reason"
}
""".strip()

def _b64_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _extract_json(text: str) -> dict:
    """
    Tries strict json.loads first.
    If that fails, extracts the first {...} block and tries again.
    """
    t = (text or "").strip()

    # 1) strict parse
    try:
        return json.loads(t)
    except Exception:
        pass

    # 2) extract first JSON object block
    # find first "{" and last "}" after it
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        candidate = m.group(0).strip()
        return json.loads(candidate)

    # 3) give up
    raise json.JSONDecodeError("No JSON object found", t, 0)

def vlm_suggest(payload: dict) -> dict:
    """
    Calls local Ollama VLM (LLaVA) with image + structured prompt.
    Returns dict with keys: field, old_value, new_value, confidence, evidence
    Safe fallback on any failure.
    """
    try:
        img_path = payload.get("image_path") or ""
        images = []
        if img_path and os.path.exists(img_path):
            images = [_b64_image(img_path)]

        prompt_obj = {
            "row_id": payload.get("row_id"),
            "attributes": payload.get("attributes", {}),
            "flag_reason": payload.get("flag_reason", ""),
            "diagnostics": payload.get("diagnostics", {}),
            "editable_fields": payload.get("editable_fields", []),
        }

        prompt = (
            SYSTEM_RULES
            + "\n\nIMPORTANT: Respond with ONLY the JSON object. No extra words.\n\nINPUT:\n"
            + json.dumps(prompt_obj, ensure_ascii=False)
        )

        req = {
            "model": MODEL,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "options": {"temperature": 0.2},
            # If your Ollama supports it, this helps a lot:
            "format": "json",
        }

        r = requests.post(f"{OLLAMA_URL}/api/generate", json=req, timeout=180)
        r.raise_for_status()
        out = (r.json().get("response") or "").strip()

        sug = _extract_json(out)

        # normalize keys (just in case model returns weird types)
        sug.setdefault("field", "")
        sug.setdefault("old_value", "")
        sug.setdefault("new_value", "")
        sug.setdefault("confidence", 0.0)
        sug.setdefault("evidence", "")

        return sug

    except Exception as e:
        return {
            "field": "",
            "old_value": "",
            "new_value": "",
            "confidence": 0.0,
            "evidence": f"vlm_error:{type(e).__name__}"
        }

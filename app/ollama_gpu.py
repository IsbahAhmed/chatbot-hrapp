"""Helpers to verify Ollama is using GPU (AMD on Windows: Vulkan / ROCm via Ollama, not this app)."""

import os
import sys

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def check_ollama_connection(model: str) -> dict:
  """Return status dict: connected, models, running_models, hints."""
  out = {
    "connected": False,
    "models": [],
    "running": [],
    "hints": [],
  }

  try:
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    if tags.status_code != 200:
      out["hints"].append(f"Ollama /api/tags returned {tags.status_code}")
      return out
    out["connected"] = True
    out["models"] = [m.get("name") for m in tags.json().get("models", [])]
  except requests.RequestException as e:
    out["hints"].append(f"Cannot reach Ollama at {OLLAMA_URL}: {e}")
    return out

  if model and model not in out["models"] and not any(
    (m or "").startswith(model.split(":")[0]) for m in out["models"]
  ):
    out["hints"].append(f"Model '{model}' not pulled. Run: ollama pull {model}")

  try:
    ps = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10)
    if ps.status_code == 200:
      for entry in ps.json().get("models", []):
        name = entry.get("name", "?")
        processor = entry.get("processor", entry.get("size_vram", "unknown"))
        out["running"].append(f"{name} ({processor})")
  except requests.RequestException:
    pass

  if sys.platform == "win32":
    vulkan = os.getenv("OLLAMA_VULKAN", "").lower() in ("1", "true", "yes")
    override = os.getenv("OLLAMA_GPU_OVERIDE", "")
    if not vulkan and override.lower() != "vulkan":
      out["hints"].append(
        "AMD on Windows: quit Ollama, set OLLAMA_VULKAN=1 (or OLLAMA_GPU_OVERIDE=vulkan), "
        "then restart Ollama. See scripts/start-ollama-amd.ps1"
      )

  return out


def log_ollama_status(model: str) -> None:
  status = check_ollama_connection(model)
  if not status["connected"]:
    for h in status["hints"]:
      print(f"Ollama: {h}")
    print("Start Ollama, then: ollama pull", model)
    return

  print(f"Ollama connected at {OLLAMA_URL}")
  print(f"Available models: {status['models']}")
  if status["running"]:
    print(f"Loaded models: {', '.join(status['running'])}")
  for h in status["hints"]:
    print(f"Ollama hint: {h}")

# handler.py
import base64
import glob
import os
import shlex
import subprocess
import uuid
from pathlib import Path

import runpod

# ---- Volume + model locations (weights MUST live on Network Volume) ----
VOLUME_ROOT = os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")

# Directory on the Network Volume that contains:
#   hunyuan-video-t2v-720p/, vae/, text_encoder/, text_encoder_2/, ...
MODEL_BASE = os.getenv("HUNYUAN_MODEL_BASE", f"{VOLUME_ROOT}/ckpts")

# Default DiT checkpoint (adjust via env var if your layout differs)
DIT_WEIGHT = os.getenv(
    "HUNYUAN_DIT_WEIGHT",
    f"{MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
)

# ---- Code locations inside the container ----
WORKDIR = os.getenv("HUNYUAN_WORKDIR", "/workspace")
SCRIPT_NAME = os.getenv("HUNYUAN_SCRIPT", "sample_video.py")

# ---- Output ----
OUTPUT_ROOT = os.getenv("HUNYUAN_OUTPUT_ROOT", "/tmp/hunyuan_results")

def _ls(p):
    try:
        return sorted(os.listdir(p))[:200]
    except Exception as e:
        return [f"<err: {e}>"]

print("CWD:", os.getcwd())
print("LS /:", _ls("/"))
print("LS /workspace:", _ls("/workspace"))
print("LS /workspace (recursive shallow):")
try:
    base = Path("/workspace")
    hits = list(base.rglob("sample_video.py"))
    print("FOUND sample_video.py:", [str(h) for h in hits[:20]])
except Exception as e:
    print("rglob err:", e)
    
def _coerce_video_size(v):
    """
    Accept:
      - [720, 1280]
      - "720 1280"
      - "720x1280"
      - 720  (single int -> square)
    Returns: (h, w)
    """
    if v is None:
        return (720, 1280)

    if isinstance(v, int):
        return (v, v)

    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return (int(v[0]), int(v[0]))
        if len(v) >= 2:
            return (int(v[0]), int(v[1]))
        raise ValueError("video_size list/tuple must have 1 or 2 ints")

    if isinstance(v, str):
        s = v.strip().lower().replace("×", "x")
        if "x" in s:
            a, b = s.split("x", 1)
            return (int(a.strip()), int(b.strip()))
        parts = s.split()
        if len(parts) == 1:
            n = int(parts[0])
            return (n, n)
        if len(parts) >= 2:
            return (int(parts[0]), int(parts[1]))
        raise ValueError("video_size string must be like '720 1280' or '720x1280'")

    raise TypeError("video_size must be int, list/tuple, or string")


def _find_script():
    # 1) Respect explicit env
    p = Path(WORKDIR) / SCRIPT_NAME
    if p.exists():
        return p

    # 2) Common repo locations in different images
    candidates = [
        Path("/HunyuanVideo") / SCRIPT_NAME,
        Path("/workspace/HunyuanVideo") / SCRIPT_NAME,
        Path("/opt/HunyuanVideo") / SCRIPT_NAME,
        Path("/root/HunyuanVideo") / SCRIPT_NAME,
        Path.cwd() / SCRIPT_NAME,
    ]
    for c in candidates:
        if c.exists():
            return c

    # 3) Shallow search (avoid expensive full FS scan)
    for base in ["/workspace", "/HunyuanVideo", "/opt", "/root"]:
        try:
            root = Path(base)
            if root.exists():
                hits = list(root.rglob(SCRIPT_NAME))
                if hits:
                    return hits[0]
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not find {SCRIPT_NAME}. Set HUNYUAN_WORKDIR or HUNYUAN_SCRIPT."
    )


def _latest_mp4(out_dir: Path) -> Path:
    mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        mp4s = sorted(
            (Path(p) for p in glob.glob(str(out_dir / "**" / "*.mp4"), recursive=True)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not mp4s:
        raise FileNotFoundError(f"No .mp4 generated under {out_dir}")
    return mp4s[0]


def handler(event):
    job_input = (event or {}).get("input") or {}

    prompt = job_input.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return {"error": "Missing required string: input.prompt"}

    h, w = _coerce_video_size(job_input.get("video_size"))
    video_length = int(job_input.get("video_length", 129))
    infer_steps = int(job_input.get("infer_steps", 50))

    # Optional knobs (pass-through when present)
    seed = job_input.get("seed", None)
    neg_prompt = job_input.get("negative_prompt", None)
    cfg_scale = float(job_input.get("cfg_scale", 1.0))
    embedded_cfg_scale = float(job_input.get("embedded_cfg_scale", 6.0))

    # Model selection (depends on your Hunyuan repo/script)
    model = job_input.get("model", "HYVideo-T/2-cfgdistill")
    model_resolution = job_input.get("model_resolution", "720p")

    # Output format
    return_base64 = bool(job_input.get("return_base64", False))

    # --- Preflight: verify Network Volume + weights exist ---
    vol = Path(VOLUME_ROOT)
    base = Path(MODEL_BASE)
    dit = Path(DIT_WEIGHT)

    if not vol.exists():
        return {"error": f"Network Volume not mounted at {VOLUME_ROOT}"}
    if not base.exists():
        return {"error": f"Model base missing: {MODEL_BASE} (check HUNYUAN_MODEL_BASE)"}
    if not dit.exists():
        return {"error": f"DiT weight missing: {DIT_WEIGHT} (check HUNYUAN_DIT_WEIGHT)"}

    # Locate sample_video.py inside the container
    script_path = _find_script()
    script_dir = script_path.parent

    # Per-job output dir (ephemeral by default)
    run_id = (event or {}).get("id") or str(uuid.uuid4())
    out_dir = Path(OUTPUT_ROOT) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build the CLI call (weights explicitly from Network Volume) ---
    cmd = [
        "python3",
        str(script_path),
        "--model",
        str(model),
        "--model-resolution",
        str(model_resolution),
        "--model-base",
        str(base),
        "--dit-weight",
        str(dit),
        "--video-size",
        str(h),
        str(w),
        "--video-length",
        str(video_length),
        "--infer-steps",
        str(infer_steps),
        "--prompt",
        prompt,
        "--cfg-scale",
        str(cfg_scale),
        "--embedded-cfg-scale",
        str(embedded_cfg_scale),
        "--save-path",
        str(out_dir),
    ]

    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if neg_prompt:
        cmd += ["--neg-prompt", str(neg_prompt)]

    # --- Run inference ---
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(script_dir),  # script-relative imports still work
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return {
            "error": "HunyuanVideo inference failed",
            "exit_code": e.returncode,
            "cmd": " ".join(shlex.quote(x) for x in cmd),
            "log_tail": (e.stdout or "")[-8000:],
        }

    # Find resulting mp4
    try:
        mp4_path = _latest_mp4(out_dir)
    except Exception as e:
        return {
            "error": f"Inference completed but no mp4 found: {e}",
            "output_dir": str(out_dir),
            "log_tail": (proc.stdout or "")[-8000:],
        }

    # Return base64 or path (quality is identical; only transport differs)
    if return_base64:
        data = mp4_path.read_bytes()
        return {
            "video_base64": base64.b64encode(data).decode("utf-8"),
            "filename": mp4_path.name,
            "mime_type": "video/mp4",
            "log_tail": (proc.stdout or "")[-2000:],
        }

    return {
        "video_path": str(mp4_path),
        "output_dir": str(out_dir),
        "log_tail": (proc.stdout or "")[-2000:],
    }


runpod.serverless.start({"handler": handler})

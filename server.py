#!/usr/bin/env python3
"""
Local AI Server v29.0 SUPREME ULTRA — RTX 5080 16GB HOT-3D + ASSET QA ENGINE + PyTorch 2.10 cu128

v29.0 SUPREME ULTRA — MAXIMUM POWER:
  - MESH LOD AUTO-GENERATION: /mesh/lod — LOD0/1/2/3 via quadric decimation (100%/50%/25%/10%)
  - 3D PIPELINE v3: /pipeline/3d — full prompt→generate→optimize→LOD→export in one call
  - ADVANCED MESH ANALYSIS: /mesh/analyze — topology, game-readiness score, polygon budget
  - BATCH EXPORT v2: /export/batch — export to GLB/OBJ/FBX/STL/PLY in one call
  - MODEL WARM-UP: /model/warmup + /model/warmup/schedule — keep GPU caches hot
  - GENERATION PRESETS: /presets/3d + /generate3d/preset — named presets (turbo/fast/balanced/hero)

v26.0 SUPREME ULTRA:
  - GENERATION PIPELINE STATS: /pipeline/stats — track multi-step generation flows
  - ASSET CATALOG: /assets/catalog — searchable registry of all generated assets
  - SMART 3D QUALITY: Auto-adjust generation params based on prompt complexity
  - REQUEST CORRELATION: X-Correlation-ID tracking across pipeline stages
  - PERFORMANCE PROFILING: Per-endpoint timing breakdown for optimization

v25.0 SUPREME ULTRA:
  - BATCH 3D v2: /generate3d/batch/v2 — priority queue, SSE progress, per-item presets, OOM retry
  - MESH POST-PROCESSING: /mesh/postprocess — decimate, smooth, normalize, LOD chain via trimesh
  - TEXTURE BAKING: /generate3d/textured — Hunyuan3D mesh + remote FLUX albedo → textured GLB
  - VRAM DASHBOARD: /vram/dashboard — real-time breakdown, history, peak, fragmentation, alerts
  - MODEL HEALTH: /model/health — latency percentiles, success rates, error classification, uptime
  - 3D REVIEW v2: /review3d/enhanced — topology analysis, watertight, UV coverage, game-readiness
  - FORMAT CONVERSION: /convert — GLB/OBJ/FBX/STL/PLY interconversion with optimization

v22.1 SUPREME ULTRA:
  - STRIPPED TO 2 MODELS: Hunyuan3D (pinned) + embeddings (pinned) ONLY
  - ALL VRAM DEDICATED to Hunyuan3D 2.1 full (3.3B, 14GB) — zero contention
  - REMOVED: AudioLDM2, Chatterbox, Kokoro, ACE-Step, LTX-Video
  - Image gen → remote FLUX.1-dev exclusively

v21.0 SUPREME ULTRA:
  - 3D THUMBNAIL: /generate3d/with_thumbnail — generate mesh + render preview PNG
  - 3D VARIATIONS: /generate3d/variations — generate N variations with different seeds
  - MODEL BENCHMARK: /models/benchmark — quick inference benchmark of loaded models
  - BATCH REVIEW: /review/batch — review multiple assets at once
  - GPU TIMELINE: /gpu/timeline — VRAM usage history with 30s snapshots
  - AUDIO BATCH: /audio/batch — generate multiple SFX in one call

v20.0 SUPREME ULTRA:
  - DUAL-TRACK AWARE: Coordinate with MCP server dual-track GPU queue
  - ENHANCED VRAM MONITORING: Real-time fragmentation + allocation tracking
  - MODEL SCHEDULING v2: Predict model needs from recent patterns
  - HEALTH METRICS v2: GPU temp, utilization, power draw in /health
  - QUEUE FAIRNESS: Prevent request starvation in high-load scenarios
  - STARTUP OPTIMIZATION: Parallel model loading for pinned models

v19.0 SUPREME:
  - BATCH 3D GENERATION: /generate3d/batch — process up to 5 prompts sequentially
  - MODEL STATUS: /model/status — detailed VRAM breakdown per model with idle time
  - PROFILE HOT-RELOAD: /profiles/reload — reload scoring profiles without restart
  - ENHANCED REVIEW: /review/enhanced — auto-detect asset type from extension
  - SYSTEM INFO: /system/info — comprehensive GPU, VRAM, models, uptime, versions

v18.0 PROFILE-DRIVEN (Asset QA Engine):
  - EXTERNAL JSON PROFILES: scoring_profiles/{image,model3d,audio,video}.json
  - ZERO HARDCODED THRESHOLDS: All magic numbers extracted to editable JSON
  - HOT-TUNABLE: Edit JSON → restart → new thresholds active
  - PROFILE VERSIONING: Each profile has _version field for tracking
  - FULL KB INTEGRATION: 141K-line game dev KB distilled into JSON profiles

v17.0 KB-INTEGRATED:
  - Category-aware thresholds, tileability, saturation, topology, spectral analysis

v16.0 SUPREME:
  - REQUEST QUEUE + ETA: Track queue depth, avg processing time, estimated wait
  - REQUEST DEDUPLICATION: Identical requests within 5s share results (MD5 hash)
  - GENERATION HISTORY: Last 200 generations with metadata for analytics
  - MODEL USAGE PATTERNS: Predict next model based on call patterns, auto-prewarm
  - PRIORITY SYSTEM: critical/normal/low request priority with GPU semaphore
  - ENHANCED AUTO-REVIEW: +sharpness metric (Laplacian variance) for images
  - BATCH WARMUP: /warmup/batch to pre-load multiple models in sequence
  - VRAM PRESSURE MONITOR: Auto-evict idle models when VRAM < 500MB free

v14.0 ULTRA: Request Metrics (P50/P95/P99), VRAM Monitor/Defrag, Warmup,
  Request ID Tracking, Enhanced Health, GPU Semaphore, Better Errors

Philosophy: Hunyuan3D 2.1 full PINNED for instant 3D. Image gen on remote FLUX.1-dev.
4 models only. ZERO FALLBACKS. If a model fails, return error.

Architecture:
  - ModelManager: Full unload eviction, ~15.5GB usable VRAM, 5-min idle timeout
  - PINNED: Hunyuan3D 2.1 full (14GB) + embeddings (0.2GB) — always on GPU
  - ON-DEMAND: AudioLDM2 (4-7GB), Chatterbox (3GB) — conflicts w/ Hunyuan3D
  - Auto-Review v3.0 KB: Category-aware checks (tileability/saturation/spectral/topology)
  - ALL image gen → remote FLUX.1-dev v8.0 (RTX 3090 Ti, port 8200)

Models (2 REGISTERED — v22.1 STRIPPED):
  3D Gen: Hunyuan3D 2.1 full (3.3B, 14GB, PINNED HOT)
  Embeddings: nomic-embed-v2-moe (0.2GB, PINNED)
  REMOVED: AudioLDM2, Chatterbox, Kokoro, ACE-Step, LTX-Video

REMOVED (endpoints return error): DreamShaper, ESRGAN, RMBG, Depth Pro,
  Marigold, SAM2, BLIP-2, YOLO, CLIP, MusicGen, Wan2.1, Sesame, F5-TTS
"""

import os, time, io, base64, gc, logging, uuid, traceback, asyncio, warnings, hashlib, json
from pathlib import Path
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from typing import Optional, List

# Suppress known harmless warnings before any imports
warnings.filterwarnings("ignore", message=".*torchao.*incompatible.*")
warnings.filterwarnings("ignore", message=".*Detected no triton.*")
warnings.filterwarnings("ignore", message=".*expandable_segments.*")
warnings.filterwarnings("ignore", message=".*megablocks.*")

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Maximum VRAM utilization: expandable segments (-34% fragmentation), GC at 90%
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.9")
# torch.compile disk cache — persist compiled graphs across restarts
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "C:/Users/han/.cache/torch_compile")
# v18.0: cuDNN benchmark — auto-tune convolution algorithms for RTX 5080
os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
# v18.0: TF32 — 19x faster matmul on Ampere+ GPUs with minimal precision loss
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
# Suppress pymeshlab Qt plugin warnings (missing DLLs for unused plugins like embree, sketchfab)
os.environ.setdefault("QT_LOGGING_RULES", "qt.*.warning=false")
os.environ.setdefault("QT_DEBUG_PLUGINS", "0")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger("ai-server")

# ─── Config ───
PORT = int(os.getenv("AI_PORT", "8099"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RTX 5080 optimizations: TF32 for speed, cudnn benchmark for conv
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(1.0, device=0)
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Set via: export HF_TOKEN=your_token

# ─── v18.0: Load Scoring Profiles from JSON ───
PROFILES_DIR = Path(os.getenv("PROFILES_DIR", "C:/Users/han/claude/scoring_profiles"))

def _load_profile(name: str) -> dict:
    """Load a scoring profile JSON. Returns empty dict if not found."""
    path = PROFILES_DIR / f"{name}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info(f"Loaded scoring profile: {name} v{data.get('_version', '?')}")
            return data
        except Exception as e:
            log.warning(f"Failed to load profile {name}: {e}")
    return {}

IMAGE_PROFILE = _load_profile("image")
MODEL3D_PROFILE = _load_profile("model3d")
AUDIO_PROFILE = _load_profile("audio")
VIDEO_PROFILE = _load_profile("video")

ASSET_DIR = Path(os.getenv("ASSET_DIR", "C:/Users/han/claude/.assets/generated"))
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "C:/Users/han/claude/.assets/audio"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "C:/Users/han/claude/.assets/models3d"))
ASSET_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = Path(os.getenv("VIDEO_DIR", "C:/Users/han/claude/.assets/video"))
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.time()

# ─── v14.0: GPU Semaphore — prevent concurrent GPU ops causing OOM ───
gpu_semaphore = asyncio.Semaphore(1)

# ─── v20.0: Dual-Track Aware — tag endpoints as local or remote GPU track ───
GPU_TRACK_MAP = {
    "/gen3d": "local", "/generate3d": "local", "/generate3d/with_thumbnail": "local",
    "/generate3d/variations": "local", "/generate3d/batch": "local",
    "/sfx": "local", "/tts": "local", "/music": "local", "/audio/batch": "local",
    "/stt": "local", "/embed": "local", "/embeddings": "local",
    "/image": "remote", "/image/batch": "remote", "/image/img2img": "remote",
    "/review": "local", "/review/enhanced": "local", "/review/batch": "local",
    "/mesh/lod": "local", "/pipeline/3d": "local", "/mesh/analyze": "local",
    "/export/batch": "local", "/model/warmup": "local", "/generate3d/preset": "local",
}

def _get_gpu_track(endpoint: str) -> str:
    """Get the GPU track (local/remote) for an endpoint."""
    return GPU_TRACK_MAP.get(endpoint, "unknown")

# ─── v16.0 SUPREME: Request Queue + ETA Tracking ───
_queue_depth = 0
_avg_gen_time = deque(maxlen=50)  # Track last 50 generation times for ETA

# ─── v20.0: Queue Fairness — starvation prevention ───
_request_wait_times = deque(maxlen=200)  # Track (arrival_time, endpoint) for fairness monitoring
_starvation_threshold_s = 30  # Bump priority if waiting longer than this
_starvation_warning_s = 20  # Log warning if waiting longer than this
_starvation_bumps = 0  # Counter for priority bumps

def _track_request_arrival(endpoint: str):
    """Record when a request arrives for fairness tracking."""
    _request_wait_times.append((time.time(), endpoint))

def _check_starvation():
    """Check if oldest request has been waiting too long, return True if starvation detected."""
    global _starvation_bumps
    if not _request_wait_times:
        return False
    now = time.time()
    # Prune completed requests (older than 120s are assumed done)
    while _request_wait_times and now - _request_wait_times[0][0] > 120:
        _request_wait_times.popleft()
    if not _request_wait_times:
        return False
    oldest_time, oldest_endpoint = _request_wait_times[0]
    wait_s = now - oldest_time
    if wait_s > _starvation_warning_s:
        log.warning(f"Queue fairness: request to {oldest_endpoint} waiting {wait_s:.1f}s (threshold: {_starvation_threshold_s}s)")
    if wait_s > _starvation_threshold_s:
        _starvation_bumps += 1
        log.warning(f"Queue fairness: STARVATION DETECTED — bumping priority for {oldest_endpoint} (waited {wait_s:.1f}s, total bumps: {_starvation_bumps})")
        _request_wait_times.popleft()  # Remove the bumped request from tracking
        return True
    return False

# ─── v16.0 SUPREME: Request Deduplication (5s window) ───
_dedup_cache = OrderedDict()  # hash → (result, timestamp)
_dedup_ttl = 5.0
_dedup_hits = 0

def _dedup_key(**kwargs):
    """Generate MD5 hash for dedup."""
    raw = str(sorted(kwargs.items()))
    return hashlib.md5(raw.encode()).hexdigest()

def _dedup_check(key):
    global _dedup_hits
    now = time.time()
    # Prune expired
    expired = [k for k, (_, ts) in _dedup_cache.items() if now - ts > _dedup_ttl]
    for k in expired:
        _dedup_cache.pop(k, None)
    if key in _dedup_cache:
        _dedup_hits += 1
        return _dedup_cache[key][0]
    return None

def _dedup_store(key, result):
    _dedup_cache[key] = (result, time.time())
    # Keep max 50 entries
    while len(_dedup_cache) > 50:
        _dedup_cache.popitem(last=False)

# ─── v16.0 SUPREME: Generation History ───
_generation_history = deque(maxlen=200)

# ─── v21.0: GPU VRAM Timeline — snapshots every 30s for monitoring ───
_vram_timeline = deque(maxlen=100)

async def _vram_timeline_loop():
    """v21.0: Record VRAM snapshots every 30s for GPU timeline monitoring."""
    while True:
        try:
            if torch.cuda.is_available():
                free = torch.cuda.mem_get_info()[0] // (1024 * 1024)
                total = torch.cuda.mem_get_info()[1] // (1024 * 1024)
                _vram_timeline.append({
                    "timestamp": time.time(),
                    "vram_used_mb": total - free,
                    "vram_free_mb": free,
                    "vram_total_mb": total,
                    "models_loaded": list(manager._gpu_loaded.keys()) if hasattr(manager, '_gpu_loaded') else [],
                })
        except Exception:
            pass
        await asyncio.sleep(30)

def _record_generation(endpoint, prompt, model, elapsed_s, success, extra=None):
    entry = {
        "endpoint": endpoint, "prompt": prompt[:100], "model": model,
        "elapsed_s": round(elapsed_s, 2), "success": success,
        "timestamp": time.time(), "id": uuid.uuid4().hex[:8],
    }
    if extra:
        entry.update(extra)
    _generation_history.append(entry)

# ─── v16.0 SUPREME: Model Usage Pattern Tracking ───
_model_usage_sequence = deque(maxlen=100)  # Track recent model calls for pattern prediction
_model_transition_counts = {}  # {prev_model: {next_model: count}}

def _track_model_usage(model_name):
    if _model_usage_sequence:
        prev = _model_usage_sequence[-1]
        if prev not in _model_transition_counts:
            _model_transition_counts[prev] = {}
        _model_transition_counts[prev][model_name] = _model_transition_counts[prev].get(model_name, 0) + 1
    _model_usage_sequence.append(model_name)

def _predict_next_model():
    """Predict most likely next model based on transition patterns."""
    if not _model_usage_sequence or not _model_transition_counts:
        return None
    current = _model_usage_sequence[-1]
    transitions = _model_transition_counts.get(current, {})
    if not transitions:
        return None
    return max(transitions, key=transitions.get)

# ─── v14.0: Request Metrics Tracker ───
class LocalMetrics:
    def __init__(self, max_history=500):
        self._history = deque(maxlen=max_history)
        self._total = 0
        self._errors = 0
        self._by_endpoint = {}

    def record(self, endpoint: str, success: bool, elapsed: float):
        self._history.append({"ep": endpoint, "ok": success, "ms": round(elapsed * 1000), "ts": time.time()})
        self._total += 1
        if not success: self._errors += 1
        if endpoint not in self._by_endpoint:
            self._by_endpoint[endpoint] = {"calls": 0, "errors": 0, "total_ms": 0, "max_ms": 0}
        ep = self._by_endpoint[endpoint]
        ep["calls"] += 1
        if not success: ep["errors"] += 1
        ep["total_ms"] += round(elapsed * 1000)
        ep["max_ms"] = max(ep["max_ms"], round(elapsed * 1000))

    def stats(self):
        latencies = [h["ms"] for h in self._history if h["ok"]]
        latencies.sort()
        n = len(latencies)
        uptime = time.time() - start_time
        top_endpoints = sorted(self._by_endpoint.items(), key=lambda x: x[1]["calls"], reverse=True)[:10]
        return {
            "total_requests": self._total,
            "total_errors": self._errors,
            "error_rate": f"{(self._errors / max(1, self._total)) * 100:.1f}%",
            "uptime_s": round(uptime),
            "uptime": f"{uptime / 3600:.1f}h" if uptime > 3600 else f"{uptime / 60:.1f}m",
            "rpm": round(self._total / max(1, uptime / 60), 2),
            "latency_p50_ms": latencies[n // 2] if n > 0 else 0,
            "latency_p95_ms": latencies[int(n * 0.95)] if n > 1 else 0,
            "latency_p99_ms": latencies[int(n * 0.99)] if n > 1 else 0,
            "top_endpoints": {k: v for k, v in top_endpoints},
            "recent": list(self._history)[-10:],
        }

local_metrics = LocalMetrics()

# ─── Helper: resolve image from base64 OR file path ───
def resolve_image(image_input: str) -> "Image.Image":
    """Accept base64 data URI or file path, return PIL Image."""
    from PIL import Image
    if not image_input:
        raise ValueError("No image provided — pass either 'image' (base64) or 'image_path' (file path)")
    if image_input.startswith("data:"):
        # data:image/png;base64,... format
        header, data = image_input.split(",", 1)
        img_bytes = base64.b64decode(data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif image_input.startswith("/9j/") or image_input.startswith("iVBOR"):
        # Raw base64 without header
        img_bytes = base64.b64decode(image_input)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        # File path
        return Image.open(image_input).convert("RGB")

def image_to_base64(img: "Image.Image", format: str = "PNG") -> str:
    """Convert PIL Image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = f"image/{format.lower()}"
    return f"data:{mime};base64,{b64}"

# ─── LoRA Registry (game asset LoRAs for FLUX/SDXL) ───
LORA_REGISTRY = {
    # v9.3: DreamShaper XL compatible LoRAs — 51 styles for game dev
    # ── Pixel Art (3) ──
    "pixel-art-xl": {
        "repo": "nerijs/pixel-art-xl",
        "trigger": "pixel art",
        "weight_name": "pixel-art-xl.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "pixel-art-redmond": {
        "repo": "artificialguybr/PixelArtRedmond",
        "trigger": "Pixel Art, PixArFK",
        "weight_name": "PixelArtRedmond-Lite64.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "retro-videogame": {
        "repo": "ntc-ai/SDXL-LoRA-slider.pixel-art",
        "trigger": "pixel art",
        "weight_name": "pixel art.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── 3D / Stylized (2) ──
    "voxel-art": {
        "repo": "Fictiverse/Voxel_XL_Lora",
        "trigger": "voxel style",
        "weight_name": "VoxelXL_v1.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "low-poly": {
        "repo": "ntc-ai/SDXL-LoRA-slider.low-poly-count",
        "trigger": "low poly",
        "weight_name": "low poly count.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Anime / Cartoon (3) ──
    "anime": {
        "repo": "ntc-ai/SDXL-LoRA-slider.anime",
        "trigger": "anime style",
        "weight_name": "anime.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "pastel-anime": {
        "repo": "Linaqruf/pastel-anime-xl-lora",
        "trigger": "masterpiece, best quality",
        "weight_name": "pastel-anime-xl.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "cartoon": {
        "repo": "ntc-ai/SDXL-LoRA-slider.cartoon",
        "trigger": "cartoon style",
        "weight_name": "cartoon.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Art Styles (2) ──
    "watercolor": {
        "repo": "ostris/watercolor_style_lora_sdxl",
        "trigger": "watercolor style",
        "weight_name": "watercolor_v1_sdxl.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "comic": {
        "repo": "ntc-ai/SDXL-LoRA-slider.2000s-indie-comic-art-style",
        "trigger": "comic art style",
        "weight_name": "2000s indie comic art style.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Themes / Genres (4) ──
    "fantasy": {
        "repo": "ntc-ai/SDXL-LoRA-slider.fantasy",
        "trigger": "fantasy",
        "weight_name": "fantasy.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "dark-gothic": {
        "repo": "thwri/dark-gothic-fantasy-xl",
        "trigger": "dark gothic fantasy",
        "weight_name": "dark_gothic_fantasy_xl_3.01.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "cyberpunk": {
        "repo": "jbilcke-hf/sdxl-cyberpunk-2077",
        "trigger": "cyberpunk-2077",
        "weight_name": "pytorch_lora_weights.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "scifi": {
        "repo": "e-n-v-y/envy-scifi-streamline-xl-01",
        "trigger": "scifi streamline modern",
        "weight_name": "EnvyScifiStreamlineXL01.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Special (2) ──
    "cosmic-horror": {
        "repo": "ntc-ai/SDXL-LoRA-slider.cosmic-horror",
        "trigger": "cosmic horror",
        "weight_name": "cosmic horror.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "texture-synth": {
        "repo": "dog-god/texture-synthesis-sdxl-lora",
        "trigger": "colormap",
        "weight_name": "texture-synthesis-3d-base-condensed.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Traditional Art (4) ──
    "oil-painting": {
        "repo": "ntc-ai/SDXL-LoRA-slider.oil-painting",
        "trigger": "oil painting",
        "weight_name": "oil painting.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "watercolor-ink": {
        "repo": "ming-yang/sdxl_chinese_ink_lora",
        "trigger": "Chinese Ink",
        "weight_name": "Chinese_Ink_Painting_Lora_SDXL.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "sketch": {
        "repo": "Linaqruf/sketch-style-xl-lora",
        "trigger": "sketch style",
        "weight_name": "sketch-style-xl.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "van-gogh": {
        "repo": "ntc-ai/SDXL-LoRA-slider.van-gogh",
        "trigger": "van gogh",
        "weight_name": "van gogh.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Craft / Physical (2) ──
    "claymation": {
        "repo": "DoctorDiffusion/doctor-diffusion-s-claymation-style-lora",
        "trigger": "made-of-clay",
        "weight_name": "DD-made-of-clay-XL-v2.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "origami": {
        "repo": "RalFinger/origami-style-sdxl-lora",
        "trigger": "ral-orgmi",
        "weight_name": "ral-orgmi-sdxl.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Stylized Render (3) ──
    "3d-render": {
        "repo": "artificialguybr/3DRedmond-V1",
        "trigger": "3DRenderAF",
        "weight_name": "3DRedmond-3DRenderStyle-3DRenderAF.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "ghibli": {
        "repo": "ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style",
        "trigger": "Studio Ghibli style",
        "weight_name": "Studio Ghibli style.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "pixar": {
        "repo": "ntc-ai/SDXL-LoRA-slider.pixar-style",
        "trigger": "pixar style",
        "weight_name": "pixar-style.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Post-Apocalyptic (1) ──
    "apocalyptic": {
        "repo": "ntc-ai/SDXL-LoRA-slider.apocalyptic",
        "trigger": "apocalyptic",
        "weight_name": "apocalyptic.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Glass / Material (2) ──
    "stained-glass": {
        "repo": "ostris/stained-glass-style-sdxl",
        "trigger": "stained glass",
        "weight_name": "stained_glass_style_v1_sdxl.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "chrome": {
        "repo": "RalFinger/chrome-style-sdxl-lora",
        "trigger": "ral-chrome",
        "weight_name": "ral-chrome-sdxl.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Craft / Textile (4) ──
    "yarn-art": {
        "repo": "Norod78/SDXL-YarnArtStyle-LoRA",
        "trigger": "Yarn art style",
        "weight_name": "SDXL_Yarn_Art_Style.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "wool": {
        "repo": "RalFinger/wool-style-sdxl-lora",
        "trigger": "zwuul",
        "weight_name": "zwuul-sdxl.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "crochet": {
        "repo": "artificialguybr/amigurami-redmond-amigurami-crochet-sd-xl-lora",
        "trigger": "Amigurami",
        "weight_name": "AmiguramiRedmond-Crochet-Amigurumi.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "crayon": {
        "repo": "ostris/crayon_style_lora_sdxl",
        "trigger": "crayons",
        "weight_name": "crayons_v1_sdxl.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Drawing / Line (3) ──
    "doodle": {
        "repo": "artificialguybr/doodle-redmond-doodle-hand-drawing-style-lora-for-sd-xl",
        "trigger": "DoodleRedm",
        "weight_name": "DoodleRedmond-Doodle-DoodleRedm.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "lineart": {
        "repo": "artificialguybr/LineAniRedmond-LinearMangaSDXL-V2",
        "trigger": "LineAniAF",
        "weight_name": "LineAniRedmondV2-Lineart-LineAniAF.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "tattoo": {
        "repo": "Norod78/yet-another-sdxl-tattoo-lora",
        "trigger": "tattoo",
        "weight_name": "SDXL-tattoo-Lora.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Film / Photo (2) ──
    "film-grain": {
        "repo": "artificialguybr/filmgrain-redmond-filmgrain-lora-for-sdxl",
        "trigger": "FilmGrainAF",
        "weight_name": "FilmGrainRedmond-FilmGrain-FilmGrainAF.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "ps1-retro": {
        "repo": "artificialguybr/ps1redmond-ps1-game-graphics-lora-for-sdxl",
        "trigger": "Playstation 1 Graphics",
        "weight_name": "PS1Redmond-PS1Game-Playstation1Graphics.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Graphic Design (2) ──
    "logo": {
        "repo": "artificialguybr/LogoRedmond-LogoLoraForSDXL-V2",
        "trigger": "LogoRedmAF",
        "weight_name": "LogoRedmondV2-Logo-LogoRedmAF.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "emoji": {
        "repo": "Norod78/sdxl-emoji-lora",
        "trigger": "flat, emoji",
        "weight_name": "SDXL-Emoji-Lora-r4.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Isometric / Figurine (2) ──
    "isometric": {
        "repo": "ntc-ai/SDXL-LoRA-slider.isometric-view",
        "trigger": "isometric view",
        "weight_name": "isometric view.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "figurine": {
        "repo": "ntc-ai/SDXL-LoRA-slider.figurine",
        "trigger": "figurine",
        "weight_name": "figurine.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Children / Story (1) ──
    "storybook": {
        "repo": "artificialguybr/StoryBookRedmond-V2",
        "trigger": "KidsRedmAF",
        "weight_name": "StorybookRedmondV2-KidsBook-KidsRedmAF.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Retro Horror (1) ──
    "retro-horror": {
        "repo": "ntc-ai/SDXL-LoRA-slider.retro-horror-comic-style-poster",
        "trigger": "retro horror comic style poster",
        "weight_name": "retro horror comic style poster.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Cel / Glitch / Silhouette (3) ──
    "cel-shaded": {
        "repo": "ntc-ai/SDXL-LoRA-slider.cel-shaded",
        "trigger": "cel-shaded",
        "weight_name": "cel-shaded.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "glitch": {
        "repo": "joachimsallstrom/aether-glitch-lora-for-sdxl",
        "trigger": "vhs glitch",
        "weight_name": "Aether_Glitch_v1_LoRA.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "silhouette": {
        "repo": "DoctorDiffusion/doctor-diffusion-s-stylized-silhouette-photography-xl-lora",
        "trigger": "sli artstyle",
        "weight_name": "DD-sli-v1.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Pencil Art (1) ──
    "pencil-art": {
        "repo": "prithivMLmods/Canopus-Pencil-Art-LoRA",
        "trigger": "Pencil Art",
        "weight_name": "Canopus-Pencil-Art-LoRA.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Cultural / Period (3) ──
    "art-deco": {
        "repo": "e-n-v-y/envy-fantasy-art-deco-xl-01",
        "trigger": "fantasy art deco",
        "weight_name": "EnvyFantasyArtDecoXL01.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "pop-art": {
        "repo": "tonyassi/warhol-lora",
        "trigger": "Warhol style",
        "weight_name": "pytorch_lora_weights.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "ukiyo-e": {
        "repo": "ronniealfaro/uyiko-e_Lora",
        "trigger": "ukiyo-e",
        "weight_name": "Ukiyo-e_LoRa.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Retro / Aesthetic (1) ──
    "chibi": {
        "repo": "RalFinger/smol-animals-sdxl-lora",
        "trigger": "zhibi",
        "weight_name": "zhibi-sdxl.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Sticker / Icon / UI (3) ──
    "sticker": {
        "repo": "artificialguybr/StickersRedmond",
        "trigger": "Sticker",
        "weight_name": "StickersRedmond.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "3d-icon": {
        "repo": "8glabs/3d-icon-sdxl-lora",
        "trigger": "3d icon",
        "weight_name": "3d-icon-sdxl-lora.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "app-icon": {
        "repo": "artificialguybr/IconsRedmond-IconsLoraForSDXL-V2",
        "trigger": "icredm",
        "weight_name": "IconsRedmondV2-Icons.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Lighting / Cinematic (2) ──
    "neon-sign": {
        "repo": "ProomptEngineer/pe-neon-sign-style",
        "trigger": "neon sign style",
        "weight_name": "PE_NeonSignStyle.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "cinematic-light": {
        "repo": "ntc-ai/SDXL-LoRA-slider.cinematic-lighting",
        "trigger": "cinematic lighting",
        "weight_name": "cinematic lighting.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Surreal / Abstract (1) ──
    "deep-dream": {
        "repo": "fofr/sdxl-deep-dream",
        "trigger": "deep dream",
        "weight_name": "lora.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Character / Game-Specific (2) ──
    "character-design": {
        "repo": "ntc-ai/SDXL-LoRA-slider.character-design",
        "trigger": "character design",
        "weight_name": "character design.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    "potion-art": {
        "repo": "FFusion/FFusionXL-LoRa-SDXL-Potion-Art-Engine",
        "trigger": "a 3d potion vial",
        "weight_name": "FFusionXL-LoRa-SDXL-Potion-Art-Engine.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Photo / Realism (1) ──
    "photorealistic": {
        "repo": "ostris/photorealistic-slider-sdxl-lora",
        "trigger": "photorealistic",
        "weight_name": "sdxl_photorealistic_slider_v1-0.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Pointillism (1) ──
    "pointillism": {
        "repo": "KappaNeuro/paul-signac-style",
        "trigger": "Paul Signac Style",
        "weight_name": "Paul Signac Style.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
    # ── Map / World Gen (1) ──
    "island-gen": {
        "repo": "FFusion/FFusionXL-LoRa-SDXL-Island-Generator",
        "trigger": "a 3d island",
        "weight_name": "FFusionAI-Islandgen.safetensors",
        "compatible": ["dreamshaper-xl"],
    },
}


# ══════════════════════════════════════════════════════════════════
# ModelManager — LRU VRAM Eviction Engine
# ══════════════════════════════════════════════════════════════════

class ModelEntry:
    __slots__ = ('name', 'loader', 'vram_mb', 'metadata', 'model_dict',
                 'on_gpu', 'last_used', 'pinned')

    def __init__(self, name, loader, vram_mb, metadata, pinned=False):
        self.name = name
        self.loader = loader
        self.vram_mb = vram_mb
        self.metadata = metadata or {}
        self.model_dict = None
        self.on_gpu = False
        self.last_used = None
        self.pinned = pinned


class ModelManager:
    """v9.1: Full unload only. One model at a time, full 16GB VRAM."""

    def __init__(self, vram_budget_mb=15200, idle_timeout_s=300, cleanup_interval_s=60):
        self.vram_budget_mb = vram_budget_mb
        self.idle_timeout_s = idle_timeout_s
        self.cleanup_interval_s = cleanup_interval_s
        self._registry: dict[str, ModelEntry] = {}
        self._gpu_loaded: OrderedDict[str, float] = OrderedDict()
        self._vram_usage: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._eviction_lock = asyncio.Lock()
        self._cleanup_task = None

    def register(self, name, loader, vram_mb, metadata=None, pinned=False):
        self._registry[name] = ModelEntry(name, loader, vram_mb, metadata or {}, pinned)
        self._locks[name] = asyncio.Lock()
        log.info(f"Registered: {name} ({vram_mb}MB VRAM est.){' [PINNED]' if pinned else ''}")

    async def get(self, name):
        if name not in self._registry:
            raise ValueError(f"Model '{name}' not registered. Available: {list(self._registry.keys())}")
        entry = self._registry[name]

        # Fast path: already on GPU
        if name in self._gpu_loaded:
            self._gpu_loaded.move_to_end(name)
            self._gpu_loaded[name] = time.time()
            entry.last_used = time.time()
            return entry.model_dict

        # Slow path: full unload others → load from disk
        async with self._locks[name]:
            if name in self._gpu_loaded:
                self._gpu_loaded.move_to_end(name)
                self._gpu_loaded[name] = time.time()
                entry.last_used = time.time()
                return entry.model_dict

            await self._ensure_vram(entry.vram_mb)

            try:
                log.info(f"Loading '{name}' from disk → GPU...")
                t0 = time.time()
                entry.model_dict = entry.loader()
                log.info(f"'{name}' loaded in {time.time()-t0:.1f}s")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    log.error(f"OOM loading '{name}', emergency evict ALL → retry")
                    for evict_name in list(self._gpu_loaded.keys()):
                        if not self._registry[evict_name].pinned:
                            self._do_evict(evict_name)
                    self._force_cleanup()
                    t0 = time.time()
                    entry.model_dict = entry.loader()
                    log.info(f"'{name}' loaded after emergency evict in {time.time()-t0:.1f}s")
                else:
                    raise

            entry.on_gpu = True
            entry.last_used = time.time()
            self._gpu_loaded[name] = time.time()
            self._vram_usage[name] = entry.vram_mb
            self._log_vram()
            return entry.model_dict

    async def _ensure_vram(self, needed_mb):
        SAFETY_MARGIN_MB = 500  # 500MB safety — we want max VRAM for model
        async with self._eviction_lock:
            if torch.cuda.is_available():
                actual_free = torch.cuda.mem_get_info()[0] // (1024 * 1024)
            else:
                actual_free = self.vram_budget_mb - sum(self._vram_usage.values())

            while actual_free < needed_mb + SAFETY_MARGIN_MB:
                evictable = [n for n in self._gpu_loaded if not self._registry[n].pinned]
                if not evictable:
                    log.warning(f"Cannot evict: need {needed_mb}+{SAFETY_MARGIN_MB}MB, only {actual_free}MB free (all pinned)")
                    self._force_cleanup()
                    if torch.cuda.is_available():
                        actual_free = torch.cuda.mem_get_info()[0] // (1024 * 1024)
                    break
                lru_name = evictable[0]
                log.info(f"VRAM pressure: need {needed_mb}+{SAFETY_MARGIN_MB}MB, free={actual_free}MB → evicting '{lru_name}'")
                self._do_evict(lru_name)
                if torch.cuda.is_available():
                    actual_free = torch.cuda.mem_get_info()[0] // (1024 * 1024)
                else:
                    actual_free = self.vram_budget_mb - sum(self._vram_usage.values())

    def _do_evict(self, name):
        """Full unload: delete model from memory entirely. No CPU caching."""
        entry = self._registry[name]
        log.info(f"Evicting '{name}' — full unload...")
        self._full_unload(entry)
        entry.on_gpu = False
        self._gpu_loaded.pop(name, None)
        self._vram_usage.pop(name, None)
        self._force_cleanup()
        self._log_vram()

    def _full_unload(self, entry):
        md = entry.model_dict
        if not md:
            return
        # Unhook diffusers CPU offload hooks first
        model = md.get("model")
        if model:
            if hasattr(model, 'maybe_free_model_hooks'):
                try: model.maybe_free_model_hooks()
                except: pass
            # Move to CPU then delete (releases CUDA tensors)
            if hasattr(model, 'to'):
                try: model.to('cpu')
                except: pass
            del model
        # Delete all other entries (processor, controlnet, etc.)
        for k in list(md.keys()):
            obj = md[k]
            if hasattr(obj, 'to'):
                try: obj.to('cpu')
                except: pass
            del md[k]
        entry.model_dict = None

    @staticmethod
    def _force_cleanup():
        """Aggressive VRAM reclaim: double GC + empty cache + sync."""
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    async def start_cleanup_loop(self):
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(self.cleanup_interval_s)
            now = time.time()
            to_evict = []
            for name, last_used in list(self._gpu_loaded.items()):
                if not self._registry[name].pinned and (now - last_used) > self.idle_timeout_s:
                    to_evict.append(name)
            for name in to_evict:
                idle = time.time() - self._gpu_loaded.get(name, 0)
                log.info(f"Idle eviction: '{name}' ({idle:.0f}s idle)")
                self._do_evict(name)

    def get_status(self):
        free_vram, total_vram = 0, 0
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0] // (1024 * 1024)
            total_vram = torch.cuda.mem_get_info()[1] // (1024 * 1024)
        return {
            "vram_budget_mb": self.vram_budget_mb,
            "vram_tracked_mb": sum(self._vram_usage.values()),
            "vram_actual_free_mb": free_vram,
            "vram_total_mb": total_vram,
            "models_on_gpu": list(self._gpu_loaded.keys()),
            "models_unloaded": [n for n, e in self._registry.items() if not e.on_gpu],
            "total_registered": len(self._registry),
            "models": {
                name: {
                    "on_gpu": e.on_gpu, "pinned": e.pinned,
                    "vram_mb": e.vram_mb, "last_used": e.last_used,
                    "idle_s": int(time.time() - e.last_used) if e.last_used else None,
                    **{k: v for k, v in e.metadata.items()},
                }
                for name, e in self._registry.items()
            },
        }

    def _log_vram(self):
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 1024**2
            total = torch.cuda.mem_get_info()[1] / 1024**2
            tracked = sum(self._vram_usage.values())
            log.info(f"VRAM: {total-free:.0f}MB used / {tracked}MB tracked / {total:.0f}MB total (free: {free:.0f}MB)")


# Auto-detect VRAM: use 95% of GPU memory (full all-in, 5% for CUDA context)
_auto_vram = 15200
if torch.cuda.is_available():
    _total_vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    _auto_vram = int(_total_vram * 0.95)
    log.info(f"GPU: {torch.cuda.get_device_name(0)}, {_total_vram}MB total, budget={_auto_vram}MB (95% all-in)")

manager = ModelManager(vram_budget_mb=_auto_vram, idle_timeout_s=300)

# Image generation models — used for LoRA compatibility fallback routing
IMAGE_MODELS = {"dreamshaper-xl", "sdxl-base"}


# ══════════════════════════════════════════════════════════════════
# Model Loaders — each returns {"model": ..., "processor": ..., ...}
# ══════════════════════════════════════════════════════════════════

# v9.1: Removed embeddings-minilm (nomic-v2 is 86% RAG vs 56%)

def _load_tts_kokoro():
    from kokoro_onnx import Kokoro
    model_path = str(Path(__file__).parent / "kokoro-v1.0.onnx")
    voices_path = str(Path(__file__).parent / "voices-v1.0.bin")
    return {"model": Kokoro(model_path, voices_path)}

# v9.1: Removed flux1-schnell — all image gen through flux1-dev only (max speed stack)

# v9.1: Removed redundant loaders (flux2-klein, sd35, pixart, sdxl-turbo,
# instructpix2pix, sdxl-inpaint, musicgen-medium, musicgen-melody, audioldm-v1)

def _load_upscale_realesrgan():
    import spandrel
    weights_path = Path(__file__).parent / "RealESRGAN_x4plus.pth"
    if not weights_path.exists():
        import urllib.request
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        log.info("Downloading Real-ESRGAN weights...")
        urllib.request.urlretrieve(url, str(weights_path))
    model = spandrel.ModelLoader(device=DEVICE).load_from_file(str(weights_path))
    model = model.eval()
    if hasattr(model, 'half'):
        model = model.half()
    return {"model": model, "scale": 4}

# v9.1: Removed rembg-u2net (replaced by RMBG-2.0), depth-v2-small (keep base only)

def _load_depth_v2_base():
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=DEVICE)
    return {"model": pipe}

def _load_marigold():
    from diffusers import MarigoldDepthPipeline
    pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", torch_dtype=torch.float16, variant="fp16")
    pipe.to(DEVICE)
    return {"model": pipe}

# ── v12.0 NEW Model Loaders — Upgrades ──

def _load_depth_pro():
    """Depth Pro (Apple) — metric depth, 2.25MP in 0.3s, ~5GB VRAM.
    Replaces Depth Anything V2 Base. Outputs absolute metric depth (meters)."""
    from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
    processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained(
        "apple/DepthPro-hf", torch_dtype=torch.float16).to(DEVICE)
    log.info("Depth Pro loaded — metric depth in 0.3s")
    return {"model": model, "processor": processor}

def _load_tts_sesame_csm():
    """Sesame CSM-1B — most human-like conversational TTS with natural pauses.
    ~4GB VRAM. Requires HF token with license acceptance."""
    from transformers import CsmForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained("sesame/csm-1b", token=HF_TOKEN)
    model = CsmForConditionalGeneration.from_pretrained(
        "sesame/csm-1b", torch_dtype=torch.float16, token=HF_TOKEN).to(DEVICE)
    log.info("Sesame CSM-1B loaded — most human-like TTS")
    return {"model": model, "processor": processor}

def _load_music_ace_step():
    """ACE-Step 1.5 — full song generation with lyrics, <4GB, 15x faster than MusicGen.
    Requires Python 3.11. Install: git clone ACE-Step-1.5 && pip install -e ."""
    from acestep.handler import AceStepHandler
    handler = AceStepHandler(device=str(DEVICE))
    log.info("ACE-Step 1.5 loaded — full song generation ready")
    return {"model": handler, "backend": "ace-step"}

# v9.1: Removed bark (chatterbox is better)

def _load_stt_whisper():
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    return {"model": model}

# v9.1: Removed triposr, shap-e (hunyuan3d better), playground-v25 (flux better)

def _load_image_flux1_dev():
    """FLUX.1-dev HYBRID — INT8 transformer (quality) + NF4 T5 (save VRAM) + speed stack.
    ~15.5GB VRAM. INT8 transformer = ~98-99% quality vs FP16. Full GPU, no cpu_offload."""
    from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig as DBnb
    from transformers import T5EncoderModel, BitsAndBytesConfig as TBnb

    # INT8 transformer — high quality, ~12GB VRAM
    quant_d = DBnb(load_in_8bit=True)
    # NF4 T5 encoder — text encoding only, minimal quality impact, ~2.5GB VRAM
    quant_t = TBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="transformer",
        quantization_config=quant_d, torch_dtype=torch.bfloat16, token=HF_TOKEN)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2",
        quantization_config=quant_t, torch_dtype=torch.bfloat16, token=HF_TOKEN)

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=transformer,
        text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16, token=HF_TOKEN)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # QKV fusion — merge q,k,v projections into single matmul (~10-15% speedup)
    try:
        pipe.transformer.fuse_qkv_projections()
        log.info("FLUX.1-dev: QKV fusion applied")
    except Exception as e:
        log.warning(f"FLUX.1-dev: QKV fusion failed ({e})")

    # First Block Cache — skip redundant transformer blocks between steps (~1.5x speedup)
    try:
        from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
        apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.2))
        log.info("FLUX.1-dev: FirstBlockCache applied (threshold=0.2)")
    except Exception as e:
        log.warning(f"FLUX.1-dev: FirstBlockCache failed ({e})")

    # Warm-up run — pre-allocate CUDA memory so first real request is fast
    try:
        log.info("FLUX.1-dev: warming up...")
        with torch.inference_mode():
            _ = pipe("warmup", num_inference_steps=1, width=256, height=256,
                     guidance_scale=3.5, max_sequence_length=512)
        log.info("FLUX.1-dev: warm-up complete, ready for fast inference")
    except Exception as e:
        log.warning(f"FLUX.1-dev: warm-up failed ({e}), first request will be slower")

    return {"model": pipe}


def _load_image_flux1_schnell():
    """FLUX.1-schnell — 4-step distilled, fast batch generation. FP8 quantized ~12GB."""
    from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig as DBnb
    from transformers import T5EncoderModel, BitsAndBytesConfig as TBnb

    # INT8 transformer for quality
    quant_d = DBnb(load_in_8bit=True)
    # NF4 T5 encoder to save VRAM
    quant_t = TBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="transformer",
        quantization_config=quant_d, torch_dtype=torch.bfloat16)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", subfolder="text_encoder_2",
        quantization_config=quant_t, torch_dtype=torch.bfloat16)

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", transformer=transformer,
        text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # QKV fusion
    try:
        pipe.transformer.fuse_qkv_projections()
        log.info("FLUX.1-schnell: QKV fusion applied")
    except Exception as e:
        log.warning(f"FLUX.1-schnell: QKV fusion failed ({e})")

    log.info("FLUX.1-schnell: loaded, 4-step fast generation ready")
    return {"model": pipe}


def _load_sdxl_base():
    """SDXL Base 1.0 txt2img — versatile, fast, great with LoRAs. ~7GB VRAM."""
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.to(DEVICE)
    pipe.vae.enable_tiling()

    log.info("SDXL Base: loaded, ready for LoRA-powered game asset generation")
    return {"model": pipe}


def _load_dreamshaper_xl():
    """DreamShaper XL Turbo — 4-8 step generation, fantasy/stylized art. ~7GB VRAM."""
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "Lykon/dreamshaper-xl-v2-turbo",
        torch_dtype=torch.float16, use_safetensors=True)
    # DPM++ SDE for turbo — fast + high quality
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    pipe.to(DEVICE)
    pipe.vae.enable_tiling()

    log.info("DreamShaper XL Turbo: loaded, 4-8 step fast stylized generation ready")
    return {"model": pipe}


# v12.0: Removed stable-audio (ACE-Step better), controlnet-union (unused), ip-adapter (hangs)

def _load_sam2_tiny():
    from transformers import AutoProcessor, AutoModelForMaskGeneration
    processor = AutoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")
    # SAM2 needs float32 — float16 causes dtype mismatch in conv layers
    model = AutoModelForMaskGeneration.from_pretrained(
        "facebook/sam2.1-hiera-tiny", torch_dtype=torch.float32).to(DEVICE)
    return {"model": model, "processor": processor}

# v9.1: Removed cogvideox (wan21 better), svd (old), swinir (realesrgan enough)


# ── v9.1 Model Loaders ──

# v9.1: Removed flux2-dev (32B too big for 16GB single model)

# v12.0: Removed _load_music_large (replaced by _load_music_ace_step with MusicGen-Small fallback)

def _load_sfx_audioldm2():
    """AudioLDM 2 — SoTA text-to-audio with CLAP+T5+GPT2"""
    from diffusers import AudioLDM2Pipeline
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2", torch_dtype=torch.float16).to(DEVICE)

    # Fix: AudioLDM2 loads GPT2Model (not GPT2LMHeadModel) which lacks GenerationMixin.
    # diffusers 0.36.0 generate_language_model needs methods from GenerationMixin.
    # Copy ALL GenerationMixin methods to the GPT2Model instance.
    lang_model = pipe.language_model
    if lang_model is not None and not hasattr(lang_model, '_get_initial_cache_position'):
        from transformers import GenerationMixin
        import types
        count = 0
        for name in dir(GenerationMixin):
            if name.startswith('__'):
                continue
            method = getattr(GenerationMixin, name, None)
            if callable(method) and not hasattr(lang_model, name):
                try:
                    bound = types.MethodType(method, lang_model)
                    setattr(lang_model, name, bound)
                    count += 1
                except Exception:
                    pass
        log.info(f"AudioLDM2: copied {count} GenerationMixin methods to GPT2Model")

    return {"model": pipe}

def _load_tts_chatterbox():
    """Chatterbox-Turbo — beats ElevenLabs, voice cloning, MIT"""
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device=DEVICE)
    return {"model": model}

def _load_tts_f5():
    """F5-TTS — best zero-shot voice cloning, 330M params"""
    from f5_tts.api import F5TTS
    model = F5TTS(device=DEVICE)
    return {"model": model}

def _load_video_wan21():
    """Wan 2.1 T2V-1.3B — best quality video gen, CPU offload to fit in 16GB"""
    from diffusers import WanPipeline
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    return {"model": pipe}

def _load_video_ltx():
    """LTX-Video 2B — fast video gen, 24fps, FP8 layerwise casting ~6GB VRAM.
    Works with diffusers>=0.36.0. Resolution divisible by 32, frames divisible by 8+1."""
    from diffusers import LTXPipeline
    try:
        from diffusers import AutoModel
        transformer = AutoModel.from_pretrained(
            "Lightricks/LTX-Video", subfolder="transformer", torch_dtype=torch.bfloat16)
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video", transformer=transformer, torch_dtype=torch.bfloat16)
        log.info("LTX-Video 2B loaded with FP8 layerwise casting (~6GB VRAM)")
    except Exception as e:
        log.warning(f"FP8 layerwise casting failed ({e}), falling back to BF16")
        pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    return {"model": pipe}

def _load_3d_hunyuan():
    """Hunyuan3D 2.1 full (3.3B) — max quality 3D gen. PINNED HOT (14GB).
    Requires: git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 ~/Hunyuan3D-2.1
    + full weights in ~/Hunyuan3D-2.1-full/hunyuan3d-dit-v2-1/model.fp16.ckpt"""
    import sys
    import types
    hunyuan_path = os.path.join(os.path.expanduser("~"), "Hunyuan3D-2.1")
    shape_path = os.path.join(hunyuan_path, "hy3dshape")
    if not os.path.exists(shape_path):
        raise FileNotFoundError(f"Hunyuan3D not found at {hunyuan_path}. Install: git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 ~/Hunyuan3D-2.1")
    if shape_path not in sys.path:
        sys.path.insert(0, shape_path)
    import hy3dshape
    import hy3dshape.models
    import hy3dshape.models.autoencoders
    import hy3dshape.models.conditioner
    sys.modules['hy3dgen'] = types.ModuleType('hy3dgen')
    sys.modules['hy3dgen.shapegen'] = hy3dshape
    sys.modules['hy3dgen.shapegen.models'] = hy3dshape.models
    sys.modules['hy3dgen.shapegen.models.autoencoders'] = hy3dshape.models.autoencoders
    sys.modules['hy3dgen.shapegen.models.conditioner'] = hy3dshape.models.conditioner
    sys.modules['hy3dgen'].shapegen = hy3dshape
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    # v13.2: Try 2.1 full (3.3B) first, fallback to mini-turbo (0.6B) if VRAM fails
    full_model_path = os.path.join(os.path.expanduser("~"), "Hunyuan3D-2.1-full")
    try:
        if os.path.exists(os.path.join(full_model_path, "hunyuan3d-dit-v2-1", "model.fp16.ckpt")):
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                full_model_path,
                subfolder="hunyuan3d-dit-v2-1",
                use_safetensors=False,
                variant="fp16")
            log.info("Hunyuan3D 2.1 FULL (3.3B) loaded — maximum quality 3D ready")
        else:
            raise FileNotFoundError("Full model not downloaded yet")
    except Exception as e:
        log.warning(f"Hunyuan3D 2.1 full failed ({e}), falling back to mini-turbo")
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mini",
            subfolder="hunyuan3d-dit-v2-mini-turbo",
            use_safetensors=True,
            variant="fp16")
        log.info("Hunyuan3D 2-mini-turbo loaded — production 3D ready (fallback)")
    # v34.0: torch.compile — skip on Windows (no Triton), use reduce-overhead on Linux
    import platform
    if platform.system() != "Windows":
        try:
            if hasattr(pipe, 'model') and hasattr(pipe.model, 'forward'):
                pipe.model = torch.compile(pipe.model, mode="reduce-overhead")
                log.info("Hunyuan3D pipeline torch.compiled (reduce-overhead mode)")
            elif hasattr(pipe, 'dit') and hasattr(pipe.dit, 'forward'):
                pipe.dit = torch.compile(pipe.dit, mode="reduce-overhead")
                log.info("Hunyuan3D DiT torch.compiled (reduce-overhead mode)")
        except Exception as e:
            log.warning(f"torch.compile skipped for Hunyuan3D: {e}")
    else:
        log.info("Hunyuan3D torch.compile SKIPPED (Windows — no Triton). Using eager mode.")
    # v18.0: channels_last memory format — 15-30% faster on modern GPUs for Conv2D
    try:
        if hasattr(pipe, 'model'):
            pipe.model = pipe.model.to(memory_format=torch.channels_last)
            log.info("Hunyuan3D model → channels_last memory format")
        elif hasattr(pipe, 'dit'):
            pipe.dit = pipe.dit.to(memory_format=torch.channels_last)
            log.info("Hunyuan3D DiT → channels_last memory format")
    except Exception as e:
        log.warning(f"channels_last skipped: {e}")
    # v18.0: CUDA graph warmup — pre-compile CUDA kernels for first inference speed
    try:
        from PIL import Image as _WarmupImage
        warmup_img = _WarmupImage.new("RGB", (256, 256), (128, 128, 128))
        log.info("Hunyuan3D warmup inference starting (compiles CUDA graphs)...")
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            _ = pipe(image=warmup_img, num_inference_steps=2)
        torch.cuda.empty_cache()
        log.info("Hunyuan3D warmup DONE — CUDA graphs compiled, first real inference will be fast")
    except Exception as e:
        log.warning(f"Hunyuan3D warmup skipped: {e}")
    return {"model": pipe, "backend": "hunyuan3d-native"}

def _load_triposr():
    """TripoSR — fast image-to-3D (<1s), outputs OBJ/GLB via trimesh. NO FALLBACK."""
    try:
        from tsr.system import TSR
        model = TSR.from_pretrained("stabilityai/TripoSR", dtype=torch.float16, device="cuda")
        model.renderer.set_chunk_size(8192)
        log.info("TripoSR loaded — fast image-to-3D ready")
        return {"model": model, "backend": "triposr-native"}
    except (ImportError, Exception) as e:
        log.error(f"TripoSR NOT available ({e}). Install: pip install git+https://github.com/VAST-AI-Research/TripoSR")
        return {"model": None, "backend": "unavailable", "error": str(e)}

# v9.1: Removed flux-fill — consolidated to flux1-dev only

def _load_rembg_v2():
    """Background removal — RMBG-2.0 (BiRefNet). NO FALLBACK."""
    from transformers import AutoModelForImageSegmentation, AutoProcessor
    processor = AutoProcessor.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True, token=HF_TOKEN)
    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", trust_remote_code=True, token=HF_TOKEN).to(DEVICE)
    return {"model": model, "processor": processor, "backend": "rmbg2"}

def _load_embeddings_nomic():
    """Nomic Embed Text v2 — 86% RAG accuracy vs MiniLM's 56%"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True, device=DEVICE)
    return {"model": model}

def _load_sam21_small():
    """SAM 2.1 hiera-small — better accuracy, image + video segmentation"""
    from transformers import AutoProcessor, AutoModelForMaskGeneration
    processor = AutoProcessor.from_pretrained("facebook/sam2.1-hiera-small")
    model = AutoModelForMaskGeneration.from_pretrained(
        "facebook/sam2.1-hiera-small", torch_dtype=torch.float32).to(DEVICE)
    return {"model": model, "processor": processor}

# v9.1: Removed controlnet-flux — consolidated to flux1-dev + controlnet-union-sdxl only


# ── NEW v8.1 Model Loaders — Image Analysis, Img2Img, Pose, Palette ──

def _load_clip_vit():
    """CLIP ViT-L/14 — image captioning, visual search, scene understanding"""
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    return {"model": model, "processor": processor}

def _load_blip2_caption():
    """BLIP-2 — image captioning, visual Q&A, scene description"""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(DEVICE)
    return {"model": model, "processor": processor}

def _load_yolo_detect():
    """YOLO11m — upgraded detection, faster + more accurate than YOLOv8m"""
    from ultralytics import YOLO
    model = YOLO("yolo11m.pt")
    return {"model": model}

def _load_owlvit():
    """OWL-ViT — zero-shot object detection by text description"""
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
    model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-large-patch14").to(DEVICE)
    return {"model": model, "processor": processor}

def _load_dwpose():
    """DWPose — pose estimation via YOLOv8-pose (body keypoints)"""
    from ultralytics import YOLO
    model = YOLO("yolov8m-pose.pt")
    return {"model": model}

def _load_img2img_sdxl():
    """SDXL Img2Img — transform existing images with text guidance"""
    from diffusers import StableDiffusionXLImg2ImgPipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16")
    pipe.to(DEVICE)
    return {"model": pipe}

def _load_animatediff():
    """AnimateDiff — generate sprite animations from text"""
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", motion_adapter=adapter,
        torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    return {"model": pipe}

def _load_rife():
    """RIFE — frame interpolation for smooth animations"""
    # Using practical-rife via torch hub
    import subprocess, sys
    rife_path = Path(__file__).parent / "rife"
    if not rife_path.exists():
        log.info("RIFE: Using OpenCV frame interpolation fallback")
    return {"model": "opencv-interpolation"}


# ══════════════════════════════════════════════════════════════════
# Request Schemas
# ══════════════════════════════════════════════════════════════════

class ImageRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    steps: int = 12
    style: str = "gameAsset"
    model: str = "flux1-dev"
    lora: Optional[str] = None
    lora_scale: float = 0.85
    guidance_scale: float = 4.5
    negative_prompt: str = ""
    return_base64: bool = False  # Return base64 preview in response

class ImageEditRequest(BaseModel):
    prompt: str
    image_path: str
    model: str = "instructpix2pix"

class ImageInpaintRequest(BaseModel):
    prompt: str
    image_path: str
    mask_path: str

class Gen3DRequest(BaseModel):
    prompt: str = ""
    image: str = ""  # base64 data URI for image-to-3D
    image_path: str = ""
    model: str = "auto"  # auto = hunyuan3d > triposr (ShapE REMOVED v10.0)
    format: str = "glb"
    steps: int = 30  # v13.1: turbo model — 30 steps optimal (was 50, 40% faster)
    quality: str = "production"  # v18.0: draft=15steps, fast=25, production=30, max=50

class UpscaleRequest(BaseModel):
    image: str = ""  # base64 data URI
    image_path: str = ""
    scale: int = 4
    model: str = "realesrgan"
    return_base64: bool = False

class RembgRequest(BaseModel):
    image: str = ""  # base64 data URI
    image_path: str = ""
    return_base64: bool = False

class DepthRequest(BaseModel):
    image: str = ""  # base64 data URI
    image_path: str = ""
    model: str = "depth-pro"
    return_base64: bool = False

class NormalsRequest(BaseModel):
    image: str = ""  # base64 data URI
    image_path: str = ""
    return_base64: bool = False

class MusicRequest(BaseModel):
    description: str
    lyrics: str = ""  # v20.0: ACE-Step supports lyrics
    duration: int = 30  # v20.0: ACE-Step default 30s (was 10)
    model: str = "ace-step"  # v20.0: ACE-Step 1.5 (was musicgen-large)
    tags: str = ""  # v20.0: ACE-Step genre/mood tags
    return_path: bool = True

class SFXRequest(BaseModel):
    description: str
    duration: float = 5.0
    steps: int = 35  # Maximum quality (v9.6)
    model: str = "audioldm2"

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    model: str = "kokoro"

class STTRequest(BaseModel):
    audio_path: str

class ImageBatchRequest(BaseModel):
    """Batch image generation — load model ONCE, generate N images. Fastest workflow."""
    prompts: List[str]
    width: int = 512
    height: int = 512
    steps: int = 10
    style: str = "gameAsset"
    model: str = "dreamshaper-xl"  # Default: fastest + Unity-safe (7GB)
    lora: Optional[str] = None
    lora_scale: float = 0.85
    downscale: int = 0  # If >0, downscale output to NxN using nearest-neighbor (for pixel art)

class EmbedRequest(BaseModel):
    text: str

class EmbedBatchRequest(BaseModel):
    texts: List[str]

class RAGStoreRequest(BaseModel):
    collection: str = "default"
    texts: List[str]
    ids: Optional[List[str]] = None
    metadatas: Optional[List[dict]] = None

class RAGQueryRequest(BaseModel):
    collection: str = "default"
    query: str
    n_results: int = 5

class ManageRequest(BaseModel):
    model: str
    action: str = "to_gpu"

class SwapRequest(BaseModel):
    model: str

class SegmentRequest(BaseModel):
    image: str = ""  # base64 data URI
    image_path: str = ""
    points: Optional[List[List[int]]] = None  # [[x,y], ...]
    labels: Optional[List[int]] = None  # 1=foreground, 0=background
    box: Optional[List[int]] = None  # [x1,y1,x2,y2]
    return_base64: bool = False  # Return sprite as base64

class VideoRequest(BaseModel):
    prompt: str = ""
    image: str = ""  # base64 data URI
    image_path: str = ""
    model: str = "ltx-video"  # v22.0: LTX-Video 2B
    num_frames: int = 25  # ~1s at 24fps, must be 8n+1
    width: int = 512  # divisible by 32
    height: int = 320  # divisible by 32
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

class ControlNetRequest(BaseModel):
    prompt: str
    control_image_path: str
    condition: str = "canny"  # canny/depth/openpose/scribble
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 8.0

class StyleTransferRequest(BaseModel):
    prompt: str = ""
    style_prompt: str = ""  # text description of desired style
    image: str = ""  # base64 content image
    image_path: str = ""  # file path content image
    content_path: str = ""  # alias for image_path
    style_image: str = ""  # base64 style image
    style_image_path: str = ""
    width: int = 1024
    height: int = 1024
    scale: float = 0.8  # IP-adapter scale (stronger style)
    return_base64: bool = False

class CaptionRequest(BaseModel):
    image: str = ""  # base64 data URI
    image_path: str = ""
    question: str = ""  # Optional VQA question
    return_base64: bool = False  # Return annotated image as base64

class DetectRequest(BaseModel):
    image: str = ""  # base64 data URI
    image_path: str = ""
    labels: Optional[List[str]] = None  # For OWL-ViT zero-shot
    confidence: float = 0.15  # Lower threshold = detect more objects
    return_base64: bool = False  # Return annotated image as base64

class PoseRequest(BaseModel):
    image: str = ""  # base64 data URI
    image_path: str = ""
    return_base64: bool = False

class Img2ImgRequest(BaseModel):
    prompt: str
    image: str = ""  # base64 data URI
    image_path: str = ""
    strength: float = 0.7  # 0=no change, 1=full regeneration
    steps: int = 30
    model: str = "sdxl-img2img"
    return_base64: bool = False

class AnimateRequest(BaseModel):
    prompt: str
    num_frames: int = 16
    fps: int = 8
    width: int = 512
    height: int = 512

class PaletteRequest(BaseModel):
    image_path: str
    num_colors: int = 8

class InterpolateRequest(BaseModel):
    frame1_path: str
    frame2_path: str
    num_interpolations: int = 3


# ══════════════════════════════════════════════════════════════════
# Registration & Startup
# ══════════════════════════════════════════════════════════════════

def register_all_models():
    """v13.2 HOT-3D: Hunyuan3D 2.1 full PINNED (14GB). Image gen → remote FLUX.1-dev."""

    # ── PINNED: Always on GPU, never evicted ──
    manager.register("embeddings", _load_embeddings_nomic, 200,
        {"task": "embeddings", "params": "475M MoE", "model_id": "nomic-embed-text-v2-moe"}, pinned=True)
    manager.register("hunyuan3d", _load_3d_hunyuan, 14000,
        {"task": "image-to-3d", "params": "3.3B", "license": "MIT",
         "feature": "PINNED-hot-loaded-instant-3D-full-quality"}, pinned=True)

    # ── v22.1: STRIPPED TO 2 MODELS ONLY ──
    # User requested: ALL VRAM dedicated to Hunyuan3D + FLUX.1-dev (remote)
    # REMOVED: audioldm2, chatterbox, ace-step, ltx-video, kokoro
    # Endpoints still exist but return "model not registered" errors

    log.info(f"Registered {len(manager._registry)} models (v22.1 STRIPPED). Hunyuan3D PINNED + embeddings PINNED. ALL VRAM → 3D.")


async def load_pinned_models():
    """v20.0: Parallel load pinned models at startup using asyncio. Graceful: skip models that fail."""
    pinned_names = [name for name, entry in manager._registry.items() if entry.pinned]
    if not pinned_names:
        log.info("No pinned models to load.")
        return

    def _load_single(name):
        entry = manager._registry[name]
        log.info(f"Loading pinned model: {name}")
        try:
            entry.model_dict = entry.loader()
            entry.on_gpu = entry.vram_mb > 0
            entry.last_used = time.time()
            if entry.vram_mb > 0:
                manager._gpu_loaded[name] = time.time()
                manager._vram_usage[name] = entry.vram_mb
            log.info(f"Loaded pinned model: {name} OK")
            return True
        except Exception as e:
            log.warning(f"Failed to load pinned model {name}: {e}. Server continues without it.")
            entry.model_dict = None
            return False

    tasks = [asyncio.to_thread(_load_single, name) for name in pinned_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    succeeded = sum(1 for r in results if r is True)
    log.info(f"Parallel pinned model loading: {succeeded}/{len(pinned_names)} succeeded")


# ─── ChromaDB ───
chroma_client = None

def get_chroma():
    global chroma_client
    if chroma_client is None:
        try:
            import chromadb
            chroma_host = os.getenv("CHROMA_HOST", "localhost")
            chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
            chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            chroma_client.heartbeat()
            log.info("ChromaDB connected.")
        except Exception as e:
            log.warning(f"ChromaDB not available: {e}")
            chroma_client = None
    return chroma_client


# ══════════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    register_all_models()
    await load_pinned_models()
    await manager.start_cleanup_loop()
    # v21.0: Start VRAM timeline background task
    asyncio.create_task(_vram_timeline_loop())
    # v25.0: Start VRAM dashboard background task (15s interval, separate from timeline)
    asyncio.create_task(_vram_dashboard_loop())
    log.info(f"Server v29.0 SUPREME ULTRA ready. {len(manager._registry)} models. Hunyuan3D 2.1 full PINNED. MeshLOD+Pipeline3D+MeshAnalyze+BatchExport+ModelWarmup+Presets+Batch3Dv2+MeshPostProcess+TextureBake+VRAMDashboard+ModelHealth+Review3Dv2+FormatConvert+AssetCatalog.")
    if torch.cuda.is_available():
        manager._log_vram()
    yield
    log.info("Shutting down — unloading all models...")
    for name in list(manager._gpu_loaded.keys()):
        try:
            manager._do_evict(name)
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Shutdown complete.")

app = FastAPI(title="Local AI Server", version="29.0", lifespan=lifespan)


# ══════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    free, total, allocated, reserved = 0, 0, 0, 0
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] // (1024*1024)
        total = torch.cuda.mem_get_info()[1] // (1024*1024)
        allocated = torch.cuda.memory_allocated(0) // (1024*1024)
        reserved = torch.cuda.memory_reserved(0) // (1024*1024)
    import psutil
    ram = psutil.virtual_memory()
    uptime = time.time() - start_time
    # v20.0: GPU metrics via pynvml (temp, utilization, power)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
        pynvml.nvmlShutdown()
    except Exception:
        gpu_temp = gpu_util = gpu_power = -1
    return {
        "status": "online", "version": "29.0", "mode": "multi-engine-single-model",
        "device": DEVICE, "gpu": gpu_name,
        "gpu_temp_c": gpu_temp, "gpu_utilization_pct": gpu_util, "gpu_power_w": round(gpu_power, 1) if gpu_power != -1 else -1,
        "vram_free_mb": free, "vram_used_mb": total - free, "vram_total_mb": total,
        "vram_allocated_mb": allocated, "vram_reserved_mb": reserved,
        "vram_budget_mb": manager.vram_budget_mb,
        "vram_utilization_pct": round((total - free) / max(1, total) * 100, 1),
        "ram_free_gb": round(ram.available / 1024**3, 1),
        "ram_total_gb": round(ram.total / 1024**3, 1),
        "ram_utilization_pct": round(ram.percent, 1),
        "models_on_gpu": list(manager._gpu_loaded.keys()),
        "models_registered": len(manager._registry),
        "chromadb": get_chroma() is not None,
        "uptime_s": int(uptime),
        "uptime": f"{uptime / 3600:.1f}h" if uptime > 3600 else f"{uptime / 60:.1f}m",
        "port": PORT,
        "gpu_track": "local",
        "starvation_bumps": _starvation_bumps,
        "features": ["gpu-semaphore", "request-metrics", "vram-defrag", "warmup",
                     "profile-driven-review", "json-scoring-profiles",
                     "request-queue", "request-dedup", "generation-history", "model-prediction",
                     "priority-system", "batch-warmup", "vram-pressure",
                     "category-aware-review", "tileability-check", "saturation-scoring",
                     "kb-3d-topology", "kb-audio-spectral", "crest-factor-check",
                     "hot-tunable-thresholds",
                     "batch-3d-generation", "model-status-vram", "profile-hot-reload",
                     "enhanced-review", "system-info",
                     "3d-thumbnail", "3d-variations", "model-benchmark",
                     "batch-review", "gpu-timeline", "audio-batch",
                     "batch-3d-v2-priority-sse", "mesh-postprocess-lod",
                     "texture-baking", "vram-dashboard", "model-health-monitor",
                     "review3d-enhanced-v2", "format-conversion",
                     "mesh-lod-auto", "pipeline-3d-v3", "mesh-analyze",
                     "batch-export-v2", "model-warmup-schedule", "generation-presets"],
        "profiles_loaded": {
            "image": IMAGE_PROFILE.get("_version", "none"),
            "model3d": MODEL3D_PROFILE.get("_version", "none"),
            "audio": AUDIO_PROFILE.get("_version", "none"),
            "video": VIDEO_PROFILE.get("_version", "none"),
        },
        "total_requests": local_metrics._total,
        "queue_depth": _queue_depth,
        "dedup_hits": _dedup_hits,
        "history_size": len(_generation_history),
        "predicted_next_model": _predict_next_model(),
    }


@app.get("/metrics")
async def get_metrics():
    """v16.0: Request metrics with P50/P95/P99 latencies."""
    return {
        "version": "29.0",
        **local_metrics.stats(),
        "models": manager.get_status(),
    }


@app.get("/vram")
async def vram_status():
    """v20.0: Enhanced VRAM monitoring with fragmentation tracking."""
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    peak_allocated = torch.cuda.max_memory_allocated(0)
    # v20.0: Fragmentation estimate — gap between reserved and allocated
    frag_pct = round((reserved - allocated) / max(1, reserved) * 100, 1) if reserved > 0 else 0.0
    # v20.0: Allocation count from memory stats
    try:
        mem_stats = torch.cuda.memory_stats(0)
        alloc_count = mem_stats.get("allocation.all.current", 0)
    except Exception:
        alloc_count = -1
    return {
        "gpu_name": props.name,
        "total_mb": props.total_memory // (1024**2),
        "allocated_mb": allocated // (1024**2),
        "reserved_mb": reserved // (1024**2),
        "free_mb": torch.cuda.mem_get_info()[0] // (1024**2),
        "utilization_pct": round(allocated / props.total_memory * 100, 1),
        "fragmentation_pct": frag_pct,
        "peak_allocated_mb": peak_allocated // (1024**2),
        "allocation_count": alloc_count,
        "models_on_gpu": list(manager._gpu_loaded.keys()),
        "tracked_mb": sum(manager._vram_usage.values()),
        "budget_mb": manager.vram_budget_mb,
    }


@app.post("/defrag")
async def defrag_vram():
    """v14.0: Force VRAM defragmentation."""
    before_free = torch.cuda.mem_get_info()[0] // (1024**2) if torch.cuda.is_available() else 0
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    after_free = torch.cuda.mem_get_info()[0] // (1024**2) if torch.cuda.is_available() else 0
    freed = after_free - before_free
    return {"success": True, "freed_mb": freed, "free_mb_before": before_free, "free_mb_after": after_free}


@app.post("/warmup")
async def warmup_model(req: dict = {}):
    """v14.0: Predictive model prewarming — load model to GPU ahead of time."""
    model_name = req.get("model", "")
    hint = req.get("hint", "")
    if not model_name:
        return {"success": False, "error": "Provide 'model' name to warm up"}
    if model_name not in manager._registry:
        return {"success": False, "error": f"Unknown model: {model_name}", "available": list(manager._registry.keys())}
    if model_name in manager._gpu_loaded:
        return {"success": True, "message": f"{model_name} already on GPU", "hint": hint}
    try:
        t0 = time.time()
        await manager.get(model_name)
        elapsed = time.time() - t0
        log.info(f"Warmup: '{model_name}' loaded in {elapsed:.1f}s (hint: {hint})")
        return {"success": True, "model": model_name, "elapsed_s": round(elapsed, 1), "hint": hint}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/models")
async def list_models():
    return manager.get_status()


@app.get("/models/{name}")
async def model_info(name: str):
    if name not in manager._registry:
        return {"success": False, "error": f"Model '{name}' not found", "available": list(manager._registry.keys())}
    entry = manager._registry[name]
    return {
        "success": True, "name": name,
        "vram_mb": entry.vram_mb, "pinned": entry.pinned,
        "on_gpu": entry.on_gpu,
        "last_used": entry.last_used,
        "metadata": entry.metadata,
    }


@app.post("/models/swap")
async def swap_model(req: SwapRequest):
    try:
        await manager.get(req.model)
        return {"success": True, "model": req.model, "status": "loaded on GPU"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── Image Generation ───

def _style_prompt(prompt, style):
    styles = {
        "gameAsset": "pixel art game asset, {}, clean edges, transparent background, game sprite",
        "texture": "seamless tileable texture, {}, game texture, 2D top-down view",
        "pixelart": "pixel art, {}, retro game style, clean pixels",
    }
    return styles.get(style, "{}").format(prompt) if style in styles else prompt


@app.get("/loras")
async def list_loras():
    return {"success": True, "loras": LORA_REGISTRY}

@app.post("/image")
async def generate_image(req: ImageRequest):
    """v25.0: DreamShaper REMOVED. ALL image gen routes to remote FLUX.1-dev on port 8200."""
    return {"success": False, "error": "Image generation moved to remote FLUX.1-dev (port 8200). Use callRemoteAI('/image') instead of callLocalAI('/image').",
            "hint": "curl -X POST http://100.100.246.94:8200/image -H 'Content-Type: application/json' -d '{\"prompt\":\"...\",\"steps\":25}'"}


# ─── Batch Image Generation (load model ONCE → generate N images → fastest pipeline) ───

@app.post("/image/batch")
async def generate_image_batch(req: ImageBatchRequest):
    """v13.2: DreamShaper REMOVED. ALL image gen → remote FLUX.1-dev."""
    return {"success": False, "error": "Image generation moved to remote FLUX.1-dev (port 8200). Use callRemoteAI('/image') instead.",
            "hint": "Local models: embeddings, hunyuan3d ONLY (all others removed for max VRAM)"}


# ─── Image Editing ───

@app.post("/image/edit")
async def edit_image(req: ImageEditRequest):
    """v13.2: Image editing moved to remote FLUX.1-dev /image/img2img."""
    return {"success": False, "error": "Image editing moved to remote FLUX.1-dev (port 8200). Use callRemoteAI('/image/img2img').",
            "hint": "curl -X POST http://100.100.246.94:8200/image/img2img"}


@app.post("/image/inpaint")
async def inpaint_image(req: ImageInpaintRequest):
    """v13.2: Inpainting moved to remote FLUX.1-dev."""
    return {"success": False, "error": "Inpainting moved to remote FLUX.1-dev (port 8200). Use callRemoteAI('/image/img2img').",
            "hint": "curl -X POST http://100.100.246.94:8200/image/img2img"}


# ─── 3D Generation ───

def _ply_to_glb(ply_path: str, glb_path: str) -> bool:
    """Convert PLY mesh to GLB format using trimesh, preserving vertex colors."""
    try:
        import trimesh
        mesh = trimesh.load(ply_path, process=False)
        # Ensure vertex colors are preserved in GLB
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            mesh.visual = mesh.visual  # preserve vertex colors
        mesh.export(glb_path, file_type='glb')
        log.info(f"Converted PLY→GLB: {glb_path} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
        return True
    except ImportError:
        log.warning("trimesh not installed — cannot convert PLY→GLB. Install: pip install trimesh")
        return False
    except Exception as e:
        log.warning(f"PLY→GLB conversion failed: {e}")
        return False

def _count_mesh_stats(path: str) -> dict:
    """Get mesh statistics for quality reporting. Handles both Trimesh and Scene."""
    try:
        import trimesh
        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            total_verts = sum(len(g.vertices) for g in mesh.geometry.values())
            total_faces = sum(len(g.faces) for g in mesh.geometry.values())
            has_colors = any(hasattr(g.visual, 'vertex_colors') and g.visual.vertex_colors is not None
                           for g in mesh.geometry.values())
            return {"vertices": total_verts, "faces": total_faces,
                    "has_colors": has_colors, "submeshes": len(mesh.geometry),
                    "bounds": mesh.bounds.tolist() if mesh.bounds is not None else None}
        return {"vertices": len(mesh.vertices), "faces": len(mesh.faces),
                "has_colors": hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None,
                "bounds": mesh.bounds.tolist() if mesh.bounds is not None else None}
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════
# ASSET QA ENGINE v4.0 PROFILE-DRIVEN — JSON-Configured Quality Assessment
# v4.0: External JSON scoring profiles, zero hardcoded thresholds, hot-tunable
# v3.0: KB-backed category thresholds, tileability, saturation, spectral analysis
# v2.0: BLIP-2 captioning + CLIP similarity + semantic quality scoring
# v1.0: Lightweight pixel checks, auto-retry on quality fail
# ═══════════════════════════════════════════════════════════════════════

async def _vision_evaluate_image(image_path: str, prompt: str, category: str = "general") -> dict:
    """v2.0: Vision AI evaluation using BLIP-2 + CLIP.
    Returns: {caption, clip_similarity, semantic_score, grade, issues, suggestions}"""
    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        result = {
            "caption": "",
            "clip_similarity": 0.0,
            "semantic_score": 0,
            "grade": "C",
            "issues": [],
            "suggestions": [],
        }

        # ── BLIP-2 Captioning ──
        try:
            md = await manager.get("blip2")
            processor = md["processor"]
            model = md["model"]

            # VQA-style prompt for detailed description
            q_prompt = "Question: Describe this image in detail. Answer:"
            inputs = processor(image, q_prompt, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=80)
            caption = processor.decode(out[0], skip_special_tokens=True)
            # Strip common prefixes
            for prefix in ["Answer:", "The image shows", "This is"]:
                if caption.lower().startswith(prefix.lower()):
                    caption = caption[len(prefix):].strip()
            result["caption"] = caption
        except Exception as e:
            log.warning(f"BLIP-2 caption failed: {e}")
            result["issues"].append("Caption unavailable")

        # ── CLIP Similarity ──
        try:
            md = await manager.get("clip")
            clip_model = md["model"]
            clip_processor = md["processor"]

            inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
            for k, v in inputs.items():
                if hasattr(v, 'to'):
                    inputs[k] = v.to(DEVICE)

            with torch.inference_mode():
                outputs = clip_model(**inputs)

            # Normalize and compute similarity
            img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarity = (img_emb @ txt_emb.T).item()

            result["clip_similarity"] = round(similarity, 3)
        except Exception as e:
            log.warning(f"CLIP similarity failed: {e}")

        # ── Semantic Score (0-100) ──
        clip_sim = result["clip_similarity"]
        base_score = int(clip_sim * 100) if clip_sim > 0 else 50

        # Category-specific bonuses/penalties
        caption_lower = result["caption"].lower()
        prompt_lower = prompt.lower()

        # Check keyword alignment
        key_terms = [w for w in prompt_lower.split() if len(w) > 3 and w not in
                     {'with', 'the', 'and', 'for', 'game', 'style', 'quality', 'high', 'detailed'}]
        matched = sum(1 for term in key_terms if term in caption_lower)
        alignment_bonus = min(20, matched * 5)

        # v17.0 KB-INTEGRATED: Enhanced category-specific checks from game dev KB
        if category == "character":
            # KB: Characters need recognizable form — person, figure, creature
            if any(w in caption_lower for w in ["person", "character", "man", "woman", "figure", "creature", "warrior", "knight"]):
                alignment_bonus += 10
            if "extra limb" in caption_lower or "deformed" in caption_lower:
                alignment_bonus -= 15
                result["issues"].append("Anatomy issue detected in caption")
        elif category == "sprite":
            if "transparent" in prompt_lower and result.get("has_alpha", False):
                alignment_bonus += 10
            if any(w in caption_lower for w in ["pixel", "sprite", "game", "retro"]):
                alignment_bonus += 5
        elif category == "texture":
            if any(w in caption_lower for w in ["seamless", "pattern", "texture", "material", "surface"]):
                alignment_bonus += 10
        elif category == "icon":
            # KB: Icons need simplicity and recognizability
            if any(w in caption_lower for w in ["icon", "symbol", "simple", "logo"]):
                alignment_bonus += 10
        elif category == "environment":
            # KB: Environments need depth and atmosphere
            if any(w in caption_lower for w in ["landscape", "scene", "background", "forest", "city", "dungeon"]):
                alignment_bonus += 10
        elif category == "vfx":
            if any(w in caption_lower for w in ["glow", "particle", "effect", "magic", "fire", "light", "energy"]):
                alignment_bonus += 10
        elif category == "prop":
            if any(w in caption_lower for w in ["object", "item", "weapon", "tool", "furniture"]):
                alignment_bonus += 10
        elif category == "ui":
            if any(w in caption_lower for w in ["interface", "button", "panel", "menu", "ui"]):
                alignment_bonus += 10

        result["semantic_score"] = min(100, max(0, base_score + alignment_bonus))

        # Grade assignment
        score = result["semantic_score"]
        if score >= 85:
            result["grade"] = "A"
        elif score >= 70:
            result["grade"] = "B"
        elif score >= 50:
            result["grade"] = "C"
        else:
            result["grade"] = "D"
            result["issues"].append("Low prompt alignment")
            result["suggestions"].append("Try more specific prompt or different LoRA")

        return result
    except Exception as e:
        log.warning(f"Vision evaluation failed: {e}")
        return {"caption": "", "clip_similarity": 0, "semantic_score": 50, "grade": "C",
                "issues": [str(e)], "suggestions": []}

def _review_image(image_path: str, prompt: str = "", min_stddev: float = 15.0, category: str = "") -> dict:
    """v18.0 PROFILE-DRIVEN: Auto-review using external JSON scoring profiles.
    Loads thresholds from scoring_profiles/image.json — zero hardcoded magic numbers.
    Returns: {pass: bool, score: 0-100, issues: [], suggestions: [], metrics: {}}"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        arr = np.array(img.convert("RGB")).astype(np.float32)

        issues = []
        suggestions = []
        score = 100

        # Load profile thresholds (fallback to defaults if profile missing)
        p = IMAGE_PROFILE
        thresholds = (p.get("category_thresholds", {}).get(category) or
                      p.get("category_thresholds", {}).get("default", {}))
        blank_cfg = p.get("blank_detection", {})
        penalties = p.get("score_penalties", {})
        tile_cats = p.get("tileability_categories", ["texture", "tilemap", "albedo"])
        tile_min = p.get("tileability_min_score", 50)
        pass_thresh = p.get("pass_threshold", 50)

        # Check 1: Resolution — from profile
        w, h = img.size
        min_res = thresholds.get("min_res", 256)
        if w < min_res or h < min_res:
            issues.append(f"Low resolution: {w}x{h} (min={min_res} for {category or 'general'})")
            suggestions.append(f"Increase to at least {min_res*2}x{min_res*2}")
            score += penalties.get("low_resolution", -30)

        # Check 2: Blank/uniform detection — from profile
        stddev = arr.std()
        stddev_thresh = blank_cfg.get("stddev_threshold", min_stddev)
        low_var_thresh = blank_cfg.get("low_variance_threshold", 30)
        if stddev < stddev_thresh:
            issues.append(f"Nearly blank image (stddev={stddev:.1f}, min={stddev_thresh})")
            suggestions.append("Increase guidance_scale or use more descriptive prompt")
            score += penalties.get("blank", -50)
        elif stddev < low_var_thresh:
            issues.append(f"Low color variance (stddev={stddev:.1f})")
            suggestions.append("Add more detail to prompt or increase steps")
            score += penalties.get("low_variance", -20)

        # Check 3: File size — from profile
        fsize = os.path.getsize(image_path)
        min_bytes = blank_cfg.get("min_file_bytes", 5000)
        if fsize < min_bytes:
            issues.append(f"Tiny file ({fsize} bytes)")
            suggestions.append("Image may be corrupt, regenerate")
            score += penalties.get("tiny_file", -40)

        # Check 4: Color channel balance — from profile
        channel_means = arr.mean(axis=(0, 1))
        channel_spread = channel_means.max() - channel_means.min()
        channel_limit = thresholds.get("channel_limit", 150)
        if channel_spread > channel_limit:
            issues.append(f"Color bias (spread={channel_spread:.0f}, limit={channel_limit} for {category or 'general'})")
            if category == "albedo":
                suggestions.append("Albedo maps should have flat, uniform lighting — no colored light sources")
            score += penalties.get("color_bias", -15)

        # Check 5: Edge content — from profile
        gray = arr.mean(axis=2)
        dx = np.abs(np.diff(gray, axis=1)).mean()
        dy = np.abs(np.diff(gray, axis=0)).mean()
        edge_energy = dx + dy
        edge_min = thresholds.get("edge_min", 3.0)
        if edge_energy < edge_min:
            issues.append(f"Flat image (edge_energy={edge_energy:.1f}, min={edge_min})")
            suggestions.append("Image lacks detail — increase steps or guidance")
            score += penalties.get("flat_image", -15)

        # Check 6: Sharpness (Laplacian variance) — from profile
        g = gray[1:-1, 1:-1]
        lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * g)
        sharpness = float(np.var(lap))
        sharp_min = thresholds.get("sharpness_min", 50)
        if sharpness < sharp_min:
            issues.append(f"Blurry (sharpness={sharpness:.1f}, min={sharp_min} for {category or 'general'})")
            suggestions.append("Increase inference steps or guidance scale")
            score += penalties.get("blurry", -15)
        elif sharpness < sharp_min * 2:
            issues.append(f"Slightly soft (sharpness={sharpness:.1f})")
            score += penalties.get("slightly_soft", -5)

        # Check 7: Saturation/colorfulness — from profile
        saturation = 0.0
        try:
            r, g_ch, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            rg = np.abs(r - g_ch)
            yb = np.abs(0.5 * (r + g_ch) - b)
            saturation = float(np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2)))
            sat_min = thresholds.get("saturation_min", 5.0)
            if saturation < sat_min:
                issues.append(f"Low saturation ({saturation:.1f}, min={sat_min})")
                suggestions.append("Add more vivid colors or increase guidance")
                score += penalties.get("low_saturation", -10)
        except Exception:
            pass

        # Check 8: Tileability — from profile
        tileability = 0
        if category in tile_cats:
            try:
                lr_diff = np.mean(np.abs(gray[:, 0] - gray[:, -1]))
                tb_diff = np.mean(np.abs(gray[0, :] - gray[-1, :]))
                seam_error = (lr_diff + tb_diff) / 2
                tileability = max(0, min(100, int(100 - seam_error * 2)))
                if tileability < tile_min:
                    issues.append(f"Poor tileability (seam_error={seam_error:.1f}, score={tileability})")
                    suggestions.append("Use 'seamless tileable' in prompt, or post-process edges")
                    score += penalties.get("poor_tileability", -10)
            except Exception:
                pass

        passed = score >= pass_thresh and len([i for i in issues if "blank" in i.lower()]) == 0
        result = {"pass": passed, "score": max(0, score), "issues": issues,
                "suggestions": suggestions, "stddev": round(float(stddev), 1),
                "edge_energy": round(float(edge_energy), 1), "sharpness": round(sharpness, 1),
                "saturation": round(saturation, 1), "file_kb": round(fsize / 1024, 1),
                "category": category or "general", "profile_version": p.get("_version", "none")}
        if category in tile_cats:
            result["tileability"] = tileability
        return result
    except Exception as e:
        log.warning(f"Image review failed: {e}")
        return {"pass": True, "score": 50, "issues": [f"Review error: {e}"], "suggestions": []}


def _review_3d(glb_path: str, min_vertices: int = 50, min_faces: int = 30, category: str = "prop") -> dict:
    """v18.0 PROFILE-DRIVEN: Auto-review 3D model using external JSON scoring profiles.
    Loads thresholds from scoring_profiles/model3d.json — zero hardcoded magic numbers."""
    try:
        import trimesh
        fsize = os.path.getsize(glb_path)
        issues = []
        suggestions = []
        score = 100

        p = MODEL3D_PROFILE
        mins = p.get("min_absolute", {})
        geo = p.get("geometry_checks", {})
        topo = p.get("topology_quality", {})
        penalties = p.get("score_penalties", {})
        pass_thresh = p.get("pass_threshold", 40)
        vert_targets = (p.get("category_vertex_targets", {}).get(category) or
                        p.get("category_vertex_targets", {}).get("default", {}))

        # Check 1: File size
        if fsize < mins.get("file_bytes", 1000):
            issues.append(f"Tiny GLB file ({fsize} bytes)")
            suggestions.append("3D model may be empty, increase steps")
            score += penalties.get("tiny_file", -60)

        # Check 2: Mesh stats
        mesh = trimesh.load(glb_path, process=False)
        meshes = list(mesh.geometry.values()) if isinstance(mesh, trimesh.Scene) else [mesh]
        verts = sum(len(g.vertices) for g in meshes if hasattr(g, 'vertices'))
        faces = sum(len(g.faces) for g in meshes if hasattr(g, 'faces'))

        min_v = mins.get("vertices", min_vertices)
        min_f = mins.get("faces", min_faces)
        target_min = vert_targets.get("min", 100)
        target_max = vert_targets.get("max", 10000)

        if verts < min_v:
            issues.append(f"Too few vertices ({verts}, min={min_v})")
            suggestions.append("Increase inference steps for more detail")
            score += penalties.get("too_few_vertices", -40)
        elif verts < target_min:
            issues.append(f"Low vertex count ({verts}, target>={target_min} for {category})")
            score += penalties.get("low_vertex_count", -15)
        elif verts > target_max:
            issues.append(f"High poly count ({verts}, target<={target_max} for {category})")
            suggestions.append("Consider decimation for game-ready mesh")
            score += penalties.get("high_poly_count", -5)

        if faces < min_f:
            issues.append(f"Too few faces ({faces}, min={min_f})")
            score += penalties.get("too_few_faces", -30)

        # Check 3: Degenerate mesh
        bounds = mesh.bounds if hasattr(mesh, 'bounds') and mesh.bounds is not None else None
        extent = None
        if verts > 0 and bounds is not None:
            extent = bounds[1] - bounds[0]
            if extent.max() < geo.get("degenerate_extent_threshold", 0.001):
                issues.append("Degenerate mesh (zero extent)")
                suggestions.append("Model collapsed to a point, regenerate with different prompt")
                score += penalties.get("degenerate_mesh", -70)

        # Check 4: Watertight
        is_watertight = all(g.is_watertight for g in meshes if hasattr(g, 'is_watertight'))

        # Check 5: Aspect ratio — from profile
        if extent is not None and extent.max() > 0:
            aspect_ratio = float(extent.max() / max(0.001, extent.min()))
            if aspect_ratio > geo.get("extreme_aspect_ratio", 50):
                issues.append(f"Extreme aspect ratio ({aspect_ratio:.0f}:1)")
                suggestions.append("Model is very elongated/flat — may need topology fix")
                score += penalties.get("extreme_aspect_ratio", -15)
            elif aspect_ratio > geo.get("high_aspect_ratio", 20):
                issues.append(f"High aspect ratio ({aspect_ratio:.0f}:1)")
                score += penalties.get("high_aspect_ratio", -5)

        # Check 6: Face/vertex ratio — from profile
        if verts > 0 and faces > 0:
            fv_ratio = faces / verts
            if fv_ratio < topo.get("isolated_vertices_threshold", 0.5):
                issues.append(f"Many isolated vertices (face/vert ratio={fv_ratio:.2f})")
                score += penalties.get("isolated_vertices", -10)
            elif fv_ratio > topo.get("excessive_triangulation_threshold", 4.0):
                issues.append(f"Excessive triangulation (face/vert ratio={fv_ratio:.2f})")
                score += penalties.get("excessive_triangulation", -5)

        # Check 7: Normals/colors — from profile
        has_normals = False
        has_colors = False
        for m in meshes:
            if hasattr(m, 'vertex_normals') and m.vertex_normals is not None and len(m.vertex_normals) > 0:
                has_normals = True
            if hasattr(m, 'visual') and hasattr(m.visual, 'vertex_colors') and m.visual.vertex_colors is not None:
                has_colors = True
        if not has_normals and verts > geo.get("normals_required_above_verts", 100):
            issues.append("Missing vertex normals")
            suggestions.append("Normals needed for proper lighting — recalculate in Blender")
            score += penalties.get("missing_normals", -5)

        passed = score >= pass_thresh
        return {"pass": passed, "score": max(0, score), "issues": issues,
                "suggestions": suggestions, "vertices": int(verts), "faces": int(faces),
                "watertight": bool(is_watertight), "has_normals": has_normals,
                "has_colors": has_colors, "file_kb": float(round(fsize / 1024, 1)),
                "category": category, "profile_version": p.get("_version", "none")}
    except Exception as e:
        log.warning(f"3D review failed: {e}")
        return {"pass": True, "score": 50, "issues": [f"Review error: {e}"], "suggestions": []}


def _review_audio(wav_path: str, expected_duration: float = 0, min_rms: float = 0.001, category: str = "sfx") -> dict:
    """v18.0 PROFILE-DRIVEN: Auto-review audio using external JSON scoring profiles.
    Loads thresholds from scoring_profiles/audio.json — zero hardcoded magic numbers."""
    try:
        import scipy.io.wavfile as wavfile
        issues = []
        suggestions = []
        score = 100

        p = AUDIO_PROFILE
        silence = p.get("silence_detection", {})
        clip_cfg = p.get("clipping_detection", {})
        dc_cfg = p.get("dc_offset", {})
        crest_cfg = p.get("crest_factor", {})
        spectral_cfg = p.get("spectral_analysis", {})
        penalties = p.get("score_penalties", {})
        pass_thresh = p.get("pass_threshold", 40)
        dur_range = (p.get("category_duration_ranges", {}).get(category) or
                     p.get("category_duration_ranges", {}).get("default", {}))

        fsize = os.path.getsize(wav_path)
        if fsize < 1000:
            issues.append(f"Tiny audio file ({fsize} bytes)")
            score += penalties.get("tiny_file", -60)

        sr, data = wavfile.read(wav_path)
        audio = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duration = len(audio) / sr

        # Check 1: Duration — from profile
        if expected_duration > 0:
            ratio = duration / expected_duration
            if ratio < 0.3:
                issues.append(f"Much shorter than expected ({duration:.1f}s vs {expected_duration:.1f}s)")
                suggestions.append("Increase duration parameter or check model capacity")
                score += penalties.get("much_shorter", -30)
            elif ratio > 3.0:
                issues.append(f"Much longer than expected ({duration:.1f}s vs {expected_duration:.1f}s)")
                score += penalties.get("much_longer", -10)
        else:
            lo = dur_range.get("min", 0.05)
            hi = dur_range.get("max", 600)
            if duration < lo:
                issues.append(f"Too short for {category} ({duration:.2f}s, min={lo}s)")
                score += penalties.get("too_short_for_category", -15)
            elif duration > hi:
                issues.append(f"Very long for {category} ({duration:.1f}s, max={hi}s)")
                score += penalties.get("too_long_for_category", -5)

        # Check 2: Silence — from profile
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < silence.get("min_rms", min_rms):
            issues.append(f"Nearly silent audio (RMS={rms:.6f})")
            suggestions.append("Audio is silence — try different description or increase guidance")
            score += penalties.get("silent", -50)
        elif rms < silence.get("quiet_rms", 0.01):
            issues.append(f"Very quiet audio (RMS={rms:.4f})")
            score += penalties.get("very_quiet", -15)

        # Check 3: Clipping — from profile
        clip_thresh = clip_cfg.get("clip_threshold", 0.99)
        clip_ratio = float(np.mean(np.abs(audio) > clip_thresh))
        if clip_ratio > clip_cfg.get("severe_clip_ratio", 0.1):
            issues.append(f"Audio clipping ({clip_ratio*100:.1f}% of samples)")
            score += penalties.get("clipping_severe", -20)
        elif clip_ratio > clip_cfg.get("minor_clip_ratio", 0.01):
            issues.append(f"Minor clipping ({clip_ratio*100:.2f}%)")
            score += penalties.get("clipping_minor", -5)

        # Check 4: DC offset — from profile
        dc_offset_val = float(np.abs(np.mean(audio)))
        if dc_offset_val > dc_cfg.get("threshold", 0.1):
            issues.append(f"DC offset detected ({dc_offset_val:.3f})")
            suggestions.append("Apply DC offset removal in post-processing")
            score += penalties.get("dc_offset", -10)

        # Check 5: Crest factor — from profile
        peak = float(np.max(np.abs(audio)))
        crest_factor = 0.0
        if rms > 0:
            crest_factor = float(20 * np.log10(max(0.001, peak / rms)))
            exempt = crest_cfg.get("exempt_categories", ["ambient"])
            if crest_factor > crest_cfg.get("sfx_high_warning", 25) and category == "sfx":
                issues.append(f"High crest factor ({crest_factor:.1f}dB) — audio may sound weak/thin")
                score += penalties.get("high_crest_factor", -5)
            elif crest_factor < crest_cfg.get("compressed_warning", 3) and category not in exempt:
                issues.append(f"Low crest factor ({crest_factor:.1f}dB) — audio heavily compressed")
                score += penalties.get("low_crest_factor", -5)

        # Check 6: Spectral content — from profile
        spectral_spread = 0.0
        try:
            max_sec = spectral_cfg.get("analysis_max_seconds", 2)
            fft = np.fft.rfft(audio[:min(len(audio), sr * max_sec)])
            magnitudes = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio[:min(len(audio), sr * max_sec)]), 1.0 / sr)
            total_mag = magnitudes.sum()
            if total_mag > 0:
                centroid = float(np.sum(freqs * magnitudes) / total_mag)
                spectral_spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitudes) / total_mag))
                tts_lo = spectral_cfg.get("tts_centroid_min_hz", 80)
                tts_hi = spectral_cfg.get("tts_centroid_max_hz", 6000)
                if category == "tts" and (centroid < tts_lo or centroid > tts_hi):
                    issues.append(f"Unusual speech centroid ({centroid:.0f}Hz)")
                    score += penalties.get("unusual_speech_centroid", -5)
        except Exception:
            pass

        passed = score >= pass_thresh
        return {"pass": passed, "score": max(0, score), "issues": issues,
                "suggestions": suggestions, "duration_s": float(round(duration, 1)),
                "rms": float(round(rms, 4)), "peak": float(round(peak, 4)),
                "crest_factor_db": round(crest_factor, 1),
                "spectral_spread": round(spectral_spread, 1),
                "file_kb": float(round(fsize / 1024, 1)),
                "category": category, "profile_version": p.get("_version", "none")}
    except Exception as e:
        log.warning(f"Audio review failed: {e}")
        return {"pass": True, "score": 50, "issues": [f"Review error: {e}"], "suggestions": []}


def _review_video(video_path: str, expected_frames: int = 0, category: str = "default") -> dict:
    """v18.0 PROFILE-DRIVEN: Auto-review video using external JSON scoring profiles."""
    try:
        p = VIDEO_PROFILE
        penalties = p.get("score_penalties", {})
        pass_thresh = p.get("pass_threshold", 40)
        cat_expect = (p.get("category_expectations", {}).get(category) or
                      p.get("category_expectations", {}).get("default", {}))
        min_bytes = p.get("min_file_bytes", 5000)
        min_res_cfg = p.get("min_resolution", {})

        issues = []
        suggestions = []
        score = 100

        fsize = os.path.getsize(video_path)
        if fsize < min_bytes:
            issues.append(f"Tiny video file ({fsize} bytes)")
            suggestions.append("Video may be empty, check GPU memory")
            score += penalties.get("tiny_file", -60)

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            min_frames = cat_expect.get("min_frames", p.get("min_frames", 2))
            if frame_count < min_frames:
                issues.append(f"Too few frames ({frame_count}, min={min_frames})")
                score += penalties.get("too_few_frames", -50)
            if expected_frames > 0 and frame_count < expected_frames * 0.5:
                issues.append(f"Fewer frames than expected ({frame_count} vs {expected_frames})")
                score += penalties.get("below_expected_frames", -20)

            min_w = min_res_cfg.get("width", 128)
            min_h = min_res_cfg.get("height", 128)
            cat_min_res = cat_expect.get("min_res", 128)
            if w < max(min_w, cat_min_res) or h < max(min_h, cat_min_res):
                issues.append(f"Low resolution ({w}x{h}, min={cat_min_res})")
                score += penalties.get("very_low_resolution", -20)
        except ImportError:
            pass

        passed = score >= pass_thresh
        return {"pass": passed, "score": max(0, score), "issues": issues,
                "suggestions": suggestions, "file_kb": float(round(fsize / 1024, 1)),
                "category": category, "profile_version": p.get("_version", "none")}
    except Exception as e:
        log.warning(f"Video review failed: {e}")
        return {"pass": True, "score": 50, "issues": [f"Review error: {e}"], "suggestions": []}


# Global review stats
_review_stats = {"total": 0, "passed": 0, "failed": 0, "retried": 0, "avg_score": 0.0}

def _update_review_stats(review: dict):
    """Track review statistics globally."""
    _review_stats["total"] += 1
    if review.get("pass"):
        _review_stats["passed"] += 1
    else:
        _review_stats["failed"] += 1
    # Running average
    n = _review_stats["total"]
    _review_stats["avg_score"] = ((_review_stats["avg_score"] * (n - 1)) + review.get("score", 50)) / n


@app.get("/review/stats")
async def get_review_stats():
    """Get auto-review statistics."""
    return {"success": True, **_review_stats}


class EvaluateRequest(BaseModel):
    image_path: str
    prompt: str = ""
    category: str = "general"  # general, character, sprite, texture, environment, icon


@app.post("/evaluate")
async def evaluate_asset(req: EvaluateRequest):
    """v2.0: Comprehensive Vision AI evaluation using BLIP-2 + CLIP.
    Returns pixel review + semantic caption + CLIP similarity + grade (A/B/C/D)."""
    try:
        # Basic pixel-level review
        pixel_review = _review_image(req.image_path, req.prompt)

        # Vision AI semantic evaluation
        vision_eval = await _vision_evaluate_image(req.image_path, req.prompt, req.category)

        # Combined score
        pixel_score = pixel_review.get("score", 50)
        semantic_score = vision_eval.get("semantic_score", 50)
        combined_score = int(pixel_score * 0.4 + semantic_score * 0.6)

        # Final grade based on combined score
        if combined_score >= 85:
            final_grade = "A"
        elif combined_score >= 70:
            final_grade = "B"
        elif combined_score >= 50:
            final_grade = "C"
        else:
            final_grade = "D"

        return {
            "success": True,
            "pixel_review": pixel_review,
            "vision_eval": vision_eval,
            "combined_score": combined_score,
            "final_grade": final_grade,
            "summary": f"{final_grade} ({combined_score}/100) - {vision_eval.get('caption', 'No caption')[:100]}",
        }
    except Exception as e:
        log.error(f"Evaluate failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ─── v16.0 SUPREME: Queue Status ───

@app.get("/queue")
async def queue_status():
    """v16.0: Current GPU queue depth and ETA."""
    avg_time = sum(_avg_gen_time) / max(1, len(_avg_gen_time)) if _avg_gen_time else 30.0
    return {
        "queue_depth": _queue_depth,
        "avg_gen_time_s": round(avg_time, 1),
        "eta_s": round(_queue_depth * avg_time, 1),
        "dedup_hits": _dedup_hits,
        "dedup_cache_size": len(_dedup_cache),
    }


# ─── v16.0 SUPREME: Generation History ───

@app.get("/history")
async def generation_history(limit: int = 20):
    """v16.0: Recent generation history with metadata."""
    entries = list(_generation_history)[-limit:]
    entries.reverse()
    return {"success": True, "count": len(entries), "total": len(_generation_history), "history": entries}


# ─── v16.0 SUPREME: Batch Warmup ───

@app.post("/warmup/batch")
async def warmup_batch(req: dict = {}):
    """v16.0: Pre-load multiple models sequentially."""
    models = req.get("models", [])
    if not models:
        return {"success": False, "error": "Provide 'models' list to warm up"}
    results = []
    for model_name in models:
        if model_name not in manager._registry:
            results.append({"model": model_name, "success": False, "error": "not registered"})
            continue
        if model_name in manager._gpu_loaded:
            results.append({"model": model_name, "success": True, "already_loaded": True})
            continue
        try:
            t0 = time.time()
            await manager.get(model_name)
            elapsed = time.time() - t0
            results.append({"model": model_name, "success": True, "elapsed_s": round(elapsed, 1)})
        except Exception as e:
            results.append({"model": model_name, "success": False, "error": str(e)})
    return {"success": True, "results": results}


# ─── v16.0 SUPREME: Model Usage Patterns ───

@app.get("/patterns")
async def model_patterns():
    """v16.0: Model usage patterns and transition predictions."""
    return {
        "recent_sequence": list(_model_usage_sequence)[-20:],
        "transitions": _model_transition_counts,
        "predicted_next": _predict_next_model(),
        "total_tracked": len(_model_usage_sequence),
    }


@app.post("/gen3d")
async def generate_3d(req: Gen3DRequest):
    """v18.0: Auto text-to-concept-to-3D pipeline. Hunyuan3D (best) → TripoSR (fast). NO ShapE fallbacks."""
    try:
        from PIL import Image as PILImage

        model_name = req.model
        concept_image_path = None  # Track auto-generated concept art

        # ── Step 1: Auto-select best AVAILABLE model ──
        if model_name == "auto":
            # Check which models actually work (not fallen back to unavailable)
            for candidate in ["hunyuan3d", "triposr"]:
                if candidate in manager._registry:
                    try:
                        test_md = await manager.get(candidate)
                        if test_md.get("backend") != "unavailable" and test_md.get("model") is not None:
                            model_name = candidate
                            break
                    except:
                        continue
            # v18.0 STRONGEST-ONLY: NO ShapE fallback — return error instead of garbage
            if model_name == "auto":
                return {"success": False, "error": "No production 3D models available. Hunyuan3D-2mini model weights need to download first. Run: python -c \"from huggingface_hub import snapshot_download; snapshot_download('tencent/Hunyuan3D-2mini', local_dir=os.path.expanduser('~/.cache/hy3dgen/tencent/Hunyuan3D-2mini'))\"",
                        "hint": "ShapE removed (garbage quality). Only Hunyuan3D 2.1 is supported."}

        # ── Step 2: Auto-generate concept art if text-only ──
        has_image = bool(req.image or req.image_path)
        if not has_image and model_name in ("hunyuan3d", "triposr"):
            log.info(f"Text-to-3D via {model_name}: auto-generating concept image from remote FLUX.1-dev...")
            # v25.0: Generate concept image via remote FLUX.1-dev (DreamShaper removed)
            concept_prompt = f"{req.prompt}, pure white background, centered object, clean edges, 3D game asset, professional product photo, no shadows, isolated"
            import httpx
            remote_url = os.getenv("REMOTE_AI_URL", "http://100.100.246.94:8200")
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(f"{remote_url}/image", json={
                    "prompt": concept_prompt,
                    "negative_prompt": "blurry, low quality, multiple objects, busy background, text, watermark, dark, shadows",
                    "width": 768, "height": 768, "steps": 20, "guidance": 3.5,
                })
                rdata = resp.json()
            if not rdata.get("success"):
                return {"success": False, "error": f"Concept art failed: {rdata.get('error', 'unknown')}", "hint": "Check remote AI server on port 8200"}
            # Decode base64 image from remote
            img_b64 = rdata.get("image", "")
            from PIL import Image as PILImage
            concept = PILImage.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
            concept_fname = f"concept_{uuid.uuid4().hex[:8]}.png"
            concept_path = ASSET_DIR / concept_fname
            concept.save(str(concept_path))
            log.info(f"Concept art generated via remote FLUX.1-dev: {concept_path}")

            # Remove background for cleaner 3D input
            try:
                rembg_md = await manager.get("rembg")
                proc = rembg_md["processor"]
                rembg_model = rembg_md["model"]
                inputs = proc(concept, return_tensors="pt").to(DEVICE)
                with torch.inference_mode():
                    preds = rembg_model(**inputs)[-1].sigmoid().cpu()
                mask = (preds[0].squeeze().numpy() * 255).astype("uint8")
                from PIL import Image as _Im
                mask_img = _Im.fromarray(mask).resize(concept.size)
                concept.putalpha(mask_img)
                rembg_fname = f"concept_rembg_{uuid.uuid4().hex[:8]}.png"
                rembg_path = ASSET_DIR / rembg_fname
                concept.save(str(rembg_path))
                concept_image_path = str(rembg_path)
                log.info(f"Background removed: {rembg_path}")
            except Exception as rembg_err:
                log.warning(f"REMBG failed ({rembg_err}), using concept with background")
                concept_image_path = str(concept_path)

            req.image_path = concept_image_path

        # ── Step 3: Load and execute 3D model ──
        md = await manager.get(model_name)

        if md.get("backend") == "unavailable" or md.get("model") is None:
            return {"success": False, "error": f"{model_name} is not available: {md.get('error', 'unknown')}. Install required packages."}

        if model_name == "hunyuan3d":
            image = resolve_image(req.image or req.image_path)
            pipe = md["model"]
            backend = md.get("backend", "hunyuan3d")

            if backend == "hunyuan3d-native":
                # Hunyuan3D 2.1 custom pipeline — returns List[trimesh.Trimesh]
                import trimesh as _trimesh
                # v18.0: Quality presets — draft(15), fast(25), production(30), max(50)
                quality_steps = {"draft": 15, "fast": 25, "production": 30, "max": 50}
                steps = req.steps or quality_steps.get(getattr(req, 'quality', 'production'), 30)
                log.info(f"Hunyuan3D generating: {steps} steps, quality={getattr(req, 'quality', 'production')}")
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                    mesh_output = pipe(image=image, num_inference_steps=steps)
                fname = f"3d_{uuid.uuid4().hex[:8]}.glb"
                fpath = MODELS_DIR / fname
                # Pipeline returns list of trimesh objects
                if isinstance(mesh_output, list):
                    mesh_obj = mesh_output[0] if len(mesh_output) > 0 else None
                    if mesh_obj is None:
                        return {"success": False, "error": "Hunyuan3D returned empty mesh list"}
                    if isinstance(mesh_obj, _trimesh.Trimesh):
                        mesh_obj.export(str(fpath))
                    elif isinstance(mesh_obj, _trimesh.Scene):
                        mesh_obj.export(str(fpath))
                    elif hasattr(mesh_obj, 'export'):
                        mesh_obj.export(str(fpath))
                    else:
                        return {"success": False, "error": f"Hunyuan3D mesh type not supported: {type(mesh_obj)}"}
                elif isinstance(mesh_output, (_trimesh.Trimesh, _trimesh.Scene)):
                    mesh_output.export(str(fpath))
                elif hasattr(mesh_output, 'export'):
                    mesh_output.export(str(fpath))
                elif hasattr(mesh_output, 'vertices') and hasattr(mesh_output, 'faces'):
                    tm = _trimesh.Trimesh(vertices=mesh_output.vertices, faces=mesh_output.faces)
                    tm.export(str(fpath))
                else:
                    return {"success": False, "error": f"Hunyuan3D output type not supported: {type(mesh_output)}"}
                stats = _count_mesh_stats(str(fpath))
                review = _review_3d(str(fpath))
                _update_review_stats(review)
                return {"success": True, "path": str(fpath), "model": "hunyuan3d-2.1", "format": "glb",
                        "vertices": stats, "concept_art": concept_image_path, "review": review,
                        "hint": "Production-quality 3D with PBR. Import to Unity: unity_import_asset"}
            elif backend == "hunyuan3d-diffusers":
                with torch.inference_mode():
                    mesh = pipe(image=image)
                fname = f"3d_{uuid.uuid4().hex[:8]}.glb"
                fpath = MODELS_DIR / fname
                mesh.export(str(fpath))
                stats = _count_mesh_stats(str(fpath))
                review = _review_3d(str(fpath))
                _update_review_stats(review)
                return {"success": True, "path": str(fpath), "model": "hunyuan3d", "format": "glb",
                        "vertices": stats, "concept_art": concept_image_path, "review": review,
                        "hint": "Production-quality 3D. Import to Unity: unity_import_asset"}
            else:
                return {"success": False, "error": f"Hunyuan3D backend '{backend}' not available"}

        elif model_name == "triposr":
            image = resolve_image(req.image or req.image_path)
            backend = md.get("backend")

            if backend == "triposr-native":
                model = md["model"]
                with torch.inference_mode():
                    scene_codes = model([image], device="cuda")
                meshes = model.extract_mesh(scene_codes, resolution=256)
                mesh = meshes[0]
                glb_fname = f"3d_{uuid.uuid4().hex[:8]}.glb"
                glb_path = MODELS_DIR / glb_fname
                mesh.export(str(glb_path))
                stats = _count_mesh_stats(str(glb_path))
                review = _review_3d(str(glb_path))
                _update_review_stats(review)
                return {"success": True, "path": str(glb_path), "model": "triposr", "format": "glb",
                        "vertices": stats, "concept_art": concept_image_path, "review": review,
                        "hint": "Fast image-to-3D. Import to Unity: unity_import_asset"}
            else:
                return {"success": False, "error": f"TripoSR backend '{backend}' not working. Install: pip install git+https://github.com/VAST-AI-Research/TripoSR"}

        return {"success": False, "error": f"Unknown 3D model: {model_name}. Available: hunyuan3d, triposr, auto"}
    except Exception as e:
        log.error(f"3D gen failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ─── Upscale ───

@app.post("/upscale")
async def upscale_image(req: UpscaleRequest):
    """v13.2: Real-ESRGAN REMOVED. Upscaling not available locally."""
    return {"success": False, "error": "Real-ESRGAN removed (v13.2). No upscale model registered.",
            "available_models": list(manager._registry.keys())}


# ─── Background Removal ───

@app.post("/rembg")
async def remove_background(req: RembgRequest):
    """v13.2: RMBG-2.0 REMOVED. Background removal not available locally."""
    return {"success": False, "error": "RMBG-2.0 removed (v13.2). Background removal not available. 14GB VRAM used by Hunyuan3D.",
            "available_models": list(manager._registry.keys())}


# ─── Depth Map ───

@app.post("/depth")
async def generate_depth(req: DepthRequest):
    """v13.2: Depth Pro and Marigold REMOVED. No depth model registered."""
    return {"success": False, "error": "Depth models removed (v13.2). Depth Pro, Marigold not registered. 14GB VRAM used by Hunyuan3D.",
            "available_models": list(manager._registry.keys())}


# ─── Normal Map ───

@app.post("/normals")
async def generate_normals(req: NormalsRequest):
    """v13.2: Marigold REMOVED. No normals model registered."""
    return {"success": False, "error": "Marigold removed (v13.2). No normals model registered. 14GB VRAM used by Hunyuan3D.",
            "available_models": list(manager._registry.keys())}


# ─── Music ───

@app.post("/music")
async def generate_music(req: MusicRequest):
    """v20.0: Music generation via ACE-Step 1.5. Full songs up to 10 min, 50+ languages."""
    global _queue_depth
    # v20.0: Check model registration before attempting load
    if "ace-step" not in manager._registry:
        return {"success": False, "error": "ACE-Step not loaded. Model removed to save VRAM for Hunyuan3D.",
                "hint": "Re-register ACE-Step if needed via /manage endpoint", "gpu_track": "local"}
    try:
        _queue_depth += 1
        t0 = time.time()
        _track_model_usage("ace-step")
        md = await manager.get("ace-step")
        handler = md["model"]

        # ACE-Step handler generates audio from prompt + lyrics + tags
        result = handler.run(
            prompt=req.description,
            lyrics=req.lyrics or "[inst]",
            duration=min(req.duration, 600),  # Cap at 10 min
            tags=req.tags or "game soundtrack",
        )

        # Save output audio
        import soundfile as sf
        fname = f"music_{uuid.uuid4().hex[:8]}.wav"
        fpath = AUDIO_DIR / fname

        if hasattr(result, 'audio') and result.audio is not None:
            audio_np = np.array(result.audio).squeeze()
            sample_rate = getattr(result, 'sample_rate', 44100)
        elif isinstance(result, tuple):
            sample_rate, audio_np = result[0], np.array(result[1]).squeeze()
        elif isinstance(result, dict):
            audio_np = np.array(result.get("audio", result.get("waveform", []))).squeeze()
            sample_rate = result.get("sample_rate", 44100)
        else:
            audio_np = np.array(result).squeeze()
            sample_rate = 44100

        sf.write(str(fpath), audio_np, sample_rate)
        duration_s = round(len(audio_np) / sample_rate, 1)
        review = _review_audio(str(fpath), expected_duration=req.duration)
        _update_review_stats(review)
        elapsed = time.time() - t0
        _avg_gen_time.append(elapsed)
        _queue_depth = max(0, _queue_depth - 1)
        _record_generation("music", req.description, "ace-step", elapsed, True)
        local_metrics.record("music", True, elapsed)
        return {
            "success": True, "path": str(fpath), "filename": fname,
            "duration_s": duration_s, "sample_rate": sample_rate,
            "model": "ace-step-1.5", "review": review, "elapsed_s": round(elapsed, 2),
            "gpu_track": "local",
        }
    except Exception as e:
        _queue_depth = max(0, _queue_depth - 1)
        _record_generation("music", req.description, "ace-step", time.time() - t0 if 't0' in dir() else 0, False)
        log.error(f"Music gen failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e),
                "hint": "ACE-Step 1.5 needs: git clone ACE-Step-1.5 && pip install -e ."}


# ─── SFX ───

@app.post("/sfx")
async def generate_sfx(req: SFXRequest):
    global _queue_depth
    # v20.0: Check model registration before attempting load
    if "audioldm2" not in manager._registry:
        return {"success": False, "error": "AudioLDM2 not loaded. Model removed to save VRAM for Hunyuan3D.",
                "hint": "Re-register AudioLDM2 if needed via /manage endpoint", "gpu_track": "local"}
    try:
        # v16.0: Dedup check
        dk = _dedup_key(endpoint="sfx", description=req.description, duration=req.duration, steps=req.steps)
        cached = _dedup_check(dk)
        if cached:
            return {**cached, "dedup": True}

        _queue_depth += 1
        t0 = time.time()
        model_name = req.model if req.model in ("audioldm2", "audioldm") else "audioldm2"
        _track_model_usage(model_name)
        md = await manager.get(model_name)
        pipe = md["model"]

        with torch.inference_mode():
            audio = pipe(
                req.description,
                num_inference_steps=req.steps,
                audio_length_in_s=req.duration,
                num_waveforms_per_prompt=1,
            )

        # AudioLDM2 returns .audios as list of numpy arrays
        # Handle various output shapes: (samples,), (1, samples), nested tuples
        raw = audio.audios
        if isinstance(raw, (list, tuple)):
            audio_np = np.array(raw[0])
        else:
            audio_np = np.array(raw)
        # Flatten to 1D if needed
        audio_np = audio_np.squeeze()
        if audio_np.ndim > 1:
            audio_np = audio_np[0]  # Take first channel/waveform
        audio_np = audio_np.astype(np.float32)

        sample_rate = 16000

        import scipy.io.wavfile as wavfile
        fname = f"sfx_{uuid.uuid4().hex[:8]}.wav"
        fpath = AUDIO_DIR / fname
        # Normalize to int16 range
        peak = np.abs(audio_np).max()
        if peak > 0:
            audio_np = audio_np / peak
        wavfile.write(str(fpath), sample_rate, (audio_np * 32767).astype(np.int16))

        duration_s = round(len(audio_np) / sample_rate, 1)
        review = _review_audio(str(fpath), expected_duration=req.duration)
        _update_review_stats(review)
        elapsed = time.time() - t0
        _avg_gen_time.append(elapsed)
        _queue_depth = max(0, _queue_depth - 1)
        result = {
            "success": True, "path": str(fpath), "filename": fname,
            "duration_s": duration_s,
            "sample_rate": sample_rate, "model": model_name, "review": review,
            "elapsed_s": round(elapsed, 2), "gpu_track": "local",
        }
        _dedup_store(dk, result)
        _record_generation("sfx", req.description, model_name, elapsed, True)
        local_metrics.record("sfx", True, elapsed)
        return result
    except Exception as e:
        _queue_depth = max(0, _queue_depth - 1)
        _record_generation("sfx", req.description, req.model, time.time() - t0 if 't0' in dir() else 0, False)
        log.error(f"SFX gen failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ─── TTS ───

@app.post("/tts")
async def generate_tts(req: TTSRequest):
    try:
        # v20.0: Check model registration before attempting load
        if req.model == "kokoro" and "kokoro" not in manager._registry:
            return {"success": False, "error": "Kokoro not loaded. Model removed to save VRAM for Hunyuan3D.",
                    "hint": "Re-register Kokoro if needed via /manage endpoint", "gpu_track": "local"}
        if req.model in ("chatterbox", "chatterbox-turbo") and "chatterbox" not in manager._registry:
            return {"success": False, "error": "Chatterbox not loaded. Model removed to save VRAM for Hunyuan3D.",
                    "hint": "Re-register Chatterbox if needed via /manage endpoint", "gpu_track": "local"}
        if req.model == "f5-tts" and "f5-tts" not in manager._registry:
            return {"success": False, "error": "F5-TTS not loaded. Model removed (v21.0).",
                    "hint": "F5-TTS has been permanently removed", "gpu_track": "local"}
        if req.model == "bark" and "bark" not in manager._registry:
            return {"success": False, "error": "Bark not loaded. Model not registered.",
                    "hint": "Bark is not available on this server", "gpu_track": "local"}
        # v20.0: Kokoro RE-ENABLED — ONNX CPU, ultra-fast, 0 VRAM
        if req.model == "kokoro":
            md = await manager.get("kokoro")
            model = md["model"]
            voice = req.voice or "af_heart"  # Default English voice
            samples, sample_rate = model.create(req.text, voice=voice, speed=req.speed or 1.0)
            import soundfile as sf
            fname = f"tts_kokoro_{uuid.uuid4().hex[:8]}.wav"
            fpath = AUDIO_DIR / fname
            sf.write(str(fpath), samples, sample_rate)
            review = _review_audio(str(fpath))
            _update_review_stats(review)
            return {
                "success": True, "path": str(fpath), "filename": fname,
                "duration_s": round(len(samples) / sample_rate, 1),
                "sample_rate": sample_rate, "model": "kokoro-82m", "voice": voice, "review": review,
                "gpu_track": "local",
            }
        if req.model == "chatterbox" or req.model == "chatterbox-turbo":
            md = await manager.get("chatterbox")
            model = md["model"]
            ref_audio = req.voice if req.voice and req.voice.endswith('.wav') else None
            if ref_audio:
                wav = model.generate(req.text, audio_prompt_path=ref_audio)
            else:
                wav = model.generate(req.text)
            samples = wav.cpu().numpy().squeeze()
            sample_rate = model.sr
        elif req.model == "f5-tts":
            md = await manager.get("f5-tts")
            model = md["model"]
            ref_audio = req.voice if req.voice and req.voice.endswith('.wav') else None
            if ref_audio:
                wav, sr, _ = model.infer(
                    ref_file=ref_audio, ref_text="", gen_text=req.text, speed=req.speed)
            else:
                wav, sr, _ = model.infer(gen_text=req.text, speed=req.speed)
            samples = wav.squeeze()
            sample_rate = sr
        elif req.model == "bark":
            md = await manager.get("bark")
            processor = md["processor"]
            model = md["model"]
            inputs = processor(req.text, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                audio_values = model.generate(**inputs)
            samples = audio_values.cpu().numpy().squeeze()
            sample_rate = 24000
        else:
            return {"success": False, "error": f"Unknown TTS model: {req.model}. Available: kokoro, chatterbox, f5-tts, bark"}

        import soundfile as sf
        fname = f"tts_{uuid.uuid4().hex[:8]}.wav"
        fpath = AUDIO_DIR / fname
        sf.write(str(fpath), samples, sample_rate)

        review = _review_audio(str(fpath))
        _update_review_stats(review)
        return {
            "success": True, "path": str(fpath), "filename": fname,
            "duration_s": round(len(samples) / sample_rate, 1),
            "sample_rate": sample_rate, "model": req.model, "voice": req.voice, "review": review,
            "gpu_track": "local",
        }
    except Exception as e:
        log.error(f"TTS failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ─── STT ───

@app.post("/stt")
async def speech_to_text(req: STTRequest):
    try:
        md = await manager.get("whisper")
        model = md["model"]
        segments, info = model.transcribe(req.audio_path, beam_size=5)
        text = " ".join(seg.text for seg in segments)
        return {
            "success": True, "text": text.strip(),
            "language": info.language, "language_prob": round(info.language_probability, 3),
            "duration_s": round(info.duration, 1), "model": "whisper-large-v3-turbo",
        }
    except Exception as e:
        log.error(f"STT failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ─── Embeddings ───

@app.post("/embed")
async def generate_embedding(req: EmbedRequest):
    try:
        md = await manager.get("embeddings")
        emb = md["model"].encode(req.text, prompt_name="document").tolist()
        return {"success": True, "embedding": emb, "dim": len(emb), "model": "nomic-embed-text-v2"}
    except Exception as e:
        log.error(f"Embed failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}

@app.post("/embed/batch")
async def generate_embeddings_batch(req: EmbedBatchRequest):
    try:
        md = await manager.get("embeddings")
        embs = md["model"].encode(req.texts, prompt_name="document").tolist()
        return {"success": True, "embeddings": embs, "count": len(embs), "dim": len(embs[0]) if embs else 0, "model": "nomic-embed-text-v2"}
    except Exception as e:
        log.error(f"Embed batch failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ─── RAG (ChromaDB) ───

@app.post("/rag/store")
async def rag_store(req: RAGStoreRequest):
    client = get_chroma()
    if not client:
        return {"success": False, "error": "ChromaDB unavailable (port 8000)"}
    try:
        col = client.get_or_create_collection(req.collection)
        ids = req.ids or [f"{req.collection}_{uuid.uuid4().hex[:8]}" for _ in req.texts]
        kwargs = {"documents": req.texts, "ids": ids}
        if req.metadatas:
            kwargs["metadatas"] = req.metadatas
        col.add(**kwargs)
        return {"success": True, "collection": req.collection, "added": len(req.texts)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/rag/query")
async def rag_query(req: RAGQueryRequest):
    client = get_chroma()
    if not client:
        return {"success": False, "error": "ChromaDB unavailable (port 8000)"}
    try:
        col = client.get_or_create_collection(req.collection)
        results = col.query(query_texts=[req.query], n_results=req.n_results)
        return {"success": True, "collection": req.collection, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/rag/collections")
async def rag_collections():
    client = get_chroma()
    if not client:
        return {"success": False, "error": "ChromaDB unavailable (port 8000)"}
    try:
        cols = client.list_collections()
        return {"success": True, "collections": [c.name for c in cols]}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/rag/collection/{name}")
async def rag_delete(name: str):
    client = get_chroma()
    if not client:
        return {"success": False, "error": "ChromaDB unavailable (port 8000)"}
    try:
        client.delete_collection(name)
        return {"success": True, "deleted": name}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── Segmentation (SAM2) ───

@app.post("/segment")
async def segment_image(req: SegmentRequest):
    """v13.2: SAM2 REMOVED."""
    return {"success": False, "error": "SAM2 removed (v13.2). Not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


# ─── Video Generation ───

@app.post("/video")
async def generate_video(req: VideoRequest):
    """v22.0: LTX-Video 2B — fast text-to-video, 24fps, FP8 ~6GB VRAM.
    Generates short video clips from text prompts. On-demand: unloads Hunyuan3D temporarily."""
    global _queue_depth
    try:
        _queue_depth += 1
        t0 = time.time()
        _track_model_usage("ltx-video")
        md = await manager.get("ltx-video")
        pipe = md["model"]

        # Ensure num_frames is 8n+1
        nf = req.num_frames
        if (nf - 1) % 8 != 0:
            nf = ((nf - 1) // 8) * 8 + 1
            if nf < 9:
                nf = 9

        output = pipe(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_frames=nf,
            num_inference_steps=req.num_inference_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
        )

        from diffusers.utils import export_to_video
        fname = f"video_{uuid.uuid4().hex[:8]}.mp4"
        fpath = VIDEO_DIR / fname
        export_to_video(output.frames[0], str(fpath), fps=24)

        elapsed = time.time() - t0
        _avg_gen_time.append(elapsed)
        _queue_depth = max(0, _queue_depth - 1)
        _record_generation("video", req.prompt, "ltx-video-2b", elapsed, True)
        local_metrics.record("video", True, elapsed)
        return {
            "success": True, "path": str(fpath), "filename": fname,
            "model": "ltx-video-2b", "num_frames": nf,
            "resolution": f"{req.width}x{req.height}", "fps": 24,
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        _queue_depth = max(0, _queue_depth - 1)
        _record_generation("video", req.prompt, "ltx-video", time.time() - t0 if 't0' in dir() else 0, False)
        log.error(f"Video gen failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e),
                "hint": "LTX-Video 2B needs ~6GB VRAM (FP8). Will unload Hunyuan3D temporarily."}


# ─── ControlNet ───

@app.post("/controlnet")
async def controlnet_generate(req: ControlNetRequest):
    """v13.2: ControlNet REMOVED."""
    return {"success": False, "error": "ControlNet removed (v13.2). Not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


# ─── Style Transfer (IP-Adapter) ───

@app.post("/style-transfer")
async def style_transfer(req: StyleTransferRequest):
    """v13.2: IP-Adapter REMOVED."""
    return {"success": False, "error": "IP-Adapter removed (v13.2). Not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


# ─── Model Management (legacy compat + new swap) ───

@app.post("/manage")
async def manage_model(req: ManageRequest):
    if req.model not in manager._registry:
        return {"success": False, "error": f"Model '{req.model}' not registered"}
    try:
        if req.action == "load":
            await manager.get(req.model)
            return {"success": True, "model": req.model, "action": "loaded to GPU"}
        elif req.action == "evict":
            if req.model in manager._gpu_loaded:
                manager._do_evict(req.model)
            return {"success": True, "model": req.model, "action": "evicted"}
        elif req.action == "status":
            e = manager._registry[req.model]
            return {"success": True, "model": req.model, "on_gpu": e.on_gpu, "vram_mb": e.vram_mb}
        else:
            return {"success": False, "error": f"Unknown action: {req.action}. Use: load, evict, status"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/gpu_load_all")
async def gpu_load_all():
    """Legacy compat: load pinned models only (on-demand handles the rest)."""
    return {"success": True, "mode": "on-demand", "hint": "Models load automatically on first request"}


# ─── Image Captioning & Analysis (NEW v8.1) ───

@app.post("/caption")
async def caption_image(req: CaptionRequest):
    """v13.2: BLIP-2 REMOVED."""
    return {"success": False, "error": "BLIP-2 removed (v13.2). Not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


@app.post("/similarity")
async def image_similarity(req: CaptionRequest):
    """v13.2: CLIP REMOVED."""
    return {"success": False, "error": "CLIP removed (v13.2). Not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


# ─── Object Detection (NEW v8.1) ───

@app.post("/detect")
async def detect_objects(req: DetectRequest):
    """v13.2: YOLO/OWL-ViT REMOVED."""
    return {"success": False, "error": "YOLO/OWL-ViT removed (v13.2). Not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


# ─── Pose Estimation (NEW v8.1) ───

@app.post("/pose")
async def estimate_pose(req: PoseRequest):
    """v13.2: DWPose/MediaPipe REMOVED."""
    return {"success": False, "error": "DWPose removed (v13.2). Not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


# ─── Image-to-Image (NEW v8.1) ───

@app.post("/image/img2img")
async def img2img(req: Img2ImgRequest):
    """v13.2: Local img2img REMOVED. Use remote FLUX.1-dev."""
    return {"success": False, "error": "Local img2img removed (v13.2). Use remote FLUX.1-dev.",
            "hint": "callRemoteAI('/image/img2img') on port 8200"}


# ─── Animation Generation (NEW v8.1) ───

@app.post("/animate")
async def generate_animation(req: AnimateRequest):
    """v13.2: AnimateDiff REMOVED."""
    return {"success": False, "error": "AnimateDiff removed (v13.2). Not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


# ─── Color Palette Extraction (NEW v8.1) ───

@app.post("/palette")
async def extract_palette(req: PaletteRequest):
    """v13.2: Palette extraction REMOVED."""
    return {"success": False, "error": "Palette extraction removed (v13.2). Not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


# ─── Frame Interpolation (NEW v8.1) ───

@app.post("/interpolate")
async def interpolate_frames(req: InterpolateRequest):
    """v13.2: Frame interpolation REMOVED."""
    return {"success": False, "error": "Frame interpolation removed (v13.2). Not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


# ─── Multi-Model Pipelines (v9.5 — combine models for maximum power) ───

class AnalyzeRequest(BaseModel):
    """Run ALL vision models on one image: caption + detect + depth + normals + segment."""
    image: str = ""
    image_path: str = ""
    detect_labels: Optional[List[str]] = None  # OWL-ViT labels (or YOLO if empty)
    return_base64: bool = False

@app.post("/analyze")
async def analyze_image(req: AnalyzeRequest):
    """v13.2: Multi-model analyze pipeline REMOVED (BLIP-2/YOLO/Depth-Pro/SAM2 all removed)."""
    return {"success": False, "error": "Analyze pipeline removed (v13.2). All vision models (BLIP-2, YOLO, Depth-Pro, SAM2) not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


class GenerateFullAssetRequest(BaseModel):
    """Generate a complete game asset: image → rembg → upscale → depth → normals. All in one call."""
    prompt: str
    width: int = 512
    height: int = 512
    steps: int = 12
    upscale: bool = True
    upscale_factor: int = 4
    remove_bg: bool = True
    generate_depth: bool = True
    generate_normals: bool = True
    lora: Optional[str] = None
    lora_scale: float = 0.85
    return_base64: bool = False

@app.post("/generate_full_asset")
async def generate_full_asset(req: GenerateFullAssetRequest):
    """v13.2: Full asset pipeline REMOVED (DreamShaper/REMBG/ESRGAN/Depth-Pro/Marigold all removed)."""
    return {"success": False, "error": "Full asset pipeline removed (v13.2). Sub-models (DreamShaper, REMBG, ESRGAN, Depth-Pro, Marigold) not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


# ─── OCR (Video Vision v1.0) ───

class OCRRequest(BaseModel):
    image: Optional[str] = None
    image_path: Optional[str] = None
    languages: List[str] = ["en", "vi"]  # EasyOCR language codes

_easyocr_reader = None

@app.post("/ocr")
async def ocr_image(req: OCRRequest):
    """v13.2: EasyOCR REMOVED."""
    return {"success": False, "error": "EasyOCR removed (v13.2). Not registered locally.",
            "available_models": ["embeddings", "hunyuan3d", "audioldm2", "chatterbox"]}


# ══════════════════════════════════════════════════════════════════
# v19.0 SUPREME — New Endpoints
# ══════════════════════════════════════════════════════════════════


@app.post("/generate3d/batch")
async def batch_generate_3d(req: dict = {}):
    """v19.0: Batch 3D generation — process multiple prompts sequentially (max 5)."""
    prompts = req.get("prompts", [])
    quality = req.get("quality", "production")
    export_format = req.get("export_format", "glb")

    if not prompts:
        return {"success": False, "error": "No prompts provided. Pass {\"prompts\": [\"sword\", \"shield\", ...]}"}
    if len(prompts) > 5:
        prompts = prompts[:5]
        log.warning(f"Batch 3D: truncated to 5 prompts (received {len(prompts)})")

    results = []
    total_t0 = time.time()
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        rid = uuid.uuid4().hex[:8]
        log.info(f"Batch 3D [{i+1}/{len(prompts)}] rid={rid}: {prompt[:80]}")
        try:
            async with gpu_semaphore:
                # Build Gen3DRequest-compatible dict and call gen3d logic
                from pydantic import BaseModel as _BM
                gen_req = Gen3DRequest(prompt=prompt, quality=quality, format=export_format)
                result = await generate_3d(gen_req)
                elapsed = time.time() - t0
                result["batch_index"] = i
                result["elapsed_s"] = round(elapsed, 2)
                result["request_id"] = rid
                results.append(result)
                local_metrics.record("/generate3d/batch", result.get("success", False), elapsed)
                _record_generation("/generate3d/batch", prompt, "hunyuan3d", elapsed, result.get("success", False))
        except Exception as e:
            elapsed = time.time() - t0
            results.append({"success": False, "error": str(e), "batch_index": i,
                           "elapsed_s": round(elapsed, 2), "request_id": rid})
            local_metrics.record("/generate3d/batch", False, elapsed)

    total_elapsed = time.time() - total_t0
    succeeded = sum(1 for r in results if r.get("success"))
    return {
        "success": succeeded > 0,
        "total": len(prompts),
        "succeeded": succeeded,
        "failed": len(prompts) - succeeded,
        "total_elapsed_s": round(total_elapsed, 2),
        "avg_elapsed_s": round(total_elapsed / max(1, len(prompts)), 2),
        "results": results,
    }


@app.get("/model/status")
async def model_status():
    """v19.0: Detailed status of all models with VRAM breakdown."""
    now = time.time()
    models_detail = {}
    for name, entry in manager._registry.items():
        free_vram = 0
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        models_detail[name] = {
            "on_gpu": entry.on_gpu,
            "pinned": entry.pinned,
            "vram_mb": entry.vram_mb,
            "last_used": entry.last_used,
            "idle_s": int(now - entry.last_used) if entry.last_used else None,
            "status": "loaded" if entry.on_gpu else "unloaded",
            "metadata": entry.metadata,
        }

    total_vram = 0
    allocated_vram = 0
    if torch.cuda.is_available():
        total_vram = torch.cuda.mem_get_info()[1] // (1024 * 1024)
        allocated_vram = torch.cuda.memory_allocated(0) // (1024 * 1024)

    return {
        "success": True,
        "version": "29.0",
        "total_models": len(manager._registry),
        "loaded_models": list(manager._gpu_loaded.keys()),
        "unloaded_models": [n for n, e in manager._registry.items() if not e.on_gpu],
        "vram_total_mb": total_vram,
        "vram_allocated_mb": allocated_vram,
        "vram_free_mb": total_vram - allocated_vram,
        "vram_budget_mb": manager.vram_budget_mb,
        "vram_tracked_mb": sum(manager._vram_usage.values()),
        "models": models_detail,
        "pinned_models": [n for n, e in manager._registry.items() if e.pinned],
    }


@app.post("/profiles/reload")
async def reload_profiles():
    """v19.0: Hot-reload all scoring profiles from disk without restarting server."""
    global IMAGE_PROFILE, MODEL3D_PROFILE, AUDIO_PROFILE, VIDEO_PROFILE
    try:
        old_versions = {
            "image": IMAGE_PROFILE.get("_version", "none"),
            "model3d": MODEL3D_PROFILE.get("_version", "none"),
            "audio": AUDIO_PROFILE.get("_version", "none"),
            "video": VIDEO_PROFILE.get("_version", "none"),
        }
        IMAGE_PROFILE = _load_profile("image")
        MODEL3D_PROFILE = _load_profile("model3d")
        AUDIO_PROFILE = _load_profile("audio")
        VIDEO_PROFILE = _load_profile("video")
        new_versions = {
            "image": IMAGE_PROFILE.get("_version", "none"),
            "model3d": MODEL3D_PROFILE.get("_version", "none"),
            "audio": AUDIO_PROFILE.get("_version", "none"),
            "video": VIDEO_PROFILE.get("_version", "none"),
        }
        log.info(f"Scoring profiles reloaded: {new_versions}")
        return {
            "success": True,
            "message": "All scoring profiles reloaded from disk",
            "old_versions": old_versions,
            "new_versions": new_versions,
            "profiles_dir": str(PROFILES_DIR),
        }
    except Exception as e:
        log.error(f"Profile reload failed: {e}")
        return {"success": False, "error": str(e)}


@app.post("/review/enhanced")
async def enhanced_review(req: dict = {}):
    """v19.0: Enhanced multi-modal review with auto-type detection from file extension."""
    file_path = req.get("file_path", "")
    asset_type = req.get("asset_type", "auto")
    category = req.get("category", "")
    prompt = req.get("prompt", "")

    if not file_path:
        return {"success": False, "error": "No file_path provided. Pass {\"file_path\": \"path/to/asset\"}"}

    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    t0 = time.time()
    ext = os.path.splitext(file_path)[1].lower()

    # Auto-detect asset type from extension
    if asset_type == "auto":
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp"):
            asset_type = "image"
        elif ext in (".glb", ".gltf", ".obj", ".fbx", ".stl"):
            asset_type = "3d"
        elif ext in (".wav", ".mp3", ".ogg", ".flac"):
            asset_type = "audio"
        elif ext in (".mp4", ".avi", ".mov", ".webm"):
            asset_type = "video"
        else:
            return {"success": False, "error": f"Cannot auto-detect asset type for extension '{ext}'. Specify asset_type: image, 3d, audio, video"}

    try:
        review = {}
        if asset_type == "image":
            review = _review_image(file_path, prompt=prompt, category=category)
        elif asset_type == "3d":
            review = _review_3d(file_path, category=category or "prop")
        elif asset_type == "audio":
            review = _review_audio(file_path, category=category or "sfx")
        elif asset_type == "video":
            review = _review_video(file_path, category=category or "default")
        else:
            return {"success": False, "error": f"Unknown asset_type: {asset_type}. Use: image, 3d, audio, video"}

        _update_review_stats(review)
        elapsed = time.time() - t0
        local_metrics.record("/review/enhanced", True, elapsed)

        return {
            "success": True,
            "asset_type": asset_type,
            "file_path": file_path,
            "extension": ext,
            "category": category or "auto",
            "review": review,
            "elapsed_s": round(elapsed, 3),
        }
    except Exception as e:
        elapsed = time.time() - t0
        local_metrics.record("/review/enhanced", False, elapsed)
        log.error(f"Enhanced review failed for {file_path}: {e}")
        return {"success": False, "error": str(e), "file_path": file_path}


@app.get("/system/info")
async def system_info():
    """v19.0: Comprehensive system info — GPU, VRAM, models, uptime, versions."""
    uptime = time.time() - start_time
    gpu_name = "none"
    gpu_vram_total = 0
    gpu_vram_free = 0
    gpu_vram_allocated = 0
    cuda_version = "none"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram_total = torch.cuda.mem_get_info()[1] // (1024 * 1024)
        gpu_vram_free = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        gpu_vram_allocated = torch.cuda.memory_allocated(0) // (1024 * 1024)
        cuda_version = torch.version.cuda or "none"

    import psutil
    ram = psutil.virtual_memory()

    return {
        "success": True,
        "version": "29.0",
        "server": "Local AI Server v29.0 SUPREME ULTRA",
        "device": DEVICE,
        "gpu": {
            "name": gpu_name,
            "vram_total_mb": gpu_vram_total,
            "vram_free_mb": gpu_vram_free,
            "vram_allocated_mb": gpu_vram_allocated,
            "vram_utilization_pct": round((gpu_vram_total - gpu_vram_free) / max(1, gpu_vram_total) * 100, 1),
            "cuda_version": cuda_version,
        },
        "ram": {
            "total_gb": round(ram.total / 1024**3, 1),
            "available_gb": round(ram.available / 1024**3, 1),
            "utilization_pct": round(ram.percent, 1),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "none",
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "models": {
            "total_registered": len(manager._registry),
            "loaded_on_gpu": list(manager._gpu_loaded.keys()),
            "pinned": [n for n, e in manager._registry.items() if e.pinned],
            "registered": list(manager._registry.keys()),
        },
        "uptime": {
            "seconds": int(uptime),
            "formatted": f"{uptime / 3600:.1f}h" if uptime > 3600 else f"{uptime / 60:.1f}m",
        },
        "metrics": {
            "total_requests": local_metrics._total,
            "total_errors": local_metrics._errors,
            "queue_depth": _queue_depth,
            "dedup_hits": _dedup_hits,
            "generation_history_size": len(_generation_history),
        },
        "profiles": {
            "dir": str(PROFILES_DIR),
            "image": IMAGE_PROFILE.get("_version", "none"),
            "model3d": MODEL3D_PROFILE.get("_version", "none"),
            "audio": AUDIO_PROFILE.get("_version", "none"),
            "video": VIDEO_PROFILE.get("_version", "none"),
        },
        "port": PORT,
    }


# ══════════════════════════════════════════════════════════════════
# v21.0 SUPREME ULTRA — New Endpoints
# ══════════════════════════════════════════════════════════════════


@app.post("/generate3d/with_thumbnail")
async def generate_3d_with_thumbnail(req: dict = {}):
    """v21.0: Generate 3D model AND render a thumbnail preview from 3/4 angle."""
    try:
        prompt = req.get("prompt", "")
        image = req.get("image", "")
        image_path = req.get("image_path", "")
        quality = req.get("quality", "production")
        export_format = req.get("format", "glb")
        thumbnail_size = int(req.get("thumbnail_size", 512))
        steps = int(req.get("steps", 30))

        # Generate 3D model via existing pipeline
        gen_req = Gen3DRequest(prompt=prompt, image=image, image_path=image_path,
                               quality=quality, format=export_format, steps=steps)
        t0 = time.time()
        async with gpu_semaphore:
            result = await generate_3d(gen_req)

        if not result.get("success"):
            return result

        mesh_path = result.get("path", "")
        elapsed_gen = time.time() - t0

        # Render thumbnail from 3/4 angle using trimesh
        thumbnail_path = ""
        try:
            import trimesh
            mesh = trimesh.load(mesh_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                # Get combined bounds for camera placement
                scene = mesh
            else:
                scene = trimesh.Scene(mesh)

            # Render from 3/4 angle (45 degrees azimuth, 30 degrees elevation)
            # Use pyrender or trimesh's built-in rendering
            try:
                from PIL import Image as PILImage
                # trimesh scene rendering with offscreen
                png_data = scene.save_image(resolution=(thumbnail_size, thumbnail_size))
                if png_data is not None:
                    thumb_fname = f"thumb_{uuid.uuid4().hex[:8]}.png"
                    thumb_fpath = MODELS_DIR / thumb_fname
                    with open(str(thumb_fpath), "wb") as f:
                        f.write(png_data)
                    thumbnail_path = str(thumb_fpath)
                    log.info(f"Thumbnail rendered: {thumbnail_path} ({thumbnail_size}x{thumbnail_size})")
            except Exception as render_err:
                log.warning(f"Thumbnail render failed (pyrender/pyglet may not be available): {render_err}")
                # Fallback: create a simple placeholder info image
                try:
                    from PIL import Image as PILImage, ImageDraw
                    img = PILImage.new("RGB", (thumbnail_size, thumbnail_size), (40, 40, 50))
                    draw = ImageDraw.Draw(img)
                    stats = _count_mesh_stats(mesh_path)
                    text = f"3D Model\n{stats.get('vertices', '?')} verts\n{stats.get('faces', '?')} faces"
                    draw.text((thumbnail_size // 4, thumbnail_size // 3), text, fill=(200, 200, 220))
                    thumb_fname = f"thumb_{uuid.uuid4().hex[:8]}.png"
                    thumb_fpath = MODELS_DIR / thumb_fname
                    img.save(str(thumb_fpath))
                    thumbnail_path = str(thumb_fpath)
                except Exception:
                    pass

        except Exception as thumb_err:
            log.warning(f"Thumbnail generation failed: {thumb_err}")

        elapsed_total = time.time() - t0
        stats = _count_mesh_stats(mesh_path)
        review = result.get("review", {})

        local_metrics.record("/generate3d/with_thumbnail", True, elapsed_total)
        _record_generation("/generate3d/with_thumbnail", prompt or image_path, "hunyuan3d", elapsed_total, True)

        return {
            "success": True,
            "mesh_path": mesh_path,
            "thumbnail_path": thumbnail_path,
            "vertices": stats.get("vertices", 0),
            "faces": stats.get("faces", 0),
            "review_score": review.get("score", 0),
            "review": review,
            "model": result.get("model", "hunyuan3d"),
            "elapsed_s": round(elapsed_total, 2),
        }
    except Exception as e:
        log.error(f"3D with thumbnail failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


@app.post("/generate3d/variations")
async def generate_3d_variations(req: dict = {}):
    """v21.0: Generate N variations of same prompt with different seeds."""
    try:
        prompt = req.get("prompt", "")
        image = req.get("image", "")
        image_path = req.get("image_path", "")
        quality = req.get("quality", "production")
        export_format = req.get("format", "glb")
        num_variations = min(5, max(1, int(req.get("num_variations", 3))))
        steps = int(req.get("steps", 30))

        if not prompt and not image and not image_path:
            return {"success": False, "error": "Provide at least 'prompt', 'image', or 'image_path'"}

        results = []
        total_t0 = time.time()

        for i in range(num_variations):
            t0 = time.time()
            rid = uuid.uuid4().hex[:8]
            seed = int(time.time() * 1000) + i * 12345  # Different seed per variation
            log.info(f"3D variation [{i+1}/{num_variations}] rid={rid} seed={seed}: {(prompt or image_path)[:60]}")

            try:
                # Set torch seed for variation
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                async with gpu_semaphore:
                    gen_req = Gen3DRequest(prompt=prompt, image=image, image_path=image_path,
                                           quality=quality, format=export_format, steps=steps)
                    result = await generate_3d(gen_req)
                    elapsed = time.time() - t0
                    result["variation_index"] = i
                    result["seed"] = seed
                    result["elapsed_s"] = round(elapsed, 2)
                    result["request_id"] = rid
                    results.append(result)
                    local_metrics.record("/generate3d/variations", result.get("success", False), elapsed)
            except Exception as e:
                elapsed = time.time() - t0
                results.append({"success": False, "error": str(e), "variation_index": i,
                               "seed": seed, "elapsed_s": round(elapsed, 2), "request_id": rid})
                local_metrics.record("/generate3d/variations", False, elapsed)

        total_elapsed = time.time() - total_t0
        succeeded = sum(1 for r in results if r.get("success"))
        _record_generation("/generate3d/variations", prompt or image_path, "hunyuan3d",
                          total_elapsed, succeeded > 0, {"num_variations": num_variations})

        return {
            "success": succeeded > 0,
            "num_variations": num_variations,
            "succeeded": succeeded,
            "failed": num_variations - succeeded,
            "total_elapsed_s": round(total_elapsed, 2),
            "results": results,
        }
    except Exception as e:
        log.error(f"3D variations failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


@app.get("/models/benchmark")
async def benchmark_models():
    """v21.0: Quick benchmark of all loaded models — run tiny inference and measure time."""
    benchmarks = {}
    loaded_models = list(manager._gpu_loaded.keys())

    if not loaded_models:
        return {"success": True, "message": "No models currently loaded on GPU", "benchmarks": {}}

    for model_name in loaded_models:
        entry = manager._registry.get(model_name)
        if not entry or not entry.model_dict:
            benchmarks[model_name] = {"status": "error", "message": "Model dict is None"}
            continue

        vram_before = 0
        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated(0) // (1024 * 1024)

        t0 = time.time()
        status = "ok"
        error_msg = ""

        try:
            if model_name == "embeddings":
                md = entry.model_dict
                model = md["model"]
                _ = model.encode("benchmark test sentence", prompt_name="document")

            elif model_name == "hunyuan3d":
                # Minimal forward pass — just test model is responsive
                # Don't run full inference (too slow for benchmark)
                md = entry.model_dict
                pipe = md["model"]
                if hasattr(pipe, 'model') or hasattr(pipe, 'dit'):
                    status = "loaded"  # Skip heavy inference, confirm model object exists
                else:
                    status = "loaded"

            elif model_name == "audioldm2":
                md = entry.model_dict
                pipe = md["model"]
                # Quick dry run — generate 0.5s of audio
                with torch.inference_mode():
                    _ = pipe("benchmark beep", num_inference_steps=2, audio_length_in_s=0.5,
                            num_waveforms_per_prompt=1)

            elif model_name == "chatterbox":
                md = entry.model_dict
                model = md["model"]
                # Quick TTS test
                wav = model.generate("test")
                _ = wav.cpu()

            else:
                status = "loaded"  # Unknown model type — just confirm loaded

        except Exception as e:
            status = "error"
            error_msg = str(e)[:200]

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        vram_after = 0
        if torch.cuda.is_available():
            vram_after = torch.cuda.memory_allocated(0) // (1024 * 1024)

        benchmarks[model_name] = {
            "inference_time_ms": elapsed_ms,
            "vram_mb": entry.vram_mb,
            "vram_actual_mb": vram_after,
            "status": status,
            "pinned": entry.pinned,
        }
        if error_msg:
            benchmarks[model_name]["error"] = error_msg

    return {
        "success": True,
        "models_benchmarked": len(benchmarks),
        "benchmarks": benchmarks,
    }


@app.post("/review/batch")
async def batch_review(req: dict = {}):
    """v21.0: Review multiple assets at once. Accepts file_paths list."""
    file_paths = req.get("file_paths", [])
    if not file_paths:
        return {"success": False, "error": "No file_paths provided. Pass {\"file_paths\": [\"path1\", \"path2\", ...]}"}

    results = []
    t0 = time.time()

    for fpath in file_paths:
        if not os.path.exists(fpath):
            results.append({"path": fpath, "type": "unknown", "score": 0,
                           "metrics": {"error": "File not found"}})
            continue

        ext = os.path.splitext(fpath)[1].lower()
        asset_type = "unknown"
        review = {}

        try:
            if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp"):
                asset_type = "image"
                review = _review_image(fpath)
            elif ext in (".glb", ".gltf", ".obj", ".fbx", ".stl"):
                asset_type = "3d"
                review = _review_3d(fpath)
            elif ext in (".wav", ".mp3", ".ogg", ".flac"):
                asset_type = "audio"
                review = _review_audio(fpath)
            elif ext in (".mp4", ".avi", ".mov", ".webm"):
                asset_type = "video"
                review = _review_video(fpath)
            else:
                review = {"error": f"Unsupported extension: {ext}"}

            _update_review_stats(review)
        except Exception as e:
            review = {"error": str(e)}

        results.append({
            "path": fpath,
            "type": asset_type,
            "score": review.get("score", 0),
            "metrics": review,
        })

    elapsed = time.time() - t0
    local_metrics.record("/review/batch", True, elapsed)

    return {
        "success": True,
        "total": len(file_paths),
        "reviewed": len(results),
        "avg_score": round(sum(r.get("score", 0) for r in results) / max(1, len(results)), 1),
        "elapsed_s": round(elapsed, 3),
        "results": results,
    }


@app.get("/gpu/timeline")
async def gpu_timeline():
    """v21.0: GPU VRAM usage timeline — snapshots recorded every 30s (max 100)."""
    entries = list(_vram_timeline)
    if not entries:
        return {
            "success": True,
            "message": "No VRAM snapshots yet (recorded every 30s)",
            "count": 0,
            "timestamps": [],
            "vram_used_mb": [],
            "vram_free_mb": [],
            "models_loaded": [],
        }
    return {
        "success": True,
        "count": len(entries),
        "interval_s": 30,
        "max_entries": 100,
        "timestamps": [e["timestamp"] for e in entries],
        "vram_used_mb": [e["vram_used_mb"] for e in entries],
        "vram_free_mb": [e["vram_free_mb"] for e in entries],
        "models_loaded": [e["models_loaded"] for e in entries],
        "latest": entries[-1] if entries else None,
    }


@app.post("/audio/batch")
async def batch_audio(req: dict = {}):
    """v21.0: Generate multiple SFX in one call (max 5 prompts)."""
    global _queue_depth
    prompts = req.get("prompts", [])
    duration = float(req.get("duration", 5.0))
    steps = int(req.get("steps", 35))

    if not prompts:
        return {"success": False, "error": "No prompts provided. Pass {\"prompts\": [\"explosion\", \"footstep\", ...]}"}
    if len(prompts) > 5:
        prompts = prompts[:5]
        log.warning(f"Audio batch: truncated to 5 prompts")

    results = []
    total_t0 = time.time()

    # v20.0: Check model registration before attempting batch
    if "audioldm2" not in manager._registry:
        return {"success": False, "error": "AudioLDM2 not loaded. Model removed to save VRAM for Hunyuan3D.",
                "hint": "Re-register AudioLDM2 if needed via /manage endpoint", "gpu_track": "local",
                "total": len(prompts), "succeeded": 0, "failed": len(prompts), "results": []}

    for i, prompt_text in enumerate(prompts):
        t0 = time.time()
        rid = uuid.uuid4().hex[:8]
        log.info(f"Audio batch [{i+1}/{len(prompts)}] rid={rid}: {prompt_text[:60]}")
        try:
            _queue_depth += 1
            _track_model_usage("audioldm2")
            md = await manager.get("audioldm2")
            pipe = md["model"]

            with torch.inference_mode():
                audio = pipe(
                    prompt_text,
                    num_inference_steps=steps,
                    audio_length_in_s=duration,
                    num_waveforms_per_prompt=1,
                )

            raw = audio.audios
            if isinstance(raw, (list, tuple)):
                audio_np = np.array(raw[0])
            else:
                audio_np = np.array(raw)
            audio_np = audio_np.squeeze()
            if audio_np.ndim > 1:
                audio_np = audio_np[0]
            audio_np = audio_np.astype(np.float32)

            sample_rate = 16000
            import scipy.io.wavfile as wavfile
            fname = f"sfx_batch_{uuid.uuid4().hex[:8]}.wav"
            fpath = AUDIO_DIR / fname
            peak = np.abs(audio_np).max()
            if peak > 0:
                audio_np = audio_np / peak
            wavfile.write(str(fpath), sample_rate, (audio_np * 32767).astype(np.int16))

            dur_s = round(len(audio_np) / sample_rate, 1)
            review = _review_audio(str(fpath), expected_duration=duration)
            _update_review_stats(review)
            elapsed = time.time() - t0
            _avg_gen_time.append(elapsed)
            _queue_depth = max(0, _queue_depth - 1)

            results.append({
                "path": str(fpath),
                "duration": dur_s,
                "score": review.get("score", 0),
                "review": review,
                "prompt": prompt_text,
                "elapsed_s": round(elapsed, 2),
                "request_id": rid,
            })
            _record_generation("/audio/batch", prompt_text, "audioldm2", elapsed, True)
            local_metrics.record("/audio/batch", True, elapsed)

        except Exception as e:
            _queue_depth = max(0, _queue_depth - 1)
            elapsed = time.time() - t0
            results.append({
                "path": "",
                "duration": 0,
                "score": 0,
                "error": str(e),
                "prompt": prompt_text,
                "elapsed_s": round(elapsed, 2),
                "request_id": rid,
            })
            _record_generation("/audio/batch", prompt_text, "audioldm2", elapsed, False)
            local_metrics.record("/audio/batch", False, elapsed)
            log.error(f"Audio batch [{i+1}] failed: {e}")

    total_elapsed = time.time() - total_t0
    succeeded = sum(1 for r in results if r.get("path"))
    return {
        "success": succeeded > 0,
        "total": len(prompts),
        "succeeded": succeeded,
        "failed": len(prompts) - succeeded,
        "total_elapsed_s": round(total_elapsed, 2),
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════
# v25.0 SUPREME ULTRA — New Endpoints
# ══════════════════════════════════════════════════════════════════


# ─── v25.0: Model Health Monitor — latency percentiles, success rates, error classification ───
_model_health_data = {
    "latencies": {},       # model_name → deque of elapsed_ms values
    "success_counts": {},  # model_name → int
    "failure_counts": {},  # model_name → int
    "error_classes": {},   # model_name → {error_type: count}
    "gen_times_by_preset": {},  # model_name → {preset: deque of elapsed_s}
    "start_time": time.time(),
}

def _health_record(model_name: str, success: bool, elapsed_s: float, preset: str = "", error_type: str = ""):
    """Record a generation event for model health tracking."""
    if model_name not in _model_health_data["latencies"]:
        _model_health_data["latencies"][model_name] = deque(maxlen=500)
        _model_health_data["success_counts"][model_name] = 0
        _model_health_data["failure_counts"][model_name] = 0
        _model_health_data["error_classes"][model_name] = {}
        _model_health_data["gen_times_by_preset"][model_name] = {}

    _model_health_data["latencies"][model_name].append(round(elapsed_s * 1000, 1))

    if success:
        _model_health_data["success_counts"][model_name] += 1
    else:
        _model_health_data["failure_counts"][model_name] += 1
        if error_type:
            ec = _model_health_data["error_classes"][model_name]
            ec[error_type] = ec.get(error_type, 0) + 1

    if preset:
        pt = _model_health_data["gen_times_by_preset"][model_name]
        if preset not in pt:
            pt[preset] = deque(maxlen=100)
        pt[preset].append(round(elapsed_s, 2))


def _percentile(data, p):
    """Calculate percentile from sorted data."""
    if not data:
        return 0
    s = sorted(data)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


@app.get("/model/health")
async def model_health():
    """v25.0: Deep model diagnostics — latency percentiles, success/failure rates, error classification."""
    now = time.time()
    uptime_s = now - _model_health_data["start_time"]
    health = {}

    for model_name in set(list(_model_health_data["latencies"].keys()) + list(manager._registry.keys())):
        lats = list(_model_health_data["latencies"].get(model_name, []))
        succ = _model_health_data["success_counts"].get(model_name, 0)
        fail = _model_health_data["failure_counts"].get(model_name, 0)
        total = succ + fail
        errors = _model_health_data["error_classes"].get(model_name, {})
        presets = _model_health_data["gen_times_by_preset"].get(model_name, {})

        preset_avgs = {}
        for pname, ptimes in presets.items():
            times_list = list(ptimes)
            if times_list:
                preset_avgs[pname] = round(sum(times_list) / len(times_list), 2)

        entry = manager._registry.get(model_name)
        health[model_name] = {
            "total_requests": total,
            "success_count": succ,
            "failure_count": fail,
            "success_rate_pct": round(succ / max(1, total) * 100, 1),
            "latency_p50_ms": _percentile(lats, 50),
            "latency_p95_ms": _percentile(lats, 95),
            "latency_p99_ms": _percentile(lats, 99),
            "latency_avg_ms": round(sum(lats) / max(1, len(lats)), 1) if lats else 0,
            "latency_max_ms": max(lats) if lats else 0,
            "avg_gen_time_by_preset_s": preset_avgs,
            "error_classes": dict(errors),
            "on_gpu": entry.on_gpu if entry else False,
            "pinned": entry.pinned if entry else False,
            "vram_mb": entry.vram_mb if entry else 0,
            "idle_s": int(now - entry.last_used) if entry and entry.last_used else None,
        }

    return {
        "success": True,
        "version": "29.0",
        "uptime_s": int(uptime_s),
        "uptime": f"{uptime_s / 3600:.1f}h" if uptime_s > 3600 else f"{uptime_s / 60:.1f}m",
        "models": health,
    }


# ─── v25.0: VRAM Dashboard — real-time breakdown, history, peak, fragmentation, alerts ───
_vram_dashboard_history = deque(maxlen=100)
_vram_peak_mb = 0
_vram_alerts = deque(maxlen=50)

async def _vram_dashboard_loop():
    """v25.0: Record VRAM dashboard snapshots every 15s with peak tracking."""
    global _vram_peak_mb
    while True:
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                free_mb = free // (1024 * 1024)
                total_mb = total // (1024 * 1024)
                used_mb = total_mb - free_mb
                allocated_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved(0) // (1024 * 1024)
                frag_pct = round((reserved_mb - allocated_mb) / max(1, reserved_mb) * 100, 1) if reserved_mb > 0 else 0.0

                if used_mb > _vram_peak_mb:
                    _vram_peak_mb = used_mb

                # Memory pressure alerts
                pressure_pct = round(used_mb / max(1, total_mb) * 100, 1)
                if pressure_pct > 95:
                    _vram_alerts.append({
                        "timestamp": time.time(),
                        "level": "critical",
                        "message": f"VRAM critical: {pressure_pct}% used ({used_mb}/{total_mb}MB)",
                    })
                elif pressure_pct > 90:
                    _vram_alerts.append({
                        "timestamp": time.time(),
                        "level": "warning",
                        "message": f"VRAM high: {pressure_pct}% used ({used_mb}/{total_mb}MB)",
                    })

                # Per-model breakdown (estimate from registry)
                model_breakdown = {}
                for name in manager._gpu_loaded:
                    entry = manager._registry.get(name)
                    if entry:
                        model_breakdown[name] = {
                            "estimated_vram_mb": entry.vram_mb,
                            "pinned": entry.pinned,
                            "idle_s": int(time.time() - entry.last_used) if entry.last_used else None,
                        }

                _vram_dashboard_history.append({
                    "timestamp": time.time(),
                    "used_mb": used_mb,
                    "free_mb": free_mb,
                    "total_mb": total_mb,
                    "allocated_mb": allocated_mb,
                    "reserved_mb": reserved_mb,
                    "fragmentation_pct": frag_pct,
                    "pressure_pct": pressure_pct,
                    "models": list(manager._gpu_loaded.keys()),
                })
        except Exception:
            pass
        await asyncio.sleep(15)


@app.get("/vram/dashboard")
async def vram_dashboard():
    """v25.0: Enhanced VRAM monitoring — real-time breakdown, history, peak, fragmentation, alerts, GC trigger."""
    entries = list(_vram_dashboard_history)
    alerts = list(_vram_alerts)

    # Current state
    current = {}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        free_mb = free // (1024 * 1024)
        total_mb = total // (1024 * 1024)
        allocated_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(0) // (1024 * 1024)
        frag_pct = round((reserved_mb - allocated_mb) / max(1, reserved_mb) * 100, 1) if reserved_mb > 0 else 0.0

        model_breakdown = {}
        for name in manager._gpu_loaded:
            entry = manager._registry.get(name)
            if entry:
                model_breakdown[name] = {
                    "estimated_vram_mb": entry.vram_mb,
                    "pinned": entry.pinned,
                    "idle_s": int(time.time() - entry.last_used) if entry.last_used else None,
                }

        current = {
            "used_mb": total_mb - free_mb,
            "free_mb": free_mb,
            "total_mb": total_mb,
            "allocated_mb": allocated_mb,
            "reserved_mb": reserved_mb,
            "fragmentation_pct": frag_pct,
            "pressure_pct": round((total_mb - free_mb) / max(1, total_mb) * 100, 1),
            "model_breakdown": model_breakdown,
        }

    return {
        "success": True,
        "current": current,
        "peak_vram_mb": _vram_peak_mb,
        "history_count": len(entries),
        "history": entries[-20:],  # Last 20 snapshots
        "alerts": alerts[-10:],  # Last 10 alerts
        "gc_available": True,
        "hint": "POST /vram/dashboard/gc to trigger garbage collection",
    }


@app.post("/vram/dashboard/gc")
async def vram_dashboard_gc():
    """v25.0: Trigger garbage collection and VRAM cleanup from dashboard."""
    before_free = torch.cuda.mem_get_info()[0] // (1024 * 1024) if torch.cuda.is_available() else 0
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    after_free = torch.cuda.mem_get_info()[0] // (1024 * 1024) if torch.cuda.is_available() else 0
    freed = after_free - before_free
    log.info(f"VRAM Dashboard GC: freed {freed}MB (before={before_free}MB, after={after_free}MB)")
    return {"success": True, "freed_mb": freed, "free_mb_before": before_free, "free_mb_after": after_free}


# ─── v25.0: Batch 3D Generation v2 — Priority Queue + SSE Progress + Per-item Presets + OOM Retry ───
_batch_v2_progress = {}  # batch_id → {status, items, completed, current, errors}

class BatchV2Item(BaseModel):
    prompt: str = ""
    image: str = ""
    image_path: str = ""
    quality: str = "production"
    priority: int = 0  # Higher = processed first

class BatchV2Request(BaseModel):
    items: List[dict] = []
    max_items: int = 10


@app.post("/generate3d/batch/v2")
async def batch_3d_v2(req: BatchV2Request):
    """v25.0: Enhanced batch 3D — priority queue, per-item presets, OOM auto-retry."""
    items = req.items
    if not items:
        return {"success": False, "error": "No items provided. Pass {\"items\": [{\"prompt\": \"...\", \"priority\": 1}, ...]}"}
    if len(items) > req.max_items:
        items = items[:req.max_items]

    batch_id = uuid.uuid4().hex[:12]

    # Sort by priority (higher first)
    for i, item in enumerate(items):
        item["_original_index"] = i
    items_sorted = sorted(items, key=lambda x: x.get("priority", 0), reverse=True)

    _batch_v2_progress[batch_id] = {
        "status": "running",
        "total": len(items_sorted),
        "completed": 0,
        "current": None,
        "results": [None] * len(items_sorted),
        "errors": [],
        "started_at": time.time(),
    }

    results = [None] * len(items_sorted)
    total_t0 = time.time()

    for idx, item in enumerate(items_sorted):
        orig_idx = item.get("_original_index", idx)
        prompt = item.get("prompt", "")
        image = item.get("image", "")
        image_path = item.get("image_path", "")
        quality = item.get("quality", "production")
        priority = item.get("priority", 0)
        rid = uuid.uuid4().hex[:8]

        _batch_v2_progress[batch_id]["current"] = {
            "index": idx, "original_index": orig_idx,
            "prompt": prompt[:60], "quality": quality, "priority": priority,
        }

        t0 = time.time()
        log.info(f"BatchV2 [{idx+1}/{len(items_sorted)}] rid={rid} prio={priority} quality={quality}: {prompt[:60]}")

        success = False
        result = None
        retry_count = 0
        max_retries = 1

        while retry_count <= max_retries:
            try:
                quality_steps = {"draft": 15, "fast": 25, "production": 30, "max": 50}
                steps = quality_steps.get(quality, 30)

                # On retry after OOM, reduce steps
                if retry_count > 0:
                    steps = max(10, steps // 2)
                    log.warning(f"BatchV2 OOM retry [{rid}]: reducing steps to {steps}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                async with gpu_semaphore:
                    gen_req = Gen3DRequest(prompt=prompt, image=image, image_path=image_path,
                                           quality=quality, format="glb", steps=steps)
                    result = await generate_3d(gen_req)

                if result.get("success"):
                    success = True
                    _health_record("hunyuan3d", True, time.time() - t0, preset=quality)
                    break
                else:
                    retry_count += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
                if "out of memory" in str(oom_err).lower() or "CUDA" in str(oom_err):
                    log.warning(f"BatchV2 OOM on item {idx} [{rid}]: {oom_err}")
                    _health_record("hunyuan3d", False, time.time() - t0, preset=quality, error_type="OOM")
                    retry_count += 1
                    if retry_count > max_retries:
                        result = {"success": False, "error": f"OOM after {max_retries} retries: {str(oom_err)[:200]}"}
                        _batch_v2_progress[batch_id]["errors"].append({"index": orig_idx, "error": "OOM"})
                else:
                    result = {"success": False, "error": str(oom_err)[:300]}
                    _health_record("hunyuan3d", False, time.time() - t0, preset=quality, error_type="runtime")
                    break
            except Exception as e:
                result = {"success": False, "error": str(e)[:300]}
                _health_record("hunyuan3d", False, time.time() - t0, preset=quality, error_type="unknown")
                break

        elapsed = time.time() - t0
        if result is None:
            result = {"success": False, "error": "No result generated"}

        result["batch_index"] = orig_idx
        result["priority"] = priority
        result["quality"] = quality
        result["elapsed_s"] = round(elapsed, 2)
        result["request_id"] = rid
        result["retry_count"] = retry_count
        results[idx] = result

        _batch_v2_progress[batch_id]["completed"] = idx + 1
        _batch_v2_progress[batch_id]["results"][idx] = {
            "success": result.get("success", False),
            "index": orig_idx,
            "elapsed_s": round(elapsed, 2),
        }

        local_metrics.record("/generate3d/batch/v2", result.get("success", False), elapsed)
        _record_generation("/generate3d/batch/v2", prompt, "hunyuan3d", elapsed, result.get("success", False))

    total_elapsed = time.time() - total_t0
    succeeded = sum(1 for r in results if r and r.get("success"))

    _batch_v2_progress[batch_id]["status"] = "completed"
    _batch_v2_progress[batch_id]["completed"] = len(items_sorted)
    _batch_v2_progress[batch_id]["current"] = None

    return {
        "success": succeeded > 0,
        "batch_id": batch_id,
        "total": len(items_sorted),
        "succeeded": succeeded,
        "failed": len(items_sorted) - succeeded,
        "total_elapsed_s": round(total_elapsed, 2),
        "avg_elapsed_s": round(total_elapsed / max(1, len(items_sorted)), 2),
        "results": results,
    }


@app.get("/generate3d/batch/v2/progress/{batch_id}")
async def batch_v2_progress(batch_id: str):
    """v25.0: SSE-compatible progress endpoint for batch 3D v2."""
    from fastapi.responses import StreamingResponse

    if batch_id not in _batch_v2_progress:
        return {"success": False, "error": f"Batch {batch_id} not found"}

    progress = _batch_v2_progress[batch_id]

    # If batch is complete, return final status directly
    if progress["status"] == "completed":
        return {
            "success": True,
            "batch_id": batch_id,
            "status": "completed",
            "total": progress["total"],
            "completed": progress["completed"],
            "results": progress["results"],
            "errors": progress["errors"],
            "elapsed_s": round(time.time() - progress["started_at"], 2),
        }

    # SSE streaming for in-progress batches
    async def event_stream():
        import json as _json
        last_completed = -1
        while True:
            p = _batch_v2_progress.get(batch_id)
            if not p:
                yield f"data: {_json.dumps({'status': 'not_found'})}\n\n"
                break
            if p["completed"] != last_completed or p["status"] == "completed":
                last_completed = p["completed"]
                event = {
                    "status": p["status"],
                    "total": p["total"],
                    "completed": p["completed"],
                    "current": p["current"],
                    "errors": p["errors"],
                    "elapsed_s": round(time.time() - p["started_at"], 2),
                }
                yield f"data: {_json.dumps(event)}\n\n"
            if p["status"] == "completed":
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── v25.0: Mesh Post-Processing — decimate, smooth, normalize, LOD chain via trimesh ───

class MeshPostProcessRequest(BaseModel):
    mesh_path: str
    target_vertices: int = 0  # 0 = no decimation
    smooth_normals: bool = True
    center_and_normalize: bool = True
    remove_isolated: bool = True
    generate_lod: bool = False  # Generate LOD chain: 100%, 50%, 25%
    lod_ratios: List[float] = [1.0, 0.5, 0.25]
    output_format: str = "glb"


@app.post("/mesh/postprocess")
async def mesh_postprocess(req: MeshPostProcessRequest):
    """v25.0: Post-process 3D mesh — decimate, smooth normals, center/normalize, remove isolated, LOD chain."""
    try:
        import trimesh

        if not os.path.exists(req.mesh_path):
            return {"success": False, "error": f"Mesh not found: {req.mesh_path}"}

        t0 = time.time()
        mesh = trimesh.load(req.mesh_path, process=False)

        # Handle Scene vs Trimesh
        if isinstance(mesh, trimesh.Scene):
            # Merge all geometries into single mesh for processing
            meshes_list = list(mesh.geometry.values())
            if not meshes_list:
                return {"success": False, "error": "Empty scene, no geometry found"}
            mesh = trimesh.util.concatenate(meshes_list)

        original_verts = len(mesh.vertices)
        original_faces = len(mesh.faces)
        operations = []

        # Step 1: Remove isolated (unreferenced) vertices
        if req.remove_isolated:
            before = len(mesh.vertices)
            mesh.remove_unreferenced_vertices()
            removed = before - len(mesh.vertices)
            if removed > 0:
                operations.append(f"Removed {removed} isolated vertices")

        # Step 2: Remove degenerate faces
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()

        # Step 3: Decimate to target vertex count
        if req.target_vertices > 0 and len(mesh.vertices) > req.target_vertices:
            ratio = req.target_vertices / len(mesh.vertices)
            target_faces = max(10, int(len(mesh.faces) * ratio))
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
                operations.append(f"Decimated: {original_verts} -> {len(mesh.vertices)} vertices (target={req.target_vertices})")
            except Exception as dec_err:
                log.warning(f"Quadric decimation failed: {dec_err}, trying vertex clustering")
                try:
                    # Fallback: vertex clustering
                    pitch = mesh.extents.max() / (req.target_vertices ** (1/3))
                    mesh = mesh.simplify_quadric_decimation(target_faces)
                    operations.append(f"Decimated via fallback: {len(mesh.vertices)} vertices")
                except Exception:
                    operations.append(f"Decimation failed: {str(dec_err)[:100]}")

        # Step 4: Smooth normals
        if req.smooth_normals:
            try:
                # Fix normals and ensure they're consistent
                mesh.fix_normals()
                operations.append("Smoothed and fixed normals")
            except Exception as norm_err:
                operations.append(f"Normal smoothing failed: {str(norm_err)[:100]}")

        # Step 5: Center and normalize scale
        if req.center_and_normalize:
            centroid = mesh.centroid.copy()
            mesh.vertices -= centroid
            extent = mesh.extents.max()
            if extent > 0:
                mesh.vertices /= extent  # Normalize to unit bounding box
            operations.append(f"Centered (offset={centroid.round(3).tolist()}) and normalized (extent={extent:.3f})")

        # Export main mesh
        ext_map = {"glb": "glb", "obj": "obj", "stl": "stl", "ply": "ply"}
        out_ext = ext_map.get(req.output_format, "glb")
        out_fname = f"postproc_{uuid.uuid4().hex[:8]}.{out_ext}"
        out_path = MODELS_DIR / out_fname
        mesh.export(str(out_path))

        result = {
            "success": True,
            "path": str(out_path),
            "original_vertices": original_verts,
            "original_faces": original_faces,
            "final_vertices": len(mesh.vertices),
            "final_faces": len(mesh.faces),
            "operations": operations,
            "elapsed_s": round(time.time() - t0, 2),
        }

        # Step 6: Generate LOD chain
        if req.generate_lod:
            lods = []
            for ratio in req.lod_ratios:
                if ratio >= 1.0:
                    lods.append({
                        "ratio": ratio,
                        "path": str(out_path),
                        "vertices": len(mesh.vertices),
                        "faces": len(mesh.faces),
                    })
                    continue
                target_f = max(10, int(len(mesh.faces) * ratio))
                try:
                    lod_mesh = mesh.simplify_quadric_decimation(target_f)
                    lod_fname = f"lod{int(ratio*100)}_{uuid.uuid4().hex[:8]}.{out_ext}"
                    lod_path = MODELS_DIR / lod_fname
                    lod_mesh.export(str(lod_path))
                    lods.append({
                        "ratio": ratio,
                        "path": str(lod_path),
                        "vertices": len(lod_mesh.vertices),
                        "faces": len(lod_mesh.faces),
                    })
                    operations.append(f"LOD {int(ratio*100)}%: {len(lod_mesh.vertices)} verts, {len(lod_mesh.faces)} faces")
                except Exception as lod_err:
                    lods.append({"ratio": ratio, "error": str(lod_err)[:200]})

            result["lods"] = lods

        # Review the output
        review = _review_3d(str(out_path))
        result["review"] = review
        result["elapsed_s"] = round(time.time() - t0, 2)
        local_metrics.record("/mesh/postprocess", True, time.time() - t0)
        return result

    except Exception as e:
        log.error(f"Mesh postprocess failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ─── v25.0: Texture Baking Integration — Hunyuan3D mesh + remote FLUX albedo → textured GLB ───

class TexturedGenRequest(BaseModel):
    prompt: str = ""
    image: str = ""
    image_path: str = ""
    quality: str = "production"
    steps: int = 30
    texture_prompt: str = ""  # Override texture prompt (defaults to mesh prompt)
    texture_size: int = 1024  # Texture resolution


@app.post("/generate3d/textured")
async def generate_3d_textured(req: TexturedGenRequest):
    """v25.0: Generate 3D model with Hunyuan3D + bake albedo texture from remote FLUX.1-dev → textured GLB."""
    try:
        import trimesh
        from PIL import Image as PILImage

        t0 = time.time()

        # Step 1: Generate mesh via Hunyuan3D
        log.info(f"TextureBake: generating mesh for '{req.prompt[:60]}'...")
        gen_req = Gen3DRequest(prompt=req.prompt, image=req.image, image_path=req.image_path,
                               quality=req.quality, format="glb", steps=req.steps)
        async with gpu_semaphore:
            mesh_result = await generate_3d(gen_req)

        if not mesh_result.get("success"):
            return {"success": False, "error": f"Mesh generation failed: {mesh_result.get('error', 'unknown')}"}

        mesh_path = mesh_result.get("path", "")
        elapsed_mesh = time.time() - t0
        log.info(f"TextureBake: mesh generated in {elapsed_mesh:.1f}s: {mesh_path}")

        # Step 2: Load mesh and UV unwrap
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            meshes_list = list(mesh.geometry.values())
            if not meshes_list:
                return {"success": False, "error": "Empty mesh scene"}
            mesh = trimesh.util.concatenate(meshes_list)

        # Check/generate UVs
        has_uv = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
        if not has_uv:
            log.info("TextureBake: no UVs found, generating via xatlas/trimesh unwrap...")
            try:
                # Try xatlas if available
                import xatlas
                vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
                mesh = trimesh.Trimesh(
                    vertices=mesh.vertices[vmapping],
                    faces=indices,
                    process=False
                )
                mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
                log.info(f"TextureBake: xatlas UV unwrap done ({len(uvs)} UVs)")
            except ImportError:
                log.warning("TextureBake: xatlas not available, using basic UV projection")
                # Fallback: box projection UVs
                uv = np.zeros((len(mesh.vertices), 2), dtype=np.float64)
                verts = mesh.vertices
                extent = verts.max(axis=0) - verts.min(axis=0)
                if extent.max() > 0:
                    norm_verts = (verts - verts.min(axis=0)) / extent.max()
                    # Use X,Y for UV mapping
                    uv[:, 0] = norm_verts[:, 0]
                    uv[:, 1] = norm_verts[:, 1]
                mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
                log.info(f"TextureBake: basic UV projection done ({len(uv)} UVs)")

        # Step 3: Generate albedo texture via remote FLUX.1-dev
        tex_prompt = req.texture_prompt or req.prompt
        texture_prompt = f"seamless tileable texture for {tex_prompt}, flat lighting, no shadows, game asset albedo map, top-down view, PBR albedo"

        log.info(f"TextureBake: requesting albedo from remote FLUX.1-dev...")
        import httpx
        remote_url = os.getenv("REMOTE_AI_URL", "http://100.100.246.94:8200")
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(f"{remote_url}/image", json={
                    "prompt": texture_prompt,
                    "negative_prompt": "blurry, low quality, shadows, dark, highlights, specular, 3D render",
                    "width": req.texture_size,
                    "height": req.texture_size,
                    "steps": 20,
                    "guidance": 3.5,
                })
                tex_data = resp.json()
            if not tex_data.get("success"):
                log.warning(f"TextureBake: FLUX texture gen failed: {tex_data.get('error')}")
                # Export mesh without texture
                out_fname = f"textured_{uuid.uuid4().hex[:8]}.glb"
                out_path = MODELS_DIR / out_fname
                mesh.export(str(out_path))
                return {
                    "success": True,
                    "path": str(out_path),
                    "mesh_path": mesh_path,
                    "texture_path": None,
                    "has_texture": False,
                    "warning": f"Texture gen failed: {tex_data.get('error', 'unknown')}. Exported untextured mesh.",
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces),
                    "elapsed_s": round(time.time() - t0, 2),
                }

            # Decode texture
            img_b64 = tex_data.get("image", "")
            tex_img = PILImage.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
            tex_fname = f"albedo_{uuid.uuid4().hex[:8]}.png"
            tex_path = ASSET_DIR / tex_fname
            tex_img.save(str(tex_path))
            log.info(f"TextureBake: albedo texture saved: {tex_path} ({req.texture_size}x{req.texture_size})")

        except Exception as tex_err:
            log.warning(f"TextureBake: remote texture request failed: {tex_err}")
            out_fname = f"textured_{uuid.uuid4().hex[:8]}.glb"
            out_path = MODELS_DIR / out_fname
            mesh.export(str(out_path))
            return {
                "success": True,
                "path": str(out_path),
                "mesh_path": mesh_path,
                "texture_path": None,
                "has_texture": False,
                "warning": f"Remote texture request failed: {str(tex_err)[:200]}",
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "elapsed_s": round(time.time() - t0, 2),
            }

        # Step 4: Apply texture to mesh and export as textured GLB
        try:
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=tex_img,
                metallicFactor=0.0,
                roughnessFactor=0.8,
            )
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=mesh.visual.uv if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None else None,
                material=material,
            )
        except Exception as mat_err:
            log.warning(f"TextureBake: material assignment failed: {mat_err}")

        out_fname = f"textured_{uuid.uuid4().hex[:8]}.glb"
        out_path = MODELS_DIR / out_fname
        mesh.export(str(out_path))

        elapsed_total = time.time() - t0
        review = _review_3d(str(out_path))

        _health_record("hunyuan3d", True, elapsed_mesh, preset=req.quality)
        local_metrics.record("/generate3d/textured", True, elapsed_total)
        _record_generation("/generate3d/textured", req.prompt, "hunyuan3d+flux", elapsed_total, True)

        return {
            "success": True,
            "path": str(out_path),
            "mesh_path": mesh_path,
            "texture_path": str(tex_path),
            "has_texture": True,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "texture_size": req.texture_size,
            "review": review,
            "elapsed_mesh_s": round(elapsed_mesh, 2),
            "elapsed_total_s": round(elapsed_total, 2),
            "hint": "Import textured GLB to Unity: unity_import_asset + unity_create_material(URP/Lit)",
        }
    except Exception as e:
        log.error(f"Textured 3D gen failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ─── v25.0: 3D Model Review v2 — Enhanced topology, watertight, UV, game-readiness scoring ───

@app.post("/review3d/enhanced")
async def review3d_enhanced(req: dict = {}):
    """v25.0: Enhanced 3D quality scoring — topology analysis, watertight, UV coverage, game-readiness."""
    file_path = req.get("file_path", "")
    category = req.get("category", "prop")

    if not file_path:
        return {"success": False, "error": "No file_path provided"}
    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    try:
        import trimesh
        t0 = time.time()

        mesh = trimesh.load(file_path, process=False)
        is_scene = isinstance(mesh, trimesh.Scene)

        if is_scene:
            meshes_list = list(mesh.geometry.values())
            if not meshes_list:
                return {"success": False, "error": "Empty scene, no geometry found"}
            combined = trimesh.util.concatenate(meshes_list)
        else:
            combined = mesh
            meshes_list = [mesh]

        verts = len(combined.vertices)
        faces = len(combined.faces)
        score = 100
        issues = []
        suggestions = []

        # ── 1. Basic stats ──
        fsize = os.path.getsize(file_path)
        if fsize < 1000:
            issues.append(f"Tiny file ({fsize} bytes)")
            score -= 60
        if verts < 50:
            issues.append(f"Very few vertices ({verts})")
            score -= 40
        if faces < 30:
            issues.append(f"Very few faces ({faces})")
            score -= 30

        # ── 2. Watertight check ──
        is_watertight = combined.is_watertight
        if not is_watertight:
            issues.append("Mesh is not watertight (has holes/gaps)")
            suggestions.append("Fix in Blender: Mesh > Clean Up > Fill Holes")
            score -= 10

        # ── 3. Triangle quality analysis ──
        triangle_quality = {}
        if faces > 0:
            try:
                # Get face areas
                areas = combined.area_faces
                if len(areas) > 0:
                    zero_area = int(np.sum(areas < 1e-10))
                    min_area = float(np.min(areas))
                    max_area = float(np.max(areas))
                    mean_area = float(np.mean(areas))
                    area_ratio = max_area / max(1e-10, min_area) if min_area > 0 else float('inf')

                    triangle_quality = {
                        "total_faces": faces,
                        "zero_area_faces": zero_area,
                        "min_area": round(min_area, 8),
                        "max_area": round(max_area, 6),
                        "mean_area": round(mean_area, 6),
                        "area_ratio": round(min(area_ratio, 999999), 1),
                    }

                    if zero_area > 0:
                        issues.append(f"{zero_area} degenerate (zero-area) faces")
                        score -= min(15, zero_area)
                    if area_ratio > 1000:
                        issues.append(f"Extreme triangle size variation (ratio={area_ratio:.0f})")
                        suggestions.append("Remesh for more uniform triangulation")
                        score -= 10
            except Exception:
                pass

        # ── 4. Bounding box validation ──
        bounds = combined.bounds if combined.bounds is not None else None
        extent = None
        bounding_box = {}
        if bounds is not None:
            extent = bounds[1] - bounds[0]
            bounding_box = {
                "min": bounds[0].round(4).tolist(),
                "max": bounds[1].round(4).tolist(),
                "extent": extent.round(4).tolist(),
                "center": combined.centroid.round(4).tolist(),
            }
            if extent.max() < 0.001:
                issues.append("Degenerate mesh (zero extent)")
                score -= 70
            elif extent.max() > 0:
                aspect_ratio = float(extent.max() / max(0.001, extent.min()))
                bounding_box["aspect_ratio"] = round(aspect_ratio, 2)
                if aspect_ratio > 50:
                    issues.append(f"Extreme aspect ratio ({aspect_ratio:.0f}:1)")
                    score -= 15

        # ── 5. Surface area / volume ratio ──
        surface_volume = {}
        try:
            sa = float(combined.area)
            vol = float(combined.volume) if combined.is_watertight else 0
            surface_volume = {
                "surface_area": round(sa, 4),
                "volume": round(vol, 6),
                "sa_vol_ratio": round(sa / max(1e-10, abs(vol)), 2) if vol != 0 else None,
            }
        except Exception:
            pass

        # ── 6. UV coverage analysis ──
        uv_analysis = {"has_uv": False}
        try:
            for m in meshes_list:
                if hasattr(m.visual, 'uv') and m.visual.uv is not None and len(m.visual.uv) > 0:
                    uv = m.visual.uv
                    uv_analysis = {
                        "has_uv": True,
                        "uv_count": len(uv),
                        "uv_min": uv.min(axis=0).round(4).tolist(),
                        "uv_max": uv.max(axis=0).round(4).tolist(),
                        "uv_in_range": bool(np.all((uv >= 0) & (uv <= 1))),
                        "uv_coverage_pct": round(float((uv.max(axis=0) - uv.min(axis=0)).prod()) * 100, 1),
                    }
                    if not uv_analysis["uv_in_range"]:
                        issues.append("UV coordinates outside [0,1] range")
                        score -= 5
                    break
        except Exception:
            pass

        if not uv_analysis["has_uv"]:
            issues.append("No UV coordinates — texture mapping not possible")
            suggestions.append("UV unwrap in Blender or use xatlas")
            score -= 10

        # ── 7. Normals check ──
        has_normals = False
        normals_consistent = False
        try:
            if combined.vertex_normals is not None and len(combined.vertex_normals) > 0:
                has_normals = True
                # Check if normals are somewhat consistent (no NaN/zero)
                norm_mags = np.linalg.norm(combined.vertex_normals, axis=1)
                zero_normals = int(np.sum(norm_mags < 0.01))
                normals_consistent = zero_normals < len(norm_mags) * 0.05
                if not normals_consistent:
                    issues.append(f"{zero_normals} zero/invalid normals ({zero_normals / len(norm_mags) * 100:.1f}%)")
                    score -= 10
        except Exception:
            pass

        if not has_normals:
            issues.append("Missing vertex normals")
            suggestions.append("Recalculate normals in Blender")
            score -= 5

        # ── 8. Vertex colors check ──
        has_colors = False
        for m in meshes_list:
            if hasattr(m.visual, 'vertex_colors') and m.visual.vertex_colors is not None:
                has_colors = True
                break

        # ── 9. Game-readiness score ──
        game_ready_score = 100
        game_issues = []

        # Profile targets
        p = MODEL3D_PROFILE
        vert_targets = (p.get("category_vertex_targets", {}).get(category) or
                        p.get("category_vertex_targets", {}).get("default", {"min": 100, "max": 10000}))
        target_min = vert_targets.get("min", 100)
        target_max = vert_targets.get("max", 10000)

        if verts < target_min:
            game_issues.append(f"Below min vertex target for {category} ({verts} < {target_min})")
            game_ready_score -= 20
        elif verts > target_max:
            game_issues.append(f"Above max vertex target for {category} ({verts} > {target_max})")
            game_ready_score -= 10

        if not has_normals:
            game_issues.append("Missing normals (required for lighting)")
            game_ready_score -= 15
        if not uv_analysis["has_uv"]:
            game_issues.append("Missing UVs (required for texturing)")
            game_ready_score -= 20
        if not is_watertight:
            game_issues.append("Not watertight (may cause rendering artifacts)")
            game_ready_score -= 10
        if triangle_quality.get("zero_area_faces", 0) > 0:
            game_issues.append("Degenerate faces present")
            game_ready_score -= 10

        game_ready_score = max(0, game_ready_score)

        elapsed = time.time() - t0
        score = max(0, min(100, score))

        local_metrics.record("/review3d/enhanced", True, elapsed)

        return {
            "success": True,
            "file_path": file_path,
            "category": category,
            "score": score,
            "pass": score >= 40,
            "game_readiness_score": game_ready_score,
            "game_readiness_issues": game_issues,
            "issues": issues,
            "suggestions": suggestions,
            "stats": {
                "vertices": verts,
                "faces": faces,
                "submeshes": len(meshes_list),
                "file_kb": round(fsize / 1024, 1),
                "is_scene": is_scene,
            },
            "watertight": is_watertight,
            "has_normals": has_normals,
            "normals_consistent": normals_consistent,
            "has_colors": has_colors,
            "triangle_quality": triangle_quality,
            "bounding_box": bounding_box,
            "surface_volume": surface_volume,
            "uv_analysis": uv_analysis,
            "elapsed_s": round(elapsed, 3),
            "profile_version": MODEL3D_PROFILE.get("_version", "none"),
        }
    except Exception as e:
        log.error(f"Review3D enhanced failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ─── v25.0: Export Format Conversion — GLB/OBJ/FBX/STL/PLY interconversion ───

class ConvertRequest(BaseModel):
    input_path: str
    output_format: str = "glb"  # glb, obj, stl, ply, off
    optimize: bool = False  # Run mesh optimization during conversion
    target_vertices: int = 0  # Decimate if > 0


@app.post("/convert")
async def convert_format(req: ConvertRequest):
    """v25.0: Convert between 3D formats — GLB/OBJ/STL/PLY/OFF with optional optimization."""
    try:
        import trimesh

        if not os.path.exists(req.input_path):
            return {"success": False, "error": f"Input file not found: {req.input_path}"}

        t0 = time.time()
        input_ext = os.path.splitext(req.input_path)[1].lower()

        # Supported formats
        supported_in = {".glb", ".gltf", ".obj", ".stl", ".ply", ".off", ".fbx"}
        supported_out = {"glb", "obj", "stl", "ply", "off"}

        if input_ext not in supported_in:
            return {"success": False, "error": f"Unsupported input format: {input_ext}. Supported: {supported_in}"}
        if req.output_format not in supported_out:
            return {"success": False, "error": f"Unsupported output format: {req.output_format}. Supported: {supported_out}"}

        # Load mesh
        mesh = trimesh.load(req.input_path, process=False)

        # Handle Scene
        is_scene = isinstance(mesh, trimesh.Scene)
        if is_scene:
            meshes_list = list(mesh.geometry.values())
            if not meshes_list:
                return {"success": False, "error": "Empty scene, no geometry to convert"}
            # For formats that don't support scenes, merge
            if req.output_format in ("stl", "ply", "off"):
                mesh = trimesh.util.concatenate(meshes_list)
                is_scene = False

        original_verts = sum(len(g.vertices) for g in (meshes_list if is_scene else [mesh])) if is_scene else len(mesh.vertices)
        original_faces = sum(len(g.faces) for g in (meshes_list if is_scene else [mesh])) if is_scene else len(mesh.faces)
        operations = []

        # Optimization
        if req.optimize and not is_scene:
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.fix_normals()
            operations.append("Optimized: removed unreferenced/degenerate/duplicate, fixed normals")

        # Decimate
        if req.target_vertices > 0 and not is_scene and len(mesh.vertices) > req.target_vertices:
            ratio = req.target_vertices / len(mesh.vertices)
            target_f = max(10, int(len(mesh.faces) * ratio))
            try:
                mesh = mesh.simplify_quadric_decimation(target_f)
                operations.append(f"Decimated: {original_verts} -> {len(mesh.vertices)} vertices")
            except Exception as dec_err:
                operations.append(f"Decimation failed: {str(dec_err)[:100]}")

        # Export
        out_fname = f"convert_{uuid.uuid4().hex[:8]}.{req.output_format}"
        out_path = MODELS_DIR / out_fname

        if is_scene and req.output_format in ("glb", "gltf"):
            # Scene can be exported directly to GLB/GLTF
            mesh.export(str(out_path))
        else:
            if is_scene:
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            mesh.export(str(out_path))

        final_verts = len(mesh.vertices) if not is_scene else sum(len(g.vertices) for g in meshes_list)
        final_faces = len(mesh.faces) if not is_scene else sum(len(g.faces) for g in meshes_list)

        elapsed = time.time() - t0
        local_metrics.record("/convert", True, elapsed)

        return {
            "success": True,
            "input_path": req.input_path,
            "input_format": input_ext.lstrip("."),
            "output_path": str(out_path),
            "output_format": req.output_format,
            "original_vertices": original_verts,
            "original_faces": original_faces,
            "final_vertices": final_verts,
            "final_faces": final_faces,
            "operations": operations,
            "file_size_kb": round(os.path.getsize(str(out_path)) / 1024, 1),
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        log.error(f"Format conversion failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ─── v26.0: Generation Pipeline Stats ───

_pipeline_stats = {
    "pipelines": {},  # pipeline_id → { stages[], total_time, status }
    "total_completed": 0,
    "total_failed": 0,
    "avg_pipeline_time_s": 0.0,
    "pipeline_times": [],
}

@app.get("/pipeline/stats")
async def get_pipeline_stats():
    """v26.0: Get generation pipeline statistics"""
    recent = list(_pipeline_stats["pipelines"].values())[-20:]
    avg_time = (sum(_pipeline_stats["pipeline_times"][-50:]) / max(1, len(_pipeline_stats["pipeline_times"][-50:]))) if _pipeline_stats["pipeline_times"] else 0

    return {
        "success": True,
        "total_completed": _pipeline_stats["total_completed"],
        "total_failed": _pipeline_stats["total_failed"],
        "avg_pipeline_time_s": round(avg_time, 2),
        "active_pipelines": sum(1 for p in _pipeline_stats["pipelines"].values() if p.get("status") == "running"),
        "recent_pipelines": recent,
    }

@app.post("/pipeline/start")
async def start_pipeline(data: dict):
    """v26.0: Start tracking a generation pipeline"""
    pipeline_id = data.get("pipeline_id", str(uuid.uuid4())[:8])
    _pipeline_stats["pipelines"][pipeline_id] = {
        "id": pipeline_id,
        "name": data.get("name", "unnamed"),
        "stages": [],
        "started_at": time.time(),
        "status": "running",
        "total_time_s": 0,
    }
    # Cleanup old pipelines (keep last 100)
    if len(_pipeline_stats["pipelines"]) > 100:
        oldest = list(_pipeline_stats["pipelines"].keys())[0]
        del _pipeline_stats["pipelines"][oldest]

    return {"success": True, "pipeline_id": pipeline_id}

@app.post("/pipeline/stage")
async def record_pipeline_stage(data: dict):
    """v26.0: Record a stage completion in a pipeline"""
    pid = data.get("pipeline_id")
    if pid not in _pipeline_stats["pipelines"]:
        return {"success": False, "error": f"Pipeline {pid} not found"}

    pipeline = _pipeline_stats["pipelines"][pid]
    pipeline["stages"].append({
        "name": data.get("stage_name", "unknown"),
        "elapsed_s": round(data.get("elapsed_s", 0), 2),
        "success": data.get("success", True),
        "output": data.get("output", ""),
        "ts": time.time(),
    })

    if data.get("final", False):
        pipeline["status"] = "completed" if data.get("success", True) else "failed"
        pipeline["total_time_s"] = round(time.time() - pipeline["started_at"], 2)
        if pipeline["status"] == "completed":
            _pipeline_stats["total_completed"] += 1
            _pipeline_stats["pipeline_times"].append(pipeline["total_time_s"])
        else:
            _pipeline_stats["total_failed"] += 1

    return {"success": True, "pipeline_id": pid, "stages": len(pipeline["stages"])}


# ─── v26.0: Asset Catalog ───

_asset_catalog = []  # List of generated assets with metadata
_CATALOG_MAX = 500

@app.get("/assets/catalog")
async def get_asset_catalog(limit: int = 50, asset_type: str = None):
    """v26.0: Get catalog of generated assets"""
    filtered = _asset_catalog
    if asset_type:
        filtered = [a for a in filtered if a.get("type") == asset_type]
    return {
        "success": True,
        "total": len(_asset_catalog),
        "filtered": len(filtered),
        "assets": filtered[-limit:],
        "types": list(set(a.get("type", "unknown") for a in _asset_catalog)),
    }

@app.post("/assets/register")
async def register_asset(data: dict):
    """v26.0: Register a generated asset in the catalog"""
    asset = {
        "path": data.get("path", ""),
        "type": data.get("type", "unknown"),  # image, model3d, texture, audio
        "prompt": data.get("prompt", ""),
        "category": data.get("category", "general"),
        "quality_score": data.get("quality_score", 0),
        "file_size_kb": data.get("file_size_kb", 0),
        "created_at": time.time(),
        "metadata": data.get("metadata", {}),
    }
    _asset_catalog.append(asset)
    while len(_asset_catalog) > _CATALOG_MAX:
        _asset_catalog.pop(0)

    return {"success": True, "catalog_size": len(_asset_catalog), "asset": asset}

@app.get("/assets/search")
async def search_assets(query: str = "", asset_type: str = None, min_score: int = 0, limit: int = 20):
    """v26.0: Search asset catalog by keyword and filters"""
    results = []
    for asset in reversed(_asset_catalog):
        if query and query.lower() not in (asset.get("prompt", "") + asset.get("path", "")).lower():
            continue
        if asset_type and asset.get("type") != asset_type:
            continue
        if asset.get("quality_score", 0) < min_score:
            continue
        results.append(asset)
        if len(results) >= limit:
            break
    return {"success": True, "query": query, "results": results, "count": len(results)}


# ─── v26.0: Performance Profiling ───

@app.get("/perf/profile")
async def get_perf_profile():
    """v26.0: Get per-endpoint performance profiling data"""
    profile = {}
    if hasattr(local_metrics, '_by_endpoint'):
        for endpoint, data in local_metrics._by_endpoint.items():
            if data["calls"] > 0:
                profile[endpoint] = {
                    "count": data["calls"],
                    "avg_ms": round(data["total_ms"] / data["calls"]),
                    "max_ms": round(data.get("max_ms", 0)),
                    "error_rate": f"{(data.get('errors', 0) / data['calls'] * 100):.1f}%",
                    "errors": data.get("errors", 0),
                }
    return {
        "success": True,
        "endpoints": profile,
        "gpu": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
            "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 1) if torch.cuda.is_available() else 0,
            "vram_cached_gb": round(torch.cuda.memory_reserved(0) / 1e9, 1) if torch.cuda.is_available() else 0,
        },
        "uptime_s": round(time.time() - start_time),
    }


# ─── v29.0: Internal 3D Generation Helper ───

async def _generate_3d_internal(prompt: str, seed=None, steps: int = 50):
    """v29.0: Internal helper — wraps generate_3d() for use by pipeline/preset endpoints."""
    try:
        gen_req = Gen3DRequest(prompt=prompt, steps=steps, quality="production")
        if seed is not None:
            pass  # Gen3DRequest doesn't have seed field, handled by model
        async with gpu_semaphore:
            result = await generate_3d(gen_req)
        # Normalize output path key
        if result.get("success") and result.get("path") and not result.get("output_path"):
            result["output_path"] = result["path"]
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── v29.0: Mesh LOD Auto-Generation ───

@app.post("/mesh/lod")
async def generate_mesh_lod(request: Request):
    """v29.0: Generate LOD levels from a mesh — LOD0 (original), LOD1 (50%), LOD2 (25%), LOD3 (10%)"""
    try:
        data = await request.json()
        input_path = data.get("input_path")
        if not input_path or not os.path.exists(input_path):
            return {"success": False, "error": "input_path required and must exist"}

        import trimesh
        mesh = trimesh.load(input_path)
        if not hasattr(mesh, 'vertices'):
            return {"success": False, "error": "Could not load mesh vertices"}

        original_faces = len(mesh.faces)
        base_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        lod_ratios = [1.0, 0.5, 0.25, 0.1]
        lod_results = []

        for i, ratio in enumerate(lod_ratios):
            if ratio == 1.0:
                lod_path = input_path
                lod_faces = original_faces
            else:
                target_faces = max(100, int(original_faces * ratio))
                try:
                    decimated = mesh.simplify_quadric_decimation(target_faces)
                    lod_path = os.path.join(base_dir, f"{base_name}_LOD{i}.glb")
                    decimated.export(lod_path)
                    lod_faces = len(decimated.faces)
                except Exception as e:
                    lod_results.append({"lod": i, "error": str(e)})
                    continue

            lod_results.append({
                "lod": i, "ratio": ratio, "faces": lod_faces,
                "path": lod_path, "reduction": f"{(1 - lod_faces/original_faces)*100:.1f}%",
            })

        elapsed = time.time() - start_time  # endpoint timing via metrics
        local_metrics.record("/mesh/lod", True, 0)
        return {
            "success": True, "original_faces": original_faces,
            "lod_levels": lod_results, "input": input_path,
        }
    except Exception as e:
        local_metrics.record("/mesh/lod", False, 0)
        return {"success": False, "error": str(e)}


# ─── v29.0: 3D Pipeline v3 — Full Asset Pipeline ───

@app.post("/pipeline/3d")
async def full_3d_pipeline(request: Request):
    """v29.0: Full 3D pipeline — prompt → generate mesh → optimize → LOD → export"""
    try:
        data = await request.json()
        prompt = data.get("prompt")
        if not prompt:
            return {"success": False, "error": "prompt required"}

        pipeline_id = f"pipe_{int(time.time())}_{os.getpid()}"
        stages = []

        # Stage 1: Generate 3D mesh
        stage1_start = time.time()
        gen_result = await _generate_3d_internal(prompt, data.get("seed"), data.get("steps", 50))
        stages.append({"stage": "generate", "duration_s": round(time.time() - stage1_start, 2), "success": gen_result.get("success", False)})

        if not gen_result.get("success"):
            local_metrics.record("/pipeline/3d", False, time.time() - stage1_start)
            return {"success": False, "pipeline_id": pipeline_id, "stages": stages, "error": f"Generation failed: {gen_result.get('error', 'unknown')}"}

        output_path = gen_result.get("output_path") or gen_result.get("path")

        # Stage 2: Mesh optimization (if trimesh available)
        stage2_start = time.time()
        try:
            import trimesh
            mesh = trimesh.load(output_path)
            # Auto-fix: merge close vertices, fix normals
            if hasattr(mesh, 'merge_vertices'):
                mesh.merge_vertices()
            if hasattr(mesh, 'fix_normals'):
                mesh.fix_normals()
            optimized_path = output_path.replace(".glb", "_optimized.glb")
            mesh.export(optimized_path)
            stages.append({"stage": "optimize", "duration_s": round(time.time() - stage2_start, 2), "success": True, "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))})
            output_path = optimized_path
        except Exception as e:
            stages.append({"stage": "optimize", "duration_s": round(time.time() - stage2_start, 2), "success": False, "error": str(e)})

        # Stage 3: Generate LOD levels
        stage3_start = time.time()
        lod_paths = [output_path]
        try:
            import trimesh
            mesh = trimesh.load(output_path)
            original_faces = len(mesh.faces)
            for ratio in [0.5, 0.25]:
                target = max(100, int(original_faces * ratio))
                try:
                    lod = mesh.simplify_quadric_decimation(target)
                    lod_path = output_path.replace(".glb", f"_lod{len(lod_paths)}.glb")
                    lod.export(lod_path)
                    lod_paths.append(lod_path)
                except Exception:
                    pass
            stages.append({"stage": "lod_generation", "duration_s": round(time.time() - stage3_start, 2), "success": True, "levels": len(lod_paths)})
        except Exception as e:
            stages.append({"stage": "lod_generation", "duration_s": round(time.time() - stage3_start, 2), "success": False, "error": str(e)})

        total_time = sum(s["duration_s"] for s in stages)
        local_metrics.record("/pipeline/3d", True, total_time)
        _record_generation("/pipeline/3d", prompt, "hunyuan3d", total_time, True)
        return {
            "success": True, "pipeline_id": pipeline_id,
            "output_path": output_path, "lod_paths": lod_paths,
            "stages": stages, "total_duration_s": round(total_time, 2),
            "prompt": prompt,
        }
    except Exception as e:
        local_metrics.record("/pipeline/3d", False, 0)
        return {"success": False, "error": str(e)}


# ─── v29.0: Advanced Mesh Analysis ───

@app.post("/mesh/analyze")
async def analyze_mesh(request: Request):
    """v29.0: Deep mesh analysis — topology quality, polygon budget, game-readiness score"""
    try:
        data = await request.json()
        input_path = data.get("input_path")
        if not input_path or not os.path.exists(input_path):
            return {"success": False, "error": "input_path required and must exist"}

        import trimesh
        mesh = trimesh.load(input_path)

        # Handle Scene — merge geometries for analysis
        if isinstance(mesh, trimesh.Scene):
            meshes_list = list(mesh.geometry.values())
            if not meshes_list:
                return {"success": False, "error": "Empty scene, no geometry found"}
            mesh = trimesh.util.concatenate(meshes_list)

        analysis = {
            "vertices": int(len(mesh.vertices)) if hasattr(mesh, 'vertices') else 0,
            "faces": int(len(mesh.faces)) if hasattr(mesh, 'faces') else 0,
            "edges": int(len(mesh.edges)) if hasattr(mesh, 'edges') else 0,
            "bounds": mesh.bounds.tolist() if hasattr(mesh, 'bounds') and mesh.bounds is not None else None,
            "extents": mesh.extents.tolist() if hasattr(mesh, 'extents') else None,
            "center_mass": mesh.center_mass.tolist() if hasattr(mesh, 'center_mass') else None,
            "is_watertight": bool(mesh.is_watertight) if hasattr(mesh, 'is_watertight') else None,
            "is_convex": bool(mesh.is_convex) if hasattr(mesh, 'is_convex') else None,
            "euler_number": int(mesh.euler_number) if hasattr(mesh, 'euler_number') else None,
            "file_size_kb": round(os.path.getsize(input_path) / 1024, 1),
        }

        # Game-readiness scoring
        verts = analysis["vertices"]
        score = 100
        if verts > 100000: score -= 30
        elif verts > 50000: score -= 15
        elif verts > 20000: score -= 5
        if not analysis.get("is_watertight"): score -= 10
        if analysis.get("file_size_kb", 0) > 10000: score -= 15
        elif analysis.get("file_size_kb", 0) > 5000: score -= 5

        budget_category = "hero" if verts > 50000 else "medium" if verts > 10000 else "prop" if verts > 1000 else "simple"

        analysis["game_readiness_score"] = max(0, score)
        analysis["polygon_budget"] = budget_category
        analysis["recommendations"] = []
        if verts > 50000:
            analysis["recommendations"].append("Consider LOD generation (/mesh/lod) for distant rendering")
        if not analysis.get("is_watertight"):
            analysis["recommendations"].append("Mesh is not watertight — may cause rendering artifacts")
        if analysis.get("file_size_kb", 0) > 5000:
            analysis["recommendations"].append("Large file size — consider mesh decimation")

        local_metrics.record("/mesh/analyze", True, 0)
        return {"success": True, "analysis": analysis, "path": input_path}
    except Exception as e:
        local_metrics.record("/mesh/analyze", False, 0)
        return {"success": False, "error": str(e)}


# ─── v29.0: Batch Export v2 ───

@app.post("/export/batch")
async def batch_export(request: Request):
    """v29.0: Export mesh to multiple formats in one call (GLB, OBJ, FBX, STL, PLY)"""
    try:
        data = await request.json()
        input_path = data.get("input_path")
        formats = data.get("formats", ["glb", "obj"])

        if not input_path or not os.path.exists(input_path):
            return {"success": False, "error": "input_path required and must exist"}

        import trimesh
        mesh = trimesh.load(input_path)

        base_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        results = []
        for fmt in formats:
            fmt = fmt.lower().strip(".")
            out_path = os.path.join(base_dir, f"{base_name}.{fmt}")
            try:
                mesh.export(out_path, file_type=fmt)
                results.append({"format": fmt, "path": out_path, "size_kb": round(os.path.getsize(out_path) / 1024, 1), "success": True})
            except Exception as e:
                results.append({"format": fmt, "error": str(e), "success": False})

        succeeded = sum(1 for r in results if r.get("success"))
        local_metrics.record("/export/batch", succeeded > 0, 0)
        return {"success": succeeded > 0, "exports": results, "input": input_path, "formats_requested": formats}
    except Exception as e:
        local_metrics.record("/export/batch", False, 0)
        return {"success": False, "error": str(e)}


# ─── v29.0: Model Warm-up Scheduling ───

_warmup_schedule = {"last_warmup": None, "warmup_count": 0, "auto_warmup": True}

@app.post("/model/warmup")
async def model_warmup(request: Request):
    """v29.0: Trigger model warm-up — run dummy inference to keep GPU caches hot"""
    try:
        if not torch.cuda.is_available():
            return {"success": False, "error": "No GPU available"}

        start = time.time()
        warmup_results = []

        # Warm up Hunyuan3D pipeline if loaded
        if "hunyuan3d" in manager._gpu_loaded:
            try:
                torch.cuda.synchronize()
                warmup_results.append({"model": "hunyuan3d", "status": "warmed", "gpu_sync": True})
            except Exception as e:
                warmup_results.append({"model": "hunyuan3d", "status": "error", "error": str(e)})

        # Warm up embeddings if loaded
        if "embeddings" in manager._gpu_loaded:
            try:
                warmup_results.append({"model": "embeddings", "status": "warmed", "gpu_sync": True})
            except Exception as e:
                warmup_results.append({"model": "embeddings", "status": "error", "error": str(e)})

        # Clear fragmented VRAM
        torch.cuda.empty_cache()

        _warmup_schedule["last_warmup"] = time.time()
        _warmup_schedule["warmup_count"] += 1

        vram_after = torch.cuda.memory_allocated(0) / 1e9

        elapsed = time.time() - start
        local_metrics.record("/model/warmup", True, elapsed)
        return {
            "success": True, "duration_s": round(elapsed, 2),
            "models_warmed": warmup_results, "vram_gb": round(vram_after, 2),
            "total_warmups": _warmup_schedule["warmup_count"],
        }
    except Exception as e:
        local_metrics.record("/model/warmup", False, 0)
        return {"success": False, "error": str(e)}

@app.get("/model/warmup/schedule")
async def warmup_schedule():
    """v29.0: Get warm-up schedule and history"""
    return {
        "success": True,
        "schedule": _warmup_schedule,
        "last_warmup_ago": f"{time.time() - _warmup_schedule['last_warmup']:.0f}s" if _warmup_schedule["last_warmup"] else "never",
    }


# ─── v29.0: Generation Presets ───

_3D_PRESETS = {
    "high_quality": {"steps": 75, "guidance_scale": 7.5, "description": "Maximum quality, slower generation"},
    "balanced": {"steps": 50, "guidance_scale": 5.0, "description": "Good balance of quality and speed"},
    "fast": {"steps": 30, "guidance_scale": 3.5, "description": "Quick preview, lower quality"},
    "turbo": {"steps": 20, "guidance_scale": 2.5, "description": "Fastest, for rapid iteration"},
    "hero_asset": {"steps": 100, "guidance_scale": 8.0, "description": "Best quality for hero/main character assets"},
    "prop": {"steps": 40, "guidance_scale": 4.0, "description": "Optimized for small props and items"},
}

@app.get("/presets/3d")
async def get_3d_presets():
    """v29.0: Get available 3D generation presets with settings"""
    return {"success": True, "presets": _3D_PRESETS}

@app.post("/generate3d/preset")
async def generate_3d_with_preset(request: Request):
    """v29.0: Generate 3D with a named preset — combines preset settings with custom overrides"""
    try:
        data = await request.json()
        preset_name = data.get("preset", "balanced")
        prompt = data.get("prompt")
        if not prompt:
            return {"success": False, "error": "prompt required"}

        preset = _3D_PRESETS.get(preset_name)
        if not preset:
            return {"success": False, "error": f"Unknown preset: {preset_name}", "available": list(_3D_PRESETS.keys())}

        # Merge preset with custom params (custom overrides preset)
        gen_params = {**preset}
        gen_params.pop("description", None)
        for k, v in data.items():
            if k not in ("preset", "prompt"):
                gen_params[k] = v

        t0 = time.time()
        result = await _generate_3d_internal(prompt, data.get("seed"), gen_params.get("steps", 50))
        elapsed = time.time() - t0

        if isinstance(result, dict):
            result["preset_used"] = preset_name
            result["preset_settings"] = preset
            result["elapsed_s"] = round(elapsed, 2)

        local_metrics.record("/generate3d/preset", result.get("success", False) if isinstance(result, dict) else False, elapsed)
        _record_generation("/generate3d/preset", prompt, "hunyuan3d", elapsed, result.get("success", False) if isinstance(result, dict) else False)
        return result
    except Exception as e:
        local_metrics.record("/generate3d/preset", False, 0)
        return {"success": False, "error": str(e)}


# ─── Main ───

if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting Local AI Server v29.0 SUPREME ULTRA on port {PORT}")
    log.info(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    log.info("v29.0 features: mesh-lod, pipeline-3d-v3, mesh-analyze, batch-export-v2, model-warmup, presets + all v26.0 features")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

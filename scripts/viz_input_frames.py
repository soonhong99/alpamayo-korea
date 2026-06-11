"""
viz_input_frames.py  —  Visualize the actual camera frames fed into the model

Saves recognizable RGB images and an animated GIF of the driving scene.
The PhysicalAI dataset stores images pre-normalized for the VLM.
This script tries multiple denormalization methods and saves the best.

Output:
  evaluation/results/streaming/exp1_decode_skip/input_viz/
  ├── cameras_t0.png          4-camera grid at prediction time (t=0)
  ├── cam{i}_sequence.png     4-frame time strip per camera
  ├── front_left_video.gif    animated GIF  (front-left cam, 4 frames)
  ├── all_cameras_video.gif   2×2 grid animated GIF
  ├── debug_normalization.png shows 4 denorm methods side-by-side
  └── info.txt                tensor stats, timestamps, camera indices

Usage:
  cd ~/alpamayo1.5
  source a1_5_venv/bin/activate
  python3 scripts/viz_input_frames.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.1,
})

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US   = 5_100_000
OUT_DIR = ROOT / "evaluation/results/streaming/exp1_decode_skip/input_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Source: ~/alpamayo1.5/src/alpamayo1_5/helper.py  CAMERA_DISPLAY_NAMES (confirmed)
CAM_LABELS = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}


# ── Image normalization ────────────────────────────────────────────────────────

def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """Convert (3,H,W) → (H,W,3)."""
    if arr.ndim == 3 and arr.shape[0] == 3:
        return arr.transpose(1, 2, 0)
    return arr


def denorm_imagenet(arr: np.ndarray) -> np.ndarray:
    """Reverse ImageNet mean/std normalization."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out  = arr * std + mean
    return np.clip(out, 0, 1)


def denorm_minmax(arr: np.ndarray) -> np.ndarray:
    """Global min-max stretch to [0,1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def denorm_percentile(arr: np.ndarray) -> np.ndarray:
    """Robust 1–99 percentile stretch (handles outlier pixels)."""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def denorm_symrange(arr: np.ndarray) -> np.ndarray:
    """Convert [-1, 1] → [0, 1]."""
    return np.clip((arr + 1.0) / 2.0, 0, 1)


def auto_denorm(arr: np.ndarray) -> np.ndarray:
    """
    Heuristic: pick the most likely denormalization based on value range.
      [0, 1]    → display as-is
      [0, 255]  → divide by 255
      [-1, 1]   → (x+1)/2
      ImageNet  → reverse mean/std  (values outside [-3, 3] range → use percentile)
    """
    vmin, vmax = float(arr.min()), float(arr.max())

    if vmax <= 1.01 and vmin >= -0.01:
        # already [0,1]
        return np.clip(arr, 0, 1)

    if vmax > 10:
        # [0, 255] stored as float
        return np.clip(arr / 255.0, 0, 1)

    if vmin >= -1.05 and vmax <= 1.05:
        # [-1, 1]
        return denorm_symrange(arr)

    if vmin > -3.5 and vmax < 3.5:
        # Likely ImageNet normalization
        out = denorm_imagenet(arr)
        # Sanity check: if result looks valid (not mostly saturated), use it
        if out.mean() > 0.05 and out.mean() < 0.95:
            return out

    # Fallback: robust percentile stretch
    return denorm_percentile(arr)


def tensor_to_uint8(t, method: str = "auto") -> np.ndarray:
    """
    Convert (3,H,W) tensor/array → (H,W,3) uint8 displayable image.
    method: "auto" | "imagenet" | "minmax" | "percentile" | "symrange"

    NOTE: load_physical_aiavdataset returns raw uint8 [0,255] images.
    auto_denorm handles this: vmax > 10 → divide by 255.
    """
    import torch
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy()
    else:
        arr = np.array(t)

    # If already uint8, just reorder channels and return
    if arr.dtype == np.uint8:
        return _to_hwc(arr)

    arr = arr.astype(np.float32)
    arr = _to_hwc(arr)

    fn = {"auto":       auto_denorm,
          "imagenet":   denorm_imagenet,
          "minmax":     denorm_minmax,
          "percentile": denorm_percentile,
          "symrange":   denorm_symrange}[method]

    out = fn(arr)
    return (out * 255).astype(np.uint8)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> dict:
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    print(f"Loading PhysicalAI clip  {CLIP_ID}  t0={T0_US/1e6:.1f} s ...")
    return load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)


def extract_frames(data: dict) -> np.ndarray:
    """Return (N_CAM, N_FRAME, H, W, 3) uint8 with auto denorm."""
    import torch
    raw = data["image_frames"]          # (N_CAM, N_FRAME, 3, H, W)
    if isinstance(raw, torch.Tensor):
        raw = raw.cpu()
    nc, nf = raw.shape[0], raw.shape[1]
    H, W   = raw.shape[3], raw.shape[4]

    out = np.zeros((nc, nf, H, W, 3), dtype=np.uint8)
    for ci in range(nc):
        for fi in range(nf):
            out[ci, fi] = tensor_to_uint8(raw[ci, fi], method="auto")
    return out


def get_timestamps(data: dict) -> np.ndarray | None:
    import torch
    ts = data.get("relative_timestamps")
    if ts is None:
        return None
    if isinstance(ts, torch.Tensor):
        ts = ts.cpu().numpy()
    return ts


def get_cam_label(idx: int, camera_indices) -> str:
    import torch
    ci = camera_indices
    if isinstance(ci, torch.Tensor):
        ci = ci.cpu().numpy()
    ci = np.array(ci)
    if idx < len(ci):
        return CAM_LABELS.get(int(ci[idx]), f"Camera {int(ci[idx])}")
    return f"Camera {idx}"


# ── Figure A: 4-camera grid at t=0 (most recent frame) ───────────────────────

def fig_cameras_t0(frames: np.ndarray, camera_indices, timestamps):
    nc = frames.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ci in range(nc):
        ax = axes[ci]
        img = frames[ci, -1]   # most recent frame (t = 0)
        ax.imshow(img, interpolation="bilinear")
        ax.axis("off")
        ts_str = ""
        if timestamps is not None and ci < timestamps.shape[0]:
            ts_str = f"  (t = {timestamps[ci, -1]:.2f} s)"
        label = get_cam_label(ci, camera_indices)
        ax.set_title(f"{label}{ts_str}", fontsize=12, fontweight="bold", pad=5)

        # Subtle border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor("#444")

    fig.suptitle(
        f"Model Input  —  4-Camera View at Prediction Time  (t = 0)\n"
        f"Clip: {CLIP_ID}   t₀ = {T0_US/1e6:.1f} s\n"
        f"Resolution: {frames.shape[3]}×{frames.shape[4]} px  per camera",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out = OUT_DIR / "cameras_t0.png"
    plt.savefig(str(out))
    plt.close()
    print(f"Saved: {out}")


# ── Figure B: Per-camera 4-frame time strip ───────────────────────────────────

def fig_camera_sequences(frames: np.ndarray, camera_indices, timestamps):
    nc, nf, H, W, _ = frames.shape

    for ci in range(nc):
        label = get_cam_label(ci, camera_indices)
        fig, axes = plt.subplots(1, nf, figsize=(nf * 5, 3.6))
        if nf == 1:
            axes = [axes]

        for fi, ax in enumerate(axes):
            ax.imshow(frames[ci, fi], interpolation="bilinear")
            ax.axis("off")
            ts_str = ""
            if timestamps is not None:
                ts_str = f"t = {timestamps[ci, fi]:.2f} s"
            else:
                ts_str = f"frame {fi}"
            ax.set_title(ts_str, fontsize=11)

            # Highlight last frame (t = 0, used for prediction)
            if fi == nf - 1:
                rect = plt.Rectangle((0, 0), W - 1, H - 1,
                                     linewidth=3, edgecolor="#e74c3c",
                                     facecolor="none",
                                     transform=ax.transData, zorder=10)
                ax.add_patch(rect)
                ax.set_title(ts_str + "  ← prediction time",
                             fontsize=11, color="#c0392b", fontweight="bold")

        fig.suptitle(f"{label}  —  4-frame input sequence  (oldest → newest)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()

        out = OUT_DIR / f"cam{ci}_{label.replace(' ', '_').lower()}_sequence.png"
        plt.savefig(str(out))
        plt.close()
        print(f"Saved: {out}")


# ── Debug: side-by-side normalization comparison ──────────────────────────────

def fig_normalization_debug(data: dict):
    """Show the same frame under 4 different normalizations."""
    import torch
    raw = data["image_frames"]
    if isinstance(raw, torch.Tensor):
        raw = raw.cpu()

    frame = raw[0, -1]  # front-left, last frame

    methods = [
        ("auto (heuristic)",    "auto"),
        ("ImageNet denorm",     "imagenet"),
        ("Percentile stretch",  "percentile"),
        ("Min-max stretch",     "minmax"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (title, method) in zip(axes, methods):
        img = tensor_to_uint8(frame, method=method)
        ax.imshow(img, interpolation="bilinear")
        ax.axis("off")
        arr = _to_hwc(frame.numpy() if isinstance(frame, torch.Tensor)
                      else np.array(frame, dtype=np.float32))
        ax.set_title(f"{title}\nrange [{arr.min():.2f}, {arr.max():.2f}]",
                     fontsize=10)

    fig.suptitle("Normalization Debug  —  Front-Left Camera, t=0\n"
                 "Use whichever panel shows the most recognizable road scene",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = OUT_DIR / "debug_normalization.png"
    plt.savefig(str(out))
    plt.close()
    print(f"Saved: {out}")


# ── Animated GIF: front-left camera ──────────────────────────────────────────

def make_gif_front(frames: np.ndarray, camera_indices, timestamps):
    """Save front-left camera frames as animated GIF."""
    try:
        import imageio
    except ImportError:
        print("  imageio not found — installing...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "imageio", "-q"])
        import imageio

    label = get_cam_label(0, camera_indices)
    nc, nf, H, W, _ = frames.shape

    # Resize to 960×540 for manageable GIF file size (1920×1080 → too large)
    GIF_W, GIF_H = 960, 540

    gif_frames = []
    for fi in range(nf):
        img = frames[0, fi]

        # Resize using numpy (simple area average via slicing)
        img_small = img[::2, ::2]   # 1080→540, 1920→960

        fig, ax = plt.subplots(figsize=(GIF_W / 100, GIF_H / 100), dpi=100)
        fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(img_small, interpolation="bilinear")
        ax.axis("off")

        ts_str = ""
        if timestamps is not None:
            ts = float(timestamps[0, fi])
            ts_str = f"{label}   t = {ts:.2f} s"
            if fi == nf - 1:
                ts_str += "  (prediction frame)"

        ax.text(0.01, 0.98, ts_str, transform=ax.transAxes,
                fontsize=11, color="white", va="top",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.65))
        if fi == nf - 1:
            ax.text(0.01, 0.02,
                    'CoC: "Keep distance to the lead vehicle\n'
                    '       since it is directly ahead in our lane"',
                    transform=ax.transAxes,
                    fontsize=10, color="#f1c40f",
                    va="bottom", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.72))

        fig.canvas.draw()
        # buffer_rgba() replaces deprecated tostring_rgb()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        gif_frames.append(buf[:, :, :3])   # drop alpha
        plt.close(fig)

    out = OUT_DIR / "front_left_video.gif"
    imageio.mimsave(str(out), gif_frames, fps=4, loop=0)
    print(f"Saved: {out}")


# ── Animated GIF: all 4 cameras 2×2 ─────────────────────────────────────────

def make_gif_all_cameras(frames: np.ndarray, camera_indices, timestamps):
    """Save 2×2 camera grid as animated GIF."""
    try:
        import imageio
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "imageio", "-q"])
        import imageio

    nc, nf, H, W, _ = frames.shape
    gif_frames = []

    for fi in range(nf):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        axes = axes.flatten()

        for ci in range(nc):
            ax = axes[ci]
            img_small = frames[ci, fi][::2, ::2]   # downsample to 540×960
            ax.imshow(img_small, interpolation="bilinear")
            ax.axis("off")
            label = get_cam_label(ci, camera_indices)
            ts_str = ""
            if timestamps is not None:
                ts_str = f"  t={timestamps[ci, fi]:.2f}s"
            title_color = "#c0392b" if fi == nf - 1 else "black"
            ax.set_title(label + ts_str, fontsize=9,
                         color=title_color, fontweight="bold")

        suptitle = (f"All Cameras — Frame {fi + 1} / {nf}"
                    + ("  ← PREDICTION FRAME" if fi == nf - 1 else ""))
        fig.suptitle(suptitle, fontsize=11, fontweight="bold",
                     color="#c0392b" if fi == nf - 1 else "black")
        plt.tight_layout()

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        gif_frames.append(buf[:, :, :3])
        plt.close(fig)

    out = OUT_DIR / "all_cameras_video.gif"
    imageio.mimsave(str(out), gif_frames, fps=3, loop=0)
    print(f"Saved: {out}")


# ── info.txt ──────────────────────────────────────────────────────────────────

def save_info(data: dict):
    import torch
    lines = [
        "=== Model Input Data Info ===",
        f"clip_id       : {CLIP_ID}",
        f"t0            : {T0_US} us  ({T0_US/1e6:.2f} s)",
        "",
        "Tensors:",
    ]
    for k, v in data.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            lines.append(f"  {k:28s}: shape={list(v.shape)}"
                         f"  dtype={v.dtype}"
                         f"  min={v.min():.4f}  max={v.max():.4f}"
                         f"  mean={v.mean():.4f}")
        else:
            lines.append(f"  {k:28s}: {type(v).__name__} = {v}")

    ts = data.get("relative_timestamps")
    if ts is not None:
        import torch
        if isinstance(ts, torch.Tensor):
            ts = ts.cpu().numpy()
        lines += ["", "Timestamps per camera (relative_timestamps):"]
        for ci in range(ts.shape[0]):
            label = get_cam_label(ci, data.get("camera_indices"))
            lines.append(f"  {label:12s}: {[f'{v:.3f}s' for v in ts[ci]]}")

    txt = "\n".join(lines)
    (OUT_DIR / "info.txt").write_text(txt, encoding="utf-8")
    print(f"Saved: {OUT_DIR / 'info.txt'}")
    print()
    print(txt)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Input Frame Visualization")
    print(f"  Clip : {CLIP_ID}")
    print(f"  t0   : {T0_US / 1e6:.1f} s")
    print("=" * 60)

    data = load_data()
    save_info(data)

    ci   = data.get("camera_indices")
    ts   = get_timestamps(data)
    frames = extract_frames(data)

    nc, nf, H, W, _ = frames.shape
    print(f"\nImage array: {nc} cameras × {nf} frames × {H}×{W} px")

    print("\nRendering figures...")
    fig_normalization_debug(data)    # save first so user can verify image quality
    fig_cameras_t0(frames, ci, ts)
    fig_camera_sequences(frames, ci, ts)

    print("\nCreating animated GIFs...")
    make_gif_front(frames, ci, ts)
    make_gif_all_cameras(frames, ci, ts)

    print(f"\nDone. Output: {OUT_DIR.resolve()}")
    print("\nFetch on Windows (WSL):")
    print("  scp -r 'ice401@100.95.177.101:"
          "~/alpamayo1.5/evaluation/results/streaming/exp1_decode_skip/input_viz/' "
          "/mnt/c/Users/nanay/Desktop/Alphamayo/evaluation/results/streaming/"
          "exp1_decode_skip/")


if __name__ == "__main__":
    main()

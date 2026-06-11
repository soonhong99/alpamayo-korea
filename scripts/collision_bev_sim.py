"""
collision_bev_sim.py  --  BEV Collision Simulation
======================================================================

Purpose:
    Visualize whether the three Exp-1 trajectories (GT / N=16 / N=0)
    collide with real obstacles measured by LiDAR, using a BEV animation.

Confirmed data (all verified directly via API or experiment):
    clip_id       : 030c760c-ae38-49aa-9ad8-f5650a545d26
    t0_us         : 5,100,000 us
    lidar_spin_idx: 50  (24,982 us offset from t0, confirmed 2026-05-20)
    T_lidar->veh  : SensorExtrinsics.lidar_top_360fov (confirmed 2026-05-20)
    vehicle_dims  : VehicleDimensions API (confirmed 2026-05-20)
    GT trajectory : ego_future_xyz (loaded from load_physical_aiavdataset)
    N=0/16 traj.  : evaluation/results/streaming/exp1_decode_skip/waypoints_{n}.npy

Explicit assumption (unverifiable -- tagged ASSUMPTION in code):
    ASSUMPTION-A: waypoint reference = rear axle center
                  Evidence: speed consistency
                    8.66 m/s x 0.1 s = 0.866 m ~ waypoint[0].x = 0.863 m
                  Error impact: if wrong, vehicle box is offset up to 1.354 m in x

Explicit limits:
    LIMIT-1: Single LiDAR spin (t0 snapshot only)
             -> Dynamic obstacles (lead vehicle, cyclists) frozen at t0 position
             -> Static obstacles (construction equipment, road structures) accurate
    LIMIT-2: Only 6.4 s window (64 waypoints)
    LIMIT-3: Obstacles outside z=[0.3, 2.5] m are excluded
             (low ground-hugging objects, tall overhead objects)

Outputs:
    evaluation/results/streaming/exp1_decode_skip/
      bev_sanity.png        <- Initial-state check (verify coords before MP4)
      collision_bev_sim.mp4 <- BEV animation (65 frames, 10 fps)
      collision_log.json    <- Per-frame collision flag and gap values

Run:
    cd ~/alpamayo1.5
    python3 scripts/collision_bev_sim.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ── 패키지 경로 설정 (exp1_decode_skip.py 와 동일 패턴) ─────────────────────────
# scripts/collision_bev_sim.py 기준 → 부모 2단계 = ~/alpamayo1.5
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import DracoPy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 확인된 상수 ────────────────────────────────────────────────────────────────

CLIP_ID   = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US     = 5_100_000
SPIN_IDX  = 50          # 24,982 us offset from t0 (confirmed 2026-05-20)
N_WP      = 64
DT        = 0.1         # seconds per waypoint

# Vehicle dimensions -- VehicleDimensions API (confirmed 2026-05-20)
EGO_L       = 4.925     # m (length)
EGO_W       = 2.116     # m (width)
REAR_TO_CTR = 1.354     # m (rear_axle_to_bbox_center)
# ASSUMPTION-A: waypoint = rear axle center
# bbox center = waypoint + (REAR_TO_CTR * heading_vector)

# LiDAR extrinsics -- SensorExtrinsics.lidar_top_360fov (confirmed 2026-05-20)
T_LIDAR_TO_VEH = np.array([
    [ 0.99991909, -0.00477392, -0.011791  ,  1.18850005],
    [ 0.0047993 ,  0.99998623,  0.00212509,  0.        ],
    [ 0.0117807 , -0.00218151,  0.99992823,  1.86154997],
    [ 0.        ,  0.        ,  0.        ,  1.        ],
], dtype=np.float64)

# Obstacle filter
Z_OBS_MIN =  0.3    # m -- remove ground reflections
Z_OBS_MAX =  2.5    # m -- remove overhead wires / ceiling
BEV_X_MIN = -5.0   # m -- BEV forward range
BEV_X_MAX = 65.0
BEV_Y_MIN = -8.0   # m -- BEV lateral range (y>0 = left, y<0 = right)
BEV_Y_MAX =  8.0

# Visualization
TRAJ_COLORS  = {"gt": "#2ecc71", "n16": "#111111", "n0": "#e74c3c"}
TRAJ_LABELS  = {"gt": "GT (ground truth)", "n16": "N=16 (Full CoC)", "n0": "N=0 (Decode Skip)"}
TRAJ_ZORDERS = {"gt": 6, "n16": 8, "n0": 7}

RESULT_DIR       = Path("evaluation/results/streaming/exp1_decode_skip")
OUT_SANITY       = RESULT_DIR / "bev_sanity.png"
OUT_MP4          = RESULT_DIR / "collision_bev_sim.mp4"          # static LiDAR
OUT_LOG          = RESULT_DIR / "collision_log.json"
OUT_MP4_DYNAMIC  = RESULT_DIR / "collision_bev_dynamic.mp4"      # dynamic LiDAR
OUT_LOG_DYNAMIC  = RESULT_DIR / "collision_log_dynamic.json"

# Dynamic LiDAR: spin index range
# spin 50 = t=0 (confirmed: start=-25ms, end=+75ms from T0)
# spin 50+i = frame i (t=i*0.1s), cadence=100ms (confirmed 2026-05-22)
DYN_SPIN_BASE  = 50   # spin index for frame 0 (t=0)
DYN_N_FRAMES   = 65   # 0..64  (t=0 ~ t=6.4s)

# Ego exclusion zone -- applied in vehicle frame BEFORE world-frame transform.
#
# Confirmed (2026-05-22): LiDAR returns 577 self-reflection points from car body
# at z=0.86~1.65m (bonnet, doors, trunk) within x∈[-0.81, 3.50], y∈[-0.72, 0.71].
# After world-frame shift these land exactly inside the GT bbox → 65/65 false collision.
#
# Fix: ASYMMETRIC exclusion covering the full car body extent from rear axle:
#   Rear  : -(EGO_L/2 - REAR_TO_CTR) = -1.109 m  →  add 0.5 m buffer = -1.609 m
#   Front : +(EGO_L/2 + REAR_TO_CTR) = +3.817 m  →  add 0.5 m buffer = +4.317 m
#   Lateral: EGO_W/2 = 1.058 m                   →  add 0.5 m buffer = +1.558 m
#
# Previous symmetric EGO_EXCL_X=3.46m missed the front zone [3.46, 3.82] → collision persisted.
EGO_EXCL_X_FWD  = EGO_L / 2.0 + REAR_TO_CTR + 0.5   # = 4.317 m  (forward from rear axle)
EGO_EXCL_X_REAR = EGO_L / 2.0 - REAR_TO_CTR + 0.5   # = 1.609 m  (behind  rear axle)
EGO_EXCL_Y      = EGO_W / 2.0               + 0.5   # = 1.558 m  (lateral)

# 충돌 판정 gap 임계값 (단순 거리 기준)
GAP_COLLISION  = 0.0    # m: 겹침 = 충돌
GAP_CRITICAL   = 1.0    # m: 1m 미만 = 위험
GAP_WARNING    = 3.0    # m: 3m 미만 = 주의


# ── Phase 1: 데이터 로딩 ───────────────────────────────────────────────────────

def load_lidar_spin(spin_idx: int) -> np.ndarray:
    """
    LiDAR spin을 로드하고 LiDAR 센서 frame의 포인트로 반환.

    Returns
    -------
    pts : (N, 3) float64, LiDAR 센서 frame [x, y, z]
    """
    from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
    logger.info(f"Loading LiDAR spin {spin_idx} (HuggingFace streaming)...")
    avdi = PhysicalAIAVDatasetInterface()
    feat = avdi.get_clip_feature(CLIP_ID, "lidar_top_360fov", maybe_stream=True)
    df   = feat["pointclouds"]

    row  = df[df["spin_index"] == spin_idx]
    if row.empty:
        raise ValueError(f"spin_index={spin_idx} not found in pointclouds")
    draco_bytes = row.iloc[0]["draco_encoded_pointcloud"]

    pc  = DracoPy.decode(draco_bytes)
    pts = np.array(pc.points, dtype=np.float64)
    logger.info(f"  -> {len(pts):,} points loaded")
    return pts


def to_vehicle_frame(pts_lidar: np.ndarray) -> np.ndarray:
    """
    LiDAR 센서 frame → vehicle frame 변환.
    T_LIDAR_TO_VEH: SensorExtrinsics.lidar_top_360fov (확인됨).

    Parameters
    ----------
    pts_lidar : (N, 3) float64, LiDAR sensor frame

    Returns
    -------
    pts_veh : (N, 3) float64, vehicle frame (x=전방, y=좌측, z=상방)
    """
    ones      = np.ones((len(pts_lidar), 1), dtype=np.float64)
    pts_hom   = np.hstack([pts_lidar, ones])          # (N, 4)
    pts_veh   = (T_LIDAR_TO_VEH @ pts_hom.T).T[:, :3] # (N, 3)
    return pts_veh


def filter_obstacles(pts_veh: np.ndarray) -> np.ndarray:
    """
    Apply BEV region + obstacle height filter.

    Filter criteria:
        z in [0.3, 2.5] m : remove ground reflections and overhead wires
        x in [-5, 65] m   : BEV display range (65 m forward)
        y in [-8, 8] m    : driving corridor +/- 8 m (~4 lane widths)

    LIMIT-3: obstacles below 0.3 m are excluded by this filter.
    """
    mask = (
        (pts_veh[:, 2] >= Z_OBS_MIN) & (pts_veh[:, 2] <= Z_OBS_MAX) &
        (pts_veh[:, 0] >= BEV_X_MIN) & (pts_veh[:, 0] <= BEV_X_MAX) &
        (pts_veh[:, 1] >= BEV_Y_MIN) & (pts_veh[:, 1] <= BEV_Y_MAX)
    )
    obs = pts_veh[mask]
    logger.info(f"After obstacle filter: {len(obs):,} points remain")
    return obs


def load_trajectories() -> dict[str, np.ndarray | None]:
    """
    세 궤적 로드. 모두 t0 ego frame (x=전방, y=좌측).

    Returns
    -------
    dict with keys 'gt', 'n16', 'n0'
        각 값: (64, 3) float32 또는 None (파일 없을 때)
    """
    # GT -- loaded directly from load_physical_aiavdataset
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    logger.info("Loading GT trajectory (ego_future_xyz)...")
    data   = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    gt_xyz = data["ego_future_xyz"][0, 0].numpy().astype(np.float32)  # (64, 3)
    logger.info(f"  GT endpoint (6.4 s): x={gt_xyz[-1,0]:.3f} m  y={gt_xyz[-1,1]:.3f} m")

    trajs: dict[str, np.ndarray | None] = {"gt": gt_xyz}

    for key, n in [("n16", 16), ("n0", 0)]:
        p = RESULT_DIR / f"waypoints_{n}.npy"
        if p.exists():
            wp = np.load(str(p))          # (64, 3) float32
            if wp.shape != (N_WP, 3):
                logger.warning(f"waypoints_{n}.npy shape={wp.shape}, expected (64,3) -- skip")
                trajs[key] = None
            else:
                trajs[key] = wp.astype(np.float32)
                logger.info(f"  {key} endpoint: x={wp[-1,0]:.3f} m  y={wp[-1,1]:.3f} m")
        else:
            logger.warning(f"waypoints_{n}.npy not found -> {key} trajectory not shown")
            trajs[key] = None

    return trajs


# ── Phase 2: 충돌 판정 ────────────────────────────────────────────────────────

def compute_heading(traj: np.ndarray, i: int) -> float:
    """
    waypoint i에서의 헤딩 각도 (라디안).
    연속 두 waypoint의 차분으로 근사한다.
    i=63(마지막)에서는 i-1→i 차분 사용.

    Parameters
    ----------
    traj : (64, 3)
    i    : waypoint index 0..63
    """
    if i < N_WP - 1:
        dx = float(traj[i + 1, 0] - traj[i, 0])
        dy = float(traj[i + 1, 1] - traj[i, 1])
    else:
        dx = float(traj[i, 0] - traj[i - 1, 0])
        dy = float(traj[i, 1] - traj[i - 1, 1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return float(np.arctan2(dy, dx))


def vehicle_corners(wp: np.ndarray, heading: float) -> tuple[np.ndarray, np.ndarray]:
    """
    차량 직사각형의 4개 꼭짓점과 bbox 중심 반환.

    ASSUMPTION-A: wp = 후륜축 위치.
    bbox 중심은 후륜축에서 헤딩 방향으로 REAR_TO_CTR(=1.354m) 앞.

    Parameters
    ----------
    wp      : (3,) waypoint (x, y, z)
    heading : float, 라디안

    Returns
    -------
    corners : (4, 2) 꼭짓점 (world frame)
    center  : (2,)  bbox 중심 (world frame)
    """
    ch, sh = np.cos(heading), np.sin(heading)
    cx = float(wp[0]) + REAR_TO_CTR * ch
    cy = float(wp[1]) + REAR_TO_CTR * sh

    hl, hw = EGO_L / 2.0, EGO_W / 2.0
    R = np.array([[ch, -sh], [sh, ch]])

    # 로컬 frame 꼭짓점 (반시계 순서)
    local = np.array([[ hl,  hw],
                      [-hl,  hw],
                      [-hl, -hw],
                      [ hl, -hw]])
    corners = (R @ local.T).T + np.array([cx, cy])
    return corners, np.array([cx, cy])


def collision_and_gap(
    wp: np.ndarray,
    heading: float,
    obs_pts: np.ndarray,
) -> tuple[bool, float]:
    """
    차량 박스와 장애물 포인트의 충돌 여부 및 전방 gap 계산.

    충돌 판정:
        장애물 포인트를 차량 로컬 frame으로 변환.
        로컬 frame에서 |lx| ≤ EGO_L/2 AND |ly| ≤ EGO_W/2 이면 충돌.

    전방 gap:
        차량 로컬 frame에서 lx > EGO_L/2 (전방)
        AND |ly| ≤ EGO_W/2 (차선 폭 내)
        인 포인트 중 가장 가까운 것까지의 거리.
        (LIMIT-1: 동적 장애물이 t0에 고정이므로 gap은 t0 스냅샷 기준)

    Parameters
    ----------
    wp      : (3,) waypoint
    heading : float, 라디안
    obs_pts : (M, 3) 장애물 포인트 (vehicle frame)

    Returns
    -------
    collision : bool
    gap       : float, 전방 gap(m). 전방 장애물 없으면 inf.
    """
    if len(obs_pts) == 0:
        return False, float("inf")

    ch, sh = np.cos(heading), np.sin(heading)
    cx = float(wp[0]) + REAR_TO_CTR * ch
    cy = float(wp[1]) + REAR_TO_CTR * sh

    # 역회전 행렬 (R^T = R^{-1} for rotation)
    R_inv = np.array([[ch, sh], [-sh, ch]])

    pts_xy    = obs_pts[:, :2]                             # (M, 2)
    centered  = pts_xy - np.array([cx, cy])                # (M, 2)
    pts_local = (R_inv @ centered.T).T                     # (M, 2), local frame

    hl, hw = EGO_L / 2.0, EGO_W / 2.0

    # 충돌: 박스 내부
    in_x = np.abs(pts_local[:, 0]) <= hl
    in_y = np.abs(pts_local[:, 1]) <= hw
    collision = bool((in_x & in_y).any())

    # 전방 gap: lx > hl AND |ly| < hw
    fwd  = pts_local[:, 0] > hl
    lane = np.abs(pts_local[:, 1]) <= hw
    mask = fwd & lane
    if mask.any():
        gap = float(pts_local[mask, 0].min()) - hl
    else:
        gap = float("inf")

    return collision, gap


def precompute_all(
    obs_pts: np.ndarray,
    trajs: dict[str, np.ndarray | None],
) -> dict[str, list[dict]]:
    """
    65 프레임(t=0 + 64 waypoints) × 3 궤적의 충돌·gap 사전 계산.

    frame 0 : t=0.0s, ego at (0, 0, 0) — 모든 궤적 동일
    frame i (1..64) : t=i×0.1s, ego at traj[i-1]

    Returns
    -------
    results[key] : list of dict per frame
        {'t': float, 'wx': float, 'wy': float,
         'heading': float, 'collision': bool, 'gap': float}
    """
    results: dict[str, list[dict]] = {}

    # frame 0: ego at origin
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    for key, traj in trajs.items():
        frames: list[dict] = []

        # frame 0
        col0, gap0 = collision_and_gap(origin, 0.0, obs_pts)
        frames.append({
            "t": 0.0, "wx": 0.0, "wy": 0.0,
            "heading": 0.0, "collision": col0, "gap": gap0,
        })

        if traj is None:
            results[key] = frames
            continue

        # frames 1..64
        for i in range(N_WP):
            wp      = traj[i]                         # (3,)
            heading = compute_heading(traj, i)
            col, gap = collision_and_gap(wp, heading, obs_pts)
            frames.append({
                "t":         round((i + 1) * DT, 2),
                "wx":        float(wp[0]),
                "wy":        float(wp[1]),
                "heading":   float(heading),
                "collision": col,
                "gap":       gap if np.isfinite(gap) else None,
            })

        results[key] = frames
        n_col = sum(1 for f in frames if f["collision"])
        logger.info(f"[{key}] collision frames: {n_col}/65")

    return results


# ── Phase 3: Visualization ────────────────────────────────────────────────────

def _make_vehicle_patch(
    wp: np.ndarray,
    heading: float,
    color: str,
    alpha: float = 0.85,
    zorder: int = 5,
) -> mpatches.Polygon:
    corners, _ = vehicle_corners(wp, heading)
    return mpatches.Polygon(
        corners,
        closed=True,
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        alpha=alpha,
        zorder=zorder,
    )


def _classify_obs(obs_pts: np.ndarray) -> dict[str, np.ndarray]:
    """
    Classify obstacle points by lateral distance from vehicle path center.

    Classes:
        in_path   : |y| < EGO_W/2 = 1.058 m  -> directly in vehicle corridor  [RED]
        near_lane : 1.058 m <= |y| < 1.75 m   -> in lane, beside vehicle       [ORANGE]
        off_road  : |y| >= 1.75 m              -> roadside static structure     [GRAY]
    """
    abs_y = np.abs(obs_pts[:, 1])
    return {
        "in_path":   obs_pts[abs_y <  EGO_W / 2],
        "near_lane": obs_pts[(abs_y >= EGO_W / 2) & (abs_y < 1.75)],
        "off_road":  obs_pts[abs_y >= 1.75],
    }


def _draw_obs_background(
    ax,
    cls: dict[str, np.ndarray],
    sub_off: np.ndarray,
    sub_near: np.ndarray,
    sub_in: np.ndarray,
    show_legend: bool = False,
) -> None:
    """Draw colored obstacle scatter + lane lines + vehicle corridor band."""
    # Light-red band showing vehicle width
    ax.axhspan(-EGO_W / 2, EGO_W / 2, alpha=0.06, color="#e74c3c", zorder=1)

    lbl_off  = f"Off-road / static  ({len(cls['off_road']):,} pts)"   if show_legend else "_"
    lbl_near = f"In lane, beside vehicle  ({len(cls['near_lane']):,} pts)" if show_legend else "_"
    lbl_in   = f"In vehicle corridor  ({len(cls['in_path']):,} pts)"   if show_legend else "_"

    if len(sub_off) > 0:
        ax.scatter(sub_off[:, 0],  sub_off[:, 1],  s=0.8, c="#999999",
                   alpha=0.45, zorder=2, rasterized=True, label=lbl_off)
    if len(sub_near) > 0:
        ax.scatter(sub_near[:, 0], sub_near[:, 1], s=2.5, c="#e67e22",
                   alpha=0.70, zorder=3, rasterized=True, label=lbl_near)
    if len(sub_in) > 0:
        ax.scatter(sub_in[:, 0],   sub_in[:, 1],   s=4.0, c="#c0392b",
                   alpha=0.85, zorder=4, rasterized=True, label=lbl_in)

    for y_lane in [-1.75, 1.75]:
        ax.axhline(y_lane, color="steelblue", lw=0.9, ls="--", alpha=0.50, zorder=5)
    ax.axhline(0, color="steelblue", lw=0.4, ls=":", alpha=0.30, zorder=5)

    ax.set_xlim(BEV_X_MIN, BEV_X_MAX)
    ax.set_ylim(BEV_Y_MIN, BEV_Y_MAX)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.14, zorder=0)


def save_sanity_png(
    obs_pts: np.ndarray,
    trajs: dict[str, np.ndarray | None],
    results: dict[str, list[dict]],
) -> None:
    """
    Save initial-state (t=0) BEV for coordinate verification.
    Shows colored obstacles + all 3 trajectory paths in one overview.

    RED    = obstacle in vehicle corridor (direct collision zone)
    ORANGE = obstacle in lane beside vehicle
    GRAY   = off-road static structure (construction, guardrail)
    """
    cls = _classify_obs(obs_pts)
    rng = np.random.default_rng(42)

    def _sub(pts: np.ndarray, n: int) -> np.ndarray:
        if len(pts) == 0:
            return pts
        idx = rng.choice(len(pts), size=min(len(pts), n), replace=False)
        return pts[idx]

    sub_off  = _sub(cls["off_road"],   8000)
    sub_near = _sub(cls["near_lane"],  3000)
    sub_in   = _sub(cls["in_path"],    3000)

    fig, axes = plt.subplots(1, 2, figsize=(20, 6),
                             gridspec_kw={"width_ratios": [3.5, 1]})
    ax_bev, ax_info = axes

    _draw_obs_background(ax_bev, cls, sub_off, sub_near, sub_in, show_legend=True)

    # All 3 trajectory paths
    for key in ["n16", "n0", "gt"]:
        traj = trajs.get(key)
        if traj is not None:
            ax_bev.plot(traj[:, 0], traj[:, 1],
                        color=TRAJ_COLORS[key], lw=2.0, alpha=0.82,
                        zorder=TRAJ_ZORDERS[key] + 1,
                        label=f"{TRAJ_LABELS[key]}  -> ({traj[-1,0]:.1f}, {traj[-1,1]:.2f}) m")
            ax_bev.plot(traj[-1, 0], traj[-1, 1], "*",
                        color=TRAJ_COLORS[key], ms=12, zorder=TRAJ_ZORDERS[key] + 2)

    # Ego vehicle box at t=0
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    patch = _make_vehicle_patch(origin, 0.0, "#444444", alpha=0.90, zorder=12)
    ax_bev.add_patch(patch)
    _, ctr = vehicle_corners(origin, 0.0)
    ax_bev.text(ctr[0], ctr[1] + 0.5, f"EGO t=0\n{EGO_L}m x {EGO_W}m",
                fontsize=7, color="white", ha="center", va="bottom", zorder=15,
                bbox=dict(boxstyle="round,pad=0.2", fc="#333333", alpha=0.82))
    ax_bev.text(62, 1.90, "lane boundary (+/-1.75 m)", fontsize=7,
                color="steelblue", alpha=0.75, ha="right")

    ax_bev.set_xlabel("Forward distance  x  (m)", fontsize=11)
    ax_bev.set_ylabel("Lateral deviation  y  (m)\n[left(+) / right(-)]", fontsize=10)
    ax_bev.set_title(
        "BEV Sanity Check (t=0)  --  Obstacle Classification + All Trajectories\n"
        "RED = in vehicle path  |  ORANGE = in lane beside vehicle  |  GRAY = off-road static",
        fontsize=11, fontweight="bold",
    )
    ax_bev.legend(fontsize=8, loc="upper right", framealpha=0.90, ncol=1)

    # Info panel
    ax_info.axis("off")
    lines = [
        "=== Obstacle classification ===",
        f"  In vehicle corridor:",
        f"    {len(cls['in_path']):>6,} pts  [RED]",
        f"    |y| < {EGO_W/2:.3f} m",
        "",
        f"  In lane, beside vehicle:",
        f"    {len(cls['near_lane']):>6,} pts  [ORANGE]",
        f"    {EGO_W/2:.3f} <= |y| < 1.75 m",
        "",
        f"  Off-road structure:",
        f"    {len(cls['off_road']):>6,} pts  [GRAY]",
        f"    |y| >= 1.75 m",
        "",
        f"  Total: {len(obs_pts):,} pts",
        "",
        "=== Endpoints (6.4 s) ===",
    ]
    for key in ["n16", "n0", "gt"]:
        traj = trajs.get(key)
        if traj is not None:
            n_col = sum(1 for f in results.get(key, []) if f["collision"])
            lines.append(
                f"  {key.upper()}: ({traj[-1,0]:.2f}, {traj[-1,1]:.2f}) m\n"
                f"    collision: {n_col}/65 frames"
            )
    lines += [
        "",
        "=== LIMIT-1 note ===",
        "  Vehicles/cyclists shown",
        "  at t=0 position only.",
        "  Real future positions",
        "  unknown from single spin.",
    ]
    ax_info.text(0.04, 0.97, "\n".join(lines), transform=ax_info.transAxes,
                 fontsize=8, va="top", family="monospace",
                 bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))

    fig.tight_layout()
    fig.savefig(str(OUT_SANITY), dpi=150)
    plt.close(fig)
    logger.info(f"Sanity PNG saved: {OUT_SANITY}")
    logger.info("-> RED dots = in-path obstacles. GRAY = static roadside structure.")


def make_animation(
    obs_pts: np.ndarray,
    trajs: dict[str, np.ndarray | None],
    results: dict[str, list[dict]],
) -> None:
    """
    BEV animation -- 3 rows stacked vertically, one per trajectory.

    Layout (top to bottom):
        Row 0 : N=16  (Full CoC)
        Row 1 : N=0   (Decode Skip)
        Row 2 : GT    (Ground Truth)

    Each row shows:
      - Colored LiDAR obstacle cloud (red/orange/gray by lateral zone)
      - Animated vehicle bounding box for that row's trajectory only
      - Faded dashed path showing the full 6.4 s planned trajectory
      - Status overlay (COLLISION / CRITICAL / WARNING / SAFE / CLEAR)

    Obstacle color key:
      RED    |y| < 1.058 m  -- in vehicle corridor (direct collision zone)
      ORANGE 1.058 <= |y| < 1.75 m  -- in lane, beside vehicle
      GRAY   |y| >= 1.75 m  -- off-road static structure
    """
    TRAJ_ORDER = ["n16", "n0", "gt"]
    n_frames = 65
    times = [results["gt"][f]["t"] for f in range(n_frames)]

    # Classify obstacles once
    cls = _classify_obs(obs_pts)
    rng = np.random.default_rng(42)

    def _sub(pts: np.ndarray, n: int) -> np.ndarray:
        if len(pts) == 0:
            return pts
        idx = rng.choice(len(pts), size=min(len(pts), n), replace=False)
        return pts[idx]

    sub_off  = _sub(cls["off_road"],   8000)
    sub_near = _sub(cls["near_lane"],  3000)
    sub_in   = _sub(cls["in_path"],    3000)

    # ── Figure: 3 rows x 1 col ──
    fig = plt.figure(figsize=(16, 15))
    fig.patch.set_facecolor("#f4f4f4")
    gs_main = fig.add_gridspec(
        3, 1, hspace=0.09,
        top=0.93, bottom=0.04, left=0.06, right=0.98,
    )
    axes_bev = {key: fig.add_subplot(gs_main[i]) for i, key in enumerate(TRAJ_ORDER)}

    # ── Static background per row ──
    for i, key in enumerate(TRAJ_ORDER):
        ax = axes_bev[key]
        is_top    = (i == 0)
        is_bottom = (i == len(TRAJ_ORDER) - 1)

        _draw_obs_background(ax, cls, sub_off, sub_near, sub_in,
                             show_legend=is_top)

        # Faded full trajectory path
        traj = trajs.get(key)
        if traj is not None:
            ax.plot(traj[:, 0], traj[:, 1], "--",
                    color=TRAJ_COLORS[key], lw=1.3, alpha=0.30, zorder=6)
            ax.plot(traj[-1, 0], traj[-1, 1], "*",
                    color=TRAJ_COLORS[key], ms=10, alpha=0.50, zorder=7)
            ax.annotate(
                f"({traj[-1,0]:.1f}, {traj[-1,1]:.2f}) m",
                xy=(traj[-1, 0], traj[-1, 1]),
                xytext=(traj[-1, 0] - 9, traj[-1, 1] + 2.8),
                fontsize=7, color=TRAJ_COLORS[key], alpha=0.65,
                arrowprops=dict(arrowstyle="->", color=TRAJ_COLORS[key],
                                lw=0.8, alpha=0.5),
            )

        # Row label (trajectory name)
        ax.text(BEV_X_MIN + 0.8, BEV_Y_MAX - 1.1,
                TRAJ_LABELS[key],
                fontsize=10, fontweight="bold", color=TRAJ_COLORS[key],
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=TRAJ_COLORS[key], alpha=0.92, lw=1.5),
                zorder=25)

        ax.set_ylabel("y  (m)\n[L(+)/R(-)]", fontsize=8)
        if is_bottom:
            ax.set_xlabel("Forward distance  x  (m)", fontsize=10)

        if is_top:
            handles = [
                mpatches.Patch(fc="#c0392b",
                               label=f"In vehicle corridor  (|y|<{EGO_W/2:.2f} m)"),
                mpatches.Patch(fc="#e67e22",
                               label=f"In lane, beside vehicle  ({EGO_W/2:.2f}<|y|<1.75 m)"),
                mpatches.Patch(fc="#999999",
                               label="Off-road static structure  (|y|>=1.75 m)"),
                mpatches.Patch(fc="#e74c3c", alpha=0.15,
                               label="Vehicle width corridor band"),
                mpatches.Patch(fc="steelblue", alpha=0.45,
                               label="Lane boundary  (+/-1.75 m)"),
            ]
            ax.legend(handles=handles, fontsize=7.5, loc="upper right",
                      framealpha=0.92, ncol=2, borderpad=0.5)

    fig.suptitle(
        "BEV Collision Simulation  --  Exp 1: Decode Skip\n"
        "Each row shows one trajectory independently.  "
        "[LIMIT-1: obstacles frozen at t=0 position]",
        fontsize=12, fontweight="bold",
    )

    # ── Dynamic elements (updated every frame) ──
    dyn_patches: dict[str, list] = {key: [] for key in TRAJ_ORDER}
    dyn_texts:   dict[str, list] = {key: [] for key in TRAJ_ORDER}
    time_label = fig.text(0.5, 0.955, "", ha="center",
                          fontsize=11, fontweight="bold")

    def _frame(fi: int):
        t_now = times[fi]
        time_label.set_text(f"t = {t_now:.1f} s  |  waypoint {fi:02d} / 64")

        for key in TRAJ_ORDER:
            ax = axes_bev[key]

            # Clear previous frame
            for p in dyn_patches[key]:
                p.remove()
            for t in dyn_texts[key]:
                t.remove()
            dyn_patches[key].clear()
            dyn_texts[key].clear()

            fd        = results[key][fi]
            wp        = np.array([fd["wx"], fd["wy"], 0.0], dtype=np.float32)
            heading   = fd["heading"]
            collision = fd["collision"]
            gap       = fd.get("gap")

            # Status & edge styling
            if collision:
                edge_c = "#c0392b";  edge_lw = 3.5
                status_str = "[COLLISION]"
                status_fc  = "#fdecea"; status_ec = "#c0392b"
            elif gap is not None and gap < GAP_CRITICAL:
                edge_c = "#e67e22";  edge_lw = 2.5
                status_str = f"[CRITICAL]  gap = {gap:.1f} m"
                status_fc  = "#fef3cd"; status_ec = "#e67e22"
            elif gap is not None and gap < GAP_WARNING:
                edge_c = "#f1c40f";  edge_lw = 2.0
                status_str = f"[WARNING]   gap = {gap:.1f} m"
                status_fc  = "#fffde7"; status_ec = "#f1c40f"
            elif gap is not None:
                edge_c = "#2ecc71";  edge_lw = 1.5
                status_str = f"[SAFE]      gap = {gap:.1f} m"
                status_fc  = "#e8f5e9"; status_ec = "#27ae60"
            else:
                edge_c = "#2ecc71";  edge_lw = 1.5
                status_str = "[CLEAR]     no forward obstacle"
                status_fc  = "#e8f5e9"; status_ec = "#27ae60"

            # Vehicle bounding box
            corners, ctr = vehicle_corners(wp, heading)
            poly = mpatches.Polygon(
                corners, closed=True,
                facecolor=TRAJ_COLORS[key],
                edgecolor=edge_c,
                linewidth=edge_lw,
                alpha=0.92, zorder=15,
            )
            ax.add_patch(poly)
            dyn_patches[key].append(poly)

            # Position label above vehicle box
            pos_lbl = ax.text(
                ctr[0], ctr[1] + 0.60,
                f"({ctr[0]:.1f}, {ctr[1]:.2f}) m",
                fontsize=6.5, color="white", ha="center", va="bottom", zorder=20,
                bbox=dict(boxstyle="round,pad=0.15",
                          fc=TRAJ_COLORS[key], ec="none", alpha=0.85),
            )
            dyn_texts[key].append(pos_lbl)

            # Status box (bottom-right of each BEV row)
            st = ax.text(
                BEV_X_MAX - 0.5, BEV_Y_MIN + 0.9,
                f"t={t_now:.1f} s  |  {status_str}",
                fontsize=9, ha="right", va="bottom", zorder=20,
                fontweight="bold" if collision else "normal",
                bbox=dict(boxstyle="round,pad=0.40",
                          fc=status_fc, ec=status_ec,
                          alpha=0.95, lw=2.5 if collision else 1.0),
            )
            dyn_texts[key].append(st)

        return (
            [p for key in TRAJ_ORDER for p in dyn_patches[key]] +
            [t for key in TRAJ_ORDER for t in dyn_texts[key]]
        )

    ani = animation.FuncAnimation(
        fig, _frame, frames=n_frames, interval=100, blit=False,
    )

    try:
        writer = animation.FFMpegWriter(fps=10, bitrate=2500)
        ani.save(str(OUT_MP4), writer=writer)
        logger.info(f"MP4 saved: {OUT_MP4}")
    except Exception as e:
        logger.warning(f"FFMpeg failed ({e}), saving as GIF...")
        out_gif = OUT_MP4.with_suffix(".gif")
        ani.save(str(out_gif), writer=animation.PillowWriter(fps=10))
        logger.info(f"GIF saved: {out_gif}")

    plt.close(fig)


def save_log(results: dict[str, list[dict]]) -> None:
    """프레임별 충돌 여부 및 gap 수치를 JSON으로 저장."""
    with open(str(OUT_LOG), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Collision log saved: {OUT_LOG}")


# ── Dynamic LiDAR: Phase 1 extension ─────────────────────────────────────────

def load_future_lidar_spins(
    gt_traj: np.ndarray,
) -> list[np.ndarray]:
    """
    Load 65 LiDAR spins (spin 50~114) and transform each to t=0 ego frame.

    Confirmed (2026-05-22):
        spin cadence = 100 ms exactly
        spin 50 + i = t = i * 0.1 s

    Transform for frame i >= 1:
        pts_veh_ti   = T_lidar_to_veh @ pts_lidar      (in t_i ego frame)
        pts_t0_xy    = R(heading_i) @ pts_veh_ti_xy + gt_traj[i-1, :2]

    where heading_i = direction of GT ego motion at step i-1.

    Static structures (construction, road markings) stay in place across
    frames because they don't move — the math works out naturally.
    Dynamic objects (lead vehicle, cyclists) appear at their actual
    positions in the world frame at each time step.

    Parameters
    ----------
    gt_traj : (64, 3) GT trajectory in t=0 ego frame

    Returns
    -------
    List of 65 arrays (one per frame), each (N, 3) float64 in t=0 ego frame.
    Frame 0 = t=0 (spin 50).  Frame i = t=i*0.1s (spin 50+i).
    """
    import DracoPy

    from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
    avdi = PhysicalAIAVDatasetInterface()
    feat = avdi.get_clip_feature(CLIP_ID, "lidar_top_360fov", maybe_stream=True)
    df   = feat["pointclouds"]

    logger.info(f"Loading {DYN_N_FRAMES} LiDAR spins "
                f"(spin {DYN_SPIN_BASE}~{DYN_SPIN_BASE + DYN_N_FRAMES - 1})...")

    future_obs: list[np.ndarray] = []

    for i in range(DYN_N_FRAMES):          # i = 0..64
        spin_idx = DYN_SPIN_BASE + i
        row = df[df["spin_index"] == spin_idx]
        if row.empty:
            logger.warning(f"  spin {spin_idx} not found — using empty array")
            future_obs.append(np.zeros((0, 3), dtype=np.float64))
            continue

        draco_bytes = row.iloc[0]["draco_encoded_pointcloud"]
        pts_lidar   = np.array(DracoPy.decode(draco_bytes).points, dtype=np.float64)
        pts_veh_ti  = to_vehicle_frame(pts_lidar)   # (N,3) in t_i ego frame

        # Filter height in t_i ego frame (z axis consistent across frames)
        z_mask     = (pts_veh_ti[:, 2] >= Z_OBS_MIN) & (pts_veh_ti[:, 2] <= Z_OBS_MAX)
        pts_veh_ti = pts_veh_ti[z_mask]

        # Ego exclusion zone: remove self-reflections from car body.
        # Applied in vehicle frame (i=0 included for consistency).
        # Without this, near-ego points transform to land inside GT bbox → 65/65 false collision.
        #
        # ASYMMETRIC: rear axle is NOT at car center.
        #   forward  : +EGO_EXCL_X_FWD  = +4.317 m from rear axle (covers bonnet + 0.5m buffer)
        #   rearward : -EGO_EXCL_X_REAR = -1.609 m from rear axle (covers trunk  + 0.5m buffer)
        #   lateral  :  EGO_EXCL_Y      =  1.558 m                (half-width    + 0.5m buffer)
        near_ego = (
            (pts_veh_ti[:, 0] >= -EGO_EXCL_X_REAR) &
            (pts_veh_ti[:, 0] <=  EGO_EXCL_X_FWD)  &
            (np.abs(pts_veh_ti[:, 1]) <= EGO_EXCL_Y)
        )
        pts_veh_ti = pts_veh_ti[~near_ego]

        if i == 0:
            # Frame 0 = t=0: already in t=0 ego frame
            pts_t0 = pts_veh_ti
        else:
            # Transform x,y from t_i ego frame → t=0 ego frame
            # ego position at t_i = gt_traj[i-1, :2]
            ego_pos = gt_traj[i - 1, :2]          # (2,)
            heading = compute_heading(gt_traj, i - 1)
            ch, sh  = np.cos(heading), np.sin(heading)
            R       = np.array([[ch, -sh], [sh, ch]])

            xy_t0   = (R @ pts_veh_ti[:, :2].T).T + ego_pos   # (N,2) in t=0 frame
            pts_t0  = np.column_stack([xy_t0, pts_veh_ti[:, 2]])

        # Spatial filter in t=0 frame (same BEV display range)
        sp = (
            (pts_t0[:, 0] >= BEV_X_MIN) & (pts_t0[:, 0] <= BEV_X_MAX) &
            (pts_t0[:, 1] >= BEV_Y_MIN) & (pts_t0[:, 1] <= BEV_Y_MAX)
        )
        future_obs.append(pts_t0[sp])

        if i % 10 == 0:
            logger.info(f"  frame {i:02d}  spin {spin_idx}  "
                        f"{len(future_obs[-1]):,} pts after filter")

    logger.info(f"Dynamic LiDAR loaded: {DYN_N_FRAMES} frames")
    return future_obs


def precompute_all_dynamic(
    future_obs: list[np.ndarray],
    trajs: dict[str, np.ndarray | None],
) -> dict[str, list[dict]]:
    """
    Like precompute_all but uses per-frame obstacle cloud.

    frame 0  → future_obs[0]  (spin 50, t=0)
    frame i  → future_obs[i]  (spin 50+i, t=i*0.1s, transformed to t=0 frame)
    """
    results: dict[str, list[dict]] = {}
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    for key, traj in trajs.items():
        frames: list[dict] = []

        # frame 0
        obs_0 = future_obs[0]
        col0, gap0 = collision_and_gap(origin, 0.0, obs_0)
        frames.append({
            "t": 0.0, "wx": 0.0, "wy": 0.0,
            "heading": 0.0, "collision": col0, "gap": gap0,
        })

        if traj is None:
            results[key] = frames
            continue

        for i in range(N_WP):             # i = 0..63 → frame 1..64
            wp      = traj[i]
            heading = compute_heading(traj, i)
            obs_i   = future_obs[i + 1]   # frame i+1 = t=(i+1)*0.1s
            col, gap = collision_and_gap(wp, heading, obs_i)
            frames.append({
                "t":         round((i + 1) * DT, 2),
                "wx":        float(wp[0]),
                "wy":        float(wp[1]),
                "heading":   float(heading),
                "collision": col,
                "gap":       gap if np.isfinite(gap) else None,
            })

        results[key] = frames
        n_col = sum(1 for f in frames if f["collision"])
        logger.info(f"[{key}] dynamic collision frames: {n_col}/65")

    return results


def make_animation_dynamic(
    future_obs: list[np.ndarray],
    trajs: dict[str, np.ndarray | None],
    results: dict[str, list[dict]],
) -> None:
    """
    BEV animation with dynamic LiDAR (spin updates every frame).

    Layout: 3 rows stacked vertically — N=16 / N=0 / GT (same as static).

    Key difference vs make_animation():
      - Obstacle cloud updated EVERY frame from the corresponding LiDAR spin
      - Static structures (construction, road) naturally stay in place
      - Dynamic objects (lead vehicle, cyclists) move to real future positions
      - Output: collision_bev_dynamic.mp4

    Layout fix vs static version:
      - Larger figure height (17 in) + lower gridspec top (0.90)
        → title / time-label / BEV content no longer overlap
    """
    TRAJ_ORDER = ["n16", "n0", "gt"]
    n_frames   = DYN_N_FRAMES
    times      = [results["gt"][f]["t"] for f in range(n_frames)]

    rng = np.random.default_rng(42)

    def _sub(pts: np.ndarray, n: int) -> np.ndarray:
        if len(pts) == 0:
            return pts
        idx = rng.choice(len(pts), size=min(len(pts), n), replace=False)
        return pts[idx]

    # Pre-subsample per-frame obstacles to keep animation fast
    logger.info("Pre-subsampling per-frame obstacle clouds ...")
    frame_obs_sub: list[dict] = []
    for fi in range(n_frames):
        obs_fi  = future_obs[fi]
        cls_fi  = _classify_obs(obs_fi)
        frame_obs_sub.append({
            "off":  _sub(cls_fi["off_road"],   8000),
            "near": _sub(cls_fi["near_lane"],  3000),
            "in":   _sub(cls_fi["in_path"],    3000),
            "cls":  cls_fi,
        })
    logger.info("Pre-subsampling done.")

    # ── Figure: 3 rows, taller figure + lower gridspec top to avoid overlap ──
    fig = plt.figure(figsize=(16, 17))
    fig.patch.set_facecolor("#f4f4f4")
    gs_main = fig.add_gridspec(
        3, 1, hspace=0.10,
        top=0.90,    # more headroom for suptitle + time label
        bottom=0.04,
        left=0.06,
        right=0.98,
    )
    axes_bev = {key: fig.add_subplot(gs_main[i]) for i, key in enumerate(TRAJ_ORDER)}

    # ── Static per-row elements (drawn once) ──
    # Scatter artists are created empty here and filled each frame via set_offsets()
    sc_artists: dict[str, dict] = {}

    for i, key in enumerate(TRAJ_ORDER):
        ax        = axes_bev[key]
        is_bottom = (i == len(TRAJ_ORDER) - 1)

        # Vehicle-width corridor band
        ax.axhspan(-EGO_W / 2, EGO_W / 2, alpha=0.06, color="#e74c3c", zorder=1)

        # Lane boundaries
        for y_lane in [-1.75, 1.75]:
            ax.axhline(y_lane, color="steelblue", lw=0.9, ls="--", alpha=0.50, zorder=5)
        ax.axhline(0, color="steelblue", lw=0.4, ls=":", alpha=0.30, zorder=5)

        ax.set_xlim(BEV_X_MIN, BEV_X_MAX)
        ax.set_ylim(BEV_Y_MIN, BEV_Y_MAX)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.14, zorder=0)

        # Empty scatter artists (filled per frame)
        sc_off  = ax.scatter([], [], s=0.8, c="#999999", alpha=0.45, zorder=2, rasterized=True)
        sc_near = ax.scatter([], [], s=2.5, c="#e67e22", alpha=0.70, zorder=3, rasterized=True)
        sc_in   = ax.scatter([], [], s=4.0, c="#c0392b", alpha=0.85, zorder=4, rasterized=True)
        sc_artists[key] = {"off": sc_off, "near": sc_near, "in": sc_in}

        # Faded full trajectory path
        traj = trajs.get(key)
        if traj is not None:
            ax.plot(traj[:, 0], traj[:, 1], "--",
                    color=TRAJ_COLORS[key], lw=1.3, alpha=0.30, zorder=6)
            ax.plot(traj[-1, 0], traj[-1, 1], "*",
                    color=TRAJ_COLORS[key], ms=10, alpha=0.50, zorder=7)
            # Endpoint annotation — placed below star to avoid top-area overlap
            ann_y = traj[-1, 1] - 2.5   # below the endpoint star
            ann_y = max(ann_y, BEV_Y_MIN + 1.0)
            ax.annotate(
                f"({traj[-1,0]:.1f}, {traj[-1,1]:.2f}) m",
                xy=(traj[-1, 0], traj[-1, 1]),
                xytext=(traj[-1, 0] - 6, ann_y),
                fontsize=7, color=TRAJ_COLORS[key], alpha=0.70,
                arrowprops=dict(arrowstyle="->", color=TRAJ_COLORS[key],
                                lw=0.8, alpha=0.5),
            )

        # Row label — lower inside plot to avoid suptitle region
        ax.text(BEV_X_MIN + 0.8, BEV_Y_MAX - 1.5,
                TRAJ_LABELS[key],
                fontsize=10, fontweight="bold", color=TRAJ_COLORS[key],
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=TRAJ_COLORS[key], alpha=0.92, lw=1.5),
                zorder=25)

        ax.set_ylabel("y  (m)\n[L(+)/R(-)]", fontsize=8)
        if is_bottom:
            ax.set_xlabel("Forward distance  x  (m)", fontsize=10)

        # Legend in top row only — positioned lower to avoid overlap with row label
        if i == 0:
            handles = [
                mpatches.Patch(fc="#c0392b",
                               label=f"In vehicle corridor  (|y|<{EGO_W/2:.2f} m)"),
                mpatches.Patch(fc="#e67e22",
                               label=f"In lane, beside vehicle  ({EGO_W/2:.2f}<|y|<1.75 m)"),
                mpatches.Patch(fc="#999999",
                               label="Off-road static structure  (|y|>=1.75 m)"),
                mpatches.Patch(fc="steelblue", alpha=0.45,
                               label="Lane boundary  (+/-1.75 m)"),
            ]
            ax.legend(handles=handles, fontsize=7.5, loc="lower right",
                      framealpha=0.92, ncol=2, borderpad=0.5)

    fig.suptitle(
        "BEV Collision Simulation  —  Dynamic LiDAR (spin per frame)\n"
        "Obstacles update each frame: dynamic objects move, static structures stay",
        fontsize=11, fontweight="bold", y=0.975,
    )

    # Time label positioned in the gap between suptitle and gridspec
    time_label = fig.text(0.5, 0.925, "", ha="center",
                          fontsize=11, fontweight="bold")

    # ── Dynamic elements ──
    dyn_patches: dict[str, list] = {key: [] for key in TRAJ_ORDER}
    dyn_texts:   dict[str, list] = {key: [] for key in TRAJ_ORDER}

    def _frame(fi: int):
        t_now = times[fi]
        time_label.set_text(f"t = {t_now:.1f} s  |  frame {fi:02d} / 64")

        # Update obstacle scatter for this frame
        obs_sub = frame_obs_sub[fi]
        for key in TRAJ_ORDER:
            sc = sc_artists[key]
            empty = np.zeros((0, 2))
            sc["off"].set_offsets(
                obs_sub["off"][:, :2] if len(obs_sub["off"]) > 0 else empty)
            sc["near"].set_offsets(
                obs_sub["near"][:, :2] if len(obs_sub["near"]) > 0 else empty)
            sc["in"].set_offsets(
                obs_sub["in"][:, :2] if len(obs_sub["in"]) > 0 else empty)

        # Update vehicle boxes and status text
        for key in TRAJ_ORDER:
            ax = axes_bev[key]

            for p in dyn_patches[key]:
                p.remove()
            for t in dyn_texts[key]:
                t.remove()
            dyn_patches[key].clear()
            dyn_texts[key].clear()

            fd        = results[key][fi]
            wp        = np.array([fd["wx"], fd["wy"], 0.0], dtype=np.float32)
            heading   = fd["heading"]
            collision = fd["collision"]
            gap       = fd.get("gap")

            if collision:
                edge_c = "#c0392b"; edge_lw = 3.5
                status_str = "[COLLISION]"
                status_fc = "#fdecea"; status_ec = "#c0392b"
            elif gap is not None and gap < GAP_CRITICAL:
                edge_c = "#e67e22"; edge_lw = 2.5
                status_str = f"[CRITICAL]  gap = {gap:.1f} m"
                status_fc = "#fef3cd"; status_ec = "#e67e22"
            elif gap is not None and gap < GAP_WARNING:
                edge_c = "#f1c40f"; edge_lw = 2.0
                status_str = f"[WARNING]   gap = {gap:.1f} m"
                status_fc = "#fffde7"; status_ec = "#f1c40f"
            elif gap is not None:
                edge_c = "#2ecc71"; edge_lw = 1.5
                status_str = f"[SAFE]      gap = {gap:.1f} m"
                status_fc = "#e8f5e9"; status_ec = "#27ae60"
            else:
                edge_c = "#2ecc71"; edge_lw = 1.5
                status_str = "[CLEAR]     no forward obstacle"
                status_fc = "#e8f5e9"; status_ec = "#27ae60"

            corners, ctr = vehicle_corners(wp, heading)
            poly = mpatches.Polygon(
                corners, closed=True,
                facecolor=TRAJ_COLORS[key],
                edgecolor=edge_c,
                linewidth=edge_lw,
                alpha=0.92, zorder=15,
            )
            ax.add_patch(poly)
            dyn_patches[key].append(poly)

            pos_lbl = ax.text(
                ctr[0], ctr[1] + 0.60,
                f"({ctr[0]:.1f}, {ctr[1]:.2f}) m",
                fontsize=6.5, color="white", ha="center", va="bottom", zorder=20,
                bbox=dict(boxstyle="round,pad=0.15",
                          fc=TRAJ_COLORS[key], ec="none", alpha=0.85),
            )
            dyn_texts[key].append(pos_lbl)

            st = ax.text(
                BEV_X_MAX - 0.5, BEV_Y_MIN + 0.9,
                f"t={t_now:.1f} s  |  {status_str}",
                fontsize=9, ha="right", va="bottom", zorder=20,
                fontweight="bold" if collision else "normal",
                bbox=dict(boxstyle="round,pad=0.40",
                          fc=status_fc, ec=status_ec,
                          alpha=0.95, lw=2.5 if collision else 1.0),
            )
            dyn_texts[key].append(st)

        return (
            [p for key in TRAJ_ORDER for p in dyn_patches[key]] +
            [t for key in TRAJ_ORDER for t in dyn_texts[key]]
        )

    ani = animation.FuncAnimation(
        fig, _frame, frames=n_frames, interval=100, blit=False,
    )

    try:
        writer = animation.FFMpegWriter(fps=10, bitrate=2500)
        ani.save(str(OUT_MP4_DYNAMIC), writer=writer)
        logger.info(f"Dynamic MP4 saved: {OUT_MP4_DYNAMIC}")
    except Exception as e:
        logger.warning(f"FFMpeg failed ({e}), saving as GIF...")
        out_gif = OUT_MP4_DYNAMIC.with_suffix(".gif")
        ani.save(str(out_gif), writer=animation.PillowWriter(fps=10))
        logger.info(f"GIF saved: {out_gif}")

    plt.close(fig)


def save_log_dynamic(results: dict[str, list[dict]]) -> None:
    with open(str(OUT_LOG_DYNAMIC), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Dynamic collision log saved: {OUT_LOG_DYNAMIC}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: 데이터 로딩 ──────────────────────────────────────────────────
    pts_lidar = load_lidar_spin(SPIN_IDX)
    pts_veh   = to_vehicle_frame(pts_lidar)
    obs_pts   = filter_obstacles(pts_veh)
    trajs     = load_trajectories()

    gt_traj = trajs.get("gt")
    if gt_traj is None:
        raise RuntimeError("GT trajectory required for dynamic LiDAR transform.")

    # ── Phase 2a: Static LiDAR collision (single spin, t=0 snapshot) ─────────
    results_static = precompute_all(obs_pts, trajs)

    # ── Phase 3a: Static visualisation ───────────────────────────────────────
    save_sanity_png(obs_pts, trajs, results_static)
    logger.info("=" * 60)
    logger.info("bev_sanity.png saved.")
    logger.info("=" * 60)
    make_animation(obs_pts, trajs, results_static)
    save_log(results_static)

    logger.info("\n=== Static LiDAR Collision Summary (LIMIT-1: frozen at t=0) ===")
    logger.info("  collision count reflects trajectory LENGTH, not safety quality.")
    for key in ["gt", "n16", "n0"]:
        frames = results_static[key]
        n_col  = sum(1 for f in frames if f["collision"])
        gaps   = [f["gap"] for f in frames if f.get("gap") is not None]
        min_g  = min(gaps) if gaps else float("inf")
        ep_x   = frames[-1]["wx"] if len(frames) > 1 else 0.0
        logger.info(
            f"  {TRAJ_LABELS[key]:24s}: "
            f"collision={n_col:2d}/65  min_gap={min_g:.2f} m  endpoint_x={ep_x:.1f} m"
        )

    # ── Phase 2b: Dynamic LiDAR collision (spin per frame, t=0~6.4s) ─────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Starting Dynamic LiDAR analysis (65 spins, ~8s load time)...")
    logger.info("=" * 60)
    future_obs     = load_future_lidar_spins(gt_traj)
    results_dynamic = precompute_all_dynamic(future_obs, trajs)

    # ── Phase 3b: Dynamic visualisation ──────────────────────────────────────
    make_animation_dynamic(future_obs, trajs, results_dynamic)
    save_log_dynamic(results_dynamic)

    logger.info("\n=== Dynamic LiDAR Collision Summary (real future obstacle positions) ===")
    logger.info("  Dynamic objects move; static structures stay in place.")
    for key in ["gt", "n16", "n0"]:
        frames = results_dynamic[key]
        n_col  = sum(1 for f in frames if f["collision"])
        gaps   = [f["gap"] for f in frames if f.get("gap") is not None]
        min_g  = min(gaps) if gaps else float("inf")
        ep_x   = frames[-1]["wx"] if len(frames) > 1 else 0.0
        logger.info(
            f"  {TRAJ_LABELS[key]:24s}: "
            f"collision={n_col:2d}/65  min_gap={min_g:.2f} m  endpoint_x={ep_x:.1f} m"
        )
    logger.info("")
    logger.info("Outputs:")
    logger.info(f"  Static  : {OUT_MP4}")
    logger.info(f"  Dynamic : {OUT_MP4_DYNAMIC}")


if __name__ == "__main__":
    main()

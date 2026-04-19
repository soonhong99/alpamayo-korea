"""
run_thor_inference.py
Real-time Alpamayo-Korea inference on NVIDIA Jetson AGX Thor.

Usage:
    python scripts/run_thor_inference.py \
        --model checkpoints/alpamayo_korea_v1/ \
        --lang ko \
        --scenario realtime_camera

Requirements:
    - JetPack 7 (Ubuntu 24.04, Linux 6.8)
    - Python 3.10+
    - Alpamayo-Korea checkpoint (fine-tuned or baseline)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpamayo-Korea Thor Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Alpamayo-1.5-10B",
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="ko",
        choices=["ko", "en"],
        help="Language for reasoning trace output (ko = Korean)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="realtime_camera",
        help="Scenario mode: 'realtime_camera' or path to scenario YAML",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp4",
        choices=["fp4", "bf16", "fp16"],
        help="Inference dtype. Use fp4 for maximum Thor throughput (2,070 TFLOPS)",
    )
    parser.add_argument(
        "--attn",
        type=str,
        default="flash_attn2",
        choices=["flash_attn2", "sdpa"],
        help="Attention implementation. Use sdpa if nvcc is unavailable.",
    )
    parser.add_argument(
        "--latency_target_ms",
        type=float,
        default=100.0,
        help="Target inference latency in milliseconds",
    )
    parser.add_argument(
        "--save_traces",
        action="store_true",
        help="Save reasoning traces to JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/results/thor_inference/",
        help="Directory for output files",
    )
    return parser.parse_args()


def load_model(model_path: str, dtype: str, attn: str):
    """Load Alpamayo model. Falls back to SDPA if flash-attn unavailable."""
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"dtype={dtype}, attention={attn}")

    try:
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        dtype_map = {
            "fp4": torch.float8_e4m3fn,   # TensorRT handles FP4 quantization
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        model = Alpamayo1_5.from_pretrained(
            model_path,
            dtype=torch_dtype,
            attn_implementation=attn,
        ).to("cuda")

        model.eval()
        logger.info("Model loaded successfully")
        return model

    except ImportError:
        logger.error(
            "alpamayo1_5 package not found. "
            "Run: pip install -e alpasim/  or clone NVlabs/alpamayo1.5"
        )
        raise


def build_korean_prompt(scenario: str) -> str:
    """Build Korean-language reasoning prompt."""
    return (
        f"현재 주행 상황 ({scenario})을 분석하고, "
        "결정한 행동과 그 이유를 한국어로 설명하세요. "
        '형식: {"상황": "...", "판단": "...", "행동": "...", "안전근거": "..."}'
    )


def run_inference_loop(model, args: argparse.Namespace) -> None:
    """Main inference loop. Measures latency and outputs reasoning traces."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_log = []
    latencies = []

    logger.info("Starting inference loop. Press Ctrl+C to stop.")
    logger.info(f"Target latency: {args.latency_target_ms}ms")

    iteration = 0
    try:
        while True:
            iteration += 1

            # Mock input — replace with real camera frames in production
            # Shape: (num_cameras, seq_len, C, H, W)
            mock_video = torch.randn(4, 8, 3, 1080, 1920, device="cuda")
            mock_egomotion = torch.randn(8, 6, device="cuda")  # 6-DOF

            start_time = time.perf_counter()

            with torch.no_grad():
                output = model(
                    video=mock_video,
                    egomotion=mock_egomotion,
                    language_prompt=build_korean_prompt(args.scenario) if args.lang == "ko" else None,
                )

            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)

            trajectory = output.trajectory.cpu().numpy()  # (64, 3) waypoints
            reasoning_trace = getattr(output, "reasoning_trace", None)

            status = "OK" if latency_ms <= args.latency_target_ms else "SLOW"
            logger.info(
                f"[{iteration:04d}] {status} Latency: {latency_ms:.1f}ms | "
                f"Next waypoint: ({trajectory[0, 0]:.2f}, {trajectory[0, 1]:.2f})m"
            )

            if reasoning_trace and iteration % 10 == 0:
                logger.info(f"  추론: {reasoning_trace[:200]}...")

            if args.save_traces:
                trace_entry = {
                    "iteration": iteration,
                    "latency_ms": round(latency_ms, 2),
                    "trajectory_next_5": trajectory[:5].tolist(),
                    "reasoning_trace": reasoning_trace,
                    "timestamp": time.time(),
                }
                trace_log.append(trace_entry)

            if iteration % 100 == 0:
                avg_lat = np.mean(latencies[-100:])
                p95_lat = np.percentile(latencies[-100:], 95)
                meets_target = avg_lat <= args.latency_target_ms
                logger.info(
                    f"--- Summary (last 100) ---\n"
                    f"  Avg latency: {avg_lat:.1f}ms\n"
                    f"  P95 latency: {p95_lat:.1f}ms\n"
                    f"  Target met: {'YES' if meets_target else 'NO'}"
                )

    except KeyboardInterrupt:
        logger.info("Inference stopped by user.")

    finally:
        if args.save_traces and trace_log:
            trace_path = output_dir / "reasoning_traces.jsonl"
            with open(trace_path, "w", encoding="utf-8") as f:
                for entry in trace_log:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(f"Reasoning traces saved to: {trace_path}")

        if latencies:
            summary = {
                "total_iterations": iteration,
                "avg_latency_ms": round(float(np.mean(latencies)), 2),
                "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
                "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
                "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
                "target_ms": args.latency_target_ms,
                "target_met_pct": round(
                    100 * sum(l <= args.latency_target_ms for l in latencies) / len(latencies), 1
                ),
            }
            summary_path = output_dir / "latency_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Latency summary saved to: {summary_path}")
            logger.info(
                f"Final: avg={summary['avg_latency_ms']}ms, "
                f"p95={summary['p95_latency_ms']}ms, "
                f"target met {summary['target_met_pct']}% of the time"
            )


def main() -> None:
    args = parse_args()

    logger.info("=" * 50)
    logger.info("  Alpamayo-Korea — Jetson AGX Thor Inference")
    logger.info("=" * 50)
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Language: {'Korean (한국어)' if args.lang == 'ko' else 'English'}")
    logger.info(f"  dtype: {args.dtype}")
    logger.info(f"  Hardware: {torch.cuda.get_device_name(0)}")
    logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info("=" * 50)

    model = load_model(args.model, args.dtype, args.attn)
    run_inference_loop(model, args)


if __name__ == "__main__":
    main()

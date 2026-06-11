# Jetson AGX Thor — Hardware Specification
**보드**: ice401@100.95.177.101 | **측정일**: 2026-05-10

---

## 공식 스펙 (NVIDIA Datasheet DS-11945-001)

| 항목 | 값 |
|------|----|
| **GPU 아키텍처** | Blackwell (SM 11.0) |
| **GPU CUDA 코어** | 2,560 (20 SM × 128) |
| **GPU Tensor 코어** | 96개 (5세대) |
| **GPU FP4 성능** | 2,070 TFLOPS |
| **CPU 아키텍처** | Arm Neoverse V3AE |
| **CPU 코어 수** | 14코어, 최대 2.6 GHz |
| **CPU L1 캐시** | 64 KB I + 64 KB D (per core) |
| **CPU L2 캐시** | 1 MB per core (= 14 MB total) |
| **CPU L3 캐시** | 16 MB shared |
| **메모리 용량** | 128 GB LPDDR5X |
| **메모리 대역폭** | 273 GB/s |
| **메모리 버스** | 256-bit |
| **JetPack** | 7 (Ubuntu 24.04, CUDA 13.0) |
| **소비전력** | 75–120 W (max 130 W) |

---

## 실측값 (CUDA API / psutil)

| 항목 | 실측값 | 비고 |
|------|--------|------|
| GPU 이름 | NVIDIA Thor | `prop.name` |
| Unified Memory | 131.9 GB | `prop.total_memory` |
| GPU L2 Cache | **33.6 MB** | `prop.L2_cache_size` — 공식 미공개, 실측 |
| GPU SM 수 | 20 | `prop.multi_processor_count` |
| GPU Compute Cap | 11.0 | `prop.major.minor` |
| GPU Shared+L1/SM | 228 KB | `prop.shared_memory_per_multiprocessor` |
| CPU 코어 수 | 14 | `psutil.cpu_count()` |
| CPU L3 캐시 | /sys 미탐지 | JetPack 7에서 `/sys` 미노출, 공식 16 MB 사용 |
| 시스템 RAM | 131.9 GB | `psutil.virtual_memory()` |

> **Unified Memory**: CPU와 GPU가 동일한 LPDDR5X를 물리적으로 공유. GPU VRAM이 별도로 없음.

---

## 측정 방법

```bash
# 스펙만 빠르게 확인 (모델 로드 불필요, 수 초)
python scripts/profiling/260510_profile_memory_utilization.py --spec-only

# 전체 프로파일링 (모델 로드 포함, ~40분)
python scripts/profiling/260510_profile_memory_utilization.py --warmup 2 --runs 4
```

GPU 메모리는 `torch.cuda.memory_allocated()`로 측정합니다.  
`pynvml.nvmlDeviceGetMemoryInfo()`는 Jetson (Unified Memory SoC)에서 `NVMLError_NotSupported`를 반환합니다.

---

## Alpamayo 1.5 모델 분석

| 항목 | 값 |
|------|----|
| 실제 파라미터 수 | **11.08 B** (HuggingFace 명칭 "10B"는 마케팅) |
| 모델 크기 (bf16) | **22.16 GB** (= 11.08B × 2 bytes) |
| GPU L2 대비 | **660배** 초과 → 매 Decode step DRAM 직접 접근 |
| CPU L3 대비 | **1,385배** 초과 |
| Decode 이론 하한 | **81.2 ms/step** (= 22.16 GB ÷ 273 GB/s) |

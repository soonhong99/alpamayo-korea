# GPU 메모리 프로파일링 실험 런북
**스크립트**: `scripts/profiling/260510_profile_memory_utilization.py`

---

## 파일 전송 (WSL)

```bash
scp /mnt/c/Users/nanay/Desktop/Alphamayo/scripts/profiling/260510_profile_memory_utilization.py \
    ice401@100.95.177.101:~/alpamayo1.5/scripts/profiling/
```

## Thor에서 실행

```bash
ssh ice401@100.95.177.101
source ~/alpamayo1.5/a1_5_venv/bin/activate
cd ~/alpamayo1.5

# 스펙 확인 (빠름)
python scripts/profiling/260510_profile_memory_utilization.py --spec-only

# 전체 프로파일링
python scripts/profiling/260510_profile_memory_utilization.py --warmup 2 --runs 4
```

## 결과 가져오기 (WSL)

```bash
scp -r ice401@100.95.177.101:~/alpamayo1.5/profiling_results/260510_memory_utilization \
    /mnt/c/Users/nanay/Desktop/Alphamayo/profiling_results/
```

## 출력 파일

```
profiling_results/260510_memory_utilization/
  ├── hardware_spec.json
  ├── memory_timeline.json
  ├── summary.json
  └── figures/
       ├── fig_memory_timeline.png
       └── fig_llc_analysis.png
```

## 측정 항목

| 항목 | API | 설명 |
|------|-----|------|
| GPU 메모리 사용량 | `torch.cuda.memory_allocated()` | 모델+KV+활성화 합계 |
| GPU SM 활용률 | `tegrastats GR3D_FREQ` | GPU 3D 엔진 활성 비율 |
| 시스템 RAM | `psutil.virtual_memory()` | OS 전체 RAM |

> **참고**: `pynvml.nvmlDeviceGetMemoryInfo()`는 Jetson에서 `NVMLError_NotSupported`.  
> `GR3D_FREQ`는 compute/BW 구분 불가 — 정밀 분석은 `nsys` 사용.

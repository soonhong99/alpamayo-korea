# 프로파일링 최종 수치 — 논문 기재용
**측정일**: 2026-05-10 | **보드**: Jetson AGX Thor | **모델**: Alpamayo 1.5

---

## 1. 확정 수치 (실측, 직접 인용 가능)

| 항목 | 값 | 측정 방법 |
|------|----|-----------|
| 모델 파라미터 수 | **11.08 B** | `sum(p.numel())` |
| 모델 크기 (bf16) | **22.16 GB** | `numel × 2 bytes` |
| 추론 중 GPU 메모리 피크 | **23.20 GB** | `memory_stats['active_bytes.all.peak']` |
| 모델 대비 추가 메모리 (KV+활성화) | **0.96 GB** | peak − baseline |
| 추론 완료 후 잔류 메모리 | 9 MB | after − before |
| 총 가용 메모리 | 131.9 GB | Unified LPDDR5X |
| **메모리 활용률** | **17.6%** | peak / total |
| 메모리 여유분 | 109 GB | total − peak |
| 1회 추론 지연 (4-run 평균) | **4934 ± 144 ms** | CUDA Events |
| Decode 생성 토큰 수 | 17 tokens | generate() 훅 |
| Decode 이론 최소 시간/step | 81.2 ms/step | 22.16 GB ÷ 273 GB/s |

---

## 2. Phase별 메모리 (tegrastats 4-run 평균)

| Phase | 평균 메모리 | Peak 메모리 | 모델 대비 |
|-------|-------------|-------------|-----------|
| Vision | 22.13 GB | 22.18 GB | +28 MB |
| Prefill | 22.29 GB | 22.50 GB | +348 MB |
| Decode | 22.25 GB | 22.26 GB | +105 MB |
| Flow | 22.49 GB | 22.55 GB | +400 MB |

> **주의**: tegrastats는 100ms 샘플링이므로 단기 spike 누락 가능.
> 정밀 peak은 `memory_stats` 기반 23.20 GB 참고.

---

## 3. Decode BW 추정 (phase 비율 가정)

현재 측정은 vision+prefill+decode+flow 전체를 하나로 측정해 BW가 희석됨.
Phase 비율 가정에 따른 decode-only BW 추정:

| Decode 비율 가정 | Decode 시간 | 추정 BW | 이론 최대 대비 | 판정 |
|-----------------|------------|---------|--------------|------|
| 30% | 1.7s | 219 GB/s | 80% | **BW-bound** |
| 40% | 2.3s | 164 GB/s | 60% | compute-bound 혼재 |
| 50% | 2.9s | 131 GB/s | 48% | compute-bound 혼재 |

> **결론**: nsys kernel 분석으로 decode 실제 비율 확인 후 BW 확정 예정.
> 현재 증거로는 decode가 전체 시간의 30~50%라면 **BW-bound 확실**.

---

## 4. 논문 기재 시 주의사항

- `mem_kv_activation_GB`: KV cache + 활성화 + 임시 버퍼 합산. KV 단독 수치 아님.
- `inference_latency`: single-sample, bf16, torch.autocast 조건.
- BW 수치는 phase 비율 가정 포함 → 논문에서 'estimated' 명시.
- GR3D_FREQ (92.7%)는 SM compute 효율 ≠ compute utilization. 별도 표기.

---

## 5. 다음 실험 (미확정 수치 확정용)

| 수치 | 현재 상태 | 확정 방법 |
|------|-----------|-----------|
| Phase별 GPU 시간 | 미측정 | nsys kernel 패턴 분석 |
| Decode-only BW | 추정 | nsys + phase 분리 |
| KV cache 단독 크기 | 미분리 | memory_history_dump.pickle 분석 |
| SM compute 효율 | GR3D_FREQ만 | ncu (--metrics sm__throughput) |
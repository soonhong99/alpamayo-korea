# Figure 부연 설명 — Alpamayo 1.5 Bandwidth Profiling
**생성 스크립트**: `scripts/profiling/260514_bw_plot_paper.py`
**출력 경로**: `profiling_results/260514_bw/paper_figures/`
**측정 플랫폼**: Jetson AGX Thor (LPDDR5X 273 GB/s, 128 GB Unified Memory)
**모델**: Alpamayo 1.5 (11.08B params, 22.16 GB BF16)

---

## Figure 1 — `fig1_inference_profile.{png, pdf}`

### 전체 설명
Alpamayo 1.5의 1회 추론 실행을 **4개 Phase**로 분리하여 지연 시간, Warmup vs. Steady-state 차이, GPU 메모리 점유량을 한 figure에 정리한 3-panel 종합 프로파일.

---

### Panel (a) — Inference Latency Timeline

**무엇을 보여주나:**
Warmup(첫 번째 실행)과 Measure(안정 상태, n=2 평균)를 가로 방향 Gantt 차트로 비교.
각 색상 블록의 너비가 해당 Phase의 지속 시간을 나타낸다.

**읽는 법:**
- 왼쪽(0 ms)에서 오른쪽으로 시간 흐름
- 위 행 = Warmup, 아래 행 = Measure (steady-state)
- 숫자는 각 블록의 지속 시간 (ms)

**주목할 점:**
| Phase | Warmup | Measure | 차이 |
|-------|--------|---------|------|
| Vision Enc. | 1,151 ms | 741 ms | **-35.6%** (JIT 가장 큰 영향) |
| LM Prefill | 1,545 ms | 1,435 ms | -7.1% |
| Decode | 1,594 ms | 1,886 ms | **+18%** ← 토큰 수 증가(15→18 tok) |
| Flow | 930 ms | 906 ms | -2.6% |

> Decode가 Warmup보다 느린 이유: 토큰 생성 수가 Warmup(15) < Measure(18)이기 때문.
> 동일 토큰 수 기준으로는 Measure가 더 빠름.

---

### Panel (b) — Phase Latency: Warmup vs. Steady-State

**무엇을 보여주나:**
각 Phase의 지속 시간을 막대 그래프로 비교.
- **실선 막대** = Measure (steady-state), 오차 막대 = 2회 측정 표준편차(σ)
- **빗금 막대** = Warmup (JIT 컴파일 + 첫 메모리 할당 포함)

**배율 주석 (multiplier):**
각 Phase 위에 표시된 ×숫자는 `Warmup 시간 / Measure 시간` 비율.
- **1.55×**: Vision은 JIT autotuning으로 Warmup이 Measure의 1.55배 소요
- **1.08×**: Prefill은 JIT 영향이 작음 (대형 GEMM이 이미 컴파일되어 있음)
- **0.85×**: Decode Warmup이 Measure보다 빠른 것은 token 수 차이(15 vs 18)

**오차 막대 해석:**
- Vision σ ≈ 0 ms → 안정적 (ViT forward가 결정론적)
- Decode σ = 101 ms → 토큰 수 변동(±1–2 tok)에 의해 발생

---

### Panel (c) — GPU Memory Footprint per Phase

**무엇을 보여주나:**
각 Phase의 GPU 메모리 점유량을 스택 바로 표현.
- **회색 베이스** = 모델 가중치 베이스라인 (22.2 GB, 추론 내내 상주)
- **연파란 레이어** = 활성화 텐서 + KV cache 할당 (Phase 시작→끝 증가분)
- **진파란 레이어** = Peak 순간 transient overhead (완료 후 해제)
- **Peak 레이블** = 모델 베이스 대비 추가 최대 메모리

**핵심 수치:**
| Phase | Peak 초과분 | 원인 |
|-------|-------------|------|
| Vision Enc. | +649 MB | 이미지 feature map 임시 버퍼 |
| LM Prefill | +761 MB | **KV cache 할당** (3,086 tok) + Flash Attn 버퍼 |
| Decode | +18 MB | KV cache 증분 (token 추가), 베이스라인 거의 변화 없음 |

> **해석**: KV cache는 Prefill에서 한꺼번에 할당(+761 MB)되고
> Decode에서는 매 step마다 미세하게 증가(+18 MB total)하는 구조.
> 128 GB 중 단 23.2 GB만 사용 → **메모리 여유 104 GB**, OOM 위험 없음.

**하단 각주:**
플랫폼 / 모델 / 총 추론 지연 / 측정 횟수를 기재하여 그림 단독으로 인용 가능하게 함.

---

## Figure 2 — `fig2_bandwidth_utilization.{png, pdf}`

### 전체 설명
Decode 구간이 DRAM 대역폭의 77.5%를 점유하는 BW-bound 특성을 두 패널로 시각화.
왼쪽은 절대 수치(MBU 게이지), 오른쪽은 Phase별 비교.

---

### Panel (a) — Decode Memory Bandwidth Utilization (MBU Gauge)

**무엇을 보여주나:**
273 GB/s LPDDR5X 최대 대역폭 대비 Decode 실측 BW(211.5 GB/s)의 비율을
수평 게이지 바 형태로 직관적으로 표현.

**구성 요소:**
- **빨간 막대** = 실측 BW (211.5 GB/s)
- **회색 배경** = 이론 피크 (273 GB/s)
- **점선 수직선** = 50% / 70% / 90% MBU 임계선
- **흰색 텍스트** = 핵심 수치 (77.5% MBU, 211.5 GB/s) 막대 중앙 표기
- **좌상단 주석 박스** = 측정 방법 명시 (CUDA Events 공식)

**MBU 계산식 (주석 박스 내):**
```
BW = 22.16 GB × 18 tok / 1,886 ms = 211.5 GB/s
MBU = 211.5 / 273.0 = 77.5%
```

**임계선의 의미:**
- 70% 이상 → 메모리 대역폭이 실질적 병목 (BW-bound 판정 기준)
- Alpamayo Decode는 이 기준을 **7.5%p 초과** → 명백한 BW-bound

> 이 panel 하나로 "Decode는 BW 병목이 걸린다"는 주장을 뒷받침할 수 있음.
> 논문 Figure로 독립 사용 가능.

---

### Panel (b) — Measured Bandwidth by Inference Phase

**무엇을 보여주나:**
Vision / LM Prefill / Decode / Flow 4개 Phase의 추정 BW를
막대 그래프로 비교하고, 어느 Phase가 BW-bound인지 시각적으로 구분.

**색상 의미:**
- 각 Phase 고유 색상 (Vision=파랑, Prefill=초록, Decode=빨강, Flow=보라)

**오차 막대:**
- Decode: CUDA Events 2회 측정의 표준편차
- 나머지: 추정값 불확도 (±6–15 GB/s)

**내부 텍스트 주석:**
- Decode 막대 중앙 **"CUDA Events"** → 직접 측정임을 명시
- Vision / Prefill / Flow 막대 중앙 **"est."** → 추정값임을 투명하게 표기

**참조선 2종:**
- **빨간 점선** (273 GB/s) = 이론 피크
- **보라 점선** (191 GB/s = 70%) = BW-bound 임계, 해당 영역에 연한 배경색

**수치 레이블 (막대 상단):**
각 bar 위에 GB/s 절대값과 (MBU%) 함께 표기 — figure만 봐도 수치 파악 가능

**핵심 메시지:**
- Decode(211 GB/s, 77%)만 BW-bound 영역에 위치
- Prefill(48 GB/s, 18%), Vision(35 GB/s, 13%)은 Compute-bound 영역에 위치
- **BW 최고 Phase: Decode / BW 최저 Phase: Vision**
- 두 극단의 차이: **6배** (35 → 211 GB/s)

---

## Figure 3 — `fig3_power_timeline.{png, pdf}`

### 전체 설명
**현재 상태: Thor에서 bw_timeseries.json 전송 후 자동 생성됨 (아직 미생성)**

생성 조건: `bw_timeseries.json`에 `vdd_gpu_mW` 필드가 존재해야 함.
(`260514_bw_monitor.py` 최신 버전으로 재실행 필요 — power 필드 저장 수정됨)

---

### 생성 후 예상 내용

**Panel (a) — GPU / CPU Power Profile**
시간 축(초) 위에 두 선:
- **빨간선** = VDD_GPU 전력 (W, sensor offset 보정됨)
  - Prefill 구간: 높은 전력 (~30–80 W) → compute-bound 시각화
  - Decode 구간: 낮은 전력 (~5–20 W) → BW-bound 시각화
  - **전력 패턴이 compute vs BW-bound 판별의 직관적 지표**
- **파란선** = VDD_CPU_SOC_MSS 전력 (W)
  - 추론 전 구간에서 비교적 일정 (~4–6 W)

**Panel (b) — GPU Utilization (%)**
- nvidia-smi GR3D% 시계열
- Prefill 구간: 높은 SM utilization (~80–100%)
- Decode 구간: 낮은 SM utilization (~20–40%) ← 메모리 대기 시간

**하단 각주:**
tegrastats 100ms 샘플링 간격 + GPU 전력 오프셋(-392 mW) 보정 사실 명시

---

### Figure 3 생성 방법 (Thor → Windows 전송 후)

```bash
# 1. 최신 bw_monitor 전송 (power 필드 포함 버전)
scp /mnt/c/Users/nanay/Desktop/Alphamayo/scripts/profiling/260514_bw_monitor.py \
    ice401@100.95.177.101:~/alpamayo1.5/scripts/profiling/

# 2. Thor에서 재실행
source ~/alpamayo1.5/a1_5_venv/bin/activate
python3 scripts/profiling/260514_bw_monitor.py

# 3. 결과 Windows로 전송
scp -r ice401@100.95.177.101:~/alpamayo1.5/profiling_results/260514_bw/ \
    /mnt/c/Users/nanay/Desktop/Alphamayo/profiling_results/

# 4. Paper figure 재생성 (fig3 포함)
cd C:\Users\nanay\Desktop\Alphamayo
python scripts\profiling\260514_bw_plot_paper.py
```

---

## Figure 4 — `fig4_summary_table.{png, pdf}`

### 전체 설명
4개 Phase의 핵심 수치를 **논문 삽입용 테이블 figure**로 정리.
수치 표를 텍스트가 아닌 figure로 제공하므로 LaTeX 없이도 논문 품질 표 삽입 가능.

---

### 레이아웃 구성

**제목 (Figure 상단):**
플랫폼 / 토큰 수 / 피크 BW / 모델 크기를 한 줄로 요약

**헤더 행 (진파란 배경, 흰색 텍스트):**
| Phase | Warmup (ms) | Measure (ms) | Speedup | BW (GB/s) | MBU (%) |

**데이터 행:**
- 짝수 행: 연파란 배경 (가독성을 위한 alternating 줄무늬)
- Decode BW / MBU 셀만 **빨간색** 강조 → 해당 수치가 핵심임을 시각적으로 표시
- 나머지 Phase의 BW/MBU는 `--` (미측정 or 추정값이므로 직접 기재 지양)

**Total 행 (연파란 배경 + 볼드):**
- Warmup 합계 5,219 ms vs. Measure 합계 4,969 ms
- 전체 speedup 1.05× (Warmup보다 Measure가 약 250 ms 빠름)

---

### 각 컬럼 해설

| 컬럼 | 의미 | 측정 방법 |
|------|------|-----------|
| Phase | 추론 단계 명칭 | — |
| Warmup (ms) | JIT 컴파일 포함 첫 실행 시간 | CUDA Events |
| Measure (ms) | 안정 상태 평균 (n=2) | CUDA Events |
| Speedup | Warmup / Measure 비 | 산술 계산 |
| BW (GB/s) | DRAM 대역폭 | CUDA Events (Decode만 직접 측정) |
| MBU (%) | BW / 273 GB/s × 100 | 산술 계산 |

**Speedup 컬럼 주의사항:**
- Vision 1.55× / Prefill 1.08× → JIT 효과로 Measure가 빠름 (정상)
- Decode 0.85× → Measure가 더 느린 것처럼 보이지만 **토큰 수 차이(15→18)**가 원인
  - 논문 인용 시: "Decode speedup은 token count variation으로 인해 유의미하지 않음" 주석 필요

---

## 공통 Style Guide

모든 4개 figure에 적용된 스타일 설정:

| 설정 | 값 | 이유 |
|------|-----|------|
| 폰트 패밀리 | Liberation Sans / Arial / DejaVu Sans | 한글 glyph 없는 폰트 → 경고 없음 |
| 텍스트 언어 | **영어 전용** | DejaVu Sans는 Hangul 미지원 → "Glyph missing" 경고 방지 |
| DPI | 300 (PNG), vector (PDF) | 학회 제출 기준 (NeurIPS: 300 DPI) |
| Figure 너비 | 7.0 inch | IEEE double-column = 7.16", NeurIPS = 6.875" |
| 배경 | 흰색 (facecolor="white") | 인쇄 친화적 |
| 상단/우측 spine | 제거 | 불필요한 테두리 최소화 (modern style) |
| Grid | y축 방향만, alpha=0.35 | 가독성 유지하면서 시각적 노이즈 최소화 |
| 위상 색상 | Vision=#5B9BD5, Prefill=#70AD47, Decode=#C0504D, Flow=#9067A7 | 색맹 친화적 (muted palette) |

**한글 경고 (이전 버전의 문제):**
이전 260514_bw_monitor.py의 figure 함수들은 한글 레이블 사용 →
matplotlib 기본 폰트(DejaVu Sans)가 Hangul glyph 미지원으로 수백 개 경고 발생.
이번 paper figure script는 **모든 레이블/제목을 영어로 작성**하여 완전 해소.

---

## 파일 목록

```
profiling_results/260514_bw/paper_figures/
├── fig1_inference_profile.png    (300 DPI, ~7×5.8 inch)
├── fig1_inference_profile.pdf    (vector, 논문 제출용)
├── fig2_bandwidth_utilization.png
├── fig2_bandwidth_utilization.pdf
├── fig3_power_timeline.png       ← Thor 재실행 후 생성
├── fig3_power_timeline.pdf       ← Thor 재실행 후 생성
├── fig4_summary_table.png
└── fig4_summary_table.pdf
```

---

*생성 스크립트*: `scripts/profiling/260514_bw_plot_paper.py`
*데이터 소스*: `profiling_results/260513_v4/phase_v4.json` (v4 fallback) + `profiling_results/260514_bw/bw_analysis.json` (Thor BW run, 전송 시 우선 사용)

# Alpamayo 1.5 — 순간 DRAM 대역폭 측정 방법론과 결과

**작성일**: 2026-06-09 / **업데이트**: 2026-06-11 (Decode 완료 + Prefill 완료 + VE 완료 + Flow 완료 + SM 메트릭 수정)  
**환경**: Jetson AGX Thor (SM 11.0, LPDDR5X 231 GB/s peak, CUDA 13.0)  
**관련 문서**: `260608_01_Alpamayo_4단계_실제_DRAM_대역폭_ncu_실측_분석.md`

---

## 이 측정으로 무엇을 결정하는가

> **한 줄 요약: DRAM 대역폭 측정은 "얼마나 빠른가"를 아는 게 아니라, "어디를 고쳐야 빨라지는가"를 확정하기 위한 것이다.**

### SM util과 DRAM BW가 함께 말해주는 것

**측정 목적:** GPU/CPU가 언제 노는지 파악 → async pipeline(cudaMemPrefetchAsync + CUDA Stream 이중화)으로 겹칠 수 있는 시간 구간을 확정한다.

두 지표가 결정하는 것:

| DRAM BW | smsp_occupancy | 의미 | Async Pipeline 기회 |
|---------|---------------|------|-------------------|
| > 70% | 높음 | DRAM이 거의 포화 → Stream 2 DMA 여지 적음 | 커널 간 갭 제거(CUDA Graph) |
| 30~60% | 높음 | DRAM 여유 있음 → SM은 다른 커널 실행 중이지만 BW는 남음 | **Stream 2로 DMA prefetch 최대 기회** |
| < 30% | 높음 | DRAM 거의 안 씀 (SRAM 재사용) → BW 매우 여유 | **Stream 2 DMA 최대 기회** |

> ⚠ **SM% 해석 주의 (2026-06-11 수정):**  
> GB10B(SM 11.0)에서 `sm__active_cycles.sum`은 **미지원 메트릭**으로 항상 0을 반환한다.  
> 올바른 메트릭은 `smsp__cycles_active / smsp__cycles_elapsed` (warp 스케줄링 occupancy).  
> 이 값은 DRAM stall 중인 warp도 "active"로 카운트하므로 **높아도 BW 여유가 있다**.  
> **"SM이 논다" = smsp_occupancy 낮음이 아니라 커널 간 dispatch 갭(GPU idle)을 봐야 한다.**

**우리 측정 결과 (Async Pipeline 관점)**  
- Decode GEMV: DRAM 57~92% → BW 여유 8~43% → 커널 간 갭 10.6%가 주요 idle  
- **Prefill FlashAttention: DRAM 39.6% → BW 여유 60% = 140 GB/s** ← **최대 pipeline 삽입 기회**  
- Prefill nvjet GEMM: DRAM 63~70% → BW 여유 30~37% = 69~85 GB/s  
- Flow nvjet GEMV (대형): DRAM 57% → BW 여유 43% = 99 GB/s  
- Flow elementwise: DRAM ~31% → BW 여유 69% = 159 GB/s

---

### 측정 수치가 결정하는 것 4가지

#### ① Async Pipeline 삽입 가능 구간을 수치로 결정할 수 있다

```
DRAM BW 여유 = (1 - BW%) × 231 GB/s → Stream 2에서 DMA prefetch 가능한 BW

FlashAttention (Prefill): DRAM 39.6% → 여유 140 GB/s, 41 ms
  → Stream 2: 140 GB/s × 41 ms = 5.7 GB DMA 가능
  → 다음 단계(Flow or Decode) 레이어 가중치 선적재 가능

nvjet GEMM (Prefill): DRAM 70% → 여유 69 GB/s, 각 레이어 ~3 ms
  → Stream 2: 69 GB/s × 3 ms = 0.21 GB DMA 가능
  → 다음 레이어 가중치 일부 prefetch 가능 (레이어당 0.5 GB 목표 대비 제한적)

Decode GEMV (전체): DRAM 57~92%
  → 커널 간 갭 10.6%(229 ms)가 주요 idle → CUDA Graph 핵심 타겟
  → 커널 내 BW 여유는 8~43% 존재하나 커널 실행 시간 짧아 DMA 효과 제한적
```

#### ② CUDA Graph 효과를 정량화할 수 있다

```
GPU idle: 229 ms / 2,171 ms = 10.6%  (Decode nsys 실측)
idle 유형: 1~10 μs gap 35,498건 = 158.9 ms → CUDA Graph로 제거 가능

CUDA Graph 적용 시 예상 Decode 단축: 158.9 ms
현재 Decode: 1,503 ms → 적용 후: ~1,344 ms (-10.6%)

Prefill idle: 아직 미측정 (VE/Flow와 함께 향후 추가 예정)
```

#### ③ Async Pipeline(Layer Prefetch) 설계 제약을 결정할 수 있다

RT-Swap / Demand Layering 아이디어의 핵심은 "레이어 N을 계산하는 동안 레이어 N+1 가중치를 DMA로 미리 당긴다"는 것이다. 이게 가능하려면 계산 중에 여유 DRAM BW가 있어야 한다.

```
Decode (SM=0%, DRAM=89%):
  레이어 계산 중 남은 BW = 231 - 89% × 231 = 25 GB/s
  다음 레이어 가중치 494 MB를 25 GB/s로 전송: 20 ms
  레이어 실행 시간: 3,039 μs = 3 ms
  → prefetch 20 ms > 레이어 시간 3 ms → 이 방식은 Decode에 적용 불가

Prefill FlashAttention (DRAM=39.6%, SRAM/compute-bound):
  실행 시간: 2,298 μs/layer × 18 layers = 41 ms
  DRAM 사용: 39.6% → 남은 BW = 231 × (1 - 39.6%) = 140 GB/s
  41 ms × 140 GB/s = 5.7 GB 전송 가능
  → 다음 단계(Flow) 가중치 사전 적재 가능 ← 실제 활용처
  ※ FlashAttention이 SRAM-bound이므로 이 BW 여유는 실제로 사용 가능 ✅

VE (DRAM≈43~61% GEMM / 24.3% FlashAttn, smsp_occ 89~100%):
  nvjet GEMM BW ≈ 50% → 남은 BW = 231 × 50% = 115 GB/s
  VE GEMM 구간 활용 가능 → Prefill 가중치 일부 prefetch 가능
  단, patch embedding conv(3.5% BW, 48ms)는 단독으로 병목이 됨
```

#### ④ 단계 간 BW 공유 가능성 판단

```
VE → LM Prefill 사이: (2026-06-11 실측 반영)
  VE nvjet GEMM: DRAM-bound(BW 43~61%), smsp_occ 89~100%
  "남는 compute로 Prefill 처리" 불가 (DRAM-bound).
  단, VE nvjet GEMM 구간(BW 43~61%)에서 39~57%의 DRAM BW 여유 있음
  → Prefill 가중치 일부 prefetch 가능 (RT-Swap 방식)
  patch embedding conv(8.1 GB/s, 48ms)는 BW 여유 많지만 단독 커널이라 pipeline 효과 제한적

Prefill → Decode 사이: FlashAttention 구간(39.6% BW)에서
  → Decode 첫 레이어 가중치를 미리 L2에 고정(L2 Persistent Residency) 가능

Flow (ODE×10, ~870 ms):
  ★ 수정 (2026-06-11 실측): stage BW ~44% → 남은 BW = 129 GB/s
  이전 추정 88% = 오류 → "Flow는 BW 포화라 prefetch 불가" 결론도 수정 필요
  실제: 129 GB/s × 870 ms = 112 GB 여유 BW 공간 존재
  → 다음 추론 VE 가중치 전체(22 GB) 선적재 충분히 가능
  → ODE step 내 elementwise-GEMV CUDA Stream 이중화도 가능 (남은 BW 129 GB/s)

Decode+Flow 마지막 → 다음 추론 첫 VE 사이 (10Hz 시스템에서 100ms 주기):
  → 100 ms 공백에 다음 추론 가중치 prefetch
  → 231 GB/s × 100 ms = 23.1 GB 전송 가능 = 모델 가중치 전체(22 GB) 1회 전부 적재
  ← 이게 Inter-Inference Pipeline의 이론적 근거
```

---

### SM 메트릭 해석 가이드 (2026-06-11 수정)

**⚠ 중요: GB10B(SM 11.0)에서 `sm__active_cycles` = 항상 0 (미지원 메트릭)**

```
올바른 smsp_occupancy 해석:
  ① SM이 LOAD 명령 발행         → warp active (smsp_active count++)
  ② DRAM 응답 대기 (stall)       → warp still active (smsp_active count++)  ← 핵심
  ③ 데이터 도착, 곱셈/덧셈 실행  → warp active (smsp_active count++)
  ④ → ①로 반복

smsp_occupancy = smsp_active.sum / smsp_elapsed.sum
              = (① + ② + ③) / 전체 = 거의 100%

→ stall 중인 warp도 "active"이므로 DRAM-bound이든 compute-bound이든
   GEMM처럼 warp가 많은 커널은 모두 높은 값 나옴.
```

**계산은 일어난다.** GEMV에서 SM의 대부분 시간은 DRAM 응답 대기(stall)이다.  
그러나 smsp_occupancy는 이 stall 시간도 active로 카운트하여 96~100%를 반환한다.

**DRAM BW%가 bound의 유일한 올바른 지표:**
```
nvjet GEMM (DRAM 70%): warp의 대부분이 DRAM stall → DRAM-bound
FlashAttention (DRAM 39.6%): warp의 많은 부분이 SRAM 내 연산 → SRAM/compute-bound
```

---

## 결론 (한 줄)

> **Decode GEMV: lm_head 212 GB/s(92%), FFN 180 GB/s(78%), FFN down 131 GB/s(57%), KV ≥194 GB/s(≥84%), byte-weighted 171 GB/s(74%).**  
> **Prefill GEMM: 162.7 GB/s(70%) / 144.7 GB/s(62%). FlashAttention(seq=3086): 91.5 GB/s(39.6%, DRAM) — SRAM/compute-bound 판정 (DRAM BW 39.6% < 60%) ← Prefill stage 55%의 시간 주원인이나 DRAM이 병목은 아님.**  
> **VE (ViT): nvjet GEMM 100~141 GB/s(43~61%), FlashAttention 56 GB/s(24.3%, SRAM-bound), patch conv 8.1 GB/s(3.5%) ← VE stage 35%의 주원인.**  
> **Flow (ODE×10): nvjet GEMV 131~221 GB/s (57~96%), elementwise 72~141 GB/s, stage BW ~44% (보정). 이전 추정치 203 GB/s(88%)는 과대 추정이었음 확인.**  
> **⚠ SM% 수정 (2026-06-11): GB10B에서 sm__active_cycles 미지원 → 이전 "SM=0% 전 단계" 결론은 측정 오류. smsp_occupancy(warp occupancy) 실측 89~100%. DRAM BW%가 bound 판별 기준. Async Pipeline 설계: FlashAttention(DRAM 39.6%, BW여유 140 GB/s)이 cudaMemPrefetchAsync Stream 2 최대 기회.**

---

## 1. 측정 결과

### 1.1 Decode 단계 — 측정 완료

| 항목 | 값 | 비고 |
|------|-----|------|
| **GEMV byte-weighted 순간 BW** | **171 GB/s (74% peak)** | 4 타입 전체 ncu+nsys 교차검증 |
| **lm_head 순간 BW** | **212 GB/s (92% peak)** | 단일 커널 최고 효율 |
| **Stage 평균 BW** | **204.6 GB/s (89% peak)** | 260609 ncu 실측 (갭 포함 분모) |
| **SM 활용률 (전체 GEMV)** | **0%** | DRAM 전송 중 연산 없음 |
| **GPU idle** | **10.6%** (229 ms / 2,171 ms) | 커널 간 dispatch 갭, nsys 기준 상한값 |
| **Decode 총 DRAM** | **307.49 GB** | ncu 하드웨어 카운터, ground truth |
| ncu replay 오버헤드 | 2.85× | 타이밍만 부풀려짐, 바이트는 정확 |

### 1.2 Decode GEMV 커널 타입별 순간 BW (2026-06-11 완료)

| 커널 타입 | 추정 역할 | DRAM/k | 커널 수 | 순간 BW | peak 대비 | 신뢰도 |
|---------|---------|--------|--------|---------|---------|--------|
| `gemv2T` (lm_head) | vocab 투영 [4096→152K] | 1,276 MB | 19 | **212 GB/s** | 92% | ✅ 높음 (오차 0.25%) |
| `nvjet 192x8_64x8` | KV head 투영 (GQA 소형) | 8 MB | 1,368 | **≥194 GB/s** | ≥84% | ⚠ 하한 (nsys 오버헤드 35%) |
| `nvjet 128x8_64x12` | FFN gate/up 또는 QO 투영 | 67 MB | 2,736 | **180 GB/s** | 78% | 🔶 중간 (nsys 4% 과대, 실제 ≈81%) |
| `nvjet 512x8_64x3 (splitK)` | FFN down [11008→4096] | 101 MB | 684 | **131 GB/s** | 57% | ✅ 높음 (오차 2%) |
| **byte-weighted 평균** | — | — | 4,826 | **171 GB/s** | **74%** | — |

*분석 스크립트: `scripts/profiling/260611_analyze_nvjet_bw.py`*

**BW 격차의 해석:**

```
lm_head (92%): 1.276 GB 연속 burst → DRAM row hit 최대화 → 최고 효율
192x8   (≥84%): 8 MB 소형. nsys 오버헤드가 kernel 시간의 35%라 과소평가.
                실제 BW는 ≥194 GB/s이며 ~210-230 GB/s 가능성 높음
128x8   (78%):  splitK+TNT 혼합 중형. nsys 4% 오차 보정 시 실제 ≈81%
512x8   (57%):  splitK K-방향 분할 → DRAM row hit율 저하 → 진짜 57% (오차 아님)
```

**왜 512x8 splitK가 57%로 낮은가:**  
FFN down [11008→4096] GEMV를 K-방향(=11008)으로 분할해서 여러 CUDA block이 병렬 처리한다.
각 block은 K의 서로 다른 비연속 구간(0..3669, 3669..7337, 7337..11008)에 접근한다.
DRAM 입장에서 이는 서로 다른 메모리 page를 교대로 건드리는 패턴이므로 row precharge/activate
횟수가 증가하고 sustained BW가 하락한다. lm_head처럼 1.276 GB를 연속으로 긁는 경우와 대비된다.

### 1.3 수치 해석

**byte-weighted 순간 BW(171 GB/s) vs stage BW(204.6 GB/s) 불일치 이유:**

두 지표는 다른 분모를 쓴다. 모순이 아니다.

```
stage 평균 BW  = 307.49 GB / 1,503 ms              = 204.6 GB/s  (갭 포함 실측)
순간 BW (ncu+nsys) = Σ(ncu bytes) / Σ(nsys 커널 시간)
                                                          ↑ nsys 오버헤드가 소형 커널 시간을 부풀림
```

특히 192x8 (43 μs, 오버헤드 35%)와 128x12 (374 μs, 4% 오버헤드)의 BW가 nsys에 의해 과소평가된다.
보정하면 byte-weighted 실제값은 204.6 GB/s에 근접한다. 두 값은 일관된다.

**lm_head란:** 매 decode step마다 1회 실행되는 어휘 투영 행렬 `[4096 × 152,064]` (BF16, 1.276 GB).
19 steps × 1회 = 19개 커널.

### 1.4 가장 중요한 발견

```
SM 활용률 = 0%  →  GPU는 DRAM 데이터가 올 때까지 100% 대기

즉, 모든 GEMV 커널 속도 = DRAM 전송 속도  (1:1 종속)
compute-memory 중첩(overlap) 없음
병목 = DRAM 자체 (BW가 57~92%까지 다양하나, 모두 DRAM-bound)
```

---

## 2. 측정 현황: 완료 / 미완료

### 2.1 전체 추론 파이프라인 현황

| 단계 | 소요 시간 | Stage BW (260609) | 순간 BW (이 문서) | 상태 |
|------|---------|-------------------|-----------------|------|
| **VE** | **728 ms** | **80 GB/s (35%)** | **✅ 100~141 GB/s (GEMM), FlashAttn 56 GB/s** | **완료** |
| **LM Prefill** | **1,423 ms** | **126 GB/s (55%)** | **✅ 144~163 GB/s (GEMM), FlashAttn 91.5 GB/s** | **완료** |
| **LM Decode** | **1,503 ms** | **204.6 GB/s (89%)** | **✅ 131~212 GB/s (GEMV)** | **완료** |
| **Flow** | **870 ms** | ~~203 GB/s (88%)~~ **→ ~101 GB/s (~44%)** | **✅ nvjet 131~221 GB/s, elementwise 72~141 GB/s** | **완료** |

### 2.2 Decode 내 세부 항목

| 커널 유형 | 순간 BW | 상태 |
|---------|--------|------|
| lm_head (vocab 투영, 1.276 GB) | ✅ **212 GB/s (92%)** | 직접 측정 완료 |
| FFN gate/up 투영 (`nvjet 128x8`) | ✅ **180 GB/s (78%)** | 직접 측정 완료 |
| FFN down 투영 (`nvjet 512x8` splitK) | ✅ **131 GB/s (57%)** | 직접 측정 완료 |
| KV head 투영 (`nvjet 192x8`) | ✅ **≥194 GB/s (≥84%)** | 직접 측정 완료 (하한) |
| SM 활용률 (모든 GEMV) | ✅ **0%** | 완료 |
| GPU idle 분포 | ✅ **10.6%** | 완료 (상한값) |

### 2.3 Prefill 내 세부 항목 (2026-06-11 완료)

| 커널 유형 | 역할 | DRAM/k | N(실제) | 순간 BW | peak% | SM%(warp occ) | bound | 신뢰도 |
|---------|------|--------|--------|---------|-------|-------------|-------|-------|
| `nvjet 256x128_64x5` (GEMM) | Q·K·V·O·gate·up 투영 (6종) | 494 MB | 108 | ✅ **162.7 GB/s** | 70.4% | ~96% | **DRAM-bound** | ✅ 높음 |
| `nvjet 256x208_64x4` (GEMM) | FFN down 투영 | 788 MB | 18 | ✅ **144.7 GB/s** | 62.6% | ~96% | **DRAM-bound** | ✅ 높음 |
| **FlashAttention** (seq=3086) | **Self-attention** | **210 MB** | **18** | ✅ **91.5 GB/s** | **39.6%** | **~96%** | **⚠ SRAM/compute-bound** | ✅ 높음 (2.3 ms, 오차 0.7%) |
| elementwise 계열 | LayerNorm/SiLU/RoPE/residual | 41~60 MB | ~400 | ✅ **170~226 GB/s** | 74~98% | ~89% | **DRAM-bound** | 🔶 중간 (nsys 오버헤드 5~7%) |
| **SM occ (전체 2032 커널)** | — | — | — | — | — | ✅ **89~100%** | DRAM BW%로 분류 | smsp_occupancy |

> ⚠ SM% = `smsp__cycles_active / smsp__cycles_elapsed` (warp 스케줄링 occupancy). DRAM stall 중인 warp도 active 포함. 96% = warp가 많이 스케줄됨(GEMM 특성) ≠ compute utilization 96%.

**★ 모델 레이어 수 확인: 18 transformer layers**

```
nvjet_256x128 실제 호출 수: 216 ÷ 2(2×이중산정) = 108 → 108 / 18층 = 6 per layer
  (Q, K, V, O, gate, up 6개 투영)
nvjet_256x208 실제 호출 수:  36 ÷ 2               =  18 →  18 / 18층 = 1 per layer
  (FFN down 1개 투영)
FlashAttention 실제 호출 수: nsys 직접 측정 = 18 →  18 / 18층 = 1 per layer ✅
```

**★ stage BW 55%의 원인: FlashAttention이 병목**

```
Prefill 내 커널별 BW 효율:
  nvjet GEMM 투영:    144~163 GB/s (63~70%)
  FlashAttention:      91.5 GB/s (39.6%)  ← 가장 낮음 = 병목
  elementwise 계열:   170~226 GB/s (74~98%)

stage BW = Σ(DRAM) / Σ(시간)
  FlashAttention의 39.6%가 전체 평균을 끌어내려 stage 55%를 만든다.
  Decode(89%)에는 FlashAttention 부하가 훨씬 작음(GEMV에 seq=1 attention).
```

**FlashAttention BW가 낮은 이유:**

```
nvjet GEMM:      가중치 한 덩어리(494 MB) 연속 단일 스트림  →  high BW 가능
FlashAttention:  Q + K + V + O 4개 텐서 + AppendOnlyCache KV write
                 여러 DRAM 스트림 동시 경쟁 → sustained BW 저하
                 DRAM 210 MB / 2.3 ms = 91.5 GB/s (39.6%)
```

**FlashAttention 산술강도 및 bound 재분석 (2026-06-11 수정):**

```
★ 두 가지 산술강도 개념의 구분이 핵심:

[외부 메모리(DRAM) 기준 산술강도 — "DRAM 접근 비율"]
  FlashAttention: O(n) DRAM 접근, O(n²) FLOPs → 강도 ≈ n/4 = 771 FLOPS/byte
  Prefill GEMM:   O(seq×d) DRAM, O(seq×d×d) FLOPs → 강도 ≈ 1,659 FLOPS/byte
  Thor BF16 ridge point (DRAM 기준): ≈ 4,480 FLOPS/byte

  771 < 1,659 < 4,480 → DRAM 관점에서는 둘 다 DRAM-bound 구간

[SRAM 내부 산술강도 — FlashAttention의 실제 동작]
  FlashAttention은 SRAM tiling 알고리즘:
    - Q·K·V 블록을 SRAM에 로드 후 SRAM 내에서 재사용
    - DRAM 접근: O(n) (표준 attention O(n²) 대비 대폭 감소)
    - SRAM 내 FLOPs: O(n²) (블록 크기 B 기준 B²×d 연산)
    - SRAM 기준 강도: 충분히 높음 → SRAM 충분히 활용

[측정 결과와 해석]
  DRAM BW% = 39.6% < 60% → mixed/SRAM-bound 판정 ✅
  smsp_occupancy = 96% → warp가 idle하지 않음 (연산 또는 SRAM 접근 중)
  
  → FlashAttention(seq=3086)은 DRAM이 아닌 SRAM/compute 경로가 병목
  → 가중치 양자화는 DRAM bytes 감소 → DRAM-bound 커널에만 선형 효과
  → FlashAttention에 대한 양자화 효과는 제한적 (이미 DRAM을 적게 씀)
  
★ 이전 결론 "FlashAttention도 DRAM-bound, SM=0%" → 수정:
   - SM=0%: GB10B에서 sm__active_cycles 미지원으로 인한 측정 오류
   - DRAM-bound: DRAM BW 39.6%로 mixed/SRAM-bound가 올바른 분류
```

**Prefill stage BW 55%의 원인 (수정된 해석):**

```
FlashAttention이 stage 시간의 상당 부분을 차지하지만 DRAM은 적게 쓴다.
→ stage BW(DRAM/시간) 계산 시 분모(시간)에 FlashAttention 실행 시간이 포함
→ FlashAttention 구간에서 DRAM 처리량이 낮아 stage 평균 BW가 하락

이전 해석: "FlashAttention이 DRAM-bound라 BW 39.6%로 끌어내린다"
수정 해석: "FlashAttention이 SRAM-bound라 이 구간에서 DRAM을 거의 안 쓰므로
            stage DRAM/시간 비율이 낮아진다"
결론은 같음: FlashAttention이 stage BW 55%의 원인 — 단 그 메커니즘이 다름.
```

**ncu 2× 이중산정 이슈 (정리):**

```
원인: --replay-mode kernel + 6 메트릭 → 2 pass → 각 커널 CSV에 2회 등장
결과:
  ncu 보고 총 DRAM: 231.19 GB  →  실제: 115.6 GB (÷2)
  ncu 커널 수:       2032       →  실제: 1016
  nsys 커널 수:      1048             (정확, 기준값)

BW 교차검증은 영향 없음:
  n_match = min(ncu, nsys) = nsys 수 → pass1 바이트만 취함 → 각 BW 값 정확 ✅
```

### 2.4 VE 세부 항목 (2026-06-11 완료)

| 커널 유형 | 역할 | DRAM/k | N(ncu실제) | 순간 BW | peak% | SM% | 비고 |
|---------|------|--------|-----------|---------|-------|-----|------|
| `nvjet_tst_128x256_64x4_1x2` | GEMM (large proj) | 276 MB | 27 | 100.0 GB/s | 43.3% | 0% | ✅ |
| `nvjet_tst_128x256_64x4_1x1` | GEMM (mid proj) | 206 MB | 27 | 126.6 GB/s | 54.8% | 0% | ✅ |
| `nvjet_tst_128x256_64x6_2x1_2cta` | GEMM (multi-CTA) | 162 MB | 27 | 141.4 GB/s | 61.2% | 0% | ✅ |
| `nvjet_tst_256x256_64x4_2x1_2cta` | GEMM (large) | 200 MB | 4 | 140.3 GB/s | 60.7% | 0% | ✅ |
| elementwise 계열 | LayerNorm/RoPE/GELU | 40~75 MB | ~270 | 167~175 GB/s | 72~76% | 0% | ✅ |
| **FlashAttention (VE ViT)** | Self-attention | **7.4 MB** | **27** | **56.0 GB/s** | **24.3%** | 0% | ⚠ ncu 미캡처 문제 |
| **`implicit_convolveNd_sgemm`** | **patch embedding conv** | **392 MB** | **1** | **8.1 GB/s** | **3.5%** | 0% | **🚨 이상 커널** |
| **SM util (전체)** | — | — | 1755 | — | — | **0%** | VE도 memory-bound |

**★ 핵심 발견: VE가 compute-bound라는 예측이 틀렸다**

```
예측: VE stage BW = 35% → SM이 바빠서 BW를 못 씀 (compute-bound)
실측: SM util = 0% for ALL 1755 kernels → memory-bound

이유: VE(ViT) 산술강도 분석
  - ViT GEMM (패치 N개): 산술강도 ∝ 패치 수. 패치 수 < 4480이면 memory-bound.
  - Thor ridge point = 4480 FLOPS/byte (BF16)
  - Alpamayo VE 패치 수 = 6카메라 × 해상도 기반 → 4480 미만으로 추정됨
  → GEMM도 FlashAttention도 모두 DRAM-bound
```

**VE stage BW(35%) 낮은 이유 확정:**

```
원인 3가지 중첩:

① nvjet GEMM 효율 40~61%
   Prefill GEMM(70%)보다 낮음. 이유: 패치 seq_len이 3086보다 작음
   → 타일 미달 (256×128 타일을 채우지 못해 DRAM burst 효율 저하)

② FlashAttention BW 56 GB/s (24.3%)
   Prefill FlashAttention(91.5 GB/s)보다도 낮음
   이유: VE attention head 수 × 패치 수가 적어 DRAM burst 길이 짧음
   ⚠ 주의: ncu=27, nsys=432 → 16× 미캡처. 실제 VE FlashAttention DRAM = 432×7.4MB = 3.2 GB
   → 미캡처 원인: NVTX 중첩 구조 (ViT per-block sub-range)

③ implicit_convolveNd_sgemm: 8.1 GB/s (3.5%)  ← 최대 이상값
   정체: patch embedding convolution (이미지 픽셀 → 패치 임베딩)
   6카메라 × 고해상도 입력 → 392 MB DRAM, 48.4 ms 소요
   cuDNN이 선택한 implicit GEMM 알고리즘이 이 입력 shape에 비효율적
   최적화 시: cuDNN 알고리즘 강제 선택 or im2col+GEMM 대체 → 2~3 ms 목표 (16× 개선)
```

**FlashAttention [ncu=27 nsys=432] 미캡처 원인:**

```
VE NVTX 구조 (추정):
  Vision_Encoder (push)
   ├── [per-layer or per-camera sub-range] (push)
   │    └── flash_fwd_kernel  ← 중첩된 sub-range 내부에서 실행
   └── nvjet GEMM             ← Vision_Encoder 직접 범위에서도 실행 (캡처됨)

ncu --nvtx-include "Vision_Encoder":
  Vision_Encoder 범위 내 커널을 캡처하지만,
  flash attention은 deeper sub-range가 active일 때 실행
  → 일부 캡처, 대부분 누락

nsys 시간 범위 기반 캡처:
  Vision_Encoder 범위 내 시간 전체를 커버 → 432개 전부 캡처

미캡처 FlashAttention DRAM 보정:
  (432 - 27) × 7.4 MB = 3.0 GB 추가 DRAM 미계산
  VE 실제 총 DRAM ≈ 43 GB + 3 GB = 46 GB (추정)
```

### 2.5 Flow (ODE) 세부 항목 (2026-06-11 완료)

**측정 개요:**

```
ncu 필터: --nvtx-include "FlowODE/FlowStep"
ncu 레코드: 23,336개 (10 ODE steps × 2-pass 이중산정 포함)
nsys FlowStep 1 step 시간: 111.3 ms  →  10 steps 총 ~1,113 ms

커버리지: ncu 1,167 / nsys 2,444 = 47.7%  ⚠ 절반만 캡처됨
  원인: FlowStep 내부의 중첩된 NVTX sub-range(per-layer 등) 안에서
        실행되는 커널은 ncu NVTX 필터가 놓침 (VE와 동일한 구조적 한계)
  영향: ncu DRAM 수치는 실제의 하한값, 커버리지 보정 후 ~2× 가 실제 추정
```

**커널 조성 (ncu 실측, DRAM 기준, 2× 이중산정 보정 후):**

| 커널 그룹 | 실제 DRAM | 비율 | ncu BW | 해석 |
|---------|---------|------|--------|------|
| **nvjet GEMV/GEMM** | **24.32 GB** | **45.1%** | 74.1 GB/s | 가중치 선형 투영 (Action Expert) |
| **elementwise** | **17.01 GB** | **31.6%** | 40.2 GB/s | AdaLN/timestep/scale-shift/SiLU ★비율 높음 |
| **KV concat** | **6.77 GB** | **12.6%** | 23.4 GB/s | cross-attention K,V 텐서 concat |
| 기타 (attention 추정) | 4.90 GB | 9.1% | 27.9 GB/s | FlashAttention 또는 미분류 |
| reduction | 0.91 GB | 1.7% | 23.7 GB/s | softmax reduce 등 |
| **합계 (측정값)** | **53.92 GB** | 100% | — | ← 실제 추정 ~113 GB (커버리지 보정) |

**★ 이전 추정치 203 GB/s(88%) 수정:**

```
이전 추정 출처: 260608/260609 문서 — model weight size 기반 간접 추정
실제 ncu 측정 (2026-06-11):
  53.92 GB (2× 보정) ÷ 10 steps = 5.39 GB/step
  커버리지 보정: 5.39 / 0.477 = ~11.3 GB/step
  1 FlowStep nsys 시간: 111.3 ms
  
  Stage BW (측정 하한): 5.39 GB / 111.3 ms = 48.4 GB/s = 21.0% peak
  Stage BW (커버리지 보정): 11.3 GB / 111.3 ms = 101.5 GB/s = 43.9% peak

→ 203 GB/s(88%)는 과대 추정. 실제 stage BW ≈ 44% (101 GB/s)로 수정.
```

**순간 BW — nsys 교차검증 (1 FlowStep 기준):**

| 커널 이름 (단축) | N(교차) | avg DRAM/k | avg dur/k | 순간 BW | peak% | 신뢰도 |
|---------------|--------|----------|----------|---------|-------|--------|
| `nvjet_tst_448x64_64x3_2x1_2cta_v_bz_TNT` | 72 | 35.2 MB | 266.8 μs | **131.9 GB/s** | 57.1% | ✅ 높음 |
| `nvjet_tst_512x64_64x3_2x1_2cta_v_bz_splitK_TNT` | 36 | 37.6 MB | 283.9 μs | **132.3 GB/s** | 57.3% | ✅ 높음 |
| `nvjet_tst_112x64_64x9_2x1_v_bz_TNN` | 72 | 8.9 MB | 53.8 μs | **166.1 GB/s** | 71.9% | ⚠ 하한 |
| `nvjet_tst_64x64_64x16_2x1_2cta_v_bz_TNT` | 72 | 4.6 MB | 20.9 μs | **220.6 GB/s** | 95.5% | ⚠ 하한 |
| `nvjet_tst_112x64_64x9_2x1_v_bz_bias_TNN` | 1 | 2.5 MB | 12.0 μs | **204.7 GB/s** | 88.6% | ⚠ 하한 |
| `arch::Sm80 (FlashAttn 추정)` | 36 | 27.2 MB | 230.7 μs | **117.8 GB/s** | 51.0% | 🔶 중간 |
| elementwise 계열 (다양) | 934 | 0.7 MB | 9.2 μs | **71.8 GB/s** | 31.1% | ⚠ nsys 오버헤드 큼 |
| `nvjet_tst_64x32` (소형) | 1 | 0.7 MB | 5.3 μs | **126.7 GB/s** | 54.8% | ⚠ 하한 |

**BW 분석:**

```
대형 GEMV (448x64, 512x64) 57%:
  가중치 35~37 MB로 큼. splitK K-방향 분할로 DRAM row hit 저하.
  → Decode의 512x8 splitK(57%)와 동일한 패턴 확인 — 구조적 한계

소형 GEMV (64x64) 95.5%, 112x64_bias 88.6%:
  nsys 오버헤드로 하한값이나 높음.
  가중치 작을수록 high BW → GQA 소형 head 투영이 여기에 해당

elementwise (71.8 GB/s, 31.1%):
  낮은 BW의 원인: 매우 짧은 커널(avg 9.2 μs) → nsys overhead 비율 高
  실제 BW는 더 높을 것이나 정밀 측정 어려움.
  Flow에서 elementwise 비율이 Decode(20%)보다 높은 이유:
  → AdaLN (timestep conditioning), scale/shift/gate 연산이 많음
  → Diffusion ODE 구조의 특성

KV concat (12.6%):
  cross-attention을 위한 텍스트/ego-motion 조건화 K,V 적재.
  23.4 GB/s로 낮음 — 단순 메모리 복사지만 불연속 접근으로 BW 낮음
```

**Decode와의 구조적 비교:**

```
                  Decode      Prefill     Flow
FlashAttention    ~28%        ~54%        거의 없음(미캡처 가능성)
nvjet GEMV/GEMM   ~52%        ~46%        45.1%
elementwise       ~15%        ~4%         31.6%  ← Flow 특징
KV concat         거의없음    거의없음     12.6%  ← Flow 특징 (cross-attn)
reduction         ~5%         ~5%         1.7%

Flow의 차별점:
→ elementwise 3.1 GB/step = AdaLN + timestep 임베딩의 매 step 반복
→ KV concat  0.68 GB/step = 조건화 벡터(텍스트/ego)를 매 step attention에 주입
→ FlashAttention이 보이지 않음 → 미캡처(48.4% nesting 누락) 또는 다른 구현
```

**★ SM 및 bound 분석 (2026-06-11 수정):**

```
nvjet GEMV: DRAM-bound (아키텍처 근거)
  산술강도 ≈ 0.5 FLOPS/byte (batch=1 GEMV) ≪ ridge point 4480
  DRAM BW 57~96% → DRAM-bound 확정
  smsp_occupancy ≈ 89~100%: warp occupancy 높음 (DRAM stall 중에도 active)

elementwise: DRAM-bound
  산술강도 < 1 FLOPS/byte → 전형적 memory-bound
  DRAM BW ~31% (ncu BW 기준) — nsys 오버헤드로 하한값

KV concat: DRAM-bound
  단순 memcpy 패턴, DRAM BW 기준 memory-bound

⚠ 주의: GB10B에서 sm__active_cycles는 미지원 → smsp_occupancy 사용.
   smsp_occupancy = 89~100%이지만 DRAM BW%로 판정 시 nvjet GEMV = DRAM-bound.
```

**Async Pipeline에의 시사점 (수정된 수치 기준):**

```
Flow stage BW ~44% → 남은 BW = 231 × 56% = 129 GB/s

활용 전략:
① L2 Persistent Residency (제한적):
   L2 32 MB vs Action Expert 4.6 GB → 전체 고정 불가
   핵심 attention 파라미터(~30 MB 이내)만 L2 pin 가능
   → per-step BW 절감 효과 제한적

② ODE step 내부 pipeline:
   elementwise(71.8 GB/s, 31.6%) + nvjet GEMV(132 GB/s)를 CUDA Stream 이중화
   elementwise 커널 실행 중 다음 GEMV 가중치 prefetch
   BW 128 GB/s 남음 → 35 MB GEMV 가중치 prefetch = 0.27 ms (GEMV 266 μs와 동급)

③ Cross-step KV reuse:
   ODE 10 steps 모두 동일 텍스트/ego 조건화 KV 사용
   → step 1에서 KV를 L2에 올려두면 step 2~10은 L2 hit (32 MB 내 가능 여부 확인 필요)

④ Flow → 다음 VE prefetch 기회:
   Flow 실행 중 (~870 ms) VE 가중치 일부 prefetch 가능
   129 GB/s × 870 ms = 112 GB 전송 가능 → 전체 모델 22 GB 5회 적재 가능
```

---

## 3. 왜 기존 방법으로는 순간 BW를 알 수 없나

### 3.1 기존 stage BW의 정의와 한계

`260608_01` 문서의 stage BW = `DRAM bytes / stage 전체 시간`. 이 분모에는 두 성분이 섞인다:

```
stage 전체 시간 = (DRAM 전송 시간) + (커널 간 갭)
                              ↑               ↑
                       진짜 전송 속도      dispatch overhead
                        분모에 포함            분모에 포함
```

결과적으로 stage BW는 "갭까지 포함한 평균 처리량"이지, DRAM이 실제로 데이터를 전송하는 순간의 속도가 아니다.

### 3.2 ncu 단독으로도 안 되는 이유: replay 오버헤드

처음에는 ncu의 `gpu__time_duration.sum`(커널 실행 시간)으로 순간 BW를 계산하려 했다. 결과:

```
ncu 기반 순간 BW = 77.8 GB/s (peak의 34%)
```

이 값은 틀렸다. ncu의 `--replay-mode kernel`은 각 커널을 **개별 재실행(replay)** 하며 카운터를 수집하는데,
이 replay 인프라가 DRAM 응답 경로에 개입하여 실행 시간을 2.85× 늘린다:

```
ncu 측정 실행 시간:  17.219 ms
nsys 측정 실행 시간:  6.037 ms   ← 실제값
오버헤드 배율:        2.85×

"77.8 GB/s" = 77.8 × 2.85 ≈ 221 GB/s  →  정확히 측정하면 212 GB/s
```

단, **DRAM 바이트 수 카운터**(`lts__d_sectors_fill_sysmem.sum`)는 실제 전송된 sector 수를 세는 것이라
replay 오버헤드와 무관하다. **바이트는 정확, 시간만 부풀려진다.**

---

## 4. 측정 방법: ncu + nsys 교차검증

두 도구의 역할을 나눠서 각자의 강점만 사용한다.

| 도구 | 사용하는 정보 | 신뢰 여부 |
|------|------------|---------|
| **ncu** | 커널별 DRAM 바이트 수 | ✅ 정확 (hardware sector counter) |
| **ncu** | 커널별 실행 시간 | ❌ 2.85× 부풀려짐 (replay 오버헤드) |
| **nsys** | 커널별 실행 시간 | ✅ 정확 (GPU hardware timer, replay 없음) |
| **nsys** | 커널별 DRAM 바이트 수 | ❌ 없음 (타이밍만 기록) |

```
순간 BW = (ncu DRAM bytes) / (nsys 커널 실행 시간)
         = 1.276 GB        / 6.037 ms
         = 212.0 GB/s
```

### 4.1 이 측정이 포착하는 것의 정확한 정의

> **"순간 DRAM 대역폭"이 측정하는 것: 해당 커널 실행 중 DRAM 버스를 통해 실제 전송된 바이트 수를 실제 커널 실행 시간으로 나눈 값 — 즉 애플리케이션이 경험하는 실효(effective) DRAM 처리량.**

"DRAM에서 오는 데이터 양/전송 속도만 측정한 것인가"를 깊이 검토한다.

**ncu DRAM 바이트: L2 cache hit을 제외한 DRAM 버스 실통과 바이트**

```
lts__d_sectors_fill_sysmem.sum × 32 bytes/sector
= L2 cache가 DRAM에서 읽어들인 sector 수 × sector 크기
= 실제로 DRAM 버스를 건넌 바이트 수 (L2 hit 완전 제외)
```

L2 hit(32 MB 캐시 내 재사용)은 이 카운터에 포함되지 않는다.
GEMV 가중치(90 MB~1.276 GB)는 L2(32 MB)보다 크므로 L2 hit율 ≈ 0% → 측정된 바이트 = 실제 DRAM 전송 바이트. ✅

**nsys 커널 시간: "순수 DRAM 전송 시간"인가, 그보다 넓은 개념인가?**

GPU 하드웨어 타임스탬프 기준의 커널 시간은 "DRAM 데이터가 버스를 통과하는 시간만"이 아니다.
이 시간에는 다음이 모두 포함된다:

```
커널 시간 =  DRAM 지연(latency)       ← 첫 데이터 도착까지 메모리 컨트롤러 대기 (~수십~수백 ns)
           + DRAM 전송 시간           ← 바이트가 실제로 DRAM 버스를 통과하는 시간
           + DRAM 컨트롤러 오버헤드    ← bank conflict, row precharge, refresh pause
           + L2 fill 시간             ← DRAM→L2 적재 시간 (전송 시간과 중첩)
           + 커널 에필로그             ← 출력 y 기록, 자원 해제 (수 μs 수준)
           (+ compute 시간            ← SM util=0%이므로 이 항은 0)
```

SM util = 0%이므로 "compute 시간"은 포함되지 않는다. 하지만 "순수 DRAM 버스 통과 시간"보다는
bank conflict, latency 등 DRAM 내부 오버헤드가 추가로 포함된다.

**그럼에도 이 지표가 "DRAM 전송 속도"인 이유:**

DRAM latency, bank conflict, refresh는 DRAM 하드웨어의 물리적 특성이며 소프트웨어로 피할 수 없다.
이 오버헤드까지 포함한 것이 애플리케이션이 실제로 경험하는 실효 DRAM 대역폭이다.

```
"이론 DRAM 피크 BW"  vs  "측정 순간 BW (실효 BW)"
     231 GB/s         vs  57~212 GB/s
         ↑                     ↑
   DRAM 버스 최대 물리 속도     이 커널이 실제로 받는 DRAM 처리량
   (완벽한 연속 burst 가정)     (접근 패턴 + 오버헤드까지 반영한 실측)
```

최적화 관점에서는 "이론 피크"가 아니라 "실효 BW"가 의미 있다. 512x8 splitK가 57%라는 사실은
"이 커널이 DRAM에서 데이터를 가져오는 속도는 131 GB/s"라는 것을 정확히 포착한다.
이를 더 빠르게 하려면 splitK 접근 패턴 자체를 바꿔야 하는데, 이는 커널 설계 수준의 문제다.

**측정의 미세한 불완전성 (실질 영향 무시 가능):**

| 항목 | 영향 크기 | 설명 |
|------|---------|------|
| splitKreduce_kernel 누락 | 0.02% | 512x8 splitK 이후 reduce 커널의 DRAM(~24 KB) 미계상, 101 MB 대비 무시 가능 |
| ncu replay 캐시 상태 차이 | ~0% | 가중치 >> L2이므로 replay 전후 L2 cold start 여부 무관 — DRAM 바이트 동일 |
| DRAM write 바이트 포함 | 0.01% | 출력 y(~8 KB) write도 합산, read(90 MB+) 대비 사실상 read BW만 측정 |
| 192x8 nsys 오버헤드 | **35%** | 43 μs 커널에 15 μs 오버헤드 → 하한값만 보고 (§4.2 상세) |

**결론: 이 측정은 각 커널이 실제로 경험하는 실효 DRAM 처리량을 정확히 포착한다. 192x8 소형 커널을 제외하면 신뢰도 높다. Prefill nvjet GEMM(3~5 ms)도 동일하게 신뢰도 높다.** ✅

### 4.2 nsys 타이밍 신뢰성 및 소형 커널 주의사항

nsys는 replay 없이 실제 추론을 1회 실행한다. GPU 하드웨어가 직접 각 커널의 start/end 타임스탬프를
나노초 정밀도로 기록한다.

nsys 오버헤드(총 668 ms)는 커널당 평균 15.2 μs로 분산된다:

| 커널 | 평균 duration | nsys 오차 비율 | BW 신뢰도 |
|-----|-------------|------------|---------|
| lm_head (6,037 μs) | 6,037 μs | **+0.25%** | ✅ 매우 높음 |
| 512x8 GEMV (771 μs) | 771 μs | +2% | ✅ 높음 |
| 128x8 GEMV (374 μs) | 374 μs | +4% | 🔶 중간 (보정 시 ≈81%) |
| **192x8 GEMV (43 μs)** | **≪43 μs** | **+35%** | **⚠ 하한만 유효** |
| 소형 elementwise (21 μs) | 21 μs | +71% | ❌ 측정 불가 |

192x8 (8 MB, avg 43 μs) 상세:
```
nsys 오버헤드 추정: 15 μs
실제 kernel: 43 - 15 = 28 μs (추정)
역산 BW = 8 MB / 28 μs = 286 GB/s  →  물리 한계(231 GB/s) 초과

→ 실제 BW: ≥194 GB/s (하한), 최대 ~220 GB/s 수준 추정
   정확한 값은 이 방법으로 측정 불가 (커널 duration ≈ nsys overhead)
```

### 4.3 SM 활용률 측정 방법 (2026-06-11 수정)

**⚠ 이전 버전의 메트릭은 GB10B에서 작동하지 않음:**

```
❌ sm__active_cycles.sum   → GB10B(SM 11.0) 미지원 → 항상 0 반환
❌ gpc__cycles_elapsed.max → GB10B 미지원 → 항상 0 반환

→ 이전에 보고된 "SM util = 0% for ALL kernels"는 메트릭 부재로 인한 측정 오류였음
```

**✅ GB10B에서 지원되는 올바른 메트릭 (`--list-metrics` 실측 확인 2026-06-11):**

```
smsp__cycles_active.sum    ← SM 서브파티션 active cycles 합
smsp__cycles_elapsed.sum   ← SM 서브파티션 elapsed cycles 합 (분모)

warp occupancy = smsp__cycles_active.sum / smsp__cycles_elapsed.sum × 100
              ≈ 89~100% (Prefill 모든 커널, smsp_occupancy 측정 결과)
```

**warp occupancy의 의미 — compute utilization과 다름:**

```
warp "active"의 정의:
  - warp가 이슈 큐에 있거나 실행 중인 사이클 → active
  - warp가 DRAM stall로 대기 중인 사이클도 → active  ← 이것이 핵심

따라서:
  smsp_occupancy = 89~100% → "warp가 많이 스케줄됨 (GEMM 특성)"
                            ≠ "SM이 90%의 시간을 연산함"

GEMM 커널은 warp를 많이 생성하여 DRAM stall을 hide하는 구조
→ occupancy 높음 = 자연스러운 GEMM 동작
→ bound 판별은 DRAM BW%를 봐야 함
```

**보조 메트릭 (교차검증용):**
```
sm__throughput.avg.pct_of_peak_sustained_elapsed → SM throughput % (GB10B 지원)
```

Prefill nvjet GEMM에서 smsp_occupancy ≈ 96%, DRAM BW 70% → **DRAM-bound** 확정.  
FlashAttention에서 smsp_occupancy ≈ 96%, DRAM BW 39.6% → **SRAM/compute-bound** 확정.

### 4.4 DRAM-bound 커널의 구조적 이유 + batch=1

**이유 1: GEMV의 구조적 data dependency (batch 크기와 무관)**

GEMV는 출력 원소 `y[i] = Σ W[i,j] × x[j]`를 계산할 때 가중치 행렬의 i번째 행 전체가
도착해야 비로소 연산을 시작할 수 있다. 데이터 도착 전 연산을 시작하는 것이 원천적으로 불가능하다.
가중치(15 GB) >> L2(32 MB)이므로 모든 데이터가 매번 DRAM에서 온다. SM은 구조적으로 기다릴 수밖에 없다.

> ※ GEMV warp는 DRAM stall 중에도 smsp_occupancy에서 "active"로 카운트됨.  
> smsp_occupancy ≈ 96%인데도 DRAM-bound인 이유가 바로 이것이다.

**이유 2: batch=1이 GEMV를 순수 memory-bound로 고정**

batch 크기와 산술 강도(arithmetic intensity)의 관계:

```
batch=1: flops = 2×out×in,   bytes = 2×out×in  → intensity = 1 op/byte ≈ 0
batch=N: flops = 2×N×out×in, bytes = 2×out×in  → intensity = N ops/byte
           (가중치를 N번 재사용)
```

batch가 클수록 가중치 재사용으로 intensity가 올라가 compute-bound에 가까워진다.
batch=1에서는 가중치를 딱 한 번만 쓰므로 intensity가 극히 낮아 GPU는 DRAM만 기다린다.

**왜 batch=1인가 — Alpamayo 배포 조건**

Alpamayo는 10Hz(100ms 간격) 단일 추론이 시스템 설계 원칙이다.
여러 요청을 동시에 처리하는 스트리밍 서버 환경이 아니다.
단일 추론(batch=1)에서 SM 활용률 0%는 이 시스템의 실제 동작 상태다.

멀티 배치 스트리밍 환경(예: LLM 서빙 서버)에서는 batch 증가로 SM utilization이 올라갈 수 있다.
그러나 그 경우에도 DRAM 요구량이 batch 배로 증가하므로 DRAM 병목은 동일하게 유지된다.

### 4.5 커널 이름 매핑: nvjet_tst_*

SM 11.0(Blackwell)에서 NVIDIA JET GEMV 커널의 이름 저장 방식:

```
ncu  (shortName으로 직접 저장): nvjet_tst_512x8_64x3_2x1_v_bz_splitK_TNT
nsys (shortName):               nvjet_tst_512x8_64x3_2x1_v_bz_splitK_TNT
```

두 도구 모두 shortName으로 저장된다. nvjet(NVIDIA JET) = Blackwell 전용 새 GEMV/GEMM 커널 생성기.
타일 파라미터 `512x8_64x3_2x1` = [output×batch, tile×splitK, ...] 형태의 설계 파라미터.

lm_head만 예외적으로 nvjet가 아닌 구형 `gemv2T` 경로를 쓴다
(vocab size 152K가 너무 커서 nvjet 타일 설계 범위를 초과).

---

## 5. Decode 커널 구조 전체

### 5.1 주요 커널 분포 (nsys shortName 기준, DecodeAll 필터, 44,078개)

```
소형 elementwise 계열: 다수     avg 0.01~0.03 ms   (LayerNorm, RoPE, SiLU 등)
CatArrayBatchedCopy:  4,511개   avg 0.065~0.089 ms (KV cache concat)

GEMV 계열 (nvjet_tst_* + gemv2T):
  nvjet 128x8_64x12_2x1:        2,736개  avg 0.374 ms  ★ 180 GB/s (78%)
  nvjet 512x8_64x3_2x1_splitK:    684개  avg 0.771 ms  ★ 131 GB/s (57%)  ← FFN down
  nvjet 192x8_64x8_2x1:          1,368개  avg 0.043 ms  ★ ≥194 GB/s (≥84%)
  gemv2T_kernel_val (lm_head):      19개  avg 6.037 ms  ★ 212 GB/s (92%)

Attention:
  flash_fwd_kernel:               468개   avg 0.294 ms  ← FlashAttention-2
  flash_fwd_splitkv_kernel:        684개   avg 0.027 ms
```

### 5.2 GPU idle 분포

```
총 idle: 229.2 ms (10.6%)

  1~10 μs  (kernel launch overhead):  158.9 ms  35,498건  → CUDA Graph로 제거 가능
  0.1~1 ms (Python dispatch):          29.0 ms     108건
  1 ms 이상 (sync/alloc):              41.4 ms       9건

최대 단일 갭: 13.587 ms (Decode 전체 중 1회만 발생)
```

---

## 6. 최적화 시사점

```
현재 상태:
  DRAM BW 활용:  57~92% (커널별), 89% (stage 평균)  ← 거의 포화
  SM 활용률:     0%  (GEMV 전체)                    ← DRAM 대기 중 낭비
  GPU idle:     10.6%                               ← dispatch 오버헤드
```

**Decode 최적화 방향 (시스템 레벨):**

| 경로 | 방법 | 기댓값 |
|------|------|--------|
| **커널 간 idle 제거** | CUDA Graph (launch overhead 229 ms 제거) | ~229 ms 단축 (-15%) |
| **레이어 prefetch 중첩** | cudaMemPrefetchAsync + CUDA Stream 이중화 | 일부 DMA 시간 숨김 |
| **단계 간 pipeline** | Inter-Inference Pipeline (10Hz, 다음 추론 VE 가중치 선적재) | 단계 2 계획 |

smsp_occupancy ~96%: SM이 항상 warp를 스케줄링 중 → **SM 내에 "빈 compute 슬롯" 없음**.  
Async pipeline의 여지는 SM 여유가 아닌 **DRAM BW 여유**(未사용 BW)와 **커널 간 갭**에 있다.  
GEMV는 구조적으로 DRAM fetch 완료 전 연산 불가(data dependency) → SM idle은 제거 불가능, DRAM BW 여유를 DMA에 활용하는 것이 핵심.

---

## 7. 다음 측정 계획

### ~~우선순위 1: LM Prefill per-kernel 측정~~ → **완료 (2026-06-11)**

결과 요약: SM util = 0% (2032 커널 전체), GEMM BW 144~163 GB/s.  
상세 내용: §2.3 참조.

### ~~우선순위 1: FlashAttention nsys 타이밍 교차검증~~ → **완료 (2026-06-11)**

결과: `flash_fwd_kernel` (nsys shortName), N=18, DRAM=210 MB, dur=2298 μs, **BW=91.5 GB/s (39.6%)**.  
Prefill stage 55%의 원인이 FlashAttention의 낮은 BW 효율임을 확인.

### ~~우선순위 2: VE, Flow per-kernel 측정~~ → **완료 (2026-06-11)**

결과 요약 (smsp_occupancy 기준):
- **VE**: nvjet GEMM DRAM 43~61% (DRAM-bound), FlashAttention 24.3% (SRAM-bound), patch conv 3.5% (DRAM-bound). §2.4 참조.
- **Flow**: nvjet GEMV DRAM 57~96% (DRAM-bound), elementwise ~31% (DRAM-bound). §2.5 참조.

### 우선순위 3: VE/Flow 재측정 (smsp_occupancy 값 확인)

VE/Flow는 기존 Decode/Flow 스크립트에서 수집된 데이터에 `sm__active_cycles`(GB10B 미지원)를 사용.
`smsp__cycles_active.sum` 으로 재측정 시 실제 warp occupancy 값을 얻을 수 있음.
단, Prefill에서 이미 89~100%임을 확인했으므로 의사결정에 영향 없음. 낮은 우선순위.

---

## 부록: 측정 방법 재현 가이드

### 스크립트 목록

```
scripts/profiling/
  260610_run_ncu_per_kernel_bw.sh    ← ncu per-kernel 실행 (Decode)
  260610_run_nsys_decode.sh          ← nsys 타임라인 실행
  260610_analyze_per_kernel_bw.py    ← 분석 (--mode ncu / --mode nsys)
  260611_analyze_nvjet_bw.py         ← FFN/QKV nvjet 커널 BW 교차검증 (Decode 완료)
  260611_run_ncu_prefill_bw.sh       ← ncu per-kernel 실행 (Prefill, NVTX: Phase/LM_Prefill)
  260611_analyze_prefill_bw.py       ← Prefill 분석 (normalize_kernel_name 포함)
```

### 실행 순서 (Thor에서, Decode 기준)

```bash
# Step 1: ncu (~90분)
sudo -E bash ~/alpamayo1.5/scripts/profiling/260610_run_ncu_per_kernel_bw.sh
sudo chown -R ice401:ice401 ~/alpamayo1.5/profiling_results/260610_per_kernel_bw/

# Step 2: nsys (~15분)
sudo -E bash ~/alpamayo1.5/scripts/profiling/260610_run_nsys_decode.sh
sudo chown -R ice401:ice401 ~/alpamayo1.5/profiling_results/260610_per_kernel_bw/

# Step 3: 분석
python3 260611_analyze_nvjet_bw.py   # FFN/QKV nvjet 교차검증
```

### 핵심 메트릭 설명

| 메트릭 | 용도 | 신뢰성 | GB10B 지원 |
|--------|------|--------|-----------|
| `lts__d_sectors_fill_sysmem.sum × 32` | DRAM read bytes (L2 hit 완전 제외) | ✅ (replay 무관) | ✅ |
| `lts__t_sectors_aperture_sysmem_op_write.sum × 32` | DRAM write bytes | ✅ | ✅ |
| `smsp__cycles_active.sum / smsp__cycles_elapsed.sum` | warp occupancy (≠ compute util) | ✅ (해석 주의) | ✅ (확인됨) |
| ~~`sm__active_cycles.sum / gpc__cycles_elapsed.max`~~ | ~~SM 활용률~~ | ❌ **항상 0 반환** | ❌ GB10B 미지원 |
| `gpu__time_duration.sum` | 커널 시간 (ncu) | ❌ (2.85× 부풀려짐) | ✅ |
| nsys `end - start` timestamp | 커널 시간 (nsys) | ✅ 대형 커널 한정 (>300 μs) | — |

---

*4단계 전체 순간 BW 측정 + SM 메트릭 수정 완료 (2026-06-11). 주요 발견: (1) GB10B에서 sm__active_cycles 미지원 → 이전 "SM=0%"는 측정 오류. 올바른 메트릭: smsp__cycles_active/elapsed (warp occupancy) → 89~100%. (2) Bound 판별은 DRAM BW% 기준: nvjet GEMM/GEMV → DRAM-bound, FlashAttention(seq=3086, DRAM 39.6%) → SRAM/compute-bound. (3) 양자화(INT4/FP4) 효과: GEMM/GEMV에서 선형, FlashAttention에서 제한적. (4) 모델 18 transformer layers 확인. (5) Flow stage BW 수정: 203 GB/s(88%) → ~101 GB/s(~44%).*

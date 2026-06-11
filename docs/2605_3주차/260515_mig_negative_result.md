# Thor Blackwell iGPU — MIG GI 파티셔닝: JetPack 7.2에서 지원 예정

**실험 일시**: 2026-05-15  
**플랫폼**: Jetson AGX Thor (SM 11.0, LPDDR5X 128 GB, JetPack 7)  
**드라이버**: 580.00 / CUDA 13.0

---

## 실험 결과

### 시도한 명령어와 결과

| 명령어 | 결과 |
|---|---|
| `sudo nvidia-smi -mig 1` | ✅ `Enabled MIG Mode for GPU 00000000:01:00.0` |
| `nvidia-smi --query-gpu=mig.mode.current` | ✅ `Enabled` |
| `sudo nvidia-smi -pm 1` | ✅ Persistence mode 활성화 |
| `sudo nvidia-smi mig -lgip` | ❌ `Failed to display GPU instance profiles: Unknown Error` (RC=255) |
| `sudo nvidia-smi mig -lcip` | ❌ `No MIG-enabled devices found` |
| `sudo nvidia-smi mig -lgi` | ❌ `No MIG-enabled devices found` |

### 실제 출력

```
+-------------------------------------------------------------------------------+
| GPU instance profiles:                                                        |
| GPU   Name               ID    Instances   Memory     P2P    SM    DEC   ENC  |
|                                Free/Total   GiB              CE    JPEG  OFA  |
|===============================================================================|
Failed to display GPU instance profiles: Unknown Error
```

GI 프로파일 테이블이 완전히 비어 있음. 헤더 행만 출력되고 실제 프로파일 없음.

---

## 원인 분석 (2026-05-15 조사 결과)

### 결론: 하드웨어 지원 O, 소프트웨어(JetPack 7.1) 미구현

Thor 하드웨어는 MIG를 지원한다 (10 TPC, MIG 회로 탑재).  
문제는 **JetPack 7.0/7.1까지 NVML MIG 소프트웨어 스택이 미구현**이다.

NVIDIA 공식 포럼 스태프 발언 (1차 소스):

| 출처 | 발언 |
|---|---|
| kayccc (NVIDIA, thread 344978) | "It is not supported on JetPack 7.0GA. We are planning it." |
| AastaLLL (NVIDIA, thread 359667) | "The MIG is not yet supported." |
| kayccc (NVIDIA, thread 367194) | **"We plan to support MIG at JetPack 7.2, released before June."** |

→ **JetPack 7.2 (2026년 6월 이전 출시 예정)에서 최초 지원**.  
→ 현재 JetPack 7.1 기준으로는 lgip 명령이 Unknown Error를 반환하는 것이 **정상적 동작**.

### MIG 지원의 두 가지 레이어

```
레이어 1: MIG 모드 플래그 (on/off)
  → nvidia-smi -mig 1/0 으로 제어
  → JetPack 7.1에서 작동함 ✅

레이어 2: GI(GPU Instance) 파티셔닝
  → 실제로 GPU를 독립 슬라이스로 분할
  → nvidia-smi mig -cgi, -lgip 으로 제어
  → JetPack 7.1에서 미구현 ❌ → JetPack 7.2에서 구현 예정 ✅
```

### 실패 메커니즘 상세

```
nvidia-smi mig -lgip 실패 경로:
  nvidia-smi
    → NVML API: nvmlDeviceGetGpuInstanceProfiles()
    → RM 드라이버
    → DCE(Display Controller Engine) RPC 레이어  ← 여기서 차단
    → NVRM: rpcRmApiControl_dce: Failed RM ctrl call
             cmd:0x731341 result 0xffff (Generic Error)
```

Thor iGPU는 GPU가 SoC 패브릭에 내장되어 있어  
DCE RPC 레이어를 경유하는데, 이 경로에 MIG 파티션 명령이  
JetPack 7.1까지 구현되어 있지 않음.

### Fabric Manager는 관계없음

`nvidia-fabricmanager`는 NVLink 다중 GPU 시스템(DGX, HGX)용 서비스.  
Thor는 단일 iGPU → Fabric Manager 불필요, 이 문제와 무관.

### 재부팅은 해결책이 아님

MIG 모드 플래그는 재부팅 후 유지되지만  
GI 파티셔닝 API 미구현 자체는 재부팅으로 해결되지 않음.

### GPU 종류별 MIG 지원 상태

| GPU | MIG 모드 플래그 | GI 파티셔닝 |
|---|---|---|
| A100, H100, B200 (데이터센터) | ✅ | ✅ |
| RTX PRO 6000 Blackwell | ✅ | ✅ |
| RTX 4090 (소비자) | ❌ | ❌ |
| Jetson AGX Orin (Ampere) | ❌ | ❌ |
| **Jetson AGX Thor (JetPack 7.1)** | **✅ 플래그** | **❌ 미구현** |
| **Jetson AGX Thor (JetPack 7.2 예정)** | **✅** | **✅ 예정** |

### Thor MIG와 A100 MIG의 차이점 (예상)

A100은 HBM 전용 메모리 분할 → 프로파일에 GB 수 명시 (예: `3g.20gb`)  
Thor는 CPU/GPU 공유 LPDDR5X → 메모리 파티셔닝 방식이 다름  
→ JetPack 7.2에서도 프로파일이 `Xg.0gb` 형태로 나올 수 있음 (메모리 accounting 방식 상이)

### "MIG 지원"이라고 했던 EXP-0 결과 재해석

EXP-0에서 `has_gi_profiles: True`로 판정된 이유:
```python
# EXP-0 판정 코드
gi_result = ...  # nvidia-smi mig -lgip
has_gi_profiles = gi_result["rc"] == 0 and len(gi_result["stdout"]) > 10
# rc=0 이고 stdout이 10자 이상 → True 로 판정됨
```

그러나 실제로는 rc=0이 아니라 **RC=255** (오류).  
`nvidia-smi mig -lgip` 명령이 오류 시에도 테이블 헤더를 stdout에 출력하고  
실제 오류 메시지 "Failed to..."도 **stdout**으로 출력 (stderr가 아님).

→ EXP-0 스크립트의 판정 로직 버그:
  - RC 체크가 잘못됨 (RC=255인데 rc==0 판정)
  - "Unknown Error" 텍스트 확인 누락

---

## 수정된 EXP-0 판정 로직

```python
# 수정 전 (틀림)
has_gi_profiles = gi_result["rc"] == 0 and len(gi_result["stdout"]) > 10

# 수정 후 (올바름)
gi_stdout = gi_result.get("stdout", "")
gi_rc = gi_result.get("rc", 1)
has_gi_profiles = (
    gi_rc == 0
    and len(gi_stdout) > 10
    and "Unknown Error" not in gi_stdout
    and "Failed to" not in gi_stdout
)
```

---

## 실험 계획 수정

### 현재 (JetPack 7.1) 가능한 것

| 실험 | 상태 | 방법 |
|---|---|---|
| EXP-1 (MIG 슬라이스 스케일링) | ⏳ JetPack 7.2 대기 | MIG GI 생성 필요 |
| EXP-2 (MIG 인스턴스 배정) | ⏳ JetPack 7.2 대기 | MIG GI 생성 필요 |
| **EXP-3 (크로스프레임 파이프라인)** | **✅ 지금 가능** | MIG 불필요 |
| EXP-4 (GPU 용량 스케일링) | ✅ 지금 가능 | MIG 불필요 |
| CUDA MPS 격리 (대체) | ✅ 지금 가능 | MIG 없이 SM% 제한 |

### JetPack 7.2 출시 시 즉시 진행할 실험 순서

```bash
# 1. JetPack 7.2 설치 후
sudo nvidia-smi -mig 1
sudo reboot

# 2. 재부팅 후 GI 프로파일 확인
sudo nvidia-smi mig -lgip
# → 이제 프로파일 목록이 나와야 함

# 3. Vision=작은 슬라이스, VLM=큰 슬라이스, Action=작은 슬라이스 생성
sudo nvidia-smi mig -cgi <small_id>,<large_id>,<small_id> -C

# 4. EXP-2 실제 측정 실행
python3 ~/alpamayo1.5/scripts/profiling/260515_exp2_mig_real_measure.py
```

### 현재 대안: CUDA MPS (SM 점유율 제한)

```bash
# MPS 활성화
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# SM 점유율 제한으로 MIG 슬라이스 효과 근사
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=15 python3 run_vision.py &   # ~3 SM
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70 python3 run_vlm.py &      # ~14 SM
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=15 python3 run_action.py &   # ~3 SM
```

MPS는 메모리 격리가 없어 실제 MIG와 다르지만,  
SM 할당 비율에 따른 성능 변화는 측정 가능.

---

## 논문에서의 활용

이 부정적 결과는 그 자체로 기여다:

> "We attempted to evaluate MIG-based module partitioning on the Jetson AGX Thor.
>  While MIG mode can be enabled via `nvidia-smi -mig 1`, the actual GPU instance
>  partitioning (GI profile creation) is not supported on the Thor Blackwell iGPU
>  as of JetPack 7 (driver 580.00). `nvidia-smi mig -lgip` returns an Unknown Error
>  with no profiles listed, despite MIG mode reporting 'Enabled'.
>  This distinguishes the Thor iGPU from data-center Blackwell GPUs where full
>  MIG partitioning is available."

**함의**: 향후 Jetson에서 MIG를 계획하는 연구자들에게 실용적 경고.  
NVIDIA의 공식 문서에 이 제약이 명시되어 있지 않아 직접 확인이 필요했음.

---

## 다음 단계

```
1. sudo nvidia-smi -mig 0    ← MIG 비활성화 (필수)
2. EXP-3 크로스프레임 파이프라인 실행
3. EXP-4 GPU 용량 스케일링 실험
4. (선택) CUDA MPS 기반 SM 점유율 제한 실험
```

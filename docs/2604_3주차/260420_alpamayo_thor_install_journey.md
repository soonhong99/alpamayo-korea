# Alpamayo 1.5를 Jetson AGX Thor에 설치하기까지 — 완전한 기록

> 작성일: 2026-04-19~20  
> 환경: Jetson AGX Thor (aarch64, JetPack R38.4.0, CUDA 13.0, SM 11.0)  
> 목표: NVIDIA Alpamayo 1.5 (10B VLA 자율주행 모델)를 Thor 엣지 보드에서 CUDA 추론으로 실행하기

---

## 들어가며

이 문서는 NVIDIA의 Alpamayo 1.5 모델을 Jetson AGX Thor 보드에 설치하는 과정에서 마주친 모든 문제와 해결책을 기록한 것이다. 단순한 `pip install` 한 줄로 끝날 줄 알았던 작업이, CUDA 컴파일러 내부 구조를 뜯어고치는 수준의 작업으로 이어졌다. 그 전 과정을 하나도 빠짐없이 기록한다.

**왜 Thor인가?**  
Alpamayo 1.5는 10B(100억) 파라미터짜리 모델이다. 이걸 실시간 자율주행에 쓰려면 ≤100ms 추론 지연이 필요하다. Jetson AGX Thor는 128GB 통합 메모리와 2,070 FP4 TFLOPS를 제공하는 유일한 엣지 플랫폼으로, 10B 모델을 엣지에서 실시간으로 돌릴 수 있는 몇 안 되는 선택지다.

---

## 목차

1. [환경 설명](#1-환경-설명)
2. [문제 0: 애초에 왜 이렇게 어려운가?](#2-문제-0-애초에-왜-이렇게-어려운가)
3. [문제 1: AlpaSim이 Thor에서 설치 불가](#3-문제-1-alpasim이-thor에서-설치-불가)
4. [문제 2: PyTorch aarch64 CUDA wheel 없음](#4-문제-2-pytorch-aarch64-cuda-wheel-없음)
5. [문제 3: NGC L4T 컨테이너도 미출시](#5-문제-3-ngc-l4t-컨테이너도-미출시)
6. [결정: 소스에서 직접 빌드한다](#6-결정-소스에서-직접-빌드한다)
7. [문제 4: PEP 668 — 시스템 pip 차단](#7-문제-4-pep-668--시스템-pip-차단)
8. [문제 5: CUDA 13.0이 부순 것들 — CCCL 통합](#8-문제-5-cuda-130이-부순-것들--cccl-통합)
9. [패치 1: CMake가 CUB를 못 찾음](#9-패치-1-cmake가-cub를-못-찾음)
10. [패치 2: cudaDeviceProp::computeMode 제거](#10-패치-2-cudadevicepropcomputemode-제거)
11. [패치 3: cuFFT 에러코드 제거](#11-패치-3-cufft-에러코드-제거)
12. [패치 4~10: CUB Iterator 전면 교체](#12-패치-410-cub-iterator-전면-교체)
13. [빌드 성공 & 검증](#13-빌드-성공--검증)
14. [문제 6: CPU torch가 CUDA torch를 덮어씀](#14-문제-6-cpu-torch가-cuda-torch를-덮어씀)
15. [현재 상태 및 다음 단계](#15-현재-상태-및-다음-단계)

---

## 1. 환경 설명

### 하드웨어

| 항목 | 사양 |
|---|---|
| 보드 | NVIDIA Jetson AGX Thor |
| CPU | Cortex-A78AE (aarch64) |
| GPU | Blackwell GPU (SM 11.0) |
| 메모리 | 128GB LPDDR5X (CPU+GPU 통합) |
| FP4 연산 | 2,070 TFLOPS |
| 메모리 대역폭 | 900 GB/s |

### 소프트웨어

| 항목 | 버전 |
|---|---|
| JetPack | R38.4.0 |
| OS | Ubuntu 24.04 LTS |
| CUDA | 13.0 |
| Python | 3.12.13 |
| PyTorch (목표) | 2.8.0 |

### 용어 정리

- **aarch64**: ARM 64-bit 아키텍처. 스마트폰, Jetson 등 대부분의 임베디드 장치가 쓰는 CPU 명령어 집합. x86_64(인텔/AMD PC)와 바이너리 호환이 안 된다.
- **SM (Streaming Multiprocessor)**: NVIDIA GPU의 연산 단위. SM 버전이 GPU 세대를 나타낸다. SM 8.0 = Ampere(A100), SM 9.0 = Hopper(H100), SM 11.0 = Blackwell(Thor).
- **JetPack**: Jetson 보드용 NVIDIA 소프트웨어 패키지. OS + CUDA + cuDNN + TensorRT 등을 한 번에 설치해준다.
- **VLA (Vision-Language-Action) 모델**: 카메라 영상을 보고 언어로 상황을 설명하면서 동시에 행동(주행 경로)을 출력하는 대형 AI 모델.

---

## 2. 문제 0: 애초에 왜 이렇게 어려운가?

시작하기 전에 왜 이 작업이 어려운지 큰 그림을 이해해야 한다.

### "바이너리 삼각형" 문제

PyTorch를 설치하려면 세 가지 조건이 맞아야 한다:

```
CPU 아키텍처 (aarch64)
        × 
CUDA 버전 (13.0)
        × 
Python 버전 (3.12)
```

이 세 조건이 **모두 맞는 pre-built wheel**이 존재해야 `pip install torch`가 된다. 그런데:

- PyPI: `torch 2.8.0+cpu` (aarch64용 CPU 버전만 있음)
- download.pytorch.org/whl/cu130: aarch64 없음
- NVIDIA JP7 공식 서버: **미출시** (2026년 4월 기준)

Thor는 출시된 지 얼마 안 된 최신 보드다. NVIDIA가 JetPack 7용 PyTorch wheel을 아직 만들지 않은 것이 근본 원인이다. **결론: 소스 빌드 외에 방법이 없다.**

---

## 3. 문제 1: AlpaSim이 Thor에서 설치 불가

### 시도한 것

AlpaSim은 Alpamayo를 훈련/평가하는 NVIDIA의 시뮬레이터다.

```bash
cd ~/alpamayo-korea/alpasim
uv sync
```

### 에러

```
pyqt5-qt5: no matching distribution found for linux_aarch64
tensordict: no matching distribution found for linux_aarch64
```

### 원인

AlpaSim은 NVlabs가 **x86_64 Linux 전용**으로 개발한 시뮬레이터다. 핵심 GUI 라이브러리(`pyqt5-qt5`)와 텐서 연산 라이브러리(`tensordict`)가 aarch64 wheel을 배포하지 않는다.

### 결론

AlpaSim은 Thor에서 실행 불가다. 이는 구조적 문제로 우회가 불가능하다. AlpaSim이 필요한 **훈련/평가는 Cloud GPU(A100)에서**, Thor는 **완성된 모델의 추론(inference)만 담당**하는 역할 분리가 필요하다.

---

## 4. 문제 2: PyTorch aarch64 CUDA wheel 없음

### 시도한 것 (모두 실패)

```bash
# 1) 표준 PyPI
pip install torch==2.8.0
# → torch 2.8.0+cpu 설치됨 (CUDA 없음)

# 2) PyTorch 공식 CUDA 인덱스
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu130
# → No matching distribution found

# 3) NVIDIA JetPack 6.1 서버
pip install torch --index-url \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/
# → No matching distribution found

# 4) NVIDIA JetPack 7 서버
pip install torch --index-url \
  https://developer.download.nvidia.com/compute/redist/jp/v70/pytorch/
# → No matching distribution found
```

### 확인

```python
import torch
print(torch.__version__)        # 2.8.0+cpu
print(torch.cuda.is_available())  # False
```

### 결론

JetPack 7(CUDA 13.0)용 PyTorch wheel이 존재하지 않는다.

---

## 5. 문제 3: NGC L4T 컨테이너도 미출시

### 시도한 것

Docker 컨테이너를 쓰면 사전 빌드된 환경을 가져올 수 있다. NVIDIA가 Jetson용 `l4t-pytorch` 컨테이너를 제공한다.

```bash
docker pull nvcr.io/nvidia/l4t-pytorch:r38.1.0-pth2.8-py3
# → manifest unknown

docker pull dustynv/pytorch:2.8-r38.1.0
# → manifest unknown
```

### 결론

커뮤니티 빌드(dusty-nv)를 포함해 JetPack R38 기반 컨테이너가 전혀 없다. Thor가 너무 신형이라 생태계가 따라오지 못한 상황이다.

---

## 6. 결정: 소스에서 직접 빌드한다

세 가지 방법을 모두 시도했지만 모두 막혔다. 남은 선택지는:

**PyTorch 2.8.0을 소스에서 직접 컴파일한다.**

이 방법은:
- 시간이 오래 걸리지만 (Thor에서 약 6시간)
- 확실하게 동작하고
- SM 11.0에 최적화된 바이너리가 만들어진다

### 소스 빌드 기본 설정

```bash
git clone --recursive https://github.com/pytorch/pytorch ~/pytorch
cd ~/pytorch
git checkout v2.8.0
git submodule sync && git submodule update --init --recursive

# 핵심 환경변수
export TORCH_CUDA_ARCH_LIST="11.0"   # Thor GPU 아키텍처
export MAX_JOBS=8                      # 병렬 컴파일 수
export USE_TENSORPIPE=0               # 나중에 설명
export USE_DISTRIBUTED=0
export USE_MPI=0
```

**`TORCH_CUDA_ARCH_LIST="11.0"` 이 왜 중요한가?**  
이 변수는 "이 GPU 아키텍처 전용 코드만 컴파일하라"는 뜻이다. 지정하지 않으면 PyTorch가 SM 6.0부터 9.0까지 모든 버전을 컴파일해서 빌드 시간이 수십 시간이 된다. SM 11.0만 지정하면 Thor에서만 동작하지만 빌드가 훨씬 빠르다.

---

## 7. 문제 4: PEP 668 — 시스템 pip 차단

### 에러

```bash
pip3 install cmake ninja
# → error: externally-managed-environment
```

### 원인

Ubuntu 24.04는 **PEP 668** 규약을 따른다. 시스템 Python의 패키지를 pip으로 직접 설치하면 OS 패키지 관리자와 충돌할 수 있어서 Ubuntu가 차단한다.

### 해결

venv(가상환경) 안에서 설치한다.

```bash
python3 -m venv ~/alpamayo1.5/a1_5_venv
source ~/alpamayo1.5/a1_5_venv/bin/activate
# (a1_5_venv) 프롬프트로 바뀜

# venv 내부에서는 pip 자유롭게 사용 가능
pip install cmake ninja pyyaml typing_extensions
```

**venv란?**  
프로젝트별 독립된 Python 환경. 시스템 Python에 영향을 주지 않고 패키지를 설치할 수 있다. `source activate`로 진입, `deactivate`로 나온다.

---

## 8. 문제 5: CUDA 13.0이 부순 것들 — CCCL 통합

이것이 이번 작업의 핵심이자 가장 어려운 부분이다.

### 배경: CCCL이란 무엇인가?

NVIDIA CUDA 생태계에는 GPU 연산을 위한 세 개의 C++ 라이브러리가 있었다:

| 라이브러리 | 역할 |
|---|---|
| **CUB** | GPU 블록/워프 수준 알고리즘 (정렬, 스캔, 리덕션) |
| **Thrust** | GPU용 STL (벡터, 알고리즘, 이터레이터) |
| **libcu++** | CUDA용 C++ 표준 라이브러리 |

**CUDA 13.0**에서 NVIDIA는 이 세 라이브러리를 **CCCL(CUDA C++ Core Libraries)** 로 통합했다. 통합 과정에서 CUB의 일부 기능이 Thrust와 중복된다는 이유로 **CUB에서 이터레이터 클래스들을 완전히 제거**했다.

### 제거된 API 목록

| 제거된 API | 역할 | Thrust 대체 |
|---|---|---|
| `cub::TransformInputIterator<O,F,I>` | 이터레이터를 읽을 때 함수를 적용 | `thrust::make_transform_iterator(iter, func)` |
| `cub::CountingInputIterator<T>` | 0, 1, 2, ... 를 생성하는 이터레이터 | `thrust::counting_iterator<T>` |
| `cub::ConstantInputIterator<T>` | 항상 같은 값을 반환하는 이터레이터 | `thrust::make_constant_iterator<T>(val)` |
| `cub::Sum{}` | 덧셈 함수 객체 | `thrust::plus<T>()` |
| `cub::Max{}` | 최댓값 함수 객체 | `::cuda::maximum<>{}` |
| `cub::Equality()` | 동등 비교 함수 객체 | `::cuda::std::equal_to<>{}` |

**PyTorch 2.8.0은 이 제거된 API를 직접 사용한다.** 따라서 CUDA 13.0 환경에서 컴파일하면 에러가 난다.

### 이터레이터란 무엇인가?

이터레이터는 "데이터를 순서대로 접근하는 포인터 같은 것"이다.

예를 들어 `TransformInputIterator`는 이렇게 동작한다:

```cpp
// 원본 배열: [1, 2, 3, 4, 5]
// 변환 함수: x -> x * 2

TransformInputIterator<int, double_fn, int*> iter(data, double_fn{});
// iter를 읽으면 → [2, 4, 6, 8, 10]
// 배열을 직접 변환하지 않고 "읽을 때" 변환이 일어남
```

이걸 GPU 커널에 넘기면 GPU가 데이터를 읽으면서 동시에 변환할 수 있다. 메모리를 추가로 쓰지 않아도 돼서 효율적이다.

---

## 9. 패치 1: CMake가 CUB를 못 찾음

### 에러

```
Could not find CUB (or CUDA Toolkit).
```

### 원인 분석

CMake가 CUB 헤더 파일을 찾는 방법을 살펴봤다:

```bash
find /usr/local/cuda-13.0 -name "cub.cuh" 2>/dev/null
# → /usr/local/cuda-13.0/targets/sbsa-linux/include/cccl/cub/cub.cuh
```

CUDA 13.0은 CUB를 CCCL 안으로 옮기면서 경로가 바뀌었다:
- **기존 경로**: `/usr/local/cuda/include/cub/cub.cuh`
- **새 경로**: `/usr/local/cuda/targets/sbsa-linux/include/cccl/cub/cub.cuh`

(`sbsa` = Server-Base System Architecture, ARM 서버용 표준 ABI)

PyTorch의 CMake 스크립트는 기존 경로만 알고 있어서 새 경로를 못 찾는다.

### 패치: `cmake/Modules/FindCUB.cmake`

```cmake
# 기존 코드
find_path(CUB_INCLUDE_DIR
    HINTS "${CUDA_TOOLKIT_INCLUDE}"
    NAMES cub/cub.cuh
    ...
)

# 패치 후: sbsa 경로를 HINTS에 추가
find_path(CUB_INCLUDE_DIR
    HINTS "${CUDA_TOOLKIT_INCLUDE}"
          "/usr/local/cuda-13.0/targets/sbsa-linux/include/cccl"
          "/usr/local/cuda/targets/sbsa-linux/include/cccl"
    NAMES cub/cub.cuh
    ...
)
```

**왜 HINTS인가?** CMake의 `find_path`는 HINTS에 적힌 경로를 먼저 찾아보고, 없으면 기본 경로를 탐색한다. HINTS에 추가함으로써 기존 동작을 보존하면서 새 경로도 탐색하게 된다.

---

## 10. 패치 2: cudaDeviceProp::computeMode 제거

### 에러 (빌드 초반)

```
error: 'cudaDeviceProp' has no member named 'computeMode'
```

### 원인

`cudaDeviceProp`은 CUDA 장치 속성을 담는 구조체다. `computeMode` 필드가 CUDA 13.0에서 제거됐다. 이 필드를 사용하는 코드가 PyTorch의 `tensorpipe` 모듈에 있었다.

### 해결

`tensorpipe`는 분산 학습을 위한 모듈인데, 단일 GPU 추론에는 필요 없다. 빌드 환경변수로 통째로 비활성화했다:

```bash
export USE_TENSORPIPE=0
export USE_DISTRIBUTED=0
export USE_MPI=0
```

**이렇게 해도 되는 이유?** Thor에서 Alpamayo를 단일 GPU 추론으로만 쓰기 때문에 분산 학습 기능이 전혀 필요 없다.

---

## 11. 패치 3: cuFFT 에러코드 제거

### 에러

```
error: 'CUFFT_INCOMPLETE_PARAMETER_LIST' was not declared in this scope
error: 'CUFFT_PARSE_ERROR' was not declared in this scope
error: 'CUFFT_LICENSE_ERROR' was not declared in this scope
```

### 원인

`aten/src/ATen/native/cuda/CuFFTUtils.h` 파일에는 cuFFT(CUDA 고속 푸리에 변환) 에러를 문자열로 변환하는 switch 문이 있다. CUDA 13.0에서 이 에러코드 3개가 제거됐다.

### 패치: `aten/src/ATen/native/cuda/CuFFTUtils.h`

```cpp
// 제거 전: 존재하지 않는 에러코드 case들
switch (error) {
    case CUFFT_INCOMPLETE_PARAMETER_LIST: // ← CUDA 13.0에서 제거됨
        return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_PARSE_ERROR:               // ← 제거됨
        return "CUFFT_PARSE_ERROR";
    case CUFFT_LICENSE_ERROR:             // ← 제거됨
        return "CUFFT_LICENSE_ERROR";
    ...
}

// 패치 후: 해당 case 3개를 완전히 삭제
switch (error) {
    // 유효한 CUDA 13.0 에러코드만 남김
    case CUFFT_INVALID_PLAN: ...
    case CUFFT_ALLOC_FAILED: ...
    ...
}
```

**왜 단순히 삭제하면 되는가?** 에러코드 case를 삭제해도 나머지 에러 처리는 그대로다. 해당 에러코드가 발생할 수 없으니 case가 없어도 런타임에 문제가 없다.

---

## 12. 패치 4~10: CUB Iterator 전면 교체

이것이 가장 많은 시간이 걸린 부분이다. 에러가 파일마다 하나씩 터지면서 총 7개 파일을 수정했다.

### 패턴 파악

처음엔 파일마다 개별적으로 터지는 것처럼 보였지만, 사실 동일한 패턴이다:

```bash
# 남은 파일 전체 검색
grep -rn "cub::TransformInputIterator\|cub::CountingInputIterator\|cub::ConstantInputIterator" \
  ~/pytorch/aten/src/ --include="*.cu" --include="*.cuh" | grep -v "hipcub"
```

이 명령으로 앞으로 에러 날 파일을 미리 확인하고 일괄 패치했다.

---

### 패치 4: `aten/src/ATen/cuda/cub.cuh` — 핵심 CUB 래퍼

이 파일은 PyTorch가 CUB를 쓰는 모든 코드의 진입점이다.

#### 4-1. TransformInputIterator 교체

```cpp
// 변경 전 (CUDA 13.0에서 컴파일 불가)
NO_ROCM(at_cuda_detail)::cub::TransformInputIterator<
    input_t, decltype(input_iter_transform), ArgIndexInputIterator>(
    ArgIndexInputIterator(input + i), input_iter_transform)

// 변경 후
thrust::make_transform_iterator(
    ArgIndexInputIterator(input + i), input_iter_transform)
```

**`thrust::make_transform_iterator` 설명:**  
`make_transform_iterator(iterator, func)`는 이터레이터와 함수를 받아서 "읽을 때 func를 적용하는 이터레이터"를 만든다. 기존 `TransformInputIterator<OutputType, Func, Iterator>`와 동일한 기능이지만 타입을 명시하지 않아도 된다(자동 추론).

#### 4-2. Equality() 제거와 equal_to 추가

이 부분에서 미묘한 버그가 생겼다.

**배경:** `InclusiveSumByKey` 함수의 시그니처:
```cpp
InclusiveSumByKey(
    d_temp_storage,
    temp_storage_bytes,
    keys,         // 키 배열
    input,        // 입력
    output,       // 출력
    num_items,    // 원소 수
    equality_op,  // 키 비교 함수 (선택적, 기본값: Equality())
    stream        // CUDA 스트림
)
```

기존 코드는 `Equality()`를 명시했는데, 이걸 단순히 제거하면:
```cpp
// 잘못된 제거
CUB_WRAPPER(DeviceScan::InclusiveSumByKey,
    keys, input, output, num_items, at::cuda::getCurrentCUDAStream());
//                                  ↑ 이게 equality_op 위치로 들어감!
```

`getCurrentCUDAStream()`이 반환하는 `CUDAStream` 타입이 `EqualityOpT` 템플릿 파라미터로 추론되면서 컴파일 에러가 났다:

```
error: EqualityOp=c10::cuda::CUDAStream
call of an object of a class type without appropriate operator()
```

**올바른 패치:**
```cpp
// include 추가
#include <cuda/std/functional>

// equality_op 자리에 ::cuda::std::equal_to<>{}를 명시
CUB_WRAPPER(at_cuda_detail::cub::DeviceScan::InclusiveSumByKey,
    keys, input, output, num_items,
    ::cuda::std::equal_to<>{},           // ← 명시적으로 추가
    at::cuda::getCurrentCUDAStream());
```

---

### 패치 5: `aten/src/ATen/cuda/cub.cu` — Sum 제거

```cpp
// 변경 전
using NO_ROCM(at_cuda_detail)::cub::Sum;  // ← 제거된 타입

void inclusive_sum_truncating(...) {
    inclusive_scan(input, output, Sum{}, num_items);
}

// 변경 후: 로컬에 동등한 구조체 직접 정의
template <typename T>
struct SumOp {
    __device__ T operator()(T a, T b) const { return a + b; }
};

void inclusive_sum_truncating(...) {
    inclusive_scan(input, output, SumOp<output_t>{}, num_items);
}
```

**왜 `thrust::plus<T>()`가 아니라 로컬 구조체인가?**  
`inclusive_scan`의 템플릿 파라미터 추론 문제로 `thrust::plus`가 맞지 않는 경우가 있어서 명시적인 로컬 구조체를 사용했다.

---

### 패치 6 & 7: `EmbeddingBag.cu`, `Embedding.cu` — ConstantInputIterator, Max

```cpp
// ConstantInputIterator: 항상 1을 반환하는 이터레이터
// 변경 전
NO_ROCM(at_cuda_detail)ROCM_HIPCUB(::cub)::ConstantInputIterator<index_t>(1)

// 변경 후
thrust::make_constant_iterator<index_t>(1)
```

```cpp
// Max: 두 값 중 최댓값을 반환하는 함수 객체
// 변경 전
NO_ROCM(at_cuda_detail)ROCM_HIPCUB(::cub)::Max()

// 변경 후
::cuda::maximum<>{}
```

**`::cuda::maximum<>{}` 설명:**  
CCCL의 `cuda/functional` 헤더에 있는 표준 함수 객체. `std::max`의 CUDA 버전이다. 빈 `<>` 는 C++17의 "투명 함수 객체"로, 어떤 타입이든 자동으로 처리한다.

추가한 include:
```cpp
#include <thrust/iterator/constant_iterator.h>
#include <cuda/functional>
```

---

### 패치 8: `Nonzero.cu` — TransformInputIterator, CountingInputIterator

`Nonzero.cu`는 텐서에서 0이 아닌 값의 인덱스를 찾는 연산을 구현한다.

```cpp
// 타입 alias (커널 내부)
// 변경 전
using TransformInputIteratorT = ROCM_HIPCUB(at_cuda_detail::cub)::
    TransformInputIterator<int, NonZeroOp<T>, const T*>;

// 변경 후
using TransformInputIteratorT = thrust::transform_iterator<NonZeroOp<T>, const T*>;
```

```cpp
// 변수 선언 (1)
// 변경 전
cub::TransformInputIterator<bool, NonZeroOp<scalar_t>, const scalar_t*> itr(
    self_.const_data_ptr<scalar_t>() + idx * chunk_size,
    NonZeroOp<scalar_t>());

// 변경 후
auto itr = thrust::make_transform_iterator(
    self_.const_data_ptr<scalar_t>() + idx * chunk_size,
    NonZeroOp<scalar_t>());
```

```cpp
// CountingInputIterator
// 변경 전
cub::CountingInputIterator<int64_t> counting_itr(idx * chunk_size);

// 변경 후
thrust::counting_iterator<int64_t> counting_itr(idx * chunk_size);
```

**`auto`를 쓰는 이유:**  
`thrust::make_transform_iterator`의 반환 타입은 `thrust::transform_iterator<NonZeroOp<scalar_t>, const scalar_t*>`인데 이걸 직접 쓰면 너무 길다. C++11의 `auto` 키워드로 타입 추론을 컴파일러에게 맡긴다.

---

### 패치 9: `TensorTopK.cu` — 타입 alias 교체

```cpp
// TopK 연산 내부
// 변경 전
using counting_iter_t = cub::CountingInputIterator<uint32_t, uint32_t>;
using slice_idx_iter_t = cub::TransformInputIterator<
    uint32_t, BlockIdxToKey, counting_iter_t>;
slice_idx_iter_t slice_idx_iter(counting_iter_t(0), BlockIdxToKey(blocks_per_slice));

// 변경 후
using counting_iter_t = thrust::counting_iterator<uint32_t>;
using slice_idx_iter_t = thrust::transform_iterator<BlockIdxToKey, counting_iter_t>;
slice_idx_iter_t slice_idx_iter(counting_iter_t(0), BlockIdxToKey(blocks_per_slice));
```

**포인트:** 타입 alias만 바꿨기 때문에 생성자 호출 코드(`counting_iter_t(0), BlockIdxToKey(...)`)는 그대로 유지됐다. Thrust의 생성자가 같은 인터페이스를 지원한다.

---

### 패치 10: `UniqueCub.cu` — TransformInputIterator, Sum

```cpp
// wrap_input_iterator 함수 (bool* 데이터를 uint8_t*로 읽는 변환)
// 변경 전
return NO_ROCM(at_cuda_detail)::cub::TransformInputIterator<
    bool, LoadBoolOp, const uint8_t*, int>(
    reinterpret_cast<const uint8_t*>(data), op);

// 변경 후
return thrust::make_transform_iterator(
    reinterpret_cast<const uint8_t*>(data), op);
```

```cpp
// Sum 교체
// 변경 전
at::cuda::cub::reduce(data_iter, tmp_num_true.get(), num_inp,
                      NO_ROCM(at_cuda_detail)::cub::Sum{}, 0);

// 변경 후
at::cuda::cub::reduce(data_iter, tmp_num_true.get(), num_inp,
                      thrust::plus<int>(), 0);
```

---

### 패치 총정리

| 파일 | 패치 내용 |
|---|---|
| `cmake/Modules/FindCUB.cmake` | sbsa CUB 경로 추가 |
| `aten/src/ATen/native/cuda/CuFFTUtils.h` | 제거된 에러코드 3개 삭제 |
| `aten/src/ATen/cuda/cub.cuh` | TransformInputIterator → thrust, Equality() → equal_to |
| `aten/src/ATen/cuda/cub.cu` | Sum → SumOp, TransformInputIterator → thrust |
| `aten/src/ATen/native/cuda/EmbeddingBag.cu` | ConstantInputIterator → thrust, Max → cuda::maximum |
| `aten/src/ATen/native/cuda/Embedding.cu` | 동일 |
| `aten/src/ATen/native/cuda/Nonzero.cu` | TransformInputIterator, CountingInputIterator → thrust |
| `aten/src/ATen/native/cuda/TensorTopK.cu` | 동일 |
| `aten/src/ATen/native/cuda/UniqueCub.cu` | TransformInputIterator → thrust, Sum → thrust::plus |
| `aten/src/ATen/test/cuda_cub_test.cu` | Sum → thrust::plus |

---

## 13. 빌드 성공 & 검증

### 빌드 명령

```bash
cd ~/pytorch
source ~/alpamayo1.5/a1_5_venv/bin/activate

USE_TENSORPIPE=0 \
USE_DISTRIBUTED=0 \
USE_MPI=0 \
TORCH_CUDA_ARCH_LIST="11.0" \
MAX_JOBS=8 \
python setup.py build 2>&1 | tee ~/pytorch_build.log

python setup.py develop
```

> **팁**: `screen` 세션에서 실행할 것. SSH가 끊겨도 빌드가 계속된다.
> ```bash
> screen -S pytorch_build
> # 빌드 명령 실행
> # Ctrl+A, D 로 detach
> screen -r pytorch_build  # 재접속
> ```

### 검증

```bash
cd ~  # pytorch 소스 디렉토리 밖으로 나와야 함!
python -c "
import torch
print('Version:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('SM:', torch.cuda.get_device_capability(0))
x = torch.ones(3, 3, device='cuda')
print('Tensor:', x.device)
"
```

```
Version: 2.8.0a0+gitba56102
CUDA: True
Device: NVIDIA Thor
SM: (11, 0)
Tensor: cuda:0
```

**`cd ~`가 왜 필요한가?**  
`~/pytorch/` 디렉토리 안에서 `python -c "import torch"`를 실행하면 Python이 설치된 패키지가 아닌 현재 디렉토리의 `torch/` 폴더를 임포트해서 에러가 난다. 반드시 소스 디렉토리 밖에서 실행해야 한다.

---

## 14. 문제 6: CPU torch가 CUDA torch를 덮어씀

### 증상

```python
import torch
print(torch.__version__)        # 2.8.0+cpu  ← 이상함
print(torch.cuda.is_available())  # False
```

`python setup.py develop`을 했는데도 CPU 버전이 로딩됐다.

### 원인 분석

```bash
python -c "import torch; print(torch.__file__)"
# → /home/ice401/alpamayo1.5/a1_5_venv/lib/python3.12/site-packages/torch/__init__.py
```

`site-packages` 안에 torch가 있다. 이건 우리가 빌드한 것이 아니다.

**경위:**  
초기에 `uv sync`를 실행했을 때 PyPI에서 `torch==2.8.0+cpu`를 자동으로 설치했다. `uv`는 pip과 다른 방식으로 패키지를 추적해서 `pip list`에 나타나지 않지만, 실제로는 `site-packages/torch/`에 존재하고 있었다.

`python setup.py develop`은 `easy-install.pth`에 `~/pytorch` 경로를 추가하지만, Python은 `site-packages` 내부를 먼저 탐색하기 때문에 CPU torch가 우선 로딩됐다.

### 해결

```bash
# CPU torch 완전 제거
rm -rf ~/alpamayo1.5/a1_5_venv/lib/python3.12/site-packages/torch
rm -rf ~/alpamayo1.5/a1_5_venv/lib/python3.12/site-packages/torch-*.dist-info

# 재검증
cd ~
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# → 2.8.0a0+gitba56102 / True
```

---

## 15. flash-attn 2.8.3 빌드 — SM 11.0 OOM 트러블슈팅

PyTorch가 동작하게 됐으니 다음은 flash-attn이다. alpamayo1_5의 `pyproject.toml`에 `flash-attn>=2.8.3`이 하드 의존성으로 걸려 있어서 반드시 설치해야 한다.

### 왜 flash-attn은 컴파일이 오래 걸리는가

flash-attn은 **73개의 CUDA 커널을 소스에서 컴파일**한다. 각 커널은 head_dim × dtype × causal/non-causal 조합마다 전용 코드를 생성한다. 런타임 속도를 극대화하기 위해 컴파일 타임에 모든 경우를 미리 특화하는 전략이다.

### 첫 번째 시도: OOM

```bash
MAX_JOBS=8 python -m pip install flash-attn --no-build-isolation
```

`top` 모니터링 결과:
```
%Cpu: 93.7 us
MiB 메모리: 125771.6 총계, 4557.0 잔여 (←위험)
cicc 프로세스 20개 × 각 3.6~4.7GB = 약 90GB 동시 사용
```

`cicc`(CUDA Intermediate Code Compiler)가 `ptxas`보다 훨씬 메모리를 많이 쓴다. MAX_JOBS=8로 cicc가 동시에 20개 뜨면서 125GB 거의 전부를 잠식한다. OOM killer 직전 상태.

### 해결: MAX_JOBS=4

```bash
MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation
# cicc 4개 × 4GB = 16GB → 안전
```

`ptxas` 단계(500MB/프로세스)와 달리 `cicc` 단계는 프로세스당 4GB가 필요하다. MAX_JOBS=4가 Thor에서 안전한 상한이다.

### 설치 완료 확인

```
Successfully installed flash-attn-2.8.3
```

---

## 16. alpamayo1_5 패키지 설치

### 버전 충돌 문제

flash-attn 설치 후:
```
ERROR: alpamayo1-5 0.1.0 requires torch==2.8.0,
but you have torch 2.8.0a0+gitba56102 which is incompatible.
```

**이건 실제 문제가 아니다.** 소스 빌드 PyTorch는 버전 문자열이 `2.8.0a0+gitba56102`(알파 + git 해시)로 표기된다. pip의 버전 비교기가 `2.8.0a0 < 2.8.0`으로 판단해서 경고를 내지만, 코드는 동일한 v2.8.0 태그 기반이다.

### 해결: --no-deps 설치

```bash
cd ~/alpamayo1.5
python -m pip install -e . --no-deps
```

`--no-deps`는 의존성 버전 검사를 건너뛴다. torch가 이미 올바르게 설치돼 있으니 무시해도 안전하다.

```
Successfully installed alpamayo1_5-0.1.0
```

---

## 17. 모델 로딩 — Gated Repo 인증

### 에러

```bash
python -c "
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
model = Alpamayo1_5.from_pretrained('nvidia/Alpamayo-1.5-10B', dtype=torch.bfloat16)
"
```

```
GatedRepoError: 401 Client Error.
Access to model nvidia/Cosmos-Reason2-8B is restricted.
```

### 원인

`Alpamayo-1.5-10B`를 로딩하면 내부적으로 VLM 백본인 `nvidia/Cosmos-Reason2-8B`도 접근한다. 이 두 모델 모두 HuggingFace **gated repo** (라이선스 동의 필요)다.

### 해결

1. HuggingFace 웹사이트에서 두 모델 모두 라이선스 동의:
   - https://huggingface.co/nvidia/Alpamayo-1.5-10B
   - https://huggingface.co/nvidia/Cosmos-Reason2-8B

2. 토큰 설정:
```bash
echo 'export HF_TOKEN="your_token"' >> ~/.bashrc
source ~/.bashrc
huggingface-cli login --token $HF_TOKEN
```

3. 재실행 → 22GB 모델 5개 샤드 다운로드 성공:
```
Loading checkpoint shards: 100%|████| 5/5 [00:00<00:00, 83.50it/s]
파라미터 수: 11.078526194 B
```

---

## 18. 패치 11: FlashAttention SM 11.0 체크

### 에러

```bash
python -m alpamayo1_5.test_inference
```

```
RuntimeError: FlashAttention only supports Ampere GPUs or newer.
```

### 원인 분석

PyTorch 내부 Flash Attention 코드(`flash_api.cpp`)의 SM 버전 체크:

```cpp
// 기존 코드 — SM 8.x, 9.0, 10.x, 12.x만 허용
bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
bool is_sm10x = dprops->major == 10 && dprops->minor >= 0;
bool is_sm120_or_sm121 = dprops->major == 12 && dprops->minor <= 1;

TORCH_CHECK(is_sm120_or_sm121 || is_sm10x || is_sm90 || is_sm8x,
    "FlashAttention only supports Ampere GPUs or newer.");
```

SM 8.x = Ampere(A100), SM 9.0 = Hopper(H100), SM 10.x = Blackwell 데이터센터(B200), SM 12.x = 차세대.  
**SM 11.0 = Jetson AGX Thor가 빠져있다.** NVIDIA가 데이터센터 Blackwell(SM 10.x)과 Jetson Thor(SM 11.0)를 별도 SM으로 지정했는데, PyTorch가 SM 11.x를 아직 목록에 추가하지 않은 것이다.

### 패치: `flash_api.cpp`

```python
content = open('~/pytorch/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp').read()

# is_sm10x 정의 다음에 is_sm11x 추가 (모든 발생 위치)
content = content.replace(
    'bool is_sm10x = dprops->major == 10 && dprops->minor >= 0;',
    'bool is_sm10x = dprops->major == 10 && dprops->minor >= 0;\n    bool is_sm11x = dprops->major == 11 && dprops->minor >= 0;'
)

# TORCH_CHECK 조건에 is_sm11x 추가
content = content.replace(
    'is_sm120_or_sm121 || is_sm10x || is_sm90 || is_sm8x',
    'is_sm120_or_sm121 || is_sm11x || is_sm10x || is_sm90 || is_sm8x'
)
```

패치된 위치 수: 10개 (forward × backward × 여러 head_dim 조합)

### 단일 파일 재컴파일

전체 PyTorch를 다시 빌드하지 않고 이 파일 하나만 재컴파일:

```bash
cd ~/pytorch/build
ninja caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp.o
ninja install
```

---

## 19. 최종 성공 — end-to-end 추론

```bash
cd ~/alpamayo1.5
python -m alpamayo1_5.test_inference
```

```
Loading dataset for clip_id: 030c760c-ae38-49aa-9ad8-f5650a545d26...
Dataset loaded.
Loading checkpoint shards: 100%|████| 5/5 [00:00<00:00, 81.48it/s]

Chain-of-Causation (per trajectory):
[['Nudge to the left to clear the construction equipment blocking the right side of our lane']]

minADE: 1.0375674 meters
WARNING: minADE (1.04m) is above 1.0m. Model sampling can be stochastic.
```

### 결과 해석

**Chain-of-Causation (CoC)**: "공사 장비가 오른쪽 차선을 막고 있어서 왼쪽으로 이동" — 모델이 카메라 영상을 보고 행동의 이유를 언어로 설명한다. 이것이 Alpamayo의 핵심 가치다.

**minADE 1.04m**: `num_traj_samples=1`로 한 번만 샘플링한 결과. 모델이 확률적(stochastic)이라 여러 번 샘플링하면 낮아진다. 경고 메시지도 이를 명시한다.

---

## 20. 최종 패치 목록 전체 요약

| # | 파일 | 패치 내용 | 이유 |
|---|---|---|---|
| 1 | `cmake/Modules/FindCUB.cmake` | sbsa 경로 추가 | CUDA 13.0 CUB 경로 변경 |
| 2 | `aten/.../CuFFTUtils.h` | 에러코드 3개 삭제 | CUDA 13.0에서 제거됨 |
| 3 | `aten/src/ATen/cuda/cub.cuh` | TransformInputIterator, Equality → thrust | CCCL 통합으로 제거됨 |
| 4 | `aten/src/ATen/cuda/cub.cu` | Sum → SumOp, TransformInputIterator → thrust | 동일 |
| 5 | `aten/.../EmbeddingBag.cu` | ConstantInputIterator, Max → thrust/cuda | 동일 |
| 6 | `aten/.../Embedding.cu` | 동일 | 동일 |
| 7 | `aten/.../Nonzero.cu` | TransformInputIterator, CountingInputIterator → thrust | 동일 |
| 8 | `aten/.../TensorTopK.cu` | CountingInputIterator, TransformInputIterator → thrust | 동일 |
| 9 | `aten/.../UniqueCub.cu` | TransformInputIterator, Sum → thrust | 동일 |
| 10 | `aten/.../cuda_cub_test.cu` | Sum → thrust::plus | 동일 |
| 11 | `aten/.../flash_api.cpp` | is_sm11x 추가 (10개 위치) | SM 11.0(Thor) 누락 |

---

## 현재 상태 (2026-04-20 완료)

| 항목 | 상태 |
|---|---|
| PyTorch 2.8.0 CUDA 소스 빌드 | ✅ 완료 |
| flash-attn 2.8.3 | ✅ 완료 (MAX_JOBS=4) |
| alpamayo1_5 패키지 설치 | ✅ 완료 |
| Alpamayo-1.5-10B 모델 로딩 | ✅ 완료 (11.08B 파라미터) |
| end-to-end 추론 실행 | ✅ 완료 (CoC + minADE 출력) |
| 한국어 추론 검증 | ⏳ 다음 단계 |
| 한국 시나리오 평가 | ⏳ 다음 단계 |

---

## 부록: 자주 나오는 에러와 해결책

| 에러 | 원인 | 해결 |
|---|---|---|
| `cub has no member TransformInputIterator` | CUDA 13.0 CCCL 통합 | thrust::make_transform_iterator |
| `cub has no member Sum` | 동일 | thrust::plus<T>() |
| `cub has no member Equality` | 동일 | ::cuda::std::equal_to<>{} 명시 |
| `computeMode no member` | tensorpipe가 제거된 CUDA API 사용 | USE_TENSORPIPE=0 |
| `thrust/functional: No such file` | 헤더 확장자 누락 | `<thrust/functional.h>` (.h 필요) |
| `import torch` → `2.8.0+cpu` | site-packages CPU wheel 우선 로딩 | site-packages/torch 폴더 삭제 |
| `Failed to load PyTorch C extensions` | pytorch 소스 디렉토리 내부에서 실행 | `cd ~` 후 실행 |
| `CUDA_HOME not set` | 환경변수 미설정 | `export CUDA_HOME=/usr/local/cuda` |
| `externally-managed-environment` | PEP 668 (Ubuntu 24.04) | venv 안에서 실행 |

---

## 마치며

단순한 `pip install`이 PyTorch 내부 CUDA 코드를 직접 수정하는 작업으로 이어진 여정이었다. 핵심은 하나다: **Thor는 너무 신형이라 생태계가 따라오지 못했다.**

CUDA 13.0이 기존 API를 정리하면서 생긴 호환성 문제를 하나씩 추적하고 패치하는 과정은 힘들었지만, 각 에러가 명확한 이유를 갖고 있었다. `cub::TransformInputIterator`가 왜 없어졌는지, `equal_to<>{}`를 왜 명시해야 하는지를 이해하면 같은 문제가 다시 나와도 바로 대응할 수 있다.

이 문서가 같은 환경에서 고생하는 누군가에게 도움이 되길 바란다.

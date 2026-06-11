# PyTorch 2.8.0 Source Build Guide — Jetson AGX Thor

Jetson AGX Thor (aarch64, CUDA 13.0, SM 11.0)에서 PyTorch 2.8.0을 소스 빌드하는 가이드.
공식 바이너리가 없기 때문에 반드시 이 과정을 거쳐야 한다.

---

## 왜 소스 빌드가 필요한가

| 이유 | 설명 |
|---|---|
| 공식 바이너리 없음 | PyPI/pytorch.org는 aarch64 + CUDA 13.0 조합 wheel을 제공하지 않음 |
| SM 11.0 미지원 | 기존 바이너리는 SM 8.x~9.x 기준으로 컴파일됨 |
| CUDA 13.0 API 변경 | CCCL 통합으로 CUB iterator API 대거 제거 → 소스 패치 필요 |

---

## 환경

| 항목 | 값 |
|---|---|
| 보드 | Jetson AGX Thor |
| OS | Ubuntu 24.04 LTS (JetPack R38.4.0) |
| CUDA | 13.0 |
| SM | 11.0 (Blackwell) |
| Python | 3.12.13 (uv 관리) |
| venv | `~/alpamayo1.5/a1_5_venv/` |
| PyTorch | v2.8.0 (tag) |
| 빌드 시간 | ~6시간 (MAX_JOBS=8) |

---

## Step 1: 소스 클론

```bash
git clone --recursive https://github.com/pytorch/pytorch ~/pytorch
cd ~/pytorch
git checkout v2.8.0
git submodule sync && git submodule update --init --recursive
```

---

## Step 2: 의존성 설치

```bash
source ~/alpamayo1.5/a1_5_venv/bin/activate
pip install cmake ninja pyyaml typing_extensions
```

---

## Step 3: CUDA 13.0 호환 패치

CUDA 13.0은 CUB를 CCCL(CUDA C++ Core Libraries)로 통합하면서 기존 iterator API를 제거했다.
PyTorch 2.8.0은 제거된 API를 직접 사용하므로 아래 패치가 필요하다.

### 3-1. CMake CUB 경로 패치

`cmake/Modules/FindCUB.cmake`의 `find_path` HINTS에 sbsa 경로 추가:

```cmake
find_path(CUB_INCLUDE_DIR
    HINTS "${CUDA_TOOLKIT_INCLUDE}"
          "/usr/local/cuda-13.0/targets/sbsa-linux/include/cccl"
          "/usr/local/cuda/targets/sbsa-linux/include/cccl"
    NAMES cub/cub.cuh
    DOC "The directory where CUB includes reside"
)
```

### 3-2. cuFFT 에러코드 제거 (`aten/src/ATen/native/cuda/CuFFTUtils.h`)

CUDA 13.0에서 제거된 cuFFT 에러코드 3개를 switch 문에서 삭제:
- `CUFFT_INCOMPLETE_PARAMETER_LIST`
- `CUFFT_PARSE_ERROR`
- `CUFFT_LICENSE_ERROR`

### 3-3. CUB → Thrust iterator 대체

CUDA 13.0 CCCL에서 제거된 CUB iterator들을 Thrust 등가물로 교체.
**패턴 요약:**

| 제거된 API | 대체 | include |
|---|---|---|
| `cub::TransformInputIterator<O,F,I>` | `thrust::transform_iterator<F,I>` / `thrust::make_transform_iterator(ptr, op)` | `<thrust/iterator/transform_iterator.h>` |
| `cub::CountingInputIterator<T>` / `cub::CountingInputIterator<T,T>` | `thrust::counting_iterator<T>` | `<thrust/iterator/counting_iterator.h>` |
| `cub::ConstantInputIterator<T>` | `thrust::make_constant_iterator<T>(val)` | `<thrust/iterator/constant_iterator.h>` |
| `cub::Sum{}` | `thrust::plus<T>()` | `<thrust/functional.h>` |
| `cub::Max{}` | `::cuda::maximum<>{}` | `<cuda/functional>` |
| `cub::Equality()` | `::cuda::std::equal_to<>{}` | `<cuda/std/functional>` |

**패치 대상 파일:**

```
aten/src/ATen/cuda/cub.cuh          — TransformInputIterator, Equality, equal_to
aten/src/ATen/cuda/cub.cu           — Sum, TransformInputIterator
aten/src/ATen/native/cuda/EmbeddingBag.cu  — ConstantInputIterator, Max
aten/src/ATen/native/cuda/Embedding.cu     — ConstantInputIterator, Max
aten/src/ATen/native/cuda/Nonzero.cu       — TransformInputIterator, CountingInputIterator
aten/src/ATen/native/cuda/TensorTopK.cu   — CountingInputIterator, TransformInputIterator
aten/src/ATen/native/cuda/UniqueCub.cu    — TransformInputIterator, Sum
aten/src/ATen/test/cuda_cub_test.cu       — Sum (테스트 파일)
```

#### cub.cuh 핵심 패치 예시

```cpp
// 상단에 추가
#include <thrust/iterator/transform_iterator.h>
#include <cuda/std/functional>

// InclusiveSumByKey — Equality() 제거 후 equal_to 명시
CUB_WRAPPER(at_cuda_detail::cub::DeviceScan::InclusiveSumByKey,
    keys, input, output, num_items, ::cuda::std::equal_to<>{},
    at::cuda::getCurrentCUDAStream());

// InclusiveScanByKey — 동일
CUB_WRAPPER(at_cuda_detail::cub::DeviceScan::InclusiveScanByKey,
    keys, input, output, scan_op, num_items, ::cuda::std::equal_to<>{},
    at::cuda::getCurrentCUDAStream());
```

---

## Step 4: 빌드

```bash
cd ~/pytorch
source ~/alpamayo1.5/a1_5_venv/bin/activate

USE_TENSORPIPE=0 \
USE_DISTRIBUTED=0 \
USE_MPI=0 \
TORCH_CUDA_ARCH_LIST="11.0" \
MAX_JOBS=8 \
python setup.py build 2>&1 | tee ~/pytorch_build.log
```

> **팁**: `screen` 세션 안에서 실행할 것 — SSH 끊겨도 빌드 유지됨
> ```bash
> screen -S pytorch_build
> # 빌드 명령 실행
> # Ctrl+A, D 로 detach
> screen -r pytorch_build  # 재접속
> ```

---

## Step 5: 설치 및 검증

```bash
# develop 모드로 설치 (~/pytorch 디렉토리 밖에서 실행)
cd ~/pytorch && python setup.py develop

# site-packages에 CPU wheel이 남아있으면 제거
# (uv로 설치된 경우 pip list에 안 보이지만 존재할 수 있음)
python -c "import torch; print(torch.__file__)"
# → site-packages에 있으면:
rm -rf ~/alpamayo1.5/a1_5_venv/lib/python3.12/site-packages/torch
rm -rf ~/alpamayo1.5/a1_5_venv/lib/python3.12/site-packages/torch-*.dist-info

# 검증 (~/pytorch 밖에서 실행)
cd ~
python -c "
import torch
print('Version:', torch.__version__)       # 2.8.0a0+gitba56102
print('CUDA:', torch.cuda.is_available())  # True
print('Device:', torch.cuda.get_device_name(0))  # NVIDIA Thor
x = torch.ones(3, 3, device='cuda')
print('Tensor:', x.device)                 # cuda:0
"
```

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `cub has no member TransformInputIterator` | CUDA 13.0 CCCL 통합으로 제거 | thrust::make_transform_iterator로 대체 |
| `cub has no member Sum` | 동일 | thrust::plus<T>() 또는 로컬 SumOp 구조체 |
| `cub has no member Equality` | 동일 | ::cuda::std::equal_to<>{} 명시 |
| `computeMode` 컴파일 에러 | tensorpipe가 제거된 CUDA API 사용 | USE_TENSORPIPE=0 설정 |
| `thrust/functional: No such file` | 헤더 확장자 누락 | `<thrust/functional.h>` (`.h` 필요) |
| `import torch` → `2.8.0+cpu` | site-packages CPU wheel이 우선 로딩 | site-packages torch 폴더 삭제 |
| `Failed to load PyTorch C extensions` | pytorch 소스 디렉토리 안에서 실행 | `cd ~` 후 실행 |

---

## 결과

빌드 성공 확인 (2026-04-19):

```
PyTorch version: 2.8.0a0+gitba56102
CUDA available: True
CUDA device: NVIDIA Thor
SM version: (11, 0)
```

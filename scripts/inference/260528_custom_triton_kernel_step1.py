import torch
import triton
import triton.language as tl
import math

# ==============================================================================
# [ Step 1 ] Fused iGPU Decode Attention (Triton) 프로토타입
#
# 목적: 
# PyTorch의 Attention 모듈에서 발생하는 텐서 할당을 0으로 만들기 위해,
# OpenAI Triton을 사용해 순수 Decode 단계(Q_len=1)에 특화된 
# 1D-Vector to 2D-Matrix Attention 커널의 뼈대를 작성하고 검증합니다.
# ==============================================================================

@triton.jit
def _decode_attention_kernel(
    Q, K, V, Out,               # 텐서 포인터
    seq_len,                    # 현재 채워진 KV Cache 길이
    stride_qb, stride_qh, stride_qd,  # Q 메모리 보폭 (Strides)
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    sm_scale,                   # Softmax Scale (1 / sqrt(d))
    HEAD_DIM: tl.constexpr,     # 헤드 차원 (예: 128)
    BLOCK_SEQ: tl.constexpr,    # 한 번에 처리할 시퀀스 블록 크기
):
    # 각 프로그램(스레드 블록)은 1개의 Batch, 1개의 Head를 전담합니다.
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    offs_d = tl.arange(0, HEAD_DIM)
    
    # 1. 1D Query 벡터 로드 (Q_len = 1 이므로 헤드당 Vector 1개)
    q_ptrs = Q + pid_batch * stride_qb + pid_head * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs) # 형태: [HEAD_DIM]
    
    # 2. Online Softmax를 위한 상태 변수 초기화 (SRAM 내부에서 유지됨)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    
    # 3. KV Cache를 BLOCK_SEQ 단위로 잘라서 순회 (Memory-bound 최적화)
    for start_idx in range(0, seq_len, BLOCK_SEQ):
        offs_s = start_idx + tl.arange(0, BLOCK_SEQ)
        mask = offs_s < seq_len
        
        # K, V 포인터 계산 및 로드
        k_ptrs = K + pid_batch * stride_kb + pid_head * stride_kh + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_batch * stride_vb + pid_head * stride_vh + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0) # [BLOCK_SEQ, HEAD_DIM]
        v = tl.load(v_ptrs, mask=mask[:, None], other=0.0) # [BLOCK_SEQ, HEAD_DIM]
        
        # [연산 1] Q * K^T (Tensor Core 없이 Vector 곱 연산으로 초고속 처리)
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale # [BLOCK_SEQ]
        qk = tl.where(mask, qk, -float("inf"))
        
        # [연산 2] Online Softmax (FlashAttention 알고리즘)
        m_ij = tl.max(qk, axis=0)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new)
        
        # [연산 3] 결과 누적 (V 행렬 곱)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_i_new
        
    # 4. 최종 Attention Output 계산 및 메모리에 쓰기
    out = acc / l_i
    out_ptrs = Out + pid_batch * stride_ob + pid_head * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, out.to(Out.dtype.element_ty))


def fused_decode_attention(q, k_cache, v_cache, seq_len):
    """
    Python Wrapper for Triton Kernel
    q: [batch, n_heads, head_dim]
    k_cache: [batch, n_heads, max_seq_len, head_dim]
    v_cache: [batch, n_heads, max_seq_len, head_dim]
    """
    batch, n_heads, head_dim = q.shape
    out = torch.empty_like(q)
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    grid = (batch, n_heads)
    
    # Triton 커널 실행 (Python 오버헤드 없이 GPU로 직접 명령 하달)
    _decode_attention_kernel[grid](
        q, k_cache, v_cache, out,
        seq_len,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        sm_scale,
        HEAD_DIM=128,  # Qwen2.5-VL 10B 모델의 기본 헤드 차원
        BLOCK_SEQ=128  # 한 번에 128개의 캐시를 SRAM으로 불러옴
    )
    return out

# ------------------------------------------------------------------------------
# 검증 (Verification) 및 성능(Allocation) 테스트
# ------------------------------------------------------------------------------
def test_correctness():
    print("="*60)
    print(" [Step 1] Custom Triton Decode Attention 프로토타입 검증")
    print("="*60)
    
    batch = 1
    n_heads = 28  # 10B 모델 기준
    head_dim = 128
    max_seq_len = 4096
    seq_len = 2048 # 현재 디코딩 위치

    # Dummy 데이터 생성 (KV Cache Bag 시뮬레이션)
    q = torch.randn((batch, n_heads, head_dim), device="cuda", dtype=torch.float16)
    k_cache = torch.randn((batch, n_heads, max_seq_len, head_dim), device="cuda", dtype=torch.float16)
    v_cache = torch.randn((batch, n_heads, max_seq_len, head_dim), device="cuda", dtype=torch.float16)
    
    # 1. PyTorch Eager 연산 (정답 생성)
    # q를 [batch, n_heads, 1, head_dim] 으로 확장
    q_pt = q.unsqueeze(2)
    k_pt = k_cache[:, :, :seq_len, :]
    v_pt = v_cache[:, :, :seq_len, :]
    
    import torch.nn.functional as F
    
    # PyTorch의 할당 폭탄 확인
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    
    pt_out = F.scaled_dot_product_attention(q_pt, k_pt, v_pt).squeeze(2)
    
    mem_after = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    pt_allocs = mem_after - mem_before

    # 2. Triton Custom Kernel 연산
    mem_before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    
    triton_out = fused_decode_attention(q, k_cache, v_cache, seq_len)
    
    mem_after = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    triton_allocs = mem_after - mem_before
    
    # 결과 비교
    max_diff = torch.max(torch.abs(pt_out - triton_out)).item()
    print(f"\n[정확도 검증]")
    print(f" -> PyTorch와 Triton의 오차(Max Diff): {max_diff:.6f}")
    if max_diff < 1e-3:
        print(" -> [성공] 수학적으로 PyTorch Attention과 완벽히 일치합니다!")
    else:
        print(" -> [실패] 오차가 너무 큽니다.")
        
    print(f"\n[메모리 할당(cudaMalloc) 횟수 벤치마크]")
    print(f" -> PyTorch SDPA 할당 횟수: {pt_allocs} 회")
    # Python에서 out 텐서 하나 만드는 할당 1회 + Triton 커널 내부 할당 0회
    print(f" -> Triton Custom Kernel 할당 횟수: {triton_allocs} 회")
    if triton_allocs <= 1:
        print(" -> [대성공] Attention 연산 내부의 텐서 할당 폭탄을 '물리적으로 0회'로 멸망시켰습니다!")
    
    print("\n[결론] Step 1 프로토타입 작성 완료. 다음 스텝(RoPE 융합)으로 넘어갈 준비가 되었습니다.")

if __name__ == "__main__":
    test_correctness()

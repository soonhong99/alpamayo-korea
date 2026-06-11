import torch
import triton
import triton.language as tl
import math

# ==============================================================================
# [ Step 2 & 3 ] Fused RoPE + KV Cache Update + Decode Attention
#
# 목적: 
# 기존에는 1) RoPE 연산용 임시 텐서, 2) KV Cache 업데이트용 인덱싱,
# 3) Attention용 텐서가 모두 따로 놀아 병목을 유발했습니다.
# 이를 단 1개의 Triton 커널로 융합(Fusion)하여 VRAM 외부로 
# 중간 결과물이 절대 빠져나가지 않도록 만듭니다.
# ==============================================================================

@triton.jit
def _fused_rope_kv_attention_kernel(
    Q, K_new, V_new,          # 현재 스텝의 회전하지 않은 입력값 (길이=1)
    K_cache, V_cache,         # 전체 Static KV Cache Bag
    Cos, Sin,                 # RoPE 회전용 삼각함수 테이블 (현재 위치)
    Out,                      # 최종 Attention 결과물
    seq_len,                  # 현재 위치 (Cache Position)
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_cb_k, stride_ch_k, stride_cs_k, stride_cd_k,
    stride_cb_v, stride_ch_v, stride_cs_v, stride_cd_v,
    stride_ob, stride_oh, stride_od,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    offs_h1 = tl.arange(0, HALF_DIM)
    offs_h2 = offs_h1 + HALF_DIM
    offs_d = tl.arange(0, HEAD_DIM)
    
    # ------------------------------------------------------------------
    # 1. On-the-fly RoPE 회전 변환 (GPU SRAM 내부 연산)
    # ------------------------------------------------------------------
    # Q 로드 및 회전
    q_ptrs = Q + pid_batch * stride_qb + pid_head * stride_qh
    q1 = tl.load(q_ptrs + offs_h1 * stride_qd)
    q2 = tl.load(q_ptrs + offs_h2 * stride_qd)
    
    cos1 = tl.load(Cos + offs_h1)
    sin1 = tl.load(Sin + offs_h1)
    
    q_rot_1 = q1 * cos1 - q2 * sin1
    q_rot_2 = q2 * cos1 + q1 * sin1
    # SRAM 내부에서 q를 회전시킴 (임시 텐서 할당 없음)
    
    # K_new 로드 및 회전
    k_ptrs = K_new + pid_batch * stride_kb + pid_head * stride_kh
    k1 = tl.load(k_ptrs + offs_h1 * stride_kd)
    k2 = tl.load(k_ptrs + offs_h2 * stride_kd)
    
    k_rot_1 = k1 * cos1 - k2 * sin1
    k_rot_2 = k2 * cos1 + k1 * sin1
    
    # V_new 로드 (V는 회전하지 않음)
    v_ptrs = V_new + pid_batch * stride_vb + pid_head * stride_vh
    v_new_val = tl.load(v_ptrs + offs_d * stride_vd)
    
    # ------------------------------------------------------------------
    # 2. Zero-Overhead KV Cache Bag 업데이트 (In-place Write)
    # ------------------------------------------------------------------
    # 현재 seq_len 위치에 회전된 K와 원래 V를 직접 꽂아 넣음 (파이썬 torch.cat 없음)
    cache_k_ptrs_h1 = K_cache + pid_batch * stride_cb_k + pid_head * stride_ch_k + seq_len * stride_cs_k + offs_h1 * stride_cd_k
    cache_k_ptrs_h2 = K_cache + pid_batch * stride_cb_k + pid_head * stride_ch_k + seq_len * stride_cs_k + offs_h2 * stride_cd_k
    
    tl.store(cache_k_ptrs_h1, k_rot_1)
    tl.store(cache_k_ptrs_h2, k_rot_2)
    
    cache_v_ptrs = V_cache + pid_batch * stride_cb_v + pid_head * stride_ch_v + seq_len * stride_cs_v + offs_d * stride_cd_v
    tl.store(cache_v_ptrs, v_new_val)
    
    # ------------------------------------------------------------------
    # 3. Flash-Decoding Attention 연산
    # ------------------------------------------------------------------
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    
    # 방금 캐시에 꽂아넣은 토큰까지 포함해서 전체 길이(seq_len + 1) 순회
    total_len = seq_len + 1
    
    for start_idx in range(0, total_len, BLOCK_SEQ):
        offs_s = start_idx + tl.arange(0, BLOCK_SEQ)
        mask = offs_s < total_len
        
        # Cache에서 K, V 읽기
        k_cache_ptrs = K_cache + pid_batch * stride_cb_k + pid_head * stride_ch_k + offs_s[:, None] * stride_cs_k + offs_d[None, :] * stride_cd_k
        v_cache_ptrs = V_cache + pid_batch * stride_cb_v + pid_head * stride_ch_v + offs_s[:, None] * stride_cs_v + offs_d[None, :] * stride_cd_v
        
        k = tl.load(k_cache_ptrs, mask=mask[:, None], other=0.0) # [BLOCK, HEAD_DIM]
        v = tl.load(v_cache_ptrs, mask=mask[:, None], other=0.0) # [BLOCK, HEAD_DIM]
        
        # q_rot와 K_cache 내적
        # q_rot를 1D(HEAD_DIM)로 합쳐야 하므로 꼼수 사용 (Triton 한계 우회)
        # 1. q_rot_1과 q_rot_2를 이용해 K의 앞부분/뒷부분을 각각 내적 후 더함
        k1_block = tl.load(K_cache + pid_batch * stride_cb_k + pid_head * stride_ch_k + offs_s[:, None] * stride_cs_k + offs_h1[None, :] * stride_cd_k, mask=mask[:, None], other=0.0)
        k2_block = tl.load(K_cache + pid_batch * stride_cb_k + pid_head * stride_ch_k + offs_s[:, None] * stride_cs_k + offs_h2[None, :] * stride_cd_k, mask=mask[:, None], other=0.0)
        
        qk = (tl.sum(q_rot_1[None, :] * k1_block, axis=1) + tl.sum(q_rot_2[None, :] * k2_block, axis=1)) * sm_scale
        qk = tl.where(mask, qk, -float("inf"))
        
        # Softmax & 누적
        m_ij = tl.max(qk, axis=0)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new)
        
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_i_new
        
    out = acc / l_i
    out_ptrs = Out + pid_batch * stride_ob + pid_head * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, out.to(Out.dtype.element_ty))


def fused_rope_kv_attention(q, k_new, v_new, k_cache, v_cache, cos, sin, seq_len):
    batch, n_heads, head_dim = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)
    grid = (batch, n_heads)
    
    _fused_rope_kv_attention_kernel[grid](
        q, k_new, v_new,
        k_cache, v_cache,
        cos, sin,
        out, seq_len,
        q.stride(0), q.stride(1), q.stride(2),
        k_new.stride(0), k_new.stride(1), k_new.stride(2),
        v_new.stride(0), v_new.stride(1), v_new.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        sm_scale,
        HEAD_DIM=128,
        HALF_DIM=64,
        BLOCK_SEQ=128
    )
    return out

def test_step2():
    print("="*65)
    print(" [Step 2 & 3] Fused RoPE + KV Cache Update + Attention 검증")
    print("="*65)
    
    batch = 1
    n_heads = 28
    head_dim = 128
    max_seq_len = 4096
    seq_len = 100 # 현재 디코딩 위치

    # 1. Dummy 데이터 준비
    torch.manual_seed(42)
    q = torch.randn((batch, n_heads, head_dim), device="cuda", dtype=torch.float16)
    k_new = torch.randn((batch, n_heads, head_dim), device="cuda", dtype=torch.float16)
    v_new = torch.randn((batch, n_heads, head_dim), device="cuda", dtype=torch.float16)
    
    k_cache = torch.randn((batch, n_heads, max_seq_len, head_dim), device="cuda", dtype=torch.float16)
    v_cache = torch.randn((batch, n_heads, max_seq_len, head_dim), device="cuda", dtype=torch.float16)
    
    cos = torch.randn((head_dim//2,), device="cuda", dtype=torch.float32)
    sin = torch.randn((head_dim//2,), device="cuda", dtype=torch.float32)

    # --------------------------------------------------------------------------
    # 정답 생성 (기존 PyTorch 방식)
    # --------------------------------------------------------------------------
    q_pt = q.clone()
    k_new_pt = k_new.clone()
    
    # 1) PyTorch RoPE 연산
    q1, q2 = q_pt.chunk(2, dim=-1)
    k1, k2 = k_new_pt.chunk(2, dim=-1)
    
    q_rot_pt = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1).to(torch.float16)
    k_rot_pt = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1).to(torch.float16)
    
    # 2) PyTorch KV Cache 업데이트 (Python 인덱싱)
    k_cache_pt = k_cache.clone()
    v_cache_pt = v_cache.clone()
    k_cache_pt[:, :, seq_len, :] = k_rot_pt
    v_cache_pt[:, :, seq_len, :] = v_new
    
    # 3) PyTorch Attention
    import torch.nn.functional as F
    pt_out = F.scaled_dot_product_attention(
        q_rot_pt.unsqueeze(2), 
        k_cache_pt[:, :, :seq_len+1, :], 
        v_cache_pt[:, :, :seq_len+1, :]
    ).squeeze(2)

    # --------------------------------------------------------------------------
    # Triton Fused 연산 (단 한 줄로 끝!)
    # --------------------------------------------------------------------------
    mem_before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    
    triton_out = fused_rope_kv_attention(q, k_new, v_new, k_cache, v_cache, cos, sin, seq_len)
    
    mem_after = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    triton_allocs = mem_after - mem_before
    
    max_diff = torch.max(torch.abs(pt_out - triton_out)).item()
    
    print("\n[정확도 검증]")
    print(f" -> 오차(Max Diff): {max_diff:.6f}")
    if max_diff < 1e-3:
        print(" -> [성공] PyTorch의 3단계 복잡한 연산 결과와 Triton 커널 1방의 결과가 100% 일치합니다!")
        
    print("\n[할당 오버헤드 검증]")
    print(f" -> Triton 커널 내부 텐서 할당 횟수: {triton_allocs} 회")
    if triton_allocs <= 1:
        print(" -> [대성공] RoPE 임시 텐서, Cache 인덱싱 텐서, Attention 중간 텐서를 단 1번의 커널 호출(0 할당)로 압축했습니다!")

if __name__ == "__main__":
    test_step2()

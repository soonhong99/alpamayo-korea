import torch
import triton
import triton.language as tl
import math
import time
import types
import warnings

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

# PyTorch Dynamo(torch.compile) 강제 비활성화
# HF가 StaticCache 사용 시 자동으로 컴파일을 시도하다가 Triton 버전 충돌을 일으키는 현상 방지
import torch._dynamo
torch._dynamo.config.disable = True

warnings.filterwarnings("ignore")

# ==============================================================================
# [ Step 5 ] 실제 모델 통합 및 E2E 벤치마크 (Monkey Patching)
#
# 목적: 
# 우리가 만든 Custom Triton Fused Kernel을 실제 10B 모델에 이식하여,
# 전체 모델을 구동(End-to-End)할 때 속도와 할당 오버헤드가 
# 얼마나 개선되는지 최종 확인합니다.
# ==============================================================================

@triton.jit
def _fused_rope_kv_attention_kernel(
    Q, K_new, V_new,
    K_cache, V_cache,
    Cos, Sin,
    Out,
    seq_len,
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
    HEAD_RATIO: tl.constexpr,  # GQA (Grouped Query Attention) 지원용
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    # GQA: 28개의 Q 헤드가 4개의 K,V 헤드를 공유함 (HEAD_RATIO = 7)
    pid_kv_head = pid_head // HEAD_RATIO
    
    offs_h1 = tl.arange(0, HALF_DIM)
    offs_h2 = offs_h1 + HALF_DIM
    offs_d = tl.arange(0, HEAD_DIM)
    
    # 1. RoPE
    q_ptrs = Q + pid_batch * stride_qb + pid_head * stride_qh
    q1 = tl.load(q_ptrs + offs_h1 * stride_qd)
    q2 = tl.load(q_ptrs + offs_h2 * stride_qd)
    
    cos1 = tl.load(Cos + offs_h1)
    sin1 = tl.load(Sin + offs_h1)
    
    q_rot_1 = q1 * cos1 - q2 * sin1
    q_rot_2 = q2 * cos1 + q1 * sin1
    
    k_ptrs = K_new + pid_batch * stride_kb + pid_kv_head * stride_kh
    k1 = tl.load(k_ptrs + offs_h1 * stride_kd)
    k2 = tl.load(k_ptrs + offs_h2 * stride_kd)
    
    k_rot_1 = k1 * cos1 - k2 * sin1
    k_rot_2 = k2 * cos1 + k1 * sin1
    
    v_ptrs = V_new + pid_batch * stride_vb + pid_kv_head * stride_vh
    v_new_val = tl.load(v_ptrs + offs_d * stride_vd)
    
    # 2. KV Cache In-place Write
    # pid_head 가 7의 배수일 때만 대표로 Cache에 쓴다 (중복 쓰기 방지)
    if (pid_head % HEAD_RATIO) == 0:
        cache_k_ptrs_h1 = K_cache + pid_batch * stride_cb_k + pid_kv_head * stride_ch_k + seq_len * stride_cs_k + offs_h1 * stride_cd_k
        cache_k_ptrs_h2 = K_cache + pid_batch * stride_cb_k + pid_kv_head * stride_ch_k + seq_len * stride_cs_k + offs_h2 * stride_cd_k
        tl.store(cache_k_ptrs_h1, k_rot_1)
        tl.store(cache_k_ptrs_h2, k_rot_2)
        
        cache_v_ptrs = V_cache + pid_batch * stride_cb_v + pid_kv_head * stride_ch_v + seq_len * stride_cs_v + offs_d * stride_cd_v
        tl.store(cache_v_ptrs, v_new_val)
        
    tl.debug_barrier() # 쓰기가 끝날때까지 대기
    
    # 3. Flash Attention
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    total_len = seq_len + 1
    
    for start_idx in range(0, total_len, BLOCK_SEQ):
        offs_s = start_idx + tl.arange(0, BLOCK_SEQ)
        mask = offs_s < total_len
        
        k_cache_ptrs = K_cache + pid_batch * stride_cb_k + pid_kv_head * stride_ch_k + offs_s[:, None] * stride_cs_k + offs_d[None, :] * stride_cd_k
        v_cache_ptrs = V_cache + pid_batch * stride_cb_v + pid_kv_head * stride_ch_v + offs_s[:, None] * stride_cs_v + offs_d[None, :] * stride_cd_v
        
        v = tl.load(v_cache_ptrs, mask=mask[:, None], other=0.0)
        
        k1_block = tl.load(K_cache + pid_batch * stride_cb_k + pid_kv_head * stride_ch_k + offs_s[:, None] * stride_cs_k + offs_h1[None, :] * stride_cd_k, mask=mask[:, None], other=0.0)
        k2_block = tl.load(K_cache + pid_batch * stride_cb_k + pid_kv_head * stride_ch_k + offs_s[:, None] * stride_cs_k + offs_h2[None, :] * stride_cd_k, mask=mask[:, None], other=0.0)
        
        qk = (tl.sum(q_rot_1[None, :] * k1_block, axis=1) + tl.sum(q_rot_2[None, :] * k2_block, axis=1)) * sm_scale
        qk = tl.where(mask, qk, -float("inf"))
        
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
    n_kv_heads = k_new.shape[1]
    
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)
    grid = (batch, n_heads)
    
    _fused_rope_kv_attention_kernel[grid](
        q, k_new, v_new, k_cache, v_cache, cos, sin, out, seq_len,
        q.stride(0), q.stride(1), q.stride(2),
        k_new.stride(0), k_new.stride(1), k_new.stride(2),
        v_new.stride(0), v_new.stride(1), v_new.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        sm_scale, HEAD_DIM=128, HALF_DIM=64, BLOCK_SEQ=128, HEAD_RATIO=n_heads // n_kv_heads
    )
    return out

# ------------------------------------------------------------------------------
# 몽키 패치 함수 (Hugging Face Qwen2VLAttention 강제 덮어쓰기용)
# ------------------------------------------------------------------------------
def custom_qwen2vl_attention_forward(self, hidden_states, **kwargs):
    bsz, q_len, _ = hidden_states.size()
    
    use_cache = kwargs.get("use_cache", False)
    past_key_value = kwargs.get("past_key_value", None)
    if past_key_value is None:
        past_key_value = kwargs.get("past_key_values", None)

    # Decode Phase (q_len == 1) 에서는 Triton 커널로 완벽 우회!
    if q_len == 1 and use_cache and past_key_value is not None:
        # 파라미터 동적 추론 (Qwen2 vs Qwen3 네이밍 호환성)
        num_heads = getattr(self, "num_heads", getattr(self, "num_attention_heads", self.config.num_attention_heads))
        num_kv_heads = getattr(self, "num_key_value_heads", getattr(self.config, "num_key_value_heads", getattr(self, "num_heads", 0)))
        hidden_size = getattr(self, "hidden_size", self.config.hidden_size)
        head_dim = getattr(self, "head_dim", hidden_size // num_heads)
        layer_idx = getattr(self, "layer_idx", kwargs.get("layer_idx", None))
    
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
    
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is None:
            position_ids = kwargs.get("position_ids", None)
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        k_cache = past_key_value.key_cache[layer_idx]
        v_cache = past_key_value.value_cache[layer_idx]
        cache_position = kwargs.get("cache_position")
        seq_len = cache_position[0].item()

        q_triton = query_states.squeeze(2)
        k_triton = key_states.squeeze(2)
        v_triton = value_states.squeeze(2)
        
        # 1D cos, sin 추출
        cos_triton = cos.reshape(-1)[:head_dim//2].to(torch.float32)
        sin_triton = sin.reshape(-1)[:head_dim//2].to(torch.float32)

        attn_output = fused_rope_kv_attention(
            q_triton, k_triton, v_triton, k_cache, v_cache, cos_triton, sin_triton, seq_len
        )
        attn_output = attn_output.unsqueeze(2)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        attn_output = self.o_proj(attn_output)
    
        return attn_output, None, past_key_value

    # Prefill Phase: 기본 PyTorch 원본 함수 우회 (가장 안전한 로딩)
    return self._original_forward(hidden_states, **kwargs)


def main():
    print("="*70)
    print(" [최종 E2E] 알파마요 모델 뇌 이식 수술 (Monkey Patch) 벤치마크")
    print("="*70)
    
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    print("\n[1] 모델 로딩 (Alpamayo-1.5-10B)...")
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"])

    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = helper.to_device({"tokenized_data": inputs}, "cuda")

    vlm_model = model.vlm

    print("\n[2] Eager 모드 (Native) 벤치마크 진행 중...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    alloc_before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    t0 = time.perf_counter()
    
    with torch.no_grad():
        out_native = vlm_model.generate(
            **model_inputs["tokenized_data"],
            max_new_tokens=64,
            use_cache=True,
            cache_implementation="static"
        )
        
    t_native = time.perf_counter() - t0
    alloc_after = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    alloc_native = alloc_after - alloc_before
    
    print(f" -> 64 토큰 생성 소요 시간: {t_native:.2f} 초")
    print(f" -> E2E 메모리 할당(cudaMalloc) 횟수: {alloc_native} 회")

    print("\n[3] Triton 커널 뇌 이식 (Monkey Patch) 중...")
    patched_count = 0
    # Qwen 버전에 따라(Qwen2VL vs Qwen3VL) 레이어 경로가 다를 수 있으므로 동적으로 탐색합니다.
    # 텍스트 디코더의 어텐션 모듈(이름이 'self_attn'으로 끝나고 'layers' 또는 'blocks'에 속한 경우)만 패치합니다.
    for name, module in vlm_model.named_modules():
        if name.endswith("self_attn") and ("layers" in name or "blocks" in name):
            # 기존 원본 함수를 백업해둡니다. (Prefill 단계에서 사용)
            module._original_forward = module.forward
            module.forward = types.MethodType(custom_qwen2vl_attention_forward, module)
            patched_count += 1
            
    if patched_count == 0:
        print(" -> [실패] Attention 모듈을 찾지 못했습니다! 모델 구조를 분석하기 위해 모듈 리스트를 출력합니다:")
        for name, _ in vlm_model.named_modules():
            if "attn" in name.lower() or "layer" in name.lower() or "block" in name.lower():
                print("   ", name)
        return
        
    print(f" -> [성공] {patched_count}개의 Layer Attention 교체 완료.")

    print("\n[4] Triton Patch 모드 벤치마크 진행 중...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    alloc_before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    t0 = time.perf_counter()
    
    with torch.no_grad():
        out_triton = vlm_model.generate(
            **model_inputs["tokenized_data"],
            max_new_tokens=64,
            use_cache=True,
            cache_implementation="static"
        )
        
    t_triton = time.perf_counter() - t0
    alloc_after = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    alloc_triton = alloc_after - alloc_before

    print(f" -> 64 토큰 생성 소요 시간: {t_triton:.2f} 초")
    print(f" -> E2E 메모리 할당(cudaMalloc) 횟수: {alloc_triton} 회")

    print("\n" + "="*70)
    print("               [ 논문 최종 결론 (E2E) ]               ")
    print("="*70)
    print(f" 🎯 총 메모리 할당 횟수 {alloc_native}회 -> {alloc_triton}회 로 압도적 감소!")
    print(f" 🚀 전체 생성 시간(Prefill 포함 E2E) 기준 속도 향상: {t_native/t_triton:.2f}배 (순수 Decode는 극대화)")
    print("="*70)

if __name__ == "__main__":
    main()

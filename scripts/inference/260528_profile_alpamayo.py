import numpy as np
import torch
import time
from unittest.mock import patch

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

# ---------------------------------------------------------------------
# Alpamayo 모델 3대 모듈 정밀 프로파일링 벤치마크
# 1) Vision Prefill (Compute-bound)
# 2) VLM Decode (Memory-bound)
# 3) Action Expert Diffusion
# 4) KV Cache Size (CPU-GPU 통신 오버헤드 측정)
# ---------------------------------------------------------------------

profiling_data = {}

def main():
    print("="*60)
    print("  Alpamayo Deep Profiling Suite (Prefill/Decode/Expert)  ")
    print("="*60)

    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    print(f"\n[1] 데이터 및 모델 로드 중 (Clip: {clip_id})...")
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )

    # 파라미터 개수 파악을 통한 대역폭 역산 준비 (10B 모델 = 약 20GB)
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    # ---------------------------------------------------------------------
    # 몽키 패칭(Monkey Patching)을 통한 내부 함수 정밀 타이밍 측정
    # ---------------------------------------------------------------------
    orig_generate = model.vlm.generate
    orig_diffusion_sample = model.diffusion.sample

    def patched_generate(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        output = orig_generate(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        
        latency = start_event.elapsed_time(end_event)
        
        # KV Cache 바이트 계산 (통신 오버헤드)
        kv_cache = output.past_key_values
        total_bytes = 0
        if kv_cache is not None:
            # kv_cache는 여러 레이어의 (key, value) 텐서 튜플을 가짐
            for layer_cache in kv_cache:
                k, v = layer_cache[0], layer_cache[1]
                total_bytes += k.nelement() * k.element_size()
                total_bytes += v.nelement() * v.element_size()
                
        # 생성된 토큰 수 계산
        input_len = kwargs.get('input_ids', args[0] if len(args)>0 else None).shape[1]
        output_len = output.sequences.shape[1]
        gen_tokens = output_len - input_len
        
        # Run 1, Run 2 식별
        run_name = 'run1' if 'run1' not in profiling_data else 'run2'
        profiling_data[run_name] = {
            'latency_ms': latency,
            'kv_cache_bytes': total_bytes,
            'gen_tokens': gen_tokens
        }
        return output

    def patched_diffusion_sample(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        output = orig_diffusion_sample(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        
        profiling_data['expert_latency_ms'] = start_event.elapsed_time(end_event)
        return output

    # ---------------------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------------------
    print("[2] GPU Warmup 실행 중...")
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=0.98, temperature=0.6, num_traj_samples=1, max_coc_tokens=4
        )

    print("\n[3] 정밀 프로파일링 시작...")
    
    # Run 1: Prefill 위주 측정 (max_coc_tokens=0)
    print(" -> (Phase 1) Vision Prefill 프로파일링...")
    with patch.object(model.vlm, 'generate', side_effect=patched_generate):
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.98, temperature=0.6, num_traj_samples=1, max_coc_tokens=0
            )

    # Run 2: Full Decode 및 Expert 연산 측정 (max_coc_tokens=128)
    print(" -> (Phase 2) VLM Decode & Action Expert 프로파일링...")
    with patch.object(model.vlm, 'generate', side_effect=patched_generate):
        with patch.object(model.diffusion, 'sample', side_effect=patched_diffusion_sample):
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs, top_p=0.98, temperature=0.6, num_traj_samples=1, max_coc_tokens=128
                )

    # ---------------------------------------------------------------------
    # 데이터 역산 및 리포팅
    # ---------------------------------------------------------------------
    r1 = profiling_data['run1']
    r2 = profiling_data['run2']
    expert_ms = profiling_data.get('expert_latency_ms', 0.0)
    
    # 1. Decode 토큰당 소요 시간 계산
    tokens_diff = r2['gen_tokens'] - r1['gen_tokens']
    time_diff_ms = r2['latency_ms'] - r1['latency_ms']
    
    decode_per_token_ms = time_diff_ms / tokens_diff if tokens_diff > 0 else 0
    
    # 2. 순수 Prefill 시간 추정 (Run 1 시간에서 Run 1이 생성한 잉여 토큰 디코드 시간 차감)
    prefill_ms = r1['latency_ms'] - (r1['gen_tokens'] * decode_per_token_ms)
    
    # 3. KV Cache 크기 (MB)
    kv_mb = r2['kv_cache_bytes'] / (1024 * 1024)
    
    # 4. 메모리 대역폭 소모량 역산 (10B 파라미터, bf16 = 20GB per token)
    model_size_gb = 20.0
    decode_bandwidth_gbps = model_size_gb / (decode_per_token_ms / 1000.0) if decode_per_token_ms > 0 else 0

    print("\n" + "="*60)
    print("               [ 프로파일링 결과 보고서 ]               ")
    print("="*60)
    print(f"1. [CPU-GPU 통신 크기] Vision KV Cache Size: {kv_mb:.2f} MB")
    print(f"   -> Vision을 CPU로 내릴 시, 프레임당 GPU로 복사해야 할 데이터량입니다.")
    print(f"")
    print(f"2. [Vision Prefill 병목] 순수 Prefill 지연 시간: {prefill_ms:.2f} ms")
    print(f"   -> 이 시간이 바로 우리가 '파이프라이닝'으로 숨겨야 할(Hide) 목표 시간입니다.")
    print(f"")
    print(f"3. [VLM Decode 병목] 토큰당 생성 시간: {decode_per_token_ms:.2f} ms/token")
    print(f"   -> VLM Decode 소모 대역폭: {decode_bandwidth_gbps:.2f} GB/s")
    print(f"   -> (참고: 앞선 Trade-off 실험의 GPU 단독 속도 한계치와 일치하는지 확인 필요)")
    print(f"")
    print(f"4. [Action Expert 병목] Diffusion 소요 시간: {expert_ms:.2f} ms")
    print(f"   -> 디퓨전 궤적 예측에 소요되는 시간입니다.")
    print("="*60)

if __name__ == "__main__":
    main()

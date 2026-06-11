import multiprocessing as mp
import time
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------------------------------------------------------
# GPU-CPU 대역폭 Trade-off Sweep 벤치마크
# ---------------------------------------------------------
# GPU가 100% 부하로 VRAM을 긁어오는 동안,
# CPU 코어를 0개부터 12개까지 순차적으로 투입하면서
# 양측의 대역폭 변화 곡선을 그려 "최적의 교차점(Sweet Spot)"을 찾습니다.
# ---------------------------------------------------------

def cpu_worker(worker_id, num_iters, data_size_mb, start_event):
    torch.set_num_threads(1)
    num_elements = (data_size_mb * 1024 * 1024) // 4
    x = torch.ones(num_elements, dtype=torch.float32)
    
    for _ in range(2):
        _ = x.sum()
        
    start_event.wait()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = x.sum()
    end = time.perf_counter()
    
    duration = end - start
    bytes_read = num_iters * data_size_mb * 1024 * 1024
    return worker_id, bytes_read, duration

def gpu_worker(num_iters, data_size_mb, start_event, result_dict):
    num_elements = (data_size_mb * 1024 * 1024) // 4
    x = torch.ones(num_elements, dtype=torch.float32, device='cuda')
    
    for _ in range(5):
        _ = x.sum()
    torch.cuda.synchronize()
    
    start_event.wait()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = x.sum()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    bytes_read = num_iters * data_size_mb * 1024 * 1024
    result_dict['gpu_bytes'] = bytes_read
    result_dict['gpu_duration'] = end - start


def run_tradeoff_sweep(max_cores, cpu_iters, gpu_iters, data_size_mb):
    print("\n[Trade-off Sweep] CPU 활성 코어 증가에 따른 대역폭 경합 곡선 측정")
    print(f"조건: {data_size_mb}MB 독립 배열. GPU는 100% 부하 유지.")
    
    # 0코어(GPU 단독)부터 12코어까지 점진적으로 늘려갑니다.
    core_sweep = [0, 1, 2, 4, 6, 8, 10, 12]
    core_sweep = [c for c in core_sweep if c <= max_cores]
    if max_cores not in core_sweep:
        core_sweep.append(max_cores)
        
    results = {
        'cpu_bw': [],
        'gpu_bw': [],
        'total_bw': []
    }
    
    for num_cpu_cores in core_sweep:
        print(f" -> 측정 중... [ GPU 100% 부하 + CPU {num_cpu_cores}코어 개입 ]")
        manager = mp.Manager()
        result_dict = manager.dict()
        start_event = manager.Event()
        
        cpu_results = []
        pool = None
        if num_cpu_cores > 0:
            pool = mp.Pool(processes=num_cpu_cores)
            for i in range(num_cpu_cores):
                res = pool.apply_async(cpu_worker, args=(i, cpu_iters, data_size_mb, start_event))
                cpu_results.append(res)
                
        gpu_proc = mp.Process(target=gpu_worker, args=(gpu_iters, data_size_mb, start_event, result_dict))
        gpu_proc.start()
        
        # 메모리 할당 및 준비 대기
        time.sleep(3)
        start_event.set() # 완벽한 동시 출발
        
        total_cpu_bytes = 0
        max_cpu_duration = 0
        if num_cpu_cores > 0:
            for res in cpu_results:
                w_id, b, d = res.get()
                total_cpu_bytes += b
                max_cpu_duration = max(max_cpu_duration, d)
            pool.close()
            pool.join()
            
        gpu_proc.join()
        
        # 처리율(대역폭) 계산
        cpu_bw_gbps = (total_cpu_bytes / 1e9) / max_cpu_duration if max_cpu_duration > 0 else 0.0
        gpu_bw_gbps = (result_dict['gpu_bytes'] / 1e9) / result_dict['gpu_duration']
        total_bw = cpu_bw_gbps + gpu_bw_gbps
        
        results['cpu_bw'].append(cpu_bw_gbps)
        results['gpu_bw'].append(gpu_bw_gbps)
        results['total_bw'].append(total_bw)
        
        print(f"    결과: CPU {cpu_bw_gbps:.1f} GB/s | GPU {gpu_bw_gbps:.1f} GB/s | Total {total_bw:.1f} GB/s")
        
    return core_sweep, results

def plot_pareto_curve(core_sweep, results):
    print("\n[Plotting] 교수님 보고용 트레이드오프 파레토 곡선 렌더링 중...")
    save_path = os.path.join(os.path.dirname(__file__), '260528_대역폭_파레토_곡선.png')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#F8F9FA')
    
    # 꺾은선 그래프 그리기
    ax1.plot(core_sweep, results['gpu_bw'], marker='s', linewidth=3, markersize=10, color='#E67E22', label='GPU Bandwidth (VLM Decode)')
    ax1.plot(core_sweep, results['cpu_bw'], marker='o', linewidth=3, markersize=10, color='#2B5B84', label='CPU Bandwidth (Vision Pipeline)')
    ax1.plot(core_sweep, results['total_bw'], marker='^', linewidth=2, markersize=8, color='#27AE60', linestyle='--', label='System Total Bandwidth')
    
    ax1.set_title('CPU-GPU Memory Bandwidth Pareto Trade-off', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel('Number of Active CPU Cores (Interference Level)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12, fontweight='bold')
    ax1.set_xticks(core_sweep)
    
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11)
    
    # 주요 지점 수치 표기 (0, 4, 8, 12 코어)
    for i, c in enumerate(core_sweep):
        if c in [0, 4, 8, 12]: 
            ax1.annotate(f"{results['gpu_bw'][i]:.1f}", (c, results['gpu_bw'][i]), textcoords="offset points", xytext=(0,10), ha='center', color='#D35400', fontweight='bold')
            ax1.annotate(f"{results['cpu_bw'][i]:.1f}", (c, results['cpu_bw'][i]), textcoords="offset points", xytext=(0,-15), ha='center', color='#1A5276', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" -> 렌더링 완료: {save_path}")

if __name__ == '__main__':
    MAX_CORES = 12
    DATA_SIZE_MB = 500
    
    # Iteration 조정 (측정 시간을 길게 가져가 안정도 향상)
    CPU_ITERS = 100
    GPU_ITERS = 300
    
    mp.set_start_method('spawn', force=True)
    
    print("=" * 65)
    print("   Alpamayo CPU-GPU Bandwidth Trade-off Sweep Benchmark   ")
    print("=" * 65)
    
    core_sweep, results = run_tradeoff_sweep(MAX_CORES, CPU_ITERS, GPU_ITERS, DATA_SIZE_MB)
    plot_pareto_curve(core_sweep, results)
    
    print("\n[벤치마크 완료]")

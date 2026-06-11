import multiprocessing as mp
import time
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------------------------------------------------------
# 실험 설계 설명 (대역폭 측정의 정석)
# ---------------------------------------------------------
# 왜 각 프로세스마다 별도의 거대한 데이터를 할당해야 하는가?
# 만약 모든 CPU 코어가 같은 메모리 주소를 읽는다면, 가장 처음 읽은 코어가 데이터를 L3/LLC 캐시에 올리고,
# 나머지 코어들은 초고속 캐시에서 데이터를 가져오게 됩니다. 이는 "캐시 대역폭"을 측정하는 꼴이 되어,
# 진정한 의미의 DRAM 대역폭(통합 메모리 한계)을 측정할 수 없습니다.
# 따라서 각 코어(프로세스)마다 독립적으로 500MB 이상의 거대한 데이터를 물리적으로 할당(torch.ones)하여
# L3 캐시(일반적으로 수십 MB 이하)를 완벽히 뚫어버리고(Cache thrashing), 무조건 DRAM에서 데이터를 긁어오도록 강제합니다.
# ---------------------------------------------------------

def cpu_worker(worker_id, num_iters, data_size_mb, start_event):
    """
    개별 CPU 코어가 독립적인 메모리 블록을 할당받아 읽기 연산을 수행하는 워커
    """
    # PyTorch의 내부 멀티스레딩이 개입하여 CPU 코어를 과다점유하지 않도록 1로 고정
    torch.set_num_threads(1)
    
    # 독립적인 물리 메모리 공간 할당 (float32 = 4 bytes)
    # torch.empty 대신 torch.ones를 써서 OS가 실제로 페이지를 물리 메모리에 매핑하도록 강제함
    num_elements = (data_size_mb * 1024 * 1024) // 4
    x = torch.ones(num_elements, dtype=torch.float32)
    
    # Warmup (캐시 초기화 및 OS 오버헤드 완화)
    for _ in range(2):
        _ = x.sum()
        
    # 완벽한 동시 출발을 위해 다른 프로세스가 모두 준비될 때까지 대기
    start_event.wait()
    
    # 순수 DRAM 대역폭 측정 시작
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = x.sum()
    end = time.perf_counter()
    
    duration = end - start
    bytes_read = num_iters * data_size_mb * 1024 * 1024
    return worker_id, bytes_read, duration

def gpu_worker(num_iters, data_size_mb, start_event, result_dict):
    """
    GPU가 거대한 텐서를 연속적으로 읽어 대역폭을 최대치로 끌어올리는 워커
    """
    num_elements = (data_size_mb * 1024 * 1024) // 4
    # GPU VRAM (iGPU 환경에선 CPU와 동일한 물리적 통합 메모리 공간)에 할당
    x = torch.ones(num_elements, dtype=torch.float32, device='cuda')
    
    # Warmup
    for _ in range(5):
        _ = x.sum()
    torch.cuda.synchronize()
    
    # CPU 프로세스들과 동시 출발 대기
    start_event.wait()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = x.sum()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    bytes_read = num_iters * data_size_mb * 1024 * 1024
    result_dict['gpu_bytes'] = bytes_read
    result_dict['gpu_duration'] = end - start


def run_cpu_scaling_test(max_cores, cpu_iters, data_size_mb):
    print("\n[Part A] 다중 CPU 코어 대역폭 스케일링 측정 (CPU Only)")
    print(f"조건: {data_size_mb}MB 크기의 독립적인 배열 / 코어당, 반복 {cpu_iters}회")
    
    core_counts = [1, 2, 4, 8, 12]
    # 실험 환경 코어 수에 맞게 필터링
    core_counts = [c for c in core_counts if c <= max_cores]
    if max_cores not in core_counts:
        core_counts.append(max_cores)
        
    results = {}
    
    for num_cores in core_counts:
        print(f" -> {num_cores} 코어 측정 중...")
        manager = mp.Manager()
        start_event = manager.Event()
        pool = mp.Pool(processes=num_cores)
        async_results = []
        
        for i in range(num_cores):
            res = pool.apply_async(cpu_worker, args=(i, cpu_iters, data_size_mb, start_event))
            async_results.append(res)
            
        # 모든 프로세스가 거대 배열 할당을 완료할 수 있도록 여유 부여
        time.sleep(3)
        start_event.set() # 동시에 출발!
        
        total_bytes = 0
        max_duration = 0
        for res in async_results:
            w_id, b, d = res.get()
            total_bytes += b
            max_duration = max(max_duration, d)
            
        pool.close()
        pool.join()
        
        # 총 읽어들인 데이터를 가장 늦게 끝난 프로세스의 시간으로 나누어 묶음 대역폭 도출
        bandwidth_gbps = (total_bytes / 1e9) / max_duration
        results[num_cores] = bandwidth_gbps
        print(f"    결과: {bandwidth_gbps:.2f} GB/s")
        
    return core_counts, results


def run_contention_test(num_cpu_cores, cpu_iters, gpu_iters, data_size_mb):
    mode = "CPU + GPU 동시 실행" if num_cpu_cores > 0 else "GPU 단독 실행"
    print(f"\n[Part B] {mode} 측정")
    
    manager = mp.Manager()
    result_dict = manager.dict()
    start_event = manager.Event()
    
    # 1. CPU Pool 설정 (필요 시)
    cpu_results = []
    pool = None
    if num_cpu_cores > 0:
        pool = mp.Pool(processes=num_cpu_cores)
        for i in range(num_cpu_cores):
            res = pool.apply_async(cpu_worker, args=(i, cpu_iters, data_size_mb, start_event))
            cpu_results.append(res)
            
    # 2. GPU 프로세스 시작
    gpu_proc = mp.Process(target=gpu_worker, args=(gpu_iters, data_size_mb, start_event, result_dict))
    gpu_proc.start()
    
    # 메모리 할당 대기
    print(" -> 거대 메모리 공간 확보 중... (잠시 대기)")
    time.sleep(5)
    
    # 3. 측정 시작
    print(" -> 대역폭 벤치마크 시작!")
    start_event.set()
    
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
    
    # 결과 계산
    cpu_bw_gbps = (total_cpu_bytes / 1e9) / max_cpu_duration if max_cpu_duration > 0 else 0.0
    gpu_bw_gbps = (result_dict['gpu_bytes'] / 1e9) / result_dict['gpu_duration']
    
    if num_cpu_cores > 0:
        print(f"    동시 실행 결과 -> CPU: {cpu_bw_gbps:.2f} GB/s | GPU: {gpu_bw_gbps:.2f} GB/s")
        print(f"    총 대역폭 (Total): {cpu_bw_gbps + gpu_bw_gbps:.2f} GB/s")
    else:
        print(f"    GPU 단독 대역폭: {gpu_bw_gbps:.2f} GB/s")
        
    return cpu_bw_gbps, gpu_bw_gbps


def plot_results(cpu_core_counts, cpu_only_bw, cpu_contention_bw, gpu_only_bw, gpu_contention_bw, max_cores):
    print("\n[Part C] 실험 결과 시각화 (Figure 생성)")
    save_path = os.path.join(os.path.dirname(__file__), 'bandwidth_benchmark_results.png')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#F8F9FA')
    
    # ---------------------------------------------------------
    # Figure 1: CPU 코어 수에 따른 대역폭 스케일링 (Line Plot)
    # ---------------------------------------------------------
    x_cores = cpu_core_counts
    y_bw = [cpu_only_bw[c] for c in x_cores]
    
    ax1.plot(x_cores, y_bw, marker='o', linewidth=3, markersize=10, color='#2B5B84', label='Total CPU Bandwidth')
    ax1.set_title('CPU Core Scaling: Independent Data Read Bandwidth', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Number of CPU Cores', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_cores)
    ax1.set_ylim(0, max(y_bw) * 1.2)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 데이터 포인트 위에 수치 표시
    for i, txt in enumerate(y_bw):
        ax1.annotate(f'{txt:.1f}', (x_cores[i], y_bw[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

    # ---------------------------------------------------------
    # Figure 2: 통합 메모리에서의 CPU-GPU 경합 (Stacked Bar)
    # ---------------------------------------------------------
    labels = ['GPU Only', f'CPU Only\n({max_cores} cores)', 'CPU + GPU\n(Simultaneous)']
    
    gpu_bars = [gpu_only_bw, 0, gpu_contention_bw]
    cpu_bars = [0, cpu_only_bw[max_cores], cpu_contention_bw]
    
    x_pos = np.arange(len(labels))
    width = 0.55
    
    p1 = ax2.bar(x_pos, gpu_bars, width, label='GPU Bandwidth', color='#E67E22', edgecolor='white', linewidth=2)
    p2 = ax2.bar(x_pos, cpu_bars, width, bottom=gpu_bars, label='CPU Bandwidth', color='#2B5B84', edgecolor='white', linewidth=2)
    
    ax2.set_title('iGPU Shared Memory Contention (Bandwidth Starvation)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax2.set_ylim(0, max([g+c for g,c in zip(gpu_bars, cpu_bars)]) * 1.25)
    ax2.legend(loc='upper right', fontsize=11)
    
    # Bar 위에 숫자 표기 및 Total 표기
    for i, (g_val, c_val) in enumerate(zip(gpu_bars, cpu_bars)):
        total_val = g_val + c_val
        if g_val > 0:
            ax2.text(i, g_val/2, f'{g_val:.1f}', ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        if c_val > 0:
            ax2.text(i, g_val + c_val/2, f'{c_val:.1f}', ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        ax2.text(i, total_val + (total_val * 0.02), f'Total: {total_val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12, color='#333333')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" -> 시각화 이미지가 저장되었습니다: {save_path}")


if __name__ == '__main__':
    # Thor 보드는 12코어 ARM CPU이므로, 기본 설정을 12로 둡니다. (사용자 환경에 맞게 자동 조절 가능)
    MAX_CORES = 12
    # L3/LLC 캐시 히트를 완벽히 방지하기 위해 1코어당 500MB의 데이터를 할당합니다.
    DATA_SIZE_MB = 500
    
    # 실험에 걸리는 시간을 맞추기 위해 반복 횟수 조절
    CPU_ITERS = 100
    GPU_ITERS = 300 # GPU 연산 속도가 빠르므로 횟수를 늘림
    
    mp.set_start_method('spawn', force=True)
    
    print("=" * 65)
    print("   Alpamayo iGPU Memory Bandwidth Benchmark Suite (Strict)   ")
    print("=" * 65)
    print("※ 본 벤치마크는 모든 프로세스(CPU, GPU)에 독립적인 초대형(500MB+) ")
    print("물리 메모리를 할당하여 캐시 오버헤드를 원천 차단하고 순수 DRAM 대역폭을 측정합니다.")
    print("=" * 65)
    
    # 1. 다중 CPU 코어 대역폭 측정
    core_counts, cpu_only_results = run_cpu_scaling_test(
        max_cores=MAX_CORES, 
        cpu_iters=CPU_ITERS, 
        data_size_mb=DATA_SIZE_MB
    )
    
    # 2. GPU 단독 대역폭 측정
    print("\n[Part B-1] GPU 단독 대역폭 측정 준비...")
    _, gpu_only_bw = run_contention_test(
        num_cpu_cores=0, 
        cpu_iters=0, 
        gpu_iters=GPU_ITERS, 
        data_size_mb=DATA_SIZE_MB
    )
    
    # 3. CPU + GPU 동시 경합 측정
    print(f"\n[Part B-2] CPU ({MAX_CORES}코어) + GPU 동시 측정 준비...")
    cpu_contention_bw, gpu_contention_bw = run_contention_test(
        num_cpu_cores=MAX_CORES, 
        cpu_iters=CPU_ITERS, 
        gpu_iters=GPU_ITERS, 
        data_size_mb=DATA_SIZE_MB
    )
    
    # 4. 시각화 (Figure 생성)
    plot_results(
        cpu_core_counts=core_counts,
        cpu_only_bw=cpu_only_results,
        cpu_contention_bw=cpu_contention_bw,
        gpu_only_bw=gpu_only_bw,
        gpu_contention_bw=gpu_contention_bw,
        max_cores=MAX_CORES
    )
    
    print("\n[모든 벤치마크가 성공적으로 완료되었습니다.]")

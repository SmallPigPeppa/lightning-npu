import torch
import time
import torch_npu
import transfer_to_npu

# 检查是否有足够的GPU
num_gpus = 8
assert torch.cuda.device_count() >= num_gpus, "不足8个可用的GPU"


# 定义在每个GPU上执行的计算函数
def run_computation(device_id):
    # 设置当前线程的默认GPU
    torch.cuda.set_device(device_id)

    # 创建两个随机矩阵
    a = torch.randn(2048, 2048, device=f'cuda:{device_id}')
    b = torch.randn(2048, 2048, device=f'cuda:{device_id}')

    while True:
        # 执行矩阵乘法和加法
        c = torch.matmul(a, b)
        a += c


# 导入线程库
import threading

# 创建和启动线程
threads = []
for i in range(num_gpus):
    thread = threading.Thread(target=run_computation, args=(i,))
    thread.start()
    threads.append(thread)

# 主线程中不进行计算，仅保持子线程活跃
for thread in threads:
    thread.join()

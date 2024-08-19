import torch
import time
import torch_npu
import transfer_to_npu

num_gpus = 8
assert torch.cuda.device_count() >= num_gpus, "不足8个可用的GPU"


def run_computation(device_id):
    torch.cuda.set_device(device_id)
    a = torch.randn(2500, 1024, 1024, device=f'cuda:{device_id}')
    b = torch.randn(2500, 1024, 1024, device=f'cuda:{device_id}')

    while True:
        c = torch.matmul(a, b)
        a += c


import threading

threads = []
for i in range(num_gpus):
    thread = threading.Thread(target=run_computation, args=(i,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

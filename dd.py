import torch
import time
import torch_npu
from torch_npu.contrib import transfer_to_npu

num_gpus = 8
assert torch.cuda.device_count() >= num_gpus, "不足8个可用的GPU"

def run_computation(device_id):
    a = torch.randn(2500, 1024, 1024, device=f'cuda:{device_id}')
    b = torch.randn(2500, 1024, 1024, device=f'cuda:{device_id}')
    c = torch.matmul(a, b)
    a += c

while True:
    for i in range(num_gpus):
        print(f"Running on device {i}")
        run_computation(i)

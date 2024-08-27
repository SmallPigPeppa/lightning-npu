import torch
import torch_npu

num_gpus = 8

def run_computation(device_id):
    a = torch.randn(2500, 1024, 1024, device=f'npu:{device_id}')
    b = torch.randn(2500, 1024, 1024, device=f'npu:{device_id}')
    c = torch.matmul(a, b)
    a += c

while True:
    for i in range(num_gpus):
        print(f"Running on device {i}")
        run_computation(i)

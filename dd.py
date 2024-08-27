import os
import torch
import torch_npu
os.system('export ASCEND_LAUNCH_BLOCKING=1')
num_gpus = 8


def run_computation(device_id):
    a = torch.ones(size=[2500, 1024, 1024], device=f'npu:{device_id}')
    b = torch.ones(size=[2500, 1024, 1024], device=f'npu:{device_id}')
    c = a * b


while True:
    for i in range(num_gpus):
        run_computation(i)

import random
import torch
import time

import torch_npu
from torch_npu.contrib import transfer_to_npu

import os

os.system('export ASCEND_LAUNCH_BLOCKING=1')
os.system('export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False')


def gpu_turbo(eps):
    a=[]
    b=[]
    dev_cnt = torch.cuda.device_count()
    for i in range(dev_cnt):
        #a.append(torch.rand(3000,900000).to('cuda:'+str(i)))
        a.append(torch.rand(1000,400000).to('cuda:'+str(i)))
        b.append(torch.rand(1000,400000).to('cuda:'+str(i)))

    while True:
        if random.random() < eps:
            for i in range(dev_cnt):
                b[i] = torch.exp(a[i])
        # time.sleep(random.random() * 0.005)


if __name__ == '__main__':
    print("Occupying everything")
    gpu_turbo(0.5)
import torch
import torch_npu
# from torch_npu.contrib i
mport (transfer_to_npu)


a = torch.tensor([1, 2]).cuda()
b = a * a
print(b)

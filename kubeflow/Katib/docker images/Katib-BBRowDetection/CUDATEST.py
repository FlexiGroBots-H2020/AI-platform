import torch
import sys
print("--------------------PYTHON VERZIJA-----------------------------")
print(sys.path)
print(sys.executable)
print(sys.version)
print("--------------------CUDA VERZIJA-----------------------------")
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print("-----------------------------------------------------")
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))

x = torch.randn(1, 3, 224, 224, device='cuda')
conv = torch.nn.Conv2d(3, 3, 3).cuda()

out = conv(x)
print(out.shape)
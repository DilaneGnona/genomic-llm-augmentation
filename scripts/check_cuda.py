import torch
print('cuda_available=', torch.cuda.is_available())
print('cuda_version=', torch.version.cuda)
print('device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device_name=', torch.cuda.get_device_name(0))


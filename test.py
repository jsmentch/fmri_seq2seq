import torch

a = torch.empty(16, 9)

for i in range(a.size()[0]):
    a[i,:] = i

b = torch.reshape(a,(a.size()[0],3,3))

print(b)
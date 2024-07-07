import torch
import numpy as np
a = [torch.tensor(1),torch.tensor(2),torch.tensor(3)]
b = [4,5,6]
c = torch.stack(a)
print(c)

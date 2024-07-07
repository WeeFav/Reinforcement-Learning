import torch
import numpy as np
a = np.array([(1,2,3), (4,5,6)])
a = torch.from_numpy(a)
idx = torch.from_numpy(np.array([0,1], dtype=np.int64)).unsqueeze(1)
b = torch.gather(input=a, dim=1, index=idx)
print(b)
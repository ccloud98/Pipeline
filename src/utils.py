import torch
import numpy as np
def padded_avg(X, pad):
  X_s = X.shape
  n = torch.sum(pad, dim=1).unsqueeze(1)
  while (len(pad.shape) < len(X_s)):
    pad = pad.unsqueeze(-1)
  return torch.sum(X * pad, dim=1) / n

def mean_FM(E):
  return torch.mean(torch.stack(E, dim=1), dim=1)

def get_device():
  if torch.cuda.is_available():
    dev = torch.device('cuda')
  else:
    dev = torch.device('cpu')
  return dev

# def array_mapping(source, target):
#   return  np.array([np.where(target== pid)[0][0] for pid in source])
def array_mapping(source, target):
    result = []
    for pid in source:
        matches = np.where(target == pid)[0]
        if len(matches) > 0:
            result.append(matches[0])
        else:
            #print(f"Warning: pid {pid} not found in target array.")
            # 필요한 경우 다른 값을 추가하거나, None으로 처리할 수 있습니다.
            result.append(-1)  # -1로 처리할 경우 나중에 필터링 필요

    return np.array(result)
import torch
import numpy as np

from torch.utils.data.dataloader import default_collate

def fc_collate_fn(batch):  
  result_batch = {}
  
  for key in batch[0].keys():
    if 'other' in key or 'center_info' in key:
        result_batch[key] = [item[key] for item in batch]
    else:        
        result_batch[key] = default_collate([item[key] for item in batch])

  return result_batch


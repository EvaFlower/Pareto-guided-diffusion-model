import torch
import numpy as np


PATCH_SIZE = 8

def function_11(x, patch_size=PATCH_SIZE, bound=1e-4, reduce=True, con_t=1):
    begin_index =int((32-patch_size)/2-1)
    patch_x = x[:, :, begin_index:begin_index+patch_size, begin_index:begin_index+patch_size]
    if reduce:
        var = torch.mean((patch_x-con_t).pow(2))-bound
    else:
        var = torch.mean((patch_x-con_t).pow(2), dim=(1, 2, 3))-bound
    return var

def function_12(x, patch_size=PATCH_SIZE, bound=1e-4, reduce=True, con_t=0.5):
    begin_index =int((32-patch_size)/2-1)
    patch_x = x[:, :, begin_index:begin_index+patch_size, begin_index:begin_index+patch_size]
    if reduce:
        var = torch.mean((patch_x-con_t).pow(2))-bound
    else:
        var = torch.mean((patch_x-con_t).pow(2), dim=(1, 2, 3))-bound
    return var

def bi_objectives(x, bound=1e-4, reduce=True):
    f1 = function_11(x, reduce=reduce)
    f2 = function_12(x, reduce=reduce)
    return [f1, f2]




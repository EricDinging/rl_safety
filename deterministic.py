import numpy
import torch
import torch.nn as nn
n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01

data_x = torch.randn(batch_size, n_input)
data_y = (torch.rand(size=(batch_size, 1)) < 0.5).float()   
print(data_x)
print(data_y)
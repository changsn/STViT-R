import torch.nn.functional as F
import torch.nn as nn
import torch


class multi_scale_semantic_token1(nn.Module):
    def __init__(self, sample_window_size):
        super().__init__()
        self.sample_window_size = sample_window_size
        self.num_samples = sample_window_size * sample_window_size

    def forward(self, x):
        B, C, _, _ = x.size()
        pool_x = F.adaptive_max_pool2d(x, (self.sample_window_size, self.sample_window_size)).view(B, C, self.num_samples).transpose(2, 1)
        return pool_x

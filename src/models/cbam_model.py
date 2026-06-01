import torch
import torch.nn as nn

class DummyAttentionLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.ones(1, channels, 1, 1))
        
    def forward(self, x):
        return x * self.attention_weights
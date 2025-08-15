# model_be.py
import torch.nn as nn
from model import my_model as _BaseModel

class my_model(nn.Module):
    """
    B+E (expansion-only): just pass expansion_factor=2.0 into your baseline.
    Everything else (width, depth/heads, attention, SFCA) stays baseline.
    """
    def __init__(self,
                 num_blocks=[2, 3, 3, 4],
                 num_heads=[1, 2, 4, 8],
                 channels=[16, 32, 64, 128],
                 num_refinement=4,
                 expansion_factor=2.0,   # <-- the only change vs baseline (2.66)
                 ch=[64, 32, 16, 64],
                 *args, **kwargs):
        super().__init__()
        self.inner = _BaseModel(num_blocks=num_blocks,
                                num_heads=num_heads,
                                channels=channels,
                                num_refinement=num_refinement,
                                expansion_factor=expansion_factor,
                                ch=ch, *args, **kwargs)

    def forward(self, x):
        return self.inner(x)

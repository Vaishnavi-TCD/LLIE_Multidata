# model_bse.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import my_model as _BaseModel
from model import AGSSF  # reuse your AGSSF

class SFCA_SE(nn.Module):
    """
    SFCA variant without FFT branch: lightweight SE gate + AGSSF.
    I/O shapes match the original SFCA.
    """
    def __init__(self, channels, relu_slope=0.2, gamma=2):
        super().__init__()
        self.identity1 = nn.Conv2d(channels, channels, 1)
        self.conv_1 = nn.Conv2d(channels, 2*channels, kernel_size=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope)
        self.conv_2 = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope)

        # SE gate (squeeze-excite)
        squeeze = max(1, channels // 16)
        self.se_reduce = nn.Conv2d(channels, squeeze, 1, bias=True)
        self.se_expand = nn.Conv2d(squeeze, channels, 1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.agssf = AGSSF(channels)

    def forward(self, x):
        out = self.conv_1(x)
        a, b = torch.chunk(out, 2, dim=1)
        out = torch.cat([a, b], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out = out + self.identity1(x)

        w = torch.sigmoid(self.se_expand(F.leaky_relu(self.se_reduce(self.pool(out)), negative_slope=0.2)))
        out = out * w
        return self.agssf(out)


class my_model(nn.Module):
    """
    Wrapper that builds the baseline, then replaces the SFCA modules
    in 'self.attention' with SFCA_SE for each channel in 'ch'.
    """
    def __init__(self,
                 num_blocks=[2, 3, 3, 4],
                 num_heads=[1, 2, 4, 8],
                 channels=[16, 32, 64, 128],
                 num_refinement=4,
                 expansion_factor=2.66,
                 ch=[64, 32, 16, 64],
                 *args, **kwargs):
        super().__init__()
        self.inner = _BaseModel(num_blocks=num_blocks,
                                num_heads=num_heads,
                                channels=channels,
                                num_refinement=num_refinement,
                                expansion_factor=expansion_factor,
                                ch=ch, *args, **kwargs)
        # Replace SFCA modules
        assert hasattr(self.inner, 'attention'), "Expected 'attention' ModuleList on baseline model."
        for i, c in enumerate(ch):
            self.inner.attention[i] = SFCA_SE(c)

    def forward(self, x):
        return self.inner(x)

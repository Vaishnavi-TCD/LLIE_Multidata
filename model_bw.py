# model_bw.py
import torch
import torch.nn as nn

# Reuse your existing baseline definitions
from model import my_model as _BaseModel
from model import UpS  # reusing your UpS implementation so behavior stays identical


class my_model(nn.Module):
    """
    B+W (width-only) ablation wrapper:
      • channels set to [12, 24, 48, 96]
      • UpS and reduce 1x1 layers re-wired to match the new widths
      • output head adjusted to take channels[1] instead of fixed 32
    Everything else remains baseline (depth, heads, expansion, attention, etc.).
    """
    def __init__(
        self,
        num_blocks=[2, 3, 3, 4],
        num_heads=[1, 2, 4, 8],
        channels=[12, 24, 48, 96],   # <- width-only change
        num_refinement=4,
        expansion_factor=2.66,
        ch=[64, 32, 16, 64],
        *args, **kwargs
    ):
        super().__init__()
        self.channels = channels

        # Build the baseline model BUT with our narrower channels
        self.inner = _BaseModel(
            num_blocks=num_blocks,
            num_heads=num_heads,
            channels=channels,
            num_refinement=num_refinement,
            expansion_factor=expansion_factor,
            ch=ch,
            *args, **kwargs
        )

        # Rewire upsample modules to match width change
        # (baseline had: UpS(128), UpS(64), UpS(32))
        self.inner.ups_1 = UpS(channels[3])   # stage3 -> stage2
        self.inner.ups_2 = UpS(channels[2])   # stage2 -> stage1
        self.inner.ups_3 = UpS(channels[1])   # stage1 -> stage0

        # Fix reducer 1x1 convs after concat of (up + skip)
        # reducer1 input = (channels[3] // 2) + channels[2]  →  channels[2]
        # reducer2 input = (channels[2] // 2) + channels[1]  →  channels[1]
        self.inner.reduces1 = nn.Conv2d(
            (channels[3] // 2) + channels[2], channels[2], kernel_size=1, bias=False
        )
        self.inner.reduces2 = nn.Conv2d(
            (channels[2] // 2) + channels[1], channels[1], kernel_size=1, bias=False
        )

        # Final head: ensure the penultimate conv takes channels[1] (24) instead of fixed 32
        self.inner.outputl = nn.Conv2d(channels[1], 8, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.inner(x)

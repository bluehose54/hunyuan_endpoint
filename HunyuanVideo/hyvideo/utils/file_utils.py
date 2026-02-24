import os
from pathlib import Path
from einops import rearrange

import torch
import numpy as np
import imageio

CODE_SUFFIXES = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
}


def safe_dir(path):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def safe_file(path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def make_grid_torch(x: torch.Tensor, nrow=1):
    """
    Minimal replacement for torchvision.utils.make_grid.
    x: [B, C, H, W]
    """
    if nrow <= 1:
        return x[0]

    B, C, H, W = x.shape
    rows = []
    for i in range(0, B, nrow):
        row = torch.cat(x[i:i+nrow], dim=2)  # concat width
        rows.append(row)
    grid = torch.cat(rows, dim=1)  # concat height
    return grid


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """
    Save video tensor as mp4 without torchvision dependency.

    videos: [B, C, T, H, W]
    """

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []

    for x in videos:
        # x: [B, C, H, W]
        x = make_grid_torch(x, nrow=n_rows)

        x = x.permute(1, 2, 0)  # [H, W, C]

        if rescale:
            x = (x + 1.0) / 2.0

        x = torch.clamp(x, 0, 1)

        if x.dtype.is_floating_point:
            x = (x * 255).to(torch.uint8)

        outputs.append(x.cpu().numpy())

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

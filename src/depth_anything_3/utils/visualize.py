# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

import matplotlib
import numpy as np
import torch

from einops import rearrange

from depth_anything_3.utils.logger import logger


# ============================================================
# VISUALIZAÇÃO NUMPY
# ============================================================

def visualize_depth(
    depth: np.ndarray,
    depth_min=None,
    depth_max=None,
    percentile=2,
    ret_minmax=False,
    ret_type=np.uint8,

    # ========================================================
    # ALTERAÇÃO PRINCIPAL
    # ========================================================

    cmap="Spectral",
):
    """
    Visualize depth map using Spectral colormap.

    PADRONIZAÇÃO:
    - Igual ao FoundationStereo / DA2
    - Sem inversão
    - Sem 1/depth
    - Sem log
    - Normalização linear min/max
    """

    depth = depth.copy()

    # ========================================================
    # MÁSCARA VÁLIDA
    # ========================================================

    valid_mask = depth > 0

    # ========================================================
    # DEPTH MIN
    # ========================================================

    if depth_min is None:

        if valid_mask.sum() <= 10:
            depth_min = 0

        else:
            depth_min = depth[valid_mask].min()

    # ========================================================
    # DEPTH MAX
    # ========================================================

    if depth_max is None:

        if valid_mask.sum() <= 10:
            depth_max = 0

        else:
            depth_max = depth[valid_mask].max()

    # ========================================================
    # EVITAR DIVISÃO POR ZERO
    # ========================================================

    if depth_min == depth_max:

        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6

    # ========================================================
    # NORMALIZAÇÃO LINEAR
    # ========================================================

    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)

    # ========================================================
    # COLORMAP SPECTRAL
    # ========================================================

    cm = matplotlib.colormaps[cmap]

    img_colored_np = cm(
        depth[None],
        bytes=False
    )[:, :, :, 0:3]

    # ========================================================
    # CONVERSÃO DE TIPO
    # ========================================================

    if ret_type == np.uint8:

        img_colored_np = (
            img_colored_np[0] * 255.0
        ).astype(np.uint8)

    elif ret_type == np.float32 or ret_type == np.float64:

        img_colored_np = img_colored_np[0]

    else:

        raise ValueError(f"Invalid return type: {ret_type}")

    # ========================================================
    # RETORNO
    # ========================================================

    if ret_minmax:

        return img_colored_np, depth_min, depth_max

    else:

        return img_colored_np


# ============================================================
# VISUALIZAÇÃO TENSOR
# ============================================================

def vis_depth_map_tensor(
    result: torch.Tensor,
    color_map: str = "Spectral",
) -> torch.Tensor:
    """
    Tensor visualization using Spectral colormap.

    Padronizado com:
    - FoundationStereo
    - DA2
    - saída NumPy acima
    """

    flat = result.reshape(-1).float()

    valid = flat[flat > 0]

    # ========================================================
    # SEM VALORES VÁLIDOS
    # ========================================================

    if valid.numel() == 0:

        logger.error("No valid depth values found.")

        near = torch.zeros(
            1,
            device=result.device,
            dtype=result.dtype
        )

        far = torch.ones(
            1,
            device=result.device,
            dtype=result.dtype
        )

    else:

        near = valid.min().to(result)
        far = valid.max().to(result)

    # ========================================================
    # NORMALIZAÇÃO LINEAR
    # ========================================================

    result = (result - near) / (far - near + 1e-6)

    result = result.clamp(0, 1)

    # ========================================================
    # APPLY COLORMAP
    # ========================================================

    return apply_color_map_to_image(
        result,
        color_map
    )


# ============================================================
# APPLY COLOR MAP
# ============================================================

def apply_color_map(
    x: torch.Tensor,
    color_map: str = "Spectral",
) -> torch.Tensor:

    cmap = matplotlib.colormaps[color_map]

    mapped = cmap(
        x.float()
         .detach()
         .clip(min=0, max=1)
         .cpu()
         .numpy()
    )[..., :3]

    return torch.tensor(
        mapped,
        device=x.device,
        dtype=torch.float32
    )


# ============================================================
# APPLY COLOR MAP TO IMAGE
# ============================================================

def apply_color_map_to_image(
    image: torch.Tensor,
    color_map: str = "Spectral",
) -> torch.Tensor:

    image = apply_color_map(
        image,
        color_map
    )

    return rearrange(
        image,
        "... h w c -> ... c h w"
    )
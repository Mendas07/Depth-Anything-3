import os
import imageio
import numpy as np
import matplotlib


def export_to_depth_vis(
    prediction,
    export_dir: str,
    **kwargs
):

    save_dir = os.path.join(
        export_dir,
        "depth_vis"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # ========================================================
    # CONTADOR DE ARQUIVOS
    # ========================================================

    existing_count = len(
        os.listdir(save_dir)
    )

    # ========================================================
    # COLORMAP SPECTRAL
    # ========================================================

    cmap = matplotlib.colormaps["Spectral"]

    # ========================================================
    # TOTAL
    # ========================================================

    N = prediction.depth.shape[0]

    for idx in range(N):

        # ====================================================
        # DEPTH
        # ====================================================

        depth = prediction.depth[idx].astype(
            np.float32
        )

        # ====================================================
        # NORMALIZAÇÃO LINEAR
        # ====================================================

        depth_norm = (
            depth - depth.min()
        ) / (
            depth.max() - depth.min() + 1e-8
        )

        depth_norm = np.clip(
            depth_norm,
            0,
            1
        )

        # ====================================================
        # SPECTRAL
        # ====================================================

        depth_color = cmap(
            depth_norm
        )[:, :, :3]

        # ====================================================
        # FLOAT -> UINT8
        # ====================================================

        depth_color = (
            depth_color * 255
        ).astype(np.uint8)

        # ====================================================
        # SAVE NAME
        # ====================================================

        save_name = (
            f"depth_{existing_count + idx:06d}.png"
        )

        # ====================================================
        # SAVE
        # ====================================================

        imageio.imwrite(
            os.path.join(save_dir, save_name),
            depth_color
        )
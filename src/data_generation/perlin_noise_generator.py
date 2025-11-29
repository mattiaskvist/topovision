"""Synthetic Contour Map Generator based on Perlin noise."""

import json
import os
import random

import matplotlib.pyplot as plt
import noise
import numpy as np

# --- PART 1: The Mock Terrain Generator ---


def generate_perlin_terrain(
    shape: tuple[int, int] = (512, 512),
    scale: float = 500.0,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    z_scale: float = 600,
) -> np.ndarray:
    """Generates a 2D array representing terrain height using Perlin noise.

    Args:
        shape (tuple): The dimensions of the generated array (height, width).
        scale (float): Controls the "zoom" level of the noise. Higher values result in
                       broader, larger features (zoomed in). Lower values result in
                       more frequent features (zoomed out).
        octaves (int): The number of layers of noise to combine. Higher values add more
                       fine-grained detail (roughness). Lower values result in smoother
                       terrain.
        persistence (float): Controls how much each successive octave contributes to the
                            final shape. Values < 1.0 mean higher frequency octaves have
                            less impact (smoother).
        lacunarity (float): Controls the frequency increase for each successive octave.
                            Values > 1.0 mean detail becomes finer at each step.
        z_scale (float): Multiplier for the height values. Controls the vertical relief
                         of the terrain.

    Returns:
        np.ndarray: A 2D float32 array of height values.
    """
    data = np.zeros(shape)

    # Random offsets to ensure every generation is different
    x_offset = random.randint(0, 10000)
    y_offset = random.randint(0, 10000)

    for i in range(shape[0]):
        for j in range(shape[1]):
            n = noise.pnoise2(
                (i + x_offset) / scale,
                (j + y_offset) / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=0,
            )
            data[i][j] = n

    # Normalize data to range 0.0 to 1.0
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Scale to desired height
    data = data * z_scale
    return data.astype("float32")


# --- PART 2: The Annotation Generator ---


def generate_synthetic_pair(
    data_array: np.ndarray,
    output_dir: str,
    file_id: int | str,
    contour_interval: int = 80,
) -> None:
    """Generates a synthetic contour map image, a mask, and OCR annotations.

    Args:
        data_array (np.ndarray): The 2D height array.
        output_dir (str): Directory to save the outputs.
        file_id (int or str): Identifier for the generated files.
        contour_interval (int): The vertical distance between contour lines.
                                Larger values result in fewer, more spaced-out
                                lines (sparse map). Smaller values result in many
                                lines (dense map).
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"sparse_{file_id}"
    h, w = data_array.shape

    dpi = 100
    figsize = (w / dpi, h / dpi)

    # Determine contour levels
    z_min = np.floor(np.min(data_array) / contour_interval) * contour_interval
    z_max = np.ceil(np.max(data_array) / contour_interval) * contour_interval
    levels = np.arange(z_min, z_max + contour_interval, contour_interval)

    # --- PASS 1: Generate Segmentation Mask (Lines Only) ---
    fig_mask = plt.figure(figsize=figsize, dpi=dpi)
    ax_mask = plt.Axes(fig_mask, [0.0, 0.0, 1.0, 1.0])
    ax_mask.set_axis_off()
    fig_mask.add_axes(ax_mask)

    ax_mask.imshow(np.zeros_like(data_array), cmap="gray", vmin=0, vmax=1)
    # Thicker lines look better on sparse maps
    ax_mask.contour(data_array, levels=levels, colors="white", linewidths=2.0)

    mask_filename = os.path.join(output_dir, f"{base_name}_mask.png")
    fig_mask.savefig(mask_filename, dpi=dpi, pad_inches=0)
    plt.close(fig_mask)

    # --- PASS 2: Generate OCR Image & Extract BBoxes ---
    fig_img = plt.figure(figsize=figsize, dpi=dpi)
    ax_img = plt.Axes(fig_img, [0.0, 0.0, 1.0, 1.0])
    ax_img.set_axis_off()
    fig_img.add_axes(ax_img)

    ax_img.imshow(np.zeros_like(data_array), cmap="gray", vmin=0, vmax=1)
    cs = ax_img.contour(data_array, levels=levels, colors="white", linewidths=2.0)

    # Increased font size and spacing for better readability
    clabels = ax_img.clabel(
        cs, inline=True, fontsize=14, fmt="%1.0f", colors="white", inline_spacing=15
    )

    fig_img.canvas.draw()
    renderer = fig_img.canvas.get_renderer()

    ocr_data = []
    height_pixels = h

    print(f"Generating {base_name}...", end=" ")

    for label in clabels:
        text_content = label.get_text()
        bbox = label.get_window_extent(renderer)

        padding = 3
        x0 = bbox.x0 - padding
        y0 = bbox.y0 - padding
        x1 = bbox.x1 + padding
        y1 = bbox.y1 + padding

        img_y0 = height_pixels - y1
        img_y1 = height_pixels - y0

        img_y0 = max(0, img_y0)
        img_y1 = min(height_pixels, img_y1)
        x0 = max(0, x0)
        x1 = min(w, x1)

        annotation = {
            "text": text_content,
            "bbox_xyxy": [round(x0), round(img_y0), round(x1), round(img_y1)],
        }
        ocr_data.append(annotation)

    image_filename = os.path.join(output_dir, f"{base_name}_image.png")
    json_filename = os.path.join(output_dir, f"{base_name}_labels.json")

    fig_img.savefig(image_filename, dpi=dpi, pad_inches=0)

    metadata = {
        "image_path": image_filename,
        "mask_path": mask_filename,
        "image_size": [w, h],
        "annotations": ocr_data,
    }
    with open(json_filename, "w") as f:
        json.dump(metadata, f, indent=4)

    plt.close(fig_img)
    print(f"Found {len(ocr_data)} labels.")


def main() -> None:
    """Main execution function to generate synthetic Perlin noise terrain data."""
    output_folder = "data/synthetic/perlin_noise"
    num_images_to_generate = 5
    image_size = (512, 512)

    print(
        f"Starting generation of {num_images_to_generate} SPARSE synthetic datasets..."
    )
    print(f"Output directory: {output_folder}")

    for i in range(num_images_to_generate):
        # Scale 500-700 ensures big, smooth hills
        rand_scale = random.uniform(300.0, 500.0)
        rand_z = random.uniform(400, 800)

        # octaves=3 is default in the function, giving smoother terrain
        mock_dem = generate_perlin_terrain(
            shape=image_size, scale=rand_scale, z_scale=rand_z
        )

        # High interval (75-125) ensures very few lines (sparse)
        rand_interval = random.choice([75, 100, 125])
        generate_synthetic_pair(
            mock_dem, output_folder, file_id=i, contour_interval=rand_interval
        )

    print(f"\nDone! Check the '{output_folder}' directory.")


if __name__ == "__main__":
    main()

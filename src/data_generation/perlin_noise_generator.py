"""Synthetic Contour Map Generator based on Perlin noise."""

import json
import os
import random

import matplotlib.pyplot as plt
import noise
import numpy as np
from matplotlib.patches import Polygon

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
    file_id: int,
    annotation_id_start: int,
    contour_interval: int = 80,
) -> tuple[dict, list[dict], int]:
    """Generates a synthetic contour map image, a mask, and OCR annotations.

    Args:
        data_array (np.ndarray): The 2D height array.
        output_dir (str): Directory to save the outputs.
        file_id (int): Identifier for the generated files.
        annotation_id_start (int): Starting ID for annotations.
        contour_interval (int): The vertical distance between contour lines.

    Returns:
        tuple: (image_info, annotations, next_annotation_id)
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

    # Get the inverse transformation to convert from display to data coordinates
    inv_trans = ax_img.transData.inverted()

    coco_annotations = []
    current_ann_id = annotation_id_start

    print(f"Generating {base_name}...", end=" ")

    for label in clabels:
        text_content = label.get_text()

        # 1. Get rotation and anchor point
        rotation = label.get_rotation()
        transform = label.get_transform()
        # The position is usually in data coordinates, transform to display
        pos_display = transform.transform(label.get_position())

        # 2. Get unrotated bounding box in display coordinates
        label.set_rotation(0)
        bbox_unrotated = label.get_window_extent(renderer)
        label.set_rotation(rotation)  # Restore rotation

        # 3. Calculate the 4 corners relative to the anchor point
        # The anchor point in display coords corresponds to pos_display.
        # However, bbox_unrotated is absolute display coords of the unrotated text.
        # We can just take the corners of bbox_unrotated and rotate them around
        # pos_display.

        # Corners of unrotated box:
        # p1 (BL), p2 (BR), p3 (TR), p4 (TL) - usually standard order
        corners_display = np.array(
            [
                [bbox_unrotated.x0, bbox_unrotated.y0],
                [bbox_unrotated.x1, bbox_unrotated.y0],
                [bbox_unrotated.x1, bbox_unrotated.y1],
                [bbox_unrotated.x0, bbox_unrotated.y1],
            ]
        )

        # 4. Rotate corners around the anchor point
        # Create rotation matrix
        rot_rad = np.radians(rotation)
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)

        # Translate to origin (relative to anchor), rotate, translate back
        # Note: We assume rotation is around the anchor point (pos_display)
        # Matplotlib rotates text around its anchor.

        centered = corners_display - pos_display
        rotated = np.zeros_like(centered)
        rotated[:, 0] = centered[:, 0] * cos_r - centered[:, 1] * sin_r
        rotated[:, 1] = centered[:, 0] * sin_r + centered[:, 1] * cos_r
        corners_rotated_display = rotated + pos_display

        # 5. Transform to data coordinates
        corners_data = inv_trans.transform(corners_rotated_display)

        # 6. Clamp to image boundaries?
        # For segmentation, we might want exact points even if slightly out.
        # But for bbox we definitely want clamped.
        # Let's clamp the points for safety, but maybe not strictly necessary for
        # segmentation if we accept out of bounds. Let's clamp to be safe.
        corners_data[:, 0] = np.clip(corners_data[:, 0], 0, w)
        corners_data[:, 1] = np.clip(corners_data[:, 1], 0, h)

        # Flatten for segmentation: [x1, y1, x2, y2, x3, y3, x4, y4]
        segmentation = corners_data.flatten().tolist()
        segmentation = [round(x, 2) for x in segmentation]

        # Calculate axis-aligned bbox from the polygon
        x_min = np.min(corners_data[:, 0])
        x_max = np.max(corners_data[:, 0])
        y_min = np.min(corners_data[:, 1])
        y_max = np.max(corners_data[:, 1])

        rect_x = x_min
        rect_y = y_min
        rect_w = x_max - x_min
        rect_h = y_max - y_min

        # Skip invalid boxes
        if rect_w <= 0 or rect_h <= 0:
            continue

        annotation = {
            "id": current_ann_id,
            "image_id": file_id,
            "category_id": 1,
            "bbox": [int(rect_x), int(rect_y), int(rect_w), int(rect_h)],
            "area": int(rect_w * rect_h),  # Approx area of AABB
            "segmentation": [segmentation],
            "iscrowd": 0,
            "text": text_content,  # Extra field for OCR
        }
        coco_annotations.append(annotation)
        current_ann_id += 1

    image_filename = f"{base_name}_image.png"
    full_image_path = os.path.join(output_dir, image_filename)
    fig_img.savefig(full_image_path, dpi=dpi, pad_inches=0)
    plt.close(fig_img)

    # --- PASS 3: Generate Debug Image with BBoxes ---
    # Load the actual generated image to ensure we are debugging what we saved
    actual_image = plt.imread(full_image_path)

    fig_debug = plt.figure(figsize=figsize, dpi=dpi)
    ax_debug = plt.Axes(fig_debug, [0.0, 0.0, 1.0, 1.0])
    ax_debug.set_axis_off()
    fig_debug.add_axes(ax_debug)

    ax_debug.imshow(actual_image)

    # Draw boxes
    for ann in coco_annotations:
        # Draw Polygon from segmentation
        seg_points = ann["segmentation"][0]
        # Reshape to (4, 2)
        poly_points = np.array(seg_points).reshape((4, 2))

        poly = Polygon(
            poly_points,
            linewidth=1,
            edgecolor="red",
            facecolor="none",
        )
        ax_debug.add_patch(poly)

        # Draw text above the box (use first point)
        ax_debug.text(
            poly_points[0][0],
            poly_points[0][1],
            ann["text"],
            color="yellow",
            fontsize=8,
            verticalalignment="bottom",
        )

    debug_filename = os.path.join(output_dir, f"{base_name}_debug.png")
    fig_debug.savefig(debug_filename, dpi=dpi, pad_inches=0)
    plt.close(fig_debug)

    print(f"Found {len(coco_annotations)} labels.")

    image_info = {
        "id": file_id,
        "file_name": image_filename,
        "width": w,
        "height": h,
    }

    return image_info, coco_annotations, current_ann_id


def main() -> None:
    """Main execution function to generate synthetic Perlin noise terrain data."""
    output_folder = "data/synthetic/perlin_noise"
    num_images_to_generate = 5
    image_size = (512, 512)

    print(
        f"Starting generation of {num_images_to_generate} SPARSE synthetic datasets..."
    )
    print(f"Output directory: {output_folder}")

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "elevation_text", "supercategory": "text"}],
    }

    next_annotation_id = 1

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

        image_info, annotations, next_annotation_id = generate_synthetic_pair(
            mock_dem,
            output_folder,
            file_id=i,
            annotation_id_start=next_annotation_id,
            contour_interval=rand_interval,
        )

        coco_output["images"].append(image_info)
        coco_output["annotations"].extend(annotations)

    # Save COCO JSON
    coco_filename = os.path.join(output_folder, "coco_annotations.json")
    with open(coco_filename, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"\nDone! Check the '{output_folder}' directory.")


if __name__ == "__main__":
    main()

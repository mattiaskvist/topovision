"""Synthetic Contour Map Generator with Rotated Text Annotations (COCO Format)."""

import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import noise
import numpy as np
from matplotlib.patches import Polygon

# --- Constants ---
NOISE_REPEAT_X = 1024
NOISE_REPEAT_Y = 1024
CONTOUR_LINE_WIDTH = 2.0
CONTOUR_LABEL_FONT_SIZE = 14
CONTOUR_LABEL_INLINE_SPACING = 15
DEFAULT_DPI = 100

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

    Note:
        This function uses nested loops to generate noise values, which can be slow for
        very large arrays (e.g., > 1024x1024). The `noise` library does not currently
        support vectorization, and `np.vectorize` offers negligible performance benefits
        due to the underlying C extension implementation. For standard use cases
        (e.g., 512x512), performance is acceptable.
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
                repeatx=NOISE_REPEAT_X,
                repeaty=NOISE_REPEAT_Y,
                base=0,
            )
            data[i][j] = n

    # Normalize data to range 0.0 to 1.0
    data_min = np.min(data)
    data_max = np.max(data)
    if np.isclose(data_max, data_min):
        # Avoid division by zero if terrain is flat
        data = np.zeros_like(data)
    else:
        data = (data - data_min) / (data_max - data_min)

    # Scale to desired height
    data = data * z_scale
    return data.astype("float32")


# --- PART 2: The Annotation Generator ---


def _calculate_contour_levels(
    data_array: np.ndarray, contour_interval: int
) -> np.ndarray:
    """Calculates contour levels based on data range and interval.

    Args:
        data_array (np.ndarray): The 2D height array.
        contour_interval (int): The vertical distance between contour lines.

    Returns:
        np.ndarray: An array of contour levels.
    """
    z_min = np.floor(np.min(data_array) / contour_interval) * contour_interval
    z_max = np.ceil(np.max(data_array) / contour_interval) * contour_interval
    return np.arange(z_min, z_max + contour_interval, contour_interval)


def _generate_mask_image(
    data_array: np.ndarray,
    levels: np.ndarray,
    output_dir: str,
    base_name: str,
    figsize: tuple[float, float],
    dpi: int,
) -> None:
    """Generates and saves the segmentation mask image.

    Args:
        data_array (np.ndarray): The 2D height array.
        levels (np.ndarray): The contour levels to draw.
        output_dir (str): Directory to save the mask image.
        base_name (str): Base name for the output file.
        figsize (tuple): Figure size in inches (width, height).
        dpi (int): Dots per inch for the figure.
    """
    fig_mask = plt.figure(figsize=figsize, dpi=dpi)
    try:
        ax_mask = plt.Axes(fig_mask, [0.0, 0.0, 1.0, 1.0])
        ax_mask.set_axis_off()
        fig_mask.add_axes(ax_mask)

        ax_mask.imshow(np.zeros_like(data_array), cmap="gray", vmin=0, vmax=1)
        ax_mask.contour(
            data_array, levels=levels, colors="white", linewidths=CONTOUR_LINE_WIDTH
        )

        mask_filename = os.path.join(output_dir, f"{base_name}_mask.png")
        try:
            fig_mask.savefig(mask_filename, dpi=dpi, pad_inches=0)
        except OSError as e:
            print(f"Error saving mask image to {mask_filename}: {e}")
            raise
    finally:
        plt.close(fig_mask)


def _generate_ocr_image_and_annotations(
    data_array: np.ndarray,
    levels: np.ndarray,
    output_dir: str,
    base_name: str,
    file_id: int,
    annotation_id_start: int,
    figsize: tuple[float, float],
    dpi: int,
    h: int,
    w: int,
) -> tuple[dict, list[dict], int, str]:
    """Generates the OCR image and extracts annotations.

    Args:
        data_array (np.ndarray): The 2D height array.
        levels (np.ndarray): The contour levels to draw.
        output_dir (str): Directory to save the image.
        base_name (str): Base name for the output file.
        file_id (int): Identifier for the generated file.
        annotation_id_start (int): Starting ID for annotations.
        figsize (tuple): Figure size in inches (width, height).
        dpi (int): Dots per inch for the figure.
        h (int): Height of the image in pixels.
        w (int): Width of the image in pixels.

    Returns:
        tuple: A tuple containing:
            - image_info (dict): COCO image info dictionary.
            - coco_annotations (list[dict]): List of COCO annotation dictionaries.
            - next_ann_id (int): The next available annotation ID.
            - full_image_path (str): The absolute path to the saved image.
    """
    fig_img = plt.figure(figsize=figsize, dpi=dpi)
    try:
        ax_img = plt.Axes(fig_img, [0.0, 0.0, 1.0, 1.0])
        ax_img.set_axis_off()
        fig_img.add_axes(ax_img)

        ax_img.imshow(np.zeros_like(data_array), cmap="gray", vmin=0, vmax=1)
        cs = ax_img.contour(
            data_array, levels=levels, colors="white", linewidths=CONTOUR_LINE_WIDTH
        )

        # Note: inline_spacing puts a gap in the contour line for the text
        clabels = ax_img.clabel(
            cs,
            inline=True,
            fontsize=CONTOUR_LABEL_FONT_SIZE,
            fmt="%1.0f",
            colors="white",
            inline_spacing=CONTOUR_LABEL_INLINE_SPACING,
        )

        # Force a draw so the renderer calculates text positions
        fig_img.canvas.draw()
        renderer = fig_img.canvas.get_renderer()

        # Get transform to convert from Display Pixels -> Image Data Indices
        # This handles the Y-axis flip (Bottom-Left to Top-Left) automatically.
        inv_trans = ax_img.transData.inverted()

        coco_annotations = []
        current_ann_id = annotation_id_start

        print(f"Generating {base_name}...", end=" ")

        for label in clabels:
            text_content = label.get_text()

            # --- ROTATED BOX CALCULATION ---

            # 1. Capture current rotation and anchor
            rotation = label.get_rotation()
            transform = label.get_transform()
            # pos_display is the anchor point of the text in Display Coords (pixels)
            pos_display = transform.transform(label.get_position())

            # 2. Get the UN-ROTATED box dimensions
            # We temporarily set rotation to 0 to get true width/height of the
            # text block
            label.set_rotation(0)
            bbox_unrotated = label.get_window_extent(renderer)
            label.set_rotation(rotation)  # Restore rotation immediately

            # 3. Define the 4 corners of the unrotated box
            # Matplotlib text is usually anchored at the center
            # (ha='center', va='center') for clabels.
            # So we rotate the corners of the unrotated box around the anchor point.
            corners_display = np.array(
                [
                    [bbox_unrotated.x0, bbox_unrotated.y0],  # Bottom-Left
                    [bbox_unrotated.x1, bbox_unrotated.y0],  # Bottom-Right
                    [bbox_unrotated.x1, bbox_unrotated.y1],  # Top-Right
                    [bbox_unrotated.x0, bbox_unrotated.y1],  # Top-Left
                ]
            )

            # 4. Apply Rotation Matrix
            # Translate corners so anchor is at (0,0) -> Rotate -> Translate back
            rot_rad = np.radians(rotation)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)

            centered = corners_display - pos_display
            rotated = np.zeros_like(centered)
            rotated[:, 0] = centered[:, 0] * cos_r - centered[:, 1] * sin_r
            rotated[:, 1] = centered[:, 0] * sin_r + centered[:, 1] * cos_r
            corners_rotated_display = rotated + pos_display

            # 5. Convert Display Coords -> Image Coords
            # This maps the plot pixels to the numpy array indices (0,0 is top-left)
            corners_data = inv_trans.transform(corners_rotated_display)

            # 6. Clamp to image boundaries
            corners_data[:, 0] = np.clip(corners_data[:, 0], 0, w)
            corners_data[:, 1] = np.clip(corners_data[:, 1], 0, h)

            # --- FORMATTING FOR COCO ---

            # Segmentation: Flattened list of polygon points [x1, y1, x2, y2, ...]
            segmentation = corners_data.flatten().tolist()
            segmentation = [round(x, 2) for x in segmentation]

            # BBox: Axis-Aligned Bounding Box [x_min, y_min, width, height]
            x_min = np.min(corners_data[:, 0])
            x_max = np.max(corners_data[:, 0])
            y_min = np.min(corners_data[:, 1])
            y_max = np.max(corners_data[:, 1])

            rect_w = x_max - x_min
            rect_h = y_max - y_min

            # Use the unrotated area for strict polygon area
            # (more accurate than AABB area)
            true_area = bbox_unrotated.width * bbox_unrotated.height

            # Filter out tiny artifacts
            if rect_w <= 1 or rect_h <= 1:
                continue

            annotation = {
                "id": current_ann_id,
                "image_id": file_id,
                "category_id": 1,
                "bbox": [
                    round(x_min, 2),
                    round(y_min, 2),
                    round(rect_w, 2),
                    round(rect_h, 2),
                ],
                "area": round(true_area, 2),
                "segmentation": [segmentation],
                "iscrowd": 0,
                "attributes": {"text": text_content, "rotation": round(rotation, 2)},
            }
            coco_annotations.append(annotation)
            current_ann_id += 1

        # Save the actual image
        image_filename = f"{base_name}_image.png"
        full_image_path = os.path.join(output_dir, image_filename)
        try:
            fig_img.savefig(full_image_path, dpi=dpi, pad_inches=0)
        except OSError as e:
            print(f"Error saving OCR image to {full_image_path}: {e}")
            raise

        print(f"Found {len(coco_annotations)} labels.")
    finally:
        plt.close(fig_img)

    image_info = {
        "id": file_id,
        "file_name": image_filename,
        "width": w,
        "height": h,
    }

    return image_info, coco_annotations, current_ann_id, full_image_path


def _generate_debug_image(
    full_image_path: str,
    coco_annotations: list[dict],
    output_dir: str,
    base_name: str,
    figsize: tuple[float, float],
    dpi: int,
) -> None:
    """Generates a debug image with bounding boxes and polygons.

    Args:
        full_image_path (str): Path to the source image.
        coco_annotations (list[dict]): List of COCO annotations to visualize.
        output_dir (str): Directory to save the debug image.
        base_name (str): Base name for the output file.
        figsize (tuple): Figure size in inches (width, height).
        dpi (int): Dots per inch for the figure.
    """
    if os.path.exists(full_image_path):
        actual_image = plt.imread(full_image_path)
        fig_debug = plt.figure(figsize=figsize, dpi=dpi)
        try:
            ax_debug = plt.Axes(fig_debug, [0.0, 0.0, 1.0, 1.0])
            ax_debug.set_axis_off()
            fig_debug.add_axes(ax_debug)
            ax_debug.imshow(actual_image)

            for ann in coco_annotations:
                # Reconstruct polygon from segmentation list
                poly_coords = np.array(ann["segmentation"][0]).reshape((4, 2))

                # Draw the Polygon (Rotated Box)
                poly_patch = Polygon(
                    poly_coords, linewidth=1, edgecolor="cyan", facecolor="none"
                )
                ax_debug.add_patch(poly_patch)

                # Draw the AABB (Standard Box) - Optional, usually red
                x, y, w_box, h_box = ann["bbox"]
                rect_patch = plt.Rectangle(
                    (x, y),
                    w_box,
                    h_box,
                    linewidth=1,
                    edgecolor="red",
                    facecolor="none",
                    linestyle="--",
                )
                ax_debug.add_patch(rect_patch)

            debug_filename = os.path.join(output_dir, f"{base_name}_debug.png")
            try:
                fig_debug.savefig(debug_filename, dpi=dpi, pad_inches=0)
            except OSError as e:
                print(f"Error saving debug image to {debug_filename}: {e}")
                raise
        finally:
            plt.close(fig_debug)


def generate_synthetic_pair(
    data_array: np.ndarray,
    output_dir: str,
    file_id: int,
    annotation_id_start: int,
    contour_interval: int = 80,
    dpi: int = DEFAULT_DPI,
) -> tuple[dict, list[dict], int]:
    """Generates a synthetic contour map image, a mask, and OCR annotations.

    Args:
        data_array (np.ndarray): The 2D height array.
        output_dir (str): Directory to save the outputs.
        file_id (int): Identifier for the generated files.
        annotation_id_start (int): Starting ID for annotations.
        contour_interval (int): The vertical distance between contour lines.
        dpi (int): Dots per inch for the output images. Defaults to DEFAULT_DPI.

    Returns:
        tuple: (image_info_dict, list_of_annotation_dicts, next_annotation_id)
    """
    if contour_interval <= 0:
        raise ValueError("contour_interval must be positive")

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")
        raise
    base_name = f"sparse_{file_id}"
    h, w = data_array.shape

    # DPI must match between figure creation and savefig to ensure
    # pixel coordinates are accurate.
    figsize = (w / dpi, h / dpi)

    # Determine contour levels
    levels = _calculate_contour_levels(data_array, contour_interval)

    # --- PASS 1: Generate Segmentation Mask (Lines Only) ---
    _generate_mask_image(data_array, levels, output_dir, base_name, figsize, dpi)

    # --- PASS 2: Generate OCR Image & Extract BBoxes ---
    image_info, coco_annotations, next_ann_id, full_image_path = (
        _generate_ocr_image_and_annotations(
            data_array,
            levels,
            output_dir,
            base_name,
            file_id,
            annotation_id_start,
            figsize,
            dpi,
            h,
            w,
        )
    )

    # --- PASS 3: Generate Debug Image (Verify BBoxes) ---
    _generate_debug_image(
        full_image_path, coco_annotations, output_dir, base_name, figsize, dpi
    )

    return image_info, coco_annotations, next_ann_id


def main() -> None:
    """Main execution function."""
    random.seed(42)

    output_folder = "data/synthetic/perlin_noise"
    num_images_to_generate = 5
    image_size = (512, 512)

    print(f"Starting generation of {num_images_to_generate} datasets...")
    print(f"Output directory: {output_folder}")

    # Standard COCO Header
    coco_output = {
        "info": {
            "description": "Synthetic Contour Text Dataset",
            "year": datetime.now().year,
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "text", "supercategory": "ocr"}],
    }

    next_annotation_id = 1

    for i in range(num_images_to_generate):
        # Scale 500-700 ensures big, smooth hills
        rand_scale = random.uniform(500.0, 700.0)
        rand_z = random.uniform(400, 800)

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
    try:
        with open(coco_filename, "w") as f:
            json.dump(coco_output, f, indent=4)
    except OSError as e:
        print(f"Error saving COCO annotations to {coco_filename}: {e}")
        raise

    print(f"\nDone! Check the '{output_folder}' directory.")


if __name__ == "__main__":
    main()

from collections import Counter

import cv2
import numpy as np
from skimage.color import rgb2lab


def rgb_to_lab(rgb):
    """
    Converts an RGB color to CIE-Lab color space.

    :param rgb: A tuple of (R, G, B) values in the range [0, 255].
    :return: A tuple of (L, a, b) values in the Lab color space.
    """
    # Normalize RGB to [0, 1]
    rgb = np.array(rgb) / 255.0

    # Convert to Lab using skimage
    lab = rgb2lab(rgb[np.newaxis, np.newaxis, :])
    return lab[0, 0, :]


def delta_e_lab(lab1, lab2):
    """
    Computes the ΔE value between two Lab colors using the Euclidean distance (ΔE76).

    :param lab1: Lab color as a tuple (L, a, b).
    :param lab2: Lab color as a tuple (L, a, b).
    :return: The ΔE value.
    """
    return np.linalg.norm(np.array(lab1) - np.array(lab2))


def delta_e_rgb(rgb1, rgb2):
    """
    Computes the ΔE value between two RGB colors.

    :param rgb1: First RGB color as a tuple (R, G, B) in the range [0, 255].
    :param rgb2: Second RGB color as a tuple (R, G, B) in the range [0, 255].
    :return: The ΔE value.
    """
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    return delta_e_lab(lab1, lab2)


def delta_e(color1, color2):
    rgb1, rgb2 = color1[::-1], color2[::-1]  # convert cv2 bgr to rgb
    lab1, lab2 = rgb2lab(rgb1), rgb2lab(rgb2)

    delta_e = np.linalg.norm(lab1.astype(np.float32) - lab2.astype(np.float32))
    return delta_e


def extract_and_find_most_common(mask_img, content_img):
    """
    Extract pixel values from a content image based on a binary mask image,
    count them, and return the most common pixel value.

    Parameters:
    - mask_img: A binary mask image with a white background (255) and black lines (0).
    - content_img: A natural content image.

    Returns:
    - The most common pixel value.
    """
    # Ensure both images have the same dimensions
    if mask_img.shape[:2] != content_img.shape[:2]:
        raise ValueError("Mask image and content image must have the same dimensions.")

    # Find the indices of the black regions in the mask (mask == 0)
    # Ensure mask_img is single-channel grayscale
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    mask_indices = (mask_img == 0)

    # Extract the corresponding pixel values from the content image
    if len(content_img.shape) == 3 and content_img.shape[2] == 3:
        rows, cols = np.where(mask_indices)
        selected_pixels = content_img[rows, cols, :]
        selected_pixels = [tuple(pixel) for pixel in selected_pixels]
    else:
        selected_pixels = content_img[mask_indices].tolist()

    # Count the occurrences of each pixel value
    pixel_counts = Counter(selected_pixels)

    # Find the most common pixel value
    most_common_pixel = pixel_counts.most_common(1)[0][0] if pixel_counts else None

    return most_common_pixel


def contrast_color(mask_image, content_image, exclude_colors=None):
    candidate_colors = {
        'black': (0, 0, 0),
        'blue': (255, 0, 0),
        'green': (0, 128, 0),
        'red': (0, 0, 255)
    }
    if exclude_colors:
        candidate_colors = {k: v for k, v in candidate_colors.items() if k not in exclude_colors}
    background_color = extract_and_find_most_common(mask_image, content_image)
    best_delta_e = 0
    contrast_color, contrast_value = list(candidate_colors.keys())[0], candidate_colors[
        list(candidate_colors.keys())[0]]
    for color, value in candidate_colors.items():
        delta_e_value = delta_e_rgb(background_color, value)
        # print(f"Background Color: {background_color}, Color: {color}, Value: {value}, Delta E: {delta_e_value}")
        if delta_e_value > best_delta_e:
            best_delta_e = delta_e_value
            contrast_color, contrast_value = color, value
    return contrast_color, contrast_value


def polygon_to_coco_bbox(polygons):
    """
    Convert a list of polygons in nested list format
    [[x1, y1, x2, y2, ...], [x7, y7, x8, y8, ...]] to a COCO format bounding box
    [x, y, width, height] that merges the bounding boxes of all polygons.

    Parameters:
    polygons (list of list): A list of polygons where each polygon is a list of (x, y) coordinates.

    Returns:
    list: Merged bounding box in COCO format [x, y, width, height].
    """
    if not polygons or not all(polygon for polygon in polygons):  # Handle empty input
        return [0, 0, 0, 0]

    # Initialize variables for tracking the global bounding box
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Iterate over each polygon and compute the bounding box
    for polygon in polygons:
        if not polygon:  # Skip empty polygons
            continue

        # Extract coordinates (assuming all polygons have the same structure)
        coords = polygon

        x_coords = coords[0::2]  # Extract X values (every even index)
        y_coords = coords[1::2]  # Extract Y values (every odd index)

        # Update the global bounding box
        min_x = min(min_x, min(x_coords))
        max_x = max(max_x, max(x_coords))
        min_y = min(min_y, min(y_coords))
        max_y = max(max_y, max(y_coords))

    # Compute the merged bounding box dimensions
    width = max_x - min_x
    height = max_y - min_y

    return [min_x, min_y, width, height]


def scale_to_1080p_with_annotations(image, bbox=None, polygons=None):
    # Get the current dimensions of the image
    height, width = image.shape[:2]

    # Define the maximum dimensions for 1080p
    max_width = 1920
    max_height = 1080

    # Check if the image is larger than 1080p in either dimension
    if width > max_width or height > max_height:
        # Calculate the scaling factor while preserving the aspect ratio
        scale_factor = min(max_width / width, max_height / height)

        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image
        image_resized = cv2.resize(image, (new_width, new_height))

        # Scale the bounding box if provided
        if bbox:
            x, y, w, h = bbox
            bbox_scaled = (int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor))
        else:
            bbox_scaled = None

        # Scale the polygons if provided
        if polygons:
            polygons_scaled = []
            for poly in polygons:
                poly_scaled = [(int(poly[i] * scale_factor), int(poly[i+1] * scale_factor)) for i in range(0, len(poly), 2)]
                polygons_scaled.append(poly_scaled)
        else:
            polygons_scaled = None

        return image_resized, bbox_scaled, polygons_scaled
    else:
        # If the image is already smaller or equal to 1080p, return it as is
        return image, bbox, polygons

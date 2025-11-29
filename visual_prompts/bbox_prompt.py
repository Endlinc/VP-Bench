import cv2
import numpy as np

from demo.constant import COLOR_DICT
from vp_bench.common.utils import contrast_color
from vp_bench.visual_prompts.visual_prompt import VisualPrompt


class BoundingBoxPrompt(VisualPrompt):
    def __init__(self, prominence_threshold=0.05):
        super().__init__(dimming_enabled=True)
        self.prominence_threshold = prominence_threshold

    def apply_prompt(self, image, annotations, image_info, mask, colors, thickness, **kwargs):
        image = image.copy()
        image_height, image_width, _ = image.shape
        image_area = self.calculate_area(image_width, image_height)
        shape = kwargs.get("shape", "square").lower()  # Get shape from kwargs (default to square)

        used_colors = []
        for i, ann in enumerate(annotations):
            color = colors[i]
            color_value = COLOR_DICT[color]
            bbox = list(map(int, ann['bbox']))  # [x_min, y_min, width, height]
            bbox_area = self.calculate_area(bbox[2], bbox[3])

            # Bounding box coordinates
            x_min, y_min, width, height = bbox
            x_max, y_max = x_min + width, y_min + height

            # Apply thickness to the rectangle
            thickness_value = thickness

            # Draw bounding box
            if color == "contrast":
                white_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                cv2.rectangle(white_image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness_value)
                color, color_value = contrast_color(white_image, image, used_colors)
            used_colors.append(color)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color_value, thickness_value)

            if shape == "square":
                # Size of the solid square vertices
                vertex_size = max(thickness_value + 4, 6)
                half_vertex = vertex_size // 2

                # Top-left vertex
                cv2.rectangle(image, (x_min - half_vertex, y_min - half_vertex),
                              (x_min + half_vertex, y_min + half_vertex), color_value, -1)
                # Top-right vertex
                cv2.rectangle(image, (x_max - half_vertex, y_min - half_vertex),
                              (x_max + half_vertex, y_min + half_vertex), color_value, -1)
                # Bottom-left vertex
                cv2.rectangle(image, (x_min - half_vertex, y_max - half_vertex),
                              (x_min + half_vertex, y_max + half_vertex), color_value, -1)
                # Bottom-right vertex
                cv2.rectangle(image, (x_max - half_vertex, y_max - half_vertex),
                              (x_max + half_vertex, y_max + half_vertex), color_value, -1)

            elif shape == "round":
                radius = max(thickness_value, 5)  # Radius for the round vertices

                # Top-left corner circle
                cv2.circle(image, (x_min, y_min), radius, color_value, -1)
                # Top-right corner circle
                cv2.circle(image, (x_max, y_min), radius, color_value, -1)
                # Bottom-left corner circle
                cv2.circle(image, (x_min, y_max), radius, color_value, -1)
                # Bottom-right corner circle
                cv2.circle(image, (x_max, y_max), radius, color_value, -1)

            elif shape == "none":
                # Only draw bounding box, no vertices
                if mask is not None:
                    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
            else:
                raise ValueError("Shape should be 'square', 'round', or 'none'.")

        return image, used_colors

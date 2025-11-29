import cv2
import numpy as np

from demo.constant import COLOR_DICT
from vp_bench.common.utils import contrast_color
from vp_bench.visual_prompts.visual_prompt import VisualPrompt


class SinglePointPrompt(VisualPrompt):
    def __init__(self, prominence_threshold=0.05):
        """
        SinglePointPrompt places a single shape (circle or square) at the center of each instance.
        :param prominence_threshold: Minimum area required for an instance to be considered prominent.
        """
        super().__init__(dimming_enabled=False)
        self.prominence_threshold = prominence_threshold

    def apply_prompt(self, image, annotations, image_info, mask, colors, thickness, **kwargs):
        image = image.copy()
        image_height, image_width, _ = image.shape
        image_area = self.calculate_area(image_width, image_height)

        # Get the shape from kwargs (default is "circle")
        shape = kwargs.get("shape", "circle").lower()

        # Map thickness values
        thickness_value = thickness

        used_colors = []
        for i, ann in enumerate(annotations):
            color = colors[i]
            color_value = COLOR_DICT[color]
            bbox = list(map(int, ann['bbox']))  # [x_min, y_min, width, height]
            bbox_area = self.calculate_area(bbox[2], bbox[3])

            # Calculate the center of the object
            x_min, y_min, width, height = bbox
            center = (x_min + width // 2, y_min + height // 2)

            if shape == "circle":
                # Draw a circle at the center
                if color == "contrast":
                    white_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                    cv2.circle(white_image, center, radius=thickness_value * 2, color=(0, 0, 0), thickness=-1)
                    color, color_value = contrast_color(white_image, image, used_colors)
                used_colors.append(color)
                cv2.circle(image, center, radius=thickness_value * 2, color=color_value, thickness=-1)

            elif shape == "square":
                # Draw a square at the center

                side_length = thickness_value * 4  # Side length of the square
                top_left = (center[0] - side_length // 2, center[1] - side_length // 2)
                bottom_right = (center[0] + side_length // 2, center[1] + side_length // 2)

                if color == "contrast":
                    white_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                    cv2.circle(white_image, center, radius=thickness_value * 2, color=(0, 0, 0), thickness=-1)
                    color, color_value = contrast_color(white_image, image, used_colors)
                used_colors.append(color)
                cv2.rectangle(image, top_left, bottom_right, color_value, thickness=-1)

            else:
                raise ValueError("Invalid shape. Use 'circle' or 'square'.")
        return image, used_colors

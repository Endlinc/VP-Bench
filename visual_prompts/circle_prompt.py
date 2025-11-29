import cv2
import numpy as np

from demo.constant import COLOR_DICT
from vp_bench.common.utils import contrast_color
from vp_bench.visual_prompts.visual_prompt import VisualPrompt


class CirclePrompt(VisualPrompt):
    def __init__(self, prominence_threshold=0.1):
        super().__init__(dimming_enabled=False)  # Dimming is disabled for circles
        self.prominence_threshold = prominence_threshold

    def apply_prompt(self, image, annotations, image_info, mask, colors, thickness, shape=None):
        image = image.copy()
        image_height, image_width, _ = image.shape
        image_area = self.calculate_area(image_width, image_height)

        # Map thickness values
        thickness_value = thickness

        used_colors = []
        for i, ann in enumerate(annotations):
            color = colors[i]
            color_value = COLOR_DICT[color]
            bbox = list(map(int, ann['bbox']))  # [x_min, y_min, width, height]
            bbox_area = self.calculate_area(bbox[2], bbox[3])

            # Calculate center and axes for the ellipse (circle)
            x_min, y_min, width, height = bbox
            center = (x_min + width // 2, y_min + height // 2)
            axes = (width // 2, height // 2)  # Axes are half of the width and height

            # Draw the ellipse (circle prompt) with the specified thickness
            if color == "contrast":
                white_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                cv2.ellipse(white_image, center, axes, 0, 0, 360, (0, 0, 0), thickness_value)
                color, color_value = contrast_color(white_image, image, used_colors)
            used_colors.append(color)
            cv2.ellipse(image, center, axes, 0, 0, 360, color_value, thickness_value)
            if mask is not None:
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)  # Add to mask for dimming
        return image, used_colors

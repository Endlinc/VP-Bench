import cv2
import numpy as np

from demo.constant import COLOR_DICT
from vp_bench.common.utils import contrast_color
from vp_bench.visual_prompts.visual_prompt import VisualPrompt


class TagAlphabet(VisualPrompt):
    def __init__(self, prominence_threshold=0.05, bg_color=(0, 0, 0)):
        """
        Alphabet label prompt places an alphabetic label (A, B, C, ...) at the center of each instance.
        :param prominence_threshold: The minimum area required for an instance to be considered prominent.
        :param bg_color: Default background color for the label rectangle.
        """
        super().__init__(dimming_enabled=False)
        self.prominence_threshold = prominence_threshold
        self.bg_color = bg_color

    def get_contrary_color(self, bg_color):
        """
        Returns a contrasting text color (black or white) based on the background brightness.
        :param bg_color: Background color in BGR format.
        :return: Contrasting BGR color for text.
        """
        brightness = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
        return (0, 0, 0) if brightness > 128 else (255, 255, 255)

    def apply_prompt(self, image, annotations, image_info, mask, colors, thickness, **kwargs):
        image = image.copy()
        image_height, image_width, _ = image.shape
        image_area = self.calculate_area(image_width, image_height)
        shape = kwargs.get("shape", "square").lower()  # Get shape from kwargs (default to square)

        # Generate alphabet labels: A, B, C, ...
        labels = [chr(65 + i) for i in range(len(annotations))]  # Capital letters (ASCII 65 = 'A')

        used_colors = []
        for i, ann in enumerate(annotations):
            bbox = list(map(int, ann['bbox']))  # [x_min, y_min, width, height]
            bbox_area = self.calculate_area(bbox[2], bbox[3])

            # Calculate center of the bounding box
            x_min, y_min, width, height = bbox
            center_x, center_y = x_min + width // 2, y_min + height // 2

            # Adjust text and background sizes based on thickness
            font_scale = max(0.4 * thickness, 0.5)  # Ensure minimum font scale for readability
            font_thickness = max(thickness, 1)
            text_size, baseline = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            text_width, text_height = text_size
            rect_width = text_width + 15  # Add padding to width
            rect_height = text_height + 15  # Add padding to height

            # Calculate background dimensions and position
            rect_x_min = max(0, center_x - rect_width // 2)
            rect_y_min = max(0, center_y - rect_height // 2)
            rect_x_max = min(image_width, rect_x_min + rect_width)
            rect_y_max = min(image_height, rect_y_min + rect_height)

            color = colors[i]
            color_value = COLOR_DICT[color]
            if color == "contrast":
                white_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                cv2.rectangle(white_image, (rect_x_min, rect_y_min), (rect_x_max, rect_y_max), (0, 0, 0), -1)
                color, color_value = contrast_color(white_image, image, used_colors)
            used_colors.append(color)
            if shape == "square":
                # Draw solid rectangle background
                cv2.rectangle(image, (rect_x_min, rect_y_min), (rect_x_max, rect_y_max), color_value, -1)

            elif shape == "round":
                # Draw solid circular background
                radius = max(rect_width, rect_height) // 2
                cv2.circle(image, (center_x, center_y), radius, color_value, -1)

            # Get contrasting text color
            text_color = self.get_contrary_color(self.bg_color)

            # Calculate exact text position to center it inside the background
            text_x = center_x - text_width // 2 + thickness // 2
            text_y = center_y + text_height // 2 + thickness // 2

            # Draw text exactly centered
            cv2.putText(image, labels[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

        return image, used_colors

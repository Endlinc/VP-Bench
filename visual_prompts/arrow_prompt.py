import cv2
import numpy as np
import math

from demo.constant import COLOR_DICT
from vp_bench.common.utils import contrast_color
from vp_bench.visual_prompts.visual_prompt import VisualPrompt


class ArrowPrompt(VisualPrompt):
    def __init__(self, prominence_threshold=0.1, arrow_length=50):
        super().__init__(dimming_enabled=False)  # Dimming is disabled for arrows
        self.prominence_threshold = prominence_threshold
        self.arrow_length = arrow_length

    def apply_prompt(self, image, annotations, image_info, mask, colors, thickness, **kwargs):
        image = image.copy()
        image_height, image_width, _ = image.shape
        image_area = self.calculate_area(image_width, image_height)

        arrow_shape = kwargs.get("arrow_shape", "standard").lower()  # "standard" or "custom"

        # Map thickness values
        thickness_value = thickness

        used_colors = []
        for i, ann in enumerate(annotations):
            color = colors[i]
            color_value = COLOR_DICT[color]
            bbox = list(map(int, ann['bbox']))  # [x_min, y_min, width, height]
            bbox_area = self.calculate_area(bbox[2], bbox[3])

            x_min, y_min, width, height = bbox

            # Define possible start and end points for arrows
            start_points = {
                'left': (x_min - self.arrow_length, y_min + height // 2),
                'right': (x_min + width + self.arrow_length, y_min + height // 2),
                'top': (x_min + width // 2, y_min - self.arrow_length),
                'bottom': (x_min + width // 2, y_min + height + self.arrow_length)
            }

            edge_points = {
                'left': (x_min, y_min + height // 2),
                'right': (x_min + width, y_min + height // 2),
                'top': (x_min + width // 2, y_min),
                'bottom': (x_min + width // 2, y_min + height)
            }

            side, start_point, end_point = self.find_valid_arrow_direction(start_points, edge_points, image_width,
                                                                           image_height)
            if start_point is None or end_point is None:
                start_point = start_points['left']
                end_point = edge_points['left']

            if color == "contrast":
                white_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                self.draw_standard_arrow(white_image, start_point, end_point, (0, 0, 0), thickness_value)
                color, color_value = contrast_color(white_image, image, used_colors)
            used_colors.append(color)
            # Draw the arrow on the image
            if arrow_shape == "standard":
                self.draw_standard_arrow(image, start_point, end_point, color_value, thickness_value)
            elif arrow_shape == "custom":
                self.draw_custom_arrow(image, start_point, end_point, color_value, thickness_value)
            else:
                raise ValueError("Invalid arrow shape. Use 'standard' or 'custom'.")
        return image, used_colors

    @staticmethod
    def draw_standard_arrow(image, start_point, end_point, color=(0, 255, 0), thickness=2, tip_length=0.2):
        """
        Draw a standard arrow with an arrowhead using OpenCV's arrowedLine.
        """
        cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=tip_length)

    @staticmethod
    def draw_custom_arrow(image, start_point, end_point, color=(0, 255, 0), thickness=2):
        """
        Draw a custom arrow with a larger arrowhead using lines and polygons.
        """
        # Draw the main arrow line
        cv2.line(image, start_point, end_point, color, thickness)

        # Calculate the arrowhead points
        arrow_angle = 30  # Degrees
        arrow_head_length = 20  # Length of the arrowhead

        # Calculate the direction vector of the arrow
        dx, dy = end_point[0] - start_point[0], end_point[1] - start_point[1]
        angle = math.atan2(dy, dx)

        # Left arrowhead point
        left_x = int(end_point[0] - arrow_head_length * math.cos(angle - math.radians(arrow_angle)))
        left_y = int(end_point[1] - arrow_head_length * math.sin(angle - math.radians(arrow_angle)))

        # Right arrowhead point
        right_x = int(end_point[0] - arrow_head_length * math.cos(angle + math.radians(arrow_angle)))
        right_y = int(end_point[1] - arrow_head_length * math.sin(angle + math.radians(arrow_angle)))

        # Draw the arrowhead
        arrowhead_points = np.array([[end_point[0], end_point[1]], [left_x, left_y], [right_x, right_y]], np.int32)
        cv2.fillPoly(image, [arrowhead_points], color)

    @staticmethod
    def find_valid_arrow_direction(start_points, edge_points, image_width, image_height):
        """
        Find a valid arrow direction that stays within the image boundaries.
        """
        for side in start_points.keys():
            start_point = start_points[side]
            end_point = edge_points[side]

            if (0 <= start_point[0] < image_width and 0 <= start_point[1] < image_height and
                    0 <= end_point[0] < image_width and 0 <= end_point[1] < image_height):
                return side, start_point, end_point

        return None, None, None

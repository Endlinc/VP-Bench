import cv2
import numpy as np
from pycocotools import mask as cocomask
from scipy.interpolate import splprep, splev

from common.constant import COLOR_DICT
from common.utils import contrast_color
from visual_prompts.visual_prompt import VisualPrompt


class ContourPrompt(VisualPrompt):
    def __init__(self, prominence_threshold=0.05):
        super().__init__(dimming_enabled=False)  # Dimming is not enabled for contours
        self.prominence_threshold = prominence_threshold

    def apply_prompt(self, image, annotations, image_info, mask, colors, thickness, **kwargs):
        image_height, image_width, _ = image.shape
        image_area = self.calculate_area(image_width, image_height)

        # Get style from kwargs (default to "contour")
        style = kwargs.get("style", "contour").lower()

        used_colors = []
        for i, ann in enumerate(annotations):
            color = colors[i]
            color_value = COLOR_DICT[color]
            segmentation = ann.get('segmentation', None)
            bbox_area = self.calculate_area(ann['bbox'][2], ann['bbox'][3])  # Fallback to bounding box area

            if segmentation:
                if isinstance(segmentation, list):  # Handle polygon segmentation
                    for seg in segmentation:

                        polygon = np.array(seg).reshape((-1, 2)).astype(np.int32)
                        if color == "contrast":
                            white_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                            cv2.polylines(white_image, [polygon], isClosed=True, color=(0, 0, 0), thickness=thickness)
                            color, color_value = contrast_color(white_image, image, used_colors)
                        used_colors.append(color)
                        if style == "contour":
                            # Draw exact contour
                            cv2.polylines(image, [polygon], isClosed=True, color=color_value, thickness=thickness)

                        elif style == "loose_contour":
                            # Randomly expand contour points outward
                            loose_contour = self.expand_and_smooth_polygon(polygon, image_width, image_height)
                            cv2.polylines(image, [loose_contour], isClosed=True, color=color_value, thickness=thickness)

                elif isinstance(segmentation, dict) and 'counts' in segmentation:  # Handle RLE segmentation
                    if isinstance(segmentation['counts'], list):
                        rle = cocomask.frPyObjects(segmentation, image_height, image_width)
                        mask_decoded = cocomask.decode(rle)
                    else:
                        mask_decoded = cocomask.decode(segmentation)

                    # Extract contours from the mask
                    contours, _ = cv2.findContours(mask_decoded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if color == "contrast":
                        white_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                        cv2.polylines(white_image, [contours], isClosed=True, color=(0, 0, 0), thickness=thickness)
                        color, color_value = contrast_color(white_image, image, used_colors)
                    used_colors.append(color)

                    if style == "contour":
                        # Draw exact contour
                        for contour in contours:
                            cv2.polylines(image, [contour], isClosed=True, color=color_value, thickness=thickness)

                    elif style == "loose_contour":
                        # Randomly expand contour points outward
                        for contour in contours:
                            loose_contour = self.expand_and_smooth_polygon(contour.squeeze(), image_width, image_height)
                            cv2.polylines(image, [loose_contour], isClosed=True, color=color_value, thickness=thickness)

        return image, used_colors

    @staticmethod
    def expand_and_smooth_polygon(polygon, image_width, image_height, expansion_factor=1.2, smoothing=True):
        polygon = np.atleast_2d(polygon)
        centroid = polygon.mean(axis=0)  # Calculate the centroid of the polygon

        expanded_polygon = []
        for point in polygon:
            # Calculate the vector from the centroid to the point
            vector = point - centroid
            # Scale the vector by the expansion factor
            expanded_point = centroid + vector * expansion_factor
            # Clip the expanded point to the image boundaries
            expanded_point[0] = np.clip(expanded_point[0], 0, image_width - 1)
            expanded_point[1] = np.clip(expanded_point[1], 0, image_height - 1)
            expanded_polygon.append(expanded_point)

        expanded_polygon = np.array(expanded_polygon, dtype=np.float32)

        if smoothing:
            # Handle edge cases for insufficient or degenerate polygons
            if len(expanded_polygon) < 3:
                print("Polygon has less than 3 points, returning original points.")
                return expanded_polygon.astype(np.int32)

            # Ensure points are not degenerate (all points identical)
            if np.allclose(expanded_polygon, expanded_polygon[0]):
                print("Polygon points are degenerate, returning original points.")
                return expanded_polygon.astype(np.int32)

            # Use spline interpolation to create smooth curves
            try:
                tck, u = splprep(expanded_polygon.T, s=0.1, per=True)  # `per=True` ensures closed curve
                smooth_points = splev(np.linspace(0, 1, len(expanded_polygon) * 10), tck)
                smooth_polygon = np.vstack(smooth_points).T  # Convert back to Nx2 format
                return smooth_polygon.astype(np.int32)
            except ValueError as e:
                print(f"Error during spline interpolation: {e}. Returning original polygon.")
                return expanded_polygon.astype(np.int32)

        return expanded_polygon.astype(np.int32)

import cv2
import numpy as np
from pycocotools import mask as cocomask

from common.constant import COLOR_DICT
from common.utils import contrast_color
from visual_prompts.visual_prompt import VisualPrompt


class MaskPrompt(VisualPrompt):
    def __init__(self, prominence_threshold=0.05):
        super().__init__(dimming_enabled=True)  # Dimming is enabled for masks
        self.prominence_threshold = prominence_threshold

    def apply_prompt(self, image, annotations, image_info, mask, colors, thickness, **kwargs):
        image = image.copy()
        image_height, image_width, _ = image.shape
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        style = kwargs.get("style", "contour_fill").lower()  # "contour_fill" or "fill_only"

        used_colors = []
        for i, ann in enumerate(annotations):
            color = colors[i]
            color_value = COLOR_DICT[color]
            segmentation = ann.get('segmentation', None)

            if segmentation:
                if isinstance(segmentation, list):  # Segmentation polygons
                    for seg in segmentation:
                        polygon = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)

                        if color == "contrast":
                            white_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                            cv2.polylines(white_image, [polygon], isClosed=True, color=(0, 0, 0), thickness=thickness)
                            color, color_value = contrast_color(white_image, image, used_colors)
                        used_colors.append(color)

                        if style == "contour_fill":
                            # Draw contour
                            cv2.polylines(image, [polygon], isClosed=True, color=color_value, thickness=thickness)
                            # Fill the inner area with a lighter translucent version of the contour color
                            fill_color = tuple([min(255, int(c + (255 - c) * 0.5)) for c in color_value])  # Lighter color
                            temp_fill = image.copy()
                            cv2.fillPoly(temp_fill, [polygon], color=fill_color)
                            # Blend the filled area with the original image
                            image = cv2.addWeighted(temp_fill, 0.5, image, 0.5, 0)

                        elif style == "fill_only":
                            # Fill the inner area with a lighter translucent version of the contour color
                            fill_color = tuple([min(255, int(c + (255 - c) * 0.5)) for c in color_value])  # Lighter color
                            temp_fill = image.copy()
                            cv2.fillPoly(temp_fill, [polygon], color=fill_color)
                            # Blend the filled area with the original image
                            image = cv2.addWeighted(temp_fill, 0.5, image, 0.5, 0)

                elif isinstance(segmentation, dict) and 'counts' in segmentation:  # RLE segmentation
                    if isinstance(segmentation['counts'], list):
                        rle = cocomask.frPyObjects(segmentation, image_height, image_width)
                        mask_decoded = cocomask.decode(rle)
                    else:
                        mask_decoded = cocomask.decode(segmentation)

                    mask_decoded = (mask_decoded * 255).astype(np.uint8)

                    contours, _ = cv2.findContours(mask_decoded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if color == "contrast":
                        white_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                        cv2.fillPoly(white_image, contours, color=(0, 0, 0))
                        color, color_value = contrast_color(white_image, image, used_colors)
                    used_colors.append(color)

                    if style == "contour_fill":
                        # Draw contour
                        cv2.drawContours(image, contours, -1, color_value, thickness=thickness)

                        # Fill the inner area with a lighter translucent version of the contour color
                        fill_color = tuple([min(255, int(c + (255 - c) * 0.5)) for c in color_value])  # Lighter color
                        temp_fill = image.copy()
                        cv2.fillPoly(temp_fill, contours, color=fill_color)
                        # Blend the filled area with the original image
                        image = cv2.addWeighted(temp_fill, 0.5, image, 0.5, 0)

                    elif style == "fill_only":
                        # Fill the inner area with a lighter translucent version of the contour color
                        fill_color = tuple([min(255, int(c + (255 - c) * 0.5)) for c in color_value])  # Lighter color
                        cv2.fillPoly(image, contours, color=fill_color)
        return image, used_colors

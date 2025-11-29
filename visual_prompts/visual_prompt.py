import os
from abc import ABC, abstractmethod

import cv2
import numpy as np


class VisualPrompt(ABC):
    def __init__(self, dimming_enabled=False):
        self.dimming_enabled = dimming_enabled  # Whether dimming is enabled for this prompt

    @abstractmethod
    def apply_prompt(self, image, annotations, image_info, mask, colors, thickness):
        """Apply the specific prompt (bounding box, arrow, circle, mask) on the image and mask.
        :param colors: List of colors corresponding to each annotation.
        :param mask:
        :param image_info:
        :param image:
        :param annotations:
        :param thickness:
        """
        pass

    def apply(self, image, annotations, image_info, colors, thickness='medium', save_path=None, **kwargs):
        """
        Apply the prompt without optional dimming.
        :param colors: List of colors corresponding to each annotation.
        :param thickness: Controls the thickness of the visual prompt visualization ('low', 'medium', 'high').
        :param image: The image to apply the prompt on.
        :param annotations: The COCO-style annotations for objects in the image.
        :param image_info: Image metadata.
        :param save_path: If specified, save the resulting image.
        """
        thickness_value = thickness

        if len(colors) < len(annotations):
            raise ValueError("Length of 'colors' must greater than the number of annotations.", len(colors), len(annotations))

        image, used_color = self.apply_prompt(image, annotations, image_info, None, colors, thickness_value,
                          **kwargs)  # Direct drawing on the image
        if save_path:
            cv2.imwrite(save_path, image)  # Save the image to the specified path
        else:
            self.display_image('Image', image)
        return used_color

    @staticmethod
    def load_image(image_dir, file_name):
        image_file = os.path.join(image_dir, file_name)
        image = cv2.imread(image_file)
        if image is None:
            raise FileNotFoundError(f"Error: Could not load image from {image_file}")
        return image

    @staticmethod
    def calculate_area(width, height):
        return width * height

    @staticmethod
    def is_prominent(area, image_area, threshold):
        return area > threshold * image_area

    @staticmethod
    def draw_text(image, text, position, color=(0, 255, 0), font_scale=0.9, thickness=2):
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    @staticmethod
    def display_image(window_name, image):
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

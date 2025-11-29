import os
from abc import ABC, abstractmethod
from PIL import Image  # Add PIL import

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
        :param image: The image to apply the prompt on (PIL Image or numpy array in BGR format).
        :param annotations: The COCO-style annotations for objects in the image.
        :param image_info: Image metadata.
        :param save_path: If specified, save the resulting image.
        :return: Visualized image as a PIL Image.
        """
        thickness_value = thickness

        if len(colors) < len(annotations):
            raise ValueError("Length of 'colors' must be greater than the number of annotations.",
                             len(colors), len(annotations))

        # Convert PIL Image to numpy array (BGR format) if needed
        if isinstance(image, Image.Image):
            # PIL uses RGB, convert to BGR for OpenCV processing
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Process the image with the prompt
        image, used_color = self.apply_prompt(
            image, annotations, image_info, None, colors, thickness_value,** kwargs
        )

        # Save image if path is provided (uses OpenCV's BGR format)
        if save_path:
            cv2.imwrite(save_path, image)
        else:
            self.display_image('Image', image)

        # Convert back to PIL Image (RGB format)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)

    # Rest of the class methods remain unchanged...
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

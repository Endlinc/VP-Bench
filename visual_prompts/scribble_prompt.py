import random
import numpy as np
import cv2
from pycocotools import mask as cocomask

from common.constant import COLOR_DICT
from common.utils import contrast_color
from visual_prompts.visual_prompt import VisualPrompt


class ScribblePrompt(VisualPrompt):
    def __init__(self, prominence_threshold=0.05, erode_scale=0.1):
        """
        Scribble prompt generates scribbles based on an improved generation method.
        :param prominence_threshold: Minimum area for an instance to be considered prominent.
        :param erode_scale: Scale factor for binary erosion.
        """
        super().__init__(dimming_enabled=False)
        self.prominence_threshold = prominence_threshold
        self.erode_scale = erode_scale

    def apply_prompt(self, image, annotations, image_info, mask, colors, thickness):
        image = image.copy()
        image_height, image_width, _ = image.shape
        image_area = self.calculate_area(image_width, image_height)

        # Map thickness values
        thickness_value = thickness

        used_colors = []
        for i, ann in enumerate(annotations):
            color = colors[i]
            color_value = COLOR_DICT[color]
            segmentation = ann.get('segmentation', None)
            bbox_area = self.calculate_area(ann['bbox'][2], ann['bbox'][3])

            if segmentation:
                mask_decoded = self.get_mask_from_segmentation(segmentation, image_info)

                if mask_decoded.sum() == 0:
                    raise ValueError("Decoded mask is empty.")

                # Perform connected component analysis
                num_labels, labels = cv2.connectedComponents(mask_decoded.astype(np.uint8))

                for label in range(1, num_labels):
                    component_mask = (labels == label).astype(np.uint8)

                    # Skip small components
                    if component_mask.sum() < 50:
                        continue

                    # Binary erosion
                    eroded_mask = self.binary_erosion(component_mask)

                    if eroded_mask.sum() == 0:
                        continue

                    # Ensure binary and single-channel input for skeletonization
                    eroded_mask = (eroded_mask > 0).astype(np.uint8) * 255
                    if len(eroded_mask.shape) == 3:
                        eroded_mask = cv2.cvtColor(eroded_mask, cv2.COLOR_BGR2GRAY)

                    # Skeletonize the eroded mask
                    skeleton = cv2.ximgproc.thinning(eroded_mask)
                    if skeleton.sum() == 0:
                        raise ValueError("Skeleton is empty after thinning.")

                    # Generate scribbles
                    if color == "contrast":
                        white_image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                        self.generate_scribble(white_image, skeleton, (0, 0, 0), thickness_value)
                        color, color_value = contrast_color(white_image, image, used_colors)
                    used_colors.append(color)
                    self.generate_scribble(image, skeleton, color_value, thickness_value)

                    if mask is not None:
                        mask[component_mask > 0] = 255
        return image, used_colors

    def binary_erosion(self, mask):
        """
        Perform binary erosion on the mask. Stop if the mask becomes too small.
        :param mask: Binary mask to erode.
        :return: Eroded mask.
        """
        erode_kernel_size = max(1, int(self.erode_scale * np.sqrt(mask.sum())))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))

        eroded = mask.copy()
        while eroded.sum() > 10:  # Stop erosion if fewer than 10 pixels remain
            new_eroded = cv2.erode(eroded, kernel)
            if new_eroded.sum() == eroded.sum():  # No further erosion possible
                break
            eroded = new_eroded

        if eroded.sum() == 0:
            return mask  # Return original mask if erosion completely removes it

        return eroded

    def generate_scribble(self, image, skeleton, color, thickness):
        """
        Generate scribbles by fitting polynomial curves to skeleton points.
        :param image: Image to draw on.
        :param skeleton: Skeleton of the mask.
        :param color: Scribble color.
        :param thickness: Line thickness.
        """
        points = np.column_stack(np.where(skeleton > 0))

        if len(points) < 2:
            return

        x = points[:, 1]  # X-coordinates
        y = points[:, 0]  # Y-coordinates

        # Fit a polynomial of degree 2
        try:
            coeffs = np.polyfit(x, y, 2)
            poly_func = np.poly1d(coeffs)

            # Generate smooth curve points
            curve_x = np.linspace(x.min(), x.max(), num=100).astype(np.int32)
            curve_y = poly_func(curve_x).astype(np.int32)

            # Filter valid points within the image boundaries
            valid_indices = (curve_x >= 0) & (curve_x < image.shape[1]) & (curve_y >= 0) & (curve_y < image.shape[0])
            curve_points = np.column_stack((curve_x[valid_indices], curve_y[valid_indices]))

            if len(curve_points) < 2:
                return

            # Draw the smooth curve
            for j in range(len(curve_points) - 1):
                cv2.line(image, tuple(curve_points[j]), tuple(curve_points[j + 1]), color, thickness)
        except Exception as e:
            raise ValueError(f"Error during polynomial fitting: {e}")

    def get_mask_from_segmentation(self, segmentation, image_info):
        """
        Decode the segmentation data into a mask.
        :param segmentation: COCO segmentation data.
        :param image_info: Metadata of the image.
        :return: Mask with the instance area filled.
        """
        image_height = image_info['height']
        image_width = image_info['width']

        if isinstance(segmentation, list):
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            for seg in segmentation:
                polygon = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [polygon], 1)
        elif isinstance(segmentation, dict) and 'counts' in segmentation:
            rle = cocomask.frPyObjects(segmentation, image_height, image_width)
            mask = cocomask.decode(rle).astype(np.uint8)
        else:
            mask = np.zeros((image_height, image_width), dtype=np.uint8)

        return mask

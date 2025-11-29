from visual_prompts.tag_alphabet import TagAlphabet
from visual_prompts.arrow_prompt import ArrowPrompt
from visual_prompts.bbox_prompt import BoundingBoxPrompt
from visual_prompts.circle_prompt import CirclePrompt
from visual_prompts.mask_prompt import MaskPrompt
from visual_prompts.contour_prompt import ContourPrompt
from visual_prompts.tag_digit import TagDigit
from visual_prompts.point_prompt import SinglePointPrompt
from visual_prompts.scribble_prompt import ScribblePrompt


class PromptManager:
    def __init__(self):
        self.prompts = {
            'bounding_box': BoundingBoxPrompt,
            'mask': ContourPrompt,
            'fill_contour': MaskPrompt,
            'circle': CirclePrompt,
            'arrow': ArrowPrompt,
            'number_label': TagDigit,  # Add NumberLabelPrompt
            'alphabet_label': TagAlphabet,  # Add AlphabetLabelPrompt
            'single_point': SinglePointPrompt,
            'scribble': ScribblePrompt
        }

    def get_prompt(self, prompt_type, prominence_threshold=0.05, **kwargs):
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        return self.prompts[prompt_type](prominence_threshold=prominence_threshold, **kwargs)

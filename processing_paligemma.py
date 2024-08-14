from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch


# We can Change them latter
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompts(prefix_prompt, bos_token, image_seq_len, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(image: np.ndarray, 
        scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    rescaled_image = image * rescale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def resize(image: Image,
           size: Tuple[int, int],
           resamle: Image.Resampling = None,
           reducing_gap: Optional[int] = None,
           ) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resamle, reducing_gap=reducing_gap
    )
    
    return resized_image

def normalize(
    image: np.ndarray, 
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def process_images(image: List[Image.Image],
                   size: Dict[str, int]= None,
                   resample: Image.Resampling=None,
                   rescale_factor: float= None,
                   image_mean: Optional[Unionp[float, List[float]]] = None,
                   image_std: Optional[Union[float, List[float]]] = None)-> List[np.ndarray]:
    height, width = size[0], size[1]

    # Convert each image to numpy array
    images = [np.array(image) for image in images]

    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images

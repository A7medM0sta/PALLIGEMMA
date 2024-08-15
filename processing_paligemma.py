from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Rescales the pixel values of an image by a given scale factor.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        scale (float): The factor by which to scale the pixel values.
        dtype (np.dtype, optional): The desired data type of the output image. Defaults to np.float32.

    Returns:
        np.ndarray: The rescaled image as a NumPy array with the specified data type.
    """
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    """
    Resizes the input image to the specified size.

    Args:
        image (Image): The input image as a PIL Image object.
        size (Tuple[int, int]): The desired size as a tuple (height, width).
        resample (Image.Resampling, optional): The resampling filter to use. Defaults to None.
        reducing_gap (Optional[int], optional): The reducing gap parameter for the resize method. Defaults to None.

    Returns:
        np.ndarray: The resized image as a NumPy array.
    """
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """
    Normalizes the pixel values of an image by subtracting the mean and dividing by the standard deviation.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        mean (Union[float, Iterable[float]]): The mean value(s) to subtract from the image.
        std (Union[float, Iterable[float]]): The standard deviation value(s) to divide the image by.

    Returns:
        np.ndarray: The normalized image as a NumPy array.
    """
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Processes a list of images by resizing, rescaling, normalizing, and reordering the channel dimension.

    Args:
        images (List[Image.Image]): List of input images as PIL Image objects.
        size (Dict[str, int]): Dictionary containing the desired size with keys 'height' and 'width'.
        resample (Image.Resampling, optional): The resampling filter to use. Defaults to None.
        rescale_factor (float, optional): The factor by which to rescale the pixel values. Defaults to None.
        image_mean (Optional[Union[float, List[float]]], optional): The mean value(s) for normalization. Defaults to None.
        image_std (Optional[Union[float, List[float]]], optional): The standard deviation value(s) for normalization. Defaults to None.

    Returns:
        List[np.ndarray]: List of processed images as NumPy arrays.
    """
    height, width = size['height'], size['width']
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


class PaliGemmaProcessor:
    """
    A processor class for handling text and image inputs for the PaliGemma model.

    Attributes:
        IMAGE_TOKEN (str): Special token used to represent images.
        image_seq_length (int): The number of image tokens to prepend to the text prompt.
        image_size (int): The size to which images should be resized.
        image_token_id (int): The token ID for the image token.
        tokenizer: The tokenizer used for processing text inputs.

    Methods:
        __call__(text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True) -> dict:
            Processes the text and image inputs and returns a dictionary containing the processed inputs.
    """

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        """
        Initializes the PaliGemmaProcessor with the given tokenizer, number of image tokens, and image size.

        Args:
            tokenizer: The tokenizer used for processing text inputs.
            num_image_tokens (int): The number of image tokens to prepend to the text prompt.
            image_size (int): The size to which images should be resized.
        """
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """
        Processes the text and image inputs and returns a dictionary containing the processed inputs.

        Args:
            text (List[str]): List of input text prompts.
            images (List[Image.Image]): List of input images as PIL Image objects.
            padding (str, optional): Padding strategy for the tokenizer. Defaults to "longest".
            truncation (bool, optional): Whether to truncate the text inputs. Defaults to True.

        Returns:
            dict: A dictionary containing the processed inputs, including pixel values and tokenized text.
        """
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images,
            size={"height": self.image_size, "width": self.image_size},
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        # Convert the list of numpy arrays to a single numpy array with shape [Batch_Size, Channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
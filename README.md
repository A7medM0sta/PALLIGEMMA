# PALLIGEMA

## Overview
PaliGemma is an open Vision-Language Model (VLM) that is based on the SigLIP-So400m vision encoder and the Gemma-2B language model. It is trained to be a versatile and broadly knowledgeable base model that is effective to transfer. It achieves strong performance on a wide variety of open-world tasks. We evaluate PaliGemma on almost 40 diverse tasks including standard VLM benchmarks, but also more specialized tasks such as remote-sensing and segmentation.
## Features

- **Image Processing**: Resize, normalize, and rescale images.
- **Text Tokenization**: Tokenize text inputs and prepend special tokens.
- **Model Inference**: Run inference on the PaliGemma model with text and image inputs.
- **CPU/GPU Support**: Run inference on CPU or GPU.
- **Sampling Options**: Control sampling temperature and top-p value.
- **Output Generation**: Generate text output from the model.
- **Task Evaluation**: Evaluate the model on a variety of tasks.
- **Task-Specific Inputs**: Process task-specific inputs for the model.
## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/A7medM0sta/PALLIGEMA.git
    cd PALLIGEMA
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running Inference

To run inference using the PaliGemma model, use the provided `launch_inference.sh` script:

```sh
bash Setup/launch_inference.sh
```

### Script Parameters

- `MODEL_PATH`: Path to the model weights.
- `PROMPT`: Text prompt for the model.
- `IMAGE_FILE_PATH`: Path to the input image.
- `MAX_TOKENS_TO_GENERATE`: Maximum number of tokens to generate.
- `TEMPERATURE`: Sampling temperature.
- `TOP_P`: Top-p sampling value.
- `DO_SAMPLE`: Whether to sample or use greedy decoding.
- `ONLY_CPU`: Whether to force the use of CPU only.

### Example

```sh
#!/bin/bash

MODEL_PATH="/path/to/model"
PROMPT="this building is "
IMAGE_FILE_PATH="images/pic1.jpeg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU
```
## Results



## Code Structure

- `inference.py`: Main script for running model inference.
- `Evaluation.py`: Contains the logic for processing images and text, and running the model.
- `Setup/launch_inference.sh`: Shell script to launch the inference process.

## Classes and Functions

### `PaliGemmaProcessor`

A processor class for handling text and image inputs for the PaliGemma model.

#### Methods

- `__init__(self, tokenizer, num_image_tokens: int, image_size: int)`: Initializes the processor.
- `__call__(self, text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True) -> dict`: Processes the text and image inputs and returns a dictionary containing the processed inputs.

### `process_images`

Processes a list of images by resizing, rescaling, normalizing, and reordering the channel dimension.

### `resize`

Resizes the input image to the specified size.

### `normalize`

Normalizes the pixel values of an image by subtracting the mean and dividing by the standard deviation.

## License

This project is licensed under the MIT License.
```
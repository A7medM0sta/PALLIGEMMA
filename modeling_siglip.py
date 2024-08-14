from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiglipVisionConfig:
    """
    Configuration class for the Siglip Vision model.

    Args:
        hidden_size (int, optional): The size of the hidden layers in the transformer. Defaults to 768.
        intermediate_size (int, optional): The size of the "intermediate" (i.e., feed-forward) layer in the transformer architecture. Defaults to 3072.
        num_hidden_layers (int, optional): Number of hidden layers in the transformer. Defaults to 12.
        num_attention_heads (int, optional): Number of attention heads for each attention layer in the transformer's encoder. Defaults to 12.
        num_channels (int, optional): Number of input channels for the image. Defaults to 3.
        image_size (int, optional): The size of the input images. Defaults to 224.
        patch_size (int, optional): The size of the patches to be extracted from the images. Defaults to 16.
        layer_norm_eps (float, optional): The epsilon used by the layer normalization layers. Defaults to 1e-6.
        attention_dropout (float, optional): The dropout probability for the attention probabilities. Defaults to 0.0.
        num_image_tokens (int, optional): The total number of distinct image tokens. If None, it will be computed from the image size and patch size.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            # this indicate no padding is added
            bias=False
        )

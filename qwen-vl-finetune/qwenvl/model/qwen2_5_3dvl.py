from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
import os
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.distributed as dist

from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionBlock, 
    Qwen2_5_VisionTransformerPretrainedModel, 
    Qwen2_5_VLPreTrainedModel, 
    Qwen2_5_VisionRotaryEmbedding, 
    Qwen2_5_VisionPatchEmbed, 
    Qwen2_5_VLPatchMerger,
    QWEN2_5_VL_INPUTS_DOCSTRING,
    Qwen2_5_VLCausalLMOutputWithPast,
)
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig


def is_rank0():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


import torch
import torch.nn as nn
import math

class CamAwarePositionEmbedding(nn.Module):
    def __init__(self, hidden_size: int, temperature: float = 10000.0):
        """
        Initializes the positional embedding module.

        Args:
            hidden_size (int): The target hidden dimension of the model (C_out). Must be an even number.
            temperature (float): The temperature hyperparameter in the sinusoidal encoding, used to control the frequency range.
        """
        super().__init__()
        
        if hidden_size % 2 != 0:
            raise ValueError(f"The hidden_size must be an even number, but received {hidden_size}")
            
        # Halve the hidden_size to encode x and y dimensions separately
        self.dim_t = hidden_size // 2
        self.temperature = temperature

        # Create the division term for frequency scaling
        # Shape: [dim_t / 2]
        div_term = torch.exp(
            torch.arange(0, self.dim_t, 2, dtype=torch.float) *
            (-math.log(self.temperature) / self.dim_t)
        )
        # Register as a buffer, so it moves to the correct device with the model but is not a trainable parameter
        self.register_buffer("div_term", div_term)
        
        # Define a small MLP to add learnability to the fixed sinusoidal encoding
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Generates positional embeddings for the input coordinates.

        Args:
            coords (torch.Tensor): A tensor of shape [N, 2] containing the (x, y) coordinates
                                   for N points. It's recommended that the coordinates are
                                   normalized to a range like [0, 1].

        Returns:
            torch.Tensor: A tensor of shape [N, hidden_size] containing the positional embeddings.
        """
        # Assume input coordinates are normalized; scale them by 2*pi for better periodic features.
        # This scaling can be adjusted based on the actual coordinate range.
        coords = coords * 2.0 * math.pi
        
        # Separate x and y coordinates and prepare for broadcasting
        # Shape of coords_x, coords_y: [N] -> [N, 1]
        coords_x = coords[:, 0].unsqueeze(1)
        coords_y = coords[:, 1].unsqueeze(1)
        
        # Calculate the product of coordinates and frequencies
        # [N, 1] * [dim_t / 2] -> [N, dim_t / 2]
        pos_x = coords_x * self.div_term
        pos_y = coords_y * self.div_term
        
        # Initialize embedding vectors
        embedding_x = torch.zeros(coords.shape[0], self.dim_t, device=coords.device)
        embedding_y = torch.zeros(coords.shape[0], self.dim_t, device=coords.device)
        
        # Encode using interleaved sine and cosine functions
        # Use sin for even indices, cos for odd indices
        embedding_x[:, 0::2] = torch.sin(pos_x)
        embedding_x[:, 1::2] = torch.cos(pos_x)
        embedding_y[:, 0::2] = torch.sin(pos_y)
        embedding_y[:, 1::2] = torch.cos(pos_y)
        
        # Concatenate the x and y embeddings to form the full [N, hidden_size] encoding
        positional_embedding = torch.cat((embedding_x, embedding_y), dim=1)
        positional_embedding = positional_embedding.to(coords.dtype)
        
        # Add learnability through the MLP
        output = self.mlp(positional_embedding)
        
        return output


class Qwen2_5_3DVL_ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # Add ray direction embedding module
        self.cam_aware_embedding_module = CamAwarePositionEmbedding(config.hidden_size)
        self.cam_aware_merger = Qwen2_5_VLPatchMerger(
            dim=config.hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=self.visual.spatial_merge_size,
        )


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        # Force it to output loading_info so we can see which keys were missing/unexpected
        kwargs["output_loading_info"] = True
        
        # Load model + loading_info (which has missing/unexpected keys, etc.)
        model, loading_info = super().from_pretrained(  # type: ignore
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

        missing_keys = loading_info["missing_keys"]
        return model

    @add_start_docstrings_to_model_forward(QWEN2_5_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2_5_VLCausalLMOutputWithPast, config_class="Qwen2_5_VLConfig")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        camera_aware_position_embeddings: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            camera_aware_position_embeddings (`torch.Tensor` of shape `(batch_size, height, width, 2)`, *optional*):
                TODO

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                # Get visual features from the visual encoder
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                
                # Add 3D camera-aware positional embedding if ray directions are provided
                if camera_aware_position_embeddings is not None:
                    # Generate ray direction embeddings
                    ray_embeds = self.cam_aware_embedding_module(camera_aware_position_embeddings)
                    ray_embeds = self.cam_aware_merger(ray_embeds)
                    
                    # Add ray embeddings to image embeddings
                    image_embeds = image_embeds + ray_embeds
                
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

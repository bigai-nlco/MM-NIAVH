import math
from typing import Optional, List, Tuple, Union
from einops import rearrange, repeat, pack, unpack

import torch
from torch import nn
from transformers.activations import ACT2FN

from .self_segment import segment

class Residual(nn.Module):
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.layernorm = nn.LayerNorm(output_size, eps=config.mm_layer_norm_eps)
        self.dropout = nn.Dropout(config.mm_hidden_dropout_prob)

    def forward(
            self,
            hidden_states: torch.Tensor,
            input_tensor: torch.Tensor,
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.mm_hidden_size
        self.num_attention_heads = config.mm_num_attention_heads
        self.attention_head_size = config.mm_hidden_size // config.mm_num_attention_heads

        assert config.mm_hidden_size % config.mm_num_attention_heads == 0

        self.k_proj = nn.Linear(config.mm_hidden_size, config.mm_hidden_size)
        self.v_proj = nn.Linear(config.mm_hidden_size, config.mm_hidden_size)
        self.q_proj = nn.Linear(config.mm_hidden_size, config.mm_hidden_size)
        self.dropout = nn.Dropout(config.mm_attention_probs_dropout_prob)

        self.residual = Residual(config.mm_hidden_size, config.mm_hidden_size, config)

    def transpose_for_scores(self, x):
        # B, L, D --> B, H, L, HD
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ):
        query = self.transpose_for_scores(self.q_proj(hidden_states))

        if encoder_hidden_states is not None:
            # cross attention
            if past_key_value is not None:
                # use cache
                key = past_key_value[0]
                value = past_key_value[1]
                attention_mask = encoder_attention_mask
            else:
                key = self.transpose_for_scores(self.k_proj(encoder_hidden_states))
                value = self.transpose_for_scores(self.v_proj(encoder_hidden_states))
                attention_mask = encoder_attention_mask
            # cache key & value for crossattention
            past_key_value = (key, value)
        else:
            # self attention
            if past_key_value is not None:
                # use cache
                key = self.transpose_for_scores(self.k_proj(hidden_states))
                value = self.transpose_for_scores(self.v_proj(hidden_states))
                key = torch.cat([past_key_value[0], key], dim=2)
                value = torch.cat([past_key_value[1], value], dim=2)
            else:
                key = self.transpose_for_scores(self.k_proj(hidden_states))
                value = self.transpose_for_scores(self.v_proj(hidden_states))

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) # B, H, N, M

        # TODO position encoding

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores += attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        output = torch.matmul(attention_probs, value)
        output = output.permute(0, 2, 1, 3).contiguous()
        output_shape = output.size()[:-2] + (self.hidden_size,)
        output = output.view(output_shape)

        output = self.residual(output, hidden_states)

        outputs = (output, attention_probs) if output_attentions else (output, )
        if past_key_value is not None:
            outputs = outputs + (past_key_value,)


        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.selfattention = Attention(config)
        self.crossattention = Attention(config)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.mm_intermediate_size),
            ACT2FN[config.mm_hidden_act],
        )
        self.residual = Residual(config.mm_intermediate_size, config.mm_hidden_size, config)

    def ffn(self, attention_output):
        intermediate_output = self.mlp(attention_output)
        layer_output = self.residual(intermediate_output, attention_output)
        return layer_output

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ):
        if past_key_value is not None:
            self_past_key_value = past_key_value[:2]
        else:
            self_past_key_value = None
        self_attention_outputs = self.selfattention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_past_key_value
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        
        present_key_value = None
        if encoder_hidden_states is not None:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
            if past_key_value is not None:
                cross_past_key_value = past_key_value[:2]
            else:
                cross_past_key_value = None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            
            outputs = outputs + cross_attention_outputs[1:-1]
            present_key_value = present_key_value + cross_attention_outputs[-1]

        output = self.ffn(attention_output)
        outputs = (output,) + outputs
        return outputs

class TransformerProjector(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            ACT2FN[config.mm_hidden_act],
        )

        # configuration for memory
        self.num_memory_tokens = 32
        self.read_memory_emb = nn.Parameter(torch.zeros(self.num_memory_tokens, config.mm_hidden_size))
        self.memory_tokens = nn.Parameter(torch.randn(self.num_memory_tokens, config.mm_hidden_size))

    def init_memory(self, batch_size):
        return repeat(self.memory_tokens, 'm d -> b m d', b = batch_size)


    def forward(
            self,
            hidden_states: torch.Tensor,
            read_memories: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            is_image: Optional[bool] = False,
    ):
        # store intermediate results
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None

        # use cache
        next_cache = () if use_cache else None

        # if not is_image:
        # set up memories for encoder: memory + x
        B = hidden_states.shape[0]
        if read_memories is not None:
            if read_memories.ndim == 2:
                read_memories = repeat(read_memories, 'n d -> b n d', b = B)
                read_mem_length = self.num_memory_tokens
                read_memories = read_memories + self.read_memory_emb
        else:
            red_mem_length = self.num_memory_tokens
            read_memories = repeat(self.read_memory_emb, 'n d -> b n d', b = B)
        if attention_mask is not None:
            attention_mask = nn.functional.pad(attention_mask, (read_mem_length, self.num_memory_tokens), value=True)
        # TODO
        assert encoder_attention_mask is None
        hidden_states, ps = pack([read_memories, hidden_states], 'b * d')

        for i, layer in enumerate(self.layers):
            if output_hidden_states: all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_cache += (layer_outputs[-1], )
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # if not is_image:
        proj_hidden_states = self.proj(hidden_states)
        read_memories, split_hidden_states = unpack(hidden_states, ps, 'b * d')
        
        return proj_hidden_states, read_memories, split_hidden_states
        
        # TODO: save intermedia results
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)

class RMTTransformerProjector(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.config = config
        self.projector = TransformerProjector(config, depth)
        
         # absolute positional encoding for segment (TODO: fixed length)
        self.abs_vid_pe = nn.Embedding(1024, config.mm_hidden_size)
        nn.init.constant_(self.abs_vid_pe.weight, 0.0)

        self.h, self.w = 14, 14
        # self.h, self.w = 12, 12
        self.pooler = nn.AdaptiveAvgPool2d((self.h, self.w))

    def forward(
            self,
            hidden_states: torch.Tensor,
            read_memories: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
    ):
        b, t, n, d = hidden_states.shape
        # print(hidden_states.shape)
        # if t > 1:
        cls_states = hidden_states[:, :, 0, :]
        cls_states = rearrange(cls_states, 'b t d -> (b t) d')
        hidden_states = hidden_states[:, :, 1:, :]
        # hidden_states = rearrange(hidden_states, 'b t n d -> (b t) n d')


        # spatial pooling
        b, t, n, d = hidden_states.shape
        w = int(math.sqrt(n * 336 // 336))
        h = int(w * 336 // 336)
        # w = int(math.sqrt(n * 224 // 224))
        # h = int(w * 224 // 224)
        hidden_states = rearrange(hidden_states, 'b t (h w) d -> (b t) d h w', b=b, t=t, h=h, w=w, d=d)
        hidden_states = self.pooler(hidden_states)
        hidden_states = rearrange(hidden_states, '(b t) d h w -> (b t) (h w) d', b=b, t=t, h=self.h, w=self.w, d=d)


        assert b == 1
        
        enc_memories = []
        all_last_hidden_states = []
        
        if t <= 1:
            is_image = True
            # hidden_state = hidden_states
            hidden_state = hidden_states.reshape(1, -1, d)
            
            # position_ids = torch.tensor([0]).long().to(hidden_state.device)
            # position_ids = position_ids.unsqueeze(0).expand(hidden_state.shape[0], -1)
            # hidden_state = hidden_state + self.abs_vid_pe(position_ids)
            
            image_output = self.projector(
                hidden_state,
                read_memories,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values,
                use_cache,
                output_attentions,
                output_hidden_states,
                is_image,
            )
            # last_hidden_states = image_output[0]
            
            last_hidden_states = image_output[1]
            last_hidden_states = self.projector.proj(last_hidden_states)
            
        else:
            # by semantic segment
            is_image = False
            # cls_states = hidden_states[:, 1, :]
            
            # k = cls_states.shape[0] // 8 - 1
            # assert cls_states.shape[0] % 8 == 0
            # boundaries = segment(cls_states, k=k)
            
            # boundaries = segment(cls_states, k=3)
            # boundaries = segment(cls_states, alpha=1.0)
            # boundaries = segment(cls_states, alpha=1.0)
            boundaries = segment(cls_states, alpha=1.0)
            
            print(boundaries)
            
            index = 0
            # hidden_states = hidden_states[:,1:,:]
            for s, bi in enumerate(boundaries):
                # # prevent from exceed max length
                steps = torch.linspace(index, bi, min(8, bi-index+1), dtype=torch.int).to(hidden_states.device)
                hidden_state = torch.index_select(hidden_states, 0, steps)
                # # vanila
                # hidden_state = hidden_states[index: bi+1]
                
                # # abs position
                # position_ids = torch.tensor([s]).long().to(hidden_state.device)
                # position_ids = position_ids.unsqueeze(0).expand(hidden_state.shape[0], -1)
                # hidden_state = hidden_state + self.abs_vid_pe(position_ids)
                
                hidden_state = hidden_state.reshape(1, -1, d)
                index = bi+1
                segment_output = self.projector(
                    hidden_state,
                    read_memories,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    output_hidden_states,
                    is_image,
                )
                flast_hidden_states = segment_output[0]
                # all_last_hidden_states.append(last_hidden_states)
                read_memories = segment_output[1]
                
                enc_memories.append(read_memories.clone())
                
            last_hidden_states = torch.cat(enc_memories, dim=1)
            last_hidden_states = self.projector.proj(last_hidden_states)
            # last_hidden_states = torch.cat([last_hidden_states, flast_hidden_states], dim=1)
            
            print(last_hidden_states.shape)
                    
        
        return last_hidden_states


    
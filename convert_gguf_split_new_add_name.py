import argparse
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import pygguf
import numpy as np
import openvino as ov
import torch
from openvino.runtime import Model, Type
from openvino.runtime import opset15 as opset
from openvino.runtime import serialize
from openvino.runtime.op import Constant
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

OV_XML_FILE_NAME="openvino_model.xml"

# GGML_QUANTIZATION_GROUP_SIZE = 32

def show_model(m):
    print("inputs of the model:")
    for port, _input in enumerate(m.inputs):
        print("	[{}] {}".format(port, _input))
    print("outputs of the model:")
    for port, _output in enumerate(m.outputs):
        print("	[{}] {}".format(port, _output))


def create_causal_mask(attention_mask, keys, hidden_dim, input_shape):
    t130 = opset.shape_of(attention_mask, output_type=Type.i64, name=f"{keys}.t130")
    t131 = opset.constant(1, dtype=np.int64, name=f"{keys}.t131")
    t132 = opset.constant(0, dtype=np.int64, name=f"{keys}.t132")
    t133 = opset.gather(t130, t131, t132, batch_dims=0, name=f"{keys}.t133")
    
    t134 = opset.constant([1], dtype=np.int64, name=f"{keys}.t134")
    t135 = opset.reshape(t133, t134, special_zero=False, name=f"{keys}.t135")
    t40 = input_shape
    t127 = opset.gather(t40, opset.constant(1, dtype=np.int64, name=f"{keys}.const1"), axis=0, name=f"{keys}.t127")
    t129 = opset.reshape(t127, opset.constant([1], dtype=np.int64, name=f"{keys}.const1_reshape"), special_zero=False, name=f"{keys}.t129")
    t136 = opset.concat([t129, t135], axis=0, name=f"{keys}.t136")
    min_shape_val = opset.constant([1, 1], dtype=np.int64, name=f"{keys}.min_shape_val")
    t137 = opset.maximum(min_shape_val, t136, auto_broadcast="numpy", name=f"{keys}.t137")
    t138 = opset.broadcast(opset.constant(-65504, dtype=np.float32, name=f"{keys}.const_mask_val"), t137, broadcast_spec="NUMPY", name=f"{keys}.t138")
    
    t139 = opset.shape_of(t138, output_type=Type.i32, name=f"{keys}.t139")
    t140 = opset.constant(1, dtype=np.int64, name=f"{keys}.t140")
    t141 = opset.constant(0, dtype=np.int64, name=f"{keys}.t141")
    t142 = opset.gather(t139, t140, t141, batch_dims=0, name=f"{keys}.t142")
    t143 = opset.constant(1, dtype=np.int32, name=f"{keys}.t143")
    
    zero_const = opset.constant(0, dtype=np.int32, name=f"{keys}.zero_const")
    t144 = opset.range(zero_const, t142, t143, output_type=Type.i32, name=f"{keys}.t144")
    t145 = opset.unsqueeze(t144, axes=zero_const, name=f"{keys}.t145")
    t146 = opset.constant(1, dtype=np.int32, name=f"{keys}.t146")
    t147 = opset.constant(0, dtype=np.int64, name=f"{keys}.t147")
    t148 = opset.constant(0, dtype=np.int64, name=f"{keys}.t148")
    
    t149 = opset.gather(t139, t147, t148, batch_dims=0, name=f"{keys}.t149")
    t150 = opset.add(t149, t146, auto_broadcast="numpy", name=f"{keys}.t150")
    t151 = opset.range(t146, t150, t143, output_type=Type.i32, name=f"{keys}.t151")
    t152 = opset.unsqueeze(t151, axes=t143, name=f"{keys}.t152")
    t153 = opset.greater_equal(t145, t152, auto_broadcast="numpy", name=f"{keys}.t153")
    
    t154 = opset.constant(0.0, dtype=np.float32, name=f"{keys}.t154")
    t155 = opset.select(t153, t138, t154, auto_broadcast="numpy", name=f"{keys}.t155")
    
    t156 = opset.constant(0, dtype=np.int32, name=f"{keys}.t156")
    t157 = opset.constant(1, dtype=np.int32, name=f"{keys}.t157")
    t158 = opset.range(t156, t133, t157, output_type=Type.f32, name=f"{keys}.t158")
    t159 = opset.convert(t158, destination_type=np.int64, name=f"{keys}.t159")
    t160 = opset.convert(t159, destination_type=np.float32, name=f"{keys}.t160")
    t161 = opset.shape_of(keys, output_type=Type.i64, name=f"{keys}.t161")
    t162 = opset.constant(2, dtype=np.int64, name=f"{keys}.t162")
    t163 = opset.constant(0, dtype=np.int64, name=f"{keys}.t163")
    t164 = opset.gather(t161, t162, t163, batch_dims=0, name=f"{keys}.t164")
    t165 = opset.add(t164, t127, auto_broadcast='numpy', name=f"{keys}.t165")
    t166 = opset.constant(1, dtype=np.int32, name=f"{keys}.t166")
    t167 = opset.range(t164, t165, t166, output_type=Type.f32, name=f"{keys}.t167")
    t168 = opset.constant([-1, 1], dtype=np.int64, name=f"{keys}.t168")
    t169 = opset.reshape(t167, t168, special_zero=False, name=f"{keys}.t169")
    t170 = opset.greater(t160, t169, auto_broadcast='numpy', name=f"{keys}.t170")
    t171 = opset.convert(t170, destination_type=np.float32, name=f"{keys}.t171")
    
    t172 = opset.multiply(t155, t171, auto_broadcast='numpy', name=f"{keys}.t172")
    t173 = opset.constant(0, dtype=np.int64, name=f"{keys}.t173")
    t174 = opset.unsqueeze(t172, t173, name=f"{keys}.t174")
    t48 = opset.constant(1, dtype=np.int64, name=f"{keys}.t48")
    t175 = opset.unsqueeze(t174, t48, name=f"{keys}.t175")
    t41 = opset.constant([0], dtype=np.int64, name=f"{keys}.t41")
    t42 = opset.constant(0, dtype=np.int64, name=f"{keys}.t42")
    t43 = opset.gather(t40, t41, t42, batch_dims=0, name=f"{keys}.t43")
    t176 = opset.constant([1], dtype=np.int64, name=f"{keys}.t176")
    t177 = opset.constant([1], dtype=np.int64, name=f"{keys}.t177")
    t178 = opset.constant([1], dtype=np.int64, name=f"{keys}.t178")
    t179 = opset.concat([t43, t176, t177, t178], axis=0, name=f"{keys}.t179")
    t180 = opset.broadcast(t175, t179, broadcast_spec="bidirectional", name=f"{keys}.t180")
    t181 = opset.constant([-1], dtype=np.int64, name=f"{keys}.t181")
    t182 = opset.reshape(t180, t181, special_zero=False, name=f"{keys}.t182")
    t183 = opset.constant(0, dtype=np.int64, name=f"{keys}.t183")
    t184 = opset.shape_of(t180, output_type=Type.i64, name=f"{keys}.t184")
    t185 = opset.reduce_prod(t184, t183, keep_dims=False, name=f"{keys}.t185")
    t186 = opset.constant(1, dtype=np.int64, name=f"{keys}.t186")
    t187 = opset.range(t183, t185, t186, output_type=Type.i64, name=f"{keys}.t187")
    t188 = opset.reshape(t187, t184, special_zero=False, name=f"{keys}.t188")
    t189 = opset.constant([0], dtype=np.int64, name=f"{keys}.t189")
    t190 = opset.constant([1], dtype=np.int64, name=f"{keys}.t190")
    t191 = opset.slice(t188, t189, t135, t190, hidden_dim, name=f"{keys}.t191")
    t192 = opset.constant([-1, 1], dtype=np.int64, name=f"{keys}.t192")
    t193 = opset.reshape(t191, t192, special_zero=False, name=f"{keys}.t193")
    t194 = opset.constant([0], dtype=np.int64, name=f"{keys}.t194")
    t195 = opset.constant([1], dtype=np.int64, name=f"{keys}.t195")
    t196 = opset.slice(t180, t194, t135, t195, hidden_dim, name=f"{keys}.t196")

    t197 = opset.unsqueeze(attention_mask, t48, name=f"{keys}.t197")
    t198 = opset.constant(2, dtype=np.int64, name=f"{keys}.t198")
    t199 = opset.unsqueeze(t197, t198, name=f"{keys}.t199")
    t200 = opset.convert(t199, destination_type=np.float32, name=f"{keys}.t200")
    t201 = opset.add(t196, t200, auto_broadcast='numpy', name=f"{keys}.t201")
    t202 = opset.constant([[[[0.0]]]], dtype=np.float32, name=f"{keys}.t202")
    t203 = opset.equal(t201, t202, auto_broadcast='numpy', name=f"{keys}.t203")
    t204 = opset.constant(-65504.0, dtype=np.float32, name=f"{keys}.t204")
    t205 = opset.select(t203, t204, t196, auto_broadcast='numpy', name=f"{keys}.t205")
    t206 = opset.shape_of(t196, output_type=Type.i64, name=f"{keys}.t206")
    t207 = opset.broadcast(t205, t206, broadcast_spec="NUMPY", name=f"{keys}.t207")
    t208 = opset.constant([-1], dtype=np.int64, name=f"{keys}.t208")
    t209 = opset.reshape(t207, t208, special_zero=False, name=f"{keys}.t209")
    t210 = opset.scatter_nd_update(t182, t193, t209, name=f"{keys}.t210")
    t211 = opset.reshape(t210, t184, special_zero=False, name=f"{keys}.t211")
    t212 = opset.constant([0], dtype=np.int64, name=f"{keys}.t212")
    t213 = opset.constant([1], dtype=np.int64, name=f"{keys}.t213")
    t214 = opset.reshape(t164, t213, special_zero=False, name=f"{keys}.t214")
    t215 = opset.add(t214, t129, auto_broadcast='numpy', name=f"{keys}.t215")
    t216 = opset.constant([1], dtype=np.int64, name=f"{keys}.t216")
    t217 = opset.slice(t211, t212, t215, t216, hidden_dim, name=f"{keys}.t217")
    return t217


def rotate_half(x, head_size, axis):
    """Rotates half the hidden dimensions of the input tensor."""
    t58 = opset.constant([head_size // 2], dtype=np.int64)
    t59 = opset.constant([9223372036854775807], dtype=np.int64)
    t60 = opset.constant([1], dtype=np.int64)
    t62 = opset.slice(x, t58, t59, t60, axis)
    t63 = opset.constant([[[[-1.0]]]], dtype=np.float32)
    t64 = opset.multiply(t62, t63)
    t65 = opset.constant([0], dtype=np.int64)
    t66 = opset.constant([head_size // 2], dtype=np.int64)
    t67 = opset.constant([1], dtype=np.int64)
    t68 = opset.slice(x, t65, t66, t67, axis)
    rotated = opset.concat([t64, t68], axis=-1)
    return rotated


def apply_rotary_pos_emb(q, k, cos, sin, head_size, hidden_dim, cos_sin_cached, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors using OpenVINO API.

    Args:
        q: Query tensor (OpenVINO node).
        k: Key tensor (OpenVINO node).
        cos: Cosine part of rotary embedding (OpenVINO node).
        sin: Sine part of rotary embedding (OpenVINO node).
        unsqueeze_dim: Dimension to unsqueeze cos and sin for broadcasting.
    Returns:
        Tuple of (q_rotated, k_rotated) tensors.
    """
    # Unsqueeze cos and sin along the specified dimension
    if cos_sin_cached is None:
        cos_unsqueezed = opset.unsqueeze(cos, np.int64(unsqueeze_dim))
        sin_unsqueezed = opset.unsqueeze(sin, np.int64(unsqueeze_dim))
    else:
        cos_unsqueezed = cos_sin_cached[0]
        sin_unsqueezed = cos_sin_cached[1]

    # Apply Rotary Positional Embedding
    q_rotated = opset.add(opset.multiply(q, cos_unsqueezed), opset.multiply(rotate_half(q, head_size, hidden_dim), sin_unsqueezed))
    k_rotated = opset.add(opset.multiply(k, cos_unsqueezed), opset.multiply(rotate_half(k, head_size, hidden_dim), sin_unsqueezed))

    return q_rotated, k_rotated, (cos_unsqueezed, sin_unsqueezed)


def rope_emb(x, rope_const, position_ids, batch_dim):
    """
    Generates Rotary Position Embedding (RoPE) cosine and sine components using OpenVINO.

    Args:
        x: The input tensor to determine the device and dtype (OpenVINO node).
        rope_const: Tensor containing the rotary embedding constants (OpenVINO node).
        position_ids: Tensor with position IDs (OpenVINO node).
        batch_dim: batch dimension

    Returns:
        cos: Cosine component of the rotary embedding.
        sin: Sine component of the rotary embedding.
    """
    # Expand dimensions for broadcasting
    position_ids_expanded = opset.convert(
        opset.unsqueeze(position_ids, 1, name="rope.position_ids_unsqueeze"),
        Type.f32,
        name="rope.position_ids_f32"
    )  # Add head_dim axis: [batch_size, 1, seq_len]

    # Broadcast RoPE to batch dimension
    inv_freq_expanded = opset.broadcast(
        rope_const,
        target_shape=opset.concat([
            batch_dim,
            np.int64([1]),
            np.int64([1])
        ], axis=0, name="rope.concat_broadcast_shape"),
        broadcast_spec="BIDIRECTIONAL",
        name="rope.inv_freq_broadcast"
    )

    # Compute frequencies
    freqs = opset.matmul(
        inv_freq_expanded,
        position_ids_expanded,
        transpose_a=False,
        transpose_b=False,
        name="rope.freqs_matmul"
    )  # Shape: [batch_size, seq_len, head_dim]

    freqs_transposed = opset.transpose(
        freqs,
        np.int32([0, 2, 1]),
        name="rope.freqs_transpose"
    )  # Transpose to shape: [batch_size, head_dim, seq_len]

    # Concatenate frequencies along the last dimension
    emb = opset.concat(
        [freqs_transposed, freqs_transposed],
        axis=-1,
        name="rope.emb_concat"
    )  # Shape: [batch_size, head_dim, seq_len * 2]

    # Compute cosine and sine
    cos = opset.cos(emb, name="rope.emb_cos")
    sin = opset.sin(emb, name="rope.emb_sin")

    return cos, sin



def multi_head_attention(query, key, value,
                        configs,
                        batch_dim,
                        layer_idx,
                        hidden_dim,
                        input_shape,
                        output_shape,
                        attention_mask,
                        mask=None,
                        position_ids=None,
                        rope_const=None,
                        beam_idx=None,
                        cos_sin_cached=None):
    """
    Implements multi-head self-attention using OpenVINO operations
    
    Args:
        query: Query tensor of shape [batch_size, seq_len, hidden_dim]
        key: Key tensor of shape [batch_size, seq_len, hidden_dim]
        value: Value tensor of shape [batch_size, seq_len, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        mask: Optional attention mask
        sin: Optional sine component for rotary embeddings
        cos: Optional cosine component for rotary embeddings
        opset: OpenVINO operation set to use
    """
    num_heads = configs["head_num"]
    head_dim = configs["head_size"]
    num_heads_kv = configs["head_num_kv"]
    
    # 1. Reshape Q, K, V to split heads
    def split_heads(x, num_heads, head_dim):
        # Reshape from [batch, seq_len, hidden_dim] to [batch, seq_len, num_heads, head_dim]
        x = opset.reshape(x, [0, 0, num_heads, head_dim], special_zero=True)
        # Transpose to [batch, num_heads, seq_len, head_dim]
        x = opset.transpose(x, [0, 2, 1, 3])
        return x
    
    q = split_heads(query, num_heads, head_dim)
    k = split_heads(key, num_heads_kv, head_dim)
    v = split_heads(value, num_heads_kv, head_dim)

    cos = None
    sin = None
    if cos_sin_cached is None:
        cos, sin = rope_emb(v, rope_const, position_ids, batch_dim)
    
    # 2. Apply rotary embeddings
    q, k, cos_sin_cached = apply_rotary_pos_emb(q, k, cos, sin, head_dim, hidden_dim, cos_sin_cached)

    default_key = opset.broadcast(opset.constant(0.0, dtype=np.float32),
                                opset.concat([batch_dim,
                                            np.int64([num_heads_kv]),
                                            np.int64([0]),
                                            np.int64([head_dim])],
                                            axis=0))
    k_cache_val = opset.read_value(default_key, 
                                   variable_shape=ov.PartialShape([-1,num_heads_kv,-1,head_dim]),
                                   variable_type=Type.f32,
                                   variable_id=f"past_key_values.{layer_idx}.keypresent.{layer_idx}.key")
    k_cache = opset.gather(k_cache_val, beam_idx, opset.constant(0, dtype=np.int64), batch_dims=0)
    default_value = opset.broadcast(opset.constant(0.0, dtype=np.float32),
                                opset.concat([batch_dim,
                                            np.int64([num_heads_kv]),
                                            np.int64([0]),
                                            np.int64([head_dim])],
                                            axis=0))
    v_cache_val = opset.read_value(default_value, 
                                   variable_shape=ov.PartialShape([-1,num_heads_kv,-1,head_dim]),
                                   variable_type=Type.f32,
                                   variable_id=f"past_key_values.{layer_idx}.valuepresent.{layer_idx}.key")
    v_cache = opset.gather(v_cache_val, beam_idx, opset.constant(0, dtype=np.int64), batch_dims=0)

    k_combined = opset.concat([k_cache, k], axis=2)
    v_combined = opset.concat([v_cache, v], axis=2)

    k_assigned = opset.assign(k_combined, f"past_key_values.{layer_idx}.keypresent.{layer_idx}.key")
    v_assigned = opset.assign(v_combined, f"past_key_values.{layer_idx}.valuepresent.{layer_idx}.key")

    if num_heads == num_heads_kv:
        k_reshaped = k_combined
        v_reshaped = v_combined
    else: # Group Query Attention branch
        kv_per_head = num_heads // num_heads_kv
        k_combined_unsq = opset.unsqueeze(k_combined, opset.constant(2, dtype=np.int64))
        k_combined_broad = opset.broadcast(k_combined_unsq,
                                           opset.concat([batch_dim,
                                                np.int64([num_heads_kv]),
                                                np.int64([kv_per_head]),
                                                np.int64([0]),
                                                np.int64([head_dim])],
                                                axis=0),
                                            broadcast_spec="BIDIRECTIONAL")
        k_reshaped = opset.reshape(k_combined_broad, [0, num_heads, -1, head_dim], special_zero=True)

        v_combined_unsq = opset.unsqueeze(v_combined, opset.constant(2, dtype=np.int64))
        v_combined_broad = opset.broadcast(v_combined_unsq,
                                           opset.concat([batch_dim,
                                                np.int64([num_heads_kv]),
                                                np.int64([kv_per_head]),
                                                np.int64([0]),
                                                np.int64([head_dim])],
                                                axis=0),
                                            broadcast_spec="BIDIRECTIONAL")
        v_reshaped = opset.reshape(v_combined_broad, [0, num_heads, -1, head_dim], special_zero=True)

    if mask is None:
        mask = create_causal_mask(attention_mask, k_cache, hidden_dim, input_shape)

    # 3. Calculate attention
    #scale = opset.constant(np.float32(1.0 / np.sqrt(head_dim)))
    
    attention_output = opset.scaled_dot_product_attention(q, k_reshaped, v_reshaped, mask)
    
    # 4. Reshape output
    # Transpose back to [batch, seq_len, num_heads, head_dim]
    context_transposed = opset.transpose(attention_output, [0, 2, 1, 3])
    
    # Combine heads: [batch, seq_len, hidden_dim]
    output = opset.reshape(context_transposed,
                          output_shape,
                          special_zero=False)
    
    return output, [k_assigned, v_assigned], cos_sin_cached, mask


# Reorder weight rows from 0,2,1,3 -> 0,1,2,3 for each head
def reorder_interleaved_format(weights, head_size):
    shape = weights.shape
    num_heads = shape[0] // head_size
    weights = weights.reshape((num_heads, head_size, -1))
    weights = np.moveaxis(weights, 0, 1)
    new_weight = np.empty_like(weights)
    new_weight[0:head_size // 2] = weights[0::2]
    new_weight[(head_size//2):head_size] = weights[1::2]
    new_weight = np.moveaxis(new_weight, 0, 1)
    return new_weight.reshape(shape)


def make_fp16_weights(key, consts, reorder, head_size):
    weight = consts[f"{key}.weight"]
    if reorder:
        weight = reorder_interleaved_format(weight, head_size)

    weights = opset.constant(weight, dtype=np.float16)
    weights.set_friendly_name(name=f"{key}.weight")
    w_f32 = opset.convert(weights, Type.f32)
    return w_f32

def make_int8_weights(key, consts, reorder, head_size, group_size):#weight = ov.Tensor(weight, weight.shape, const_dtype)
    weight = consts[f"{key}.weight"]
    # weight = weight.view(np.uint8)
    orig_weight_shape = list(weight.shape)
    weight = weight.reshape(orig_weight_shape[0], -1, group_size)
    scale = np.expand_dims(consts[f"{key}.scales"], axis=2)
    bias = np.expand_dims(consts[f"{key}.biases"], axis=2)
    if reorder:
        weight = reorder_interleaved_format(weight, head_size)
        scale = reorder_interleaved_format(scale, head_size)
        bias = reorder_interleaved_format(bias, head_size)

    weights = opset.constant(weight, dtype=np.uint8)
    weights.set_friendly_name(name=f"{key}.weight")
    weights_f16 = opset.convert(weights, Type.f16)

    zero_point =  np.round(-bias / scale).astype(np.uint8)
    zero_points = opset.constant(zero_point, dtype=np.uint8)
    zero_points_f16 = opset.convert(zero_points, Type.f16)

    scales = opset.constant(scale, dtype=np.float16)

    w_zp = opset.subtract(weights_f16, zero_points_f16, auto_broadcast="numpy")
    w_zp_s = opset.multiply(w_zp, scales, auto_broadcast="numpy")

    # w_zp_s_r = opset.reshape(w_zp_s, opset.constant(orig_weight_shape, dtype=np.int64), special_zero=False)
    w_zp_s_r = opset.reshape(w_zp_s, opset.constant(orig_weight_shape, dtype=np.int32), special_zero=False)
    w_zp_s_f32 = opset.convert(w_zp_s_r, Type.f32)
    return w_zp_s_f32


def make_int4_weights(key, consts, reorder, head_size, group_size):
    weight = consts[f"{key}.weight"]
    # weight = weight.view(np.uint8)
    orig_weight_shape = list(weight.shape)
    orig_weight_shape[1] = orig_weight_shape[1] * 2  # double number of columns as it is 4-bit tensor

    weight = weight.reshape(orig_weight_shape[0], -1, group_size // 2)
    scale = np.expand_dims(consts[f"{key}.scales"], axis=2)
    bias = np.expand_dims(consts[f"{key}.biases"], axis=2)

    if reorder:
        weight = reorder_interleaved_format(weight, head_size)
        scale = reorder_interleaved_format(scale, head_size)
        bias = reorder_interleaved_format(bias, head_size)

    shape = (orig_weight_shape[0], orig_weight_shape[1] // group_size, group_size)
    weight_tensor = ov.Tensor(weight.reshape(-1), shape, Type.u4)
    weights = opset.constant(weight_tensor, name=f"{key}.weight_const", shared_memory=False)  # Don't use shared_memory=True
    weights_f16 = opset.convert(weights, Type.f16, name=f"{key}.weights_to_f16")

    zero_point = np.round(-bias / scale).astype(np.uint8)
    zero_point_shape = list(zero_point.shape)
    zero_point = zero_point.reshape(-1)
    # Pack zero points: two subsequent values into one
    mask = np.array(0b00001111, dtype=np.uint8)
    zero_point_packed = (zero_point[1::2] << 4) | (zero_point[0::2] & mask)
    zero_point_tensor = ov.Tensor(zero_point_packed, tuple(zero_point_shape), Type.u4)
    zero_points = opset.constant(zero_point_tensor, name=f"{key}.zp_const", shared_memory=False)  # Don't use shared_memory=True
    zero_points_f16 = opset.convert(zero_points, Type.f16, name=f"{key}.zp_to_f16")

    scales = opset.constant(scale, dtype=np.float16, name=f"{key}.scales_const", shared_memory=False)

    w_zp = opset.subtract(weights_f16, zero_points_f16, auto_broadcast="numpy", name=f"{key}.wzp_sub")
    w_zp_s = opset.multiply(w_zp, scales, auto_broadcast="numpy", name=f"{key}.wzp_scale_mul")
    w_zp_s_r = opset.reshape(
        w_zp_s,
        opset.constant(orig_weight_shape, dtype=np.int64, name=f"{key}.orig_shape_const"),
        special_zero=False,
        name=f"{key}.wzp_reshaped"
    )
    w_zp_s_f32 = opset.convert(w_zp_s_r, Type.f32, name=f"{key}.wzp_to_f32")
    return w_zp_s_f32



def make_weights_subgraph(key, consts, qtype, reorder, head_size):
    if "FP16" == qtype:
        final_node = make_fp16_weights(key, consts, reorder, head_size)
    elif "Q8_0" == qtype:
        final_node = make_int8_weights(key, consts, reorder, head_size, 32)
    elif "Q4_0" == qtype:
        final_node = make_int4_weights(key, consts, reorder, head_size, 32)
    elif "Q4_K" == qtype:
        final_node = make_int4_weights(key, consts, reorder, head_size, 32)
    elif "Q4_K_int8" == qtype:
        final_node = make_int8_weights(key, consts, reorder, head_size, 32)
    elif "Q6_K" == qtype:
        final_node = make_int8_weights(key, consts, reorder, head_size, 16)
    else:
        raise ValueError("Unsupported quantization type: ", qtype)
    
    return final_node


def make_fc(key, input, consts, qtype, reorder=False, head_size=-1):
    # weight const f32 NxK
    w_f32 = make_weights_subgraph(key, consts, qtype, reorder, head_size)
    matmul = opset.matmul(
        input, w_f32,
        transpose_a=False, transpose_b=True,
        name=f"{key}.make_fc.matmul"
    )

    if consts[f"{key}.bias"] is not None:
        bias = opset.constant(
            consts[f"{key}.bias"], Type.f32,
            name=f"{key}.make_fc.bias"
        )
        matmul = opset.add(
            matmul, bias,
            auto_broadcast="numpy",
            name=f"{key}.make_fc.bias_add"
        )

    return matmul



def make_lm_head(key, input, consts, embeddings_node, qtype):
    if consts.get(f"{key}.weight", None) is not None:
        if consts.get(f"{key}.scales", None) is not None:
            lm_head_qtype = qtype
        else:
            lm_head_qtype = "FP16"
        w_f32 = make_weights_subgraph(key, consts, lm_head_qtype, False, -1)
    else:
        w_f32 = embeddings_node # shared weights with embeddings
    return opset.matmul(input, w_f32, transpose_a=False, transpose_b=True)


def make_mvn(key, input, consts, configs, name_suffix=""):
    mvn = opset.mvn(input, axes=[-1], normalize_variance=True, eps=configs["layer_norm_eps"], eps_mode="inside_sqrt", name=f"{key}.mvn{name_suffix}")
    if consts[f"{key}.weight"] is not None:
        weights = opset.constant(consts[f"{key}.weight"], Type.f16, name=f"{key}.weight{name_suffix}")
        mvn = opset.multiply(mvn, weights, auto_broadcast="numpy", name=f"{key}.mul{name_suffix}")
    if consts[f"{key}.bias"] is not None:
        bias = opset.constant(consts[f"{key}.bias"], Type.f16, name=f"{key}.bias{name_suffix}")
        mvn = opset.add(mvn, bias, auto_broadcast="numpy", name=f"{key}.add{name_suffix}")
    return mvn


def make_rms_norm(key, input, consts, epsilon):
    epsilon_c = opset.constant(np.float16([[[epsilon]]]), name=f"{key}.epsilon")
    pow = opset.power(input, opset.convert(np.array([[[2]]], np.float16), Type.f32), name=f"{key}.pow")
    variance = opset.reduce_mean(pow, opset.constant([-1], dtype=np.int64, name=f"{key}.reduce_axis"), keep_dims=True, name=f"{key}.variance")
    add = opset.add(variance, opset.convert(epsilon_c, Type.f32), name=f"{key}.add_epsilon")
    sqrt = opset.sqrt(add, name=f"{key}.sqrt")
    reciprocal = opset.divide(
        opset.convert(opset.constant([[[1]]], 
                                     dtype=np.float16, 
                                     name=f"{key}.reciprocal.const"), 
                      Type.f32, 
                      name=f"{key}.reciprocal.convert"),
        sqrt,
        name=f"{key}.reciprocal.divide"
    )
    mul = opset.multiply(reciprocal, input, auto_broadcast="numpy", name=f"{key}.normalize")

    if not np.all(consts[f"{key}.weight"] == 1.0):
        weights = opset.convert(
            opset.constant(consts[f"{key}.weight"].reshape((1, 1, -1)), np.float16, name=f"{key}.weight_raw"),
            Type.f32,
            name=f"{key}.weight"
        )
        mul = opset.multiply(mul, weights, auto_broadcast="numpy", name=f"{key}.scale")

    return mul



def make_embedding(key, input, consts, qtype):
    if consts.get(f"{key}.scales", None) is not None:
        embedding_type = qtype
    else:
        embedding_type = "FP16"
    embed_f32 = make_weights_subgraph(key, consts, embedding_type, False, -1)
    input_int32 = opset.convert(input, Type.i32, name=f"{key}.input_int32")
    inputs_embeds = opset.gather(embed_f32, indices=input_int32, axis=0, name=f"{key}.inputs_embeds")
    return inputs_embeds, embed_f32


def save_tokenzier(orig_model_path, ov_model_path):
    tokenizer = AutoTokenizer.from_pretrained(orig_model_path)
    tokenizer.save_pretrained(ov_model_path)

    from openvino_tokenizers import convert_tokenizer
    OV_TOKENIZER_NAME = "openvino_tokenizer.xml"
    OV_DETOKENIZER_NAME = "openvino_detokenizer.xml"

    converted = convert_tokenizer(tokenizer, with_detokenizer=True)
    for model, file_name in zip(converted, (OV_TOKENIZER_NAME, OV_DETOKENIZER_NAME)):
        ov.save_model(model, Path(ov_model_path) / file_name)


def layer(configs, consts, layer_idx, hidden_states, attn_mask, causal_mask, position_ids, rope_const, beam_idx, batch_dim, hidden_dim, cos_sin_cached, output_shape):
    name_suffix = f".layer{layer_idx}"
    name_prefix = "model.layers.self_attn"
    # layerNorm operation
    input_layernorm = make_rms_norm(f"model.layers.{layer_idx}.input_layernorm", hidden_states, consts["layers"][layer_idx], configs["rms_norm_eps"])

    reorder = False
    if configs["architecture"] == "llama":
        reorder = True
    q = make_fc(f"model.layers.{layer_idx}.self_attn.q_proj", input_layernorm, consts["layers"][layer_idx], config[f"blk.{layer_idx}.attn_q.weight_qtype"], reorder, configs["head_size"])
    k = make_fc(f"model.layers.{layer_idx}.self_attn.k_proj", input_layernorm, consts["layers"][layer_idx], config[f"blk.{layer_idx}.attn_k.weight_qtype"], reorder, configs["head_size"])
    v = make_fc(f"model.layers.{layer_idx}.self_attn.v_proj", input_layernorm, consts["layers"][layer_idx], config[f"blk.{layer_idx}.attn_v.weight_qtype"])

    input_shape = opset.shape_of(input_layernorm)
    if output_shape is None:
        output_shape = opset.concat([opset.gather(input_shape,
                                                 opset.constant([0, 1], dtype=np.int64),
                                                 axis=0,
                                                 batch_dims=0,
                                                 name = f"model.layers.{layer_idx}.output_shape.gather"),
                                    opset.constant([-1], dtype=np.int64, name = f"model.layers.{layer_idx}.output_shape.constant")],
                                    axis=0,
                                    name = f"model.layers.{layer_idx}.output_shape.concat")


    attn_output, sinks, cos_sin_cached, causal_mask = multi_head_attention(q, k, v,
                        configs,
                        batch_dim=batch_dim,
                        layer_idx=layer_idx,
                        hidden_dim=hidden_dim,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        attention_mask=attn_mask,
                        mask=causal_mask,
                        position_ids=position_ids,
                        rope_const=rope_const,
                        beam_idx=beam_idx,
                        cos_sin_cached=cos_sin_cached)

    attn_output = make_fc(f"model.layers.{layer_idx}.self_attn.o_proj", attn_output, consts["layers"][layer_idx], config[f"blk.{layer_idx}.attn_output.weight_qtype"])

    attn_output = opset.add(hidden_states, attn_output, auto_broadcast="numpy", name=f"{name_prefix}.add0{name_suffix}")
    post_attention_layernorm = make_rms_norm(f"model.layers.{layer_idx}.post_attention_layernorm", attn_output, consts["layers"][layer_idx], configs["rms_norm_eps"])

    # mlp
    def mlp(states):

        gate_proj = make_fc(f"model.layers.{layer_idx}.mlp.gate_proj", states, consts["layers"][layer_idx], config[f"blk.{layer_idx}.ffn_gate.weight_qtype"])
        silu = opset.swish(gate_proj)
        up_proj = make_fc(f"model.layers.{layer_idx}.mlp.up_proj", states, consts["layers"][layer_idx], config[f"blk.{layer_idx}.ffn_up.weight_qtype"])
        mul = opset.multiply(silu, up_proj, auto_broadcast="numpy", name=f"{name_prefix}.mlp.mul{name_suffix}")
        down_proj = make_fc(f"model.layers.{layer_idx}.mlp.down_proj", mul, consts["layers"][layer_idx], config[f"blk.{layer_idx}.ffn_down.weight_qtype"])
        return down_proj

    mlp_output = mlp(post_attention_layernorm)
    # residual connection.
    output = opset.add(attn_output, mlp_output, auto_broadcast="numpy", name=f"{name_prefix}.add1{name_suffix}")
    return output, sinks, causal_mask, cos_sin_cached, output_shape


def init_rope(head_dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim))
    inv_freq = inv_freq[None, :, None]
    # For BC we register cos and sin cached
    rope_const = opset.constant(inv_freq.numpy(), Type.f32, name="rope.inv_freq")
    return rope_const

def init_rope_numpy(head_dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
    # 等价于 torch.arange(0, head_dim, 2, dtype=torch.int64)
    arange = np.arange(0, head_dim, 2, dtype=np.float32)
    
    # 计算 inv_freq
    inv_freq = 1.0 / (base ** (arange / head_dim))  # shape: (head_dim // 2,)
    
    # 添加维度，使其形状与 PyTorch 的 inv_freq[None, :, None] 相同，即 (1, head_dim // 2, 1)
    inv_freq = inv_freq[None, :, None]

    # 假设你仍然使用 ONNX 的 OpenVINO opset，这里保持不变
    rope_const = opset.constant(inv_freq, Type.f32)

    return rope_const

def create_model(configs, consts):
    print("start generate ov model...")
    beg = time.time()
    # [batch, query_len]
    input_ids = opset.parameter([-1, -1], Type.i64, name="input_ids")
    # [batch, query_len+past_len]
    attention_mask = opset.parameter([-1, -1], Type.i64, name="attention_mask")
    # [batch, query_len+past_len]
    position_ids = opset.parameter([-1, -1], Type.i64, name="position_ids")
    # [batch, max_kv_len]
    beam_idx = opset.parameter([-1], Type.i32, name="beam_idx")

    inputs_embeds, embeddings = make_embedding("model.embed_tokens", input_ids, consts, configs["model.embed_tokens.weight"+"_qtype"])
    hidden_states = inputs_embeds
    
    rope_const = init_rope(configs["head_size"], configs["max_position_embeddings"], configs["rope_freq_base"])

    input_shape = opset.shape_of(input_ids, name = "shape_of.input_shape")
    batch_size = opset.gather(input_shape,
                         opset.constant([0], dtype=np.int64, name = "constant0.batch_size"),
                         opset.constant([0], dtype=np.int64, name = "constant1.batch_size"),
                         name = "gather.batch_size")
    hidden_dim = opset.constant([3], dtype=np.int64, name = "constant.hidden_dim")

    sinks = []
    # Shared tensors across all the Transformer blocks
    causal_mask = None
    cos_sin_cached = None
    output_shape = None
    for i in tqdm(range(configs["layer_num"])):
        hidden_states, layer_sinks, causal_mask, cos_sin_cached, output_shape = layer(configs, consts, i, hidden_states, attention_mask, causal_mask, position_ids, rope_const, beam_idx, batch_size, hidden_dim, cos_sin_cached, output_shape)
        sinks = sinks + layer_sinks
    # final_layernorm
    final_layernorm = make_rms_norm("model.norm", hidden_states, consts, configs["rms_norm_eps"])
    # embed_out
    embed_out = make_lm_head("lm_head", final_layernorm, consts, embeddings, config["lm_head.weight"+"_qtype"])#TODO
    logits = opset.result(embed_out, name="logits")
    logits.set_friendly_name("logits")
    cost = time.time() - beg
    print(f"generate ov model done, cost {cost:.2f} seconds.")
    model = Model([logits], sinks,
                 [input_ids, attention_mask, position_ids, beam_idx])
    model.outputs[0].get_tensor().set_names({"logits"})

    # set runtime options
    model.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
    model.set_rt_info("8.0", ["runtime_options", "ACTIVATIONS_SCALE_FACTOR"])

    return model

def create_partial_model_before_mha(configs, consts, layer_idx=0):
    print("Generating partial model with qkv make_fc...")

    # Inputs
    input_ids = opset.parameter([-1, -1], Type.i64, name="input_ids")
    attention_mask = opset.parameter([-1, -1], Type.i64, name="attention_mask")
    position_ids = opset.parameter([-1, -1], Type.i64, name="position_ids")
    beam_idx = opset.parameter([-1], Type.i32, name="beam_idx")

    # Step 1: Embedding
    inputs_embeds, embeddings = make_embedding(
        "model.embed_tokens",
        input_ids,
        consts,
        configs["model.embed_tokens.weight" + "_qtype"]
    )
    hidden_states = inputs_embeds

    # Step 2: First Layer Norm (before attention)
    input_layernorm = make_rms_norm(
        f"model.layers.{layer_idx}.input_layernorm",
        hidden_states,
        consts["layers"][layer_idx],
        configs["rms_norm_eps"]
    )

    # Step 3: Q, K, V projections
    reorder = configs.get("architecture", "") == "llama"

    q_proj = make_fc(
        f"model.layers.{layer_idx}.self_attn.q_proj",
        input_layernorm,
        consts["layers"][layer_idx],
        configs[f"blk.{layer_idx}.attn_q.weight_qtype"],
        reorder,
        configs["head_size"]
    )
    # k_proj = make_fc(
    #     f"model.layers.{layer_idx}.self_attn.k_proj",
    #     input_layernorm,
    #     consts["layers"][layer_idx],
    #     configs[f"blk.{layer_idx}.attn_k.weight_qtype"],
    #     reorder,
    #     configs["head_size"]
    # )
    # v_proj = make_fc(
    #     f"model.layers.{layer_idx}.self_attn.v_proj",
    #     input_layernorm,
    #     consts["layers"][layer_idx],
    #     configs[f"blk.{layer_idx}.attn_v.weight_qtype"]
    # )

    # Output results
    result_q = opset.result(q_proj, name="q_proj")
    result_q.set_friendly_name("q_proj")

    # result_k = opset.result(k_proj, name="k_proj")
    # result_k.set_friendly_name("k_proj")

    # result_v = opset.result(v_proj, name="v_proj")
    # result_v.set_friendly_name("v_proj")

    # model = Model([result_q, result_k, result_v], [], [input_ids, attention_mask, position_ids, beam_idx])
    model = Model([result_q], [], [input_ids, attention_mask, position_ids, beam_idx])
    model.outputs[0].get_tensor().set_names({"q_proj"})
    # model.outputs[1].get_tensor().set_names({"k_proj"})
    # model.outputs[2].get_tensor().set_names({"v_proj"})
    return model

def create_partial_model_before_fc(configs, consts, layer_idx=0):
    print("Generating partial model up to RMSNorm...")

    # === Inputs ===
    input_ids = opset.parameter([-1, -1], Type.i64, name="input_ids")
    attention_mask = opset.parameter([-1, -1], Type.i64, name="attention_mask")
    position_ids = opset.parameter([-1, -1], Type.i64, name="position_ids")
    beam_idx = opset.parameter([-1], Type.i32, name="beam_idx")

    # === Step 1: Embedding ===
    inputs_embeds, embeddings = make_embedding(
        "model.embed_tokens",
        input_ids,
        consts,
        configs["model.embed_tokens.weight" + "_qtype"]
    )
    hidden_states = inputs_embeds

    # === Step 2: Inline RMSNorm ===
    epsilon = configs["rms_norm_eps"]
    epsilon_c = opset.constant(np.float16([[[epsilon]]]))

    powed = opset.power(hidden_states, opset.convert(np.array([[[2]]], np.float16), Type.f32))
    variance = opset.reduce_mean(powed, opset.constant([-1], dtype=np.int64), keep_dims=True)
    add = opset.add(variance, opset.convert(epsilon_c, Type.f32))
    sqrt = opset.sqrt(add)
    inv_sqrt = opset.divide(opset.convert(opset.constant([[[1]]], dtype=np.float16), Type.f32), sqrt)
    normed = opset.multiply(inv_sqrt, hidden_states, auto_broadcast="numpy")

    weight_key = f"model.layers.{layer_idx}.input_layernorm.weight"
    weight_values = consts["layers"][layer_idx][weight_key]
    if not np.all(weight_values == 1.0):
        weights = opset.convert(
            opset.constant(weight_values.reshape((1, 1, -1)), np.float16),
            Type.f32
        )
        normed = opset.multiply(normed, weights, auto_broadcast="numpy")

    input_layernorm = normed

    # === Output ===
    result = opset.result(input_layernorm, name="input_layernorm")
    result.set_friendly_name("input_layernorm")

    model = Model(
        [result],
        [],
        [input_ids, attention_mask, position_ids, beam_idx]
    )
    model.outputs[0].get_tensor().set_names({"input_layernorm"})

    return model

def create_model_before_final_layernorm(configs, consts):
    print("Generating partial OV model before final_layernorm (layer 0 only)...")
    beg = time.time()

    # Input nodes
    input_ids = opset.parameter([-1, -1], Type.i64, name="input_ids")
    attention_mask = opset.parameter([-1, -1], Type.i64, name="attention_mask")
    position_ids = opset.parameter([-1, -1], Type.i64, name="position_ids")
    beam_idx = opset.parameter([-1], Type.i32, name="beam_idx")

    # Embeddings
    inputs_embeds, embeddings = make_embedding(
        "model.embed_tokens", input_ids, consts,
        configs["model.embed_tokens.weight" + "_qtype"]
    )
    hidden_states = inputs_embeds

    # RoPE constants
    rope_const = init_rope(
        configs["head_size"],
        configs["max_position_embeddings"],
        configs["rope_freq_base"]
    )

    # Batch and hidden dim
    input_shape = opset.shape_of(input_ids)
    batch_size = opset.gather(input_shape, opset.constant([0], dtype=np.int64), opset.constant([0], dtype=np.int64))
    hidden_dim = opset.constant([3], dtype=np.int64)

    # Initial shared tensors
    causal_mask = None
    cos_sin_cached = None
    output_shape = None
    sinks = []

    # Only apply the first layer (i = 0)
    hidden_states, layer_sinks, causal_mask, cos_sin_cached, output_shape = layer(
        configs, consts, 0, hidden_states, attention_mask, causal_mask,
        position_ids, rope_const, beam_idx, batch_size, hidden_dim,
        cos_sin_cached, output_shape
    )
    sinks = sinks + layer_sinks

    # Output result before final_layernorm
    result_node = opset.result(hidden_states, name="layer0_output")
    result_node.set_friendly_name("layer0_output")

    cost = time.time() - beg
    print(f"Partial model generated. Time taken: {cost:.2f}s")

    model = Model([result_node], sinks, [input_ids, attention_mask, position_ids, beam_idx])
    model.outputs[0].get_tensor().set_names({"layer0_output"})

    return model

def get_quantizaiton_type(gguf_type):
    if gguf_type == 0 or gguf_type == 1:
        qtype = "F16"
        print("Working with FP16 model")
    elif gguf_type == 2 or gguf_type == 3 or gguf_type == 12 or gguf_type == 15:
        # MOSTLY_Q4_0 or MOSTLY_Q4_1
        qtype = "Q4_0"
        # print bits value
        print("Working with INT4 quantized model")
    elif gguf_type == 7:
        # MOSTLY_Q8_0 = 7
        qtype = "Q8_0"
        print("Working with INT8 quantized model")
    elif gguf_type == 18:
        qtype ="Q6_K"
    else:
        qtype = None
        raise ValueError("Using unsupported GGUF quantization: ", gguf_type)
    return qtype

def check_q_layer(name):
    names = ["attn_q", "attn_output"]
    for n in names:
        if n in name:
            return True
    return False

def load_gguf_model(model_path: str, all_layer: bool) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract configurations and weights from GGUF model"""
    print(f"extracting from GGUF model '{model_path}'...")
    beg = time.time()
    config = {}
    # Load GGUF model
    with open(model_path, "rb") as f:
        metadata, tensorinfo = pygguf.load_gguf(f)
        weights={}
        for name in tensorinfo: 
            weight, scales, biases, ggml_name = pygguf.load_gguf_tensor(f, tensorinfo, name, all_layer)
            config[name+"_qtype"] = ggml_name
            shape = tensorinfo[name]["shape"]
            if scales is not None:
                if check_q_layer(name):#TODO                
                    weights[name] = weight.reshape([shape[0], -1])
                    weights[name.replace(".weight", ".scales")] = scales.reshape([shape[0], -1])
                    weights[name.replace(".weight", ".biases")] = biases.reshape([shape[0], -1])
                else:                
                    weights[name] = weight.reshape([shape[-1], -1])
                    weights[name.replace(".weight", ".scales")] = scales.reshape([shape[-1], -1])
                    weights[name.replace(".weight", ".biases")] = biases.reshape([shape[-1], -1])
            else:
                if name == "output.weight":
                    weights[name] = weight.reshape([shape[-1], -1])
                else:
                    weights[name] = weight

    architecture = metadata["general.architecture"]
    config.update({
        "architecture": architecture,
        "layer_num": metadata[architecture+".block_count"],
        "head_num": metadata[architecture+".attention.head_count"],
        "head_size": metadata[architecture+".embedding_length"] // metadata[architecture+".attention.head_count"],
        "head_num_kv": metadata.get(architecture+".attention.head_count_kv", metadata[architecture+".attention.head_count"]),
        "hidden_size": metadata[architecture+".embedding_length"],
        "max_position_embeddings": metadata.get(architecture+".context_length", np.int32([2048])),
        "rotary_dims": 128,
        "rms_norm_eps": metadata[architecture+".attention.layer_norm_rms_epsilon"],
        "rope_freq_base": metadata.get(architecture+".rope.freq_base", np.float32(10000)),
        "file_type": get_quantizaiton_type(metadata.get("general.file_type")),
        # "model_id": model_id,        
    })


    # Extract weights and biases
    print("Extract weights and biases")
    print("============= Weight keys ============")
    # print(list(weights.keys()))
    consts = {
        "model.embed_tokens.weight": np.array(weights["token_embd.weight"]),
        "model.norm.weight": np.array(weights["output_norm.weight"]),
        "lm_head.weight": np.array(weights["output.weight"]) if weights.get("output.weight", None) is not None else None,
        "layers": []
    }
    if weights.get("token_embd.scales", None) is not None:
        consts["model.embed_tokens.scales"] = np.array(weights["token_embd.scales"])
        consts["model.embed_tokens.biases"] = np.array(weights["token_embd.biases"])

    if weights.get("output.scales", None) is not None:
        consts["lm_head.scales"] = np.array(weights["output.scales"])
        consts["lm_head.biases"] = np.array(weights["output.biases"])
    
    config["model.embed_tokens.weight"+"_qtype"] = config["token_embd.weight"+"_qtype"]
    
    if config.get("output.weight"+"_qtype", None) is not None:
        config["lm_head.weight"+"_qtype"] = config["output.weight"+"_qtype"] 

    # Extract layer weights
    print("Extract layer weights")
    if architecture == "qwen2":    
        for i in range(config["layer_num"]):
            layer_weights = {
                f"model.layers.{i}.input_layernorm.weight": np.array(weights[f"blk.{i}.attn_norm.weight"]),
                f"model.layers.{i}.post_attention_layernorm.weight": np.array(weights[f"blk.{i}.ffn_norm.weight"]),
                # "model.layers.self_attn.q_proj.bias": None,
                f"model.layers.{i}.self_attn.q_proj.bias": np.array(weights[f"blk.{i}.attn_q.bias"]),
                f"model.layers.{i}.self_attn.q_proj.weight": np.array(weights[f"blk.{i}.attn_q.weight"]),
                # "model.layers.self_attn.k_proj.bias": None,
                f"model.layers.{i}.self_attn.k_proj.bias": np.array(weights[f"blk.{i}.attn_k.bias"]),
                f"model.layers.{i}.self_attn.k_proj.weight": np.array(weights[f"blk.{i}.attn_k.weight"]),
                # "model.layers.self_attn.v_proj.bias": None,
                f"model.layers.{i}.self_attn.v_proj.bias": np.array(weights[f"blk.{i}.attn_v.bias"]),
                f"model.layers.{i}.self_attn.v_proj.weight": np.array(weights[f"blk.{i}.attn_v.weight"]),
                f"model.layers.{i}.self_attn.o_proj.bias": None,
                f"model.layers.{i}.self_attn.o_proj.weight": np.array(weights[f"blk.{i}.attn_output.weight"]),
                f"model.layers.{i}.mlp.gate_proj.bias": None,
                f"model.layers.{i}.mlp.gate_proj.weight": np.array(weights[f"blk.{i}.ffn_gate.weight"]),
                f"model.layers.{i}.mlp.up_proj.bias": None,
                f"model.layers.{i}.mlp.up_proj.weight": np.array(weights[f"blk.{i}.ffn_up.weight"]),
                f"model.layers.{i}.mlp.down_proj.bias": None,
                f"model.layers.{i}.mlp.down_proj.weight": np.array(weights[f"blk.{i}.ffn_down.weight"])
            }
            if not ("F16" in config["file_type"] or "F32" in config["file_type"]):
                q_weights = {
                    f"model.layers.{i}.self_attn.q_proj.scales": np.array(weights[f"blk.{i}.attn_q.scales"]),
                    f"model.layers.{i}.self_attn.k_proj.scales": np.array(weights[f"blk.{i}.attn_k.scales"]),
                    f"model.layers.{i}.self_attn.v_proj.scales": np.array(weights[f"blk.{i}.attn_v.scales"]),
                    f"model.layers.{i}.self_attn.o_proj.scales": np.array(weights[f"blk.{i}.attn_output.scales"]),
                    f"model.layers.{i}.mlp.gate_proj.scales": np.array(weights[f"blk.{i}.ffn_gate.scales"]),
                    f"model.layers.{i}.mlp.up_proj.scales": np.array(weights[f"blk.{i}.ffn_up.scales"]),
                    f"model.layers.{i}.mlp.down_proj.scales": np.array(weights[f"blk.{i}.ffn_down.scales"]),

                    f"model.layers.{i}.self_attn.q_proj.biases": np.array(weights[f"blk.{i}.attn_q.biases"]),
                    f"model.layers.{i}.self_attn.k_proj.biases": np.array(weights[f"blk.{i}.attn_k.biases"]),
                    f"model.layers.{i}.self_attn.v_proj.biases": np.array(weights[f"blk.{i}.attn_v.biases"]),
                    f"model.layers.{i}.self_attn.o_proj.biases": np.array(weights[f"blk.{i}.attn_output.biases"]),
                    f"model.layers.{i}.mlp.gate_proj.biases": np.array(weights[f"blk.{i}.ffn_gate.biases"]),
                    f"model.layers.{i}.mlp.up_proj.biases": np.array(weights[f"blk.{i}.ffn_up.biases"]),
                    f"model.layers.{i}.mlp.down_proj.biases": np.array(weights[f"blk.{i}.ffn_down.biases"])
                }
                layer_weights = {**layer_weights, **q_weights}
            consts["layers"].append(layer_weights)
    elif architecture == "llama":  
        for i in range(config["layer_num"]):
            layer_weights = {
                f"model.layers.{i}.input_layernorm.weight": np.array(weights[f"blk.{i}.attn_norm.weight"]),
                f"model.layers.{i}.post_attention_layernorm.weight": np.array(weights[f"blk.{i}.ffn_norm.weight"]),
                f"model.layers.{i}.self_attn.q_proj.bias": None,
                f"model.layers.{i}.self_attn.q_proj.weight": np.array(weights[f"blk.{i}.attn_q.weight"]),
                f"model.layers.{i}.self_attn.k_proj.bias": None,
                f"model.layers.{i}.self_attn.k_proj.weight": np.array(weights[f"blk.{i}.attn_k.weight"]),
                f"model.layers.{i}.self_attn.v_proj.bias": None,
                f"model.layers.{i}.self_attn.v_proj.weight": np.array(weights[f"blk.{i}.attn_v.weight"]),
                f"model.layers.{i}.self_attn.o_proj.bias": None,
                f"model.layers.{i}.self_attn.o_proj.weight": np.array(weights[f"blk.{i}.attn_output.weight"]),
                f"model.layers.{i}.mlp.gate_proj.bias": None,
                f"model.layers.{i}.mlp.gate_proj.weight": np.array(weights[f"blk.{i}.ffn_gate.weight"]),
                f"model.layers.{i}.mlp.up_proj.bias": None,
                f"model.layers.{i}.mlp.up_proj.weight": np.array(weights[f"blk.{i}.ffn_up.weight"]),
                f"model.layers.{i}.mlp.down_proj.bias": None,
                f"model.layers.{i}.mlp.down_proj.weight": np.array(weights[f"blk.{i}.ffn_down.weight"])
            }
            if not ("F16" in config["file_type"] or "F32" in config["file_type"]):
                q_weights = {
                    f"model.layers.{i}.self_attn.q_proj.scales": np.array(weights[f"blk.{i}.attn_q.scales"]),
                    f"model.layers.{i}.self_attn.k_proj.scales": np.array(weights[f"blk.{i}.attn_k.scales"]),
                    f"model.layers.{i}.self_attn.v_proj.scales": np.array(weights[f"blk.{i}.attn_v.scales"]),
                    f"model.layers.{i}.self_attn.o_proj.scales": np.array(weights[f"blk.{i}.attn_output.scales"]),
                    f"model.layers.{i}.mlp.gate_proj.scales": np.array(weights[f"blk.{i}.ffn_gate.scales"]),
                    f"model.layers.{i}.mlp.up_proj.scales": np.array(weights[f"blk.{i}.ffn_up.scales"]),
                    f"model.layers.{i}.mlp.down_proj.scales": np.array(weights[f"blk.{i}.ffn_down.scales"]),

                    f"model.layers.{i}.self_attn.q_proj.biases": np.array(weights[f"blk.{i}.attn_q.biases"]),
                    f"model.layers.{i}.self_attn.k_proj.biases": np.array(weights[f"blk.{i}.attn_k.biases"]),
                    f"model.layers.{i}.self_attn.v_proj.biases": np.array(weights[f"blk.{i}.attn_v.biases"]),
                    f"model.layers.{i}.self_attn.o_proj.biases": np.array(weights[f"blk.{i}.attn_output.biases"]),
                    f"model.layers.{i}.mlp.gate_proj.biases": np.array(weights[f"blk.{i}.ffn_gate.biases"]),
                    f"model.layers.{i}.mlp.up_proj.biases": np.array(weights[f"blk.{i}.ffn_up.biases"]),
                    f"model.layers.{i}.mlp.down_proj.biases": np.array(weights[f"blk.{i}.ffn_down.biases"])
                }
                layer_weights = {**layer_weights, **q_weights}
            consts["layers"].append(layer_weights)

    else:
        raise ValueError("Unsupported architecture: ", architecture)
    
    
    
    cost = time.time() - beg
    print(f"extracting done, cost {cost:.2f} seconds.\nmodel configs:")

    return config, consts


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--org_model_path", type=str, default="Model ID (can be a Hugginface Hub id, or a local directory)")
    parser.add_argument("--ov_model_path", type=str, nargs="?", default="./gen/llama-2-7b-chat/")
    parser.add_argument("--model_id", type=str, nargs="?", default=None)
    parser.add_argument("--all_layer", action='store_true', default=False )
    args = parser.parse_args()
    beg = time.time()
    os.makedirs(args.ov_model_path, exist_ok=True)

    config, consts = load_gguf_model(args.org_model_path, args.all_layer)
    # model = create_model(config, consts)
    model = create_partial_model_before_fc(config, consts)
    cost = time.time() - beg
    print(f"convert done, cost {cost:.2f} seconds.")
    show_model(model)

    print(f"serialize ov model to '{args.ov_model_path}'...")
    beg = time.time()
    serialize(model, os.path.join(args.ov_model_path, OV_XML_FILE_NAME))
    cost = time.time() - beg
    print(f"serialize done, cost {cost:.2f} seconds.")

    # save tokenizer and config to load with GenAI and Optimum
    model_id = args.model_id #or config["model_id"] #"HuggingFaceTB/SmolLM2-135M" #"meta-llama/Llama-2-7b-chat-hf"
    # if model_id:
    #     print(f"save tokenzier to '{args.ov_model_path}' ...")
    #     save_tokenzier(model_id, args.ov_model_path)
    #     config = AutoConfig.from_pretrained(model_id)
    #     config.save_pretrained(args.ov_model_path)
    # else:
    #     print("[WARNING]: Tokenizer and config.json were not saved because model_id was not found or provided as an option.")
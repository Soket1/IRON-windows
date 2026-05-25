# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pack Q4_0 weights into the post_attn_fused DDR layout.

The IRON design loads three packed weight buffers (w_o, w_gu, w_d) using
linear TAPs of the form sizes=[1,1,1,bytes_col_*], strides=[0,0,0,1].

Per-column layout for each weight:
  w_o  (O_proj):    [col0_tiles ... col_{C-1}_tiles]
                    each col has  tiles_per_col_o  tiles of packed_tile_o bytes
  w_gu (gate+up):   [col0_gate, col0_up, col1_gate, col1_up, ...]
                    each col-phase has bytes_col_gu of tiles
  w_d  (Down):      [col0_tiles ... col_{C-1}_tiles]  (K=hidden_dim)

Within each tile (m_input consecutive output rows):
  [m_input * K/2 bytes packed weights] then [m_input * groups * 2 bytes bf16 scales]

Packed weight bytes: byte k of a row = elem[2k] | (elem[2k+1] << 4), with
each element being an unsigned uint4 nibble in [0, 15]. The kernel
subtracts 8 internally (signed range [-8, 7]) and multiplies by the scale.

This packer matches the IRON design.py compile-time parameters:
  m_input_o = 2, m_input_d = 2, m_input_gu = 8, group_size = 32
"""

import numpy as np
from ml_dtypes import bfloat16


Q4_0_BLOCK_BYTES = 18   # 2-byte fp16 scale + 16-byte qs
Q4_0_BLOCK_ELEMS = 32


def _fp16_to_bf16_uint16(fp16_u16: np.ndarray) -> np.ndarray:
    """Convert raw fp16 bit patterns to bf16 bit patterns. Mirrors the
    C++ implementation in ggml-xdna.cpp::fp16_to_bf16 exactly, including
    its subnormal normalization (NumPy's native fp16→fp32 conversion
    differs from the C++ version on subnormals)."""
    h = fp16_u16.astype(np.uint32, copy=True)
    sign     = (h & 0x8000) << 16
    exp_f16  = (h >> 10) & 0x1F
    mant_f16 = h & 0x3FF

    f32 = np.zeros_like(h)

    # Case A: exp == 0
    zero_mask = (exp_f16 == 0)
    # A1: signed zero — leave f32 = sign (already zero where applicable since
    #     mant=0 means the mantissa contributes nothing; sign-only is fine).
    f32 = np.where(zero_mask, sign, f32)
    # A2: subnormals — normalise. Vectorize the count-leading-zero shift loop.
    sub_mask = zero_mask & (mant_f16 != 0)
    if np.any(sub_mask):
        m = mant_f16[sub_mask].copy()
        e = -np.ones_like(m, dtype=np.int32)
        # Each iteration shifts m left and decrements e while bit 10 is unset.
        # fp16 mantissa is 10 bits, so worst case ~10 iterations.
        for _ in range(11):
            still = (m & 0x400) == 0
            if not np.any(still):
                break
            m = np.where(still, (m << 1) & 0xFFFFFFFF, m)
            e = np.where(still, e - 1, e)
        m_norm = m & 0x3FF
        exp_f32 = (127 - 15 + e + 1).astype(np.uint32)
        sub_bits = sign[sub_mask] | (exp_f32 << 23) | (m_norm << 13)
        f32_sub = f32.copy()
        f32_sub[sub_mask] = sub_bits
        f32 = f32_sub

    # Case B: exp == 0x1F (inf / NaN). Preserve mantissa.
    inf_mask = (exp_f16 == 0x1F)
    f32 = np.where(inf_mask,
                   sign | (np.uint32(0xFF) << 23) | (mant_f16 << 13),
                   f32)

    # Case C: normal numbers (covers everything else).
    norm_mask = ~zero_mask & ~inf_mask
    exp_f32_norm = (exp_f16.astype(np.int32) - 15 + 127).astype(np.uint32)
    f32 = np.where(norm_mask,
                   sign | (exp_f32_norm << 23) | (mant_f16 << 13),
                   f32)

    # Round-to-nearest-even down to bf16 (top 16 bits).
    f32 = (f32 + (0x7FFF + ((f32 >> 16) & 1))) & 0xFFFFFFFF
    return (f32 >> 16).astype(np.uint16)


def _q4_0_to_packed_row(q4_0_row: np.ndarray, K: int, group_size: int):
    """Unpack one row of Q4_0 blocks into (weight_bytes, scale_bf16).

    Q4_0 byte j of qs holds two nibbles: elem[j] = lo,  elem[j+16] = hi.
    IRON expects linear pairing: byte k = elem[2k] | (elem[2k+1] << 4).
    """
    num_groups = K // group_size
    assert q4_0_row.size == num_groups * Q4_0_BLOCK_BYTES, (
        f"row size mismatch: got {q4_0_row.size}, expected "
        f"{num_groups * Q4_0_BLOCK_BYTES}")
    blocks = q4_0_row.reshape(num_groups, Q4_0_BLOCK_BYTES)
    scales_fp16 = blocks[:, :2].copy().view(np.uint16).reshape(-1)
    qs = blocks[:, 2:]   # (num_groups, 16)
    # e[j] = qs[j] & 0xF; e[j+16] = (qs[j] >> 4) & 0xF
    elems = np.empty((num_groups, group_size), dtype=np.uint8)
    elems[:, :16] = qs & 0x0F
    elems[:, 16:] = (qs >> 4) & 0x0F
    # Repack pairwise: byte k = e[2k] | (e[2k+1] << 4)
    elems_flat = elems.reshape(-1)             # K elements per row
    even = elems_flat[0::2]
    odd  = elems_flat[1::2]
    weight_bytes = (even | (odd << 4)).astype(np.uint8)   # K/2 bytes
    scales_bf16 = _fp16_to_bf16_uint16(scales_fp16)        # num_groups bf16
    return weight_bytes, scales_bf16


def pack_one_weight(q4_0_buf: np.ndarray, M: int, K: int,
                    m_input: int, cols: int, group_size: int = 32) -> np.ndarray:
    """Pack a single Q4_0 weight tensor into the IRON per-col tile layout.

    Args:
      q4_0_buf:    raw Q4_0 bytes, length M * (K // group_size) * 18.
      M:           output dimension (rows in unpacked weight).
      K:           input dimension.
      m_input:     rows per tile.
      cols:        number of AIE columns the kernel splits M across.
      group_size:  Q4_0 block size, always 32.

    Returns:
      A 1-D uint8 buffer of total size cols * tiles_per_col * packed_tile bytes.
      Layout: col0 tiles 0..T-1, then col1, etc.
    """
    assert group_size == 32
    assert M % cols == 0
    rows_per_col = M // cols
    assert rows_per_col % m_input == 0
    tiles_per_col = rows_per_col // m_input

    num_groups = K // group_size
    row_stride_q40 = num_groups * Q4_0_BLOCK_BYTES
    weight_bytes_per_row = K // 2
    scale_bytes_per_row  = num_groups * 2
    packed_tile = m_input * weight_bytes_per_row + m_input * scale_bytes_per_row
    bytes_col   = tiles_per_col * packed_tile
    total       = cols * bytes_col

    assert q4_0_buf.size == M * row_stride_q40, (
        f"Q4_0 buffer size mismatch: got {q4_0_buf.size}, expected "
        f"{M * row_stride_q40}")
    src = q4_0_buf.reshape(M, row_stride_q40)
    out = np.zeros(total, dtype=np.uint8)

    for col in range(cols):
        col_base = col * bytes_col
        for tile_idx in range(tiles_per_col):
            row_start  = col * rows_per_col + tile_idx * m_input
            tile_off   = col_base + tile_idx * packed_tile
            scale_off  = tile_off + m_input * weight_bytes_per_row
            for r in range(m_input):
                w_bytes, s_bf16 = _q4_0_to_packed_row(
                    src[row_start + r], K, group_size)
                out[tile_off + r * weight_bytes_per_row :
                    tile_off + (r + 1) * weight_bytes_per_row] = w_bytes
                s_dst = out[scale_off + r * scale_bytes_per_row :
                            scale_off + (r + 1) * scale_bytes_per_row]
                s_dst.view(np.uint16)[:] = s_bf16
    return out


def pack_gate_up_interleaved(
        gate_q4_0: np.ndarray, up_q4_0: np.ndarray,
        N: int, K: int, m_input: int, cols: int,
        group_size: int = 32) -> np.ndarray:
    """Pack gate and up weights into the col-interleaved DDR layout that the
    Phase B IRON design expects.

    Output layout: [col0_gate, col0_up, col1_gate, col1_up, ...].
    Each col-phase chunk is bytes_col_gu bytes.

    Args:
      gate_q4_0, up_q4_0: raw Q4_0 weight buffers, each of size
                          N * (K // group_size) * 18 bytes.
      N:                  hidden_dim (gate/up output dim).
      K:                  embed_dim (gate/up input dim).
      m_input:            m_input_gu (typically 8).
      cols:               num AIE columns (typically 2).
    """
    gate_packed = pack_one_weight(gate_q4_0, N, K, m_input, cols, group_size)
    up_packed   = pack_one_weight(up_q4_0,   N, K, m_input, cols, group_size)
    bytes_col = gate_packed.size // cols
    assert up_packed.size == cols * bytes_col

    out = np.zeros(2 * cols * bytes_col, dtype=np.uint8)
    for col in range(cols):
        out[(2*col)   * bytes_col : (2*col+1) * bytes_col] = \
            gate_packed[col * bytes_col : (col+1) * bytes_col]
        out[(2*col+1) * bytes_col : (2*col+2) * bytes_col] = \
            up_packed  [col * bytes_col : (col+1) * bytes_col]
    return out


def pack_post_attn_fused_weights(
        o_proj_q4_0: np.ndarray,
        gate_q4_0:   np.ndarray,
        up_q4_0:     np.ndarray,
        down_q4_0:   np.ndarray,
        embed_dim: int = 2048,
        hidden_dim: int = 8192,
        cols: int = 2,
        group_size: int = 32,
        m_input_o:  int = 2,
        m_input_gu: int = 8,
        m_input_d:  int = 2):
    """Produce the three DDR buffers (w_o, w_gu, w_d) that the post_attn_fused
    IRON design consumes.

    Returns:
      (w_o, w_gu, w_d): three 1-D uint8 numpy arrays.
    """
    w_o  = pack_one_weight(o_proj_q4_0, embed_dim, embed_dim,
                           m_input_o, cols, group_size)
    w_gu = pack_gate_up_interleaved(gate_q4_0, up_q4_0,
                                    hidden_dim, embed_dim,
                                    m_input_gu, cols, group_size)
    w_d  = pack_one_weight(down_q4_0, embed_dim, hidden_dim,
                           m_input_d, cols, group_size)
    return w_o, w_gu, w_d


def expected_sizes(embed_dim=2048, hidden_dim=8192, cols=2, group_size=32,
                   m_input_o=2, m_input_gu=8, m_input_d=2):
    """Compute the expected DDR sizes for sanity checks."""
    groups_embed = embed_dim // group_size
    groups_hidden = hidden_dim // group_size
    packed_tile_o = m_input_o * embed_dim // 2 + m_input_o * groups_embed * 2
    packed_tile_gu = m_input_gu * embed_dim // 2 + m_input_gu * groups_embed * 2
    packed_tile_d = m_input_d * hidden_dim // 2 + m_input_d * groups_hidden * 2
    tiles_per_col_o  = (embed_dim  // cols) // m_input_o
    tiles_per_col_gu = (hidden_dim // cols) // m_input_gu
    tiles_per_col_d  = (embed_dim  // cols) // m_input_d
    bytes_col_o  = tiles_per_col_o  * packed_tile_o
    bytes_col_gu = tiles_per_col_gu * packed_tile_gu
    bytes_col_d  = tiles_per_col_d  * packed_tile_d
    return {
        "w_o":  cols * bytes_col_o,
        "w_gu": cols * 2 * bytes_col_gu,
        "w_d":  cols * bytes_col_d,
        "packed_tile_o":  packed_tile_o,
        "packed_tile_gu": packed_tile_gu,
        "packed_tile_d":  packed_tile_d,
        "tiles_per_col_o":  tiles_per_col_o,
        "tiles_per_col_gu": tiles_per_col_gu,
        "tiles_per_col_d":  tiles_per_col_d,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Pack Q4_0 weights for post_attn_fused")
    p.add_argument("--embed-dim",  type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=8192)
    p.add_argument("--cols",       type=int, default=2)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--print-sizes", action="store_true",
                   help="Print expected DDR buffer sizes and exit.")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out-prefix", type=str,
                   help="If set, write random test weights to "
                        "<prefix>_w_o.bin / w_gu.bin / w_d.bin.")
    args = p.parse_args()

    sizes = expected_sizes(args.embed_dim, args.hidden_dim, args.cols,
                            args.group_size)
    print(f"DDR buffer sizes (cols={args.cols}, embed={args.embed_dim}, "
          f"hidden={args.hidden_dim}):")
    print(f"  w_o:  {sizes['w_o']:>12,} bytes  "
          f"(packed_tile={sizes['packed_tile_o']}, "
          f"tiles/col={sizes['tiles_per_col_o']})")
    print(f"  w_gu: {sizes['w_gu']:>12,} bytes  "
          f"(packed_tile={sizes['packed_tile_gu']}, "
          f"tiles/col={sizes['tiles_per_col_gu']})")
    print(f"  w_d:  {sizes['w_d']:>12,} bytes  "
          f"(packed_tile={sizes['packed_tile_d']}, "
          f"tiles/col={sizes['tiles_per_col_d']})")

    if args.print_sizes:
        raise SystemExit(0)

    if args.out_prefix:
        rng = np.random.default_rng(args.seed)
        E, H, G = args.embed_dim, args.hidden_dim, args.group_size
        def rand_q4_0(M: int, K: int) -> np.ndarray:
            blocks = (M * K) // G
            buf = rng.integers(0, 256, size=blocks * Q4_0_BLOCK_BYTES,
                                dtype=np.uint8).copy()
            # Make scales small fp16 values (not garbage NaNs)
            scales = buf.reshape(-1, Q4_0_BLOCK_BYTES)[:, :2].view(np.float16)
            scales[:] = rng.uniform(-0.1, 0.1, size=scales.shape).astype(np.float16)
            return buf

        w_o_q4  = rand_q4_0(E, E)
        gate_q4 = rand_q4_0(H, E)
        up_q4   = rand_q4_0(H, E)
        down_q4 = rand_q4_0(E, H)

        w_o, w_gu, w_d = pack_post_attn_fused_weights(
            w_o_q4, gate_q4, up_q4, down_q4,
            embed_dim=E, hidden_dim=H, cols=args.cols, group_size=G)

        for name, buf in [("w_o", w_o), ("w_gu", w_gu), ("w_d", w_d)]:
            path = f"{args.out_prefix}_{name}.bin"
            buf.tofile(path)
            print(f"  wrote {path}: {buf.size:,} bytes")

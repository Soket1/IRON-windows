# SPDX-License-Identifier: Apache-2.0
"""decode_layer_f3best — design callback.

Unlike the other operators, this one is authored as RAW aie-dialect MLIR (below
the IRON ObjectFifo layer) so the per-phase weight-stream time-mux and the single
continuous dispatch can be expressed. The callback returns the MLIR text directly;
DesignGenerator does str() on the return value, so a string passes through verbatim.

The MLIR generator is f3best_emit.OFold8F3BestEmitter, validated byte-identical to
the standalone build and NPU-PASS (915.5us, rel_L2 0.02942 random / 0.01886 real
gguf layer-0). Shape is fixed to the Llama-3.2-1B geometry; the emitter raises on
any other shape rather than silently generalizing.
"""


def my_decode_layer_f3best(dev, embed_dim=2048, hidden_dim=8192, group_size=32,
                           head_dim=64, num_kv_heads=8, attn_group=4, seq_len=256,
                           with_npu_kv=False, flowkv_obj_name=None, rope_obj_name=None,
                           relay_obj_name=None):
    import os
    import sys

    if dev == "npu":
        raise ValueError("decode_layer_f3best targets npu2 (AIE2P); device_type 'npu' (NPU1) unsupported")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if with_npu_kv:
        from f3best_emit import OFold8F3BestEmitter
    else:
        from f3best_emit_nokv import OFold8F3BestEmitter

    return OFold8F3BestEmitter(
        NH=8, E=embed_dim, H=hidden_dim, G=group_size, M=4, HD=head_dim,
        AG=attn_group, SEQ=seq_len, flowkv_obj_name=flowkv_obj_name,
        rope_obj_name=rope_obj_name, relay_obj_name=relay_obj_name,
    ).emit_mlir()

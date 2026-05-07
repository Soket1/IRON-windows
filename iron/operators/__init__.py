# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Devel operators (no AIE prefix)
from .axpy.op import AXPY
from .dequant.op import Dequant
from .elementwise_add.op import ElementwiseAdd
from .elementwise_mul.op import ElementwiseMul
from .gelu.op import GELU
from .gemm.op import GEMM
from .gemv.op import GEMV
from .layer_norm.op import LayerNorm
from .leaky_relu.op import LeakyReLU
from .mem_copy.op import MemCopy
from .mha.op import MHA
from .relu.op import ReLU
from .rms_norm.op import RMSNorm
from .rope.op import RoPE
from .sigmoid.op import Sigmoid
from .silu.op import SiLU
from .softmax.op import Softmax
from .swiglu_decode.op import AIESwiGLUDecode
from .swiglu_prefill.op import AIESwiGLUPrefill
from .tanh.op import Tanh
from .transpose.op import Transpose
from .strided_copy.op import StridedCopy
from .repeat.op import Repeat

# AIE-prefixed aliases used by Llama application
AIEGEMM = GEMM
AIEGEMV = GEMV
AIESiLU = SiLU
AIERMSNorm = RMSNorm
AIEElementwiseMul = ElementwiseMul
AIEElementwiseAdd = ElementwiseAdd

# New operators from decode-fusion-llama (AIE prefix)
from .dual_gemv_silu_mul.op import AIEDualGEMVSiLUMul
from .flowkv_decode.op import AIEFlowKVDecode
from .fused_dequant_gemv.op import AIEFusedDequantGEMV
from .fused_qkv_proj.op import AIEFusedQKVProj
from .silu_mul.op import AIESiLUMul
from .swiglu_fused_decode.op import AIESwiGLUFusedDecode

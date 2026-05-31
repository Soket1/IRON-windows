# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense transformer layer kernel — time-multiplexed BD chains per shim col.

Goal: replicate FFLM's per-layer DMA topology (1-2 shim MM2S per col total
for ALL output streams) by chaining BDs on a single shim DMA channel via
aiex.dma_configure_task_for + bds() + multiple shim_dma_bd blocks per task.
This mirrors npu_dma_memcpy_nd + cmds2seq from the FFLM RE.
"""

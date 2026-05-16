# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from pathlib import Path
from ml_dtypes import bfloat16

from iron.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)


class AIEEchoV1(AIEOperatorBase):
    """Minimal DMA echo: DDR -> tile memcpy -> DDR.
    Tests basic DMA path correctness."""

    def __init__(self, size=256, context=None):
        self.size = size
        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"echo_v1_{self.size}"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="echo_v1",
            callback_args=["npu2", self.size],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    "echo.o",
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir / "aie_kernels" / "aie2p" / "echo.cc"
                        )
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{file_name_base}.bin", depends=[mlir_artifact]
        )

        self.xclbin_artifact = xclbin_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        self.add_buffer("input", self.size)
        self.add_buffer("output", self.size)
        self.add_kernel(
            "echo_v1",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
        )
        self.add_to_runlist("echo_v1", "input", "output")

    def forward(self, input_data):
        return self.run(input_data)


class AIEEchoV2(AIEOperatorBase):
    """Dual ObjectFifo echo: bo_a + bo_b -> tile concat -> DDR.
    Tests K+V style multi-DMA pattern."""

    def __init__(self, size=128, context=None):
        self.size = size
        AIEOperatorBase.__init__(self, context=context)

    def set_up_artifacts(self):
        operator_dir = Path(__file__).parent
        file_name_base = f"echo_v2_{self.size}"

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{file_name_base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="echo_v2",
            callback_args=["npu2", self.size],
        )

        xclbin_artifact = XclbinArtifact.new(
            f"{file_name_base}.xclbin",
            depends=[
                mlir_artifact,
                KernelObjectArtifact.new(
                    "echo.o",
                    depends=[
                        SourceArtifact.new(
                            self.context.base_dir / "aie_kernels" / "aie2p" / "echo.cc"
                        )
                    ],
                ),
            ],
        )

        insts_artifact = InstsBinArtifact.new(
            f"{file_name_base}.bin", depends=[mlir_artifact]
        )

        self.xclbin_artifact = xclbin_artifact
        self.add_artifacts([xclbin_artifact, insts_artifact])

    def set_up_runtime(self):
        self.add_buffer("input_a", self.size)
        self.add_buffer("input_b", self.size)
        self.add_buffer("output", 2 * self.size)
        self.add_kernel(
            "echo_v2",
            self.xclbin_artifact,
            self.xclbin_artifact.kernel_name,
        )
        self.add_to_runlist("echo_v2", "input_a", "input_b", "output")

    def forward(self, input_a, input_b):
        return self.run(input_a, input_b)

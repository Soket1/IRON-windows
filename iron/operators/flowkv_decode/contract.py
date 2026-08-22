# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free compile-time contract for FlowKV kernel specializations.

This module deliberately has no AIE, XRT, NumPy, or Torch dependency.  It is
shared by the FlowKV operator and the Python compilation bridge, and can be
covered on a CPU-only machine.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable
from pathlib import Path


class FlowKVContractError(ValueError):
    """Raised when FlowKV compile-time specialization inputs are invalid."""


_PROTECTED_GEOMETRY_MACROS = frozenset({"HEAD_DIM", "MAX_Q_HEADS", "MAX_CHUNK"})
_ALLOWED_TUNING_MACROS = frozenset({
    "FLOWKV_PRESCALE_Q",
    "FLOWKV_SCORE_STUB",
    "FLOWKV_SCORE_NOREDUCE",
    "FLOWKV_VEC_EXP",
    "FLOWKV_NOEXP",
    "FLOWKV_VALUE_STUB",
    "FLOWKV_VALUE_LEGACY",
    "FLOWKV_NOVALUE",
    "FLOWKV_VALUE_AMAC",
    # #267i: seed the score dot product from the c == 0 product instead of a
    # hoistable aie::zeros(), so no accumulator initialization survives the
    # unroll at HEAD_DIM=128 (n_chunks=4). Correctness A/B for #187.
    "FLOWKV_DOT_MULINIT",
    # #268: read Q directly from q_in in flowkv_score_chunk_bf16 instead of
    # static rotated_q buffer, fixing stale Q in 2-chunk mode.
    "FLOWKV_Q_IN_DIRECT",
    # #271: Q sync protocol v2 - separate lock pairs for center->mem (Qc2m/Qp_c2m)
    # and center->score (Qc2s/Qp_c2s) to prevent stale Q in 2-chunk continuous dispatch.
    "FLOWKV_Q_SYNC_V2",
})
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALUE_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise FlowKVContractError(f"{name} must be a positive integer")


def flowkv_geometry_flags(
    head_dim: int,
    max_q_heads: int,
    max_chunk: int,
) -> tuple[str, str, str]:
    """Return the mandatory geometry definitions in their canonical order."""
    _require_positive_int("head_dim", head_dim)
    _require_positive_int("max_q_heads", max_q_heads)
    _require_positive_int("max_chunk", max_chunk)
    return (
        f"-DHEAD_DIM={head_dim}",
        f"-DMAX_Q_HEADS={max_q_heads}",
        f"-DMAX_CHUNK={max_chunk}",
    )


def flowkv_effective_chunk(seq_len: int, preferred_chunk: int = 32) -> int:
    """Return the static FlowKV chunk capacity selected by live wrappers."""
    _require_positive_int("seq_len", seq_len)
    _require_positive_int("preferred_chunk", preferred_chunk)
    return preferred_chunk if seq_len % preferred_chunk == 0 else seq_len


def flowkv_is_npu1_device(dev: object) -> bool:
    """Return whether an IRON device name or device object represents NPU1."""
    resolved = dev.resolve() if hasattr(dev, "resolve") else dev
    name = getattr(resolved, "name", resolved)
    return name in ("npu", "npu1")


def flowkv_tuning_tokens_from_environment() -> tuple[str, ...]:
    """Read and validate optional FlowKV tuning tokens without exposing raw text."""
    return normalize_flowkv_tuning_tokens(os.environ.get("FLOWKV_CFLAGS", "").split())


def _protected_geometry_name(name: str) -> bool:
    # Reject joined forms such as -DHEAD_DIM128 as well as an exact define.
    return any(name.startswith(protected) for protected in _PROTECTED_GEOMETRY_MACROS)


def _invalid_tuning() -> FlowKVContractError:
    return FlowKVContractError("FLOWKV_CFLAGS contains an unsupported tuning flag")


def _protected_geometry_override() -> FlowKVContractError:
    return FlowKVContractError("FLOWKV_CFLAGS must not override FlowKV geometry")


def normalize_flowkv_tuning_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    """Validate and canonicalize FlowKV's optional non-geometry tuning flags.

    Only the FlowKV kernel's explicitly supported feature defines are accepted.
    Definitions of the static-capacity macros, including split and joined
    ``-D``/``-U`` spellings, are rejected before they influence a compiler
    command, object name, or cache key.  Duplicate tuning definitions are also
    rejected so a cache identity cannot hide order-dependent preprocessor state.
    """
    raw_tokens = tuple(tokens)
    if len(raw_tokens) > 32:
        raise _invalid_tuning()
    normalized: list[str] = []
    seen_macros: set[str] = set()
    index = 0

    while index < len(raw_tokens):
        token = raw_tokens[index]
        if not isinstance(token, str) or not token or len(token) > 128:
            raise _invalid_tuning()

        if token in ("-D", "-U"):
            if index + 1 >= len(raw_tokens):
                raise _invalid_tuning()
            kind = token
            definition = raw_tokens[index + 1]
            index += 2
        elif token.startswith("-D") or token.startswith("-U"):
            kind = token[:2]
            definition = token[2:]
            index += 1
        else:
            raise _invalid_tuning()

        if not definition or len(definition) > 120:
            raise _invalid_tuning()

        if kind == "-U":
            if _IDENTIFIER_RE.fullmatch(definition) and _protected_geometry_name(definition):
                raise _protected_geometry_override()
            raise _invalid_tuning()

        name, separator, value = definition.partition("=")
        if not _IDENTIFIER_RE.fullmatch(name):
            raise _invalid_tuning()
        if _protected_geometry_name(name):
            raise _protected_geometry_override()
        if name not in _ALLOWED_TUNING_MACROS:
            raise _invalid_tuning()
        if separator:
            if not value or not _VALUE_RE.fullmatch(value) or value == "0":
                raise _invalid_tuning()
        if name in seen_macros:
            raise _invalid_tuning()
        seen_macros.add(name)
        normalized.append(f"-D{name}{separator}{value}" if separator else f"-D{name}")

    return tuple(sorted(normalized))


def flowkv_tuning_fingerprint(tokens: Iterable[str]) -> str:
    """Return the stable FNV-1a identity of normalized tuning tokens."""
    normalized = normalize_flowkv_tuning_tokens(tokens)
    canonical = "flowkv-cflags-v1;" + str(len(normalized)) + ";"
    canonical += "".join(f"{len(token)}:{token}" for token in normalized)

    value = 0xCBF29CE484222325
    for byte in canonical.encode("ascii"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _flowkv_source_hash(base_dir: Path | None = None) -> str:
    """Return a short hash of FlowKV kernel source files.

    This hash is mixed into the cache key so that any change to the kernel
    source invalidates the cached artifact, preventing stale .o reuse.
    """
    if base_dir is None:
        # contract.py is at IRON-windows/iron/operators/flowkv_decode/contract.py
        # flowkv.cc is at IRON-windows/aie_kernels/aie2p/flowkv.cc
        contract_path = Path(__file__).resolve()
        base_dir = contract_path.parents[3] / "aie_kernels" / "aie2p"
    source_files = [
        base_dir / "flowkv.cc",
    ]
    hasher = hashlib.sha256()
    for src in source_files:
        if src.is_file():
            hasher.update(src.read_bytes())
    return hasher.hexdigest()[:16]


def flowkv_bundle_cache_key(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    chunk_size: int,
    num_cols: int,
    tuning_fingerprint: str = "63624e81faf2e175",
    source_hash: str | None = None,
) -> str:
    """Return the directory key shared by Python and native FlowKV builds."""
    _require_positive_int("num_heads", num_heads)
    _require_positive_int("num_kv_heads", num_kv_heads)
    _require_positive_int("head_dim", head_dim)
    _require_positive_int("seq_len", seq_len)
    _require_positive_int("chunk_size", chunk_size)
    _require_positive_int("num_cols", num_cols)
    if not isinstance(tuning_fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{16}", tuning_fingerprint
    ):
        raise FlowKVContractError("tuning_fingerprint must be a 16-character lowercase hex value")
    if source_hash is None:
        source_hash = _flowkv_source_hash()
    return (
        f"flowkv_H{num_heads}_KV{num_kv_heads}_d{head_dim}_S{seq_len}"
        f"_C{chunk_size}_{num_cols}col_t{tuning_fingerprint}_s{source_hash}"
    )


def flowkv_artifact_identity(
    head_dim: int,
    max_q_heads: int,
    max_chunk: int,
    tuning_tokens: Iterable[str] = (),
    source_hash: str | None = None,
) -> str:
    """Return the cache-safe identity for FlowKV generated artifacts."""
    flowkv_geometry_flags(head_dim, max_q_heads, max_chunk)
    if source_hash is None:
        source_hash = _flowkv_source_hash()
    # Include source hash in the artifact identity so it propagates to xclbin cache key
    return f"{flowkv_tuning_fingerprint(tuning_tokens)}_s{source_hash}"


def flowkv_kernel_object_name(
    head_dim: int,
    max_q_heads: int,
    max_chunk: int,
    tuning_tokens: Iterable[str] = (),
    source_hash: str | None = None,
) -> str:
    """Return an object identity covering FlowKV geometry, tuning, and source."""
    flowkv_geometry_flags(head_dim, max_q_heads, max_chunk)
    normalized = normalize_flowkv_tuning_tokens(tuning_tokens)
    base = f"flowkv_{head_dim}d_h{max_q_heads}_c{max_chunk}"
    if source_hash is None:
        source_hash = _flowkv_source_hash()
    if normalized:
        return f"{base}_t{flowkv_tuning_fingerprint(normalized)}_s{source_hash}.o"
    return f"{base}_s{source_hash}.o"

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from dataclasses import dataclass, field

from ..index_build_parameters import SpaceType

from .gpu_index_cagra_config import GPUIndexCagraConfig
from .index_hnsw_cagra_config import IndexHNSWCagraConfig


@dataclass
class GPUIndexBuildConfig:
    index_hnsw_cagra_config: IndexHNSWCagraConfig = field(
        default_factory=IndexHNSWCagraConfig
    )
    gpu_index_cagra_config: GPUIndexCagraConfig = field(
        default_factory=GPUIndexCagraConfig
    )

    # type of metric the gpuIndex is created with
    metric: SpaceType

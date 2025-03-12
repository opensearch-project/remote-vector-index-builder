# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from dataclasses import dataclass
from abc import ABC


@dataclass
class GPUIndexConfig(ABC):
    """Base class for GPU Index configurations"""

    # GPU Device on which the index is resident
    device: int = 0

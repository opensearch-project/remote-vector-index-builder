# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC
from dataclasses import dataclass


@dataclass
class CPUIndexResponse(ABC):
    """Extend this base class to implement the response to the CPUIndexBuildService method
    to create a CPU Index
    """

    pass

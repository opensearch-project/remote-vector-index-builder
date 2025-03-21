# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from enum import Enum


class CagraGraphBuildAlgo(Enum):
    """
    Use this enum to switch the type of FAISS GPU Cagra Index to build
    in the FaissGpuIndexCagraBuilder
    """

    IVF_PQ = "IVF_PQ"
    NN_DESCENT = "NN_DESCENT"

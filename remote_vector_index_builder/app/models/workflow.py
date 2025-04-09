# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from pydantic import BaseModel
from core.common.models.index_build_parameters import IndexBuildParameters


class BuildWorkflow(BaseModel):
    """
    Represents a workflow for building a vector index with specified resource requirements.

    This class encapsulates all necessary information for executing a vector index build,
    including job identification, resource requirements, and build parameters.

    Attributes:
        job_id (str): Unique identifier for the build job
        gpu_memory_required (float): Amount of GPU memory required for the build process in bytes
        cpu_memory_required (float): Amount of CPU memory required for the build process in bytes
        index_build_parameters (IndexBuildParameters): Parameters specifying how to build the index

    """

    job_id: str
    gpu_memory_required: float
    cpu_memory_required: float
    index_build_parameters: IndexBuildParameters

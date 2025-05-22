# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from pydantic_settings import BaseSettings
from app.storage.types import RequestStoreType
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Settings class for the application. Pulls the settings
    from the Docker container environment variables
    """

    # Request Store settings
    request_store_type: RequestStoreType = os.environ.get(
        "REQUEST_STORE_TYPE", RequestStoreType.MEMORY
    )

    # In memory settings
    request_store_max_size: int = int(os.environ.get("REQUEST_STORE_MAX_SIZE", "10000"))
    request_store_ttl_seconds: Optional[int] = int(
        os.environ.get("REQUEST_STORE_TTL_SECONDS", "1800")
    )

    # Resource Manager settings, in GB
    # Value later gets multiplied by 10**9
    gpu_memory_limit: float = float(os.environ.get("GPU_MEMORY_LIMIT", "24.0"))
    cpu_memory_limit: float = float(os.environ.get("CPU_MEMORY_LIMIT", "32.0"))

    # Workflow Executor settings
    max_workers: int = int(os.environ.get("MAX_WORKERS", "2"))

    # Service settings
    service_name: str = "remote-vector-index-builder-api"
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")

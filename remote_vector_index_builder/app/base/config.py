# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from pydantic_settings import BaseSettings
from app.storage.types import RequestStoreType
from typing import Optional


class Settings(BaseSettings):
    """
    Settings class for the application. Pulls the settings
    from the Docker container environment variables
    """

    # Request Store settings
    request_store_type: RequestStoreType = RequestStoreType.MEMORY

    # In memory settings
    request_store_max_size: int = 1000000
    request_store_ttl_seconds: Optional[int] = 600

    # Resource Manager settings, in bytes
    gpu_memory_limit: float = 24.0 * 10**9
    cpu_memory_limit: float = 32.0 * 10**9

    # Workflow Executor settings
    max_workers: int = 5

    # Service settings
    service_name: str = "remote-vector-index-builder-api"
    log_level: str = "INFO"

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from fastapi import APIRouter
from app.base.config import Settings

import logging

router = APIRouter()
settings = Settings()
logger = logging.getLogger(__name__)


@router.get("/_heart_beat")
def heart_beat() -> str:
    """
    Health check endpoint to verify the service is running.

    This endpoint provides a simple health check mechanism to verify that the service
    is up and responding to requests.

    Returns:
        str: A simple response indicating the service is alive and responding.
    """
    return settings.service_name

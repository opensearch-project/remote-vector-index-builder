# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from app.base.exceptions import HashCollisionError, CapacityError
from fastapi import APIRouter, HTTPException, Request
from app.schemas.api import CreateJobResponse
from core.common.models import IndexBuildParameters

import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/_build")
def create_job(
    index_build_parameters: IndexBuildParameters, request: Request
) -> CreateJobResponse:
    """
    Create a new index build job with the provided parameters.

    This endpoint initiates a new job for building an index based on the provided
    parameters. It handles job creation through the job service and manages
    potential error scenarios.

    Args:
        index_build_parameters (IndexBuildParameters): Parameters for the index build job
        request (Request): FastAPI request object containing application state

    Returns:
        CreateJobResponse: Response object containing the created job ID

    Raises:
        HTTPException:
            - 429 status code if a hash collision occurs
            - 507 status code if system memory capacity is exceeded
    """
    try:
        job_service = request.app.state.job_service
        job_id = job_service.create_job(index_build_parameters)
    except HashCollisionError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except CapacityError as e:
        raise HTTPException(status_code=507, detail=str(e))
    return CreateJobResponse(job_id=job_id)

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from fastapi import APIRouter, Request
import json

import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/_jobs")
def get_jobs(request: Request) -> str:
    """
    Retrieve all jobs from the job service.

    This endpoint returns a list of all jobs in the system. The jobs are retrieved
    from the job service stored in the application state and serialized to JSON.

    Args:
        request (Request): The request object containing the application state
                         with the job service.

    Returns:
        str: A JSON string containing all jobs, with each job's attributes serialized
             into a readable format with proper indentation.

    """
    job_service = request.app.state.job_service
    jobs = job_service.get_jobs()
    return json.dumps(jobs, default=lambda o: o.__dict__, indent=4)

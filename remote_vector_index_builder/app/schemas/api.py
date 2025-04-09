# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from pydantic import BaseModel
from typing import Optional

from app.models.job import JobStatus


class CreateJobResponse(BaseModel):
    """
    Response model for job creation endpoint.

    Attributes:
        job_id (str): Unique identifier for the created job.
    """

    job_id: str


class GetStatusResponse(BaseModel):
    """
    Response model for retrieving job status.

    Attributes:
        task_status (JobStatus): Current status of the task.
        file_name (Optional[str]): Name of the file uploaded to remote storage, if present
        error_message (Optional[str]): Error message if task encountered an error.
    """

    task_status: JobStatus
    file_name: Optional[str] = None
    error_message: Optional[str] = None

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from enum import Enum
from pydantic import BaseModel
from app.models.request import RequestParameters
from typing import Optional


class JobStatus(str, Enum):
    """
    Enumeration of possible job status states.

    Attributes:
        RUNNING (str): Indicates the index build is currently in progress
        FAILED (str): Indicates the index build failed to complete
        COMPLETED (str): Indicates the index build completed successfully
    """

    RUNNING = "RUNNING_INDEX_BUILD"
    FAILED = "FAILED_INDEX_BUILD"
    COMPLETED = "COMPLETED_INDEX_BUILD"


class Job(BaseModel):
    """
    Represents a job in the remote vector index building system.

    This class tracks the state and parameters of an index building job,
    including its status, associated request parameters, and any error information.

    Attributes:
        id (str): Unique identifier for the job
        status (JobStatus): Current status of the job
        request_parameters (RequestParameters): Parameters specified in the original request
        file_name (Optional[str]): Name of the output file, if any
        error_message (Optional[str]): Error message if the job failed
    """

    id: str
    status: JobStatus
    request_parameters: RequestParameters
    file_name: Optional[str] = None
    error_message: Optional[str] = None

    def compare_request_parameters(self, other: RequestParameters) -> bool:
        """
        Compare this job's request parameters with another set of parameters.

        This method is used to check if a new request matches an existing job's parameters.

        Args:
            other (RequestParameters): The request parameters to compare against

        Returns:
            bool: True if the parameters match, False otherwise
        """
        return self.request_parameters == other

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import logging
import time
from typing import Dict, Any
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout

from app.models.job import JobStatus
from app.schemas.api import CreateJobResponse, GetStatusResponse


class RemoteVectorAPIClient:
    """
    Test Client to mock requests to Remote Vector API Server
    """

    def __init__(
        self, base_url: str = "http://localhost:1025", http_request_timeout: int = 30
    ):
        """
        Initialize the Remote Vector API Client

        Args:
            base_url: Base URL for the API server
            http_request_timeout: Timeout in seconds for HTTP requests
        """
        self.base_url = base_url
        self.http_request_timeout = http_request_timeout

    def wait_for_job_completion(
        self, job_id: str, status_request_timeout: int = 1200, interval: int = 10
    ) -> GetStatusResponse:
        """
        Method to Poll Job Status sent to the Index Builder workflow executor

        Args:
            job_id (str): Required field. Job Id to request status for
            status_request_timeout (int): Max seconds to Poll for status
            interval (int): Interval in seconds between consecutive requests
        Returns:
            Dict containing job status information

        Raises:
            TimeoutError: If job doesn't complete within timeout period
            RuntimeError: If job fails or has unknown status
        """
        start_time = time.time()

        logger = logging.getLogger(__name__)

        while True:
            if time.time() - start_time > status_request_timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {status_request_timeout} seconds"
                )

            status_response = self.get_job_status(job_id)

            task_status = status_response.task_status

            if task_status == JobStatus.COMPLETED:
                logger.info(f"Job {job_id} completed successfully")
                return status_response
            elif task_status == JobStatus.FAILED:
                raise RuntimeError(
                    f"Job {job_id} failed: {status_response.error_message}"
                )
            elif task_status == JobStatus.RUNNING:
                logger.debug(
                    f"Job {job_id} still running , waiting {interval} seconds..."
                )
                time.sleep(interval)
            else:
                raise RuntimeError(f"Unknown job status: {task_status}")

    def get_job_status(self, job_id: str) -> GetStatusResponse:
        """
        Method to make an Index Build Job Status request to the _status API
        Args:
            job_id (str): Required field. Job Id to request status for

        Raises:
            APIError: If the status request fails
        """

        logger = logging.getLogger(__name__)
        try:
            response = self._make_request(method="GET", endpoint=f"/_status/{job_id}")
            return GetStatusResponse.model_validate(response.json())
        except APIError:
            logger.error(f"Failed to get status for job {job_id}")
            raise

    def build_index(self, index_build_parameters: Dict[str, Any]) -> str:
        """
        Submits an Index Build Job via HTTP request to the _build API
        Args:
            index_build_parameters: Parameters for the index build job

        Returns:
            str: Job ID of the created build job

        Raises:
            APIError: If the status request fails
        """
        logger = logging.getLogger(__name__)
        try:
            response = self._make_request(
                method="POST", endpoint="/_build", json=index_build_parameters
            )
            create_response = CreateJobResponse.model_validate(response.json())
            return create_response.job_id
        except APIError:
            logger.error("Failed to create index build job")
            raise

    def _make_request(
        self, method: str, endpoint: str, max_retries: int = 1, **kwargs
    ) -> requests.Response:
        """
        Generic method to make HTTP request with error handling and retries
        Args:
            method (str): HTTP method
            endpoint (str): HTTP request endpoint
            max_retries: Max retries with exponential backoff to loop for HTTP failed request errors

        Returns:
            requests.Response object

        Raises:
            APIError: If the request fails after all retries
        """

        logger = logging.getLogger(__name__)
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        for attempt in range(max_retries + 1):
            try:
                response = requests.request(
                    method=method, url=url, timeout=self.http_request_timeout, **kwargs
                )
                response.raise_for_status()
                return response
            except HTTPError as e:
                error_detail = None
                try:
                    error_detail = e.response.json()
                except:
                    error_detail = e.response.text

                logger.error(
                    f"HTTP {e.response.status_code} Error: "
                    f"URL: {url}, "
                    f"Method: {method}, "
                    f"Detail: {error_detail}"
                )
                if attempt == max_retries:
                    raise APIError(f"API request failed: {str(e)}") from e
            except ConnectionError as e:
                logger.error(f"Connection failed to {url}: {str(e)}")
                if attempt == max_retries:
                    raise APIError("Could not connect to API server") from e
            except Timeout as e:
                logger.error(f"Request timed out to {url}: {str(e)}")
                if attempt == max_retries:
                    raise APIError("API request timed out") from e
            except Exception as e:
                logger.error(f"Unexpected error making request to {url}: {str(e)}")
                if attempt == max_retries:
                    raise APIError("Unexpected error during API request") from e

            retry_delay = 2 ** (attempt + 1)
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

        raise APIError("All retries exhausted with no specific error captured")


class APIError(Exception):
    """Base exception for API errors"""

    pass

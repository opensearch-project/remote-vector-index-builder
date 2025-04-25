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

class RemoteVectorAPIClient:
    def __init__(self, base_url: str = "http://localhost:1025", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout


    def wait_for_job_completion(
        self, 
        job_id: str, 
        timeout: int = 1200, 
        interval: int = 10
    ) -> Dict[str, Any]:
        """Wait for job to complete with timeout"""
        start_time = time.time()
        attempts = 0

        logger = logging.getLogger(__name__)
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout} seconds"
                )
            
            try:
                attempts += 1
                status_response = self.get_job_status(job_id)

                task_status = status_response.get("task_status")
                
                if task_status == JobStatus.COMPLETED:
                    logger.info(f"Job {job_id} completed successfully")
                    return status_response
                elif task_status == JobStatus.FAILED:
                    raise RuntimeError(
                        f"Job {job_id} failed: {status_response.get('error_message')}"
                    )
                elif task_status == JobStatus.RUNNING:
                    logger.debug(
                        f"Job {job_id} still running (attempt {attempts}), "
                        f"waiting {interval} seconds..."
                    )
                    time.sleep(interval)
                else:
                    raise RuntimeError(f"Unknown job status: {task_status}")
                
            except APIError as e:
                if time.time() - start_time > timeout:
                    raise
                logger.warning(
                    f"Error checking job status (attempt {attempts}): {str(e)}, "
                    f"retrying in {interval} seconds..."
                )
                time.sleep(interval)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job"""
        logger = logging.getLogger(__name__)
        try:
            response = self._make_request(
                method="GET",
                endpoint=f"/_status/{job_id}"
            )
            return response.json()
        except APIError:
            logger.error(f"Failed to get status for job {job_id}")
            raise

    def build_index(self, index_build_parameters: Dict[str, Any]) -> str:
        """Create a new index build job"""
        logger = logging.getLogger(__name__)
        try:
            response = self._make_request(
                method="POST",
                endpoint="/_build",
                json=index_build_parameters
            )
            return response.json()["job_id"]
        except APIError:
            logger.error("Failed to create index build job")
            raise
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling"""

        logger = logging.getLogger(__name__)
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
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
            raise APIError(f"API request failed: {str(e)}") from e
        except ConnectionError as e:
            logger.error(f"Connection failed to {url}: {str(e)}")
            raise APIError("Could not connect to API server") from e
        except Timeout as e:
            logger.error(f"Request timed out to {url}: {str(e)}")
            raise APIError("API request timed out") from e
        except Exception as e:
            logger.error(f"Unexpected error making request to {url}: {str(e)}")
            raise APIError("Unexpected error during API request") from e

class APIError(Exception):
    """Base exception for API errors"""
    pass

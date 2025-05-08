# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from app.models.job import Job


class RequestStore(ABC):
    """
    Abstract base class defining the interface for job request storage operations.

    This class provides the contract for implementing storage operations for job requests,
    including adding, retrieving, updating, and deleting jobs, as well as cleaning up
    expired entries.
    """

    @abstractmethod
    def add(self, job_id: str, job: Job) -> bool:
        """
        Add a job to the store.

        Args:
            job_id (str): Unique identifier for the job
            job (Job): Job object containing the job details

        Returns:
            bool: True if addition was successful, False otherwise
        """
        """Add a job to the store"""
        return True

    @abstractmethod
    def get(self, job_id: str) -> Optional[Job]:
        """
        Retrieve a job from the store.

        Args:
            job_id (str): Unique identifier of the job to retrieve

        Returns:
            Optional[Job]: The job if found, None otherwise
        """
        pass

    @abstractmethod
    def update(self, job_id: str, data: Dict[str, Any]) -> bool:
        """
        Update a job in the store.

        Args:
            job_id (str): Unique identifier of the job to update
            data (Dict[str, Any]): Dictionary containing the fields to update

        Returns:
            bool: True if update was successful, False otherwise
        """
        return True

    @abstractmethod
    def delete(self, job_id: str) -> bool:
        """
        Delete a job from the store.

        Args:
            job_id (str): Unique identifier of the job to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        return True

    @abstractmethod
    def get_jobs(self) -> Dict[str, Job]:
        """
        Retrieve all jobs from the store.

        Returns:
            Dict[str, Job]: A dictionary of all jobs, with job IDs as keys and Job objects as values
        """
        pass

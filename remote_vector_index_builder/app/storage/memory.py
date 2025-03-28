# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from typing import Dict, Optional, Any
from datetime import datetime, timedelta, timezone
import threading
import time

from app.models.job import Job
from app.storage.base import RequestStore
from app.base.config import Settings

import logging

logger = logging.getLogger(__name__)


class InMemoryRequestStore(RequestStore):
    """
    In-memory implementation of the RequestStore interface.

    This class provides a thread-safe, in-memory storage solution for job requests
    with configurable maximum size and TTL-based cleanup capabilities.

    Attributes:
        _store (Dict[str, tuple[Job, datetime]]): Internal dictionary storing jobs and their timestamps
        _lock (threading.Lock): Thread lock for synchronization
        _max_size (int): Maximum number of jobs that can be stored
        _ttl_seconds (int): Time-to-live in seconds for stored jobs
    """

    def __init__(self, settings: Settings):
        """
        Initialize the in-memory request store.

        Args:
            settings (Settings): Configuration settings containing request_store_max_size
                and request_store_ttl_seconds
        """
        self._store: Dict[str, tuple[Job, datetime]] = {}
        self._lock = threading.Lock()
        self._max_size = settings.request_store_max_size
        self._ttl_seconds = settings.request_store_ttl_seconds

        # Start cleanup thread, TTL seconds is not None
        if self._ttl_seconds is not None:
            logger.info(
                f"Starting cleanup thread for request store with TTL {self._ttl_seconds} seconds"
            )
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop, daemon=True
            )
            self._cleanup_thread.start()

    def add(self, job_id: str, job: Job) -> bool:
        """
        Add a new job to the store with current timestamp.

        Args:
            job_id (str): Unique identifier for the job
            job (Job): Job object to store

        Returns:
            bool: True if job was added successfully, False if store is at capacity
        """
        with self._lock:
            if len(self._store) >= self._max_size:
                return False
            else:
                self._store[job_id] = (job, datetime.now(timezone.utc))
                return True

    def get(self, job_id: str) -> Optional[Job]:
        """
        Retrieve a job from the store.

        Args:
            job_id (str): Unique identifier of the job to retrieve

        Returns:
            Optional[Job]: The job if found and not expired, None otherwise
        """
        with self._lock:
            if job_id in self._store:
                job, timestamp = self._store[job_id]
                if self._ttl_seconds is None or (
                    self._ttl_seconds
                    and datetime.now(timezone.utc) - timestamp
                    < timedelta(seconds=self._ttl_seconds)
                ):
                    return job
                else:
                    del self._store[job_id]
            return None

    def update(self, job_id: str, data: Dict[str, Any]) -> bool:
        """
        Update an existing job in the store with new data.

        Args:
            job_id (str): Unique identifier of the job to update
            data (Dict[str, Any]): Dictionary containing the fields and values to update

        Returns:
            bool: True if job was found and updated successfully, False if job not found
        """
        with self._lock:
            if job_id not in self._store:
                return False

            job, timestamp = self._store[job_id]
            for key, value in data.items():
                setattr(job, key, value)
            self._store[job_id] = (job, timestamp)
            return True

    def delete(self, job_id: str) -> bool:
        """
        Delete a job from the store.

        Args:
            job_id (str): Unique identifier of the job to delete

        Returns:
            bool: True if job was found and deleted successfully, False if job not found
        """
        with self._lock:
            if job_id in self._store:
                del self._store[job_id]
                return True
            return False

    def cleanup_expired(self) -> None:
        """
        Remove all expired entries from the store based on TTL.
        Thread-safe implementation using the store's lock.
        """
        with self._lock:
            current_time = datetime.now(timezone.utc)
            expiration_threshold = current_time - timedelta(seconds=self._ttl_seconds)

            self._store = {
                job_id: data
                for job_id, data in self._store.items()
                if data[1] > expiration_threshold
            }

    def _cleanup_loop(self) -> None:
        """
        Background thread that periodically removes expired entries from the store.
        Runs continuously while the store is active.
        """
        while True:
            time.sleep(5)  # Run cleanup every 5 seconds
            self.cleanup_expired()

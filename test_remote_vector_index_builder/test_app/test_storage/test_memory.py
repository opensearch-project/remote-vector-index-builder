# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest
import time
from unittest.mock import Mock, patch

from app.storage.memory import InMemoryRequestStore
from app.base.config import Settings
from app.models.job import JobStatus


@pytest.fixture
def settings():
    return Settings(request_store_max_size=2, request_store_ttl_seconds=1)


@pytest.fixture
def settings_no_ttl():
    return Settings(request_store_max_size=2, request_store_ttl_seconds=None)


@pytest.fixture
def sample_job():
    job = Mock()
    job.status = JobStatus.COMPLETED
    return job


def test_init_with_ttl(settings):
    store = InMemoryRequestStore(settings)
    assert store._max_size == 2
    assert store._ttl_seconds == 1
    assert len(store._store) == 0


def test_init_without_ttl(settings_no_ttl):
    store = InMemoryRequestStore(settings_no_ttl)
    assert store._max_size == 2
    assert store._ttl_seconds is None
    assert len(store._store) == 0


def test_add_and_get(settings, sample_job):
    store = InMemoryRequestStore(settings)
    assert store.add("job1", sample_job) is True
    retrieved_job = store.get("job1")
    assert retrieved_job == sample_job


def test_add_max_size(settings, sample_job):
    store = InMemoryRequestStore(settings)
    assert store.add("job1", sample_job) is True
    assert store.add("job2", sample_job) is True
    assert store.add("job3", sample_job) is False  # Should fail, store is full


def test_get_nonexistent(settings):
    store = InMemoryRequestStore(settings)
    assert store.get("nonexistent") is None


def test_get_expired(settings, sample_job):
    store = InMemoryRequestStore(settings)
    store.add("job1", sample_job)
    time.sleep(1.1)  # Wait for expiration
    assert store.get("job1") is None


def test_update(settings, sample_job):
    store = InMemoryRequestStore(settings)
    store.add("job1", sample_job)

    update_data = {"status": JobStatus.COMPLETED}
    assert store.update("job1", update_data) is True

    updated_job = store.get("job1")
    assert updated_job.status == JobStatus.COMPLETED


def test_update_nonexistent(settings):
    store = InMemoryRequestStore(settings)
    assert store.update("nonexistent", {"status": JobStatus.COMPLETED}) is False


def test_delete(settings, sample_job):
    store = InMemoryRequestStore(settings)
    store.add("job1", sample_job)
    assert store.delete("job1") is True
    assert store.get("job1") is None


def test_delete_nonexistent(settings):
    store = InMemoryRequestStore(settings)
    assert store.delete("nonexistent") is False


def test_cleanup_expired(settings, sample_job):
    store = InMemoryRequestStore(settings)
    store.add("job1", sample_job)
    time.sleep(1.1)  # Wait for expiration
    store.cleanup_expired()
    assert store.get("job1") is None


def test_do_not_clean_up_in_progress_job(settings, sample_job):
    store = InMemoryRequestStore(settings)
    sample_job.status = JobStatus.RUNNING
    store.add("job1", sample_job)
    store.cleanup_expired()
    assert store.get("job1") == sample_job


@patch("time.sleep")
def test_cleanup_loop(mock_sleep, settings):
    """Test cleanup loop"""
    # Create a store but patch the thread creation
    with patch("threading.Thread.start"):
        store = InMemoryRequestStore(settings)

    # Now test the cleanup loop directly
    mock_sleep.side_effect = [None, Exception("Stop loop")]

    with pytest.raises(Exception):
        store._cleanup_loop()

    mock_sleep.assert_called_with(5)


def test_get_no_ttl(settings_no_ttl, sample_job):
    store = InMemoryRequestStore(settings_no_ttl)
    store.add("job1", sample_job)
    time.sleep(1.1)  # Even after waiting, job should still be there
    assert store.get("job1") == sample_job


def test_get_jobs(settings, sample_job):
    store = InMemoryRequestStore(settings)
    store.add("job1", sample_job)
    store.add("job2", sample_job)
    jobs = store.get_jobs()
    assert len(jobs) == 2
    assert jobs["job1"] == sample_job
    assert jobs["job2"] == sample_job

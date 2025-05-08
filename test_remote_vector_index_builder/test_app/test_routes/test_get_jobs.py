# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock
from app.routes import get_jobs


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(get_jobs.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_job_service():
    return Mock()


class JobMock:
    """Mock job class for testing"""

    def __init__(self, job_id, status, file_name=None, error_message=None):
        self.job_id = job_id
        self.status = status
        self.file_name = file_name
        self.error_message = error_message


def test_get_jobs_basic(client, mock_job_service):
    """Test basic retrieval of jobs"""
    # Arrange
    app = client.app
    app.state.job_service = mock_job_service
    job_id = "test-job-1"
    mock_job = JobMock(job_id, "RUNNING_INDEX_BUILD", "test_file.index")
    mock_job_service.get_jobs.return_value = {job_id: mock_job}

    response = client.get("/_jobs")

    assert response.status_code == 200
    response_data = response.json()
    response_data = json.loads(response_data)
    assert isinstance(response_data, dict)
    assert len(response_data) == 1
    assert job_id in response_data
    assert response_data[job_id]["job_id"] == job_id
    assert response_data[job_id]["status"] == "RUNNING_INDEX_BUILD"
    assert response_data[job_id]["file_name"] == "test_file.index"
    mock_job_service.get_jobs.assert_called_once()


def test_get_multiple_jobs(client, mock_job_service):
    """Test retrieval of multiple jobs"""
    # Arrange
    app = client.app
    app.state.job_service = mock_job_service

    job1_id = "test-job-1"
    job2_id = "test-job-2"
    job3_id = "test-job-3"

    job1 = JobMock(job1_id, "RUNNING_INDEX_BUILD", "file1.index")
    job2 = JobMock(job2_id, "COMPLETED", "file2.index")
    job3 = JobMock(job3_id, "FAILED", None, "Error occurred")

    mock_job_service.get_jobs.return_value = {
        job1_id: job1,
        job2_id: job2,
        job3_id: job3,
    }

    response = client.get("/_jobs")

    assert response.status_code == 200
    response_data = response.json()
    response_data = json.loads(response_data)
    assert isinstance(response_data, dict)
    assert len(response_data) == 3
    assert job1_id in response_data
    assert job2_id in response_data
    assert job3_id in response_data
    assert response_data[job1_id]["job_id"] == job1_id
    assert response_data[job2_id]["job_id"] == job2_id
    assert response_data[job3_id]["job_id"] == job3_id
    mock_job_service.get_jobs.assert_called_once()


def test_get_empty_jobs_list(client, mock_job_service):
    """Test retrieval when there are no jobs"""

    app = client.app
    app.state.job_service = mock_job_service
    mock_job_service.get_jobs.return_value = {}

    response = client.get("/_jobs")

    assert response.status_code == 200
    response_data = response.json()
    response_data = json.loads(response_data)
    assert isinstance(response_data, dict)
    assert len(response_data) == 0
    mock_job_service.get_jobs.assert_called_once()

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock
from app.routes import status
from app.models.job import JobStatus, Job


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(status.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_job_service():
    return Mock()


@pytest.fixture
def mock_job():
    job = Mock(spec=Job)
    job.status = JobStatus.RUNNING
    return job


def test_get_status_basic(client, mock_job_service, mock_job):
    """Test basic status retrieval with only status field"""
    # Arrange
    app = client.app
    app.state.job_service = mock_job_service
    mock_job_service.get_job.return_value = mock_job
    job_id = "test_job_123"

    # Act
    response = client.get(f"/_status/{job_id}")

    # Assert
    assert response.status_code == 200
    assert response.json()["task_status"] == mock_job.status
    assert response.json()["file_name"] is None
    assert response.json()["error_message"] is None
    mock_job_service.get_job.assert_called_once_with(job_id)


def test_get_status_with_filename(client, mock_job_service, mock_job):
    """Test status retrieval with filename"""
    # Arrange
    app = client.app
    app.state.job_service = mock_job_service
    mock_job.file_name = "test_file.index"
    mock_job_service.get_job.return_value = mock_job
    job_id = "test_job_123"

    # Act
    response = client.get(f"/_status/{job_id}")

    # Assert
    assert response.status_code == 200
    assert response.json()["task_status"] == mock_job.status
    assert response.json()["file_name"] == mock_job.file_name
    assert response.json()["error_message"] is None


def test_get_status_with_error(client, mock_job_service, mock_job):
    """Test status retrieval with error message"""
    # Arrange
    app = client.app
    app.state.job_service = mock_job_service
    mock_job.status = JobStatus.FAILED
    mock_job.error_message = "Test error message"
    mock_job_service.get_job.return_value = mock_job
    job_id = "test_job_123"

    # Act
    response = client.get(f"/_status/{job_id}")

    # Assert
    assert response.status_code == 200
    assert response.json()["task_status"] == mock_job.status
    assert response.json()["file_name"] is None
    assert response.json()["error_message"] == mock_job.error_message


def test_get_status_with_all_fields(client, mock_job_service, mock_job):
    """Test status retrieval with all optional fields"""
    # Arrange
    app = client.app
    app.state.job_service = mock_job_service
    mock_job.status = JobStatus.COMPLETED
    mock_job.file_name = "test_file.index"
    mock_job.error_message = "Warning: something happened"
    mock_job_service.get_job.return_value = mock_job
    job_id = "test_job_123"

    # Act
    response = client.get(f"/_status/{job_id}")

    # Assert
    assert response.status_code == 200
    assert response.json()["task_status"] == mock_job.status
    assert response.json()["file_name"] == mock_job.file_name
    assert response.json()["error_message"] == mock_job.error_message


def test_get_status_job_not_found(client, mock_job_service):
    """Test status retrieval for non-existent job"""
    # Arrange
    app = client.app
    app.state.job_service = mock_job_service
    mock_job_service.get_job.return_value = None
    job_id = "nonexistent_job"

    # Act
    response = client.get(f"/_status/{job_id}")

    # Assert
    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}

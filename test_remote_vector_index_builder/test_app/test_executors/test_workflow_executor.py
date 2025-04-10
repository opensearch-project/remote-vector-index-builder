# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
import pytest
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
from app.models.workflow import BuildWorkflow
from app.models.job import JobStatus
from app.executors.workflow_executor import WorkflowExecutor


@pytest.fixture
def mock_request_store():
    store = Mock()
    store.get.return_value = True  # Default to existing job
    return store


@pytest.fixture
def mock_resource_manager():
    return Mock()


@pytest.fixture
def mock_build_index_fn():
    return Mock(return_value=(True, "test/path", None))


@pytest.fixture
def workflow_executor(mock_request_store, mock_resource_manager, mock_build_index_fn):
    return WorkflowExecutor(
        max_workers=2,
        request_store=mock_request_store,
        resource_manager=mock_resource_manager,
        build_index_fn=mock_build_index_fn,
    )


@pytest.fixture
def sample_workflow():
    workflow = Mock(spec=BuildWorkflow)
    workflow.job_id = "test_job_123"
    workflow.gpu_memory_required = 1000
    workflow.cpu_memory_required = 2000
    return workflow


def test_workflow_executor_initialization(
    mock_request_store, mock_resource_manager, mock_build_index_fn
):
    """Test proper initialization of WorkflowExecutor"""
    executor = WorkflowExecutor(
        max_workers=2,
        request_store=mock_request_store,
        resource_manager=mock_resource_manager,
        build_index_fn=mock_build_index_fn,
    )

    assert isinstance(executor._executor, ThreadPoolExecutor)
    assert executor._request_store == mock_request_store
    assert executor._resource_manager == mock_resource_manager
    assert executor._build_index_fn == mock_build_index_fn


def test_successful_workflow_execution(
    workflow_executor,
    sample_workflow,
    mock_request_store,
    mock_resource_manager,
    mock_build_index_fn,
):
    """Test successful execution of a workflow"""
    mock_build_index_fn.return_value = (True, "/path/to/index", None)

    # Execute workflow
    workflow_executor.submit_workflow(sample_workflow)
    # Wait for execution to complete
    workflow_executor._executor.shutdown(wait=True)

    # Verify build function was called
    mock_build_index_fn.assert_called_once_with(sample_workflow)

    # Verify status update
    mock_request_store.update.assert_called_with(
        sample_workflow.job_id,
        {
            "status": JobStatus.COMPLETED,
            "file_name": "/path/to/index",
            "error_message": None,
        },
    )

    # Verify resource release
    mock_resource_manager.release.assert_called_with(
        sample_workflow.gpu_memory_required, sample_workflow.cpu_memory_required
    )


def test_failed_workflow_execution(
    workflow_executor,
    sample_workflow,
    mock_request_store,
    mock_resource_manager,
    mock_build_index_fn,
):
    """Test workflow execution with failure"""
    error_message = "Build failed"
    mock_build_index_fn.return_value = (False, None, error_message)

    workflow_executor.submit_workflow(sample_workflow)
    workflow_executor._executor.shutdown(wait=True)

    mock_request_store.update.assert_called_with(
        sample_workflow.job_id,
        {"status": JobStatus.FAILED, "file_name": None, "error_message": error_message},
    )


def test_deleted_job_before_execution(
    workflow_executor,
    sample_workflow,
    mock_request_store,
    mock_resource_manager,
    mock_build_index_fn,
):
    """Test handling of job that was deleted before execution"""
    mock_request_store.get.return_value = False

    workflow_executor.submit_workflow(sample_workflow)
    workflow_executor._executor.shutdown(wait=True)

    mock_build_index_fn.assert_not_called()
    mock_request_store.update.assert_not_called()


def test_deleted_job_during_execution(
    workflow_executor,
    sample_workflow,
    mock_request_store,
    mock_resource_manager,
    mock_build_index_fn,
):
    """Test handling of job that was deleted during execution"""
    # First get returns True, second returns False
    mock_request_store.get.side_effect = [True, False]

    workflow_executor.submit_workflow(sample_workflow)
    workflow_executor._executor.shutdown(wait=True)

    mock_build_index_fn.assert_called_once()
    mock_request_store.update.assert_not_called()


def test_exception_during_execution(
    workflow_executor,
    sample_workflow,
    mock_request_store,
    mock_resource_manager,
    mock_build_index_fn,
):
    """Test handling of exceptions during execution"""
    error_message = "Unexpected error"
    mock_build_index_fn.side_effect = Exception(error_message)

    workflow_executor.submit_workflow(sample_workflow)
    workflow_executor._executor.shutdown(wait=True)

    # mock_request_store.update.assert_called_with(
    #     sample_workflow.job_id,
    #     {"status": JobStatus.FAILED, "error_message": error_message}
    # )
    mock_request_store.update.assert_called_once()

    # Verify resources are released even after exception
    mock_resource_manager.release.assert_called_with(
        sample_workflow.gpu_memory_required, sample_workflow.cpu_memory_required
    )


def test_shutdown(workflow_executor):
    """Test executor shutdown"""
    with patch.object(workflow_executor._executor, "shutdown") as mock_shutdown:
        workflow_executor.shutdown()
        mock_shutdown.assert_called_once_with(wait=True)

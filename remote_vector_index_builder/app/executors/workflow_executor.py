# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Optional, Callable
from app.models.workflow import BuildWorkflow
from app.base.resources import ResourceManager
from app.storage.base import RequestStore
from app.models.job import JobStatus

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """
    Executes build workflows in a thread pool while managing system resources.

    This class handles the concurrent execution of build workflows, managing thread pools,
    and coordinating resource allocation for index building operations.

    Attributes:
        _executor (ThreadPoolExecutor): Thread pool for executing concurrent workflows
        _request_store (RequestStore): Interface for storing request data
        _resource_manager (ResourceManager): Manager for system resource allocation
        _build_index_fn (Callable): Function that performs the actual index building
    """

    def __init__(
        self,
        max_workers: int,
        request_store: RequestStore,
        resource_manager: ResourceManager,
        build_index_fn: Callable[
            [BuildWorkflow], tuple[bool, Optional[str], Optional[str]]
        ],
    ):
        """
        Initialize the WorkflowExecutor with specified parameters.

        Args:
            max_workers (int): Maximum number of concurrent worker threads
            request_store (RequestStore): Interface for storing request data
            resource_manager (ResourceManager): Manager for system resources
            build_index_fn (Callable): Function that builds the index, returns a tuple of
                (success: bool, error_message: Optional[str], result: Optional[str])
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._request_store = request_store
        self._resource_manager = resource_manager
        self._build_index_fn = build_index_fn

    def submit_workflow(self, workflow: BuildWorkflow) -> None:
        """
        Submit a workflow for execution in the thread pool.

        This method queues the workflow for execution.
        The workflow will be executed asynchronously in a thread pool

        Args:
            workflow (BuildWorkflow): The workflow to be executed

        """

        # Submit the workflow to thread pool
        self._executor.submit(self._execute_workflow, workflow)

    def _execute_workflow(self, workflow: BuildWorkflow) -> None:
        """
        Execute the workflow and handle results.

        This method handles the actual execution of the build workflow, including:
        - Executing the build process
        - Updating the job status in the request store
        - Logging the execution status

        Args:
            workflow (BuildWorkflow): The workflow to execute containing job parameters
                and resource requirements

        Note:
            This method is intended to be run in a separate thread.
        """

        # TODO: Block until memory resource is available, instead of failing immediately
        if not self._resource_manager.allocate(
            workflow.gpu_memory_required, workflow.cpu_memory_required
        ):
            self._request_store.update(
                workflow.job_id,
                {
                    "status": JobStatus.FAILED,
                    "error_message": "Worker has not enough memory available at this time",
                },
            )
            return

        logger.info(
            f"Worker resource status after allocating memory for job id {workflow.job_id}: - "
            f"GPU: {self._resource_manager.get_available_gpu_memory():,} bytes, "
            f"CPU: {self._resource_manager.get_available_cpu_memory():,} bytes"
        )

        try:
            logger.info(f"Starting execution of job {workflow.job_id}")

            success, index_path, msg = self._build_index_fn(workflow)

            # Job may have been deleted by request store TTL, so we need to check if job
            # still exists before updating status.
            if self._request_store.get(workflow.job_id):
                status = JobStatus.COMPLETED if success else JobStatus.FAILED
                self._request_store.update(
                    workflow.job_id,
                    {"status": status, "file_name": index_path, "error_message": msg},
                )

                logger.info(
                    f"Job {workflow.job_id} completed with status: {status}, index path: "
                    f"{index_path}, and error message: {msg}"
                )
            else:
                logger.error(
                    f"[ERROR] Job {workflow.job_id} was deleted during execution"
                )

        except Exception as e:
            logger.error(f"Build process failed for job {workflow.job_id}: {str(e)}")
            self._request_store.update(
                workflow.job_id,
                {"status": JobStatus.FAILED, "error_message": str(e)},
            )
        finally:
            self._resource_manager.release(
                workflow.gpu_memory_required, workflow.cpu_memory_required
            )

    def shutdown(self) -> None:
        """
        Shutdown the executor and wait for all pending tasks to complete.

        This method initiates a graceful shutdown of the thread pool executor.
        It blocks until all pending tasks are completed and releases all resources.
        No new tasks will be accepted after calling this method.

        Note:
            - This is a blocking call that waits for all tasks to finish
            - The executor cannot be reused after shutdown
        """
        self._executor.shutdown(wait=True)

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from typing import Optional
from app.base.exceptions import HashCollisionError, CapacityError
from app.base.resources import ResourceManager
from app.executors.workflow_executor import WorkflowExecutor
from app.models.job import Job, JobStatus
from app.models.request import RequestParameters
from app.models.workflow import BuildWorkflow
from app.utils.hash import generate_job_id
from app.utils.memory import calculate_memory_requirements
from app.utils.request import create_request_parameters
from app.storage.base import RequestStore
from core.common.models import IndexBuildParameters

import logging

logger = logging.getLogger(__name__)


class JobService:
    """
    Service class for managing job operations including creation, validation, and resource management.

    This service handles job lifecycle operations, resource allocation, and workflow execution
    while maintaining state in the request store.

    Attributes:
        request_store (RequestStore): Store for persisting job requests and their states
        workflow_executor (WorkflowExecutor): Executor for handling workflow operations
        total_gpu_memory (float): Total available GPU memory for job allocation, in bytes
        total_cpu_memory (float): Total available CPU memory for job allocation, in bytes
        resource_manager (ResourceManager): Manager for handling system resources
    """

    def __init__(
        self,
        request_store: RequestStore,
        workflow_executor: WorkflowExecutor,
        resource_manager: ResourceManager,
        total_gpu_memory: float,
        total_cpu_memory: float,
    ):
        """
        Initialize the JobService with required dependencies.

        Args:
            request_store (RequestStore): Store for managing job requests
            workflow_executor (WorkflowExecutor): Executor for workflow operations
            resource_manager (ResourceManager): Manager for system resources
            total_gpu_memory (float): Total GPU memory available, in bytes
            total_cpu_memory (float): Total CPU memory available, in bytes
        """
        self.request_store = request_store
        self.workflow_executor = workflow_executor
        self.total_gpu_memory = total_gpu_memory
        self.total_cpu_memory = total_cpu_memory
        self.resource_manager = resource_manager

    def _validate_job_existence(
        self, job_id: str, request_parameters: RequestParameters
    ) -> bool:
        """
        Validate if a job exists and check for hash collisions.

        Args:
            job_id (str): Unique identifier for the job
            request_parameters (RequestParameters): Parameters of the request to validate

        Returns:
            bool: True if job exists with matching parameters, False otherwise

        Raises:
            HashCollisionError: If job exists but with different parameters
        """
        job = self.request_store.get(job_id)
        if job:
            if job.compare_request_parameters(request_parameters):
                return True
            raise HashCollisionError(f"Hash collision detected for job_id: {job_id}")
        return False

    def _add_to_request_store(
        self, job_id: str, request_parameters: RequestParameters
    ) -> None:
        """
        Add a new job to the request store with initial running status.

        Args:
            job_id (str): Unique identifier for the job
            request_parameters (RequestParameters): Parameters of the job request

        Raises:
            CapacityError: If the job cannot be added to the request store
        """
        result = self.request_store.add(
            job_id,
            Job(
                id=job_id,
                status=JobStatus.RUNNING,
                request_parameters=request_parameters,
            ),
        )

        if not result:
            raise CapacityError("Could not add item to request store")

    def _create_workflow(
        self,
        job_id: str,
        gpu_mem: float,
        cpu_mem: float,
        index_build_parameters: IndexBuildParameters,
    ) -> BuildWorkflow:
        """
        Create a new build workflow with the specified parameters.

        Args:
            job_id (str): Unique identifier for the job
            gpu_mem (float): Required GPU memory for the job, in bytes
            cpu_mem (float): Required CPU memory for the job, in bytes
            index_build_parameters (IndexBuildParameters): Parameters for building the index

        Returns:
            BuildWorkflow: Configured workflow instance ready for execution
        Raises:
            CapacityError: If the service does not have enough GPU or CPU memory to process the job
        """
        workflow = BuildWorkflow(
            job_id=job_id,
            gpu_memory_required=gpu_mem,
            cpu_memory_required=cpu_mem,
            index_build_parameters=index_build_parameters,
        )

        # Allocate resources
        allocation_success = self.resource_manager.allocate(
            workflow.gpu_memory_required, workflow.cpu_memory_required
        )

        if not allocation_success:
            self.request_store.delete(job_id)
            raise CapacityError(
                f"Insufficient available resources to process workflow"
            )

        return workflow

    def create_job(self, index_build_parameters: IndexBuildParameters) -> str:
        """
        Creates and initiates a new index building job.

        This method handles the complete job creation workflow including:
        - Generating and validating job ID
        - Checking for existing jobs
        - Calculating required resources
        - Creating and submitting the workflow

        Args:
            index_build_parameters (IndexBuildParameters): Parameters for building the index,
                including dimensions, document count, and algorithm settings

        Returns:
            str: Unique job identifier for the created job

        Raises:
            CapacityError: If there are insufficient resources to process the job
            HashCollisionError: If a job exists with same ID but different parameters
        """
        # Create parameters and validate job
        request_parameters = create_request_parameters(index_build_parameters)
        job_id = generate_job_id(request_parameters)
        job_exists = self._validate_job_existence(job_id, request_parameters)
        if job_exists:
            logger.info(f"Job with id {job_id} already exists")
            return job_id

        self._add_to_request_store(job_id, request_parameters)
        logger.info(f"Added job to request store with job id: {job_id}")

        gpu_mem, cpu_mem = calculate_memory_requirements(index_build_parameters)

        logger.info(
            f"Job id requirements: GPU memory: {gpu_mem}, CPU memory: {cpu_mem}"
        )

        workflow = self._create_workflow(
            job_id, gpu_mem, cpu_mem, index_build_parameters
        )

        logger.info(
            f"Worker resource status for job id {job_id}: - "
            f"GPU: {self.resource_manager.get_available_gpu_memory():,} bytes, "
            f"CPU: {self.resource_manager.get_available_cpu_memory():,} bytes"
        )

        self.workflow_executor.submit_workflow(workflow)
        logger.info(f"Successfully created workflow with job id: {job_id}")

        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Retrieves a job by its unique identifier.

        Args:
            job_id (str): Unique identifier of the job to retrieve

        Returns:
            Optional[Job]: The job object if found, None otherwise
        """
        return self.request_store.get(job_id)

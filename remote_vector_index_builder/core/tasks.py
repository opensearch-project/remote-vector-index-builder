# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
core.tasks
~~~~~~~~~~~~~~~~~


This module contains the core tasks for building a GPU-accelerated vector index.

The tasks execute in the following sequence:
1. create_vectors_dataset: Downloads and parses vector data and document IDs from remote storage
2. build_index: Constructs the vector index on GPU using the downloaded data
3. upload_index: Uploads the built index back to remote storage

The run_tasks function orchestrates these tasks and returns a TaskResult object containing:
- file_name: The file name of the uploaded index
- error: Any error message if the process failed

Consumers of this module can either import and call the run_tasks function, or import and call each
individual task in sequence.

One reason to call each task in sequence is to persist the vectors to disk,
before building the index. The default run_tasks function does not do this, for simplicity.

One reason to call the default run_tasks function is that it optimizes the memory usage, by cleaning
up the vectors after the index is built in memory

"""
import logging
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from timeit import default_timer as timer
import traceback
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager


from core.common.models import (
    IndexBuildParameters,
    IndexSerializationMode,
    VectorsDataset,
)
from core.index_builder.faiss.faiss_index_build_service import FaissIndexBuildService
from core.object_store.object_store_factory import ObjectStoreFactory

from remote_vector_index_builder.core.object_store.object_store import ObjectStore

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Represents the result of an index building task.

    This class encapsulates the outcome of an index building operation, containing either
    the file name of the index in remote storage, or an error message if the operation failed.

    Note: The file name does not include the full path of the index in remote storage. It is just the
    base name

    Attributes:
        file_name (Optional[str]): The file name of the index in remote storage.
            None if the operation failed.
        error (Optional[str]): Error message if the operation failed. None if the operation
            was successful.
    """

    file_name: Optional[str] = None
    error: Optional[str] = None


def run_tasks(
    index_build_params: IndexBuildParameters,
    object_store_config: Optional[Dict[str, Any]] = None,
    index_serialization_mode: IndexSerializationMode = IndexSerializationMode.DISK,
) -> TaskResult:
    """Execute the index building tasks using the provided parameters.

    This function orchestrates the index building process by:
    1. Creating a temporary directory for processing
    2. Setting up byte buffers for vectors and document IDs
    3. Downloading vector and document ID data from remote storage
    4. Building the index, then storing the index in memory or disk
    5. Uplaoding index to remote storage

    Args:
        index_build_params (IndexBuildParameters): Parameters for building the index,
            including vector path and other configuration settings.
        object_store_config (Dict[str, Any], optional): Configuration settings for the
            object store. Defaults to None, in which case an empty dictionary is used.
        index_serialization_mode (IndexSerializationMode): The storage location for
            the constructed vector index. Defaults to disk

    Returns:
        TaskResult: An object containing either:
            - file_name: The name of the successfully created index file in remote storage
            - error: An error message if the operation failed

    """
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        index_storage_context(
            index_serialization_mode,
            temp_dir,
            index_build_params.vector_path,
        ) as index_storage,
    ):
        if object_store_config is None:
            object_store_config = {}

        vector_buffer = BytesIO()
        doc_id_buffer = BytesIO()
        vectors_dataset = None
        try:
            logger.debug(
                f"Starting task execution for vector path: {index_build_params.vector_path}"
            )

            object_store = ObjectStoreFactory.create_object_store(
                index_build_params, object_store_config
            )

            logger.debug(
                f"Downloading vector and doc id blobs for vector path: {index_build_params.vector_path}"
            )
            t1 = timer()
            vectors_dataset = create_vectors_dataset(
                index_build_params=index_build_params,
                object_store=object_store,
                vector_bytes_buffer=vector_buffer,
                doc_id_bytes_buffer=doc_id_buffer,
            )

            t2 = timer()
            download_time = t2 - t1
            logger.debug(
                f"Vector download time for vector path {index_build_params.vector_path}: {download_time:.2f} seconds"
            )

            logger.debug(
                f"Building GPU index for vector path: {index_build_params.vector_path}"
            )

            # First build the faiss index

            t1 = timer()
            faiss_service = FaissIndexBuildService()
            cpu_index = faiss_service.build_index(
                index_build_params,
                vectors_dataset,
            )

            # now that cpu index is in memory, free the vectors to optimize memory usage

            # free the memory view to the vector and doc id buffer
            vectors_dataset.free_vectors_space()

            # close the buffers
            vector_buffer.close()
            doc_id_buffer.close()

            # finally, write the index to index_storage
            faiss_service.write_cpu_index(
                cpu_index, index_build_params, index_serialization_mode, index_storage
            )

            t2 = timer()
            build_time = t2 - t1
            logger.debug(
                f"Total index build time for path {index_build_params.vector_path}: {build_time:.2f} seconds"
            )

            logger.debug(
                f"Uploading index for vector path: {index_build_params.vector_path}"
            )

            t1 = timer()
            remote_path = upload_index(
                index_build_params=index_build_params,
                object_store=object_store,
                data=index_storage,
            )
            t2 = timer()
            upload_time = t2 - t1
            logger.debug(
                f"Total upload time for path {index_build_params.vector_path}: {upload_time:.2f} seconds"
            )

            logger.debug(
                f"Ending task execution for vector path: {index_build_params.vector_path}"
            )
            return TaskResult(file_name=os.path.basename(remote_path))
        except Exception as e:
            logger.error(
                f"Error running tasks for vector path {index_build_params.vector_path}: {e}. "
                f"Index build parameters: {index_build_params}. "
                f"Traceback: {traceback.format_exc()}"
            )
            return TaskResult(error=str(e))
        finally:
            if vectors_dataset is not None:
                vectors_dataset.free_vectors_space()
            vector_buffer.close()
            doc_id_buffer.close()


def build_index(
    index_build_params: IndexBuildParameters,
    vectors_dataset: VectorsDataset,
    output_destination: Union[BytesIO, str],
    index_serialization_mode: IndexSerializationMode = IndexSerializationMode.DISK,
) -> None:
    """Builds an index using the provided vectors dataset and parameters.

    This function wraps the FaissIndexBuildService build_index function. In the future,
    if there are other engines that can support building an index on GPUs/hardware accelerators,
    this function will abstract the engine type from the caller.
    Currently, only FAISS is supported via the index_build_params.engine parameter

    Args:
        index_build_params (IndexBuildParameters): Contains the configuration for the index build,
            including:
            - dimension: The dimension of the vectors
            - index_parameters: Parameters for the index, such as space type
        index_serialization_mode
        vectors_dataset (VectorsDataset): The dataset containing the vectors and document IDs
        output_destination: The output destination - either a file path (str) or buffer (BytesIO) to write the graph to


    Returns:
        None

    Note:
        - The caller is responsible for closing the vectors_dataset object
        - The caller is responsible for removing the CPU index file after upload to remote storage
        - This method

    Raises:
        Exception: If the index build fails

    """

    faiss_service = FaissIndexBuildService()
    cpu_index = faiss_service.build_index(
        index_build_params,
        vectors_dataset,
    )

    faiss_service.write_cpu_index(
        cpu_index, index_build_params, index_serialization_mode, output_destination
    )


def _determine_streaming_buffer(
    index_build_params: IndexBuildParameters, vector_bytes_buffer
):
    from remote_vector_index_builder.core.common.models.index_build_parameters import (
        DataType,
    )
    from remote_vector_index_builder.core.fp32_to_fp16_converting_bytes_io import (
        FP32ToFP16ConvertingBytesIO,
    )

    if index_build_params.data_type == DataType.FLOAT16:
        return FP32ToFP16ConvertingBytesIO(
            index_build_params.doc_count * index_build_params.dimension
        )

    return vector_bytes_buffer


def create_vectors_dataset(
    index_build_params: IndexBuildParameters,
    object_store: ObjectStore,
    vector_bytes_buffer: BytesIO,
    doc_id_bytes_buffer: BytesIO,
) -> VectorsDataset:
    """
    Downloads vector and document ID data from object storage and creates a VectorsDataset.

    This function performs the first step in the index building process by:
    1. Creating an appropriate object store instance
    2. Downloading vector data from the specified vector_path, into the vector_bytes_buffer
    3. Downloading document IDs from the specified doc_id_path, into the doc_id_bytes_buffer
    4. Combining them into a VectorsDataset object

    Args:
        index_build_params (IndexBuildParameters): Contains the configuration for the index build,
            including:
            - vector_path: Path to the vector data in object storage
            - doc_id_path: Path to the document IDs in object storage
            - repository_type: Type of object store to use
        object_store (ObjectStore): Object store instance
        vector_bytes_buffer: Buffer for storing vector binary data
        doc_id_bytes_buffer: Buffer for storing doc id binary data

    Returns:
        VectorsDataset: A dataset object containing:
            - The downloaded vectors in the specified format
            - Associated document IDs for each vector

    Note:
        - Uses BytesIO buffers for memory-efficient data handling
            - The caller is responsible for closing each buffer
            - Before closing the buffers, caller must call free_vector_space on VectorDataset object,
                to remove all references to the underlying data.
        - Both vector and document ID files must exist in object storage
        - The number of vectors must match the number of document IDs
        - Memory usage scales with the size of the vector and document ID data

    Raises:
        BlobError: If there are issues accessing or reading from object storage
        VectorDatasetError: If there are issues parsing the vectors and/or doc IDs into a VectorDataset
        UnsupportedVectorsDataTypeError: If the index_build_params.data_type is not supported
        UnsupportedObjectStoreTypeError: If the index_build_params.repository_type is not supported

    """
    vector_bytes_buffer = _determine_streaming_buffer(
        index_build_params, vector_bytes_buffer
    )
    object_store.read_blob(index_build_params.vector_path, vector_bytes_buffer)
    object_store.read_blob(index_build_params.doc_id_path, doc_id_bytes_buffer)

    return VectorsDataset.parse(
        vector_bytes_buffer,
        doc_id_bytes_buffer,
        index_build_params.dimension,
        index_build_params.doc_count,
        index_build_params.data_type,
    )


def upload_index(
    index_build_params: IndexBuildParameters,
    object_store: ObjectStore,
    data: Union[BytesIO, str],
) -> str:
    """
    Uploads a built index from a local path to the configured object store.

    Args:
        index_build_params (IndexBuildParameters): Parameters for the index build process,
            containing the vector path which is used to determine the upload destination
        object_store (ObjectStore): Object Store instance
        data (Union[str, BytesIO]): Either a string representing a local file path
            or a BytesIO object representing a buffer


    Returns:
        None

    Note:
        - Creates an object store instance based on the provided configuration
        - Uses the vector_path from index_build_params to determine the upload destination
            - The upload destination has the same file path as the vector_path
              except for the file extension. The file extension is based on the engine
        - The data store must be a buffer or a filepath
        - The function assumes index_build_params has already been validated by Pydantic

    Raises:
        BlobError: If there are issues uploading to the object store
        UnsupportedObjectStoreTypeError: If the index_build_params.repository_type is not supported
    """

    # vector path has already been validated that it ends with '.knnvec' by pydantic regex
    vector_root_path = ".".join(index_build_params.vector_path.split(".")[0:-1])

    # the index path is in the same root location as the vector path
    index_remote_path = vector_root_path + "." + index_build_params.engine

    object_store.write_blob(data, index_remote_path)

    return index_remote_path


@contextmanager
def index_storage_context(
    storage_mode: IndexSerializationMode, temp_dir: str, vector_path: str
):
    """Context manager for index storage setup and cleanup."""
    if storage_mode == IndexSerializationMode.MEMORY:
        logger.debug(
            f"Build is configured to store index in memory for vector path {vector_path}"
        )
        index_buffer = BytesIO()
        try:
            yield index_buffer
        finally:
            index_buffer.close()
    else:  # DISK
        logger.debug(
            f"Build is configured to store index on disk for vector path {vector_path}"
        )
        index_local_path = os.path.join(temp_dir, vector_path)
        directory = os.path.dirname(index_local_path)
        os.makedirs(directory, exist_ok=True)
        try:
            yield index_local_path
        finally:
            if os.path.exists(index_local_path):
                os.remove(index_local_path)

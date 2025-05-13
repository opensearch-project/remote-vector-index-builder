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
individual task in sequence. One reason to call each task in sequence is to persist the vectors to external storage,
before building the index. The default run_tasks function supports storing the vectors on host disk or memory.

"""
import logging
import os
from tempfile import TemporaryDirectory
from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, Optional

from core.common.models import IndexBuildParameters
from core.common.models.index_build_parameters import DataType
from core.common.models import VectorsDataset
from core.index_builder.faiss.faiss_index_build_service import FaissIndexBuildService
from core.object_store.object_store_factory import ObjectStoreFactory
from core.binary_source.binary_source_factory import BinarySourceFactory, StorageMode
from core.binary_source.binary_source import BinarySource

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
    storage_mode: StorageMode = StorageMode.MEMORY,
) -> TaskResult:
    """Execute the index building tasks using the provided parameters.

    This function orchestrates the index building process by:
    1. Downloading vector and doc ID binaries from remote storage
    2. Building an index
    3. Uploading index to remote storage

    Args:
        index_build_params (IndexBuildParameters): Parameters for building the index,
            including vector path and other configuration settings.
        object_store_config (Dict[str, Any], optional): Configuration settings for the
            object store. Defaults to None, in which case an empty dictionary is used.
        storage_mode: Where to store the downloaded vectors. Can be either CPU memory or disk, defaults to memory

    Returns:
        TaskResult: An object containing either:
            - file_name: The name of the successfully created index file in remote storage
            - error: An error message if the operation failed

    """
    with (
        TemporaryDirectory() as temp_dir,
        BinarySourceFactory.create_binary_source(storage_mode) as doc_id_source,
        BinarySourceFactory.create_binary_source(storage_mode) as vector_source,
    ):
        if object_store_config is None:
            object_store_config = {}

        vectors_dataset = None
        try:
            logger.info(
                f"Starting task execution for vector path: {index_build_params.vector_path}"
            )

            index_path = _get_index_path_from_vector_path(
                index_build_params.vector_path, index_build_params.engine
            )
            index_local_path = os.path.join(temp_dir, index_path)
            directory = os.path.dirname(index_local_path)
            os.makedirs(directory, exist_ok=True)

            vectors_dataset = create_vectors_dataset(
                index_build_params, object_store_config, vector_source, doc_id_source
            )

            logger.info(
                f"Building GPU index for vector path: {index_build_params.vector_path}"
            )
            build_index(
                index_build_params=index_build_params,
                vectors_dataset=vectors_dataset,
                cpu_index_output_file_path=index_local_path,
            )

            vectors_dataset.free_vectors_space()
            logger.info(
                f"Uploading index for vector path: {index_build_params.vector_path}"
            )
            remote_path = upload_index(
                index_build_params=index_build_params,
                object_store_config=object_store_config,
                index_local_path=index_local_path,
            )

            os.remove(index_local_path)
            logger.info(
                f"Ending task execution for vector path: {index_build_params.vector_path}"
            )

            return TaskResult(file_name=os.path.basename(remote_path))

        except Exception as e:
            logger.error(f"Error running tasks: {e}")
            return TaskResult(error=str(e))
        finally:
            if vectors_dataset is not None:
                vectors_dataset.free_vectors_space()


def build_index(
    index_build_params: IndexBuildParameters,
    vectors_dataset: VectorsDataset,
    cpu_index_output_file_path: str,
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
        vectors_dataset (VectorsDataset): The dataset containing the vectors and document IDs
        cpu_index_output_file_path (str): The file path where the CPU compatible index will be saved

    Returns:
        None

    Note:
        - The caller is responsible for closing the vectors_dataset object
        - The caller is responsible for removing the CPU index file after upload to remote storage

    Raises:
        Exception: If the index build fails

    """
    faiss_service = FaissIndexBuildService()
    faiss_service.build_index(
        index_build_params, vectors_dataset, cpu_index_output_file_path
    )


def create_vectors_dataset(
    index_build_params: IndexBuildParameters,
    object_store_config: Dict[str, Any],
    vector_source: BinarySource,
    doc_id_source: BinarySource,
) -> VectorsDataset:
    """
    Downloads vector and document ID data from object storage and creates a VectorsDataset.
    Vector and doc ID data are first read into the corresponding BinarySource data structures,
    before being assigned to the VectorsDataset object. It is the callers responsibility to
    close the BinarySource data structures.

    Args:

        index_build_params (IndexBuildParameters): Build configuration
        object_store_config (Dict[str, Any]): Object store configuration
        vector_source (BinarySource): The data structure that stores the vectors
        doc_id_source (BinarySource): The data structure that stores the doc ids

    Returns:
        VectorsDataset: Dataset containing the vectors and document IDs
    """

    object_store = ObjectStoreFactory.create_object_store(
        index_build_params, object_store_config
    )

    np_docs = doc_id_source.transform_to_numpy_array(
        object_store,
        index_build_params.doc_id_path,
        "<i4",
        index_build_params.doc_count,
    )
    _check_dimensions(np_docs, index_build_params.doc_count)

    np_vectors = vector_source.transform_to_numpy_array(
        object_store,
        index_build_params.vector_path,
        _get_numpy_dtype(index_build_params.data_type),
        index_build_params.doc_count * index_build_params.dimension,
    )
    _check_dimensions(
        np_vectors, index_build_params.doc_count * index_build_params.dimension
    )
    np_vectors = np_vectors.reshape(
        index_build_params.doc_count, index_build_params.dimension
    )
    vectors_dataset = VectorsDataset(np_vectors, np_docs)
    return vectors_dataset


def upload_index(
    index_build_params: IndexBuildParameters,
    object_store_config: Dict[str, Any],
    index_local_path: str,
) -> str:
    """
    Uploads a built index from a local path to the configured object store.

    Args:
        index_build_params (IndexBuildParameters): Parameters for the index build process,
            containing the vector path which is used to determine the upload destination
        object_store_config (Dict[str, Any]): Configuration dictionary for the object store
            containing connection details
        index_local_path (str): Local filesystem path where the built index is stored

    Returns:
        None

    Note:
        - Creates an object store instance based on the provided configuration
        - Uses the vector_path from index_build_params to determine the upload destination
            - The upload destination has the same file path as the vector_path
              except for the file extension. The file extension is based on the engine
        - The index_local_path must exist and be readable

    Raises:
        BlobError: If there are issues uploading to the object store
        UnsupportedObjectStoreTypeError: If the index_build_params.repository_type is not supported
    """
    object_store = ObjectStoreFactory.create_object_store(
        index_build_params, object_store_config
    )

    index_remote_path = _get_index_path_from_vector_path(
        index_build_params.vector_path, index_build_params.engine
    )

    object_store.write_blob(index_local_path, index_remote_path)

    return index_remote_path


def _get_index_path_from_vector_path(vector_path: str, engine: str) -> str:
    """
    Helper function to get the index path from the vector path.

    Args:
        vector_path (str): The vector path
    Returns:
        str: The index path
    """

    # vector path has already been validated that it ends with '.knnvec' by pydantic regex
    # so this will extract everything before the .knnvec extension
    vector_root_path = ".".join(vector_path.split(".")[0:-1])

    return vector_root_path + "." + engine


def _get_numpy_dtype(data_type: DataType) -> str:
    """
    Helper function to get the numpy data type from the DataType enum

    Args:
        data_type: The DataType enum value
    Returns:
        The numpy data type as a string
    Raises:
        ValueError: If the data type is not supported
    """
    if data_type == DataType.FLOAT:
        return "<f4"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def _check_dimensions(array: np.ndarray, expected_length: int):
    """
    Helper function to check the numpy array dimensions

    Args:
        array (np.ndarray): The numpy array to check
        expected_length: The expected length of the numpy array
    Raises:
        ValueError: If the length of the numpy array does not match expected length

    """
    if len(array) != expected_length:
        raise ValueError(f"Expected {expected_length} elements, but got {len(array)}")

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
core.tasks
~~~~~~~~~~~~~~~~~

This module contains the tasks necessary to build an index on GPUs.
These tasks must be run in the following sequence, for a given build request:
1. create_vectors_dataset
2. build_index
3. upload_index

"""
import logging
from io import BytesIO
from typing import Any, Dict

from core.common.models.index_build_parameters import IndexBuildParameters
from core.common.models.vectors_dataset import VectorsDataset
from core.object_store.object_store_factory import ObjectStoreFactory

logger = logging.getLogger(__name__)

# TODO: Create build_index task


def create_vectors_dataset(
    index_build_params: IndexBuildParameters,
    object_store_config: Dict[str, Any],
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
        object_store_config (Dict[str, Any]): Configuration for the object store
            containing connection details
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
    object_store = ObjectStoreFactory.create_object_store(
        index_build_params, object_store_config
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
    object_store_config: Dict[str, Any],
    index_local_path: str,
) -> None:
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
        - The function assumes index_build_params has already been validated by Pydantic

    Raises:
        BlobError: If there are issues uploading to the object store
        UnsupportedObjectStoreTypeError: If the index_build_params.repository_type is not supported
    """
    object_store = ObjectStoreFactory.create_object_store(
        index_build_params, object_store_config
    )

    # vector path has already been validated that it ends with '.knnvec' by pydantic regex
    vector_root_path = ".".join(index_build_params.vector_path.split(".")[0:-1])

    # the index path is in the same root location as the vector path
    index_remote_path = vector_root_path + "." + index_build_params.engine

    object_store.write_blob(index_local_path, index_remote_path)

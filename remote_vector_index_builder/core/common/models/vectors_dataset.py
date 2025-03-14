# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from core.common.exceptions import UnsupportedVectorsDataTypeError, VectorsDatasetError
from core.common.models.index_build_parameters import DataType


@dataclass
class VectorsDataset:
    """A class for handling vector datasets and their associated document IDs.

    This class provides functionality to parse, validate, and store vector data along with
    their corresponding document IDs. It supports multiple data types including FLOAT32,
    FLOAT16, BYTE, and BINARY formats.

    Attributes:
        vectors (numpy.ndarray): The array of vectors, where each row represents a vector.
        doc_ids (numpy.ndarray): Array of document IDs corresponding to the vectors.
    """

    vectors: np.ndarray
    doc_ids: np.ndarray

    def free_vectors_space(self):
        """Free up memory by deleting the vectors and document IDs arrays."""
        del self.vectors
        del self.doc_ids

    @staticmethod
    def get_numpy_dtype(dtype: DataType):
        """Convert DataType enum to numpy dtype string.

        Args:
            dtype (DataType): The data type enum value to convert.

        Returns:
            str: The corresponding numpy dtype string.

        Raises:
            UnsupportedVectorsDataTypeError: If the provided data type is not supported.
        """
        if dtype == DataType.FLOAT:
            return "<f4"
        else:
            raise UnsupportedVectorsDataTypeError(f"Unsupported data type: {dtype}")

    @staticmethod
    def check_dimensions(vectors, expected_length):
        """Validate that the vector array has the expected length.

        Args:
            vectors: Array-like object to check.
            expected_length (int): The expected length of the vectors array.

        Raises:
            VectorsDatasetError: If the vectors length doesn't match the expected length.
        """
        if len(vectors) != expected_length:
            raise VectorsDatasetError(
                f"Expected {expected_length} vectors, but got {len(vectors)}"
            )

    @staticmethod
    def parse(
        vectors: BytesIO,
        doc_ids: BytesIO,
        dimension: int,
        doc_count: int,
        vector_dtype: DataType,
    ):
        """Parse binary vector data and document IDs into numpy arrays.

        This method reads binary data for vectors and document IDs, validates their
        dimensions, and creates a new VectorsDataset instance.

        Args:
            vectors (BytesIO): Binary stream containing vector data.
            doc_ids (BytesIO): Binary stream containing document IDs.
            dimension (int): The dimensionality of each vector.
            doc_count (int): Expected number of vectors/documents.
            vector_dtype (DataType): The data type of the vector values.

        Returns:
            VectorsDataset: A new instance containing the parsed vectors and document IDs.

        Raises:
            VectorsDatasetError: If there are any errors during parsing or validation.
        """
        try:
            # Create a view into the buffer, to prevent additional allocation of memory
            vector_view = vectors.getbuffer()
            np_vectors = np.frombuffer(
                vector_view, dtype=VectorsDataset.get_numpy_dtype(vector_dtype)
            )
            VectorsDataset.check_dimensions(np_vectors, doc_count * dimension)
            np_vectors = np_vectors.reshape(doc_count, dimension)

            # Do the same for doc ids
            doc_id_view = doc_ids.getbuffer()
            np_doc_ids = np.frombuffer(doc_id_view, dtype="<i4")
            VectorsDataset.check_dimensions(np_doc_ids, doc_count)

        except (ValueError, TypeError, MemoryError, RuntimeError) as e:
            raise VectorsDatasetError(f"Error parsing vectors: {e}") from e
        return VectorsDataset(np_vectors, np_doc_ids)

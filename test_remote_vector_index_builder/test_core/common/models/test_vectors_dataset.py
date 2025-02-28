# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from core.common.exceptions import UnsupportedVectorsDataTypeError, VectorsDatasetError
from core.common.models.index_build_parameters import DataType
from core.common.models.vectors_dataset import VectorsDataset


@pytest.fixture
def sample_vectors():
    # Create sample float32 vectors (2 vectors of dimension 3)
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4")


@pytest.fixture
def sample_doc_ids():
    return np.array([1, 2], dtype="<i4")


@pytest.fixture
def vectors_dataset(sample_vectors, sample_doc_ids):
    return VectorsDataset(vectors=sample_vectors, doc_ids=sample_doc_ids)


def test_initialization(sample_vectors, sample_doc_ids):
    dataset = VectorsDataset(vectors=sample_vectors, doc_ids=sample_doc_ids)
    assert np.array_equal(dataset.vectors, sample_vectors)
    assert np.array_equal(dataset.doc_ids, sample_doc_ids)


def test_free_vectors_space(vectors_dataset):
    vectors_dataset.free_vectors_space()
    with pytest.raises(AttributeError):
        _ = vectors_dataset.vectors
    with pytest.raises(AttributeError):
        _ = vectors_dataset.doc_ids


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (DataType.FLOAT32, "<f4"),
        (DataType.FLOAT16, "<f2"),
        (DataType.BYTE, "<i1"),
        (DataType.BINARY, "<i1"),
    ],
)
def test_get_numpy_dtype_valid(dtype, expected):
    assert VectorsDataset.get_numpy_dtype(dtype) == expected


def test_get_numpy_dtype_invalid():
    with pytest.raises(UnsupportedVectorsDataTypeError):
        VectorsDataset.get_numpy_dtype("invalid_dtype")


def test_check_dimensions_valid():
    vectors = np.zeros(5)
    VectorsDataset.check_dimensions(vectors, 5)  # Should not raise


def test_check_dimensions_invalid():
    vectors = np.zeros(5)
    with pytest.raises(VectorsDatasetError):
        VectorsDataset.check_dimensions(vectors, 10)


@pytest.mark.parametrize(
    "vector_dtype", [DataType.FLOAT32, DataType.FLOAT16, DataType.BYTE, DataType.BINARY]
)
def test_parse_valid_data(vector_dtype):
    # Prepare test data
    dimension = 3
    doc_count = 2

    arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    if vector_dtype == DataType.BYTE:
        arr = [[1, 2, 3], [4, 5, 6]]
    elif vector_dtype == DataType.BINARY:
        arr = [[0, 0, 0], [1, 1, 1]]

    test_vectors = np.array(arr, dtype=VectorsDataset.get_numpy_dtype(vector_dtype))
    test_doc_ids = np.array([1, 2], dtype="<i4")

    # Convert to binary
    vectors_binary = BytesIO(test_vectors.tobytes())
    doc_ids_binary = BytesIO(test_doc_ids.tobytes())

    # Parse
    dataset = VectorsDataset.parse(
        vectors=vectors_binary,
        doc_ids=doc_ids_binary,
        dimension=dimension,
        doc_count=doc_count,
        vector_dtype=vector_dtype,
    )

    # Verify
    assert isinstance(dataset, VectorsDataset)
    assert dataset.vectors.shape == (doc_count, dimension)
    assert len(dataset.doc_ids) == doc_count
    assert len(dataset.vectors) == doc_count
    assert np.array_equal(dataset.doc_ids, test_doc_ids)
    assert np.array_equal(dataset.vectors, test_vectors)

    dataset.free_vectors_space()
    vectors_binary.close()
    doc_ids_binary.close()


def test_parse_invalid_doc_count():
    with pytest.raises(VectorsDatasetError):
        vectors = BytesIO(np.zeros(6, dtype="<f4").tobytes())
        doc_ids = BytesIO(np.array([1], dtype="<i4").tobytes())
        dataset = VectorsDataset.parse(
            vectors=vectors,
            doc_ids=doc_ids,
            dimension=2,
            doc_count=2,
            vector_dtype=DataType.FLOAT32,
        )
        dataset.free_vectors_space()
        vectors.close()
        doc_ids.close()


def test_parse_invalid_vector_dimensions():
    with pytest.raises(VectorsDatasetError):
        vectors = BytesIO(np.zeros(4, dtype="<f4").tobytes())
        doc_ids = BytesIO(np.array([1, 2], dtype="<i4").tobytes())
        dataset = VectorsDataset.parse(
            vectors=vectors,
            doc_ids=doc_ids,
            dimension=3,  # Expecting 6 values (2*3), but only provided 4
            doc_count=2,
            vector_dtype=DataType.FLOAT32,
        )
        dataset.free_vectors_space()
        vectors.close()
        doc_ids.close()


#
def test_parse_invalid_data():
    with patch("numpy.frombuffer") as mock_frombuffer:
        mock_frombuffer.side_effect = ValueError("Invalid data")
        with pytest.raises(VectorsDatasetError):
            vectors = BytesIO(np.zeros(6, dtype="<f4").tobytes())
            doc_ids = BytesIO(np.array([1, 2], dtype="<i4").tobytes())
            dataset = VectorsDataset.parse(
                vectors=vectors,
                doc_ids=doc_ids,
                dimension=3,
                doc_count=2,
                vector_dtype=DataType.FLOAT32,
            )
            dataset.free_vectors_space()
            vectors.close()
            doc_ids.close()

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


def test_free_vectors_space_when_vectors_and_doc_ids_already_deleted(vectors_dataset):
    vectors_dataset.free_vectors_space()
    with pytest.raises(AttributeError):
        _ = vectors_dataset.vectors
    with pytest.raises(AttributeError):
        _ = vectors_dataset.doc_ids
    # test idempotency
    vectors_dataset.free_vectors_space()


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (DataType.FLOAT, "<f4"),
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


def test_parse_valid_data(sample_vectors, sample_doc_ids):
    # Prepare test data
    dimension = len(sample_vectors[0])
    doc_count = len(sample_vectors)
    vector_dtype = DataType.FLOAT

    # Convert to binary
    vectors_binary = BytesIO(sample_vectors.tobytes())
    doc_ids_binary = BytesIO(sample_doc_ids.tobytes())

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
    assert np.array_equal(dataset.doc_ids, sample_doc_ids)
    assert np.array_equal(dataset.vectors, sample_vectors)

    dataset.free_vectors_space()
    vectors_binary.close()
    doc_ids_binary.close()


def test_parse_invalid_doc_count():
    with pytest.raises(VectorsDatasetError):
        vectors = BytesIO(np.zeros(6, dtype="<f4").tobytes())
        doc_ids = BytesIO(np.array([1, 2, 3, 4, 5, 6], dtype="<i4").tobytes())
        dataset = VectorsDataset.parse(
            vectors=vectors,
            doc_ids=doc_ids,
            dimension=1,
            doc_count=5,
            vector_dtype=DataType.FLOAT,
        )
        dataset.free_vectors_space()
        vectors.close()
        doc_ids.close()


def test_parse_invalid_vector_dimensions():
    with pytest.raises(VectorsDatasetError):
        vectors = BytesIO(np.zeros(5, dtype="<f4").tobytes())
        doc_ids = BytesIO(np.array([1, 2, 3, 4, 5], dtype="<i4").tobytes())
        dataset = VectorsDataset.parse(
            vectors=vectors,
            doc_ids=doc_ids,
            dimension=2,  # Expecting 10 values (5*2), but only provided 5
            doc_count=5,
            vector_dtype=DataType.FLOAT,
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
            doc_ids = BytesIO(np.array([1, 2, 3, 4, 5, 6], dtype="<i4").tobytes())
            dataset = VectorsDataset.parse(
                vectors=vectors,
                doc_ids=doc_ids,
                dimension=1,
                doc_count=6,
                vector_dtype=DataType.FLOAT,
            )
            dataset.free_vectors_space()
            vectors.close()
            doc_ids.close()

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
    dataset = VectorsDataset(
        vectors=sample_vectors, doc_ids=sample_doc_ids, dtype=DataType.FLOAT
    )
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
        (DataType.BYTE, "<i1"),
        (DataType.BINARY, "<u1"),
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


def test_parse_valid_fp32_data(sample_vectors, sample_doc_ids):
    _do_test_parse_valid_data(sample_vectors, sample_doc_ids, DataType.FLOAT)


def test_parse_valid_byte_data(sample_byte_vectors, sample_doc_ids):
    _do_test_parse_valid_data(sample_byte_vectors, sample_doc_ids, DataType.BYTE)


def test_parse_valid_binary_data(sample_binary_vectors, sample_doc_ids):
    _do_test_parse_valid_data(sample_binary_vectors, sample_doc_ids, DataType.BINARY)


def _do_test_parse_valid_data(sample_vectors, sample_doc_ids, vector_dtype):
    # Prepare test data
    dimension = len(sample_vectors[0])
    if vector_dtype == DataType.BINARY:
        # For binary case, one vector is bit, so we need to multiply 8 for dimension
        dimension = dimension * 8
    doc_count = len(sample_vectors)

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
    last_shape = dimension
    if vector_dtype == DataType.BINARY:
        last_shape = last_shape / 8

    assert isinstance(dataset, VectorsDataset)
    assert dataset.vectors.shape == (doc_count, last_shape)
    assert len(dataset.doc_ids) == doc_count
    assert len(dataset.vectors) == doc_count
    assert np.array_equal(dataset.doc_ids, sample_doc_ids)
    assert np.array_equal(dataset.vectors, sample_vectors)
    assert dataset.dtype == vector_dtype

    dataset.free_vectors_space()
    vectors_binary.close()
    doc_ids_binary.close()


@pytest.mark.parametrize(
    "vector_dtype, numpy_dtype",
    [
        (DataType.FLOAT, "<f4"),
        (DataType.BYTE, "<i1"),
        (DataType.BINARY, "<u1"),
    ],
)
def _do_test_parse_invalid_doc_count(vector_dtype, numpy_dtype):
    with pytest.raises(VectorsDatasetError):
        num_vectors = 8
        if vector_dtype == DataType.BINARY:
            # In binary vector, one bit represents one vector so one byte represents 8 vectors each.
            num_vectors = 1

        vectors = BytesIO(np.zeros(num_vectors, dtype=numpy_dtype).tobytes())
        doc_ids = BytesIO(np.array([1, 2, 3, 4, 5, 6], dtype="<i4").tobytes())
        dataset = VectorsDataset.parse(
            vectors=vectors,
            doc_ids=doc_ids,
            dimension=1,
            doc_count=num_vectors,
            vector_dtype=vector_dtype,
        )
        dataset.free_vectors_space()
        vectors.close()
        doc_ids.close()


@pytest.mark.parametrize(
    "vector_dtype, numpy_dtype",
    [
        (DataType.FLOAT, "<f4"),
        (DataType.BYTE, "<i1"),
    ],
)
def test_parse_invalid_vector_dimensions(vector_dtype, numpy_dtype):
    with pytest.raises(VectorsDatasetError):
        vectors = BytesIO(np.zeros(5, dtype=numpy_dtype).tobytes())
        doc_ids = BytesIO(np.array([1, 2, 3, 4, 5], dtype="<i4").tobytes())
        dataset = VectorsDataset.parse(
            vectors=vectors,
            doc_ids=doc_ids,
            dimension=2,  # Expecting 10 values (5*2), but only provided 5
            doc_count=5,
            vector_dtype=vector_dtype,
        )
        dataset.free_vectors_space()
        vectors.close()
        doc_ids.close()


@pytest.mark.parametrize("num_docs", [10, 3])
def test_parse_invalid_binary_vector_dimensions(num_docs):
    with pytest.raises(VectorsDatasetError):
        vectors = BytesIO(np.zeros(num_docs, dtype="<u1").tobytes())
        doc_ids = BytesIO(np.array([1, 2, 3, 4, 5], dtype="<i4").tobytes())
        dataset = VectorsDataset.parse(
            vectors=vectors,
            doc_ids=doc_ids,
            dimension=8,  # e.g. one vector element would occupy 1 byte (= 8 bits)
            doc_count=num_docs,
            vector_dtype=DataType.BINARY,
        )
        dataset.free_vectors_space()
        vectors.close()
        doc_ids.close()


@pytest.mark.parametrize(
    "vector_dtype, numpy_dtype",
    [
        (DataType.FLOAT, "<f4"),
        (DataType.BYTE, "<i1"),
        (DataType.BINARY, "<u1"),
    ],
)
def test_parse_invalid_data(vector_dtype, numpy_dtype):
    with patch("numpy.frombuffer") as mock_frombuffer:
        mock_frombuffer.side_effect = ValueError("Invalid data")
        with pytest.raises(VectorsDatasetError):
            vectors = BytesIO(np.zeros(6, dtype=numpy_dtype).tobytes())
            doc_ids = BytesIO(np.array([1, 2, 3, 4, 5, 6], dtype="<i4").tobytes())
            dataset = VectorsDataset.parse(
                vectors=vectors,
                doc_ids=doc_ids,
                dimension=1,
                doc_count=6,
                vector_dtype=vector_dtype,
            )
            dataset.free_vectors_space()
            vectors.close()
            doc_ids.close()

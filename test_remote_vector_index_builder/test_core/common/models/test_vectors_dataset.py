# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import numpy as np
import pytest
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

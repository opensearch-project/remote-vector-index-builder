# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import numpy as np
import pytest
from core.common.models import VectorsDataset
from core.common.models.index_build_parameters import (
    AlgorithmParameters,
    IndexBuildParameters,
    IndexParameters,
    SpaceType,
)


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing"""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)


@pytest.fixture
def sample_doc_ids():
    """Generate sample document IDs for testing"""
    return np.array([1, 2], dtype=np.int64)


@pytest.fixture
def vectors_dataset(sample_vectors, sample_doc_ids):
    """Create a VectorsDataset instance for testing"""
    return VectorsDataset(vectors=sample_vectors, doc_ids=sample_doc_ids)


@pytest.fixture
def index_build_parameters():
    """Create sample IndexBuildParameters for testing"""
    return IndexBuildParameters(
        container_name="testbucket",
        vector_path="vec.knnvec",
        doc_id_path="doc.knndid",
        dimension=3,
        doc_count=2,
        index_parameters=IndexParameters(
            space_type=SpaceType.INNERPRODUCT,
            algorithm_parameters=AlgorithmParameters(
                ef_construction=200, ef_search=200
            ),
        ),
        repository_type="s3",
    )


@pytest.fixture
def object_store_config():
    """Create a sample object store configuration for testing"""
    return {
        "region": "us-west-2",
    }

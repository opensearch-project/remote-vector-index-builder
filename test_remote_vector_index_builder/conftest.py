# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest
import os
from core.common.models.index_build_parameters import (
    AlgorithmParameters,
    IndexBuildParameters,
    IndexParameters,
    SpaceType,
)

from remote_vector_index_builder.core.common.models.index_build_parameters import (
    DataType,
)


@pytest.fixture
def index_build_parameters():
    """Create sample IndexBuildParameters for testing"""
    return IndexBuildParameters(
        container_name="testbucket",
        vector_path="vec.knnvec",
        doc_id_path="doc.knndid",
        dimension=3,
        doc_count=5,
        index_parameters=IndexParameters(
            space_type=SpaceType.INNERPRODUCT,
            algorithm_parameters=AlgorithmParameters(
                ef_construction=200, ef_search=200
            ),
        ),
        data_type=DataType.FLOAT,
        repository_type="s3",
    )


@pytest.fixture
def byte_index_build_parameters():
    """Create sample IndexBuildParameters for testing"""
    return IndexBuildParameters(
        container_name="testbucket",
        vector_path="vec.knnvec",
        doc_id_path="doc.knndid",
        dimension=3,
        doc_count=5,
        index_parameters=IndexParameters(
            space_type=SpaceType.INNERPRODUCT,
            algorithm_parameters=AlgorithmParameters(
                ef_construction=200, ef_search=200
            ),
        ),
        data_type=DataType.BYTE,
        repository_type="s3",
    )


@pytest.fixture
def binary_index_build_parameters():
    """Create sample IndexBuildParameters for testing"""
    return IndexBuildParameters(
        container_name="testbucket",
        vector_path="vec.knnvec",
        doc_id_path="doc.knndid",
        dimension=24,
        doc_count=5,
        index_parameters=IndexParameters(
            space_type=SpaceType.INNERPRODUCT,
            algorithm_parameters=AlgorithmParameters(
                ef_construction=200, ef_search=200
            ),
        ),
        data_type=DataType.BINARY,
        repository_type="s3",
    )


@pytest.fixture(autouse=True)
def aws_credentials():
    """Mocked AWS Credentials for tests."""
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    yield
    os.environ.pop("AWS_DEFAULT_REGION", None)

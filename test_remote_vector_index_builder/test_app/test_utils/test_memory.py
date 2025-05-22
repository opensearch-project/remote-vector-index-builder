# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest
from core.common.models.index_build_parameters import (
    AlgorithmParameters,
    IndexBuildParameters,
    IndexParameters,
    SpaceType,
    DataType,
)
from app.utils.memory import calculate_memory_requirements


@pytest.fixture
def base_index_parameters():
    """Create base IndexParameters for testing"""
    return IndexParameters(
        space_type=SpaceType.INNERPRODUCT,
        algorithm_parameters=AlgorithmParameters(
            ef_construction=200, ef_search=200, m=16
        ),
    )


def create_index_build_parameters(
    dimension: int, doc_count: int, m: int = 16, data_type: DataType = DataType.FLOAT
) -> IndexBuildParameters:
    """Helper function to create IndexBuildParameters with different values"""
    return IndexBuildParameters(
        container_name="testbucket",
        vector_path="test.knnvec",
        doc_id_path="test_ids.txt",
        dimension=dimension,
        doc_count=doc_count,
        data_type=data_type,
        index_parameters=IndexParameters(
            space_type=SpaceType.INNERPRODUCT,
            algorithm_parameters=AlgorithmParameters(
                ef_construction=200, ef_search=200, m=m
            ),
        ),
    )


def test_basic_calculation():
    """Test calculation with typical values"""
    params = create_index_build_parameters(dimension=128, doc_count=1000)

    gpu_memory, cpu_memory = calculate_memory_requirements(params)

    # Calculate expected values
    entry_size = 4  # FLOAT32 size
    vector_memory = 128 * 1000 * entry_size
    expected_gpu_memory = ((128 * entry_size + 16 * 8) * 1.1 * 1000) * 0.5
    expected_cpu_memory = (128 * entry_size + 16 * 8) * 1.1 * 1000 + vector_memory

    assert gpu_memory == expected_gpu_memory
    assert cpu_memory == expected_cpu_memory

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import numpy as np
import pytest
import sys
from types import ModuleType
from unittest.mock import Mock

from core.common.models import VectorsDataset


class DeletionTracker:
    """Helper class to track object deletions"""

    def __init__(self):
        self.deleted_objects = set()

    def mark_deleted(self, obj_id):
        self.deleted_objects.add(obj_id)

    def is_deleted(self, obj_id):
        return obj_id in self.deleted_objects

    def reset(self):
        self.deleted_objects.clear()


# Create global deletion tracker
_deletion_tracker = DeletionTracker()


@pytest.fixture
def deletion_tracker():
    """Fixture to provide access to deletion tracker"""
    _deletion_tracker.reset()  # Reset before each test
    return _deletion_tracker


@pytest.fixture(autouse=True)
def reset_deletion_tracker():
    """Reset deletion tracker before each test"""
    _deletion_tracker.reset()
    yield


class MockGpuIndexCagra:
    """Mock for faiss.GpuIndexCagra with deletion tracking"""

    def __init__(self, *args, **kwargs):
        self.id = id(self)
        self.thisown = False
        self.args = args
        self.kwargs = kwargs

    def __del__(self):
        print("deleting MockGpuIndexCagra:", self.id)
        _deletion_tracker.mark_deleted(self.id)

    @property
    def is_deleted(self):
        return _deletion_tracker.is_deleted(self.id)

    def copyTo(self, cpu_index):
        """Mock implementation of copyTo method"""
        if not isinstance(cpu_index, MockIndexHNSWCagra):
            raise TypeError("Target must be IndexHNSWCagra")
        # Simulate copying data to CPU index
        return True


class MockIndexIDMap:
    """Mock for faiss.IndexIDMap with deletion tracking"""

    def __init__(self, *args, **kwargs):
        self.id = id(self)
        self.own_fields = False
        self.index = None
        self.args = args
        self.kwargs = kwargs

    def __del__(self):
        print("deleting MockIndexIDMap:", self.id)
        _deletion_tracker.mark_deleted(self.id)

    @property
    def is_deleted(self):
        return _deletion_tracker.is_deleted(self.id)

    def add_with_ids(self, vectors, ids):
        pass


class MockIndexHNSWCagra(Mock):
    """Mock for faiss.IndexHNSWCagra"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hnsw = Mock()
        self.base_level_only = True

    def __del__(self):
        _deletion_tracker.mark_deleted(self.id)

    @property
    def is_deleted(self):
        return _deletion_tracker.is_deleted(self.id)


class MockIVFPQBuildCagraConfig:
    """Mock class for faiss.IVFPQBuildCagraConfig"""

    def __init__(self):
        self.n_lists = 1024
        self.kmeans_n_iters = 20
        self.kmeans_trainset_fraction = 0.5
        self.pq_bits = 8
        self.pq_dim = 0
        self.conservative_memory_allocation = True


class MockIVFPQSearchCagraConfig:
    """Mock class for faiss.IVFPQSearchCagraConfig"""

    def __init__(self):
        self.n_probes = 20


class MockGpuIndexCagraConfig:
    """Mock class for faiss.GpuIndexCagraConfig"""

    def __init__(self):
        self.intermediate_graph_degree = 64
        self.graph_degree = 32
        self.store_dataset = False
        self.device = 0
        self.refine_rate = 2.0
        self.build_algo = None
        self.ivf_pq_build_config = None
        self.ivf_pq_search_config = None


class FaissMock(ModuleType):
    """Complete mock for faiss module"""

    def __init__(self):
        super().__init__("faiss")
        # Classes
        self.StandardGpuResources = Mock()
        self.GpuIndexCagra = MockGpuIndexCagra
        self.IndexIDMap = MockIndexIDMap
        self.IndexHNSWCagra = MockIndexHNSWCagra
        self.IVFPQBuildCagraConfig = MockIVFPQBuildCagraConfig
        self.IVFPQSearchCagraConfig = MockIVFPQSearchCagraConfig
        self.GpuIndexCagraConfig = MockGpuIndexCagraConfig

        # Enums
        self.graph_build_algo_IVF_PQ = 1

        self.METRIC_L2 = 0
        self.METRIC_INNER_PRODUCT = 1

        self._num_threads = None
        self.omp_set_num_threads = self._omp_set_num_threads
        self.omp_get_num_threads = self._omp_get_num_threads

        self.write_index = self._write_index

    def _omp_set_num_threads(self, num_threads: int) -> None:
        self._num_threads = num_threads

    def _omp_get_num_threads(self) -> int:
        return self._num_threads

    def _write_index(self, index, filepath):
        if not isinstance(filepath, str):
            raise TypeError("Filepath must be a string")
        if not index:
            raise ValueError("Index cannot be None")
        try:
            with open(filepath, "wb") as f:
                f.write(b"MOCK_INDEX")
        except IOError as e:
            raise IOError(f"Failed to write to {filepath}: {str(e)}")


# Create the mock and patch faiss
faiss_mock = FaissMock()
sys.modules["faiss"] = faiss_mock


@pytest.fixture
def object_store_config():
    """Create a sample object store configuration for testing"""
    return {
        "region": "us-west-2",
    }


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing"""
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_doc_ids():
    """Generate sample document IDs for testing"""
    return np.array([1, 2, 3, 4, 5], dtype=np.int32)


@pytest.fixture
def vectors_dataset(sample_vectors, sample_doc_ids):
    """Create a VectorsDataset instance for testing"""
    return VectorsDataset(vectors=sample_vectors, doc_ids=sample_doc_ids)

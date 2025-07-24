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
from core.common.models.index_build_parameters import DataType
from core.object_store.s3.s3_object_store_config import S3ClientConfig


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


class MockGpuIndexBinaryCagra:
    """Mock for faiss.GpuIndexBinaryCagra with deletion tracking"""

    def __init__(self, *args, **kwargs):
        self.id = id(self)
        self.thisown = False
        self.args = args
        self.kwargs = kwargs

    def __del__(self):
        print("deleting MockGpuIndexBinaryCagra:", self.id)
        _deletion_tracker.mark_deleted(self.id)

    @property
    def is_deleted(self):
        return _deletion_tracker.is_deleted(self.id)

    def copyTo(self, cpu_index):
        """Mock implementation of copyTo method"""
        if not isinstance(cpu_index, MockIndexBinaryHNSW):
            raise TypeError("Target must be IndexBinaryHNSW")
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

    def add_with_ids(self, vectors, ids, numeric_type):
        pass


class MockIndexBinaryIDMap:
    """Mock for faiss.IndexBinaryIDMap with deletion tracking"""

    def __init__(self, *args, **kwargs):
        self.id = id(self)
        self.own_fields = False
        self.index = None
        self.args = args
        self.kwargs = kwargs

    def __del__(self):
        print("deleting IndexBinaryIDMap:", self.id)
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


class MockIndexBinaryHNSW(Mock):
    """Mock for faiss.IndexBinaryHNSW"""

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
        self.GpuIndexBinaryCagra = MockGpuIndexBinaryCagra
        self.IndexIDMap = MockIndexIDMap
        self.IndexBinaryIDMap = MockIndexBinaryIDMap
        self.IndexHNSWCagra = MockIndexHNSWCagra
        self.IVFPQBuildCagraConfig = MockIVFPQBuildCagraConfig
        self.IVFPQSearchCagraConfig = MockIVFPQSearchCagraConfig
        self.GpuIndexCagraConfig = MockGpuIndexCagraConfig
        self.IndexBinaryHNSW = MockIndexBinaryHNSW
        self.Float32 = Mock()
        self.Float16 = Mock()
        self.Int8 = Mock()

        # Enums
        self.graph_build_algo_IVF_PQ = 0
        self.graph_build_algo_NN_DESCENT = 1

        self.METRIC_L2 = 0
        self.METRIC_INNER_PRODUCT = 1

        self._num_threads = None
        self.omp_set_num_threads = self._omp_set_num_threads
        self.omp_get_num_threads = self._omp_get_num_threads

        self.write_index = self._write_index
        self.write_index_binary = self._write_index_binary
        self.index_binary_gpu_to_cpu = self._index_binary_gpu_to_cpu

    def _omp_set_num_threads(self, num_threads: int) -> None:
        self._num_threads = num_threads

    def _omp_get_num_threads(self) -> int:
        return self._num_threads

    def _index_binary_gpu_to_cpu(self, index):
        if not index:
            raise ValueError("Index cannot be None")
        if not isinstance(index, MockGpuIndexBinaryCagra):
            raise TypeError("Target must be GpuIndexBinaryCagra")
        return self.IndexBinaryHNSW()

    def _write_index_binary(self, index, filepath):
        if not isinstance(filepath, str):
            raise TypeError("Filepath must be a string")
        if not index:
            raise ValueError("Index cannot be None")
        if not isinstance(index, MockIndexBinaryIDMap):
            raise TypeError("Target must be IndexBinaryIDMap")
        try:
            with open(filepath, "wb") as f:
                f.write(b"MOCK_INDEX_BINARY")
        except IOError as e:
            raise IOError(f"Failed to write to {filepath}: {str(e)}")

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
        "s3_client_config": S3ClientConfig(region_name="us-west-2", max_retries="4")
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
def sample_binary_vectors():
    """Generate sample binary vectors for testing"""
    return np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def sample_byte_vectors():
    """Generate sample byte vectors for testing"""
    return np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
        ],
        dtype=np.int8,
    )


@pytest.fixture
def sample_fp16_vectors():
    """Generate sample fp16 vectors for testing"""
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ],
        dtype=np.float16,
    )


@pytest.fixture
def sample_doc_ids():
    """Generate sample document IDs for testing"""
    return np.array([1, 2, 3, 4, 5], dtype=np.int32)


@pytest.fixture
def vectors_dataset(sample_vectors, sample_doc_ids):
    """Create a VectorsDataset instance for testing"""
    return VectorsDataset(
        vectors=sample_vectors, doc_ids=sample_doc_ids, dtype=DataType.FLOAT
    )


@pytest.fixture
def byte_vectors_dataset(sample_byte_vectors, sample_doc_ids):
    """Create a VectorsDataset instance for testing"""
    return VectorsDataset(
        vectors=sample_byte_vectors, doc_ids=sample_doc_ids, dtype=DataType.BYTE
    )


@pytest.fixture
def binary_vectors_dataset(sample_binary_vectors, sample_doc_ids):
    """Create a VectorsDataset instance for testing"""
    return VectorsDataset(
        vectors=sample_binary_vectors, doc_ids=sample_doc_ids, dtype=DataType.BINARY
    )


@pytest.fixture
def fp16_vectors_dataset(sample_fp16_vectors, sample_doc_ids):
    """Create a VectorsDataset instance for testing"""
    return VectorsDataset(
        vectors=sample_fp16_vectors, doc_ids=sample_doc_ids, dtype=DataType.FLOAT16
    )

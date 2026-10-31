# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
import pytest
from typing import Dict, Any

from core.common.models.index_builder.faiss import FaissIndexHNSWCagraBuilder
from core.common.models.index_builder import (
    FaissGpuBuildIndexOutput,
    FaissCpuBuildIndexOutput,
)


class TestFaissIndexHNSWCagraBuilder:

    @pytest.fixture
    def default_builder(self, request):
        return FaissIndexHNSWCagraBuilder()

    @pytest.fixture
    def custom_params(self) -> Dict[str, Any]:
        return {"ef_search": 200, "ef_construction": 150, "base_level_only": False}

    @pytest.fixture
    def mock_gpu_index(self):
        """Mock GPU index with tracking"""
        gpu_index = faiss.GpuIndexCagra()
        return gpu_index

    @pytest.fixture
    def mock_index_id_map(self):
        """Mock index ID map"""
        return faiss.IndexIDMap()

    def test_default_initialization(self, default_builder):
        assert default_builder.ef_search == 100
        assert default_builder.ef_construction == 100
        assert default_builder.base_level_only is True

    def test_custom_initialization(self, custom_params):
        custom_builder = FaissIndexHNSWCagraBuilder(**custom_params)
        assert custom_builder.ef_search == custom_params["ef_search"]
        assert custom_builder.ef_construction == custom_params["ef_construction"]
        assert custom_builder.base_level_only == custom_params["base_level_only"]

    def test_from_dict_partial(self):
        partial_params = {"ef_search": 200, "ef_construction": 158}
        builder = FaissIndexHNSWCagraBuilder.from_dict(partial_params)
        assert builder.ef_search == 200
        assert builder.ef_construction == 158
        assert builder.base_level_only is True  # default value

    def test_convert_gpu_to_cpu_index_success(
        self, default_builder, mock_gpu_index, mock_index_id_map
    ):
        """Test successful GPU to CPU index conversion"""
        gpu_build_output = FaissGpuBuildIndexOutput(
            gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
        )

        # Perform conversion
        result = default_builder.convert_gpu_to_cpu_index(gpu_build_output)

        # Verify result type and structure
        assert isinstance(result, FaissCpuBuildIndexOutput)
        assert isinstance(result.cpu_index, faiss.IndexHNSWCagra)
        assert isinstance(result.index_id_map, faiss.IndexIDMap)

        # Verify CPU index configuration
        assert result.cpu_index.hnsw.efConstruction == default_builder.ef_construction
        assert result.cpu_index.hnsw.efSearch == default_builder.ef_search
        assert result.cpu_index.base_level_only == default_builder.base_level_only

        assert gpu_build_output.index_id_map is None

    def test_convert_gpu_to_cpu_index_copy_error(
        self, default_builder, mock_gpu_index, mock_index_id_map
    ):
        def failing_copy(*args):
            raise RuntimeError("Simulated copy error")

        mock_gpu_index.copyTo = failing_copy

        with pytest.raises(Exception) as exc_info:
            default_builder.convert_gpu_to_cpu_index(
                FaissGpuBuildIndexOutput(
                    gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
                )
            )
        assert "Failed to convert GPU index to CPU index" in str(exc_info.value)
        assert "Simulated copy error" in str(exc_info.value)

    def test_skip_stored_vectors_passes_skip_storage(
        self, mock_gpu_index, mock_index_id_map
    ):
        """Test that skip_stored_vectors=True passes skip_storage=True to copyTo"""
        builder = FaissIndexHNSWCagraBuilder(skip_stored_vectors=True)

        copy_calls = []
        original_copyTo = mock_gpu_index.copyTo

        def tracking_copyTo(cpu_index, skip_storage=False):
            copy_calls.append(skip_storage)
            return original_copyTo(cpu_index, skip_storage)

        mock_gpu_index.copyTo = tracking_copyTo

        builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
            )
        )

        assert len(copy_calls) == 1
        assert copy_calls[0] is True

    def test_skip_stored_vectors_false_passes_skip_storage_false(
        self, default_builder, mock_gpu_index, mock_index_id_map
    ):
        """Test that skip_stored_vectors=False (default) passes skip_storage=False to copyTo"""
        copy_calls = []
        original_copyTo = mock_gpu_index.copyTo

        def tracking_copyTo(cpu_index, skip_storage=False):
            copy_calls.append(skip_storage)
            return original_copyTo(cpu_index, skip_storage)

        mock_gpu_index.copyTo = tracking_copyTo

        default_builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
            )
        )

        assert len(copy_calls) == 1
        assert copy_calls[0] is False

    def test_binary_skip_stored_vectors_passes_skip_storage(self):
        """Test that skip_stored_vectors=True passes skip_storage=True to binary copyTo"""
        from core.common.models.index_build_parameters import DataType

        builder = FaissIndexHNSWCagraBuilder(
            skip_stored_vectors=True, vector_dtype=DataType.BINARY
        )

        mock_gpu_binary_index = faiss.GpuIndexBinaryCagra()
        mock_binary_id_map = faiss.IndexBinaryIDMap()

        copy_calls = []
        original_copyTo = mock_gpu_binary_index.copyTo

        def tracking_copyTo(cpu_index, skip_storage=False):
            copy_calls.append(skip_storage)
            return original_copyTo(cpu_index, skip_storage)

        mock_gpu_binary_index.copyTo = tracking_copyTo

        builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_binary_index, index_id_map=mock_binary_id_map
            )
        )

        assert len(copy_calls) == 1
        assert copy_calls[0] is True

    def test_binary_skip_stored_vectors_false_does_not_skip_storage(self):
        """Test that skip_stored_vectors=False (default) does not skip storage for binary copyTo"""
        from core.common.models.index_build_parameters import DataType

        builder = FaissIndexHNSWCagraBuilder(vector_dtype=DataType.BINARY)

        mock_gpu_binary_index = faiss.GpuIndexBinaryCagra()
        mock_binary_id_map = faiss.IndexBinaryIDMap()

        copy_calls = []
        original_copyTo = mock_gpu_binary_index.copyTo

        def tracking_copyTo(cpu_index, skip_storage=False):
            copy_calls.append(skip_storage)
            return original_copyTo(cpu_index, skip_storage)

        mock_gpu_binary_index.copyTo = tracking_copyTo

        builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_binary_index, index_id_map=mock_binary_id_map
            )
        )

        assert len(copy_calls) == 1
        assert copy_calls[0] is False

    @staticmethod
    def _track_copyTo(gpu_index):
        """Replaces gpu_index.copyTo with a wrapper that records every call's arguments"""
        calls = []
        original_copyTo = gpu_index.copyTo

        def tracking_copyTo(cpu_index, skip_storage=False, *rest):
            calls.append((cpu_index, skip_storage, *rest))
            return original_copyTo(cpu_index, skip_storage, *rest)

        gpu_index.copyTo = tracking_copyTo
        return calls

    def test_upper_level_projection_defaults(self, default_builder):
        assert default_builder.project_link_to_base_layer is False
        assert default_builder.m == 0
        assert default_builder.upper_level_ef_construction == 0
        assert default_builder.upper_level_max_share == 0.5
        assert default_builder.upper_level_protect_last_in_edge is True

    def test_upper_level_projection_from_dict(self):
        builder = FaissIndexHNSWCagraBuilder.from_dict(
            {"project_link_to_base_layer": True, "m": 16}
        )
        assert builder.project_link_to_base_layer is True
        assert builder.m == 16
        assert builder.upper_level_max_share == 0.5  # default value

    def test_upper_level_projection_calls_projection_copyTo(
        self, mock_gpu_index, mock_index_id_map
    ):
        """project_link_to_base_layer=True calls copyTo(cpu_index, skip_storage, params, stats)"""
        builder = FaissIndexHNSWCagraBuilder(
            project_link_to_base_layer=True,
            m=24,
            upper_level_ef_construction=256,
            upper_level_max_share=0.25,
            upper_level_protect_last_in_edge=False,
        )
        calls = self._track_copyTo(mock_gpu_index)

        result = builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
            )
        )

        assert len(calls) == 1
        cpu_index, skip_storage, params, stats = calls[0]
        assert cpu_index is result.cpu_index
        assert result.cpu_index.base_level_only is True
        assert skip_storage is False
        assert isinstance(params, faiss.CagraUpperLevelProjectionParams)
        assert params.M == 24
        assert params.ef_construction == 256
        assert params.max_share == 0.25
        assert params.protect_last_in_edge is False
        assert isinstance(stats, faiss.CagraUpperLevelProjectionStats)
        assert stats.promoted == 1  # filled in by copyTo

    def test_upper_level_projection_default_params(
        self, mock_gpu_index, mock_index_id_map
    ):
        """Default projection settings are passed through unchanged"""
        builder = FaissIndexHNSWCagraBuilder(project_link_to_base_layer=True, m=16)
        calls = self._track_copyTo(mock_gpu_index)

        builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
            )
        )

        params = calls[0][2]
        assert params.M == 16
        assert params.ef_construction == 0
        assert params.max_share == 0.5
        assert params.protect_last_in_edge is True

    def test_upper_level_projection_with_skip_stored_vectors(
        self, mock_gpu_index, mock_index_id_map
    ):
        """skip_stored_vectors is still passed as skip_storage with the projection overload"""
        builder = FaissIndexHNSWCagraBuilder(
            project_link_to_base_layer=True, m=16, skip_stored_vectors=True
        )
        calls = self._track_copyTo(mock_gpu_index)

        builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
            )
        )

        assert len(calls) == 1
        assert calls[0][1] is True
        assert isinstance(calls[0][2], faiss.CagraUpperLevelProjectionParams)

    def test_upper_level_projection_disabled_uses_plain_copyTo(
        self, default_builder, mock_gpu_index, mock_index_id_map
    ):
        """project_link_to_base_layer=False (default) keeps the original copyTo call"""
        calls = self._track_copyTo(mock_gpu_index)

        default_builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
            )
        )

        assert len(calls) == 1
        assert len(calls[0]) == 2  # (cpu_index, skip_storage): no projection params

    def test_upper_level_projection_skipped_without_base_level_only(
        self, mock_gpu_index, mock_index_id_map
    ):
        """Projection requires base_level_only; otherwise the plain copyTo is used"""
        builder = FaissIndexHNSWCagraBuilder(
            project_link_to_base_layer=True, base_level_only=False
        )
        calls = self._track_copyTo(mock_gpu_index)

        builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
            )
        )

        assert len(calls) == 1
        assert len(calls[0]) == 2

    def test_upper_level_projection_not_used_for_binary(self):
        """Binary indexes never use the projection overload"""
        from core.common.models.index_build_parameters import DataType

        builder = FaissIndexHNSWCagraBuilder(
            project_link_to_base_layer=True, vector_dtype=DataType.BINARY
        )
        mock_gpu_binary_index = faiss.GpuIndexBinaryCagra()
        calls = self._track_copyTo(mock_gpu_binary_index)

        builder.convert_gpu_to_cpu_index(
            FaissGpuBuildIndexOutput(
                gpu_index=mock_gpu_binary_index,
                index_id_map=faiss.IndexBinaryIDMap(),
            )
        )

        assert len(calls) == 1
        assert len(calls[0]) == 2

    def test_upper_level_projection_requires_m(self, mock_gpu_index, mock_index_id_map):
        """project_link_to_base_layer=True without a positive m fails the conversion"""
        builder = FaissIndexHNSWCagraBuilder(project_link_to_base_layer=True)

        with pytest.raises(Exception) as exc_info:
            builder.convert_gpu_to_cpu_index(
                FaissGpuBuildIndexOutput(
                    gpu_index=mock_gpu_index, index_id_map=mock_index_id_map
                )
            )
        assert "m must be positive" in str(exc_info.value)

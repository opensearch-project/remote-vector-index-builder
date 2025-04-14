# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
import pytest
from typing import Dict, Any
from unittest.mock import patch
import os

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

    def test_write_cpu_index_io_error(self, default_builder, tmp_path):

        cpu_output = FaissCpuBuildIndexOutput(
            cpu_index=faiss.IndexHNSWCagra(), index_id_map=faiss.IndexIDMap()
        )

        # Create an invalid path
        invalid_path = str(tmp_path / "nonexistent" / "index.faiss")

        # Mock write_index to raise IOError
        with patch.object(faiss, "write_index", side_effect=IOError("IO Error")):
            with pytest.raises(Exception) as exc_info:
                default_builder.write_cpu_index(cpu_output, invalid_path)

            assert f"Failed to write index to file {invalid_path}" in str(
                exc_info.value
            )
            assert "IO Error" in str(exc_info.value)

    def test_write_cpu_index_file_content(self, default_builder, tmp_path):
        """Test the content of written file"""
        # Create mock CPU index and output
        cpu_index = faiss.IndexHNSWCagra()
        index_id_map = faiss.IndexIDMap()
        index_id_map.index = cpu_index

        cpu_output = FaissCpuBuildIndexOutput(
            cpu_index=cpu_index, index_id_map=index_id_map
        )

        output_path = str(tmp_path / "index.faiss")

        # Write index
        default_builder.write_cpu_index(cpu_output, output_path)

        # Verify file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

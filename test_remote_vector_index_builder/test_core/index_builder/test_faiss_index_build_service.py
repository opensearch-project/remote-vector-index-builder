# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
import pytest
from unittest.mock import patch
import os

from core.index_builder.faiss.faiss_index_build_service import FaissIndexBuildService


class TestFaissIndexBuildService:

    @pytest.fixture
    def service(self):
        with patch("os.cpu_count", return_value=8):
            service = FaissIndexBuildService()
            assert service.omp_num_threads == 6
            return service

    def test_build_index_success(
        self, service, vectors_dataset, index_build_parameters, tmp_path
    ):
        output_path = str(tmp_path / "output.index")
        service.build_index(index_build_parameters, vectors_dataset, output_path)

        # Verify OMP threads were set correctly
        assert faiss.omp_get_num_threads() == 6  # 8 CPUs - 2 = 6 threads
        assert os.path.exists(output_path)

    def test_build_index_gpu_creation_error(
        self, service, vectors_dataset, index_build_parameters, tmp_path
    ):
        with patch(
            "core.common.models.index_builder.faiss.FaissGPUIndexCagraBuilder.build_gpu_index",
            side_effect=Exception("GPU creation failed"),
        ):

            with pytest.raises(Exception) as exc_info:
                service.build_index(
                    index_build_parameters,
                    vectors_dataset,
                    str(tmp_path / "index.faiss"),
                )

            assert "GPU creation failed" in str(exc_info.value)

    def test_build_index_cpu_conversion_error(
        self, service, vectors_dataset, index_build_parameters, tmp_path
    ):
        with patch(
            "core.common.models.index_builder.faiss.FaissIndexHNSWCagraBuilder.convert_gpu_to_cpu_index",
            side_effect=Exception("Conversion failed"),
        ):

            with pytest.raises(Exception) as exc_info:
                service.build_index(
                    index_build_parameters,
                    vectors_dataset,
                    str(tmp_path / "index.faiss"),
                )

            assert "Conversion failed" in str(exc_info.value)

    def test_build_index_write_error(
        self, service, vectors_dataset, index_build_parameters, tmp_path
    ):
        """Test error handling during index writing"""
        with patch(
            "core.common.models.index_builder.faiss.FaissIndexHNSWCagraBuilder.write_cpu_index",
            side_effect=Exception("Write failed"),
        ):

            with pytest.raises(Exception) as exc_info:
                service.build_index(
                    index_build_parameters,
                    vectors_dataset,
                    str(tmp_path / "index.faiss"),
                )

            assert "Write failed" in str(exc_info.value)

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

from core.common.models import IndexSerializationMode
from core.common.models.index_build_parameters import DataType
from core.common.models.index_builder import CagraGraphBuildAlgo
from core.common.models.index_builder.faiss import FaissGPUIndexCagraBuilder
from core.index_builder.faiss.faiss_index_build_service import FaissIndexBuildService
from core.index_builder.index_builder_utils import calculate_ivf_pq_n_lists
from core.common.models.index_builder.faiss import FaissIndexHNSWCagraBuilder


class TestFaissIndexBuildService:

    @pytest.fixture
    def service(self):
        with patch("os.cpu_count", return_value=8):
            service = FaissIndexBuildService()
            assert service.omp_num_threads == 2
            return service

    def test_build_index_success(
        self, service, vectors_dataset, index_build_parameters, tmp_path
    ):
        self._do_test_build_index_success(
            service, vectors_dataset, index_build_parameters, tmp_path
        )

    def test_build_byte_index_success(
        self, service, byte_vectors_dataset, byte_index_build_parameters, tmp_path
    ):
        self._do_test_build_index_success(
            service, byte_vectors_dataset, byte_index_build_parameters, tmp_path
        )

    def test_build_fp16_index_success(
        self, service, fp16_vectors_dataset, byte_index_build_parameters, tmp_path
    ):
        self._do_test_build_index_success(
            service, fp16_vectors_dataset, byte_index_build_parameters, tmp_path
        )

    def test_build_binary_index_success(
        self, service, binary_vectors_dataset, binary_index_build_parameters, tmp_path
    ):
        self._do_test_build_index_success(
            service, binary_vectors_dataset, binary_index_build_parameters, tmp_path
        )

    def _do_test_build_index_success(
        self, service, vectors_dataset, index_build_parameters, tmp_path
    ):
        with patch(
            "core.common.models.index_builder.faiss.FaissGPUIndexCagraBuilder.from_dict"
        ) as mock_gpu_from_dict:

            output_path = str(tmp_path / "output.index")
            mock_gpu_from_dict.return_value = FaissGPUIndexCagraBuilder()

            cpu_index_output = service.build_index(
                index_build_parameters, vectors_dataset
            )
            service.write_cpu_index(
                cpu_index_output,
                index_build_parameters,
                IndexSerializationMode.DISK,
                output_path,
            )

            # Ensuring that FaissGPUIndexCagraBuilder parameters are set correctly
            expected_params = self._get_expected_gpu_params(
                service, index_build_parameters
            )
            mock_gpu_from_dict.assert_called_once_with(expected_params)

            assert faiss.omp_get_num_threads() == 2  # 8 CPUs/4 = 2 threads
            assert os.path.exists(output_path)

    def test_write_cpu_index_memory_mode(
        self, service, vectors_dataset, index_build_parameters
    ):
        from io import BytesIO

        cpu_index_output = service.build_index(index_build_parameters, vectors_dataset)

        buffer = BytesIO()
        service.write_cpu_index(
            cpu_index_output,
            index_build_parameters,
            IndexSerializationMode.MEMORY,
            buffer,
        )

        assert buffer.tell() > 0
        buffer.close()

    def test_write_binary_cpu_index_memory_mode(
        self, service, binary_vectors_dataset, binary_index_build_parameters
    ):
        from core.common.models import IndexSerializationMode
        from io import BytesIO

        cpu_index_output = service.build_index(
            binary_index_build_parameters, binary_vectors_dataset
        )

        buffer = BytesIO()
        service.write_cpu_index(
            cpu_index_output,
            binary_index_build_parameters,
            IndexSerializationMode.MEMORY,
            buffer,
        )

        assert buffer.tell() > 0
        buffer.close()

    def _get_expected_gpu_params(self, service, index_build_parameters):
        if index_build_parameters.data_type != DataType.BINARY:
            return {
                "ivf_pq_params": {
                    "n_lists": calculate_ivf_pq_n_lists(
                        index_build_parameters.doc_count
                    ),
                    "pq_dim": int(
                        index_build_parameters.dimension
                        / service.PQ_DIM_COMPRESSION_FACTOR
                    ),
                },
                "graph_degree": index_build_parameters.index_parameters.algorithm_parameters.m
                * 4,
                "intermediate_graph_degree": index_build_parameters.index_parameters.algorithm_parameters.m
                * 4,
            }
        else:
            return {
                "graph_build_algo": CagraGraphBuildAlgo.NN_DESCENT,
                "graph_degree": index_build_parameters.index_parameters.algorithm_parameters.m
                * 4,
                "intermediate_graph_degree": index_build_parameters.index_parameters.algorithm_parameters.m
                * 4,
            }

    def test_build_index_gpu_creation_error(
        self, service, vectors_dataset, index_build_parameters, tmp_path
    ):
        self._do_test_build_index_gpu_creation_error(
            service, vectors_dataset, index_build_parameters, tmp_path
        )

    def test_build_byte_index_gpu_creation_error(
        self, service, byte_vectors_dataset, byte_index_build_parameters, tmp_path
    ):
        self._do_test_build_index_gpu_creation_error(
            service, byte_vectors_dataset, byte_index_build_parameters, tmp_path
        )

    def test_build_fp16_index_gpu_creation_error(
        self, service, fp16_vectors_dataset, byte_index_build_parameters, tmp_path
    ):
        self._do_test_build_index_gpu_creation_error(
            service, fp16_vectors_dataset, byte_index_build_parameters, tmp_path
        )

    def test_build_binary_index_gpu_creation_error(
        self, service, binary_vectors_dataset, binary_index_build_parameters, tmp_path
    ):
        pass
        self._do_test_build_index_gpu_creation_error(
            service, binary_vectors_dataset, binary_index_build_parameters, tmp_path
        )

    def _do_test_build_index_gpu_creation_error(
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
                )

            assert "GPU creation failed" in str(exc_info.value)

    def test_build_index_cpu_conversion_error(
        self, service, vectors_dataset, index_build_parameters, tmp_path
    ):
        self._do_test_build_index_cpu_conversion_error(
            service, vectors_dataset, index_build_parameters, tmp_path
        )

    def test_build_byte_index_cpu_conversion_error(
        self, service, byte_vectors_dataset, byte_index_build_parameters, tmp_path
    ):
        self._do_test_build_index_cpu_conversion_error(
            service, byte_vectors_dataset, byte_index_build_parameters, tmp_path
        )

    def test_build_fp16_index_cpu_conversion_error(
        self, service, fp16_vectors_dataset, byte_index_build_parameters, tmp_path
    ):
        self._do_test_build_index_cpu_conversion_error(
            service, fp16_vectors_dataset, byte_index_build_parameters, tmp_path
        )

    def test_build_binary_index_cpu_conversion_error(
        self, service, binary_vectors_dataset, binary_index_build_parameters, tmp_path
    ):
        self._do_test_build_index_cpu_conversion_error(
            service, binary_vectors_dataset, binary_index_build_parameters, tmp_path
        )

    def _do_test_build_index_cpu_conversion_error(
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
                )

            assert "Conversion failed" in str(exc_info.value)

    def test_build_index_write_error(
        self, service, vectors_dataset, index_build_parameters, tmp_path
    ):
        self._do_test_build_index_write_error(
            service, vectors_dataset, index_build_parameters, tmp_path
        )

    def test_build_byte_index_write_error(
        self, service, byte_vectors_dataset, byte_index_build_parameters, tmp_path
    ):
        self._do_test_build_index_write_error(
            service, byte_vectors_dataset, byte_index_build_parameters, tmp_path
        )

    def test_build_fp16_index_write_error(
        self, service, fp16_vectors_dataset, byte_index_build_parameters, tmp_path
    ):
        self._do_test_build_index_write_error(
            service, fp16_vectors_dataset, byte_index_build_parameters, tmp_path
        )

    def test_build_binary_index_write_error(
        self, service, binary_vectors_dataset, binary_index_build_parameters, tmp_path
    ):
        self._do_test_build_index_write_error(
            service, binary_vectors_dataset, binary_index_build_parameters, tmp_path
        )

    def _do_test_build_index_write_error(
        self, service, vectors_dataset, index_build_parameters, tmp_path
    ):
        from core.common.models import IndexSerializationMode

        """Test error handling during index writing"""
        output_path = str(tmp_path / "index.faiss")

        # Build index successfully first
        cpu_index_output = service.build_index(index_build_parameters, vectors_dataset)

        # Mock faiss.write_index to fail
        write_func = (
            "faiss.write_index"
            if index_build_parameters.data_type != DataType.BINARY
            else "faiss.write_index_binary"
        )
        with patch(write_func, side_effect=Exception("Write failed")):
            with pytest.raises(Exception) as exc_info:
                service.write_cpu_index(
                    cpu_index_output,
                    index_build_parameters,
                    IndexSerializationMode.DISK,
                    output_path,
                )

            assert "Write failed" in str(exc_info.value)

    def test_build_index_skip_stored_vectors(
        self,
        service,
        vectors_dataset,
        skip_stored_vectors_index_build_parameters,
        tmp_path,
    ):
        """Test that skip_stored_vectors=True is passed through to CPU index builder"""
        with patch(
            "core.common.models.index_builder.faiss.FaissGPUIndexCagraBuilder.from_dict"
        ) as mock_gpu_from_dict, patch(
            "core.common.models.index_builder.faiss.FaissIndexHNSWCagraBuilder.from_dict"
        ) as mock_cpu_from_dict:
            from core.common.models.index_builder.faiss import FaissGPUIndexCagraBuilder

            mock_gpu_from_dict.return_value = FaissGPUIndexCagraBuilder()
            mock_cpu_from_dict.return_value = FaissIndexHNSWCagraBuilder(
                skip_stored_vectors=True
            )

            service.build_index(
                skip_stored_vectors_index_build_parameters, vectors_dataset
            )

            cpu_call_args = mock_cpu_from_dict.call_args[0][0]
            assert cpu_call_args["skip_stored_vectors"] is True

    def test_build_binary_index_skip_stored_vectors(
        self,
        service,
        binary_vectors_dataset,
        skip_stored_vectors_binary_index_build_parameters,
        tmp_path,
    ):
        """Test that skip_stored_vectors=True is passed through to CPU index builder for binary"""
        with patch(
            "core.common.models.index_builder.faiss.FaissGPUIndexCagraBuilder.from_dict"
        ) as mock_gpu_from_dict, patch(
            "core.common.models.index_builder.faiss.FaissIndexHNSWCagraBuilder.from_dict"
        ) as mock_cpu_from_dict:
            from core.common.models.index_builder.faiss import FaissGPUIndexCagraBuilder

            mock_gpu_from_dict.return_value = FaissGPUIndexCagraBuilder()
            mock_cpu_from_dict.return_value = FaissIndexHNSWCagraBuilder(
                skip_stored_vectors=True, vector_dtype=DataType.BINARY
            )

            service.build_index(
                skip_stored_vectors_binary_index_build_parameters,
                binary_vectors_dataset,
            )

            cpu_call_args = mock_cpu_from_dict.call_args[0][0]
            assert cpu_call_args["skip_stored_vectors"] is True

    def test_write_cpu_index_skip_stored_vectors_passes_io_flag(
        self,
        service,
        vectors_dataset,
        skip_stored_vectors_index_build_parameters,
        tmp_path,
    ):
        """Test that write_cpu_index passes IO_FLAG_SKIP_STORAGE when skip_stored_vectors=True"""
        cpu_index_output = service.build_index(
            skip_stored_vectors_index_build_parameters, vectors_dataset
        )

        write_calls = []
        original_write = faiss.write_index

        def tracking_write(index, writer, io_flags=0):
            write_calls.append(io_flags)
            return original_write(index, writer)

        faiss.write_index = tracking_write
        try:
            output_path = str(tmp_path / "output.index")
            service.write_cpu_index(
                cpu_index_output,
                skip_stored_vectors_index_build_parameters,
                IndexSerializationMode.DISK,
                output_path,
            )
            assert len(write_calls) == 1
            assert write_calls[0] == faiss.IO_FLAG_SKIP_STORAGE
        finally:
            faiss.write_index = original_write

    def test_write_binary_cpu_index_skip_stored_vectors_passes_io_flag(
        self,
        service,
        binary_vectors_dataset,
        skip_stored_vectors_binary_index_build_parameters,
        tmp_path,
    ):
        """Test that write_cpu_index passes IO_FLAG_SKIP_STORAGE for binary when skip_stored_vectors=True"""
        cpu_index_output = service.build_index(
            skip_stored_vectors_binary_index_build_parameters, binary_vectors_dataset
        )

        write_calls = []
        original_write = faiss.write_index_binary

        def tracking_write(index, writer, io_flags=0):
            write_calls.append(io_flags)
            return original_write(index, writer)

        faiss.write_index_binary = tracking_write
        try:
            output_path = str(tmp_path / "output.index")
            service.write_cpu_index(
                cpu_index_output,
                skip_stored_vectors_binary_index_build_parameters,
                IndexSerializationMode.DISK,
                output_path,
            )
            assert len(write_calls) == 1
            assert write_calls[0] == faiss.IO_FLAG_SKIP_STORAGE
        finally:
            faiss.write_index_binary = original_write

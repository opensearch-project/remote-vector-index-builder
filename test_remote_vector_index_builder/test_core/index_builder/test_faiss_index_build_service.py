# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
import logging
import pytest
from unittest.mock import patch
import os

from core.common.models import IndexSerializationMode
from core.common.models.index_build_parameters import DataType
from core.common.models.index_builder import CagraGraphBuildAlgo
from core.common.models.index_builder.faiss import FaissGPUIndexCagraBuilder
from core.index_builder.faiss.faiss_index_build_service import FaissIndexBuildService
from core.common.models.index_builder.faiss.ivf_pq_search_cagra_config import (
    IVFPQSearchCagraConfig,
)
from core.index_builder.index_builder_utils import (
    GRAPH_DEGREE_PER_M,
    calculate_effective_m,
    calculate_ivf_pq_n_lists,
)
from core.common.models.index_builder.faiss import FaissIndexHNSWCagraBuilder

PROJECTION_TARGET = "core.common.models.index_builder.faiss.faiss_index_hnsw_cagra_builder.project_upper_level"
GPU_BUILDER_FROM_DICT_TARGET = (
    "core.common.models.index_builder.faiss.FaissGPUIndexCagraBuilder.from_dict"
)
CPU_BUILDER_FROM_DICT_TARGET = (
    "core.common.models.index_builder.faiss.FaissIndexHNSWCagraBuilder.from_dict"
)
SERVICE_LOGGER = "core.index_builder.faiss.faiss_index_build_service"


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
        with patch(GPU_BUILDER_FROM_DICT_TARGET) as mock_gpu_from_dict:

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
            n_lists = calculate_ivf_pq_n_lists(index_build_parameters.doc_count)
            effective_m = calculate_effective_m(
                index_build_parameters.index_parameters.algorithm_parameters.m,
                index_build_parameters.doc_count,
                n_lists,
                IVFPQSearchCagraConfig.n_probes,
            )
            return {
                "ivf_pq_params": {
                    "n_lists": n_lists,
                    "pq_dim": int(
                        index_build_parameters.dimension
                        / service.PQ_DIM_COMPRESSION_FACTOR
                    ),
                },
                "graph_degree": effective_m * GRAPH_DEGREE_PER_M,
                "intermediate_graph_degree": effective_m * GRAPH_DEGREE_PER_M,
            }
        else:
            return {
                "graph_build_algo": CagraGraphBuildAlgo.NN_DESCENT,
                "graph_degree": index_build_parameters.index_parameters.algorithm_parameters.m
                * GRAPH_DEGREE_PER_M,
                "intermediate_graph_degree": index_build_parameters.index_parameters.algorithm_parameters.m
                * GRAPH_DEGREE_PER_M,
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
        with patch(GPU_BUILDER_FROM_DICT_TARGET) as mock_gpu_from_dict, patch(
            CPU_BUILDER_FROM_DICT_TARGET
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
        with patch(GPU_BUILDER_FROM_DICT_TARGET) as mock_gpu_from_dict, patch(
            CPU_BUILDER_FROM_DICT_TARGET
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

    @staticmethod
    def _with_projection(index_build_parameters, enabled=True):
        """Copy of the request with algorithm_parameters.project_link_to_base_layer set"""
        params = index_build_parameters.model_copy(deep=True)
        params.index_parameters.algorithm_parameters.project_link_to_base_layer = (
            enabled
        )
        return params

    def test_project_link_to_base_layer_defaults_to_false(self, index_build_parameters):
        """The API parameter is opt-in"""
        assert (
            index_build_parameters.index_parameters.algorithm_parameters.project_link_to_base_layer
            is False
        )

    def test_project_link_to_base_layer_parsed_from_request(
        self, index_build_parameters
    ):
        """project_link_to_base_layer is accepted next to m in algorithm_parameters"""
        from core.common.models import IndexBuildParameters

        request = index_build_parameters.model_dump(mode="json")
        request["index_parameters"]["algorithm_parameters"][
            "project_link_to_base_layer"
        ] = True
        parsed = IndexBuildParameters.model_validate(request)
        assert (
            parsed.index_parameters.algorithm_parameters.project_link_to_base_layer
            is True
        )

    @pytest.mark.parametrize(
        "dataset_fixture,params_fixture",
        [
            ("vectors_dataset", "index_build_parameters"),
            ("byte_vectors_dataset", "byte_index_build_parameters"),
            ("fp16_vectors_dataset", "byte_index_build_parameters"),
        ],
    )
    def test_build_index_passes_project_link_to_base_layer(
        self, service, request, dataset_fixture, params_fixture
    ):
        """The request flag and the effective m reach the CPU index builder"""
        vectors_dataset = request.getfixturevalue(dataset_fixture)
        index_build_parameters = self._with_projection(
            request.getfixturevalue(params_fixture)
        )
        with patch(
            CPU_BUILDER_FROM_DICT_TARGET,
            wraps=FaissIndexHNSWCagraBuilder.from_dict,
        ) as mock_cpu_from_dict, patch(PROJECTION_TARGET):
            service.build_index(index_build_parameters, vectors_dataset)

            cpu_call_args = mock_cpu_from_dict.call_args[0][0]
            assert cpu_call_args["project_link_to_base_layer"] is True
            # the m the CAGRA graph was built with
            n_lists = calculate_ivf_pq_n_lists(index_build_parameters.doc_count)
            assert cpu_call_args["m"] == calculate_effective_m(
                index_build_parameters.index_parameters.algorithm_parameters.m,
                index_build_parameters.doc_count,
                n_lists,
                IVFPQSearchCagraConfig.n_probes,
            )

    def test_build_index_without_flag_does_not_project(
        self, service, vectors_dataset, index_build_parameters
    ):
        """Without project_link_to_base_layer, build_index only runs the plain copyTo"""
        with patch(PROJECTION_TARGET) as mock_projection:
            service.build_index(index_build_parameters, vectors_dataset)

        mock_projection.assert_not_called()

    def test_build_index_with_flag_projects_the_dataset_vectors(
        self, service, vectors_dataset, index_build_parameters
    ):
        """End to end through build_index: the projection gets the CPU index, the dataset vectors and m"""
        with patch(PROJECTION_TARGET) as mock_projection:
            output = service.build_index(
                self._with_projection(index_build_parameters), vectors_dataset
            )

        mock_projection.assert_called_once()
        args, kwargs = mock_projection.call_args
        assert args[0] is output.cpu_index
        assert args[1] is vectors_dataset.vectors
        assert kwargs["m"] > 0
        assert (
            kwargs["ef_construction"]
            == index_build_parameters.index_parameters.algorithm_parameters.ef_construction
        )

    def test_build_binary_index_ignores_project_link_to_base_layer(
        self, service, binary_vectors_dataset, binary_index_build_parameters
    ):
        """Binary builds never project, even when the request asks for it"""
        with patch(PROJECTION_TARGET) as mock_projection:
            service.build_index(
                self._with_projection(binary_index_build_parameters),
                binary_vectors_dataset,
            )

        mock_projection.assert_not_called()

    @staticmethod
    def _with_base_level_only(index_build_parameters, base_level_only=False, **updates):
        """Deep copy of the request with algorithm_parameters.base_level_only set, plus top-level updates"""
        params = index_build_parameters.model_copy(deep=True, update=updates)
        params.index_parameters.algorithm_parameters.base_level_only = base_level_only
        return params

    @staticmethod
    def _build_and_capture(service, index_build_parameters, vectors_dataset, caplog):
        """
        Runs build_index. Returns the config dict given to the CPU index builder, the build output
        and the build service's log records about base_level_only.
        """
        with caplog.at_level(logging.DEBUG, logger=SERVICE_LOGGER), patch(
            CPU_BUILDER_FROM_DICT_TARGET,
            wraps=FaissIndexHNSWCagraBuilder.from_dict,
        ) as mock_cpu_from_dict:
            output = service.build_index(index_build_parameters, vectors_dataset)

        records = [
            record
            for record in caplog.records
            if record.name == SERVICE_LOGGER
            and "base_level_only" in record.getMessage()
        ]
        return mock_cpu_from_dict.call_args[0][0], output, records

    def test_base_level_only_defaults_to_true(self, index_build_parameters):
        """Requests that do not set the parameter keep today's base-level-only graph"""
        assert (
            index_build_parameters.index_parameters.algorithm_parameters.base_level_only
            is True
        )

    def test_base_level_only_is_optional_in_request(self, index_build_parameters):
        """A request from a client that predates the parameter still validates"""
        from core.common.models import IndexBuildParameters

        request = index_build_parameters.model_dump(mode="json")
        del request["index_parameters"]["algorithm_parameters"]["base_level_only"]
        parsed = IndexBuildParameters.model_validate(request)
        assert parsed.index_parameters.algorithm_parameters.base_level_only is True

    def test_base_level_only_parsed_from_request(self, index_build_parameters):
        """base_level_only is accepted in index_parameters.algorithm_parameters"""
        from core.common.models import IndexBuildParameters

        request = index_build_parameters.model_dump(mode="json")
        request["index_parameters"]["algorithm_parameters"]["base_level_only"] = False
        parsed = IndexBuildParameters.model_validate(request)
        assert parsed.index_parameters.algorithm_parameters.base_level_only is False

    @pytest.mark.parametrize(
        "dataset_fixture,params_fixture",
        [
            ("vectors_dataset", "index_build_parameters"),
            ("binary_vectors_dataset", "binary_index_build_parameters"),
        ],
    )
    def test_build_index_honors_base_level_only_false(
        self, service, request, caplog, dataset_fixture, params_fixture
    ):
        """float and binary vectors with stored vectors: the CPU index is built with upper levels"""
        index_build_parameters = self._with_base_level_only(
            request.getfixturevalue(params_fixture)
        )

        cpu_call_args, output, records = self._build_and_capture(
            service,
            index_build_parameters,
            request.getfixturevalue(dataset_fixture),
            caplog,
        )

        assert cpu_call_args["base_level_only"] is False
        assert output.cpu_index.base_level_only is False
        # Honored request logs are visible only at DEBUG
        assert [record.levelno for record in records] == [logging.DEBUG]
        assert index_build_parameters.vector_path in records[0].getMessage()

    @pytest.mark.parametrize(
        "dataset_fixture,params_fixture",
        [
            ("vectors_dataset", "index_build_parameters"),
            ("binary_vectors_dataset", "binary_index_build_parameters"),
        ],
    )
    def test_build_index_keeps_base_level_only_by_default(
        self, service, request, caplog, dataset_fixture, params_fixture
    ):
        """Without the parameter the build is unchanged and logs nothing about it"""
        cpu_call_args, output, records = self._build_and_capture(
            service,
            request.getfixturevalue(params_fixture),
            request.getfixturevalue(dataset_fixture),
            caplog,
        )

        assert cpu_call_args["base_level_only"] is True
        assert output.cpu_index.base_level_only is True
        assert records == []

    @pytest.mark.parametrize(
        "dataset_fixture,params_fixture,updates",
        [
            ("byte_vectors_dataset", "byte_index_build_parameters", {}),
            (
                "fp16_vectors_dataset",
                "index_build_parameters",
                {"data_type": DataType.FLOAT16},
            ),
            ("vectors_dataset", "skip_stored_vectors_index_build_parameters", {}),
            (
                "binary_vectors_dataset",
                "skip_stored_vectors_binary_index_build_parameters",
                {},
            ),
        ],
        ids=["byte", "half_float", "float_skip_stored", "binary_skip_stored"],
    )
    def test_build_index_falls_back_to_base_level_only(
        self, service, request, caplog, dataset_fixture, params_fixture, updates
    ):
        """
        faiss cannot build upper levels for half_float or byte vectors, or without the stored
        vectors: the request still succeeds but as a base-level-only build, and says so at INFO
        """
        index_build_parameters = self._with_base_level_only(
            request.getfixturevalue(params_fixture), **updates
        )

        cpu_call_args, output, records = self._build_and_capture(
            service,
            index_build_parameters,
            request.getfixturevalue(dataset_fixture),
            caplog,
        )

        assert cpu_call_args["base_level_only"] is True
        assert output.cpu_index.base_level_only is True
        assert [record.levelno for record in records] == [logging.INFO]
        message = records[0].getMessage()
        assert index_build_parameters.vector_path in message
        assert f"data_type={index_build_parameters.data_type.value}" in message
        assert (
            f"skip_stored_vectors={index_build_parameters.skip_stored_vectors}"
            in message
        )

    def test_build_index_with_upper_levels_does_not_project(
        self, service, vectors_dataset, index_build_parameters
    ):
        """An honored base_level_only=False takes precedence over project_link_to_base_layer"""
        with patch(PROJECTION_TARGET) as mock_projection:
            output = service.build_index(
                self._with_base_level_only(
                    self._with_projection(index_build_parameters)
                ),
                vectors_dataset,
            )

        mock_projection.assert_not_called()
        assert output.cpu_index.base_level_only is False

    def test_build_index_projects_when_base_level_only_falls_back(
        self, service, byte_vectors_dataset, byte_index_build_parameters
    ):
        """A base_level_only=False that is not honored leaves project_link_to_base_layer in effect"""
        with patch(PROJECTION_TARGET) as mock_projection:
            # Index build service/faiss does not support upper level build for byte vectors
            output = service.build_index(
                self._with_base_level_only(
                    self._with_projection(byte_index_build_parameters)
                ),
                byte_vectors_dataset,
            )

        mock_projection.assert_called_once()
        assert output.cpu_index.base_level_only is True

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

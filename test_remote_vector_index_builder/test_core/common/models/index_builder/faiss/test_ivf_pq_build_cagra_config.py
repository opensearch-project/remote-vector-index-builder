# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
import pytest
from typing import Dict, Any
import re
from core.common.models.index_builder.faiss import IVFPQBuildCagraConfig


class TestIVFPQBuildCagraConfig:

    @pytest.fixture
    def default_config(self):
        return IVFPQBuildCagraConfig()

    @pytest.fixture
    def custom_params(self) -> Dict[str, Any]:
        return {
            "n_lists": 2048,
            "kmeans_n_iters": 30,
            "kmeans_trainset_fraction": 0.7,
            "pq_bits": 6,
            "pq_dim": 16,
            "conservative_memory_allocation": False,
        }

    def test_default_initialization(self, default_config):
        assert default_config.n_lists == 1024
        assert default_config.kmeans_n_iters == 20
        assert default_config.kmeans_trainset_fraction == 0.5
        assert default_config.pq_bits == 8
        assert default_config.pq_dim == 0
        assert default_config.conservative_memory_allocation is True

    def test_custom_initialization(self, custom_params):
        config = IVFPQBuildCagraConfig(**custom_params)
        for key, value in custom_params.items():
            assert getattr(config, key) == value

    def test_validate_params_valid(self, custom_params):
        # Should not raise error
        IVFPQBuildCagraConfig._validate_params(custom_params)

    def test_validate_params_pq_dim_constraint(self):
        params = {"pq_bits": 6, "pq_dim": 7}
        with pytest.raises(
            ValueError, match="When pq_bits is not 8, pq_dim must be a multiple of 8"
        ):
            IVFPQBuildCagraConfig._validate_params(params)

    @pytest.mark.parametrize(
        "param,value,error_msg",
        [
            ("n_lists", 0, "n_lists must be positive"),
            ("n_lists", -1, "n_lists must be positive"),
            ("kmeans_n_iters", 0, "kmeans_n_iters must be positive"),
            ("kmeans_n_iters", -1, "kmeans_n_iters must be positive"),
            (
                "kmeans_trainset_fraction",
                0,
                "kmeans_trainset_fraction must be between 0 and 1",
            ),
            (
                "kmeans_trainset_fraction",
                1.1,
                "kmeans_trainset_fraction must be between 0 and 1",
            ),
            ("pq_bits", 3, re.escape("pq_bits must be one of [4, 5, 6, 7, 8]")),
            ("pq_bits", 9, re.escape("pq_bits must be one of [4, 5, 6, 7, 8]")),
            ("pq_dim", -1, "pq_dim must be non-negative"),
        ],
    )
    def test_validate_params_invalid(self, param, value, error_msg):
        params = {param: value}
        with pytest.raises(
            ValueError, match=f"IVFPQBuildCagraConfig param: {error_msg}"
        ):
            IVFPQBuildCagraConfig._validate_params(params)

    def test_to_faiss_config(self, custom_params):
        config = IVFPQBuildCagraConfig(**custom_params).to_faiss_config()

        assert isinstance(config, faiss.IVFPQBuildCagraConfig)
        assert config.n_lists == custom_params["n_lists"]
        assert config.kmeans_n_iters == custom_params["kmeans_n_iters"]
        assert (
            config.kmeans_trainset_fraction == custom_params["kmeans_trainset_fraction"]
        )
        assert config.pq_bits == custom_params["pq_bits"]
        assert config.pq_dim == custom_params["pq_dim"]
        assert (
            config.conservative_memory_allocation
            == custom_params["conservative_memory_allocation"]
        )

    def test_from_dict_partial(self):
        partial_params = {"n_lists": 2048, "kmeans_n_iters": 30}
        config = IVFPQBuildCagraConfig.from_dict(partial_params)
        assert config.n_lists == 2048
        assert config.kmeans_n_iters == 30
        assert config.kmeans_trainset_fraction == 0.5  # default value
        assert config.pq_bits == 8  # default value
        assert config.pq_dim == 0  # default value
        assert config.conservative_memory_allocation is True  # default value

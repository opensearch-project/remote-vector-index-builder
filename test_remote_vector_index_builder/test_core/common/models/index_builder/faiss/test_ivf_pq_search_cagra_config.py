# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
import pytest
from typing import Dict, Any

from core.common.models.index_builder.faiss import IVFPQSearchCagraConfig


class TestIVFPQSearchCagraConfig:

    @pytest.fixture
    def default_config(self):
        return IVFPQSearchCagraConfig()

    @pytest.fixture
    def custom_params(self) -> Dict[str, Any]:
        return {"n_probes": 40}

    def test_default_initialization(self, default_config):
        assert default_config.n_probes == 20

    def test_custom_initialization(self, custom_params):
        config = IVFPQSearchCagraConfig(**custom_params)
        assert config.n_probes == custom_params["n_probes"]

    @pytest.mark.parametrize(
        "n_probes,error_expected",
        [
            (-1, True),  # Negative value
            (0, True),  # Zero value
            (1, False),  # Valid minimum value
            (20, False),  # Default value
            (100, False),  # Large value
        ],
    )
    def test_n_probes_validation(self, n_probes, error_expected):
        params = {"n_probes": n_probes}

        if error_expected:
            with pytest.raises(
                ValueError,
                match="IVFPQSearchCagraConfig param: n_probes must be positive",
            ):
                IVFPQSearchCagraConfig.from_dict(params)
        else:
            config = IVFPQSearchCagraConfig.from_dict(params)
            assert config.n_probes == n_probes

    def test_to_faiss_config(self, custom_params):
        config = IVFPQSearchCagraConfig(**custom_params)
        faiss_config = config.to_faiss_config()

        assert isinstance(faiss_config, faiss.IVFPQSearchCagraConfig)
        assert faiss_config.n_probes == config.n_probes

    def test_from_dict_empty(self):
        config = IVFPQSearchCagraConfig.from_dict(None)
        assert isinstance(config, IVFPQSearchCagraConfig)
        assert config.n_probes == 20  # default value

    def test_from_dict_custom(self, custom_params):
        config = IVFPQSearchCagraConfig.from_dict(custom_params)
        assert isinstance(config, IVFPQSearchCagraConfig)
        assert config.n_probes == custom_params["n_probes"]

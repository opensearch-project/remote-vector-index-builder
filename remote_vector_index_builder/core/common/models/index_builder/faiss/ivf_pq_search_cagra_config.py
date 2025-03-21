# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import faiss
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class IVFPQSearchCagraConfig:
    """Configuration class for IVF-PQ GPU Cagra Index Search params"""

    # The number of clusters to search.
    n_probes: int = 20

    def to_faiss_config(self) -> faiss.IVFPQSearchCagraConfig:
        """
        Creates and configures the equivalent FAISS IVFPQSearchCagraConfig from the
        IVFPQSearchCagraConfig core datamodel.
        Returns:
            A configured FAISS IVFPQSearchCagraConfig object with search parameters for:
            - n_probes The number of clusters to search
        """

        config = faiss.IVFPQSearchCagraConfig()
        config.n_probes = self.n_probes
        return config

    @classmethod
    def from_dict(
        cls, params: Dict[str, Any] | None = None
    ) -> "IVFPQSearchCagraConfig":
        """
        Constructs a IVFPQSearchCagraConfig object from a dictionary of parameters.

        Args:
            params: Dictionary containing configuration parameters

        Returns:
            IVFPQSearchCagraConfig instance
        """
        if not params:
            return cls()

        # Validate parameters
        if "n_probes" in params:
            if params["n_probes"] <= 0:
                raise ValueError(
                    "IVFPQSearchCagraConfig param: n_probes must be positive"
                )
        return cls(**params)

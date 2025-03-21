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
class IVFPQBuildCagraConfig:
    """Configuration class for IVF-PQ GPU Cagra Index build params"""

    # The number of inverted lists (clusters)
    # Hint: the number of vectors per cluster (`n_rows/n_lists`) should be
    # approximately 1,000 to 10,000.
    n_lists: int = 1024

    # The number of iterations searching for kmeans centers (index building).
    kmeans_n_iters: int = 20
    # The fraction of data to use during iterative kmeans building.
    kmeans_trainset_fraction: float = 0.5

    # The bit length of the vector element after compression by PQ.
    # Possible values: [4, 5, 6, 7, 8].
    # Hint: the smaller the 'pq_bits', the smaller the index size and the
    # better the search performance, but the lower the recall.
    pq_bits: int = 8

    # The dimensionality of the vector after compression by PQ. When zero, an
    # optimal value is selected using a heuristic.
    # pq_bits` must be a multiple of 8.
    # Hint: a smaller 'pq_dim' results in a smaller index size and better
    # search performance, but lower recall. If 'pq_bits' is 8, 'pq_dim' can be
    # set to any number, but multiple of 8 are desirable for good performance.
    # If 'pq_bits' is not 8, 'pq_dim' should be a multiple of 8. For good
    # performance, it is desirable that 'pq_dim' is a multiple of 32
    # Ideally 'pq_dim' should be also a divisor of the dataset dim.
    pq_dim: int = 0

    # By default, the algorithm allocates more space than necessary for
    # individual clusters
    # This allows to amortize the cost of memory allocation and
    # reduce the number of data copies during repeated calls to `extend`
    # (extending the database).
    #
    # The alternative is the conservative allocation behavior; when enabled,
    # the algorithm always allocates the minimum amount of memory required to
    # store the given number of records. Set this flag to `true` if you prefer
    # to use as little GPU memory for the database as possible.
    conservative_memory_allocation: bool = True

    @staticmethod
    def _validate_params(params: Dict[str, Any]) -> None:
        """
        Pre-validates IVFPQBuildCagraConfig configuration parameters before object creation.

        Args:
            params: Dictionary of parameters to validate

        Raises:
            ValueError: If any parameter fails validation
        """
        if "n_lists" in params:
            if params["n_lists"] <= 0:
                raise ValueError(
                    "IVFPQBuildCagraConfig param: n_lists must be positive"
                )
        if "kmeans_n_iters" in params:
            if params["kmeans_n_iters"] <= 0:
                raise ValueError(
                    "IVFPQBuildCagraConfig param: kmeans_n_iters must be positive"
                )
        if "kmeans_trainset_fraction" in params:
            if not 0 < params["kmeans_trainset_fraction"] <= 1:
                raise ValueError(
                    "IVFPQBuildCagraConfig param: kmeans_trainset_fraction must be between 0 and 1"
                )
        # Validation Ref: https://github.com/facebookresearch/faiss/blob/main/faiss/gpu/GpuIndexCagra.h#L67
        if "pq_bits" in params:
            if params["pq_bits"] not in [4, 5, 6, 7, 8]:
                raise ValueError(
                    "IVFPQBuildCagraConfig param: pq_bits must be one of [4, 5, 6, 7, 8]"
                )
        if "pq_dim" in params:
            pq_dim = params["pq_dim"]
            if pq_dim < 0:
                raise ValueError(
                    "IVFPQBuildCagraConfig param: pq_dim must be non-negative"
                )

            # Check pq_dim constraints based on pq_bits
            pq_bits = params.get("pq_bits", 8)  # Use 8 if not specified
            if pq_bits != 8 and pq_dim % 8 != 0:
                raise ValueError(
                    "IVFPQBuildCagraConfig param: When pq_bits is not 8, pq_dim must be a multiple of 8"
                )

    def to_faiss_config(self) -> faiss.IVFPQBuildCagraConfig:
        """
        Creates and configures the equivalent FAISS IVFPQBuildCagraConfig from the
        IVFPQBuildCagraConfig core datamodel.

        Returns:
             A configured FAISS IVFPQBuildCagraConfig object with parameters for:
            - kmeans training set fraction
            - kmeans iteration count
            - Product Quantization bits and dimensions
            - Number of inverted lists (kmeans clusters)
            - Memory allocation strategy
        """

        config = faiss.IVFPQBuildCagraConfig()
        config.kmeans_trainset_fraction = self.kmeans_trainset_fraction
        config.kmeans_n_iters = self.kmeans_n_iters
        config.pq_bits = self.pq_bits
        config.pq_dim = self.pq_dim
        config.n_lists = self.n_lists
        config.conservative_memory_allocation = self.conservative_memory_allocation
        return config

    @classmethod
    def from_dict(cls, params: Dict[str, Any] | None = None) -> "IVFPQBuildCagraConfig":
        """
        Constructs a IVFPQBuildCagraConfig object from a dictionary of parameters.

        Args:
            params: Dictionary containing configuration parameters

        Returns:
            IVFPQBuildCagraConfig instance
        """
        if not params:
            return cls()

        # Validate parameters
        cls._validate_params(params)
        return cls(**params)

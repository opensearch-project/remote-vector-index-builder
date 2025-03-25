# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from enum import Enum
from typing import Annotated

from core.object_store.types import ObjectStoreType
from pydantic import BaseModel, ConfigDict, Field

# Type annotation for vector file paths that must end with .knnvec
VectorPathRegex = Annotated[str, Field(pattern=".+\\.knnvec$")]


class DataType(str, Enum):
    """Supported data types for vector values.

    Attributes:
        FLOAT: 32-bit floating point values
    """

    FLOAT = "float"


class SpaceType(str, Enum):
    """Distance method used for measuring vector similarities.

    Attributes:
        L2: Euclidean distance
        INNERPRODUCT: Dot product similarity
    """

    L2 = "l2"
    INNERPRODUCT = "innerproduct"


class Algorithm(str, Enum):
    """Supported algorithms for vector indexing.

    Attributes:
        HNSW: Hierarchical Navigable Small World graph
    """

    HNSW = "hnsw"


class Engine(str, Enum):
    """Available vector search engines.

    Attributes:
        FAISS: Facebook AI Similarity Search
    """

    FAISS = "faiss"


class AlgorithmParameters(BaseModel):
    """Configuration parameters for the HNSW algorithm.

    Attributes:
        ef_construction (int): Size of the dynamic candidate list for constructing
            the HNSW graph. Higher values lead to better quality but slower
            index construction. Defaults to 100.
        ef_search (int): The size of the dynamic list used during k-NN searches.
            Higher values result in more accurate but slower searches.
        m (int): Number of bi-directional links created for every new element
            during construction. Higher values lead to better search speed but
            more memory consumption. Defaults to 16.
    Note:
        The class is configured to allow extra attributes using the ConfigDict class.
    """

    ef_construction: int = 100
    ef_search: int = 100
    m: int = 16
    model_config = ConfigDict(extra="allow")


class IndexParameters(BaseModel):
    """Configuration parameters for vector index construction.

    This class defines the core index configuration including the algorithm type,
    distance metric, and algorithm-specific parameters.

    Attributes:
        algorithm (Algorithm): The vector indexing algorithm to use.
            Defaults to HNSW (Hierarchical Navigable Small World).
        space_type (SpaceType): The distance metric to use for vector comparisons.
            Defaults to L2 (Euclidean distance).
        algorithm_parameters (AlgorithmParameters): Specific parameters for the chosen
            algorithm. Defaults to standard HNSW parameters (ef_construction=128, m=16).
    """

    algorithm: Algorithm = Algorithm.HNSW
    space_type: SpaceType = SpaceType.L2
    algorithm_parameters: AlgorithmParameters = Field(
        default_factory=AlgorithmParameters
    )


class IndexBuildParameters(BaseModel):
    """Parameters required for building a vector index.

    This class encapsulates all necessary parameters for constructing a vector index,
    including data source information, vector specifications, and index configuration.

    Attributes:
        repository_type (str): The type of repository where the vector data is stored.
            Defaults to s3
        container_name (str): Name of the container (e.g., S3 bucket) containing the vector data.
        vector_path (VectorPathRegex): Path to the vector data file. Must end with .knnvec extension.
        doc_id_path (str): Path to the document IDs corresponding to the vectors.
        tenant_id (str): Optional identifier for multi-tenant scenarios. Defaults to empty string.
        dimension (int): The dimensionality of the vectors to be indexed.
        doc_count (int): Total number of documents/vectors to be indexed.
        data_type (DataType): The numerical format of the vector data.
            Defaults to FLOAT32.
        engine (Engine): The vector search engine to use for indexing.
            Defaults to FAISS.
        index_parameters (IndexParameters): Configuration for the index structure
            and algorithm. Defaults to standard HNSW configuration.

    Note:
        The class is configured to forbid extra attributes using the ConfigDict class,
        ensuring strict parameter validation.
    """

    repository_type: ObjectStoreType = ObjectStoreType.S3
    container_name: str
    vector_path: VectorPathRegex
    doc_id_path: str
    tenant_id: str = ""
    dimension: int = Field(gt=0)
    doc_count: int = Field(gt=1)
    data_type: DataType = DataType.FLOAT
    engine: Engine = Engine.FAISS
    index_parameters: IndexParameters = Field(default_factory=IndexParameters)
    model_config = ConfigDict(extra="forbid")

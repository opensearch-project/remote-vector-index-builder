# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from core.common.models import IndexBuildParameters
from app.models.request import RequestParameters


def create_request_parameters(
    index_build_parameters: IndexBuildParameters,
) -> RequestParameters:
    """Create a RequestParameters object from IndexBuildParameters.

    This function transforms an IndexBuildParameters object into a RequestParameters
    object, extracting and mapping the necessary fields for request processing.
    The RequestParameters object is later used for generating a request hash

    Args:
        index_build_parameters (IndexBuildParameters): The input parameters containing
            unique request attributes (such as vector path) for index building.

    Returns:
        RequestParameters: A new RequestParameters object containing the mapped
            attributes from the input parameters.

    """
    return RequestParameters(
        vector_path=index_build_parameters.vector_path,
        tenant_id=index_build_parameters.tenant_id,
    )

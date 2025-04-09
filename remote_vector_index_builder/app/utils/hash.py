# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import hashlib
from app.models.job import RequestParameters


def generate_job_id(request_parameters: RequestParameters) -> str:
    """Generate a unique hash-based job identifier from request parameters.

    This function creates a SHA-256 hash from the string representation of the
    provided request parameters, ensuring a unique and consistent identifier
    for each unique set of parameters.

    Args:
        request_parameters (RequestParameters): The request parameters object
            containing the job configuration details.

    Returns:
        str: A 64-character hexadecimal string representing the SHA-256 hash
            of the request parameters.
    """
    combined = str(request_parameters).encode()
    return hashlib.sha256(combined).hexdigest()

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
class ApiError(Exception):
    """Base exception for api errors"""

    pass


class HashCollisionError(ApiError):
    """Raised when there's a hash collision in the Request Store"""

    pass


class CapacityError(ApiError):
    """Raised when the worker does not have enough capacity to fulfill the request"""

    pass

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
class BuildServiceError(Exception):
    """Base exception for build service errors"""
    pass

class BuildError(BuildServiceError):
    """Raised when there's an error during index building"""
    pass

class ObjectStoreError(BuildServiceError):
    """Raised when there's an error with object store operations"""
    pass

class HashCollisionError(BuildServiceError):
    """Raised when there's a hash collision in the Request Store"""
    pass

class CapacityError(BuildServiceError):
    """Raised when the worker does not have enough capacity to fulfill the request"""
    pass
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
Expose public exceptions & warnings
"""


class BlobError(Exception):
    """Generic error raised when blob is downloaded to or uploaded from Object Store"""

    def __init__(self, message: str):
        super().__init__(message)


class UnsupportedObjectStoreTypeError(ValueError):
    """Error raised when creating an Object Store object"""

    pass


class VectorsDatasetError(Exception):
    """Generic error raised when converting a buffer into a Vector Dataset"""

    def __init__(self, message: str):
        super().__init__(message)


class UnsupportedVectorsDataTypeError(ValueError):
    """Error raised when creating a Vector Dataset because of unsupported data type"""

    pass
